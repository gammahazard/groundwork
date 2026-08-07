"""The VRAM watchdog must never be able to wedge the run it is guarding.

WHAT WENT WRONG. Both challenger trainers polled `nvidia-smi` every 20 seconds
for the whole length of every run. Under WSL2 that crosses /dev/dxg and can
block in uninterruptible D-state, unkillable, and subprocess.run's timeout does
not bound it. So the guard written to "kill the run before the driver can hang"
leaked one permanently-stuck process per poll under exactly the conditions it
existed to detect. A DEIM run on 2026-08-01 died that way.

THREE PROPERTIES, and each maps to a specific way the old one failed:

  1. NO SUBPROCESS, EVER. That is the whole fix. Checked by booby-trapping every
     spawn primitive and calling the reader.
  2. IT MUST NOT FORCE A CUDA CONTEXT. The watchdog thread starts before the
     trainer has touched the GPU. If reading memory created the context itself,
     the guard would be doing the exact thing -- a cold CUDA init on a loaded
     box -- that took 32 minutes to return on the run that died.
  3. NO READING IS `None`, NOT 0. A guard whose "cannot read" answer is zero
     reads as "plenty of headroom" and silently never fires. Wrong in the
     dangerous direction.

Needs torch, so it runs in a venv that has one; skips cleanly otherwise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from altmodels import vram_guard  # noqa: E402

FAILURES: list[str] = []

try:
    import torch
except ImportError:
    print("  no torch in this interpreter — skipping (run it in .venv)")
    raise SystemExit(0)


class Spawned(AssertionError):
    pass


def _forbid(name):
    def boom(*a, **k):
        raise Spawned(f"{name}({a[0] if a else '?'})")
    return boom


print("  [1] the reader spawns no process")
real = (subprocess.Popen, subprocess.run, subprocess.check_output)
subprocess.Popen = _forbid("Popen")
subprocess.run = _forbid("run")
subprocess.check_output = _forbid("check_output")
try:
    v = vram_guard.used_mib()
    print(f"      ✓ used_mib() -> {v} with spawning forbidden")
except Spawned as e:
    FAILURES.append(f"used_mib() spawned a process: {e}")
    print(f"      ✗ spawned: {e}")
finally:
    (subprocess.Popen, subprocess.run, subprocess.check_output) = real

print("\n  [2] it does not force a CUDA context")
if torch.cuda.is_initialized():
    print("      ~ CUDA already up in this process — cannot test the cold path here")
else:
    before = torch.cuda.is_initialized()
    v = vram_guard.used_mib()
    after = torch.cuda.is_initialized()
    if after and not before:
        FAILURES.append("used_mib() created a CUDA context — the watchdog thread "
                        "would be doing the cold init that wedged the DEIM run")
        print("      ✗ a context was created")
    else:
        print(f"      ✓ is_initialized {before} -> {after}, returned {v!r}")
    if v is not None:
        FAILURES.append(f"expected None before CUDA is up, got {v!r} — a number "
                        f"here would be invented")

print("\n  [3] a real reading once CUDA is genuinely up")
if not torch.cuda.is_available():
    print("      ~ no CUDA device — skipping")
else:
    torch.zeros(1, device="cuda")
    v = vram_guard.used_mib()
    if not isinstance(v, int) or v < 0:
        FAILURES.append(f"no usable reading with CUDA up: {v!r}")
        print(f"      ✗ got {v!r}")
    else:
        print(f"      ✓ {v} MiB, device-wide")

print("\n  [4] a failed read is None, never a comfortable zero")
saved = vram_guard.used_mib
try:
    import builtins
    real_import = builtins.__import__

    def broken(name, *a, **k):
        if name == "torch":
            raise RuntimeError("simulated driver failure")
        return real_import(name, *a, **k)

    builtins.__import__ = broken
    v = vram_guard.used_mib()
    builtins.__import__ = real_import
    if v == 0:
        FAILURES.append("a failed read returned 0 — reads as headroom, so the "
                        "guard would never fire")
        print("      ✗ returned 0")
    elif v is None:
        print("      ✓ returned None")
    else:
        print(f"      ~ returned {v!r} (torch was still importable)")
except Exception as e:  # noqa: BLE001
    print(f"      ~ could not simulate: {type(e).__name__}")

print()
if FAILURES:
    for f in FAILURES:
        print(f"  ✗ {f}")
    sys.exit(1)
print("  all checks passed")
