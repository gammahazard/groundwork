"""A polled endpoint must never spawn a process. Behavioural, not a grep.

WHY THIS CHECK EXISTS. `/api/retrain` called nvidia-smi once per poll while a
run was training. Under WSL2 that wedges in unkillable D-state, and because
`subprocess.run(timeout=)` finishes by waiting on a process that can never die,
each poll permanently ate one of anyio's 40 worker threads. The worker's cockpit
stopped serving every sync endpoint about five minutes into a run — twice now,
2026-07-30 and 2026-08-01.

A grep for "nvidia-smi" would not have caught it: the call site said
`run_stats.gpu()`. So this check does the only thing that generalises — it
makes spawning ANY process fatal and then calls the polled function.

IT PROVES IT CAN FAIL. The second half re-introduces the old behaviour and
asserts the check catches it. A guard that has never been seen to fail is not
evidence of anything (this repo has shipped two of those).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundwork.web import gpu_facts, retrain_job  # noqa: E402

FAILURES: list[str] = []


class Spawned(AssertionError):
    pass


def _forbid(name):
    def boom(*a, **k):
        raise Spawned(f"{name}({a[0] if a else '?'})")
    return boom


def call_status_with_spawning_forbidden():
    """Run retrain_job.status() with every spawn primitive booby-trapped."""
    real = (subprocess.Popen, subprocess.run, subprocess.check_output)
    subprocess.Popen = _forbid("Popen")
    subprocess.run = _forbid("run")
    subprocess.check_output = _forbid("check_output")
    try:
        return retrain_job.status(), None
    except Spawned as e:
        return None, str(e)
    finally:
        (subprocess.Popen, subprocess.run, subprocess.check_output) = real


# ---------------------------------------------------------------- the check
res, spawned = call_status_with_spawning_forbidden()
if spawned:
    FAILURES.append(f"status() spawned a process: {spawned}")
else:
    print(f"  ✓ retrain_job.status() spawned nothing  (gpu={res.get('gpu')})")

# The field must still be USEFUL, or we fixed the hang by deleting the feature.
sample = ("      Epoch    GPU_mem   box_loss\n"
          "\x1b[K     51/250      7.15G      1.153      0.797     0.9984   265   1280")
mem = gpu_facts.from_log(sample)
if not mem or mem["vram_used_gb"] != 7.15 or mem["epoch"] != 51:
    FAILURES.append(f"gpu_facts.from_log did not parse the ultralytics line: {mem}")
else:
    print(f"  ✓ VRAM still reported, from the log: {mem}")

if gpu_facts.from_log("") is not None:
    FAILURES.append("from_log('') should be None, not a fabricated reading")
else:
    print("  ✓ no log yet -> None (a missing field, not a made-up one)")

b = gpu_facts.busy(True, sample)
if not b.get("busy") or b.get("vram_used_gb") != 7.15:
    FAILURES.append(f"busy() lost the VRAM: {b}")
elif gpu_facts.busy(False).get("busy"):
    FAILURES.append("busy(False) claimed the card is in use")
else:
    print(f"  ✓ busy() reads the state file, not the driver: {b}")

# ------------------------------------------------- prove the check can fail
from groundwork.web import run_stats  # noqa: E402

# status() lives in retrain_job.progress, which reads `gpu_facts.busy` off the
# shared module object at call time — so patching the module attribute is what
# re-introduces the old behaviour for every consumer at once.
_saved = gpu_facts.busy
gpu_facts.busy = lambda running, tail="": run_stats.gpu()
_, spawned_now = call_status_with_spawning_forbidden()
gpu_facts.busy = _saved
if spawned_now:
    print(f"  ✓ CAN FAIL: restoring the nvidia-smi call was caught -> {spawned_now}")
else:
    FAILURES.append("the check did not catch a deliberately restored spawn — "
                    "it verifies nothing")

# run_stats.gpu() must refuse a second attempt once wedged, or one stuck
# process becomes forty.
run_stats._SMI_WEDGED = True
if run_stats.gpu() is not None:
    FAILURES.append("gpu() sampled the driver after a wedge — the leak repeats")
else:
    print("  ✓ gpu() stays off after a wedge (one leak, never forty)")
run_stats._SMI_WEDGED = False

print()
if FAILURES:
    for f in FAILURES:
        print(f"  ✗ {f}")
    sys.exit(1)
print("  all checks passed")
