"""GW_DEIM_GLOO must produce a real gloo process group that satisfies DEIM.

WHY THIS EXISTS. The claim behind the flag is a chain of four links, and three
of them are about code we do not own:

  1. the vendor never names a backend  (DEIMv2 engine/misc/dist_utils.py:44 is
     `init_process_group(init_method='env://')`, backend argument absent);
  2. torch therefore resolves it through Backend.default_device_backend_map,
     which is {"cpu": GLOO, "cuda": NCCL} — so CUDA goes over NCCL by default;
  3. our patch turns that absent backend into "gloo";
  4. and the ONE thing the vendor actually requires — engine/backbone/hgnetv2.py
     calling torch.distributed.get_rank() directly, outside the safe wrapper —
     is satisfied by any initialised group, gloo included.

Link 3 is ours and is the one that can rot. This tests the REAL function, pulled
out of deim_entry.py by AST rather than copied here, because a copy would keep
passing after the original changed.

NO GPU IS TOUCHED. gloo is a CPU transport; the whole point of the flag is that
this path never asks the driver for anything. That is also why this check is
safe to run on a box whose cards are busy — which, on the lab, they usually are.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []

# torchrun sets these; the vendor reads them via init_method='env://'.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")


def real_force_gloo():
    """_force_gloo, lifted out of deim_entry.py itself.

    deim_entry.py cannot be imported — its last line runpy's the vendor's
    train.py at module scope — so the function is extracted and exec'd. That
    still tests the shipped code, which a transcription would not.
    """
    src = (ROOT / "altmodels" / "deim_entry.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "_force_gloo"), None)
    if fn is None:
        FAILURES.append("deim_entry._force_gloo is gone — this check is stale")
        return None
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<deim_entry>", "exec"), ns)
    return ns["_force_gloo"]


print("  [1] torch's own default: an unnamed backend on CUDA means NCCL")
import torch                      # noqa: E402
import torch.distributed as dist  # noqa: E402

# CAPTURED BEFORE ANYTHING PATCHES IT. Step [4] rebuilds an inert patch to prove
# this check can fail — and the first version of it wrapped whatever was already
# installed, which by then was the gloo patch from [2]. So the "unpatched" call
# inherited the injection and reported gloo, and [4] declared the check useless.
# It was right: it just had the wrong culprit. Keep the pristine function.
_PRISTINE = dist.init_process_group

dmap = dist.Backend.default_device_backend_map
print(f"      default_device_backend_map = {dict(dmap)}")
if dmap.get("cuda") != "nccl":
    FAILURES.append(f"torch no longer defaults cuda to nccl (got {dmap.get('cuda')!r})"
                    " — the flag may now be solving a problem that is gone")
else:
    print("      ✓ cuda -> nccl, which is what the flag exists to override")
if "cuda" not in dist.Backend.backend_capability.get("gloo", []):
    FAILURES.append("torch says gloo can no longer carry CUDA tensors — DDP in "
                    "dist_utils.warp_model would break under this flag")
else:
    print("      ✓ gloo is declared capable of cuda, so the DDP wrap still works")

print("\n  [2] the patch turns the vendor's exact call into a gloo group")
force = real_force_gloo()
if force is not None:
    force()
    # THE VENDOR'S CALL, character for character (dist_utils.py:44).
    dist.init_process_group(init_method="env://")
    backend = dist.get_backend()
    print(f"      get_backend() = {backend}")
    if backend != "gloo":
        FAILURES.append(f"patched init produced {backend!r}, not gloo")
    else:
        print("      ✓ gloo")

    print("\n  [3] the one thing the vendor actually requires")
    try:
        rank = torch.distributed.get_rank()      # hgnetv2.py:562, unguarded
        print(f"      torch.distributed.get_rank() = {rank}  ✓ does not raise")
        if rank != 0:
            FAILURES.append(f"get_rank() returned {rank}, expected 0")
    except Exception as e:  # noqa: BLE001
        FAILURES.append(f"get_rank() still raises under gloo: {e!r} — the flag "
                        f"does not remove the reason torchrun was mandatory")
    dist.destroy_process_group()

print("\n  [4] prove it can fail")
# A patch that forgot to override anything must NOT pass [2]. Rebuild the same
# shape with the injection removed and confirm the assertion catches it.
probe = {"seen": None}


def _noop_patch():
    dist.init_process_group = _PRISTINE      # undo [2] first
    real = dist.init_process_group

    def patched(backend=None, *a, **kw):
        probe["seen"] = backend          # records None, injects nothing
        return real(backend, *a, **kw)
    dist.init_process_group = patched


_noop_patch()
os.environ["MASTER_PORT"] = "29578"
try:
    dist.init_process_group(init_method="env://")
    got = dist.get_backend()
    dist.destroy_process_group()
except Exception as e:  # noqa: BLE001 — no CUDA/NCCL here is itself the answer
    got = f"failed to init ({type(e).__name__})"
print(f"      an unpatched call resolves to: {got}")
if got == "gloo":
    FAILURES.append("an inert patch ALSO produced gloo — this check cannot tell "
                    "a working override from a broken one, so it verifies nothing")
else:
    print(f"      ✓ differs from gloo, so [2] is measuring the override itself")

print()
if FAILURES:
    for f in FAILURES:
        print(f"  ✗ {f}")
    sys.exit(1)
print("  gloo is a real, sufficient substitute for NCCL here")
