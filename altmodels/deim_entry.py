"""Run DEIMv2's own train.py, with our torchvision compat shim applied first.

    torchrun --nproc_per_node=1 altmodels/deim_entry.py -c <cfg> --use-amp ...

WHY A WRAPPER AT ALL. `.venv-deim13` exists so DEIM can use the worker's 5070 Ti —
`.venv-deim`'s torch has no sm_120 kernels. Getting there needs two torchvision
v2 hooks bridged (`_transform`→`transform`, `_get_params`→`make_params`), and
the shim has to be applied INSIDE the process that runs the vendor code.

`sitecustomize` does not work for this — measured, twice: the module loads and
bridges nothing, while the identical code imported at launch patches both hooks.
So the shim is imported here, explicitly, and the vendor entry is executed after
it. This is the mechanism that was verified training on the 5070 Ti; see
altmodels/deim_tv_compat.py for the three wrong answers on the way.

HARMLESS UNDER THE OLD VENV. On torchvision 0.20 the base class still has the
old hook names, so the shim finds nothing to bridge and says so. That means both
DEIM entries can go through the same entry point, which is the only way the two
are actually comparable — a wrapper used by one and not the other would be a
difference between them that is not the venv.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# THIS FILE IS EXECUTED AS A SCRIPT, not imported as a module: trainers/deim.py
# runs `torchrun … altmodels/deim_entry.py` with cwd set to the VENDOR repo. So
# sys.path[0] is altmodels/ and the cwd is somebody else's tree — neither makes
# `altmodels` importable. A repo import added here without this line died with
# "No module named 'altmodels'" at rank 0, AFTER the sync, split and convert had
# all succeeded (measured 2026-08-10). Anything imported below needs this first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from altmodels.deim_vendor import vendor_dir  # noqa: E402
VENDOR = vendor_dir()


# ── TWO OPT-IN EXPERIMENTS, both default OFF ─────────────────────────────────
#
# DEIM is the only challenger family that has wedged the worker's /dev/dxg while
# the other card trained (twice, 2026-08-01). Reading all four trainers, it is
# also the only one that does either of these two things, and both are removable
# without touching vendor code. They are SEPARATE flags on purpose: turning both
# on at once and seeing no wedge would not say which one mattered.
#
#   GW_DEIM_GLOO=1        force the process group onto gloo instead of NCCL.
#   GW_DEIM_CHILD_GUARD=1 the launcher stays CUDA-free; the VRAM guard and the
#                           sharing strategy move here, into the process that
#                           actually trains.
#
# Neither is on by default, so a normal run is byte-identical to every one of
# the 35 DEIM runs already in the ledger.


def _force_gloo() -> None:
    """Make an unspecified backend mean gloo rather than NCCL.

    WHY THIS IS THE WHOLE CHANGE. The vendor never chooses a backend --
    engine/misc/dist_utils.py:44 is `init_process_group(init_method='env://')`
    with the backend argument left off, and the explicit-backend line above it
    is commented out. torch then resolves it through
    Backend.default_device_backend_map, which is {"cpu": GLOO, "cuda": NCCL}, so
    CUDA traffic goes over NCCL by default rather than by decision.

    gloo is a legitimate target, not a workaround: torch's own
    Backend.backend_capability lists GLOO as supporting ["cpu", "cuda"], so the
    DDP wrap in dist_utils.warp_model keeps working on CUDA tensors.

    AND IT KEEPS THE RUN COMPARABLE. The alternative -- not initialising
    distributed at all -- would make warp_model skip DDP and warp_loader skip
    DistributedSampler, which changes the data ORDER and so cannot be compared
    against the existing runs. This changes only the transport, and at
    world_size 1 an allreduce over one rank is an identity operation.
    """
    import torch.distributed as dist
    real = dist.init_process_group

    def patched(backend=None, *a, **kw):
        if backend is None and kw.get("backend") is None:
            kw.pop("backend", None)
            backend = "gloo"
        return real(backend, *a, **kw)

    dist.init_process_group = patched
    print("[deim_entry] GW_DEIM_GLOO=1 — process group forced to gloo",
          flush=True)


def _child_guard() -> None:
    """Own the VRAM guard here instead of in the launcher.

    The launcher called torch.cuda.get_device_properties(0) purely to compute a
    ceiling, which CREATES A CUDA CONTEXT in a process whose only other job is
    to fork torchrun -- so DEIM ran with two live contexts on one card, which no
    other family does (rtmdet and rfdetr make the same call but then train in
    that same process). vram_guard derives the ceiling itself now.

    set_sharing_strategy moves with it, and is arguably more correct here: the
    dataloader workers belong to THIS process, and the launcher's setting never
    reached them.
    """
    import torch
    torch.multiprocessing.set_sharing_strategy("file_system")
    from altmodels import vram_guard
    vram_guard.start(None, label="deim-vram", headroom_mib=900)
    print("[deim_entry] GW_DEIM_CHILD_GUARD=1 — guard runs here; the launcher "
          "never touched CUDA", flush=True)

# BEFORE anything imports torchvision transforms. The vendor's train.py pulls in
# engine.data at import time, which is where its three custom transforms are
# defined — patch the base class first or they are already bound to a method
# that raises.
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altmodels.deim_tv_compat  # noqa: E402,F401 — applies on import

# AFTER the path setup above (both need `altmodels` importable) and BEFORE the
# vendor entry runs, which is the only window in which either can take effect.
if os.environ.get("GW_DEIM_GLOO") == "1":
    _force_gloo()
if os.environ.get("GW_DEIM_CHILD_GUARD") == "1":
    _child_guard()

# argv[0] becomes train.py's own name; torchrun already stripped its own flags,
# so everything after it is the vendor's (-c, --use-amp, --seed, -t).
sys.argv[0] = str(VENDOR / "train.py")
runpy.run_path(str(VENDOR / "train.py"), run_name="__main__")
