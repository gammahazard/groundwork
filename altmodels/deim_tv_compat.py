"""Let DEIMv2 run on a modern torchvision — which is what unlocks the 5070 Ti.

USE IT BY IMPORTING IT BEFORE THE VENDOR CODE RUNS:

    PYTHONPATH=~/DEIMv2 .venv-deim13/bin/python -c "
    import altmodels.deim_tv_compat, runpy, sys
    sys.argv = ['train.py', '-c', CFG]
    runpy.run_path('train.py', run_name='__main__')"

VERIFIED TWICE this way — 2 epochs on the 5070 Ti, COCO eval clean, 1:01 wall.

NOT VIA sitecustomize, and that is a measured negative rather than a preference.
Dropping this file in as `.venv-deim13/.../sitecustomize.py` loads (it IS in
sys.modules) and bridges NOTHING — `_install()` runs, finds both conditions
false, and returns having done nothing, while an explicit import of the same
code a moment later patches both hooks. Something about the state of
`torchvision.transforms.v2` at interpreter-startup time differs; I did not chase
it, because a launch-time import is the mechanism our own trainer controls and
is the one proven to work. Recorded so the next person does not spend the
evening I nearly did.

It is a FILE IN THE REPO rather than something hand-typed into a venv, because a
fix that lives only inside a gitignored directory is a fix nobody can find,
review or reproduce on the next machine.

=============================================================================
WHY — measured 2026-07-31, and the first three answers were wrong
=============================================================================

The worker has two cards: CUDA 0 is an RTX 5070 Ti (sm_120) and CUDA 1 an RTX 3090
(sm_86). `.venv-deim` is torch 2.5.1+cu121 whose compiled arch list stops at
sm_90 and carries no `compute_XX` PTX to JIT forward from, so DEIM could only
ever use the 3090. Building `.venv-deim13` (torch 2.13.0+cu130, archs up to
sm_120) was supposed to fix that.

It did not, and the failure looked like three different things in turn:

  1. Launching the trainer with the new interpreter changed NOTHING —
     `altmodels/trainers/deim.py` hardcodes `PYBIN = REPO/".venv-deim"/...` and
     re-execs, so the run was still on torch 2.5.1. The log said the right card
     and the wrong venv, and only a /proc walk showed it.
  2. Then: `NotImplementedError` from a DataLoader worker, with the real
     traceback swallowed. torchvision RENAMED the v2 hooks between 0.20 and
     0.28 — `_transform` -> `transform`, `_get_params` -> `make_params`. DEIM's
     three custom transforms override the old names, so the base class's
     `transform()` ran instead and raised. Nothing to do with CUDA at all.
  3. A first version of this shim did nothing, because it was installed inside
     a `try/except Exception: pass` that hid its own failure. A compat shim that
     cannot report not having applied is worse than none.

With both hooks bridged, DEIM trains on the 5070 Ti: 2 epochs, COCO eval clean,
1:01 wall. Verified, not inferred.

=============================================================================
WHAT THIS DOES NOT ESTABLISH
=============================================================================

That it RUNS is not that it produces COMPARABLE numbers. Every one of the 54
challenger runs was trained under torch 2.5.1 / torchvision 0.20, and a
torchvision major can move augmentation behaviour and numerics. A run under
`.venv-deim13` is a NEW baseline until a like-for-like comparison says
otherwise — same seed, same config, both venvs, holdout MAE compared. Until
then `.venv-deim` stays the registry's venv for deimv2-n, and this is opt-in.

`.venv-deim` is NEVER touched by any of this. The shim exists only in the venv
you copy it into.
"""
from __future__ import annotations

import os
import sys

_QUIET = os.environ.get("DEIM_TV_COMPAT_QUIET") == "1"


def _note(msg: str) -> None:
    # To stderr, and never silenced by default. The whole failure mode this file
    # exists inside is "a thing that did not apply and said nothing".
    if not _QUIET:
        print(f"[deim-tv-compat] {msg}", file=sys.stderr, flush=True)


def _install() -> None:
    from torchvision.transforms import v2

    base = v2.Transform
    bridged = []

    # _transform(self, inpt, params) -> transform(self, inpt, params)
    if hasattr(base, "transform") and not hasattr(base, "_transform"):
        _orig_transform = base.transform

        def transform(self, inpt, params):          # noqa: ANN001
            fn = getattr(type(self), "_transform", None)
            if fn is not None:
                return fn(self, inpt, params)
            return _orig_transform(self, inpt, params)

        base.transform = transform
        bridged.append("_transform->transform")

    # _get_params(self, flat_inputs) -> make_params(self, flat_inputs)
    if hasattr(base, "make_params") and not hasattr(base, "_get_params"):
        _orig_params = base.make_params

        def make_params(self, flat_inputs):         # noqa: ANN001
            fn = getattr(type(self), "_get_params", None)
            if fn is not None:
                return fn(self, flat_inputs)
            return _orig_params(self, flat_inputs)

        base.make_params = make_params
        bridged.append("_get_params->make_params")

    if bridged:
        import torchvision
        _note(f"torchvision {torchvision.__version__}: bridged "
              f"{', '.join(bridged)} for DEIMv2")


try:
    _install()
except ImportError:
    # No torchvision in this interpreter — nothing to bridge, and sitecustomize
    # runs in EVERY python that uses this venv, including pip's.
    pass
except Exception as e:  # noqa: BLE001
    # Report and continue. Refusing to start the interpreter over a compat shim
    # would be worse; a silent pass is what made the first attempt untraceable.
    _note(f"FAILED to install ({type(e).__name__}: {e}) — DEIM transforms will "
          f"raise NotImplementedError on the first batch")
