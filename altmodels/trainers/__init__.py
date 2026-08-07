"""One module per challenger family — the only model-specific code in altmodels/.

    deim.py    DEIMv2-N (Apache-2.0). The active challenger; best holdout
               0.0154. Needs .venv-deim.
    rtmdet.py  RTMDet-tiny via mmdet/mmengine. Deprecated but runnable; still
               owns `_vram_guard`, the per-card watchdog deim reuses.
    dfine.py   D-FINE via HF transformers — the ONE family that trains in the
               MAIN .venv. Concluded (~0.10, not the path).
    rfdetr.py  RF-DETR. Concluded; CLI-only, see the README History.

Everything AROUND them is deliberately model-agnostic and stays at altmodels/
root: `convert` (one COCO tree), `harness` (the exam, mirroring
dataset.pipeline.count_eval), `gpu` (admission locks), `nms`, `mmdet_zoo`, and
`predictors/`. Adding a family should mean adding one file here and one
predictor — if it means touching the harness, the abstraction has slipped.

TWO THINGS EVERY TRAINER MUST HONOUR, both learned expensively:

  - `gpu.acquire()` for the WHOLE run, non-blocking. Locks are per-card
    (gpu<N>.lock keyed on CUDA_VISIBLE_DEVICES), which is what lets two
    families train at once on different cards of one machine.
  - the dataset must be SPLIT before it is converted. The mirror deliberately
    excludes the machine-local split dirs, so on the lab they go stale the
    moment HQ gains an image — two runs once trained without the 19 newest images
    and reported numbers about the previous night's dataset. api/lab_ops.py
    splits first and verifies the image count; any new entry point must too.

These are launched as `python -m altmodels.trainers.<name>` from
groundwork/web/api/lab_ops.py, which builds the module path as a STRING — so a
rename here is invisible to every import checker. Grep lab_ops before moving one.
"""
from __future__ import annotations

__all__ = ["deim", "dfine", "rfdetr", "rtmdet"]
