"""Machines labelling data for machines — the two generations of teacher.

Nothing here serves a count. Every module writes LABELS that a student model
later trains on, which is why they belong together and away from the runtime.

    autolabel.py      LocateAnything-3B -> dataset/raw/. Generation one: the
                      3B VLM bootstrapped this whole dataset, then a human
                      corrected it. Driven by scratch/label_new.sh, one image
                      per fresh process so VRAM is released between images.
    la_probe.py       LocateAnything on ONE stored image, points to JSON, exit.
                      The dot editor's crosshair second opinion; advisory, never
                      writes a label. Launched by web/la_job.py.
    teacher_label.py  Generation two: the PINNED yolo champion labels casual
                      capture photos into dataset/distill/. Challengers opt in with
                      --include-distill.

THE RULE THAT MATTERS: machine labels never enter raw/ silently.

  - `teacher_label` writes to its own `distill/` bucket, never `raw/`, never the
    holdout, and skips any stem a human already verified in ANY collection.
  - `autolabel` DOES write straight into `raw/` — by design, because raw/ is the
    hand-correction queue and that is the first step of the loop, not the last.
    An image is only training data once a human has been through its dots.
  - `la_probe` writes no labels at all.

`teacher_label` runs the full serving pipeline (tuned conf/iou + whatever
`groundwork/serving_filters.py` enables + the stacked-dot dedupe) so its labels
match what the champion would actually count. It was the SIXTH copy of the hull
filter and the switch that disabled the other five missed it — see
`dataset/filters/__init__.py`. If you add a filter, this file is the one people
forget.
"""
from __future__ import annotations

# Deliberately NOT importing the submodules here, unlike dataset/synth/__init__.
# autolabel imports ..worker at module level, which loads torch + transformers —
# so an eager re-export would make `import groundwork.dataset.teacher` drag the
# 3B stack into any process that merely touched the package. These are launched
# as `python -m groundwork.dataset.teacher.<name>` and imported directly by the
# few callers that want them. Keep it that way.
__all__ = ["autolabel", "la_probe", "teacher_label"]
