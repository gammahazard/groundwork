"""The serving filter switch — ONE definition, read by the runtime AND the exam.

Why this module exists at all: the counting pipeline is model + filters, and the
project's central rule is that the offline exam must apply exactly the filters
that ship, or the certified MAE measures a system nobody uses. Enforcing that
rule by hand in two independent code paths — `serve.yolo_counter.YoloCounter
.count` and `dataset.pipeline.count_eval.evaluate` — is precisely the
arrangement that lets them drift. Putting the switch here makes divergence a
code change rather than an oversight.

DEDUPE = True
    Two detections closer than the dedupe threshold (see
    dataset/filters/dot_dedupe.py) are one object counted twice. Cheap, fires
    only on genuinely degenerate spacing, and the per-image `dupes` count is
    recorded in every eval so its activity is measurable after the fact.

If you change this, change it for BOTH paths — that is the whole point of this
file — and re-run count_eval, because every certified MAE assumes the setting
in force when it was measured.
"""
from __future__ import annotations

DEDUPE = True
