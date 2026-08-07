"""The detection filters that sit between the detector and the count.

THE INVARIANT THIS PACKAGE EXISTS TO PROTECT:

    Runtime counting = model + filters, and the filters MUST be identical
    between eval and runtime. If they ever diverge, count_eval grades a system
    nobody runs and every certified MAE becomes a number about nothing.

Two call sites must agree, and there are only two:

    groundwork/serve/yolo_counter.py              what the bots and cockpit serve
    groundwork/dataset/pipeline/count_eval.py     what the holdout exam measures

(`deim_counter` inherits `YoloCounter.count()` and overrides only `_detect`, so
it is the same code path by construction — which is the point. `altmodels/
harness.py` mirrors count_eval for challengers and imports from here too.)

WHICH filters run is decided in ONE place, `groundwork/serve/serving_filters.py`,
read by every call site above. There is one filter:

    dot_dedupe    two detections closer than a fraction of the typical object
                  spacing are one object counted twice (an undertrained
                  checkpoint, or an elongated object getting a box per end)

Parameters that must match on both sides, stated once:

    dedupe threshold   max(dot_radius * 1.2, 0.4 x median NN spacing)
    dot_radius         max(3, round(min(w, h) / 200)), banker's rounding

`dot_radius` and the dedupe are additionally ported to JavaScript
(frontend/editor/editor_mirrors.js) so the editor overlay draws exactly the box
the save will write. `checks/check_editor_mirrors.py` EXECUTES both against
shared fixtures and diffs the output value by value — a real equivalence proof,
not a claim. A claim about a check is worth exactly as much as the check.
"""
from __future__ import annotations

from . import dot_dedupe   # noqa: F401

__all__ = ["dot_dedupe"]
