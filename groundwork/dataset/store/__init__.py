"""Reading and writing image data — the layer that owns what is on disk.

    labelio.py          points <-> YOLO boxes for one image, per collection.
                        THE definition of that conversion: it preserves each
                        dot's CLASS index on the round trip. A second, private
                        implementation once did not, and rewrote every non-zero
                        class as class 0 on save — hence "one implementation".
    collect.py          the routing verbs. pending -> raw / testset /
                        needs_fix, and the truth-count + test-earmark sidecars.
    testset_buckets.py  per-holdout-image eval buckets (tags). An EVAL LENS
                        ONLY — the model never sees a bucket.
    exam.py             freeze the holdout as a NAMED, re-runnable exam.
    prelabel.py         pre-label an upload with the project's own model.

`paths` and `sidecar` deliberately stay at `dataset/` root rather than living
here, even though the plan originally put `paths` in this package. They are
foundation, not storage: `paths` has import sites spanning filters/, teacher/,
web/ and altmodels/, and `sidecar` is used by the web job-state writers too.
Burying either one behind `store.` would add a hop for every module in the
repo and imply a layering that isn't real.

WHAT THIS LAYER IS FOR: every collection is an exclusive bucket, and an image
moves between them by FILE MOVE (collect._move). That is what makes "the same
photo is never on both sides of train/test" true by construction rather than
by convention — there is only ever one copy. Anything that duplicates an image
instead of moving it breaks the guarantee the frozen holdout rests on.

The JSON sidecars here are written through `dataset/sidecar.py`, atomically:
the web service, the Telegram bots, the detached retrain pipeline and the
timers all read and write them from separate processes.
"""
from __future__ import annotations

from . import collect, labelio, testset_buckets   # noqa: F401

__all__ = ["collect", "labelio", "testset_buckets"]
