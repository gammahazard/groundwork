"""Looking at the data and the runs — review, visualisation, housekeeping.

Nothing here is on the counting path. Every module exists so a human can SEE
what happened, or so the working tree stays tidy. If one of these breaks, counts
keep working; that is the line that puts a module in this package.

    gallery.py         contact sheets of labelled images for review.
    dedup_check.py     perceptual-hash near-duplicate finder. This one guards a
                       RULE rather than just showing something: the same capture on
                       both sides of train/test is the one thing that makes a
                       holdout score a lie, and `split` calls it on every run.
    timelapse.py       "watch it learn" — the run's periodic checkpoints scored
                       on the same images, as a filmstrip.
    heatmap.py         "where it looks" — detection density per image.
    live_trainer.py    an ultralytics trainer subclass that dumps augmented
                       batches mid-run for the cockpit's Watch modal.
    pending_janitor.py nightly: discard pending images nobody routed, and prune
                       the bot's per-count overlay PNGs (they hit ~1GB once).

Two of these are NOT just viewers, and both are launched as subprocesses rather
than imported:

  - timelapse and heatmap run at the tail of every retrain. Their module paths
    live in web/retrain_job.py's `_STAGE`, alongside the pipeline stages, because
    cancel()'s pkill has to match the same strings the launches use.
  - pending_janitor is the ExecStart of the janitor job's systemd unit on HQ, and it
    DELETES things. It only ever touches `pending/` (counted but never routed
    with a verdict) plus overlay images — never raw/, never the holdout. That
    boundary is the whole reason it is safe to run unattended at midnight.

live_trainer is imported by pipeline/train.py inside a try/except: if a future
ultralytics changes the trainer hook, training must fall back to stock rather
than fail. Keep that shape.
"""
from __future__ import annotations

__all__ = ["dedup_check", "gallery", "heatmap", "live_trainer",
           "pending_janitor", "timelapse"]
