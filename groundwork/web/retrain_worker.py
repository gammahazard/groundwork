"""Detached entrypoint for a retrain — its OWN process, not a thread inside the web service.

Why this module exists: the pipeline used to run as a `threading.Thread` inside
the web service, so `systemctl --user restart the web service` took the whole retrain down
mid-epoch. Burned 2026-07-30: yolov8n-47 died 6 seconds after a restart, 24
minutes into a 250-epoch run, and the state file said only "interrupted (owner
process died)". A thread can never outlive its process, so no amount of care in
the launcher fixes that — the work has to live somewhere else. It lives here.

BOTH of these are required, and that was measured rather than assumed:

  * this module — the pipeline stops being the web service's thread, and
  * `KillMode=process` in the web service.service — systemd's default kill mode
    SIGTERMs the service's whole CGROUP, and setsid does not change a process's
    cgroup. Tested all four combinations: a setsid'd child under the default
    `KillMode=control-group` still died; only the unit setting saved it.

So `start_new_session=True` in the launcher is NOT what protects this process.
It is there to give the pipeline its own process group, which is what lets
cancel() kill the pipeline and whatever stage subprocess it is inside with a
single killpg instead of pattern-matching pkill.

The JSON state file stays the only channel between here and the cockpit; it was
already written atomically for cross-process readers (the Telegram bot reads it
too), so nothing about the UI or the bot had to change.
"""
from __future__ import annotations

import argparse

from . import retrain_job


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", required=True,
                    help="imgsz values to train back-to-back on ONE split")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="train/val membership seed (split._val_rank). 0 is what "
                         "every run so far used, so a non-zero value is a "
                         "DIFFERENT experiment, not a re-run")
    from ..project import add_argument
    add_argument(ap)
    a = ap.parse_args()
    # _worker owns all state/log writing and catches its own exceptions, so a
    # failure here still leaves the cockpit an accurate "error" state.
    retrain_job.run_pipeline(a.sizes, a.val_frac,
                             a.epochs, a.batch, a.project, a.split_seed)


if __name__ == "__main__":
    main()
