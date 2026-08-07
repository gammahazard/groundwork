"""Scrap a training run — precisely, and completely.

TWO KINDS OF RUN, ONE MEANING OF "CANCEL". A yolo retrain is a detached pipeline
tracked by `owner_pid` in the retrain state file; a challenger is a detached
trainer tracked by a `gpu_holder.json` crumb in its own run dir. Different
bookkeeping, but the user pressing Cancel means the same thing either way: stop
burning the card, and leave nothing behind that looks like a result.

WHY "COMPLETELY" IS HALF THE JOB. A half-trained run dir with weights in it is
indistinguishable from a finished one at a glance, and the Models list will
happily offer it to serve. So cancelling deletes the directory the run created.
A run that produced a `count_eval.json` is FINISHED and is never touched — that
is the one artifact that only exists after scoring.

WHY killpg AND NOT pkill. `pkill -f <pattern>` matches every process on the box
whose command line contains the pattern, including jobs started by something
else — it killed an unrelated retrain at epoch 186/250 on 2026-07-30 — and
`pgrep -f` additionally matches its own command line, which deadlocked a wait
loop on 2026-07-31. Both launchers use `start_new_session=True`, so every run
already has its own process group and can be taken down by ID with no pattern
matching at all. If a pid is missing, the honest answer is "I could not target
this precisely", not a wider net.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import time
from pathlib import Path
from ..procs import alive as _alive


def kill_group(pid) -> tuple[bool, str]:
    """SIGKILL the whole process group of `pid`. Returns (killed, why-not).

    The group, not the process: a trainer spawns dataloader workers and (for
    torchrun) an elastic agent, and killing only the parent leaves those holding
    the GPU — which then blocks the next run on a lock whose owner looks dead.
    """
    if not pid:
        return False, "no pid recorded for this run"
    if not _alive(pid):
        return False, "already exited"
    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
        return True, ""
    except OSError as e:
        # No process group of its own means the launcher did not use
        # start_new_session. Kill the single process rather than widening to a
        # pattern; the strays are visible in the next status poll.
        try:
            os.kill(int(pid), signal.SIGKILL)
            return True, f"killed pid only, not its group ({e})"
        except OSError as e2:
            return False, f"could not kill {pid}: {e2}"


def _scrap_dir(run_dir: Path) -> bool:
    """Delete a half-finished run dir. Refuses anything that got as far as
    scoring — `count_eval.json` is written only by a completed evaluation, so
    its presence means this is a real result and not wreckage."""
    if not run_dir.is_dir():
        return False
    if (run_dir / "count_eval.json").exists():
        return False
    shutil.rmtree(run_dir, ignore_errors=True)
    return True


def challenger(alt_dirs, *, card: int | None = None, scrap: bool = True) -> dict:
    """Stop the alt/challenger run holding a GPU on THIS machine.

    `alt_dirs` is passed in rather than derived here so this module stays free
    of project layout (and of an import cycle with lab_ops, which owns the
    machine-level scan). Machine-level is correct for the scan itself: two
    projects must not both think the card is theirs.

    `card` narrows to one GPU, for the per-card Cancel buttons. A crumb written
    before the card was recorded has `card: None`, and those are matched by ANY
    request — refusing to cancel a job we can see running because we cannot
    prove which card it is on would be the worse failure. Once both cards can
    hold a run at the same time, an unknown-card crumb is genuinely ambiguous,
    so the result says which run was taken and the UI shows it back.
    """
    for alt in alt_dirs:
        for crumb in sorted(Path(alt).glob("*/gpu_holder.json")):
            try:
                h = json.loads(crumb.read_text())
            except Exception:  # noqa: BLE001
                continue
            if not _alive(h.get("pid")):
                continue                      # stale crumb; not a running job
            if card is not None and h.get("card") is not None \
                    and int(h["card"]) != int(card):
                continue                      # a different card's job
            run_dir = crumb.parent
            from .spawn import request_cancel
            if request_cancel(run_dir.name):
                # Spool mode: the trainer daemon is the run's parent (its own
                # PID namespace under Docker) — it does the killpg on our
                # behalf; the crumb/scrap bookkeeping below is still ours.
                killed, why = True, "cancel handed to the trainer daemon"
            else:
                killed, why = kill_group(h.get("pid"))
            if not killed:
                return {"ok": False, "error": why, "run": run_dir.name}
            # The crumb goes first: if the rmtree is slow, a poll arriving
            # mid-delete must not still see a live-looking holder.
            crumb.unlink(missing_ok=True)
            time.sleep(0.3)                   # let the group actually die
            scrapped = _scrap_dir(run_dir) if scrap else False
            return {"ok": True, "run": run_dir.name, "kind": "challenger",
                    "scrapped": scrapped, "note": why,
                    "card": h.get("card"), "what": h.get("what")}
    return {"ok": False, "error": "no challenger is training on this machine"}


def yolo() -> dict:
    """Stop the yolo retrain here. Delegates to the pipeline's own cancel,
    which already kills by group and prunes the dirs it created."""
    from . import retrain_job
    r = retrain_job.cancel()
    return {**r, "kind": "yolo"}


def anything(alt_dirs, *, card: int | None = None) -> dict:
    """Whatever is training on this machine. Used by the per-card button, which
    knows a card is busy but not which kind of job made it busy.

    Yolo first: it is the only kind whose state file records a definite
    "running", so if it says so there is nothing to search for. A challenger is
    found by scanning crumbs, which is the fuzzier answer of the two.
    """
    from . import retrain_job
    if retrain_job.status().get("status") == "running":
        return yolo()
    return challenger(alt_dirs, card=card)
