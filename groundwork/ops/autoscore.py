"""Score every challenger run that finished training and was never scored.

    .venv/bin/python -m groundwork.ops.autoscore            # score one, if due
    .venv/bin/python -m groundwork.ops.autoscore --dry-run  # say what it would do
    .venv/bin/python -m groundwork.ops.autoscore --all      # keep going, not one

RUNS ON THE WORKER, on a timer. That is the only place it can work: the checkpoints
never leave the machine that trained them (adopt_run copies results home with
`--exclude train/`), so HQ physically cannot score a worker run.

=============================================================================
THE GAP THIS CLOSES, and it was circular
=============================================================================

Training and scoring have always been separate wirings, and nothing joined them
once `scratch/night_queue.py` was deleted (2026-08-02). So a challenger finished,
wrote its checkpoint, and stopped — and then:

  * `the adopt job` on HQ only adopts a run that ALREADY has count_eval.json, so an
    unscored run never comes home;
  * `/api/lab/runs` reads the LOCAL alt tree, so an un-adopted run has no row in
    HQ's table;
  * no row means no `score` button.

Unscorable because unadopted, unadopted because unscored. Measured 2026-08-03:
seven runs sat finished-and-unscored on the worker, and the worker had 87 alt dirs
against HQ's 76. The owner's ask was blunt and correct — "I'd want every run we
do come back scored".

With this, the chain completes by itself: train -> score -> the adopt job (5 min) ->
the run appears in HQ's Runs table, charts and detail modal.

=============================================================================
WHAT IT TREATS AS "FINISHED", and why that is not the trap it looks like
=============================================================================

`meta.json`. Not the absence of a GPU crumb.

That distinction is the single most expensive lesson in this repo. night_queue
decided a run had finished when its `gpu_holder.json` disappeared — which is
equally what a CRASH looks like — so it logged "finished (card released)",
skipped scoring, and moved on, hiding a crash for an hour. That is the
rule it cost: **absence of a crumb is not completion; meta.json is.**

It is a safe marker because every trainer writes it LAST, after the training loop
returns, carrying `train_seconds` and `best_checkpoint` (verified in all four:
deim.py:353, rtmdet.py:244, dfine.py:223, rfdetr.py:160). A crashed run has no
meta.json and is therefore never scored — it is reported instead, because a run
that died deserves to be noticed rather than quietly retried forever.

=============================================================================
IT NEVER STARTS GPU WORK BESIDE GPU WORK
=============================================================================

Scoring is a short inference pass, not a 250-epoch training — but this box has
wedged its paravirtualised /dev/dxg twice and faulted a run once when two jobs
shared it, and the rule that came out of that was ONE AUTHORITY PER CARD. A timer
that starts GPU work is a second authority unless it is strictly conservative, so
it is: it refuses while ANY card holds a live crumb, and while a yolo retrain is
running. It waits rather than competing. A late score costs nothing; a wedged
card costs a run.

Unknown is not idle, everywhere: a crumb it cannot parse counts as busy.

=============================================================================
IT WILL NOT POLLUTE THE LEDGER
=============================================================================

Scoring writes count_eval.json, which is what puts a run on the leaderboard and
in the cross-model charts. So the accuracy question is not "did it score" but
"should this run have a score at all", and two kinds must not:

  * SMOKE RUNS (`*-smoke`) — deliberately tiny, and their MAE is meaningless.
    Scoring them would add fake competitors to a chart whose whole purpose is
    ranking real ones.
  * runs whose meta records no usable checkpoint — nothing to score.

Both are skipped by name and by fact, and both are printed rather than silently
passed over.

ONE RUN PER TICK by default. Scoring holds a card; taking them one at a time
keeps this a background courtesy rather than something that seizes the box, and
the timer comes round again in minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import HTTPException

from ..dataset import paths
from ..procs import alive as _alive

# The cockpit on THIS machine. Going through HTTP rather than importing the
# endpoint is deliberate: /api/lab/score already refuses an unrecognised arch, an
# unwired predictor and a missing venv, and re-deriving any of that here would be
# a second copy of the rule this repo keeps paying to keep in one place.
# Kept for reference only — scoring no longer goes over HTTP (see score()).
LOCAL = os.environ.get("GW_SELF_URL", "http://127.0.0.1:8000")

# A run whose MAE would be a lie on a chart built to rank real ones.
SKIP_SUFFIXES = ("-smoke",)
SKIP_NAMES = {"datasets", "pretrain", "parity"}


def _busy(alt_dirs) -> str | None:
    """What is holding a card right now, or None. MACHINE-WIDE, every project.

    A card is held by whichever project got there first, so asking through one
    project's tree would miss a job that is plainly running — the same reason
    lab_ops._every_alt exists.
    """
    for root in alt_dirs:
        if not root.exists():
            continue
        for crumb in sorted(root.glob("*/gpu_holder.json")):
            try:
                d = json.loads(crumb.read_text())
            except (OSError, ValueError):
                # Unreadable is NOT free. It means a job wrote a crumb we cannot
                # parse, which is a worse reason to start a second one, not a
                # better one.
                return f"{crumb.parent.name} (unreadable crumb)"
            if _alive(d.get("pid")):
                return f"{crumb.parent.name} on card {d.get('card', '?')}"
    return None


def _yolo_running() -> bool:
    """Is the cockpit's own retrain going? Read from disk, never nvidia-smi."""
    try:
        from ..config import OUTPUTS_DIR
        s = json.loads((OUTPUTS_DIR / "retrain_state.json").read_text())
    except (OSError, ValueError, ImportError):
        return False
    return s.get("status") == "running" and _alive(s.get("owner_pid"))


def candidates(alt_dirs_by_slug: dict[str, Path]) -> list[dict]:
    """Every run that trained cleanly and has no score, newest first.

    Newest first on purpose: tonight's run matters more than a backlog entry, and
    with one scored per tick the order IS the priority.
    """
    out = []
    for slug, root in alt_dirs_by_slug.items():
        if not root.exists():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if d.name in SKIP_NAMES:
                continue
            if (d / "count_eval.json").exists():
                continue                       # already scored
            meta_f = d / "meta.json"
            if not meta_f.exists():
                continue                       # never finished — see the header
            try:
                meta = json.loads(meta_f.read_text())
            except (OSError, ValueError):
                print(f"  ? {d.name}: meta.json unreadable — skipping")
                continue
            if d.name.endswith(SKIP_SUFFIXES):
                print(f"  - {d.name}: smoke run, not a competitor — skipping")
                continue
            best = meta.get("best_checkpoint")
            if not best or not Path(best).exists():
                print(f"  ! {d.name}: finished but no usable checkpoint "
                      f"({best or 'none recorded'}) — cannot score")
                continue
            out.append({"run": d.name, "slug": slug, "dir": d,
                        "arch": meta.get("arch"),
                        "created": meta.get("created") or 0})
    out.sort(key=lambda r: r["created"], reverse=True)
    return out


def score(run: str, slug: str, timeout: float = 180.0) -> dict:
    """Score one run, by CALLING the endpoint function rather than POSTing to it.

    IT USED TO GO OVER HTTP to this machine's own cockpit, and the reason given
    was sound: "/api/lab/score already refuses an unrecognised arch, an unwired
    predictor and a missing venv, and re-deriving any of that here would be a
    second copy of the rule this repo keeps paying to keep in one place."

    That reason is about not duplicating the RULES, and importing the function
    satisfies it better than HTTP did — it is literally the same code, with the
    same refusals, and it cannot drift.

    WHAT FORCED THE CHANGE: the cockpit is gated now. A timer POSTing to
    127.0.0.1:8000 gets 401 the moment this machine runs the current build, and
    scoring would stop — silently, because the failure is a status code nobody is
    watching, on the mechanism that makes "every run comes back scored" true.

    The alternatives were worse. Exempting localhost from the gate would let
    anything on the box drive the cockpit; giving the timer a key means a
    credential on disk for a call that never needed to leave the process.
    """
    from ..web.api.lab_ops import ScoreReq, score as _score
    from .. import project as project_mod
    try:
        return _score(ScoreReq(run_name=run), project_mod.load(slug))
    except HTTPException as e:
        return {"ok": False, "error": f"{e.status_code}: {e.detail}"}
    except Exception as e:  # noqa: BLE001 — one bad run must not stop the timer
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="list what is due and start nothing")
    ap.add_argument("--all", action="store_true",
                    help="score every due run, not just the first")
    a = ap.parse_args()

    by_slug = {pp.slug: pp.ALT_DIR for pp in paths.every_project()}
    print(f"[autoscore] {time.strftime('%H:%M:%S')} · "
          f"{len(by_slug)} project(s): {', '.join(by_slug) or 'none'}")

    due = candidates(by_slug)
    if not due:
        print("[autoscore] nothing to score")
        return
    print(f"[autoscore] {len(due)} run(s) finished and unscored: "
          + ", ".join(r["run"] for r in due[:6])
          + (" …" if len(due) > 6 else ""))

    # THE GATE. Checked AFTER listing, so the log always says what is waiting
    # even on a busy box — otherwise a run could sit unscored for hours with
    # nothing anywhere saying why.
    if (who := _busy(list(by_slug.values()))):
        print(f"[autoscore] {who} is training — waiting. Scoring holds a card, "
              f"and two GPU jobs at once have failed 3 of 4 trials on this box.")
        return
    if _yolo_running():
        print("[autoscore] a yolo retrain is running — waiting for the same reason")
        return
    if a.dry_run:
        for r in due:
            print(f"  would score {r['run']} ({r['arch']}, project {r['slug']})")
        return

    for r in due:
        print(f"[autoscore] scoring {r['run']} ({r['arch']}, project {r['slug']})")
        res = score(r["run"], r["slug"])
        if not res.get("ok"):
            print(f"  FAILED: {res.get('error')}", file=sys.stderr)
        else:
            print(f"  started, pid {res.get('pid')} — waiting for it")
            _await(r, res.get("pid"), by_slug[r["slug"]])
        if not a.all:
            # One per tick: scoring holds a card, and the timer returns shortly.
            break


# How long a score may take before this tick gives up WATCHING it. Not a kill:
# the child is detached and may still be working. Measured: a 68-image deim
# score is ~30-60s including model load, so ten minutes is generous.
WATCH_S = 600
POLL_S = 5


def _await(r: dict, pid, alt_dir) -> None:
    """Wait for the score to actually land, and SAY what happened.

    THIS IS THE WHOLE POINT OF THE FUNCTION, and it was missing. `score()`
    launches a detached child and returns a pid; nothing ever looked at it
    again. So a scorer that died instantly was indistinguishable from one that
    was working — and because the run stayed unscored, the next tick started
    another, and the next. Measured 2026-08-05: deimv2n13-1280-wtrx, a finished
    400-epoch run, was relaunched every five minutes for hours. Each child wrote
    ZERO bytes to lab.log (which takes both its stdout and its stderr), the
    owner's run was invisible on HQ the whole time, and the only reason it was
    found is that he remembered training it.

    A run that cannot be scored must fail LOUDLY once, not quietly twelve times
    an hour. The check is the crumb, not the exit status: `count_eval.json` is
    what adoption keys off, so "did it appear" is the only question that matters
    — and it stays true if the child is replaced by something that forks.
    """
    out = alt_dir / r["run"] / "count_eval.json"
    log = alt_dir / r["run"] / "lab.log"
    before = log.stat().st_size if log.exists() else 0
    t0 = time.time()
    while time.time() - t0 < WATCH_S:
        if out.exists():
            print(f"  scored in {time.time() - t0:.0f}s — the adopt job carries it to HQ")
            return
        if pid and not _alive(pid):
            break
        time.sleep(POLL_S)

    if out.exists():                       # landed as the child exited
        print(f"  scored in {time.time() - t0:.0f}s")
        return
    # It is gone (or hung) and produced no score. Say so, with whatever it wrote
    # — the tail of its own log is the only evidence there is.
    tail = ""
    try:
        if log.exists() and log.stat().st_size > before:
            with open(log, "rb") as f:
                f.seek(before)
                tail = f.read()[-600:].decode("utf-8", "replace").strip()
    except OSError:
        pass
    print(f"[autoscore] {r['run']}: the scorer exited without writing "
          f"count_eval.json after {time.time() - t0:.0f}s. It will be retried "
          f"next tick — if this repeats, that is the bug.", file=sys.stderr)
    print(f"  its output: {tail or '(it wrote nothing at all — the child died '
          f'before Python produced a line)'}", file=sys.stderr)


if __name__ == "__main__":
    main()
