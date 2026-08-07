"""The auto-scorer must never score a crashed run, and must never score beside a
training one.

    .venv/bin/python checks/check_autoscore.py

WHY THIS EXISTS. groundwork/autoscore.py runs unattended on the worker every five
minutes and starts GPU work. Two ways it could quietly do harm:

  1. SCORE SOMETHING THAT NEVER FINISHED. A crashed run leaves a checkpoint
     directory and no meta.json. If "has a checkpoint" were the test, a run that
     died at epoch 9 would be scored and its MAE would enter the leaderboard as
     though it were a finished model. That is the silent-wrong-data failure this
     repo is most expensive at, and it is exactly the mistake night_queue made in
     the other direction — it read the ABSENCE of a GPU crumb as completion and
     hid a crash for an hour. The recorded rule from that: absence of a
     crumb is not completion; meta.json is.

  2. START A SECOND GPU JOB. Two at once have failed 3 of 4 trials on this box —
     two /dev/dxg wedges and a CUDA fault at epoch 292 of 300. A timer that
     starts GPU work is a second authority over the cards unless it is strictly
     conservative.

THIS CHECK CAN FAIL, which is the whole point. Asserting that today's tree
produces today's answer would pass just as green with the meta.json test deleted,
because none of the fixtures would change. So part 4 MUTATES the rule out of the
source and asserts a crashed run then DOES appear — proving the check is reading
the rule rather than agreeing with a coincidence. This repo has shipped two
checks that verified nothing (one asserted a gate was textually present, so
`if False:` left it passing; one built its route list from `app.routes`, which is
empty), and the lesson was logged twice.

READ-ONLY AND GPU-FREE: every fixture is a temp directory. It starts nothing.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from groundwork.ops import autoscore  # noqa: E402

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


def _run(root: Path, name: str, *, meta=True, scored=False, ckpt=True,
         created=0.0, crumb_pid=None) -> Path:
    """One fixture run directory, in whichever state we need to describe."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    w = d / "best.pth"
    if ckpt:
        w.write_text("not really a checkpoint")
    if meta:
        (d / "meta.json").write_text(json.dumps({
            "run": name, "arch": "deimv2-n", "created": created,
            "best_checkpoint": str(w) if ckpt else None}))
    if scored:
        (d / "count_eval.json").write_text(json.dumps({"mae": 0.0}))
    if crumb_pid is not None:
        (d / "gpu_holder.json").write_text(json.dumps(
            {"pid": crumb_pid, "card": 1, "what": f"train {name}"}))
    return d


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="autoscore-check-"))
    alt = tmp / "alt"
    alt.mkdir()

    print("1. WHAT COUNTS AS FINISHED — meta.json, never a checkpoint on disk")
    _run(alt, "finished-unscored", created=100)
    _run(alt, "crashed-no-meta", meta=False, created=200)   # has a checkpoint!
    _run(alt, "already-scored", scored=True, created=300)
    _run(alt, "no-checkpoint", ckpt=False, created=400)
    _run(alt, "thing-smoke", created=500)
    got = {c["run"] for c in autoscore.candidates({"projcheck": alt})}

    note("finished-unscored" in got, "a finished, unscored run IS due")
    note("crashed-no-meta" not in got,
         "a CRASHED run (checkpoint on disk, no meta.json) is NOT scored")
    note("already-scored" not in got, "an already-scored run is not re-scored")
    note("no-checkpoint" not in got,
         "a run whose meta records no usable checkpoint is not scored")
    note("thing-smoke" not in got,
         "a *-smoke run is not scored — its MAE would be a fake competitor")

    print("\n2. ORDER — newest first, because one is scored per tick")
    for i, n in enumerate(["old", "middle", "newest"]):
        _run(alt / "ord", n, created=100 * (i + 1))
    order = [c["run"] for c in autoscore.candidates({"projcheck": alt / "ord"})]
    note(order == ["newest", "middle", "old"],
         f"newest first, so tonight's run beats the backlog (got {order})")

    print("\n3. THE GPU GATE — a live crumb anywhere means busy")
    busy_root = tmp / "busy"
    busy_root.mkdir()
    _run(busy_root, "training-now", crumb_pid=os.getpid())      # a REAL live pid
    who = autoscore._busy([busy_root])
    note(who is not None and "training-now" in who,
         f"a live crumb reports the card busy ({who})")

    dead_root = tmp / "dead"
    dead_root.mkdir()
    # A pid that cannot be alive. 2**22 is above every default pid_max.
    _run(dead_root, "finished-earlier", crumb_pid=2 ** 22)
    note(autoscore._busy([dead_root]) is None,
         "a STALE crumb (pid gone) does not block scoring forever")

    unread_root = tmp / "unreadable"
    unread_root.mkdir()
    d = _run(unread_root, "garbled")
    (d / "gpu_holder.json").write_text("{not json")
    note(autoscore._busy([unread_root]) is not None,
         "an UNREADABLE crumb counts as busy — unknown is not idle")

    print("\n4. MUTATION — is the rule being read, or is this a coincidence?")
    # THE COMPLETION RULE IS DEFENDED TWICE, which the first version of this
    # mutation discovered by failing. Deleting `if not meta_f.exists()` alone
    # does NOT let a crashed run through: the very next statement reads the file
    # and the `except (OSError, ValueError)` catches the missing one and skips it
    # anyway. That is a good property of autoscore.py — but it means a mutation
    # that removes only the first layer proves nothing, because the answer is
    # unchanged for the wrong reason. A fair test removes BOTH.
    import inspect
    src = inspect.getsource(autoscore.candidates)
    mutated = src.replace(
        "            if not meta_f.exists():\n                continue",
        "            if False:\n                continue", 1)
    note(mutated != src, "layer 1 (the exists test) is findable in the source")
    before = mutated
    mutated = mutated.replace(
        "                meta = json.loads(meta_f.read_text())",
        "                meta = {'best_checkpoint': str(d / 'best.pth')}", 1)
    note(mutated != before, "layer 2 (the guarded read) is findable too")
    ns = {"json": json, "Path": Path, "print": print,
          "SKIP_NAMES": autoscore.SKIP_NAMES,
          "SKIP_SUFFIXES": autoscore.SKIP_SUFFIXES}
    exec(compile(mutated, "<mutated>", "exec"), ns)
    try:
        mut_got = {c["run"] for c in ns["candidates"]({"projcheck": alt})}
    except Exception as e:  # noqa: BLE001
        mut_got = set()
        note(False, f"the mutated copy did not run ({type(e).__name__}: {e})")
    note("crashed-no-meta" in mut_got,
         "WITHOUT the meta.json test the crashed run IS returned — so part 1 "
         "is checking the rule, not agreeing with a coincidence")

    print()
    if FAIL:
        print("AUTO-SCORER IS UNSAFE:")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)
    print("only finished runs are scored, and never beside a training one")


if __name__ == "__main__":
    main()
