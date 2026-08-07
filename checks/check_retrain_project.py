"""A retrain must read and write ITS OWN project's tree, never another's.

    .venv/bin/python checks/check_retrain_project.py

WHAT THIS CATCHES. Every STAGE of a retrain has been told `--project`
explicitly since the pivot, and each one prints its project before doing work.
The ORCHESTRATOR around them was the half that still assumed one fixed
project, at eight sites, and none of them would have raised:

  * `run_pipeline` asked whether a holdout existed by looking at the FIRST
    project's testset, so a second project with no holdout would still have
    been scored `--source testset` (or a project WITH one scored on `val`);
  * `find_latest_weights()` found the newest run in the first project's
    RUNS_DIR, so the run that got scored, snapshotted and written to the
    ledger under project B's name could be a different project's entirely;
  * `cancel()` pruned the first project's RUNS_DIR against project B's
    `pre_runs`, where every run absent from that snapshot reads as
    half-trained — and gets deleted;
  * `runtime_model._in_progress()` — in the BOT's process, not the cockpit's —
    subtracted B's `pre_runs` from the first project's run dirs, so every one
    of them read as "still cooking". `list_runs()` hid all of them and
    `resolve()` found no weights: a machine serving perfectly would stop
    serving because something else began training.

None of that throws. It produces a number about the wrong dataset, which is
the failure this repo is most expensive at.

HOW THIS CHECK CAN FAIL, which is the point. Every assertion MUTATES the
project and requires the answer to MOVE with it. Asserting today's answer
would pass just as green against un-fixed code — the standing lesson is that a
check that verifies nothing must FAIL, not pass.

NO GPU, NO NETWORK, NO TRAINING. It writes a state file and two empty run dirs
inside THIS checkout's outputs/ and restores both, so it is safe to run while
a card is busy — which is exactly when you want to ask.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


from groundwork import project as project_mod          # noqa: E402
from groundwork.config import OUTPUTS_DIR              # noqa: E402
from groundwork.dataset import paths                   # noqa: E402
from groundwork.web import retrain_job                 # noqa: E402
from groundwork.serve import runtime_model             # noqa: E402

A, B = "retraincheck-a", "retraincheck-b"
STATE = OUTPUTS_DIR / "retrain_state.json"


def _write_state(**kw) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"status": "running", "owner_pid": os.getpid(),
                                 "pre_runs": [], **kw}), encoding="utf-8")


def _mk(slug: str, made: list[Path]):
    p = project_mod.Project(slug=slug, name=slug,
                            dataset_root=OUTPUTS_DIR / "projects" / slug / "dataset")
    project_mod.manifest_path(slug).parent.mkdir(parents=True, exist_ok=True)
    project_mod.save(p)
    made.append(project_mod.manifest_path(slug).parent)
    return paths.for_project(slug)


def main() -> int:
    made: list[Path] = []
    state_backup = STATE.read_bytes() if STATE.exists() else None

    pp_a = _mk(A, made)
    pp_b = _mk(B, made)

    # Two run dirs in A's tree, so "did it look at the wrong one" is observable
    # rather than an empty-vs-empty tie.
    for name in ("checkrun-a", "checkrun-b"):
        (pp_a.RUNS_DIR / name).mkdir(parents=True, exist_ok=True)
    pp_b.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[check] project A {pp_a.slug} -> {pp_a.RUNS_DIR}")
        print(f"[check] project B {pp_b.slug} -> {pp_b.RUNS_DIR}")

        # ---- 1. _pp() FOLLOWS the state file, and moves when it changes ------
        #
        # GUARDED rather than called straight. Against code that predates the
        # fix these attributes simply do not exist, and an AttributeError here
        # would abort the run before parts 3 and 4 — which are the ones that
        # describe the expensive failures. A check that stops at its first
        # finding cannot tell you how much is broken.
        print("\n1. the orchestrator resolves the project from the state file")
        _pp = getattr(retrain_job, "_pp", None)
        if _pp is None:
            note(False, "retrain_job._pp is missing — the orchestrator has no "
                        "way to know whose retrain is running")
        else:
            _write_state(project=B)
            got_b = _pp().RUNS_DIR
            _write_state(project=A)
            got_a = _pp().RUNS_DIR
            note(got_b == pp_b.RUNS_DIR, f"project={B} -> {got_b}")
            note(got_a == pp_a.RUNS_DIR, f"project={A} -> {got_a}")
            note(got_a != got_b,
                 "the answer MOVES with the project (this is what makes 1 a check)")

        # ---- 2. a state with no project is a REFUSAL, not a guess ------------
        # There is no default project to fall back to any more; an orchestrator
        # that guessed would be binding somebody's retrain to somebody else's
        # tree, which is the exact bug the pivot removed.
        print("\n2. a state that names no project refuses to resolve")
        _write_state()                      # no `project` key at all
        if _pp is None:
            note(False, "skipped — no _pp to ask")
        else:
            try:
                got = _pp().RUNS_DIR
                note(False, f"no project in the state resolved to {got} — "
                            f"_pp() guessed a tree instead of refusing")
            except ValueError as e:
                note("project" in str(e).lower(),
                     f"_pp() raises and says why ({str(e)[:60]})")

        # ---- 3. pre_runs is snapshotted from the RIGHT tree ------------------
        print("\n3. pre_runs snapshots the project being trained")
        # _run_dirs lives in the package's control module (not re-exported —
        # only start() and cancel() need it).
        _run_dirs = getattr(retrain_job, "_run_dirs", None) \
            or retrain_job.control._run_dirs
        try:
            seen_a = set(_run_dirs(pp_a))
            seen_b = set(_run_dirs(pp_b))
        except TypeError:
            # The pre-fix signature takes no argument: it can only ever answer
            # about one fixed project, whoever is training.
            note(False, "_run_dirs() takes no project — pre_runs is snapshotted "
                        "from one fixed tree whatever is training")
            seen_a = seen_b = set()
        note({"checkrun-a", "checkrun-b"} <= seen_a,
             f"project A sees its own {len(seen_a)} run dir(s)")
        note(not (seen_b & {"checkrun-a", "checkrun-b"}),
             f"project B does NOT see A's runs (got {sorted(seen_b)})")

        # ---- 4. the SERVING path is not poisoned by another project's retrain
        # The one that bites in a different process: the bot's counter resolves
        # weights through list_runs() -> _in_progress().
        print("\n4. another project's retrain must not hide this project's models")
        _write_state(project=B, pre_runs=[])
        cooking = runtime_model._in_progress()
        note(not (cooking & {"checkrun-a", "checkrun-b"}),
             f"{B} training -> A's runs stay servable (cooking={sorted(cooking)})")

        # ...and the filter must STILL work for the project that IS training.
        _write_state(project=A, pre_runs=["checkrun-a"])
        cooking = runtime_model._in_progress()
        note("checkrun-b" in cooking and "checkrun-a" not in cooking,
             f"its own in-flight run is still hidden (cooking={sorted(cooking)})")
    finally:
        for slug in (A, B):
            shutil.rmtree(OUTPUTS_DIR / "projects" / slug, ignore_errors=True)
        for d in reversed(made):
            shutil.rmtree(d, ignore_errors=True)
        if state_backup is None:
            STATE.unlink(missing_ok=True)
        else:
            STATE.write_bytes(state_backup)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("PASS — a retrain reads and writes only its own project's tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
