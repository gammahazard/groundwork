"""Cancel must kill THIS retrain or nothing — never everything that looks like one.

    .venv/bin/python checks/check_cancel_targeting.py

WHAT THIS PROTECTS. cancel() takes down the pipeline by process GROUP, which is
precise: `_launch_pipeline` gives each one its own session, so one killpg ends
that pipeline and whichever stage it is inside, and nothing else. The fallback
when that fails used to be:

    pkill -9 -f "<split>|<train>|<count_eval>|<run_snapshot>|<timelapse>|<heatmap>"

which matches every training stage on the box with no way to tell whose job it
is. It already killed an unrelated live retrain at epoch 186/250 on 2026-07-30 —
a stubbed test called cancel(), and no amount of file redirection can stub pkill.
It is the same species as `pgrep -f` matching its own command line: a pattern
cannot express ownership.

That path only ever fires when precise targeting has ALREADY failed — owner_pid
missing, dead, or belonging to something that is not a pipeline. Every one of
those means we do not know which processes are ours, and the honest response is
to say so. The moment two retrains can exist, the shotgun stops being a last
resort and becomes a way to end someone else's run from a button marked Cancel.

HOW IT CAN FAIL, which is the point: every spawn primitive is booby-trapped, so
a cancel that shells out at all raises. Restore the pkill and part 2 goes red —
the same technique check_no_subprocess_on_poll.py uses on the polled endpoint.

NO GPU, NO KILLING. The state it writes names a pid that is alive but is NOT a
pipeline (this checker's own), which is precisely the case the fallback used to
handle by firing blind. Nothing is signalled, because nothing is targetable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


from groundwork.config import OUTPUTS_DIR              # noqa: E402
from groundwork.web import retrain_job                 # noqa: E402

STATE = OUTPUTS_DIR / "retrain_state.json"


class Spawned(Exception):
    """Raised by the booby traps. Reaching this is the finding."""


def main() -> int:
    backup = STATE.read_bytes() if STATE.exists() else None
    saved = {n: getattr(subprocess, n) for n in ("run", "Popen", "call",
                                                 "check_call", "check_output")}
    saved_system, saved_kill, saved_killpg = os.system, os.kill, os.killpg
    try:
        # ---- 1. the dead pattern is gone from the package entirely ------------
        print("\n1. no kill-by-pattern anywhere in the package")
        pkg = ROOT / "groundwork" / "web" / "retrain_job"
        files = sorted(pkg.glob("*.py"))
        note(bool(files), f"retrain_job package found ({len(files)} modules)")
        code = "\n".join(ln for f in files for ln in f.read_text().splitlines()
                         if not ln.lstrip().startswith("#"))
        note("pkill" not in code and "pgrep" not in code,
             "retrain_job contains no pkill/pgrep outside comments")
        note("_STAGE_KILL_RE" not in code,
             "_STAGE_KILL_RE is gone (it existed only to feed the pkill)")

        # ---- 2. an untargetable cancel spawns NOTHING and says why ------------
        print("\n2. a cancel that cannot identify its pipeline refuses")
        # Alive, but this process is a checker, not a pipeline — so _is_pipeline
        # says no and the old code fell straight through to pkill.
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"status": "running", "step": "training",
                                     "owner_pid": os.getpid(), "pre_runs": [],
                                     "project": "cancelcheck"}), encoding="utf-8")

        def trap(*a, **k):
            raise Spawned(f"cancel() shelled out: {a[:1]}")
        for n in saved:
            setattr(subprocess, n, trap)
        os.system = trap

        # SIGNAL 0 IS NOT A KILL — it is the liveness probe `_alive()` uses, and
        # trapping it indiscriminately broke cancel() before it could reach the
        # path under test (the checker reported a bug that was its own). Let the
        # probe through; refuse anything that actually delivers a signal.
        def kill_trap(pid, sig, *a, **k):
            if sig == 0:
                return saved_kill(pid, sig)
            raise Spawned(f"cancel() sent signal {sig} to pid {pid}")
        os.kill = kill_trap
        # killpg is the PRECISE path and is legitimate — but not here: this
        # state's owner is not a pipeline, so reaching it would mean cancel()
        # skipped _is_pipeline and killed a group on a recycled pid.
        os.killpg = lambda pgid, sig: (_ for _ in ()).throw(
            Spawned(f"cancel() killed process group {pgid} without "
                    f"confirming it was a pipeline"))

        try:
            r = retrain_job.cancel()
            spawned = None
        except Spawned as e:
            r, spawned = {}, str(e)

        note(spawned is None, f"nothing was spawned or signalled ({spawned or 'clean'})")
        # A live pid that is NOT a pipeline means the process we recorded has
        # been recycled — our pipeline is gone, so ending the job is correct.
        note(r.get("ok") is True,
             f"a recycled owner_pid cancels cleanly (ok={r.get('ok')})")
        note(json.loads(STATE.read_text()).get("status") != "running",
             "and the job is released, because nothing of ours is on the card")

        # ---- 3. THE FAIL-OPEN TEST -----------------------------------------
        #
        # owner_pid MISSING is the case where we genuinely cannot tell whether a
        # stage is still training. altmodels/gpu.py::_busy_reason admits a
        # sidecar as soon as this file stops saying "running", and start() reads
        # the same field — so marking the job failed here would release every
        # guard on the card while an unidentified process may still hold it.
        # That is two GPU jobs on one card, which this box has paid for three
        # times. A stuck "running" costs a manual fix; a false "error" costs a
        # run, so the state must be left ALONE.
        print("\n3. an unidentifiable job is left running, never marked failed")
        STATE.write_text(json.dumps({"status": "running", "step": "training",
                                     "pre_runs": [], "project": "cancelcheck"}),
                         encoding="utf-8")          # no owner_pid at all
        try:
            r = retrain_job.cancel()
            spawned = None
        except Spawned as e:
            r, spawned = {}, str(e)
        note(spawned is None, f"still nothing spawned or signalled ({spawned or 'clean'})")
        note(r.get("ok") is False,
             f"it reports failure rather than claiming success (ok={r.get('ok')})")
        note(json.loads(STATE.read_text()).get("status") == "running",
             "THE STATE IS UNTOUCHED — the GPU stays claimed (fail-safe)")
        note("running" in (r.get("error") or "").lower(),
             "and the message says the job was left running, and why")
    finally:
        for n, f in saved.items():
            setattr(subprocess, n, f)
        os.system, os.kill, os.killpg = saved_system, saved_kill, saved_killpg
        if backup is None:
            STATE.unlink(missing_ok=True)
        else:
            STATE.write_bytes(backup)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("PASS — cancel targets its own pipeline or refuses; it never fires blind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
