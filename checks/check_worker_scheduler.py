"""A joined worker must end up running the scheduler, not only the cockpit.

    .venv/bin/python checks/check_worker_scheduler.py

WHY THIS EXISTS. `autoscore` runs on the WORKER and nowhere else: a
challenger's checkpoints never leave the machine that trained them (adoption
copies results home with `--exclude train/`), so HQ physically cannot score
one — and HQ only adopts a run that already has `count_eval.json`. The
scheduler is therefore the middle link of `train → score → adopt`.

`join_worker._start_service` has two branches. The systemd branch calls
`install()`, which renders a timer per job from the scheduler's table. The
FALLBACK branch — taken on any box without a user systemd, which includes
containers and plenty of headless rigs — used to spawn the cockpit and
nothing else. A worker joined that way trains perfectly and never scores
anything: the run is finished, correct and invisible, with no row on HQ and
no error anywhere to explain it. Measured 2026-08-10 on a real worker whose
`outputs/jobs_status.json` had never been written at all.

Tested BEHAVIOURALLY — spawns are recorded and asserted — rather than by
grepping the source for a module name, because this repo has shipped two
checks that verified nothing.

READ-ONLY: nothing is spawned, no network, no GPU. Every process launch and
health probe is replaced.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groundwork.web import join_worker  # noqa: E402
from groundwork.web import spawn as spawn_mod  # noqa: E402
from groundwork.web import botsup  # noqa: E402

FAILS = 0


def expect(cond: bool, what: str) -> None:
    global FAILS
    print(("  ok  " if cond else "  FAIL") + f" {what}")
    if not cond:
        FAILS += 1


def main() -> int:
    launched: list[str] = []
    real_spawn, real_mode, real_healthy = (spawn_mod.spawn_detached,
                                           botsup.mode, join_worker._wait_healthy)
    spawn_mod.spawn_detached = lambda name, cmd, **k: (
        launched.append(" ".join(cmd[1:])) or 4242)
    botsup.mode = lambda: "detached"          # the no-systemd branch
    join_worker._wait_healthy = lambda *a, **k: True
    try:
        note = join_worker._start_service(8000)
    finally:
        spawn_mod.spawn_detached = real_spawn
        botsup.mode = real_mode
        join_worker._wait_healthy = real_healthy

    print("[1] the fallback branch starts BOTH long-running processes")
    expect(any("groundwork.web" in c for c in launched),
           "the cockpit is started (as it always was)")
    expect(any("groundwork.ops.scheduler" in c for c in launched),
           "the scheduler is started — without it autoscore never runs, so a "
           "challenger trains and is never scored, and HQ never adopts it")

    print("[2] the caveat still names the durable fix")
    expect("install --role worker" in (note or ""),
           "the returned note points at the systemd install for reboot survival")

    print("[3] autoscore is a worker job in the one job table")
    from groundwork.ops.scheduler import JOBS
    au = next((j for j in JOBS if j.name == "autoscore"), None)
    expect(au is not None and au.worker_only,
           "autoscore is worker_only — so HQ's own scheduler will not run it "
           "and the worker's must")

    print("PASS" if not FAILS else f"{FAILS} FAILURE(S)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
