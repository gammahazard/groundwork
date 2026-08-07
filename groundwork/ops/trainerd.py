"""The trainer daemon — the process that PARENTS detached training runs.

WHY IT EXISTS. A container is a cgroup: when the app container is recreated on
upgrade, everything inside dies — `start_new_session` does not help, exactly
as KillMode taught the native deployment (a 250-epoch run once died 24 minutes
in, six seconds after a service restart). So under Docker the app process must
not be the training run's parent. With GW_TRAINER=spool, the app WRITES a job
file instead of forking; this daemon — its own container, untouched by app
upgrades — claims the file and forks the run.

The contract stays what it always was: the launched worker owns
outputs/retrain_state.json (the cross-process channel the cockpit, the bots
and the stale-healer already read), so nothing downstream changes.

CANCEL crosses PID namespaces by file too: the app writes
jobs/cancel-<name>; this daemon — the run's parent, in its namespace — does
the exact killpg cancel always did.

Claiming is an atomic rename (jobs/x.json -> jobs/x.claimed): two daemons
racing get one winner and one FileNotFoundError, never a double launch.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..config import DATA_DIR, ROOT

JOBS_DIR = DATA_DIR / "jobs"
POLL_S = 1.0

_children: dict[str, subprocess.Popen] = {}


def spool_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def _launch(claim: Path) -> None:
    try:
        job = json.loads(claim.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[trainerd] unreadable job {claim.name}: {e}", flush=True)
        claim.unlink(missing_ok=True)
        return
    argv = job.get("argv") or []
    if not argv:
        claim.unlink(missing_ok=True)
        return
    env = dict(os.environ)
    env.update(job.get("env") or {})
    log_path = job.get("log")
    out = open(log_path, "a") if log_path else subprocess.DEVNULL
    # start_new_session: the run gets its own process GROUP, so a cancel can
    # take down the pipeline and whichever stage subprocess it is inside with
    # one killpg — same reasoning as the direct launcher.
    p = subprocess.Popen(argv, cwd=job.get("cwd") or str(ROOT), env=env,
                         stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)
    _children[job.get("name") or claim.stem] = p
    print(f"[trainerd] launched {job.get('name') or claim.stem} "
          f"(pid {p.pid}): {' '.join(map(str, argv))[:120]}", flush=True)


def _handle_cancels(d: Path) -> None:
    for c in d.glob("cancel-*"):
        name = c.name.removeprefix("cancel-")
        p = _children.get(name)
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                time.sleep(1.0)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                print(f"[trainerd] cancelled {name}", flush=True)
            except ProcessLookupError:
                pass
        c.unlink(missing_ok=True)


def loop() -> None:
    d = spool_dir()
    print(f"[trainerd] watching {d}", flush=True)
    while True:
        for f in sorted(d.glob("*.json")):
            claim = f.with_suffix(".claimed")
            try:
                f.rename(claim)              # atomic: one winner
            except FileNotFoundError:
                continue
            _launch(claim)
            claim.unlink(missing_ok=True)
        _handle_cancels(d)
        # reap finished children so they don't sit as zombies
        for name, p in list(_children.items()):
            if p.poll() is not None:
                _children.pop(name, None)
        time.sleep(POLL_S)


if __name__ == "__main__":
    loop()
