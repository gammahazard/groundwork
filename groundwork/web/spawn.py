"""HOW a detached GPU job is started — direct fork, or the trainer-daemon spool.

GW_TRAINER=direct (default) is the native path: fork with start_new_session
and rely on the unit's KillMode=process to keep the child alive across service
restarts. GW_TRAINER=spool is the container path: a container is one cgroup,
so the app must not be the run's parent — write a job file and let
groundwork.ops.trainerd (its own container) do the forking.

Both callers (the retrain launcher and the challenger launcher) go through
here so the mode is decided once. In spool mode there is NO PID to return —
the state files' pid fields stay None, which the stale-healers already treat
as "not mine to judge", and cancellation goes by NAME through a cancel file
the daemon (the real parent) honours with the same killpg the direct path
uses.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ..config import DATA_DIR, ROOT


def spool_mode() -> bool:
    return os.environ.get("GW_TRAINER", "direct").strip() == "spool"


def _jobs_dir() -> Path:
    d = DATA_DIR / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def spawn_detached(name: str, argv: list[str], log_path: Path,
                   env: dict | None = None, cwd: Path | None = None,
                   preexec_fn=None) -> int | None:
    """Start `argv` detached. Returns the pid, or None in spool mode."""
    if spool_mode():
        job = {"name": name, "argv": [str(a) for a in argv],
               "cwd": str(cwd or ROOT), "log": str(log_path),
               "env": {k: str(v) for k, v in (env or {}).items()
                       if k.startswith(("CUDA_", "GW_"))},
               "queued": time.time()}
        d = _jobs_dir()
        tmp = d / f".{name}.tmp"
        tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        tmp.replace(d / f"{name}.json")     # atomic hand-off to trainerd
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        p = subprocess.Popen([str(a) for a in argv], cwd=str(cwd or ROOT),
                             env=env, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True, preexec_fn=preexec_fn)
    return p.pid


def request_cancel(name: str) -> bool:
    """Spool-mode cancel: ask the daemon (the run's parent) to killpg it.
    Returns False in direct mode — the caller's own pid-based kill applies."""
    if not spool_mode():
        return False
    (_jobs_dir() / f"cancel-{name}").write_text(str(time.time()),
                                                encoding="utf-8")
    return True
