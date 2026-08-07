"""The cross-process retrain state: files, locks, and who reads them."""

from __future__ import annotations
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ...config import OUTPUTS_DIR, ROOT
from ...serve.yolo_counter import find_latest_weights
from ...dataset import paths
from ...dataset.pipeline import training_history
from .. import gpu_facts, run_stats


LOG = OUTPUTS_DIR / "retrain.log"
STATE = OUTPUTS_DIR / "retrain_state.json"

# The stage modules a retrain launches as subprocesses — named ONCE.
#
# They used to be spelled out twice: as literal strings at each launch site, and
# again inside cancel()'s last-resort pkill regex. Two lists of the same six
# module paths, hundreds of lines apart, and a drift between them failed
# silently — the launches still worked while the kill pattern matched nothing.
#
# THE SECOND CONSUMER IS GONE (the pkill was deleted; see cancel()), so this is
# now read only by the launches. It stays a table rather than six literals for
# the original reason: moving a `python -m` target breaks strings no import
# checker sees, and moving these into dataset/pipeline/ is exactly the edit that
# proved it.
_STAGE = {n: f"groundwork.dataset.pipeline.{n}"
          for n in ("split", "train", "count_eval", "run_snapshot")}
_STAGE.update({n: f"groundwork.dataset.viz.{n}" for n in ("timelapse", "heatmap")})
# The pipeline's own stdout/stderr. Separate from LOG on purpose: run_pipeline
# opens LOG itself, and a crash BEFORE that (bad import, missing venv) would
# otherwise vanish with no trace of why the retrain never started.
WORKER_LOG = OUTPUTS_DIR / "retrain_worker.log"


def _lockfile(card: int | None = None):
    """Per-card admission locks. On a multi-GPU box every job is pinned to a
    card (an explicit `card` from the request, else the service's own
    CUDA_VISIBLE_DEVICES), so jobs on different cards get different lock files
    and never queue on each other. Unpinned single-card machines keep the one
    shared gpu.lock — behavior unchanged there."""
    if card is not None:
        return OUTPUTS_DIR / f"gpu{int(card)}.lock"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return OUTPUTS_DIR / (f"gpu{cvd}.lock" if cvd else "gpu.lock")


LOCKFILE = _lockfile()
# `project` is listed so the shape of a state file is readable in one place. It
# is None when idle and written by start(); see _pp() for why absent means the
# default project rather than an error.
_IDLE = {"status": "idle", "step": "", "mae": None, "prev_mae": None,
         "started": None, "finished": None, "sizes": None, "batch_results": None,
         "project": None}
_lock = threading.Lock()
_PP_CACHE: dict = {}


@contextmanager
def gpu_lock(card: int | None = None):
    """Cross-PROCESS mutex around GPU-job admission. The bot and the web UI are
    separate processes sharing only the state files, and threading.Lock is
    per-process — two near-simultaneous starts could both read "not running"
    and both train on the 8GB card. flock makes the check-then-start atomic;
    la_job takes the same lock so a probe and a retrain can't slip past each
    other's busy checks either."""
    lf = _lockfile(card)
    lf.parent.mkdir(parents=True, exist_ok=True)
    with open(lf, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _read() -> dict:
    if STATE.exists():
        try:
            return {**_IDLE, **json.loads(STATE.read_text())}
        except Exception:  # noqa: BLE001
            pass
    return dict(_IDLE)


def _write(d: dict) -> None:
    # The bot, the web service and runtime_model all read this from other processes, so
    # a torn half-write must never be observable — and the tmp must be
    # pid-scoped, because the web service and the detached pipeline are different
    # processes and a shared tmp name is not atomic between them.
    from ...dataset import sidecar
    sidecar.write_json(STATE, d)


def _update(**kw) -> None:
    _write({**_read(), **kw})


def _pp(project: str | None = None):
    """THIS JOB's project layout — not the default project's.

    Every stage subprocess has been told `--project` explicitly since the pivot
    (see `proj_arg` in run_pipeline). The ORCHESTRATOR's own bookkeeping was the
    half that still assumed the first project, and it fails in the way this repo pays
    most for — silently, looking like nothing. A second project's retrain would
    hunt for the run it had just produced in the OBJECT COUNTER's RUNS_DIR, find
    whatever was newest there, and then score it, snapshot it and write it to
    the ledger under the other project's name.

    The slug is read from the state file when not given, because everything that
    asks — status(), _progress(), cancel() — is describing a job that is already
    running and has no other way to learn whose it is. start() writes it.

    A state with no `project` predates this field and IS the default project, on
    the same principle as the ledger's un-tagged rows: every one of them was
    written when only one project existed, so that reading is not a guess.

    Cached per slug because _progress() sits on the most-polled endpoint in the
    cockpit and for_project() re-reads project.json on every call — which is the
    same reason paths.default() caches.
    """
    slug = project or _read().get("project")
    if not slug:
        raise ValueError("no project: none passed and none recorded in the "
                         "retrain state")
    if slug not in _PP_CACHE:
        _PP_CACHE[slug] = paths.for_project(slug)
    return _PP_CACHE[slug]


def _tail(n: int = 40) -> str:
    return "\n".join(LOG.read_text(errors="ignore").splitlines()[-n:]) if LOG.exists() else ""


