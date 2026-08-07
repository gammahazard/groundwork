"""start/cancel — admission, the detached launch, and teardown."""

from __future__ import annotations

from .state import (LOG, STATE, WORKER_LOG, _STAGE, _lockfile,  # noqa: F401
                    gpu_lock, _alive, _read, _write, _update,
                    _pp, _lock)
from .progress import status  # noqa: F401
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

def _launch_pipeline(sizes: list[int], val_frac: float,
                     epochs: int | None, batch: int | None,
                     project: str, split_seed: int = 0,
                     card: int | None = None) -> int:
    """Start the retrain as a detached process and return its pid.

    start_new_session is NOT what makes it survive a the web service restart (systemd
    kills by cgroup, which setsid does not leave — see retrain_worker). It buys
    the pipeline its own process group, so cancel() can take down the pipeline
    and whichever stage subprocess it is inside with one killpg.

    Environment and cwd are INHERITED deliberately: on the lab, the web service.service
    pins CUDA_DEVICE_ORDER + CUDA_VISIBLE_DEVICES=0 (the 5070 Ti), and
    _lockfile() derives the per-card lock from that. A systemd-run transient
    unit would have started from the user manager's environment instead and
    quietly trained on the wrong card.
    """
    cmd = [sys.executable, "-m", "groundwork.web.retrain_worker",
           "--sizes", *[str(s) for s in sizes], "--val-frac", str(val_frac)]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    if batch:
        cmd += ["--batch", str(batch)]
    if split_seed:
        cmd += ["--split-seed", str(split_seed)]
    if project:
        cmd += ["--project", project]
    env = dict(os.environ)
    if card is not None:
        # An explicit card wins over whatever the service inherited. PCI order
        # ALWAYS — CUDA's fastest-first default once landed a pin on the wrong
        # card and cost a run.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(int(card))
    from ..spawn import spawn_detached
    pid = spawn_detached("retrain", cmd, WORKER_LOG, env=env, cwd=ROOT)
    if pid is not None:
        # Reap it when it exits so a finished retrain doesn't sit as a zombie
        # for as long as the web service lives. (Spool mode has no child here —
        # the trainer daemon is the parent and does its own reaping.)
        import os as _os
        threading.Thread(target=lambda: _try_wait(pid), daemon=True).start()
    return pid or 0


def _try_wait(pid: int) -> None:
    try:
        import os as _os
        _os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _is_pipeline(pid) -> bool:
    """Does `pid` really belong to a retrain pipeline? Guards cancel()'s killpg:
    a stale pid from a state file that outlived a reboot could otherwise have
    been recycled by anything, and killpg would take out its whole group."""
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except Exception:  # noqa: BLE001
        return False
    return b"groundwork.web.retrain_worker" in cmdline


def start(imgsz: int = 960, val_frac: float = 0.2,
          sizes: list[int] | None = None,
          epochs: int | None = None,
          batch: int | None = None,
          project: str | None = None,
          split_seed: int = 0,
          card: int | None = None) -> dict:
    # sizes (e.g. [1280, 960]) queues up to 2 runs back-to-back; else a single imgsz.
    queue = [_clamp(s) for s in sizes][:2] if sizes else [_clamp(imgsz)]
    if epochs is not None:
        epochs = max(50, min(600, int(epochs)))   # sane band; None = train.py default
    with _lock, gpu_lock(card):
        if status().get("status") == "running":   # status() also clears stale states
            return {"ok": False, "error": "a retrain is already running"}
        # LA-3B nearly fills the card; checked HERE, inside the cross-process
        # lock, so a probe admitted a moment ago can't be missed. (Lazy import:
        # la_job imports this module.)
        from .. import la_job
        if la_job.status().get("status") == "running":
            return {"ok": False, "error": "a LocateAnything probe is running — "
                                          "the GPU can't fit both; retry in ~a minute"}
        prev = _read().get("mae")            # last completed run's MAE, to show improvement
        if not project:
            return {"ok": False, "error": "no project named — a retrain is "
                                          "always for a project"}
        slug = project
        # owner_pid is filled in below, once the pipeline exists. Between here
        # and then the state has none, which the stale healer treats as "legacy,
        # leave alone" — so a poll landing in that window can't false-heal.
        _write({"status": "running", "step": "starting", "mae": None, "imgsz": queue[0],
                # EXPLICIT, not _pp()'s read-from-state fallback: the state below
                # has not been written yet, so resolving from it would snapshot
                # the PREVIOUS job's project and then compare this run against it.
                "sizes": queue, "batch_results": None,
                "pre_runs": _run_dirs(_pp(slug)),
                # WHOSE RETRAIN THIS IS. Everything that later asks about a
                # running job — status(), _progress(), cancel(), and
                # runtime_model._in_progress() in another process entirely — has
                # only the state file to learn it from. Without this they all
                # answered about the first project, whatever was actually training.
                "project": slug,
                "epochs": epochs,
                # RECORDED, because a run trained on a different split is not
                # comparable to one that was not, and nothing else on disk would
                # say which it was.
                "split_seed": split_seed,
                # The card this run holds — lab_ops.held_cards() reads it
                # when deciding where a concurrent sidecar job may land.
                "card": card,
                "prev_mae": prev, "started": time.time(), "finished": None})
    try:
        pid = _launch_pipeline(queue, val_frac, epochs,
                               batch, slug, split_seed, card=card)
    except Exception as e:  # noqa: BLE001 — never leave a phantom "running"
        _update(status="error", step=f"could not start the pipeline: {e}",
                finished=time.time())
        return {"ok": False, "error": f"could not start the pipeline: {e}"}
    _update(owner_pid=pid, detached=True)
    return {"ok": True, "sizes": queue,
            "epochs": epochs, "pid": pid}


def _run_dirs(pp=None) -> list[str]:
    """This project's existing run dirs — the snapshot `pre_runs` is made from.

    TAKES the layout rather than resolving it, because start() calls this BEFORE
    writing the state, and _pp()'s read-from-state fallback would therefore
    answer with the previous job's project.
    """
    pp = pp or _pp()
    return sorted(p.name for p in pp.RUNS_DIR.glob("*") if p.is_dir()) \
        if pp.RUNS_DIR.exists() else []


def cancel() -> dict:
    """Kill a running retrain's GPU subprocesses, and DELETE the half-finished run
    dir(s) it created — so a cancelled run never shows up in Models to serve."""
    s = _read()
    if s.get("status") != "running":
        return {"ok": False, "error": "nothing running"}
    # Kill the PIPELINE first, and by process group: that is the pipeline plus
    # whichever stage subprocess it is currently inside, in one signal. Killing
    # the pipeline before its children also closes an old race — the pipeline
    # used to notice its dead stage, raise, and overwrite "cancelled" in the
    # state file with a misleading "train @1280 failed".
    pid, killed_group = s.get("owner_pid"), False
    from ..spawn import request_cancel
    if request_cancel("retrain"):
        # Spool mode: no pid was ever ours — the trainer daemon parents the
        # pipeline and performs the exact killpg below in its own namespace.
        killed_group = True
    if not killed_group and pid and _alive(pid) and _is_pipeline(pid):
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            killed_group = True          # stages die with the group; done here
        except OSError:
            pass
    if not killed_group:
        # THE BLIND pkill IS GONE, and this refuses instead.
        #
        # It was `pkill -9 -f _STAGE_KILL_RE` — a pattern matching all six stage
        # module paths with NO way to tell whose job a match belongs to. It
        # already killed an unrelated live retrain at epoch 186/250 on
        # 2026-07-30, and it is the same species as `pgrep -f` matching its own
        # command line: pattern-based process matching cannot express ownership.
        #
        # It only ever fires when precise targeting failed — owner_pid absent,
        # dead, or not a pipeline. Every one of those means WE DO NOT KNOW which
        # processes are ours, and the honest response to that is to say so, not
        # to shoot everything that looks similar. The moment a second retrain can
        # exist the shotgun stops being a last resort and becomes a way to kill
        # somebody else's run from a button labelled Cancel.
        #
        # WHETHER TO MARK THE JOB DEAD IS A SEPARATE QUESTION, and getting it
        # wrong is fail-OPEN. altmodels/gpu.py admits a sidecar when this file
        # says anything but "running" (gpu.py::_busy_reason), and start() checks
        # the same field — so flipping to "error" here releases every guard on
        # the card. Do that while an unidentified process is still training and
        # two jobs land on one GPU, which is the failure this box has already
        # paid for three times.
        #
        # The three sub-cases are NOT the same, so they are not treated alike:
        if pid and (not _alive(pid) or not _is_pipeline(pid)):
            # We know our pipeline is gone — either the pid is dead, or it has
            # been recycled by something that is not a retrain, which means the
            # process we recorded no longer exists. Nothing to kill, and the job
            # is genuinely over, so marking it is correct and releasing the
            # card is safe. Fall through to the run-dir prune below.
            killed_group = True
        else:
            # owner_pid is MISSING — a state written before the pipeline was
            # detached. A stage may well still be running and we cannot name it.
            # Leave the state exactly as it is: a stuck "running" costs a manual
            # fix, a false "error" costs a run. status()'s stale healer clears
            # it by itself the moment the owner pid dies.
            return {"ok": False, "error":
                    "this retrain has no recorded process id, so nothing could "
                    "be identified to stop — and killing by name would match "
                    "every training stage on this machine, including another "
                    "project's. The job is left RUNNING rather than marked "
                    "failed, because saying it stopped would let a second job "
                    "onto the same card. End the stage by pid, or delete "
                    "outputs/retrain_state.json once you are sure it is idle."}
    pre = _read().get("pre_runs")            # runs that existed BEFORE this retrain
    if pre:                                  # guard: only prune if we have a snapshot
        pre = set(pre)
        # The cancelled run is in ITS project's tree. Pruning the default
        # project's instead would delete nothing it meant to and could delete
        # something it did not — every the first project run absent from a second
        # project's pre_runs snapshot looks half-trained from here.
        for p in _pp(s.get("project")).RUNS_DIR.glob("*"):
            # A count_eval.json means a FINISHED run — e.g. a lab run the
            # adoption timer landed mid-retrain (not in pre_runs). Never
            # cancel's to delete: only half-trained dirs this retrain made.
            if p.is_dir() and p.name not in pre \
                    and not (p / "count_eval.json").exists():
                shutil.rmtree(p, ignore_errors=True)
    _update(status="error", step="cancelled", finished=time.time())
    return {"ok": True}
