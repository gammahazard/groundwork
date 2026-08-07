"""Live progress + status(), parsed from the in-flight run's artifacts."""

from __future__ import annotations

from .state import (_tail, _write,  # noqa: F401
                    LOG, STATE, _read, _pp, _alive, _update,  # noqa: F401
                    _IDLE, _lock)
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

def _progress() -> dict | None:
    """Live training progress for the dashboard, parsed from the in-progress
    run's results.csv (ultralytics appends one row per epoch). Runs are fixed
    length (250, no early stop — deliberate, for comparability), so ETA is
    just recent pace x remaining epochs. None when not mid-training."""
    s = _read()
    if s.get("status") != "running":
        return None
    pre = set(s.get("pre_runs") or [])
    pp = _pp(s.get("project"))
    cooking = [p for p in pp.RUNS_DIR.glob("*")
               if p.is_dir() and p.name not in pre]
    if not cooking:
        return None
    run = max(cooking, key=lambda p: p.stat().st_mtime)
    out = run_stats.stats(run)
    # Universal signals, same meaning as the challenger rows: filesystem and
    # /proc facts that work whatever the trainer prints.
    try:
        out["log_age_s"] = max(0, int(time.time() - LOG.stat().st_mtime))
    except OSError:
        pass
    pid = s.get("owner_pid")
    if pid:
        from ...web.api.lab_progress import _cpu_ticks
        out["ticks"] = _cpu_ticks(pid)
    if s.get("started"):
        out["elapsed_s"] = int(time.time() - s["started"])
    times = out.pop("times", [])
    done, total = out["epoch"], out["total"]
    if times and done:
        # recent pace beats the overall average once caching/GPU warm up
        per = (times[-1] - times[-11]) / 10 if done > 10 else times[-1] / done
        out["epoch_s"] = round(per, 1)
        if total:
            out["eta_s"] = int(per * (total - done))
            out["eta_total_s"] = out["eta_s"] + run_stats.eval_overhead_s()
    # Trajectory vs the previous run AT THE SAME EPOCH — an honest leading
    # indicator (val mAP; the real judgment is still the holdout MAE at the
    # end). "similar" band = ±0.01 mAP over the last-10-epoch means.
    prev = run_stats.prev_completed(pre, pp)
    if prev is not None:
        pmaps = [e["map50"] for e in training_history.get_curve(prev.name, pp)
                 if e.get("map50") is not None]
        if pmaps:
            out["prev_run"] = prev.name
            out["prev_map50_curve"] = pmaps
            cur = out.get("map50_curve") or []
            n = min(len(cur), len(pmaps))
            if n >= 5:
                k = min(10, n)
                d = sum(cur[n - k:n]) / k - sum(pmaps[n - k:n]) / k
                out["vs_prev"] = {
                    "run": prev.name, "delta": round(d, 4), "epoch": n,
                    "verdict": "ahead" if d > 0.01 else
                               "behind" if d < -0.01 else "similar"}
    return out




def status() -> dict:
    s = _read()
    # A "running" state whose pipeline process is gone (crashed, killed, or OOM)
    # would block retrains AND LA probes forever. start() records the pipeline's
    # owner_pid; states written by older code lack it and are left untouched.
    if s.get("status") == "running" and s.get("owner_pid") \
            and not _alive(s.get("owner_pid")):
        s.update(status="error", step="interrupted (owner process died)",
                 finished=time.time())
        _write(s)
    # THE GPU FIELD IS A FILESYSTEM READ, AND MUST STAY ONE. This used to be
    # `run_stats.gpu()` — an nvidia-smi subprocess — gated on status == running.
    # /api/retrain is the most-polled endpoint in the cockpit (every browser on
    # the dashboard, plus HQ's lab_proxy every 8s), so that gate fired the call
    # continuously during exactly the window when the GPU is busy and the WSL2
    # driver call wedges in unkillable D-state. Each wedge permanently consumed
    # one of anyio's 40 worker threads; five minutes into a run the worker served
    # nothing sync at all. See gpu_facts for the full measurement.
    tail = _tail()
    return {**s, "log_tail": tail, "progress": _progress(),
            "gpu": gpu_facts.busy(s.get("status") == "running", tail)}


