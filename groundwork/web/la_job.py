"""Single-flight background LocateAnything probe (crosshair second opinion).

Emergency/rescue tool for the dot editor: when the YOLO counter messes up a
image (huge captures especially), run LA-3B point mode on that one image and hand
the points to the browser as an advisory overlay. Mirrors retrain_job's
state-file pattern, but the model runs in a FRESH subprocess that exits when
done — the only safe way to use the 3B on the 8GB card (VRAM fully released,
same discipline as scratch/label_new.sh). Slow (~30-90s) by design; rare use.

Never writes labels: the result JSON goes to the editor, which decides whether
to adopt the points, and persistence stays on the normal Save path.
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import threading
import time

from ..config import OUTPUTS_DIR
from . import retrain_job

LOG = OUTPUTS_DIR / "la.log"
STATE = OUTPUTS_DIR / "la_state.json"
RESULT = OUTPUTS_DIR / "la_result.json"
# Proven-stable envelope for batch LA on this 8GB card (see label_new.sh).
_PATCH_CEILING = "2200"
_IDLE = {"status": "idle", "collection": None, "stem": None, "desc": None,
         "pid": None, "started": None, "finished": None, "error": None}
_lock = threading.Lock()


def _read() -> dict:
    if STATE.exists():
        try:
            return {**_IDLE, **json.loads(STATE.read_text())}
        except Exception:  # noqa: BLE001
            pass
    return dict(_IDLE)


def _write(d: dict) -> None:
    # This file is a GPU-ADMISSION gate read by other processes (retrain_job's
    # start() check, altmodels/gpu.py). A torn read makes _read() report IDLE,
    # and a retrain gets admitted while a LocateAnything probe still holds the
    # card — which nearly fills it. Both halves matter: atomic replace AND a
    # pid-scoped tmp, since a shared tmp name is not atomic between processes.
    from ..dataset import sidecar
    sidecar.write_json(STATE, d)


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def status() -> dict:
    s = _read()
    # A "running" state whose process is gone (e.g. the web service restarted mid-probe)
    # is stale — surface it as an error instead of blocking forever.
    if s["status"] == "running" and not _alive(s.get("pid")):
        s.update(status="error", error="probe process died (interrupted?)",
                 finished=time.time())
        _write(s)
    if s["status"] == "done" and RESULT.exists():
        try:
            s["result"] = json.loads(RESULT.read_text())
        except Exception:  # noqa: BLE001
            s.update(status="error", error="unreadable result")
    tail = LOG.read_text(errors="ignore").splitlines()[-12:] if LOG.exists() else []
    s["log_tail"] = "\n".join(tail)
    return s


def running() -> bool:
    return status()["status"] == "running"


def start(collection: str, stem: str, desc: str) -> dict:
    # gpu_lock: cross-process admission mutex shared with retrain_job, so a
    # probe and a retrain can't both pass the other's busy check concurrently.
    with _lock, retrain_job.gpu_lock():
        if _read()["status"] == "running" and _alive(_read().get("pid")):
            return {"ok": False, "error": "a LocateAnything probe is already running"}
        if retrain_job.status().get("status") == "running":
            return {"ok": False, "error": "a retrain is running — the GPU can't fit "
                                          "both; try again after it finishes"}
        RESULT.unlink(missing_ok=True)
        log = open(LOG, "w")
        cmd = [sys.executable, "-m", "groundwork.dataset.teacher.la_probe",
               "--collection", collection, "--stem", stem,
               "--desc", desc, "--out", str(RESULT)]
        log.write(f"$ {' '.join(cmd)}\n"); log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ,
                                     "LA_SAFE_PATCH_CEILING": _PATCH_CEILING})
        _write({**_IDLE, "status": "running", "collection": collection,
                "stem": stem, "desc": desc, "pid": proc.pid,
                "started": time.time()})

    def watch() -> None:
        rc = proc.wait()
        log.close()
        s = _read()
        if s.get("pid") != proc.pid:      # cancelled/superseded; don't clobber
            return
        if rc == 0 and RESULT.exists():
            _write({**s, "status": "done", "finished": time.time()})
        else:
            _write({**s, "status": "error", "finished": time.time(),
                    "error": f"probe exited {rc} — see log"})

    threading.Thread(target=watch, daemon=True).start()
    return {"ok": True, "collection": collection, "stem": stem, "desc": desc}


def cancel() -> dict:
    s = _read()
    if s["status"] != "running":
        return {"ok": False, "error": "nothing running"}
    if _alive(s.get("pid")):
        # SIGKILL, not SIGTERM: the point is to release VRAM immediately.
        os.kill(int(s["pid"]), signal.SIGKILL)
    # pid=None so the watcher thread sees it's been superseded and doesn't
    # clobber "cancelled" with the killed process's exit-code message.
    _write({**s, "status": "error", "error": "cancelled", "pid": None,
            "finished": time.time()})
    return {"ok": True}
