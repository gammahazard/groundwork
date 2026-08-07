"""Dashboard state: dataset counts, current model, retrain status."""
from __future__ import annotations
import json
import os
import re
import subprocess

from .. import safe_proc
import time
import urllib.request

from fastapi import APIRouter, Depends, HTTPException

from ... import config as _config

from ...dataset.pipeline import adopt_run
from ...dataset.store import labelio, collect
from ...dataset import paths as dpaths
from ...dataset.paths import ProjectPaths
from ...serve.yolo_counter import DEFAULT_CONF, DEFAULT_IOU
from ...serve import runtime_model
from .. import retrain_job
from .deps import project_paths

router = APIRouter()


@router.get("/api/state")
def state(pp: ProjectPaths = Depends(project_paths)) -> dict:
    c = collect.counts(pp)
    raw = labelio.list_images("raw", pp)
    testset = labelio.list_images("testset", pp)
    served_w, served_imgsz, served_conf, served_iou = runtime_model.resolve(pp)
    # The engine switch decides which ARCHITECTURE answers a photo
    # (runtime_model.make_counter). This block described the pinned YOLO run
    # regardless, so with the challenger serving, the hero card named a model
    # that was not counting anything — under a line claiming the bot and the
    # phone use it. Describe whichever engine is actually live.
    engine = runtime_model.get_engine(pp)
    run_dir = served_w.parent.parent if served_w else None
    if engine == "deim":
        from ...deim_counter import DEIM_CONF, DEIM_IOU   # local: keeps /api/state light
        dw = runtime_model.deim_weights(pp)
        # <alt>/<run>/train/detector.torchscript.pt -> <alt>/<run>
        run_dir = dw.parent.parent if dw else None
        served_conf, served_iou = DEIM_CONF, DEIM_IOU
    served_mae, served_exact = None, None
    try:   # the served run's holdout MAE + perfect-image rate (the hero stats)
        if run_dir:
            ce = json.loads((run_dir / "count_eval.json").read_text())
            served_mae = ce.get("mae")
            t = ce.get("images") or []
            if t:
                served_exact = [sum(1 for x in t if x["pred"] == x["gt"]), len(t)]
            if engine == "deim":
                # DeimCounter reads its tuned pair from this same file, with the
                # same fallbacks — keep the card and the counter on one source.
                served_conf = float(ce.get("conf", served_conf))
                served_iou = float(ce.get("iou", served_iou))
                served_imgsz = int(ce.get("imgsz") or served_imgsz)
    except Exception:  # noqa: BLE001
        pass
    return {
        "project": pp.slug,
        "classes": list(pp.CLASS_NAMES),
        "extensions": list(pp.extensions),
        # Which collections this project HAS — the UI hides the tabs it does
        # not, and hiding is driven by data rather than a hardcoded list.
        "collections": list(labelio.collection_names(pp)),
        "raw": len(raw),
        "pending": c["pending"],
        "needs_fix": c["needs_fix"],
        "testset": len(testset),
        # OBJECTS — the count is a sum of label rows.
        "objects": sum(t["count"] for t in raw),
        "served": {"name": run_dir.name if run_dir else None,
                   "imgsz": served_imgsz, "mae": served_mae,
                   "exact": served_exact,
                   "engine": engine,
                   "experimental": engine != "yolo",
                   "pinned": runtime_model.get_active(pp) is not None},
        "odometer": _odometer(max((t["count"] for t in raw + testset), default=0)),
        # show the served run's tuned conf/iou (falls back to the defaults)
        "conf": served_conf if served_conf is not None else DEFAULT_CONF,
        "iou": served_iou if served_iou is not None else DEFAULT_IOU,
        "retrain": retrain_job.status(),
        # Worker identity: set GW_ROLE=worker in that machine's service so its
        # cockpit announces itself and the two UIs can never be confused.
        "lab": _config.is_worker(),
        "last_sync": _last_sync(pp),
        "mirror": _mirror_health(),
    }


def _challenger_mileage() -> dict:
    """Challenger runs trained ON THIS MACHINE, from each run's meta.json.

    The yolo ledger only ever counted yolo, so on the Trainer — where the
    challenger programme actually lives — the odometer was hiding most of the
    worker's work (measured 2026-07-29: 3.8 yolo hours shown, 20.1 challenger
    hours invisible).

    Attribution: training artifacts (a per-epoch log, or checkpoints) stay on
    the machine that trained the run — the adopt rsync and results sync both
    exclude train/. Their presence is the honest test for "did THIS box burn
    these hours?"; without it HQ would report the lab's GPU time as its own.
    A bare train/ dir is NOT enough: a serving export gets copied home into one
    (deimv2-n-1280-k's TorchScript detector did exactly that, and was briefly
    credited to HQ), so require a log or a real checkpoint.
    """
    runs = hours = epochs = 0
    by_arch: dict[str, float] = {}
    # EVERY project's alt tree, deliberately. "How many GPU hours has this box
    # burned" is a machine question, and the yolo half of the odometer is
    # already machine-wide (one ledger for every project, by design — see
    # training_history's header). Reading only the default project's tree would
    # under-count the moment a second project trains anything, which is the same
    # invisible-mileage bug that motivated counting challengers at all.
    for meta in sorted(m for pp in dpaths.every_project()
                       for m in pp.ALT_DIR.glob("*/meta.json")):
        t = meta.parent / "train"
        if meta.parent.name == "datasets" or not t.is_dir():
            continue                      # results-only copy: not our mileage
        trained_here = ((t / "log.txt").exists() or (t / "log.json").exists()
                        or any(t.glob("*.pth")) or any(t.glob("*/*.log")))
        if not trained_here:
            continue                      # export artifacts only, not training
        try:
            m = json.loads(meta.read_text())
        except Exception:  # noqa: BLE001
            continue
        secs = m.get("train_seconds") or 0
        if not secs:
            continue                      # an exam-only dir, not a training run
        runs += 1
        hours += secs / 3600
        epochs += m.get("epochs") or 0
        arch = m.get("arch") or "challenger"
        by_arch[arch] = round(by_arch.get(arch, 0) + secs / 3600, 2)
    return {"runs": runs, "hours": round(hours, 1), "epochs": epochs,
            "by_arch": dict(sorted(by_arch.items(), key=lambda kv: -kv[1]))}


def _remote_challenger_mileage() -> dict | None:
    """The OTHER machine's challenger hours, or None if we could not ask.

    THE ODOMETER WAS NEVER "THIS MACHINE", AND THE OWNER WANTS THE FLEET (2026-08-02).
    Measured that day: HQ showed 23.0h labelled "this machine" while the worker had
    burned 72.6h, and 11.9h of HQ's own figure was worker-trained yolo it had adopted.

    The yolo half needs no fixing for a fleet total: every one of the worker's 17
    ledger rows already appears in HQ's 35 by the same run name, because adoption
    writes the row home. HQ's ledger IS the union, so yolo hours must NOT be added
    across machines — that double-counts 16.3h. Only the CHALLENGER half is
    machine-local, because _challenger_mileage attributes by training artifacts
    that never leave the box that trained.

    Through lab_proxy: /api/state is polled, and a blocking cross-network call on
    a polled endpoint is this repo's most repeated service bug. Cached and
    refreshed off-thread, so a slow worker costs nothing here.

    None means "could not ask", and the caller must SAY so rather than quietly
    reporting a smaller total — today a cached-but-stale reading was mistaken for
    a live one for six minutes, which is the same failure in a different hat.
    """
    from ...config import is_worker
    if is_worker():
        return None                     # a worker does not phone itself
    try:
        from .. import lab_proxy
        wk = lab_proxy.first_worker_key()
        if wk is None:
            return None                 # no workers registered — nothing to ask
        d, _age = lab_proxy.get(wk, "/api/state")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(d, dict):
        return None
    ch = (d.get("odometer") or {}).get("challenger")
    return ch if isinstance(ch, dict) else None


def _odometer(biggest: int) -> dict:
    """Lifetime tallies — yolo generations from the ledger PLUS challenger runs
    (see _challenger_mileage), epochs, biggest capture, and the FLEET total across
    both machines (see _remote_challenger_mileage for why only challengers add)."""
    ch = _challenger_mileage()
    try:
        from ...dataset.pipeline import training_history
        from ...config import KWH_RATE, TRAIN_WATTS
        recs = training_history.load()
        yolo_h = sum(r.get("duration_s") or 0 for r in recs) / 3600
        yolo_ep = sum(r.get("epochs_trained") or 0 for r in recs)
        hours = yolo_h + ch["hours"]
        # THE FLEET, because HQ is becoming the only control plane (owner,
        # 2026-08-02 — the Trainer UI is to be retired and everything driven from
        # here). YOLO HOURS ARE NOT ADDED ACROSS MACHINES: adoption writes every
        # lab run into this ledger, so all 17 of the worker's rows are already among
        # these 35 by name and summing both would double-count 16.3h. Only the
        # challenger halves add, because those attribute by training artifacts
        # that never leave the box that trained.
        remote_ch = _remote_challenger_mileage()
        fleet_h = hours + (remote_ch or {}).get("hours", 0)
        fleet = {"hours": round(fleet_h, 1),
                 # NOT SILENTLY SMALLER. An unreachable worker means this total is
                 # missing its largest component (56.3h of 79.3h when measured),
                 # and a number that quietly shrinks is worse than one that says
                 # it is incomplete.
                 "complete": remote_ch is not None,
                 "remote_hours": round((remote_ch or {}).get("hours", 0), 1),
                 "remote_runs": (remote_ch or {}).get("runs", 0)}
        # Estimated, not measured (see config.py for why) — whole-system watts
        # x hours x rate. Flagged so the UI can never imply a meter reading.
        # Charged on the FLEET when we have it: the electricity was spent whoever
        # owns the card.
        kwh = TRAIN_WATTS * (fleet_h if fleet["complete"] else hours) / 1000
        return {"runs": len(recs),
                "gpu_hours": round(yolo_h, 1),
                "epochs": yolo_ep,
                "challenger": ch,
                "fleet": fleet,
                "total_runs": len(recs) + ch["runs"],
                "total_hours": round(hours, 1),
                "total_epochs": yolo_ep + ch["epochs"],
                "power": {"kwh": round(kwh, 1), "usd": round(kwh * KWH_RATE, 2),
                          "rate": KWH_RATE, "watts": TRAIN_WATTS,
                          "measured": False},
                "biggest": biggest}
    except Exception:  # noqa: BLE001
        return {"biggest": biggest, "challenger": ch}


def _last_sync(pp: ProjectPaths) -> int | None:
    """Epoch of the last HQ->worker dataset sync (the mirror stamps it).

    Per project AND per machine: the stamp is `.last_sync.<machine>` inside the
    dataset, so one healthy path cannot mask a broken one. This dashboard
    figure is the NEWEST stamp — "has anything received this dataset lately" —
    and the Machines tab is where per-worker freshness lives.
    """
    stamps = []
    for f in pp.DATASET_DIR.glob(".last_sync*"):
        try:
            stamps.append(int(f.read_text().strip()))
        except Exception:  # noqa: BLE001
            pass
    return max(stamps) if stamps else None


_TS_AUTH = re.compile(r"https://login\.tailscale\.com/a/[a-z0-9]+")


def _mirror_health() -> dict:
    """Why the mirror is failing, in words the cockpit can show.

    READ FROM THE SCHEDULER'S OWN RECORD, not from systemctl/journalctl: the
    scheduler (groundwork.ops.scheduler) stamps every job's last outcome into
    outputs/jobs_status.json — {job: {last_ok, last_error, running, at}} — so
    this works identically under systemd timers, the Docker scheduler service
    and the in-process fallback. One special case is preserved: a VPN SSH
    re-auth demand buried in the error text becomes a tappable link, because
    a headless job can never satisfy it and a human can, in one tap.
    HQ ONLY (workers receive; they don't run the mirror).
    """
    if _config.is_worker():
        return {}
    try:
        d = json.loads((OUTPUTS_DIR / "jobs_status.json").read_text())
    except Exception:  # noqa: BLE001 — no scheduler record yet is a fresh box
        return {}
    j = (d or {}).get("mirror") or {}
    if not j:
        return {}
    if not j.get("last_error"):
        return {"ok": True}
    out = {"ok": False, "error": str(j["last_error"])[:200]}
    if m := _TS_AUTH.findall(str(j["last_error"])):
        out["auth_url"] = m[-1]          # newest link wins; older ones expire
    return out


@router.get("/api/lab_status")
def lab_status(pp: ProjectPaths = Depends(project_paths)) -> dict:
    """HOME ONLY: peek at the Trainer's retrain over the network, so HQ's
    dashboard can say "lab is cooking / verdict waiting to adopt" without
    anyone opening the Trainer UI. Read-only; an unreachable lab is not an
    error — the card just doesn't render."""
    from ...config import is_worker
    if is_worker():
        raise HTTPException(400, "a worker doesn't watch itself — HQ only")
    # Cached + background-refreshed: this endpoint is polled every ~8s and a
    # cold network path took 3s inline, which made the whole HQ cockpit crawl.
    from .. import lab_proxy
    wk = lab_proxy.first_worker_key()
    if wk is None:
        return {"ok": False, "no_workers": True}
    d, age = lab_proxy.get(wk, "/api/retrain")
    if d is None:
        return {"ok": False}
    p = d.get("progress") or {}
    # THE WHOLE progress BLOCK, so 🔬 Watch training works for a run on the worker.
    # peek.js renders from `progress` and builds every image URL off
    # progress.peek.base — so absolutising that one string to the worker's /outputs
    # mount is the entire port. Same move /api/lab/vis makes for its images, and
    # the same class of bug as run_log/run_vis: the panel followed the MACHINE
    # instead of the run, so it vanished the day training moved to the lab.
    peek_prog = p
    pk = p.get("peek") or {}
    base = pk.get("base")
    if base and not str(base).startswith("http"):
        from .. import machines as machines_mod
        wm = machines_mod.get(wk)
        peek_prog = {**p, "peek": {**pk, "base": (wm.url if wm else "") + base}}
    fin = d.get("finished")
    # ADOPTION IS RECORDED IN THE LEDGER. ASK THE LEDGER.
    #
    # This was `any(q.stat().st_mtime > fin for q in pp.RUNS_DIR.glob("yolov8n-*"))`
    # — a directory timestamp standing in for a fact, and it could essentially
    # never be true of the run it was asking about. adopt_run copies the run with
    # `rsync -a`, which PRESERVES mtimes, so an adopted directory arrives carrying
    # the WORKER's timestamps; and the worker writes its `finished` marker after the
    # directory is complete, so the copy is always slightly OLDER than `fin`.
    # Measured 2026-08-02: yolov8n-59's dir read 11:56:26 against fin 11:56:43 —
    # 17 seconds the wrong way. The card therefore said "verdict in, adopting…"
    # for the full 15-minute `recent` window after EVERY lab run, including ones
    # that had been fully adopted minutes earlier (that one was in the ledger,
    # with its MAE, at 11:57:29).
    #
    # A ledger row IS the adoption — it is what adopt_run writes, and it carries
    # both the project and a locally-generated `finished`, so this keeps the
    # per-project scoping the old test had (a second project's dashboard must not
    # report adoption because the first project adopted something) without depending on a
    # timestamp that rsync deliberately copies.
    from ...dataset.pipeline import training_history
    try:
        rows = training_history.load()
    except Exception:  # noqa: BLE001 — an unreadable ledger is "cannot tell"
        rows = []
    adopted = bool(fin) and any(
        r.get("project") == pp.slug and (r.get("finished") or 0) >= fin
        for r in rows)
    return {"ok": True, "age_s": age,
            "status": d.get("status"), "step": d.get("step"),
            "imgsz": d.get("imgsz"), "finished": fin, "adopted": adopted,
            "progress": peek_prog,
            "epoch": p.get("epoch"), "total": p.get("total"),
            "eta_s": p.get("eta_total_s") or p.get("eta_s"),
            # THE WORKER'S OWN LOG TAIL, passed straight through. lab_proxy already
            # fetches the whole /api/retrain payload for the fields above, so
            # this costs nothing extra — no second call to a machine that is
            # deliberately busy, which is the pattern that has wedged this
            # cockpit three times. It arrives in ~8s steps and can be up to
            # ~20s stale during heavy epochs; `age_s` above says how stale, and
            # the UI shows it rather than implying live.
            "log_tail": d.get("log_tail"),
            "next_name": adopt_run._next_local_name(pp)}


