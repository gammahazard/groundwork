"""Retrain endpoints + active-model selection (pin a run / revert)."""
from __future__ import annotations
import json

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import retrain_job, la_job
from ...serve import runtime_model

from ...dataset.paths import ProjectPaths
from .deps import current_project, project_paths

router = APIRouter()


class RetrainOpts(BaseModel):
    imgsz: int | None = None       # single-run size (required unless `sizes` given)
    sizes: list[int] | None = None  # queue up to 2 back-to-back, e.g. [1280, 960]
    epochs: int | None = None      # None = train.py default (250); 400 = long experiment
    batch: int | None = None       # None = 8GB-safe defaults; Trainer sends 8/12
    # THE TRAIN/VAL MEMBERSHIP SEED. 0 is what every run in the ledger used, so a
    # non-zero value is a DIFFERENT experiment rather than a repeat — see
    # retrain_job.run_pipeline. Exposed because the five champions re-scored on
    # 2026-08-02 all hit 0.0 while sharing one fixed split, so "does it survive a
    # different 139/35" is the question they cannot answer.
    split_seed: int = 0
    # WHICH CARD, PCI order. None = the service's own device (single-card
    # machines and the historical default). An explicit card travels into the
    # detached pipeline's CUDA_VISIBLE_DEVICES and its per-card lock.
    card: int | None = None


class ActivateOpts(BaseModel):
    weights: str
    imgsz: int


@router.post("/api/retrain")
def start(opts: RetrainOpts, p=Depends(current_project)):
    if not opts.sizes and opts.imgsz is None:
        return {"ok": False, "error": "specify imgsz or sizes"}
    # The 3B probe nearly fills the 8GB card — a retrain on top would spill.
    # (Guarded here, web-side only, so bot-shared retrain_job stays untouched.)
    if la_job.running():
        return {"ok": False, "error": "a LocateAnything probe is running — "
                                      "wait ~a minute and retry"}
    # NOTHING TO TRAIN ON is a refusal, not a failed run. Without this the
    # launch "succeeded", a detached pipeline started, split exited, and the
    # cockpit showed a failed run whose reason was only in the log — the first
    # thing a new install does is press Train, and this was its answer.
    # The floor is TWO, not one: select_val() always reserves at least one
    # image for validation (max(1, …)), so a single labelled image launches a
    # pipeline whose training side is empty — measured: ultralytics dies with
    # a FileNotFoundError on images/train, a crash where a refusal belongs.
    from ...dataset.pipeline.split import _stems_with_labels
    from ...dataset import paths as _paths
    n = len(_stems_with_labels(_paths.for_project(p)))
    if n == 0:
        return {"ok": False, "error":
                "no labelled images yet — open Images ▸ Fix queue, mark the "
                "objects in a few, and press Done → training. Training needs "
                "at least two labelled images (the split reserves one for "
                "validation)."}
    if n == 1:
        return {"ok": False, "error":
                "only one labelled image — the split always reserves one for "
                "validation, which would leave the training side empty. "
                "Label at least one more."}
    return retrain_job.start(imgsz=opts.imgsz or 960, sizes=opts.sizes,
                             epochs=opts.epochs, batch=opts.batch,
                             project=p.slug,
                             split_seed=opts.split_seed, card=opts.card)


@router.get("/api/models")
def models(pp: ProjectPaths = Depends(project_paths)):
    """This project's trained runs (newest first) + which one it serves."""
    return {"runs": runtime_model.list_runs(pp),
            "active": runtime_model.get_active(pp), "project": pp.slug}


@router.post("/api/model/activate")
def activate(opts: ActivateOpts, pp: ProjectPaths = Depends(project_paths)):
    """Pin the run THIS PROJECT serves (takes effect on the next count).

    THE WEIGHTS PATH IS CHECKED AGAINST THIS PROJECT'S RUNS. It arrives in the
    request body as a string, and without this a project could be pinned to
    another project's checkpoint — which is the machine-global bug this commit
    removes, re-introduced through the front door and with a worse blast radius,
    because the pin persists and the bot reads it too.
    """
    w = Path(opts.weights).resolve()
    if pp.RUNS_DIR.resolve() not in w.parents:
        raise HTTPException(422, f"{opts.weights} is not a run of {pp.slug}")
    runtime_model.set_active(w, opts.imgsz, pp)
    return {"ok": True, "active": runtime_model.get_active(pp)}


@router.delete("/api/model/activate")
def deactivate(pp: ProjectPaths = Depends(project_paths)):
    """Unpin — go back to serving the newest run."""
    runtime_model.clear(pp)
    return {"ok": True, "active": None}


@router.get("/api/retrain")
def status():
    return retrain_job.status()


@router.delete("/api/retrain")
def cancel():
    return retrain_job.cancel()


class EngineOpts(BaseModel):
    engine: str


@router.get("/api/engine")
def engine(pp: ProjectPaths = Depends(project_paths)):
    """Which architecture answers photos FOR THIS PROJECT, and what else could.
    The challenger only appears as available when this project actually has an
    exported detector — no phantom option that would fail on click."""
    cur = runtime_model.get_engine(pp)
    deim = runtime_model.deim_weights(pp)
    opts = [{"engine": "yolo", "label": "YOLOv8n", "license": "AGPL-3.0",
             "available": True}]
    if deim is not None:
        run = deim.parent.parent
        mae = None
        try:
            mae = json.loads((run / "count_eval.json").read_text()).get("mae")
        except Exception:  # noqa: BLE001
            pass
        opts.append({"engine": "deim", "label": run.name, "license": "Apache-2.0",
                     "available": True, "mae": mae})
    return {"engine": cur, "options": opts}


@router.post("/api/engine")
def set_engine(opts: EngineOpts, pp: ProjectPaths = Depends(project_paths)):
    """Swap the serving architecture. The yolo pin is untouched either way, so
    flipping back is instant (takes effect on the next count; the bot reloads
    on /restart)."""
    try:
        return {"ok": True, "engine": runtime_model.set_engine(opts.engine, pp)}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
