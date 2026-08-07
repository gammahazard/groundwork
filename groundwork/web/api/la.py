"""LocateAnything probe endpoints: run LA point-mode on one stored image as a
crosshair second opinion for the dot editor. Advisory only — the result goes to
the browser; labels change only via the editor's normal Save. Single-flight,
subprocess-isolated (see la_job), mutually exclusive with retrains."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...dataset.store import labelio
from ...dataset.paths import ProjectPaths
from .deps import project_paths
from .. import la_job
from .. import lab_guard

router = APIRouter()


class ProbeOpts(BaseModel):
    desc: str = "object"            # free-form description LA looks for


@router.post("/api/la/{collection}/{stem}")
def start(collection: str, stem: str, opts: ProbeOpts,
          pp: ProjectPaths = Depends(project_paths)):
    lab_guard.no_lab_edits()
    desc = opts.desc.strip() or "object"
    if collection not in labelio.collection_names(pp):
        return {"ok": False, "error": f"unknown collection {collection!r}"}
    if labelio.image_path(collection, stem, pp) is None:
        return {"ok": False, "error": f"no image for {stem!r} in {collection!r}"}
    return la_job.start(collection, stem, desc, pp.slug)


@router.get("/api/la")
def status():
    return la_job.status()


@router.delete("/api/la")
def cancel():
    return la_job.cancel()
