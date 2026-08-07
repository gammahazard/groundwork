"""Dataset-snapshot endpoints for the Runs tab: what a run trained/tested on,
what has changed since, and single-image restore from the blob store."""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException

from ...dataset.pipeline import run_snapshot
from ...dataset.paths import ProjectPaths
from .images import _holdout_locked
from .. import lab_guard
from .deps import project_paths

router = APIRouter()

_RUN_OK = re.compile(r"^[A-Za-z0-9._-]+$")
_STEM_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def _check(name: str, pat: re.Pattern) -> str:
    if not pat.match(name):
        raise HTTPException(422, "bad name")
    return name


@router.get("/api/runs/{run}/snapshot")
def snapshot(run: str, pp: ProjectPaths = Depends(project_paths)):
    """Snapshot summary + diff vs the current dataset."""
    d = run_snapshot.diff_vs_current(_check(run, _RUN_OK), pp)
    if d is None:
        raise HTTPException(404, "no dataset snapshot for this run "
                                 "(snapshots start with runs after the feature shipped)")
    return d


@router.post("/api/runs/{run}/restore/{stem}")
def restore(run: str, stem: str, pp: ProjectPaths = Depends(project_paths)):
    """Restore one image byte-exact as the snapshot recorded it. Never
    overwrites an existing image; holdout restores respect the mid-exam lock."""
    lab_guard.no_lab_edits()
    _check(run, _RUN_OK)
    _check(stem, _STEM_OK)
    m = run_snapshot.load(run, pp)
    if m is None:
        raise HTTPException(404, "no snapshot for this run")
    in_testset = stem in m["collections"].get("testset", {})
    if in_testset and (msg := _holdout_locked()):
        return {"ok": False, "error": msg}
    try:
        return run_snapshot.restore_stem(run, stem, pp=pp)
    except FileExistsError as e:
        return {"ok": False, "error": str(e)}
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
