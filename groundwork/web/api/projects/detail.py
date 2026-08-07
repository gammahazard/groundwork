"""GET /api/projects/{slug} — one project's card."""

from __future__ import annotations

from .shared import (router, _counts, _best_run, _thumb,  # noqa: F401
                     _last_activity, _card, _visible)

import json
import os
import re
import time

from fastapi import APIRouter, Depends, HTTPException

from .... import config as _config
from pydantic import BaseModel

from ..deps import current_user, readable_by, why_readable
from ....config import OUTPUTS_DIR

from .... import project
from ....dataset import paths
from ....dataset.pipeline import training_history
from ....dataset.store import labelio
from ....models import metrics
from ... import lab_guard


@router.get("/api/projects/{slug}")
def get_project(slug: str, user: str | None = Depends(current_user)):
    """One project's card — OWNERSHIP-CHECKED like every other project read.

    This was the one project endpoint with no `readable_by` gate, so any
    signed-in account could read any project's name, classes, counts and best
    run. 403 (not 404) matches deps.current_project: the slug's existence is
    not the secret; its contents are.
    """
    try:
        p = project.load(slug)
    except KeyError:
        raise HTTPException(404, f"no such project {slug!r}") from None
    if not readable_by(p, user):
        raise HTTPException(403, f"{slug!r} belongs to another account")
    try:
        return _card(slug)
    except KeyError:
        raise HTTPException(404, f"no such project {slug!r}") from None
