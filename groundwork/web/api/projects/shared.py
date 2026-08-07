"""Per-project card facts: counts, best run, thumb, visibility."""

from __future__ import annotations

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


router = APIRouter()


def _counts(pp) -> dict:
    """Images per collection + total labelled objects, for one project.

    IMAGES and OBJECTS — one .txt file per image, one line per object.

    Only the collections this project HAS — `collection_names` drops the
    ones behind an extension it does not enable.
    """
    out, objects = {}, 0
    for name in labelio.collection_names(pp):
        _, lbl_dir = labelio._dirs(name, pp)
        stems = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        out[name] = len(stems)
        if name in ("raw", "testset"):        # the two that are really the dataset
            for f in stems:
                objects += sum(1 for ln in f.read_text(errors="ignore").splitlines()
                               if ln.strip())
    return {"collections": out, "objects": objects}


def _best_run(slug: str) -> dict | None:
    """The best COMPLETED run for this project, by its headline metric.

    THE TIE-BREAK IS LOAD-BEARING, because the holdout is saturated: five runs
    score a perfect 0.0 and a plain `min()` returns whichever the ledger happens
    to list first. It returned yolov8n-36 — a real 0.0, but measured on 63
    holdout images, over runs 49 and 50 which are 0.0 on 65. Presenting the older,
    less-tested run as "best" is the kind of quietly-wrong answer that survives
    for months because nothing about it looks wrong.

    Order: lowest MAE, then MOST holdout images (training_history records
    test_trays precisely because "a low MAE on a tiny testset means little"),
    then most recent. `ties` says how many runs matched on all of it, so a card
    can say "5 runs tied" rather than implying a winner.

    """
    m = metrics.get(project.load(slug).headline_metric)
    rows = [r for r in training_history.load()
            if r.get("project") == slug
            and r.get(m.field) is not None]
    if not rows:
        return None
    best = min(rows, key=m.sort_key)
    tied = sum(1 for r in rows if r[m.field] == best[m.field]) - 1
    return {"run": best["run"], "mae": best.get("mae"),
            "metric": m.key, "metric_label": m.label,
            "metric_source": m.source,
            "value": best[m.field],
            "mae_source": best.get("mae_source"),
            "test_trays": best.get("test_trays"),
            "machine": best.get("machine"), "finished": best.get("finished"),
            # Kept under its old name so nothing that reads it breaks; it now
            # counts ties on whichever metric this project actually ranks by.
            "tied_on_mae": tied}


def _thumb(pp) -> dict | None:
    """One image to put on the card, or None for a project with no data yet.

    The MOST RECENT image, not a random one. Random was the first instinct — it
    shows off the variety — but a card that changes picture on every poll is
    noise, and worse, it makes "did this project change?" unanswerable at a
    glance. The newest image is stable between edits and happens to be the more
    useful answer anyway: it is what you were last working on.

    `raw` before `testset` because the holdout is frozen — its newest image can
    be months old while the project is busy. Falls through the project's own
    collections so a project without either still gets a picture if it has one.

    Returns the stem plus its mtime; the browser passes the mtime as ?v= so the
    thumbnail can be cached immutably and still change the moment the image does.
    """
    order = ["raw", "testset"] + [c for c in labelio.collection_names(pp)
                                  if c not in ("raw", "testset")]
    for name in order:
        if name not in labelio.collection_names(pp):
            continue
        img_dir, _ = labelio._dirs(name, pp)
        if not img_dir.exists():
            continue
        newest, stamp = None, -1.0
        for f in img_dir.iterdir():
            if f.suffix.lower() == ".txt" or f.name.startswith("."):
                continue
            m = f.stat().st_mtime
            if m > stamp:
                newest, stamp = f, m
        if newest is not None:
            return {"collection": name, "stem": newest.stem, "v": int(stamp)}
    return None


def _last_activity(pp, slug: str) -> float | None:
    """Most recent sign of life: an image added or a run finished."""
    stamps = []
    for name in labelio.collection_names(pp):
        _, lbl_dir = labelio._dirs(name, pp)
        if lbl_dir.exists():
            stamps += [f.stat().st_mtime for f in lbl_dir.glob("*.txt")]
    stamps += [r["finished"] for r in training_history.load()
               if r.get("project") == slug and r.get("finished")]
    return max(stamps) if stamps else None


def _card(slug: str) -> dict:
    p = project.load(slug)
    pp = paths.for_project(p)
    c = _counts(pp)
    return {
        "slug": p.slug, "name": p.name, "owner": p.owner,
        "classes": list(p.classes), "nc": p.nc,
        "labelling": p.labelling, "headline_metric": p.headline_metric,
        "extensions": list(p.extensions),
        "dataset_root": str(p.dataset_root),
        "created": p.created,
        "collections": c["collections"], "objects": c["objects"],
        # The two collections that ARE the dataset. `pending` and `needs_fix`
        # are a queue, not data.
        "images": c["collections"].get("raw", 0) + c["collections"].get("testset", 0),
        "best_run": _best_run(slug),
        "last_activity": _last_activity(pp, slug),
        "thumb": _thumb(pp),
    }


def _visible(user: str | None) -> list[str]:
    """The slugs this account may open. THE LANDING PAGE IS A LIST OF NAMES.

    Filtered on the server rather than marked-and-hidden in the browser: a card
    carries the project's name, its classes, its image counts and a THUMBNAIL
    LIFTED FROM ITS DATA. Sending all of that and hiding it with CSS would leak
    exactly what ownership exists to protect — and `?project=` would 403 anyway,
    so the browser would be drawing a door it cannot open.

    A manifest that will not parse is kept in the list on purpose: it cannot be
    read to find an owner, and dropping it silently is how a project disappears.
    The caller reports it as broken instead.
    """
    out = []
    for slug in project.slugs():
        try:
            if readable_by(project.load(slug), user):
                out.append(slug)
        except Exception:  # noqa: BLE001
            out.append(slug)
    return out


