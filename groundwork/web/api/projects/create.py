"""POST /api/projects — the one place a project is born."""

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


class NewProject(BaseModel):
    name: str
    classes: list[str] = ["object"]
    labelling: str = "dots"
    headline_metric: str = "count_mae"
    extensions: list[str] = []


# Filesystem-safe, URL-safe, and short enough to read in a path. The slug
# becomes a DIRECTORY NAME, so this is the only thing standing between a form
# field and a path — hence a strict allowlist rather than a blocklist.
_SLUG_OK = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")
# Names that would collide with something meaningful, or read as a mistake.
# The CON/PRN/AUX family are Windows reserved DEVICE names: harmless on this
# Linux box, and unopenable if the tree is ever synced to or restored on
# Windows. Cheap to refuse now, miserable to discover later.
_RESERVED = ({"new", "all", "none", "default", "api", "static", "outputs",
              "projects", "dataset", "runs", "alt", "admin"}
             | {"con", "prn", "aux", "nul"}
             | {f"com{i}" for i in range(1, 10)}
             | {f"lpt{i}" for i in range(1, 10)})

# A name containing a path separator is REFUSED, not cleaned. Slugifying it is
# safe — `../../etc/passwd` becomes `etc-passwd`, which cannot escape
# outputs/projects/ — but silently creating a project under a name that is
# nothing like what was typed is its own bug. Say no and say why.
_PATHY = re.compile(r"[/\\]|\.\.")


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)


# _SLUG_OK caps the whole thing at 40 characters, so a base long enough to need a
# suffix has to give up room for it rather than produce a slug the allowlist then
# rejects — which would present as "your project name is invalid" for a name that
# was fine until somebody else took it.
_MAX_SLUG = 40


def _unique_slug(base: str) -> str:
    """`base`, or the first free `base-2`, `base-3`, … — never a collision.

    THE NAME AND THE ID STOP BEING THE SAME THING HERE, and that is the point.
    They were lockstep, so two accounts could not both have a project called
    "Bolts": the second got 409 "a project called 'bolts' already exists" — about
    a project it is not allowed to see. That is both a dead end for the second
    user and a small existence oracle, contradicting the listing filter that
    exists to keep other people's projects private.

    The SLUG stays globally unique because it is a directory name, a `?project=`
    parameter, a ledger column and a `GW_PROJECT` value; the NAME is what
    a human typed and has no such duty. Nothing in the codebase derives one from
    the other outside this module (verified), so decoupling them costs nothing
    downstream.

    Reserved ids are skipped as well as taken ones — `admin-2` is fine, `admin`
    is not, and the loop must not stop on a name it would then be refused for.
    """
    base = base[:_MAX_SLUG]
    n, slug = 1, base
    while project.manifest_path(slug).exists() or slug in _RESERVED:
        n += 1
        suffix = f"-{n}"
        slug = base[:_MAX_SLUG - len(suffix)].rstrip("-") + suffix
        if n > 999:      # a caller in a loop, not a real collision
            raise HTTPException(409, f"too many projects named like {base!r}")
    return slug


@router.post("/api/projects")
def create_project(req: NewProject, user: str | None = Depends(current_user)):
    """Create a project: a manifest and an empty dataset tree. Nothing else.

    The earlier plan said this needed the upload/label flow first: a POST that
    minted an empty manifest would just be a way to make a broken project. That
    was true when it was written and is not now: Phase 4 gave every panel an
    empty state, so a project with no data renders as a coherent empty project —
    "Nothing here", a drawn no-images-yet thumbnail, no runs — rather than a
    half-drawn page. Verified by scratch/check_multiproject.mjs, which creates
    exactly this and asserts every tab reads empty.

    THE SLUG IS THE ONLY DANGEROUS FIELD, because it becomes a directory name.
    It is DERIVED from the name and then validated against an allowlist — never
    taken from the request — so no request can choose a path. `dataset_root` is
    likewise derived, never accepted: a project cannot be pointed at another
    project's data, which would be the worst possible way to lose a dataset.

    Refused on the lab for the same reason every other mutation is: its tree is
    a one-way mirror, so anything created there is deleted by the next sync.
    """
    lab_guard.no_lab_edits()
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(422, "a project needs a name")
    if _PATHY.search(name):
        raise HTTPException(422, "a project name cannot contain / \\ or ..")
    base = _slugify(name)
    if not _SLUG_OK.match(base):
        raise HTTPException(422, f"{name!r} does not make a usable id (got "
                                 f"{base!r}) — use letters and numbers")
    # UNIQUE PER OWNER, by NAME. Two accounts may both have a "Bolts"; ONE
    # account may not, because the landing page renders the name and two
    # identical cards is a question the reader cannot answer. This is scoped to
    # the caller's own projects, so it tells nobody what anyone else has.
    for other in project.slugs():
        try:
            o = project.load(other)
        except Exception:  # noqa: BLE001
            continue
        if o.owner is not None and o.owner == user and o.name.strip().lower() == name.lower():
            raise HTTPException(409, f"you already have a project called {name!r}")
    # The ID is disambiguated rather than refused — see _unique_slug.
    slug = _unique_slug(base)

    classes = tuple(c.strip() for c in req.classes if c and c.strip())
    if not classes:
        raise HTTPException(422, "a project must declare at least one class — "
                                 "it is what the model will be trained to find")
    if req.labelling not in ("dots", "boxes"):
        raise HTTPException(422, "labelling must be 'dots' or 'boxes'")
    if req.headline_metric not in metrics.METRICS:
        raise HTTPException(422, "headline_metric must be one of "
                                 + ", ".join(sorted(metrics.METRICS)))
    # No built-in extensions ship; the field is refused non-empty until a
    # deployment registers some (the mechanism stays — see project.py).
    if req.extensions:
        raise HTTPException(422, f"unknown extension(s): {', '.join(req.extensions)}")

    p = project.Project(
        slug=slug, name=name,
        # WHOEVER CREATED IT OWNS IT, taken from the session and never from the
        # request — a body field would let any account create a project owned by
        # someone else, which is the same class of mistake as accepting a slug.
        #
        # THE OWNER IS NOT IN THE PATH, deliberately. `dataset_root` stays
        # `outputs/projects/<slug>/dataset` rather than gaining a `<user>/`
        # level, because ownership CHANGES and paths do not: a transfer, or a
        # rename, would become a directory move — of a dataset, mid-mirror,
        # while `dataset_root` in the manifest and a worker's copy both point
        # at the old place. Point the code at the data, not the data at the
        # code.
        owner=user,
        # DERIVED, never supplied. See the docstring.
        dataset_root=OUTPUTS_DIR / "projects" / slug / "dataset",
        classes=classes, labelling=req.labelling,
        headline_metric=req.headline_metric,
        extensions=tuple(req.extensions), created=time.time())
    project.save(p)

    # The collections this project actually has — `collection_names` already
    # drops the ones behind an extension it did not enable.
    pp = paths.for_project(p)
    for cname in labelio.collection_names(pp):
        img, lbl = labelio._dirs(cname, pp)
        img.mkdir(parents=True, exist_ok=True)
        lbl.mkdir(parents=True, exist_ok=True)
    pp.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[projects] created {slug!r} ({name}) classes={list(classes)} "
          f"root={p.dataset_root}", flush=True)
    return {"ok": True, "project": _card(slug)}


