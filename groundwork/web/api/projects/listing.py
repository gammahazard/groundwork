"""GET /api/projects + /api/overview."""

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


@router.get("/api/projects")
def list_projects(user: str | None = Depends(current_user)):
    """Every project THIS ACCOUNT may open, default first — one card each.

    A project whose manifest is unreadable is reported rather than skipped: a
    projects list that quietly omits a project is how you lose one.
    """
    out, broken = [], []
    for slug in _visible(user):
        try:
            c = _card(slug)
            # WHY it is visible — "owner", "unowned" or "admin". The UI can then
            # say "visible to you as an admin" instead of implying it is yours.
            c["readable_because"] = why_readable(project.load(slug), user)
            out.append(c)
        except Exception as e:  # noqa: BLE001
            broken.append({"slug": slug, "error": f"{type(e).__name__}: {e}"})
    return {"projects": out, "broken": broken,
            "user": user, "now": time.time()}


@router.get("/api/overview")
def overview(user: str | None = Depends(current_user)):
    """Platform-wide totals — the Dashboard's headline, ACROSS projects.

    The Dashboard used to describe one project while sitting in the global half
    of the nav, which was the pivot showing through: it was written when there
    was one project and "the dataset" meant the first project's. What belongs on a
    global tab is the answer to "what is on this machine" — how many projects,
    how much labelled data, how many runs, how many model families.

    Everything is summed from the SAME `_card()` the landing page renders, so a
    total can never disagree with the cards it is a total of. That matters more
    than it sounds: two independent walks of the same directories is exactly how
    a dashboard starts quietly contradicting the page below it.

    Runs are counted per MODEL, not per family-of-privilege. There is no
    champion-vs-challenger here by design — a project has models and one of them
    serves, which is what Phase 2's move to <project>/models/<model>/<run>/ makes
    literal. yolov8n is simply the model that happens to serve the first project today.
    """
    from ....models import registry
    from ....web import bots as bots_mod

    # SCOPED TO WHAT THIS ACCOUNT MAY SEE. "What is on this machine" is a
    # platform question, but the answer is assembled from project cards — so an
    # unscoped total would report another account's image and object counts, and
    # `cards` below carries their NAMES and best runs outright.
    cards, broken = [], []
    visible = list(_visible(user))
    for slug in visible:
        try:
            cards.append(_card(slug))
        except Exception as e:  # noqa: BLE001
            broken.append({"slug": slug, "error": f"{type(e).__name__}: {e}"})

    # EVERY run, from both ledgers, keyed by MODEL. Not "champion + challengers":
    # a project has models and one of them serves, and counting them in two
    # buckets is the framing Phase 2 exists to delete. The yolo ledger lives in
    # training_history.json; the others under outputs/alt, each with its own
    # count_eval.json — two locations today, one models/ tree after Phase 2.
    per_model: dict[str, dict] = {}

    def _tally(run_name: str, arch, mae):
        m = registry.by_name(arch) or registry.for_run(run_name)
        name = m.name if m else (arch or "unknown")
        e = per_model.setdefault(name, {"model": name, "runs": 0, "best_mae": None,
                                        "license": m.license if m else None})
        e["runs"] += 1
        if mae is not None and (e["best_mae"] is None or mae < e["best_mae"]):
            e["best_mae"] = mae

    rows = training_history.load()
    for r in rows:
        _tally(r.get("run", ""), r.get("arch"), r.get("mae"))

    # EVERY project's alt tree, not `outputs/alt`. This endpoint's whole claim is
    # "what is on this machine, across projects", and it was reading one
    # project's directory — so a second project's challenger runs would be
    # missing from a total that says it counts them. Same `runs` key that the
    # per-model table and the "runs" tile are both built from, so the fix lands
    # in one place. (the first project's ALT_DIR resolves to that same outputs/alt.)
    alt_runs = 0
    seen = set()
    for pp in paths.every_project():
        alt_dir = pp.ALT_DIR
        if not alt_dir.exists():
            continue
        for f in list(alt_dir.glob("*/count_eval.json")) + list(alt_dir.glob("*/meta.json")):
            d = f.parent
            # Keyed by (project, run): two projects may each have a run called
            # "a", and collapsing them would silently drop one.
            key = (pp.slug, d.name)
            if key in seen or d.name == "datasets":
                continue
            seen.add(key)
            arch = mae = None
            for name in ("meta.json", "count_eval.json"):
                try:
                    j = json.loads((d / name).read_text())
                except Exception:  # noqa: BLE001
                    continue
                arch = arch or j.get("arch")
                mae = j.get("mae", mae)
            _tally(d.name, arch, mae)
            alt_runs += 1

    bot_rows = bots_mod.status(visible) if visible else []
    images = sum(c.get("images") or 0 for c in cards)
    objects = sum(c.get("objects") or 0 for c in cards)
    # "labelled objects" is dots-or-boxes: a project labelling with boxes counts
    # the same rows in the same .txt files. The word is deliberately not "objects".
    return {
        "projects": len(cards),
        "images": images,
        "objects": objects,
        "runs": len(rows) + alt_runs,
        "models": len(per_model),
        "per_model": sorted(per_model.values(),
                            key=lambda m: (-(m["runs"]), m["model"])),
        # ONCE, and scoped. This was two unscoped calls — so the headline
        # counted every account's bots, and each call spawns one
        # `systemctl is-active` per bot, making a dashboard poll cost 2N
        # subprocesses to answer a two-number question.
        "bots": len(bot_rows),
        "bots_running": sum(1 for b in bot_rows if b["state"] == "active"),
        "last_activity": max((c.get("last_activity") or 0 for c in cards), default=0) or None,
        "cards": [{"slug": c["slug"], "name": c["name"], "images": c.get("images"),
                   "objects": c.get("objects"), "classes": c.get("classes"),
                   "best_run": c.get("best_run")} for c in cards],
        "broken": broken,
        # WHICH MACHINE this is. A platform fact, and it has to be answerable
        # WITHOUT a project: the Dashboard no longer shows any project's data
        # until one is opened, so the lab's ember theme and read-only banner
        # would otherwise not apply until the operator clicked into a project.
        # /api/state still reports it too — that one is project-scoped and this
        # is not, which is the whole reason it belongs here as well.
        "lab": _config.is_worker(),
        "now": time.time(),
    }


