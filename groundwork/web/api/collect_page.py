"""The Data collection page: YOUR bots, and what you could add.

IT USED TO BE THE MACHINE'S BOTS, and that was a deliberate decision (owner,
2026-07-31) taken when there was one account: "a bot is a thing running on this
box, and the interesting question is what is collecting right now and where it
is putting things — which spans projects by nature." That reasoning was right
for one operator and is wrong for two. The owner asked for separation on
2026-08-05, so this is a reversal, recorded rather than quiet.

WHAT REPLACES IT: a bot belongs to a project, a project belongs to a user, so
the page shows the bots of the projects you can open. An admin still sees
everything — the same rule `deps.readable_by` applies to projects, and for the
same reason: deleting an account must not strand its work with no route back.
The page says WHICH reason, via `why_readable`, rather than implying everything
listed is yours.

FILTERED BEFORE `bots_mod.status`, NOT AFTER, and that is not tidiness:
`status()` runs one `systemctl is-active` per bot. Filtering afterwards would
spawn a subprocess for every bot on the machine on every poll of this page, most
of them for bots the caller is not allowed to know exist.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import project as project_mod
from .. import bot_roles, bots as bots_mod
from ..auth import users as users_mod
from .bots import setup_steps
from .deps import current_user, readable_by, why_readable

router = APIRouter()


def _process_model() -> str:
    from .. import botsup
    return botsup.mode()


@router.get("/api/collect")
def collect_overview(user: str | None = Depends(current_user)):
    """Every bot in a project this account may open, plus what is installable."""
    projects, slugs = [], []
    for slug in project_mod.slugs():
        try:
            p = project_mod.load(slug)
        except Exception:  # noqa: BLE001 — a broken manifest is not a permission
            continue
        if not readable_by(p, user):
            continue
        slugs.append(p.slug)
        used = {b.get("role") for b in p.bots if b.get("role")}
        projects.append({
            "slug": p.slug, "name": p.name,
            "why": why_readable(p, user),
            # ONLY THE ROLES THIS PROJECT HAS NOT USED. A project holds at most
            # one bot per role, so offering a used one would produce a form that
            # can only 409. The cap is enforced in api/bots.register_bot; this
            # is what stops the UI asking for something the server refuses.
            "roles": [{"key": r.key, "name": r.name, "what": r.what}
                      for r in bot_roles.for_project(p) if r.key not in used],
        })

    rows = bots_mod.status(slugs) if slugs else []
    # A bot behind an extension its project no longer enables is not "missing",
    # it is not applicable. api/bots.list_bots has always dropped these; this
    # page did not, so a retired extension showed a permanently dead card.
    enabled = {}
    for slug in slugs:
        try:
            enabled[slug] = project_mod.load(slug)
        except Exception:  # noqa: BLE001
            pass
    rows = [b for b in rows
            if not b.get("extension")
            or (b["project"] in enabled and enabled[b["project"]].has(b["extension"]))]

    return {"process_model": _process_model(),
            
        "projects": projects,
        "bots": rows,
        # The legacy machine-wide id is an ADMIN fact now — it governs only the
        # hand-written units and is admin-writable. Non-admins are not shown a
        # control they cannot use and do not need.
        # Written against the first project this account can actually use, so
        # step 2 does not name a project it is forbidden from opening.
        "steps": setup_steps(slugs[0] if slugs else None),
        "roles": [{"key": r.key, "name": r.name, "what": r.what,
                   "extension": r.extension} for r in bot_roles.ROLES.values()],
    }



