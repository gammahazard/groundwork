"""How a request says WHICH bot, and whether it may touch it.

ONE SEAM. `bot_setup._find(key)` scanned every project on the machine and
returned the first entry whose key matched, with no reference to who was asking
— which is how `POST /api/bots/{key}/project` could move somebody else's bot
into your project, rewrite its unit with your slug and restart it, quietly
redirecting their Telegram photos into your dataset. Every endpoint in that file
had the same hole because they all resolved a bot the same way.

Resolving through `current_project` instead means the ownership check that
already exists IS the bot's ownership check. No second concept, no second
implementation to drift, and the 403 and its audit line come from the path that
already writes them (deps.current_project).

A bot belongs to a project; a project belongs to a user; so a bot belongs to a
user. That is the whole model, and it is why nothing here is admin-gated —
gating on admin would stop the person whose project it is from setting up their
own bot, which is the point of the feature.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Query, Request

from ... import project as project_mod
from ..auth import audit as audit_mod
from .deps import current_user, readable_by

# A Telegram user id. Digits only, and bounded — Telegram's ids are ~10 digits
# today and were 9 not long ago, so the range is deliberately loose at both ends
# without admitting a string that is obviously something else pasted by mistake.
ID_MIN, ID_MAX = 4, 15


def required_project(slug: str = Query(..., alias="project",
                                       description="project slug (required)"),
                     request: Request = None,
                     user: str | None = Depends(current_user)):
    """The named project, IF this account may open it. `?project=` is MANDATORY.

    NOT `deps.current_project`, which defaults the slug to the default project.
    Defaulting is right for the sixty routes reached from inside an open
    project; it is wrong here. Data collection is a machine-level tab, so
    `core.js`'s PROJECT is null and a caller that forgot the parameter would
    silently ask about the first project — getting a spurious 403 about a project it
    never named, AND filing a false `denied` row against it in the one log that
    has to stay high-signal.

    A missing parameter is therefore 422 from FastAPI, which says what is wrong.
    """
    try:
        p = project_mod.load(slug)
    except Exception:  # noqa: BLE001 — a broken manifest is not a permission answer
        raise HTTPException(404, f"no such project {slug!r}") from None
    if not readable_by(p, user):
        audit_mod.record(audit_mod.DENIED, actor=user, target=slug, ok=False,
                         via=(request.scope.get("auth_via") if request else None),
                         ip=(getattr(getattr(request, "client", None), "host", None)
                             if request else None),
                         detail=f"bot setup; owned by {p.owner!r}")
        raise HTTPException(
            403, f"project {slug!r} belongs to {p.owner!r} — ask them to "
                 f"transfer it, or open one of your own")
    return p


def owned_bot(key: str, p=Depends(required_project)) -> tuple:
    """(project, bot entry) for a bot in a project you may open.

    404 rather than 403 when the key is not in THIS project, deliberately: by
    the time we are here the caller has already been cleared for the project, so
    the only remaining question is whether the bot is in it. Saying "no bot X in
    your project" reveals nothing about anyone else's.
    """
    b = next((x for x in p.bots if x.get("key") == key), None)
    if b is None:
        raise HTTPException(404, f"no bot {key!r} in {p.slug}")
    return p, dict(b)


def clean_ids(raw) -> list[str]:
    """Telegram ids from a request, validated. Raises HTTPException(422).

    Here AND again in web/bot_units before anything reaches a unit file, because
    a unit file is executed. One of the two is redundant on any given day; which
    one depends on a bug nobody has written yet.
    """
    if raw is None:
        return []
    if isinstance(raw, (str, int)):
        raw = [raw]
    out: list[str] = []
    for v in raw:
        s = str(v).strip()
        if not s:
            continue
        if not s.isdigit() or not (ID_MIN <= len(s) <= ID_MAX):
            raise HTTPException(422, f"{s!r} is not a Telegram user id — they are "
                                     f"{ID_MIN}-{ID_MAX} digits. Send /whoami to "
                                     f"your bot to get yours.")
        if s not in out:
            out.append(s)
    if len(out) > 16:
        raise HTTPException(422, "at most 16 people per bot")
    return out
