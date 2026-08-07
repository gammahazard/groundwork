"""API keys, sessions, audit + per-account stats."""

from __future__ import annotations

from .guards import (router, current_user, require_admin,  # noqa: F401
                     refuse_key_auth, _refuse_key_auth, _ip,
                     _device, _set_cookie, Credentials,
                     PasswordChange, UsernameChange, NewUser)

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from .. import audit, keys, sessions, throttle, users


class NewKey(BaseModel):
    # OPTIONAL. Forcing a label before you can have a key is friction for the
    # common case ("I just need one for this box"); keys.mint dates it instead,
    # which is a better default than "unnamed" because it is still identifying.
    name: str = ""
    expires_days: float | None = None
    # WHAT IT MAY DO — read / train / full. Defaults to `train`, which is what
    # a key is actually minted for; `full` has to be asked for, because it is
    # the one that can register a machine and write bot config.
    scope: str = keys.DEFAULT_SCOPE


@router.get("/api/keys/scopes")
def key_scopes():
    """What a key may be scoped to, and what each one means.

    Served rather than hardcoded in the browser: the rule lives in keys.py and a
    second copy in JavaScript would be a second thing to update when a scope is
    added, and the wrong one would still render.
    """
    return {"scopes": [
        {"key": keys.READ, "label": "Read only",
         "what": "Every GET and nothing else. The safest thing to leave on a "
                 "machine you do not control — a dashboard or a monitor."},
        {"key": keys.TRAIN, "label": "Train and export",
         "what": "Read, plus starting and stopping runs, exporting, counting "
                 "and asking LocateAnything. What a key is usually for."},
        {"key": keys.FULL, "label": "Full",
         "what": "Everything a key may ever do, including registering a machine "
                 "and writing bot config. Registering a machine makes the "
                 "dataset rsync to it, so give this out deliberately."},
    ], "default": keys.DEFAULT_SCOPE, "legacy": keys.LEGACY_SCOPE}


@router.get("/api/keys")
def list_keys(user: str = Depends(current_user)):
    """YOUR keys. An admin does not see everyone's here — a key is a personal
    credential, and a list of other people's is not a thing this page is for."""
    return {"keys": keys.listing(user)}


@router.post("/api/keys")
def create_key(body: NewKey, request: Request, user: str = Depends(current_user)):
    """Mint a key. The raw value is in THIS response and nowhere else, ever.

    SESSION ONLY. A key cannot mint another key, for the same reason it cannot
    change a password: that would make one leaked key regenerate itself forever,
    and revocation would stop meaning anything.
    """
    _refuse_key_auth(request)
    if body.scope not in keys.SCOPES:
        raise HTTPException(422, f"scope must be one of {', '.join(keys.SCOPES)}")
    kid, raw = keys.mint(user, body.name, body.expires_days, scope=body.scope)
    audit.record(audit.KEY_MINTED, actor=user, target=kid, via="session",
                 ip=_ip(request),
                 detail=f"scope {body.scope}" + (f" · {body.name}" if body.name else ""))
    return {"ok": True, "id": kid, "key": raw, "scope": body.scope}


@router.delete("/api/keys/{kid}")
def delete_key(kid: str, request: Request, user: str = Depends(current_user)):
    _refuse_key_auth(request)
    if not keys.revoke(kid, user):
        raise HTTPException(404, f"no key {kid!r} on this account")
    audit.record(audit.KEY_REVOKED, actor=user, target=kid, via="session",
                 ip=_ip(request))
    return {"ok": True}


# --------------------------------------------------------------- sessions ---

@router.get("/api/sessions")
def list_sessions(request: Request, user: str = Depends(current_user)):
    """MY live sessions. Never anyone else's, not even for an admin.

    A session list is a list of places a person is signed in from. An admin can
    already END another account's sessions by deleting it, which is the power
    that matters; being able to WATCH them is a different thing and not one this
    page needs.
    """
    return {"sessions": sessions.listing(user, request.scope.get("auth_token"))}


@router.delete("/api/sessions/{sid}")
def revoke_session(sid: str, request: Request, user: str = Depends(current_user)):
    """Sign one browser out. Refuses the one you are using.

    THE CURRENT SESSION IS NOT REVOCABLE HERE, and that is a usability choice
    rather than a security one: "revoke" on the row marked *this browser* would
    log you out mid-task and look like a crash. Sign out is the button for that,
    it is elsewhere, and it says what it does.
    """
    _refuse_key_auth(request)
    mine = sessions.listing(user, request.scope.get("auth_token"))
    row = next((r for r in mine if r["id"] == sid), None)
    if row is None:
        raise HTTPException(404, "no such session on this account")
    if row.get("current"):
        raise HTTPException(409, "that is this browser — use Sign out")
    if not sessions.end_one(sid, user):
        raise HTTPException(404, "no such session on this account")
    audit.record(audit.SESSION_REVOKED, actor=user, target=sid, via="session",
                 ip=_ip(request))
    return {"ok": True}


@router.post("/api/sessions/revoke-others")
def revoke_other_sessions(request: Request, user: str = Depends(current_user)):
    """Sign out everywhere except here — the "I left it open somewhere" button."""
    _refuse_key_auth(request)
    n = sessions.end_all_for(user, keep=request.scope.get("auth_token"))
    audit.record(audit.SESSION_REVOKED, actor=user, via="session", ip=_ip(request),
                 detail=f"{n} other session(s)")
    return {"ok": True, "ended": n}


# ------------------------------------------------------------------ audit ---

@router.get("/api/audit")
def audit_trail(n: int = 200, event: str | None = None,
                actor: str | None = None,
                _: str = Depends(require_admin)):
    """The security trail, newest first. ADMIN ONLY.

    It names who did what to whom, so it is exactly as sensitive as the account
    list and gated the same way. Keys are refused by require_admin — a trail
    readable by a credential that lives in a config file would defeat its own
    purpose.
    """
    return {"events": audit.tail(n, event=event, actor=actor),
            "summary": audit.summary(),
            "throttle": throttle.state()}


@router.get("/api/users/stats")
def user_stats(_: str = Depends(require_admin)):
    """Per-account activity, for the admin panel.

    ASSEMBLED FROM WHAT ALREADY EXISTS — the account record, live sessions, key
    metadata, the audit trail and project ownership. Nothing new is stored to
    make this page possible: a counter maintained only so a screen can render it
    is a second source of truth that will eventually disagree with the first.
    """
    from .... import project as project_mod
    from ....dataset import paths as paths_mod
    from ....dataset.store import labelio
    from ....dataset.pipeline import training_history

    # ONE PASS OVER THE LEDGER, not one per user. It is a single file read; doing
    # it inside the per-user loop would re-read it for every account.
    runs_by_project: dict[str, list[dict]] = {}
    try:
        for row in training_history.load():
            runs_by_project.setdefault(
                row.get("project") or "?", []).append(row)
    except Exception:  # noqa: BLE001 — an unreadable ledger must not 500 this
        pass

    owners: dict[str, list[dict]] = {}
    for slug in project_mod.slugs():
        try:
            p = project_mod.load(slug)
        except Exception:  # noqa: BLE001 — a broken manifest must not break this
            continue
        if not p.owner:
            continue
        # WHAT THIS PERSON ACTUALLY HAS. Counted from the same collections the
        # project page counts, so the two cannot disagree: `raw` is the training
        # set and `testset` the holdout, and the queue collections are not data.
        pp = paths_mod.for_project(p)
        counts, objects = {}, 0
        try:
            for name in labelio.collection_names(pp):
                _img, lbl = labelio._dirs(name, pp)
                if not lbl.exists():
                    continue
                txts = list(lbl.glob("*.txt"))
                counts[name] = len(txts)
                if name in ("raw", "testset"):
                    for t in txts:
                        try:
                            objects += sum(1 for ln in t.read_text().splitlines()
                                           if ln.strip())
                        except OSError:
                            pass
        except Exception:  # noqa: BLE001
            pass
        rows = runs_by_project.get(p.slug, [])
        scored = [r for r in rows if r.get("mae") is not None]
        owners.setdefault(p.owner, []).append({
            "slug": p.slug, "name": p.name,
            "training": counts.get("raw", 0), "holdout": counts.get("testset", 0),
            "needs_fix": counts.get("needs_fix", 0),
            "objects": objects,
            "runs": len(rows),
            "best_mae": min((r["mae"] for r in scored), default=None),
            "last_run": max((r.get("finished") or 0 for r in rows), default=0) or None,
        })

    trail = audit.tail(2000)
    out = []
    for name in users.usernames():
        rec = users.record(name) or {}
        mine = [d for d in trail if d.get("actor") == name]
        fails = [d for d in trail
                 if d.get("event") == audit.LOGIN_FAIL and d.get("target") == name]
        klist = keys.listing(name)
        # LAST ACTIVE IS NOT LAST LOGIN, and the difference is the useful one.
        # Sessions last 30 days, so someone can sign in once and use the cockpit
        # every day after — "last seen 3 weeks ago" would be true and useless.
        # The freshest session touch is when they were actually here.
        live = sessions.listing(name)
        last_active = max([r.get("last_seen") or 0 for r in live] or [0]) or None
        # Where and what, from the most recent successful sign-in. Answers "is
        # that them?" without opening the trail.
        last_in = next((d for d in mine if d.get("event") == audit.LOGIN_OK), None)
        out.append({
            **rec,
            "last_active": last_active,
            "last_ip": (last_in or {}).get("ip"),
            "last_device": (last_in or {}).get("device"),
            # Every distinct browser this account has signed in from. A new one
            # appearing is the thing worth noticing.
            "devices": sorted({d.get("device") for d in mine
                               if d.get("event") == audit.LOGIN_OK and d.get("device")}),
            "sessions": sessions.count_for(name),
            "keys": len(klist),
            "keys_expired": sum(1 for k in klist if k.get("expired")),
            "key_last_used": max([k.get("last_used") or 0 for k in klist] or [0]) or None,
            # The projects themselves, not just their names — the admin panel
            # opens each account to show what it owns and how big it is.
            "projects": sorted(owners.get(name, []), key=lambda x: x["slug"]),
            "images": sum(p["training"] + p["holdout"] for p in owners.get(name, [])),
            "runs": sum(p["runs"] for p in owners.get(name, [])),
            "events": len(mine),
            "last_event": mine[0].get("ts") if mine else None,
            # Failures against THIS username, whoever was trying. One is a typo;
            # a run of them is the thing an admin screen exists to surface.
            "failed_logins": len(fails),
        })
    # Newest activity first — the account someone is using right now is the one
    # an admin most likely opened this page about.
    out.sort(key=lambda r: (r.get("last_active") or r.get("last_login") or 0),
             reverse=True)
    return {"users": out}
