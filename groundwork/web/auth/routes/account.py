"""Sign-in/out, /api/me, password/username changes, user admin."""

from __future__ import annotations

from .guards import (router, current_user, require_admin,  # noqa: F401
                     refuse_key_auth, _refuse_key_auth, _ip,
                     _device, _set_cookie, Credentials,
                     PasswordChange, UsernameChange, NewUser)

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from .. import audit, keys, sessions, throttle, users


@router.post("/api/login")
def login(body: Credentials, request: Request, response: Response):
    name = body.username.strip().lower()
    ip = _ip(request)
    # THROTTLE BEFORE VERIFYING, so a blocked caller never reaches scrypt.
    # scrypt is deliberately expensive; running it for attempts we have already
    # decided to refuse turns this endpoint into the cheapest way to load the
    # machine, which is the opposite of what a login limit is for.
    wait = throttle.blocked_for(name, ip)
    if wait > 0:
        audit.record(audit.LOGIN_THROTTLED, target=name, ok=False, ip=ip,
                     device=_device(request), detail=f"{round(wait)}s remaining")
        raise HTTPException(429, f"too many failed attempts — try again in "
                                 f"{max(1, round(wait / 60))} min")
    if not users.check(body.username, body.password):
        # ONE MESSAGE FOR BOTH FAILURES. See the module docstring.
        blocked = throttle.record_failure(name, ip)
        # The username TRIED is the point of this line; the password never is.
        audit.record(audit.LOGIN_FAIL, target=name, ok=False, ip=ip,
                     device=_device(request),
                     detail="blocked after this attempt" if blocked else None)
        raise HTTPException(401, "wrong username or password")
    throttle.clear(name, ip)
    users.touch_login(name)
    _set_cookie(response, sessions.begin(
        name, ua=request.headers.get("user-agent"), ip=ip),
        remember=body.remember)
    audit.record(audit.LOGIN_OK, actor=name, via="session", ip=ip,
                 device=_device(request))
    return {"ok": True, "user": users.record(name)}


@router.post("/api/logout")
def logout(request: Request, response: Response):
    sessions.end(request.scope.get("auth_token"))
    response.delete_cookie(sessions.COOKIE, path="/")
    audit.record(audit.LOGOUT, actor=request.scope.get("auth_user"),
                 via="session", ip=_ip(request), device=_device(request))
    return {"ok": True}


@router.get("/api/me")
def me(request: Request):
    """Who am I — the SPA's first call, and the one that decides login-or-app.

    OPEN AT THE GATE, 401 HERE. It has to be reachable while signed out, or the
    browser has no way to discover that it is signed out; middleware.py lists it
    as open for exactly that reason and this is where the answer is given.
    """
    u = request.scope.get("auth_user")
    if not u:
        raise HTTPException(401, "not signed in")
    r = users.record(u)
    if r is None:
        # The account was deleted while a session was live. Ending it here means
        # the browser sees a clean 401 next request instead of a user that does
        # not exist, which several panels would render as blanks.
        sessions.end(request.scope.get("auth_token"))
        raise HTTPException(401, "this account no longer exists")
    return {**r, "sessions": sessions.count_for(u)}


@router.post("/api/me/password")
def change_password(body: PasswordChange, request: Request,
                    user: str = Depends(current_user)):
    """Change my own password. Requires the current one.

    RE-AUTHENTICATION IS NOT CEREMONY HERE. Without it, an unattended open tab
    is a permanent account takeover: anyone who walks up can set a new password
    and own the account. Asking for the old one costs one field.
    """
    _refuse_key_auth(request)
    if not users.check(user, body.old_password):
        raise HTTPException(403, "current password is wrong")
    try:
        users.set_password(user, body.new_password)
    except users.UserError as e:
        raise HTTPException(422, str(e)) from None
    # Every OTHER session for this account dies. If the password changed because
    # it might be known to someone else, leaving their session alive would make
    # the change pointless — but the browser doing the changing is spared, so a
    # successful action does not log you out of it.
    dropped = sessions.end_all_for(user, keep=request.scope.get("auth_token"))
    audit.record(audit.PASSWORD_CHANGED, actor=user, via="session", ip=_ip(request),
                 detail=f"{dropped} other session(s) ended")
    return {"ok": True, "other_sessions_ended": dropped}


@router.post("/api/me/username")
def change_username(body: UsernameChange, request: Request,
                    user: str = Depends(current_user)):
    """Change my own username. Requires my password.

    IT MOVES PROJECT OWNERSHIP TOO. `Project.owner` stores a username, so
    users.rename rewrites every manifest this account owns in the same call —
    without that you would rename yourself and lose access to your own work,
    with no route back through the UI. Sessions and API keys follow inside the
    same call, so the rename does not eject the browser that asked for it —
    this route no longer does that itself, because the CLI needs it too.
    """
    _refuse_key_auth(request)
    if not users.check(user, body.password):
        raise HTTPException(403, "password is wrong")
    try:
        new = users.rename(user, body.new_username)
    except users.UserError as e:
        raise HTTPException(409, str(e)) from None
    audit.record(audit.USERNAME_CHANGED, actor=new, target=user, via="session",
                 ip=_ip(request), detail=f"{user} -> {new}")
    return {"ok": True, "user": users.record(new)}


@router.get("/api/users")
def list_users(_: str = Depends(require_admin)):
    """Every account. `record()` omits the digest, so no hash can leave here."""
    return {"users": [users.record(u) for u in users.usernames()]}


@router.post("/api/users")
def create_user(body: NewUser, request: Request,
                admin: str = Depends(require_admin)):
    try:
        name = users.add(body.username, body.password, admin=body.admin)
    except users.UserError as e:
        # 409 for a taken name, 422 for a bad one — the UI says different things.
        raise HTTPException(409 if "exists" in str(e) else 422, str(e)) from None
    audit.record(audit.USER_CREATED, actor=admin, target=name, via="session",
                 ip=_ip(request), detail="admin" if body.admin else None)
    return {"ok": True, "user": users.record(name)}


@router.delete("/api/users/{username}")
def delete_user(username: str, request: Request,
                admin: str = Depends(require_admin)):
    """Remove an account, and every session it had.

    THE LAST-ADMIN GUARD IS IN users.remove(), not here — a guard the HTTP layer
    owns is one the CLI does not consult, and this repo has paid twice for a rule
    that lived in only one of its callers.
    """
    if username.strip().lower() == admin:
        raise HTTPException(409, "you cannot delete the account you are using — "
                                 "ask another admin, or use the CLI")
    try:
        gone = users.remove(username)
    except users.UserError as e:
        raise HTTPException(409, str(e)) from None
    if not gone:
        raise HTTPException(404, f"no such user {username!r}")
    # Their live sessions go too, or a deleted account keeps working until its
    # cookie expires.
    sessions.end_all_for(username.strip().lower())
    keys.revoke_all_for(username.strip().lower())
    audit.record(audit.USER_DELETED, actor=admin, target=username.strip().lower(),
                 via="session", ip=_ip(request))
    return {"ok": True}


# ------------------------------------------------------------------- keys ---

