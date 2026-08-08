"""The auth dependencies + request helpers every route shares."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from .. import audit, keys, sessions, throttle, users


router = APIRouter()


def current_user(request: Request) -> str:
    """The signed-in username, or 401. The one place routes ask."""
    u = request.scope.get("auth_user")
    if not u:
        raise HTTPException(401, "not signed in")
    return u


def require_admin(request: Request, user: str = Depends(current_user)) -> str:
    """...and may manage other accounts. 403, never 401 — see the header.

    KEYS ARE REFUSED HERE, at the single point both user-management routes pass
    through, rather than repeated on each — a rule enforced in some of its
    callers is the shape this repo has paid for twice.
    """
    _refuse_key_auth(request)
    if not users.is_admin(user):
        raise HTTPException(403, "this account cannot manage users")
    return user


def refuse_key_auth(request: Request, what: str) -> None:
    """THE ASYMMETRY THAT MAKES KEYS SAFE TO HAND OUT.

    A session belongs to a person who typed a password minutes ago. A key lives
    in a config file on another machine and is the credential most likely to be
    committed, pasted or backed up by accident. So a key may do the WORK — train,
    read, report — and may not do the things that would turn a leak into a
    permanent takeover.

    Two families qualify, and the second was added 2026-08-05:

      accounts and keys   add an account, change a password or username, mint
                          another key. A leaked key must not be able to mint its
                          own successor, or revoking it means nothing.
      systemd units       write a unit file and start it. That is arbitrary code
                          running as the user who owns .env and private/ — the
                          closest thing this cockpit has to a shell, and not
                          something a string in someone's config file should
                          reach. It costs a key nothing real: registering a bot,
                          storing its token and probing it all still work.

    403 rather than 401: the caller IS authenticated, just not by a means that
    is allowed to do this, and bouncing them to a login form would be nonsense
    for a script.

    `what` completes "an API key cannot ___", so the refusal says which rule it
    hit rather than making the caller guess.
    """
    if request.scope.get("auth_via") == "key":
        raise HTTPException(
            403, f"an API key cannot {what} — sign in with a password to do this")


def _refuse_key_auth(request: Request) -> None:
    """The accounts-and-keys case, at the single point require_admin passes."""
    refuse_key_auth(request, "manage accounts or keys")


class Credentials(BaseModel):
    username: str
    password: str
    # Default TRUE keeps the long-standing behaviour (a 30-day persistent
    # cookie) for every existing caller; a browser sends it False for a
    # session-only cookie that dies when the browser closes — the "keep me
    # signed in" checkbox, unchecked.
    remember: bool = True


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


class UsernameChange(BaseModel):
    new_username: str
    password: str          # re-authenticate: renaming moves project ownership


class NewUser(BaseModel):
    username: str
    password: str
    admin: bool = False


def _ip(request: Request) -> str | None:
    """Best guess at where a request came from.

    NO X-Forwarded-For read of our own. Trusting that header without a proxy
    in front is worse than having no IP at all: the client sets it, so an
    attacker would simply send a different one per attempt and walk straight
    through the per-source throttle. Behind a reverse proxy, GW_TRUST_PROXY=1
    makes uvicorn itself rewrite request.client from the forwarded headers —
    so this function stays one line and stays honest either way.
    """
    return getattr(getattr(request, "client", None), "host", None)


def _device(request: Request) -> str:
    """This request's browser, reduced. One resolver, so the trail and the
    session list can never describe the same browser two different ways."""
    return sessions.describe_agent(request.headers.get("user-agent"))


def _set_cookie(response: Response, token: str, remember: bool = True) -> None:
    """HttpOnly so JavaScript cannot read it — an XSS in the cockpit then cannot
    exfiltrate the session. SameSite=lax stops another site POSTing to /api/train
    with your cookie attached, while normal navigation still works.

    `secure` only under GW_TRUST_PROXY=1 (TLS terminated upstream). On the
    default plain-HTTP LAN deployment a Secure cookie would simply never be
    sent, locking everyone out — which is one more reason the box stays off
    the public internet (see the package docstring).

    `remember` decides only the COOKIE's lifetime, never the server session's:
    True → Max-Age 30 days (survives a browser restart — the default, and what
    a phone wants); False → a session cookie the browser drops on close. The
    server-side session lasts 30 days either way, so an unchecked box logs you
    out client-side, it does not shorten what revocation must cover.
    """
    import os as _os
    response.set_cookie(sessions.COOKIE, token, httponly=True, samesite="lax",
                        secure=_os.environ.get("GW_TRUST_PROXY", "") == "1",
                        max_age=sessions.TTL_S if remember else None, path="/")


