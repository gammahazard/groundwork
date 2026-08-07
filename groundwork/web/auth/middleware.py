"""The gate. It runs before every route AND every mount — that is the point.

WHY ASGI MIDDLEWARE AND NOT A FastAPI DEPENDENCY.

`apps/webui.py` mounts the whole `outputs/` tree as StaticFiles: eighteen
gigabytes of image photographs, eval previews, challenger checkpoints and the
CoreML exports. A dependency guards ROUTES. It would have left every one of
those files reachable by anyone who could open the port — which is not a weaker
version of being secured, it is the same exposure with a login page in front.
Middleware wraps the application, so one rule covers routes and mounts together
and there is no list of static paths to keep in sync with anything.

WHY RAW ASGI AND NOT BaseHTTPMiddleware. Starlette's convenience wrapper reads
the response body to hand it to your function, so it buffers. On a route that is
fine; on a 6 MB `.mlpackage` or a 40 MB image photograph it is a memory copy per
request, and on the streaming responses StaticFiles produces it defeats the
streaming entirely. This one touches `scope` and then gets out of the way.

WHAT IS OPEN, and why each one has to be:

    /               the SPA shell. It is markup with no data in it, and it must
                    load so it can ASK whether you are signed in.
    /api/me         how it asks. Returns 401 when you are not — so it has to be
                    reachable to be able to say so.
    /api/login      obviously.
    /static/*       the login form needs its own CSS and JS.
    /favicon.ico    a browser requests it before anything else; a 401 here just
                    fills the console with noise.

EVERYTHING ELSE IS CLOSED, `/outputs` included.

THE PATH IS NORMALISED BEFORE IT IS MATCHED. `/static/../outputs/dataset/raw/
images/x.jpg` starts with `/static/`, and a plain prefix test would wave it
through — after which the router happily serves it from the outputs mount. That
is a hole in the gate opened by the gate's own allow-list, so the path is
collapsed with posixpath.normpath first and the allow-list sees what the router
will see.
"""
from __future__ import annotations

import json
import posixpath
from http.cookies import SimpleCookie

from . import keys, sessions

# Exact paths anyone may reach.
_OPEN_EXACT = {"/", "/login", "/api/login", "/api/me", "/favicon.ico",
               "/healthz"}

# The two SETUP endpoints are open only while the instance has zero accounts —
# a DYNAMIC membership, asked per request through the router's one-way latch
# (web/api/setup.setup_open). Once any account exists they need a session like
# everything else, and deleting the accounts file does not reopen them without
# a deliberate restart.
_SETUP_OPEN = {"/api/setup/status", "/api/setup/claim"}


def _setup_allows(path: str) -> bool:
    if path not in _SETUP_OPEN:
        return False
    try:
        from ..api.setup import setup_open
        return setup_open()
    except Exception:  # noqa: BLE001 — a broken import must fail CLOSED
        return False
# Prefixes anyone may reach. Trailing slash is deliberate: "/static" alone would
# also open "/staticfiles" or any future route that merely starts with it.
_OPEN_PREFIX = ("/static/",)


def _is_open(path: str) -> bool:
    # NORMALISE FIRST — see the module docstring. A path that traverses out of an
    # open prefix must be judged where it lands, not where it starts.
    p = posixpath.normpath(path or "/")
    if not p.startswith("/"):
        p = "/" + p
    return p in _OPEN_EXACT or p.startswith(_OPEN_PREFIX) \
        or _setup_allows(p)


def _cookie_token(scope) -> str | None:
    for k, v in scope.get("headers") or ():
        if k == b"cookie":
            try:
                c = SimpleCookie()
                c.load(v.decode("latin-1"))
            except Exception:  # noqa: BLE001 — a malformed cookie is no cookie
                return None
            m = c.get(sessions.COOKIE)
            return m.value if m else None
    return None


async def _forbid(scope, send, detail: str) -> None:
    """403 for a key that is authenticated but not permitted to do THIS.

    403, never 401. The caller IS authenticated; bouncing a script to a login
    form would be nonsense, and core.js reads 401 as "show the login view" —
    which would log a person out because their key lacked a scope.
    """
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": 403,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": body})


async def _deny(scope, send) -> None:
    """401, in the shape the caller can act on.

    JSON for the API because core.js's fetch wrapper is the one place every
    request passes through and it can drop to the login view on a 401. A bare
    HTML page there would be parsed as a failed panel instead.
    """
    wants_json = scope.get("path", "").startswith("/api/")
    body = (json.dumps({"detail": "not signed in"}).encode() if wants_json
            else b"<!doctype html><meta charset=utf-8>"
                 b"<title>Sign in</title><p>Please <a href=\"/\">sign in</a>.")
    await send({"type": "http.response.start", "status": 401,
                "headers": [(b"content-type",
                             b"application/json" if wants_json else b"text/html"),
                            (b"content-length", str(len(body)).encode()),
                            # A 401 must never be reused from cache — a stale one
                            # would keep showing the login page after signing in.
                            (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": body})


class AuthGate:
    """Wraps the whole app. `scope["auth_user"]` is set for anything that passes.

    Non-HTTP scopes (lifespan, and websockets if any ever appear) are handed on
    untouched: there is no cookie to read and a gate that 401s the lifespan
    event would stop the app booting.
    """

    def __init__(self, app, enabled: bool = True):
        self.app = app
        # OFF IS A REAL STATE, and it is how the worker runs. Once its UI is retired
        # the worker serves only machine-to-machine traffic from HQ, and HQ has no
        # credential to present yet — see FRONTEND.md §3.5. Explicit, so nobody
        # has to infer whether a box is gated.
        self.enabled = enabled

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            return await self.app(scope, receive, send)
        # TWO DOORS, ONE ANSWER. A cookie is a person at a keyboard; a bearer
        # key is a script or another machine. Both resolve to a username here so
        # that everything downstream — project ownership, the admin checks,
        # lab_guard, the busy gates — applies identically. `auth_via` is carried
        # so the few routes that must NOT accept a key can say so; see
        # routes._refuse_key_auth for why that asymmetry exists.
        token = _cookie_token(scope)
        user = sessions.user_for(token)
        via = "session" if user else None
        key_scope = None
        if user is None:
            raw = keys.bearer(scope.get("headers"))
            # record_for, not user_for: the gate needs the SCOPE as well as the
            # owner, and asking twice would double the cost of the busiest path.
            rec = keys.record_for(raw)
            if rec:
                user = rec.get("username")
                key_scope = rec.get("scope") or keys.LEGACY_SCOPE
                via = "key"
        # Carried on the scope for both the open and closed paths: /api/me is
        # open precisely so it can report WHICH of the two you are.
        scope["auth_user"] = user
        scope["auth_token"] = token
        scope["auth_via"] = via
        scope["auth_scope"] = key_scope
        if user is None and not _is_open(scope.get("path", "/")):
            return await _deny(scope, send)
        # SCOPE IS ENFORCED HERE, at the gate, for the same reason the gate
        # exists at all: it covers routes AND mounts with one rule. A dependency
        # would guard handlers and leave /outputs — 18 GB of photographs and
        # checkpoints — reachable by a read key that should not be writing
        # anyway, but more importantly it would need a list of paths kept in
        # sync with the router forever.
        if via == "key" and not keys.scope_allows(
                key_scope, scope.get("method", "GET"), scope.get("path", "/")):
            from . import audit
            audit.record(audit.DENIED, actor=user, target=scope.get("path"),
                         ok=False, via="key",
                         detail=f"key scope {key_scope!r} cannot "
                                f"{scope.get('method', 'GET')} this")
            return await _forbid(
                scope, send,
                f"this API key is scoped {key_scope!r} and cannot "
                f"{scope.get('method', 'GET')} {scope.get('path')} — mint a key "
                f"with a wider scope on the API tab")
        await self.app(scope, receive, send)
