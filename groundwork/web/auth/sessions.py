"""A logged-in browser, and how it stops being one.

SERVER-SIDE, NOT A SIGNED COOKIE. A signed cookie needs no storage and cannot be
revoked, and both of the things this system actually needs are revocations:
"log out" and "changing my password ends my other sessions". A signed token
stays valid until it expires no matter what the user does, so logging out of a
lost phone would be a lie. The cost is one small JSON file.

It also avoids a dependency. Starlette's SessionMiddleware needs `itsdangerous`,
which is not installed, and the worker's tree is written by rsync rather than git —
so every new package is a manual install on two machines (see
scratch/check_deploy_parity.py). `secrets.token_urlsafe` plus a dict is enough.

THE FILE STORES HASHES, NOT TOKENS. The cookie value is 256 bits of urandom, so
it needs no KDF — but keying the store by the raw token would mean anyone who
could read `outputs/sessions.json` could paste a live session into their own
browser. Keying by SHA-256 of the token makes the file useless for that: it can
say a session EXISTS and cannot say what its cookie is. Same principle as
users.json holding digests rather than passwords, and it costs one hash per
request.

CONCURRENCY. Two processes can write this file — the web service serves requests in a
thread pool and nothing else touches it today, but the read-modify-write below
is not atomic across processes. It is acceptable here and would not be for the
retrain state: the worst case is a lost `last_seen` bump or one session that
outlives a logout by the width of one request, whereas a torn retrain state
loses a run. Revisit if a second process ever writes sessions.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path

from ...config import OUTPUTS_DIR  # noqa: F401 — kept for callers
from . import store as _store

# NOT under outputs/ — that tree is mounted as StaticFiles and this
# file would be served to anyone the gate lets through. See auth/store.py.
SESSIONS_PATH = _store.path("sessions.json")

COOKIE = "gw_session"
# Thirty days. The owner uses this from a phone, where the tab is reopened
# constantly and a short expiry means retyping a password one-handed in a
# workplace. Revocation is what makes a long TTL safe, and revocation is exactly
# what a server-side store buys.
TTL_S = 30 * 24 * 3600
# Below this many seconds, a `last_seen` bump is not worth rewriting the file
# for — every request would otherwise be a write.
_TOUCH_EVERY_S = 300


def _hash(token: str) -> str:
    return hashlib.sha256((token or "").encode()).hexdigest()


def _read() -> dict:
    try:
        d = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d.get("sessions") or {} if isinstance(d, dict) else {}


def _write(sessions: dict) -> None:
    """Atomic, and 0600 before the rename — same rule as users.py and env_file:
    a file that grants access must never exist at its final path world-readable,
    not even for the width of one syscall."""
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"sessions": sessions}, indent=0, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=SESSIONS_PATH.parent, prefix=".sessions.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, 0o600)
        os.replace(tmp, SESSIONS_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _live(sessions: dict, now: float) -> dict:
    return {k: v for k, v in sessions.items()
            if now - (v.get("created") or 0) < TTL_S}


def describe_agent(ua: str | None) -> str:
    """A User-Agent string, reduced to something a person can recognise.

    "Chrome on Windows" is what makes a session list usable — an id is enough to
    tell rows apart and tells you nothing about WHICH browser to go and close.
    The raw header is not stored: it is long, it is fingerprinting-grade detail
    nobody here needs, and a security store should hold the least that answers
    the question.

    ORDER MATTERS in both lists. Every Chromium browser also says "Safari", and
    Edge says "Chrome" as well, so the more specific name has to win — this is
    the classic trap with UA sniffing and the reason this is a fixed ladder
    rather than a loop over a dict.
    """
    u = (ua or "")[:400]
    if not u:
        return "unknown"
    browser = "browser"
    for needle, name in (("Edg/", "Edge"), ("OPR/", "Opera"),
                         ("Firefox/", "Firefox"), ("Chrome/", "Chrome"),
                         ("Safari/", "Safari"), ("curl/", "curl"),
                         ("python-requests", "python"), ("Wget/", "wget")):
        if needle in u:
            browser = name
            break
    system = ""
    for needle, name in (("iPhone", "iPhone"), ("iPad", "iPad"),
                         ("Android", "Android"), ("Windows", "Windows"),
                         ("Mac OS X", "Mac"), ("Macintosh", "Mac"),
                         ("CrOS", "ChromeOS"), ("Linux", "Linux")):
        if needle in u:
            system = name
            break
    return f"{browser} on {system}" if system else browser


def begin(username: str, ua: str | None = None, ip: str | None = None) -> str:
    """Start a session and return the COOKIE VALUE — the only time it exists.

    Returned rather than stored: after this call the raw token lives in exactly
    one place, the user's browser. The store keeps its hash.
    """
    token = secrets.token_urlsafe(32)
    now = time.time()
    s = _live(_read(), now)
    # WHAT, NOT WHO-IN-DETAIL: a reduced agent string and the address it came
    # from. Enough to recognise a session you want to end; not a fingerprint.
    s[_hash(token)] = {"username": username, "created": now, "last_seen": now,
                       "agent": describe_agent(ua), "ip": (ip or "")[:45]}
    _write(s)
    return token


def user_for(token: str | None) -> str | None:
    """Whose session is this, or None. Expired sessions are not renewed.

    THE TTL IS FROM `created`, NOT `last_seen`, so a session has a hard lifetime
    rather than a sliding one. A sliding window means a stolen cookie stays
    valid forever as long as it is used, which is precisely the case the
    expiry exists for.
    """
    if not token:
        return None
    now = time.time()
    rec = _read().get(_hash(token))
    if not rec or now - (rec.get("created") or 0) >= TTL_S:
        return None
    if now - (rec.get("last_seen") or 0) > _TOUCH_EVERY_S:
        s = _live(_read(), now)
        if _hash(token) in s:
            s[_hash(token)]["last_seen"] = now
            _write(s)
    return rec.get("username")


def end(token: str | None) -> None:
    """Log out THIS browser."""
    if not token:
        return
    s = _live(_read(), time.time())
    if s.pop(_hash(token), None) is not None:
        _write(s)


def end_all_for(username: str, keep: str | None = None) -> int:
    """Drop every session for this account, optionally sparing the caller's.

    Called on a password change, and that is the point of the whole module: if
    the password changed because it might be known to someone else, the sessions
    it created have to go with it or the change achieved nothing. `keep` spares
    the browser doing the changing, so the user is not logged out by their own
    successful action.
    """
    now = time.time()
    s = _live(_read(), now)
    spare = _hash(keep) if keep else None
    drop = [k for k, v in s.items() if v.get("username") == username and k != spare]
    for k in drop:
        del s[k]
    if drop:
        _write(s)
    return len(drop)


def rename_user(old: str, new: str) -> None:
    """Follow a username change, so a rename does not log the user out.

    Sessions store a username, so without this every session for `old` would
    resolve to an account that no longer exists — the user would rename
    themselves and be ejected mid-click.
    """
    s = _live(_read(), time.time())
    hit = False
    for v in s.values():
        if v.get("username") == old:
            v["username"] = new
            hit = True
    if hit:
        _write(s)


def count_for(username: str) -> int:
    return sum(1 for v in _live(_read(), time.time()).values()
               if v.get("username") == username)


def listing(username: str, current: str | None = None) -> list[dict]:
    """This account's live sessions, newest first, for the Account page.

    THE ID IS A PREFIX OF THE STORED HASH, not the token and not the hash. It
    has to be something the browser can send back to say "revoke that one", and
    both alternatives are wrong: the token exists only in its own browser, and
    the full hash is the store's key — handing it out would let anyone who saw a
    screenshot revoke someone else's session by pasting it. Twelve hex chars is
    ample to tell four sessions apart and useless as a credential, since
    revoking still requires being signed in AS this account.

    `current` marks the caller's own row so the UI can say "this browser" and
    refuse to revoke it by accident.
    """
    now = time.time()
    me = _hash(current) if current else None
    out = []
    for k, v in _live(_read(), now).items():
        if v.get("username") != username:
            continue
        out.append({"id": k[:12], "created": v.get("created"),
                    "last_seen": v.get("last_seen"),
                    # Absent on sessions created before this was recorded —
                    # rendered as "unknown" rather than backfilled with a guess.
                    "agent": v.get("agent") or "unknown",
                    "ip": v.get("ip") or "",
                    "current": k == me,
                    # How long this one has left, since the TTL runs from
                    # `created` rather than sliding — see user_for().
                    "expires_in": max(0, round(TTL_S - (now - (v.get("created") or 0))))})
    out.sort(key=lambda r: r.get("last_seen") or 0, reverse=True)
    return out


def end_one(session_id: str, username: str) -> bool:
    """Revoke ONE session of this account by the id `listing` gave out.

    SCOPED TO THE ACCOUNT. The id is a hash prefix, so a caller could in
    principle collide with somebody else's — matching on username as well means
    the worst case is a failed revoke rather than logging out a stranger.
    """
    sid = (session_id or "").strip().lower()
    if len(sid) < 8:                       # too short to identify anything
        return False
    s = _live(_read(), time.time())
    hit = [k for k, v in s.items()
           if k.startswith(sid) and v.get("username") == username]
    for k in hit:
        del s[k]
    if hit:
        _write(s)
    return bool(hit)
