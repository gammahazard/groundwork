"""The security trail: who signed in, who changed an account, what was refused.

WHY THIS EXISTS SEPARATELY FROM THE LOGS THIS REPO ALREADY KEEPS. The ledger
records what the MACHINE did — runs, scores, exports. Nothing recorded what
PEOPLE did, and the moment there is more than one account that is the gap that
matters: "who deleted that user", "is someone guessing my password", "when was
this key last used and by whom".

APPEND-ONLY JSONL, like training_history and vram_observed. A security trail
that can be edited in place is not a trail; every other store in this package is
read-modify-write (sessions.json, api_keys.json) and that shape is wrong here —
a bug in the writer could silently drop yesterday. One line, one event, and the
only mutation this module performs is dropping the OLDEST lines when the file
grows past its cap.

WRITES ARE A SINGLE os.write TO AN O_APPEND HANDLE. Two processes share this
file — the web service and the counting bot both authenticate — and on Linux an O_APPEND write
below PIPE_BUF (4096) is atomic, so two events cannot interleave into one
corrupt line. Every field here is short and bounded for that reason; `detail` is
truncated rather than trusted.

WHAT IS NEVER WRITTEN: passwords, key values, session tokens. The store holds a
username, an event name, and a target — enough to reconstruct who did what to
whom, and nothing that would turn a leaked log into a credential. A failed login
records the username that was TRIED, which is the whole point of it, and never
the password that was tried with it.

IT MUST NEVER BREAK THE THING IT IS WATCHING. record() swallows its own errors:
an audit line that cannot be written is worth a lost line, not a failed login.
The alternative — a full disk locking everyone out of their own cockpit — is a
worse failure than a gap in the trail.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ...config import OUTPUTS_DIR  # noqa: F401 — kept for callers
from . import store as _store

# NOT under outputs/ — that tree is mounted as StaticFiles and this
# file would be served to anyone the gate lets through. See auth/store.py.
AUDIT_PATH = _store.path("audit.jsonl")

# Roughly a year of ordinary use for one household, and about 2 MB. The cap
# exists so an unattended box cannot fill a disk with sign-in lines; it is not a
# retention policy, and anything that matters should be read long before this.
MAX_LINES = 20_000
# How much slack to allow before the trim runs. Trimming rewrites the file, so
# doing it on every write past the cap would rewrite on every single event.
TRIM_SLACK = 2_000

# The events this module knows how to name. A closed vocabulary rather than free
# text: a trail you have to grep with a regex because the same thing is spelled
# three ways is a trail nobody reads.
LOGIN_OK = "login.ok"
LOGIN_FAIL = "login.fail"
LOGIN_THROTTLED = "login.throttled"
LOGOUT = "logout"
PASSWORD_CHANGED = "password.changed"
USERNAME_CHANGED = "username.changed"
USER_CREATED = "user.created"
SETUP_CLAIM = "setup.claim"
USER_DELETED = "user.deleted"
KEY_MINTED = "key.minted"
KEY_REVOKED = "key.revoked"
SESSION_REVOKED = "session.revoked"
DENIED = "denied"

# BOTS — privileged MACHINE actions, sharing the trail with the account ones
# because they are the same kind of fact: a person caused this box to do
# something it cannot undo by itself. Writing a token, writing a systemd unit
# and starting a service are the three things in this cockpit that reach
# outside its own process, and until 2026-08-05 none of them was recorded at
# all. `target` is "<project>/<key>"; `detail` carries the action, never a token.
BOT_TOKEN = "bot.token"
BOT_INSTALLED = "bot.installed"
BOT_SERVICE = "bot.service"
BOT_MOVED = "bot.moved"
BOT_ALLOWED = "bot.allowed"


def _line(d: dict) -> bytes:
    # separators: no spaces, so the line stays as small as possible — see the
    # note on atomic appends in the module docstring.
    return (json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def record(event: str, *, actor: str | None = None, target: str | None = None,
           ok: bool = True, via: str | None = None, ip: str | None = None,
           device: str | None = None, detail: str | None = None) -> None:
    """Append one event. Never raises — see the module docstring.

    actor   who did it (a username), or None when nobody was authenticated yet
    target  what it was done to — another username, a key id, a project slug
    via     "session" | "key", how the actor authenticated
    ip      where from, when the server could tell
    device  the browser, already reduced by sessions.describe_agent — "Chrome on
            Mac", never the raw User-Agent. Its own field rather than folded into
            `detail` so the admin trail can column it, and because a FAILED
            sign-in wants it too: "someone tried mongo from Safari on iPhone" is
            a different story from the same name failing from curl.
    detail  a short reason; truncated, and never anything secret
    """
    rec = {"ts": round(time.time(), 3), "event": event}
    if actor:  rec["actor"] = actor[:64]
    if target: rec["target"] = str(target)[:96]
    if not ok: rec["ok"] = False
    if via:    rec["via"] = via[:16]
    if ip:     rec["ip"] = ip[:45]           # 45 = longest IPv6 text form
    if device: rec["device"] = str(device)[:48]
    if detail: rec["detail"] = str(detail)[:200]
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(AUDIT_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, _line(rec))
        finally:
            os.close(fd)
    except OSError:
        return                                # a lost line beats a failed login
    _maybe_trim()


def _maybe_trim() -> None:
    """Drop the oldest lines when the file has grown well past the cap.

    Cheap check first: a stat, and only then a read. This runs after EVERY
    event, so it must not read a 2 MB file each time — the size guard means the
    read happens roughly once per TRIM_SLACK events.
    """
    try:
        # ~110 bytes a line in practice; the guard only has to be in the right
        # order of magnitude, since the real count is checked below.
        if AUDIT_PATH.stat().st_size < (MAX_LINES + TRIM_SLACK) * 80:
            return
        lines = AUDIT_PATH.read_bytes().splitlines(keepends=True)
        if len(lines) <= MAX_LINES + TRIM_SLACK:
            return
        keep = lines[-MAX_LINES:]
        tmp = AUDIT_PATH.with_suffix(".jsonl.tmp")
        tmp.write_bytes(b"".join(keep))
        os.replace(tmp, AUDIT_PATH)           # atomic: no window with no trail
    except OSError:
        return


def tail(n: int = 200, event: str | None = None,
         actor: str | None = None) -> list[dict]:
    """The most recent events, newest first.

    READS THE WHOLE FILE. At the cap that is ~2 MB, and this is an admin screen
    rather than a polled endpoint — a seek-backwards reader would be the right
    answer at a hundred times this size and premature complexity at this one.

    A malformed line is SKIPPED, not fatal. Two processes append here, and while
    the writes are atomic today, a trail that refuses to render because of one
    bad byte is a trail that fails exactly when something has gone wrong.
    """
    try:
        raw = AUDIT_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(raw):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if event and d.get("event") != event:
            continue
        if actor and d.get("actor") != actor:
            continue
        out.append(d)
        if len(out) >= max(1, min(n, 2000)):
            break
    return out


def summary(window_s: float = 24 * 3600) -> dict:
    """A few numbers an admin screen can lead with, over a recent window.

    Deliberately small. The trail itself is the answer to most questions; this
    exists so the panel can say "3 failed sign-ins today" without the reader
    having to scan for them.
    """
    since = time.time() - window_s
    rows = [d for d in tail(2000) if d.get("ts", 0) >= since]
    fails = [d for d in rows if d.get("event") == LOGIN_FAIL]
    return {
        "window_s": window_s,
        "events": len(rows),
        "logins": sum(1 for d in rows if d.get("event") == LOGIN_OK),
        "failed_logins": len(fails),
        # WHICH USERNAMES were tried and failed. One failure is a typo; the same
        # name twenty times is somebody working through a list.
        "failed_for": sorted({d.get("target") or d.get("actor") or "?"
                              for d in fails}),
        "throttled": sum(1 for d in rows if d.get("event") == LOGIN_THROTTLED),
        "admin_actions": sum(1 for d in rows if d.get("event") in
                             (USER_CREATED, USER_DELETED)),
    }
