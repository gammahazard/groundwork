"""Slowing down password guessing, and nothing else.

THE ONE RATE LIMIT WORTH HAVING IN THE APP. General request limiting belongs in
a reverse proxy — it is a transport concern, it needs to see traffic this
process never does, and every implementation inside an app is a worse version of
nginx's. But a FAILED-LOGIN limit is different: only this code knows a password
was wrong, and without it a single machine can try passwords as fast as the
network allows. That is the attack this cockpit is actually exposed to the
moment it stops being network-only.

KEYED ON USERNAME **AND** SOURCE, together and separately:

  username   stops one attacker working through a word list against `mongo`,
             wherever they are calling from.
  source     stops one machine working through a list of USERNAMES, which a
             per-username counter cannot see at all.

Both are needed. Either alone leaves an obvious hole, and the pair costs one
dictionary.

IT DELAYS, IT DOES NOT LOCK OUT. A hard lockout is a denial-of-service anyone
can trigger against you by failing your login five times — the operator is
locked out of their own cockpit by a stranger. The window here EXPIRES on its
own, and a correct password clears the counter immediately, so the only person
who waits is the one still getting it wrong.

IN MEMORY, ON PURPOSE. It is per-process and it resets when the web service restarts,
which is a real limit and an acceptable one: an attacker cannot restart the
service, and persisting it would mean a disk write on every failed attempt —
turning a guessing attempt into an I/O amplifier, which is its own problem.
"""
from __future__ import annotations

import threading
import time

# Five wrong answers is far more than a person mistyping and far fewer than a
# word list. The window is short enough to be a nuisance rather than a punishment
# and long enough that guessing at scale stops being worth doing.
MAX_FAILS = 5
WINDOW_S = 15 * 60          # a failure is forgotten after this
BLOCK_S = 10 * 60           # how long the door stays shut once tripped

_lock = threading.Lock()
# key -> [timestamps of recent failures]
_fails: dict[str, list[float]] = {}
# key -> when the block lifts
_blocked: dict[str, float] = {}


def _keys(username: str, ip: str | None) -> list[str]:
    """The two buckets this attempt counts against — see the module docstring."""
    out = [f"u:{(username or '').strip().lower()[:64]}"]
    if ip:
        out.append(f"i:{ip[:45]}")
    return out


def _prune(now: float) -> None:
    """Drop expired entries. Called under the lock, from both paths.

    Without it this is a memory leak with a username-shaped key: an attacker
    choosing a new username each attempt would grow the dict forever.
    """
    for k in [k for k, until in _blocked.items() if until <= now]:
        _blocked.pop(k, None)
    for k in list(_fails):
        recent = [t for t in _fails[k] if t > now - WINDOW_S]
        if recent:
            _fails[k] = recent
        else:
            _fails.pop(k, None)


def blocked_for(username: str, ip: str | None) -> float:
    """Seconds remaining before this attempt may be tried, or 0.

    Checked BEFORE the password is verified, so a blocked caller costs nothing —
    no scrypt, no disk. That matters: scrypt is deliberately expensive, and
    running it for every attempt is how a login endpoint becomes the cheapest
    way to load the machine.
    """
    now = time.time()
    with _lock:
        _prune(now)
        return max((_blocked.get(k, 0) - now for k in _keys(username, ip)),
                   default=0)


def record_failure(username: str, ip: str | None) -> float:
    """Count one wrong password. Returns the block in seconds, or 0."""
    now = time.time()
    with _lock:
        _prune(now)
        worst = 0.0
        for k in _keys(username, ip):
            hits = _fails.setdefault(k, [])
            hits.append(now)
            if len(hits) >= MAX_FAILS:
                _blocked[k] = now + BLOCK_S
                _fails.pop(k, None)          # the block replaces the count
                worst = max(worst, float(BLOCK_S))
        return worst


def clear(username: str, ip: str | None) -> None:
    """A correct password forgets every failure for these keys.

    So the honest user who mistyped four times and then got it right is not
    carrying a near-block around for the next quarter of an hour.
    """
    now = time.time()
    with _lock:
        _prune(now)
        for k in _keys(username, ip):
            _fails.pop(k, None)
            _blocked.pop(k, None)


def state() -> dict:
    """What is currently held down — for the admin panel, not for callers."""
    now = time.time()
    with _lock:
        _prune(now)
        return {
            "blocked": [{"key": k, "seconds_left": round(until - now)}
                        for k, until in sorted(_blocked.items())],
            "watching": {k: len(v) for k, v in sorted(_fails.items())},
            "max_fails": MAX_FAILS, "window_s": WINDOW_S, "block_s": BLOCK_S,
        }
