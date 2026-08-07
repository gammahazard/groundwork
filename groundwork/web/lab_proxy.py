"""Cached, NEVER-BLOCKING reads of a worker's API from HQ.

HQ's dashboard polls worker status every ~8s, and those views used to call the
worker inline with a short urlopen timeout. When the network path is cold
(first contact on a VPN can take seconds) each poll cycle burned seconds
waiting on another machine, and the whole HQ cockpit felt stuck — even though
HQ's own endpoints answer in milliseconds.

This is the third instance of the same bug in this codebase: an expensive
external call sitting on a polled endpoint (nvidia-smi twice, then the remote
fetch). The rule: polled endpoints return what is already known and refresh
out of band.

Contract: get(machine_key, path) answers instantly with the last known value
(or None if nothing has been fetched yet) and triggers at most one background
refresh per (machine, path). Callers get `_age_s` so the UI can say how fresh
it is instead of implying live. With no registered workers it fetches nothing
and spawns nothing.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

TTL = 8.0           # matches the dashboard's poll interval
# MEASURED, not guessed: first contact on a cold VPN path took 36.1s once, and
# the two calls after it took 0.06s. A 6s timeout meant the FIRST refresh after
# any idle period always failed, so nothing was ever cached and the view stayed
# blank until something else happened to warm the path. This fetch is
# off-thread by construction — a long timeout costs one sleeping daemon thread,
# and _inflight already stops it being retried in parallel. The only number
# that must stay small is how long a CALLER waits, and that is zero.
TIMEOUT = 45.0
_cache: dict[tuple[str, str], dict] = {}
_inflight: set[tuple[str, str]] = set()
_lock = threading.Lock()


def _refresh(key: str, path: str) -> None:
    data = None
    try:
        # AUTHENTICATED, because the far end may be gated — and this function
        # swallows failures by design, so without a credential the cockpit
        # would show a blank live view rather than an error. Silent is the
        # wrong failure for "is anything training".
        from . import machines as M
        m = M.get(key)
        if m is None or not m.url:
            raise LookupError(key)
        req = urllib.request.Request(f"{m.url}{path}",
                                     headers=M.auth_headers(key))
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read())
    except Exception:  # noqa: BLE001 — an unreachable worker is not an error
        pass
    with _lock:
        # A failed fetch records the ATTEMPT (so we don't hammer) but keeps the
        # last good payload — a brief blip shouldn't blank the dashboard.
        ck = (key, path)
        prev = _cache.get(ck) or {}
        _cache[ck] = {"at": time.time(),
                      "data": data if data is not None else prev.get("data"),
                      "ok": data is not None}
        _inflight.discard(ck)


def get(key: str, path: str) -> tuple[dict | None, float]:
    """(payload, age_seconds) for machine `key`. Never blocks; may be stale."""
    ck = (key, path)
    with _lock:
        entry = _cache.get(ck)
        fresh = entry and (time.time() - entry["at"] < TTL)
        if not fresh and ck not in _inflight:
            _inflight.add(ck)
            threading.Thread(target=_refresh, args=(key, path),
                             daemon=True).start()
    if not entry:
        return None, 0.0
    return entry["data"], round(time.time() - entry["at"], 1)


def first_worker_key():
    """The single registered worker's key, or None — the common one-worker
    fleet's default for views that show 'the trainer'."""
    from . import machines as M
    ws = list(M.workers())
    return ws[0] if ws else None
