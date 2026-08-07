"""API keys: a second way to answer "who is asking", for things without a browser.

    .venv/bin/python -m groundwork.web.auth.keys add hq-control
    .venv/bin/python -m groundwork.web.auth.keys list
    .venv/bin/python -m groundwork.web.auth.keys revoke k7f3a1

WHY THIS IS SMALL. `middleware.py` resolves the caller ONCE, for the whole app,
and sets `scope["auth_user"]`. A key is one more branch there — after which
project ownership, the admin checks, `lab_guard` and every busy gate apply
unchanged. That is the property the owner asked for: same rules, different door.

=============================================================================
WHAT A KEY MAY NOT DO, AND WHY THAT ASYMMETRY IS THE POINT
=============================================================================

A key CANNOT manage accounts. Not `/api/users`, not `/api/me/password`, not
`/api/me/username`.

A session belongs to a person at a keyboard who typed a password minutes ago. A
key belongs to a script, lives in a config file on another machine, and is
exactly the credential most likely to be committed, pasted or backed up by
accident. If a leaked training key could add an admin, one careless `git add`
would be a permanent takeover rather than an inconvenience you revoke.

So the blast radius of a key is deliberately bounded to the work: train, read,
report. Escalation requires the password.

=============================================================================
MINTED ON THE MACHINE THAT WILL HONOUR IT
=============================================================================

There is no shipped key and no shared secret. Each machine mints its own, and
the caller stores it. A key baked into the repo would be a backdoor in every
install and revoking it would mean cutting a release.

For HQ->worker that means: mint on the WORKER, paste into HQ. The direction is one-way
— HQ calls the worker, the worker never calls HQ — so there is exactly one key per
machine pair, and the CLI above exists so nobody has to open the worker's cockpit to
create it. Needing a second UI is the thing being removed, not reinforced.

=============================================================================
STORAGE
=============================================================================

`outputs/api_keys.json`, keyed by SHA-256 of the key. 32 bytes of urandom need no
KDF — there is no low-entropy secret to stretch — but hashing means the file
cannot hand anyone a working credential, the same reasoning as sessions.py. The
raw key exists exactly once, in the response to the call that created it.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import time

from ...config import OUTPUTS_DIR  # noqa: F401 — kept for callers
from . import store as _store

# NOT under outputs/ — that tree is mounted as StaticFiles and this
# file would be served to anyone the gate lets through. See auth/store.py.
KEYS_PATH = _store.path("api_keys.json")

# A visible, greppable prefix. Secret scanners key on shapes like this, and it
# lets middleware tell "this is meant to be an API key" from "this is a mangled
# cookie" without a lookup — which is the difference between a clear 401 and a
# confusing one.
PREFIX = "gw_"
HEADER = "authorization"


def _hash(raw: str) -> str:
    return hashlib.sha256((raw or "").encode()).hexdigest()


def _read() -> dict:
    try:
        d = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d.get("keys") or {} if isinstance(d, dict) else {}


def _write(keys: dict) -> None:
    """Atomic, 0600 before the rename — same rule as users.py and sessions.py."""
    KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"keys": keys}, indent=1, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=KEYS_PATH.parent, prefix=".api_keys.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, 0o600)
        os.replace(tmp, KEYS_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- scopes ---
#
# WHAT A KEY IS ALLOWED TO DO, and the reason this exists at all: a key was
# all-or-nothing, so one pasted into a CI job could also register a machine —
# which makes the dataset rsync to a host of the caller's choosing — delete
# labelled images, and install systemd units. None of that is what anyone mints
# a key for.
#
# THREE, NOT TEN. A scope list nobody can hold in their head gets set to the
# widest option every time, which is the state we are leaving. These map onto
# the three real reasons to have a key:
#
#   read    watch. Every GET, nothing else. A dashboard, a monitor, a status
#           poller — the safest thing to leave on a machine you do not control.
#   train   work. read, plus starting and stopping runs, exporting, counting and
#           asking LocateAnything. The default, because it is what a key is for.
#   full    everything a session could do except manage accounts and keys, which
#           no key may ever do (see routes._refuse_key_auth). Registering a
#           machine and writing bot config live here.
#
# THE WRITE LIST IS A WHITELIST, not a blacklist of dangerous paths. A blacklist
# is wrong by default: the next endpoint anyone adds is permitted until somebody
# remembers to forbid it, and this repo has already paid twice for a rule
# enforced in only some of its callers.
READ, TRAIN, FULL = "read", "train", "full"
SCOPES = (READ, TRAIN, FULL)

# KEYS WITHOUT A SCOPE ARE `full`. Every key minted before this existed had no
# limit, and silently narrowing them would break whatever is using them with a
# 403 nobody could explain. They are shown as `full` in the UI so the state is
# visible rather than implied.
DEFAULT_SCOPE = TRAIN
LEGACY_SCOPE = FULL

# What `train` may write to. Prefixes, and each one is a job someone mints a key
# to do. Anything not here needs `full`.
_TRAIN_WRITE = (
    "/api/train",        # start / cancel, either family, either machine
    "/api/retrain",      # the yolo job directly
    "/api/lab/train",    # a challenger directly
    "/api/lab/score",    # score a finished run
    "/api/export",       # turn a run into a file
    "/api/count",        # count one image, saves nothing
    "/api/la",           # ask LocateAnything for a second opinion
    "/api/machine/pair",  # one call, guarded again inside by the single-use
                          # pairing ticket — see web/api/pairing.accept_pair
)


def scope_allows(scope: str | None, method: str, path: str) -> bool:
    """May a key with this scope make this request?

    METHOD FIRST, because it is the half that cannot be got wrong: everything
    that changes state is a POST, DELETE, PUT or PATCH, and every scope may GET.
    Only then does the path matter, and only for `train`.
    """
    if (method or "GET").upper() in ("GET", "HEAD", "OPTIONS"):
        return True
    s = scope or LEGACY_SCOPE
    if s == FULL:
        return True
    if s == TRAIN:
        p = (path or "/").rstrip("/")
        return any(p == w or p.startswith(w + "/") or p.startswith(w + "?")
                   for w in _TRAIN_WRITE)
    return False                     # read: no writes at all


def _expired(rec: dict, now: float) -> bool:
    exp = rec.get("expires")
    return bool(exp) and now >= exp


def mint(username: str, name: str, expires_days: float | None = None,
         scope: str = DEFAULT_SCOPE) -> tuple[str, str]:
    """Create a key for `username`. Returns (id, RAW KEY) — the only time it exists.

    The id is what the UI and the CLI refer to afterwards, because the key itself
    is never readable again. Without it, revoking would mean pasting the secret
    back in, which defeats the storage design.
    """
    # A DATED DEFAULT, not "unnamed". The name is the only thing distinguishing
    # two keys in a list once the secrets are gone, so an unnamed one is a key
    # nobody dares revoke.
    name = (name or "").strip() or time.strftime("key-%Y-%m-%d")
    raw = PREFIX + secrets.token_urlsafe(32)
    kid = secrets.token_hex(3)
    now = time.time()
    keys = _read()
    keys[_hash(raw)] = {
        "id": kid, "name": name[:60], "username": username, "created": now,
        "last_used": None,
        # An unknown scope becomes the DEFAULT, never `full`. A typo must not
        # quietly widen a credential.
        "scope": scope if scope in SCOPES else DEFAULT_SCOPE,
        "expires": now + expires_days * 86400 if expires_days else None,
    }
    _write(keys)
    return kid, raw


def record_for(raw: str | None) -> dict | None:
    """The whole key record, or None. What the gate needs to judge a request.

    Separate from user_for because the gate wants the SCOPE as well as the owner,
    and re-reading the store twice per request to get both would double the cost
    of the busiest path there is.
    """
    if not raw or not raw.startswith(PREFIX):
        return None
    now = time.time()
    h = _hash(raw)
    rec = _read().get(h)
    if not rec or _expired(rec, now):
        return None
    last = rec.get("last_used") or 0
    if now - last > 3600:
        keys = _read()
        if h in keys:
            keys[h]["last_used"] = now
            _write(keys)
    return rec


def user_for(raw: str | None) -> str | None:
    """Whose key is this, or None. Bumps last_used.

    `last_used` is written on a coarse schedule rather than every call: this runs
    on EVERY request that presents a key, and a file write per request would turn
    the busiest path into the slowest. A day's resolution is enough to answer the
    only question it exists for — "is this key still in use, or did I forget it?"
    """
    if not raw or not raw.startswith(PREFIX):
        return None
    now = time.time()
    h = _hash(raw)
    rec = _read().get(h)
    if not rec or _expired(rec, now):
        return None
    last = rec.get("last_used") or 0
    if now - last > 3600:
        keys = _read()
        if h in keys:
            keys[h]["last_used"] = now
            _write(keys)
    return rec.get("username")


def bearer(headers) -> str | None:
    """The token from an `Authorization: Bearer …` header, or None.

    NEVER FROM A QUERY PARAMETER, and that is deliberate rather than an omission.
    A secret in a URL lands in access logs, in browser history, and in the
    Referer header of any outbound link — three places nobody thinks to rotate.
    """
    for k, v in headers or ():
        if k.lower() == HEADER.encode() if isinstance(k, bytes) else k.lower() == HEADER:
            val = v.decode("latin-1") if isinstance(v, bytes) else v
            parts = val.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
    return None


def listing(username: str | None = None) -> list[dict]:
    """Keys, WITHOUT their hashes — safe to return from an endpoint.

    An expired key is still listed, flagged, rather than hidden: "why did my
    script stop working" is answered by seeing it there and expired, not by it
    having silently vanished.
    """
    now = time.time()
    out = []
    for rec in _read().values():
        if username and rec.get("username") != username:
            continue
        out.append({"id": rec.get("id"), "name": rec.get("name"),
                    "username": rec.get("username"), "created": rec.get("created"),
                    "last_used": rec.get("last_used"), "expires": rec.get("expires"),
                    # Absent means it predates scopes — reported as what it
                    # actually is rather than as a blank.
                    "scope": rec.get("scope") or LEGACY_SCOPE,
                    "legacy_scope": "scope" not in rec,
                    "expired": _expired(rec, now)})
    out.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return out


def revoke(kid: str, username: str | None = None) -> bool:
    """Delete a key by id. `username` restricts it to that account's own keys."""
    keys = _read()
    for h, rec in list(keys.items()):
        if rec.get("id") == kid and (username is None
                                     or rec.get("username") == username):
            del keys[h]
            _write(keys)
            return True
    return False


def revoke_all_for(username: str) -> int:
    """Every key of an account — called when the account is deleted, or a key
    holder is renamed away. A key outliving its owner is a credential nobody
    can see in a list any more."""
    keys = _read()
    gone = [h for h, r in keys.items() if r.get("username") == username]
    for h in gone:
        del keys[h]
    if gone:
        _write(keys)
    return len(gone)


def rename_user(old: str, new: str) -> None:
    """Follow a username change, so a rename does not silently kill every key."""
    keys = _read()
    hit = False
    for rec in keys.values():
        if rec.get("username") == old:
            rec["username"] = new
            hit = True
    if hit:
        _write(keys)


# --------------------------------------------------------------------- CLI ---

def main() -> None:
    import argparse
    from . import users
    ap = argparse.ArgumentParser(
        description="API keys. The key is printed ONCE and never stored in the clear.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add", help="mint a key")
    s.add_argument("name", help="what it is for, e.g. hq-control")
    s.add_argument("--user", default=None,
                   help="account it acts as (default: the only admin)")
    s.add_argument("--days", type=float, default=None, help="expire after N days")
    s = sub.add_parser("revoke", help="delete a key by id")
    s.add_argument("id")
    sub.add_parser("list", help="every key (never shows a key)")

    a = ap.parse_args()
    if a.cmd == "list":
        rows = listing()
        if not rows:
            print("no api keys")
            return
        for r in rows:
            when = time.strftime("%Y-%m-%d", time.localtime(r["created"] or 0))
            used = ("never" if not r["last_used"] else
                    time.strftime("%Y-%m-%d", time.localtime(r["last_used"])))
            print(f"  {r['id']}  {r['name']:<24} {r['username']:<12} "
                  f"created {when}  used {used}"
                  + ("  EXPIRED" if r["expired"] else ""))
        return
    if a.cmd == "revoke":
        print(f"[keys] revoked {a.id}" if revoke(a.id)
              else f"[keys] no such key {a.id!r}")
        return

    user = a.user
    if user is None:
        # DEFAULT TO THE ONLY ADMIN, and refuse when that is ambiguous. Guessing
        # which account a key acts as is guessing what it may reach.
        admins = users.admins()
        if len(admins) != 1:
            raise SystemExit("[keys] --user is required (there is not exactly "
                             f"one admin: {admins or 'none'})")
        user = admins[0]
    if users.record(user) is None:
        raise SystemExit(f"[keys] no such account {user!r}")
    kid, raw = mint(user, a.name, a.days)
    print(f"[keys] {kid}  {a.name}  (acts as {user})")
    print(f"\n    {raw}\n")
    print("Copy it now — it is stored only as a hash and cannot be shown again.")


if __name__ == "__main__":
    main()
