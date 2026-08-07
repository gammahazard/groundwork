"""Accounts: who may open this cockpit at all.

    .venv/bin/python -m groundwork.web.auth.users add mongo
    .venv/bin/python -m groundwork.web.auth.users passwd mongo
    .venv/bin/python -m groundwork.web.auth.users list
    .venv/bin/python -m groundwork.web.auth.users remove someone

=============================================================================
WHY THE STDLIB AND NOT bcrypt/argon2
=============================================================================

`hashlib.scrypt` is a memory-hard KDF and it is already here. Nothing else in
this repo needs a password library, and a new dependency is not free on this
system specifically: the worker's working tree is written by rsync rather than git
(scratch/check_deploy_parity.py), so a dependency has to be installed by hand on
both machines, and a cockpit that will not import is a cockpit that will not
train. The parameters below cost ~0.1-0.3s per attempt, which is the point.

The digest carries its own parameters:

    scrypt$<n>$<r>$<p>$<salt-b64>$<key-b64>

so N can be raised later without invalidating a single existing password —
verify() reads the cost out of the stored hash rather than assuming today's.

=============================================================================
NO SELF-SERVICE SIGNUP. ADMINS MANAGE USERS.
=============================================================================

This section used to read "NO SIGNUP ENDPOINT. EVER." — the owner has overruled
it (2026-08-05) and it is recorded here rather than quietly reversed, same as
web/env_file.py records the token-writing decision.

What was right about it survives: there is still no route by which an
UNAUTHENTICATED caller creates an account. What changes is that an account
flagged `admin` may add and remove others through the cockpit, because the
alternative is an ssh session every time a person is added — which is the same
argument that won for bot tokens, and it is the owner's machine.

So there are two ways an account comes into being, and no third:

    terminal    `python -m groundwork.web.auth.users add <name>` — always available,
                and the only way to make the FIRST one by hand
    an admin    POST /api/users, authenticated, admin-only

THE PASSWORD NEVER TRAVELS ON ARGV. `add mongo` prompts; it does not take the
password as an argument. A password in argv is in `ps`, in shell history, and in
the systemd journal if a unit ever ran it.

ADMIN IS A FLAG, NOT A SEPARATE STORE, because there is one operator and a role
table for one person is furniture. `is_admin()` is the only reader; every route
that manages users asks it and nothing else does.

=============================================================================
FIRST-RUN BOOTSTRAP, for a fresh checkout (and for open-sourcing this)
=============================================================================

A cockpit nobody can log into is not a safe default, it is a brick. So:

    GW_ADMIN_USER=admin
    GW_ADMIN_PASSWORD=a-real-password

in `.env` creates that ONE account, and ONLY when no users exist yet.

That condition is the whole security of it. If the env could override an
existing account, the variable would be a permanent password-reset backdoor that
anyone who could write `.env` — or set a variable on the service — could use. It
cannot: once `users.json` has a single entry, `bootstrap()` is a no-op forever
and says so. Deleting users.json to re-trigger it requires filesystem access, at
which point the attacker did not need the backdoor.

The bootstrapped account is flagged `must_change`, so the UI can insist on a real
password before it is useful.

=============================================================================
WHAT THIS MODULE WILL NOT DO
=============================================================================

Return a password or a hash to a caller. `record()` exists and omits the digest.
Same rule as web/env_file.py: there is no function here that hands a secret out,
so no endpoint can grow one by accident.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from ...config import OUTPUTS_DIR  # noqa: F401 — kept for callers
from . import store as _store

# NOT under outputs/ — that tree is mounted as StaticFiles and this
# file would be served to anyone the gate lets through. See auth/store.py.
USERS_PATH = _store.path("users.json")

# ~0.15s on this laptop. n is the work factor; 128 * r * n bytes of RAM, so
# 2**15 * 8 * 128 = 32 MiB — hence the explicit maxmem, because OpenSSL's
# default is exactly 32 MiB and lands on the wrong side of the comparison.
_N, _R, _P, _DKLEN = 2 ** 15, 8, 1, 32
_MAXMEM = 128 * _R * _N * 2
_SALT_BYTES = 16

# A username becomes a JSON key and appears in URLs and logs. Deliberately
# narrow — this is not a display name, and there is exactly one operator.
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")

# Short enough to be typed on a phone, long enough not to be guessed in the
# ~0.15s per attempt this KDF costs. Not a complexity rule: length is what
# matters and complexity rules push people toward Passw0rd!.
MIN_PASSWORD = 8
# The passwords everyone tries first. Compared with case, hyphens and
# underscores stripped, so "Change-Me" is as refused as "changeme".
BANNED_PASSWORDS = {"changeme", "password", "password1", "12345678",
                    "admin123", "groundwork", "letmein1"}

BOOTSTRAP_USER_ENV = "GW_ADMIN_USER"
BOOTSTRAP_PASS_ENV = "GW_ADMIN_PASSWORD"


class UserError(ValueError):
    """A username or password this module refuses."""


# ----------------------------------------------------------------- hashing ---

def hash_password(password: str) -> str:
    """`scrypt$n$r$p$salt$key`. A fresh random salt every call."""
    if len(password or "") < MIN_PASSWORD:
        raise UserError(f"password must be at least {MIN_PASSWORD} characters")
    if (password or "").lower().replace("-", "").replace("_", "") in BANNED_PASSWORDS:
        raise UserError("that password is on every guess list — pick another")
    salt = secrets.token_bytes(_SALT_BYTES)
    key = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P,
                         dklen=_DKLEN, maxmem=_MAXMEM)
    b = lambda x: base64.b64encode(x).decode()   # noqa: E731
    return f"scrypt${_N}${_R}${_P}${b(salt)}${b(key)}"


def verify(password: str, stored: str) -> bool:
    """Does `password` match this digest? Constant-time, never raises.

    The COST COMES FROM THE STORED HASH, not from the constants above, so
    raising _N later leaves every existing password working — it just verifies
    at the cost it was created with until it is next changed.

    A malformed or unknown-scheme digest is False rather than an exception: this
    is on the login path, and the difference between "wrong password" and "your
    user record is corrupt" is not something to leak to an unauthenticated
    caller. The CLI's `list` is where a bad record should be noticed.
    """
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        want = base64.b64decode(key)
        got = hashlib.scrypt(
            (password or "").encode(), salt=base64.b64decode(salt),
            n=int(n), r=int(r), p=int(p), dklen=len(want),
            maxmem=128 * int(r) * int(n) * 2)
    except Exception:  # noqa: BLE001 — see the docstring
        return False
    return hmac.compare_digest(want, got)


# ------------------------------------------------------------------- store ---

def _read() -> dict:
    try:
        d = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d.get("users") or {} if isinstance(d, dict) else {}


def _write(users: dict) -> None:
    """Atomic, and 0600 BEFORE the rename.

    Same reasoning as env_file.set_key: a file full of password digests must
    never exist at the final path with default permissions, not even for the
    width of one syscall.
    """
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"users": users}, indent=1, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=USERS_PATH.parent, prefix=".users.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, 0o600)
        os.replace(tmp, USERS_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def usernames() -> list[str]:
    return sorted(_read())


def count() -> int:
    return len(_read())


def record(username: str) -> dict | None:
    """This account WITHOUT its digest — safe to return from an endpoint.

    NORMALISED, like every other lookup here. It was not, and that was a real
    bug rather than untidiness: check() lower-cases before matching, so signing
    in as "Mongo" SUCCEEDS and the very next call — record("Mongo"), which
    /api/me makes — returned None. routes.me reads None as "this account no
    longer exists" and ends the session, so you would log in and be thrown
    straight back out, with the login itself having worked.
    """
    username = (username or "").strip().lower()
    u = _read().get(username)
    if u is None:
        return None
    return {"username": username, "created": u.get("created"),
            "must_change": bool(u.get("must_change")),
            "admin": bool(u.get("admin")),
            "last_login": u.get("last_login")}


def is_admin(username: str) -> bool:
    """May this account manage other accounts? The ONLY reader of the flag.

    Absent means False, which is the safe direction and is also correct for
    every record written before the flag existed — except the bootstrap account,
    which sets it explicitly, so a fresh install is never left with nobody able
    to add a user.
    """
    return bool((_read().get((username or "").strip().lower()) or {}).get("admin"))


def admins() -> list[str]:
    return sorted(u for u, r in _read().items() if r.get("admin"))


def _check_name(username: str) -> str:
    username = (username or "").strip().lower()
    if not _NAME.match(username):
        raise UserError(f"{username!r} is not a usable username — lower-case "
                        "letters, digits, dot, dash or underscore (2-32)")
    return username


def add(username: str, password: str, must_change: bool = False,
        admin: bool = False) -> str:
    """Create an account. USERNAMES ARE UNIQUE — a collision is refused, never
    merged or suffixed, because silently landing on someone else's account is
    the worst outcome available here."""
    username = _check_name(username)
    users = _read()
    if username in users:
        raise UserError(f"{username!r} already exists — use `passwd` to change "
                        f"its password, or pick another name")
    users[username] = {"hash": hash_password(password), "created": time.time(),
                       "must_change": must_change, "admin": admin}
    _write(users)
    return username


def set_password(username: str, password: str) -> None:
    username = _check_name(username)
    users = _read()
    if username not in users:
        raise UserError(f"no such user {username!r}")
    users[username]["hash"] = hash_password(password)
    users[username]["must_change"] = False
    _write(users)


def rename(old: str, new: str) -> str:
    """Change a username, keeping the same password, history and OWNED PROJECTS.

    The record MOVES rather than being recreated, so the digest, the created
    stamp and last_login travel with it — a rename is not a new account, and
    making it one would silently reset the password.

    IT REWRITES `Project.owner` TOO, and that is not a nicety. The owner field
    stores a username, so a rename without this leaves every project owned by a
    name that no longer exists — and since an owned project refuses everyone but
    its owner, renaming yourself would lock you out of your own work with no way
    back through the UI. Done in the same call because a half-applied rename is
    worse than either state.
    """
    old = (old or "").strip().lower()
    new = _check_name(new)
    users = _read()
    if old not in users:
        raise UserError(f"no such user {old!r}")
    if new == old:
        return new
    if new in users:
        raise UserError(f"{new!r} already exists — usernames are unique")
    users[new] = users.pop(old)
    _write(users)
    # Lazy import: `project` reaches the dataset layer, and this module is on the
    # login path where a heavy import chain is a cost paid per process start.
    from ... import project as project_mod
    moved = []
    for slug in project_mod.slugs():
        try:
            p = project_mod.load(slug)
        except Exception:  # noqa: BLE001 — an unreadable manifest is not this
            continue       # operation's problem, and must not half-abort it
        if p.owner == old:
            project_mod.save(replace(p, owner=new))
            moved.append(slug)
    if moved:
        print(f"[users] {old!r} -> {new!r}; re-owned {', '.join(moved)}",
              flush=True)
    return new


def remove(username: str, force: bool = False) -> bool:
    """Delete an account. Refuses to remove the last ADMIN.

    THE GUARD LIVES HERE, NOT IN THE CALLER. It used to be an `if` in the CLI,
    which was fine while the CLI was the only caller — and stopped being fine the
    moment an admin could delete users over HTTP, because a guard only one caller
    consults is decoration. This repo has paid for that exact shape twice (the
    GPU gate that lived in one of two drivers, and a check that asserted a
    function was MENTIONED rather than calling it).

    Last ADMIN, not last account: a machine with three ordinary users and nobody
    who can add a fourth is locked just as thoroughly, and the recovery is the
    same ssh session. `force` exists because someone with shell can delete the
    file anyway — it is a guardrail, not a lock.
    """
    username = (username or "").strip().lower()
    users = _read()
    if username not in users:
        return False
    if not force and users[username].get("admin") and len(admins()) <= 1:
        raise UserError(
            f"{username!r} is the only admin — removing it leaves nobody able "
            f"to add a user, and the fix would be an ssh session. Promote "
            f"someone else first.")
    del users[username]
    _write(users)
    return True


def check(username: str, password: str) -> bool:
    """The auth primitive. False for an unknown user, at the same cost.

    A MISSING USER STILL PAYS FOR A HASH. Returning early would make "no such
    user" measurably faster than "wrong password", which is a username oracle —
    and on a single-operator system, confirming the operator's username is most
    of the work. The dummy verify below is against a real digest so the timing
    matches.
    """
    u = _read().get((username or "").strip().lower())
    if u is None:
        verify(password, _timing_dummy())
        return False
    return verify(password, u.get("hash", ""))


_DUMMY: str | None = None


def _timing_dummy() -> str:
    """A real digest of a random string — never matches, costs the same.

    LAZY, not a module constant. Built at import it charged every process that
    so much as imports this module ~0.15s of scrypt, including the CLI's `list`
    and anything that only wanted `record()`. Built here it is paid once, by the
    first failed login, which is exactly who it exists for.
    """
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_password(secrets.token_urlsafe(24))
    return _DUMMY


def touch_login(username: str) -> None:
    users = _read()
    if username in users:
        users[username]["last_login"] = time.time()
        _write(users)


# --------------------------------------------------------------- bootstrap ---

def bootstrap() -> str | None:
    """Create the first account from the environment, if there are none.

    Returns the username created, or None. See the module docstring for why this
    is safe ONLY because of the emptiness test: an env var that could overwrite
    an existing account would be a permanent reset backdoor.

    Reads .env as well as the process environment, because that is what every
    other reader here does (env_file.has) — the bots call load_dotenv(), the web
    process never has.
    """
    if count():
        return None
    from .. import env_file
    user = env_file.public(BOOTSTRAP_USER_ENV)
    # Through the NAMED exception rather than env_file's private parser:
    # `public()` documents itself as being for values that are NOT secrets, and
    # a password is not one of those. See env_file.bootstrap_secret.
    pw = env_file.bootstrap_secret(BOOTSTRAP_PASS_ENV)
    if not user or not pw:
        return None
    try:
        # ADMIN, necessarily: this is the only account on the machine, and one
        # that cannot add a second is a dead end on a fresh install.
        name = add(user, pw, must_change=True, admin=True)
    except UserError as e:
        # LOUD, and without the value. A typo'd bootstrap that fails silently
        # leaves a cockpit nobody can enter and no reason on screen.
        print(f"[users] {BOOTSTRAP_USER_ENV}/{BOOTSTRAP_PASS_ENV} could not "
              f"create the first account: {e}", flush=True)
        return None
    print(f"[users] created the first account {name!r} from "
          f"{BOOTSTRAP_USER_ENV} — change this password, then remove both "
          f"variables from .env", flush=True)
    return name


# --------------------------------------------------------------------- CLI ---

def _prompt(confirm: bool = True) -> str:
    """A password, twice, off the terminal. Never from argv — see the header."""
    import getpass
    pw = getpass.getpass("password: ")
    if confirm and pw != getpass.getpass("again: "):
        raise SystemExit("[users] the two entries did not match")
    return pw


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Cockpit accounts. Passwords are prompted, never arguments.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="create an account")
    s.add_argument("username")
    s.add_argument("--admin", action="store_true",
                   help="may add and remove other accounts")

    s = sub.add_parser("passwd", help="change an account's password")
    s.add_argument("username")

    s = sub.add_parser("rename", help="change a username (keeps the password)")
    s.add_argument("old")
    s.add_argument("new")

    s = sub.add_parser("remove", help="delete an account")
    s.add_argument("username")
    s.add_argument("--force", action="store_true",
                   help="allow removing the last admin")

    sub.add_parser("list", help="who exists (never shows a hash)")

    a = ap.parse_args()
    try:
        if a.cmd == "list":
            rows = [record(u) for u in usernames()]
            if not rows:
                print("no accounts — create one with `add`, or set "
                      f"{BOOTSTRAP_USER_ENV}/{BOOTSTRAP_PASS_ENV} in .env")
                return
            for r in rows:
                when = time.strftime("%Y-%m-%d", time.localtime(r["created"] or 0))
                print(f"  {r['username']:<20} created {when}"
                      + ("  ADMIN" if r["admin"] else "")
                      + ("  MUST CHANGE PASSWORD" if r["must_change"] else ""))
        elif a.cmd == "add":
            name = add(a.username, _prompt(), admin=a.admin)
            print(f"[users] created {name!r}{' (admin)' if a.admin else ''} "
                  f"-> {USERS_PATH}")
        elif a.cmd == "passwd":
            set_password(a.username, _prompt())
            print(f"[users] password changed for {a.username!r}")
        elif a.cmd == "rename":
            print(f"[users] {a.old!r} -> {rename(a.old, a.new)!r}")
        elif a.cmd == "remove":
            print(f"[users] removed {a.username!r}"
                  if remove(a.username, force=a.force)
                  else f"[users] no such user {a.username!r}")
    except UserError as e:
        raise SystemExit(f"[users] {e}")


if __name__ == "__main__":
    main()
