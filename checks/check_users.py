"""Accounts: hashed, unique, and the bootstrap env is not a backdoor.

    .venv/bin/python checks/check_users.py

THE ONE THAT MATTERS is part 5. `GW_ADMIN_USER` / `_PASSWORD` create the
first account so a fresh checkout is not a brick — and if they could touch an
EXISTING account they would be a permanent password-reset backdoor for anyone
who can write `.env` or set a variable on the service. The only thing standing
between those two readings is the emptiness test in bootstrap(), so this asserts
it directly: with a user present, the env is inert, and the existing password
still works afterwards.

Everything else is the ordinary contract: a digest is not a password, usernames
are unique, a rename keeps the password AND the projects, and the last admin
cannot be removed.

HOW IT CAN FAIL: no assertion here is "today's answer". Each one changes an input
and requires the output to change with it — a wrong password must be rejected, a
duplicate name must raise, the env must be honoured when empty and ignored when
not. Deleting bootstrap()'s `if count()` makes part 5 red.

SAFE: USERS_PATH is redirected to a temp file for the whole run, so it never
reads or writes the real outputs/users.json. No network, no GPU.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


from groundwork.web.auth import users                  # noqa: E402

REAL = users.USERS_PATH
PW = "correct-horse"
OTHER = "battery-staple"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="checkusers-")) / "users.json"
    users.USERS_PATH = tmp                    # never touch the real store
    saved_env = {k: os.environ.get(k)
                 for k in (users.BOOTSTRAP_USER_ENV, users.BOOTSTRAP_PASS_ENV)}
    try:
        note(tmp != REAL, f"redirected to {tmp} (real store untouched)")

        # ---- 1. a digest is not a password ---------------------------------
        print("\n1. passwords are hashed, salted, and never stored")
        h1, h2 = users.hash_password(PW), users.hash_password(PW)
        note(h1 != h2, "the same password hashes differently twice (random salt)")
        note(PW not in h1, "the password does not appear in its own digest")
        note(users.verify(PW, h1) and users.verify(PW, h2),
             "both digests verify the right password")
        note(not users.verify(OTHER, h1), "a wrong password is rejected")
        note(not users.verify(PW, "garbage"),
             "a malformed digest is False, not an exception")

        # ---- 2. rules: 8 chars, unique names -------------------------------
        print("\n2. the rules: 8 characters, not on the guess list, unique name")
        try:
            users.hash_password("short7x")            # 7
            note(False, "a 7-character password was accepted")
        except users.UserError:
            note(True, "under 8 characters is refused")
        note(users.hash_password("abcdefgh") != "", "exactly 8 is accepted")
    for bad in ("changeme", "Change-Me", "change_me", "12345678", "PASSWORD"):
        try:
            users.hash_password(bad)
            note(False, f"guess-list password {bad!r} must be refused")
        except users.UserError:
            note(True, f"guess-list password {bad!r} is refused")
        note(users.hash_password("!!!!!!!!") != "",
             "no complexity rule — 8 identical symbols is fine")
        users.add("casey", PW, admin=True)
        try:
            users.add("casey", OTHER)
            note(False, "a duplicate username was accepted")
        except users.UserError:
            note(True, "a duplicate username is refused, not merged or suffixed")
        note(users.check("casey", PW), "the account authenticates")
        note(not users.check("casey", OTHER), "with the wrong password it does not")
        note(not users.check("nobody", PW), "an unknown user does not authenticate")

        # ---- 3. nothing readable lands on disk ------------------------------
        print("\n3. the store holds no plaintext")
        raw = tmp.read_text()
        note(PW not in raw, "the password is absent from users.json")
        note(json.loads(raw)["users"]["casey"]["hash"].startswith("scrypt$"),
             "it holds a parameterised scrypt digest")
        note(oct(tmp.stat().st_mode)[-3:] == "600",
             f"users.json is 0600 (got {oct(tmp.stat().st_mode)[-3:]})")

        # ---- 4. rename keeps the password and the projects ------------------
        print("\n4. rename moves the account, it does not recreate it")
        users.rename("casey", "jordan")
        note(users.check("jordan", PW), "the password survives the rename")
        note(not users.check("casey", PW), "the old name no longer authenticates")
        try:
            users.add("second", OTHER)
            users.rename("second", "jordan")
            note(False, "renaming onto an existing name was allowed")
        except users.UserError:
            note(True, "renaming onto a taken name is refused (names are unique)")

        # ---- 5. THE BACKDOOR TEST ------------------------------------------
        #
        # TWO INDEPENDENT BARRIERS, asserted as one PROPERTY. bootstrap() tests
        # emptiness, and add() refuses a duplicate name — so deleting either one
        # alone leaves the system safe. That is defence in depth and it is why
        # an earlier version of this check passed a mutation that removed the
        # emptiness test: it was asserting the mechanism, and the other barrier
        # held. What must be true regardless of mechanism is that bootstrap
        # NEVER MODIFIES a store that already has users — so assert the bytes.
        print("\n5. the bootstrap env cannot touch an existing account")
        os.environ[users.BOOTSTRAP_USER_ENV] = "jordan"
        os.environ[users.BOOTSTRAP_PASS_ENV] = "attacker-chosen"
        before = tmp.read_bytes()
        note(users.bootstrap() is None,
             "bootstrap() is a no-op while any user exists")
        note(tmp.read_bytes() == before,
             "it did not write to the store AT ALL (the property, not the how)")
        note(users.check("jordan", PW),
             "the real password STILL works (the env did not reset it)")
        note(not users.check("jordan", "attacker-chosen"),
             "the env-supplied password does NOT authenticate")

        # ...and it does work on an empty store, or a fresh install is a brick.
        users.USERS_PATH = tmp.with_name("empty.json")
        os.environ[users.BOOTSTRAP_USER_ENV] = "admin"
        os.environ[users.BOOTSTRAP_PASS_ENV] = "Change-Me"
        made = users.bootstrap()
        note(made is None,
             "a guess-list password ('Change-Me') is REFUSED, loudly not fatally")
        os.environ[users.BOOTSTRAP_PASS_ENV] = "quartz-heron-42"
        made = users.bootstrap()
        note(made == "admin", f"on an EMPTY store it creates the account ({made})")
        note(users.check("admin", "quartz-heron-42"), "and that account can log in")
        r = users.record("admin")
        note(r and r["admin"], "the first account is an admin (it must be able to add a second)")
        note(r and r["must_change"], "and is flagged must-change")
        users.USERS_PATH = tmp

        # ---- 6. the last admin cannot be removed ---------------------------
        print("\n6. you cannot lock yourself out")
        note(users.remove("second"), "an ordinary account can be removed")
        try:
            users.remove("jordan")
            note(False, "the last admin was removed")
        except users.UserError:
            note(True, "removing the last admin is refused")
        note(users.remove("jordan", force=True),
             "...unless forced, because shell access can delete the file anyway")
    finally:
        users.USERS_PATH = REAL
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("PASS — accounts are hashed and unique; the bootstrap env is not a backdoor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
