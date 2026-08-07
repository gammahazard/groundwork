"""The `users` command line: add / passwd / rename / remove / list.

Split out of users.py, which is imported on every request through the auth
middleware — the CLI's argparse and getpass have no business being resident
there, and the module was over the repo's 500-line cap with them in it.

Entry point is unchanged: `python -m groundwork.web.auth.users <cmd>`
delegates here.
"""
from __future__ import annotations

import time

from .users import (BOOTSTRAP_PASS_ENV, BOOTSTRAP_USER_ENV, USERS_PATH,
                    UserError, add, record, remove, rename, set_password,
                    usernames)


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


if __name__ == "__main__":       # pragma: no cover — `-m ...auth.users` is the door
    main()
