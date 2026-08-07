#!/usr/bin/env python3
"""No credential store may live under the tree that is served over HTTP.

WHY THIS EXISTS. groundwork/web/app.py mounts the whole of outputs/ as
StaticFiles. The accounts work once put users.json, sessions.json,
api_keys.json and audit.jsonl in there, and a signed-in NON-ADMIN could GET
/outputs/users.json and read every account's scrypt digest. File permissions could not help: the files
are 0600 and the server owns them, so the only party excluded was the one party
handing them out.

The fix was structural — the stores moved to private/, which is not mounted — and
this check is what keeps it that way. It fails if any store resolves inside
OUTPUTS_DIR, and it fails if the running server will serve one.

IT MUST BE ABLE TO FAIL, which is this repo's standing rule for checks. Run with
--self-test to point a store back at outputs/ in memory and confirm the check
goes red; a check that cannot fail is decoration.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundwork.config import OUTPUTS_DIR                       # noqa: E402
from groundwork.web.auth import audit, keys, sessions, users    # noqa: E402

STORES = {
    "users.json": users.USERS_PATH,
    "sessions.json": sessions.SESSIONS_PATH,
    "api_keys.json": keys.KEYS_PATH,
    "audit.jsonl": audit.AUDIT_PATH,
}

FAIL: list[str] = []


def note(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def check_paths(force_bad: str | None = None) -> None:
    print("1. no store resolves inside the served tree")
    served = OUTPUTS_DIR.resolve()
    for name, p in STORES.items():
        path = (OUTPUTS_DIR / name) if force_bad == name else p
        inside = served == path.resolve().parent or served in path.resolve().parents
        note(not inside, f"{name} -> {path}")


def check_http() -> None:
    """And the running server agrees. A path check is about intent; this is
    about behaviour, and they have disagreed before."""
    print("\n2. the running server does not serve them")
    try:
        import urllib.error
        import urllib.request
        for name in STORES:
            url = f"http://127.0.0.1:8000/outputs/{name}"
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code
            # 401 (gate) or 404 (absent) are both fine; 200 is the bug.
            note(code != 200, f"GET /outputs/{name} -> {code}")
    except OSError as e:
        print(f"  ..   server not reachable ({e}); path check above still applies")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        print("SELF-TEST: pretending users.json is back under outputs/\n")
        check_paths(force_bad="users.json")
        print()
        print("FAILED as expected" if FAIL else "*** SELF-TEST DID NOT FAIL ***")
        raise SystemExit(0 if FAIL else 1)
    check_paths()
    check_http()
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        raise SystemExit(1)
    print("credential stores are outside the served tree")
