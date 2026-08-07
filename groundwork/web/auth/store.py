"""Where a credential store lives, and how it got there.

ONE PLACE DECIDES, because the answer has already been wrong once. users.json,
sessions.json, api_keys.json and audit.jsonl were all created under outputs/ —
which apps/webui.py mounts as StaticFiles — so every one of them was served over
HTTP to anyone the gate let through. Measured 2026-08-05: a signed-in NON-ADMIN
could GET /outputs/users.json and read every account's scrypt digest.

File permissions did not save it and could not: the files are 0600 and the
server process owns them, so the only party the permissions exclude is the one
party that was handing them out.

MIGRATION IS AUTOMATIC AND ONE-WAY. An existing install has live sessions and
keys in the old location; moving them by hand would sign everyone out and break
whatever is holding a key. path() moves the file the first time it is asked for,
with os.replace so there is no window where it exists in neither place, and only
when the destination does not already exist — a half-migrated pair must never
silently prefer the stale one.
"""
from __future__ import annotations

import os
from pathlib import Path

from ...config import OUTPUTS_DIR, PRIVATE_DIR


def path(name: str) -> Path:
    """The private path for `name`, migrating an outputs/ copy if one is there."""
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    dst = PRIVATE_DIR / name
    src = OUTPUTS_DIR / name
    if not dst.exists() and src.exists():
        try:
            os.replace(src, dst)          # atomic within one filesystem
            os.chmod(dst, 0o600)
            print(f"[auth] moved {name} out of the served outputs tree -> {dst}",
                  flush=True)
        except OSError as e:              # a failed move must not break login
            print(f"[auth] could NOT move {name} to {dst}: {e}", flush=True)
            return src
    return dst
