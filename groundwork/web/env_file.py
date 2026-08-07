"""Read and write the repo's `.env`, for the settings the cockpit is allowed to own.

WHY THIS EXISTS AT ALL, since the previous answer was "it does not".
`api/bots.py::setup_steps` used to say the cockpit deliberately would not write
a token, because this server binds to 0.0.0.0 and is reachable over the network
with no auth. That reasoning was sound and the owner has overruled it
(2026-07-31): the network is theirs, every machine on it is theirs, and the
alternative was hand-editing a dotfile over ssh to add a bot. Their call, and it
is recorded here rather than quietly reversed.

WHAT IS STILL TRUE, and is why this is a module and not three inline open()s:

  NEVER READ A VALUE BACK OUT. `has()` returns a boolean and `public()` returns
  only keys that are not secrets. There is no function here that returns a token
  to a caller, so no endpoint can grow one by accident. A token goes in and only
  Telegram ever sees it again.

  ONE KEY AT A TIME, PRESERVING EVERYTHING ELSE. The file is parsed, one
  assignment is replaced or appended, and it is written back whole. A rewrite
  that dropped an unrelated line would take out a running bot, and .env is not
  in git — there is no undo.

  ATOMIC, AND 0600. Written to a temp file in the same directory and renamed, so
  a crash mid-write cannot leave a truncated .env that every service then fails
  to parse. chmod happens BEFORE the rename, so the secret is never briefly
  world-readable.

  QUOTING IS DELIBERATELY MINIMAL. python-dotenv is what reads this file, and it
  strips one layer of matching quotes. Values are written bare unless they
  contain whitespace, a quote or a hash — a Telegram token is `\\d+:[A-Za-z0-9_-]+`
  and a user id is digits, so in practice nothing here is ever quoted. Refusing
  a value that would need escaping is safer than emitting an escape the reader
  might interpret differently.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from ..config import ENV_PATH  # noqa: F401 — one home for the path

# A key we are willing to touch at all. Upper-case, because dotenv lookups are
# case-sensitive and the rest of the codebase reads upper-case names.
_KEY = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")

# Characters that would need quoting or escaping. See the docstring: rather than
# emit an escape whose reader we would have to agree with, refuse the value.
_UNSAFE = re.compile(r"[\s\"'#\\\x00-\x1f]")

# `KEY=value`, tolerating `export KEY=` and leading whitespace, which dotenv also
# accepts. Comments and blank lines match nothing and are preserved verbatim.
_ASSIGN = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=)(.*)$")


class EnvError(ValueError):
    """A key or value this module refuses to write."""


def _check(key: str, value: str) -> tuple[str, str]:
    key = (key or "").strip()
    value = (value or "").strip()
    if not _KEY.match(key):
        raise EnvError(f"{key!r} is not a usable environment variable name — "
                       "upper-case letters, digits and underscores")
    if not value:
        raise EnvError("empty value")
    if _UNSAFE.search(value):
        raise EnvError("value contains whitespace, quotes or a comment marker, "
                       "which this writer will not escape")
    return key, value


def has(key: str) -> bool:
    """Is this key set, in the process environment OR in .env?

    BOTH, and in that order, because that is what the readers do: the bots call
    load_dotenv() and then read os.environ, so a value exported into the service
    wins over the file. A status panel that only looked at .env would report
    "not configured" for a bot that is running fine — which is exactly the bug
    web/bots.py::_token was written to fix.
    """
    if os.environ.get(key, "").strip():
        return True
    return bool(_read().get(key, "").strip())


def public(key: str) -> str:
    """The VALUE of a key that is not a secret — a user id, a slug.

    Separate function from has() on purpose: reading a value out is the
    dangerous operation, so it is not something a caller does by passing a flag.
    Callers are expected to have thought about whether this key is safe to
    return, and the endpoint that uses it says so.
    """
    v = os.environ.get(key, "").strip()
    return v or _read().get(key, "").strip()


def bootstrap_secret(key: str) -> str:
    """The ONE exception to "never read a value back out", named so it cannot
    spread.

    `public()` deliberately refuses to be used for secrets, and that rule is why
    no endpoint here can grow a token-reader by accident. But the first-run
    admin password has to come from somewhere, and `.env` is where this repo
    already keeps per-machine configuration.

    So this exists, with a name that describes its ONLY legitimate caller
    (auth/users.bootstrap) rather than a generic `secret()` that would read as
    an invitation. Its value is turned into a scrypt digest and discarded inside
    that function — it is never returned to a request, logged, or echoed. Grep
    for callers before adding a second one; if there is ever a second, this was
    the wrong shape.
    """
    return os.environ.get(key, "").strip() or _read().get(key, "").strip()


def _read() -> dict[str, str]:
    """Parsed .env, or {} if there is none. Values are NOT returned to callers."""
    out: dict[str, str] = {}
    try:
        text = ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        m = _ASSIGN.match(line)
        if not m:
            continue
        v = m.group(4).strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[m.group(2)] = v
    return out


def set_key(key: str, value: str) -> None:
    """Write one assignment, preserving every other line byte for byte.

    Replaces the LAST assignment of the key if it appears more than once —
    dotenv takes the last one, so replacing an earlier duplicate would write a
    value that nothing reads and report success.
    """
    key, value = _check(key, value)
    try:
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []

    hit = -1
    for i, line in enumerate(lines):
        m = _ASSIGN.match(line)
        if m and m.group(2) == key:
            hit = i
    if hit >= 0:
        m = _ASSIGN.match(lines[hit])
        lines[hit] = f"{m.group(1)}{key}{m.group(3)}{value}"
    else:
        lines.append(f"{key}={value}")

    body = "\n".join(lines) + "\n"
    d = ENV_PATH.parent
    d.mkdir(parents=True, exist_ok=True)   # a fresh box has no config/ yet
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        # BEFORE the rename: a secret must never exist at the final path with
        # default permissions, not even for the width of one syscall.
        os.chmod(tmp, 0o600)
        os.replace(tmp, ENV_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
