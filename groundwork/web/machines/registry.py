"""Machines added at runtime — the extension point pairing writes."""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass

from ...config import OUTPUTS_DIR, PRIVATE_DIR
from .model import MACHINES, Machine  # noqa: F401

# ======================================================================== #
# THE REGISTRY — machines added at runtime, alongside the built-in two.
# ======================================================================== #
#
# MACHINES above is a CLOSED SET, and its docstring says why: "a request picks
# WHICH machine, never a URL. Otherwise 'train over there' would be an endpoint
# that POSTs a job to an arbitrary host." That property is kept exactly.
#
# What changes is that the set can be ADDED TO — deliberately, once, through a
# guarded endpoint that writes a file — rather than only by editing Python. A
# request still names a KEY and the URL is resolved here from stored config, so
# no request ever supplies a destination. Same shape as web/bot_roles.py: a
# closed set of things, extended by a commit or by an admin, never by a caller.

# PRIVATE, not outputs/: this file holds other machines' API keys, and
# outputs/ is the HTTP-served StaticFiles tree — there, any signed-in
# account (a read-scope key included) could download every worker's
# train-scope credential.
REGISTRY_PATH = PRIVATE_DIR / "machines_registry.json"
_LEGACY_REGISTRY = OUTPUTS_DIR / "machines_registry.json"


def _migrate_legacy() -> None:
    """One-time move out of the served tree. Runs on every read, does work
    only while the old file still exists and the new one does not."""
    if REGISTRY_PATH.exists() or not _LEGACY_REGISTRY.exists():
        return
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LEGACY_REGISTRY.replace(REGISTRY_PATH)
        REGISTRY_PATH.chmod(0o600)
    except OSError:
        pass                     # next read retries; worst case is status quo


def _registry() -> dict:
    _migrate_legacy()
    try:
        d = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d.get("machines") or {} if isinstance(d, dict) else {}


def _write_registry(d: dict) -> None:
    """0600 before the rename — it holds an API key for another machine."""
    import tempfile
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"machines": d}, indent=1, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=REGISTRY_PATH.parent, prefix=".machines.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, 0o600)
        os.replace(tmp, REGISTRY_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def all_machines() -> dict[str, Machine]:
    """Built-ins plus registered ones. Registered cannot shadow a built-in.

    A registered entry overriding `here` or `the worker` would let someone point
    an existing name at a different host — every saved reference to that key
    would then mean something else, silently. Refused at registration; ignored
    here as a backstop.
    """
    out = dict(MACHINES)
    for key, r in _registry().items():
        if key in MACHINES:
            continue
        out[key] = Machine(key=key, name=r.get("name") or key,
                           what=r.get("what") or "added by you",
                           url=r.get("url"),
                           role=r.get("role") or "worker",
                           trains=True,
                           verified=bool(r.get("verified")))
    return out


def workers() -> dict[str, Machine]:
    """Registered, verified machines with role=worker — the iteration source
    for the mirror, the adopt scan and the dashboard's remote panels."""
    return {k: m for k, m in all_machines().items()
            if not m.local and m.role == "worker" and m.verified}


def get(key: str) -> Machine | None:          # noqa: F811 — replaces the above
    """A machine by key, from the built-ins OR the registry."""
    return all_machines().get((key or "").strip())


def registered(key: str) -> dict:
    """The stored record — includes the API key, so NEVER return this to HTTP."""
    return dict(_registry().get(key) or {})


def add_machine(key: str, name: str, url: str, api_key: str = "",
                ssh_host: str = "", remote_root: str = "",
                role: str = "worker", transport: str = "ssh",
                ssh_port: int = 22) -> dict:
    if key in MACHINES:
        raise ValueError(f"{key!r} is a built-in machine name")
    d = _registry()
    d[key] = {"name": name, "url": url.rstrip("/"), "api_key": api_key,
              "ssh_host": ssh_host, "remote_root": remote_root,
              "role": role, "transport": transport, "ssh_port": int(ssh_port),
              "backup_target": False, "verified": None,
              "added": time.time()}
    _write_registry(d)
    return d[key]


def update_machine(key: str, **fields) -> dict:
    """Merge fields into a registered machine's record (registry only)."""
    d = _registry()
    if key not in d:
        raise KeyError(f"no registered machine {key!r}")
    d[key].update(fields)
    _write_registry(d)
    return d[key]


def mark_verified(key: str) -> None:
    update_machine(key, verified=time.time())


def set_backup_target(key: str, on: bool) -> None:
    update_machine(key, backup_target=bool(on))


def remove_machine(key: str) -> bool:
    d = _registry()
    if key not in d:
        return False
    del d[key]
    _write_registry(d)
    return True


def public_registry() -> list[dict]:
    """Every registered machine WITHOUT its API key — safe for an endpoint.

    Same rule as web/bots.py and env_file: there is no function here that hands
    a secret to a caller, so no endpoint can grow one by accident. `has_key` is
    the boolean a UI actually needs.
    """
    out = []
    for key, r in sorted(_registry().items()):
        out.append({"key": key, "name": r.get("name"), "url": r.get("url"),
                    "ssh_host": r.get("ssh_host"),
                    "remote_root": r.get("remote_root"),
                    "role": r.get("role") or "worker",
                    "transport": r.get("transport") or "ssh",
                    "backup_target": bool(r.get("backup_target")),
                    "verified": r.get("verified"),
                    "has_key": bool(r.get("api_key")), "added": r.get("added")})
    return out


def api_key_for(key: str) -> str:
    """The credential to present when calling machine `key`, or "".

    ONE PLACE ANSWERS THIS. HQ talks to another cockpit from three code paths —
    lab_proxy's cached reads, train_dispatch's dispatch, and the registry probe —
    and each growing its own idea of where the key lives is how two of them end
    up sending nothing. The registry is the one home; there is no env fallback.
    """
    r = _registry().get(key) or {}
    return r.get("api_key") or ""


def auth_headers(key: str) -> dict:
    """`{Authorization: Bearer …}` for machine `key`, or {} if we have no key.

    EMPTY IS A REAL ANSWER, not a failure: a machine running without the gate
    (GW_AUTH=0, which is how the fleet runs until every box is deployed to)
    answers perfectly well with no header, and sending a bogus one would be worse
    than sending none.
    """
    k = api_key_for(key)
    return {"Authorization": f"Bearer {k}"} if k else {}


