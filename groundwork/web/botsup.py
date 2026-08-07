"""The bot process SUPERVISOR — how bots run where systemd does not exist.

Native installs keep the systemd path (bot_units writes a user unit; systemd
restarts it). In a container there is no user systemd, so with
GW_PROCESS_MODEL=supervisor the app process itself parents each enabled bot:
Popen + restart-with-backoff, mirroring the unit's Restart=always/RestartSec
semantics closely enough that the bots panel's status words stay meaningful.

The SECURITY BOUNDARY DOES NOT MOVE. What may run is still the closed role
table (bot_roles.ROLES — module paths are literals there), and the
environment block is built by the SAME validated builder the unit writer uses
(bot_units.environment), so GW_PROJECT/GW_TOKEN_ENV/GW_ALLOWED_IDS semantics
are identical in both process models. "Install" here is just the manifest's
`enabled` flag plus a child process — no file is written into systemd's tree.

Selection: mode() reads GW_PROCESS_MODEL, defaulting by detection — a
reachable user systemd means "systemd", anything else means "supervisor".
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

from ..config import OUTPUTS_DIR, ROOT
from . import bot_roles, bot_units

_BACKOFF_MIN, _BACKOFF_MAX = 5.0, 300.0

_lock = threading.Lock()
_procs: dict[str, subprocess.Popen] = {}
_state: dict[str, str] = {}          # unit -> active|backoff|stopped|failed
_stop: set[str] = set()


_mode_cache: str | None = None


def mode() -> str:
    """systemd or supervisor. CACHED after the first real answer — the
    process model cannot change under a running service, and this is
    reachable from the polled Data-collection page, where shelling out
    per poll is the exact failure safe_proc exists to prevent."""
    global _mode_cache
    forced = (os.environ.get("GW_PROCESS_MODEL", "") or "").strip()
    if forced in ("systemd", "supervisor"):
        return forced
    if _mode_cache is None:
        from . import safe_proc
        r = safe_proc.run(["systemctl", "--user", "is-system-running"],
                          timeout=5)
        # "degraded" still means a live user manager; only a hard failure
        # to talk to it means there is nothing to hand units to.
        _mode_cache = "systemd" if (r.stdout or "").strip() else "supervisor"
    return _mode_cache


def state(unit: str) -> str | None:
    """The supervisor's answer to `is-active`, or None in systemd mode."""
    if mode() != "supervisor":
        return None
    with _lock:
        return _state.get(unit, "inactive")


def _runner(p, bot: dict, role) -> None:
    unit = bot_units.unit_of(p, bot)
    interp = sys.executable
    backoff = _BACKOFF_MIN
    while unit not in _stop:
        env = dict(os.environ)
        env.update(bot_units.environment(p, bot))
        try:
            child = subprocess.Popen(
                [interp, "-m", role.module], cwd=str(ROOT), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            with _lock:
                _state[unit] = f"failed ({type(e).__name__})"
            return
        with _lock:
            _procs[unit] = child
            _state[unit] = "active"
        started = time.time()
        child.wait()
        if unit in _stop:
            break
        # A child that lived a while earns a fresh backoff; a crash-loop
        # backs off up to five minutes, same shape as StartLimit semantics.
        backoff = _BACKOFF_MIN if time.time() - started > 60 \
            else min(backoff * 2, _BACKOFF_MAX)
        with _lock:
            _state[unit] = "backoff"
        time.sleep(backoff)
    with _lock:
        _state[unit] = "inactive"
        _procs.pop(unit, None)


def start(p, bot: dict) -> None:
    role = bot_roles.get(bot.get("role"))
    if role is None:
        raise bot_units.UnitError(f"unknown role {bot.get('role')!r}")
    unit = bot_units.unit_of(p, bot)
    _stop.discard(unit)
    with _lock:
        if _state.get(unit) == "active":
            return
    threading.Thread(target=_runner, args=(p, bot, role), daemon=True).start()


def stop(p, bot: dict) -> None:
    unit = bot_units.unit_of(p, bot)
    _stop.add(unit)
    with _lock:
        child = _procs.get(unit)
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()


def restart(p, bot: dict) -> None:
    stop(p, bot)
    time.sleep(0.5)
    start(p, bot)


def boot_enabled(projects) -> int:
    """Start every enabled bot at app startup (supervisor mode only).
    Returns how many were started. Failures are per-bot, never fatal."""
    if mode() != "supervisor":
        return 0
    n = 0
    for p in projects:
        for bot in getattr(p, "bots", ()) or ():
            if not bot.get("enabled"):
                continue
            try:
                start(p, dict(bot))
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[botsup] {p.slug}/{bot.get('key')}: "
                      f"{type(e).__name__}: {e}", flush=True)
    return n
