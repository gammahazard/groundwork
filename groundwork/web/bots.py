"""What Telegram bots exist, whether they are alive, and which project they feed.

Data reaches this system almost entirely through Telegram — the counting bot is
how images arrive. Bots were
invisible from the cockpit: to find out whether the bot that feeds your dataset
was even running, you opened an ssh session and asked systemd.

WHAT IS DELIBERATELY NOT HERE: the token. A bot's token is the bot — anyone
holding it can read every message sent to it and post as it. This module reports
`token_set` as a boolean and never returns, logs, or echoes the value, so the
token cannot leak through an endpoint that is bound to 0.0.0.0 and reachable
from a phone. `probe()` sends it to Telegram and returns only what Telegram
says back, which is public information about the bot.

LIVENESS IS TWO DIFFERENT FACTS and the panel shows both, because they fail
apart. `unit` is what systemd thinks (running / stopped / crashed). `last_seen`
is the bot's own log mtime — a bot whose unit is "active" but whose log has not
moved in a day is polling Telegram fine and receiving nothing, which is a real
and otherwise-silent failure of a data pipeline.

On calling systemctl at all: the hard rule is about `nvidia-smi`, which
under WSL2 goes through the paravirtualised /dev/dxg and can wedge in
uninterruptible D-state. `systemctl --user` talks to the user's own systemd over
a unix socket, cannot touch the GPU, and is already used this way by
state.py::_mirror_health. It still gets a short timeout and a bare except,
because a status panel must never be the thing that hangs the cockpit.
"""
from __future__ import annotations

import os
import subprocess

from . import safe_proc
from pathlib import Path

from ..config import OUTPUTS_DIR
from .. import project

ROOT = Path(__file__).resolve().parents[2]      # groundwork/web/ -> repo root

# WHERE THE LIST LIVES: each PROJECT's manifest (groundwork/project.py), not
# here. It was a module constant with `project: DEFAULT_SLUG` stamped on every
# entry, which is the shape of "the two bots this repo happens to ship" rather
# than "one bot per project" — a second project could never have one without
# editing Python. the first project's two moved into its manifest by forward-fill, so
# nothing changed for it and nothing had to be migrated by hand.
#
# A bot entry names a systemd unit, an env var, and a log file. It NEVER names a
# token; see the module docstring.


def _for(p) -> list[dict]:
    """One project's declared bots, tagged with whose they are."""
    return [{**b, "project": p.slug} for b in p.bots]


def _safe_load(slug: str):
    """A project, or an empty stand-in if its manifest will not parse.

    A broken manifest must not take the whole Data collection panel down with
    it — /api/projects already reports a bad manifest rather than dropping the
    project, and this is the same rule for the same reason.
    """
    try:
        return project.load(slug)
    except Exception:  # noqa: BLE001
        return project.Project(slug=slug, name=slug, dataset_root=ROOT)


def _token(name: str) -> str:
    """This bot's token, from the environment or from .env.

    BOTH sources, and the .env one is why this function exists. The bots call
    `load_dotenv()` at startup; the WEB process never has, so reading os.environ
    alone reported `token_set: false` for two bots that were running perfectly —
    a panel confidently saying "not configured" about a working bot, which is
    worse than no panel.

    `dotenv_values` parses the file WITHOUT mutating os.environ. That matters:
    this is a status endpoint, and a status check that silently injects secrets
    into the process it runs in is a side effect nobody asked for.
    """
    v = os.environ.get(name, "").strip()
    if v:
        return v
    try:
        from dotenv import dotenv_values
        # ENV_PATH, not ROOT/.env: with GW_DATA_DIR set the cockpit WRITES the
        # file under the data dir, so reading the repo copy reported
        # token_set:false for bots that were running fine.
        from ..config import ENV_PATH
        return (dotenv_values(ENV_PATH).get(name) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _unit_state(unit: str) -> str:
    """active / inactive / failed / activating..., or "unknown" if we cannot ask.

    `is-active` exits non-zero for anything that is not active, which is not an
    error here — the word on stdout is the answer either way.
    """
    # Supervisor mode answers from memory — no subprocess at all on this
    # POLLED path.
    from . import botsup
    s = botsup.state(unit)
    if s is not None:
        return s
    # safe_proc: subprocess.run's timeout does not bound a wedged child.
    r = safe_proc.run(["systemctl", "--user", "is-active", unit], timeout=5)
    return (r.stdout or r.stderr).strip() or "unknown"


def _log_mtime(name: str) -> float | None:
    p = OUTPUTS_DIR / name
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def status(slugs: str | list[str] | None = None) -> list[dict]:
    """Live state for these projects' bots. None still means every project.

    A LIST, not one slug, because the callers that needed "more than one but not
    all" had no way to say so and reached for None — which on a machine with two
    accounts means reporting bots the caller may not know exist. Filtering the
    RESULT would be the obvious fix and is the wrong one: every row here costs a
    `systemctl is-active` subprocess, so a filter after the fact spawns one per
    bot on the box, per poll, to throw most of them away.
    """
    if slugs is None:
        want = list(project.slugs())
    elif isinstance(slugs, str):
        want = [slugs]
    else:
        want = list(slugs)
    bots = [b for s in want for b in _for(_safe_load(s))]
    out = []
    for b in bots:
        out.append({
            "key": b["key"], "name": b["name"], "what": b["what"],
            "unit": b["unit"], "project": b["project"],
            "extension": b.get("extension"),
            # WHICH installable job this is (web/bot_roles.py). None means the
            # entry predates roles or names one this build does not ship — the
            # panel offers re-registration rather than an Install button that
            # would 400.
            "role": b.get("role"),
            "installed": (Path.home() / ".config" / "systemd" / "user"
                          / b["unit"]).exists(),
            # The NAME of the variable, so the UI can tell you what to set —
            # never its value.
            "token_env": b["token_env"],
            "token_set": bool(_token(b["token_env"])),
            # WHO MAY TALK TO IT. Telegram user ids, which are not secrets —
            # you hand yours to a bot by messaging it. Shown so an unclaimed bot
            # reads as unclaimed rather than as one that is ignoring you.
            "allowed_ids": [str(i) for i in (b.get("allowed_ids") or [])],
            "state": _unit_state(b["unit"]),
            "last_seen": _log_mtime(b["log"]),
            "log": b["log"],
        })
    return out


def probe(b: dict) -> dict:
    """Ask Telegram who this token belongs to. Explicit, never polled.

    This is the one honest test that a bot is correctly configured: a token can
    be present, well-formed and revoked. getMe answers in one round trip, and
    everything it returns is public (the bot's own @username and display name),
    so none of it is a secret being echoed back.

    TAKES THE ENTRY, not a key. It used to look the key up across every project,
    which meant any signed-in account could confirm any other account's token
    was live. Resolving the bot is the caller's job now, and the caller is
    `bot_deps.owned_bot`.
    """
    tok = _token(b.get("token_env") or "")
    if not tok:
        return {"ok": False, "error": f"{b['token_env']} is not set"}
    try:
        import httpx
        r = httpx.get(f"https://api.telegram.org/bot{tok}/getMe", timeout=10)
        d = r.json()
    except Exception as e:  # noqa: BLE001
        # The token can appear in an httpx error's request URL. Report the
        # exception TYPE only — never str(e).
        return {"ok": False, "error": f"could not reach Telegram ({type(e).__name__})"}
    if not d.get("ok"):
        # Telegram's own description is safe (it does not echo the token) but
        # keep it short and never include the URL.
        return {"ok": False, "error": str(d.get("description", "rejected"))[:120]}
    me = d.get("result", {})
    return {"ok": True, "username": me.get("username"),
            "name": me.get("first_name"), "id": me.get("id")}
