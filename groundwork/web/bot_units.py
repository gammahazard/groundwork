"""Writing, reloading and driving a bot's systemd unit. One copy of the dance.

THE INVARIANT THIS MODULE OWNS: a unit this cockpit writes or touches is one it
NAMED, and every value inside it has been validated.

Both halves were broken, and they broke differently:

  THE NAME. `unit` came straight off the request (`^[A-Za-z0-9@._-]+\\.service$`)
  and was checked against a DENYLIST of seven infrastructure units — which did
  not include a hand-written counting-bot unit or `a legacy unit`. So a second account
  could register a bot in its OWN project claiming the owner's unit name, then
  stop it, or install over it and repoint the owner's Telegram photos at their
  own dataset. It is an ALLOWLIST now: `gw-<slug>-<role>.service`, derived and
  never accepted, plus the two hand-written legacy names honoured ONLY in the
  default project. The denylist stays underneath as a second, redundant check.

  THE CONTENTS. See web/bot_roles.unit_text — a bot NAME containing newlines
  appended arbitrary directives to a unit systemd then ran. Values are validated
  there AND here, in two modules, on purpose.

WHY THE NAMES ARE DERIVED RATHER THAN CHECKED FOR COLLISIONS. Project slugs are
already globally unique (a slug is a directory name, `_unique_slug` guarantees
it), and a project holds at most one bot per role. So `gw-<slug>-<role>` is
unique by construction — no cross-project scan, and no 409 that would tell a
caller a bot exists in a project they cannot see.

ALWAYS THROUGH safe_proc. systemctl talks to the user's own systemd over a unix
socket and cannot touch the GPU, so this is not the /dev/dxg trap — but
`subprocess.run(timeout=)` does not bound a wedged child in any case, and this
is reached from a polled panel. See CONTRIBUTING.md's subprocess rule.
"""
from __future__ import annotations

import re

from . import bot_roles, safe_proc
from .. import project as project_mod
from ..config import ROOT

USER_UNITS = ROOT.home() / ".config" / "systemd" / "user"

# What this cockpit names. Anything else is refused, whoever asks.
DERIVED = re.compile(r"^gw-[a-z0-9][a-z0-9-]{1,54}[a-z0-9]\.service$")

# Redundant given the allowlist above, and kept anyway: if the allowlist is
# ever widened, this is the line that still refuses to restart the web server
# or the platform's own timers.
PROTECTED = {"groundwork-web.service", "groundwork-mirror.service",
             "groundwork-adopt.service", "groundwork-backup.service",
             "groundwork-clean.service", "groundwork-autoscore.service"}


class UnitError(RuntimeError):
    """A unit this module refuses to touch, or a systemctl call that failed."""


# ------------------------------------------------------------------ naming ---

def derived_unit(slug: str, role: str) -> str:
    return f"gw-{slug}-{role}.service"


def derived_log(slug: str, role: str) -> str:
    return f"{slug}-{role}.log"


def derived_token_env(slug: str, role: str) -> str:
    """The .env key holding this bot's token.

    PREFIXED, because a project slug may begin with a digit (`_SLUG_OK` is
    `^[a-z0-9]…`) and an environment variable may not. `GW_` also makes the
    namespace obvious in a .env someone is reading by hand.

    Derived rather than typed, which is what kills the overwrite: `token_env`
    used to be free text checked for uniqueness only WITHIN a project, while
    .env is one flat machine-global file — so registering a bot claiming
    TELEGRAM_BOT_TOKEN and pasting a token overwrote the owner's.
    """
    return f"GW_{slug}_{role}_TOKEN".upper().replace("-", "_")


def derived_key(slug: str, role: str) -> str:
    """A bot's stable id. One per (project, role), so it cannot collide."""
    return f"{slug}-{role}"


def unit_of(p, bot: dict) -> str:
    """The unit this bot may control. Raises UnitError for anything else."""
    unit = (bot.get("unit") or "").strip()
    if not unit:
        raise UnitError("this bot has no service name")
    if unit in PROTECTED:
        raise UnitError(f"{unit!r} is infrastructure — this cockpit will not touch it")
    if DERIVED.match(unit):
        return unit
    raise UnitError(
        f"{unit!r} is not a unit this cockpit will touch — it writes "
        f"gw-<project>-<role>.service and nothing else")


# ------------------------------------------------------------- the contents ---

def environment(p, bot: dict) -> dict[str, str]:
    """The validated Environment block for this bot.

    Everything the running process needs to know WHICH bot it is. Each value is
    checked by bot_roles.unit_text before it reaches the file; a value that
    cannot be rendered safely is dropped here rather than written raw, so a
    hand-edited manifest degrades to a bot that will not start rather than to a
    unit that runs something unexpected.
    """
    env = {"GW_PROJECT": p.slug}
    if bot.get("token_env"):
        # WHICH VARIABLE HOLDS THIS BOT'S TOKEN. Without it apps/bot/config.py
        # names no variable, so a second bot could long-poll Telegram as the
        # FIRST bot and the two would split each other's updates — one
        # person's images counted by someone else's process, with someone
        # else's model, into someone else's project.
        env["GW_TOKEN_ENV"] = str(bot["token_env"])
    if bot.get("log"):
        env["GW_BOT_LOG"] = str(bot["log"])
    ids = allowed_ids(bot)
    if ids:
        # OMITTED ENTIRELY when there are none. An empty `Environment=X=` sets
        # the variable to the empty string, which is a different thing from
        # unset — and "unset or empty" both mean the bot answers NOBODY.
        env["GW_ALLOWED_IDS"] = ",".join(ids)
    return env


def allowed_ids(bot: dict) -> list[str]:
    """This bot's owner ids, as validated digit strings. Junk is dropped."""
    raw = bot.get("allowed_ids") or []
    if isinstance(raw, (str, int)):          # tolerate a hand-edited scalar
        raw = [raw]
    out = []
    for v in raw:
        s = str(v).strip()
        if s.isdigit() and 4 <= len(s) <= 15 and s not in out:
            out.append(s)
    return out[:16]


def describe(p, bot: dict) -> str:
    """The Description line, with control characters collapsed."""
    return bot_roles.clean_description(
        f"{p.name} — {bot.get('name') or bot.get('key') or 'bot'} (Groundwork)")


def render(p, bot: dict, role) -> str:
    """The unit text for this bot. Raises ValueError on anything unsafe."""
    return bot_roles.unit_text(role, describe(p, bot), environment(p, bot))


# -------------------------------------------------------------- the systemd ---

def systemctl(*args: str) -> tuple[bool, str]:
    """Run `systemctl --user ...`. Short timeout: a status page must never hang."""
    r = safe_proc.run(["systemctl", "--user", *args], timeout=25)
    if r.abandoned:
        return False, r.stderr
    return r.returncode == 0, ((r.stderr or r.stdout).strip() or "")[:400]


def write(p, bot: dict, role) -> str:
    """Write the unit file. Returns the unit name. No systemd call."""
    unit = unit_of(p, bot)
    try:
        text = render(p, bot, role)
    except ValueError as e:
        raise UnitError(str(e)) from None
    USER_UNITS.mkdir(parents=True, exist_ok=True)
    (USER_UNITS / unit).write_text(text, encoding="utf-8")
    return unit


def install(p, bot: dict, role) -> str:
    """Write it, reload, enable and start it. Raises UnitError with the reason.

    The unit file is deliberately LEFT ON DISK when starting fails: it is what
    you read to find out why, and rewriting it away would delete the evidence.
    """
    from . import botsup
    if botsup.mode() == "supervisor":
        # No systemd here (a container, or a box with no user manager): the
        # supervisor parents the process. The role table and the validated
        # environment builder are the same; only the parent differs.
        unit = unit_of(p, bot)
        botsup.start(p, bot)
        return unit
    unit = write(p, bot, role)
    ok, msg = systemctl("daemon-reload")
    if not ok:
        raise UnitError(f"daemon-reload failed: {msg}")
    ok, msg = systemctl("enable", "--now", unit)
    if not ok:
        raise UnitError(f"{unit} would not start: {msg}")
    return unit


def refresh(p, bot: dict, role) -> bool:
    """Rewrite an ALREADY-INSTALLED unit and restart it if it was running.

    For the changes that alter a unit's environment rather than its existence —
    moving a bot to another project, setting who may use it. `try-restart`
    because a stopped bot should stay stopped: this keeps the file in step, it
    does not start anything.

    A bot with no installed unit is not an error — it picks the new value up
    when it is installed. Returns whether anything was rewritten.
    """
    if role is None:
        return False
    try:
        unit = unit_of(p, bot)
    except UnitError:
        return False
    if not (USER_UNITS / unit).exists():
        return False
    try:
        text = render(p, bot, role)
    except ValueError:
        return False
    (USER_UNITS / unit).write_text(text, encoding="utf-8")
    systemctl("daemon-reload")
    systemctl("try-restart", unit)
    return True


def installed(bot: dict) -> bool:
    """Does this bot have a unit file on disk?"""
    unit = (bot.get("unit") or "").strip()
    return bool(unit) and (USER_UNITS / unit).exists()
