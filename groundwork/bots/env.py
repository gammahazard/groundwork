"""How a bot learns WHICH bot it is.

Stdlib-only, deliberately: this resolves identity before anything heavy loads,
and it must be importable by any bot entrypoint without pulling torch.

    GW_PROJECT      which project's dataset  (REQUIRED — a bot always collects
                    FOR a project; the cockpit-written unit always sets it)
    GW_TOKEN_ENV    which .env key holds my token (REQUIRED)
    GW_ALLOWED_IDS  who may talk to me
    GW_BOT_LOG      which log file is mine

THERE IS NO MACHINE-WIDE FALLBACK, and that is the point. An unconditional
"GW_ALLOWED_IDS, else some machine-wide id" would hand a NEW user's freshly
installed bot to the machine's owner — because a new bot has nobody named yet,
and the person setting it up cannot learn their own numeric id until the bot is
running to answer /whoami. A bot that names nobody is UNCLAIMED, and unclaimed
means "answers nobody" — never "belongs to whoever owns the box".

Same shape for the token, as a refusal: a unit with no GW_TOKEN_ENV would have
to guess a variable, and polling Telegram with another bot's token makes two
processes fight over one getUpdates stream — messages split between them
silently. Refusing to start is the only safe answer.
"""
from __future__ import annotations

import os
import re

# What Telegram issues. The variable NAME travels through a systemd unit built
# from a hand-editable manifest, so without this a bad entry could name, say,
# GW_ADMIN_PASSWORD — and the bot would then send its value to
# api.telegram.org/bot<that>/getMe. Refusing anything not shaped like a token
# turns an exfiltration primitive into a startup refusal.
_TOKEN_SHAPE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")
_VAR_SHAPE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")

ID_MIN, ID_MAX = 4, 15


def project_slug() -> str | None:
    """The project this bot collects for. None means misconfigured — the
    caller refuses to start (see groundwork.bots.telegram.config.pp)."""
    return (os.environ.get("GW_PROJECT", "") or "").strip() or None


def token() -> tuple[str, str, str | None]:
    """(token, variable_name, refusal). A refusal means: do not start.

    Never returns the token in the refusal, and never logs it — the caller gets
    the VARIABLE name to print, which is the useful half anyway.
    """
    var = (os.environ.get("GW_TOKEN_ENV", "") or "").strip()
    if not var:
        return "", "", (
            "this service does not say which variable holds its token "
            "(GW_TOKEN_ENV is unset). Install the bot from the cockpit "
            "(Data collection -> Install), which writes it.")
    if not _VAR_SHAPE.match(var):
        return "", var, f"{var!r} is not a usable environment variable name"
    val = (os.environ.get(var, "") or "").strip()
    if not val:
        return "", var, f"{var} is empty — paste the token on the bot's card"
    if not _TOKEN_SHAPE.match(val):
        # Shape only. This is the check that stops a mis-named variable being
        # posted to Telegram; it says nothing about whether the token is valid.
        return "", var, (f"{var} does not hold something shaped like a Telegram "
                         f"bot token")
    return val, var, None


def allowed_ids() -> frozenset[int]:
    """Who may talk to this bot. Empty means nobody — see the module docstring.

    A SET FROM THE FIRST DAY. A single id per bot would reproduce, one level
    down, a machine-wide-single-id problem: real deployments have more than one
    person sending photos.
    """
    raw = (os.environ.get("GW_ALLOWED_IDS", "") or "").strip()
    out = set()
    for part in raw.replace(";", ",").split(","):
        s = part.strip()
        if s.isdigit() and ID_MIN <= len(s) <= ID_MAX:
            out.add(int(s))
    return frozenset(out)


def log_name(default: str) -> str:
    """This bot's log file name, relative to outputs/.

    Per-bot, because two counting bots sharing one RotatingFileHandler on the
    same file interleave their lines — and because the cockpit reads this file's
    mtime to answer "last activity", which would otherwise read as activity on
    a bot that has never been messaged.
    """
    v = (os.environ.get("GW_BOT_LOG", "") or "").strip()
    # A log name reaches a filesystem path. Anything with a separator, or that
    # does not look like a plain file, falls back rather than being cleaned.
    if v and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.log", v):
        return v
    return default


def describe(token_var: str) -> str:
    """A one-line 'which bot am I' for the startup log. Never the token itself.

    The same rule every pipeline stage follows — say which tree you are about to
    touch before touching it. Silent wrong-data is this repo's most expensive
    failure, and a bot has three ways to be the wrong one at once.
    """
    ids = sorted(allowed_ids())
    who = ", ".join(str(i) for i in ids) if ids else "NOBODY (unclaimed)"
    return (f"project={project_slug() or 'UNSET'} | token from {token_var} | "
            f"allowed: {who}")
