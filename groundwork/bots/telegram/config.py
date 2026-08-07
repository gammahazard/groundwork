"""Bot configuration: secrets from env/.env, paths, logging, error handler.

Secrets are never hard-coded — they come from the environment / .env. WHICH
variable holds this bot's token is itself configuration, because a machine can
run more than one counting bot:

  GW_<PROJECT>_COUNTER_TOKEN=...     # from @BotFather, one per bot

WHICH PROJECT this bot feeds, WHICH token is its own, WHO may talk to it and
WHICH log it writes all come from its unit environment (web/bot_units.py
writes it, web/bot_roles.py validates it). Not CLI flags, because a bot is
started by its unit or the supervisor, never by hand. There is NO fallback
token variable and no fallback project: a bot that cannot name its own token
refuses to start rather than long-polling as some other bot (see
groundwork/bots/env.py — stdlib-only, shared, imports nothing from here).
"""
from __future__ import annotations
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import ContextTypes

from groundwork.bots import env as botenv
# The SAME layout answers as the web tier — including the GW_DATA_DIR
# seam, so a Docker bot writes where the cockpit reads. Rebuilding ROOT
# from __file__ here was both a wrong depth and a second opinion.
from groundwork.config import ENV_PATH, OUTPUTS_DIR, ROOT

load_dotenv(ENV_PATH)

# WHICH BOT AM I. TOKEN_VAR matters as much as TOKEN: this used to read the
# literal TELEGRAM_BOT_TOKEN, so a SECOND counting bot — its own token verified
# and stored — long-polled Telegram as the FIRST bot. Two processes on one
# getUpdates stream split the messages between them, so the owner's images would
# be counted by someone else's process, with someone else's model, into someone
# else's project. `token_env` was on the bot record the whole time and nothing
# read it. REFUSAL is set when the unit does not name its own token;
# main() exits on it rather than starting and stealing updates.
TOKEN, TOKEN_VAR, REFUSAL = botenv.token()
# The project this bot collects FOR. One bot per project is the design; without
# this every bot on the machine would write into the default project's dataset,
# which is silent wrong-data — this repo's most expensive failure.
PROJECT = botenv.project_slug()

_PP = None


def pp():
    """THIS bot's project layout. Resolved once, on first use.

    Every collect/labelio call in this package passes it. GW_PROJECT is
    REQUIRED: a bot with no project would have to guess whose dataset its
    images belong to, and a wrong guess is the silent-wrong-data failure this
    repo treats as its most expensive. The cockpit-installed unit always
    writes GW_PROJECT; a hand-run bot must set it too.

    LAZY, because importing config must not touch the filesystem.
    """
    global _PP
    if _PP is None:
        if not PROJECT:
            raise SystemExit(
                "[bot] GW_PROJECT is not set — a bot always collects FOR a "
                "project. Install it from the cockpit's Data collection tab, "
                "or set GW_PROJECT=<slug> in the unit/environment.")
        from groundwork.dataset import paths as _paths
        _PP = _paths.for_project(PROJECT)
        log.info("collecting for project %s (%s)", _PP.slug, _PP.root)
    return _PP
# WHO MAY TALK TO ME. A SET — one id per bot would reproduce, one level down,
# the single machine-wide id that stopped a second person having a bot at all,
# and a workspace has more than one technician on it.
#
# EMPTY MEANS NOBODY, deliberately, and never "whoever owns the machine": the
# machine-wide fallback id was consulted only by a unit that
# is unclaimed. A new user's freshly installed bot has
# nobody named yet — they cannot learn their own id until it answers /whoami —
# so inheriting the owner would hand them a bot that obeys someone else.
ALLOWED_IDS: frozenset[int] = botenv.allowed_ids()

# Who to ping once we're back after a /restart (survives the os.execv).
RESTART_FLAG = OUTPUTS_DIR / "restart_notify.json"

log = logging.getLogger("the counting bot")


def setup_logging() -> None:
    """Timestamped rotating log to outputs/bot.log (survives reboots) + stdout
    (captured by systemd/journald). Call once at startup."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    # PER BOT. Two counting bots sharing one RotatingFileHandler on one file
    # interleave their lines and rotate each other's; worse, the cockpit reads
    # this file's mtime to answer "last activity", so one bot's traffic would
    # report as activity on a bot nobody has ever messaged.
    fileh = RotatingFileHandler(OUTPUTS_DIR / botenv.log_name("bot.log"),
                                maxBytes=2_000_000,
                                backupCount=5, encoding="utf-8")
    fileh.setFormatter(fmt)
    consoleh = logging.StreamHandler(sys.stdout)
    consoleh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fileh)
    root.addHandler(consoleh)
    # httpx logs every Telegram poll at INFO — too noisy; quiet it down.
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log any exception raised inside a handler instead of losing it."""
    log.error("handler error", exc_info=context.error)
