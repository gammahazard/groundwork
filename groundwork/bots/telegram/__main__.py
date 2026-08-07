#!/usr/bin/env python3
"""Telegram bot front-end for object counting — thin entrypoint.

Loads the fast YOLO-nano counter ONCE (~0.1 GB VRAM), then answers photos with a
count + a point overlay in ~1 s. Just send a photo of the image — no modes, no
prompt. Autocrop and the image-hull filter are OFF (see
groundwork/serving_filters.py), so this counts exactly like the iPhone app:
model + stacked-dot dedupe, nothing else. Nothing trims the frame any more, so
FRAME THE IMAGE TIGHTLY — background junk can be counted as objects.
Owner-locked by numeric Telegram id; long polling (no public IP).

Handlers live in `apps/bot/` (config, text, state, counting, feedback, commands);
this file only wires them onto the Application and runs the poller.

Owner-locked by numeric Telegram id — a LIST of them, per bot, set in the
cockpit. Which token this process uses, which project it feeds and who may talk
to it all come from its systemd unit; see groundwork/bots/env.py. The old single
machine-wide id survives only for the hand-written original unit.

Run:
  .venv/bin/python -m groundwork.bots.telegram
"""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

# NO sys.path games here. The old layout inserted a parent directory so its
# helper module could be found; after the move that same insert put THIS
# package's parent first on the path — and this package is named `telegram`,
# so `from telegram import Update` imported *us* instead of the library, on
# every start, with a crash-loop as the only symptom.

from telegram import Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          MessageHandler, filters)

from groundwork.serve import runtime_model
from groundwork.bots import env as botenv
from groundwork.bots.telegram import config, commands, counting, feedback, state
from groundwork.bots.telegram.text import COMMANDS


async def _post_init(app: Application) -> None:
    state.GPU_LOCK = asyncio.Lock()          # created inside the bot's event loop
    await app.bot.set_my_commands(COMMANDS)
    if config.RESTART_FLAG.exists():         # a /restart asked us to confirm when back
        try:
            chat_id = json.loads(config.RESTART_FLAG.read_text())["chat_id"]
            config.RESTART_FLAG.unlink(missing_ok=True)
            await app.bot.send_message(chat_id, f"✅ Back up — serving {state.model_desc()}.")
        except Exception:  # noqa: BLE001
            config.log.warning("could not send restart-done ping", exc_info=True)


def main() -> None:
    config.setup_logging()
    # REFUSE RATHER THAN STEAL UPDATES. A unit that names a project but no token
    # variable predates per-bot tokens; starting it would poll Telegram with the
    # ORIGINAL bot's token and split its messages. See groundwork/bots/env.token.
    if config.REFUSAL:
        sys.exit(f"[bot] refusing to start: {config.REFUSAL}")
    if not config.TOKEN:
        sys.exit(f"Missing {config.TOKEN_VAR} (paste the token on the bot's card "
                 "in the cockpit → Data collection).")
    # SAY WHICH BOT BEFORE DOING ANYTHING — the same rule every pipeline stage
    # follows. A bot has three ways to be the wrong one at once (wrong project,
    # wrong token, wrong people) and all three are now readable in one line.
    config.log.info("identity: %s", botenv.describe(config.TOKEN_VAR))

    # A PROJECT WITH NO TRAINED RUN IS NOT AN ERROR, it is a new project. This
    # used to raise FileNotFoundError out of main(), and with Restart=always /
    # StartLimitBurst=10 the unit crash-looped into `failed` within a minute —
    # where start, stop and restart all refuse until reset-failed. A new user's
    # first bot bricked its own card, with nothing on screen saying why.
    #
    # So: start anyway, with no counter. state.model_desc() already reads "no
    # model loaded", and commands._swap_counter() already exists, so the bot
    # picks up its first model when one lands without a reinstall.
    try:
        config.log.info("loading counter (%s, once) ...",
                        runtime_model.get_engine(config.pp()))
        state.COUNTER = runtime_model.make_counter(config.pp())
    except FileNotFoundError as e:
        # ONLY this. A corrupt checkpoint or a broken venv should still be loud.
        state.COUNTER = None
        config.log.warning("no trained model for project %s yet (%s) — starting "
                           "without one; it will be picked up once a run exists",
                           config.pp().slug, e)
    lock = (f"locked to {len(config.ALLOWED_IDS)} id(s)" if config.ALLOWED_IDS
            else "UNCLAIMED — answering nobody until someone is added in the cockpit")
    config.log.info("ready, %s. Polling…", lock)

    app = Application.builder().token(config.TOKEN).post_init(_post_init).build()
    app.add_error_handler(config.on_error)
    app.add_handler(CommandHandler(["start", "help"], commands.start))
    app.add_handler(CommandHandler("whoami", commands.whoami))
    app.add_handler(CommandHandler("stats", commands.stats))
    app.add_handler(CommandHandler("clearpending", commands.clear_pending_cmd))
    app.add_handler(CommandHandler("restart", commands.restart))
    app.add_handler(CommandHandler("cancel", commands.cancel))
    app.add_handler(CommandHandler("status", commands.status))
    app.add_handler(CommandHandler("engine", commands.engine_cmd))
    # /engine picks the ARCHITECTURE, /model picks WHICH RUN of it — two files
    # underneath, one question from the phone.
    app.add_handler(CommandHandler("model", commands.model_cmd))
    # Pattern handler MUST come before the generic feedback one — within a
    # group the first matching handler wins, and feedback's has no pattern.
    app.add_handler(CallbackQueryHandler(commands.on_clear_pending_confirm, pattern="^clearpend:"))
    app.add_handler(CallbackQueryHandler(commands.on_engine_switch, pattern="^eng:"))
    app.add_handler(CallbackQueryHandler(commands.on_model_switch, pattern="^mdl:"))
    app.add_handler(CallbackQueryHandler(feedback.on_feedback))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, counting.on_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, feedback.on_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
