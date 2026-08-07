"""User-facing copy: the /help text and the command menu (kept apart from logic
so wording can be tweaked without touching handlers)."""
from __future__ import annotations
from telegram import BotCommand

HELP = (
    "💊 *Object Counter*\n"
    "Send a photo and I'll reply with the count + a point overlay "
    "(a dot on each object) in about a second.\n\n"
    "• *Frame the image tightly* — fill the shot with it. Nothing is cropped for "
    "you any more, so a bench edge, your hand or a bottle cap in frame can be "
    "counted as objects.\n"
    "• Send as a *File* (not compressed photo) for best detail on dense images.\n"
    "• Send several at once — they queue and run one at a time.\n\n"
    "Under each count, tap:\n"
    "  *✓ Correct → train* — count's right; save it as training data\n"
    "  *🎯 Correct → test* — count's right; hold it out as a test image "
    "(measures accuracy, never trained on — earmark ~1 in 5)\n"
    "  *✗ Wrong → train* — I'll ask for the real count, then you fix the dots (→ training)\n"
    "  *🎯 Wrong → test* — same, but the fixed image graduates to the *test set*\n"
    "  *½ Half object(s)* — image has a partial object; queued to label in the web UI\n"
    "  *🗑 Dismiss* — discard (accidental / not needed)\n\n"
    "`/stats` — training-data totals\n"
    "`/clearpending` — discard every unrouted pending sample (asks to confirm)\n"
    "`/cancel` — stop all active + queued counts\n"
    "`/restart` — reload the bot · `/status` — health · `/whoami` — your id\n"
    "`/engine` — tap to switch the ARCHITECTURE that counts (yolo champion / deim challenger)\n"
    "`/model` — tap to switch WHICH RUN of it serves, with each run's holdout MAE"
)

# Commands shown in Telegram's menu (the ☰ button beside the text box).
COMMANDS = [
    BotCommand("help", "How to use the bot"),
    BotCommand("stats", "Training-data totals"),
    BotCommand("clearpending", "Discard all unrouted pending samples"),
    BotCommand("cancel", "Stop all active + queued counts"),
    BotCommand("restart", "Reload the bot"),
    BotCommand("status", "Model + GPU health"),
    BotCommand("engine", "Switch the counting architecture (yolo / deim)"),
    BotCommand("model", "Switch which trained run serves"),
    BotCommand("whoami", "Show your Telegram id"),
]
