"""Shared runtime state for the handlers + authorization helpers.

One process, one GPU: the resident counter, a lock that serializes counts, and
the maps that let /cancel stop tasks and a text reply set a specific image's count.
Handlers read these as `state.X` so the entrypoint can (re)assign COUNTER/GPU_LOCK
at startup and every handler sees the update.
"""
from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING

from telegram import Update

from . import config

if TYPE_CHECKING:                       # avoid importing torch just for the hint
    from groundwork.serve.yolo_counter import YoloCounter

# The resident model (loaded once in main) and the lock that serializes counts.
COUNTER: "YoloCounter | None" = None
GPU_LOCK: asyncio.Lock | None = None    # created in post_init (inside the event loop)
# Track in-flight/queued count tasks so /cancel can stop them all.
ACTIVE: set[asyncio.Task] = set()
# count-reply message id -> sample name, so a reply to a specific photo sets its count.
MSG_SAMPLE: dict[int, str] = {}
# Samples marked ✗ Wrong that still need a correct count typed in.
AWAITING: set[str] = set()


def model_desc() -> str:
    """Human-readable 'which model am I serving' for /status + restart pings."""
    if not COUNTER:
        return "no model loaded"
    run = COUNTER.weights.parent.parent.name          # e.g. yolov8n-6
    tag = "" if not getattr(COUNTER, "EXPERIMENTAL", False) \
        else f" [{COUNTER.ENGINE} — EXPERIMENTAL]"
    return (f"{run} @ imgsz {COUNTER.imgsz} "
            f"(conf {COUNTER.conf} / iou {COUNTER.iou}){tag}")


def count_warning() -> str:
    """Extra line for a count reply when a CHALLENGER answered it.

    Counts came back looking identical whichever engine served them, so a tech
    reading the number had no way to know an unproven model produced it. Read
    off the loaded counter, not the engine file — the switch can be flipped
    after the bot has loaded, and what answered this photo is what matters."""
    if not COUNTER or not getattr(COUNTER, "EXPERIMENTAL", False):
        return ""
    return (f"\n⚠️ *EXPERIMENTAL model* ({COUNTER.ENGINE}) — not the champion. "
            "Double-check this count.")


def authorized(update: Update) -> bool:
    """Is this sender on this bot's list?

    An EMPTY list means nobody, which is why this reads as a membership test
    rather than "no owner set means anyone". See groundwork/bots/env.allowed_ids.
    """
    return bool(config.ALLOWED_IDS) and update.effective_user is not None \
        and update.effective_user.id in config.ALLOWED_IDS


async def deny(update: Update) -> None:
    """One refusal, whatever the reason.

    IT USED TO EXPLAIN ITSELF, and that was the bug: with no id set it told
    whoever messaged it to `put your id in the bot's allowed list and
    restart me` — handing a stranger who found the bot the exact recipe for
    claiming it, and telling them the bot was unclaimed in the first place. Two
    different denials also let anyone probe which state a bot is in.

    The sender's own id still reaches them through /whoami, which is what it is
    for and reveals nothing they cannot get from @userinfobot. Claiming a bot
    happens in the cockpit, behind its login.

    The guard on `update.message` is not defensive noise: a callback query
    arrives with no message, and this used to AttributeError there — the capture
    bot's twin has always had it.
    """
    uid = update.effective_user.id if update.effective_user else "?"
    # Logged so an owner being messaged by strangers can see it, and so a person
    # who cannot get in has something to point at.
    config.log.info("denied telegram id %s", uid)
    if update.message:
        await update.message.reply_text("⛔ Not authorized.")
