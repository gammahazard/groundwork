"""The count flow: receive a photo, count it (serialized on the one GPU), reply
with the point overlay + the ✓/🎯/✗/½/🗑 feedback buttons."""
from __future__ import annotations
import asyncio
import io

from PIL import Image

try:                                    # iPhone Files-app sends can be true HEIC
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from . import config, state

def _try_load() -> bool:
    """Load this project's model if there is one now. True if we have a counter.

    A bot installed for a project with no trained run starts WITHOUT a model
    rather than crash-looping, so this is what lets the first successful
    training take effect on the next photo instead of on the next restart —
    which is the moment a new user most needs it to just work.

    Cheap when there is still nothing: resolving weights is a directory glob, no
    GPU is touched until a checkpoint actually exists.
    """
    if state.COUNTER is not None:
        return True
    try:
        from groundwork.serve import runtime_model
        state.COUNTER = runtime_model.make_counter(config.pp())
        config.log.info("model appeared — now serving %s", state.model_desc())
        return True
    except FileNotFoundError:
        return False
    except Exception:  # noqa: BLE001 — a broken checkpoint must not eat photos
        config.log.error("could not load the model", exc_info=True)
        return False


async def run_count(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                    img: Image.Image, img_full: Image.Image, note) -> None:
    """Count one photo, serialized behind the GPU lock. Cancellable via /cancel.

    Delivery is failure-proofed: Telegram flakes (Bad Gateway nights exist)
    must never eat a finished count silently — the photo send retries once,
    and any error still reports the NUMBER as text (the image stays staged)."""
    result = None
    try:
        async with state.GPU_LOCK:                   # queue: one count at a time
            await ctx.bot.send_chat_action(update.effective_chat.id,
                                           ChatAction.UPLOAD_PHOTO)
            result = await asyncio.to_thread(
                state.COUNTER.count, img,
                name=f"tg_{update.effective_user.id}_{update.message.message_id}")
        await note.delete()
        # TRAINING COPIES KEEP EVERY PIXEL THE PHONE SENT. The ≤2000px working
        # copy exists only for reply speed — re-stage the pending image from
        # the ORIGINAL (same crop, full resolution). Labels are normalized
        # coordinates, so they remain valid untouched.
        if img is not img_full:
            try:
                img_full.save(config.pp().PENDING_IMAGES / f"{result.name}.jpeg",
                              quality=95)
            except Exception:  # noqa: BLE001 — the small copy stays if this fails
                config.log.error("full-res restage failed (small copy kept)",
                                 exc_info=True)
        # 2x2 of {right,wrong} x {train,test}, then dismiss. The 🎯 column
        # earmarks the image for the holdout; ✗/🎯-wrong both ask for the true count.
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✓ Correct → train", callback_data=f"ok:{result.name}"),
             InlineKeyboardButton("🎯 Correct → test", callback_data=f"test:{result.name}")],
            [InlineKeyboardButton("✗ Wrong → train", callback_data=f"no:{result.name}"),
             InlineKeyboardButton("🎯 Wrong → test", callback_data=f"notest:{result.name}")],
            [InlineKeyboardButton("🗑 Dismiss", callback_data=f"rm:{result.name}")],
        ])
        caption = (f"🔢 *Count: {result.count}*\n{result.seconds}s"
                   f"{state.count_warning()}\n"
                   f"Is this right? ✓ saves it for training · ✗ to fix later · 🗑 discard")
        # Full-res overlays make multi-MB PNGs — send a lean JPEG copy instead
        # (Telegram recompresses photos anyway; the original stays on disk).
        send_path = result.overlay_path
        try:
            import os
            if os.path.getsize(send_path) > 2_500_000:
                im = Image.open(send_path)
                im.thumbnail((2560, 2560))
                send_path = send_path + ".send.jpg"
                im.convert("RGB").save(send_path, quality=88)
        except Exception:  # noqa: BLE001 — cosmetic; fall back to the original
            send_path = result.overlay_path
        sent = None
        for attempt in (1, 2):                       # one retry on a flaky upload
            try:
                with open(send_path, "rb") as f:
                    sent = await update.message.reply_photo(
                        photo=f, caption=caption, parse_mode="Markdown",
                        reply_markup=kb)
                break
            except Exception:  # noqa: BLE001
                config.log.error("overlay send failed (attempt %d)", attempt,
                                 exc_info=True)
                if attempt == 2:
                    raise
                await asyncio.sleep(2)
        # Remember which image this photo is, so a reply to it sets its count.
        state.MSG_SAMPLE[sent.message_id] = result.name
    except asyncio.CancelledError:
        try:
            await note.edit_text("🛑 Cancelled.")
        except Exception:
            pass
        raise
    except Exception as e:  # noqa: BLE001 — NEVER let a finished count vanish
        config.log.error("count delivery failed", exc_info=True)
        msg = f"⚠️ {type(e).__name__} while delivering the result."
        if result is not None:
            msg += (f"\n🔢 The count WAS *{result.count}* — image staged as "
                    f"`{result.name}` (route it in the web Fix/Training tabs).")
        try:
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:  # noqa: BLE001
            pass


async def on_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not state.authorized(update):
        return await state.deny(update)
    # NO MODEL YET is a state a new project is legitimately in, not a bug — the
    # bot starts without one rather than crash-looping (see apps/telegram_bot).
    # It was an `assert`, which would have taken the handler down with an
    # AttributeError and told the sender nothing.
    #
    # RETRY THE LOAD FIRST: the moment the project's first run finishes, the
    # next photo should just work, with no restart. web/api/count.py::_get uses
    # the same rebuild-on-demand pattern.
    # NO MODEL is not a refusal — it is INTAKE MODE. A brand-new project's
    # whole problem is that it has no images yet; the bot is precisely the
    # easiest way to fix that. So before the first trained run, photos are
    # accepted and land straight in the fix queue, unlabeled, and the moment a
    # run finishes the SAME bot starts counting (the retry-load below).
    intake_only = state.COUNTER is None and not _try_load()

    # Note goes up BEFORE the download: a large File on a flaky Telegram night
    # can take a while (or fail) — the user must never stare at silence.
    note = await update.message.reply_text("⏳ fetching photo…")
    try:
        if update.message.photo:
            tf = await update.message.photo[-1].get_file()
        elif update.message.document:
            tf = await update.message.document.get_file()
        else:
            return await note.delete()
        buf = await tf.download_as_bytearray()
        img_full = Image.open(io.BytesIO(bytes(buf))).convert("RGB")
    except Exception as e:  # noqa: BLE001 — network flake / undecodable file
        config.log.error("photo ingest failed", exc_info=True)
        return await note.edit_text(
            f"⚠️ couldn't fetch/decode that image ({type(e).__name__}) — "
            "Telegram may be flaky right now; send it again.")
    # Dual-path: COUNT on a downscaled copy (12MP buys counting nothing — the
    # model sees ~1280px — but costs seconds in overlays/upload), while the
    # ORIGINAL stays in memory so the training copy keeps every pixel the
    # phone captured.
    img = img_full
    if max(img_full.size) > 2000:
        img = img_full.copy()
        img.thumbnail((2000, 2000))

    if intake_only:
        pp = config.pp()
        stem = f"tg_{update.effective_user.id}_{update.message.message_id}"
        img_dir, lbl_dir = pp.NEEDS_FIX_IMAGES, pp.NEEDS_FIX_LABELS
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        while (img_dir / f"{stem}.jpeg").exists():
            stem += "x"
        img_full.save(img_dir / f"{stem}.jpeg", quality=95)
        (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
        n = sum(1 for _ in img_dir.iterdir())
        return await note.edit_text(
            f"📥 Saved for labeling — {n} image(s) waiting in the fix queue.\n"
            f"This project has no trained model yet, so I collect instead of "
            f"counting. Label in the cockpit, train a run, and I start "
            f"counting automatically on the next photo.")

    queued = " (queued)" if state.GPU_LOCK.locked() else ""
    await note.edit_text(f"⏳ Counting…{queued}")
    task = asyncio.create_task(run_count(update, ctx, img, img_full, note))
    state.ACTIVE.add(task)
    task.add_done_callback(state.ACTIVE.discard)
