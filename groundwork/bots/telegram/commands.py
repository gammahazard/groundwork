"""Slash-command handlers: /whoami /start /status /stats /clearpending
/engine /restart /cancel.

Training commands were removed 2026-07-29 — retrains run on the Trainer
(the worker) through the cockpit, not on the machine this bot lives on.
retrain_job is still imported: /status reports a local run and /restart
refuses to re-exec during one."""
from __future__ import annotations
import asyncio
from pathlib import Path
import json
import os
import sys

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from groundwork.dataset.store import collect, labelio
from groundwork.web import retrain_job

from . import config, state
from .text import HELP


async def whoami(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Your Telegram id: `{update.effective_user.id}`",
                                    parse_mode="Markdown")


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not state.authorized(update):
        return await state.deny(update)
    await update.message.reply_text(HELP, parse_mode="Markdown")


async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not state.authorized(update):
        return await state.deny(update)
    import torch
    used = torch.cuda.memory_reserved() / 1e9
    c = collect.counts(config.pp())
    await update.message.reply_text(
        f"🧠 serving: {state.model_desc()}\n"
        f"VRAM reserved: {used:.2f} GB\n"
        f"collected → pending: {c['pending']} · to-fix: {c['needs_fix']} · halves: {c['halves']}")


async def stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Return the training-data totals."""
    if not state.authorized(update):
        return await state.deny(update)
    raw = labelio.list_images("raw", config.pp())
    objects = sum(t["count"] for t in raw)
    c = collect.counts(config.pp())
    rt = retrain_job.status()
    await update.message.reply_text(
        f"📊 *Training data*\n"
        f"• images: *{len(raw)}*\n"
        f"• objects labelled: *{objects}*\n"
        f"• to fix (✗): {c['needs_fix']}\n"
        f"• pending: {c['pending']}\n"
        f"• model: {rt.get('mae') and 'MAE '+str(rt['mae']) or 'not evaluated'}",
        parse_mode="Markdown")


async def clear_pending_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm-then-discard every staged pending sample (never touches the
    training/test buckets — pending is only unrouted counts)."""
    if not state.authorized(update):
        return await state.deny(update)
    n = collect.counts(config.pp())["pending"]
    if not n:
        return await update.message.reply_text("Nothing pending — all clear.")
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(f"🗑 Yes, discard {n}", callback_data="clearpend:yes"),
        InlineKeyboardButton("❌ No", callback_data="clearpend:no"),
    ]])
    await update.message.reply_text(
        f"🗑 Discard all *{n}* pending sample(s)?\n\n"
        "These are counted images you never routed with ✓/✗ — discarding them "
        "can't touch training or test data.",
        parse_mode="Markdown", reply_markup=kb)


async def on_clear_pending_confirm(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not state.authorized(update):
        return await q.answer("Not allowed.")
    await q.answer()
    if q.data != "clearpend:yes":
        return await q.edit_message_text("👍 Kept — nothing discarded.")
    n = collect.clear_pending(config.pp())
    await q.edit_message_text(f"🗑 Discarded {n} pending sample(s).")


async def restart(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-exec the bot to reload code (works only while the bot is responsive)."""
    if not state.authorized(update):
        return await state.deny(update)
    # Historically load-bearing: os.execv keeps our PID, so a bot-started
    # retrain's worker thread died, its training subprocess orphaned, and the
    # stale-state healer could never fire (owner_pid == the reborn bot). The
    # pipeline is its own process now (retrain_worker) and owner_pid points at
    # it, so /restart can no longer break a run. Kept as deliberate caution —
    # nothing good comes of reloading code mid-retrain — not as a fix.
    if retrain_job.status().get("status") == "running":
        return await update.message.reply_text(
            "⏳ A retrain is running — /restart would kill it and jam the "
            "training state. Try again after it finishes (or /cancel first).")
    try:                                    # remember who to ping once we're back
        config.RESTART_FLAG.parent.mkdir(parents=True, exist_ok=True)
        config.RESTART_FLAG.write_text(json.dumps({"chat_id": update.effective_chat.id}))
    except Exception:  # noqa: BLE001
        pass
    await update.message.reply_text("♻️ Restarting… I'll message you the moment I'm back.")
    os.execv(sys.executable, [sys.executable, "-m", "groundwork.bots.telegram"])


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not state.authorized(update):
        return await state.deny(update)
    tasks = list(state.ACTIVE)
    for t in tasks:
        t.cancel()
    state.ACTIVE.clear()
    n = len(tasks)
    await update.message.reply_text(
        f"🛑 Cancelled {n} count{'s' if n != 1 else ''}." if n
        else "Nothing to cancel.")


def _engine_rows() -> tuple[str, list]:
    """(status text, keyboard rows) for the engine switch. The challenger only
    appears on a machine that actually has an exported detector, so a button
    can never offer something that would fail on tap."""
    from groundwork.serve import runtime_model
    cur = runtime_model.get_engine(config.pp())
    deim = runtime_model.deim_weights(config.pp())
    lines = [f"🔧 Counting with: *{cur}*", ""]
    rows = []
    lines.append("• *yolo* — champion, holdout MAE 0.0, AGPL-3.0")
    rows.append([InlineKeyboardButton(("● " if cur == "yolo" else "") + "YOLOv8n",
                                      callback_data="eng:yolo")])
    if deim is not None:
        run = deim.parent.parent
        mae = ""
        try:
            mae = f", holdout MAE {json.loads((run / 'count_eval.json').read_text())['mae']}"
        except Exception:  # noqa: BLE001
            pass
        lines.append(f"• *deim* — challenger {run.name}{mae}, Apache-2.0")
        rows.append([InlineKeyboardButton(("● " if cur == "deim" else "") + "DEIMv2-N",
                                          callback_data="eng:deim")])
    else:
        lines.append("• *deim* — not on this machine (export a detector first)")
    lines.append("")
    lines.append("Tap to switch — takes effect immediately, no /restart.")
    return "\n".join(lines), rows


async def engine_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """/engine — show or switch the architecture that counts photos.

    Tap-to-switch: the resident model is swapped in place behind the same lock
    that serialises counts, so a switch can never land mid-count. The pinned
    yolo run is never touched, so flipping back is instant."""
    if not state.authorized(update):
        return await state.deny(update)
    from groundwork.serve import runtime_model
    arg = (ctx.args[0].lower() if ctx.args else "")
    if arg:                                     # typed form still works
        if arg == runtime_model.get_engine(config.pp()):
            return await update.message.reply_text(f"Already counting with *{arg}*.",
                                                   parse_mode="Markdown")
        try:
            runtime_model.set_engine(arg, config.pp())
        except ValueError as e:
            return await update.message.reply_text(f"🚫 {e}")
        await _swap_counter()
        return await update.message.reply_text(
            f"✅ Now counting with *{arg}*.", parse_mode="Markdown")
    text, rows = _engine_rows()
    await update.message.reply_text(text, parse_mode="Markdown",
                                    reply_markup=InlineKeyboardMarkup(rows))


async def _swap_counter() -> None:
    """Rebuild the resident counter for the current engine, holding the count
    lock so it can never swap out from under a photo already being counted."""
    from groundwork.serve import runtime_model
    if state.GPU_LOCK is None:                  # pre-init safety
        state.COUNTER = runtime_model.make_counter(config.pp())
        return
    async with state.GPU_LOCK:
        state.COUNTER = await asyncio.to_thread(runtime_model.make_counter, config.pp())


def _model_rows() -> tuple[str, list]:
    """(status text, keyboard rows) for the RUN switch.

    /engine picks the ARCHITECTURE; this picks WHICH RUN of it serves. Two files
    underneath (serving_engine.json and active_model.json) and deliberately so —
    switching engine leaves the yolo pin untouched — but from the phone they are
    one question: what is counting my photos.

    ONLY SCORED RUNS GET A BUTTON, and every button carries its MAE. There is no
    table beside a tap, so an unscored run behind one is a model pinned on no
    evidence — which is the same silent-wrong-model failure that mtime selection
    caused on the deim side.
    """
    from groundwork.serve import runtime_model
    eng = runtime_model.get_engine(config.pp())
    lines, rows = [], []
    if eng == "deim":
        # The deim run is chosen by score, not pinned — say so rather than
        # offering a switch that does not exist.
        cur = runtime_model.deim_weights(config.pp())
        name = cur.parent.parent.name if cur else "none"
        lines += [f"🧠 Engine *deim* — serving *{name}*", "",
                  "DEIM serves the best-scoring export automatically; there is "
                  "no per-run pin for it. Use /engine to go back to yolo.", ""]
        for c in runtime_model.deim_choices(config.pp()):
            mark = "● " if cur and c["weights"] == str(cur) else "   "
            lines.append(f"{mark}`{c['name']}` — MAE {c['mae']} on {c['n_trays']} images")
        return "\n".join(lines), rows

    active = runtime_model.get_active(config.pp())
    pinned = active and Path(active["weights"]).parent.parent.name
    lines += [f"🧠 Engine *yolo* — serving *{pinned or 'newest run (unpinned)'}*", ""]
    choices = runtime_model.yolo_choices(pp=config.pp())
    # THE PINNED RUN IS ALWAYS OFFERED, even when it does not make the top N.
    # yolov8n-55 was pinned at 0.0154 while eight runs sat at 0.0, so it fell off
    # the list entirely: the header said "serving yolov8n-55" and no button was
    # marked current. Same rule cockpit/yolo.js already follows — "always keep
    # the currently-served run visible even if old".
    if pinned and not any(c["name"] == pinned for c in choices):
        extra = [c for c in runtime_model.yolo_choices(limit=999, pp=config.pp()) if c["name"] == pinned]
        choices = choices[:-1] + extra if extra else choices
    for c in choices:
        on = c["name"] == pinned
        lines.append(f"{'● ' if on else '   '}`{c['name']}` — MAE {c['mae']} "
                     f"on {c['n_trays']} images")
        # MAE *and* image count on the button: eight runs all reading "(0.0)" is
        # not a choice. The exam SIZE is the differentiator now — 0.0 on 68 images
        # is a stronger result than 0.0 on 62, and the holdout has grown four
        # times.
        rows.append([InlineKeyboardButton(
            ("● " if on else "") + f"{c['name']}  {c['mae']} / {c['n_trays']} images",
            callback_data=f"mdl:{c['name']}")])
    rows.append([InlineKeyboardButton("↩ unpin (serve newest)", callback_data="mdl:*")])
    lines += ["", "Tap to switch — takes effect immediately, no /restart."]
    return "\n".join(lines), rows


async def model_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """/model — show or switch WHICH RUN counts photos (see /engine for which
    architecture)."""
    if not state.authorized(update):
        return await state.deny(update)
    text, rows = _model_rows()
    await update.message.reply_text(
        text, parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(rows) if rows else None)


async def on_model_switch(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline-button handler for the run switch."""
    q = update.callback_query
    if not state.authorized(update):
        return await q.answer("not authorized", show_alert=True)
    want = q.data.split(":", 1)[1]
    from groundwork.serve import runtime_model
    if want == "*":
        runtime_model.clear(config.pp())
    else:
        pick = next((c for c in runtime_model.yolo_choices(pp=config.pp()) if c["name"] == want), None)
        if pick is None:
            # The run list is rebuilt on every render, so a stale button is a
            # normal race (a run adopted or re-scored since) rather than an error.
            return await q.answer("that run is no longer offered", show_alert=True)
        runtime_model.set_active(pick["weights"], pick["imgsz"], config.pp())
    await q.answer("loading…")
    # The SAME swap /engine uses — it holds the count lock, so a switch can never
    # land in the middle of a photo already being counted.
    await _swap_counter()
    text, rows = _model_rows()
    try:
        await q.edit_message_text(text, parse_mode="Markdown",
                                  reply_markup=InlineKeyboardMarkup(rows) if rows else None)
    except Exception:  # noqa: BLE001
        pass                                    # message unchanged = fine


async def on_engine_switch(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline-button handler for the engine switch."""
    q = update.callback_query
    if not state.authorized(update):
        return await q.answer("not authorized", show_alert=True)
    want = q.data.split(":", 1)[1]
    from groundwork.serve import runtime_model
    if want == runtime_model.get_engine(config.pp()):
        return await q.answer("already serving that one")
    try:
        runtime_model.set_engine(want, config.pp())
    except ValueError as e:
        return await q.answer(str(e)[:180], show_alert=True)
    await q.answer("loading…")
    await _swap_counter()
    text, rows = _engine_rows()
    try:
        await q.edit_message_text(text, parse_mode="Markdown",
                                  reply_markup=InlineKeyboardMarkup(rows))
    except Exception:  # noqa: BLE001
        pass                                    # message unchanged = fine
