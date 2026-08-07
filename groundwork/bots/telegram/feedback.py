"""Route feedback on a count: the ✓/🎯/✗/🗑 buttons, and the typed correct-count
capture for a ✗-Wrong image."""
from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from groundwork.dataset.store import collect
from groundwork.web import retrain_job

from . import config, state


async def on_feedback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the ✓/🎯/✗/🗑 buttons under a count: route the staged sample.

    The 🎯 buttons earmark an image for the holdout test set: 'test' (right) sends it
    straight there; 'notest' (wrong) sends it to the fix queue *flagged for test*,
    so after you fix its dots it graduates to testset instead of training.
    """
    q = update.callback_query
    if not state.authorized(update):
        return await q.answer("Not authorized", show_alert=True)
    action, _, name = (q.data or "").partition(":")
    if action == "ok":
        done = collect.promote(name, config.pp())
        tag = "✅ Saved for training" if done else "✅ (already handled)"
    elif action == "test":
        # Same lock as the web UI: the holdout is FROZEN while a retrain runs —
        # it's scored at the END of the run, so an image added mid-run could test
        # the model on data it just trained on. Buttons stay live to re-tap.
        if retrain_job.status().get("status") == "running":
            return await q.answer("🔒 Retrain running — the holdout is mid-exam. "
                                  "Tap 🎯 again after it finishes.", show_alert=True)
        done = collect.promote_to_testset(name, config.pp())
        tag = "🎯 Saved to the TEST SET (held out — measures accuracy, never trained on)" \
            if done else "🎯 (already handled)"
    elif action == "no":
        done = collect.reject(name, config.pp())
        if done:
            state.AWAITING.add(name)                     # expects a correct count
            tag = "✗ Flagged. *Reply to this photo* with the correct count 👇"
        else:
            tag = "✗ (already handled)"
    elif action == "notest":
        done = collect.reject(name, config.pp())
        if done:
            collect.set_earmark(name, config.pp())                    # earmark for the holdout
            state.AWAITING.add(name)                     # expects a correct count
            tag = ("🎯✗ Flagged *for the TEST SET*. *Reply to this photo* with the "
                   "correct count 👇 — fix its dots later, then it graduates to test.")
        else:
            tag = "🎯✗ (already handled)"
    else:  # rm
        done = collect.discard(name, config.pp())
        state.AWAITING.discard(name)
        tag = "🗑 Discarded" if done else "🗑 (already gone)"
    await q.answer(tag.replace("*", ""))
    cap = (q.message.caption or "").split("\nIs this right?")[0]
    await q.edit_message_caption(caption=f"{cap}\n{tag}", parse_mode="Markdown")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Capture the correct count for a ✗-Wrong image. Prefer an explicit reply
    to the count photo (unambiguous with many photos); fall back to the sole
    pending."""
    if not state.authorized(update):
        return
    # Which image? A reply to a specific count photo wins.
    reply = update.message.reply_to_message
    name = state.MSG_SAMPLE.get(reply.message_id) if reply else None
    if name is None:
        if len(state.AWAITING) == 1:
            name = next(iter(state.AWAITING))           # only one pending — no ambiguity
        elif len(state.AWAITING) > 1:
            return await update.message.reply_text(
                "You have several images to correct — long-press the specific "
                "photo, tap *Reply*, and send its count.", parse_mode="Markdown")
        else:
            return  # nothing awaiting a count — ignore stray text
    if name not in state.AWAITING:
        return await update.message.reply_text(
            "That image isn't waiting for a count (already set, or marked ✓/🗑).")
    txt = (update.message.text or "").strip()
    if not txt.isdigit():
        return await update.message.reply_text("Send just the number, e.g. `247`.",
                                               parse_mode="Markdown")
    state.AWAITING.discard(name)
    collect.set_truth(name, int(txt), config.pp())
    await update.message.reply_text(
        f"✅ Saved: correct count = *{txt}* for that image. "
        f"You'll fix its dots to match {txt} later in the editor.",
        parse_mode="Markdown")
