"""Telegram bot, split into small single-purpose modules.

Entry point is `apps/telegram_bot.py` (thin wiring). The pieces:
  config   — secrets/env, paths, logging, error handler
  text     — HELP copy + the command menu
  state    — shared runtime state (resident model, GPU lock, task/sample maps) + auth
  counting — photo -> count flow (serialized on the GPU)
  feedback — the ✓/🎯/✗/🗑 buttons + the typed correct-count capture
  commands — slash-command handlers (/stats, /retrain, /restart, …)
"""
