#!/usr/bin/env bash
# Groundwork native install — stage 0.
#
# Only what a shell must do BEFORE a working interpreter exists: check the
# basics, pick a torch wheel channel, build the venv, install. The moment
# there is a venv this script execs `groundwork.install --native`, which does
# the real setup (systemd units, timers, first-run pointers) where such logic
# belongs — in Python, tested, idempotent.
#
# Docker users don't run this; `docker compose -f docker/docker-compose.yml
# up -d --build` is the whole install there.
set -euo pipefail

say()  { printf '%s\n' "$*"; }
fail() { printf 'install: %s\n' "$*" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- preflight ---------------------------------------------------------------
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 \
    || fail "python3 not found (or set PYTHON=/path/to/python)"
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "python >= 3.10 required, found: $("$PY" -V 2>&1)"
"$PY" -m venv -h >/dev/null 2>&1 \
    || fail "the venv module is missing (debian/ubuntu: apt install python3-venv)"
command -v git >/dev/null 2>&1 \
    || fail "git is required (upgrades are a git pull; challenger stacks are checkouts)"
command -v rsync >/dev/null 2>&1 \
    || fail "rsync is required (dataset sync between machines)"

# --- torch channel -----------------------------------------------------------
# compute capability 12.x (Blackwell) needs cu128; older cards get cu126; no
# GPU means the cpu channel — a real channel, installed the same way, just
# small. TORCH_CHANNEL in the environment overrides the guess entirely.
if [ -n "${TORCH_CHANNEL:-}" ]; then
    say "torch channel: $TORCH_CHANNEL (from the environment)"
elif command -v nvidia-smi >/dev/null 2>&1; then
    # Highest capability wins on a mixed box — the newest card sets the floor.
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | tr -d '[:space:]' | sort -rn | head -n 1)" || cap=""
    major="${cap%%.*}"
    case "$major" in
        ''|*[!0-9]*)
            # nvidia-smi exists but won't answer (old driver, mid-upgrade…).
            # A GPU is probably there, so take the conservative CUDA channel.
            TORCH_CHANNEL=cu126
            say "GPU present but compute capability unreadable — using cu126 (override with TORCH_CHANNEL=...)"
            ;;
        *)
            if [ "$major" -ge 12 ]; then TORCH_CHANNEL=cu128; else TORCH_CHANNEL=cu126; fi
            say "GPU compute capability $cap -> torch channel $TORCH_CHANNEL"
            ;;
    esac
else
    TORCH_CHANNEL=cpu
    say "no nvidia-smi — CPU-only install (collect/label/manage; training needs a GPU machine)"
fi

# --- venv + install ----------------------------------------------------------
VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    say "creating venv at .venv"
    "$PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip

# Torch FIRST, from the channel index. The order is load-bearing: with torch
# already satisfied, the unpinned "torch" in the [train] extra resolves to
# this build, so nothing installed afterwards (ultralytics included) can swap
# in a generic one. TORCH_VERSION/TV_VERSION pin exact versions if set;
# unset, the channel's current build is fine.
say "installing torch + torchvision from the $TORCH_CHANNEL wheel index (large download)"
"$VENV/bin/pip" install \
    "torch${TORCH_VERSION:+==$TORCH_VERSION}" \
    "torchvision${TV_VERSION:+==$TV_VERSION}" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL"

say "installing groundwork with the [train,bots] extras"
"$VENV/bin/pip" install -e ".[train,bots]"

# --- hand over ---------------------------------------------------------------
say "python setup complete — handing over to groundwork.install"
exec "$VENV/bin/python" -m groundwork.install --native "$@"
