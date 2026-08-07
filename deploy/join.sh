#!/usr/bin/env bash
# Groundwork worker join — run ON the GPU machine:
#   curl -fsSL http://<hq>:8000/join.sh | bash -s -- http://<hq>:8000 <token>
set -euo pipefail
HUB="${1:?usage: join.sh <hub-url> <join-token> [dir]}"
TOKEN="${2:?usage: join.sh <hub-url> <join-token> [dir]}"
DIR="${3:-$HOME/groundwork}"

say() { printf '[join] %s\n' "$*"; }
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
command -v rsync   >/dev/null || { echo "rsync is required (dataset sync)"; exit 1; }
command -v sshd >/dev/null 2>&1 || [ -d /etc/ssh ] || {
  say "WARNING: no sshd — the HQ pulls runs over ssh. Fix: sudo apt install openssh-server"
  say "         (joining continues; the HQ shows the resume path once sshd exists)"
}

if [ ! -e "$DIR/pyproject.toml" ]; then
  say "downloading Groundwork from the hub"
  mkdir -p "$DIR"
  curl -fsSL "$HUB/api/join/bundle?token=$TOKEN" | tar -xz -C "$DIR" --strip-components=1
fi
cd "$DIR"

if [ ! -x .venv/bin/python ]; then
  say "creating venv"
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip

# torch channel: Blackwell needs cu128, older CUDA cards cu126, no GPU -> cpu.
if [ -z "${TORCH_CHANNEL:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | tr -d ' \r' | sort -rn | head -n1)"
    major="${cap%%.*}"
    case "$major" in (1[2-9]) TORCH_CHANNEL=cu128 ;; (*) TORCH_CHANNEL=cu126 ;; esac
  else
    TORCH_CHANNEL=cpu
  fi
fi
say "installing torch ($TORCH_CHANNEL — large download on first run)"
.venv/bin/pip install -q torch torchvision \
    --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL"
say "installing groundwork"
.venv/bin/pip install -q -e ".[train,bots]"
# D-FINE, the built-in challenger — the cockpit offers it, so a worker that
# cannot import transformers would refuse every D-FINE dispatch.
.venv/bin/pip install -q -e ".[dfine]"

exec .venv/bin/python -m groundwork.web.join_worker --hub "$HUB" --token "$TOKEN"
