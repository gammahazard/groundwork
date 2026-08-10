"""Central configuration — no magic numbers scattered across modules.

TORCH-FREE ON PURPOSE. This module is imported by the web assembler, the
scheduler and the trainer daemon; `import torch` here made every one of those
a ~1 GB process and meant the login page could not render without a working
CUDA install. Anything that needs torch lives in la/settings.py (loaded only
by the auto-labeler) or inside the functions that use it.
"""
from __future__ import annotations
import os
import socket
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent

# GW_DATA_DIR relocates everything a deployment WRITES — outputs, credentials,
# downloaded model weights, the .env — under one directory, which is what lets
# a container bind-mount a single volume and a native install keep today's
# repo-root layout by simply not setting it.
DATA_DIR = Path(os.environ.get("GW_DATA_DIR") or ROOT)
MODEL_PATH = os.environ.get("LA_MODEL_PATH",
                            str(DATA_DIR / "models" / "LocateAnything-3B"))
OUTPUTS_DIR = DATA_DIR / "outputs"
# Where sidecar (challenger-stack) venvs live: the checkout root on a native
# install — today's layout, unchanged — and venvs/ under DATA_DIR when the
# data-dir seam is active, because there the code tree is read-only to the
# runtime user (the Docker image ships /data/venvs world-writable for exactly
# this, and a named-volume venv survives an image rebuild). Everything that
# CREATES or RESOLVES a sidecar venv must go through this one answer;
# stacks.install building at ROOT inside a container died at mkdir.
VENVS_DIR = ROOT if DATA_DIR == ROOT else DATA_DIR / "venvs"
# The .env lives in a CONFIG DIRECTORY under DATA_DIR when that seam is active
# (a bind-mounted file breaks env_file's atomic os.replace — EBUSY), and at the
# repo root otherwise, which is where native installs have always kept it.
ENV_PATH = (DATA_DIR / "config" / ".env") if os.environ.get("GW_DATA_DIR") \
    else (ROOT / ".env")


def _load_env_file() -> None:
    """Fold `.env` into the process environment, without overriding it.

    THE WEB PROCESS NEVER DID THIS, and the systemd unit's own comment said it
    did ("settings come from .env in the working directory, so no Environment=
    lines belong here"). So every setting the wizard and the join flow persist
    through env_file.set_key — GW_PORT, GW_ROLE, GW_SELF_URL, GW_AUTH — was
    written to a file the service could not see. Measured: a worker joined on
    port 8420 came back after `systemctl start` listening on 8000, because the
    default won.

    override=False is the precedence rule: a real environment variable —
    systemd `Environment=`, a container's `-e`, `GW_PORT=9000 python -m ...` —
    beats the file. The file is persisted DEFAULTS, not an override.

    GW_DATA_DIR is deliberately NOT among the things this can set: it decides
    where the file itself lives, so honouring it from inside the file is a
    circle. It stays environment-only.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:            # base dep, but never fail a boot over it
        return
    try:
        load_dotenv(ENV_PATH, override=False)
    except OSError:
        pass


_load_env_file()

# --- machine role ---
# What this machine IS in a fleet: "hq" curates and dispatches (the default —
# a single-machine install is an HQ that trains locally); a "worker" trains and
# holds a read-only mirror, so every dataset mutation is refused there
# (web/lab_guard) and its scheduler runs the worker-side jobs only.
def role() -> str:
    return (os.environ.get("GW_ROLE", "hq") or "hq").strip().lower()


def is_worker() -> bool:
    return role() == "worker"


# --- machine identity ---
# Two machines share this repo and both train. A ledger row therefore has to say
# WHERE it was measured: a run adopted from the lab carries the lab's wall-clock
# duration, and using that to estimate this machine's ETA would be wrong (they
# have different GPUs). Legacy rows predate this field and were all local.
MACHINE = os.environ.get("GW_MACHINE") or socket.gethostname()

# --- model I/O conventions (from the model card / repo) ---
# Coordinates come back as normalized integers in [0, COORD_SCALE].
COORD_SCALE = 1000
# A box emitted this many identical times is a repetition-loop artifact, not
# that many real objects — drop it during parsing.
REPEAT_ARTIFACT_MIN = 4
# Merge points closer than this fraction of the mean image dimension (the model
# sometimes puts two dots on one object). Real objects — even touching — sit much
# farther apart than this, so distinct objects are never merged.
POINT_DEDUP_FRAC = 0.02
# Image processor budget: images over this many 14px patches get downscaled.
IN_TOKEN_LIMIT = 25600
PATCH_SIZE = 14

# HARD SAFETY CEILING for this machine (RTX 5060 8GB + WSL2).
# Empirically, runs stay stable while peak VRAM fits in real VRAM (<~8GB); once
# the driver spills into shared system RAM (seen at ~5k patches / ~10GB) the
# hypervisor faults and the whole machine crashes. We cap every run below that
# spill threshold so the pipeline can never crash the box. Raise/remove this on
# a bigger card (e.g. the 24GB 3090), where native resolution is safe.
SAFE_PATCH_CEILING = int(os.environ.get("LA_SAFE_PATCH_CEILING", "3000"))

# --- generation defaults (kept BOUNDED for TDR safety on the laptop GPU) ---
# 768 is plenty for an image of objects (each point ~8 tokens) and, crucially, BOUNDS
# the occasional repetition loop that otherwise drags a run to minutes. Benchmarked:
# 40 objects counted correctly at 768 in ~17s, same as 2048. hybrid keeps accuracy on
# dense/touching clusters (fast mode is weaker there for no speed gain).
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_GENERATION_MODE = "hybrid"   # "fast" | "slow" | "hybrid"
# A point costs ~4 tokens (<box><x><y></box>), so 768 tokens -> ~192 points max.
# If a count comes back near that ceiling the image was likely TRUNCATED, so we
# auto-retry once with a big budget (handles huge images without slowing normal ones).
TOKENS_PER_POINT = 4
MAX_NEW_TOKENS_LARGE = 6144            # ~1500 points on the retry pass
TRUNCATION_RATIO = 0.9                 # count >= 0.9 * ceiling -> retry bigger

# --- annotation style ---
BOX_COLOR = (255, 59, 48)      # red
POINT_COLOR = (0, 122, 255)    # blue
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_BG_COLOR = (255, 59, 48)

# force the sdpa attention backend (magi/flash aren't installed)
os.environ.setdefault("LA_FLASH_ATTN", "sdpa")


# ---- electricity accounting (a stat, not a budget) --------------------------
# Whole-programme cost so far is ~$2, so precision here is pointless — these are
# deliberately simple, overridable knobs rather than measured telemetry.
#
# Why estimated: nvidia-smi DOES expose total_energy_consumption (a cumulative
# mJ counter, ideal — two samples per run, no polling), but it is unreliable
# under WSL2 while a GPU trains, and hammering it is what wedged the NVIDIA
# driver on 2026-07-29. Not worth the risk for a two-dollar number. If the
# counter ever proves stable on an idle card, measured energy slots in beside
# train_seconds in each run's meta.json and this becomes the fallback.
#
# WATTS is WHOLE SYSTEM while training, PSU losses included — not GPU draw:
# the lab is a dual-card box (~280W GPU + ~90W rest, /0.90 efficiency); the
# laptop is a single 8GB card.
KWH_RATE = float(os.environ.get("GW_KWH_RATE", "0.18"))   # $/kWh, all-in
TRAIN_WATTS = int(os.environ.get(
    "GW_TRAIN_WATTS", "410" if is_worker() else "200"))

# ---------------------------------------------------------------- private ---
#
# CREDENTIAL STORES DO NOT LIVE UNDER outputs/, and that is not tidiness.
# apps/webui.py mounts the whole outputs tree as StaticFiles, so every file in
# it is served to anyone the gate lets through. The auth work put users.json,
# sessions.json, api_keys.json and audit.jsonl there, and measured 2026-08-05 a
# signed-in NON-ADMIN could GET /outputs/users.json and read every account's
# scrypt digest — offline cracking material, handed over by the web server.
#
# 0600 on disk did not help: the server process owns the files, so the
# permissions it enforces against other users are exactly the ones it bypasses
# itself when serving them.
#
# A DIRECTORY, NOT A DENY-LIST. Blocking the four known paths in the gate would
# be wrong by default — the next sensitive file added to outputs/ would be
# served until somebody remembered. Nothing here is reachable over HTTP because
# nothing here is mounted, which is a property rather than a rule to maintain.
PRIVATE_DIR = DATA_DIR / "private"
