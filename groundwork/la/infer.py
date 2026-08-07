"""High-level detection orchestration: resolution policy + OOM fallback.

Wraps Worker.detect with:
  - a large-image policy (crop preferred; else fit to the model's patch budget),
  - an OOM retry ladder that downscales and reports the reduced resolution.
"""
from __future__ import annotations

import torch
from PIL import Image

from . import image_ops
from ..config import (DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_LARGE,
                     TOKENS_PER_POINT, TRUNCATION_RATIO)
from .parse import Detections, parse
from .worker import Worker

# Each retry multiplies the image size by this factor after an OOM.
_OOM_LADDER = [1.0, 0.75, 0.5, 0.35]


def _looks_truncated(count: int, max_new_tokens: int) -> bool:
    """True if the count is pinned near the token ceiling (likely cut off)."""
    ceiling = max_new_tokens / TOKENS_PER_POINT
    return count >= TRUNCATION_RATIO * ceiling


def _parse_for_mode(raw: str, width: int, height: int, mode: str) -> Detections:
    """Parse, then keep only the kind the mode is supposed to return — point mode
    emits points; any stray boxes (edge/rim artifacts) are not objects, and vice
    versa. This is what stops phantom edge detections inflating the count."""
    dets = parse(raw, width, height)
    want = "point" if mode == "point" else "box"
    dets.items = [d for d in dets.items if d.kind == want]
    return dets


def run_detection(worker: Worker, image: Image.Image, description: str,
                  mode: str = "point",
                  crop_box: tuple[int, int, int, int] | None = None,
                  **kwargs) -> tuple[Detections, dict, Image.Image]:
    """Locate `description` in `image`, handling large images and OOM.

    mode: "point" (default — faster, cleaner for counting) or "box".
    Returns (Detections, stats, display_image). Detections coords are in the
    coordinate space of display_image (the full-res image to annotate on) — the
    model runs on an internally-downscaled copy, but coords are re-scaled back to
    display_image so overlays always land correctly. Annotate on display_image.
    """
    if crop_box is not None:
        image = image_ops.crop(image, crop_box)
    display = image  # full-res image the caller should annotate; coords map to it

    # Hard safety cap: keep every run inside the machine's stable VRAM envelope
    # so we can never re-enter the sysmem-spill zone that crashes the box.
    base, capped = image_ops.fit_to_safe_ceiling(image)
    locate = worker.point if mode == "point" else worker.detect

    max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    # Ceiling for the truncation-retry pass. Defaults to the big runtime budget,
    # but callers on a VRAM-tight card (e.g. batch labeling on the 8GB laptop)
    # pass a lower value: a 6144-token decode's KV cache can spill VRAM and crash
    # the machine. Capping it keeps peak memory bounded.
    max_retry_tokens = kwargs.pop("max_retry_tokens", MAX_NEW_TOKENS_LARGE)

    last_err: Exception | None = None
    for factor in _OOM_LADDER:
        attempt = base if factor == 1.0 else image_ops.downscale(base, factor)
        try:
            dets_small, stats = locate(attempt, description,
                                       max_new_tokens=max_new_tokens, **kwargs)
            # Re-scale coords from the downscaled inference image back to full res
            # (coords are normalized [0,1000], so parse against display size).
            dets = _parse_for_mode(dets_small.raw, display.width, display.height, mode)

            # Truncation guard: if the count is pinned near the token ceiling, the
            # image was likely cut off — rerun once with a big budget so huge images
            # count fully. Normal/looping images never hit this, so stay fast.
            retried = False
            if _looks_truncated(dets.count, max_new_tokens) \
                    and max_new_tokens < max_retry_tokens:
                print(f"[infer] count {dets.count} near ceiling — retry at "
                      f"{max_retry_tokens} tokens", flush=True)
                dets_small, stats = locate(attempt, description,
                                           max_new_tokens=max_retry_tokens, **kwargs)
                dets = _parse_for_mode(dets_small.raw, display.width, display.height, mode)
                retried = True

            stats.update({
                "mode": mode,
                "used_size": attempt.size,
                "oom_factor": factor,
                "safety_capped": capped,
                "oom_fallback": factor != 1.0,
                "truncation_retry": retried,
            })
            return dets, stats, display
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            last_err = e
            torch.cuda.empty_cache()
            print(f"[infer] OOM at factor {factor} ({attempt.size}); retrying smaller",
                  flush=True)

    raise RuntimeError(
        f"OOM even at smallest retry factor {_OOM_LADDER[-1]}: {last_err}")
