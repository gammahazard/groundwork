"""Image loading + resolution helpers. Pure + GPU-free.

Key facts about the model's image processor:
  - It auto-downscales any image whose 14px-patch count exceeds IN_TOKEN_LIMIT.
  - 1242x2208 screenshots (~14k patches) pass through at native resolution.
  - 12MP originals (~62k patches) get crushed ~0.64x, hurting per-object detail.

So for large images we prefer cropping to a region over blind downscaling, and
we expose a downscale helper the OOM-fallback path uses to retry smaller.
"""
from __future__ import annotations
import math

from PIL import Image

from ..config import IN_TOKEN_LIMIT, PATCH_SIZE, SAFE_PATCH_CEILING


def load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def patch_count(image: Image.Image) -> int:
    w, h = image.size
    return (w // PATCH_SIZE) * (h // PATCH_SIZE)


def exceeds_native_budget(image: Image.Image) -> bool:
    """True if the model will internally downscale this image."""
    return patch_count(image) > IN_TOKEN_LIMIT


def downscale(image: Image.Image, factor: float) -> Image.Image:
    """Downscale by a factor in (0, 1]. Used by the OOM-retry path."""
    if not 0 < factor <= 1:
        raise ValueError("factor must be in (0, 1]")
    w, h = image.size
    return image.resize((max(1, int(w * factor)), max(1, int(h * factor))),
                        Image.Resampling.BICUBIC)


def fit_to_budget(image: Image.Image, budget: int | None = None) -> Image.Image:
    """Downscale so patch_count <= budget (mirrors the model's own rescale).

    Returned image feeds the model at a resolution we control, so we can reason
    about VRAM instead of letting the processor decide silently.
    """
    budget = budget or IN_TOKEN_LIMIT
    pc = patch_count(image)
    if pc <= budget:
        return image
    scale = math.sqrt(budget / pc)
    return downscale(image, scale)


def fit_to_safe_ceiling(image: Image.Image) -> tuple[Image.Image, bool]:
    """Downscale so patch_count <= SAFE_PATCH_CEILING, keeping runs inside the
    machine's proven-stable VRAM envelope. Returns (image, was_capped)."""
    if patch_count(image) <= SAFE_PATCH_CEILING:
        return image, False
    return fit_to_budget(image, SAFE_PATCH_CEILING), True


def crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop to a pixel box (x1, y1, x2, y2) — preferred over downscaling for
    large images so per-object detail in the image region is preserved."""
    return image.crop(box)
