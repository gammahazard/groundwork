"""Draw detections onto an image with an index number at each. Pure + GPU-free.

Kept model-free so overlay bugs can be reproduced with a hand-made Detections
object and no GPU.
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..config import (BOX_COLOR, POINT_COLOR, LABEL_TEXT_COLOR, LABEL_BG_COLOR,
                     OUTPUTS_DIR)
from ..dataset.filters.dot_dedupe import dot_radius
from .parse import Detections


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
           font, anchor_above: bool = True) -> None:
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    pad = 2
    ly = y - th - 2 * pad if anchor_above else y
    draw.rectangle([x, ly, x + tw + 2 * pad, ly + th + 2 * pad], fill=LABEL_BG_COLOR)
    draw.text((x + pad, ly + pad), text, fill=LABEL_TEXT_COLOR, font=font)


def annotate(image: Image.Image, dets: Detections, numbered: bool = True) -> Image.Image:
    """Return a copy of image with boxes/points drawn on it.

    numbered=True adds an index label at each detection (good for the LA tools /
    cross-referencing). numbered=False draws plain dots only — cleaner for just
    verifying at a glance that every object got one (the count is shown elsewhere).
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    line_w = max(2, round(min(w, h) / 500))
    font = _font(max(14, round(min(w, h) / 60)))
    dot_r = dot_radius(w, h)

    for i, d in enumerate(dets.items, start=1):
        idx = str(i)
        if d.kind == "box":
            x1, y1, x2, y2 = d.coords
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_w)
            if numbered:
                _label(draw, (x1, y1), idx, font, anchor_above=True)
        else:  # point
            x, y = d.coords
            draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r],
                         fill=POINT_COLOR, outline=(255, 255, 255), width=1)
            if numbered:
                _label(draw, (x + dot_r, y - dot_r), idx, font, anchor_above=False)
    return out


def save_annotated(image: Image.Image, dets: Detections, src_path: str,
                   out_dir: Path | None = None, numbered: bool = True) -> Path:
    """Annotate and save as <name>_annotated.png. Returns the output path."""
    out_dir = out_dir or OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src_path).stem
    out_path = out_dir / f"{stem}_annotated.png"
    annotate(image, dets, numbered=numbered).save(out_path)
    return out_path
