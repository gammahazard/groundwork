"""Convert parsed box Detections into normalized YOLO label lines. Pure + GPU-free.

YOLO detection format, one line per box:
    <class> <cx> <cy> <w> <h>
all four geometry values normalized to [0, 1] by image width/height, with the
center (not the corner) as the anchor. Kept model-free so it unit-tests without a GPU.
"""
from __future__ import annotations
from pathlib import Path

from ..la.parse import Detection


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def box_to_yolo_line(det: Detection, img_w: int, img_h: int, cls: int = 0) -> str | None:
    """One pixel-space box Detection -> a normalized YOLO line, or None if degenerate.

    Coords are clamped to the frame first (LA can emit boxes a few px out of bounds),
    then a box with no area after clamping is dropped.
    """
    if det.kind != "box" or img_w <= 0 or img_h <= 0:
        return None
    x1, y1, x2, y2 = det.coords
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # Clamp to frame in pixels, then normalize.
    x1 = _clamp01(x1 / img_w); x2 = _clamp01(x2 / img_w)
    y1 = _clamp01(y1 / img_h); y2 = _clamp01(y2 / img_h)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    cx, cy = x1 + w / 2, y1 + h / 2
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def boxes_to_yolo_lines(boxes, img_w: int, img_h: int, cls: int = 0) -> list[str]:
    """Convert an iterable of box Detections into YOLO lines, dropping degenerates."""
    lines = []
    for d in boxes:
        line = box_to_yolo_line(d, img_w, img_h, cls)
        if line is not None:
            lines.append(line)
    return lines


def write_label(path: Path, lines: list[str]) -> None:
    """Write YOLO lines to a .txt (empty file = a valid 'no objects' label)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
