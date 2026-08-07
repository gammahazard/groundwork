"""Convert point Detections into box Detections when only points are available.

Points carry no size, so we synthesize a box around each. Two strategies:

  "fixed"    — every box is a fixed fraction of the image (min(W,H)). Simple, but
               a single global size mis-fits mixed object sizes and, on DENSE images,
               makes neighboring boxes overlap -> YOLO NMS merges them -> undercount.

  "adaptive" — (default) size each box from the distance to its nearest neighbor,
               so boxes shrink in dense clusters (just touch, no NMS merge) and grow
               on sparse images. Auto-adapts to object size and image density. Clamped
               to a [min, max] fraction band so isolated points don't blow up.

Real box-mode LA labels are still preferred (they carry true extent AND aspect);
this exists to salvage point-only images without re-running the 3B. Pure + GPU-free.

The "adaptive" strategy is ALSO ported to JavaScript so the editor's "show boxes"
draws the boxes this writes (apps/static/editor/editor_mirrors.js). Change either and
re-run `.venv/bin/python scratch/check_editor_mirrors.py`.
"""
from __future__ import annotations
import math

from ..la.parse import Detection


def box_at(x: float, y: float, half_w: float, half_h: float,
           img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Box centred on (x, y) with INDEPENDENT half-extents, shrunk per axis to
    stay inside the frame.

    Centre-preserving on purpose. Clamping corners after centring (the old path,
    via box_to_yolo_line's frame clamp) shifted the stored CENTRE inward for edge
    dots — every editor save→load cycle migrated rim objects toward the middle.
    Pinning the centre and shrinking the half-extent instead makes the round trip
    exact; edge objects get narrower boxes, which is truthful (they ARE partly
    out of frame). 1px floor so a box is never degenerate.

    Independent half-extents exist so a HAND-SIZED box survives the same round
    trip: labelio stores a real width and height per object, and they must clamp
    by the identical rule the synthesised ones do, or an edge box would drift on
    every save.
    """
    hx = max(1.0, min(half_w, x, img_w - x))
    hy = max(1.0, min(half_h, y, img_h - y))
    return (x - hx, y - hy, x + hx, y + hy)


def _box_at(x: float, y: float, half: float,
            img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Square version — the synthesised default, sized from dot spacing."""
    return box_at(x, y, half, half, img_w, img_h)


def boxes_for(points, img_w: int, img_h: int) -> list[tuple]:
    """THE rule for turning editor points into boxes: keep a stored extent, and
    synthesise the adaptive default only for points that have none.

    `points` are ragged rows as labelio produces them — `[x, y]`, `[x, y, cls]`
    or `[x, y, cls, bw, bh]` in PIXELS. Returns pixel `(x1, y1, x2, y2)` per
    point, in order.

    One definition on purpose. It is applied in two languages — here (what
    `labelio.save_points` writes to disk) and in `apps/static/editor/editor_mirrors.js`
    (what the editor DRAWS). If they disagree, the boxes you see are not the
    boxes you save, which is the worst possible bug in a labelling tool.
    `scratch/check_editor_mirrors.py` compares them.

    The default is measured across ALL points, including sized ones: the
    adaptive rule is about the image's spacing, so dropping sized boxes from the
    neighbour search would change the default for everything near them.
    """
    dets = [Detection(kind="point", coords=(float(p[0]), float(p[1])))
            for p in points]
    defaults = points_to_boxes(dets, img_w, img_h)
    out = []
    for p, d in zip(points, defaults):
        if len(p) > 4 and p[3] and p[4]:
            out.append(box_at(float(p[0]), float(p[1]),
                              float(p[3]) / 2, float(p[4]) / 2, img_w, img_h))
        else:
            out.append(tuple(d.coords))
    return out


def points_to_boxes(
    points: list[Detection],
    img_w: int,
    img_h: int,
    strategy: str = "adaptive",
    fixed_frac: float = 0.05,
    nn_alpha: float = 0.5,
    min_frac: float = 0.015,
    max_frac: float = 0.08,
) -> list[Detection]:
    """Turn point Detections into square box Detections (pixel coords).

    fixed_frac / min_frac / max_frac are fractions of min(img_w, img_h).
    nn_alpha scales the nearest-neighbor distance into a box half-size
    (0.5 -> touching objects get boxes that just meet).
    """
    ref = min(img_w, img_h)
    lo, hi = min_frac * ref, max_frac * ref
    pts = [(float(d.coords[0]), float(d.coords[1])) for d in points]

    if strategy == "fixed" or len(pts) < 2:
        half = fixed_frac * ref / 2
        return [Detection(kind="box", coords=_box_at(x, y, half, img_w, img_h),
                          label="object") for x, y in pts]

    boxes: list[Detection] = []
    for i, (x, y) in enumerate(pts):
        nn = min(math.hypot(x - ox, y - oy)
                 for j, (ox, oy) in enumerate(pts) if j != i)
        half = min(max(nn_alpha * nn, lo), hi)   # nn-scaled half-size, clamped to band
        boxes.append(Detection(kind="box", coords=_box_at(x, y, half, img_w, img_h),
                               label="object"))
    return boxes
