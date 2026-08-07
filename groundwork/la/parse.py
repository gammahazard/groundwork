"""Parse raw model output into pixel-space detections. Pure + GPU-free.

The model emits normalized integer coordinates in [0, COORD_SCALE]:
  box   -> <box><x1><y1><x2><y2></box>
  point -> <box><x><y></box>
We scale to pixels using the ORIGINAL image size. This module is deliberately
model-free so it can be unit-tested and debugged in isolation.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from ..config import COORD_SCALE, REPEAT_ARTIFACT_MIN, POINT_DEDUP_FRAC

# 4 coords = box, 2 coords = point. Order matters: match boxes first.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
# Semantic label blocks sometimes precede coords, e.g. <ref>object</ref>.
_REF_RE = re.compile(r"<ref>(.*?)</ref>")


@dataclass
class Detection:
    kind: str                     # "box" or "point"
    coords: tuple                 # pixel coords: (x1,y1,x2,y2) or (x,y)
    label: str | None = None


@dataclass
class Detections:
    items: list[Detection] = field(default_factory=list)
    raw: str = ""

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def boxes(self) -> list[Detection]:
        return [d for d in self.items if d.kind == "box"]

    @property
    def points(self) -> list[Detection]:
        return [d for d in self.items if d.kind == "point"]


def _scale(v: int, size: int) -> float:
    return int(v) / COORD_SCALE * size


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _suppress_near_dups(boxes: list[Detection], iou_thresh: float = 0.9) -> list[Detection]:
    """Drop boxes that near-perfectly overlap an already-kept box. Collapses
    drifting repetition loops; real (even touching) objects overlap far less."""
    kept: list[Detection] = []
    for d in boxes:
        if all(_iou(d.coords, k.coords) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def _suppress_near_points(points: list[Detection], min_dist: float) -> list[Detection]:
    """Drop points closer than min_dist to an already-kept point — the model
    sometimes places two dots on one object. Real objects sit much farther apart."""
    kept: list[Detection] = []
    for d in points:
        x, y = d.coords
        if all(math.hypot(x - k.coords[0], y - k.coords[1]) >= min_dist for k in kept):
            kept.append(d)
    return kept


def _artifact_boxes(raw: str) -> set[tuple[int, int, int, int]]:
    """Integer box tuples the model emitted >= REPEAT_ARTIFACT_MIN times.

    The model can fall into a repetition loop, re-emitting one degenerate box
    dozens/hundreds of times. That's a decoding artifact, not many real objects,
    so we drop those coordinates entirely.
    """
    counts = Counter(tuple(int(g) for g in m.groups())
                     for m in _BOX_RE.finditer(raw))
    return {coords for coords, n in counts.items() if n >= REPEAT_ARTIFACT_MIN}


def parse(raw: str, width: int, height: int) -> Detections:
    """Parse raw text into pixel-space Detections for a width x height image.

    Exact-duplicate boxes are collapsed to one (real objects have distinct
    integer coords), and repetition-loop artifacts are dropped.
    """
    artifacts = _artifact_boxes(raw)
    boxes: list[Detection] = []
    box_spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for m in _BOX_RE.finditer(raw):
        box_spans.append(m.span())
        key = tuple(int(g) for g in m.groups())
        if key in artifacts or key in seen:
            continue
        seen.add(key)
        x1, y1, x2, y2 = key
        px = (_scale(x1, width), _scale(y1, height),
              _scale(x2, width), _scale(y2, height))
        bw, bh = px[2] - px[0], px[3] - px[1]
        # Drop degenerate line/edge artifacts (an object is never a thin line):
        # near-zero dimension or extreme aspect ratio.
        if bw < 2 or bh < 2 or max(bw, bh) / min(bw, bh) > 6:
            continue
        boxes.append(Detection(kind="box", coords=px))

    # Points: 2-coord <box> NOT inside a matched 4-coord box span, deduped.
    points: list[Detection] = []
    seen_pt: set[tuple[int, int]] = set()
    for m in _POINT_RE.finditer(raw):
        if any(s <= m.start() < e for s, e in box_spans):
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key in seen_pt:
            continue
        seen_pt.add(key)
        points.append(Detection(kind="point",
                                coords=(_scale(key[0], width), _scale(key[1], height))))

    boxes = _suppress_near_dups(boxes)
    points = _suppress_near_points(points, POINT_DEDUP_FRAC * (width + height) / 2)
    return Detections(items=boxes + points, raw=raw)


def count_only(raw: str) -> int:
    """Fast count without pixel scaling (for quick prompt A/B comparisons)."""
    dets = parse(raw, COORD_SCALE, COORD_SCALE)  # scale is irrelevant to count
    return dets.count
