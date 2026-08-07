"""Score filter + greedy IoU-NMS for CHALLENGER predictors (numpy, class-agnostic).

The harness runs a challenger ONCE per image at a floor confidence, then applies
this per (conf, iou) sweep cell — instead of re-running the model 42x like the
ultralytics eval must (whose NMS lives inside predict()). Parity runs never use
this module: the ultralytics parity predictor re-runs predict() per cell so its
numbers are bit-identical to count_eval's.

Semantics (documented once, relied on by the harness):
- score filter: score > conf (strict, matches ultralytics' internal `conf > thres`)
- greedy NMS: sort desc by score, suppress IoU > iou (strict)
- single class, max_det cap applied after NMS (ultralytics default 2000 here)
"""
from __future__ import annotations
import numpy as np


def filter_and_nms(boxes: np.ndarray, scores: np.ndarray,
                   conf: float, iou: float, max_det: int = 2000) -> list[list[float]]:
    """boxes (N,4) xyxy absolute px, scores (N,) -> kept boxes as lists."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    keep = scores > conf
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    boxes = boxes[order]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    kept_idx: list[int] = []
    alive = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        kept_idx.append(i)
        if len(kept_idx) >= max_det:
            break
        rest = np.nonzero(alive)[0]
        rest = rest[rest > i]
        if len(rest) == 0:
            continue
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        ious = np.where(union > 0, inter / union, 0.0)
        alive[rest[ious > iou]] = False
    return [boxes[i].tolist() for i in kept_idx]
