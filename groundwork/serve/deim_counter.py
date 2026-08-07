"""DEIMv2-N challenger, served exactly like the champion.

Subclasses YoloCounter and overrides ONLY `_detect`, so the stacked-dot
dedupe, the overlay and the staging path are literally the same code — the
lockstep the hard rules require, rather than a second copy that can drift.

No vendor repo, no second venv, no GPU: the exported TorchScript detector is
self-contained (altmodels/export_deim_onnx.py --format torchscript
--no-postprocess) and runs on CPU in ~0.4 s here, which also keeps the card
free for training.

Two things differ from the champion and must not be cross-copied:
  - preprocessing is a SQUASHED resize to imgsz (DEIM's training transform),
    not yolo's letterbox. The squash is linear per axis, so normalised
    coordinates map straight back to the original frame.
  - the model is NMS-free and emits raw heads (600 logits + 600 cxcywh boxes
    normalised 0-1), so thresholding and the NMS composition the exam used
    happen here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from ..config import ROOT
from .yolo_counter import YoloCounter

# Serving default if a run carries no tuned pair (K's exam picked 0.25/0.3).
DEIM_CONF, DEIM_IOU = 0.25, 0.3


def _nms(boxes: list[list[float]], scores: list[float], iou_thr: float) -> list[int]:
    """Greedy class-agnostic IoU NMS, same composition as altmodels/nms.py."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        ax1, ay1, ax2, ay2 = boxes[i]
        aarea = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        drop = False
        for j in keep:
            bx1, by1, bx2, by2 = boxes[j]
            iw = min(ax2, bx2) - max(ax1, bx1)
            ih = min(ay2, by2) - max(ay1, by1)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            barea = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            if inter / (aarea + barea - inter + 1e-9) > iou_thr:
                drop = True
                break
        if not drop:
            keep.append(i)
    return keep


class DeimCounter(YoloCounter):
    """Same serving contract as YoloCounter; different detector."""

    ENGINE = "deim"
    EXPERIMENTAL = True         # a challenger, and every surface must say so

    def __init__(self, pp, weights: str | Path | None = None,
                 conf: float | None = None, iou: float | None = None,
                 imgsz: int = 1280, verbose: bool = True):
        import torch                      # already a dependency of the main venv

        w = Path(weights) if weights else None
        if w is None:
            from .. import runtime_model
            # THIS project's exports (ProjectPaths.ALT_DIR), not the machine's.
            w = runtime_model.deim_weights(pp)
        if w is None or not w.exists():
            raise SystemExit("[deim] no exported detector — run "
                             "altmodels.export_deim_onnx --format torchscript "
                             "--no-postprocess on a DEIM run first")
        self.weights = w
        # THIS __init__ DELIBERATELY DOES NOT CHAIN TO super(), so every
        # attribute the inherited count() reads has to be set here — `pp`
        # included, or staging raises instead of writing to this project.
        self.pp = pp
        # tuned pair from the run's own exam, like the champion does
        ce = w.parent.parent / "count_eval.json"
        if conf is None or iou is None:
            try:
                j = json.loads(ce.read_text())
                conf = conf if conf is not None else j.get("conf", DEIM_CONF)
                iou = iou if iou is not None else j.get("iou", DEIM_IOU)
            except Exception:  # noqa: BLE001
                conf, iou = conf or DEIM_CONF, iou or DEIM_IOU
        self.conf, self.iou, self.imgsz = float(conf), float(iou), imgsz
        self.max_det = 2000
        t0 = time.time()
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) // 2))
        self.model = torch.jit.load(str(w), map_location="cpu").eval()
        self._torch = torch
        if verbose:
            rel = w.relative_to(ROOT) if ROOT in w.parents else w
            print(f"[deim] loaded {rel} in {time.time()-t0:.1f}s | "
                  f"conf={self.conf} iou={self.iou} imgsz={imgsz} (cpu)",
                  flush=True)

    def _detect(self, image: Image.Image) -> list[list[float]]:
        torch = self._torch
        w, h = image.size
        # SQUASH to imgsz (DEIM's Resize[s,s]) — no letterbox, no padding maths
        import numpy as np
        arr = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(arr, dtype=np.uint8).copy())
        x = x.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)  # ToTensor()
        with torch.no_grad():
            logits, boxes = self.model(x)
        scores = logits.sigmoid().flatten().tolist()
        bx = boxes[0].tolist()                          # cxcywh, normalised
        cand, keep_scores = [], []
        for s, (cx, cy, bw, bh) in zip(scores, bx):
            if s <= self.conf:
                continue
            # normalised (squashed) == normalised (original) per axis
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            cand.append([max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)])
            keep_scores.append(s)
        return [cand[i] for i in _nms(cand, keep_scores, self.iou)]
