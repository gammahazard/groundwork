"""Parity predictor: the champion itself, called EXACTLY like count_eval does.

Runs only under the main .venv (ultralytics lives there). conf+NMS are baked
into predict(), so parity re-runs it per sweep cell — slow (~2.6k predicts for
a 60-image holdout) but bit-identical to count_eval.py's three call sites, which
is the whole point: if the harness reproduces the pinned run's count_eval.json
through THIS predictor, the harness is proven, and challenger scores are
trustworthy.
"""
from __future__ import annotations


class UltralyticsParity:
    def __init__(self, weights, imgsz: int):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.imgsz = imgsz

    def predict_cells(self, pil, conf: float, iou: float) -> list[list[float]]:
        r = self.model.predict(pil, imgsz=self.imgsz, conf=conf, iou=iou,
                               max_det=2000, verbose=False)[0]
        return r.boxes.xyxy.tolist()
