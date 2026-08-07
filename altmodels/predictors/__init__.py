"""Predictor registry — lazy imports so each predictor's heavy deps only load
in the venv that has them.

Protocol (duck-typed, the harness dispatches on which method exists):
- infer_raw(pil) -> (boxes_xyxy_np, scores_np)   raw pass at a floor conf;
  the harness applies per-cell score threshold + NMS (altmodels.nms).
- predict_cells(pil, conf, iou) -> list[xyxy]    re-runs the model per sweep
  cell; used ONLY by the ultralytics parity predictor, whose NMS lives inside
  predict() and must stay bit-identical to count_eval's calls.
"""
from __future__ import annotations


def get_predictor(name: str, weights, imgsz: int):
    if name == "ultralytics":
        from .ultralytics_parity import UltralyticsParity
        return UltralyticsParity(weights, imgsz)
    if name == "rfdetr":
        from .rfdetr_predict import RFDetrPredictor
        return RFDetrPredictor(weights, imgsz)
    if name == "rtmdet":
        from .rtmdet_predict import RTMDetPredictor  # exists only if the mmcv gate passed
        return RTMDetPredictor(weights, imgsz)
    if name == "dfine":
        from .dfine_predict import DFinePredictor    # main .venv, transformers
        return DFinePredictor(weights, imgsz)
    if name == "deim":
        from .deim_predict import DeimPredictor      # .venv-deim + vendor ~/DEIMv2
        return DeimPredictor(weights, imgsz)
    raise SystemExit(f"unknown predictor {name!r}")
