"""RTMDet challenger predictor (.venv-mmdet).

One near-raw pass per image: score floor 0.01 and an internal NMS opened up to
iou 0.98, so the model's own NMS keeps only true duplicates and the harness's
per-cell threshold + NMS (altmodels.nms) does the real work — composition with
the sweep grid stays exact for our conf/iou ranges.

mmdet needs the run's dumped config next to the checkpoint (the trainer's
work_dir keeps one); boxes come back rescaled to the input image already.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

FLOOR = 0.01


def _find_config(weights: Path) -> Path:
    """The mmengine Runner dumps the fully-resolved config into work_dir."""
    for d in (weights.parent, weights.parent.parent):
        cfgs = sorted(d.glob("*.py"))
        if cfgs:
            return cfgs[0]
    raise SystemExit(f"no mmdet config .py found beside {weights} — "
                     "point --weights at a trainers.rtmdet work_dir checkpoint")


class RTMDetPredictor:
    def __init__(self, weights, imgsz: int):
        from mmengine.config import Config
        from mmdet.apis import init_detector
        weights = Path(weights)
        cfg = Config.fromfile(str(_find_config(weights)))
        # Serve at the requested size regardless of what the config trained at.
        for tr in cfg.test_pipeline:
            if tr.get("type") in ("Resize", "FixScaleResize"):
                tr["scale"] = (imgsz, imgsz)
        cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline
        model = init_detector(cfg, str(weights), device="cuda:0")
        # Near-raw output: the harness owns real thresholding + NMS. And never
        # cap detections below a dense capture (stock max_per_img is 300).
        model.cfg.model.test_cfg.score_thr = FLOOR
        model.cfg.model.test_cfg.max_per_img = 1000
        # topk TOO, or CenterNet stays capped at 100 no matter what max_per_img
        # says — it selects the top-k heatmap peaks BEFORE applying max_per_img.
        # The eval and the training config must agree here, or a model trained
        # with a raised ceiling gets scored through a lowered one.
        if "topk" in model.cfg.model.test_cfg:
            model.cfg.model.test_cfg.topk = 1000
        if hasattr(model, "test_cfg") and model.test_cfg is not None:
            model.test_cfg.score_thr = FLOOR
            model.test_cfg.max_per_img = 1000
            if "topk" in model.test_cfg:
                model.test_cfg.topk = 1000
            if getattr(model.test_cfg, "nms", None) is not None:
                model.test_cfg.nms.iou_threshold = 0.98
        self.model = model

    def infer_raw(self, pil):
        from mmdet.apis import inference_detector
        img = np.asarray(pil.convert("RGB"))[:, :, ::-1]  # mmdet expects BGR
        inst = inference_detector(self.model, img).pred_instances
        keep = inst.scores >= FLOOR
        boxes = inst.bboxes[keep].cpu().numpy().astype(np.float64).reshape(-1, 4)
        scores = inst.scores[keep].cpu().numpy().astype(np.float64).reshape(-1)
        # At the floor conf the model sometimes "detects" in the letterbox
        # padding; after rescale those boxes land outside the original frame
        # (killed round B's first exam as an assert). Clip to the frame — the
        # degenerate edge slivers die at the harness's real thresholds.
        w, h = pil.size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores
