"""RF-DETR challenger predictor (.venv-alt).

One raw pass per image at a floor confidence (0.05 — below the sweep grid's
minimum 0.10 and below the 0.25 mask probe); the harness owns thresholding +
NMS per sweep cell. DETR emits a fixed set of scored queries with no NMS of its
own, so the sweep's iou axis is expected to be nearly flat for this predictor —
harmless, the grid still picks the operating point.
"""
from __future__ import annotations
import numpy as np

FLOOR = 0.05


class RFDetrPredictor:
    def __init__(self, weights, resolution: int):
        assert resolution % 56 == 0, f"rfdetr resolution must be /56, got {resolution}"
        import torch
        from rfdetr import RFDETRNano
        # Query count lives in the checkpoint (nano: group_detr=13 slots per
        # query) — read it so 600-query round-2 checkpoints load cleanly.
        sd = torch.load(str(weights), map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        nq = sd["refpoint_embed.weight"].shape[0] // 13
        self.model = RFDETRNano(pretrain_weights=str(weights),
                                resolution=resolution, num_queries=nq)
        try:
            self.model.optimize_for_inference()
        except Exception:  # noqa: BLE001 — optional fast path only
            pass
        self._checked = False
        # rfdetr trains on /56 resolutions but predicts on /32 shapes — round
        # the trained resolution up to the nearest valid inference shape.
        side = ((resolution + 31) // 32) * 32
        self.shape = (side, side)

    def infer_raw(self, pil):
        det = self.model.predict(pil, threshold=FLOOR, shape=self.shape)
        boxes = np.asarray(det.xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(det.confidence, dtype=np.float64).reshape(-1)
        if not self._checked and len(boxes):
            w, h = pil.size
            assert boxes[:, [0, 2]].max() <= w + 2 and boxes[:, [1, 3]].max() <= h + 2, \
                "rfdetr boxes are not in original-image pixels"
            self._checked = True
        return boxes, scores
