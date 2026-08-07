"""D-FINE challenger predictor (main .venv — transformers, no vendor repo).

One raw pass per image at a floor confidence; the harness owns the per-cell
threshold + NMS, exactly as for the other challengers.

Preprocessing MUST match trainers.dfine.TrayDataset: a SQUASHED resize to imgsz
and /255, with NO ImageNet normalisation (verified against the shipped image
processor, do_normalize: False). Because the squash is linear per axis, the
model's normalised cxcywh output maps straight back to original-image pixels —
no letterbox arithmetic anywhere.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

FLOOR = 0.01


class DFinePredictor:
    def __init__(self, weights, imgsz: int):
        import torch
        from transformers import DFineForObjectDetection

        self.torch = torch
        self.imgsz = imgsz
        w = Path(weights)
        if not (w / "config.json").exists():
            raise SystemExit(f"{w} is not a saved D-FINE directory — point "
                             "--weights at a trainers.dfine run's train/best or "
                             "train/last")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DFineForObjectDetection.from_pretrained(str(w)).to(self.dev).eval()

    def infer_raw(self, pil):
        torch = self.torch
        W, H = pil.size
        # BILINEAR is LOAD-BEARING, not a default worth omitting: PIL's bare
        # .resize() is BICUBIC, while trainers.dfine.TrayDataset trains on BILINEAR.
        # Leaving it implicit meant every D-FINE exam ran on a different image
        # distribution than the model was trained on (mean |delta| 3.86/255), so
        # runs A and B were both scored unfairly (found 2026-07-30 — the
        # docstring above already demanded they match).
        arr = pil.convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(arr, dtype=np.uint8).copy())
        x = x.permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            out = self.model(pixel_values=x)
        # single class -> the one logit IS the objectness
        scores = out.logits.sigmoid()[0, :, 0].float().cpu().numpy()
        b = out.pred_boxes[0].float().cpu().numpy()          # cxcywh, 0-1
        keep = scores > FLOOR
        scores, b = scores[keep], b[keep]
        if not len(b):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        xy = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H,
                       (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
        xy[:, 0::2] = xy[:, 0::2].clip(0, W)                 # keep boxes in frame
        xy[:, 1::2] = xy[:, 1::2].clip(0, H)
        return xy.astype(np.float32), scores.astype(np.float32)
