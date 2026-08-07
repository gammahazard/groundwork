"""DEIMv2 challenger predictor (.venv-deim, vendor repo ~/DEIMv2).

NMS-free decoder: one raw pass per image at a floor confidence; the harness
owns thresholding + NMS composition (iou axis near-flat, like rfdetr — the
sweep still picks the operating point). Their deployed model takes
(image_tensor, orig_size) and returns boxes ALREADY in original pixels.
HGNetv2 variants (atto..n) take unnormalized tensors; ViT variants (s+)
would need ImageNet normalization — flagged via meta arch if we ever go there.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

FLOOR = 0.01
VENDOR = Path.home() / "DEIMv2"


class DeimPredictor:
    def __init__(self, weights, imgsz: int):
        import os
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        # their HGNetv2 calls torch.distributed.get_rank() unconditionally —
        # give the process a 1-member gloo group so plain (non-torchrun)
        # inference works
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29777")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        sys.path.insert(0, str(VENDOR))
        from engine.core import YAMLConfig

        weights = Path(weights)
        cfg_path = weights.parent.parent / "run_config.yml"   # trainer saved it
        if not cfg_path.exists():
            raise SystemExit(f"no run_config.yml beside {weights} — "
                             "point --weights at a trainers.deim run checkpoint")
        # Their loader resolves __include__ relative to the config file, which
        # points into the vendor tree — run from a copy placed there.
        vcfg = VENDOR / "configs" / "deimv2" / "_predict_tmp.yml"
        vcfg.write_text(cfg_path.read_text(), encoding="utf-8")
        cfg = YAMLConfig(str(vcfg))
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        sd = (state.get("ema") or {}).get("module") or state.get("model") or state
        cfg.model.load_state_dict(sd)

        class Deploy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.post = cfg.postprocessor.deploy()

            def forward(self, images, orig_sizes):
                return self.post(self.model(images), orig_sizes)

        self.torch = torch
        self.model = Deploy().cuda().eval()
        self.tf = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])

    def infer_raw(self, pil):
        torch = self.torch
        w, h = pil.size
        with torch.no_grad():
            out = self.model(self.tf(pil.convert("RGB")).unsqueeze(0).cuda(),
                             torch.tensor([[w, h]]).cuda())
        labels, boxes, scores = out
        boxes = boxes[0].cpu().numpy().astype(np.float64).reshape(-1, 4)
        scores = scores[0].cpu().numpy().astype(np.float64).reshape(-1)
        keep = scores >= FLOOR
        boxes, scores = boxes[keep], scores[keep]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)   # same guard as rtmdet
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores
