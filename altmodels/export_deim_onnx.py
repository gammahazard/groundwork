"""DEIMv2 -> ONNX, the first leg of the iPhone question.

The vendor's tools/deployment/export_onnx.py traces a batch of 32 at full
resolution (untenable on CPU at 1280px) and re-reads its own configs; this
does the same job for OUR runs: the trainer's run_config.yml, batch 1,
static shapes (CoreML prefers them), and the same Deploy wrapper the
challenger predictor uses, so what we export is what we scored.

    .venv-deim/bin/python -m altmodels.export_deim_onnx \
        --weights outputs/alt/deimv2-n-1280-k/train/last.pth

Then coremltools converts the .onnx on any machine. The open question this
answers first: does the deformable-attention grid_sample survive tracing at
all? If ONNX fails, CoreML never happens.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from altmodels.deim_vendor import vendor_dir
VENDOR = vendor_dir()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a DEIM run to ONNX.")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-postprocess", action="store_true",
                    help="export the DETECTOR ONLY (raw logits + cxcywh boxes, "
                         "no top-k/gather). Their postprocessor gathers with "
                         "fp32 indices, which CoreML rejects — and the phone "
                         "does its own thresholding anyway (DEIM is NMS-free)")
    ap.add_argument("--format", choices=["onnx", "torchscript", "both"],
                    default="onnx",
                    help="torchscript is the CoreML path: coremltools 7+ "
                         "dropped ONNX input, and a traced module is "
                         "self-contained so the conversion can happen on any "
                         "machine without the vendor repo or this venv")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.distributed as dist
    if not dist.is_initialized():          # HGNetv2 calls get_rank() blindly
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29778")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    sys.path.insert(0, str(VENDOR))
    from engine.core import YAMLConfig

    w = args.weights
    cfg_path = w.parent.parent / "run_config.yml"
    if not cfg_path.exists():
        raise SystemExit(f"no run_config.yml beside {w}")
    vcfg = VENDOR / "configs" / "deimv2" / "_export_tmp.yml"
    vcfg.write_text(cfg_path.read_text(), encoding="utf-8")
    cfg = YAMLConfig(str(vcfg))
    if "HGNetv2" in cfg.yaml_cfg:          # weights come from the checkpoint
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # CoreML's `linear` op asserts a 2-D weight. DEIM's Integral head calls
    # F.linear(x, project) with project 1-D (the reg_max+1 distribution bin
    # centres), which is just a dot product — mathematically identical to a
    # matmul, but it traces as `linear` and blows up the conversion. Swap it
    # here, at export time only: the vendor repo stays untouched and the
    # numbers are unchanged (asserted below).
    from engine.deim.dfine_decoder import Integral

    def _integral_forward(self, x, project):
        shape = x.shape
        x = torch.nn.functional.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = torch.matmul(x, project.to(x.device).reshape(-1, 1)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])

    _orig_forward = Integral.forward
    # unit-check the swap on the op itself (building a second Deploy would
    # re-run cfg.model.deploy(), which fuses conv layers IN PLACE and dies)
    _rm = int(cfg.yaml_cfg.get("DEIMTransformer", {}).get("reg_max", 32))
    _probe, _x, _p = Integral(reg_max=_rm), torch.rand(2, 4 * (_rm + 1)), torch.rand(_rm + 1)
    with torch.no_grad():
        _ref = _probe(_x, _p)
        Integral.forward = _integral_forward
        _got = _probe(_x, _p)
    _d = (_got - _ref).abs().max().item()
    assert _d < 1e-5, f"Integral matmul patch changed the output by {_d}"
    print(f"[export] Integral matmul patch verified identical (max diff {_d:.2e})")

    state = torch.load(str(w), map_location="cpu", weights_only=False)
    sd = (state.get("ema") or {}).get("module") or state.get("model") or state
    cfg.model.load_state_dict(sd)

    class Deploy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.post(self.model(images), orig_sizes)

    class Detector(nn.Module):
        """Heads only, ONE image input. Boxes come out cxcywh normalised to
        the (squashed) input, which is also normalised original-image space —
        so the phone needs no size tensor and no letterbox arithmetic."""

        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            out = self.model(images)
            return out["pred_logits"], out["pred_boxes"]

    model = (Detector() if args.no_postprocess else Deploy()).eval()
    data = torch.rand(1, 3, args.imgsz, args.imgsz)
    size = torch.tensor([[args.imgsz, args.imgsz]])
    inputs = (data,) if args.no_postprocess else (data, size)
    with torch.no_grad():
        probe = model(*inputs)             # smoke the graph before tracing
    if args.no_postprocess:
        print(f"[export] detector-only: logits {tuple(probe[0].shape)}, "
              f"boxes {tuple(probe[1].shape)} (cxcywh, normalised)")
    if args.format in ("torchscript", "both"):
        ts = args.out if (args.out and args.format == "torchscript") \
            else w.with_suffix(".torchscript.pt")
        with torch.no_grad():
            traced = torch.jit.trace(model, inputs, strict=False)
        torch.jit.save(traced, str(ts))
        print(f"[torchscript] wrote {ts} ({ts.stat().st_size / 1e6:.1f} MB) "
              f"— feed this to coremltools (source='pytorch')")
    if args.format in ("onnx", "both"):
        out = args.out or w.with_suffix(".onnx")
        torch.onnx.export(
            model, inputs, str(out),
            input_names=["images"] if args.no_postprocess
            else ["images", "orig_target_sizes"],
            output_names=["logits", "boxes"] if args.no_postprocess
            else ["labels", "boxes", "scores"],
            opset_version=args.opset, do_constant_folding=True, verbose=False)
        mb = out.stat().st_size / 1e6
        print(f"[onnx] wrote {out} ({mb:.1f} MB, opset {args.opset}, "
              f"static 1x3x{args.imgsz}x{args.imgsz})")


if __name__ == "__main__":
    main()
