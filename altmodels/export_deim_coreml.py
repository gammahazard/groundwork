"""DEIM TorchScript -> CoreML .mlpackage, the second leg of the iPhone path.

    python -m altmodels.export_deim_coreml \
        --torchscript outputs/alt/<run>/train/detector.torchscript.pt

WHY THIS EXISTS AS A FILE. The first DEIM package was converted by hand, and the
recipe lived only in a docstring line — "feed this to coremltools
(source='pytorch')". That is not enough to reproduce it: the interface the Swift
expects is specific, and getting any part of it wrong produces a package that
loads and then returns nothing useful.

THE INTERFACE IS FIXED BY the consuming CoreML app's model class, not chosen here:

    input   "image"    an IMAGE input, 1280x1280 RGB — not a multiarray
    outputs "logits"   (1, N, 1) raw, pre-sigmoid
            "boxes"    (1, N, 4) cxcywh, already normalised 0-1

The Swift does its own sigmoid, threshold, NMS and dot-dedupe, so the model must
NOT postprocess. That is also forced on us: DEIM's own postprocessor gathers with
fp32 indices and CoreML rejects it, which is why export_deim_onnx has
--no-postprocess and why only `detector.torchscript.pt` is convertible.

N IS NOT FIXED. Run K exported 600 queries; the 2026-08-02 run uses 1000, because
the detection ceiling was raised when centernet's topk=100 was found capping
dense images. The Swift loops over `logits.count`, so it adapts — but the
description written into the package states the real number, since "600" baked
into a description that ships 1000 is exactly the kind of quiet lie that costs an
afternoon.

PREPROCESSING IS DEIM'S, NOT YOLO'S, and must not be cross-copied (IPHONE.md):
DEIM trains on a SQUASHED resize to 1280x1280 with ConvertPILImage(scale=True),
i.e. plain 0-1 with no mean/std. yolo letterboxes. Because the squash is linear
per axis, normalised coordinates need no un-letterboxing, which is why the export
takes no size input at all.

CONVERSION RUNS ON LINUX; RUNNING THE PACKAGE NEEDS APPLE HARDWARE. coremltools
here has no libcoremlpython, so this cannot predict() to check itself — the
verification that matters is on the Mac, comparing CoreML against PyTorch on one
image. IPHONE.md records that this has never been done for DEIM.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs" / "coreml"

# The Swift's contract. Changing either name breaks the app silently — it throws
# CountError.modelMissing on a nil featureValue, which reads as "no model".
IMAGE_INPUT = "image"
OUTPUTS = ("logits", "boxes")


def convert(ts_path: Path, imgsz: int = 1280, name: str | None = None) -> Path:
    import torch
    import coremltools as ct

    traced = torch.jit.load(str(ts_path), map_location="cpu").eval()
    # Probe the real output shape rather than assuming 600 or 1000 — the ceiling
    # has already changed once.
    with torch.no_grad():
        logits, boxes = traced(torch.zeros(1, 3, imgsz, imgsz))
    n_queries = int(logits.shape[1])

    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[ct.ImageType(name=IMAGE_INPUT, shape=(1, 3, imgsz, imgsz),
                             # 0-1, no mean/std: ConvertPILImage(scale=True) is
                             # the whole of DEIM's normalisation.
                             scale=1 / 255.0, bias=[0, 0, 0],
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=o) for o in OUTPUTS],
        minimum_deployment_target=ct.target.iOS16,
    )
    run = ts_path.parent.parent.name
    mlmodel.short_description = (
        f"DEIMv2-N object detector ({run}). Outputs {n_queries} logits + "
        f"{n_queries} cxcywh boxes normalised 0-1. Squashed {imgsz}x{imgsz} "
        f"input, no postprocessing — the app thresholds and dedupes.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name or run}.mlpackage"
    if out.exists():
        shutil.rmtree(out)
    mlmodel.save(str(out))

    # SAY WHAT THE INTERFACE ACTUALLY IS, from the saved package rather than from
    # what was requested — a mismatch here is the failure that presents on the
    # phone as "no model" and nowhere else.
    spec = ct.models.MLModel(str(out)).get_spec()
    ins = [(i.name, "image" if i.type.HasField("imageType") else "array")
           for i in spec.description.input]
    outs = [o.name for o in spec.description.output]
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[coreml] wrote {out} ({size:.1f} MB)")
    print(f"         inputs {ins}  outputs {outs}  queries {n_queries}")
    if ins != [(IMAGE_INPUT, "image")] or outs != list(OUTPUTS):
        raise SystemExit(
            f"[coreml] INTERFACE MISMATCH — the app expects one image input "
            f"{IMAGE_INPUT!r} and outputs {list(OUTPUTS)}; this package would "
            f"load and then return nothing the Swift can read.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--torchscript", required=True, type=Path,
                    help="detector.torchscript.pt from export_deim_onnx "
                         "--no-postprocess")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--name", default=None,
                    help="package name (default: the run directory's name)")
    args = ap.parse_args()
    if not args.torchscript.exists():
        raise SystemExit(f"no such file: {args.torchscript}")
    convert(args.torchscript, args.imgsz, args.name)


if __name__ == "__main__":
    main()
