"""Export the served model to CoreML (for an iOS/macOS app).

Exports the SAME thing the bot serves: the pinned/newest run's best.pt at its
trained imgsz, with that run's holdout-tuned conf/iou baked into the mlpackage's
NMS pipeline (still overridable at runtime via the iouThreshold /
confidenceThreshold inputs CoreML exposes). Conversion runs fine on Linux;
RUNNING the mlpackage needs Apple hardware.

    python -m groundwork.export.coreml --project <slug>
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path

from ..config import OUTPUTS_DIR
from ..serve import runtime_model

OUT_DIR = OUTPUTS_DIR / "coreml"


def export(pp, weights: Path | None = None) -> Path:
    # CPU is all conversion needs — never grab the GPU out from under a retrain.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ultralytics import YOLO   # lazy: exporting is rare, importing torch is slow

    if weights is None:
        weights, imgsz, conf, iou = runtime_model.resolve(pp)
        if weights is None:
            raise SystemExit("no trained run to export — train first")
    else:
        weights = Path(weights)
        imgsz = runtime_model.run_imgsz(weights.parent.parent)
        conf, iou = runtime_model._tuning(weights)
    run = weights.parent.parent.name
    conf = conf if conf is not None else 0.25
    iou = iou if iou is not None else 0.45
    print(f"[coreml] exporting {run} @ imgsz {imgsz} (nms baked: conf={conf} iou={iou})")

    model = YOLO(str(weights))
    exported = Path(model.export(format="coreml", imgsz=imgsz, nms=True,
                                 conf=conf, iou=iou))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"{run}-{imgsz}.mlpackage"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(exported), dst)
    size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"[coreml] wrote {dst} ({size_mb:.1f} MB)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the served model to CoreML.")
    ap.add_argument("--weights", type=Path, default=None,
                    help="a specific best.pt (default: the served pinned/newest run)")
    from ..project import add_argument, load
    from ..dataset import paths
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    print(f"[coreml] project {proj.slug} ({proj.name})")
    export(paths.for_project(proj), args.weights)


if __name__ == "__main__":
    main()
