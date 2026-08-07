"""Post-run feature-activation heatmaps — WHERE the network looks.

For the same fixed probe images as the timelapse, run the finished model once
with forward hooks on two layers and render the per-location feature-vector
magnitude as a colored overlay:

  * P3 — the highest-resolution scale feeding the Detect head (stride 8, the
    object-sized scale): bright = strong object-ish features at that spot.
  * SPPF — the deepest backbone block (stride 32): coarse scene context.

Gradient-free (activation magnitude, EigenCAM-style family), so it is robust
across ultralytics versions; the P3/SPPF indices are read from the model's own
wiring (Detect.f and module class names), not hardcoded. Frames + manifest
land in <run>/live/ for the 🔬 modal. Read-only; called post-run (GPU free).

    python -m groundwork.dataset.viz.heatmap --run yolov8n-27 [--device cpu]
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from .. import paths
from .timelapse import _probe_stems


def _pick_layers(mm) -> dict[str, int]:
    """{name: module index} for P3 (first Detect input) and the last SPPF —
    resolved from the model graph itself so version bumps don't break us."""
    layers = {}
    try:
        detect = mm[-1]
        f = getattr(detect, "f", None)
        if isinstance(f, (list, tuple)) and f:
            layers["p3"] = int(f[0])
    except Exception:  # noqa: BLE001
        pass
    for i, m in enumerate(mm):
        if type(m).__name__ == "SPPF":
            layers["sppf"] = i
    return layers


def generate(run_name: str, device: str = "", pp=None) -> dict | None:
    import cv2
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ...serve import runtime_model
    from ..store import labelio

    run = pp.RUNS_DIR / run_name
    best = run / "weights" / "best.pt"
    if not best.exists():
        print(f"[heatmap] no best.pt for {run_name}")
        return None
    stems = _probe_stems(pp)
    if not stems:
        print("[heatmap] no holdout images to probe")
        return None
    imgsz = runtime_model.run_imgsz(run)
    model = YOLO(str(best))
    mm = model.model.model
    layers = _pick_layers(mm)
    if not layers:
        print("[heatmap] could not resolve P3/SPPF layers")
        return None

    acts: dict[str, "torch.Tensor"] = {}
    hooks = []
    for name, idx in layers.items():
        hooks.append(mm[idx].register_forward_hook(
            lambda _m, _i, out, name=name: acts.__setitem__(
                name, out.detach() if isinstance(out, torch.Tensor) else None)))

    live = run / "live"
    live.mkdir(exist_ok=True)
    manifest = {"layers": layers, "images": {}}
    try:
        for stem in stems:
            p = labelio.image_path("testset", stem, pp)
            if not p:
                continue
            img = Image.open(p).convert("RGB")
            acts.clear()
            model.predict(img, imgsz=imgsz, conf=0.25, iou=0.5, max_det=2000,
                          verbose=False, **({"device": device} if device else {}))
            frames = []
            base = np.asarray(img)[:, :, ::-1]           # RGB -> BGR for cv2
            for name in layers:
                feat = acts.get(name)
                if feat is None:
                    continue
                # Per-location feature magnitude: L2 over channels, normalized.
                heat = torch.norm(feat[0].float(), dim=0).cpu().numpy()
                heat = (heat - heat.min()) / (np.ptp(heat) + 1e-6)
                heat = cv2.resize(heat, (img.width, img.height),
                                  interpolation=cv2.INTER_CUBIC)
                color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
                blend = cv2.addWeighted(base, 0.55, color, 0.45, 0)
                name_f = f"heat_{stem}_{name}.jpg"
                cv2.imwrite(str(live / name_f), blend,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append({"layer": name, "file": name_f})
            manifest["images"][stem] = frames
            print(f"[heatmap] {stem}: {len(frames)} layer(s)", flush=True)
    finally:
        for h in hooks:
            h.remove()
    (live / "heatmaps.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"[heatmap] wrote {live / 'heatmaps.json'}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature-activation heatmaps for a run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="", help="e.g. cpu (default: ultralytics pick)")
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    print(f"[heatmap] project {proj.slug} ({proj.name}) | {proj.dataset_root}")
    generate(args.run, device=args.device, pp=pp)


if __name__ == "__main__":
    main()
