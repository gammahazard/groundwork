"""Post-run probe timelapse — "watch it learn".

Runs every periodic checkpoint (weights/epoch*.pt, saved natively via
train.py's save_period) plus the final best.pt on a couple of FIXED hard
holdout images, and renders the predicted dots per checkpoint. The frames +
manifest land in <run>/live/ (served by the /outputs mount) for the 🔬
modal's filmstrip slider. Called by the retrain worker AFTER count_eval
(GPU free), best-effort; pure read-only inference — training and eval are
never affected. Checkpoints are pruned afterwards (frames are tiny, the
.pt files are 6MB each).

    python -m groundwork.dataset.viz.timelapse --run yolov8n-27 [--device cpu]
"""
from __future__ import annotations
import argparse
import json
import re

from .. import paths
from ..store import testset_buckets

# Neutral mid thresholds used for EVERY frame — per-epoch tuned values don't
# exist, and a fixed setting keeps frames comparable to each other.
_CONF, _IOU = 0.25, 0.5
_MAX_FRAMES_TRAYS = 2


def _gt(stem: str, pp=None) -> int:
    f = pp.TESTSET_LABELS / f"{stem}.txt"
    return len([l for l in f.read_text().splitlines() if l.strip()]) if f.exists() else 0


def _probe_stems(pp=None) -> list[str]:
    """Fixed, deterministic probes: the fullest capsule-tagged image + the
    fullest image overall — the two hardest exams the holdout offers."""
    counts = {p.stem: _gt(p.stem, pp) for p in pp.TESTSET_LABELS.glob("*.txt")}
    if not counts:
        return []
    tags = testset_buckets.load(pp)
    picks = []
    caps = [s for s in counts if tags.get(s) == "capsule"]
    if caps:
        picks.append(max(caps, key=counts.get))
    dense = max(counts, key=counts.get)
    if dense not in picks:
        picks.append(dense)
    return picks[:_MAX_FRAMES_TRAYS]


def generate(run_name: str, device: str = "", keep_ckpts: bool = False,
             pp=None) -> dict | None:
    from ultralytics import YOLO
    from PIL import Image
    from ...la.annotate import annotate
    from ...la.parse import Detection, Detections
    from ...serve import runtime_model
    from ..store import labelio

    run = pp.RUNS_DIR / run_name
    w = run / "weights"
    if not w.exists():
        print(f"[timelapse] no weights dir for {run_name}")
        return None
    ckpts = sorted(
        ((int(m.group(1)), p) for p in w.glob("epoch*.pt")
         if (m := re.match(r"epoch(\d+)", p.stem))), key=lambda t: t[0])
    ckpts.append((None, w / "best.pt"))          # final frame = the shipped pick
    stems = _probe_stems()
    if not stems:
        print("[timelapse] no holdout images to probe")
        return None
    imgsz = runtime_model.run_imgsz(run)
    live = run / "live"
    live.mkdir(exist_ok=True)

    imgs = {}
    for stem in stems:
        p = labelio.image_path("testset", stem)
        if p:
            imgs[stem] = Image.open(p).convert("RGB")
    manifest = {"imgsz": imgsz, "conf": _CONF, "iou": _IOU,
                "images": {s: [] for s in imgs}}
    for epoch, ckpt in ckpts:
        if not ckpt.exists():
            continue
        model = YOLO(str(ckpt))
        tag = f"ep{epoch:03d}" if epoch is not None else "final"
        for stem, img in imgs.items():
            r = model.predict(img, imgsz=imgsz, conf=_CONF, iou=_IOU,
                              max_det=2000, verbose=False,
                              **({"device": device} if device else {}))[0]
            centers = [((x1 + x2) / 2, (y1 + y2) / 2)
                       for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]
            dets = Detections(items=[Detection("point", (float(x), float(y)))
                                     for x, y in centers])
            name = f"{stem}_{tag}.jpg"
            annotate(img, dets, numbered=False).convert("RGB").save(
                live / name, quality=85)
            manifest["images"][stem].append(
                {"epoch": epoch, "file": name, "pred": len(centers),
                 "gt": _gt(stem)})
        del model
        print(f"[timelapse] {tag}: done", flush=True)
    (live / "timelapse.json").write_text(json.dumps(manifest), encoding="utf-8")
    if not keep_ckpts:
        for _, ckpt in ckpts:
            if ckpt.name.startswith("epoch"):
                ckpt.unlink(missing_ok=True)
        print(f"[timelapse] pruned periodic checkpoints (frames kept in {live})")
    print(f"[timelapse] wrote {live / 'timelapse.json'} "
          f"({sum(len(v) for v in manifest['images'].values())} frames)")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe timelapse for a finished run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="", help="e.g. cpu (default: ultralytics pick)")
    ap.add_argument("--keep-ckpts", action="store_true",
                    help="don't prune the periodic epoch*.pt afterwards")
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    print(f"[timelapse] project {proj.slug} ({proj.name}) | {proj.dataset_root}")
    generate(args.run, device=args.device, keep_ckpts=args.keep_ckpts, pp=pp)


if __name__ == "__main__":
    main()
