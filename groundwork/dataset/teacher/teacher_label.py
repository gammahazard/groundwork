"""Champion-as-teacher: auto-label new photos with the PINNED model.

The distillation trick, one generation down: the champion labels casual,
un-counted photos so challengers get a bigger diet for the cost of a shutter
press.

PROVENANCE RULES (non-negotiable):
  - Output lands in dataset/distill/{images,labels} — its OWN bucket. It
    NEVER enters raw/ (hand-verified only), NEVER the holdout, and feeds
    challengers only via an explicit --include-distill at their tree build.
  - The teacher runs the full serving pipeline (tuned conf/iou + whichever
    filters groundwork/serve/serving_filters enables) so labels match what the
    champion would actually count — box sizes come from the raw detections
    that survive filtering.

    python -m groundwork.dataset.teacher.teacher_label --project <slug> photos/*.jpeg
    python -m groundwork.dataset.teacher.teacher_label --project <slug> --src <dir>
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from .. import paths
from ...serve import serving_filters
from ..filters.dot_dedupe import find_stacked, dot_radius

# Per call: `distill/` belongs to a project, and machine labels landing in the
# wrong project's tree is the silent-wrong-data failure wearing a new hat.
def _distill(pp):
    d = pp.DATASET_DIR / "distill"
    return d / "images", d / "labels"


def label_one(model, img_path: Path, imgsz: int, conf: float, iou: float,
              pp=None) -> int | None:
    if pp is None:
        raise ValueError("label_one needs a project (pp)")
    pil = Image.open(img_path).convert("RGB")
    w, h = pil.size
    r = model.predict(pil, imgsz=imgsz, conf=conf, iou=iou,
                      max_det=2000, verbose=False)[0]
    boxes = r.boxes.xyxy.tolist()
    if not boxes:
        return None
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
    # Gated on the SAME switch as the runtime and both exams — the teacher must
    # write labels through the exact chain the counter it mirrors serves.
    dot_r = dot_radius(w, h)
    stacked = set()
    if serving_filters.DEDUPE:
        for i in find_stacked(centers, dot_r):
            stacked.add(i)
    lines = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if i in stacked:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    DISTILL_IMAGES, DISTILL_LABELS = _distill(pp)
    DISTILL_IMAGES.mkdir(parents=True, exist_ok=True)
    DISTILL_LABELS.mkdir(parents=True, exist_ok=True)
    dst = DISTILL_IMAGES / img_path.name
    if img_path.resolve() != dst.resolve():
        shutil.copy2(img_path, dst)
    (DISTILL_LABELS / f"{img_path.stem}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Champion auto-labels capture photos "
                                             "into the distill bucket.")
    ap.add_argument("photos", nargs="*", type=Path)
    ap.add_argument("--src", type=Path, help="label every image in this folder")
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    print(f"[teacher] project {proj.slug} ({proj.name}) | {proj.dataset_root}")
    todo = list(args.photos)
    if args.src:
        todo += [p for p in args.src.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic")]
    # never re-teach what a human already verified (any bucket), or holdout
    known = set()
    for coll in ("raw", "testset", "pending", "needs_fix"):
        d = pp.DATASET_DIR / coll / "labels"
        if d.exists():
            known |= {f.stem for f in d.glob("*.txt")}
    todo = [p for p in todo if p.stem not in known]
    if not todo:
        print("[teacher] nothing new to label")
        return
    from ...serve import runtime_model
    from ultralytics import YOLO
    w, imgsz, conf, iou = runtime_model.resolve(pp)
    if w is None:
        sys.exit("[teacher] no served model")
    print(f"[teacher] {w.parent.parent.name} @{imgsz} conf={conf} iou={iou} "
          f"-> {len(todo)} photo(s)")
    model = YOLO(str(w))
    for p in sorted(todo):
        n = label_one(model, p, imgsz, conf or 0.25, iou or 0.45, pp)
        print(f"[teacher] {p.stem}: "
              f"{'nothing found — skipped' if n is None else f'{n} objects'}")
    _, distill_labels = _distill(pp)
    print(f"[teacher] distill bucket now: "
          f"{len(list(distill_labels.glob('*.txt')))} image(s) — challengers "
          "opt in with --include-distill; raw/ and the holdout are untouched")


if __name__ == "__main__":
    main()
