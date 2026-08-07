"""Stage 1: auto-label source images with LocateAnything in BOX mode.

We run the 3B VLM ONCE per image to bootstrap bounding boxes, then a human
fixes misses in the dot editor. BOX mode is mandatory: YOLO needs box extents,
and point mode gives none. Reuses the live pipeline's parsing, near-dup
suppression, OOM ladder, and the SAFE_PATCH_CEILING so this batch pass stays
inside the machine's stable VRAM envelope (never crashes the box).

    python -m groundwork.dataset.teacher.autolabel --project <slug> --images <dir>
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from ...la.annotate import save_annotated
from ...la.infer import run_detection
from ...la.worker import Worker
from ...la import image_ops
from .. import paths
from ..yolo_export import boxes_to_yolo_lines, write_label
from ..point_to_box import points_to_boxes

_EXTS = {".jpeg", ".jpg", ".png", ".webp", ".bmp"}

# VRAM-safe retry ceiling for an 8GB card: a 6144-token decode can spill into
# shared RAM and crash the machine. 1536 tokens ≈ 384 points — covers any real
# scene while keeping the KV cache small. Raise on a bigger card.
_SAFE_RETRY_TOKENS = 1536


def _iter_images(src: Path):
    return sorted(p for p in src.iterdir() if p.suffix.lower() in _EXTS)


def autolabel(src: Path, description: str, pp,
              mode: str = "box", box_strategy: str = "adaptive",
              resume: bool = False, limit: int | None = None,
              only: str | None = None, verbose: bool = True) -> int:
    """Label every image in `src` into dataset/raw. Returns total boxes written.

    mode="box" (preferred): LA emits true box extents.
    mode="point": LA emits points (faster, robust on dense scenes), then
        point_to_box synthesizes boxes via `box_strategy` ("adaptive" | "fixed").
    `description` is the text prompt the teacher detects — usually the
    project's class name.
    """
    for d in (pp.RAW_IMAGES, pp.RAW_LABELS, pp.RAW_PREVIEW):
        d.mkdir(parents=True, exist_ok=True)

    images = _iter_images(src)
    if only:                       # process a single image (fresh process = clean VRAM)
        images = [p for p in images if p.stem == only]
    if limit:
        images = images[:limit]
    if not images:
        print(f"[autolabel] nothing to do (only={only}, src={src})", flush=True)
        return 0
    worker = Worker(verbose=verbose)

    total = 0
    t0 = time.time()
    for i, path in enumerate(images, 1):
        # Resume: skip images already labeled, so a mid-run crash never loses
        # progress — just re-invoke with --resume to continue where it stopped.
        if resume and (pp.RAW_LABELS / f"{path.stem}.txt").exists():
            print(f"[autolabel] {i}/{len(images)} {path.name}: skip (already labeled)",
                  flush=True)
            continue
        img = image_ops.load(str(path))
        # Cap the truncation-retry budget so a dense image can't spike VRAM.
        dets, stats, display = run_detection(worker, img, description, mode=mode,
                                             max_retry_tokens=_SAFE_RETRY_TOKENS)
        if mode == "point":
            # Points carry no size; synthesize boxes so YOLO has extents to learn.
            box_dets = points_to_boxes(dets.points, display.width, display.height,
                                       strategy=box_strategy)
            dets.items = box_dets
        lines = boxes_to_yolo_lines(dets.boxes, display.width, display.height)

        write_label(pp.RAW_LABELS / f"{path.stem}.txt", lines)
        # Save the exact image these labels describe.
        out_name = f"{path.stem}.jpeg"
        display.convert("RGB").save(pp.RAW_IMAGES / out_name, quality=95)
        save_annotated(display, dets, out_name, out_dir=pp.RAW_PREVIEW)

        total += len(lines)
        print(f"[autolabel] {i}/{len(images)} {path.name}: {len(lines)} boxes "
              f"({stats.get('seconds')}s, size {stats.get('used_size')})", flush=True)

    print(f"[autolabel] done: {total} boxes across {len(images)} images in "
          f"{time.time() - t0:.0f}s -> {pp.RAW_DIR}\n"
          f"[autolabel] NEXT: review {pp.RAW_PREVIEW}, then fix missed/wrong "
          f"labels in the web UI's dot editor.", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Auto-label images with LocateAnything (box mode).")
    ap.add_argument("--images", type=Path, required=True, help="source image dir")
    ap.add_argument("--desc", default=None,
                    help="detection description prompt; defaults to the "
                         "project's first class name")
    ap.add_argument("--mode", choices=["box", "point"], default="box",
                    help="box: true extents (preferred); point: synthesize boxes")
    ap.add_argument("--box-strategy", choices=["adaptive", "fixed"], default="adaptive",
                    help="point mode only: how to size synthesized boxes")
    ap.add_argument("--resume", action="store_true",
                    help="skip images already labeled (continue after a crash)")
    ap.add_argument("--only", default=None,
                    help="label just this one image stem (fresh process = clean VRAM)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N images (for previewing)")
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    desc = args.desc or (proj.classes[0] if proj.classes else "object")
    # autolabel WRITES INTO raw/ by design (raw/ is the hand-correction queue),
    # so it is one of the few machine-label paths that reaches the training set.
    # It says which one.
    print(f"[autolabel] project {proj.slug} ({proj.name}) | classes "
          f"{list(proj.classes)} | prompt {desc!r} | "
          f"{len(list(pp.RAW_LABELS.glob('*.txt')))} images already labelled")
    autolabel(args.images, desc, pp=pp,
              mode=args.mode, box_strategy=args.box_strategy,
              resume=args.resume, limit=args.limit, only=args.only)


if __name__ == "__main__":
    main()
