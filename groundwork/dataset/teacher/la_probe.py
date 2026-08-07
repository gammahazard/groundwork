"""One-shot LocateAnything probe for a single stored image (crosshair mode).

Runs LA-3B in POINT mode on one collection image and writes the pixel-space
points to a JSON file, then exits — so VRAM is fully released (same fresh-
process discipline as scratch/label_new.sh). Advisory only: never touches
labels; the web dot editor decides whether to adopt the points.

    python -m groundwork.dataset.teacher.la_probe --collection raw \
        --desc "object" --out outputs/la_result.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path


# VRAM-safe retry ceiling, same rationale as autolabel: a 6144-token decode can
# spill on the 8GB card; 1536 tokens ≈ 384 points covers any real image.
_SAFE_RETRY_TOKENS = 1536


def probe(collection: str, stem: str, desc: str, out: Path, pp) -> dict:
    # Heavy imports inside so --help stays instant.
    from ...la.worker import Worker
    from ...la.infer import run_detection
    from ...la import image_ops
    from ..store import labelio

    # The house rule for stage subprocesses: say whose data, before any work.
    print(f"[la_probe] project {pp.slug} | {pp.root}", flush=True)
    p = labelio.image_path(collection, stem, pp)
    if p is None:
        raise FileNotFoundError(f"no image for {stem!r} in {collection!r}")
    img = image_ops.load(str(p))

    t0 = time.time()
    worker = Worker(verbose=True)
    # Stored collection images are already image-cropped, so no autocrop pass —
    # coords come back in the stored image's own pixel space (display space).
    dets, stats, display = run_detection(worker, img, desc, mode="point",
                                         max_retry_tokens=_SAFE_RETRY_TOKENS)
    pts = [[round(float(d.coords[0]), 1), round(float(d.coords[1]), 1)]
           for d in dets.points]
    result = {
        "collection": collection, "stem": stem, "desc": desc,
        "w": display.width, "h": display.height,
        "points": pts, "count": len(pts),
        "seconds": round(time.time() - t0, 1),
        "model_seconds": stats.get("seconds"),
        "used_size": list(stats.get("used_size") or ()),
        "truncation_retry": stats.get("truncation_retry"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"[la_probe] {stem}: {len(pts)} points in {result['seconds']}s "
          f"-> {out}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe one stored image with LocateAnything (point mode).")
    ap.add_argument("--collection", default="raw")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--desc", default="object")
    ap.add_argument("--out", type=Path, required=True)
    from ... import project as project_mod
    project_mod.add_argument(ap)
    args = ap.parse_args()
    from .. import paths
    probe(args.collection, args.stem, args.desc, args.out,
          paths.for_project(args.project))


if __name__ == "__main__":
    main()
