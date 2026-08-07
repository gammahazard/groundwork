"""YOLO-txt -> COCO converter for the sidecar trainers (run under the MAIN .venv).

Train/val MEMBERSHIP is imported from groundwork.dataset.pipeline.split (_val_rank
ranking, same val_frac/seed defaults), so the challenger trains on byte-identical
image membership to the champion's runs. The frozen holdout never enters this
tree — rfdetr's required test/ dir is a mirror of valid/.

Ends with a mandatory self-check that reloads every written JSON and re-counts
boxes per image against the source txt files — a category/coordinate bug must
abort here, never silently eat labels.

    .venv/bin/python -m altmodels.convert            # -> outputs/alt/datasets/coco_rfdetr
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image

from groundwork.dataset import paths
from groundwork.dataset.pipeline.split import _stems_with_labels, _val_rank, select_val

REPO = Path(__file__).resolve().parents[1]
OUT_FALLBACK = REPO / "outputs" / "alt" / "datasets" / "coco_rfdetr"


def _dirs_of(stem: str, pp) -> tuple[Path, Path]:
    """Which collection a stem lives in — raw is the only source tree."""
    return pp.RAW_IMAGES, pp.RAW_LABELS


def _yolo_lines(stem: str, pp) -> list[list[float]]:
    """Non-empty label lines as [cls, cx, cy, w, h] floats (normalized)."""
    _, lbl_dir = _dirs_of(stem, pp)
    txt = (lbl_dir / f"{stem}.txt").read_text(encoding="utf-8")
    return [[float(v) for v in ln.split()[:5]]
            for ln in txt.splitlines() if ln.strip()]


def _bad_classes(stems: list[str], pp) -> list[str]:
    """Stems carrying a class index this project does not declare.

    THE BOUND IS THE PROJECT'S, not a hardcoded 0 — the same fix split.py already
    made, quoted here because this file had the same bug and it bites in the same
    place: "it was `!= '0'`, written when there was one project with one class…
    For a project that declares three classes, 1 and 2 are exactly what it is
    FOR." Left as-is, every multi-class project would have its labels rejected by
    the converter as corrupt.
    """
    nc = len(pp.CLASS_NAMES)
    return [s for s in stems
            if any(not (0 <= int(l[0]) < nc) for l in _yolo_lines(s, pp))]


def _membership(val_frac: float, seed: int, pp) -> tuple[list[str], list[str]]:
    stems = _stems_with_labels(pp)
    if not stems:
        raise SystemExit(f"no labeled images in {pp.RAW_LABELS}")
    bad = _bad_classes(stems, pp)
    if bad:
        raise SystemExit(f"[convert] {len(bad)} image(s) carry a class index "
                         f"outside this project's {len(pp.CLASS_NAMES)} class(es) "
                         f"({', '.join(bad[:8])}) — same guard as split.py")
    # Call split's rule rather than re-deriving it: two copies of the
    # membership rule once produced different val sets, and the whole point of
    # --match-split is byte-identical data.
    val = select_val(stems, val_frac, seed)
    train = [s for s in stems if s not in set(val)]
    return train, val


def _categories(pp) -> list[dict]:
    """COCO categories from the PROJECT's class list.

    COCO ids are 1-based and YOLO's are 0-based, so category_id == cls + 1.
    """
    return [{"id": i + 1, "name": n, "supercategory": "objects"}
            for i, n in enumerate(pp.CLASS_NAMES)]


def _write_split(out: Path, name: str, stems: list[str], copy: bool, pp) -> dict:
    d = out / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    images, annotations = [], []
    ann_id = 1
    for img_id, stem in enumerate(sorted(stems), start=1):
        img_dir, _ = _dirs_of(stem, pp)
        src = next(p for p in img_dir.glob(f"{stem}.*")
                   if p.suffix.lower() != ".txt")
        with Image.open(src) as im:
            W, H = im.size
        dst = d / src.name
        if copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)
        images.append({"id": img_id, "file_name": src.name,
                       "width": W, "height": H})
        for cls, cx, cy, w, h in _yolo_lines(stem, pp):
            x = max(0.0, (cx - w / 2) * W)
            y = max(0.0, (cy - h / 2) * H)
            bw = min(w * W, W - x)
            bh = min(h * H, H - y)
            assert bw > 0 and bh > 0, f"degenerate box in {stem}"
            annotations.append({"id": ann_id, "image_id": img_id,
                                # cls + 1: COCO is 1-based, YOLO is 0-based.
                                "category_id": int(cls) + 1,
                                "bbox": [x, y, bw, bh],
                                "area": bw * bh, "iscrowd": 0})
            ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": _categories(pp)}
    (d / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return {"images": len(images), "boxes": len(annotations)}


def _self_check(out: Path, splits: dict[str, list[str]],
                val_frac: float, seed: int, pp) -> None:
    """Reload what was written; any mismatch vs the source txts aborts loudly."""
    train, val = _membership(val_frac, seed, pp)   # re-derive independently
    assert splits["train"] == sorted(train) \
        and splits["valid"] == sorted(val), \
        "membership drifted between write and check"
    _check_written(out, splits, pp)


def _check_written(out: Path, splits: dict[str, list[str]], pp) -> None:
    """Per-image validation of the written COCO jsons vs the source txts."""
    want_ids = [c["id"] for c in _categories(pp)]
    for name, stems in splits.items():
        coco = json.loads((out / name / "_annotations.coco.json").read_text())
        assert [c["id"] for c in coco["categories"]] == want_ids
        per_img: dict[int, int] = {}
        img_dims = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
        for a in coco["annotations"]:
            assert a["category_id"] in want_ids, \
                f"{name}: stray category {a['category_id']} (declared {want_ids})"
            W, H = img_dims[a["image_id"]]
            x, y, w, h = a["bbox"]
            assert -0.01 <= x and -0.01 <= y and x + w <= W + 0.01 and y + h <= H + 0.01, \
                f"{name}: out-of-bounds bbox on image {a['image_id']}"
            per_img[a["image_id"]] = per_img.get(a["image_id"], 0) + 1
        by_name = {i["file_name"]: per_img.get(i["id"], 0) for i in coco["images"]}
        bad = [f for f in by_name
               if by_name[f] != len(_yolo_lines(Path(f).stem, pp))]
        if bad:
            raise SystemExit(f"[convert] SELF-CHECK FAILED ({name}): box-count "
                             f"mismatch vs txt for {bad[:8]}")
        assert {Path(i["file_name"]).stem for i in coco["images"]} == set(stems)
        assert all((out / name / i["file_name"]).exists() for i in coco["images"])
    print("[convert] self-check OK — every image's box count matches its txt")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO raw/ -> COCO tree for rfdetr.")
    ap.add_argument("--out", type=Path, default=OUT_FALLBACK)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    ap.add_argument("--match-split", action="store_true",
                    help="take train/val membership DIRECTLY from the split dirs "
                         "(labels/train, labels/val) — exactly what the last "
                         "champion run trained on")
    from groundwork.project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    # SAY WHICH DATASET, before doing anything. Every stage in this repo prints
    # its project, root and image count first, because silent wrong-data is the
    # most expensive failure here and a converter feeding the wrong images to a
    # trainer is exactly that failure with an extra step.
    print(f"[convert] project {proj.slug} ({proj.name}) | classes "
          f"{list(proj.classes)} | {proj.dataset_root}")

    if args.match_split:
        train = sorted(p.stem for p in (pp.LABELS_DIR / "train").glob("*.txt"))
        val = sorted(p.stem for p in (pp.LABELS_DIR / "val").glob("*.txt"))
        if not train or not val:
            raise SystemExit("[convert] --match-split: split dirs are empty — "
                             "run a champion retrain (or split) first")
        bad = _bad_classes(train + val, pp)
        if bad:
            raise SystemExit(f"[convert] class index outside this "
                             f"project's classes in split stems: {bad[:8]}")
        print(f"[convert] --match-split: {len(train)} train / {len(val)} val "
              "taken verbatim from the champion's split dirs")
        stats = {}
        for name, stems in (("train", train), ("valid", val), ("test", val)):
            stats[name] = _write_split(args.out, name, stems, args.copy, pp)
            print(f"[convert]   {name}: {stats[name]['images']} images / "
                  f"{stats[name]['boxes']} boxes")
        _check_written(args.out, {"train": train, "valid": val, "test": val}, pp)
        print(f"[convert] wrote {args.out} (holdout untouched, as always)")
        return

    train, val = _membership(args.val_frac, args.seed, pp)
    print(f"[convert] membership (split.py hash ranking, val_frac={args.val_frac} "
          f"seed={args.seed}): {len(train)} train / {len(val)} val")
    stats = {}
    for name, stems in (("train", train), ("valid", val), ("test", val)):
        stats[name] = _write_split(args.out, name, stems, args.copy, pp)
        print(f"[convert]   {name}: {stats[name]['images']} images / "
              f"{stats[name]['boxes']} boxes")
    _self_check(args.out, {"train": sorted(train), "valid": sorted(val),
                           "test": sorted(val)}, args.val_frac, args.seed, pp)
    print(f"[convert] wrote {args.out} (test/ mirrors valid/ — the frozen "
          "holdout is NOT here and never will be)")


if __name__ == "__main__":
    main()
