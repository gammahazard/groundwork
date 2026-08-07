"""Per-run dataset snapshots: what EXACTLY did this run train and test on?

Written at the end of each retrain (non-fatal, like timelapse/heatmaps):
- <run>/dataset_manifest.json.gz — every train/val/holdout stem with its image
  content-hash and full label text (labels are tiny; ~half a MB covers all).
- outputs/dataset/snapshots/blobs/<md5><ext> — one content-addressed copy of
  each image, shared across ALL runs (an unchanged image costs nothing after
  its first snapshot). Blobs are never auto-pruned.

Together they answer "what changed since run N?" instantly and make any
deleted/edited image restorable byte-exact. Train/val stems are recorded
against raw/ (the editable source of truth); the split dirs are derived
copies wiped on every split.

    python -m groundwork.dataset.pipeline.run_snapshot --run yolov8n-33
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

from .. import paths

def _blobs(pp=None):
    """The content-addressed image store for a project's snapshots.

    A function, not `BLOBS = paths.DATASET_DIR / ...`: that bound one project's
    blob store the moment this module was imported, and restoring an image into the
    wrong project's tree is not a mistake you notice quickly.
    """
    return pp.DATASET_DIR / "snapshots" / "blobs"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _img_for(stem: str, img_dir: Path) -> Path | None:
    return next((p for p in img_dir.glob(f"{stem}.*")
                 if p.suffix.lower() != ".txt"), None)


def _entry(stem: str, img_dir: Path, lbl_dir: Path, src: str, pp=None) -> dict | None:
    img = _img_for(stem, img_dir)
    lbl = lbl_dir / f"{stem}.txt"
    if img is None or not lbl.exists():
        return None
    md5 = _md5(img)
    blobs = _blobs(pp)
    blob = blobs / f"{md5}{img.suffix.lower()}"
    if not blob.exists():
        blobs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, blob)
    return {"file": img.name, "md5": md5, "src": src,
            "labels": lbl.read_text(encoding="utf-8")}


def manifest_path(run: str, pp=None) -> Path:
    return pp.RUNS_DIR / run / "dataset_manifest.json.gz"


def write_snapshot(run: str, pp=None) -> dict:
    """Record the dataset as it stands NOW against `run`. Membership for
    train/val comes from the split dirs (what the run actually consumed);
    bytes/labels come from raw/ and testset/ (the editable sources)."""
    split_stems = {
        s: sorted(p.stem for p in (pp.LABELS_DIR / s).glob("*.txt"))
        for s in ("train", "val")
    }
    out = {"run": run, "created": time.time(), "collections": {}}
    counts = {}
    for name, stems in split_stems.items():
        col = {}
        for stem in stems:
            e = _entry(stem, pp.RAW_IMAGES, pp.RAW_LABELS, "raw", pp)
            if e is not None:
                col[stem] = e
        out["collections"][name] = col
        counts[name] = len(col)
    col = {}
    for lbl in sorted(pp.TESTSET_LABELS.glob("*.txt")):
        e = _entry(lbl.stem, pp.TESTSET_IMAGES, pp.TESTSET_LABELS, "testset", pp)
        if e is not None:
            col[lbl.stem] = e
    out["collections"]["testset"] = col
    counts["testset"] = len(col)
    mp = manifest_path(run, pp)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(mp, "wt", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[snapshot] {run}: " +
          " / ".join(f"{k} {v}" for k, v in counts.items()) +
          f" -> {mp.name} (+ blobs)")
    return counts


def load(run: str, pp=None) -> dict | None:
    mp = manifest_path(run, pp)
    if not mp.exists():
        return None
    with gzip.open(mp, "rt", encoding="utf-8") as f:
        return json.load(f)


def ensure_blobs(run: str, pp=None) -> dict:
    """Back-fill the blob store for a manifest that arrived from another machine.

    An adopted run brings the lab's manifest home (that manifest is the only
    honest record of what the lab trained on), but not the blobs — and it does
    not need to: the lab's dataset is a one-way mirror of this one, so the same
    image bytes are already sitting in raw/. Copy anything the blob store is
    missing straight from the live collection the entry names.

    Cheap by construction: it md5s only the entries whose blob is absent, and
    reports the ones it could not place rather than failing — a missing blob
    costs exactly one thing, the ability to restore that image from this run.
    """
    m = load(run, pp)
    if m is None:
        return {"copied": 0, "missing": []}
    blobs = _blobs(pp)
    copied, missing = 0, []
    for name, col in m.get("collections", {}).items():
        for stem, e in col.items():
            blob = blobs / f"{e['md5']}{Path(e['file']).suffix.lower()}"
            if blob.exists():
                continue
            img_dir, _ = _dirs_for(e, name, pp)
            src = img_dir / e["file"]
            if src.exists() and _md5(src) == e["md5"]:
                blobs.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, blob)
                copied += 1
            else:
                missing.append(stem)
    return {"copied": copied, "missing": sorted(missing)}


def _current_state(pp) -> dict[str, tuple[Path, Path]]:
    return {"raw": (pp.RAW_IMAGES, pp.RAW_LABELS),
            "testset": (pp.TESTSET_IMAGES, pp.TESTSET_LABELS)}


def _dirs_for(e: dict, collection: str, pp) -> tuple[Path, Path]:
    """Where this entry's image lives today: its recorded source collection
    (raw / testset; pre-src manifests default sensibly)."""
    src = e.get("src", "testset" if collection == "testset" else "raw")
    return _current_state(pp)[src]


def diff_vs_current(run: str, pp=None) -> dict | None:
    """For each snapshot collection: stems now missing, stems whose image
    bytes changed, stems whose labels changed. train/val compare against
    raw/ (their source), testset against testset/."""
    m = load(run, pp)
    if m is None:
        return None
    out = {"run": run, "created": m["created"], "collections": {}}
    for name, col in m["collections"].items():
        missing, img_changed, lbl_changed = [], [], []
        for stem, e in col.items():
            img_dir, lbl_dir = _dirs_for(e, name, pp)
            img = _img_for(stem, img_dir)
            lbl = lbl_dir / f"{stem}.txt"
            if img is None or not lbl.exists():
                missing.append(stem)
                continue
            if _md5(img) != e["md5"]:
                img_changed.append(stem)
            if lbl.read_text(encoding="utf-8") != e["labels"]:
                lbl_changed.append(stem)
        out["collections"][name] = {
            "images": len(col), "missing": sorted(missing),
            "image_changed": sorted(img_changed),
            "label_changed": sorted(lbl_changed)}
    return out


def restore_stem(run: str, stem: str, force: bool = False, pp=None) -> dict:
    """Bring one image back exactly as the snapshot recorded it — image bytes
    from the blob store, labels from the manifest — into the collection it
    lived in (train/val stems -> raw/). Refuses to overwrite unless force."""
    m = load(run, pp)
    if m is None:
        raise KeyError(f"no snapshot for run {run!r}")
    # A restore may NEVER create a second copy anywhere: an image that was MOVED
    # (not deleted) still exists in some collection, and restoring it would put
    # the same photo on both sides of the train/test wall. Moves are undone
    # with the 🎯/↩ buttons, not with restore.
    everywhere = {"raw": pp.RAW_LABELS, "testset": pp.TESTSET_LABELS,
                  "synthetic": pp.SYNTHETIC_LABELS,
                  "needs_fix": pp.NEEDS_FIX_LABELS,
                  "halves": pp.HALVES_LABELS}
    for where, lbl_dir in everywhere.items():
        if (lbl_dir / f"{stem}.txt").exists() and not force:
            raise FileExistsError(
                f"{stem} already exists in {where} — it was moved, not deleted; "
                "use the 🎯/↩ move buttons instead of restore")
    # pending stages images before labels exist — check by image too
    if any(pp.PENDING_IMAGES.glob(f"{stem}.*")) and not force:
        raise FileExistsError(f"{stem} is sitting in pending — route it first")
    for name, col in m["collections"].items():
        if stem not in col:
            continue
        e = col[stem]
        img_dir, lbl_dir = _dirs_for(e, name, pp)
        blob = next(iter(_blobs(pp).glob(f"{e['md5']}.*")), None)
        if blob is None:
            raise FileNotFoundError(f"blob missing for {stem} ({e['md5']})")
        dst_img = img_dir / e["file"]
        dst_lbl = lbl_dir / f"{stem}.txt"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(blob, dst_img)
        dst_lbl.write_text(e["labels"], encoding="utf-8")
        return {"ok": True, "stem": stem, "collection": name,
                "restored_to": str(dst_img.parent.parent)}
    raise KeyError(f"{stem!r} not in run {run!r}'s snapshot")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a dataset snapshot for a run.")
    ap.add_argument("--run", required=True)
    from ...project import add_argument, load as load_project
    add_argument(ap)
    args = ap.parse_args()
    proj = load_project(args.project)
    pp = paths.for_project(proj)
    # This file IS the record of what a run trained on — the thing three places
    # once answered differently. It has to name its project out loud.
    print(f"[snapshot] project {proj.slug} ({proj.name}) | {proj.dataset_root}")
    write_snapshot(args.run, pp)


if __name__ == "__main__":
    main()
