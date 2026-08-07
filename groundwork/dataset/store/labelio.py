"""Read/write label points per image for a given collection (raw or needs_fix).

Shared by the web cockpit (and reusable elsewhere): points are stored as YOLO
boxes on disk (adaptive size), but the UI works in point space, so we convert on
the way in/out. Pure + GPU-free.
"""
from __future__ import annotations
import re

from PIL import Image

from .. import paths
from ..point_to_box import boxes_for
from ..yolo_export import box_to_yolo_line, write_label
from ...la.parse import Detection
from . import collect

# The full collection VOCABULARY. Kept as a plain tuple because that is what
# its two consumers actually want: a membership test (web/api/la.py) and a set
# of CLI choices (viz/point_editor.py). Neither ever wanted the directories.
#
# The core three (raw / needs_fix / testset) are universal: every project has
# a training source, a fix queue and a frozen holdout. An extension can bring
# a collection with it — register it in _NEEDS_EXTENSION so it exists only for
# projects that enable that extension.
COLLECTION_NAMES = ("raw", "needs_fix", "testset")

# collection -> the extension that has to be enabled for it to exist. Empty
# today; the mechanism stays because a collection appearing for every project
# when only one enables its tooling is a real leak, and this is where it is
# stopped.
_NEEDS_EXTENSION: dict[str, str] = {}

# Which DIRECTORIES those names point at IS per-project, so it cannot be a
# module constant. It used to be: `COLLECTIONS` was a dict built at import time
# from module path constants, which froze this module to one project the moment
# it was imported — the reason a mapping like this cannot be fixed by "just
# passing an argument" and had to be turned inside out.
_SUFFIXES = {
    "raw": ("RAW_IMAGES", "RAW_LABELS"),
    "needs_fix": ("NEEDS_FIX_IMAGES", "NEEDS_FIX_LABELS"),
    "testset": ("TESTSET_IMAGES", "TESTSET_LABELS"),
}


def collection_names(pp) -> tuple[str, ...]:
    """The collections THIS project actually has, in vocabulary order.

    Extension-gated ones are dropped when the project does not enable them."""
    ext = set(pp.extensions)
    return tuple(c for c in COLLECTION_NAMES
                 if _NEEDS_EXTENSION.get(c) in (None, *ext))


def collections(pp) -> dict:
    """{collection: (images dir, labels dir)} for one project."""
    return {name: _dirs(name, pp) for name in collection_names(pp)}


def _dirs(collection: str, pp):
    if collection not in _SUFFIXES:
        raise KeyError(f"unknown collection {collection!r}")
    img, lbl = _SUFFIXES[collection]
    return getattr(pp, img), getattr(pp, lbl)


# Stems come straight from URL path segments and are fed to glob() / path
# joins. Anything beyond plain filename characters (glob metachars *?[],
# separators, '..') could match or reach files it must not — e.g.
# DELETE /api/image/raw/* would unlink every training image. Reject early.
_SAFE_STEM = re.compile(r"^[A-Za-z0-9_.-]+$")


def check_stem(stem: str) -> str:
    if not _SAFE_STEM.fullmatch(stem) or ".." in stem:
        raise KeyError(f"unsafe stem {stem!r}")
    return stem


def image_path(collection: str, stem: str, pp):
    img_dir, _ = _dirs(collection, pp)
    if not _SAFE_STEM.fullmatch(stem):
        return None
    for p in img_dir.glob(f"{stem}.*"):
        if p.suffix.lower() != ".txt":
            return p
    return None


def _count(lbl_dir, stem: str) -> int:
    f = lbl_dir / f"{stem}.txt"
    return len([l for l in f.read_text().splitlines() if l.strip()]) if f.exists() else 0


def list_images(collection: str, pp) -> list[dict]:
    """Images in a collection with their dot count and (if any) target count.

    Ordered by when they were ADDED (oldest first, newest at the bottom of
    the grids). Birth time where the filesystem provides it; mtime fallback
    (an in-place edit like ✂ crop may bump an image down — acceptable)."""
    img_dir, lbl_dir = _dirs(collection, pp)

    def added(p):
        st = p.stat()
        return (getattr(st, "st_birthtime", None) or st.st_mtime, p.name)

    out = []
    for p in sorted((x for x in img_dir.glob("*.*")
                     if x.suffix.lower() != ".txt"), key=added):
        # mtime lets the grid ask for an immutably-cacheable thumbnail URL: it
        # changes when the image does (e.g. an in-place crop), so the browser
        # can cache hard without ever showing a stale image.
        out.append({"stem": p.stem, "count": _count(lbl_dir, p.stem),
                    "target": collect.get_truth(p.stem, pp),
                    "earmarked": collect.is_earmarked(p.stem, pp),
                    "mtime": int(p.stat().st_mtime)})
    return out


def load_points(collection: str, stem: str, pp) -> dict:
    """Return {'w','h','points':[[x, y, cls, bw, bh], ...]} + target.

    cls is the class INDEX into the project's class list. bw/bh are the box's
    width/height IN PIXELS. A label
    line that carries no extent yields a SHORT point `[x, y, cls]` rather than
    nulls — absence is the signal save_points already checks for, and the API's
    `list[list[float]]` body model would reject a null anyway.

    This used to read `cls, cx, cy, *_` — dropping the stored width and height on
    the floor — while save_points re-derived every box from dot SPACING. The
    round trip therefore destroyed any hand-sized box the moment an image was
    reopened and saved, which made a box editor impossible to build. It also
    means every label written before 2026-07-30 is a pixel-square sized by
    spacing rather than by the object: position is real, extent is synthetic.
    """
    _, lbl_dir = _dirs(collection, pp)
    img = Image.open(image_path(collection, stem, pp))
    w, h = img.size
    pts = []
    f = lbl_dir / f"{stem}.txt"
    if f.exists():
        for ln in f.read_text().splitlines():
            if not ln.strip():
                continue
            parts = ln.split()
            cls, cx, cy = parts[0], parts[1], parts[2]
            pt = [float(cx) * w, float(cy) * h, int(float(cls))]
            if len(parts) > 4:
                pt += [float(parts[3]) * w, float(parts[4]) * h]
            pts.append(pt)
    return {"w": w, "h": h, "points": pts, "target": collect.get_truth(stem, pp),
            "earmarked": collect.is_earmarked(stem, pp), "collection": collection}


def save_points(collection: str, stem: str, points: list[list[float]],
                pp) -> int:
    """Persist points as YOLO boxes, preserving per-point class AND extent.

    A point may be [x, y], [x, y, cls] or [x, y, cls, bw, bh]. One that carries
    an extent keeps it; one that does not (a dot just clicked) gets the adaptive
    default sized from its nearest neighbour. So an image nobody resized saves
    byte-identically to how it saved before this existed, and a resized box
    survives every subsequent save.

    Which box each point becomes is `point_to_box.boxes_for` — ONE definition,
    mirrored in the editor so what you see drawn is what gets written.
    """
    check_stem(stem)
    _, lbl_dir = _dirs(collection, pp)
    img = Image.open(image_path(collection, stem, pp))
    w, h = img.size
    lines = []
    for p, coords in zip(points, boxes_for(points, w, h)):
        cls = int(p[2]) if len(p) > 2 else 0
        line = box_to_yolo_line(Detection(kind="box", coords=coords, label="object"),
                                w, h, cls)
        if line:
            lines.append(line)
    write_label(lbl_dir / f"{stem}.txt", lines)
    return len(lines)


def crop_image(collection: str, stem: str, box, pp) -> dict:
    """Crop an image to `box` = pixel [x1,y1,x2,y2] in the CURRENT frame, and
    remap its dots into the cropped frame (dots outside the box are dropped).
    For trimming too-zoomed-out shots so background can't spawn false
    detections. Overwrites the image + label in place. Returns the new count
    and #dropped dots."""
    check_stem(stem)
    p = image_path(collection, stem, pp)
    if p is None:
        raise KeyError(f"no image for {stem!r} in {collection!r}")
    img = Image.open(p).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(int(round(x1)), W - 1)); y1 = max(0, min(int(round(y1)), H - 1))
    x2 = max(x1 + 1, min(int(round(x2)), W)); y2 = max(y1 + 1, min(int(round(y2)), H))
    before = load_points(collection, stem, pp)["points"]  # ragged [x,y,cls(,bw,bh)]
    img.crop((x1, y1, x2, y2)).save(
        p, **({"quality": 95} if p.suffix.lower() in (".jpg", ".jpeg") else {}))
    # Shift into the crop frame, drop what falls outside, and carry the REST of
    # each row through untouched — that is the class and, since 2026-07-30, the
    # box extent. A crop translates a box; it does not resize it. (This used to
    # unpack `for px, py, cls in before`, which raises the moment a point also
    # carries w/h.)
    kept = [[p[0] - x1, p[1] - y1, *p[2:]] for p in before
            if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
    n = save_points(collection, stem, kept, pp)        # reopens the NEW image for dims
    return {"ok": True, "count": n, "dropped": len(before) - len(kept),
            "w": x2 - x1, "h": y2 - y1}


def delete(collection: str, stem: str, pp) -> bool:
    """Remove an image (+ label + any truth count) — e.g. a duplicate."""
    check_stem(stem)
    img_dir, lbl_dir = _dirs(collection, pp)
    removed = False
    for p in img_dir.glob(f"{stem}.*"):
        if p.suffix.lower() != ".txt":
            p.unlink(missing_ok=True); removed = True
    (lbl_dir / f"{stem}.txt").unlink(missing_ok=True)
    collect.forget_truth(stem, pp)
    return removed
