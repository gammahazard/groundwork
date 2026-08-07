"""Ask the serving model to check the labels, and report where they disagree.

WHAT IT FINDS. Every image in a collection has a dot count that a human put
there. Run the model over the same image and you get a second opinion. Where
they differ, one of them is wrong — and the interesting case is when it is the
LABEL, because a mislabelled training image teaches the next model that mistake
and nothing else in the system will ever mention it. The frozen holdout is
verified by hand, but nothing has ever re-checked the 160 training images.

IT IS A SUSPICION, NOT A VERDICT, and the wording everywhere says so. The model
is wrong sometimes — that is the entire reason the fix queue exists — so a
disagreement means "look at this one", never "this label is wrong". Images are
returned sorted by how badly they disagree, because that ordering is the whole
value: the top of the list is where an hour of checking pays.

THE DOT COUNT, NOT THE TARGET. `collect.get_truth` holds a target for images that
came through the bot with a ✗ correction, and most images do not have one. Using
it would silently audit a subset while appearing to audit everything — so the
comparison is against the number of LABEL ROWS, which every image has by
construction (an image with no .txt is not an image at all).

SPEED. One inference per image, model loaded once, no staging and no overlay
written — about 40ms of GPU plus the image decode, so ~160 images is under a
minute. The caller decides whether the machine is free; this just counts.
"""
from __future__ import annotations

from PIL import Image

from ..store import labelio


def _dots(collection: str, stem: str, pp) -> int:
    """How many objects the LABEL claims. Rows in the .txt, nothing else."""
    _, lbl_dir = labelio._dirs(collection, pp)
    f = lbl_dir / f"{stem}.txt"
    try:
        return sum(1 for ln in f.read_text(errors="ignore").splitlines() if ln.strip())
    except OSError:
        return 0


def sweep(counter, collection: str, pp, limit: int | None = None,
          on_progress=None) -> dict:
    """Compare every image's dot count with what `counter` sees.

    Returns {checked, agreed, flagged:[{stem, dots, model, diff}], errors:[...]}
    with `flagged` sorted worst-first. Never raises for one bad image: an
    image that will not open is an ERROR entry, and the others still get
    checked.
    """
    img_dir, _ = labelio._dirs(collection, pp)
    images = labelio.list_images(collection, pp)
    if limit:
        images = images[:limit]

    flagged, errors = [], []
    checked = 0
    for i, t in enumerate(images):
        stem = t["stem"]
        matches = sorted(img_dir.glob(f"{stem}.*"))
        img_path = next((m for m in matches if m.suffix.lower() != ".txt"), None)
        if img_path is None:
            errors.append({"stem": stem, "why": "no image file"})
            continue
        try:
            with Image.open(img_path) as im:
                r = counter.count(im.convert("RGB"), name=f"audit_{stem}", stage=False)
        except Exception as e:  # noqa: BLE001 — one bad image must not end the sweep
            errors.append({"stem": stem, "why": type(e).__name__})
            continue
        dots = _dots(collection, stem, pp)
        checked += 1
        if r.count != dots:
            flagged.append({"stem": stem, "dots": dots, "model": r.count,
                            "diff": r.count - dots})
        if on_progress:
            on_progress(i + 1, len(images), stem)

    # Worst disagreement first — that ordering IS the product here.
    flagged.sort(key=lambda x: (-abs(x["diff"]), x["stem"]))
    return {"collection": collection, "checked": checked,
            "agreed": checked - len(flagged), "flagged": flagged,
            "errors": errors, "total": len(images)}


def _pair(mine, theirs, tol):
    """Greedy nearest-neighbour match. Returns (label-only, model-only).

    Greedy is right here: the two sets are nearly identical and the tolerance is
    tight, so an optimal assignment would cost more to explain than it could buy
    on a two-dot difference.

    IN PYTHON, not in the browser. The pairing rule decides what you are shown,
    and a second copy in JS is a second thing to keep in step — the same reason
    the collection vocabulary and the run facts are server-side. It also means
    the comparison is against the SAVED labels, which is what an audit is of.
    """
    used, missing = set(), []
    t2 = tol * tol
    for a in mine:
        best, bd = -1, t2
        for i, b in enumerate(theirs):
            if i in used:
                continue
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d <= bd:
                bd, best = d, i
        if best >= 0:
            used.add(best)
        else:
            missing.append(a)
    return missing, [b for i, b in enumerate(theirs) if i not in used]


def _low_conf(counter, image, floor: float):
    """Every box the model would produce at a much lower score floor.

    THE POINT: a label dot the served model does not report is two completely
    different situations, and they need opposite responses. If the model DID see
    something there at 0.19 and the served threshold is 0.20, the label is fine
    and the threshold cut it — nothing to fix, and lowering the threshold is
    exactly what would let a fingertip back in. If there is nothing there at any
    score, the model is genuinely blind to that object and the image is worth
    training on. Telling those apart by hand takes a separate run each time.

    Uses the counter's RESIDENT model, so this is a second inference and not a
    second model load.
    """
    try:
        r = counter.model.predict(image, imgsz=counter.imgsz, conf=floor,
                                  iou=counter.iou, max_det=counter.max_det,
                                  verbose=False)[0]
    except Exception:  # noqa: BLE001 — a counter without .model is not an error
        return []
    out = []
    for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        out.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, round(float(c), 3)))
    return out


def points_for(counter, collection: str, stem: str, pp, floor: float = 0.02) -> dict:
    """One image's MODEL points, in original-image pixels — for the editor overlay.

    The sweep answers "which images disagree"; this answers "where". Separate
    call because the sweep is a minute of GPU across the whole collection and
    this is 40ms for the one image you opened.

    Coordinates are mapped back through `crop_box` for the same reason
    prelabel.from_model does: count() may run on an autocropped image and its
    boxes are in THAT space, so saving or drawing them against the original's
    dimensions puts every mark up and left by the crop offset — which looks like
    a slightly bad model rather than a coordinate bug.
    """
    from PIL import Image
    img_dir, _ = labelio._dirs(collection, pp)
    match = next((m for m in sorted(img_dir.glob(f"{stem}.*"))
                  if m.suffix.lower() != ".txt"), None)
    if match is None:
        return {"ok": False, "error": "no image for that stem"}
    try:
        with Image.open(match) as im:
            im_for_low = im.convert("RGB")
            w, h = im_for_low.size
            r = counter.count(im_for_low, name=f"auditpts_{stem}", stage=False)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": type(e).__name__}
    cb = (r.stats or {}).get("crop_box")
    ox, oy = (cb[0], cb[1]) if cb else (0.0, 0.0)
    pts = []
    for b in (r.stats or {}).get("boxes") or []:
        try:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
        except (TypeError, ValueError):
            continue
        pts.append([round((x1 + x2) / 2 + ox, 1), round((y1 + y2) / 2 + oy, 1)])
    # The comparison itself, against the SAVED labels.
    try:
        d = labelio.load_points(collection, stem, pp)
        mine = [[float(p[0]), float(p[1])] for p in d.get("points") or []]
    except Exception:  # noqa: BLE001
        mine = []
    from ..filters import dot_dedupe
    tol = max(8.0, dot_dedupe.dot_radius(w, h) * 2)
    missing, extra = _pair(mine, pts, tol)

    # WHY each ⊖ happened: did the model nearly see it, or not at all?
    low = _low_conf(counter, im_for_low, floor)
    annotated = []
    for mx, my in missing:
        near, nd = None, (tol * 2) ** 2
        for lx, ly, c in low:
            # NO CROP OFFSET HERE. _low_conf predicts on the FULL image, so its
            # coordinates are already original-image pixels; `pts` above come
            # from count()'s boxes, which are crop-space and do need it. Adding
            # it to both applied the offset twice and, with autocrop on, matched
            # every missing dot against the wrong neighbourhood.
            dd = (mx - lx) ** 2 + (my - ly) ** 2
            if dd <= nd:
                nd, near = dd, c
        annotated.append({"x": round(mx, 1), "y": round(my, 1), "conf": near})
    served = getattr(counter, "conf", None)
    return {"ok": True, "points": pts, "model": r.count,
            "dots": _dots(collection, stem, pp), "w": w, "h": h,
            "extra": [{"x": round(x, 1), "y": round(y, 1)} for x, y in extra],
            "missing": annotated, "served_conf": served, "floor": floor}
