"""Draw a project's own model over an image, and save the result as its labels.

WHAT THIS IS FOR. A newly uploaded image arrives with an empty label file, and
labelling a 300-object image from nothing is the slowest thing in this whole system.
If the project already has a serving model, it can do the first pass — you then
FIX it, which is minutes instead of an hour. That is the same trade the
LocateAnything teacher makes for bootstrapping, at 40ms instead of 40 seconds,
using a model that has already been judged on this project's own holdout.

IT STILL LANDS IN THE FIX QUEUE. Model output is a draft, not ground truth, and
`raw/` means training-ready. Pre-labelled or not, an upload goes to `needs_fix`
and reaches the training set only when a human presses Done — see api/upload.py.
Skipping that would train a model on its own predictions, which is how a
counting error becomes a permanent one.

BOXES, NOT DOTS. `stats["boxes"]` carries the model's per-object extents, so the
saved label keeps a MEASURED width and height rather than the adaptive square
`save_points` synthesises for a hand-clicked dot. That matters for exactly the
reason PIVOT's Phase 0 existed: a box that survives a round trip is worth more
than a point, and throwing the extent away here would be discarding something
the model already knew.

THE CROP IS THE TRAP. `count()` may run on an AUTOCROPPED image, and the boxes it
reports are in THAT image's coordinate space. Saving them against the original's
dimensions would put every dot in the wrong place — squeezed toward the top-left
by exactly the crop offset, which looks like a slightly bad model rather than a
coordinate bug. Autocrop is OFF today (serving_filters) so crop_box is None and
the two spaces coincide; this maps back anyway, because "it happens to be off"
is not a reason to be wrong when it is on.
"""
from __future__ import annotations

from . import labelio


def from_model(counter, image, collection: str, stem: str, pp) -> dict:
    """Run `counter` over `image` and save its objects as this image's labels.

    Returns {ok, count, error} — never raises. One image the model chokes on
    must not lose the other nineteen in the same upload, and an image that
    arrives with no labels is still a usable image.
    """
    try:
        r = counter.count(image, name=f"prelabel_{stem}", stage=False)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "count": 0, "error": f"{type(e).__name__}"}

    boxes = (r.stats or {}).get("boxes") or []
    # Back to ORIGINAL image space. crop_box is (left, top, right, bottom) in the
    # original; the model saw only that region, so every coordinate it reports is
    # relative to the crop's top-left.
    cb = (r.stats or {}).get("crop_box")
    ox, oy = (cb[0], cb[1]) if cb else (0.0, 0.0)

    points: list[list[float]] = []
    for b in boxes:
        try:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
        except (TypeError, ValueError):
            continue
        # [cx, cy, cls, bw, bh] — cls 0 is "whole", the only class every project
        # has. save_points keeps an extent when one is given.
        points.append([(x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy, 0,
                       abs(x2 - x1), abs(y2 - y1)])
    if not points:
        # A real answer: the model found nothing. The empty label file the
        # uploader already wrote is correct, so leave it rather than rewriting.
        return {"ok": True, "count": 0}
    try:
        n = labelio.save_points(collection, stem, points, pp)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "count": 0, "error": f"save failed ({type(e).__name__})"}
    return {"ok": True, "count": n if isinstance(n, int) else len(points)}
