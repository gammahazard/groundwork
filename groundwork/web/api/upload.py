"""Put images INTO a project — the other half of Phase 6.

Creating a project was the headline gap; this is what makes one usable. Until
now every image arrived through the Telegram bot, which is exactly right for
the first project (photograph an image, get a count, ✓/✗) and useless for a project that
does not have a bot yet — which is every project on the day it is created.

WHERE UPLOADS LAND, and why it is not `raw/`. They go to the FIX QUEUE
(`needs_fix`), the same place the bot's ✗ sends an image, because an uploaded
image has no labels and `raw/` means "training-ready". A project whose training
set fills up with unlabelled images would train on empty label files and score
zero without anything looking wrong. The fix queue is the front door: label it,
press Done, and collect.promote moves it to raw/ through the same path every
other image takes.

STEMS ARE OURS, NOT THE UPLOADER'S. A filename off a phone is attacker-adjacent
input that becomes a path — labelio.check_stem exists precisely for this, and
the sanitised name is still made unique here, because two files called
IMG_0001.jpg from two phones must not silently become one image. Collisions are
resolved rather than refused: losing an upload to a name clash is a worse
outcome than a suffix.

An empty label file is written next to every image, because every consumer —
list_images, the split, the counts on the card — reads the label file to decide a
image exists. An image with no .txt is an image that is present on disk and invisible
to the system.
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ...dataset.paths import ProjectPaths
from ...dataset.store import labelio, prelabel
from .. import lab_guard
from .deps import project_paths

router = APIRouter()

# What PIL can actually open, and what a phone or a camera roll produces.
_OK_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
# 60MB: a 48MP phone JPEG is ~12MB and a raw-ish PNG can be several times that.
# A cap exists so a stray multi-GB file cannot fill the disk the dataset is on.
_MAX_BYTES = 60 * 1024 * 1024


def _stem_for(name: str, taken: set[str], img_dir: Path) -> str:
    """A safe, unique stem from an uploaded filename.

    `check_stem`'s rules are the ones the rest of the system already trusts —
    reuse them rather than inventing a second idea of what a safe stem is, and
    call it LAST so it is a real check and not decoration.

    COLLAPSING RUNS OF DOTS IS NOT COSMETIC. Replacing disallowed characters can
    CREATE a `..` that was not in the input: `..%2f..%2fx.jpg` has its `%`
    mapped to `_`, and stripping the ends leaves `2f.._2fx` — which check_stem
    rejects, correctly, because `..` in a stem is a traversal primitive. The
    guard caught it and the request 500'd, which is the guard working and the
    caller not. Collapse first so the input is genuinely safe, and let the guard
    stay the backstop it is meant to be.
    """
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)
    base = re.sub(r"\.{2,}", ".", base)              # never manufacture a ".."
    base = re.sub(r"_{2,}", "_", base).strip("._-")[:60]
    base = base or f"upload_{int(time.time())}"
    stem = base
    n = 2
    # Existing files on disk count as taken, not just this batch: uploading the
    # same photo twice on different days must not overwrite the first.
    while stem in taken or any(img_dir.glob(f"{stem}.*")):
        stem = f"{base}_{n}"
        n += 1
    taken.add(stem)
    return labelio.check_stem(stem)


@router.post("/api/upload")
async def upload(files: list[UploadFile] = File(...),
                 predict: bool = Form(False),
                 pp: ProjectPaths = Depends(project_paths)):
    """Add images to this project's fix queue, ready to label.

    `predict` runs the project's SERVING model over each one first and saves its
    objects as a draft label (see dataset/store/prelabel.py). Still the fix
    queue: model output is a draft, and promoting it unreviewed would train the
    next model on this one's mistakes. The model is loaded ONCE for the batch,
    not per file — twenty photos is twenty inferences, not twenty model loads.

    Refused on the lab like every other mutation: its tree is a one-way mirror,
    so anything uploaded there is deleted by the next sync.

    Per-file failures are REPORTED, not raised. Dropping twenty photos in and
    getting a single 400 because one was a .mov tells you nothing about the
    other nineteen — which did land, which did not, and why.
    """
    lab_guard.no_lab_edits()
    img_dir, lbl_dir = labelio._dirs("needs_fix", pp)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    counter, predict_err = None, ""
    if predict:
        try:
            from ...serve import runtime_model
            # THIS project's model. It was the machine-global one, so ticking
            # pre-label on a bolts upload ran the OBJECT counter over bolts
            # photos and saved the output as draft labels.
            counter = runtime_model.make_counter(pp)
        except Exception as e:  # noqa: BLE001
            # No pinned model, or it will not load. The upload still HAPPENS —
            # refusing twenty photos because the optional half failed would be
            # the worst of both.
            predict_err = f"could not load this project's model ({type(e).__name__})"

    added, skipped = [], []
    taken: set[str] = set()
    for f in files:
        name = f.filename or "unnamed"
        suffix = Path(name).suffix.lower()
        if suffix not in _OK_SUFFIX:
            skipped.append({"name": name, "why": f"not an image ({suffix or 'no extension'})"})
            continue
        data = await f.read()
        if not data:
            skipped.append({"name": name, "why": "empty file"})
            continue
        if len(data) > _MAX_BYTES:
            skipped.append({"name": name,
                            "why": f"{len(data) // (1024*1024)}MB is over the "
                                   f"{_MAX_BYTES // (1024*1024)}MB limit"})
            continue
        try:
            stem = _stem_for(name, taken, img_dir)
        except KeyError as e:
            # check_stem refused. That is the backstop doing its job on a name
            # the sanitiser above should already have made safe — so it is a
            # SKIPPED file with a reason, never a 500. A crafted filename must
            # not be able to make the endpoint fall over for the twenty valid
            # photos uploaded alongside it.
            skipped.append({"name": name, "why": f"unusable filename ({e.args[0]})"})
            continue
        (img_dir / f"{stem}{suffix}").write_bytes(data)
        # The empty label file is what makes it a IMAGE rather than a loose image
        # — see the module docstring. Written FIRST, so an image exists even if the
        # optional pre-label below fails.
        (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
        row = {"name": name, "stem": stem}
        if counter is not None:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(data)).convert("RGB")
                r = prelabel.from_model(counter, img, "needs_fix", stem, pp)
                row["predicted"] = r.get("count", 0)
                if not r.get("ok"):
                    row["predict_error"] = r.get("error")
            except Exception as e:  # noqa: BLE001
                row["predict_error"] = type(e).__name__
        added.append(row)

    print(f"[upload] {pp.slug}: {len(added)} added, {len(skipped)} skipped "
          f"-> {img_dir}", flush=True)
    if not added and skipped:
        # Nothing landed at all — that IS the error, and the reasons are the
        # useful part of it.
        raise HTTPException(422, "; ".join(f"{s['name']}: {s['why']}" for s in skipped[:5]))
    return {"ok": True, "added": added, "skipped": skipped,
            "collection": "needs_fix", "project": pp.slug,
            # Reported, not raised: the images landed either way, and "you asked
            # for pre-labels and did not get them" is something you must be told.
            "predict": bool(counter is not None), "predict_error": predict_err}
