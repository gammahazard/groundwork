"""Image review + in-browser dot editing: list, image, points get/save, promote."""
from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...dataset.store import labelio, collect
from ...dataset import paths
from ...dataset.paths import ProjectPaths
from .. import lab_guard, retrain_job
from .deps import project_paths

router = APIRouter()


def _holdout_locked() -> str | None:
    """The holdout is FROZEN while a retrain runs — count_eval reads testset/
    at the END of the run, so adding an image mid-run would score the model on
    data it may just have trained on (a leak), and removing one shrinks the
    in-flight exam. Returns the refusal message when locked, else None."""
    if retrain_job.status().get("status") == "running":
        return ("a retrain is running — the holdout is mid-exam (it's scored "
                "at the end of the run). Move test-set images after it finishes.")
    return None


class Points(BaseModel):
    points: list[list[float]]     # pixel-space [x, y]


class Truth(BaseModel):
    count: int | None = None      # None = clear (unknown) target


class CropBox(BaseModel):
    box: list[float]              # [x1, y1, x2, y2] pixel coords in the CURRENT image


@router.get("/api/images/{collection}")
def images(collection: str, pp: ProjectPaths = Depends(project_paths)):
    try:
        return labelio.list_images(collection, pp)
    except KeyError:
        raise HTTPException(404, "unknown collection")


@router.get("/api/points/{collection}/{stem}")
def get_points(collection: str, stem: str,
               pp: ProjectPaths = Depends(project_paths)):
    """An image's points, plus the VOCABULARY they are drawn from.

    `classes` travels with the image because the editor needs it to label at all:
    a point's third element is its class index, and without the names the editor
    can only ever write 0. That is not hypothetical — the first project has 10,247
    labels and every one of them is class 0, which is correct for a one-class
    project and would be silently wrong for a project that declared two.

    Sent here rather than fetched separately so it cannot disagree with the
    points it describes: both come from the same ProjectPaths in one request.
    """
    try:
        d = labelio.load_points(collection, stem, pp)
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "image not found")
    return {**d, "classes": list(pp.CLASS_NAMES)}


@router.post("/api/points/{collection}/{stem}")
def save_points(collection: str, stem: str, body: Points,
                pp: ProjectPaths = Depends(project_paths)):
    lab_guard.no_lab_edits()
    try:
        return {"count": labelio.save_points(collection, stem, body.points, pp)}
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/api/crop/{collection}/{stem}")
def crop(collection: str, stem: str, body: CropBox,
         pp: ProjectPaths = Depends(project_paths)):
    """Trim a too-zoomed-out image to the drawn box; dots outside are dropped."""
    lab_guard.no_lab_edits()
    if len(body.box) != 4:
        raise HTTPException(422, "box must be [x1,y1,x2,y2]")
    try:
        return labelio.crop_image(collection, stem, body.box, pp)
    except KeyError as e:
        raise HTTPException(404, str(e))


THUMB_SIZES = (320, 480)        # grid card widths; anything else is ignored


def _thumb_dir(pp=None) -> Path:
    """Where cached grid thumbnails live. Machine-global (`outputs/.thumbs`),
    because it is a cache, not data — but keyed BY PROJECT below."""
    return pp.DATASET_DIR.parent / ".thumbs"


def _thumb(src: Path, w: int, pp=None) -> Path | None:
    """A cached downscaled JPEG for grid use, regenerated when src changes.

    The grids render ~240px cards but were downloading the full original —
    ~46 MB and 155 requests for the training grid, which is slow over the
    network (i.e. from a phone). A 320px thumbnail is ~15-20 KB.

    Keyed project/collection/stem. It used to be collection/stem only, which is
    fine with one project and silently wrong with two: `raw/IMG_1` in a second
    project would hit the first project's cached thumbnail and the grid would
    show someone else's image. The freshness check is an mtime compare, so it
    would not even self-correct.
    """
    out = _thumb_dir(pp) / pp.slug / src.parent.parent.name / f"{src.stem}_{w}.jpg"
    try:
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return out                       # still fresh
        out.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        im.thumbnail((w, w * 4), Image.LANCZOS)   # cap width, keep aspect
        tmp = out.with_suffix(".tmp.jpg")         # atomic: never serve a partial
        im.save(tmp, "JPEG", quality=80, optimize=True)
        tmp.replace(out)
        return out
    except Exception:  # noqa: BLE001 — fall back to the original, never 500
        return None


@router.get("/img/{collection}/{stem}")
def image(collection: str, stem: str, w: int = 0, v: int = 0,
          pp: ProjectPaths = Depends(project_paths)):
    p = labelio.image_path(collection, stem, pp)
    if not p:
        raise HTTPException(404, "image not found")
    if w in THUMB_SIZES and (t := _thumb(p, w, pp)) is not None:
        # ?v=<mtime> makes the URL change whenever the image does, so this can
        # be cached hard — no revalidation round-trip per image.
        cache = "public, max-age=31536000, immutable" if v else "no-cache"
        return FileResponse(t, headers={"Cache-Control": cache})
    # no-cache = browsers revalidate every time (cheap 304 when unchanged), so
    # the full-size image can't go stale after an in-place edit like ✂ crop.
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


@router.post("/api/promote/{stem}")
def promote(stem: str, pp: ProjectPaths = Depends(project_paths)):
    """Mark a needs_fix image done -> move it into the training set."""
    lab_guard.no_lab_edits()
    return {"ok": collect.graduate(stem, pp)}


@router.post("/api/promote_testset/{stem}")
def promote_testset(stem: str, pp: ProjectPaths = Depends(project_paths)):
    """Mark a needs_fix image done -> move it into the holdout test set instead."""
    lab_guard.no_lab_edits()
    if (bar := _synth_barred(stem)):
        return bar
    if (msg := _holdout_locked()):
        return {"ok": False, "error": msg}
    return {"ok": collect.graduate_to_testset(stem, pp)}


@router.post("/api/truth/{stem}")
def set_truth(stem: str, body: Truth,
              pp: ProjectPaths = Depends(project_paths)):
    """Fix a wrong/missing true count you typed on Telegram (or clear it)."""
    lab_guard.no_lab_edits()
    if body.count is None:
        collect.forget_truth(stem, pp)
        return {"ok": True, "target": None}
    collect.set_truth(stem, body.count, pp)
    return {"ok": True, "target": int(body.count)}


@router.post("/api/testset/{stem}")
def to_testset(stem: str, confirm: int = 0,
               pp: ProjectPaths = Depends(project_paths)):
    """Move a raw image into the fixed holdout test set (out of training)."""
    lab_guard.no_lab_edits()
    if (msg := _holdout_locked()):
        return {"ok": False, "error": msg}
    return {"ok": collect.to_testset(stem, pp)}


@router.delete("/api/testset/{stem}")
def from_testset(stem: str, pp: ProjectPaths = Depends(project_paths)):
    """Return a test-set image to raw (back into training)."""
    lab_guard.no_lab_edits()
    if (msg := _holdout_locked()):
        return {"ok": False, "error": msg}
    return {"ok": collect.from_testset(stem, pp)}


@router.delete("/api/image/{collection}/{stem}")
def delete(collection: str, stem: str,
           pp: ProjectPaths = Depends(project_paths)):
    """Remove an image (e.g. a duplicate) from a collection."""
    lab_guard.no_lab_edits()
    if collection == "testset" and (msg := _holdout_locked()):
        return {"ok": False, "error": msg}
    try:
        return {"ok": labelio.delete(collection, stem, pp)}
    except KeyError:
        raise HTTPException(404, "unknown collection")


@router.post("/api/label_audit")
def label_audit_start(collection: str = "raw",
                      pp: ProjectPaths = Depends(project_paths)):
    """START a sweep of this collection. Returns at once; poll GET for progress.

    It used to block for the whole ~138 seconds, which is indistinguishable
    from a hang — long enough that you reload, and a reload started a SECOND
    sweep on the same card. See web/audit_job.py.

    REFUSED WHILE A RETRAIN RUNS HERE: it wants the same card, and an audit is
    never worth risking a run.
    """
    if retrain_job.status().get("status") == "running":
        raise HTTPException(409, "a retrain is running on this machine — it "
                                 "needs the card. Try again when it finishes.")
    from ...serve import runtime_model
    from ...web import audit_job
    # A FACTORY, so the model loads on the worker thread. Building it here would
    # put the one slow step back on the request this endpoint exists to keep
    # fast — but check it CAN be built, so "no model yet" is an error you get
    # now rather than a job that fails a second later.
    # WAS `get_engine() is None`, which can never be true — get_engine falls
    # back to "yolo" for any unreadable value, so the guard never fired and a
    # project with no weights got a job that failed a second later instead of
    # a clear refusal. resolve() answering None IS the question being asked.
    if runtime_model.resolve(pp)[0] is None:
        raise HTTPException(400, "no model is serving this project yet")
    # The FACTORY is bound to this project — audit_job calls it on a worker
    # thread, where there is no request to ask which project this is.
    r = audit_job.start(lambda: runtime_model.make_counter(pp), collection, pp)
    if not r.get("ok"):
        raise HTTPException(409, r.get("error", "already running"))
    return r


@router.get("/api/label_audit")
def label_audit_status():
    """Progress of the running sweep, or the last one's result.

    Machine-level on purpose: the job is single-flight for this process, and
    "is the card busy with an audit" is not a per-project question.
    """
    from ...web import audit_job
    return audit_job.status()


@router.post("/api/label_audit/{collection}/{stem}")
def label_audit_points(collection: str, stem: str,
                       pp: ProjectPaths = Depends(project_paths)):
    """WHERE this image disagrees: the serving model's points for one image.

    The sweep says which images to look at; this is what you look AT. POST for
    the same reason the sweep is: it runs the model, and that must not happen
    because something prefetched a URL.
    """
    from ...serve import runtime_model
    try:
        counter = runtime_model.make_counter(pp)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"no model is serving this project "
                                 f"({type(e).__name__})") from None
    from ...dataset.viz import label_audit as audit
    r = audit.points_for(counter, collection, stem, pp)
    # Which weights drew these marks. The overlay shows it, because the audit
    # uses the PINNED model and a different pin gives a different answer.
    r["model_name"] = runtime_model.counter_name(counter)
    if not r.get("ok"):
        raise HTTPException(404, r.get("error", "could not run the model"))
    return r


@router.get("/api/dedup")
def dedup(a: str = "raw", b: str = "testset",
          pp: ProjectPaths = Depends(project_paths)):
    """Same-capture check between buckets: perceptual image hash (re-encodes/copies)
    + label-geometry twins (re-shoots the hash misses — the dot constellation is
    a fingerprint of the capture)."""
    from ...dataset.viz import dedup_check
    return {
        "flagged": [{"a": x, "b": y, "dist": d} for x, y, d in dedup_check.near_duplicates(a, b, pp=pp)],
        "twins": [{"a": x, "b": y, "dist": d}
                  for x, y, d in dedup_check.label_twins(a, b, pp=pp)],
        "closest": [{"a": x, "b": y, "dist": d}
                    for x, y, d in dedup_check.closest(a, b, 5, pp)],
    }
