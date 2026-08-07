"""Ad-hoc count: upload an image photo, get the count + dots overlay (no staging)."""
from __future__ import annotations
import io
import time
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File
from PIL import Image

from ...serve import runtime_model
from ...dataset.paths import ProjectPaths
from ...serve.yolo_counter import YoloCounter
from .deps import project_paths

router = APIRouter()
# ONE RESIDENT COUNTER PER PROJECT, keyed by slug. It was a single global, which
# is the whole of this endpoint's share of the machine-global serving bug: ask
# project A for a count, then project B, and B got A's model — the counter was
# cached and nothing in the key said which project it belonged to.
_counters: dict[str, tuple[YoloCounter, tuple]] = {}


def _get(pp: ProjectPaths) -> YoloCounter:
    """This project's resident counter, rebuilt when its resolved model changes.

    resolve() is cheap file I/O; comparing its (weights, imgsz, conf, iou) each
    request means a new pin, a finished retrain, or re-tuned thresholds are
    served immediately — the web process never needs a restart to catch up."""
    # the engine is part of the key: flipping yolo<->deim swaps the resident
    # counter on the next request, no restart
    key = (runtime_model.get_engine(pp), runtime_model.resolve(pp))
    hit = _counters.get(pp.slug)
    if hit is None or hit[1] != key:
        hit = (runtime_model.make_counter(pp), key)
        _counters[pp.slug] = hit
    return hit[0]


@router.post("/api/count")
async def count(image: UploadFile = File(...),
                pp: ProjectPaths = Depends(project_paths)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    r = _get(pp).count(img, name=f"web_{int(time.time())}", stage=False)
    return {"count": r.count, "seconds": r.seconds, "project": pp.slug,
            "overlay": "/outputs/" + Path(r.overlay_path).name}
