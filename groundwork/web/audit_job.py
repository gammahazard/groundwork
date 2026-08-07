"""Single-flight background label audit, with progress the UI can show.

WHY A JOB AND NOT A REQUEST. The sweep runs the serving model over every image
in a collection — 160 images is ~138 seconds. As one blocking POST that is two
minutes of a spinner saying nothing, which is indistinguishable from a hang;
long enough that you reload the page, and reloading starts a second sweep on
the same card.

Mirrors la_job/retrain_job's shape — start(), status(), single-flight — but in
a THREAD with in-memory state rather than a subprocess with a state file. The
work is a model already resident in this process (the same one /api/count uses),
so a subprocess would mean loading it a second time, and the state is worth
nothing after a restart: a half-finished audit is simply re-run.

STEPS, NOT JUST A PERCENTAGE. What it is doing changes what waiting means —
listing the collection is instant, the first image includes loading the model,
and the rest is steady. So the state carries a named step as well as a count,
and the UI reads them out rather than showing a bar that sits at 0 for five
seconds and then leaps.

REFUSES WHILE A RETRAIN RUNS HERE, and refuses to start twice. Both are the
same rule as everything else that wants this machine's card.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_state: dict = {"status": "idle"}


def _idle() -> dict:
    return {"status": "idle", "step": "", "done": 0, "total": 0,
            "collection": None, "result": None, "error": None,
            "model": None, "started": None, "seconds": 0}


_state = _idle()


def status() -> dict:
    with _lock:
        s = dict(_state)
    if s.get("started") and s["status"] == "running":
        s["seconds"] = round(time.time() - s["started"], 1)
        # A rate is only meaningful once an image has actually finished — the
        # first one carries the model load and would predict a wild ETA.
        if s["done"] > 1 and s["total"]:
            per = s["seconds"] / s["done"]
            s["eta_s"] = max(0, round(per * (s["total"] - s["done"])))
    return s


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def start(counter_factory, collection: str, pp) -> dict:
    """Begin a sweep. Returns immediately; poll status().

    `counter_factory` rather than a counter, so the model is loaded on the
    WORKER thread — building it here would put the one slow step back on the
    request this function exists to keep fast.
    """
    with _lock:
        if _state["status"] == "running":
            return {"ok": False, "error": "an audit is already running",
                    "state": dict(_state)}
        _state.update(_idle())
        _state.update({"status": "running", "step": "starting",
                       "collection": collection, "started": time.time()})

    def work() -> None:
        try:
            from ..dataset.store import labelio
            from ..dataset.viz import label_audit

            _set(step="listing the images in this collection")
            images = labelio.list_images(collection, pp)
            _set(total=len(images),
                 step=f"{len(images)} image(s) found — loading the model")

            counter = counter_factory()
            # NAME THE MODEL. An audit's whole meaning depends on which weights
            # produced it, and the pin can change between runs of it.
            from ..serve import runtime_model
            name = runtime_model.counter_name(counter)
            _set(model=name, step=f"comparing each image with {name}")

            def on_progress(done: int, total: int, stem: str) -> None:
                _set(done=done, total=total,
                     step=f"comparing {done}/{total} with {name} — {stem}")

            r = label_audit.sweep(counter, collection, pp, on_progress=on_progress)
            _set(step="collecting the disagreements")
            _set(status="done", result=r, step="", done=r.get("checked", 0))
        except Exception as e:  # noqa: BLE001 — a failed audit must not take
            _set(status="error", error=f"{type(e).__name__}: {e}")  # the server

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "state": status()}
