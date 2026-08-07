"""Which model serves is a property of a PROJECT, not of the machine.

    .venv/bin/python checks/check_serving_project.py

WHAT THIS CATCHES, and it was live rather than hypothetical. `active_model.json`
and `serving_engine.json` were one pair per MACHINE, and `runtime_model.resolve()`
read one fixed project's runs whoever asked. Three project-scoped endpoints
call `make_counter()`:

    /api/upload with "pre-label with this project's model"   writes DRAFT LABELS
    /api/label_audit                                          judges a project's
                                                              images against them
    /api/count                                                the Count tab

So uploading to a bolts project with pre-labelling ticked ran the FIRST
project's counter over bolts photographs and saved the output as that
project's labels. Nothing errors; it just produces training data from the
wrong model.

PART 4 IS THE ONE THAT WOULD BITE HARDEST. `/api/model/activate` takes a weights
PATH in the request body. Scoping the file it writes to is not enough — without a
check that the path belongs to this project, the same bug walks back in through
the front door, and worse, because a pin persists and the bot reads it too.

HOW IT CAN FAIL: every assertion pins or switches in ONE project and requires the
OTHER to be unchanged. Reverting to a single global file makes parts 1-3 red,
because both projects would then read the same answer.

SAFE: two throwaway projects under a temp root, removed at the end. No GPU (no
counter is ever loaded — only the resolution that chooses one), no network.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


from groundwork import project as project_mod          # noqa: E402
from groundwork.serve import runtime_model             # noqa: E402
from groundwork.dataset import paths                   # noqa: E402

A, B = "servecheck-a", "servecheck-b"


def _mk(slug: str, root: Path):
    p = project_mod.Project(slug=slug, name=slug, dataset_root=root)
    project_mod.manifest_path(slug).parent.mkdir(parents=True, exist_ok=True)
    project_mod.save(p)
    pp = paths.for_project(slug)
    # A run with a checkpoint, so resolve() has something to find.
    w = pp.RUNS_DIR / f"{slug}-1" / "weights"
    w.mkdir(parents=True, exist_ok=True)
    (w / "best.pt").write_bytes(b"not really a checkpoint")
    return pp, w / "best.pt"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="checkserve-"))
    made = []
    try:
        pp_a, w_a = _mk(A, tmp / A / "dataset"); made.append(A)
        pp_b, w_b = _mk(B, tmp / B / "dataset"); made.append(B)

        # ---- 1. the pin --------------------------------------------------
        print("\n1. pinning a model in one project does not touch the other")
        runtime_model.set_active(w_a, 1280, pp_a)
        note(runtime_model.get_active(pp_a)["weights"] == str(w_a),
             "A serves what A pinned")
        note(runtime_model.get_active(pp_b) is None,
             f"B is still unpinned (got {runtime_model.get_active(pp_b)})")
        runtime_model.set_active(w_b, 960, pp_b)
        note(runtime_model.get_active(pp_a)["imgsz"] == 1280
             and runtime_model.get_active(pp_b)["imgsz"] == 960,
             "and now they hold DIFFERENT answers at the same time")

        # ---- 2. they are separate files ----------------------------------
        print("\n2. each project has its own file, outside every dataset tree")
        fa, _ = runtime_model._serving(pp_a)
        fb, _ = runtime_model._serving(pp_b)
        note(fa != fb, "the two pins are different paths")
        note(str(pp_a.DATASET_DIR) not in str(fa),
             "and NOT inside the dataset tree (the mirror would rsync it to the worker)")

        # ---- 3. resolve() follows the project -----------------------------
        print("\n3. resolve() returns each project's own weights")
        note(runtime_model.resolve(pp_a)[0] == w_a, "A resolves to A's checkpoint")
        note(runtime_model.resolve(pp_b)[0] == w_b, "B resolves to B's checkpoint")
        runtime_model.clear(pp_a)
        note(runtime_model.resolve(pp_a)[0] == w_a,
             "unpinned, A still finds its OWN newest run (not B's)")
        note(runtime_model.get_active(pp_b) is not None,
             "and clearing A did not clear B")

        # ---- 4. a pin cannot point at another project's run ---------------
        print("\n4. /api/model/activate refuses a foreign checkpoint")
        from groundwork.web.api import train as train_api
        from fastapi import HTTPException
        try:
            train_api.activate(train_api.ActivateOpts(weights=str(w_b), imgsz=960),
                               pp_a)
            note(False, "A was allowed to pin B's checkpoint")
        except HTTPException as e:
            note(e.status_code == 422, f"refused with {e.status_code}")
        try:
            train_api.activate(train_api.ActivateOpts(weights=str(w_a), imgsz=1280),
                               pp_a)
            note(True, "...while its own checkpoint is accepted")
        except HTTPException as e:
            note(False, f"its own checkpoint was refused: {e.detail}")

        # ---- 5. the engine switch is per project --------------------------
        print("\n5. the serving engine is per project too")
        _, ea = runtime_model._serving(pp_a)
        _, eb = runtime_model._serving(pp_b)
        note(ea != eb, "separate engine files")
        note(runtime_model.get_engine(pp_a) == "yolo"
             and runtime_model.get_engine(pp_b) == "yolo",
             "both default to yolo with nothing exported")
        # ...and no machine-global pair exists to shadow them: the per-project
        # files live under outputs/serving/<slug>/, never at outputs/ root.
        note(str(runtime_model.SERVING_DIR / A) in str(fa)
             and str(runtime_model.SERVING_DIR / B) in str(fb),
             "each pin lives under outputs/serving/<slug>/")
    finally:
        for slug in made:
            shutil.rmtree(project_mod.manifest_path(slug).parent, ignore_errors=True)
            shutil.rmtree(runtime_model.SERVING_DIR / slug, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("PASS — each project pins, switches and resolves its own model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
