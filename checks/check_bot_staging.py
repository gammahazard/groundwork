"""A counter stages into ITS OWN project's pending queue, never another's.

    .venv/bin/python checks/check_bot_staging.py

WHAT THIS CATCHES, and it was live rather than hypothetical. In the
predecessor, `YoloCounter` accepted a `pp` and used it only to resolve WEIGHTS
— it never stored it — so `count()` staged its predicted labels with no
project at all, and the store fell back to one fixed project's tree. Every bot
on the machine therefore wrote its predicted labels into that project's
`pending/labels`, whatever `GW_PROJECT` its unit set.

The half that made it nasty: the bot re-staged the FULL-RES image through its
own project, which was correct. So a second project's bot produced a SPLIT
PAIR — the image in its own project, the labels in the first project's — and
neither side errored. That is the silent-wrong-data failure this repo treats
as its most expensive, with the added bite that one user's photos land in
another user's dataset once there is more than one account.

HOW IT CAN FAIL: part 1 requires the staged pair to appear under project B,
and part 2 requires that no project-less fallback path EXISTS — save_pending
must refuse to run without a project. Dropping `pp=self.pp` from
yolo_counter.count() turns part 1 red; giving save_pending a default project
again turns part 2 red.

SAFE: no GPU and no network — `_detect` is stubbed, so ultralytics is never
imported and no checkpoint is loaded. Two throwaway projects under a temp
root, removed at the end. The annotated PNG is redirected into the temp dir.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


from groundwork import project as project_mod          # noqa: E402
from groundwork.dataset import paths                   # noqa: E402
from groundwork.dataset.store import collect           # noqa: E402
from groundwork.serve.yolo_counter import YoloCounter  # noqa: E402

A, B = "stagecheck-a", "stagecheck-b"
STEM = "stagecheck_sample"


class StubCounter(YoloCounter):
    """The REAL count(), with only the model stubbed out.

    __init__ is replaced rather than extended because the real one imports
    ultralytics and loads a checkpoint. Every attribute count() reads is set
    here, so what runs below is the shipping code path — filters, ordering,
    staging — and not a re-implementation of it.
    """

    def __init__(self, pp):
        self.pp = pp
        self.conf, self.iou, self.imgsz, self.max_det = 0.25, 0.5, 640, 300
        self.weights = Path("stub") / "weights" / "best.pt"

    def _detect(self, image):
        # Two well-separated boxes: far enough apart that the stacked-dot dedupe
        # cannot fire, so the staged label count is predictable.
        return [[10.0, 10.0, 40.0, 40.0], [200.0, 200.0, 230.0, 230.0]]


def _mk(slug: str, root: Path):
    p = project_mod.Project(slug=slug, name=slug, dataset_root=root)
    project_mod.manifest_path(slug).parent.mkdir(parents=True, exist_ok=True)
    project_mod.save(p)
    return paths.for_project(slug)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="checkstage-"))
    made: list[str] = []
    try:
        pp_a = _mk(A, tmp / A / "dataset"); made.append(A)
        pp_b = _mk(B, tmp / B / "dataset"); made.append(B)

        img = Image.new("RGB", (320, 320), "navy")

        # ---- 1. staging follows the counter's project ----------------------
        print("\n1. a counter built for B stages into B")
        StubCounter(pp_b).count(img, name=STEM, out_dir=tmp, stage=True)
        note((pp_b.PENDING_IMAGES / f"{STEM}.jpeg").exists(),
             "B's pending/images has the image")
        lab = pp_b.PENDING_LABELS / f"{STEM}.txt"
        note(lab.exists(), "B's pending/labels has the labels")
        if lab.exists():
            n = len([ln for ln in lab.read_text().splitlines() if ln.strip()])
            note(n == 2, f"and it holds both detections (got {n})")

        # ---- 2. no project-less fallback path EXISTS -----------------------
        # This is the assertion that encodes the old bug. The store used to
        # fall back to one fixed project when no pp was passed; now `pp` is a
        # required argument with no default, so the split-pair failure cannot
        # be reached silently — it is a TypeError at the call site.
        print("\n2. save_pending cannot run without a project")
        try:
            collect.save_pending(STEM + "_2", img, [])  # no pp
            note(False, "save_pending accepted a call with no project — a "
                        "fallback tree exists again")
        except TypeError:
            note(True, "save_pending REQUIRES a project (no ambient default)")

        # ---- 3. the other project is untouched -----------------------------
        print("\n3. the other project is untouched")
        note(not (pp_a.PENDING_IMAGES / f"{STEM}.jpeg").exists()
             and not (pp_a.PENDING_LABELS / f"{STEM}.txt").exists(),
             "A's pending queue is empty")

        # ---- 4. a subclass that never sets pp must not RAISE ----------------
        # DeimCounter.__init__ does not chain to super() — it assigns each
        # attribute itself — so making `pp` instance-only turned the staging
        # line into an AttributeError on every challenger count. The
        # class-level default is what keeps an unset `pp` a visible wrong
        # answer instead of a crash mid-count, and this is the assertion that
        # says so.
        print("\n4. a challenger that forgets pp does not crash mid-count")
        # getattr with a sentinel, not YoloCounter.pp — removing the default
        # should make this line report FAIL, not raise and take the run with it.
        note(getattr(YoloCounter, "pp", "MISSING") is None,
             "YoloCounter carries a class-level pp default")

        class ForgetfulCounter(StubCounter):
            def __init__(self):                    # sets no pp at all
                self.conf, self.iou = 0.25, 0.5
                self.imgsz, self.max_det = 640, 300
                self.weights = Path("stub") / "weights" / "best.pt"

        try:
            ForgetfulCounter().count(Image.new("RGB", (64, 64)), name="forgetful",
                                     out_dir=tmp, stage=False)
            note(True, "count() runs without an instance pp (stage=False)")
        except AttributeError as e:
            note(False, f"count() raised on a missing pp: {e}")

        from groundwork.serve import deim_counter                     # noqa: PLC0415
        src = (Path(deim_counter.__file__)).read_text()
        note("self.pp = pp" in src,
             "DeimCounter assigns pp too (it does not call super().__init__)")
        src = (Path(sys.modules["groundwork.serve.yolo_counter"].__file__)).read_text()
        note("pp=self.pp" in src,
             "and count()'s staging line passes the counter's OWN project")
    finally:
        for slug in made:
            shutil.rmtree(project_mod.manifest_path(slug).parent, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("all good — staging follows the project its counter was built for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
