"""split.is_current() must agree with split.build_split(), both ways.

WHAT IT GUARDS. A challenger may now start while a yolo retrain is running,
PROVIDED the split on disk is already what a fresh split would build — because
then the re-split it would otherwise run is pure destruction and recreation of
identical files, and skipping it removes the collision that made the two runs
mutually exclusive.

The entire safety of that rests on is_current() being right. A false YES trains
the challenger on a tree it then leaves alone while yolo reads it — harmless —
but more importantly it means we SKIPPED a split that was actually needed, so
the challenger trains on a stale dataset. That is this repo's most expensive
failure and it is silent. A false NO merely refuses a launch that would have
been fine.

So this builds a real split in a temp tree with the real code, then perturbs it
one way at a time and asserts the verdict flips. Uses ProjectPaths directly, so
nothing touches the actual dataset.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundwork.dataset.paths import ProjectPaths          # noqa: E402
from groundwork.dataset.pipeline import split              # noqa: E402

FAILURES: list[str] = []


def make_tree(root: Path, n: int = 20) -> ProjectPaths:
    pp = ProjectPaths(root=root)
    pp.RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    pp.RAW_LABELS.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # A 1x1 PNG is enough: split copies files, it never decodes them.
        (pp.RAW_IMAGES / f"img_{i:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (pp.RAW_LABELS / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return pp


def expect(cond: bool, msg: str, detail: str = "") -> None:
    if cond:
        print(f"  ✓ {msg}")
    else:
        FAILURES.append(f"{msg}{f' — {detail}' if detail else ''}")
        print(f"  ✗ {msg}  {detail}")


with tempfile.TemporaryDirectory() as td:
    pp = make_tree(Path(td) / "dataset")

    print("  [1] before any split has been built")
    cur, why = split.is_current(pp=pp)
    expect(not cur, "an unbuilt split is not current", why)
    expect("never been built" in why, "and it says so plainly", why)

    print("\n  [2] straight after a real build_split")
    split.build_split(val_frac=0.2, pp=pp)
    cur, why = split.is_current(pp=pp)
    expect(cur, "is_current agrees with the split that just ran", why)

    print("\n  [3] determinism: building again must not change the answer")
    split.build_split(val_frac=0.2, pp=pp)
    cur2, _ = split.is_current(pp=pp)
    expect(cur2, "still current after an identical rebuild")

    print("\n  [4] a NEW image arrives (the stale-split case that must refuse)")
    (pp.RAW_IMAGES / "img_999.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (pp.RAW_LABELS / "img_999.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    cur, why = split.is_current(pp=pp)
    expect(not cur, "a new labelled image makes the split stale", why)
    expect("missing" in why, "and the reason names what is missing", why)
    split.build_split(val_frac=0.2, pp=pp)
    expect(split.is_current(pp=pp)[0], "re-splitting makes it current again")

    print("\n  [5] an image is DELETED from the dataset")
    (pp.RAW_IMAGES / "img_999.png").unlink()
    (pp.RAW_LABELS / "img_999.txt").unlink()
    cur, why = split.is_current(pp=pp)
    expect(not cur, "a removed image also makes it stale", why)
    split.build_split(val_frac=0.2, pp=pp)

    print("\n  [6] someone deletes a file out of the split tree by hand")
    split.build_split(val_frac=0.2, pp=pp)
    victim = next(iter((pp.LABELS_DIR / "train").glob("*.txt")))
    victim.unlink()
    cur, why = split.is_current(pp=pp)
    expect(not cur, "a hand-deleted label in train is detected", why)

    print("\n  [8] a stray file in the split tree")
    split.build_split(val_frac=0.2, pp=pp)
    (pp.LABELS_DIR / "train" / "not_ours.txt").write_text("0 .5 .5 .1 .1\n")
    cur, why = split.is_current(pp=pp)
    expect(not cur, "a stray label in train is detected", why)
    (pp.LABELS_DIR / "train" / "not_ours.txt").unlink()

    print("\n  [9] classes.txt is ignored, as build_split ignores it")
    (pp.LABELS_DIR / "train" / "classes.txt").write_text("object\n")
    cur, why = split.is_current(pp=pp)
    expect(cur, "labelImg's classes.txt does not make a split look stale", why)

print()
if FAILURES:
    for f in FAILURES:
        print(f"  ✗ {f}")
    sys.exit(1)
print("  all checks passed")
