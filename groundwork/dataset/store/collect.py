"""Feedback loop: stage counted images and route them by ✓/✗ into the dataset.

Flow (driven from the front-ends' ✓/✗ buttons):
  count      -> save_pending(): image + predicted labels -> pending/
  ✓ correct  -> promote():      pending/ -> raw/ (training-ready)
  ✗ wrong    -> reject():       pending/ -> needs_fix/ (fix in point editor)
  after fix  -> promote_fixed(): needs_fix/ -> raw/

Pure file moves (no GPU). Names are unique per message (e.g. tg_<uid>_<msgid>),
so bot samples never collide with directly-uploaded images.
"""
from __future__ import annotations
import shutil
from pathlib import Path

from PIL import Image

from .. import sidecar

# Ground-truth counts the user reports at counting time (predicted count was
# wrong). Keyed by sample name; survives moves between collections. The point
# editor shows this as a target when correcting the points.
#
# Every reader/writer below takes `pp` REQUIRED and resolves the path per call.
# A module constant would bind one project's path at import time, which no
# argument can undo — and there is no default project to fall back to.
def _truth_path(pp):
    return pp.DATASET_DIR / "truth_counts.json"


def _load_truth(pp) -> dict:
    """STRICT on a corrupt file — deliberately.

    These are hand-typed correct counts, and every writer does
    load -> mutate -> save. If a bad read returned {} the very next set_truth
    would rewrite the file with ONE entry and destroy the rest. Raising is the
    safe failure: the image grid shows an error, the data is still on disk, and
    someone looks. Multiple processes write this (a bot's feedback handler, the
    web target editor, the janitor), which is why the writes go through
    sidecar's atomic replace."""
    return sidecar.read_json(_truth_path(pp), {}, strict=True)


def set_truth(name: str, count: int, pp) -> None:
    d = _load_truth(pp); d[name] = int(count)
    sidecar.write_json(_truth_path(pp), d)


def get_truth(name: str, pp) -> int | None:
    return _load_truth(pp).get(name)


def forget_truth(name: str, pp) -> None:
    d = _load_truth(pp)
    if d.pop(name, None) is not None:
        sidecar.write_json(_truth_path(pp), d)


# Test earmark: "this image is meant for the holdout, decided up front"
# (independent of whether the model got it right). Keyed by sample name so it
# survives pending -> needs_fix; cleared once the image graduates. Stored like
# the truth counts above so it's just as simple/inspectable — and per-project
# for the same reason.
def _earmark_path(pp):
    return pp.DATASET_DIR / "test_earmark.json"


def _load_earmarks(pp) -> list:
    """STRICT, same reasoning as _load_truth: this records which images are
    destined for the FROZEN holdout, and a silent reset would lose that."""
    return sidecar.read_json(_earmark_path(pp), [], strict=True)


def set_earmark(name: str, pp) -> None:
    marks = _load_earmarks(pp)
    if name not in marks:
        marks.append(name)
        sidecar.write_json(_earmark_path(pp), marks)


def is_earmarked(name: str, pp) -> bool:
    return name in _load_earmarks(pp)


def clear_earmark(name: str, pp) -> None:
    marks = _load_earmarks(pp)
    if name in marks:
        marks.remove(name)
        sidecar.write_json(_earmark_path(pp), marks)


def _move(stem: str, img_from: Path, lbl_from: Path,
          img_to: Path, lbl_to: Path) -> bool:
    """Move <stem> image+label between collections. Returns False if not found."""
    from .labelio import _SAFE_STEM      # local import: labelio imports collect
    if not _SAFE_STEM.fullmatch(stem):   # stem comes from URLs; never glob metachars
        return False
    imgs = list(img_from.glob(f"{stem}.*"))
    lbl = lbl_from / f"{stem}.txt"
    if not imgs or not lbl.exists():
        return False
    img_to.mkdir(parents=True, exist_ok=True)
    lbl_to.mkdir(parents=True, exist_ok=True)
    shutil.move(str(imgs[0]), str(img_to / imgs[0].name))
    shutil.move(str(lbl), str(lbl_to / f"{stem}.txt"))
    return True


def save_pending(name: str, image: Image.Image, lines: list[str], pp) -> None:
    """Stage a counted sample (image + predicted YOLO labels)."""
    pp.PENDING_IMAGES.mkdir(parents=True, exist_ok=True)
    pp.PENDING_LABELS.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(pp.PENDING_IMAGES / f"{name}.jpeg", quality=95)
    (pp.PENDING_LABELS / f"{name}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def promote(name: str, pp) -> bool:
    """✓ correct: pending -> raw (training set, ready for the next retrain)."""
    return _move(name, pp.PENDING_IMAGES, pp.PENDING_LABELS,
                 pp.RAW_IMAGES, pp.RAW_LABELS)


def promote_to_testset(name: str, pp) -> bool:
    """🎯 correct -> testset: count was right, but hold this image out of
    training as a frozen eval sample instead of adding it to raw (the label IS
    the count you just confirmed)."""
    return _move(name, pp.PENDING_IMAGES, pp.PENDING_LABELS,
                 pp.TESTSET_IMAGES, pp.TESTSET_LABELS)


def reject(name: str, pp) -> bool:
    """✗ wrong: pending -> needs_fix (correct later in the point editor)."""
    return _move(name, pp.PENDING_IMAGES, pp.PENDING_LABELS,
                 pp.NEEDS_FIX_IMAGES, pp.NEEDS_FIX_LABELS)


def to_testset(stem: str, pp) -> bool:
    """Move a raw image into the fixed holdout test set (out of training)."""
    return _move(stem, pp.RAW_IMAGES, pp.RAW_LABELS,
                 pp.TESTSET_IMAGES, pp.TESTSET_LABELS)


def from_testset(stem: str, pp) -> bool:
    """Return a test-set image to raw (back into training)."""
    return _move(stem, pp.TESTSET_IMAGES, pp.TESTSET_LABELS,
                 pp.RAW_IMAGES, pp.RAW_LABELS)


def discard(name: str, pp) -> bool:
    """🗑 dismiss: delete a staged sample entirely (accidental / unwanted)."""
    lbl = pp.PENDING_LABELS / f"{name}.txt"
    imgs = list(pp.PENDING_IMAGES.glob(f"{name}.*"))
    if not imgs and not lbl.exists():
        return False
    for p in imgs:
        p.unlink(missing_ok=True)
    lbl.unlink(missing_ok=True)
    d = _load_truth(pp)
    if d.pop(name, None) is not None:
        sidecar.write_json(_truth_path(pp), d)
    clear_earmark(name, pp)
    return True


def graduate(stem: str, pp) -> bool:
    """Move one corrected sample needs_fix -> raw (the web 'Done → training')."""
    ok = _move(stem, pp.NEEDS_FIX_IMAGES, pp.NEEDS_FIX_LABELS,
               pp.RAW_IMAGES, pp.RAW_LABELS)
    if ok:
        clear_earmark(stem, pp)      # earmark's job is done once it graduates
    return ok


def graduate_to_testset(stem: str, pp) -> bool:
    """Move one corrected sample needs_fix -> testset (holdout, out of training).

    An image the model got WRONG, once its dots are fixed to the true count, is
    a prime hard test case — send it straight to the frozen holdout instead of
    raw."""
    ok = _move(stem, pp.NEEDS_FIX_IMAGES, pp.NEEDS_FIX_LABELS,
               pp.TESTSET_IMAGES, pp.TESTSET_LABELS)
    if ok:
        clear_earmark(stem, pp)
    return ok


def promote_fixed(pp) -> int:
    """Move every corrected needs_fix sample into raw/. Returns how many moved."""
    n = 0
    for img in list(pp.NEEDS_FIX_IMAGES.glob("*.*")):
        if img.suffix.lower() != ".txt" and graduate(img.stem, pp):
            n += 1
    return n


def reset_collected(pp, prefix: str = "tg_") -> dict[str, int]:
    """Wipe everything the front-ends collected — pending/ + needs_fix/
    entirely, and any `prefix`-named samples promoted into raw/ — plus their
    truth counts.

    READ THIS BEFORE CALLING. When a project's images arrive mostly through a
    bot, `prefix`-named samples ARE the dataset, not scratch — this can delete
    nearly every hand-corrected label a project has. There is no undo.
    run_snapshot's blob store can restore only stems captured in some run's
    manifest, one at a time. The CLI therefore refuses without an explicit
    confirmation — see __main__.
    """
    removed = {"pending": 0, "needs_fix": 0, "raw": 0}
    for key, imgd, lbld in (("pending", pp.PENDING_IMAGES, pp.PENDING_LABELS),
                            ("needs_fix", pp.NEEDS_FIX_IMAGES, pp.NEEDS_FIX_LABELS)):
        for p in (list(imgd.glob("*.*")) if imgd.exists() else []):
            p.unlink(missing_ok=True); removed[key] += 1
        for p in (list(lbld.glob("*.txt")) if lbld.exists() else []):
            p.unlink(missing_ok=True)
    for p in list(pp.RAW_IMAGES.glob(f"{prefix}*")):   # only bot samples
        p.unlink(missing_ok=True); removed["raw"] += 1
        (pp.RAW_LABELS / f"{p.stem}.txt").unlink(missing_ok=True)
    # drop truth counts + test earmarks for anything no longer present
    d = {k: v for k, v in _load_truth(pp).items() if not k.startswith(prefix)}
    sidecar.write_json(_truth_path(pp), d)
    sidecar.write_json(_earmark_path(pp),
                       [m for m in _load_earmarks(pp) if not m.startswith(prefix)])
    return removed


def clear_pending(pp) -> int:
    """Discard EVERY staged pending sample (+ its truth/earmark). Pending
    images are counts that were never routed with ✓/✗ — deleting them cannot
    touch training or test data. Returns how many were removed."""
    n = 0
    if pp.PENDING_LABELS.exists():
        for lbl in list(pp.PENDING_LABELS.glob("*.txt")):
            if discard(lbl.stem, pp):
                n += 1
    return n


def counts(pp) -> dict[str, int]:
    """How many samples sit in each collection (for status)."""
    def n(d):
        return len(list(d.glob("*.txt"))) if d.exists() else 0
    return {
        "pending": n(pp.PENDING_LABELS),
        "needs_fix": n(pp.NEEDS_FIX_LABELS),
    }


if __name__ == "__main__":
    import argparse
    from .. import paths
    ap = argparse.ArgumentParser(description="Manage collected feedback samples.")
    ap.add_argument("--reset", action="store_true",
                    help="DESTRUCTIVE: wipe pending/ + needs_fix/ AND every tg_* image in "
                         "raw/ — which can be most of a bot-fed training set, not just "
                         "scratch. Prints what it would delete and asks for confirmation.")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt for --reset (scripts only)")
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    # SAY WHICH DATASET before touching it — doubly so here, where the
    # destructive branch below can delete most of a training set. A stage that
    # runs unannounced is how silent wrong-data happens.
    print(f"[collect] project {proj.slug} ({proj.name}) | {proj.dataset_root} | "
          f"{len(list(pp.RAW_LABELS.glob('*.txt')))} raw images")
    if args.reset:
        # Count FIRST and make the human look at it.
        doomed = sorted(p.stem for p in pp.RAW_LABELS.glob("tg_*.txt"))
        keep = len(list(pp.RAW_LABELS.glob("*.txt"))) - len(doomed)
        dots = 0
        for s in doomed:
            try:
                dots += len([ln for ln in (pp.RAW_LABELS / f"{s}.txt")
                             .read_text().splitlines() if ln.strip()])
            except Exception:  # noqa: BLE001
                pass
        print(f"[collect] --reset would DELETE {len(doomed)} raw images "
              f"({dots} hand-placed dots), leaving {keep}. This cannot be undone.")
        if not args.yes:
            if input("[collect] type DELETE to confirm: ").strip() != "DELETE":
                raise SystemExit("[collect] aborted — nothing removed")
        r = reset_collected(pp)
        print(f"[collect] reset: removed pending={r['pending']} needs_fix={r['needs_fix']} "
              f"raw(tg_*)={r['raw']} | now {counts(pp)}")
    else:                       # default: graduate corrected needs_fix samples into raw
        moved = promote_fixed(pp)
        print(f"[collect] promoted {moved} corrected sample(s) needs_fix -> raw | {counts(pp)}")
