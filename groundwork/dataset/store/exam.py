"""Freeze the holdout as a NAMED, re-runnable exam.

    python -m groundwork.dataset.store.exam freeze --name 2026-08-03
    python -m groundwork.dataset.store.exam list
    python -m groundwork.dataset.store.exam verify 2026-08-03

WHY THIS HAS TO EXIST BEFORE THE NEXT IMAGE ARRIVES.

Every MAE in this repo is denominated in an exam nobody wrote down. The holdout
can grow release over release, and each time it grows, every
number measured before it silently stopped being comparable to every number
measured after. That is not hypothetical — CLAUDE.md records the cost twice:

  * the 0.0154-vs-0.0 spread that looked like a model difference was partly the
    exam going 65 -> 68 underneath the runs;
  * `deimv2-n-1280-k` reads 0.0154 in one place and 0.0147 in another. Same
    single miss. 1/65 versus 1/68.

Growing the holdout is the RIGHT thing to do — it is saturated and needs harder
images. But the moment it grows, "the exam the champions scored 0.0 on" ceases to
exist as anything you can re-run, and ~80 historical numbers lose their common
yardstick. A manifest costs nothing now and is unrecoverable later: you cannot
reconstruct which images these were once they are mixed with the next batch.

WHAT IS RECORDED, and why each field earns its place:

    stem        which images. The exam's identity.
    gt          its object count at freeze time. An image whose LABEL is corrected
                later is no longer the same question, even though the file name
                and the image are untouched — so the count is pinned too.
    sha         short sha256 of the IMAGE bytes. Catches a re-crop or re-shoot
                that keeps the name. Cheap (68 files) and the only way to tell
                "same image" from "same filename".

A NAME, NOT A DATE-STAMPED GUESS. The file is committed to git deliberately:
outputs/ is gitignored, and a yardstick that lives only on one machine is one
disk failure from being a number nobody can reproduce.

VERIFY IS THE POINT, NOT FREEZE. Freezing is trivial; the value is being able to
ask later "is this still the same exam?" and get a specific answer — these three
images are gone, this one's label changed from 28 to 29 — instead of a silently
different denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .. import paths, sidecar

# Beside the code by default — outputs/ is gitignored and this is the one
# artifact whose whole job is to outlive the machine, so it wants to be
# committable. GW_DATA_DIR overrides it, because a Docker install's code tree
# is a read-only image and freeze() would simply fail there.
_CODE_ROOT = Path(__file__).resolve().parents[3]
EXAM_DIR = Path(os.environ.get("GW_DATA_DIR") or _CODE_ROOT) / "exams"


def _path(slug: str, name: str) -> Path:
    return EXAM_DIR / f"{slug}-{name}.json"


def _sha(p: Path) -> str:
    """Short content hash of an image. Truncated to 16 hex chars — this is a
    change detector, not a security boundary, and a full digest per image makes
    the manifest unreadable for no gain."""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _current(pp) -> dict[str, dict]:
    """The holdout as it stands right now: {stem: {gt, sha}}.

    Mirrors count_eval's own definition of the exam set — the intersection of
    labels and images — so a frozen exam cannot describe a set count_eval would
    never have scored. Deliberately re-derived rather than imported: count_eval
    builds it inside evaluate(), and lifting it out would be a refactor of the
    scoring path to serve a bookkeeping module.
    """
    lbl_dir, img_dir = pp.TESTSET_LABELS, pp.TESTSET_IMAGES
    gt = {}
    for lbl in sorted(lbl_dir.glob("*.txt")):
        lines = [x for x in lbl.read_text(encoding="utf-8").splitlines() if x.strip()]
        gt[lbl.stem] = len(lines)
    imgs = {p.stem: p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".txt"}
    return {s: {"gt": gt[s], "sha": _sha(imgs[s])}
            for s in sorted(set(gt) & set(imgs))}


def freeze(name: str, pp=None, note: str = "") -> Path:
    images = _current(pp)
    if not images:
        raise SystemExit("the holdout is empty — nothing to freeze")
    out = _path(pp.slug, name)
    if out.exists():
        raise SystemExit(f"{out} already exists — an exam is FROZEN, so this "
                         f"refuses rather than overwriting it. Pick another "
                         f"name, or delete it deliberately if it was a mistake.")
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_json(out, {
        "name": name, "project": pp.slug, "n_images": len(images),
        "total_objects": sum(t["gt"] for t in images.values()),
        "note": note, "images": images,
    })
    print(f"[exam] froze {len(images)} images "
          f"({sum(t['gt'] for t in images.values())} objects) -> {out}")
    print(f"[exam] COMMIT THIS FILE. It is the only record of which images "
          f"today's numbers were measured on.")
    return out


def load(name: str, pp=None) -> dict:
    p = _path(pp.slug, name)
    if not p.exists():
        known = ", ".join(sorted(x.stem for x in EXAM_DIR.glob(f"{pp.slug}-*.json"))) \
            or "none"
        raise SystemExit(f"no exam {name!r} for {pp.slug} at {p}. Known: {known}")
    return json.loads(p.read_text())


def verify(name: str, pp=None) -> dict:
    """Is the holdout still this exam? Reports EXACTLY what moved.

    Three distinct kinds of drift, kept apart because they mean different things:
    a missing image shrinks the exam, a changed label makes an image a different
    question, and a changed image makes it a different photo of the same image.
    """
    want = load(name, pp)["images"]
    have = _current(pp)
    missing = sorted(set(want) - set(have))
    added = sorted(set(have) - set(want))
    relabelled = sorted(s for s in set(want) & set(have)
                        if want[s]["gt"] != have[s]["gt"])
    reshot = sorted(s for s in set(want) & set(have)
                    if want[s]["sha"] != have[s]["sha"]
                    and want[s]["gt"] == have[s]["gt"])
    return {"name": name, "n_frozen": len(want), "n_now": len(have),
            "missing": missing, "added": added,
            "relabelled": [{"stem": s, "was": want[s]["gt"], "now": have[s]["gt"]}
                           for s in relabelled],
            "reshot": reshot,
            "intact": not (missing or relabelled or reshot)}


def _cmd_verify(name: str, pp) -> int:
    r = verify(name, pp)
    print(f"[exam] {r['name']}: {r['n_frozen']} images frozen, "
          f"holdout now has {r['n_now']}")
    if r["intact"] and not r["added"]:
        print("[exam] IDENTICAL — this exam is exactly reproducible today")
        return 0
    if r["intact"]:
        print(f"[exam] REPRODUCIBLE — every frozen image is unchanged; the holdout "
              f"has also GROWN by {len(r['added'])} image(s), which do not belong "
              f"to this exam and would be excluded when scoring it")
        return 0
    print("[exam] NOT reproducible as frozen:")
    for s in r["missing"]:
        print(f"    MISSING     {s} — gone from the holdout")
    for x in r["relabelled"]:
        print(f"    RELABELLED  {x['stem']} — was {x['was']} objects, now {x['now']}")
    for s in r["reshot"]:
        print(f"    IMAGE CHANGED {s} — same count, different pixels")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="snapshot the holdout as a named exam")
    f.add_argument("--name", required=True, help="e.g. 2026-08-03")
    f.add_argument("--note", default="", help="what this exam is, in words")
    sub.add_parser("list", help="known exams for this project")
    v = sub.add_parser("verify", help="is the holdout still this exam?")
    v.add_argument("name")
    from ...project import add_argument, load as load_project
    add_argument(ap)
    a = ap.parse_args()
    pp = paths.for_project(load_project(a.project))
    print(f"[exam] project {pp.slug} | {pp.TESTSET_IMAGES}")

    if a.cmd == "freeze":
        freeze(a.name, pp, a.note)
    elif a.cmd == "list":
        found = sorted(EXAM_DIR.glob(f"{pp.slug}-*.json"))
        if not found:
            print("[exam] none frozen yet")
        for p in found:
            d = json.loads(p.read_text())
            print(f"    {d['name']:14} {d['n_images']:>4} images  "
                  f"{d['total_objects']:>6} objects  {d.get('note','')}")
    else:
        raise SystemExit(_cmd_verify(a.name, pp))


if __name__ == "__main__":
    main()
