"""Holdout eval-bucket tags: label each test image (object / capsule / hard) so
count_eval can report per-bucket MAE instead of one blended number. This is an
EVALUATION lens only — the model stays single-class 'object'; tags never touch
training, weights, NMS, or the shipped counter.
"""
from __future__ import annotations

import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...dataset.store import testset_buckets
from ...dataset.paths import ProjectPaths
from .. import lab_guard
from .deps import project_paths

router = APIRouter()

# Letters, numbers, space, hyphen, underscore. No quotes, no angle brackets, no
# ampersands — nothing that can terminate an attribute or open a tag.
_TAG_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,31}$")


class Bucket(BaseModel):
    label: str | None = None      # None / "" / "untagged" -> clear the tag


@router.get("/api/buckets")
def buckets(pp: ProjectPaths = Depends(project_paths)):
    """{stem: label} for every tagged holdout image (untagged ones omitted)."""
    return testset_buckets.load(pp)


@router.post("/api/buckets/{stem}")
def set_bucket(stem: str, body: Bucket,
               pp: ProjectPaths = Depends(project_paths)):
    lab_guard.no_lab_edits()
    lbl = (body.label or "").strip().lower()
    if not lbl or lbl == testset_buckets.UNTAGGED:
        testset_buckets.remove(stem, pp)
        return {"ok": True, "label": None}
    testset_buckets.set_bucket(stem, lbl, pp)
    return {"ok": True, "label": lbl}


# ---------------------------------------------------------------------------
# THE VOCABULARY ITSELF. Until now this was a hardcoded array in buckets.js, so
# adding a tag — "scuffed", "low-light", "non-image" — meant editing JavaScript
# and redeploying. Nothing server-side ever required that: bucket_mae groups by
# whatever tags exist on disk, so the list was only ever a UI affordance and
# belongs to the project.
#
# A TAG IS NOT A CLASS, and the safety rules are the opposite way round:
#
#   a CLASS is an INDEX baked into every label file, so reordering silently
#   re-points every dot ever drawn and is refused outright (project_classes.py)
#
#   a TAG is a free-text STRING in {stem: tag}, so reordering is harmless — it
#   only sets the order the chips read in, which is deliberately a
#   simple -> hard difficulty ramp rather than alphabetical.
#
# What a tag edit CAN do is orphan images: rename "stacked" and every image still
# carrying the old string falls out of the vocabulary. So a rename MIGRATES the
# stored tags, and a removal reports how many images it would strand.
# ---------------------------------------------------------------------------

class TagVocab(BaseModel):
    buckets: list[str]


def _usage(pp) -> dict[str, int]:
    """{tag: images carrying it}, straight from the store — the only honest
    source for "is this tag still in use"."""
    used: dict[str, int] = {}
    for tag in testset_buckets.load(pp).values():
        used[tag] = used.get(tag, 0) + 1
    return used


@router.get("/api/bucket-vocab")
def get_vocab(pp: ProjectPaths = Depends(project_paths)):
    """The project's tag vocabulary, plus how many images carry each one, plus
    any tag in use that the vocabulary no longer lists."""
    from ... import project as _project
    p = _project.load(pp.slug)
    used = _usage(pp)
    return {"buckets": list(p.buckets),
            "usage": {b: used.get(b, 0) for b in p.buckets},
            # Tags on images that the vocabulary dropped. Reported rather than
            # hidden: they still group in count_eval, so a silent orphan is a
            # bucket appearing in a run report that the UI cannot offer.
            "orphans": {t: n for t, n in used.items() if t not in p.buckets}}


@router.patch("/api/bucket-vocab")
def set_vocab(body: TagVocab, pp: ProjectPaths = Depends(project_paths)):
    """Replace the vocabulary. Renames migrate the images that carry them.

    A RENAME IS DETECTED POSITIONALLY — same length, one name changed at index
    i — and the stored tags are rewritten with it. Without that, renaming
    "stacked" to "stacked-tablet" would strand every image still saying
    "stacked": they would vanish from the chips while still grouping under the
    old name in every future run report.
    """
    # The lab is a CLONE (mirror --delete), so a tag edited there is gone within
    # five minutes and a rename there would migrate images that HQ will overwrite.
    # Same rule as every other dataset-mutating endpoint.
    lab_guard.no_lab_edits()

    from dataclasses import replace as _replace
    from ... import project as _project

    p = _project.load(pp.slug)
    new = [b.strip() for b in body.buckets if b and b.strip()]
    # A CLOSED CHARSET, and this is a security boundary rather than tidiness.
    # A tag is interpolated straight into markup — `data-b="${b}"`, a title, an
    # <option> — so a name containing a quote breaks out of the attribute.
    # Measured 2026-08-05: PATCHing a tag called
    #     x" onmouseover="alert(1)
    # put a LIVE event-handler attribute on 176 <option> elements, one per
    # image, which every other signed-in user would then run by hovering. The
    # client escapes them now too, but the store is the right place to refuse:
    # a tag that cannot contain a quote cannot break any renderer, including the
    # next one somebody writes.
    bad = [b for b in new if not _TAG_OK.match(b)]
    if bad:
        raise HTTPException(
            422, f"tag names may use letters, numbers, spaces, - and _ only "
                 f"(max 32) — rejected: {', '.join(repr(b)[:40] for b in bad[:3])}")
    if not new:
        return {"ok": False, "error": "a project needs at least one tag"}
    if len(new) != len(set(new)):
        dupes = sorted({b for b in new if new.count(b) > 1})
        return {"ok": False, "error": f"duplicate tags: {', '.join(dupes)}"}
    if testset_buckets.UNTAGGED in new:
        return {"ok": False,
                "error": f"{testset_buckets.UNTAGGED!r} is what an untagged "
                         "image already reads as — it cannot also be a tag"}

    old = list(p.buckets)
    renames = {o: n for o, n in zip(old, new) if o != n} if len(old) == len(new) else {}
    migrated = 0
    if renames:
        store = testset_buckets.load(pp)
        for stem, tag in list(store.items()):
            if tag in renames:
                store[stem] = renames[tag]
                migrated += 1
        if migrated:
            testset_buckets.save(store, pp)

    _project.save(_replace(p, buckets=tuple(new)))
    used = _usage(pp)
    stranded = {t: n for t, n in used.items() if t not in new}
    print(f"[buckets] {pp.slug}: tags {old} -> {new}"
          + (f" | migrated {migrated} images" if migrated else ""), flush=True)
    return {"ok": True, "buckets": new, "migrated": migrated,
            # Named, not silently dropped: these images keep their old tag and
            # will still group under it in count_eval.
            "stranded": stranded}
