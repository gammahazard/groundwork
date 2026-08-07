"""Edit a project's class list from the UI, instead of by hand in project.json.

WHY THIS IS NOT JUST A TEXT FIELD. A class is not a name, it is an INDEX. Every
label file on disk is lines of `<cls> <cx> <cy> <w> <h>`, and that leading
integer is a position in this list. So the list is a schema for data already
written, and editing it rewrites the meaning of every dot ever placed:

    append          SAFE      indices 0..n-1 keep pointing where they did
    rename in place SAFE      the index is unchanged; only its name moves
    reorder         REFUSED   every existing label silently means something else
    remove          GUARDED   allowed only if no label uses the dropped index

Reorder is the one that has to be refused rather than warned about, because it
is invisible afterwards: nothing errors, the split succeeds, training succeeds,
and the model learns bolts as nuts. That is this repo's most expensive failure
shape — silent wrong data — and here it would be self-inflicted by a dropdown.

It is DETECTABLE, which is what makes refusing it possible: a reorder is exactly
the case where the set of names is unchanged but their order is not.

REMOVAL IS CHECKED AGAINST THE LABELS THEMSELVES, not against a count. The only
honest answer to "is class 2 still in use" comes from reading what is on disk —
raw/ and testset/ both, because the holdout is frozen and a class it still uses
can never be re-labelled away.
"""
from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ... import project
from ..auth import audit as audit_mod
from .deps import current_user, readable_by
from ...dataset import paths

router = APIRouter()


class ClassEdit(BaseModel):
    classes: list[str]
    # Removing a class that no label uses is still a schema change; requiring it
    # to be asked for means a typo that shortens the list cannot quietly do it.
    allow_removal: bool = False


def _used_indices(pp) -> dict[int, int]:
    """{class index: how many label lines use it} across raw AND testset.

    BOTH TREES, because the testset is frozen: a class still used there can
    never be re-labelled away, so it can never be removed either. Reading raw/
    alone would report a class as unused and let it be deleted out from under
    the holdout.
    """
    used: dict[int, int] = {}
    for root in (getattr(pp, "RAW_LABELS", None),
                 getattr(pp, "TESTSET_LABELS", None)):
        if not root or not root.exists():
            continue
        for f in root.glob("*.txt"):
            if f.stem == "classes":
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except OSError:
                continue
            for ln in text.splitlines():
                head = ln.split()[:1]
                if head and head[0].isdigit():
                    i = int(head[0])
                    used[i] = used.get(i, 0) + 1
    return used


@router.get("/api/projects/{slug}/classes")
def get_classes(slug: str):
    """The class list, plus how many labels actually use each one.

    The counts are the point: they tell you which classes are safe to remove and
    which are load-bearing, without anyone having to guess.
    """
    try:
        p = project.load(slug)
    except KeyError:
        raise HTTPException(404, f"no such project {slug!r}")
    used = _used_indices(paths.for_project(p))
    return {"slug": p.slug, "nc": p.nc,
            "classes": [{"index": i, "name": n, "labels": used.get(i, 0)}
                        for i, n in enumerate(p.classes)],
            # Anything beyond the declared list is already a broken split; say so
            # here rather than letting build_split be the first to mention it.
            "orphan_indices": sorted(i for i in used if i >= p.nc)}


@router.patch("/api/projects/{slug}/classes")
def set_classes(slug: str, req: ClassEdit, request: Request,
                user: str | None = Depends(current_user)):
    """Replace the class list. Append and rename freely; reorder never; remove
    only when nothing is labelled with it.

    OWNERSHIP IS CHECKED HERE rather than through `current_project`, because the
    slug arrives in the PATH — this route predates `?project=` and changing its
    shape would break the UI that calls it. The rule is the same one
    `deps.readable_by` applies everywhere else, and it was MISSING: this endpoint
    took no dependency at all, so any signed-in account could rewrite any
    project's class list. Given what this file's own docstring says a class list
    IS — a schema for every label already on disk — that is the silent-wrong-data
    failure it exists to prevent, reachable from another account.
    """
    try:
        p = project.load(slug)
    except KeyError:
        raise HTTPException(404, f"no such project {slug!r}")
    if not readable_by(p, user):
        audit_mod.record(audit_mod.DENIED, actor=user, target=slug, ok=False,
                         via=request.scope.get("auth_via"),
                         ip=getattr(getattr(request, "client", None), "host", None),
                         detail="class edit")
        raise HTTPException(403, f"project {slug!r} belongs to {p.owner!r}")

    new = [c.strip() for c in req.classes if c and c.strip()]
    if not new:
        raise HTTPException(400, "a project needs at least one class")
    if len(new) != len(set(new)):
        dupes = sorted({c for c in new if new.count(c) > 1})
        raise HTTPException(400, f"duplicate class names: {', '.join(dupes)}")
    old = list(p.classes)
    if new == old:
        return {**get_classes(slug), "changed": False}

    # REORDER: same names, different order. Refused — see the module docstring.
    if sorted(new) == sorted(old) and len(new) == len(old):
        moved = [f"{n!r} {old.index(n)}->{new.index(n)}"
                 for n in old if old.index(n) != new.index(n)]
        raise HTTPException(
            409, "reordering classes would silently re-point every label "
                 f"already drawn ({', '.join(moved[:4])}). Rename in place or "
                 "append instead; to genuinely reorder, the label files have to "
                 "be rewritten first.")

    used = _used_indices(paths.for_project(p))
    if len(new) < len(old):
        if not req.allow_removal:
            raise HTTPException(
                409, f"this drops {len(old) - len(new)} class(es) "
                     f"({', '.join(old[len(new):])}). Re-send with "
                     "allow_removal to confirm.")
        blocked = {i: used[i] for i in used if i >= len(new) and used[i]}
        if blocked:
            detail = ", ".join(f"{old[i]} (index {i}, {n} labels)"
                               for i, n in sorted(blocked.items())
                               if i < len(old))
            raise HTTPException(
                409, f"cannot remove a class that is still labelled: {detail}. "
                     "Re-label or delete those first — including in the frozen "
                     "testset, which can never be re-labelled.")

    renamed = [f"{o!r}->{n!r}" for o, n in zip(old, new) if o != n]
    added = new[len(old):]
    project.save(replace(p, classes=tuple(new)))
    print(f"[projects] {slug}: classes {old} -> {new}"
          + (f" | renamed {', '.join(renamed)}" if renamed else "")
          + (f" | added {', '.join(added)}" if added else ""), flush=True)
    return {**get_classes(slug), "changed": True,
            "renamed": renamed, "added": added,
            # The split writes data.yaml from this list, so nothing takes effect
            # in training until it runs — say so rather than implying the next
            # run is already different.
            "note": "data.yaml is rebuilt by the next split; existing runs and "
                    "their recorded class names are untouched."}
