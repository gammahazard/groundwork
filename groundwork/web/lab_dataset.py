"""Where a challenger's converted COCO tree lives — one per run, not one shared.

WHY THIS EXISTS (2026-08-02, owner: "I can't leave a full card unused").
Every challenger converted into a single directory, `outputs/alt/datasets/
coco_rfdetr`, and `altmodels.convert` rmtree's it before rebuilding. So launching
a second challenger destroyed the dataset the first one was reading, and
`lab_ops.train` had to refuse outright — which meant the worker's two GPUs could
never both run a challenger, whatever the cards allowed.

That refusal was CORRECT, not paranoid. Measured the day this was written: the
tree built at 12:45 for yolox-s already had different train/valid membership from
the split on disk, because the yolo split-seed runs had re-split underneath it. A
convert really would have pulled a live run's data out from under it.

A PER-RUN TREE REMOVES THE COLLISION rather than guarding against it. It also
fixes something quieter: a challenger's train/val membership used to depend on
whenever the shared tree was last built — so two runs of the same family could
silently train on different data, and nothing recorded which. Now the tree is
built at the run's own launch and nothing else can touch it.

THE EXAM IS UNAFFECTED EITHER WAY. count_eval scores against
`outputs/dataset/testset/`, which never enters a convert. This decides only what
a run learns from and early-stops against; it cannot move a holdout number.

COST: `--copy` (not symlinks, so curating an image mid-run cannot break a loader)
makes each tree roughly the size of the dataset — ~90MB for the first project's 174
images. `reapable()` names the ones whose run has finished so the janitor can
reclaim them; nothing here deletes anything by itself.
"""
from __future__ import annotations

from pathlib import Path

# The pre-2026-08-02 shared tree. Still the default when no run is named, so a
# bare `python -m altmodels.convert` and every existing run directory keep
# working — this adds a location, it does not move one.
LEGACY_NAME = "coco_rfdetr"


def datasets_dir(pp) -> Path:
    return pp.ALT_DIR / "datasets"


def tree_for(pp, run_name: str | None) -> Path:
    """This run's own COCO tree, or the legacy shared one when unnamed."""
    if not run_name:
        return datasets_dir(pp) / LEGACY_NAME
    # Named after the run so `ls datasets/` says which run owns what, and so a
    # stray tree can be traced back to the run that made it.
    return datasets_dir(pp) / f"coco_{run_name}"


def owner_of(tree: Path) -> str | None:
    """Which run a tree belongs to, or None for the legacy shared one."""
    n = tree.name
    return n[5:] if n.startswith("coco_") and n != LEGACY_NAME else None


def reapable(pp) -> list[Path]:
    """Per-run trees whose run has FINISHED — safe to delete, oldest first.

    Finished means the run wrote a meta.json and no longer holds a GPU crumb.
    Both conditions, deliberately: a crumb alone is not enough (a run that died
    leaves one behind), and meta.json alone is not either — scoring runs after
    training and still reads nothing from here, but a re-score might.

    The legacy shared tree is NEVER returned. Runs that predate this module have
    no tree of their own and would be reading it.
    """
    d = datasets_dir(pp)
    if not d.is_dir():
        return []
    out = []
    for tree in d.iterdir():
        run = owner_of(tree) if tree.is_dir() else None
        if not run:
            continue
        rd = pp.ALT_DIR / run
        if (rd / "meta.json").exists() and not (rd / "gpu_holder.json").exists():
            out.append(tree)
    return sorted(out, key=lambda p: p.stat().st_mtime)
