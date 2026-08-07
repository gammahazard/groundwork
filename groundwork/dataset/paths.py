"""Dataset directory layout — one place so every stage agrees on where files go.

    dataset/
      raw/          <- auto-labeler output (pre-split, human-editable)
        images/     <- copy of each source image
        labels/     <- YOLO .txt (one per image): "cls cx cy w h" normalized
        preview/    <- annotated .png for the manual review pass
      images/{train,val}/   <- split for training (populated by split.py)
      labels/{train,val}/
      data.yaml     <- ultralytics dataset descriptor
      runs/         <- training outputs (weights, curves)

PER-PROJECT, with no default. `for_project(p)` returns this layout rooted at
that project's `dataset_root`; there is deliberately NO `default()` and no
module-level constants, because both are ways to silently bind code to the
wrong project's tree — and silent wrong-data is this repo's most expensive
failure mode. A call site that does not know its project takes a
`ProjectPaths` from a caller that does; a machine-level question uses
`every_project()`.

Stages get their paths from `--project` (groundwork.project.add_argument),
because they run as subprocesses and cannot inherit ambient state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """One project's dataset layout."""
    root: Path
    CLASS_NAMES: tuple[str, ...] = ("object",)
    # Carried so the layout can answer "does THIS project have a given
    # extension collection?" without every caller also holding a Project. An
    # extension can bring a collection with it, and the collection list is a
    # property of the layout, not of the UI.
    extensions: tuple[str, ...] = ()
    # The project's slug. Carried because several things OUTSIDE the dataset
    # tree still need to be keyed by project — the thumbnail cache under
    # outputs/, a ledger row — and deriving a slug back out of `root` is
    # guesswork the moment dataset_root is customised.
    slug: str = ""

    # --- the editable source of truth ---
    @property
    def DATASET_DIR(self) -> Path: return self.root
    @property
    def RAW_DIR(self) -> Path: return self.root / "raw"
    @property
    def RAW_IMAGES(self) -> Path: return self.RAW_DIR / "images"
    @property
    def RAW_LABELS(self) -> Path: return self.RAW_DIR / "labels"
    @property
    def RAW_PREVIEW(self) -> Path: return self.RAW_DIR / "preview"

    # --- derived split, wiped and rebuilt by every split run ---
    @property
    def IMAGES_DIR(self) -> Path: return self.root / "images"
    @property
    def LABELS_DIR(self) -> Path: return self.root / "labels"
    @property
    def DATA_YAML(self) -> Path: return self.root / "data.yaml"
    @property
    def RUNS_DIR(self) -> Path: return self.root / "runs"

    # Challenger runs live beside the dataset, not inside it: `alt/` is a
    # sibling of `dataset/` under the project root, so a split that wipes the
    # derived dirs can never touch a run.
    @property
    def ALT_DIR(self) -> Path: return self.root.parent / "alt"

    # --- feedback collections; each is EXCLUSIVE, an image moves between them ---
    @property
    def PENDING_IMAGES(self) -> Path: return self.root / "pending" / "images"
    @property
    def PENDING_LABELS(self) -> Path: return self.root / "pending" / "labels"
    @property
    def NEEDS_FIX_IMAGES(self) -> Path: return self.root / "needs_fix" / "images"
    @property
    def NEEDS_FIX_LABELS(self) -> Path: return self.root / "needs_fix" / "labels"

    # --- the FROZEN holdout ---
    @property
    def TESTSET_IMAGES(self) -> Path: return self.root / "testset" / "images"
    @property
    def TESTSET_LABELS(self) -> Path: return self.root / "testset" / "labels"


def for_project(p) -> ProjectPaths:
    """This layout, rooted at `p`'s dataset_root. Accepts a Project or a slug."""
    if isinstance(p, str):
        from ..project import load
        p = load(p)
    return ProjectPaths(root=p.dataset_root, CLASS_NAMES=tuple(p.classes),
                        extensions=tuple(p.extensions), slug=p.slug)


def every_project() -> list[ProjectPaths]:
    """One layout per project with a manifest — for the MACHINE questions.

    Some facts are not about a project at all: which GPU is busy right now, how
    many hours this box has burned. Answering those from one project alone is
    silently wrong the moment a second project exists — the card is held, the
    hours were burned, and neither shows up. Answering them per project is
    worse: two projects would each conclude the GPU is free.

    A broken manifest is skipped rather than raised. These callers are asking a
    resource question, and a project whose JSON does not parse still cannot be
    the reason a card looks free when it is not — but it must not take the
    dashboard down either.
    """
    from ..project import slugs
    out = []
    for slug in slugs():
        try:
            out.append(for_project(slug))
        except Exception:  # noqa: BLE001 — an unreadable manifest is not a fact
            pass
    return out
