"""A PROJECT: what is being detected, where its data lives, how it is judged.

Everything the pipeline does — split, train, evaluate, serve — is really being
done *for a project*. There is no default project and no implicit one: a fresh
instance starts blank, and the first project exists when someone creates it in
the cockpit. Every entry point that touches a dataset therefore names its
project explicitly; "which project" can never be ambient state.

WHAT A PROJECT OWNS, and why each field is here rather than hardcoded:

    classes           the label vocabulary. One entry for a single-class
                      counter; a project that detects two things needs two
                      entries, and that is nearly the whole of "multi-class
                      support" (split writes them into data.yaml).
    labelling         "dots" (a centre per object, extent synthesised from
                      spacing) or "boxes" (the extent is measured).

                      INERT: stored, shown on the card, read by NOTHING. The
                      editor places a centre and derives an extent for both
                      values, and ⬚ Boxes sizes one afterwards either way.
                      Recorded here rather than quietly left as a field that
                      looks like it works. Making it real means a
                      draw-the-box-first gesture in the editor.
    headline_metric   what ranks a run. Counting projects rank on count-MAE;
                      others may want mAP. Cross-project comparison is
                      meaningless anyway, so this is per project by nature.
    buckets           image TAGS — the chips on the Training/Test tabs. NOT the
                      model's classes: a tag is a free-text property of an
                      IMAGE ("cluttered", "low-light"), assigned by a human,
                      that the model never sees. It is an inventory and eval
                      lens — count_eval groups holdout MAE by it so one image
                      type regressing cannot hide inside a blended average.
                      ORDER IS MEANINGFUL and preserved: chips read
                      simple -> hard as a difficulty ramp, not alphabetically.
                      Empty for a new project until the owner adds some.
    extensions        opt-in domain tools, scoped per project. Ships with zero
                      built-ins; the mechanism exists so a deployment-specific
                      tool can be enabled for exactly the projects that carry
                      it, and an extension can bring UI, logic AND a collection
                      with it.
    owner             the account that created it. Unowned projects are open to
                      every signed-in account (which is what keeps an instance
                      working the day auth is switched on).
    bots              the Telegram bots that feed THIS project. Each entry
                      names a systemd unit, an env var holding its token, and a
                      log file — never a token. In the manifest rather than in
                      code so a new project can have a bot without editing
                      Python.
"""
from __future__ import annotations

import json  # noqa: F401 — manifest debugging helpers use it
import time  # noqa: F401 — creation stamps travel through the API layer
from dataclasses import dataclass, field, replace  # noqa: F401
from pathlib import Path

from .config import OUTPUTS_DIR, ROOT
from .dataset import sidecar

PROJECTS_DIR = OUTPUTS_DIR / "projects"


@dataclass(frozen=True)
class Project:
    slug: str
    name: str
    dataset_root: Path
    classes: tuple[str, ...] = ("object",)
    buckets: tuple[str, ...] = ()
    owner: str | None = None
    labelling: str = "dots"
    headline_metric: str = "count_mae"
    extensions: tuple[str, ...] = ()
    # Tuple of dicts, frozen alongside the rest. Empty for a new project until
    # one is registered — which is the honest state, not a missing feature.
    bots: tuple[dict, ...] = ()
    created: float | None = None

    @property
    def nc(self) -> int:
        return len(self.classes)

    def has(self, extension: str) -> bool:
        """Is this domain tool enabled here?"""
        return extension in self.extensions

    def to_dict(self) -> dict:
        d = {"slug": self.slug, "name": self.name, "owner": self.owner,
             "classes": list(self.classes), "buckets": list(self.buckets),
             "labelling": self.labelling,
             "headline_metric": self.headline_metric,
             "extensions": list(self.extensions),
             "bots": [dict(b) for b in self.bots], "created": self.created}
        # Keep the repo-relative form on disk so a checkout moved to another
        # machine still resolves — absolute paths in a manifest are a
        # portability trap.
        try:
            d["dataset_root"] = str(self.dataset_root.relative_to(ROOT))
        except ValueError:
            d["dataset_root"] = str(self.dataset_root)
        return d


def _from_dict(d: dict) -> Project:
    root = Path(d["dataset_root"])
    return Project(
        slug=d["slug"], name=d.get("name") or d["slug"],
        dataset_root=root if root.is_absolute() else ROOT / root,
        classes=tuple(d.get("classes") or ("object",)),
        buckets=tuple(d.get("buckets") or ()),
        owner=d.get("owner"), labelling=d.get("labelling", "dots"),
        headline_metric=d.get("headline_metric", "count_mae"),
        extensions=tuple(d.get("extensions") or ()),
        bots=tuple(d.get("bots") or ()),
        created=d.get("created"),
    )


def manifest_path(slug: str) -> Path:
    return PROJECTS_DIR / slug / "project.json"


def load(slug: str) -> Project:
    """The project by slug. There is no default and no auto-create: an unknown
    slug is a KeyError, because acting on a project nobody named is the
    silent-wrong-data failure this repo treats as its most expensive."""
    if not (slug or "").strip():
        raise KeyError("no project named — pass a project slug")
    f = manifest_path(slug)
    if not f.exists():
        raise KeyError(f"no such project {slug!r} — manifest expected at {f}")
    return _from_dict(sidecar.read_json(f, strict=True))


def save(p: Project) -> None:
    """Write the manifest, PRESERVING keys this dataclass does not know about.

    It used to write `to_dict()` flat, which silently dropped anything else in
    the file. Foreign keys can be load-bearing (a retired-extensions marker, a
    deployment's own annotations), so the merge keeps them. Merge order
    matters: the dataclass's own fields WIN, so this preserves foreign keys
    without ever resurrecting a stale value of a real one.
    """
    f = manifest_path(p.slug)
    prior: dict = {}
    if f.exists():
        try:
            prior = sidecar.read_json(f, strict=True)
        except Exception:  # noqa: BLE001 — an unreadable manifest is replaced,
            prior = {}     # which is what the caller asked for by saving.
    sidecar.write_json(f, {**prior, **p.to_dict()}, indent=1)


def slugs() -> list[str]:
    """Every project with a manifest. Empty on a fresh instance — honestly."""
    if not PROJECTS_DIR.exists():
        return []
    return sorted(d.name for d in PROJECTS_DIR.glob("*")
                  if (d / "project.json").exists())


def add_argument(ap) -> None:
    """Give a stage CLI its --project flag, worded the same way everywhere.

    Stages run as subprocesses (web/retrain_job.py::_STAGE), so the project
    cannot be ambient state — it has to travel on the command line, or a run
    would silently use whatever the parent happened to have loaded. REQUIRED:
    there is no default project to fall back to.
    """
    ap.add_argument("--project", required=True, help="project slug")
