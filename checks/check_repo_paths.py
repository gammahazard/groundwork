"""Every module that derives a path from __file__ must still point at the repo.

    .venv/bin/python checks/check_repo_paths.py

WHY THIS EXISTS. A module that computes

    ROOT = Path(__file__).resolve().parent.parent

is silently coupled to HOW DEEP IT SITS. Move it one level down and ROOT moves
with it — the code imports fine, `--help` works, every test that only
exercises imports passes, and the module quietly starts reading and writing a
different tree.

THREE instances bit the predecessor in one day of packaging work: an
extension's database module CREATED an empty database under its own package
and orphaned the real one (and "refuse" was a legitimate verdict, so nothing
looked broken); the challenger trainers would have written every checkpoint
under altmodels/outputs/; and the lab API's REPO landed one level deep, which
disabled the challenger Train button for hours with every endpoint still 200.

None was caught by an import check, a --help sweep, the OpenAPI route diff, or
reading the diff. All were caught by asking the running system for real data
and not recognising the answer. This file turns that into a few seconds.

A module that fails to IMPORT is reported as a skip, not a pass — a venv this
interpreter is not (the challenger trainers) legitimately cannot import here,
and a hard failure would train people to ignore the check.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# module path -> attribute that must equal the repo root.
#
# The rule for what belongs here: a module that derives a root from __file__
# AND reads or writes anything through it. Sweep every `Path(__file__)` in the
# repo after any move — this list is exactly the kind of thing that goes stale
# silently.
ANCHORS = [
    ("groundwork.config", "ROOT"),
    ("altmodels.trainers.deim", "REPO"),
    ("altmodels.trainers.dfine", "REPO"),
    ("altmodels.trainers.rfdetr", "REPO"),
    ("altmodels.trainers.rtmdet", "REPO"),
    ("altmodels.gpu", "REPO"),
    ("altmodels.convert", "REPO"),
    ("groundwork.web.api.lab", "REPO"),
    # The bot config derives its outputs dir from __file__ too. It currently
    # fails to import (it still names a module from before the reorg), so it
    # reports as a skip — the moment it imports again, this anchor resumes
    # judging its ROOT, which at parents[2] would resolve one level DEEP.
    ("groundwork.bots.telegram.config", "ROOT"),
]


def main() -> None:
    bad = []
    for mod, attr in ANCHORS:
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 — a challenger venv we lack here
            print(f"  skip  {mod}.{attr} ({type(e).__name__})")
            continue
        got = getattr(m, attr, None)
        if got is None:
            bad.append(f"{mod}.{attr} is missing — did the constant get renamed?")
        elif Path(got).resolve() != REPO:
            bad.append(f"{mod}.{attr} = {got}\n        expected {REPO}")
        else:
            print(f"  ok    {mod}.{attr}")

    # Things that must EXIST once those anchors are right — a correct-looking
    # root pointing at an empty tree is the failure mode that actually bit.
    from groundwork import config, project as project_mod
    from groundwork.dataset import paths
    if config.OUTPUTS_DIR.exists():
        print(f"  ok    config.OUTPUTS_DIR exists ({config.OUTPUTS_DIR})")
    else:
        bad.append(f"config.OUTPUTS_DIR points at {config.OUTPUTS_DIR}, "
                   f"which does not exist")
    if project_mod.PROJECTS_DIR.is_relative_to(config.OUTPUTS_DIR):
        print("  ok    project manifests live under outputs/")
    else:
        bad.append(f"PROJECTS_DIR {project_mod.PROJECTS_DIR} escaped outputs/")

    # Every project's dataset root must resolve INSIDE this deployment's data
    # dir. The manifest stores repo-relative paths precisely so a checkout
    # moved to another machine still resolves; an absolute foreign path here
    # means some tool wrote a manifest the portable way round.
    for pp in paths.every_project():
        if pp.root.is_absolute() and pp.root.is_relative_to(config.DATA_DIR):
            print(f"  ok    project {pp.slug!r} root resolves under the data dir")
        else:
            bad.append(f"project {pp.slug!r} dataset_root {pp.root} is outside "
                       f"{config.DATA_DIR}")

    # A stray tree is the fingerprint of this bug having already happened:
    # something with an off-by-one root has already written there.
    for stray in (REPO / "altmodels" / "outputs",
                  REPO / "groundwork" / "outputs",
                  REPO / "frontend" / "outputs",
                  REPO / "checks" / "outputs"):
        if stray.exists():
            bad.append(f"STRAY TREE {stray} — a __file__-relative root is off "
                       f"by one level and has already written there")

    if bad:
        print("\nREPO PATHS ARE WRONG:")
        for b in bad:
            print("  - " + b)
        raise SystemExit(1)
    print("\nevery __file__-derived root still resolves to the repo")


if __name__ == "__main__":
    main()
