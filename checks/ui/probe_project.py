#!/usr/bin/env python3
"""Fixture for check_multiproject.mjs: a throwaway project, created and destroyed.

    python checks/ui/probe_project.py create     # prints the slug
    python checks/ui/probe_project.py destroy

REFUSES to name anything but its own reserved slug, so a bug here cannot reach
a real project's manifest or dataset. Teardown removes the manifest directory
and the dataset directory it created, nothing else.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from groundwork import project as project_mod          # noqa: E402
from groundwork.config import OUTPUTS_DIR              # noqa: E402

SLUG = "zz-ui-probe"          # reserved; this file touches nothing else


def create() -> None:
    p = project_mod.Project(
        slug=SLUG, name="UI Probe (throwaway)",
        dataset_root=OUTPUTS_DIR / "projects" / SLUG / "dataset")
    project_mod.manifest_path(SLUG).parent.mkdir(parents=True, exist_ok=True)
    project_mod.save(p)
    print(SLUG)


def destroy() -> None:
    assert SLUG == "zz-ui-probe"
    shutil.rmtree(project_mod.manifest_path(SLUG).parent, ignore_errors=True)
    shutil.rmtree(OUTPUTS_DIR / "projects" / SLUG, ignore_errors=True)
    print("removed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "create":
        create()
    elif cmd == "destroy":
        destroy()
    else:
        raise SystemExit(__doc__)
