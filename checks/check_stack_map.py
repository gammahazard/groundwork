"""Installing a stack must leave the machine map agreeing with the disk — and
a stale map must never claim an installed venv is missing.

    .venv/bin/python checks/check_stack_map.py

WHY THIS EXISTS. `stacks.install` used to finish by printing "probe this
machine so the new venv's kernels are recorded", and nothing made that happen.
So the map kept a measurement taken BEFORE the venv existed, and the next
launch read it and refused every card with:

    no usable card is free — card 0: .venv-deim13 is not installed on here;
                             card 1: .venv-deim13 is not installed on here

about a venv that was installed, present, and importable. Measured 2026-08-10:
a DEIM launch ran the dataset sync, the split and the COCO convert, then died
at card selection with that sentence, and the operator went looking for an
install that had already succeeded.

TWO INDEPENDENT DEFENCES, because either alone still fails someone:

  1. THE INSTALLER RE-MEASURES (`machine_self.record_local_venvs`, called by
     `stacks.install`). Root cause: the thing that changed the venvs is the
     thing that knows to re-read them.
  2. THE REFUSAL TELLS THE TRUTH (`capacity.can_run`). A venv on this box's
     disk that the map records absent means the map predates the install —
     answered permissively with that reason, which is how every other unknown
     in that module is already answered. This one also covers maps that went
     stale by some route the installer never touched.

Part 1 is tested BEHAVIOURALLY — subprocess is stubbed and the refresh is
observed being called — not by grepping the source for a call, because this
repo has shipped two checks that verified nothing (one asserted a gate was
textually present, so `if False:` left it passing).

READ-ONLY AND GPU-FREE: everything happens under a temp GW_DATA_DIR.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Set BEFORE importing config: DATA_DIR/VENVS_DIR/OUTPUTS_DIR resolve at import
# time, and this check must never write into a real install.
_TMP = tempfile.mkdtemp(prefix="check_stack_map_")
os.environ["GW_DATA_DIR"] = _TMP

from groundwork import stacks  # noqa: E402
from groundwork.config import VENVS_DIR  # noqa: E402
from groundwork.web import machine_self  # noqa: E402
from groundwork.web.machines import capacity  # noqa: E402

FAILS = 0
CARD = {"index": 0, "sm": "sm_86", "vram_gb": 24}


def expect(cond: bool, what: str) -> None:
    global FAILS
    print(("  ok  " if cond else "  FAIL") + f" {what}")
    if not cond:
        FAILS += 1


def _fake_venv(name: str) -> None:
    """A venv-shaped directory: what `_installed_locally` stats for."""
    p = VENVS_DIR / name / "bin"
    p.mkdir(parents=True, exist_ok=True)
    (p / "python").write_text("#!/bin/sh\n")


def main() -> int:
    print("[1] the refusal stops lying when the venv is on disk")
    _fake_venv(".venv-onthedisk")
    absent = {"machines": {"here": {"venvs": {
        ".venv-onthedisk": {"present": False},
        ".venv-really-gone": {"present": False}}}}}
    capacity._map = lambda: absent["machines"]                # noqa: SLF001

    ok, why = capacity.can_run("here", ".venv-onthedisk", CARD)
    expect(ok is True, "a venv present on disk is not refused on a stale map")
    expect("measured before" in why and "not installed" not in why,
           f"...and the note says why, not a falsehood (got: {why!r})")

    ok2, why2 = capacity.can_run("here", ".venv-really-gone", CARD)
    expect(ok2 is False and "is not installed" in why2,
           "a venv that is genuinely absent is still refused, plainly")

    print("[1b] a half-built stack is refused, not permitted as 'unknown'")
    half = {"machines": {"here": {"venvs": {
        ".venv-halfbuilt": {"present": True, "error": "torch import failed"},
        ".venv-oddbuild": {"present": True, "torch": "9.9"}}}}}
    capacity._map = lambda: half["machines"]                  # noqa: SLF001
    ok3, why3 = capacity.can_run("here", ".venv-halfbuilt", CARD)
    expect(ok3 is False and "cannot import torch" in why3,
           "a venv whose torch import fails cannot train, so it is refused")
    ok4, _ = capacity.can_run("here", ".venv-oddbuild", CARD)
    expect(ok4 is True,
           "...while a genuinely unreadable arch list stays permissive")
    capacity._map = lambda: absent["machines"]                # noqa: SLF001

    print("[2] a REMOTE key never gets this box's filesystem as evidence")
    expect(capacity._installed_locally("a-worker", ".venv-onthedisk") is False,
           "a remote machine's venv is not answered from local paths")

    print("[3] the installer re-measures — observed, not grepped")
    called: list[str] = []
    real_run, real_rec = subprocess.run, machine_self.record_local_venvs
    stacks.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(
        a[0] if a else [], 0, "", "")
    machine_self.record_local_venvs = lambda: (called.append("yes") or
                                               {".venv-probe-me": {"present": True,
                                                                   "torch": "9.9",
                                                                   "archs": ["sm_86"]}})
    stacks.MANIFESTS["_check"] = stacks.Stack(
        key="_check", venv=".venv-probe-me", license="Apache-2.0",
        torch_spec=("torch",), pip_specs=())
    try:
        log = stacks.install("_check", log_cb=lambda _m: None)
        expect(called == ["yes"], "install() re-measures the venv map itself")
        expect("recorded .venv-probe-me" in log,
               "...and the log says what it recorded, not 'go press Probe'")
    finally:
        stacks.subprocess.run = real_run
        machine_self.record_local_venvs = real_rec
        stacks.MANIFESTS.pop("_check", None)

    print("[4] a refresh that fails does not fail the install")
    machine_self.record_local_venvs = lambda: (_ for _ in ()).throw(
        OSError("map is read-only"))
    stacks.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(
        a[0] if a else [], 0, "", "")
    stacks.MANIFESTS["_check2"] = stacks.Stack(
        key="_check2", venv=".venv-probe-me2", license="Apache-2.0",
        torch_spec=("torch",), pip_specs=())
    try:
        log2 = stacks.install("_check2", log_cb=lambda _m: None)
        expect("could not refresh" in log2 and "press Probe" in log2,
               "the install survives and says what to do by hand instead")
    except Exception as e:  # noqa: BLE001
        expect(False, f"a failed refresh took the install down: {e}")
    finally:
        stacks.subprocess.run = real_run
        machine_self.record_local_venvs = real_rec
        stacks.MANIFESTS.pop("_check2", None)

    print("PASS" if not FAILS else f"{FAILS} FAILURE(S)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
