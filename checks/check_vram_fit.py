"""A run must land on a card it FITS in, not merely one that can launch it.

    .venv/bin/python checks/check_vram_fit.py

WHY THIS EXISTS. On the predecessor's dual-card box, a 17.2 GiB model trained
on the 16 GB card at 2.19 s/it. Seven prior runs of the same model at the same
resolution, on the 24 GB card, held 0.72-0.94. Three times slower on the
FASTER card, for four hours, and nothing anywhere raised an error — because
under WSL2 the driver does not OOM when a model overflows VRAM, it spills into
host RAM over PCIe and keeps going. The only evidence was an s/it figure
nobody was comparing.

The cause was one word in the card chooser: it took the first card that COULD
run the model, and "could" meant compute capability only.

THE TWO QUESTIONS THIS ASSERTS ARE SEPARATE:

    can_run   compute capability — a venv with no kernels for the card dies at
              the first launch, so this GATES.
    fits      memory — the model runs correctly and pays a speed tax, so this
              only STEERS the default and an explicit card still wins.

Merging them would be wrong in both directions: refusing an undersized card
would strand any family bigger than every card in the box, and ignoring size
is the bug above.

THIS CHECK CAN FAIL, which is the part worth caring about. Asserting that
today's registry produces today's answer would pass just as green if `fits`
were deleted and both models still happened to land where they do. So part 3
MUTATES the measured peak and asserts the chosen card MOVES — proving the
choice follows the number rather than coinciding with it.

SANDBOXED AND GPU-FREE: the measured map is a FAKE machines.json in a temp
directory (no real machine names, and the machine's own map is never read),
the models are synthetic registry entries, and the pure chooser is called
directly. It starts nothing.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

from groundwork.models import registry                       # noqa: E402
from groundwork.web import machines as mach                  # noqa: E402
from groundwork.web.machines import capacity as capacity_mod  # noqa: E402
from groundwork.web.api import train_dispatch as td          # noqa: E402

FAIL: list[str] = []
KEY = "workerbox"          # a machine that exists only in the sandbox map


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


def _sandbox_map(tmp: Path) -> None:
    """Point the measured map at a fake: two cards, three venvs."""
    (tmp / "machines.json").write_text(json.dumps({"machines": {KEY: {
        "measured": 0,
        "cards": [
            {"index": 0, "name": "small-fast", "vram_gb": 16, "sm": "sm_120"},
            {"index": 1, "name": "big", "vram_gb": 24, "sm": "sm_86"},
        ],
        "venvs": {
            # Drives BOTH cards — so only `fits` can separate them.
            ".venv-both": {"present": True, "torch": "x",
                           "archs": ["sm_86", "sm_120"]},
            # Compiled only up to sm_90 — card 0 is unreachable for it.
            ".venv-capped": {"present": True, "torch": "x",
                             "archs": ["sm_86"]},
        },
    }}}))
    capacity_mod.MAP_PATH = tmp / "machines.json"


def main() -> None:
    print("1. fits() on synthetic cards — the rule itself, independent of the map")
    big = {"index": 1, "vram_gb": 24}
    small = {"index": 0, "vram_gb": 16}
    note(mach.fits(big, 17.2)[0], "17.2 GiB fits a 24 GB card")
    note(not mach.fits(small, 17.2)[0], "17.2 GiB does NOT fit a 16 GB card")
    # The headroom is the point of the constant: an exact match is not a fit,
    # because the recorded peak is the largest value ever seen and a stochastic
    # trainer's peak moves several GiB across runs of one config.
    note(not mach.fits({"vram_gb": 16}, 16.0)[0],
         f"an exact match is not a fit ({mach.VRAM_HEADROOM_GB} GiB headroom)")
    note(mach.fits(small, None)[0] and "never been recorded" in mach.fits(small, None)[1],
         "an unmeasured family passes, with a reason")
    note(mach.fits({"index": 0}, 17.2)[0],
         "a card with no recorded VRAM passes rather than being refused")
    why = mach.fits(small, 17.2)[1]
    note("17.2" in why and "16" in why and "slower" in why,
         "the refusal names both numbers and the consequence")

    tmp = Path(tempfile.mkdtemp(prefix="checkvram-"))
    saved_map = capacity_mod.MAP_PATH
    try:
        _sandbox_map(tmp)

        # Synthetic families against the sandbox map. Built from a real entry
        # so the dataclass shape stays honest.
        base = registry.MODELS[0]
        hungry = dataclasses.replace(base, name="hungry", venv=".venv-both",
                                     peak_vram_gb=17.2)
        modest = dataclasses.replace(base, name="modest", venv=".venv-both",
                                     peak_vram_gb=10.4)
        capped = dataclasses.replace(base, name="capped", venv=".venv-capped",
                                     peak_vram_gb=2.0)

        print("\n2. the chooser against the sandbox map")
        cells = {c["index"]: c for c in td._cells(hungry, KEY)}
        note(cells[0]["ok"] and cells[1]["ok"],
             "both cards can RUN the hungry family (so `ok` alone cannot choose)")
        note(not cells[0]["fits"] and cells[1]["fits"], "only card 1 FITS it")
        note(td._default_card(hungry, KEY) == 1,
             "a 17.2 GiB family defaults to the 24 GB card — THE ORIGINAL BUG")
        note(td._default_card(modest, KEY) == 0,
             "a 10.4 GiB family still defaults to card 0 (first that fits)")
        # can_run GATES while fits STEERS: the capped venv cannot reach card 0
        # at all, however tiny its footprint.
        cells_c = {c["index"]: c for c in td._cells(capped, KEY)}
        note(not cells_c[0]["ok"] and cells_c[1]["ok"],
             "an sm-capped venv cannot reach the newer card")
        note(td._default_card(capped, KEY) == 1,
             "…so even a 2 GiB model lands on card 1 — capability GATES, "
             "memory only steers")

        print("\n3. MUTATION — does the choice follow the number, or just coincide?")
        # Shrink the appetite: it should move BACK to card 0. If this does not
        # change, the chooser is ignoring peak_vram_gb and part 2 was luck.
        tiny = dataclasses.replace(hungry, peak_vram_gb=2.0)
        note(td._default_card(tiny, KEY) == 0,
             "same family at a 2 GiB peak moves to card 0 — the choice tracks it")
        # Grow it past BOTH cards: no card fits, so it must still get one rather
        # than becoming unstartable.
        huge = dataclasses.replace(hungry, peak_vram_gb=64.0)
        note(td._default_card(huge, KEY) == 0,
             "at 64 GiB nothing fits, so it falls back to the first usable card")
        # And unmeasured behaves exactly as it did before this rule existed.
        unk = dataclasses.replace(hungry, peak_vram_gb=None)
        note(td._default_card(unk, KEY) == 0,
             "an unmeasured family keeps the old first-card-that-works answer")

        print("\n4. an unprobed machine degrades to card 0, never to a refusal")
        idx, reason = mach.pick_card("never-probed", venv=".venv-both")
        note(idx == 0 and "not been probed" in reason,
             f"unprobed -> card 0, and the note says so ({reason[:50]})")
    finally:
        capacity_mod.MAP_PATH = saved_map
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print("CARD CHOICE IS NOT VRAM-AWARE:")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)
    print("a run lands on a card it fits; an undersized one is a note, not a wall")


if __name__ == "__main__":
    main()
