"""The janitor's one-off sweep must delete debris and NOTHING load-bearing.

WHY THIS IS A CHECK. outputs/ root holds `training_history.json` — the ledger,
every run this repo has ever done, append-only and reproducible from nothing —
alongside `active_model.json` and `serving_engine.json` (what currently serves),
`machines.json`, `retrain_state.json` and the GPU locks. It also holds ~50 files
that a one-off script wrote in July and abandoned. A sweep that got the
distinction wrong by one glob would take the record of everything with it, and it
would do so at midnight, from a timer, with nobody watching.

So the patterns are asserted against a FIXTURE rather than reasoned about: build
a directory containing both kinds of file, run the real function over it, and
require that every decoy survives and every piece of debris does not.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []

# Must be deleted once old enough.
DEBRIS = [
    "tg_1004507388_35.raw.txt", "tg_1004507388_140.raw.txt",
    "autolabel_full.log", "dataset_autolabel.log", "autocrop_preview.log",
    "finish_labels.log", "label_new.log", "relabel_4377.log",
    "webui_test.log", "cockpit_test.log",
    "la.log", "la_result.json", "la_state.json",
    "rimfilter_check_4342.png",
]
# Must SURVIVE, however old. Losing any of these is unrecoverable or breaks a
# running service.
KEEP = [
    "training_history.json",     # the ledger — every run ever
    "active_model.json",         # what serves
    "serving_engine.json",
    "machines.json",
    "retrain_state.json",
    "retrain.log",
    "retrain_worker.log",
    "bot.log",
    "second-bot.log",
    "night_queue.log",
    "webui.log",
    "gpu.lock", "gpu0.lock", "gpu1.lock",
    ".last_backup",
    "tg_1004507388_1957_annotated.png",   # sweep_overlays' job, not this one
    "log.txt",                            # generic, and not in the pattern list
]


def build(root: Path, age_days: float) -> None:
    old = time.time() - age_days * 86400
    for name in DEBRIS + KEEP:
        f = root / name
        f.write_text("x", encoding="utf-8")
        import os
        os.utime(f, (old, old))


print("  [1] old debris goes, load-bearing files stay")
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    build(d, age_days=90)
    import groundwork.config as cfg
    from groundwork.dataset.viz import pending_janitor as pj
    real = cfg.OUTPUTS_DIR
    cfg.OUTPUTS_DIR = d                       # the function imports it at call time
    try:
        n = pj.sweep_one_offs(older_than_days=30.0)
    finally:
        cfg.OUTPUTS_DIR = real
    survived = {f.name for f in d.iterdir()}
    wrongly_kept = [x for x in DEBRIS if x in survived]
    wrongly_gone = [x for x in KEEP if x not in survived]
    print(f"      deleted {n}, {len(survived)} file(s) left")
    if wrongly_kept:
        FAILURES.append(f"debris NOT deleted: {wrongly_kept}")
        print(f"      ✗ still present: {wrongly_kept}")
    else:
        print(f"      ✓ all {len(DEBRIS)} debris files removed")
    if wrongly_gone:
        FAILURES.append(f"LOAD-BEARING FILES DELETED: {wrongly_gone}")
        print(f"      ✗✗ DELETED: {wrongly_gone}")
    else:
        print(f"      ✓ all {len(KEEP)} load-bearing files survived")

print("\n  [2] recent debris is left alone")
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    build(d, age_days=3)                      # inside the 30-day floor
    import groundwork.config as cfg
    from groundwork.dataset.viz import pending_janitor as pj
    real = cfg.OUTPUTS_DIR
    cfg.OUTPUTS_DIR = d
    try:
        n = pj.sweep_one_offs(older_than_days=30.0)
    finally:
        cfg.OUTPUTS_DIR = real
    if n != 0:
        FAILURES.append(f"deleted {n} file(s) that were only 3 days old")
        print(f"      ✗ deleted {n}")
    else:
        print("      ✓ nothing deleted at 3 days old")

print("\n  [3] prove it can fail")
# A pattern that matched the ledger must be caught by [1]. Verify the assertion
# itself works rather than trusting that it would have.
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    build(d, age_days=90)
    (d / "training_history.json").unlink()    # simulate the sweep eating it
    survived = {f.name for f in d.iterdir()}
    if "training_history.json" not in survived:
        print("      ✓ a missing ledger IS detected by the same comparison")
    else:
        FAILURES.append("the survival check cannot notice a deleted ledger")

print()
if FAILURES:
    for f in FAILURES:
        print(f"  ✗ {f}")
    sys.exit(1)
print("  the one-off sweep removes only what nothing owns")
