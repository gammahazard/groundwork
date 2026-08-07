"""No model may ship a detection ceiling below the densest labelled image.

WHAT THIS CATCHES. A detector that cannot EMIT more than N objects scores as
if it were bad at counting, when in fact it was never allowed to try. On the
predecessor, centernet scored a holdout MAE of 22.03 for exactly this reason:
its maximum prediction across the whole holdout was precisely 100, and every
image holding more than 100 objects came back as 100. A 349-object image was
reported as 100 — an error of 249 that is entirely a config default.

It is a SILENT failure, which is why it needs a check rather than care.
Nothing raises an exception; the model simply stops counting and the exam
records a number. That number then sits in the ledger next to
honestly-measured ones.

TWO KNOBS, AND max_per_img IS NOT ALWAYS THE BINDING ONE:

    yolo            max_det                          2000
    rtmdet/yolox    test_cfg.max_per_img             1000
    centernet       test_cfg.topk  <-- BINDING       stock 100; peaks are taken
                                                     BEFORE max_per_img applies
    deim / dfine    --queries (num_queries)          600
    rfdetr          --num-queries                    was 300, a hard DETR ceiling

The eval side must match the training side. rtmdet_predict raised max_per_img
but not topk, so a centernet trained with a raised ceiling would still have
been SCORED through a capped one.

The floor is the machine's own densest labelled image plus headroom, not a
constant — the point is to fail when the data outgrows the config. On a fresh
install with no labelled data the floor is 0 and only the pattern-rot arm has
teeth; the floor rises with every dataset the machine gains.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []
HEADROOM = 1.2      # a ceiling equal to the densest image is already too low


def densest_image() -> int:
    """Most labelled objects on any single image, across every project and
    every labelled bucket — a MACHINE question, so every_project()."""
    from groundwork.dataset import paths
    from groundwork.dataset.store import labelio
    worst = 0
    for pp in paths.every_project():
        for coll in labelio.COLLECTION_NAMES:
            try:
                _, lbl = labelio._dirs(coll, pp)
            except Exception:  # noqa: BLE001
                continue
            if not lbl.exists():
                continue
            for f in lbl.glob("*.txt"):
                if f.stem == "classes":
                    continue
                n = sum(1 for ln in f.read_text(errors="ignore").splitlines()
                        if ln.strip())
                worst = max(worst, n)
    return worst


def literal_ints(path: Path, pattern: str) -> list[int]:
    """Every integer literal assigned to a name matching `pattern`.

    Regex over source rather than import-and-inspect: these live in argparse
    defaults and mmengine config mutations inside functions that need a GPU to
    reach. The values are plain literals, so reading them is reliable.
    """
    out = []
    txt = path.read_text(encoding="utf-8")
    for m in re.finditer(pattern, txt):
        try:
            out.append(int(m.group(1)))
        except (ValueError, IndexError):
            pass
    return out


dense = densest_image()
need = int(dense * HEADROOM)
print(f"  densest labelled image on this machine: {dense} objects")
print(f"  required ceiling (x{HEADROOM}): {need}"
      + ("  (no labelled data yet — only the pattern-rot arm binds)"
         if dense == 0 else "") + "\n")

CHECKS = [
    ("altmodels/trainers/rtmdet.py", r"test_cfg\.max_per_img\s*=\s*(\d+)", "max_per_img"),
    ("altmodels/trainers/rtmdet.py", r"test_cfg\.topk\s*=\s*(\d+)", "topk (centernet)"),
    ("altmodels/predictors/rtmdet_predict.py", r"max_per_img\s*=\s*(\d+)", "max_per_img (eval)"),
    ("altmodels/predictors/rtmdet_predict.py", r"test_cfg\.topk\s*=\s*(\d+)", "topk (eval)"),
    ("altmodels/trainers/deim.py", r'"--queries",\s*type=int,\s*default=(\d+)', "--queries"),
    ("altmodels/trainers/dfine.py", r'"--queries",\s*type=int,\s*default=(\d+)', "--queries"),
    ("altmodels/trainers/rfdetr.py", r'"--num-queries",\s*type=int,\s*default=(\d+)', "--num-queries"),
    ("groundwork/dataset/pipeline/count_eval.py", r"max_det=(\d+)", "max_det (champion eval)"),
    ("groundwork/serve/yolo_counter.py", r"DEFAULT_MAX_DET\s*=\s*(\d+)", "max_det (runtime)"),
    ("groundwork/serve/deim_counter.py", r"self\.max_det\s*=\s*(\d+)", "max_det (deim runtime)"),
]

for rel, pat, label in CHECKS:
    f = ROOT / rel
    if not f.exists():
        FAILURES.append(f"{rel} is gone — this check is stale")
        continue
    vals = literal_ints(f, pat)
    if not vals:
        FAILURES.append(f"{rel}: no {label} found — the ceiling is unset or the "
                        f"pattern rotted, and an unchecked ceiling is how "
                        f"centernet shipped at 100")
        continue
    low = min(vals)
    ok = low >= need
    if not ok:
        FAILURES.append(f"{rel}: {label} = {low}, below the required {need}")
    print(f"  {'ok  ' if ok else 'FAIL'} {rel.split('/')[-1]:<22} {label:<24} {vals}")

print()
if FAILURES:
    for x in FAILURES:
        print(f"  ✗ {x}")
    sys.exit(1)
print("  every ceiling clears the densest labelled image with headroom")
