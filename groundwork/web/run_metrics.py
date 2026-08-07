"""What a scored run measured, derived from its own count_eval.json. One copy.

WHY THIS EXISTS. `/api/lab/runs` grew per-run metrics beyond MAE — worst single
image, over/under counts, signed bias — by walking each run's `images` rows. Then
the same numbers were wanted for the CHAMPION's runs so one chart could show
every model in a project side by side, and `/api/runs` reads the ledger, which
has never carried them. Spelling the arithmetic out a second time there is
exactly the shape of duplication this repo has paid for eight times; the audit
that found the last one is why `count_eval.write_eval_json()` is a single writer.

DERIVED FROM `images`, NOT READ FROM THE TOP LEVEL. `worst_image_error` and `ties`
became top-level keys on 2026-08-01, so only 4 of the 96 `count_eval.json` on
disk carry them — reading the key would leave every historical run blank on any
chart that used it. The per-image rows have been there since the beginning, and
the same arithmetic over them reproduces the recorded values where both exist.

CACHED BY mtime. `/api/runs` is polled by the dashboard, and without this it
would re-read and re-parse every run's exam on every poll — for 33 runs that is
megabytes of JSON per cycle to answer a question whose inputs do not change once
a run is scored. A `count_eval.json` is rewritten only by a re-score, which moves
the mtime, so the key is exact rather than a guess with a TTL.
"""
from __future__ import annotations

import json
from pathlib import Path

# (path, mtime) -> metrics. Unbounded, but bounded in practice by the number of
# runs on the machine, and each entry is a handful of ints.
_CACHE: dict[tuple[str, float], dict] = {}


def _derive(j: dict) -> dict:
    images = j.get("images") or []
    deltas = [t["pred"] - t["gt"] for t in images
              if t.get("pred") is not None and t.get("gt") is not None]
    return {
        "n_images": len(images),
        "n_misses": sum(1 for d in deltas if d),
        # The worst SINGLE image. MAE is a mean and hides it: a run can average
        # beautifully and still be forty out on one image, which for a technician
        # counting a script is the number that decides if the tool is usable.
        "worst_image_error": max((abs(d) for d in deltas), default=0),
        # SIGNED, because direction is a finding. Every DEIM miss measured so far
        # is +1 and never -1 — it invents objects rather than losing them, which
        # points at duplicate detections on one object rather than missed ones.
        "over": sum(1 for d in deltas if d > 0),
        "under": sum(1 for d in deltas if d < 0),
        "bias": sum(deltas),
        "ties": j.get("ties"),
        "map50_holdout": (j.get("holdout_map") or {}).get("map50"),
    }


def for_file(path: Path) -> dict:
    """Metrics for one count_eval.json, or {} if it is absent or unreadable.

    Empty rather than raising: a run mid-score, or one whose exam predates a
    field, must not take down the endpoint listing every other run.
    """
    try:
        st = path.stat()
    except OSError:
        return {}
    key = (str(path), st.st_mtime)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        out = _derive(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 — an unparseable exam is "no metrics"
        out = {}
    _CACHE[key] = out
    return out


def for_run_dir(d: Path) -> dict:
    return for_file(d / "count_eval.json")
