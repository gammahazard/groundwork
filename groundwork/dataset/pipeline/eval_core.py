"""THE exam. One definition of how a model is scored, for every model.

`count_eval.evaluate()` (the champion) and `altmodels/harness.run()` (the
challengers) once ran the same exam twice. harness's own docstring said so — it
"mirrors count_eval.evaluate() semantics precisely" — and listed the things it
had to keep in step by hand: the same conf x iou grids, the same
find_stacked -> count chain, the same strict-< tie-break in the same iteration
order, the same JSON schema and preview gallery.

That is a rule maintained by a comment, and it is the reason cross-model
comparison was trustworthy-but-fragile: every number in the Models & Runs table
depends on the two implementations still agreeing, and nothing checked that
they did.

WHAT ACTUALLY DIFFERED, and it is one line: how an image's boxes are obtained.

    champion     model.predict(pil, imgsz, conf, iou, max_det=2000) per cell
    challengers  ONE inference per image, then score-threshold + numpy NMS over
                 the cached raw boxes for each of the 42 cells

Everything downstream of that — centres, stacked-dot dedupe, the count, the
MAE, the winner, the buckets, the gallery — was identical prose in two files.
So this module takes `boxes_for(stem, conf, iou) -> list[xyxy]` and owns the
rest. A predictor difference stays a predictor difference; it can no longer
become an exam difference.

ONE REAL DIVERGENCE WAS FOUND AND FIXED BY UNIFYING. count_eval called
`testset_buckets.bucketize(stems, pp)` and `load(pp)`; harness called both with
no project at all, so a challenger's per-bucket MAE was always read from ONE
FIXED project's tag map. Identical while there is one project, and silently
wrong the moment there are two — precisely the class of bug project scoping
exists to remove.

NOTHING HERE MAY CHANGE WITHOUT RE-CERTIFYING. The iteration order, the strict
`<` comparison and the float arithmetic are all load-bearing: on a saturated
holdout most of the grid ties at 0.0 and the winner is simply the first cell
seen (conf outer, iou inner). Reordering these loops would silently re-tune
every run's served conf/iou.
"""
from __future__ import annotations

import shutil
from typing import Callable

from ..filters.dot_dedupe import dot_radius, find_stacked
from ..store import testset_buckets
from ...serve import serving_filters

# THE SWEEP AXES. Every certified MAE on disk was measured on exactly these
# numbers, so they are a fixed point, not a tuning knob: a winner that lands on
# an edge is FLAGGED (count_eval.grid_edge) rather than the grid being widened,
# because widening it would break every past comparison.
#
# Higher iou => fewer merges of touching objects (fights UNDER-counting). The
# low end is what fights OVER-counting: a sloppy checkpoint puts two boxes on
# one object at IoU 0.25-0.49, which every threshold >=0.5 keeps as two dots.
# Without 0.3/0.4 the sweep is one-sided and structurally unable to correct
# stacked dots.
#
# Written here ONCE and imported by count_eval, which re-exports the old private
# names for altmodels/harness. Transcribing them into this file instead of
# importing them is not hypothetical: the first draft of this module did exactly
# that, invented a 0.35 conf and a 0.55 iou, and dropped 0.10 and 0.9 — which
# would have silently re-tuned and re-scored every run in the ledger. It was
# caught by comparing the two tuples, not by reading them.
CONF_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
IOU_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# THE WIDE GRID — opt-in, never the default, and recorded in the output.
#
# WHY IT EXISTS. 28 of 29 champion runs and 4 of 5 challengers tonight won at
# iou 0.3, the floor. An axis whose optimum is always at the boundary has never
# actually been searched: the sweep has been effectively one-dimensional over
# conf for the project's whole history, and every one of those MAEs is a floor
# rather than the model's best. rtmdet-tiny additionally pinned conf=0.40, the
# ceiling, so both ends of both axes are now known to bind.
#
# WHY IT IS NOT THE DEFAULT. Every certified MAE on disk was measured on the
# tuple above, and the grid is shared with altmodels/harness — switching it
# silently would re-tune and re-score the entire ledger, which is exactly the
# accident this module's header describes being caught once already. So a run
# opts in, `grid: "wide"` lands in its count_eval.json, and a reader can always
# tell which exam produced a number.
CONF_GRID_WIDE = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
IOU_GRID_WIDE = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

GRIDS = {"standard": (CONF_GRID, IOU_GRID),
         "wide": (CONF_GRID_WIDE, IOU_GRID_WIDE)}

BoxesFor = Callable[[str, float, float], list]


def centres(boxes) -> list[tuple[float, float]]:
    return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]


def count_image(stem, pil_img, boxes) -> tuple[int, list, list]:
    """One image's count, and the pieces a gallery needs to explain it.

    Returns (count, kept_centres, stacked_indices). This IS the counting rule —
    the same chain the deployed counter runs, which is the whole point of
    serving_filters being one switch read by both.
    """
    kept = centres(boxes)
    w, h = pil_img.size
    stacked = find_stacked(kept, dot_radius(w, h)) if serving_filters.DEDUPE else []
    return len(kept) - len(stacked), kept, stacked


def score_at(stems, pil, gt, boxes_for: BoxesFor, conf: float, iou: float):
    """MAE and per-image predictions at ONE fixed (conf, iou). No search.

    This is what makes an honest held-out number possible: pick the cell on the
    val split, then measure the holdout ONCE at that cell. The sweep below
    reports `min over 42 cells measured on the same images it chose from`, which
    is an extreme-value statistic and biased low by construction.
    """
    preds, errs = {}, []
    for s in stems:
        n, *_ = count_image(s, pil[s], boxes_for(s, conf, iou))
        preds[s] = n
        errs.append(abs(n - gt[s]))
    return sum(errs) / len(errs), preds


def sweep(stems, pil, gt, boxes_for: BoxesFor, *, label: str = "eval",
          conf_grid=None, iou_grid=None):
    """The conf x iou sweep. Returns (best, grid_pred, grid_mae).

    `best` is (mae, conf, iou). The tie-break is STRICT `<` walked conf-outer /
    iou-inner, so the first cell reaching the minimum wins. That is not a
    preference — on a saturated holdout most cells reach 0.0 and this rule is
    the only thing making runs comparable to each other. It is recorded in the
    output as `ties` so a reader can tell a real winner from a coin toss.
    """
    best = None
    grid_pred: dict[tuple[float, float], dict[str, int]] = {}
    grid_mae: dict[tuple[float, float], float] = {}
    conf_grid = CONF_GRID if conf_grid is None else conf_grid
    iou_grid = IOU_GRID if iou_grid is None else iou_grid
    print("conf\\iou " + "  ".join(f"{v:>5.2f}" for v in iou_grid))
    for conf in conf_grid:
        row = []
        for iou in iou_grid:
            errs, preds = [], {}
            for s in stems:
                n, *_ = count_image(s, pil[s], boxes_for(s, conf, iou))
                preds[s] = n
                errs.append(abs(n - gt[s]))
            mae = sum(errs) / len(errs)
            grid_pred[(conf, iou)] = preds
            grid_mae[(conf, iou)] = mae
            row.append(f"{mae:>5.2f}")
            if best is None or mae < best[0]:
                best = (mae, conf, iou)
        print(f"{conf:>7.2f}  " + "  ".join(row))
    return best, grid_pred, grid_mae


def bucket_mae(stems, preds, gt, pp) -> dict:
    """Per-bucket MAE at the winning cell.

    Diagnostic only — conf/iou stay global. A blended average can hide one type
    regressing while another improves. `pp` is required because the tag map is
    a PROJECT's, not the machine's; harness used to omit it and silently read
    another project's.
    """
    errs = {s: abs(preds[s] - gt[s]) for s in stems}
    out = {}
    for bucket, members in testset_buckets.bucketize(stems, pp).items():
        es = [errs[s] for s in members]
        out[bucket] = {"mae": round(sum(es) / len(es), 4), "images": len(es)}
    return out


def render_previews(stems, pil, gt, boxes_for: BoxesFor, conf, iou,
                    preview_dir, pp) -> list[dict]:
    """The eval gallery, and the per-image records that go in the JSON.

    Shows WHERE a run is wrong, which a single MAE cannot: kept dots in the
    normal colour, stacked dots that were deduped in magenta. A -1 with a
    magenta ring means the dedupe collapsed a pair; a -1 with no dot at all is
    a detection miss. Those need opposite fixes, and the picture is the only
    thing that distinguishes them.
    """
    from ...la.annotate import annotate
    from ...la.parse import Detection, Detections
    from PIL import ImageDraw

    bmap = testset_buckets.load(pp)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in preview_dir.glob("*.png"):
        old.unlink()                                # refresh on a re-eval
    records = []
    for s in stems:
        n, kept, stacked = count_image(s, pil[s], boxes_for(s, conf, iou))
        final = [c for i, c in enumerate(kept) if i not in stacked]
        dets = Detections(items=[Detection("point", (float(x), float(y)))
                                 for x, y in final])
        overlay = annotate(pil[s], dets, numbered=False)
        w, h = pil[s].size
        rr, lw = max(4, round(min(w, h) / 160)), max(2, round(min(w, h) / 320))
        draw = ImageDraw.Draw(overlay)
        for k in stacked:                           # MAGENTA = deduped as a double-count
            x, y = kept[k]
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(235, 60, 220), width=lw)
        overlay.save(preview_dir / f"{s}.png")
        records.append({"stem": s, "pred": len(final), "gt": gt[s],
                        "dupes": len(stacked),
                        "bucket": bmap.get(s, testset_buckets.UNTAGGED)})
    return records


def holdout_map(weights, imgsz: int, source: str, pp) -> dict | None:
    """mAP on the SAME images the count exam used — or None if not applicable.

    WHY THIS EXISTS. `map50` in the ledger is `max(metrics/mAP50(B))` out of
    ultralytics' results.csv: the best epoch's score on the VAL SPLIT, which is
    the data the run early-stopped against. Every certified count-MAE comes from
    the frozen holdout instead, so a project ranking on mAP was ranking on a
    weaker number than a counting project — the gap PIVOT Phase 3 records.

    ULTRALYTICS' OWN VALIDATOR, not an AP implementation written here. Box
    matching, the 0.5 IoU sweep and the interpolation are all easy to get subtly
    wrong, this repo already carries eight duplicated-rule scars, and a
    hand-rolled AP would be the ninth — one that disagrees with the number the
    trainer printed and gives no way to tell which is right.

    The holdout already has the layout ultralytics wants (`testset/images` beside
    `testset/labels`), so this needs a data.yaml pointing at it and nothing else.

    THE CACHE IS CLEANED UP. The validator writes `labels.cache` next to the
    label dir. It is derived, it is a sibling rather than inside labels/, and
    every consumer globs `*.txt` — so it breaks nothing. It is still a write into
    a directory this repo calls FROZEN, and regenerating it costs 0.1s for 65
    files, so it is removed rather than left.

    Returns None for a non-holdout source or a non-ultralytics checkpoint: this
    is an ADDITIONAL number, and a run that cannot produce it must not be given a
    guessed one.
    """
    import tempfile
    from pathlib import Path

    if source != "testset":
        return None                    # only the frozen holdout earns the name
    weights = Path(weights)
    if weights.suffix != ".pt":
        return None                    # a challenger checkpoint; harness's job
    root = pp.DATASET_DIR / "testset"
    if not (root / "images").exists():
        return None
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(pp.CLASS_NAMES))
    tmp = Path(tempfile.mkdtemp(prefix="holdout_map_"))
    (tmp / "holdout.yaml").write_text(
        f"path: {root}\ntrain: images\nval: images\n"
        f"nc: {len(pp.CLASS_NAMES)}\nnames:\n{names}\n", encoding="utf-8")
    cache = root / "labels.cache"
    had_cache = cache.exists()
    try:
        from ultralytics import YOLO
        r = YOLO(str(weights)).val(
            data=str(tmp / "holdout.yaml"), imgsz=imgsz, split="val",
            verbose=False, plots=False, save_json=False,
            project=str(tmp), name="val", exist_ok=True)
        return {"map50": round(float(r.box.map50), 4),
                "map": round(float(r.box.map), 4)}
    except Exception as e:  # noqa: BLE001
        # An extra measurement must never fail the exam that produced the MAE.
        print(f"[eval] holdout mAP skipped ({type(e).__name__}: {e})", flush=True)
        return None
    finally:
        if not had_cache:
            cache.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
