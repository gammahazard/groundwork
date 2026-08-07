"""The alt eval: THE exam, with a pluggable predictor.

It used to *mirror* groundwork.dataset.pipeline.count_eval.evaluate() — same
conf x iou grids, same find_stacked -> count chain, same strict-< tie-break in the same
iteration order, same JSON schema and preview gallery — all maintained BY HAND
across two files. Every cross-model number in the cockpit depended on those two
implementations still agreeing, and nothing checked that they did.

They are now one: `dataset.pipeline.eval_core` owns the sweep, the counting
chain, the per-bucket MAE and the gallery, and both callers pass it
a `boxes_for(stem, conf, iou)`. What differs between champion and challenger is
exactly one thing, which is the thing that SHOULD differ:

  challengers expose infer_raw() — ONE inference per image, then the 42 sweep
  cells are score-threshold + numpy NMS over the cached raw boxes. The
  ultralytics parity predictor instead re-runs predict() per cell, because its
  NMS lives inside predict() (bit-identical to the champion; slow; one-off).

Which filters run is groundwork/serve/serving_filters' call, read by
eval_core for both exams.

    .venv/bin/python -m altmodels.harness --predictor ultralytics \
        --weights outputs/dataset/runs/yolov8n-29/weights/best.pt \
        --run-name parity/yolov8n-29-parity --restrict-ref
    .venv-alt/bin/python -m altmodels.harness --predictor rfdetr \
        --weights outputs/alt/rfdetr-nano-560-a/train/checkpoint_best_total.pth \
        --imgsz 560 --run-name rfdetr-nano-560-a --restrict-ref
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

# _dirs/_gt_counts/_images/grid_edge/write_eval_json stay with count_eval: they
# are about WHERE the exam reads and WHAT it writes. eval_core is HOW it scores.
from groundwork.dataset.pipeline.count_eval import (_dirs, _gt_counts, _images,
                                                     grid_edge, write_eval_json)
from groundwork.dataset.pipeline import eval_core
from groundwork import project
from groundwork.dataset import paths

from . import gpu, nms
from .predictors import get_predictor

# There is deliberately no ALT or REPO constant here any more. Where a challenger run
# lands is a PROJECT question and the answer is pp.ALT_DIR — one place, so the
# harness cannot write somewhere the cockpit does not read.


def _pinned_reference(pp) -> Path | None:
    """count_eval.json of the currently pinned run (reference follows the pin).

    `pp`-relative, so a project with no pin gets None — not a comparison
    against a different project's champion, which would be a number that
    looks fine and means nothing.
    """
    try:
        from groundwork.serve import runtime_model
        # WITH THE PROJECT. This called get_active() with no argument — a
        # TypeError the except below swallowed, so the follow-the-pin behaviour
        # this function exists for never happened even once.
        active = runtime_model.get_active(pp)
        if active:
            w = Path(active["weights"]).parent.parent / "count_eval.json"
            if w.is_relative_to(pp.RUNS_DIR):
                return w
    except Exception as e:  # noqa: BLE001
        print(f"[harness] no pinned reference ({type(e).__name__}: {e})", flush=True)
    # NEWEST SCORED RUN as the fallback. This named one specific run from the
    # machine this code grew up on, which exists on no other install — so every
    # fresh install silently had no reference at all.
    scored = sorted((d / "count_eval.json" for d in pp.RUNS_DIR.glob("*")
                     if (d / "count_eval.json").exists()),
                    key=lambda f: f.stat().st_mtime, reverse=True)
    return scored[0] if scored else None


def _boxes_for(pred, raw_cache, stem, pil, conf, iou):
    """Sweep-cell boxes via whichever protocol the predictor speaks."""
    if raw_cache is not None:
        boxes, scores = raw_cache[stem]
        return nms.filter_and_nms(boxes, scores, conf, iou)
    return pred.predict_cells(pil, conf, iou)


def run(predictor: str, weights: Path, imgsz: int, run_name: str,
        source: str, reference: Path | None, restrict_ref: bool,
        pp=None) -> float:
    run_dir = pp.ALT_DIR / run_name
    img_dir, lbl_dir = _dirs(source, pp)
    gt = _gt_counts(lbl_dir)
    imgs = _images(img_dir)
    stems = sorted(set(gt) & set(imgs))

    ref = json.loads(reference.read_text()) if reference and reference.exists() else None
    if restrict_ref:
        if not ref:
            raise SystemExit("--restrict-ref needs a readable --reference json")
        ref_gt = {t["stem"]: t["gt"] for t in ref["images"]}
        missing = sorted(set(ref_gt) - set(stems))
        if missing:
            raise SystemExit(f"[harness] reference stems missing from {source}: {missing}")
        drift = sorted(s for s in ref_gt if gt[s] != ref_gt[s])
        if drift:
            raise SystemExit(f"[harness] gt drifted since the reference for: {drift} "
                             "— restricted comparison would be meaningless")
        stems = sorted(ref_gt)
        print(f"[harness] restricted to the reference's {len(stems)} stems")
    if not stems:
        raise SystemExit(f"no {source} images found")
    # Say WHICH dataset this is before doing any work — a challenger run is
    # 60 epochs, and silent wrong-data is this repo's most expensive failure.
    # In run() rather than main() so it prints however the harness is entered.
    print(f"[harness] project {pp.slug} | {pp.root} | runs -> {pp.ALT_DIR}")
    print(f"[harness] {predictor} @ imgsz {imgsz} on {source} ({len(stems)} images)")

    t0 = time.time()
    with gpu.acquire(run_dir, what=f"harness {predictor} {run_name}"):
        pred = get_predictor(predictor, weights, imgsz)
        pil = {s: Image.open(imgs[s]).convert("RGB") for s in stems}

        raw_cache = None
        if hasattr(pred, "infer_raw"):
            raw_cache = {}
            for s in stems:
                raw_cache[s] = pred.infer_raw(pil[s])
            print(f"[harness] raw inference done ({time.time() - t0:.0f}s) — "
                  "sweep runs on cached boxes")

        # The ONE thing that differs from the champion's exam: where boxes come
        # from. Challengers infer ONCE per image and re-NMS the cached raw boxes
        # for each of the 42 cells; ultralytics re-runs predict() per cell
        # because its NMS lives inside predict(). Everything after this line is
        # eval_core's, shared with count_eval, so a predictor difference can no
        # longer become an exam difference.
        def boxes_for(stem, conf, iou):
            return _boxes_for(pred, raw_cache, stem, pil[stem], conf, iou)

        best, grid_pred, grid_mae = eval_core.sweep(stems, pil, gt, boxes_for)
        mae, conf, iou = best
        print(f"\n[harness] BEST: conf={conf} iou={iou} -> MAE {mae:.4f}")
        # A winner sitting on the edge of the sweep means the true optimum may
        # lie OUTSIDE it — the score is then a floor, not the model's best. The
        # grid is shared with count_eval (the champion's exam), so widening it
        # would break every past comparison: flag it, never silently fix it.
        edge = grid_edge(conf, iou)
        if edge:
            print("[harness] ⚠ WINNER AT GRID EDGE: " + ", ".join(edge)
                  + " — this MAE may understate the model; the grid is shared "
                    "with count_eval so it is deliberately not widened here")

        # Buckets + preview gallery at the winning cell — eval_core's, which is
        # count_eval's. NOTE this fixes a real divergence: the code here called
        # bucketize()/load() with NO project, so a challenger's per-bucket MAE
        # was always read from the DEFAULT project's tag map. Identical today
        # because there is one project, and silently wrong as soon as there are
        # two.
        buckets = eval_core.bucket_mae(stems, grid_pred[(conf, iou)], gt, pp)
        preview = run_dir / "eval_preview"
        image_records = eval_core.render_previews(stems, pil, gt, boxes_for,
                                                  conf, iou, preview, pp)

    run_dir.mkdir(parents=True, exist_ok=True)
    # ONE writer, shared with the champion's exam — see count_eval.write_eval_json.
    # This file used to spell the same dict out independently, and drifted
    # exactly as you would expect: a hardcoded filter flag that outlived the
    # switch turning it off. The challenger and the champion now describe
    # themselves in the same words because there is one place that decides
    # what those words are.
    ties = sum(1 for v in grid_mae.values() if v == mae) - 1
    # The same ADDITIONAL measurement the champion's exam takes, on the same
    # images. Called here too because "one writer" has to mean one set of KEYS:
    # the harness was writing `holdout_map: null` for every run including the
    # ultralytics parity one, which can produce a real number. It still returns
    # None for a challenger checkpoint — ultralytics' validator cannot read a
    # .pth — and that is the honest answer, not a gap this fills with a guess.
    # Cross-model holdout mAP needs each predictor to measure its own; this is
    # the ultralytics half of it.
    hmap = eval_core.holdout_map(weights, imgsz, source, pp)
    if hmap:
        print(f"[harness] holdout mAP50 {hmap['map50']} "
              f"(mAP50-95 {hmap['map']}) on the same {len(stems)} images")
    write_eval_json(run_dir, "", conf=conf, iou=iou, mae=mae, imgsz=imgsz,
                    source=source, weights=Path(weights).name, buckets=buckets,
                    images=image_records, ties=ties, holdout_map=hmap)
    # MERGE into any trainer-written meta.json: overwriting clobbered arch/
    # license/epochs and downstream (dashboard chart) lost the family grouping
    try:
        meta = json.loads((run_dir / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        meta = {}
    meta.update({"run": run_name, "predictor": predictor, "weights": str(weights),
                 "imgsz": imgsz, "source": source, "n_images": len(stems),
                 "restricted_to_reference": bool(restrict_ref),
                 "reference": str(reference) if reference else None,
                 "eval_seconds": round(time.time() - t0, 1),
                 "python": sys.version.split()[0], "created": time.time(),
                 "grid_edge_winner": bool(edge), "grid_edge": edge or None})
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    if ref:
        ref_pred = {t["stem"]: t for t in ref["images"]}
        rows, gained, lost = [], [], []
        for t in image_records:
            rt = ref_pred.get(t["stem"])
            rows.append({"stem": t["stem"], "pred": t["pred"], "gt": t["gt"],
                         "ref_pred": rt["pred"] if rt else None,
                         "delta_vs_ref": (t["pred"] - rt["pred"]) if rt else None})
            if rt:
                ours_miss, ref_miss = t["pred"] != t["gt"], rt["pred"] != rt["gt"]
                if ours_miss and not ref_miss:
                    gained.append(t["stem"])
                if ref_miss and not ours_miss:
                    lost.append(t["stem"])
        (run_dir / "diff_vs_ref.json").write_text(json.dumps(
            {"reference": str(reference), "ref_mae": ref["mae"], "mae": round(mae, 4),
             "ref_winner": [ref["conf"], ref["iou"]], "winner": [conf, iou],
             "misses_gained": gained, "misses_fixed": lost, "images": rows},
            indent=1), encoding="utf-8")
        print(f"[harness] vs reference {reference.name} ({ref['mae']}): "
              f"mae {mae:.4f} | new misses {gained or 'none'} | "
              f"fixed misses {lost or 'none'}")
        if predictor == "ultralytics" and restrict_ref:
            exact = all(r["delta_vs_ref"] == 0 for r in rows)
            same_winner = [conf, iou] == [ref["conf"], ref["iou"]]
            print(f"[harness] PARITY: per-image {'EXACT' if exact else 'MISMATCH'}, "
                  f"winner {'SAME' if same_winner else 'DIFFERENT'} "
                  f"-> {'PASS' if exact and same_winner else 'FAIL'}")
    print(f"[harness] wrote {run_dir}")
    return mae


def main() -> None:
    ap = argparse.ArgumentParser(description="count_eval's exam, pluggable predictor.")
    ap.add_argument("--predictor", required=True,
                    choices=["ultralytics", "rfdetr", "rtmdet", "deim", "dfine"])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="required for challengers (their trained resolution); "
                         "defaults to the run's args.yaml for ultralytics")
    ap.add_argument("--run-name", required=True,
                    help="output dir under the project's alt/ tree "
                         "(e.g. rfdetr-nano-560-a)")
    ap.add_argument("--source", choices=["val", "testset"], default="testset")
    ap.add_argument("--reference", type=Path, default=None,
                    help="count_eval.json to diff against (default: the PINNED run's)")
    ap.add_argument("--restrict-ref", action="store_true",
                    help="eval only the reference's stems (required for pinned "
                         "comparisons — the holdout keeps growing)")
    # A stage runs as a subprocess and cannot inherit ambient state, so the
    # project travels on the command line — same flag, same wording, as every
    # other stage.
    project.add_argument(ap)
    args = ap.parse_args()
    imgsz = args.imgsz
    if imgsz is None:
        if args.predictor != "ultralytics":
            raise SystemExit("--imgsz is required for challenger predictors")
        from groundwork.serve import runtime_model
        imgsz = runtime_model.run_imgsz(args.weights.parent.parent)
    pp = paths.for_project(args.project)
    reference = args.reference or _pinned_reference(pp)
    run(args.predictor, args.weights, imgsz, args.run_name,
        args.source, reference, args.restrict_ref, pp)


if __name__ == "__main__":
    main()
