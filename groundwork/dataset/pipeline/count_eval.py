"""Stage 4: evaluate what actually matters — per-image COUNT error, not box mAP.

mAP rewards precise boxes; a counter only needs one non-merged detection per
object. Default YOLO NMS (iou=0.7) merges touching objects -> undercount on
dense images, so we sweep (conf, iou) on the val set and report the setting
that minimizes count MAE. Those become the runtime defaults for the deployed
counter.

    python -m groundwork.dataset.pipeline.count_eval --project <slug> \\
        --weights <runs>/yolov8n/weights/best.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

from .. import paths, sidecar
from ..store import testset_buckets
from ..filters.dot_dedupe import find_stacked, dot_radius
from . import eval_core
from ...serve import serving_filters


# Kept as a name because altmodels/harness.py imports it; the rule itself lives
# with the filter it parameterises.
_dot_r = dot_radius

# The sweep axes live in eval_core — ONE definition, shared with the challenger
# harness, which is the whole point of the collapse. These two names remain
# because harness.py imports them from here; they are aliases, not copies, so
# the two exams cannot drift apart by an edit to one file.
_CONF_GRID = list(eval_core.CONF_GRID)
_IOU_GRID = list(eval_core.IOU_GRID)


def grid_edge(conf: float, iou: float, grid: str = "standard") -> list[str]:
    """Which axes the winning cell sits on the BOUNDARY of, if any.

    A winner on the edge means the true optimum may lie outside the sweep, so
    the score is a floor rather than the model's best. The grid is shared with
    the challenger harness, so widening it would invalidate every past
    comparison: flag it, never silently fix it.

    This lived only in altmodels/harness.py, which meant the challenger's exam
    could raise a caveat the champion's exam structurally could not — and 25 of
    26 champion runs win at iou 0.3, the grid minimum.
    """
    cg, ig = eval_core.GRIDS.get(grid, (_CONF_GRID, _IOU_GRID))
    edge = []
    if conf in (cg[0], cg[-1]):
        edge.append(f"conf={conf} (grid {cg[0]}-{cg[-1]})")
    if iou in (ig[0], ig[-1]):
        edge.append(f"iou={iou} (grid {ig[0]}-{ig[-1]})")
    return edge


def write_eval_json(run_dir, tag: str, *, conf: float, iou: float, mae: float,
                    imgsz: int, source: str, weights: str, buckets: dict,
                    images: list, ties: int | None = None,
                    holdout_map: dict | None = None,
                    tuned_on: str | None = None, grid: str = "standard"):
    """THE definition of what a count_eval.json contains. ONE writer.

    It was two — here and altmodels/harness.py — spelled out as independent
    dict literals, and they drifted exactly as predicted: the harness hardcoded
    a filter flag long after the switch turned it off. Both exams now emit the
    same keys in the same order because there is one place that decides.

    Two fields are new as of 2026-07-30, so files written before it lack them:

        grid_edge     which axes the winner sat on the boundary of (see above)
        ties          how many OTHER cells scored the same MAE. On a saturated
                      holdout many cells reach 0.0 and the documented tie-break
                      (first seen, conf-outer/iou-inner) picks one — without
                      this, an edge winner is indistinguishable from a tie, and
                      "25 of 26 runs prefer iou 0.3" cannot be read either way.

    Atomic: this file has readers across the bot, the cockpit, adopt_run and the
    harness, and it is written by a subprocess while they are live. A torn read
    makes runtime_model._tuning() fall back to hardcoded defaults instead of the
    run's tuned pair — a silent serving change.
    """
    edge = grid_edge(conf, iou, grid)
    out = run_dir / f"count_eval{'_' + tag if tag else ''}.json"
    # MISCOUNTS, not just MAE. On a saturated holdout MAE is 0, 0.0154 or 0.0308
    # — zero, one or two wrong images out of 65 — and the ratio hides both the
    # count and the size. One challenger got 28 images wrong with a single image
    # off by 27; nothing in a mean says that.
    wrong = [t for t in images if t.get("pred") != t.get("gt")]
    sidecar.write_json(out, {
        "conf": conf, "iou": iou, "mae": round(mae, 4),
        "miscounts": len(wrong), "n_images": len(images),
        "worst_image_error": max((abs(t["pred"] - t["gt"]) for t in wrong),
                                 default=0),
        # WHERE the thresholds came from. `tuned_on: "val"` means the holdout
        # number below is honest — the cell was chosen on data this MAE was not
        # measured on. Absent or equal to `source` means the sweep chose and
        # reported on the same images, which is biased low by construction.
        "tuned_on": tuned_on or source,
        "grid": grid,
        "imgsz": imgsz, "source": source, "weights": weights,
        # RECORD the regime, never assert it: this is the only thing telling a
        # future reader which filters produced a certified MAE.
        "dedupe": serving_filters.DEDUPE,
        "grid_edge": edge or None,
        "grid_edge_winner": bool(edge),
        "ties": ties,
        # mAP measured on THESE images, when the source is the frozen holdout
        # and the checkpoint is one ultralytics can validate. Absent otherwise —
        # never guessed, because the whole point is that it is a held-out number
        # and the ledger's `map50` (from results.csv) is a val-split one.
        "holdout_map": holdout_map,
        "buckets": buckets,
        "images": images,
    })
    return out


def _dirs(source: str, pp) -> tuple[Path, Path]:
    """(images_dir, labels_dir) for the eval source: 'val' split or 'testset' holdout."""
    if source == "testset":
        return pp.TESTSET_IMAGES, pp.TESTSET_LABELS
    return pp.IMAGES_DIR / "val", pp.LABELS_DIR / "val"


def _gt_counts(lbl_dir: Path) -> dict[str, int]:
    """Ground-truth object count per image = non-empty lines in its label file."""
    out = {}
    for lbl in lbl_dir.glob("*.txt"):
        lines = [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out[lbl.stem] = len(lines)
    return out


def _images(img_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".txt"}


def _restrict_to_exam(stems: list[str], name: str, gt: dict, pp, source: str) -> list[str]:
    """Cut the image set down to a FROZEN exam, or refuse.

    THE REFUSAL IS THE FEATURE. The reason to name an exam is so a number from
    today can be compared to a number from six months ago; scoring 65 of a frozen
    68 and reporting it under the same name would defeat that completely, and it
    would look exactly like a valid result. So a missing image or a changed
    ground-truth count is a hard stop, not a warning — `store.exam verify` exists
    to tell you precisely what moved.
    """
    from ..store import exam as exam_mod
    if source != "testset":
        raise SystemExit(f"--exam names a frozen HOLDOUT; --source {source} is "
                         f"not the holdout. Drop one of the two.")
    frozen = exam_mod.load(name, pp)["images"]
    have = set(stems)
    missing = sorted(set(frozen) - have)
    drift = sorted(s for s in set(frozen) & have if frozen[s]["gt"] != gt[s])
    if missing or drift:
        lines = [f"exam {name!r} cannot be reproduced from this holdout:"]
        lines += [f"    MISSING     {s}" for s in missing]
        lines += [f"    RELABELLED  {s} — exam says {frozen[s]['gt']} objects, "
                  f"holdout says {gt[s]}" for s in drift]
        lines.append("Run `python -m groundwork.dataset.store.exam verify "
                     f"{name}` for the full picture. Scoring a subset under this "
                     "name would produce a number that looks comparable and is not.")
        raise SystemExit("\n".join(lines))
    extra = len(have) - len(frozen)
    print(f"[count_eval] exam {name}: {len(frozen)} frozen images"
          + (f" (excluding {extra} image(s) added to the holdout since)"
             if extra > 0 else " — the whole holdout"))
    return sorted(frozen)


def evaluate(weights: Path, pp, imgsz: int | None = None, source: str = "val",
             tag: str = "", tune_source: str | None = None,
             grid: str = "standard", exam: str | None = None) -> float:
    """Score `weights` on `source` and write the results next to the run.

    `imgsz` defaults to the size the run was TRAINED at (its args.yaml) — the same
    size the runtime serves it at. conf/iou interact with imgsz, so sweeping at any
    other size would tune thresholds for a system that never ships.

    `tag` suffixes the outputs (count_eval_<tag>.json / eval_preview_<tag>/) so a
    second checkpoint from the same run can be scored without clobbering the first.
    Untagged writes the canonical names the runtime and UI read. Returns the MAE.
    """
    from ultralytics import YOLO
    from PIL import Image
    from ...serve import runtime_model

    if imgsz is None:
        imgsz = runtime_model.run_imgsz(weights.parent.parent)
        print(f"[count_eval] imgsz {imgsz} (the run's trained size, from args.yaml)")

    # The untagged count_eval.json is what the runtime SERVES (conf/iou). When a
    # frozen holdout exists, that file must only ever hold holdout-tuned values —
    # so a stray val-split eval is auto-tagged instead of clobbering it.
    if not tag and source == "val" and any(pp.TESTSET_LABELS.glob("*.txt")):
        tag = "val"
        print("[count_eval] holdout exists -> val results write count_eval_val.json "
              "(the served count_eval.json stays holdout-tuned)")

    img_dir, lbl_dir = _dirs(source, pp)
    gt = _gt_counts(lbl_dir)
    imgs = _images(img_dir)
    stems = sorted(set(gt) & set(imgs))
    if not stems:
        raise SystemExit(f"No {source} images/labels found in {img_dir} — "
                         "run split first (val) or add images (testset).")
    # A NAMED EXAM PINS THE IMAGE SET. Without this the holdout is whatever is
    # on disk today, so every past MAE quietly re-denominates as it grows — a
    # 65 -> 68 change is already on record as having made a spread look like a
    # model difference when it was not.
    if exam:
        stems = _restrict_to_exam(stems, exam, gt, pp, source)
    print(f"[count_eval] eval source: {source} ({len(stems)} images)")

    model = YOLO(str(weights))
    # Load each image once (the sweep re-scores it 42x: 6 conf x 7 iou) and score
    # through the SAME pipeline the bot deploys — whatever
    # groundwork/serving_filters enables. If eval and runtime ever disagree the
    # metric measures a system nobody uses, which is why both read one switch
    # instead of each deciding for themselves.
    pil = {stem: Image.open(imgs[stem]).convert("RGB") for stem in stems}
    # THE ONE DIFFERENCE between the champion's exam and a challenger's: how an
    # image's boxes are obtained. Everything downstream — centres, dedupe,
    # count, MAE, winner, buckets, gallery — is eval_core's, shared with
    # altmodels/harness. ultralytics re-runs predict() per sweep cell because
    # its NMS lives inside predict(); challengers infer once per image and
    # re-NMS cached boxes.
    def _boxes_for(images):
        """A boxes_for bound to ONE image set.

        It used to close directly over `pil`, the eval source's images. That is
        invisible until something sweeps a DIFFERENT set of stems through it —
        tuning on val raised KeyError('IMG_4347') on the first val image, because
        a val stem is not in the testset dict. Binding the images explicitly
        makes the pairing impossible to get wrong.
        """
        def f(stem, conf, iou):
            r = model.predict(images[stem], imgsz=imgsz, conf=conf, iou=iou,
                              max_det=2000, verbose=False)[0]
            return r.boxes.xyxy.tolist()
        return f

    boxes_for = _boxes_for(pil)

    cg, ig = eval_core.GRIDS[grid]
    print(f"[count_eval] {len(stems)} {source} images; sweeping conf x iou "
          f"({grid} grid, {len(cg)}x{len(ig)} cells) "
          f"(filters: model + dedupe, exactly as served) ...\n")

    if tune_source and tune_source != source:
        # HONEST HELD-OUT NUMBER. Choose the cell on `tune_source`, then measure
        # `source` ONCE at that cell.
        #
        # The default path below reports min-over-N-cells measured on the very
        # images it chose from. That is an extreme-value statistic: biased low,
        # and noisier than a single measurement because the minimum of many
        # noisy cells moves around more than any one of them. It is the same
        # discipline as "never train on the holdout", applied to hyperparameters.
        print(f"[count_eval] tuning on {tune_source}, then scoring {source} once "
              f"at the chosen cell")
        t_img, t_lbl = _dirs(tune_source, pp)
        t_gt = _gt_counts(t_lbl)
        t_imgs = _images(t_img)
        t_stems = sorted(set(t_gt) & set(t_imgs))
        if not t_stems:
            raise SystemExit(f"No {tune_source} images/labels to tune on — "
                             "run split first.")
        t_pil = {st: Image.open(t_imgs[st]).convert("RGB") for st in t_stems}
        t_boxes_for = _boxes_for(t_pil)
        print(f"[count_eval] tuning grid on {len(t_stems)} {tune_source} images:")
        t_best, _, t_grid_mae = eval_core.sweep(
            t_stems, t_pil, t_gt, t_boxes_for,
            conf_grid=cg, iou_grid=ig)
        _, conf, iou = t_best
        ties = sum(1 for v in t_grid_mae.values() if v == t_best[0]) - 1
        print(f"[count_eval] chose conf={conf} iou={iou} on {tune_source} "
              f"(its MAE there {t_best[0]:.4f}, {ties} tied cell(s))")
        mae, preds = eval_core.score_at(stems, pil, gt, boxes_for, conf, iou)
        grid_pred = {(conf, iou): preds}
        grid_mae = {(conf, iou): mae}
    else:
        best, grid_pred, grid_mae = eval_core.sweep(
            stems, pil, gt, boxes_for, conf_grid=cg, iou_grid=ig)
        mae, conf, iou = best
    print(f"\n[count_eval] BEST: conf={conf} iou={iou} -> count MAE {mae:.2f} objects/image")
    print("[count_eval] per-image at best setting (pred vs gt):")
    for stem in stems:
        p, g = grid_pred[(conf, iou)][stem], gt[stem]
        flag = "" if p == g else f"  <-- off by {p - g:+d}"
        print(f"    {stem}: {p} vs {g}{flag}")

    # Per-bucket MAE at the winning setting — a blended average can hide one type
    # regressing while another improves. Diagnostic only; conf/iou above are global.
    buckets = eval_core.bucket_mae(stems, grid_pred[(conf, iou)], gt, pp)
    if len(buckets) > 1 or testset_buckets.UNTAGGED not in buckets:
        print("[count_eval] per-bucket MAE (eval lens):")
        for bucket in sorted(buckets):
            b = buckets[bucket]
            print(f"    {bucket:>10}: MAE {b['mae']:.2f}  ({b['images']} images)")
    # ---- Eval gallery: render the PREDICTED dots on each test image so you can
    # SEE where it's wrong — a miscount with no dot on the missed object is a
    # detection miss (add data like it); a magenta ring is the dedupe collapsing
    # a pair. Re-predict once per image at the winning conf/iou.
    suffix = f"_{tag}" if tag else ""
    preview_dir = weights.parent.parent / f"eval_preview{suffix}"
    image_records = eval_core.render_previews(stems, pil, gt, boxes_for,
                                              conf, iou, preview_dir, pp)
    print(f"[count_eval] wrote {len(image_records)} eval previews -> {preview_dir}")

    # Persist the winning conf/iou next to the run so the deployed counter serves
    # THESE (not a stale hardcoded default). runtime_model.resolve() reads it.
    # Atomic. This file has 14 readers across the bot, the cockpit, the
    # challenger harness and adopt_run, and count_eval writes it as a subprocess
    # while they are live. A torn read makes runtime_model._tuning() fall back to
    # the hardcoded 0.4/0.5 defaults instead of this run's tuned pair — a silent
    # serving change, exactly what "never hardcoded" exists to prevent.
    # How many OTHER cells matched the winning MAE. On a saturated holdout most
    # of the grid reaches 0.0 and the documented tie-break (first seen,
    # conf-outer/iou-inner) picks one — so "the winner is at iou 0.3" may mean
    # the model prefers it, or may mean 0.3 was simply first. Recorded so a
    # reader can tell, instead of the two being indistinguishable.
    if not (tune_source and tune_source != source):
        ties = sum(1 for k, v in grid_mae.items() if v == mae) - 1
    # An ADDITIONAL measurement on the same images: ultralytics' own validator,
    # ~2s for 65 images. Never allowed to fail the exam (see eval_core).
    hmap = eval_core.holdout_map(weights, imgsz, source, pp)
    if hmap:
        print(f"[count_eval] holdout mAP50 {hmap['map50']} "
              f"(mAP50-95 {hmap['map']}) on the same {len(stems)} images")
    tuning = write_eval_json(
        weights.parent.parent, tag, conf=conf, iou=iou, mae=mae, imgsz=imgsz,
        source=source, weights=weights.name, buckets=buckets,
        images=image_records, ties=ties, holdout_map=hmap,
        tuned_on=tune_source or source, grid=grid)
    wrong = [t for t in image_records if t.get("pred") != t.get("gt")]
    print(f"[count_eval] {len(wrong)}/{len(image_records)} images miscounted"
          + (f"; worst off by {max(abs(t['pred'] - t['gt']) for t in wrong)}"
             if wrong else " — clean sweep"))
    if (edge := grid_edge(conf, iou, grid)):
        print("[count_eval] ⚠ WINNER AT GRID EDGE: " + ", ".join(edge)
              + " — this MAE may understate the model; the grid is shared with "
                "the challenger harness so it is deliberately not widened here")
    if ties:
        print(f"[count_eval] {ties} other cell(s) tied at MAE {mae:.4f}; the "
              "winner is the first seen (conf-outer, iou-inner) — that tie-break "
              "is load-bearing for cross-run comparability, not a preference")
    # THE TAGGED FILE IS NOT SERVED, and this line used to say it was. Serving
    # reads exactly `<run>/count_eval.json` (runtime_model.resolve(), which does
    # `weights.parent.parent / "count_eval.json"`), so a `--tag`ged re-score
    # writes count_eval_<tag>.json and changes nothing about what ships. Saying
    # "served automatically" after a tagged run invites the opposite conclusion —
    # that an experimental re-score has just altered the live thresholds — which
    # is precisely the reassurance a non-destructive flag exists to give.
    print(f"[count_eval] wrote tuned conf/iou -> {tuning}"
          + (" (NOT served — tagged runs are non-destructive; serving reads "
             "count_eval.json)" if tag else " (served automatically)"))
    return mae


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-image count-error eval + conf/NMS sweep.")
    ap.add_argument("--weights", type=Path, required=True, help="path to best.pt")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="defaults to the size the run was trained at (args.yaml)")
    ap.add_argument("--source", choices=["val", "testset"], default="val",
                    help="eval on the random val split or the fixed testset holdout")
    ap.add_argument("--tune-source", choices=["val", "testset"], default=None,
                    help="pick conf/iou on THIS source, then score --source once "
                         "at that cell. Without it the sweep chooses and reports "
                         "on the same images, which is biased low.")
    ap.add_argument("--grid", choices=["standard", "wide"], default="standard",
                    help="'wide' extends both axes past the boundaries every run "
                         "has been winning on. Not the default: every certified "
                         "MAE on disk used 'standard'.")
    ap.add_argument("--tag", default="",
                    help="suffix the outputs (count_eval_<tag>.json / eval_preview_<tag>/) "
                         "so a second checkpoint of the same run doesn't clobber the first")
    ap.add_argument("--exam", default=None,
                    help="score a FROZEN exam by name (groundwork.dataset.store.exam) "
                         "instead of whatever the holdout happens to be today. Refuses "
                         "if any frozen image is missing or relabelled — a subset scored "
                         "under the exam's name would look comparable and would not be.")
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    # The exam has to say what it examined. A certified MAE against the wrong
    # holdout looks exactly like a certified MAE against the right one.
    img_dir, lbl_dir = _dirs(args.source, pp)
    n = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    print(f"[count_eval] project {proj.slug} ({proj.name}) | classes "
          f"{list(proj.classes)} | {proj.dataset_root} | source {args.source}: "
          f"{n} images")
    evaluate(args.weights, pp, args.imgsz, args.source, args.tag,
             tune_source=args.tune_source, grid=args.grid, exam=args.exam)


if __name__ == "__main__":
    main()
