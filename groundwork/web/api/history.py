"""Training-run history endpoints: the ledger, per-run logs, and run notes."""
from __future__ import annotations
import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...dataset import paths
from ...dataset.pipeline import training_history
from ...dataset.paths import ProjectPaths
from .. import lab_guard, run_metrics
from ...models import registry
from ... import project as project_mod
from .deps import current_project, project_paths

router = APIRouter()


class Note(BaseModel):
    note: str


def _safe(run: str) -> str:
    """A run name is a single path segment. FastAPI decodes %2F into the
    param AFTER routing, so without this a caller reaches `LOGS_DIR / run`
    with separators in hand and reads any like-suffixed file on disk."""
    if not run or "/" in run or "\\" in run or ".." in run:
        raise HTTPException(422, "bad run name")
    return run


def _row_in_project(run: str, p, pp) -> dict:
    """The ledger row, refused unless it belongs to the caller's project —
    the ledger and the log dir are machine-wide, so the project dependency
    alone scopes nothing without this lookup."""
    row = next((r for r in runs(p, pp) if r.get("run") == run), None)
    if row is None:
        raise HTTPException(404, f"no run {run!r} in {p.slug}")
    return row


@router.get("/api/runs")
def runs(p=Depends(current_project), pp: ProjectPaths = Depends(project_paths)):
    """All recorded training runs, newest first."""
    # Every row carries `project` (the writer requires it); a row without one
    # is malformed and matches nobody. The comparison is exact — a default here
    # once made every un-tagged row match every project that asked, so opening
    # a second project showed the first one's runs as its own.
    #
    # Deltas are computed on the FULL chronological ledger and filtered after,
    # not the other way round: a run's predecessor is the previous run of its
    # project, and filtering first would still be correct here but reversing
    # first would silently diff against the run AFTER it.
    rows = training_history.with_deltas(training_history.load())
    rows = [r for r in rows if r.get("project") == p.slug][::-1]
    # Rows written before the model column get it resolved on READ, not by
    # rewriting the ledger. Backfilling 27 historical rows would be a migration
    # whose only gain is not doing this — and the ledger is the one file whose
    # contents are the record of what happened, so it is edited as little as
    # possible. New rows carry it; old ones are answered from the same registry
    # that would have written it.
    for r in rows:
        # ABSENT MEANS SEED 0, resolved on READ — the same treatment `model` and
        # `license` get below, and for the same stated reason: the ledger is the
        # record of what happened and is edited as little as possible. This is
        # not a guess. split_seed was hardcoded and unreachable before
        # 2026-08-02, so a row without it demonstrably trained on seed 0.
        r.setdefault("split_seed", 0)
        if not r.get("model"):
            m = registry.by_name(r.get("arch")) or registry.for_run(r.get("run", ""))
            if m:
                r["model"] = m.name
                r.setdefault("license", m.license)
        # The SAME per-run metrics the challenger endpoint reports — worst single
        # image, over/under, signed bias — read from this run's own exam. Without
        # them the champion cannot appear on a cross-model chart beside the
        # challengers, which is the whole point of having one. mtime-cached in
        # run_metrics, because this endpoint is polled.
        if r.get("run"):
            r.update(run_metrics.for_run_dir(pp.RUNS_DIR / r["run"]))
    return rows


@router.get("/api/runs/{run}")
def run_detail(run: str, p=Depends(current_project),
               pp: ProjectPaths = Depends(project_paths)):
    """EVERYTHING KNOWN ABOUT ONE RUN, in one call.

    There was no way to ask about a single run. A caller wanting one row fetched
    the whole ledger and filtered client-side — fine for a browser that wants the
    table anyway, wasteful for a script that wants one number, and impossible for
    a challenger, whose rows live in a different ledger entirely. This answers
    for BOTH families from the same name, the way registry.for_run does.

    It reports what is ON DISK as well as what is recorded, because those come
    apart: a run whose weights were pruned still has a ledger row and a score,
    and "can I export this" is answerable only by looking. Each artifact is a
    plain boolean rather than a path — a path invites a caller to construct URLs
    into the run tree, which is the containment problem all over again.
    """
    _safe(run)

    m = registry.for_run(run)
    family = getattr(m, "name", None)
    # The yolo ledger first, then this project's challenger tree. A run name is
    # a run name; asking the caller which kind it is would be asking them to
    # know something the server can answer.
    row = next((r for r in runs(p, pp) if r.get("run") == run), None)
    d = None
    for root in (pp.RUNS_DIR, pp.ALT_DIR):
        cand = (root / run).resolve()
        if root.resolve() in cand.parents and cand.is_dir():
            d = cand
            break
    if row is None and d is None:
        raise HTTPException(404, f"no run {run!r} in {p.slug}")

    out = {"run": run, "project": p.slug, "family": family,
           "model": getattr(m, "label", None),
           "kind": "yolo" if getattr(m, "predictor", "") == "ultralytics"
                   else (getattr(m, "predictor", None) or "unknown"),
           "ledger": row, "on_disk": d is not None}
    if d is None:
        return out

    def _json(path):
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — absent or unreadable is a fact, not a 500
            return None

    meta = _json(d / "meta.json")
    ce = _json(d / "count_eval.json")
    out["meta"] = meta
    # THE EXAM, SUMMARISED — not the per-image list, which is what /images is for.
    # A summary endpoint that returns 68 rows is a listing endpoint wearing a
    # different name.
    if ce:
        # THE FILE'S OWN FIELD NAMES. I guessed `per_tray[].err` first and it
        # reported 0 misses beside an MAE of 0.0147 over 68 images — which is
        # one object of error and therefore cannot be zero misses. count_eval
        # writes `miscounts` and `worst_image_error` at the top level and the
        # per-image list under `images` ({stem, pred, gt, dupes,
        # bucket}); nothing here needs deriving.
        out["eval"] = {
            "mae": ce.get("mae"), "conf": ce.get("conf"), "iou": ce.get("iou"),
            "n": ce.get("n_images"), "misses": ce.get("miscounts"),
            "worst": ce.get("worst_image_error"),
            "source": ce.get("source"), "imgsz": ce.get("imgsz"),
            # The regime the score was measured under — count_eval records it.
            "dedupe": ce.get("dedupe"),
            # OPERATING-POINT HONESTY, passed through rather than re-derived:
            # `ties` says how many other sweep cells matched the winning MAE
            # (on a saturated holdout the "winner" is a tie-break, not a
            # choice) and `grid_edge` says the optimum may lie outside the
            # sweep. Both are facts a reader needs before trusting conf/iou.
            "ties": ce.get("ties"), "grid_edge": ce.get("grid_edge"),
            "tuned_on": ce.get("tuned_on"),
            # Sum over the per-image list, not a second stored number — one of
            # these is the record and the other would be a chance to disagree.
            "dupes": sum(t.get("dupes", 0) for t in (ce.get("images") or [])),
        }
    out["artifacts"] = {
        "weights": (d / "weights" / "best.pt").exists()
                   or bool(list((d / "train").glob("*.pth"))),
        "count_eval": ce is not None,
        "meta": meta is not None,
        "eval_preview": (d / "eval_preview").is_dir(),
        "snapshot": (d / "snapshot.json").exists(),
    }
    # WHAT IT COULD BECOME. Answered from the same table the Export tab reads,
    # so this cannot drift from what pressing the button would offer.
    from .. import export_job
    fam = "deim" if getattr(m, "predictor", "") == "deim" else \
          ("yolo" if getattr(m, "predictor", "") == "ultralytics" else "")
    out["export_formats"] = [f.key for f in export_job.formats_for(fam)]
    return out


@router.get("/api/runs/{run}/log")
def run_log(run: str, p=Depends(current_project),
            pp: ProjectPaths = Depends(project_paths)):
    _row_in_project(_safe(run), p, pp)
    text = training_history.get_log(run)
    if text is None:
        raise HTTPException(404, "no saved log for this run")
    return {"run": run, "log": text}


@router.get("/api/runs/{run}/curve")
def run_curve(run: str, pp: ProjectPaths = Depends(project_paths)):
    curve = training_history.get_curve(_safe(run), pp)
    if not curve:
        raise HTTPException(404, "no results.csv for this run")
    return {"run": run, "curve": curve}


@router.post("/api/runs/{run}/note")
def set_note(run: str, body: Note, p=Depends(current_project),
             pp: ProjectPaths = Depends(project_paths)):
    # notes belong in HQ's ledger — a note typed on the Trainer would land in
    # its machine-local copy and silently never reach the ledger of record
    lab_guard.no_lab_edits()
    _row_in_project(_safe(run), p, pp)
    return {"ok": training_history.set_note(run, body.note)}


@router.get("/api/runs/{run}/peek")
def run_peek(run: str, pp: ProjectPaths = Depends(project_paths)):
    """Post-mortem 🔬 payload: curves + the training artifacts (augmented
    batches, labels plot, val pairs) ultralytics saved in the run dir."""
    from .. import run_stats
    p = run_stats.peek(_safe(run), pp)
    if p is None:
        raise HTTPException(404, "no such run")
    return p


def _preview_base(run: str, pp) -> str | None:
    """`/outputs/...` URL prefix of this run's eval_preview dir, or None."""
    from ...config import OUTPUTS_DIR
    d = pp.RUNS_DIR / run / "eval_preview"
    if not d.is_dir():
        return None
    try:
        return "/outputs/" + d.relative_to(OUTPUTS_DIR).as_posix()
    except ValueError:
        return None


@router.get("/api/runs/{run}/images")
def run_images(run: str, pp: ProjectPaths = Depends(project_paths)):
    """Per-image holdout results for the eval gallery: pred/gt/dropped/bucket + the
    overlay image path (served via the /outputs static mount). Only runs scored
    after the gallery feature shipped have these."""
    _safe(run)
    ce = pp.RUNS_DIR / run / "count_eval.json"
    if not ce.exists():
        raise HTTPException(404, "no eval for this run")
    data = json.loads(ce.read_text(encoding="utf-8"))
    images = data.get("images")
    if not images:
        raise HTTPException(404, "this run has no saved eval previews (pre-gallery)")
    return {"run": run, "conf": data.get("conf"), "iou": data.get("iou"),
            "source": data.get("source"), "images": images,
            # The BROWSER-FACING base for this run's eval previews, derived
            # from the run's own project tree — the old frontend literal
            # ("/outputs/dataset/runs/…") was one project's layout and 404'd
            # for every other project.
            "preview_base": _preview_base(run, pp)}
