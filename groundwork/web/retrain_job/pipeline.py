"""The pipeline body: stages, checkpoint scoring, the ledger write."""

from __future__ import annotations

from .state import (LOG, STATE, _STAGE, _read, _write, _update,  # noqa: F401
                    _pp, _lock)
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ...config import OUTPUTS_DIR, ROOT
from ...serve.yolo_counter import find_latest_weights
from ...dataset import paths
from ...dataset.pipeline import training_history
from .. import gpu_facts, run_stats

def _run(cmd: list[str], log) -> int:
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def _mae_of(f) -> float | None:
    try:
        return float(json.loads(f.read_text(encoding="utf-8"))["mae"])
    except Exception:  # noqa: BLE001
        return None


def _score_checkpoints(run_dir, imgsz: int, source: str, log,
                       project: str | None = None) -> tuple[float | None, str]:
    """Score BOTH checkpoints on the holdout and keep whichever COUNTS better.

    ultralytics picks best.pt by fitness (90% mAP50-95) on the random val split,
    which is not our objective. Run yolov8n-21 shipped an epoch-71 checkpoint that
    put two boxes on one object at IoU 0.25-0.49 — under NMS's 0.5 both survived, so
    images came back with stacked double dots (MAE 0.68) while that same run's
    last.pt scored 3x better. So we ask the fixed holdout instead.

    The winner BECOMES best.pt, because everything downstream (runtime_model,
    yolo_counter, the Models/Runs UI) globs */weights/best.pt — promoting in place
    means none of them need to change. The displaced checkpoint is kept alongside
    as best_valfit.pt rather than deleted. Returns (mae, which).
    """
    w = run_dir / "weights"

    def score(ckpt: str, tag: str) -> float | None:
        cmd = [sys.executable, "-m", _STAGE["count_eval"], "--weights",
               str(w / ckpt), "--imgsz", str(imgsz), "--source", source]
        if project:
            cmd += ["--project", project]
        if tag:
            cmd += ["--tag", tag]
        if _run(cmd, log):
            raise RuntimeError(f"count_eval ({ckpt}) failed")
        return _mae_of(run_dir / f"count_eval{'_' + tag if tag else ''}.json")

    mae_best = score("best.pt", "")
    if not (w / "last.pt").exists():
        return mae_best, "best"
    mae_last = score("last.pt", "last")
    log.write(f"\n[retrain] holdout MAE — best.pt {mae_best} vs last.pt {mae_last}\n")
    if mae_best is None or mae_last is None or mae_last >= mae_best:
        shutil.rmtree(run_dir / "eval_preview_last", ignore_errors=True)
        log.write("[retrain] keeping best.pt\n")
        return mae_best, "best"
    # last.pt counts better -> promote it into the canonical slots.
    shutil.copy2(w / "best.pt", w / "best_valfit.pt")
    shutil.copy2(w / "last.pt", w / "best.pt")
    (run_dir / "count_eval.json").write_text(
        (run_dir / "count_eval_last.json").read_text(encoding="utf-8"), encoding="utf-8")
    shutil.rmtree(run_dir / "eval_preview", ignore_errors=True)
    (run_dir / "eval_preview_last").rename(run_dir / "eval_preview")
    log.write(f"[retrain] PROMOTED last.pt -> best.pt (MAE {mae_last} < {mae_best}); "
              "the val-fitness pick is kept as best_valfit.pt\n")
    return mae_last, "last"


def run_pipeline(sizes: list[int], val_frac: float,
                 epochs: int | None = None,
                 batch: int | None = None, project: str | None = None,
                 split_seed: int = 0) -> None:
    """Split once, then train+eval each imgsz in `sizes` back-to-back (so a 1280 vs
    960 comparison uses the SAME train/val split). Records each to history and keeps
    a running batch_results list in state for the UI.

    Public because retrain_worker runs it as a detached process — this is the
    whole retrain, not a helper. Never call it in-process from the web service: that is
    exactly the arrangement that let a service restart kill a run."""
    py = sys.executable
    if project is None:
        raise ValueError("run_pipeline needs a project")
    # EVERY stage is told which project, explicitly. A stage subprocess cannot
    # inherit ambient state, and the cost of it guessing is a run that trained
    # the wrong data and said nothing — this repo's most expensive failure.
    proj_arg = ["--project", project]
    # ...and the orchestrator resolves the SAME project for its own bookkeeping.
    # The stages write into pp's tree; everything below that reads a run back out
    # has to look there and not at whatever paths.default() happens to be.
    pp = _pp(project)
    n = len(sizes)
    # cancel() flips status off "running"; the worker honors it at every stage
    # boundary so a cancel can't be outrun by the snapshot/timelapse tail.
    cancelled = lambda: _read().get("status") != "running"  # noqa: E731
    with open(LOG, "w") as log:
        try:
            _update(step="split")
            # THE SPLIT SEED IS A REAL EXPERIMENTAL AXIS, and it was not
            # reachable. split._val_rank ranks by md5(f"{seed}:{stem}"), so at the
            # default 0 every run this repo has ever done saw the IDENTICAL
            # train/val membership. The five champions re-scored on 2026-08-02 all
            # hit 0.0 — but they are replicates over GPU nondeterminism on one
            # fixed split, not over different splits, and nothing sets a training
            # seed either. Varying THIS asks the question those five cannot:
            # does the result survive a different 139/35?
            split_cmd = [py, "-m", _STAGE["split"], "--val-frac", str(val_frac),
                         "--seed", str(split_seed), *proj_arg]
            if _run(split_cmd, log):
                raise RuntimeError("split failed")
            # Score on the fixed holdout if it has images (comparable across runs),
            # else fall back to the random val split.
            has_testset = pp.TESTSET_LABELS.exists() and \
                any(pp.TESTSET_LABELS.glob("*.txt"))
            source = "testset" if has_testset else "val"
            results = []
            for i, imgsz in enumerate(sizes, 1):
                # Explicit batch (Trainer dropdown) wins; else the 8GB-safe
                # defaults that home has always used.
                b = batch or (8 if imgsz <= 960 else 4)
                tag = f" ({i}/{n})" if n > 1 else ""
                _update(step=f"training @{imgsz}{tag} (~few min)", imgsz=imgsz)
                train_cmd = [py, "-m", _STAGE["train"], "--model", "yolov8n",
                             "--imgsz", str(imgsz), "--batch", str(b),
                             *proj_arg]
                if epochs:
                    train_cmd += ["--epochs", str(epochs)]
                if _run(train_cmd, log):
                    raise RuntimeError(f"train @{imgsz} failed")
                if cancelled():
                    log.write("[retrain] cancelled — stopping before eval\n")
                    return
                run_dir = find_latest_weights(pp).parent.parent
                _update(step=f"evaluating @{imgsz}{tag} (on {source}, both checkpoints)")
                mae, ckpt = _score_checkpoints(run_dir, imgsz, source, log,
                                               project)
                results.append({"imgsz": imgsz, "run": run_dir.name,
                                "mae": mae, "ckpt": ckpt})
                # Dataset snapshot: manifest + content-addressed image blobs, so
                # this run's exact train/test data stays diffable and restorable.
                # It runs BEFORE the ledger write because the ledger's image and
                # object counts are read back out of this manifest — it is the
                # record of what the run consumed. Still non-fatal: if it fails,
                # run_facts falls back to counting the split dirs.
                _update(step=f"dataset snapshot @{imgsz}")
                if _run([py, "-m", _STAGE["run_snapshot"],
                         "--run", run_dir.name, *proj_arg], log):
                    log.write("[retrain] dataset snapshot failed (non-fatal)\n")
                _record_history(run_dir.name, mae, source, ckpt, project)
                # This run is complete and recorded — add it to the protected
                # snapshot so cancelling a LATER run in the batch can't delete it.
                done = set(_read().get("pre_runs") or []) | {run_dir.name}
                _update(batch_results=list(results), pre_runs=sorted(done))
                if cancelled():
                    log.write("[retrain] cancelled — skipping extras\n")
                    return
                # "Watch it learn" filmstrip + "where it looks" heatmaps —
                # best-effort (a failure never fails the retrain), GPU is free.
                _update(step=f"probe timelapse @{imgsz}")
                if _run([py, "-m", _STAGE["timelapse"],
                         "--run", run_dir.name, *proj_arg], log):
                    log.write("[retrain] timelapse failed (non-fatal)\n")
                _update(step=f"heatmaps @{imgsz}")
                if _run([py, "-m", _STAGE["heatmap"],
                         "--run", run_dir.name, *proj_arg], log):
                    log.write("[retrain] heatmaps failed (non-fatal)\n")
            last_mae = results[-1]["mae"] if results else None
            _update(status="done", step="", finished=time.time(), mae=last_mae)
        except Exception as e:  # noqa: BLE001
            log.write(f"\n[retrain] ERROR: {e}\n")
            _update(status="error", step=str(e), finished=time.time())


def _record_history(run: str, mae: float | None, source: str,
                    ckpt: str = "best", project: str | None = None) -> None:
    """Append this finished run to the persistent training-history ledger.

    images/objects are deliberately NOT passed: record_run reads them from
    this run's own dataset manifest (training_history.run_facts), which is
    what the split actually fed the trainer. Counting the live raw/ collection
    here — the old behaviour — answered a subtly different question, and
    drifted from the truth the moment an image was added or moved after the
    split."""
    try:
        st = _read()
        dur = (st.get("finished") or time.time()) - (st.get("started") or 0) \
            if st.get("started") else None
        log_text = LOG.read_text(errors="ignore") if LOG.exists() else ""
        vram = re.findall(r"peak VRAM ([0-9.]+) GB", log_text)
        pp = _pp(project)
        training_history.record_run(
            run=run, mae=mae, prev_mae=st.get("prev_mae"), mae_source=source,
            pp=pp, duration_s=dur,
            peak_vram_gb=float(vram[-1]) if vram else None, log_text=log_text,
            ckpt=ckpt,
            # From the state this pipeline wrote at start(), not re-derived: the
            # split on disk may already have been rebuilt by something else by
            # the time a run finishes.
            split_seed=st.get("split_seed") or 0,
            project=project)
    except Exception:  # noqa: BLE001
        pass          # history is best-effort; never fail a retrain over it


def _clamp(s) -> int:
    return 1280 if int(s) >= 1280 else 960          # only the two tested sizes


