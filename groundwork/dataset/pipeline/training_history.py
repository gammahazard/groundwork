"""Persistent training-run history — one appended record per completed retrain.

Each retrain overwrites the shared retrain.log/state, so without this the past is
lost. Here we keep an append-only ledger (training_history.json) plus a per-run
copy of the training log, and we can reconstruct past runs from what ultralytics
already saved in each run dir (args.yaml, results.csv). GPU-free, pure file I/O.

A record:
  {run, finished, imgsz, batch, epochs_trained, epochs_requested,
   model, license, map50, map50_holdout, mae, prev_mae, mae_source,
   images, objects, duration_s, ckpt, note}

`ckpt` is which checkpoint the holdout picked ("best" = ultralytics' val-fitness
pick, "last" = the final epoch, promoted over it because it counted better).
"""
from __future__ import annotations
import json
from pathlib import Path

from ...config import MACHINE, OUTPUTS_DIR
from .. import paths, sidecar
from . import run_snapshot

# ONE ledger for every project, not one per project — and the row carries which
# project it belongs to, exactly as Phase 2 gives it a `model` column.
#
# Every row carries `project` — the writer requires it, so a reader never
# guesses. A row without one is malformed and is skipped, not adopted.
#
# Why not per-project files: a run's artifacts already live under its project's
# dataset_root, so the ledger is an INDEX, and an index over everything is the
# more useful object. The Machines area, lifetime GPU hours and "what has this
# box been doing" are all inherently cross-project questions; splitting the file
# would mean re-joining it at every one of them. The cost is one column and a
# one-line backfill.
HISTORY = OUTPUTS_DIR / "training_history.json"
LOGS_DIR = OUTPUTS_DIR / "training_logs"       # per-run copies of the training log


def run_facts(run: str, pp) -> dict:
    """What a run ACTUALLY trained on, read from that run's own artifacts.

    One rule, one home. Each run's dataset manifest is written beside its
    weights and travels with the run, so this answers "how many images, how
    many objects" identically on the machine that trained it and on the
    machine that adopted it.

    It exists because the places that used to answer this each answered
    differently on the same run. Deriving provenance from the LIVE dataset is
    the mistake — the dataset moves on, the run does not.

    Falls back to the live split dirs for runs written before snapshots existed;
    `source` says which happened, so a caller never has to guess.
    """
    m = run_snapshot.load(run, pp)
    if m is not None:
        cols = m.get("collections", {})
        train, val = cols.get("train", {}), cols.get("val", {})
        objects = sum(len([ln for ln in e.get("labels", "").splitlines() if ln.strip()])
                      for e in (*train.values(), *val.values()))
        return {"images": len(train) + len(val), "objects": objects,
                "train": len(train), "val": len(val),
                "test_images": len(cols.get("testset", {})), "source": "manifest"}
    train = sorted((pp.LABELS_DIR / "train").glob("*.txt"))
    val = sorted((pp.LABELS_DIR / "val").glob("*.txt"))
    objects = sum(len([ln for ln in p.read_text(errors="ignore").splitlines() if ln.strip()])
                  for p in (*train, *val))
    return {"images": len(train) + len(val), "objects": objects,
            "train": len(train), "val": len(val),
            "test_images": len(list(pp.TESTSET_LABELS.glob("*.txt")))
            if pp.TESTSET_LABELS.exists() else 0,
            "source": "live"}


def with_deltas(rows: list[dict]) -> list[dict]:
    """Annotate each row with how the DATASET moved since the run before it.

    The holdout grows rather than forking into a second tier (owner's call), so
    a run's MAE is not strictly comparable to the one before it. Rather than
    pretend otherwise, every row states the movement: `delta` is a signed pair
    like "+45/0" — test images, then training images. A 0.0 beside "+45/0" reads
    as "scored on a bigger exam", which a bare 0.0 cannot say.

    Costs nothing new: `images` and `test_images` are on every row.

    `delta_trusted` is the honest caveat: a delta computed from a placeholder
    is a placeholder. Rows carrying `images_source` came from the run's own
    manifest and are trustworthy; anything else is flagged so a UI can grey it
    instead of asserting it.

    Input must be CHRONOLOGICAL (as `load()` returns it). Deltas are computed
    within a project — a run of a different project is not the predecessor of
    this one.
    """
    prev: dict[str, dict] = {}          # project slug -> previous row
    out = []
    for r in rows:
        r = dict(r)
        slug = r.get("project") or "?"
        p = prev.get(slug)
        t, te = r.get("images"), r.get("test_images")
        if p and t is not None and te is not None \
                and p.get("images") is not None and p.get("test_images") is not None:
            dte, dt = te - p["test_images"], t - p["images"]
            r["delta_test"], r["delta_train"] = dte, dt
            r["delta"] = f"{dte:+d}/{dt:+d}"
            # DOTS, not just images. A delta that says "nothing changed" when
            # the labels changed is the exact misleading-provenance failure
            # this exists to prevent, so it also tracks the object count,
            # which the ledger already stores.
            #
            # Honest limit: net dots hides offsetting edits, and says nothing
            # about images whose COUNT was unchanged but whose dots moved.
            # Only a manifest-vs-manifest comparison sees those (the snapshots
            # hold every label's text) — surfaced per-run by
            # /api/runs/{run}/snapshot rather than computed for every row here.
            # trustworthy only if BOTH ends of the subtraction are real
            r["delta_trusted"] = bool(r.get("images_source") and p.get("images_source"))
            # DOTS only when both ends are real: a `objects: 0` placeholder
            # is not a measurement, and subtracting from one produces a number
            # so wrong it discredits the honest ones beside it. Do not do
            # arithmetic on placeholders.
            pa, pb = p.get("objects"), r.get("objects")
            if r["delta_trusted"] and pa and pb:
                r["delta_objects"] = pb - pa
                if pb != pa:
                    r["delta"] += f" · {pb - pa:+d} dots"
            else:
                r["delta_objects"] = None
        else:
            r["delta_test"] = r["delta_train"] = r["delta"] = None
            r["delta_objects"] = None
            r["delta_trusted"] = None       # no predecessor: not untrusted, absent
        if t is not None:
            prev[slug] = r
        out.append(r)
    return out


def load() -> list[dict]:
    if HISTORY.exists():
        try:
            return json.loads(HISTORY.read_text())
        except Exception:  # noqa: BLE001
            pass
    return []


def _save(records: list[dict]) -> None:
    # the web service, the bot, the detached retrain pipeline and the adopt timer all
    # touch the ledger from separate processes. This used to tmp+replace through
    # a SHARED tmp name, which is not atomic between processes: one writer's
    # partial bytes could land inside the live ledger after another replaced it.
    # sidecar.write_json scopes the tmp to our pid.
    sidecar.write_json(HISTORY, records, indent=1)


def _run_meta(run_dir: Path) -> dict:
    """imgsz/batch/epochs from args.yaml + epochs_trained/map50 from results.csv."""
    meta = {"imgsz": None, "batch": None, "epochs_requested": None,
            "epochs_trained": None, "map50": None, "train_time_s": None}
    args = run_dir / "args.yaml"
    if args.exists():
        for ln in args.read_text(errors="ignore").splitlines():
            for key, field in (("imgsz:", "imgsz"), ("batch:", "batch"),
                               ("epochs:", "epochs_requested")):
                if ln.strip().startswith(key):
                    try:
                        meta[field] = int(float(ln.split(":", 1)[1].strip()))
                    except ValueError:
                        pass
    csv = run_dir / "results.csv"
    if csv.exists():
        rows = [r for r in csv.read_text(errors="ignore").splitlines() if r.strip()]
        if len(rows) >= 2:
            head = [h.strip() for h in rows[0].split(",")]
            meta["epochs_trained"] = len(rows) - 1
            if "time" in head:                       # cumulative training seconds
                col = head.index("time")
                try:
                    meta["train_time_s"] = round(float(rows[-1].split(",")[col]), 1)
                except (ValueError, IndexError):
                    pass
            if "metrics/mAP50(B)" in head:
                col = head.index("metrics/mAP50(B)")
                vals = []
                for r in rows[1:]:
                    try:
                        vals.append(float(r.split(",")[col]))
                    except (ValueError, IndexError):
                        pass
                if vals:
                    meta["map50"] = round(max(vals), 4)
    return meta


def record_run(run: str, mae: float | None, prev_mae: float | None,
               mae_source: str, pp, project: str,
               images: int | None = None, objects: int | None = None,
               duration_s: float | None = None, peak_vram_gb: float | None = None,
               log_text: str | None = None, ckpt: str = "best",
               machine: str | None = None, split_seed: int = 0) -> dict:
    """Append a finished run to the ledger (+ save its log). Returns the record.

    images/objects default to run_facts(run) — the run's own manifest. Pass
    them only when a better source exists: adoption passes the numbers the
    WORKER recorded, since that is the machine that actually ran the training.

    `project` is the slug this run belongs to. It is a COLUMN rather than a
    separate ledger file for the reason given at the top of this module, and it
    is recorded for the same reason `machine` is: a number in a shared table has
    to say what it is a number about.
    """
    run_dir = pp.RUNS_DIR / run
    facts = run_facts(run, pp)
    if images is None:
        images = facts["images"]
    if objects is None:
        objects = facts["objects"]
    # Size of the frozen holdout this run was scored against — the context that
    # makes MAE trustworthy (a low MAE on a tiny testset means little).
    test_images = facts["test_images"]
    # Per-bucket MAE breakdown (eval lens), and the HOLDOUT mAP, if count_eval
    # wrote them for this run. `map50_holdout` is deliberately a separate column
    # from `map50`: the latter comes from results.csv and is a VAL-SPLIT number,
    # and collapsing the two would erase exactly the distinction that makes one
    # of them trustworthy. Absent on every run scored before 2026-07-31 — which
    # is honest, not a gap to fill in with the val figure.
    buckets = map50_holdout = ties = grid_edge = None
    ce = run_dir / "count_eval.json"
    if ce.exists():
        try:
            j = json.loads(ce.read_text(encoding="utf-8"))
            buckets = j.get("buckets")
            hm = j.get("holdout_map") or {}
            map50_holdout = hm.get("map50")
            ties, grid_edge = j.get("ties"), j.get("grid_edge")
        except Exception:  # noqa: BLE001
            pass
    # WHICH MODEL this run is of, resolved once at write time from the registry.
    # It was derived in the BROWSER by stripping a trailing -N off the run name
    # (history.js), which works only while every run is named <model>-<n> and
    # silently mislabels anything else — a re-scored checkpoint, a hand-named
    # experiment, a second family that does not follow the convention. The row
    # is the record; it should say what it is rather than leave a reader to
    # parse it back out of a string.
    from ...models import registry as _reg
    _m = _reg.for_run(run)
    rec = {"run": run, "project": project, "model": _m.name if _m else None,
           "license": _m.license if _m else None,
           "finished": __import__("time").time(),
           **_run_meta(run_dir), "mae": mae, "prev_mae": prev_mae,
           "mae_source": mae_source, "images": images, "objects": objects,
           "train_images": facts["train"], "val_images": facts["val"],
           "images_source": facts["source"],
           "test_images": test_images, "buckets": buckets,
           # Operating-point honesty, carried onto the row so the table can
           # flag a tie-break without opening every run's eval file.
           "ties": ties, "grid_edge": grid_edge,
           "map50_holdout": map50_holdout,
           "duration_s": round(duration_s, 1) if duration_s else None,
           "peak_vram_gb": peak_vram_gb, "ckpt": ckpt,
           "machine": machine or MACHINE,
           # WHICH TRAIN/VAL SPLIT this run saw. Every row before 2026-08-02 is
           # seed 0 and that is a fact rather than a backfill guess: the parameter
           # was hardcoded and unreachable until then, so there was nothing to
           # record. Now that it varies, two rows can differ in a way no other
           # column explains — and this table is the thing every comparison reads.
           # A run trained on a 66%-different validation set is not the same
           # experiment as one that was not, however identical the rest looks.
           "split_seed": int(split_seed or 0),
           "note": ""}
    records = load()
    records = [r for r in records if r.get("run") != run]   # replace if re-run
    records.append(rec)
    _save(records)
    if log_text is not None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (LOGS_DIR / f"{run}.log").write_text(log_text, encoding="utf-8")
    return rec


def get_curve(run: str, pp) -> list[dict]:
    """Per-epoch [{epoch, map50}] from the run's results.csv (for the drill-down)."""
    csv = pp.RUNS_DIR / run / "results.csv"
    if not csv.exists():
        return []
    rows = [r for r in csv.read_text(errors="ignore").splitlines() if r.strip()]
    if len(rows) < 2:
        return []
    head = [h.strip() for h in rows[0].split(",")]
    ce = head.index("epoch") if "epoch" in head else None
    cm = head.index("metrics/mAP50(B)") if "metrics/mAP50(B)" in head else None
    out = []
    for r in rows[1:]:
        parts = r.split(",")
        try:
            out.append({"epoch": int(float(parts[ce])) if ce is not None else len(out) + 1,
                        "map50": float(parts[cm]) if cm is not None else None})
        except (ValueError, IndexError):
            pass
    return out


def get_log(run: str) -> str | None:
    f = LOGS_DIR / f"{run}.log"
    return f.read_text(errors="ignore") if f.exists() else None


def set_note(run: str, note: str) -> bool:
    records = load()
    for r in records:
        if r.get("run") == run:
            r["note"] = note
            _save(records)
            return True
    return False


def backfill(pp, project: str) -> int:
    """Add ledger stubs for existing runs not yet recorded (MAE unknown for old
    runs — their logs were overwritten). Returns how many were added."""
    known = {r.get("run") for r in load()}
    added = 0
    records = load()
    for w in sorted(pp.RUNS_DIR.glob("*/weights/best.pt"),
                    key=lambda p: p.stat().st_mtime):
        run = w.parent.parent.name
        if run in known:
            continue
        records.append({"run": run, "project": project,
                        "finished": w.stat().st_mtime,
                        **_run_meta(w.parent.parent), "mae": None, "prev_mae": None,
                        "mae_source": None, "images": None, "objects": None,
                        "duration_s": None, "note": "backfilled (pre-history)"})
        added += 1
    if added:
        _save(records)
    return added
