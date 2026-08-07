"""Which trained run the runtime counter serves — pin one, or fall back to newest.

By default the counter serves the NEWEST run (and at that run's own trained imgsz,
read from its args.yaml — so a 1280-trained model is also *served* at 1280). But
after an experiment you may want to pin a specific run, e.g. revert to the last
960 model if a 1280 retrain didn't help. That pin lives in active_model.json.

Pure file/JSON, GPU-free. A change takes effect when the counter is (re)loaded —
i.e. after a bot/webui restart, same as picking up any new weights.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from ..config import OUTPUTS_DIR

# WHICH RUN SERVES IS A PROPERTY OF A PROJECT, and until now it was a property of
# the machine: one `outputs/active_model.json` for everything on the box. With a
# second project that is not merely incomplete, it is WRONG — a bolts project
# would serve another project's weights, and `make_counter()` is what
# /api/count, the upload pre-labeller and the label audit all call.
#
# NOT UNDER dataset_root, deliberately, and this is the constraint that decides
# the layout. `groundwork/mirror.py` excludes only runs/, snapshots/, data.yaml
# and the split dirs — so anything else inside a project's dataset tree is
# rsynced to the worker with --delete. A pin is a per-MACHINE fact (the worker serves
# nothing at all), and mirroring HQ's would be both meaningless there and
# destructive on the way back.
#
# So: keyed by slug, machine-local, outside every dataset tree, and identical in
# shape for the first project and for a project created tomorrow.
SERVING_DIR = OUTPUTS_DIR / "serving"


def _slug(pp) -> str:
    return pp.slug


def _serving(pp):
    """(pin, engine) paths for this project — per-slug under outputs/serving."""
    d = SERVING_DIR / _slug(pp)
    return d / "active_model.json", d / "serving_engine.json"


def _runs(pp):
    """Where the served project's runs live.

    This was `RUNS = OUTPUTS_DIR / "dataset" / "runs"` — a SECOND spelling of
    paths.RUNS_DIR that never imported paths at all, so the "single choke point
    for where data lives" had a hole in it exactly on the serving path. Going
    through ProjectPaths means a project whose dataset_root moves is served from
    the right place instead of silently finding no weights.

    """
    from ..dataset import paths
    return pp.RUNS_DIR


_DEFAULT_IMGSZ = 960


def run_imgsz(run_dir: Path) -> int:
    """The imgsz a run was trained at (from ultralytics' args.yaml)."""
    f = run_dir / "args.yaml"
    if f.exists():
        for ln in f.read_text(errors="ignore").splitlines():
            if ln.strip().startswith("imgsz:"):
                try:
                    return int(float(ln.split(":", 1)[1].strip()))
                except ValueError:
                    break
    return _DEFAULT_IMGSZ


def _in_progress() -> set[str]:
    """Run dirs a retrain is writing RIGHT NOW. ultralytics drops best.pt into
    the new run dir minutes into training, so without this filter the
    half-trained run appears in the Models list (selectable!) and, when no pin
    is set, would even be SERVED as "newest" mid-training. The running
    retrain's pre_runs snapshot (retrain_state.json) tells us which dirs
    predate it; anything newer is still cooking. A stale "running" state whose
    owner process died is ignored (it would hide finished runs forever)."""
    f = OUTPUTS_DIR / "retrain_state.json"
    try:
        d = json.loads(f.read_text())
        if d.get("status") != "running":
            return set()
        pid = d.get("owner_pid")
        if pid:
            try:
                os.kill(int(pid), 0)
            except OSError:
                return set()               # stale state; web status() will heal it
        pre = d.get("pre_runs")
        if pre is not None:
            # THE TREE BEING WRITTEN IS THE TRAINING JOB'S, not this machine's
            # default. `pre_runs` is a snapshot of that same tree, so globbing a
            # different one subtracts two unrelated sets: while a second project
            # trained, every one of the first project's runs read as "still cooking",
            # list_runs() hid all of them, and resolve() found no weights at all
            # — a machine that was serving perfectly would stop, because
            # something else started training.
            from ..dataset import paths
            slug = d.get("project")
            runs = paths.for_project(slug).RUNS_DIR if slug else _runs()
            return {p.name for p in runs.glob("*") if p.is_dir()} - set(pre)
    except Exception:  # noqa: BLE001
        pass
    return set()


def list_runs(pp) -> list[dict]:
    """All COMPLETED runs with weights, newest first: {name, weights, imgsz,
    mtime}. Runs still being trained by the active retrain are excluded."""
    cooking = _in_progress()
    out = []
    for w in sorted(_runs(pp).glob("*/weights/best.pt"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        run = w.parent.parent
        if run.name in cooking:
            continue
        out.append({"name": run.name, "weights": str(w),
                    "imgsz": run_imgsz(run), "mtime": w.stat().st_mtime})
    return out


def yolo_choices(pp, limit: int = 8) -> list[dict]:
    """Completed yolo runs that have been SCORED, best MAE first.

    The symmetric partner of deim_choices, and it exists for the same reason:
    something has to answer "which model could serve?" with the exam attached.
    list_runs() answers by mtime, which is what the cockpit's picker wants (it
    shows everything, scored or not, with a human reading the table). A phone
    button is different — there is no table beside it, so an unscored run behind
    a tap is a model pinned on no evidence.

    UNSCORED RUNS ARE EXCLUDED, not sunk. A run mid-adoption or one whose exam
    failed will reappear here the moment it has a number.
    """
    out = []
    for r in list_runs(pp):
        try:
            j = json.loads((_runs(pp) / r["name"] / "count_eval.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        images = j.get("images") or []
        if j.get("mae") is None:
            continue
        out.append({**r, "mae": j["mae"], "n_images": len(images),
                    "conf": j.get("conf"), "iou": j.get("iou")})
    out.sort(key=lambda d: (d["mae"], -d["mtime"]))
    return out[:limit]


def get_active(pp) -> dict | None:
    """The pinned {weights, imgsz} FOR THIS PROJECT, or None if unpinned/stale."""
    cfg, _ = _serving(pp)
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text())
            if Path(d["weights"]).exists():
                return {"weights": d["weights"], "imgsz": int(d["imgsz"])}
        except Exception:  # noqa: BLE001
            pass
    return None


def set_active(weights: str | Path, imgsz: int, pp) -> None:
    # Atomic: the BOT and the web service are separate processes and both read this. A
    # torn read makes get_active() swallow the error and return None, which
    # silently un-pins the run and serves the newest one instead.
    from ..dataset import sidecar
    cfg, _ = _serving(pp)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_json(cfg, {"weights": str(weights), "imgsz": int(imgsz)})


def clear(pp) -> None:
    """Unpin — go back to serving the newest run."""
    cfg, _ = _serving(pp)
    cfg.unlink(missing_ok=True)


# ---- serving engine (champion vs challenger) --------------------------------
# Which ARCHITECTURE answers a photo. Deliberately a separate file from the
# yolo pin so the two never interfere: switching engines leaves the pinned yolo
# run exactly as it was, and switching back is instant.
ENGINES = ("yolo", "deim")


def deim_choices(pp) -> list[dict]:
    """Every exported DEIM detector, BEST SCORE FIRST, with its MAE.

    Exports are TorchScript and self-contained — no vendor repo or second venv is
    needed to serve one, which is the whole reason this path exists.

    THIS PROJECT's alt tree, via ProjectPaths.ALT_DIR — which resolves to the
    legacy `outputs/alt` for the first project, so nothing moved. It was the literal
    `OUTPUTS_DIR / "alt"`, one of the last places a second project would have
    been handed another project's models to serve.
    """
    from ..dataset import paths
    alt = pp.ALT_DIR
    out = []
    for w in alt.glob("*/train/detector.torchscript.pt"):
        run = w.parent.parent
        mae = n_images = None
        try:
            j = json.loads((run / "count_eval.json").read_text())
            images = j.get("images") or []
            mae, n_images = j.get("mae"), len(images)
        except Exception:  # noqa: BLE001 — an unscored export is still servable
            pass
        out.append({"name": run.name, "weights": str(w), "mae": mae,
                    "n_images": n_images, "mtime": w.stat().st_mtime})
    # UNSCORED SINKS. A run with no MAE is not a result, and it must never
    # outrank one that has been examined just because it is newer.
    out.sort(key=lambda d: (d["mae"] is None, d["mae"] if d["mae"] is not None else 0,
                            -d["mtime"]))
    return out


def deim_weights(pp) -> Path | None:
    """The BEST-SCORING exported DEIM detector, not the newest.

    It was the newest, by mtime, and that is a proxy for "best" which was true
    once and then was not. On 2026-08-02 the only export on HQ was run K —
    MAE 0.0154 on the old 65-image holdout — while the run that had just scored
    0.0 on 68 had no export at all. `/engine deim` would have quietly served the
    worse model, and nothing anywhere would have said so. Ranking by the exam
    removes the whole class: a newer export only wins if it also scored better.
    """
    c = deim_choices(pp)
    return Path(c[0]["weights"]) if c else None


def get_engine(pp) -> str:
    """'yolo' (default) or 'deim', FOR THIS PROJECT. Any unreadable/unknown value
    falls back to yolo — a bad write can never silently serve the challenger."""
    _, eng = _serving(pp)
    try:
        e = json.loads(eng.read_text()).get("engine")
        if e in ENGINES and (e != "deim" or deim_weights(pp)):
            return e
    except Exception:  # noqa: BLE001
        pass
    return "yolo"


def set_engine(engine: str, pp) -> str:
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    if engine == "deim" and deim_weights(pp) is None:
        raise ValueError("no exported DEIM detector for this project")
    from ..dataset import sidecar
    _, eng = _serving(pp)
    eng.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_json(eng, {"engine": engine})   # bot + web both read it
    return engine


def counter_name(counter) -> str:
    """Which RUN a live counter is serving, for UI text.

    The audit compares your labels against whatever is PINNED, not against the
    run you happen to be investigating — so an audit is only interpretable if it
    says which model produced it. Two audits of the same image legitimately
    disagree when the pin moved between them, and without the name that reads
    as the tool being unreliable.

    Derived from the counter's own loaded weights rather than re-reading the
    pin, so it cannot drift from what actually ran.
    """
    w = getattr(counter, "weights", None)
    if not w:
        return "unknown model"
    w = Path(w)
    run = w.parent.parent.name
    # best.pt is the default; anything else is worth naming (last.pt, stg1...).
    return run if w.name == "best.pt" else f"{run} ({w.name})"


def make_counter(pp, **kw):
    """The counter the bot/API should serve with, per THIS PROJECT's switch.

    `pp` travels into the counter so its own resolve() reads the same project's
    pin. Without it a bolts upload with "pre-label with this project's model"
    ticked would run the OBJECT counter over bolts photographs and write the
    result as draft labels — silent wrong-data, on the endpoint that creates
    training data.
    """
    if get_engine(pp) == "deim":
        from ..deim_counter import DeimCounter
        return DeimCounter(pp=pp, **kw)
    from ..yolo_counter import YoloCounter
    return YoloCounter(pp=pp, **kw)


def _tuning(weights: Path) -> tuple[float | None, float | None]:
    """(conf, iou) that count_eval tuned for this run, or (None, None). Written to
    <run>/count_eval.json on every evaluation, so the served thresholds track the
    served weights instead of a stale hardcoded default."""
    f = weights.parent.parent / "count_eval.json"
    if f.exists():
        try:
            d = json.loads(f.read_text())
            return float(d["conf"]), float(d["iou"])
        except Exception:  # noqa: BLE001
            pass
    return None, None


def resolve(pp) -> tuple[Path | None, int, float | None, float | None]:
    """(weights, imgsz, conf, iou) the counter should load: the pin if set, else the
    newest run. conf/iou are the run's count_eval-tuned values, or None (use defaults)."""
    a = get_active(pp)
    if a:
        w = Path(a["weights"])
        c, i = _tuning(w)
        return w, a["imgsz"], c, i
    runs = list_runs(pp)
    if runs:
        w = Path(runs[0]["weights"])
        c, i = _tuning(w)
        return w, runs[0]["imgsz"], c, i
    return None, _DEFAULT_IMGSZ, None, None
