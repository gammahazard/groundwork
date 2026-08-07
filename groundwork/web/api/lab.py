"""Challenger read side: runs list, per-run detail (curve, misses, previews),
pinned target. Live status/log + launch/score live in api_lab_ops.py.

Everything under outputs/alt/ — never the yolo ledger. The shipped exam schema
is identical to count_eval.json, so rows here compare 1:1 with the champion.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ...config import OUTPUTS_DIR
from ...dataset import paths
from ...models import registry
from .. import run_metrics
from .deps import current_project

router = APIRouter()

# Imported by lab_ops (which launches subprocesses from the repo root) —
# an in-file grep says it is unused, and it is not.
REPO = Path(__file__).resolve().parents[3]
_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")

# Challenger training needs resolutions only a big card can hold (the 8GB
# laptop proved 672px is not enough), so it must run on the worker. That used to be
# decided by shelling out to nvidia-smi for total VRAM. It no longer is, and the
# probe is gone rather than fixed — see api_lab_ops._trainable_archs for what
# replaced it and why (2026-07-30):
#   * WRONG QUESTION. Gigabytes were a proxy for "am I the lab?". The presence of
#     a challenger venv answers that directly, and the genuine card limit is not
#     size at all — the cu121 challenger venvs cannot use the 5070 Ti (sm_120).
#   * DANGEROUS ANSWER. Under WSL2 nvidia-smi goes through the paravirtualised
#     /dev/dxg and can wedge in uninterruptible D-state while a GPU trains.
#     subprocess timeouts do not help: they escalate to SIGKILL, which cannot
#     touch a D-state process. A bare 60s retry therefore leaked one unkillable
#     process per minute — found at 149 of them, load average 171, on the lab.
# The lesson generalises past this function: on a polled endpoint, prefer a fact
# you can read from the filesystem over a fact you have to ask a driver for.


def _runs(alt: Path) -> list[Path]:
    return [p.parent for p in alt.glob("*/count_eval.json")] + \
           [p.parent for p in alt.glob("*/meta.json")
            if not (p.parent / "count_eval.json").exists()]


def _pinned_eval(pp=None) -> dict | None:
    try:
        from ...serve import runtime_model
        active = runtime_model.get_active(pp)
        if active:
            f = Path(active["weights"]).parent.parent / "count_eval.json"
            return json.loads(f.read_text()) if f.exists() else None
    except Exception:  # noqa: BLE001
        pass
    return None


def _media(pp) -> str | None:
    """URL prefix for this project's run tree, or None if it is not servable.

    `dataset_root` is allowed to be an absolute path anywhere (project.py), and
    only what lives under the /outputs StaticFiles mount can be fetched. Derive
    the prefix FROM the mount rather than from the repo root, and answer None
    rather than a URL that 404s — the caller must be able to say "no previews"
    instead of quietly falling back to a different project's images.
    """
    try:
        return "/outputs/" + pp.ALT_DIR.relative_to(OUTPUTS_DIR).as_posix()
    except ValueError:
        return None


@router.get("/api/lab/runs")
def runs(p=Depends(current_project)):
    """This project's challenger runs.

    Project-scoped since 2026-07-31 via ProjectPaths.ALT_DIR, which resolves to
    the legacy `outputs/alt` for the first project and `<project>/alt` for anything else
    — so nothing moved and a second project simply has its own (empty) tree.

    Before that, `ALT` was a module constant and this endpoint had to return []
    for any project but the default: handing a second project the first project's 54
    runs, under its own name, with MAEs measured on a holdout of objects, is the
    silent-wrong-data failure this repo cares most about. The refusal was honest;
    this is the fix it was standing in for.
    """
    pp = paths.for_project(p)
    pinned = _pinned_eval(pp)
    out, seen = [], set()
    for d in _runs(pp.ALT_DIR):
        if d.name in seen or d.name == "datasets":
            continue
        seen.add(d.name)
        row = {"run": d.name}
        for f, keys in (("meta.json", ("arch", "license", "resolution",
                                       "num_queries", "epochs",
                                       "train_seconds", "created")),
                        ("count_eval.json", ("mae", "conf", "iou", "imgsz"))):
            try:
                j = json.loads((d / f).read_text())
                row.update({k: j.get(k) for k in keys})
                if f == "count_eval.json":
                    # The arithmetic lives in web/run_metrics.py, because the
                    # champion's endpoint needs the same numbers and a second
                    # copy is how this repo grew eight duplicated rules.
                    row.update(run_metrics.for_run_dir(d))
            except (OSError, ValueError, KeyError):
                # NARROWED from `except Exception`. A missing or unparseable
                # meta.json/count_eval.json is normal and must not take out the
                # listing — but the broad form also swallowed a NameError when
                # run_metrics was referenced without being imported, so every
                # challenger silently lost its derived metrics while the endpoint
                # returned 200. A typo is not a data condition.
                pass
        if not row.get("arch"):
            # Older exams overwrote the trainer meta (arch lost) and -last/-epN
            # exam dirs never had one — the run name still knows its family.
            # The prefix table that used to live here needed its entries in a
            # careful order ("yolox-tiny" before "yolox-s" or the short one wins)
            # and said so in a comment; registry.for_run matches longest-prefix
            # first, so that constraint is a property of the data now and cannot
            # be broken by an edit. Verified identical over all 54 runs on disk
            # by scratch/check_model_registry.py.
            if (m := registry.for_run(d.name)) is not None:
                row["arch"] = m.name
                row["license"] = row.get("license") or m.license
        out.append(row)
    out.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return {"runs": out,
            # Where this project's previews are served from. The browser used to
            # build `/outputs/alt/<run>/...` itself, which is only correct for
            # the first project — every other project's runs live elsewhere.
            "media": _media(pp),
            "pinned": {"mae": pinned["mae"], "conf": pinned["conf"],
                       "iou": pinned["iou"]} if pinned else None}


def _curve(d: Path) -> list[dict]:
    """Per-epoch val mAP50-95 — from lightning's metrics.csv (rfdetr) or
    mmengine's scalars.json (rtmdet), whichever the run has."""
    out = []
    mfile = d / "train" / "metrics.csv"
    if mfile.exists():
        for r in csv.DictReader(open(mfile)):
            v = r.get("val/ema_mAP_50_95")
            if v:
                out.append({"epoch": int(r["epoch"]), "map": float(v)})
        return out
    for scal in sorted(d.glob("train/*/vis_data/scalars.json"))[-1:]:
        for ln in scal.read_text(errors="ignore").splitlines():
            try:
                j = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if "coco/bbox_mAP" in j:
                out.append({"epoch": int(j.get("step", 0)),
                            "map": float(j["coco/bbox_mAP"])})
    if out:
        return out
    lg = d / "train" / "log.txt"                     # DEIMv2: json-lines/epoch
    if lg.exists():
        for ln in lg.read_text(errors="ignore").splitlines():
            try:
                j = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            ev = j.get("test_coco_eval_bbox")
            if ev and "epoch" in j:
                out.append({"epoch": int(j["epoch"]), "map": float(ev[0])})
    if out:
        return out
    lj = d / "train" / "log.json"                     # D-FINE: json array/epoch
    if lj.exists():
        try:
            for r in json.loads(lj.read_text()):
                # NOTE this is a LOSS (down is good), unlike every other curve
                # here — detail() flags it so the UI can title it honestly
                out.append({"epoch": int(r["epoch"]), "map": float(r["val_loss"]),
                            "metric": "val loss"})
        except Exception:  # noqa: BLE001
            pass
    return out


@router.get("/api/lab/runs/{run}/detail")
def detail(run: str, p=Depends(current_project)):
    if not _NAME.match(run):
        raise HTTPException(422, "bad name")
    alt = paths.for_project(p).ALT_DIR
    d = (alt / run).resolve()
    if alt.resolve() not in d.parents or not d.exists():
        raise HTTPException(404, "no such alt run")
    out = {"run": run}
    for f in ("meta.json", "count_eval.json", "diff_vs_ref.json"):
        try:
            out[f.split(".")[0]] = json.loads((d / f).read_text())
        except Exception:  # noqa: BLE001
            pass
    out["curve"] = _curve(d)
    from ...config import is_worker
    if not out["curve"] and not is_worker():
        # train/ artifacts stay on the Trainer — borrow its curve over the network,
        # THROUGH lab_proxy. This one is opened by a click rather than a timer, so
        # it is less dangerous than /api/lab/vis was, but the rule is easier to
        # keep with no exceptions: no GET handler blocks on another machine.
        # Caching also suits it — a finished run's curve never changes, and a
        # live one is 8s-stale at worst.
        from .. import lab_proxy
        wk = lab_proxy.first_worker_key()
        rj, _age = lab_proxy.get(wk, f"/api/lab/runs/{run}/detail") \
            if wk else (None, 0.0)
        out["curve"] = (rj or {}).get("curve") or []
    # curves are val mAP (up = good) everywhere except D-FINE, which logs a
    # loss — say which, so the chart never implies the wrong direction
    out["curve_metric"] = (out["curve"][0].get("metric") if out["curve"] else None) \
        or "val score"
    out["previews"] = sorted(p.name for p in (d / "eval_preview").glob("*.png")) \
        if (d / "eval_preview").exists() else []
    return out
