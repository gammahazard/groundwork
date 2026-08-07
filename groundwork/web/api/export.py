"""Export a run to a deployable format.

THIN, like every module here: it resolves which run and which family, then hands
off to web/export_job.py. The formats, their caveats and the GPU rule live there.

WHICH FAMILY A RUN BELONGS TO decides which formats it can produce, and that is
`registry.for_run` — the same longest-prefix match everything else uses, so a
run's family is never guessed twice with two answers.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...dataset import paths
from ...dataset.paths import ProjectPaths
from ...models import registry
from .. import export_job
from .deps import project_paths

router = APIRouter()


def _resolve(run: str, pp: ProjectPaths):
    """(run_dir, family, imgsz) for a run in EITHER tree, or 404.

    Looks in the yolo runs dir and this project's challenger tree, because a run
    name is a run name — asking the caller which kind it is would be asking them
    to know something the server can answer.

    THE PATH IS CONFINED to the project's own trees. `run` arrives from the
    request, and without the containment check a name with ../ in it would walk
    out — the same guard labelio.check_stem exists for.
    """
    if not run or "/" in run or "\\" in run or ".." in run:
        raise HTTPException(422, "bad run name")
    for root in (pp.RUNS_DIR, pp.ALT_DIR):
        d = (root / run).resolve()
        if root.resolve() not in d.parents or not d.is_dir():
            continue
        m = registry.for_run(run)
        # NORMALISED to the two export families this module knows, because
        # `predictor` is a longer list ("ultralytics", "deim", "rfdetr",
        # "mmdet"…) and only the exporter split matters here. Written as a
        # comparison against "yolo" first, which "ultralytics" is not — so
        # _imgsz_of silently took the challenger branch for every yolo run and
        # returned its 1280 fallback. Right by luck on a 1280 run, wrong on a
        # 960 one, and that is the export actually shipping to a phone.
        # NORMALISED to the export families, and an UNKNOWN predictor keeps its
        # own name rather than defaulting to yolo — defaulting is how an rtmdet
        # run came to be offered ultralytics' formats. formats_for() returns
        # nothing for a name it does not know, and the panel explains why.
        pred = getattr(m, "predictor", "") if m else ""
        family = {"ultralytics": "yolo", "deim": "deim"}.get(pred, pred or "yolo")
        imgsz = _imgsz_of(d, family)
        return d, family, imgsz
    raise HTTPException(404, f"no run {run!r} in {pp.slug}")


def _imgsz_of(run_dir: Path, family: str) -> int:
    """The size this run was TRAINED at — exporting at any other is a different
    model, and one that will not match the exam it was scored on."""
    if family == "yolo":
        from ...serve import runtime_model
        return runtime_model.run_imgsz(run_dir)
    import json
    try:
        meta = json.loads((run_dir / "meta.json").read_text())
        return int(meta.get("resolution") or meta.get("imgsz") or 1280)
    except Exception:  # noqa: BLE001
        return 1280


@router.get("/api/export/formats")
def formats(run: str, pp: ProjectPaths = Depends(project_paths)):
    """What this run can become, with the cost of each said up front.

    The caveats travel with the format rather than living in a doc, because the
    moment they matter is the moment someone is choosing — NCNN's accuracy cost
    and TensorRT's card-specificity are both things you want before a five-minute
    job, not after.
    """
    run_dir, family, imgsz = _resolve(run, pp)
    fmts = export_job.formats_for(family)
    return {
        "run": run, "family": family, "imgsz": imgsz,
        # EMPTY IS A STATE WITH A REASON, and it travels with the empty list.
        # A family with no exporter wired yet would otherwise render as a blank
        # dropdown, which reads as a bug in the page rather than as work not
        # done. Today that is every non-YOLO family.
        "not_wired": "" if fmts else export_job.NOT_WIRED,
        "formats": [{"key": f.key, "label": f.label, "what": f.what,
                     "caveat": f.caveat, "needs_gpu": f.needs_gpu}
                    for f in fmts],
    }


@router.get("/api/export/cards")
def cards():
    """Every card on every registered machine, for the TensorRT picker.

    ALL MACHINES, not just this one — an engine only loads on the model of GPU
    it was built on, so "which card" is exactly the question, and half the fleet
    lives on the worker. A machine that has never been probed contributes nothing
    and says so rather than appearing as a GPU-less box.

    MEASURED, NOT ENUMERATED. These come from the registry, written by an
    explicit probe. Counting cards calls into the driver, and under WSL2 that
    crosses /dev/dxg where a blocked thread cannot be killed — this fleet has
    been taken down that way twice, so a polled endpoint must never do it.
    """
    from .. import machines
    # all_machines(), NOT public_registry(): the registry holds machines someone
    # ADDED, and this box is not one of them — it is a built-in. Reading the
    # registry alone left the local card out of a picker whose whole job is
    # "which card do I build for", on the machine doing the building.
    out, unprobed = [], []
    for key, m in machines.all_machines().items():
        reg = machines.registered(key)          # local use only — carries a key
        cs = machines.cards(key) or []
        if not cs:
            unprobed.append(m.name or key)
            continue
        for c in cs:
            out.append({
                "machine": key, "machine_name": m.name,
                "local": bool(m.local), "remote": bool(reg.get("ssh_host")),
                "index": c.get("index"), "name": c.get("name"),
                "vram_gb": c.get("vram_gb"), "sm": c.get("sm"),
            })
    return {"cards": out, "unprobed": unprobed}


class ExportReq(BaseModel):
    run: str
    format: str
    # TensorRT only: which physical card to compile for. Ignored by every other
    # format, which convert on CPU.
    card: int | None = None
    # TensorRT only: WHICH MACHINE'S card. An engine is not portable between
    # models of GPU, so building for the worker's 3090 means building on the worker.
    machine: str | None = None


@router.post("/api/export")
def start(req: ExportReq, pp: ProjectPaths = Depends(project_paths)):
    run_dir, family, imgsz = _resolve(req.run, pp)
    spec = export_job._by_key(family, req.format)
    if spec is None:
        choices = [f.key for f in export_job.formats_for(family)]
        # Not "cannot export to 'coreml' — one of ''", which is what an empty
        # family produced: it blamed the format for a family-wide gap.
        raise HTTPException(422, export_job.NOT_WIRED if not choices else
                            f"{family} cannot export to {req.format!r} — "
                            f"one of {', '.join(choices)}")
    target = None
    if req.machine:
        from .. import machines
        # public_registry, NOT registered(): the stored record carries that
        # machine's API key, and this dict is handed to the export job. A
        # credential has no business travelling with a build request.
        m = machines.all_machines().get(req.machine)
        if not m:
            raise HTTPException(404, f"no machine {req.machine!r}")
        if m.local:
            target = None          # this box: the ordinary local path
        else:
            reg = machines.registered(req.machine)
            # ONLY the two fields the job needs. registered() also carries that
            # machine's API key, and a credential has no business travelling
            # with a build request.
            target = {"name": m.name, "ssh_host": reg.get("ssh_host", ""),
                      "remote_root": reg.get("remote_root", "")}
        if target is not None and not target.get("ssh_host"):
            raise HTTPException(
                422, f"{target.get('name') or req.machine} has no ssh host "
                     f"recorded, so weights cannot be sent to it. Add one on "
                     f"the Machines tab.")
    r = export_job.start(run_dir, req.format, family, imgsz, req.card, target)
    if not r.get("ok"):
        # 409: something is already using the thing this needs. Not a 500 — it
        # is a normal race, and the message says what to wait for.
        raise HTTPException(409, r.get("error", "could not start"))
    return {**r, "imgsz": imgsz, "family": family}


@router.get("/api/export")
def status():
    """Progress of the running export, or the last one's result.

    Machine-level, like the label audit: the job is single-flight for this
    process, and "is a conversion running" is not a per-project question.
    """
    return export_job.status()


@router.get("/api/export/artifacts")
def artifacts():
    """What has been exported and is sitting on disk, newest first."""
    d = export_job.EXPORT_DIR
    if not d.exists():
        return {"artifacts": []}
    out = []
    for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) \
            if p.is_dir() else p.stat().st_size
        out.append({"name": p.name, "size_mb": round(size / 1e6, 1),
                    # WHAT IT IS, not just what it is called — openvino and ncnn
                    # are extensionless directories, and .mlpackage/.engine mean
                    # nothing at a glance.
                    "format": export_job.label_for_artifact(p.name),
                    "at": p.stat().st_mtime,
                    # Served through the /outputs mount, which the auth gate
                    # covers — so a download needs a session like everything else.
                    "url": f"/outputs/exports/{p.name}"})
    return {"artifacts": out}
