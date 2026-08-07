"""DEIM export, which is REMOTE work — the weights are not on this machine.

WHY THIS IS NOT LIKE THE YOLO PATH. A yolo export is a local function call: the
champion's `best.pt` is adopted here, ultralytics is in the main venv, and the
conversion is a CPU graph rewrite. None of that is true for DEIM:

  - A DEIM run adopted to HQ is a SCORE snapshot — meta.json, count_eval.json,
    run_config.yml, eval_preview/. It carries no .pth at all. Measured
    2026-08-05: zero weights across every deim* directory here.
  - Leg 1 needs the vendor repo (~/DEIMv2) and a DEIM venv, neither of which
    exists on HQ.

So leg 1 runs where the checkpoint lives and the artifact comes back over the
same ssh/rsync data plane mirror.py and adopt_run.py already use. Leg 2 (CoreML)
then runs HERE, because a traced TorchScript module is self-contained — it needs
neither the vendor repo nor that venv, which is exactly why export_deim_onnx
offers a torchscript format at all.

WHICH VENV IS PER-FAMILY, NOT PER-MACHINE. There are two DEIM families and they
do not share an interpreter: `deimv2-n` is cu121 (.venv-deim, kernels stop at
sm_90) and `deimv2-n-tv28` is the torchvision-2.8 build (.venv-deim13) that lifts
that limit for the Blackwell card. registry.for_run already knows which a run
belongs to, so the venv is read from the model rather than guessed — picking the
wrong one is an import error at best and a silently different graph at worst.

THE EXPORT ITSELF IS CPU. export_deim_onnx loads with map_location="cpu" and
traces on a CPU tensor, so this never competes for a card and needs no busy gate
— unlike TensorRT, which compiles for a specific GPU.
"""
from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from . import export_remote

# What leg 1 can produce. CoreML is not here because it is leg 2's job.
REMOTE_FORMATS = {"onnx": ".onnx", "torchscript": ".torchscript.pt"}


def find_host(run: str) -> tuple[str, str]:
    """(ssh_host, remote_root) of a registered machine that HAS this run.

    Asks each registered machine with an ssh host rather than assuming the worker,
    so a second trainer added later works without touching this file. The run
    directory must contain a .pth — a machine holding only the score snapshot is
    not an answer, and picking it would fail three minutes later inside the
    export instead of here.
    """
    tried = []
    for r in export_remote.machines_with_ssh():
        host, root = r.get("ssh_host"), r.get("remote_root") or "~/counter1"
        d = f"{root}/outputs/alt/{run}"
        probe = export_remote.ssh(host, f"ls {shlex.quote(d)}/train/*.pth 2>/dev/null | head -1",
                     timeout=45)
        found = (probe.stdout or "").strip().splitlines()
        if found and found[0]:
            return host, root
        tried.append(f"{r.get('name') or r.get('key')} ({host})")
    raise FileNotFoundError(
        f"no registered machine has weights for {run!r}"
        + (f" — asked {', '.join(tried)}" if tried else
           " — no machine with an ssh host is registered. Add one on the "
           "Machines tab, including its ssh host, so the checkpoint can be "
           "reached."))


def _weights(host: str, root: str, run: str) -> str:
    """The checkpoint to export, PREFERRING what the run itself nominated.

    meta.json carries `best_checkpoint`, written last by the trainer; falling
    back to a glob would quietly export `last.pth` — a different model from the
    one the holdout score describes, and nothing downstream would say so.
    """
    d = f"{root}/outputs/alt/{run}"
    r = export_remote.ssh(host, f"cat {shlex.quote(d)}/meta.json 2>/dev/null", timeout=45)
    best = ""
    try:
        best = (json.loads(r.stdout or "{}") or {}).get("best_checkpoint") or ""
    except Exception:  # noqa: BLE001 — an unreadable meta is a fallback, not a stop
        best = ""
    if best:
        # It may be recorded as a bare name or a full path; normalise to a path
        # on the remote and confirm it is really there before committing to it.
        cand = best if best.startswith("/") else f"{d}/train/{Path(best).name}"
        if (export_remote.ssh(host, f"test -f {shlex.quote(cand)} && echo yes", timeout=45)
                .stdout or "").strip() == "yes":
            return cand
    # Nothing nominated, or it has been moved: take the newest .pth and SAY so
    # at the call site rather than pretending it was chosen.
    r = export_remote.ssh(host, f"ls -t {shlex.quote(d)}/train/*.pth 2>/dev/null | head -1",
             timeout=45)
    got = (r.stdout or "").strip()
    if not got:
        raise FileNotFoundError(f"{run} has no .pth on {host}")
    return got


def _venv(run: str) -> str:
    """The interpreter for this run's FAMILY — see the module docstring."""
    from ..models import registry
    m = registry.for_run(run)
    return (getattr(m, "venv", None) or ".venv-deim") if m else ".venv-deim"


def export(run: str, fmt: str, imgsz: int, dest_dir: Path) -> Path:
    """Export a DEIM run and bring the artifact back. Returns the local path.

    CoreML is the two-leg case: pull a TorchScript first, then convert here.
    """
    want_coreml = fmt == "coreml"
    remote_fmt = "torchscript" if want_coreml else fmt
    if remote_fmt not in REMOTE_FORMATS:
        raise ValueError(f"DEIM cannot export {fmt!r}")

    host, root = find_host(run)
    weights = _weights(host, root, run)
    venv = _venv(run)
    out_name = f"{run}-{imgsz}{REMOTE_FORMATS[remote_fmt]}"
    remote_out = f"{root}/outputs/exports/{out_name}"

    # --no-postprocess ALWAYS. DEIM's postprocessor gathers with fp32 indices,
    # which CoreML rejects outright — and the detector-only graph is what the
    # iPhone app's Swift already expects (it does its own sigmoid, threshold and
    # dedupe). Exporting the postprocessed graph would produce a package that
    # loads and then returns nothing the app can read.
    script = (
        f"cd {shlex.quote(root)} && mkdir -p outputs/exports && "
        f"{shlex.quote(venv)}/bin/python -m altmodels.export_deim_onnx "
        f"--weights {shlex.quote(weights)} --imgsz {int(imgsz)} "
        f"--format {remote_fmt} --no-postprocess "
        f"--out {shlex.quote(remote_out)}")
    r = export_remote.ssh(host, script, timeout=1800)
    if r.returncode != 0:
        tail = ((r.stderr or r.stdout or "").strip().splitlines()
                or ["no output"])[-1]
        raise RuntimeError(f"export failed on {host}: {tail[:400]}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / out_name
    # NO --delete, and no --delete-excluded: this pulls ONE file into a
    # directory that holds other people's artifacts.
    export_remote.pull(host, remote_out, str(local))
    if not local.exists():
        raise RuntimeError(f"exported on {host} but nothing arrived at {local}")
    if not want_coreml:
        return local

    # LEG 2, HERE. The traced module carries its own graph, so this needs
    # neither the vendor repo nor a DEIM venv — only coremltools.
    # ABSOLUTE: altmodels is a top-level package at the repo root, a sibling of
    # groundwork — a relative import would climb out of the package entirely.
    from altmodels import export_deim_coreml  # noqa: PLC0415 — heavy import
    pkg = Path(export_deim_coreml.convert(local, imgsz=imgsz,
                                          name=f"{run}-{imgsz}"))
    # convert() writes to outputs/coreml/, which is where the iPhone path looks
    # for packages. Land a copy in the exports dir too, or the thing just
    # produced is missing from the list of things produced.
    final = dest_dir / pkg.name
    if final.resolve() != pkg.resolve():
        if final.exists():
            shutil.rmtree(final) if final.is_dir() else final.unlink()
        shutil.copytree(pkg, final) if pkg.is_dir() else shutil.copy2(pkg, final)
    # The intermediate TorchScript was a means, not the artifact.
    local.unlink(missing_ok=True)
    return final
