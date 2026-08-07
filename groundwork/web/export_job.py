"""Export a trained run to a deployable format, in the background.

WHY A JOB AND NOT A REQUEST. A CoreML or TensorRT export is minutes, not
milliseconds. Doing it inside the handler parks one of anyio's 40 worker threads
for the duration — the shape that has already made this cockpit serve nothing
sync three times. Same single-flight start/status pattern as audit_job and
la_job, so there is one idea of "a long thing is running" here rather than three.

NEVER TAKES THE GPU. export_coreml.py already sets CUDA_VISIBLE_DEVICES="" with
the note "CPU is all conversion needs — never grab the GPU out from under a
retrain", and every format inherits that here. An export that stalled a training
run to save a few seconds would be a bad trade, and on this fleet two GPU jobs at
once has failed three of four trials.

THE CAVEATS ARE DATA, NOT DOCUMENTATION. Two of these formats have a cost this
repo has already measured, and a dropdown that lists them as peers would be
lying:

  ncnn      4.3x faster on a Pi and ONE OBJECT WORSE (commit 3433d89, "so we are
            not shipping it"). Fine for a preview, not for counting.
  engine    TensorRT builds an engine for the SPECIFIC GPU it ran on. An export
            from the 3090 will not load on the 5070 Ti — it is a build artifact,
            not a portable model.

Both travel with the format so the UI shows them at the moment of choosing.
"""
from __future__ import annotations

import os
import shlex
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import OUTPUTS_DIR

EXPORT_DIR = OUTPUTS_DIR / "exports"


@dataclass(frozen=True)
class Fmt:
    key: str
    label: str
    what: str                 # what it is FOR, in the user's terms
    caveat: str = ""          # a measured cost, or a portability limit
    ext: str = ""             # what lands on disk
    # DOES THE CONVERSION ITSELF NEED A GPU? Almost none do — they are graph
    # rewrites — and running them on CPU is what keeps an export from stealing a
    # card out from under a training run. TensorRT is the exception: it compiles
    # kernels FOR a specific physical card, so it must run on one, which makes it
    # the only format that has to respect the busy gate.
    needs_gpu: bool = False


# YOLO, through ultralytics' own exporter. Every one of these is a single
# `.export(format=...)` call; the list is deliberately what ultralytics actually
# supports rather than everything that exists, because an option that fails at
# the end of a five-minute job is worse than one that is absent.
YOLO_FORMATS = (
    Fmt("onnx", "ONNX",
        "The neutral interchange format. Almost every runtime reads it — "
        "ONNX Runtime, TensorRT, OpenCV, most cloud inference services — so it "
        "is the safe choice when you do not yet know where the model will run, "
        "and the usual first step into another format.",
        ext=".onnx"),
    Fmt("torchscript", "TorchScript",
        "PyTorch with the Python stripped out: the graph and the weights in one "
        "file that a C++ or mobile runtime can load on its own. Pick it to stay "
        "inside the PyTorch world without shipping a Python environment — it is "
        "what the DEIM serving path here already loads.",
        ext=".torchscript"),
    Fmt("coreml", "CoreML",
        "Apple's on-device format, for iPhone, iPad and Mac. It runs on the "
        "Neural Engine, so inference is fast and costs little battery, and the "
        "photo never leaves the phone. This is what the the original project app ships.",
        ext=".mlpackage"),
    Fmt("tflite", "TensorFlow Lite",
        "The Android counterpart — small, on-device, and the format Play "
        "Services' ML stack expects. Also the usual choice for microcontrollers "
        "and other hardware too small for a full runtime.",
        ext=".tflite"),
    Fmt("openvino", "OpenVINO",
        "Intel's CPU runtime. It makes a machine with NO GPU genuinely usable — "
        "an office desktop, a mini PC on a bench — by optimising hard for Intel "
        "chips and their built-in graphics. Pick it when the target has no "
        "NVIDIA card.",
        ext="_openvino_model"),
    Fmt("ncnn", "NCNN",
        "A tiny CPU runtime built for phones and single-board computers like a "
        "Raspberry Pi. No GPU, no big dependencies — it is what you use when "
        "the hardware is genuinely small.",
        caveat="Measured here: 4.3x faster on a Pi and one object worse. Good for "
               "a live preview, not for a count that matters.",
        ext="_ncnn_model"),
    Fmt("engine", "TensorRT",
        "NVIDIA's compiler. It rebuilds the model into kernels tuned for one "
        "exact GPU, which makes it the fastest option on that card — the right "
        "pick for a server or a workstation that will run many images.",
        # Deliberately NOT naming this fleet's cards. Someone else's machine has
        # different ones, and a caveat that says "the 3090" reads as irrelevant
        # to them — the rule is about the pairing, not about these two.
        caveat="Builds for ONE card. TensorRT compiles kernels for the exact GPU "
               "it runs on, so export on the card you will actually run "
               "inference on — an engine built on a different model of card will "
               "not load. It is a build artifact, not a portable model.",
        ext=".engine", needs_gpu=True),
)

# DEIM, through its own two-leg path — see web/export_deim.py for why it is
# remote work. Only the three legs 1 and 2 can actually produce are listed;
# ultralytics' other formats (tflite, openvino, ncnn, TensorRT) have no DEIM
# equivalent in this repo, and listing them would be an option that fails at the
# end of the job rather than one that is absent.
DEIM_FORMATS = (
    Fmt("onnx", "ONNX",
        "The neutral interchange format, straight from the DEIM exporter. "
        "Almost every runtime reads it, so it is the safe choice when you do "
        "not yet know where the model will run.",
        caveat="Exported detector-only, with no postprocessing: DEIM's own "
               "postprocessor gathers with fp32 indices, which CoreML rejects "
               "— so whatever loads this does its own thresholding. DEIM is "
               "NMS-free, so that is less work than it sounds.",
        ext=".onnx"),
    Fmt("torchscript", "TorchScript",
        "The graph and weights in one self-contained file — it needs neither "
        "the DEIM vendor repo nor its venv to load, which is why the serving "
        "path here uses it and why CoreML is built from it.",
        ext=".torchscript.pt"),
    Fmt("coreml", "CoreML",
        "iPhone, iPad and Mac, running on the Neural Engine. Built in two "
        "steps: a TorchScript trace on the machine holding the weights, then "
        "the CoreML conversion here.",
        caveat="The package's interface is fixed by the app "
               "(image input, raw logits + cxcywh boxes) and is verified after "
               "conversion — a mismatch is refused rather than shipped, "
               "because on the phone it presents only as \"no model\".",
        ext=".mlpackage"),
)

# Kept as the message for a family with NO exporter at all — every other
# challenger (rtmdet, yolox, centernet, rfdetr, d-fine) is in that position.
NOT_WIRED = ("export is not wired for this model family — only YOLO and DEIM "
             "have exporters here. YOLO converts locally; DEIM runs on the "
             "machine holding its weights.")


# THE ONLY TWO FAMILIES WITH AN EXPORTER. Written as a lookup rather than
# `DEIM_FORMATS if family == "deim" else YOLO_FORMATS`, which handed YOLO's list
# to EVERY other family — an rtmdet or yolox run was offered TensorRT and would
# have failed inside ultralytics on a checkpoint it cannot read. Anything not
# named here gets nothing, and NOT_WIRED says so.
_FAMILIES = {"yolo": YOLO_FORMATS, "deim": DEIM_FORMATS}


def formats_for(family: str) -> tuple[Fmt, ...]:
    return _FAMILIES.get(family, ())


def label_for_artifact(name: str) -> str:
    """Which format a file on disk IS, from its name.

    THE FILENAME DOES NOT SAY, for half of them. `_openvino_model` and
    `_ncnn_model` are DIRECTORIES with no extension at all, `.mlpackage` and
    `.engine` mean nothing unless you already know, and `.pt` could be a
    checkpoint or a trace. So the list of exports reads as a pile of files whose
    format you have to infer — which is exactly what the badge fixes.

    Derived from Fmt.ext rather than a second table. That field was carried on
    every format from the start and never read; this is what it was for, and
    reusing it means a format added later cannot be labelled wrongly by
    forgetting to update a lookup somewhere else.

    LONGEST SUFFIX WINS: ".torchscript.pt" must beat ".pt", and "_ncnn_model"
    must not be shadowed by a bare extension match.
    """
    best = ""
    for fam in _FAMILIES.values():
        for f in fam:
            if f.ext and name.endswith(f.ext) and len(f.ext) > len(best):
                best, out = f.ext, f.label
    if best:
        return out
    # Not something we produced — say so rather than inventing a format.
    return Path(name).suffix.lstrip(".").upper() or "file"


_state: dict = {"status": "idle", "run": "", "format": "", "path": "",
                "error": "", "started": None, "finished": None}
_lock = threading.Lock()


def status() -> dict:
    return dict(_state)


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def _by_key(family: str, fmt: str) -> Fmt | None:
    return next((f for f in formats_for(family) if f.key == fmt), None)


def _remote_yolo(run_dir: Path, fmt: str, imgsz: int, card: int,
                 machine: dict) -> Path:
    """Build a YOLO artifact ON ANOTHER MACHINE and fetch it back.

    ONLY TENSORRT NEEDS THIS, and it needs it absolutely: an engine is compiled
    for one exact GPU and will not load on a different model of card. So
    "compile for the worker's card" cannot mean anything except "compile on the
    worker". Every other format is portable and converts here.

    The weights go UP rather than being looked for there. The worker has its own
    copy of most runs, but not necessarily this one — a run trained here, or
    adopted and then re-scored, may exist only on HQ. Sending a 6 MB checkpoint
    is cheap and removes the question entirely.
    """
    from . import export_remote
    host = machine["ssh_host"]
    root = machine.get("remote_root") or "~/counter1"
    # REFUSE IF IT IS TRAINING. A TensorRT build is a real GPU job, and two GPU
    # jobs at once has failed three of four trials on this fleet. Unreachable
    # raises out of training_now rather than reading as idle.
    busy = export_remote.training_now(host)
    if busy:
        raise RuntimeError(
            f"{machine.get('name') or host} is running a GPU job ({busy}). A "
            f"TensorRT build compiles on the card, so it would be the second — "
            f"and that pairing has failed three of four trials on this fleet. "
            f"Wait for it to finish.")
    weights = run_dir / "weights" / "best.pt"
    stem = f"{run_dir.name}-{imgsz}"
    remote_pt = f"{root}/outputs/exports/_in/{stem}.pt"
    export_remote.push(str(weights), host, remote_pt)
    # The marker is printed rather than the path guessed: ultralytics decides the
    # filename, and guessing it is how an export "succeeds" and fetches nothing.
    code = ("import sys;from ultralytics import YOLO;"
            "print('EXPORTED:'+str(YOLO(sys.argv[1]).export("
            "format=sys.argv[2], imgsz=int(sys.argv[3]))))")
    script = (f"cd {shlex.quote(root)} && "
              f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={int(card)} "
              f".venv/bin/python -c {shlex.quote(code)} "
              f"{shlex.quote(remote_pt)} {shlex.quote(fmt)} {int(imgsz)}")
    r = export_remote.ssh(host, script, timeout=3600)
    line = next((l for l in (r.stdout or "").splitlines()
                 if l.startswith("EXPORTED:")), None)
    if not line:
        tail = ((r.stderr or r.stdout or "").strip().splitlines()
                or ["no output"])[-1]
        raise RuntimeError(f"export failed on {host}: {tail[:400]}")
    remote_out = line.split("EXPORTED:", 1)[1].strip()
    # Same rule as the local path: name from the FORMAT, because a directory
    # format (openvino, ncnn) has no suffix to borrow. TensorRT is the only
    # format the UI sends here today, but the API accepts a machine for any of
    # them, and an artifact that cannot say what it is is a bug either way.
    spec = _by_key("yolo", fmt)
    dst = EXPORT_DIR / f"{stem}{(spec.ext if spec else '') or Path(remote_out).suffix or ''}"
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    export_remote.pull(host, remote_out, str(dst))
    if not dst.exists():
        raise RuntimeError(f"built on {host} but nothing arrived at {dst}")
    return dst


def _run_export(run_dir: Path, fmt: str, family: str, imgsz: int,
                card: int | None = None, machine: dict | None = None) -> Path:
    """The conversion, IN A SUBPROCESS with its own environment.

    IT USED TO MUTATE os.environ IN THIS PROCESS, and that is a service-wide side
    effect from a background thread: `CUDA_VISIBLE_DEVICES=""` stayed set after
    the export returned, so every later GPU user in the web service — /api/count loading
    a counter, the label audit — would have found no device. For a CoreML export
    that window is minutes, and nothing would have said why counting had gone
    quiet.

    A child process gets its own env by construction, so the pin cannot leak. It
    also means an exporter that segfaults takes itself down rather than the
    cockpit, which is the same reason every training stage is a subprocess.
    """
    spec = _by_key(family, fmt)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if family == "deim":
        # REMOTE, and deliberately not given the env pin below: the DEIM export
        # traces on CPU on another machine, so there is no local card to hide.
        from . import export_deim
        return export_deim.export(run_dir.name, fmt, imgsz, EXPORT_DIR)
    if machine and machine.get("ssh_host"):
        return _remote_yolo(run_dir, fmt, imgsz, card or 0, machine)
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{run_dir.name} has no weights/best.pt")

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    if spec and spec.needs_gpu:
        # WHICH card is the whole point for TensorRT: the engine only loads on
        # the model of GPU it was built on.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(card if card is not None else 0)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""      # CPU: never take a training card

    from ..config import ROOT
    from . import safe_proc
    code = ("import sys,json;from ultralytics import YOLO;"
            "print('EXPORTED:'+str(YOLO(sys.argv[1]).export("
            "format=sys.argv[2], imgsz=int(sys.argv[3]))))")
    r = safe_proc.run([str(ROOT / ".venv" / "bin" / "python"), "-c", code,
                       str(weights), fmt, str(imgsz)],
                      cwd=str(ROOT), env=env, timeout=1800)
    line = next((l for l in (r.stdout or "").splitlines()
                 if l.startswith("EXPORTED:")), None)
    if not line:
        tail = ((r.stderr or r.stdout or "").strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"export produced nothing: {tail[:300]}")
    out = Path(line.split("EXPORTED:", 1)[1].strip())
    if not out.exists():
        raise FileNotFoundError(f"exporter reported {out}, which is not there")
    # NAMED FROM THE FORMAT, not from the output's suffix. Two of these formats
    # are DIRECTORIES — openvino and ncnn — and a directory has no suffix, so
    # `out.suffix` was empty and the artifact landed as a bare "yolov8n-65-960"
    # with nothing saying what it was. Measured on disk 2026-08-05: an OpenVINO
    # export sitting unlabelled beside five others, and the next extensionless
    # format would have overwritten it. Fmt.ext is exactly this name.
    dst = EXPORT_DIR / f"{run_dir.name}-{imgsz}{(spec.ext if spec else '') or out.suffix or ''}"
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    shutil.move(str(out), dst)
    return dst


def start(run_dir: Path, fmt: str, family: str, imgsz: int,
          card: int | None = None, machine: dict | None = None) -> dict:
    """Single-flight. A second export while one runs is refused, not queued.

    Refused rather than queued because an export is a thing someone is waiting
    for and watching: a queue would make the second one look stuck. There is no
    lost work either — press it again when the first finishes.
    """
    spec = _by_key(family, fmt)
    if spec and spec.needs_gpu:
        # THE ONLY FORMAT THAT COMPETES FOR A CARD, so the only one that has to
        # ask. Everything else is a CPU graph rewrite and can run beside a
        # training job all day. Two GPU jobs at once has failed three of four
        # trials on this fleet — an export is never worth risking a run.
        from . import machine_self
        # Only THIS machine's gate; a remote build checks the remote box inside
        # _remote_yolo, where it can name which machine is busy.
        busy = None if (machine and machine.get("ssh_host")) \
            else machine_self._training_now()
        if busy:
            return {"ok": False,
                    "error": f"a GPU job is running here ({busy}). A TensorRT "
                             f"export compiles on the card, so it would be the "
                             f"second GPU job — and that pairing has failed "
                             f"three of four trials on this box. Wait for it, or "
                             f"pick a format that converts on CPU."}
    with _lock:
        if _state["status"] == "running":
            return {"ok": False,
                    "error": f"already exporting {_state['run']} to "
                             f"{_state['format']} — wait for it to finish"}
        _state.update(status="running", run=run_dir.name, format=fmt, path="",
                      error="", started=time.time(), finished=None, card=card)

    def work():
        try:
            p = _run_export(run_dir, fmt, family, imgsz, card, machine)
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) \
                if p.is_dir() else p.stat().st_size
            _set(status="done", path=str(p), finished=time.time(),
                 size_mb=round(size / 1e6, 1))
        except Exception as e:  # noqa: BLE001 — an export failing is a result
            _set(status="error", error=f"{type(e).__name__}: {e}",
                 finished=time.time())

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "run": run_dir.name, "format": fmt}
