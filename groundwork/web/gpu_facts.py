"""What the GPU is doing, read from the FILESYSTEM — never from nvidia-smi.

WHY THIS MODULE EXISTS. `run_stats.gpu()` shells out to nvidia-smi. Under WSL2
that goes through the paravirtualised /dev/dxg, and while a training job owns
the card the call can wedge in uninterruptible D-state. D-state cannot be
signalled, so `subprocess.run(timeout=...)` does NOT bound it: on timeout the
implementation calls kill() and then communicate(), and communicate() blocks in
wait4() forever on a process that can never die. The calling thread is gone for
the life of the process.

That is fatal on a POLLED endpoint. `/api/retrain` was calling it once per poll
whenever `status == "running"` — that is, precisely when the GPU is busy and the
call is most likely to wedge. FastAPI runs `def` endpoints in anyio's worker
threadpool, which defaults to 40 threads. At an 8s poll interval the pool was
exhausted in about five minutes, after which EVERY sync endpoint on the box
queued forever while async ones (/openapi.json) still answered in 0.5s. Measured
on the worker 2026-08-01 mid-run: 41 threads in do_wait, 40 orphaned nvidia-smi
processes, load average 50. Two spellings of the binary appear among the
orphans because gpu() tried both in a loop — so one poll could leak two.

This is the second time this trap has been sprung (2026-07-30 was 149 stuck
processes, load 171), which is why the answer here is not "a shorter timeout"
but "do not run a subprocess on a polled path at all".

WHERE THE NUMBERS COME FROM INSTEAD. Ultralytics already prints VRAM on every
epoch line of the training log, and the log is a file we are reading anyway for
the tail:

        Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       51/250      7.15G      1.153      0.797     0.9984        265       1280

So "how much VRAM is the run using" costs one regex over text already in hand.
It is also a BETTER number than nvidia-smi's for this purpose: it is what the
training process itself has allocated, not whatever else happens to share the
card.

WHAT IS DELIBERATELY NOT HERE. Utilisation percentage. Nothing writes it to
disk, and it is the one field that cannot be had without asking the driver. A
missing field is honest; a field that can hang the cockpit is not.
"""
from __future__ import annotations

import re

# The epoch line: "  51/250      7.15G   ..." — the size suffix is what marks it
# as the memory column rather than a loss. Ultralytics emits G; accept M too so
# a small model does not silently report nothing.
_MEM = re.compile(r"^\s*(\d+)/(\d+)\s+([\d.]+)\s*([GM])\b")


def from_log(tail: str) -> dict | None:
    """VRAM the training process reports, parsed from the log tail. None if the
    log holds no epoch line yet (startup, scanning, or a non-ultralytics run).

    Reads the LAST match, not the first: the tail spans several epochs and the
    newest line is the current one.
    """
    if not tail:
        return None
    for line in reversed(tail.splitlines()):
        # Ultralytics repaints the progress bar with \r and colour codes; the
        # epoch line arrives glued to escape sequences, so scan each \r chunk.
        for chunk in line.replace("\x1b[K", "").split("\r"):
            m = _MEM.match(chunk)
            if m:
                gb = float(m.group(3)) / (1024 if m.group(4) == "M" else 1)
                return {"vram_used_gb": round(gb, 2),
                        "epoch": int(m.group(1)), "total": int(m.group(2)),
                        "source": "training log"}
    return None


def busy(running: bool, tail: str = "") -> dict:
    """Is this machine's card in use, and by what — entirely from files.

    `running` is the retrain state file's own verdict, which is already
    validated against the owner pid by retrain_job.status(). That makes "busy"
    a fact about a process we can see rather than a reading of the hardware,
    which is the right question anyway: the cockpit wants to know whether it may
    start something, not what the fans are doing.
    """
    out = {"busy": bool(running), "source": "retrain state"}
    if running:
        mem = from_log(tail)
        if mem:
            out.update(vram_used_gb=mem["vram_used_gb"], source=mem["source"])
    return out
