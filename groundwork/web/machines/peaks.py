"""Measured peak VRAM per configuration, and the machine-level can_train."""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass

from ...config import OUTPUTS_DIR
from .capacity import cards, can_run, venv_info  # noqa: F401

def measured_peak(model_name: str, imgsz: int | None,
                  batch: int | None) -> float | None:
    """The largest peak VRAM ever RECORDED for this exact configuration, or None.

    WHY NOT registry.peak_vram_gb: that is ONE number per family, taken at its
    default size, and peak scales with resolution and batch. Using it to judge a
    960/batch-4 run refuses a configuration nobody has measured — which is how
    this function came to exist: the owner had actually trained yolo at 960 on an
    8 GB laptop card, and a rule built on the 1280 figure called it impossible.

    The ledger already holds the answer per run: peak_vram_gb beside imgsz and
    batch. The MAX for a configuration is what is safe to plan against, for the
    same reason registry.py records the max — DEIM resamples resolution per batch
    and seven runs of one config spanned 13.0-17.2 GiB.

    None means "never measured at this size", and the caller must treat that as
    unknown rather than as too big.
    """
    try:
        from ...dataset.pipeline import training_history
        from ...models import registry
    except Exception:  # noqa: BLE001
        return None
    peaks = []
    # CHALLENGERS FIRST — they are not in this ledger at all. Their meta.json
    # records no memory field (checked: not one of 87 runs has one), but every
    # family PRINTS its peak into lab.log, which adoption does carry home:
    #
    #   DEIM / D-FINE   "max mem: 17555"      MB
    #   mmengine        "memory: 5029"        MB
    #
    # Read retroactively, this reproduces registry.peak_vram_gb EXACTLY for all
    # six challenger families — 4.9, 17.2, 11.6, 12.7 — which is what makes the
    # parser trustworthy rather than merely plausible. No driver call, no
    # nvidia-smi: a fact read from a file that already exists, which is this
    # repo's standing preference and the reason it has not wedged /dev/dxg here.
    if model_name != "yolov8n":
        peaks += _alt_peaks(model_name, imgsz, batch)
    for r in training_history.load():
        if not r.get("peak_vram_gb"):
            continue
        m = registry.by_name(r.get("arch") or r.get("model") or "") \
            or registry.for_run(r.get("run", ""))
        if not m or m.name != model_name:
            continue
        if imgsz is not None and r.get("imgsz") != imgsz:
            continue
        if batch is not None and r.get("batch") not in (None, batch):
            continue
        peaks.append(float(r["peak_vram_gb"]))
    return max(peaks) if peaks else None


_ALT_RE = __import__("re").compile(r"max mem:\s*([0-9.]+)|memory:\s*([0-9.]+)")


def _alt_peaks(model_name: str, imgsz: int | None,
               batch: int | None) -> list[float]:
    """Peaks in GiB for a challenger family, from each run's lab.log.

    Matched on the run's own meta.json (resolution and batch), so a 1280/batch-4
    request is judged by 1280/batch-4 runs — the whole point of measuring per
    configuration rather than per family.
    """
    import json as _json
    from ...dataset import paths as _paths
    from ...models import registry
    out = []
    for pp in _paths.every_project():
        if not pp.ALT_DIR.exists():
            continue
        for d in pp.ALT_DIR.glob("*/lab.log"):
            run = d.parent
            m = registry.for_run(run.name)
            if not m or m.name != model_name:
                continue
            try:
                meta = _json.loads((run / "meta.json").read_text())
            except Exception:  # noqa: BLE001 — a run with no meta cannot be matched
                continue
            if imgsz is not None and meta.get("resolution") not in (None, imgsz):
                continue
            if batch is not None and meta.get("batch") not in (None, batch):
                continue
            try:
                txt = d.read_text(errors="ignore")
            except OSError:
                continue
            mb = [float(a or b) for a, b in _ALT_RE.findall(txt)]
            if mb:
                out.append(max(mb) / 1024.0)          # MB -> GiB
    return out


def can_train(key: str) -> tuple[bool, str]:
    """Can this machine RUN anything at all? (ok, why not).

    RUN, NOT FIT — and that distinction is the correction. A first version asked
    whether any model FITS, which refused the laptop outright on yolov8n's 10.4
    GiB default-size peak. But the owner has trained yolo at 960 on that card:
    whether a configuration fits is a question about the REQUEST (its imgsz and
    batch), not about the machine, and answering it at machine level refused
    something that demonstrably works.

    So this is the machine-level question only: is there a card here that some
    installed venv can drive. Fit is judged per request, against a measured peak
    for that configuration, and unmeasured is not a refusal.

    DERIVED, not declared. `Machine.trains` was a hand-set flag, and on HQ it was
    False for a reason that was really about one model on one card: yolov8n peaks
    at 10.4 GiB and the laptop has 8 GB. Stated as a machine-level fact it also
    hid the honest per-model answer, and it would have stayed False on a machine
    that was later given a bigger card.

    THE RULE IS `fits`, NOT A VRAM THRESHOLD, and that distinction is the whole
    point. "8 GB and NVIDIA" would enable exactly the configuration config.py
    documents as crashing this box: "once the driver spills into shared system
    RAM the hypervisor faults and the whole machine crashes". Comparing a card
    against a MODEL's measured peak refuses that and allows a genuinely roomy
    card, with no threshold to pick.
    """
    from ...models import registry
    cs = cards(key)
    if not cs:
        return False, ("no cards measured on this machine — probe it, or it "
                       "cannot be given work")
    for m in registry.MODELS:
        if not m.trainer_module:
            continue
        for c in cs:
            ok, _ = can_run(key, m.venv, c)
            if ok:
                return True, ""
    return False, "no installed venv can drive any card on this machine"
