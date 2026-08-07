"""Card cells, default-card choice and batch defaults per family."""

from __future__ import annotations

from .model import router, TrainReq, REMOTE_TIMEOUT  # noqa: F401

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ....ops import mirror as mirror_mod
from .... import project as project_mod
from ....models import registry
from ... import machines as machines_mod
from .. import lab_ops, train as train_api
from ..deps import current_project


def _cells(m, key: str) -> list[dict]:
    """Every card on `key`: whether this model CAN use it, and whether it FITS.

    Two questions, kept apart deliberately — see machines.fits. `ok` is compute
    capability and GATES the run, because a venv with no kernels for the card
    dies at the first launch. `fits` is memory and only STEERS the default,
    because a card that is merely too small runs the model fine and pays a
    speed tax for it.
    """
    out = []
    for c in machines_mod.cards(key):
        ok, why = machines_mod.can_run(key, m.venv, c)
        roomy, tax = machines_mod.fits(c, m.peak_vram_gb)
        out.append({**c, "ok": ok, "why": why, "fits": roomy, "tax": tax})
    return out


def _default_card(m, key: str) -> int | None:
    """The card this model should land on: the first that can run it AND fits it.

    Cards are in PCI order, so "first that works" is the 5070 Ti for yolo and
    the 3090 for a challenger — the documented intended split, arrived at from
    the capability data rather than hardcoded to agree with it.

    FITTING IS PART OF "SHOULD", and it was missing until 2026-08-02. Picking the
    first card that COULD run it put deimv2-n-tv28 — 17.2 GiB — on the 16 GB
    5070 Ti, where WSL2 spilled it into host RAM and it trained at 2.19 s/it
    against 0.72-0.94 for seven runs of the same model on the 24 GB 3090. Three
    times slower on the faster card, no error, nothing to notice it by. It is the
    only family the distinction changes: everything else over 16 GiB is
    sm_90-capped and can_run already keeps it off card 0.
    """
    cells = _cells(m, key)
    for c in cells:
        if c["ok"] and c["fits"]:
            return c["index"]
    # NOTHING FITS — take the first that can run it at all. A slow run beats no
    # run, and this is a real state rather than a hypothetical: a family larger
    # than every card in the box would otherwise be unstartable.
    for c in cells:
        if c["ok"]:
            return c["index"]
    return None


# Whether a machine understands projects, cached. Keyed by machine, short TTL:
# it changes exactly once, when that box is deployed to, and until then this must
# not put a network round trip on /api/train/options, which is a page load.
_PROJECT_AWARE: dict[str, tuple[bool, float]] = {}
_AWARE_TTL = 120.0


def _batch_for(m, req) -> int:
    """The batch this run will ACTUALLY use — one rule, read by two callers.

    THEY DISAGREED, and in the dangerous direction. The VRAM fit check computed
    `8 if imgsz <= 960 else 4` for every family, which is yolo's rule
    (retrain_job.py: `b = batch or (8 if imgsz <= 960 else 4)`); challengers
    launched at `m.default_batch`, which is 8 at every size. So a DEIM run at
    1280 was checked as batch 4 and started as batch 8 — the check UNDER-states
    the memory, on the family with the largest footprint on this fleet (17.2 GiB
    measured, and stochastic to boot). An under-estimate is the bad direction:
    it lets through the run that then spills to host RAM over PCIe and trains
    three times slower without erroring, which is the exact failure the record
    records costing four hours.

    An explicit batch always wins; otherwise yolo halves at >960 and everyone
    else takes their family default (rfdetr's 2 is load-bearing — it asserts
    batch * grad_accum == 16).
    """
    if req.batch:
        return int(req.batch)
    if m.name == "yolov8n":
        return 8 if (req.imgsz or m.default_imgsz) <= 960 else 4
    return m.default_batch


