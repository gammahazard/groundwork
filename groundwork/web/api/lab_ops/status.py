"""GET /api/lab/status — the polled challenger status."""

from __future__ import annotations

from .common import (router, MAIN_PY, GPU_BASE_ENV,          # noqa: F401
                     _alt, _every_alt, _trainable_archs,
                     _card_env, _local_active, _running_seed,
                     _finishing, _score_blocked, _adoptable,
                     _split_ok)

import json
import fnmatch
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import resource
import subprocess

from ... import safe_proc
import pathlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ... import retrain_job
from .. import lab_progress
from ..lab import REPO, _NAME
from ..deps import current_project
from ....dataset import paths
from ....dataset.pipeline import split as split_mod
from ....models import registry
from .... import project as project_mod

@router.get("/api/lab/status")
def status(p=Depends(current_project)):
    """Is an alt job holding a GPU (here — or on the Trainer, when asked from
    home), and how far along is it?"""
    # EVERY live run, not the first one. Two challengers can train at once since
    # per-run dataset trees landed, and reporting only the first made the cockpit
    # show a busy card as IDLE — the panel keys jobs by card index, so a run it
    # never hears about simply is not there. `active` stays as the first for
    # callers that only ask "is anything training".
    remote_split = remote_yolo = remote_seed = None   # set on the proxy path
    remote_finishing = None
    actives = lab_progress.local_actives(_every_alt())
    from ....config import is_worker
    if not actives and not is_worker():
        # Nothing local: challenger training normally lives on the Trainer —
        # peek over the network so HQ's tab shows the same live view.
        # Cached + refreshed off-thread (see lab_proxy): inline this cost 3s per
        # poll on a cold network path and dragged the whole HQ cockpit down.
        from ... import lab_proxy
        wk = lab_proxy.first_worker_key()
        rj, age = lab_proxy.get(wk, "/api/lab/status") if wk else (None, 0.0)
        # Prefer the remote's LIST; fall back to its single `active` so an HQ
        # talking to a worker that predates this still shows the one run it knows.
        remote = (rj or {}).get("actives") or (
            [rj["active"]] if (rj or {}).get("active") else [])
        actives = [{**a, "remote": True, "age_s": age} for a in remote]
        # THE REMOTE'S SPLIT, NOT OURS. When this answer is about the Trainer,
        # every fact in it must be the Trainer's — HQ's own split state says
        # nothing about whether a challenger may start over there. Absent on a
        # worker that predates the field, which stays None: unknown, not stale.
        if rj is not None:
            remote_split = rj.get("split_current")
            remote_yolo = rj.get("yolo_retrain")
            remote_seed = rj.get("split_seed")
            # THE REMOTE'S LIST, MINUS WHAT WE ALREADY HAVE. A run the worker has
            # scored is "adopting" only until the adopt job copies it here; once the
            # directory exists locally it is in HQ's ledger and saying otherwise
            # would contradict the table on the same screen.
            here = _alt(p)
            remote_finishing = [
                {**fin, "phase": fin.get("phase") or "adopting"}
                for fin in (rj.get("finishing") or [])
                if fin.get("run") and not (here / fin["run"]).exists()]
            # And the worker's SCORED-but-unadopted runs, which its own _finishing
            # skips because they are in ITS ledger — from HQ they are the ones
            # actually in flight.
            for a in (rj.get("adoptable") or []):
                if a and not (here / a).exists():
                    remote_finishing.append({"run": a, "phase": "adopting"})
    return {"active": actives[0] if actives else None,
            "actives": actives,
            "yolo_retrain": remote_yolo if remote_yolo is not None
                            else retrain_job.status().get("status") == "running",
            # IS THE SPLIT REUSABLE — the exact fact that decides whether a
            # challenger may start while a yolo retrain runs (see train() below).
            # The cockpit could only guess before, so it blocked every case and
            # forbade the one this box has two GPUs for. 1.5 ms measured: a
            # directory listing and a set comparison, no image is opened, which
            # is why it is safe on a polled endpoint. NEVER shells out.
            # Computed ONLY while a retrain is running, because that is the
            # only time it decides anything.
            # Past training, not yet in the ledger — see _finishing().
            "finishing": remote_finishing if remote_finishing is not None
                         else _finishing(p),
            # For a caller that is watching THIS machine from elsewhere: what it
            # has finished and scored recently. See _adoptable().
            "adoptable": _adoptable(p),
            "split_current": remote_split if remote_split is not None
                             else _split_ok(p, retrain_job.status().get("status") == "running"),
            # WHICH SPLIT the running retrain built. Two runs share one split
            # directory, so a challenger asking for a different seed cannot be
            # granted — the UI warns before the press instead of after the 409.
            "split_seed": remote_seed if remote_seed is not None else (
                _running_seed() if retrain_job.status().get("status") == "running"
                else None),
            "venv_ok": _trainable_archs() != [],
            "archs": _trainable_archs(),
            # A challenger venv exists ONLY on the lab, so its presence is the
            # machine test — and unlike the nvidia-smi VRAM probe this replaced,
            # it cannot hang. (That probe wedged in D-state under WSL2 and leaked
            # one unkillable process per minute; 149 of them, load average 171,
            # 2026-07-30. It was also asking the wrong question: the real limit
            # is which card a venv's CUDA build supports — the cu121 challenger
            # venvs cannot use the 5070 Ti at all — not total gigabytes.)
            "train_unlocked": _trainable_archs() != []}


# WHAT A TRAINING RUN DRAWS OF ITSELF, per family. Surfacing what EXISTS rather
# than inventing a uniform mechanism: the frameworks genuinely differ, and
# pretending otherwise means either patching three vendor codebases or showing an
# empty box that looks broken.
#
#   mmdet (rtmdet/yolox/centernet)  DetVisualizationHook writes val predictions
#   yolo                            LiveVizTrainer writes a rolling batch stream,
#                                   and ultralytics writes the augmented mosaics
#   deim / dfine / rfdetr           nothing during training — scalars only
#
# The last row is the honest part: those three emit no images at all, and the UI
# says so instead of rendering a blank panel. Their eval gallery still exists,
# because eval_core.render_previews is shared by every family — that is the view
# that answers "where is it wrong", and it has never been yolo-only.
# NOT AN EARLY RETURN when a run dir exists but is empty. A challenger dir is
# adopted to HQ WITHOUT its training-time artifacts, which stay on the machine
# that trained — so "the dir is here and has no pictures" claimed rtmdet writes
# none when it has 51 of them on the worker.
_NO_PICTURES = ("this model writes no pictures while training — its eval "
                "gallery shows where it is wrong once scored")

_VIS_SOURCES = (
    ("train/*/vis_data/vis_image/*.png", "val predictions during training (mmdet)"),
    ("live/batch_step*.jpg", "live training batches (rolling)"),
    ("train_batch*.jpg", "augmented batches from training start"),
    ("live/*.png", "live snapshots"),
)


