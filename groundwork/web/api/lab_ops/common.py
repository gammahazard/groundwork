"""Shared state of the challenger API: run discovery, env tables, busy facts."""

from __future__ import annotations

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



def _alt(p):
    """This project's challenger tree (ProjectPaths.ALT_DIR). The project is
    REQUIRED — there is no default project to fall back to."""
    return paths.for_project(p).ALT_DIR


def _every_alt():
    """EVERY project's challenger tree. A GPU-holder crumb answers a MACHINE
    question — "is a card busy right now" — and a job in another project's tree
    holds the same card. Scoping that one by project would let two projects
    start on the same GPU, which is exactly what the crumb exists to prevent."""
    return [pp.ALT_DIR for pp in paths.every_project()]


router = APIRouter()

# Which interpreter a MODEL needs is registry.py's business (Model.python).
# MAIN_PY stays because it is not a model fact: it is this repo's own venv,
# used below to run a groundwork module as a subprocess.
MAIN_PY = REPO / ".venv" / "bin" / "python"


def _trainable_archs() -> list[str]:
    """Which challenger families this machine can actually train, by venv.

    This is the whole "am I the lab?" test. The challenger venvs are only ever
    installed on the worker, so their presence answers the question directly,
    cheaply, and without shelling out to anything that can hang.

    The champion is excluded: it is not a CHALLENGER, and it trains through the
    cockpit's own retrain path, not through /api/lab/train."""
    return [m.name for m in registry.trainable_here()
            if m.name != "yolov8n"]
GPU_BASE_ENV = {"GW_ALLOW_PARALLEL": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def _card_env(card: int) -> dict:
    """The GPU environment for a sidecar job, pinned to ONE card.

    CUDA_DEVICE_ORDER=PCI_BUS_ID is not optional and must never be dropped —
    CUDA orders by capability by default, so "card N" would silently be a
    different physical GPU than the one the lock file names, and two jobs
    would land on one card believing they are on two. The per-card lock is
    derived from CUDA_VISIBLE_DEVICES downstream (altmodels/gpu.py), so this
    one string decides both the card and the lock.

    WHICH card is the caller's problem, answered by machines.pick_card —
    explicit choice first, else a free card whose venv can actually drive it.
    There is deliberately no default index here: any fixed number is wrong on
    some machine (a single-card box has no card 1; a four-card box deserves a
    real choice).
    """
    return {**GPU_BASE_ENV, "CUDA_VISIBLE_DEVICES": str(int(card))}


# This module's version was the correct one; it now lives in groundwork/procs
# so every caller shares it instead of five near-copies disagreeing.
from ....procs import alive as _pid_alive          # noqa: E402


def held_cards() -> set[int]:
    """Card indices something on THIS machine holds right now.

    Filesystem facts only — never a driver call on a request path. Two
    sources: every sidecar run writes a gpu_holder.json crumb naming its card
    for the whole run, and the cockpit retrain records its card in the state
    file. A crumb whose pid is gone is a crash leftover, not a busy card. A
    RUNNING job with no card recorded is unpinned — it may use any visible
    card — so it marks every measured card busy rather than none of them.
    """
    from ... import machines
    held: set[int] = set()
    unpinned = False
    for alt in _every_alt():
        for crumb in alt.glob("*/gpu_holder.json"):
            try:
                d = json.loads(crumb.read_text())
            except Exception:  # noqa: BLE001 — unreadable crumb is not a card
                continue
            if not _pid_alive(d.get("pid")):
                continue
            if d.get("card") is None:
                unpinned = True
            else:
                held.add(int(d["card"]))
    st = retrain_job.status()
    if st.get("status") == "running":
        if st.get("card") is None:
            unpinned = True
        else:
            held.add(int(st["card"]))
    if unpinned:
        held |= {int(c.get("index", 0))
                 for c in machines.cards("here")} or {0}
    return held


def _local_active() -> dict | None:
    """The alt run holding a GPU on THIS MACHINE, with parsed progress.

    The parsing lives in lab_progress.py; what stays here is the one decision
    that module deliberately does not make — WHICH trees to look in. _every_alt()
    is machine-wide on purpose (see its docstring): a card is held by whichever
    project got there first.
    """
    return lab_progress.local_active(_every_alt())




def _running_seed() -> int:
    """The split seed the RUNNING retrain used — not the default.

    THE SPLIT ON DISK BELONGS TO WHOEVER BUILT IT. Membership is a seeded hash of
    the stem, so `is_current(seed=0)` against a tree built with seed 1 reports
    every swapped image as a difference and calls a perfectly reusable split
    stale. Measured on the worker 2026-08-05, minutes after the seed control shipped:
    "23 image(s) missing from train, 23 no longer in the dataset (on disk 141,
    expected 141)" — the COUNT matching while the membership differs is the
    signature of comparing against the wrong seed, and it refused a DEIM run that
    was perfectly safe to start.

    Introduced by exposing split_seed in the UI without teaching this check about
    it. The seed was always in the retrain's state; nothing was reading it.
    """
    try:
        return int(retrain_job.status().get("split_seed") or 0)
    except (TypeError, ValueError):
        return 0


def _finishing(p) -> list[dict]:
    """Challenger runs that are past training but not yet in the ledger.

    THE HOLE THIS FILLS. `actives` lists runs holding a GPU, so a challenger
    vanishes from every live view the instant training ends — and then spends
    five to ten minutes being SCORED (count_eval over the holdout) and ADOPTED
    (the adopt job, a 5-minute timer on HQ) before it appears anywhere. Asked
    2026-08-05 as "how come I don't see the deim run in the ledger, is it still
    adopting" — which is precisely the question the UI had no way to answer.

    Read from crumbs, not from a process table:

      meta.json         written LAST by every trainer, after the loop returns.
                        Its presence is what makes a run finished rather than
                        merely gone — this repo has already paid for treating a
                        disappeared process as a completed one.
      count_eval.json   written by scoring. Absent means scoring is still to
                        come or in flight.

    LOCALLY, ONLY "scoring" IS REPORTABLE, and that is a real distinction rather
    than a shortcut. /api/lab/runs lists this same directory, so the moment
    count_eval.json exists the run is ALREADY in the ledger here — announcing it
    as "adopting" would be describing a row the reader can see. "Adopting" is
    only true of a run sitting on the WORKER that HQ has not copied yet, which is
    why the proxy path below filters the remote list against what HQ already
    has, and why this returns nothing for a scored local run.

    Deliberately NOT a process check. Scoring is launched detached and the
    honest signal is the artefact it produces, not whether some pid is alive.
    """
    out = []
    try:
        alt = _alt(p)
        # local_actives takes a LIST of trees (_every_alt), not one path — a GPU
        # crumb answers a machine question, so it looks across projects.
        live = {a.get("run") for a in lab_progress.local_actives(_every_alt())}
        for d in sorted(alt.glob("*")):
            if not d.is_dir() or d.name in live:
                continue
            meta = d / "meta.json"
            if not meta.exists():
                continue                       # never finished training
            # Only the recent ones: this directory holds every run ever, and a
            # month-old unscored experiment is not "finishing".
            if time.time() - meta.stat().st_mtime > 3600:
                continue
            if (d / "count_eval.json").exists():
                continue                      # already in this machine's ledger
            # SCORING vs QUEUED, and the difference is the whole point of the
            # row. autoscore refuses to start a score while any card is held —
            # scoring holds one, and two GPU jobs at once have failed 3 of 4
            # trials here — so a finished run can sit for an hour waiting. The
            # UI said "scoring…" throughout, which reads as work in progress and
            # sends you looking for a process that was never started.
            #
            # ITS OWN GATE, imported rather than restated. A second copy of
            # "is the box busy" would eventually disagree with the one that
            # actually decides, and then the label would be confidently wrong.
            out.append({"run": d.name,
                        "phase": "queued" if _score_blocked() else "scoring"})
    except OSError:
        pass
    return out


def _score_blocked() -> str | None:
    """Why autoscore is holding off, or None — asked of autoscore itself.

    Imported inside the function: autoscore imports lab_ops.score, so a
    module-level import here would be circular.
    """
    try:
        from ....ops import autoscore
        return autoscore._busy(_every_alt()) or (
            "a yolo retrain" if autoscore._yolo_running() else None)
    except Exception:  # noqa: BLE001 — a label is never worth a 500
        return None


def _adoptable(p) -> list[str]:
    """Recently-finished, SCORED runs on this machine.

    Meaningless locally — /api/lab/runs already lists them — and the whole point
    remotely: HQ subtracts what it already has, and whatever is left is a run the
    worker has finished and scored that the adopt job has not carried home yet. That is
    the window the Overview could not describe, and the one someone actually
    asks about ("is it still adopting?").

    Bounded by mtime for the same reason as _finishing: this directory holds
    every run ever trained, and a month-old scored run is not in flight.
    """
    out = []
    try:
        for d in sorted(_alt(p).glob("*")):
            if not d.is_dir():
                continue
            meta = d / "meta.json"
            if not meta.exists() or not (d / "count_eval.json").exists():
                continue
            if time.time() - meta.stat().st_mtime > 3600:
                continue
            out.append(d.name)
    except OSError:
        pass
    return out


def _split_ok(p, only_if: bool) -> bool | None:
    """Would a fresh split match what is on disk? None if not asked or unknown.

    ONLY WHEN IT DECIDES SOMETHING. The answer gates exactly one thing — may a
    challenger start while a retrain is running — so when nothing is training it
    is not computed at all. It is only 1.5 ms (a directory listing and a set
    comparison; no image is opened, and it never shells out), but a status
    endpoint that does filesystem work nobody asked for is how polled endpoints
    get expensive one harmless addition at a time.

    None, not False, for both "not asked" and "could not tell". "Stale" and "I
    could not tell" lead to different actions — add the missing images versus go
    and look at the machine — and collapsing them would make an unreadable
    dataset look like routine staleness.
    """
    if not only_if:
        return None
    try:
        from ....dataset import paths as _paths
        from ....dataset.pipeline import split as _split
        fresh, _why = _split.is_current(_paths.for_project(p), val_frac=0.2,
                                        seed=_running_seed())
        return bool(fresh)
    except Exception:  # noqa: BLE001 — a status endpoint never fails over a hint
        return None


