"""Is anything already using a card — asked of the MACHINE, both kinds."""

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


def _busy_on(machine, slug: str) -> tuple[str | None, str | None]:
    """(what is training on `machine`, why we could not tell). Both None = idle.

    THE GATE THAT WAS MISSING, and the one this file's docstring implies exists.
    Launching a CHALLENGER while yolo trains was guarded (lab_ops.train checks the
    split). Launching a YOLO RETRAIN while a challenger trains was guarded by
    nothing at all — so on 2026-08-02 a POST here put a retrain on card 0 beside
    yolox-s on card 1, and yolox-s died 60 minutes later at epoch 292 of 300 with
    `CUDA error: unspecified launch failure`. Two hours of GPU, no meta.json, no
    score. The driver that sent it had its own gate; a gate only one caller
    consults is decoration, so this one lives at the single entry point every
    caller goes through.

    NOT A CACHED READ. lab_proxy answers /api/lab/status instantly from cache and
    returns nothing at all when cold — which is indistinguishable from "idle" and
    is exactly how a cold cache once made night_queue declare a live run finished.
    A one-shot Train press can afford to ask the machine itself.

    UNKNOWN IS NOT IDLE. A machine we cannot interrogate returns a reason, and the
    caller refuses on it — the same fail-safe the rest of this repo uses, because
    "I could not see it" and "there is nothing there" are different answers.
    """
    if machine.local:
        act = lab_ops._local_active()
        return (act.get("run") if act else None), None
    try:
        # ASKED AS A MACHINE QUESTION, NOT A PROJECT ONE. "Is a card busy" is
        # answered from lab_ops._every_alt() — every project's tree — because a
        # GPU crumb belongs to whichever project got there first. Sending the
        # CALLER's slug therefore changed nothing about the answer and could
        # only fail: a project the remote has never heard of 404s, and that 404
        # became "refusing to start a second GPU job on a machine I cannot see"
        # — for a machine that was perfectly reachable.
        #
        # That is what stopped a second person training at all. The sync that
        # would have created their project over there happens LATER in this
        # function, so the check ran before the thing that would have made it
        # pass. Measured 2026-08-05 with a non-admin account and a fresh
        # project. the first project never hit it: it has been mirrored for months.
        from .remote import _remote   # function-local: remote imports busy
        r = _remote(machine, "/api/machine/status", None, None, method="GET",
                    timeout=30.0)
    except HTTPException as e:
        return None, f"could not ask {machine.name} what is training ({e.detail})"
    if not isinstance(r, dict):
        return None, f"{machine.name} gave an unreadable status"
    act = r.get("active") or {}
    return (act.get("run") or None), None


def _yolo_busy_on(machine, slug: str) -> tuple[bool, str | None]:
    """(is a yolo retrain running on `machine`, why we could not tell)."""
    if machine.local:
        from ... import retrain_job
        return retrain_job.status().get("status") == "running", None
    try:
        # Machine-wide too — retrain_job.status() is one state file per box, not
        # per project. Same reasoning as _challenger_busy_on above.
        from .remote import _remote   # function-local: remote imports busy
        r = _remote(machine, "/api/machine/status", None, None, method="GET",
                    timeout=30.0)
    except HTTPException as e:
        return False, f"could not ask {machine.name} about its retrain ({e.detail})"
    if not isinstance(r, dict):
        return False, f"{machine.name} gave an unreadable retrain status"
    return (r.get("retrain") or {}).get("status") == "running", None


