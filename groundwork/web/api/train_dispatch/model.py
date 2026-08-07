"""The Train request shape + the router this package shares."""

from __future__ import annotations

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


router = APIRouter()

REMOTE_TIMEOUT = 60.0   # a cold network path plus the worker's own split + convert



class TrainReq(BaseModel):
    model: str = "yolov8n"
    machine: str = "here"
    card: int | None = None
    epochs: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    sizes: list[int] | None = None      # yolo only: two sizes back to back
    run_name: str = ""                  # challenger only
    split_seed: int = 0                 # yolo only: train/val membership seed
    # DELIBERATE CONCURRENCY, BY NAME. Two GPU jobs at once has failed 3 of 4
    # trials on the worker — so it is refused unless someone asks for it
    # in the request. It is not a warning and not a default: an experiment worth
    # running is worth typing.
    allow_concurrent: bool = False
    # DELIBERATE SPILL, BY NAME. A card that cannot hold the model still runs it
    # — WSL2 pages into host RAM rather than raising OOM — at a measured ~3x. On
    # a small card it has also faulted the hypervisor. Same shape as
    # allow_concurrent: refused unless someone means it.
    allow_spill: bool = False


