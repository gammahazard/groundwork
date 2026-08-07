"""Server-side curation lock for the lab machine.

The lab's dataset is a one-way mirror FROM home: any edit made on the lab is
silently overwritten by the next sync tick. Hiding the editing tabs in the UI
was the only protection — this makes the API itself refuse, so no UI gap,
stale browser, or direct call can ever curate on the wrong machine
(2026-07-26: the mirror-exclude bug showed how quietly divergence happens).
"""
from __future__ import annotations
import os

from fastapi import HTTPException


def no_lab_edits() -> None:
    """Call first in every dataset-mutating endpoint."""
    from ..config import is_worker
    if is_worker():
        raise HTTPException(
            400, "read-only mirror — curation lives on Groundwork HQ "
                 "(edits made here are overwritten by the next sync)")
