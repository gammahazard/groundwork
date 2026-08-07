"""The retrain job — a same-name package; one cross-process contract.

state owns the files and locks every process shares; progress reads,
pipeline runs (inside the detached worker), control admits and kills.
The public surface matches the old module, so no import site moves.
"""
from __future__ import annotations

from .state import (LOG, STATE, WORKER_LOG, LOCKFILE, _STAGE,   # noqa: F401
                    _lockfile, gpu_lock, _alive, _read, _write,
                    _update, _pp, _lock, _PP_CACHE, _IDLE)
from .state import _tail                                        # noqa: F401
from .progress import status, _progress                         # noqa: F401
from .pipeline import run_pipeline, _record_history             # noqa: F401
from .control import start, cancel, _launch_pipeline, _is_pipeline  # noqa: F401
