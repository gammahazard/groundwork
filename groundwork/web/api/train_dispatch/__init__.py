"""The unified Train dispatch — a same-name package, one router.

model owns the router + request shape; routes adds the endpoints on
import. The public surface matches the old module.
"""
from __future__ import annotations

from .model import router, TrainReq, REMOTE_TIMEOUT       # noqa: F401
from .busy import _busy_on, _yolo_busy_on                 # noqa: F401
from .cards import _cells, _default_card, _batch_for      # noqa: F401
from .remote import (_remote, _remote_ok,                 # noqa: F401
                     _speaks_project, refuse_if_busy)
from . import routes                                      # noqa: F401
