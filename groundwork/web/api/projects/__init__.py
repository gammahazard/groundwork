"""Projects API — a same-name package, one router."""
from __future__ import annotations

from .shared import (router, _counts, _best_run, _thumb,   # noqa: F401
                     _last_activity, _card, _visible)
from . import listing, create, detail                       # noqa: F401
