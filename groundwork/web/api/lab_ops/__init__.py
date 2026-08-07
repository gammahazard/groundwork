"""The challenger (alt-model) API — a same-name package, one router.

common owns the router and the shared helpers; status/media/train add
routes to it on import. The public surface below is exactly what the
old single module exported, so no import site moves.
"""
from __future__ import annotations

from .common import (router, MAIN_PY, GPU_BASE_ENV,        # noqa: F401
                     _alt, _every_alt, _trainable_archs,
                     _card_env, _local_active, _running_seed,
                     _finishing, _score_blocked, _adoptable, _split_ok)
from . import status, media, train                        # noqa: F401
# The REQUEST MODELS AND THE HANDLERS both — ops/autoscore.py calls score()
# in-process (a timer must not hold an API key to talk to its own machine), and
# it was importing a name this package did not re-export. Every challenger run
# a worker finished failed to score with an ImportError the moment the timer
# reached it, which is to say: the whole point of the timer never worked.
from .train import TrainReq, ScoreReq, score, train        # noqa: F401
