"""WHERE a run can be trained — which machines, which CARDS, and which models fit.

Training used to be two buttons in two places: HQ's Retrain, and a link that
opened the lab's separate cockpit so you could press ITS Retrain. That is the
split the owner is removing — the UI is a web app on the network, reachable from
the laptop, the phone or a browser at the worker, so "where you work" should stop
being tied to "which machine has the GPU". Machine and CARD become fields on the
Train action.

THE CLASS OF CONSTRAINT THIS FILE EXISTS TO ENCODE: a sidecar venv's torch
build carries kernels for a fixed set of compute capabilities, and a card whose
sm_XX is not in that list cannot run it at all — not "should not": there are no
kernels and no `compute_XX` PTX to JIT forward from, so the first kernel launch
dies. Which venv can drive which card is therefore MEASURED per machine (the
probe), never assumed.

CARD INDEXES ARE PCI ORDER, ALWAYS. CUDA numbers cards fastest-first by default,
so a plain CVD=1 landed on the 5070 Ti rather than the 3090 on this box and cost
a run (docs/models.md). Every launch this file describes sets
CUDA_DEVICE_ORDER=PCI_BUS_ID first.

A CLOSED SET, like web/bot_roles.py and for the same reason: a request picks
WHICH machine, never a URL. Otherwise "train over there" would be an endpoint
that POSTs a job to an arbitrary host.

WHAT IS FACT vs WHAT IS RECORDED. `here` is code; every other machine is a
REGISTRY entry (added by an admin, holding its URL, API key and ssh identity).
The cards and venv arch lists are MEASURED and live in outputs/machines.json,
written by the probe (POST /api/machines/{key}/probe → the remote's
/api/machine/cards). The measurement is stamped so the UI can say when, and a
missing map degrades to "unknown" rather than to a confident wrong answer.
"""
from __future__ import annotations

from .model import MACHINES, Machine, here, check          # noqa: F401
from .capacity import (MAP_PATH, _map, cards, measured_at,  # noqa: F401
                       venv_info, can_run, fits, pick_card, summary,
                       VRAM_HEADROOM_GB)
from .selfid import self_ip, vpn_ip, self_url               # noqa: F401
from .registry import (REGISTRY_PATH, _registry, all_machines,  # noqa: F401
                       workers, get, registered, add_machine,
                       update_machine, mark_verified, set_backup_target,
                       remove_machine, public_registry, api_key_for,
                       auth_headers)
from .peaks import measured_peak, can_train                 # noqa: F401
