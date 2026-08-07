"""The cockpit's HTTP surface — one router per area, and nothing else.

The split this package makes is *HTTP surface* vs *what actually runs*:

    groundwork/web/api/    request in, response out
    groundwork/web/        the machinery — retrain_job, retrain_worker, la_job,
                            run_stats, lab_proxy, lab_guard

A module here should parse its inputs, call into `groundwork/` and format the
answer. Anything that owns state, holds a lock, spawns a process or outlives a
request belongs at `web/` root instead.

`ROUTERS` is the registration list, and it is the ONE place routers are
enumerated. The web assembler used to spell the same names out twice — once in
an import tuple, once in the `for` loop that registered them — so adding a
router and updating only one of the two left it importable, greppable, and
serving nothing. Keep the import and ROUTERS adjacent so a mismatch is visible
in one screen.

Guarding writes is `lab_guard.no_lab_edits()`, called by the routers that mutate
the dataset — a worker mirrors HQ one-way, so a write there would be silently
overwritten on the next sync (see docs/machines.md).
"""
from __future__ import annotations

from . import (bot_setup, bots, buckets, collect_page, count, export,
               history, images, la, lab, lab_ops, machine, project_classes,
               pairing, projects, setup, snapshot, state, train,
               train_dispatch, upload)
# Signing in is not an "area" of the cockpit, it is how you reach any of them —
# so it lives in web/auth/ with the gate and the account store rather than as a
# sibling here. It is enumerated in the SAME list because this docstring's rule
# is that ROUTERS is the one place routers are registered, and an exception
# would be the mismatch that rule exists to prevent.
from ..auth import routes as auth_routes

# Order is not significant — the paths are disjoint and FastAPI matches on path,
# not registration order. Listed to mirror the import above.
ROUTERS = (auth_routes.router,
           bot_setup.router, bots.router, buckets.router, collect_page.router,
           count.router, export.router, history.router, images.router,
           la.router, lab.router, lab_ops.router, machine.router,
           pairing.router, project_classes.router, projects.router,
           setup.router, snapshot.router, state.router, train.router,
           train_dispatch.router, upload.router)

__all__ = ["ROUTERS", "auth_routes", "bot_setup", "bots", "buckets",
           "collect_page", "count", "export", "history", "images", "la",
           "lab", "lab_ops", "machine", "pairing", "project_classes", "projects",
           "setup", "snapshot", "state", "train", "train_dispatch", "upload"]
