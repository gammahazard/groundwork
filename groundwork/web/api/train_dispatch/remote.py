"""Talking to another machine: feature-detect, refusals, the one caller."""

from __future__ import annotations

from .model import router, TrainReq, REMOTE_TIMEOUT  # noqa: F401
from .busy import _busy_on, _yolo_busy_on  # noqa: F401
from .cards import _AWARE_TTL, _PROJECT_AWARE

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


def _speaks_project(machine) -> bool | None:
    """Does this machine's cockpit have a project dimension? None = cannot tell.

    FEATURE-DETECTED, NOT ASSUMED. The old rule here was `slug == the first project`,
    which was a stand-in for two facts that have both since changed: the mirror
    now carries every project (and its manifest), and a current cockpit takes
    `?project=` everywhere. What has NOT necessarily changed is the machine at
    the other end — the worker's tree is written by rsync, not git, so it can be any
    age at all and a version number would be a guess.

    So ask it something only a current build answers. /api/machine/self is that
    marker: it landed with this work, it is cheap, and a build without it is by
    definition a build from before projects could train remotely. Measured
    2026-08-05: the worker answers /api/retrain with 200 and /api/machine/self with
    404, which is exactly the state this must refuse.

    The moment that box is deployed to, this flips by itself. No code change, no
    flag to remember.
    """
    hit = _PROJECT_AWARE.get(machine.key)
    if hit and time.time() - hit[1] < _AWARE_TTL:
        return hit[0]
    try:
        req = urllib.request.Request(f"{machine.url}/api/machine/self",
                                     headers=machines_mod.auth_headers(machine.key))
        with urllib.request.urlopen(req, timeout=10) as r:
            ok = r.status == 200
    except urllib.error.HTTPError as e:
        # 404 = an OLD build, which is the thing this refuses.
        # 401 = a CURRENT build that we have no key for — a different problem
        # with a different fix, and calling it "old" would send someone to
        # redeploy a machine that is already up to date.
        if e.code in (401, 403):
            return None
        ok = False
    except Exception:  # noqa: BLE001 — unreachable is not "old", it is unknown
        return None
    _PROJECT_AWARE[machine.key] = (ok, time.time())
    return ok


def _remote_ok(p, machine=None) -> tuple[bool, str]:
    """May THIS project be trained on `machine`? See the module docstring."""
    if machine is None or machine.local:
        return True, ""
    aware = _speaks_project(machine)
    if aware:
        return True, ""
    if aware is None:
        return False, (f"could not reach {machine.name} to check whether it "
                       f"understands projects — refusing rather than sending "
                       f"{p.slug}'s run to a machine I cannot see.")
    return False, (
        f"{machine.name} is running an older Groundwork with no project "
        f"dimension: it would ignore ?project=, train on whatever dataset it has, "
        f"and the run would come home recorded as {p.slug}'s. Deploy the current "
        f"code to it and this allows itself — no flag to change.")


def refuse_if_busy(req, m, machine, slug: str) -> None:
    """ONE AUTHORITY PER CARD. Raises unless `machine` has no other GPU job.

    A NAMED FUNCTION, not an `if` inside start(), and that is the whole point.
    The first version of this gate was inline, and the check written to guard it
    asserted only that start() MENTIONED the probes — so disabling the gate with
    `if False:` left every call and every raise textually in place and the check
    passed green on a completely defeated guard. A check that cannot fail is not
    a check (this repo has been bitten by that twice). Pulled out here so
    scratch/check_gpu_gate.py can CALL it and watch it refuse.

    BOTH DIRECTIONS, because the hole was asymmetric: a challenger launching into
    a running retrain was guarded by lab_ops.train's split test, while a retrain
    launching into a running challenger was guarded by nothing at all. On
    2026-08-02 that unguarded direction put a retrain on card 0 beside yolox-s and
    killed it at epoch 292 of 300.

    Per-card GPU locks do NOT cover this — they exist to stop two jobs on one
    card, and permit exactly the pairing that keeps failing.
    """
    if req.allow_concurrent:
        return
    other, why = (_busy_on(machine, slug) if m.name == "yolov8n"
                  else _yolo_busy_on(machine, slug))
    if why:
        raise HTTPException(502, f"{why} — refusing to start a second GPU job on "
                                 f"a machine I cannot see. Retry, or pass "
                                 f"allow_concurrent if you mean it.")
    if other:
        what = (f"challenger {other} is training" if m.name == "yolov8n"
                else "a yolo retrain is running")
        raise HTTPException(
            409,
            f"{what} on {machine.name} — refusing to run two GPU jobs at once. "
            f"That pairing has failed 3 of 4 trials on this box (two /dev/dxg "
            f"wedges at cold init, and yolox-s dying at epoch 292/300 with an "
            f"unspecified launch failure after an hour alongside a retrain). "
            f"Wait for it, cancel it, or pass allow_concurrent:true to run the "
            f"experiment deliberately.")


def _remote(machine, path: str, body: dict, slug: str | None,
            method: str = "POST", timeout: float | None = None) -> dict:
    """Send a training request to another machine. Blocking, once, with a timeout.

    `slug=None` sends NO project — for machine-scope questions
    (/api/machine/status), which deliberately have no project dimension.

    Blocking on purpose. lab_proxy exists because an expensive call on a POLLED
    endpoint wedged this cockpit three times; this is a one-shot user action
    where a spinner is honest, and fire-and-forget would mean a Train button
    that cannot tell you it failed.
    """
    ok, why = machines_mod.check(machine)
    if not ok:
        raise HTTPException(502, why)
    url = f"{machine.url}{path}"
    if slug is not None:
        sep = "&" if "?" in path else "?"
        url += f"{sep}project={urllib.parse.quote(slug)}"
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json",
                 **machines_mod.auth_headers(machine.key)})
    try:
        with urllib.request.urlopen(req, timeout=timeout or REMOTE_TIMEOUT) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read() or b"{}").get("detail") or ""
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(502, f"{machine.name} refused: "
                                 f"{detail or e.reason}"[:300]) from None
    except Exception as e:  # noqa: BLE001 — an unreachable worker is not a crash
        raise HTTPException(
            502, f"could not reach {machine.name} ({type(e).__name__}) at "
                 f"{machine.url}. Check the network, or train on this "
                 f"machine.") from None
