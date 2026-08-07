"""First-run setup — open EXACTLY while zero accounts exist, then latched shut.

The wizard's state is DERIVED, never stored: "does setup need to run" is
`users.count() == 0`, the same emptiness test the env bootstrap uses, and the
post-claim steps read plain facts (is the local machine probed, does any
project exist, are the LA weights present). Nothing to migrate, nothing to go
stale, and on upgrade the wizard simply never fires because users exist.

THE LATCH. /api/setup/status and /api/setup/claim are reachable without a
session only while the instance has no accounts. The first successful claim —
or the first request that OBSERVES an account — closes them for the process
lifetime. Deleting users.json later does NOT reopen setup without a deliberate
restart; that asymmetry is a feature (no reset backdoor), the same philosophy
as the bootstrap's emptiness test. A race between two strangers on an exposed
port has one winner and one 409, and both are audited.

Everything else here requires an ADMIN session like any other admin surface.
"""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from ...config import MODEL_PATH, role
from .. import env_file
from ..auth import audit as audit_mod
from ..auth import sessions, users
from ..auth.routes import require_admin
from ..auth.routes.guards import _device, _ip, _set_cookie

router = APIRouter()

_claim_lock = threading.Lock()
_latched = False


def setup_open() -> bool:
    """May the two bootstrap endpoints answer without a session?"""
    global _latched
    if _latched:
        return False
    if users.count():
        _latched = True
        return False
    return True


class Claim(BaseModel):
    username: str
    password: str


@router.get("/api/setup/status")
def status():
    """What the wizard needs to decide which step to show. Open while empty;
    afterwards it answers `needed: false` and nothing more without a session."""
    if not setup_open():
        return {"needed": False}
    from .. import machines
    from ...config import MACHINE
    return {"needed": True,
            # THIS BOX'S NAME (hostname, or GW_MACHINE). here().name is the
            # registry's label — "This machine" — which reads as a placeholder
            # in the wizard's sentence and tells a first-time user nothing.
            "machine": MACHINE,
            "suggested_url": machines.self_url(),
            "role": role()}


@router.post("/api/setup/claim")
def claim(body: Claim, request: Request, response: Response):
    """Create the FIRST admin and sign them in. One winner, forever.

    must_change=False, unlike the env bootstrap: this password was typed by
    the person into a form seconds ago, not shipped in a file.
    """
    global _latched
    with _claim_lock:
        if not setup_open():
            raise HTTPException(409, "this instance already has accounts — "
                                     "sign in instead")
        try:
            users.add(body.username, body.password, admin=True,
                      must_change=False)
        except users.UserError as e:
            raise HTTPException(422, str(e)) from None
        _latched = True
    token = sessions.begin(body.username,
                           ua=request.headers.get("user-agent"),
                           ip=_ip(request))
    _set_cookie(response, token)
    try:
        audit_mod.record(audit_mod.SETUP_CLAIM, actor=body.username,
                         ip=_ip(request),
                         detail="first admin via setup wizard")
    except Exception:  # noqa: BLE001 — the trail must never block the claim
        pass
    return {"ok": True, "user": body.username}


class Instance(BaseModel):
    instance_name: str | None = None
    machine_name: str | None = None


@router.post("/api/setup/instance")
def instance(body: Instance, _: str = Depends(require_admin)):
    """Name the instance/machine — written to .env via the atomic 0600 writer.
    Values follow env_file's own no-whitespace rule; slug-like names fit."""
    wrote = {}
    if body.instance_name:
        env_file.set_key("GW_INSTANCE_NAME", body.instance_name.strip())
        wrote["GW_INSTANCE_NAME"] = body.instance_name.strip()
    if body.machine_name:
        env_file.set_key("GW_MACHINE", body.machine_name.strip())
        wrote["GW_MACHINE"] = body.machine_name.strip()
    return {"ok": True, "wrote": sorted(wrote),
            "note": "takes effect for new processes; the page keeps working"}


@router.get("/api/setup/facts")
def facts(_: str = Depends(require_admin)):
    """The derived post-claim checklist the wizard renders from."""
    from pathlib import Path
    from ... import project as project_mod
    from .. import machines
    la_dir = Path(MODEL_PATH)
    # "probed" = a probe has RUN, not "cards exist" — a CPU-only box is
    # probed and has zero cards, and the wizard must say that rather than
    # showing a spinner forever. train_env tells the two zero-card cases
    # apart: no GPU, versus no torch anywhere to ask with.
    main_env = machines.venv_info("here", ".venv")
    return {"probed": machines.measured_at("here") is not None,
            "cards": machines.cards("here"),
            "train_env": bool(main_env.get("torch")),
            "projects": project_mod.slugs(),
            "la_present": la_dir.exists() and any(la_dir.iterdir())
            if la_dir.exists() else False,
            "extras": _extras_state()}


# ---------------------------------------------------------------- extras ----
# Opt-in downloads with a license consent step. Each runs in a background
# thread and writes progress the wizard polls — the house idiom (no sockets).

_extras: dict[str, dict] = {}
_extras_lock = threading.Lock()


def _extras_state() -> dict:
    with _extras_lock:
        return {k: dict(v) for k, v in _extras.items()}


class LaReq(BaseModel):
    accept_license: bool = False
    hf_token: str | None = None
    remember_token: bool = False


@router.post("/api/setup/extras/la")
def extras_la(body: LaReq, _: str = Depends(require_admin)):
    """Download the LocateAnything-3B auto-labeler weights from Hugging Face.

    CONSENT IS THE POINT: the weights are under NVIDIA's own license, between
    the user and NVIDIA — this platform never redistributes them. Anonymous
    first; a 401/403 tells the wizard to reveal the token field.
    """
    if not body.accept_license:
        raise HTTPException(422, "accept_license must be true — the weights "
                                 "are under NVIDIA's license, which you "
                                 "accept on Hugging Face, not here")
    with _extras_lock:
        if (_extras.get("la") or {}).get("state") == "running":
            return {"ok": True, "already": True}
        _extras["la"] = {"state": "running", "pct": 0, "at": time.time()}
    if body.hf_token and body.remember_token:
        env_file.set_key("HF_TOKEN", body.hf_token.strip())

    def work():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download("nvidia/LocateAnything-3B",
                              local_dir=MODEL_PATH,
                              token=(body.hf_token or None))
            with _extras_lock:
                _extras["la"] = {"state": "done", "pct": 100, "at": time.time()}
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            needs_token = "401" in msg or "403" in msg or "gated" in msg.lower()
            with _extras_lock:
                _extras["la"] = {"state": "error", "error": msg[:300],
                                 "needs_token": needs_token, "at": time.time()}

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "started": True}


class StackReq(BaseModel):
    stack: str
    accept_license: bool = False


@router.post("/api/setup/extras/stack")
def extras_stack(body: StackReq, _: str = Depends(require_admin)):
    """Install a challenger stack (sidecar venv + vendor code) by manifest."""
    from ... import stacks
    if body.stack not in stacks.MANIFESTS:
        raise HTTPException(404, f"no stack {body.stack!r} — offered: "
                                 f"{', '.join(sorted(stacks.MANIFESTS))}")
    if not body.accept_license:
        raise HTTPException(422, "accept_license must be true")
    key = f"stack:{body.stack}"
    with _extras_lock:
        if (_extras.get(key) or {}).get("state") == "running":
            return {"ok": True, "already": True}
        _extras[key] = {"state": "running", "at": time.time()}

    def work():
        try:
            log = stacks.install(body.stack)
            with _extras_lock:
                _extras[key] = {"state": "done", "at": time.time(),
                                "log_tail": log[-1000:]}
        except Exception as e:  # noqa: BLE001
            with _extras_lock:
                _extras[key] = {"state": "error", "at": time.time(),
                                "error": f"{type(e).__name__}: {e}"[:300]}

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "started": True}
