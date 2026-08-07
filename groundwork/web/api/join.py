"""One-command worker onboarding — the JOIN flow.

The pairing code (pairing.py) is the manual door: a human mints on the worker
and pastes at HQ. This is the automatic one, and it inverts the carry: HQ
mints a JOIN TOKEN, the human runs one line on the GPU box, and the box does
everything itself — downloads Groundwork FROM HQ, installs, self-configures
as a worker, and PUSHES the same payload the pairing code carries to
/api/machines/join. Registration lands in pairing.enroll(), so both doors
run the identical register → trust → key-install → test → probe chain.

THE TOKEN is the credential, shaped exactly like the pairing ticket: stored
as a sha256, single-use for the join itself, 30-minute TTL. The BUNDLE
download validates the token without consuming it (the box needs the source
before it can join). /join.sh is open — it is the same text for everyone and
contains no secrets; the token arrives as an argument the admin copied.

WHY A BUNDLE FROM HQ AND NOT `git clone`: the repository can be private, and
a fresh box has no GitHub credential — but it can already reach HQ, and HQ
has the source it is running. The bundle is also automatically the version
the fleet actually runs.
"""
from __future__ import annotations

import hashlib
import io
import json
import secrets
import tarfile
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from ...config import PRIVATE_DIR, ROOT
from .. import machines as machines_mod
from ..auth.routes import require_admin
from .pairing import enroll

router = APIRouter()

_TOKENS = PRIVATE_DIR / "pending_join.json"
_TTL_S = 30 * 60

# What the bundle carries: the source tree, nothing that is data or secret.
_EXCLUDE = ("outputs", "private", ".git", ".env", "docker/data",
            "__pycache__", ".venv", "node_modules")


def _load_tokens() -> list[dict]:
    try:
        rows = json.loads(_TOKENS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        rows = []
    now = time.time()
    return [r for r in rows if now - float(r.get("minted", 0)) < _TTL_S]


def _save_tokens(rows: list[dict]) -> None:
    _TOKENS.parent.mkdir(parents=True, exist_ok=True)
    _TOKENS.write_text(json.dumps(rows), encoding="utf-8")


def _token_row(token: str) -> dict | None:
    sha = hashlib.sha256(token.encode()).hexdigest()
    return next((r for r in _load_tokens() if r.get("sha") == sha), None)


@router.post("/api/machines/join-token")
def mint_join_token(request: Request, user: str = Depends(require_admin)):
    """Mint the one-liner an admin pastes on a GPU box. Session-only, like
    everything that can end with a machine receiving datasets."""
    from ..auth.routes import refuse_key_auth
    refuse_key_auth(request, "mint a join token")
    token = "gwj_" + secrets.token_urlsafe(24)
    rows = _load_tokens()
    rows.append({"sha": hashlib.sha256(token.encode()).hexdigest(),
                 "minted": time.time(), "by": user})
    _save_tokens(rows)
    hub = machines_mod.self_url() or "http://<this-hq>:8000"
    return {"ok": True, "token": token, "ttl_minutes": _TTL_S // 60,
            "command": f"curl -fsSL {hub}/join.sh | bash -s -- {hub} {token}",
            "note": "Single use, 30 minutes. Run it ON the GPU machine."}


@router.get("/join.sh", response_class=PlainTextResponse)
def join_script():
    """The bootstrap script. Open: same text for everyone, no secrets — the
    token is the argument the admin carried over."""
    return _JOIN_SH


@router.get("/api/join/bundle")
def bundle(token: str = ""):
    """The source tree as a tar.gz, for a box holding a live join token.
    Validates WITHOUT consuming — the join itself is the single use."""
    if not token or _token_row(token) is None:
        raise HTTPException(403, "no live join token — mint one on the HQ's "
                                 "Machines tab (they expire after 30 minutes)")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(ROOT.iterdir()):
            if child.name in _EXCLUDE or child.name.startswith(".venv"):
                continue
            tar.add(child, arcname=f"groundwork/{child.name}",
                    filter=lambda ti: None if any(
                        part in _EXCLUDE for part in ti.name.split("/"))
                    else ti)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/gzip",
                             headers={"Content-Disposition":
                                      "attachment; filename=groundwork.tar.gz"})


class JoinReq(BaseModel):
    token: str
    payload: dict          # exactly the pairing-code payload, pushed not pasted


@router.post("/api/machines/join")
def join(body: JoinReq):
    """The worker announces itself. Auth is the join token — consumed here,
    exactly like the pairing ticket on the other side of the manual flow."""
    row = _token_row(body.token)
    if row is None:
        raise HTTPException(403, "join token is unknown, used, or expired")
    rows = [r for r in _load_tokens() if r.get("sha") != row["sha"]]
    _save_tokens(rows)                                   # single use
    p = body.payload or {}
    if int(p.get("v", 0)) != 1 or not p.get("url") or not p.get("key"):
        raise HTTPException(422, "join payload is not a v1 pairing payload")
    return enroll(p, admin=row.get("by", ""))


# The worker-side bootstrap lives as a real file so it can be shell-checked
# and read as a script, not a string. Served verbatim by /join.sh above.
_JOIN_SH = (ROOT / "deploy" / "join.sh").read_text(encoding="utf-8")
