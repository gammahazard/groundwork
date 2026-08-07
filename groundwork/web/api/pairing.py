"""Machine PAIRING — how a worker joins a fleet, in one paste.

Worker side (admin, session-only):
    POST /api/machine/pairing-code   mint a one-time string to paste at HQ
    POST /api/machine/pair           HQ calls this to install its ssh pubkey

HQ side (admin, session-only):
    POST /api/machines/pair          paste the string: register → record host
                                     keys → push our pubkey → test → probe

THE STRING is `GW1.<base64url(json)>` carrying {v, name, url, key, role,
root, ssh_user_host, ssh_host_keys}. The KEY inside is a freshly-minted
`train`-scope API key — least privilege for what HQ actually POSTs to a
worker — and the sshd host keys ride along so HQ writes known_hosts from an
already-authenticated channel instead of trusting first use.

CREDENTIALS FLOW ONE WAY. The worker ends up holding NOTHING of HQ's (its
outputs/hq.json is cosmetic — a name for refusal messages). HQ holds a
worker-scoped key and ssh access to the worker. Registration stays admin +
session-only on both sides because a registered machine receives full
datasets on the next sync.
"""
from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config import OUTPUTS_DIR, ROOT, is_worker
from .. import machine_self, machines as machines_mod
from ..auth import keys as keys_mod
from ..auth.routes import require_admin

router = APIRouter()


# ------------------------------------------------------------- worker side ---

class MintReq(BaseModel):
    url: str | None = None          # editable before minting; self_url default


@router.post("/api/machine/pairing-code")
def mint_pairing_code(body: MintReq, request: Request,
                      user: str = Depends(require_admin)):
    """Mint the paste-once pairing string. Worker role only.

    The URL is CONFIRMED BY A HUMAN here — self-IP detection can pick the
    wrong interface on a multi-homed box, and pairing is the one moment the
    person setting the machine up is guaranteed to know the reachable
    address.
    """
    if not is_worker():
        raise HTTPException(400, "pairing codes are minted on a WORKER "
                                 "(GW_ROLE=worker) — this machine is an HQ")
    url = (body.url or machines_mod.self_url() or "").strip().rstrip("/")
    if not url.startswith("http"):
        raise HTTPException(422, "no reachable URL could be detected — pass "
                                 "one explicitly (http://<this-box>:<port>)")
    kid, raw_key = keys_mod.mint(user, name="hq", scope=keys_mod.TRAIN)
    # A SINGLE-USE PAIRING TICKET: /api/machine/pair below accepts ONLY the
    # exact key minted here, for 15 minutes. Any other credential — even a
    # perfectly valid one — is refused, because "holds some API key" must
    # never escalate to "may install an ssh key on this box".
    (OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "pending_pair.json").write_text(json.dumps(
        {"sha": hashlib.sha256(raw_key.encode()).hexdigest(),
         "minted": time.time()}), encoding="utf-8")
    payload = {
        "v": 1,
        "name": os.environ.get("GW_MACHINE") or machine_self.describe()["machine"],
        "url": url,
        "key": raw_key,
        "role": "worker",
        "root": str(ROOT),
        "ssh_user_host": f"{getpass.getuser()}@{url.split('//', 1)[-1].split(':', 1)[0]}",
        "ssh_host_keys": machine_self._host_keys(),
    }
    code = "GW1." + base64.urlsafe_b64encode(
        json.dumps(payload).encode()).decode().rstrip("=")
    return {"ok": True, "code": code,
            "note": "paste this ON THE HQ's Machines tab. It contains a "
                    "train-scope API key — treat it like one."}


class PairReq(BaseModel):
    hq_pubkey: str
    hq_name: str = "hq"
    hq_url: str = ""


@router.post("/api/machine/pair")
def accept_pair(body: PairReq, request: Request):
    """Install HQ's ssh public key so the data plane works. Worker only.

    AUTH IS THE PAIRING TICKET, not a role: the ONLY credential accepted is
    the exact key minted by /api/machine/pairing-code, within 15 minutes,
    once. An admin session is deliberately NOT enough — the person pairing
    is at the HQ, and widening this to "any valid key" would turn every key
    on the box into an ssh-key installer.

    Appends to ~/.ssh/authorized_keys with a marker comment. In a container
    this file is not the HOST's sshd config — the wizard detects that (no
    host keys in /etc/ssh) and shows the manual one-liner instead.
    """
    if not is_worker():
        raise HTTPException(400, "only a worker accepts pairing")
    auth = request.headers.get("authorization", "")
    presented = auth.removeprefix("Bearer ").strip()
    try:
        pend = json.loads((OUTPUTS_DIR / "pending_pair.json")
                          .read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pend = {}
    fresh = pend and (time.time() - float(pend.get("minted", 0))) < 900
    if not (presented and fresh and hashlib.sha256(
            presented.encode()).hexdigest() == pend.get("sha")):
        raise HTTPException(403, "pairing is not open — mint a fresh pairing "
                                 "code on this machine first (15-minute, "
                                 "single-use window)")
    (OUTPUTS_DIR / "pending_pair.json").unlink(missing_ok=True)
    pub = (body.hq_pubkey or "").strip()
    if not pub.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")) \
            or "\n" in pub:
        raise HTTPException(422, "that does not look like one ssh public key")
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    ak = ssh_dir / "authorized_keys"
    have = ak.read_text(encoding="utf-8") if ak.exists() else ""
    line = f"{pub} groundwork-hq:{body.hq_name}"
    if pub.split()[1] not in have:
        with open(ak, "a", encoding="utf-8") as f:
            f.write(("" if have.endswith("\n") or not have else "\n") + line + "\n")
        ak.chmod(0o600)
    # Cosmetic record: lets refusal messages name where curation lives.
    (OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "hq.json").write_text(json.dumps(
        {"hq_name": body.hq_name, "hq_url": body.hq_url}), encoding="utf-8")
    return {"ok": True, "installed": True}


# ----------------------------------------------------------------- HQ side ---

class HqPair(BaseModel):
    code: str
    name: str | None = None         # override the worker's advertised name


@router.post("/api/machines/pair")
def hq_pair(body: HqPair, _: str = Depends(require_admin)):
    """The whole HQ half in one call: decode → register → host keys → push
    our pubkey → data-plane test → card probe. Each step reports what it did,
    and a failure leaves the machine registered-but-unverified with a resume
    path, never half-invisible."""
    code = (body.code or "").strip()
    if not code.startswith("GW1."):
        raise HTTPException(422, "that is not a Groundwork pairing code "
                                 "(expected GW1.…)")
    try:
        raw = code[4:]
        raw += "=" * (-len(raw) % 4)
        p = json.loads(base64.urlsafe_b64decode(raw.encode()))
        assert int(p.get("v", 0)) == 1
    except Exception:  # noqa: BLE001
        raise HTTPException(422, "pairing code did not decode — copy the "
                                 "whole string") from None
    import re
    name = (body.name or p.get("name") or "worker").strip()
    key = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not key:
        raise HTTPException(422, "the machine needs a usable name")
    steps = {}
    if key in machines_mod.MACHINES:
        raise HTTPException(409, f"{key!r} is a built-in name")
    if key not in machines_mod._registry():
        machines_mod.add_machine(
            key, name=name, url=p.get("url") or "",
            api_key=p.get("key") or "", ssh_host=p.get("ssh_user_host") or "",
            remote_root=p.get("root") or "", role=p.get("role") or "worker")
        steps["registered"] = True
    else:
        machines_mod.update_machine(
            key, url=p.get("url") or "", api_key=p.get("key") or "",
            ssh_host=p.get("ssh_user_host") or "",
            remote_root=p.get("root") or "")
        steps["registered"] = "updated"
    from .. import ssh_identity
    hks = p.get("ssh_host_keys") or []
    if hks and p.get("ssh_user_host"):
        steps["host_keys"] = ssh_identity.record_host_keys(
            p["ssh_user_host"], hks)
    manual = not hks
    # Push our pubkey through the worker's own API (native workers). A
    # container worker has no host sshd keys to advertise, which is also the
    # case where it cannot install into the HOST's authorized_keys — the
    # wizard shows the one manual line instead.
    if not manual:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{p['url']}/api/machine/pair", method="POST",
                data=json.dumps({
                    "hq_pubkey": ssh_identity.public_key(),
                    "hq_name": os.environ.get("GW_INSTANCE_NAME", "hq"),
                    "hq_url": machines_mod.self_url() or ""}).encode(),
                headers={"Content-Type": "application/json",
                         **machines_mod.auth_headers(key)})
            with urllib.request.urlopen(req, timeout=20) as r:
                steps["pubkey_installed"] = json.loads(r.read()).get("ok", False)
        except Exception as e:  # noqa: BLE001
            steps["pubkey_installed"] = False
            steps["pubkey_error"] = f"{type(e).__name__}: {e}"[:200]
            manual = True
    result = {"ok": True, "key": key, "steps": steps,
              "manual_step": None}
    if manual:
        result["manual_step"] = (
            f"echo '{ssh_identity.public_key()}' >> ~/.ssh/authorized_keys"
            f"   # run on {p.get('ssh_user_host') or 'the worker host'}")
    # Data-plane test + probe reuse the existing endpoints' logic through
    # their functions so the wizard shows one combined outcome.
    from .machine import test_data_plane, probe
    try:
        result["test"] = test_data_plane(key, _)
    except HTTPException as e:
        result["test"] = {"ok": False, "error": str(e.detail)}
    if result["test"].get("ok"):
        try:
            result["probe"] = probe(key, _)
        except HTTPException as e:
            result["probe"] = {"ok": False, "error": str(e.detail)}
    return result
