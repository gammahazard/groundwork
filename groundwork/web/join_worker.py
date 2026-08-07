"""Worker-side JOIN: configure this box and announce it to the hub.

The last stage of join.sh, and runnable by hand:

    .venv/bin/python -m groundwork.web.join_worker \
        --hub http://hq:8000 --token gwj_…  [--url http://this-box:8000]

Everything the manual flow does across a wizard and two pastes happens here
in-process: worker role written to .env, a bootstrap admin created (random
password, printed ONCE), a train-scope key minted with the single-use
pairing ticket beside it, the web service started detached, and the same
payload a pairing code carries POSTed to the hub's /api/machines/join.

The hub then runs the identical enroll chain the paste flow uses — register,
trust host keys, install its ssh public key through the ticketed endpoint,
test the data plane, probe the cards — and this prints the result.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import secrets
import socket
import time
import urllib.request

from ..config import OUTPUTS_DIR, PRIVATE_DIR, ROOT


def _wait_healthy(port: int, tries: int = 40) -> bool:
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Join this box to a Groundwork hub.")
    ap.add_argument("--hub", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--url", default=None,
                    help="URL the hub should call this box on "
                         "(default: http://<detected-ip>:<port>)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GW_PORT", "8000")))
    ap.add_argument("--name", default=None, help="machine name (default: hostname)")
    args = ap.parse_args()

    from . import env_file, machine_self
    from . import machines as machines_mod
    from .auth import keys as keys_mod
    from .auth import users

    # 1. Role + port, persisted — restarts must come back as a worker.
    env_file.set_key("GW_ROLE", "worker")
    env_file.set_key("GW_PORT", str(args.port))
    os.environ["GW_ROLE"] = "worker"
    os.environ["GW_PORT"] = str(args.port)

    # 2. A bootstrap admin, so the worker's own cockpit is reachable later.
    #    Random password, printed exactly once — same posture as the wizard's
    #    "this password was typed seconds ago", except here the typist is us.
    admin_pw = None
    if users.count() == 0:
        admin_pw = secrets.token_urlsafe(9)
        users.add("admin", admin_pw, admin=True, must_change=False)

    # 3. The worker-scoped key the hub will hold, plus the single-use ticket
    #    that authorizes installing the hub's ssh key — the exact artifacts
    #    /api/machine/pairing-code mints, made without the HTTP hop.
    kid, raw_key = keys_mod.mint("admin", name="hq", scope=keys_mod.TRAIN)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    (PRIVATE_DIR / "pending_pair.json").write_text(json.dumps(
        {"sha": hashlib.sha256(raw_key.encode()).hexdigest(),
         "minted": time.time()}), encoding="utf-8")

    # 4. Start the web service, detached, so the hub's enroll can call back.
    from . import spawn
    started = False
    if not _wait_healthy(args.port, tries=1):
        log = OUTPUTS_DIR / "webui.log"
        import sys
        py = ROOT / ".venv" / "bin" / "python"
        interp = str(py) if py.exists() else sys.executable
        spawn.spawn_detached(
            "webui", [interp, "-m", "groundwork.web"],
            log_path=log, env={**os.environ}, cwd=str(ROOT))
        started = _wait_healthy(args.port)
        if not started:
            print(f"[join] the web service did not come up — see {log}")
            return 1

    url = (args.url or "").strip().rstrip("/")
    if not url:
        ip = machines_mod.self_ip() or socket.gethostbyname(socket.gethostname())
        url = f"http://{ip}:{args.port}"

    payload = {
        "v": 1,
        "name": args.name or os.environ.get("GW_MACHINE")
                or machine_self.describe()["machine"],
        "url": url,
        "key": raw_key,
        "role": "worker",
        "root": str(ROOT),
        "ssh_user_host": f"{getpass.getuser()}@{url.split('//', 1)[-1].split(':', 1)[0]}",
        "ssh_host_keys": machine_self._host_keys(),
    }

    # 5. Announce. The hub consumes the join token and runs the enroll chain.
    req = urllib.request.Request(
        f"{args.hub.rstrip('/')}/api/machines/join", method="POST",
        data=json.dumps({"token": args.token, "payload": payload}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            result = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        print(f"[join] hub refused: {e.read().decode()[:300]}")
        return 1

    print("[join] hub result:")
    print(json.dumps(result, indent=2)[:1200])
    ok = bool(result.get("ok")) and (result.get("test") or {}).get("ok")
    print()
    if ok:
        print(f"[join] ✓ this machine is registered and verified as "
              f"{result.get('key')!r} — it appears in the hub's Train matrix.")
    else:
        print("[join] registered, but not fully verified — the hub's Machines "
              "tab has the resume path.")
        if result.get("manual_step"):
            print("[join] manual step:", result["manual_step"])
    if admin_pw:
        print(f"[join] this worker's own cockpit: {url}  "
              f"(user 'admin', password {admin_pw} — shown only this once)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
