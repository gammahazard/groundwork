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


def _start_service(port: int) -> str | None:
    """Bring the cockpit up. Returns how it was started, or None on failure."""
    from . import botsup
    if botsup.mode() == "systemd":
        try:
            from .. import install as installer
            installer.install(role="worker")
            if _wait_healthy(port):
                return "systemd unit groundwork-web.service (survives reboots)"
            print("[join] the unit was installed but the cockpit did not answer "
                  "— check: systemctl --user status groundwork-web")
        except SystemExit as e:
            print(f"[join] could not install the service unit ({e}) — "
                  f"falling back to a detached process")
        except Exception as e:  # noqa: BLE001
            print(f"[join] could not install the service unit "
                  f"({type(e).__name__}: {e}) — falling back to a detached process")

    from . import spawn
    log = OUTPUTS_DIR / "webui.log"
    import sys
    py = ROOT / ".venv" / "bin" / "python"
    interp = str(py) if py.exists() else sys.executable
    spawn.spawn_detached("webui", [interp, "-m", "groundwork.web"],
                         log_path=log, env={**os.environ}, cwd=str(ROOT))
    if not _wait_healthy(port):
        print(f"[join] the web service did not come up — see {log}")
        return None
    return ("a detached process — it will NOT survive a reboot. Install a "
            "service with: .venv/bin/groundwork install --role worker")


def main() -> int:
    ap = argparse.ArgumentParser(description="Join this box to a Groundwork hub.")
    ap.add_argument("--hub", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--url", default=os.environ.get("GW_SELF_URL"),
                    help="URL the hub should call this box on — honours a "
                         "GW_SELF_URL= prefix on the join command, exactly "
                         "like GW_PORT (default: http://<detected-ip>:<port>, "
                         "and on a WSL or multi-homed box the detected "
                         "interface is often one the hub cannot dial)")
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
    if args.url:
        # Persist the pinned self-address too — the announcement must stay
        # true across restarts, not just for this one join.
        env_file.set_key("GW_SELF_URL", args.url)
        os.environ["GW_SELF_URL"] = args.url

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

    # 4. Start the web service, so the hub's enroll can call back — AS A
    #    SYSTEMD UNIT where there is one. A detached process serves the join
    #    and then dies at the next reboot, which is the worst kind of broken:
    #    the machine pairs, trains, and is silently gone tomorrow with nothing
    #    in the cockpit to say why. `groundwork install` writes the unit,
    #    enables it, and turns on linger (user units stop at logout without
    #    it, and a GPU box is headless). The detached fallback stays for
    #    containers and anything without a user systemd.
    service = "already running"
    if not _wait_healthy(args.port, tries=1):
        service = _start_service(args.port)
        if service is None:
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
    except (urllib.error.URLError, OSError, ValueError) as e:
        # UNREACHABLE HUB IS THE COMMON CASE — wrong URL, firewall, HQ not
        # running — and it used to end a one-command install in a Python
        # traceback. Say what could not be reached and what to check.
        print(f"[join] could not reach the hub at {args.hub}: "
              f"{getattr(e, 'reason', e)}\n"
              f"[join] check the URL, that the HQ is running, and that this "
              f"box can reach it (curl {args.hub}/healthz).")
        return 1

    print("[join] hub result:")
    print(json.dumps(result, indent=2)[:1200])
    ok = bool(result.get("ok")) and (result.get("test") or {}).get("ok")
    print()
    if ok:
        print(f"[join] ✓ this machine is registered and verified as "
              f"{result.get('key')!r}.")
        print("[join] next: back on the hub, open the Machines tab (refresh "
              "if already open) — this machine is listed with its cards "
              "probed, ready to be picked in Train.")
    else:
        print(f"[join] registered as {result.get('key')!r}, but not fully "
              "verified.")
        if result.get("manual_step"):
            print("[join] 1. run the manual step below (in this terminal — "
                  "you are already on the right machine):")
            print("       ", result["manual_step"])
            print("[join] 2. then, back on the hub: Machines tab → refresh → "
                  "find this machine → press Probe to read its cards.")
        else:
            print("[join] next: back on the hub, Machines tab → refresh → "
                  "find this machine → press Probe to read its cards.")
        print("[join] the ssh data plane is proven at the first dispatched "
              "run — a remote train syncs first and refuses, in ssh's own "
              "words, if the path is broken.")
    print(f"[join] running as {service}")
    if admin_pw:
        print(f"[join] this worker's own cockpit: {url}  "
              f"(user 'admin', password {admin_pw} — shown only this once)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
