#!/usr/bin/env python3
"""Web cockpit for the YOLO object counter — thin assembler.

Mounts the small API routers (groundwork/web/*) + static SPA. Review the fix
queue, edit dots in the browser, and retrain in the background. Loads nothing
heavy at import; the counter model loads lazily on first ad-hoc count.

Run:
  .venv/bin/python -m groundwork.web        # then open http://localhost:8000
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from groundwork.config import OUTPUTS_DIR
from groundwork.web import asset_stamp
from groundwork.web.api import ROUTERS
from groundwork.web.auth import users
from groundwork.web.auth.middleware import AuthGate

STATIC = Path(__file__).resolve().parents[2] / "frontend"


class RevalidatingStatic(StaticFiles):
    """StaticFiles that forces the browser to ASK before reusing a file.

    Starlette's StaticFiles sends ETag and Last-Modified but no Cache-Control.
    With no explicit freshness a browser falls back to a HEURISTIC — roughly a
    tenth of the file's age — and during that window it reuses the cached copy
    WITHOUT revalidating. So an edited .js or .css keeps serving the old bytes
    for minutes to hours, and nothing in the page says so.

    That made `?v=N` in index.html the only cache-busting we had, and it is a
    number a human has to remember to bump. Forget it once — as happened while
    debugging "the burger menu doesn't open", which was verified working in a
    real browser while the owner's browser still held the previous JS — and you
    are debugging code the browser is not running. That is an expensive class
    of bug in a repo whose stated failure mode is silent wrong-data.

    `no-cache` does not mean "do not store"; it means "store, but revalidate
    every time". The browser still sends If-None-Match and still gets a cheap
    304 with no body. On a LAN cockpit that is a free round trip, and it makes
    a hard refresh never the answer to "why can't I see my change".

    Deliberately NOT applied to /outputs: those are image photographs, they are
    big, and they are addressed by a filename that changes when the content
    does. Revalidating every one of a 225-image grid would be 225 round trips
    for nothing.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app = FastAPI(title="Groundwork Cockpit")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/static", RevalidatingStatic(directory=str(STATIC)), name="static")

for router in ROUTERS:          # the one list lives in groundwork/web/api/
    app.include_router(router)

# ---------------------------------------------------------------- the gate ---
# ADDED LAST, WRAPPING EVERYTHING — including both mounts above. That ordering
# is the whole point: /outputs is eighteen gigabytes of image photographs, eval
# previews and checkpoints served by StaticFiles, and a FastAPI dependency would
# have guarded the routes and left every one of those files open. See
# groundwork/web/auth/middleware.py.
#
# GW_AUTH=0 turns it off, and that is a real state rather than an escape
# hatch: once the worker's UI is retired it serves only machine-to-machine traffic
# from HQ, which has no credential to present yet (FRONTEND.md §3.5). It is
# opt-OUT so that forgetting to set anything leaves the cockpit gated.
if os.environ.get("GW_AUTH", "1") != "0":
    # First run creates one account from GW_ADMIN_USER/_PASSWORD, and
    # only when there are none — see auth/users.py for why that emptiness test
    # is the whole difference between a convenience and a backdoor.
    users.bootstrap()
    if not users.count():
        print("[webui] NO ACCOUNTS EXIST and none could be bootstrapped — "
              "nobody can sign in. Create one:\n"
              "        .venv/bin/python -m groundwork.web.auth.users add "
              "<name> --admin", flush=True)
    app.add_middleware(AuthGate)
else:
    print("[webui] GW_AUTH=0 — the cockpit is UNAUTHENTICATED", flush=True)


# AUTO-PROBE `here` ON STARTUP — the single biggest first-run dead end was a
# Train button that stays refused until someone finds the probe. Background
# thread: the probe shells short-lived interpreters through safe_proc and can
# take ~10s; startup must not wait on it, and a hung driver call must not
# stop the server from binding. Skipped when a measurement already exists.
def _auto_probe() -> None:
    try:
        from groundwork.web import machine_self, machines
        if machines.cards("here"):
            return
        import json as _json
        d = machine_self.describe(cards=True, venvs=True)
        m = machines._map()
        m["here"] = {"cards": d.get("cards") or [],
                     "venvs": d.get("venvs") or {},
                     "busy": bool(d.get("busy")),
                     "measured": d.get("measured"),
                     "cards_stale": bool(d.get("cards_stale"))}
        machines.MAP_PATH.write_text(
            _json.dumps({"machines": m}, indent=1), encoding="utf-8")
        print(f"[web] auto-probed this machine: "
              f"{len(d.get('cards') or [])} card(s)", flush=True)
    except Exception as e:  # noqa: BLE001 — a failed probe must not kill startup
        print(f"[web] auto-probe skipped ({type(e).__name__}: {e})", flush=True)


import threading as _threading
_threading.Thread(target=_auto_probe, daemon=True).start()


@app.get("/healthz")
def healthz() -> dict:
    """Liveness for container orchestration — open (in the gate's exact set)
    because /api/me answers 401 to an anonymous probe and `curl -f` reads any
    4xx as unhealthy."""
    return {"ok": True}


# SCHEMA STAMP, from day one. private/meta.json records which on-disk schema
# this instance runs; upgrades apply ordered additive migrations against it.
# v1 has none — the hook existing before it is needed is the point.
def _migrate() -> None:
    try:
        import json as _json
        from groundwork.config import PRIVATE_DIR
        PRIVATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        meta = PRIVATE_DIR / "meta.json"
        if not meta.exists():
            meta.write_text(_json.dumps({"schema": 1}), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[web] schema stamp skipped ({type(e).__name__}: {e})", flush=True)


_migrate()


# SUPERVISOR-MODE BOTS AUTOSTART. Under systemd the units do this; in a
# container (or `groundwork run`) nothing did — botsup.boot_enabled existed
# and had no caller, so an enabled bot stayed dead until someone pressed Start
# in the cockpit. Same thread posture as the probe: never block the bind, never
# let a bad bot stop the server.
def _boot_bots() -> None:
    try:
        from groundwork import project as _project
        from groundwork.web import botsup
        n = botsup.boot_enabled([_project.load(s) for s in _project.slugs()])
        if n:
            print(f"[web] started {n} enabled bot(s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[web] bot autostart skipped ({type(e).__name__}: {e})", flush=True)


_threading.Thread(target=_boot_bots, daemon=True).start()


# INLINE SCHEDULER for installs with neither systemd timers nor the compose
# scheduler service (`groundwork run`, WSL without systemd). Explicit opt-in.
if os.environ.get("GW_SCHEDULER", "").strip() == "inline":
    from groundwork.ops import scheduler as _sched
    _threading.Thread(target=_sched.loop, daemon=True).start()
    print("[web] inline scheduler thread started", flush=True)


@app.get("/")
def index() -> HTMLResponse:
    # Never let the browser serve a stale shell: it must revalidate the HTML.
    # The shell then pulls css/js whose ?v= is a hash of that file's CONTENT
    # (asset_stamp), so an edited asset gets a new URL whether or not anyone
    # remembered to bump a version by hand. See asset_stamp.py for the bug.
    html = asset_stamp.stamp((STATIC / "index.html").read_text(encoding="utf-8"),
                             STATIC)
    return HTMLResponse(html,
                        headers={"Cache-Control": "no-cache, must-revalidate"})


def main() -> None:
    """Run the cockpit. Host/port come from GW_HOST / GW_PORT.

    GW_TRUST_PROXY=1 is the reverse-proxy opt-in: uvicorn honours
    X-Forwarded-* (so the login throttle keys on the real client, not the
    proxy) and the session cookie turns Secure. OFF by default — the default
    deployment is plain HTTP on a private network, where a Secure cookie
    would break sign-in.
    """
    import uvicorn
    host = os.environ.get("GW_HOST", "0.0.0.0")
    port = int(os.environ.get("GW_PORT", "8000"))
    trust = os.environ.get("GW_TRUST_PROXY", "") == "1"
    print(f"[web] cockpit -> http://localhost:{port}"
          + (" (proxy headers trusted)" if trust else ""), flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning",
                proxy_headers=trust,
                forwarded_allow_ips="*" if trust else None)


if __name__ == "__main__":
    main()
