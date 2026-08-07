"""Challenger training pictures + the authenticated byte-proxy + log tail."""

from __future__ import annotations

from .common import (router, MAIN_PY, GPU_BASE_ENV,          # noqa: F401
                     _alt, _every_alt, _trainable_archs,
                     _card_env, _local_active, _running_seed,
                     _finishing, _score_blocked, _adoptable,
                     _split_ok)

import json
import fnmatch
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import resource
import subprocess

from ... import safe_proc
import pathlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ... import retrain_job
from .. import lab_progress
from ..lab import REPO, _NAME
from ..deps import current_project
from ....dataset import paths
from ....dataset.pipeline import split as split_mod
from ....models import registry
from .... import project as project_mod

def _what_for(urls) -> str | None:
    """Caption for a set of image urls, by the same table _vis_local searches.

    The REMOTE payload goes through this too. The worker answers with urls and, on
    a build older than this one, no caption at all — deriving it here keeps ONE
    table describing what a picture is instead of two that drift, and means HQ
    captions a remote run correctly without the worker being redeployed first.
    """
    for pattern, what in _VIS_SOURCES:
        if any(fnmatch.fnmatch(u, "*/" + pattern) for u in urls):
            return what
    return None


def _vis_local(d: pathlib.Path, n: int):
    """(urls, what) for the first source this run actually has, newest first."""
    for pattern, what in _VIS_SOURCES:
        hits = sorted(d.glob(pattern), key=lambda x: x.stat().st_mtime,
                      reverse=True)[:n]
        if hits:
            return ["/outputs/" + str(h.relative_to(REPO / "outputs"))
                    for h in hits], what
    return [], None


@router.get("/api/lab/vis")
def vis(run: str, n: int = 12, p=Depends(current_project)):
    """What a run has drawn of itself while training — whatever kind that is.

    Looks in the challenger tree AND the yolo runs tree, so one endpoint answers
    for any run rather than the champion having a separate path. Reports WHICH
    kind of picture it found, because "val predictions" and "the augmented
    batches it trains on" are different things and a caption that says which is
    the difference between reading them and guessing.
    """
    if not _NAME.match(run):
        raise HTTPException(422, "bad name")
    pp = paths.for_project(p)
    for root in (_alt(p), pp.RUNS_DIR):
        d = (root / run).resolve()
        if root.resolve() not in d.parents or not d.is_dir():
            continue
        imgs, what = _vis_local(d, n)
        if imgs:
            return {"images": imgs, "what": what}
        break   # the dir is here but empty — the pictures may be on the worker

    from ....config import is_worker
    if not is_worker():
        # THROUGH lab_proxy: lab.js polls this every 8s while a challenger trains,
        # and a blocking urlopen parks a worker thread against a machine that is
        # busy training — the only time this view is ever used.
        from ... import lab_proxy
        wk = lab_proxy.first_worker_key()
        rj, _age = lab_proxy.get(wk, f"/api/lab/vis?run={run}&n={n}") \
            if wk else (None, 0.0)
        if not rj:
            # NOT "draws nothing" — a cold proxy cache is "I have not asked yet".
            return {"images": [], "what": None}
        # ABSOLUTE urls: the browser is on HQ and these are served by the worker's
        # /outputs mount, so a relative path would 404 against HQ.
        # THROUGH HQ, NOT STRAIGHT AT THE WORKER. These used to be absolute worker
        # URLs and the browser fetched them itself — which worked only while the
        # worker had no authentication. The moment AuthGate was deployed there,
        # every training picture on HQ became a broken image and a "please sign
        # in", because the browser has an HQ session and no worker one. HQ has the
        # worker's API key; the browser should never need one.
        out = {**rj, "images": [_proxy_img(u) for u in rj.get("images", [])]}
        if out["images"]:
            out.setdefault("what", None)
            out["what"] = out["what"] or _what_for(out["images"])
        elif not out.get("note"):
            out["note"] = _NO_PICTURES
        return out
    # On the lab itself, empty IS the answer — there is nowhere else to look.
    return {"images": [], "what": None, "note": _NO_PICTURES}


# Where a proxied picture may come from. ANCHORED and explicit: this endpoint
# fetches a URL on another machine with a stored credential, so the set of
# reachable paths is a whitelist rather than "anything under /outputs". Without
# it, a crafted `path` would make HQ an authenticated proxy into the worker.
def _proxy_ok_prefixes() -> tuple[str, ...]:
    """Browser-facing URL prefixes this proxy may fetch — DERIVED from every
    project's own layout (its alt/ and runs/ trees), never a hardcoded list.
    The old two-literal tuple was one project's legacy layout, so the image
    proxy 403'd every other project's runs even on a correctly-paired fleet."""
    from ....config import ROOT
    out = []
    for pp in paths.every_project():
        for d in (pp.ALT_DIR, pp.RUNS_DIR):
            try:
                out.append("/outputs/" + d.relative_to(ROOT / "outputs").as_posix() + "/")
            except ValueError:
                continue
    return tuple(out) or ("/outputs/__none__/",)


def _proxy_img(u: str) -> str:
    """Rewrite a worker-served picture URL to one HQ will fetch on the browser's
    behalf. Anything unexpected is left alone rather than silently rewritten."""
    from ... import lab_proxy, machines as _M
    _wk = lab_proxy.first_worker_key()
    _wm = _M.get(_wk) if _wk else None
    base = (_wm.url if _wm else "") or ""
    path = u[len(base):] if (base and u.startswith(base)) else u
    if not path.startswith("/"):
        path = "/" + path
    if not path.startswith(_proxy_ok_prefixes()):
        return u if u.startswith("http") else base + u
    return "/api/lab/img?path=" + urllib.parse.quote(path, safe="")


@router.get("/api/lab/img")
async def lab_img(path: str):
    """Fetch one of the Trainer's static files USING HQ'S CREDENTIAL.

    The browser is signed in to HQ and not to the worker, and it should stay that
    way — one login, and the worker's key never leaves the server. So HQ fetches
    the bytes and passes them on.

    NOT AN OPEN PROXY. The path must be one this cockpit actually renders (a run
    tree), must be absolute, and must contain no traversal. `..` is rejected
    before the whitelist rather than after, since `/outputs/alt/../../etc` starts
    with an allowed prefix.
    """
    if not path.startswith("/") or ".." in path or "\\" in path:
        raise HTTPException(422, "bad path")
    ok = _proxy_ok_prefixes()
    if not path.startswith(ok):
        raise HTTPException(403, "only project run/alt trees may be proxied")
    from ... import lab_proxy, machines as machines_mod
    wk = lab_proxy.first_worker_key()
    wm = machines_mod.get(wk) if wk else None
    base = wm.url if wm else None
    if not base:
        raise HTTPException(503, "no worker machine registered")
    # ASYNC, and that is the point of this whole function being `async def`.
    # urllib.urlopen BLOCKS, and FastAPI runs every `def` endpoint in anyio's
    # 40-thread pool — so a browser loading a grid of training pictures while the
    # worker is busy (which is exactly when it is slowest to answer) parked one
    # worker per image, for up to the timeout. That is the shape CLAUDE.md
    # records as costing a cockpit that serves nothing sync while async routes
    # still answer in 0.2s, and scratch/check_polled_endpoints.py exists to
    # refuse it. Awaiting yields the loop instead of holding a thread.
    try:
        import httpx
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as c:
            r = await c.get(base + path,
                            headers=machines_mod.auth_headers(wk))
        if r.status_code >= 400:
            raise HTTPException(r.status_code,
                                f"the worker refused it ({r.status_code})")
        body = r.content
        ctype = r.headers.get("Content-Type", "application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — unreachable is a 502, not a 500
        raise HTTPException(502, f"could not reach the Trainer: "
                                 f"{type(e).__name__}")
    # Cached hard: a training picture at a given path never changes its bytes.
    return Response(content=body, media_type=ctype,
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/api/lab/log")
def log(run: str, n: int = 40, p=Depends(current_project)):
    """Tail of a challenger run's training log — the equivalent of the yolo
    retrain log. Proxies to the Trainer when the run isn't local."""
    if not _NAME.match(run):
        raise HTTPException(422, "bad name")
    alt = _alt(p)
    d = (alt / run).resolve()
    if alt.resolve() not in d.parents:
        raise HTTPException(422, "bad name")
    cands = sorted(d.glob("train/*/*.log")) + sorted(d.glob("train/log.txt")) + \
        ([d / "lab.log"] if (d / "lab.log").exists() else [])
    from ....config import is_worker
    if not cands and not is_worker():
        # THROUGH lab_proxy, not a blocking urlopen. This is a POLLED endpoint —
        # the log panel refreshes it every few seconds — and lab_proxy's own
        # header already calls "an expensive external call on a polled endpoint"
        # this codebase's recurring bug. A 3s timeout bounds each call but still
        # parks a worker thread per poll whenever the worker is slow, and the worker is
        # slowest exactly while it trains, which is the only time this is read.
        # Cached + refreshed off-thread, it answers instantly and says how stale.
        from ... import lab_proxy
        wk = lab_proxy.first_worker_key()
        d, age = lab_proxy.get(wk, f"/api/lab/log?run={run}&n={n}") \
            if wk else (None, 0.0)
        return {**(d or {"tail": ""}), "age_s": age, "remote": True}
    if not cands:
        return {"tail": ""}
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    text = newest.read_bytes()[-32768:].decode(errors="ignore")
    return {"tail": "\n".join(text.splitlines()[-n:])}


