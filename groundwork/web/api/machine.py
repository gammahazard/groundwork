"""What THIS machine is, answered by the machine itself.

THE SEED OF THE MACHINE REGISTRY. `outputs/machines.json` is written today by
`scratch/probe_machines.py` over ssh FROM HQ, and machines.py's own docstring
says why: "HQ cannot ask the worker directly, because the worker's cockpit is behind
this branch and has no endpoint that would answer." This is that endpoint.

Once every cockpit answers it, adding a machine stops needing ssh keys and a
manual probe run: HQ registers a host plus an API key, asks it what it has, and
writes the answer into machines.json. A newly added machine currently reports
zero cards, which makes it unusable with no hint — `_default_card` returns None
and `/api/train` refuses — so this closes a real dead end rather than adding a
convenience.

NOT THE SAME QUESTION AS /api/train/options. That one asks "what may run WHERE",
across every machine, from HQ's recorded map. This asks "what am I", from the
hardware in front of it, and is the thing that map is made of.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...config import MACHINE
from ... import project as project_mod
from .. import machine_self, machines as machines_mod
from ..auth.routes import require_admin

router = APIRouter()


@router.get("/api/machine/self")
def self_info():
    """Address and identity — cheap, and safe to call on any page.

    `url` is what ANOTHER machine would use to reach this cockpit, and it exists
    because the browser's own origin is the wrong answer to that question: an
    operator sitting at the box sees `http://localhost:8000`, and a curl example
    built from it works nowhere except the one machine that does not need it.
    """
    from ...config import ROOT, role
    from .. import machine_self
    return {"machine": MACHINE,
            "ip": machines_mod.self_ip(),
            "url": machines_mod.self_url(),
            # Pairing facts — see machine_self.describe: role decides which
            # wizard the Machines tab shows, root is what HQ records as
            # remote_root, and the host keys close the TOFU gap.
            "role": role(), "root": str(ROOT),
            "ssh_host_keys": machine_self._host_keys()}


@router.get("/api/machine/status")
def machine_status():
    """Machine-scope busy answer — deliberately NO project dimension.

    "Is a card busy" is a fact about the BOX: a GPU crumb belongs to whichever
    project got there first, and a retrain's state file is one per machine. The
    old shape forced this question through project-scoped endpoints, which
    meant naming a slug the remote might never have heard of — a 404 that read
    as "machine unreachable". This endpoint needs no slug at all, so a fresh
    instance with zero projects still answers it.
    """
    from .. import retrain_job
    from . import lab_ops
    retrain = retrain_job.status()
    act = lab_ops._local_active()
    return {"active": act,
            "retrain": {"status": retrain.get("status"),
                        "project": retrain.get("project"),
                        "step": retrain.get("step")},
            "busy": bool(act) or retrain.get("status") == "running"}


@router.get("/api/version")
def version():
    """What code this machine is running.

    THE WORKER BEING BEHIND HQ IS THIS REPO'S RECURRING FAILURE — it gates remote
    CARD choice, it made /api/machine/self answer 404 for a build that could not
    describe itself, and "is the worker updated?" has been answered all session by
    ssh and git log. A script cannot do that; it can call this.

    Read from git, and NOT with a subprocess: a service shells out only through
    safe_proc, and this would be on the one call a monitoring loop makes most
    often. .git/HEAD and the ref file are two small reads that cannot wedge.
    """
    from ...config import ROOT
    head = ROOT / ".git" / "HEAD"
    commit = branch = None
    try:
        raw = head.read_text().strip()
        if raw.startswith("ref: "):
            ref = raw[5:].strip()
            branch = ref.rsplit("/", 1)[-1]
            p = ROOT / ".git" / ref
            if p.exists():
                commit = p.read_text().strip()
            else:                       # packed-refs, after a gc or a fresh clone
                packed = ROOT / ".git" / "packed-refs"
                if packed.exists():
                    for line in packed.read_text().splitlines():
                        if line.endswith(" " + ref):
                            commit = line.split(" ", 1)[0]
                            break
        else:
            commit = raw               # detached HEAD
    except OSError:
        pass
    return {"commit": commit, "short": (commit or "")[:9] or None,
            "branch": branch,
            # Named so two machines can be compared without either knowing the
            # other's layout — this is what a parity check reads.
            "machine": machines_mod.here().name}


@router.get("/api/machine/cards")
def cards(read: bool = False):
    """This machine's GPUs and venvs, in the shape machines.json already stores.

    SLOW BY NATURE — up to ~10s cold, because reading each venv's compiled arch
    list means starting that interpreter and importing torch. It is cached for
    five minutes and is called by a probe rather than by a page, which is the
    shape that makes the cost acceptable. `/api/machine/self` is the instant one.

    `read=false` by default, and that default is the safety property rather than
    a performance one. Enumerating cards calls into the driver — under WSL2 that
    crosses the paravirtualised /dev/dxg, where a blocked thread sits in
    uninterruptible D-state and cannot be killed by anything. This fleet has been
    taken down twice that way. So the cheap answer (identity + venv arch lists,
    which touch no device) is what you get unless you ask.

    Even with read=true it refuses while a GPU job is running, and says so —
    `cards_stale` with a reason, never a hang and never an empty list, because
    empty would read as "this machine has no GPUs" when the truth is that it has
    one and is using it.
    """
    return machine_self.describe(cards=read, venvs=True)


# ======================================================================== #
# THE REGISTRY: add a machine, probe it, remove it.
# ======================================================================== #

class NewMachine(BaseModel):
    name: str
    url: str                       # http://host:port — where its cockpit is
    api_key: str = ""              # minted ON THAT MACHINE; HQ stores it to send
    ssh_host: str = ""             # user@host, for the dataset rsync
    remote_root: str = ""          # its checkout path, for adopt_run


def _busy_reason(key: str) -> str | None:
    """Is this machine training, or can we not tell? Both block a change.

    UNKNOWN IS NOT IDLE — the rule this repo applies everywhere a job might be
    running. A machine we cannot interrogate might be mid-run, and re-pointing or
    deleting it then would strand that run: HQ would have no URL to cancel it
    through, no host to adopt it from, and `/api/train` would happily send the
    next job somewhere else entirely.

    Asked directly rather than through lab_proxy's cache: a cold cache returns
    nothing at all, which is indistinguishable from "idle" and is exactly how a
    stale read once let a live run be declared finished.
    """
    m = machines_mod.get(key)
    if m is None:
        return None
    if m.local:
        from .. import retrain_job
        from . import lab_ops
        if retrain_job.status().get("status") == "running":
            return "a yolo retrain is running here"
        act = lab_ops._local_active()
        return f"{act['run']} is training here" if act else None
    try:
        from . import train_dispatch
        r = train_dispatch._remote(m, "/api/machine/status", None, None,
                                   method="GET", timeout=20.0)
        act = (r or {}).get("active") or {}
        if act.get("run"):
            return f"{act['run']} is training on {m.name}"
        if ((r or {}).get("retrain") or {}).get("status") == "running":
            return f"a yolo retrain is running on {m.name}"
    except HTTPException as e:
        return (f"could not ask {m.name} what it is doing ({e.detail}) — "
                f"refusing to change a machine I cannot see")
    return None


@router.get("/api/machines")
def list_machines(_: str = Depends(require_admin)):
    """Built-ins and registered ones. NEVER the stored API keys."""
    out = []
    for key, m in machines_mod.all_machines().items():
        reg = machines_mod.registered(key)
        out.append({
            "key": key, "name": m.name, "what": m.what, "local": m.local,
            "url": m.url, "trains": m.trains,
            "builtin": key in machines_mod.MACHINES,
            "has_key": bool(reg.get("api_key")),
            "ssh_host": reg.get("ssh_host", ""),
            "remote_root": reg.get("remote_root", ""),
            "cards": machines_mod.cards(key),
            "measured": machines_mod.measured_at(key),
        })
    return {"machines": out}


@router.post("/api/machines")
def add_machine(body: NewMachine, _: str = Depends(require_admin)):
    """Register a machine, or update one. ADMIN ONLY, and never mid-run."""
    key = re.sub(r"[^a-z0-9-]+", "-", (body.name or "").strip().lower()).strip("-")
    if not key or len(key) > 40:
        raise HTTPException(422, "give it a short name (letters, digits, dashes)")
    if key in machines_mod.MACHINES:
        raise HTTPException(409, f"{key!r} is a built-in machine name — pick "
                                 f"another, or every saved reference to it would "
                                 f"quietly start meaning a different host")
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(422, "url must start with http:// or https://")
    if (why := _busy_reason(key)):
        raise HTTPException(409, f"{why}. Changing where a machine points while "
                                 f"it is training would strand that run.")
    machines_mod.add_machine(key, body.name.strip(), body.url, body.api_key,
                             body.ssh_host.strip(), body.remote_root.strip())
    print(f"[machines] registered {key!r} -> {body.url}", flush=True)
    return {"ok": True, "key": key}


@router.delete("/api/machines/{key}")
def remove_machine(key: str, force: bool = False,
                   _: str = Depends(require_admin)):
    """Forget a machine. Refused while it is training, or while we cannot tell.

    `force` EXISTS BECAUSE THE STRICT VERSION WAS A TRAP. "Unknown is not idle"
    is the right default — an unreachable box might be mid-run, and deleting it
    then leaves that run with no URL to cancel through and no host to adopt from.
    But applied without an override it means a machine that is genuinely DEAD can
    never be removed: the check can only ever fail, so the entry is permanent.
    Measured, not imagined — a registered host at an unroutable address refused
    its own deletion.

    So: refuse by default, say exactly why, and let an admin who knows the box is
    off override it. Same shape as users.remove(force=True), and for the same
    reason — someone with this much access can edit the file anyway, so this is a
    guardrail rather than a lock.
    """
    if key in machines_mod.MACHINES:
        raise HTTPException(409, f"{key!r} is built in and cannot be removed")
    if not force and (why := _busy_reason(key)):
        raise HTTPException(409, f"{why}. Removing it now would leave that run "
                                 f"with no way to be cancelled or adopted. Pass "
                                 f"force=true if you know the machine is off.")
    if not machines_mod.remove_machine(key):
        raise HTTPException(404, f"no machine {key!r}")
    return {"ok": True}


@router.post("/api/machines/{key}/sync")
def sync_now(key: str, project: str | None = None,
             _: str = Depends(require_admin)):
    """Push dataset(s) to one worker RIGHT NOW — the "I just labelled and want
    to train on it immediately" moment between mirror ticks.

    HQ-INITIATED, deliberately. The old shape was a worker-side /api/resync
    that ssh'd back into HQ, which meant the worker held a credential for the
    control plane. Credentials flow one way now: HQ holds the worker's, the
    worker holds nothing of HQ's.
    """
    m = machines_mod.get(key)
    if m is None or m.local:
        raise HTTPException(404, f"no remote machine {key!r}")
    from ...ops import mirror as mirror_mod
    from ... import project as project_mod
    from .. import safe_proc
    slugs = [project] if project else project_mod.slugs()
    results = []
    for s in slugs:
        try:
            results.append(mirror_mod.sync_one(s, key, run=safe_proc.run))
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "slug": s, "error": f"{type(e).__name__}: {e}"})
    ok = all(r.get("ok") for r in results)
    return {"ok": ok, "results": results}


@router.post("/api/machines/{key}/test")
def test_data_plane(key: str, _: str = Depends(require_admin)):
    """Prove the DATA PLANE works before anything depends on it: one ssh echo,
    then an rsync --dry-run against the recorded remote root. Failures come
    back verbatim — "key not installed", "wrong host" and "wrong root" all
    look different in ssh's own words, and paraphrasing them helps nobody."""
    m = machines_mod.get(key)
    if m is None or m.local:
        raise HTTPException(404, f"no remote machine {key!r}")
    from .. import safe_proc, ssh_identity
    try:
        argv = ssh_identity.ssh_argv(key)
        host, root = ssh_identity.target(key)
    except KeyError as e:
        raise HTTPException(422, str(e)) from None
    r = safe_proc.run([*argv, host, "echo", "gw-ok"], timeout=30)
    if r.returncode != 0 or "gw-ok" not in (r.stdout or ""):
        return {"ok": False, "step": "ssh",
                "error": (r.stderr or "no answer").strip()[:400]}
    r2 = safe_proc.run(["rsync", "--dry-run", "-a",
                        "-e", ssh_identity.ssh_option_string(key),
                        str(ssh_identity.KNOWN_HOSTS),
                        f"{host}:{root}/outputs/"], timeout=60)
    if r2.returncode != 0:
        return {"ok": False, "step": "rsync",
                "error": (r2.stderr or "rsync failed").strip()[:400]}
    machines_mod.mark_verified(key)
    return {"ok": True, "verified": True}


@router.post("/api/machines/{key}/probe")
def probe(key: str, _: str = Depends(require_admin)):
    """Ask a machine what it has, and write it into machines.json.

    THIS IS WHAT REPLACES THE SSH PROBE. scratch/probe_machines.py exists because
    "the worker's cockpit has no endpoint that would answer"; now it does, so
    registering a machine no longer needs shell on it. The payload is the SAME
    shape the ssh probe writes, so `machines.cards()` and `venv_info()` read it
    with no translation layer to drift.

    A machine that is TRAINING answers with its venv list and its previous cards,
    flagged stale — reading card properties crosses /dev/dxg and can wedge in
    D-state, so neither side does it mid-run.
    """
    m = machines_mod.get(key)
    if m is None:
        raise HTTPException(404, f"no machine {key!r}")
    if m.local:
        d = machine_self.describe(cards=True, venvs=True)
    else:
        reg = machines_mod.registered(key)
        req = urllib.request.Request(
            f"{m.url}/api/machine/cards?read=true",
            headers={"Authorization": f"Bearer {reg.get('api_key','')}"}
            if reg.get("api_key") else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            # 404 AND 401 ARE DIFFERENT PROBLEMS with different fixes, and
            # blaming the key for both sends someone to check a credential when
            # the machine simply predates the endpoint. Measured 2026-08-05: the
            # worker answers /api/retrain with 200 and /api/machine/cards with 404.
            if e.code == 404:
                raise HTTPException(
                    502, f"{m.name} has no /api/machine/cards — it is running an "
                         f"older Groundwork that cannot describe itself. Deploy "
                         f"the current code there, then probe again.") from None
            if e.code in (401, 403):
                raise HTTPException(
                    502, f"{m.name} refused the probe ({e.code}) — its API key is "
                         f"missing or wrong. Create one ON that machine with "
                         f"`python -m groundwork.web.auth.keys add hq` and paste "
                         f"it into this machine's entry.") from None
            raise HTTPException(502, f"{m.name} refused the probe ({e.code})") from None
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"could not reach {m.name} at {m.url} "
                                     f"({type(e).__name__})") from None
    doc = {}
    try:
        doc = json.loads(machines_mod.MAP_PATH.read_text())
    except Exception:  # noqa: BLE001
        pass
    # NEVER OVERWRITE A KNOWN INVENTORY WITH NOTHING.
    #
    # This wrote `d.get("cards") or []` unconditionally, and a machine that is
    # TRAINING deliberately answers with no cards — enumerating them crosses
    # /dev/dxg, which can wedge in unkillable D-state, so neither side reads
    # card properties mid-run. So probing a busy machine ERASED the cards HQ had
    # measured when it was idle, and the Train control then offered no card at all
    # for the machine that does all the training (measured
    # 2026-08-05).
    #
    # The docstring above always claimed a busy machine "answers with its
    # previous cards". It can only do that if IT has them stored, and the worker
    # has no machines.json — that file is written on HQ. The previous answer
    # HQ needs is HQ's own, and it is right here.
    prev = (doc.get("machines") or {}).get(key) or {}
    fresh = d.get("cards") or []
    kept = fresh or (prev.get("cards") or [])
    # Through the locked, atomic writer — a probe and a finishing stack install
    # (which re-measures the venvs) are two writers to one file, and the Extras
    # card can have three installs in flight at once. `cards_stale` is written
    # EITHER WAY rather than omitted when false: this merges into the existing
    # row now, so an omitted key would leave a previous probe's staleness
    # standing over freshly measured cards.
    machines_mod.update_entry(
        key,
        venvs=d.get("venvs") or prev.get("venvs") or {},
        cards=kept,
        busy=bool(d.get("busy")), measured=time.time(),
        # Stale when the machine refused to look, OR when we are keeping an
        # older list because it returned none. Either way the UI must not
        # present these as just-measured.
        cards_stale=bool(d.get("cards_stale") or (not fresh and kept)))
    # FINISH THE PAIRING FROM THE ONE BUTTON THAT EXISTS. When enroll's
    # verification failed (wrong address, dead network — both measured), the
    # machine sat "not finished pairing" with a message pointing at a tab
    # whose only control was this probe — which never verified, so the dead
    # end was permanent. The worker's describe() already carries its sshd
    # host keys, so the probe can complete the same chain enroll runs:
    # record keys → ssh echo → rsync dry-run → mark verified.
    verify = None
    if not m.local:
        reg2 = machines_mod.registered(key)
        if reg2.get("ssh_host") and not reg2.get("verified"):
            from .. import ssh_identity
            hks = d.get("ssh_host_keys") or []
            if hks:
                try:
                    ssh_identity.record_host_keys(reg2["ssh_host"], hks)
                except Exception as e:  # noqa: BLE001 — test below will say more
                    print(f"[machines] host-key record for {key}: {e}",
                          flush=True)
            try:
                verify = test_data_plane(key, _)
            except HTTPException as e:
                verify = {"ok": False, "error": str(e.detail)}
    return {"ok": True, "key": key, "cards": kept,
            "cards_stale": bool(d.get("cards_stale") or (not fresh and kept)),
            "verify": verify,
            "why": d.get("why") or ("kept the previously measured cards — it is "
                                    "busy and would not re-read them" if not fresh
                                    and kept else None)}
