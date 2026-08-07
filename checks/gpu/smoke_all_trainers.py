"""Train every challenger family for a couple of epochs, ONE AT A TIME.

    .venv/bin/python checks/gpu/smoke_all_trainers.py            # all of them
    .venv/bin/python checks/gpu/smoke_all_trainers.py deimv2-n   # just one
    .venv/bin/python checks/gpu/smoke_all_trainers.py --clean    # delete the runs

    GW_SMOKE_PROJECT=<slug>    which project to train (default: first project)
    GW_SMOKE_MACHINE=<key>     which machine to dispatch to (default: here)
    GW_SMOKE_CARD=<n>          which card (default: 1)

WHY. Multiple families are wired to train, and some may never have been run at
all — `trainer_module` is set and `check_model_registry` proves the argv uses
flags the trainer declares, but "the parser accepts it" is not "it trains". The
gap between those two is where an evening goes: a missing pretrained
checkpoint, a config key mmdet renamed, a venv without a dependency. Two
epochs finds all of that in minutes, because everything that breaks
structurally breaks at import, at the first batch, or at the first checkpoint
write.

ONE AT A TIME, deliberately. The point is a clean answer per family, and two
runs sharing a card would let one starve the other and turn "it works" into
"it timed out". The per-card locks would serialise them anyway — this just
makes the waiting visible.

yolov8n IS SKIPPED. It is proven end to end by real use (dispatch -> sync ->
worker split -> train -> holdout eval -> adopt -> ledger -> UI) and it is the
one family whose short run is not short: retrain_job clamps epochs to a 50-600
band, so a "2 epoch" yolo run is 50 epochs, and it would leave a junk row in a
ledger that is not throwaway.

EVERY RUN IS NAMED `zzz-smoke-*` so `--clean` can find them, and so nothing
here can be mistaken for a real experiment in the runs table.

MANUAL, NEEDS A GPU AND A RUNNING COCKPIT — deliberately excluded from the CI
set (checks/run.py skips checks/gpu/).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from groundwork.models import registry            # noqa: E402
from groundwork import project as project_mod     # noqa: E402
from groundwork.dataset import paths              # noqa: E402

BASE = os.environ.get("GW_BASE", "http://127.0.0.1:8000")
# A default cockpit has auth ON, so without a key every launch 401s and the
# whole matrix reads "REFUSED" for a reason that has nothing to do with GPUs.
_KEY = os.environ.get("GW_KEY", "")
_AUTH = {"Authorization": f"Bearer {_KEY}"} if _KEY else {}
EPOCHS = 2
PREFIX = "zzz-smoke-"
# One fixed card so a comparison between two venv builds of one family is
# about the venv and nothing else.
CARD = int(os.environ.get("GW_SMOKE_CARD", "1"))
MACHINE = os.environ.get("GW_SMOKE_MACHINE", "here")


def _project() -> str:
    slug = os.environ.get("GW_SMOKE_PROJECT")
    if slug:
        return slug
    slugs = project_mod.slugs()
    if not slugs:
        raise SystemExit("no projects exist — create one, or set GW_SMOKE_PROJECT")
    return slugs[0]


def _post(path: str, body: dict | None = None, timeout: int = 300):
    req = urllib.request.Request(
        BASE + path, method="POST",
        data=json.dumps(body).encode() if body is not None else None,
        headers={**({"Content-Type": "application/json"}
                     if body is not None else {}), **_AUTH})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return False, json.loads(e.read() or b"{}").get("detail", str(e))
        except Exception:  # noqa: BLE001
            return False, f"HTTP {e.status}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _get(path: str, timeout: int = 60, base: str | None = None):
    try:
        req = urllib.request.Request((base or BASE) + path, headers=_AUTH)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception:  # noqa: BLE001
        return {}


def _worker() -> str:
    """The base URL of the machine the runs are actually ON.

    Asking HQ about a remote run's output reported "NO OUTPUT, wrote nothing"
    for two runs that had trained perfectly well — a remote run's directory
    lives on the worker until the adopt job brings it home, and HQ's
    /api/lab/runs reads HQ's own alt dir. The check has to face the machine
    that ran it.
    """
    from groundwork.web import machines
    m = machines.get(MACHINE)
    return (m.url if m and m.url else None) or BASE


def _families(only: list[str]) -> list:
    out = []
    for m in registry.MODELS:
        if not m.trainer_module or m.name == "yolov8n":
            continue
        if only and m.name not in only:
            continue
        out.append(m)
    return out


def _wait_idle(limit_s: int = 2700) -> str:
    """Block until no challenger holds a card there. Returns why it stopped."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < limit_s:
        st = _get("/api/lab/status", base=_worker())
        a = st.get("active")
        if not a:
            return "idle"
        line = f"{a.get('run')} epoch {a.get('epoch')}/{a.get('total')}"
        if line != last:
            print(f"      …{line}", flush=True)
            last = line
        time.sleep(20)
    return "timeout"


def clean() -> None:
    """Remove every zzz-smoke-* run directory. Named prefix only, never a glob
    that could reach a real run."""
    alt = paths.for_project(_project()).ALT_DIR
    import shutil
    n = 0
    for d in sorted(alt.glob(f"{PREFIX}*")):
        assert d.name.startswith(PREFIX), d
        shutil.rmtree(d)
        print(f"  removed {d}")
        n += 1
    print(f"  {n} smoke run(s) removed")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--clean" in sys.argv:
        clean()
        return
    fams = _families(args)
    slug = _project()
    print(f"[smoke] {len(fams)} famil(ies), {EPOCHS} epochs each, one at a "
          f"time, project {slug}, machine {MACHINE}, card {CARD}\n", flush=True)
    results = []
    for m in fams:
        run = f"{PREFIX}{m.name}"
        print(f"  {m.name}: waiting for a free card…", flush=True)
        why = _wait_idle()
        if why != "idle":
            results.append((m.name, "SKIPPED", "card never freed"))
            continue
        print(f"  {m.name}: starting {run} ({m.default_imgsz}px)", flush=True)
        ok, d = _post(f"/api/train?project={slug}",
                      {"model": m.name, "machine": MACHINE, "card": CARD,
                       "imgsz": m.default_imgsz, "epochs": EPOCHS,
                       "run_name": run})
        if not ok:
            print(f"    REFUSED: {d}", flush=True)
            results.append((m.name, "REFUSED", str(d)[:120]))
            continue
        if d.get("ok") is False:
            print(f"    REFUSED: {d.get('error')}", flush=True)
            results.append((m.name, "REFUSED", str(d.get("error"))[:120]))
            continue
        # WAIT FOR IT TO APPEAR BEFORE WAITING FOR IT TO FINISH. The launch is
        # detached and the gpu_holder crumb lands a moment later, so an
        # immediate "is anything running?" answers NO and the family gets marked
        # "NO OUTPUT in 0s" — while the very next family's wait then shows that
        # same run training. A race that reports the opposite of the truth.
        t0 = time.time()
        for _ in range(60):
            if (_get("/api/lab/status", base=_worker()).get("active") or {}).get("run") == run:
                break
            time.sleep(2)
        else:
            print("    never appeared in /api/lab/status after 2 min", flush=True)
        _wait_idle()
        took = round(time.time() - t0)
        # DID IT LEAVE ANYTHING BEHIND — asked of the machine that RAN it.
        # This looked at HQ's own alt dir and reported "NO OUTPUT, wrote
        # nothing" for two runs that had trained perfectly well remotely: a
        # remote run's directory is over there until the adopt job brings it
        # home. The API answers for whichever machine served it.
        det = _get(f"/api/lab/runs/{run}/detail", base=_worker()) or {}
        trained = bool(det.get("meta") or det.get("count_eval"))
        state = "TRAINED" if trained else "NO OUTPUT"
        meta = det.get("meta") or {}
        note = (f"{took}s · arch={meta.get('arch','?')} "
                f"train={meta.get('train_seconds','?')}s") if trained else f"{took}s · nothing"
        print(f"    {state} — {note}\n", flush=True)
        results.append((m.name, state, note))

    print("\n[smoke] ---- summary ----")
    for name, state, note in results:
        print(f"  {name:16s} {state:10s} {note}")


if __name__ == "__main__":
    main()
