"""GPU admission for sidecar jobs: the repo's own gpu.lock, held for the WHOLE run.

The cockpit's retrain/LA jobs take outputs/gpu.lock only during their admission
window and then serialize via their state files (retrain_state.json /
la_state.json). A sidecar job must therefore do BOTH: take the flock
non-blocking (so two sidecars can't race, and any cockpit admission that starts
mid-run blocks until we finish) AND verify both state files are idle first (a
retrain already past admission has released the flock).

Deliberately import-light: no groundwork imports, so it runs identically in
.venv-alt / .venv-mmdet / the main venv.
"""
from __future__ import annotations
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"


def _lockfile() -> Path:
    """Per-card locks, same scheme as retrain_job._lockfile: a job pinned via
    CUDA_VISIBLE_DEVICES locks only its own card, so a 3090 sidecar and a
    5070 Ti cockpit retrain can run at once on the lab. Unpinned (single-card
    laptop) keeps the shared gpu.lock. On a dual-GPU box, ALWAYS pin — an
    unpinned job takes the shared lock, which no pinned job looks at."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return OUTPUTS / (f"gpu{cvd}.lock" if cvd else "gpu.lock")


LOCKFILE = _lockfile()
STATES = [  # (file, pid key, label)
    (OUTPUTS / "retrain_state.json", "owner_pid", "a retrain"),
    (OUTPUTS / "la_state.json", "pid", "a LocateAnything probe"),
]


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def _busy_reason() -> str | None:
    for path, key, label in STATES:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — missing/corrupt state = not running
            continue
        if d.get("status") == "running" and _alive(d.get(key)):
            return f"{label} is running (pid {d.get(key)}) — try again when the Dashboard is idle"
    return None


def peak() -> dict:
    """Peak GPU memory this process reached — BOTH ways, both in GiB.

    ONE DEFINITION, because the fleet already had three and they were compared
    as though they were one (found 2026-08-03):

        yolo   pipeline/train.py   max_memory_RESERVED()  / 1e9    -> decimal GB
        DEIM   its vendor logger   max_memory_ALLOCATED() / 1024^2 -> MiB
        mmdet  mmengine            max_memory_ALLOCATED()          -> MB

    Two axes of disagreement, and both matter. RESERVED is what the caching
    allocator holds; ALLOCATED is what is live inside it, and reserved is always
    the larger — so a table holding one of each ranks families by which metric
    they happen to report. And 1e9 is a DECIMAL gigabyte while card VRAM is
    quoted in GiB, so yolo's 11.19 "GB" is really 10.42 GiB — it was overstating
    its own appetite by 0.8 GiB against a 16 GiB card.

    Both numbers are returned rather than a choice being made here:
      · reserved is the honest planning figure — it is what the process actually
        takes off the card, and what a second job would find missing;
      · allocated is what DEIM and mmdet report, so it is the only number that
        can be compared with the seven families already measured.

    GiB throughout (2^30), because machines.json quotes card VRAM in GiB and the
    whole point of these numbers is fitting one against the other.

    Import-light like the rest of this module: torch is imported HERE, not at
    module scope, so gpu.py still loads identically in .venv-alt, .venv-mmdet
    and the main venv. Returns {} rather than raising — a trainer must never die
    at the finish line over a diagnostic.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
    except Exception:  # noqa: BLE001 — diagnostics never fail a run
        return {}


@contextmanager
def acquire(run_dir: Path, what: str):
    """Hold the GPU for the duration of `with`. Aborts (SystemExit) if busy."""
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCKFILE, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise SystemExit("[gpu] gpu.lock is held (another job mid-admission or a "
                         "sidecar run) — try again later")
    # GW_ALLOW_PARALLEL=1: dual-GPU machines only — the caller vouches that
    # this job is pinned (CUDA_VISIBLE_DEVICES) to a card no other job uses.
    # The single-card state check stays the default everywhere else.
    if os.environ.get("GW_ALLOW_PARALLEL") != "1":
        if reason := _busy_reason():
            f.close()
            raise SystemExit(f"[gpu] {reason}")
    run_dir.mkdir(parents=True, exist_ok=True)
    crumb = run_dir / "gpu_holder.json"
    # WHICH CARD, recorded rather than inferred. The cockpit shows a per-card
    # Cancel button, and without this the crumb says a job exists but not where
    # — so the button would either guess or have to cancel "whatever is running
    # on this machine", which on a dual-GPU box is the wrong job half the time.
    # It is free: the same env var already decides the lock file above.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    crumb.write_text(json.dumps({"pid": os.getpid(), "what": what,
                                 "started": time.time(),
                                 "card": int(cvd) if cvd.isdigit() else None,
                                 "lock": LOCKFILE.name}), encoding="utf-8")
    try:
        yield
    finally:
        crumb.unlink(missing_ok=True)
        f.close()  # releases the flock
