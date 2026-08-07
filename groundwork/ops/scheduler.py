"""The job table and the loop that ticks it — ONE definition of the platform's
recurring work, however it is executed.

Three execution modes run the SAME table:
  - the Docker `scheduler` service (python -m groundwork.ops.scheduler)
  - native systemd timers, RENDERED from this table by `groundwork install`
  - an in-process thread (GW_SCHEDULER=inline) for `groundwork run`

Every job runs as a BOUNDED SUBPROCESS — the same isolation the systemd
oneshots had. A wedged rsync must not take the loop down, and a hard timeout
is the only bound a wedged child respects (see web/safe_proc's header for why
`subprocess.run(timeout=)` alone is not one).

Every outcome is stamped into outputs/jobs_status.json:

    {job: {"last_ok": epoch|None, "last_error": str|None,
           "running": bool, "at": epoch}}

which is THE interface the dashboard reads (state._mirror_health) — one
contract for all three execution modes, instead of scraping journalctl.

Role predicates keep a fresh single-machine install quiet: with no workers
registered, mirror/adopt/backup skip in one line instead of erroring at a
machine that does not exist.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass

from ..config import OUTPUTS_DIR, ROOT, is_worker

STATUS_PATH = OUTPUTS_DIR / "jobs_status.json"
TICK_S = 60.0


def _has_workers() -> bool:
    from ..web import machines
    return bool(machines.workers())


def _has_backup_target() -> bool:
    from ..web import machines
    return any(machines.registered(k).get("backup_target")
               for k in machines.workers())


def _always() -> bool:
    return True


@dataclass(frozen=True)
class Job:
    name: str
    argv: tuple[str, ...]           # after `python -m`
    every_s: int
    timeout_s: int
    when: object                    # zero-arg predicate
    hq_only: bool = False
    worker_only: bool = False

    def due(self, last: float | None, now: float) -> bool:
        return last is None or (now - last) >= self.every_s

    def applies(self) -> bool:
        if self.hq_only and is_worker():
            return False
        if self.worker_only and not is_worker():
            return False
        try:
            return bool(self.when())
        except Exception:  # noqa: BLE001 — a broken predicate must not stop the loop
            return False


# THE TABLE. `groundwork install` renders the native timers from exactly this,
# so an interval changed here changes every execution mode at once.
JOBS: tuple[Job, ...] = (
    Job("mirror", ("groundwork.ops.mirror",), 300, 900, _has_workers,
        hq_only=True),
    Job("adopt", ("groundwork.dataset.pipeline.adopt_run", "--scan"),
        300, 900, _has_workers, hq_only=True),
    Job("backup", ("groundwork.ops.backup",), 108000, 2700,
        _has_backup_target, hq_only=True),
    Job("clean", ("groundwork.dataset.viz.pending_janitor",), 86400, 300,
        _always),
    Job("autoscore", ("groundwork.ops.autoscore",), 300, 300, _always,
        worker_only=True),
)


def _read_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _stamp(name: str, **fields) -> None:
    d = _read_status()
    d.setdefault(name, {})
    d[name].update(fields, at=time.time())
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def job_status(name: str) -> dict:
    """{last_ok, last_error, running, at} for one job — the dashboard's read."""
    return dict(_read_status().get(name) or {})


def run_job(job: Job) -> bool:
    """One bounded execution; outcome stamped. True on success."""
    _stamp(job.name, running=True)
    cmd = [sys.executable, "-m", *job.argv]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=job.timeout_s)
        ok = r.returncode == 0
        err = None if ok else (r.stderr or r.stdout or "failed").strip()[-400:]
    except subprocess.TimeoutExpired:
        ok, err = False, f"timed out after {job.timeout_s}s"
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    _stamp(job.name, running=False,
           **({"last_ok": time.time(), "last_error": None} if ok
              else {"last_error": err}))
    if not ok:
        print(f"[scheduler] {job.name} FAILED: {err}", flush=True)
    return ok


def loop() -> None:
    """Tick forever. Also the body of the inline-thread mode."""
    print(f"[scheduler] {len(JOBS)} job(s), tick {TICK_S:g}s "
          f"(role={'worker' if is_worker() else 'hq'})", flush=True)
    last: dict[str, float] = {}
    while True:
        now = time.time()
        for job in JOBS:
            if not job.applies() or not job.due(last.get(job.name), now):
                continue
            last[job.name] = now
            run_job(job)
        time.sleep(TICK_S)


if __name__ == "__main__":
    loop()
