"""The ONE way a long-lived service runs a subprocess. Never wedges a thread.

WHY THIS EXISTS. `subprocess.run(timeout=...)` does not do what its name
promises. CPython implements the timeout like this:

    try:
        stdout, stderr = process.communicate(input, timeout=timeout)
    except TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()      # <-- NO timeout

That second `communicate()` is unbounded, and `kill()` is a signal — which a
process in uninterruptible D-state cannot receive. So if the child is wedged in
the kernel, `subprocess.run(timeout=2)` blocks the calling thread FOREVER. The
timeout argument buys nothing in exactly the situation it was written for.

That is not theoretical here. Under WSL2, anything touching the GPU goes through
the paravirtualised /dev/dxg and can wedge in D-state while a training job owns
the card. On 2026-08-01 `/api/retrain` was calling nvidia-smi once per poll while
a run trained; every poll ate one of anyio's 40 worker threads permanently, and
about five minutes in the worker stopped serving every `def` endpoint while async
ones still answered in 0.2s. The box ended at load 50 with 36 unkillable
processes and ssh timing out. It was the second occurrence (2026-07-30: 149
stuck, load 171).

WHAT THIS DOES INSTEAD. On timeout: kill, then wait a SHORT grace period, and if
the child still has not died, ABANDON IT. A leaked process that nobody waits on
costs one process table entry. A thread parked in wait4() costs a worker thread
forever, and forty of them cost the whole cockpit. Given the choice, leak.

WHAT IT DELIBERATELY DOES NOT DO. It does not retry. Retrying a wedged call is
how one stuck process becomes forty — the pile on the worker was a poll loop, not a
single bad call. Callers get `timed_out=True` and decide.

THIS IS NOT A LICENCE TO SPAWN ON A POLLED PATH. Bounded is not free: a request
that fires a subprocess every few seconds is still doing something expensive per
poll. Prefer a fact readable from a file (see gpu_facts). Use this where a
service genuinely must shell out — systemctl, journalctl, a one-shot split.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

# How long to wait for a killed child to actually die before walking away. A
# healthy process dies from SIGKILL in microseconds, so anything still here
# after this is wedged in the kernel and will not leave no matter how long we
# wait.
GRACE_S = 0.5


@dataclass(frozen=True)
class Result:
    """What `run` returns. `timed_out` is the field callers must not ignore —
    on timeout stdout is whatever arrived before the deadline, which may be a
    partial line, so it must not be parsed as if it were complete."""
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    abandoned: bool = False          # killed but would not die: leaked on purpose

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run(cmd, timeout: float, *, cwd=None, env=None, text: bool = True) -> Result:
    """Run `cmd`, and return within roughly `timeout + GRACE_S` no matter what.

    Unlike subprocess.run this NEVER blocks indefinitely, because it never waits
    on a process it has already given up on.
    """
    cmd = [str(c) for c in cmd]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=str(cwd) if cwd else None, env=env, text=text)
    except (OSError, ValueError) as e:
        return Result(None, "", f"could not start {cmd[0]}: {e}", False)

    try:
        out, err = p.communicate(timeout=timeout)
        return Result(p.returncode, out or "", err or "", False)
    except subprocess.TimeoutExpired:
        pass

    p.kill()
    try:
        # Bounded, unlike subprocess.run's second communicate(). If the child is
        # in D-state this raises again and we leave it to the OS.
        out, err = p.communicate(timeout=GRACE_S)
        return Result(p.returncode, out or "", err or "", True)
    except subprocess.TimeoutExpired:
        # ABANDONED. Do not touch p again — reading its pipes or waiting on it
        # is the bug this module exists to prevent. The kernel reaps it if it
        # ever comes back.
        return Result(None, "", f"{cmd[0]} did not respond to SIGKILL after "
                                f"{timeout}s — abandoned", True, abandoned=True)


def out_of(cmd, timeout: float, *, cwd=None, default: str = "") -> str:
    """Just the stdout, stripped, or `default` if anything went wrong.

    For the many call sites that want one short string (`systemctl is-active`)
    and have nothing useful to do with a failure. A timeout yields `default`
    rather than a partial read, so a caller comparing == "active" can never be
    fooled by a truncated line.
    """
    r = run(cmd, timeout, cwd=cwd)
    return r.stdout.strip() if r.ok else default
