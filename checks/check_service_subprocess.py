"""Every subprocess a long-lived service runs must be unable to wedge a thread.

THE FAILURE THIS PREVENTS (twice now: 2026-07-30, 2026-08-01). A web module
called nvidia-smi with `subprocess.run(timeout=2)`. Under WSL2 that child wedged
in uninterruptible D-state; `subprocess.run` responded by calling kill() (which
D-state ignores) and then communicate() with NO timeout, parking the calling
thread in wait4() forever. On a polled endpoint that meant one dead worker
thread per poll, and FastAPI's `def` handlers all share a 40-thread pool, so the
whole cockpit stopped serving in about five minutes while async routes stayed
fast — which is why it looked like a slow endpoint and not a dead threadpool.

TWO RULES, CHECKED SEPARATELY:

  1. STRUCTURAL - no module under groundwork/web/ calls raw subprocess.run /
     check_output / call. They go through safe_proc, which abandons a child it
     cannot kill instead of waiting on it. The allowlist below is short and each
     entry states why.

  2. BEHAVIOURAL - safe_proc actually returns when the child ignores SIGKILL.
     A real D-state process cannot be created from userspace, so the test
     simulates one by making kill() a no-op and communicate() always time out.
     That is the exact shape of the bug, and the old code hangs the test.

The companion check_no_subprocess_on_poll.py covers the other half: the polled
status path spawns nothing at all.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from groundwork.web import safe_proc  # noqa: E402

FAILURES: list[str] = []

# module -> why raw subprocess is correct there. Anything not listed is a fail.
ALLOWED = {
    "safe_proc.py": "it IS the wrapper",
    "pipeline.py": "retrain_job's worker-process pipeline stages (hours long, "
                   "run inside the detached worker — no request thread "
                   "involved, and blocking on the stage is the whole job)",
    "run_stats.py": "gpu() drives Popen by hand precisely so it cannot wedge; "
                    "documented one-shot, never called from a request",
    "la_job.py": "detached Popen, never waited on",
    "ssh_identity.py": "one-shot local ssh-keygen at pairing — writes a "
                       "keypair, no GPU path, and blocking is the point",
    # mode()'s systemctl probe routes through safe_proc and is cached
    # after the first answer; what remains here is Popen for the children the
    # supervisor OWNS and reaps — spawning is its whole job.
    "botsup.py": "supervisor's own children: detached Popen, reaped by it",
}
BANNED = {"run", "check_output", "call", "check_call"}


def raw_subprocess_calls(path: Path):
    """Call sites of subprocess.<blocking>. AST, not grep — grep matches the
    word inside the comments that explain this very rule, which is how the
    first sweep for this reported false positives."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        FAILURES.append(f"{path}: will not parse: {e}")
        return []
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in BANNED
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"):
            hits.append(node.lineno)
    return hits


print("  [1] structural: raw subprocess under groundwork/web/")
web = ROOT / "groundwork" / "web"
checked = 0
for f in sorted(web.rglob("*.py")):
    hits = raw_subprocess_calls(f)
    checked += 1
    if not hits:
        continue
    if f.name in ALLOWED:
        print(f"      ~ {f.relative_to(ROOT)}:{hits} allowed — {ALLOWED[f.name]}")
    else:
        FAILURES.append(f"{f.relative_to(ROOT)} lines {hits}: raw subprocess in a "
                        f"service module — use safe_proc")
print(f"      scanned {checked} modules")

# The allowlist must not rot into a list of files that no longer exist.
for name in ALLOWED:
    if not any(p.name == name for p in web.rglob("*.py")):
        FAILURES.append(f"allowlist names {name}, which no longer exists")

def executable_strings(path: Path):
    """String literals that are actual VALUES, not docstrings.

    A line-based scan gets this wrong, and did: five modules "failed" purely for
    explaining the trap in prose. The history of why this rule exists is the
    most valuable text in these files and must never be what trips the rule.
    Comments vanish at parse time for free; docstrings need excluding by hand.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docs.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


print("\n  [2] structural: nvidia-smi may not appear on a service OR TRAINER path")
# altmodels/ is scanned too, and that is the whole point of widening this.
# The first version of this check looked only at groundwork/web/, so it passed
# while BOTH challenger trainers polled nvidia-smi every 20 seconds for the
# entire length of every run -- rtmdet.py's copy (which deim.py imported) and
# rfdetr.py's duplicate of it. That watchdog wedged a DEIM run in D-state on
# 2026-08-01. A rule enforced over one directory is not a rule.
SCAN = [web, ROOT / "altmodels", ROOT / "groundwork" / "dataset"]
ALLOWED_SMI = {"run_stats.py"}          # the one hardened, non-polling one-shot
for base in SCAN:
    for f in sorted(base.rglob("*.py")):
        if any("nvidia-smi" in s for s in executable_strings(f)) \
                and f.name not in ALLOWED_SMI:
            FAILURES.append(f"{f.relative_to(ROOT)}: references nvidia-smi outside "
                            f"the one hardened one-shot in run_stats.gpu")
print(f"      scanned {len(SCAN)} trees; only run_stats.gpu may name it "
      f"(prose about it is fine)")

print("\n  [3] behavioural: safe_proc returns even when SIGKILL does nothing")
real_kill = subprocess.Popen.kill
real_comm = subprocess.Popen.communicate
real_wait = subprocess.Popen.wait


def _wait_forever(self, *a, **k):
    """wait4() on a D-state child. subprocess.run's POSIX recovery path calls
    Popen.wait() (NOT communicate(), as an earlier version of this check
    assumed) — so this is the method that has to block, or check [4] passes
    while testing nothing."""
    if k.get("timeout") is not None or a:
        raise subprocess.TimeoutExpired("simulated-d-state", 0.1)
    threading.Event().wait()


def _never_dies(self, *a, **k):
    """A child in D-state, faithfully.

    The FIRST version of this raised TimeoutExpired unconditionally, and check
    [4] then reported that subprocess.run was fine — because run() catches that
    second raise and returns. That would have made check [3] meaningless.

    Reality is asymmetric: communicate(timeout=X) RAISES after X, while
    communicate() with no timeout BLOCKS forever on a process that never dies.
    subprocess.run's recovery path uses the second form, which is precisely why
    its timeout= does not bound anything.
    """
    if k.get("timeout") is not None or (len(a) > 1 and a[1] is not None):
        raise subprocess.TimeoutExpired("simulated-d-state", 0.1)
    threading.Event().wait()          # never returns, like wait4() on D-state


subprocess.Popen.kill = lambda self: None          # D-state ignores signals
subprocess.Popen.communicate = _never_dies
subprocess.Popen.wait = _wait_forever
done, result = threading.Event(), {}


def call():
    result["r"] = safe_proc.run(["true"], timeout=0.2)
    done.set()


threading.Thread(target=call, daemon=True).start()
returned = done.wait(timeout=10)
subprocess.Popen.kill = real_kill
subprocess.Popen.communicate = real_comm
subprocess.Popen.wait = real_wait

if not returned:
    FAILURES.append("safe_proc.run did NOT return against an unkillable child — "
                    "this is the exact hang that took the worker down")
else:
    r = result["r"]
    if not (r.timed_out and r.abandoned):
        FAILURES.append(f"safe_proc returned but did not report abandonment: {r}")
    else:
        print(f"      ✓ returned in time, abandoned=True  ({r.stderr[:60]}…)")

print("\n  [4] prove it can fail: subprocess.run against the same child")
subprocess.Popen.kill = lambda self: None
subprocess.Popen.communicate = _never_dies
subprocess.Popen.wait = _wait_forever
done2 = threading.Event()


def call_old():
    try:
        subprocess.run(["true"], capture_output=True, timeout=0.2)
    except Exception:  # noqa: BLE001
        pass
    done2.set()


threading.Thread(target=call_old, daemon=True).start()
old_returned = done2.wait(timeout=5)
subprocess.Popen.kill = real_kill
subprocess.Popen.communicate = real_comm
subprocess.Popen.wait = real_wait
if old_returned:
    FAILURES.append("subprocess.run survived the unkillable child — the "
                    "simulation is not reproducing the real failure, so check "
                    "[3] proves nothing")
else:
    print("      ✓ subprocess.run HUNG on it, as it did on the worker "
          "(thread abandoned)")

print()
if FAILURES:
    for f in FAILURES:
        print(f"  ✗ {f}")
    sys.exit(1)
print("  all checks passed")
