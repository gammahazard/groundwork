"""One authority per card: /api/train must refuse a second GPU job, both ways.

    .venv/bin/python checks/check_gpu_gate.py

WHY THIS EXISTS. On the predecessor's training box, a POST to /api/train
started a retrain on card 0 while a challenger was training on card 1. Sixty
minutes later the challenger died at epoch 292 of 300 with `CUDA error:
unspecified launch failure` — two hours of GPU, eight epochs from done, no
meta.json, no score.

The guard was ASYMMETRIC, which is why nobody noticed it was missing:

    challenger launching into a running retrain   -> guarded (lab_ops.train)
    retrain launching into a running challenger   -> guarded by NOTHING

The driver that sent that POST had its own gate, and that gate blocked only on
one model family. A gate one caller consults is decoration; the rule has to
sit at the entry point every caller goes through.

This asserts the gate in BOTH directions, asserts that "I could not ask" is
refused rather than treated as idle, and asserts that allow_concurrent still
lets a deliberate experiment through. It monkeypatches the two probes rather
than needing a live GPU, so it runs any time — including while a card is busy,
which is exactly when someone would want to check it.

READ-ONLY AND GPU-FREE: it never calls start(), only the predicates start()
consults, plus a source-level check that start() consults them at all.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import HTTPException                            # noqa: E402
from groundwork.web.api import train_dispatch as td          # noqa: E402
from groundwork.web.api.train_dispatch import busy as busy_mod    # noqa: E402
from groundwork.web.api.train_dispatch import remote as remote_mod  # noqa: E402

FAIL: list[str] = []


def note(ok: bool, msg: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {msg}")
    if not ok:
        FAIL.append(msg)


class _Mach:
    """Enough of a machine for the predicates. local=False forces the remote path."""
    def __init__(self, local: bool):
        self.local, self.name = local, "workerbox"
        self.key, self.url = "workerbox", "http://x"


def main() -> None:
    print("[1] the predicates read what the machine reports")
    # The probes import _remote from .remote AT CALL TIME, so the stub goes on
    # the remote module — patching the package attribute would change nothing,
    # which is itself worth knowing when editing this file.
    remote_mod._remote = lambda *a, **k: {"active": {"run": "chal-1280-seed3"}}
    run, why = td._busy_on(_Mach(False), "someproj")
    note(run == "chal-1280-seed3" and why is None,
         f"a live challenger is seen by name ({run})")
    remote_mod._remote = lambda *a, **k: {"active": None}
    run, why = td._busy_on(_Mach(False), "someproj")
    note(run is None and why is None, "an idle challenger card reads as idle")
    remote_mod._remote = lambda *a, **k: {"retrain": {"status": "running"}}
    busy, why = td._yolo_busy_on(_Mach(False), "someproj")
    note(busy is True and why is None, "a live yolo retrain is seen")

    print("\n[2] UNKNOWN IS NOT IDLE — the failure that makes a gate useless")

    def _boom(*a, **k):
        raise HTTPException(502, "could not reach the worker")
    remote_mod._remote = _boom
    run, why = td._busy_on(_Mach(False), "someproj")
    note(run is None and why, f"an unreachable machine returns a REASON ({bool(why)})")
    busy, why2 = td._yolo_busy_on(_Mach(False), "someproj")
    note(busy is False and why2, "…and so does the retrain probe")
    remote_mod._remote = lambda *a, **k: "not a dict"
    run, why = td._busy_on(_Mach(False), "someproj")
    note(run is None and why, "an unreadable answer is a reason, not 'idle'")

    print("\n[3] the gate REFUSES — executed, not read")

    class _Req:
        def __init__(self, allow=False):
            self.allow_concurrent = allow

    class _M:
        def __init__(self, name):
            self.name = name

    def _refuses(model, busy_run, yolo_running, allow=False):
        """Call the real gate with stubbed probes; return the HTTPException or None."""
        # refuse_if_busy reads the probes from ITS module's globals (remote.py
        # imported them at module top), so the stubs go there.
        remote_mod._busy_on = lambda *a, **k: (busy_run, None)
        remote_mod._yolo_busy_on = lambda *a, **k: (yolo_running, None)
        try:
            td.refuse_if_busy(_Req(allow), _M(model), _Mach(False), "someproj")
            return None
        except HTTPException as e:
            return e

    e = _refuses("yolov8n", "chal-1280-seed3", False)
    note(e is not None and e.status_code == 409,
         f"a RETRAIN into a live challenger is refused ({e.status_code if e else 'ALLOWED'})"
         " — the exact direction that was unguarded")
    note(e is not None and "chal-1280-seed3" in str(e.detail),
         "…and the refusal names the run that is in the way")
    e = _refuses("yolox-s", None, True)
    note(e is not None and e.status_code == 409,
         f"a CHALLENGER into a live retrain is refused ({e.status_code if e else 'ALLOWED'})")
    note(_refuses("yolov8n", None, False) is None, "an idle machine is allowed through")
    note(_refuses("yolov8n", "yolox-s", False, allow=True) is None,
         "allow_concurrent still lets a deliberate experiment run")

    remote_mod._busy_on = lambda *a, **k: (None, "could not reach the worker")
    remote_mod._yolo_busy_on = lambda *a, **k: (False, "could not reach the worker")
    try:
        td.refuse_if_busy(_Req(), _M("yolov8n"), _Mach(False), "someproj")
        e = None
    except HTTPException as ex:
        e = ex
    note(e is not None and e.status_code == 502,
         f"a machine we CANNOT ASK is refused, not assumed idle ({e.status_code if e else 'ALLOWED'})")

    src = (ROOT / "groundwork/web/api/train_dispatch/routes.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "start")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    note("refuse_if_busy" in called, "start() calls the gate")
    note(src.index("refuse_if_busy(req, m, machine, p.slug)")
         < src.index("mirror_mod.sync_one"),
         "the gate runs before the sync, not after")

    print("\n[4] prove it can fail — the mutation the first version missed")
    # Neutering the gate with `if False:` leaves every call and raise textually in
    # place, so the AST check this replaced went green on a defeated guard. Here
    # the gate is EXECUTED, so the same mutation is caught.
    # The module cannot be exec'd standalone (relative imports), so mutate just
    # the gate's own source and run THAT with the probes injected. Same mutation
    # that defeated the first version of this check.
    import inspect
    fn_src = inspect.getsource(td.refuse_if_busy)
    mutated = fn_src.replace("    if req.allow_concurrent:\n        return\n",
                             "    if True:\n        return\n", 1)
    note(mutated != fn_src, "the mutation applies (the early-return is findable)")
    ns = {"HTTPException": HTTPException,
          "_busy_on": lambda *a, **k: ("yolox-s", None),
          "_yolo_busy_on": lambda *a, **k: (False, None)}
    exec(compile(mutated, "<mutated>", "exec"), ns)
    try:
        ns["refuse_if_busy"](_Req(), _M("yolov8n"), _Mach(False), "someproj")
        refused = False
    except HTTPException:
        refused = True
    note(not refused,
         "a gate mutated to always-return does NOT refuse — so this check can "
         "tell a working gate from a defeated one")
    # …and the UNmutated source, run the same way, DOES refuse.
    ns2 = dict(ns)
    exec(compile(fn_src, "<real>", "exec"), ns2)
    try:
        ns2["refuse_if_busy"](_Req(), _M("yolov8n"), _Mach(False), "someproj")
        refused2 = False
    except HTTPException:
        refused2 = True
    note(refused2, "the real gate, run identically, DOES refuse")

    print()
    if FAIL:
        print("GPU GATE IS BROKEN:")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)
    print("one authority per card: both directions refuse, unknown is not idle")


if __name__ == "__main__":
    main()
