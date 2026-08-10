"""The DEIM entry must load the way torchrun loads it, and the vendor path must
resolve without importing the main package.

    .venv/bin/python checks/check_deim_vendor.py

WHY THIS EXISTS. `altmodels/deim_entry.py` is EXECUTED AS A SCRIPT —
`trainers/deim.py` runs `torchrun … altmodels/deim_entry.py` with the cwd set
to the vendor repo — so `sys.path[0]` is `altmodels/` and the cwd is somebody
else's tree. Neither makes `altmodels` importable. A repo import added at the
top of that file therefore works everywhere it is tested from the repo root and
fails only in the one place it runs: measured 2026-08-10, a DEIM launch did the
dataset sync, the split and the COCO convert, held the 3090, and then died at
rank 0 with `ModuleNotFoundError: No module named 'altmodels'`.

The two rules this pins:

  1. deim_entry.py loads under SCRIPT semantics. Run the real file in a
     subprocess with a foreign cwd and no vendor repo: reaching the vendor's
     train.py (a FileNotFoundError) proves the whole import block executed;
     ModuleNotFoundError proves it did not.
  2. deim_vendor imports NO `groundwork`. It loads inside the sidecar venvs,
     which hold a different torch build and none of the main package's
     dependencies — so `import groundwork.config` there is an ImportError
     waiting for the next machine. Asserted by observing sys.modules after the
     import, not by reading the source.

READ-ONLY AND GPU-FREE: temp dirs, a fake HOME, no training.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS = 0


def expect(cond: bool, what: str) -> None:
    global FAILS
    print(("  ok  " if cond else "  FAIL") + f" {what}")
    if not cond:
        FAILS += 1


def main() -> int:
    entry = ROOT / "altmodels" / "deim_entry.py"

    print("[1] deim_entry.py under torchrun's script semantics")
    with tempfile.TemporaryDirectory() as td:
        # A foreign cwd, a HOME with no ~/DEIMv2, and a data dir with no
        # vendor/ — so BOTH vendor candidates are absent and the run can only
        # die at the vendor's train.py, after every import has succeeded.
        env = {**os.environ, "HOME": td, "GW_DATA_DIR": td}
        r = subprocess.run([sys.executable, str(entry)], cwd=td, env=env,
                           capture_output=True, text=True, timeout=120)
        err = (r.stderr or "") + (r.stdout or "")
        expect("No module named 'altmodels'" not in err,
               "the repo is importable from a file run by path "
               "(sys.path[0] is altmodels/, cwd is elsewhere)")
        expect("train.py" in err and "DEIMv2" in err,
               f"...and execution reached the vendor entry, so the whole "
               f"import block ran (got: {err.strip().splitlines()[-1][:90]!r})")

    print("[2] the vendor path resolves with no groundwork import")
    with tempfile.TemporaryDirectory() as td:
        probe = (
            "import sys, json;"
            "from altmodels.deim_vendor import vendor_dir;"
            "print(json.dumps({'dir': str(vendor_dir()),"
            " 'gw': [m for m in sys.modules if m.startswith('groundwork')]}))")
        pinned = Path(td) / "vendor" / "DEIMv2"
        pinned.mkdir(parents=True)
        r = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT),
                           env={**os.environ, "GW_DATA_DIR": td, "HOME": td},
                           capture_output=True, text=True, timeout=60)
        out = json.loads(r.stdout.strip() or "{}") if r.returncode == 0 else {}
        expect(out.get("dir") == str(pinned),
               "the pinned home (DATA_DIR/vendor/DEIMv2) wins when it exists")
        expect(out.get("gw") == [],
               f"...and no groundwork module was imported to get there "
               f"(loaded: {out.get('gw')})")

    print("[3] the legacy checkout is still honoured")
    with tempfile.TemporaryDirectory() as td:
        legacy = Path(td) / "DEIMv2"
        legacy.mkdir()
        r = subprocess.run(
            [sys.executable, "-c",
             "from altmodels.deim_vendor import vendor_dir; print(vendor_dir())"],
            cwd=str(ROOT), env={**os.environ, "GW_DATA_DIR": td, "HOME": td},
            capture_output=True, text=True, timeout=60)
        expect(r.stdout.strip() == str(legacy),
               "a box with only ~/DEIMv2 keeps working (no data-dir vendor/)")

    print("PASS" if not FAILS else f"{FAILS} FAILURE(S)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
