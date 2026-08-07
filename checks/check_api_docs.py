"""docs/api.md's route catalog must match the live application.

WHY. The API reference promises "every route" — a hand-maintained table
breaks that promise the first time someone adds an endpoint and forgets the
doc. The catalog is generated (scripts/gen_api_docs.py); this check
regenerates it and fails on any difference, so the promise is enforced, not
remembered.

HOW IT CAN FAIL: add or change any route (or its docstring's first line)
without rerunning the generator.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_api_docs.py"),
                    "--check"], capture_output=True, text=True, timeout=120)
print("  " + (r.stdout or "").strip())
if r.returncode != 0:
    print((r.stderr or "").strip())
    sys.exit(1)
print("the API reference catalog matches the app")
