#!/usr/bin/env python3
"""Run the CI-safe check suite: every checks/check_*.py, one subprocess each.

    python checks/run.py                 # the whole CI-safe set
    python checks/run.py --only vocab    # one check (with or without check_/.py)

WHAT IS EXCLUDED, and why it is by DIRECTORY rather than by a list kept here:
checks/gpu/ needs a GPU and a running cockpit (manual), and checks/ui/ needs a
running instance plus playwright (manual). A list of names would go stale the
day a check is added; a directory boundary cannot.

Each check runs as its own subprocess — they monkeypatch modules, redirect
stores and mutate sys.path, and none of that may leak between them. GW_AUTH=0
is exported so web-importing checks load without the login gate; the checks
that test the gate itself set GW_AUTH=1 in their own process before importing.

Exit code: 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 600


def discover() -> list[Path]:
    """Every top-level check, sorted. gpu/ and ui/ are not CI (see docstring)."""
    return sorted(HERE.glob("check_*.py"))


def matches(f: Path, only: str) -> bool:
    want = only.removesuffix(".py").removeprefix("check_")
    return f.stem.removeprefix("check_") == want or f.stem == only


def run_one(f: Path) -> tuple[bool, float, str]:
    env = {**os.environ, "GW_AUTH": "0"}
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(f)], env=env,
                           cwd=str(HERE.parent), capture_output=True,
                           text=True, timeout=TIMEOUT_S)
        ok, tail = r.returncode == 0, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        ok, tail = False, f"timed out after {TIMEOUT_S}s"
    return ok, time.time() - t0, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="run one check by name (vocab, check_vocab, "
                                   "check_vocab.py all work)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show every check's full output, not only failures")
    args = ap.parse_args()

    files = discover()
    if args.only:
        files = [f for f in files if matches(f, args.only)]
        if not files:
            print(f"no check matches {args.only!r} — have: "
                  + ", ".join(f.stem for f in discover()))
            return 2

    failed: list[str] = []
    for f in files:
        ok, took, tail = run_one(f)
        print(f"  {'PASS' if ok else 'FAIL'}  {f.stem:<28} {took:6.1f}s")
        if args.verbose or not ok:
            for line in tail.strip().splitlines()[-25:]:
                print(f"        {line}")
        if not ok:
            failed.append(f.stem)

    print()
    if failed:
        print(f"{len(failed)} of {len(files)} FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(files)} checks passed  "
          f"(checks/gpu and checks/ui are manual — see checks/README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
