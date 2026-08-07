#!/usr/bin/env python3
"""No source file over 500 lines — the modularity rule, enforced.

The cap exists because this codebase's history shows files drifting past any
informal limit the moment nobody is looking: the predecessor repo celebrated
a 963→543 split in one file's header while five other files sailed past it.
A number in CI cannot be argued with.

Counts every tracked .py and .js under the shipping trees. checks/ itself is
exempt from nothing — a bloated check is still a bloated file.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP = 500
ROOTS = ("groundwork", "frontend", "altmodels", "checks")
SKIP_DIRS = {"__pycache__", "node_modules"}


def main() -> int:
    over = []
    for root in ROOTS:
        for p in (ROOT / root).rglob("*"):
            if p.suffix not in (".py", ".js") or not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            n = sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
            if n > CAP:
                over.append((n, p.relative_to(ROOT).as_posix()))
    if over:
        print(f"SIZE CAP: {len(over)} file(s) over {CAP} lines")
        for n, rel in sorted(over, reverse=True):
            print(f"  {n:5}  {rel}")
        return 1
    print(f"size cap: every source file is ≤{CAP} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
