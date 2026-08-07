"""Regenerate the route catalog inside docs/api.md.

The narrative half of that file is written by hand. The CATALOG between the
markers is generated from the live app — every route, method and first
docstring line — so it cannot drift from the code. checks/check_api_docs.py
fails CI when the committed catalog differs from a fresh generation.

    .venv/bin/python scripts/gen_api_docs.py          # rewrite in place
    .venv/bin/python scripts/gen_api_docs.py --check  # exit 1 on drift
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "api.md"
BEGIN = "<!-- catalog:begin (generated — edit scripts/gen_api_docs.py, not this table) -->"
END = "<!-- catalog:end -->"


def catalog() -> str:
    os.environ.setdefault("GW_AUTH", "0")
    sys.path.insert(0, str(ROOT))
    from groundwork.web.app import app

    # THE SCHEMA, not app.routes: this app includes routers through a wrapper
    # object, so a routes-walk sees two paths and lies. app.openapi() is what
    # actually serves /docs — if a route is missing there it is missing
    # everywhere, which is exactly the property a catalog wants.
    paths = app.openapi().get("paths", {})
    rows: dict[str, list[tuple[str, str, str]]] = {}
    for path, ops in paths.items():
        methods = ",".join(sorted(m.upper() for m in ops))
        summary = ""
        for op in ops.values():
            summary = op.get("summary") or (op.get("description") or "").strip().splitlines()[:1]
            summary = summary if isinstance(summary, str) else (summary[0] if summary else "")
            if summary:
                break
        summary = re.sub(r"\s+", " ", summary).replace("|", "\\|")[:110]
        seg = path.split("/")[2] if path.startswith("/api/") and len(path.split("/")) > 2 else path.strip("/") or "root"
        rows.setdefault(seg, []).append((path, methods, summary))

    out = []
    for seg in sorted(rows):
        out.append(f"\n### `{seg}`\n")
        out.append("| Route | Methods | What it does |")
        out.append("|---|---|---|")
        for path, methods, summary in sorted(rows[seg]):
            out.append(f"| `{path}` | {methods} | {summary} |")
    return "\n".join(out) + "\n"


def main() -> int:
    body = DOC.read_text(encoding="utf-8")
    i = body.index(BEGIN) + len(BEGIN)
    j = body.index(END)
    fresh = "\n" + catalog() + "\n"
    if "--check" in sys.argv:
        if body[i:j] != fresh:
            print("docs/api.md catalog is STALE — run scripts/gen_api_docs.py")
            return 1
        print("docs/api.md catalog matches the live app")
        return 0
    DOC.write_text(body[:i] + fresh + body[j:], encoding="utf-8")
    n = fresh.count("| `/")
    print(f"docs/api.md: catalog rewritten — {n} routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
