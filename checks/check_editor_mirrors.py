"""Prove frontend/editor/editor_mirrors.js still equals its Python originals.

The dot editor draws in a browser and the pipeline runs in Python, so several
functions HAVE to exist twice (see editor_mirrors.js). This drives both copies
from the same fixtures and diffs every output. Run it after touching either
side:

    .venv/bin/python checks/check_editor_mirrors.py

Requires node — it executes the JS half for real; the comparison is
behavioural, never textual. Exit code 0 = the two are equal.

The fixtures deliberately include a banker's-rounding tie, because that is the
drift this pair has actually suffered: Python's round() is half-to-even and
JS's Math.round() is half-away-from-zero, so round(0.5) is 0 in Python and 1
in JS. Without that case the check passes against a KNOWN-broken port — 750
random dots did exactly that. If you add fixtures, keep at least one that
fails when readingOrder uses Math.round, or this file is decoration.
"""
from __future__ import annotations
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundwork.la.parse import Detection                          # noqa: E402
from groundwork.dataset.point_to_box import boxes_for, points_to_boxes  # noqa: E402
from groundwork.dataset.filters.dot_dedupe import find_stacked, dot_radius  # noqa: E402
from groundwork.serve.yolo_counter import _reading_order           # noqa: E402

MIRRORS = ROOT / "frontend" / "editor" / "editor_mirrors.js"


def fixtures() -> list[dict]:
    random.seed(7)
    out = []
    for n, w, h in [(0, 1000, 800), (1, 1200, 900), (2, 900, 900), (5, 1600, 1200),
                    (40, 3024, 4032), (120, 2000, 1500), (213, 3024, 4032)]:
        out.append({"w": w, "h": h,
                    "pts": [[round(random.uniform(0, w), 3),
                             round(random.uniform(0, h), 3), 0] for _ in range(n)]})
    # coincident dots, both frame corners, a sub-pixel edge dot
    out.append({"w": 1000, "h": 1000,
                "pts": [[500, 500, 0], [500, 500, 0], [0, 0, 0],
                        [1000, 1000, 0], [999.5, 0.5, 0]]})
    out.append({"w": 1200, "h": 1200,
                "pts": [[100 + ix * 60, 100 + iy * 70, 0]
                        for iy in range(8) for ix in range(8)]})
    # Points carrying a hand-set EXTENT — the boxes_for rule. Mixed with plain
    # rows on purpose: ragged input is exactly what labelio hands over.
    out.append({"w": 1400, "h": 1000, "note": "mixed stored extents",
                "pts": [[700, 500, 0, 240, 90],      # wide, well inside
                        [300, 400, 0],               # no extent -> default
                        [60, 500, 0, 400, 120],      # must clamp on x at the edge
                        [700, 40, 1, 100, 300],      # must clamp on y, half class
                        [1000, 600, 0, 0, 0]]})      # zero extent -> falsy -> default
    # THE TIE: y / rowH lands on exactly 0.5, where the two roundings disagree
    # and the disagreement merges two rows into one band (4 of 6 dots renumber).
    gap, row_h = 20, 0.7 * 20
    out.append({"w": 2000, "h": 2000, "note": "banker's-rounding tie",
                "pts": [[200 + i * gap, row_h * (k + 0.5), 0]
                        for k in (1, 2) for i in range(3)]})
    return out


def python_side(cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        pts = [(p[0], p[1]) for p in c["pts"]]
        dets = [Detection(kind="point", coords=(x, y)) for x, y in pts]
        boxes = [list(d.coords) for d in points_to_boxes(dets, c["w"], c["h"])]
        r = dot_radius(c["w"], c["h"])
        # _reading_order returns sorted CENTRES; convert to index -> 1-based rank
        # so it lines up with the JS, which returns ranks directly.
        order: list[int | None] = [None] * len(pts)
        if pts:
            pos: dict = {}
            for k, cen in enumerate(_reading_order(list(pts), boxes)):
                pos.setdefault(cen, []).append(k + 1)
            for i, cen in enumerate(pts):
                order[i] = pos[cen].pop(0)
        out.append({"boxes": boxes, "order": order,
                    "stacked": sorted(find_stacked(pts, r)), "r": r,
                    "boxes_for": [list(b) for b in boxes_for(c["pts"], c["w"], c["h"])]})
    return out


def js_side(cases: list[dict], mirrors: Path) -> list[dict]:
    with tempfile.TemporaryDirectory() as td:
        cf, of = Path(td) / "cases.json", Path(td) / "out.json"
        cf.write_text(json.dumps(cases), encoding="utf-8")
        script = f"""
const fs = require("fs"), m = require({str(mirrors)!r});
const cases = JSON.parse(fs.readFileSync({str(cf)!r}));
fs.writeFileSync({str(of)!r}, JSON.stringify(cases.map(c => {{
  const boxes = m.adaptiveBoxes(c.pts, c.w, c.h), r = m.dotRadius(c.w, c.h);
  return {{boxes, order: m.readingOrder(c.pts, boxes),
          stacked: m.findStacked(c.pts, r).sort((a,b)=>a-b), r,
          boxes_for: m.boxesFor(c.pts, c.w, c.h)}};
}})));"""
        js = Path(td) / "run.js"
        js.write_text(script, encoding="utf-8")
        p = subprocess.run(["node", str(js)], capture_output=True, text=True,
                           timeout=120)
        if p.returncode != 0:
            raise SystemExit(f"node failed: {p.stderr.strip()[:500]}")
        return json.loads(of.read_text())


def compare(py: list[dict], js: list[dict], cases: list[dict]) -> list[str]:
    bad = []
    for i, (a, b, c) in enumerate(zip(py, js, cases)):
        tag = f"case {i} (n={len(c['pts'])}{', ' + c['note'] if c.get('note') else ''})"
        if a["r"] != b["r"]:
            bad.append(f"{tag}: dot radius {a['r']} != {b['r']}")
        if a["stacked"] != b["stacked"]:
            bad.append(f"{tag}: stacked {a['stacked']} != {b['stacked']}")
        if a["order"] != b["order"]:
            n = sum(1 for x, y in zip(a["order"], b["order"]) if x != y)
            bad.append(f"{tag}: reading order differs on {n} dot(s)")
        for k, (x, y) in enumerate(zip(a["boxes"], b["boxes"])):
            if max(abs(p - q) for p, q in zip(x, y)) > 1e-9:
                bad.append(f"{tag}: box {k} {x} != {y}")
                break
        for k, (x, y) in enumerate(zip(a["boxes_for"], b["boxes_for"])):
            if max(abs(p - q) for p, q in zip(x, y)) > 1e-9:
                bad.append(f"{tag}: boxes_for {k} {x} != {y}")
                break
    return bad


def main() -> None:
    if shutil.which("node") is None:
        raise SystemExit(
            "node not found — this check runs the JS half, so it needs it. "
            "Run it where the editor is developed.")
    cases = fixtures()
    py = python_side(cases)
    bad = compare(py, js_side(cases, MIRRORS), cases)
    dots = sum(len(c["pts"]) for c in cases)
    if bad:
        print(f"MIRRORS HAVE DRIFTED — {len(bad)} mismatch(es):")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print(f"editor_mirrors.js == Python: {len(cases)} cases, {dots} dots, "
          "boxes + boxes_for + reading order + stacked + radius all equal")

    # Self-check: the fixtures must be able to CATCH the drift they exist for.
    src = MIRRORS.read_text(encoding="utf-8")
    drifted = src.replace("bankersRound(pts[a][1] / rowH)", "Math.round(pts[a][1] / rowH)") \
                 .replace("bankersRound(pts[b][1] / rowH)", "Math.round(pts[b][1] / rowH)")
    if drifted == src:
        raise SystemExit("self-check: readingOrder no longer uses bankersRound "
                         "the way this check expects — update the check")
    with tempfile.TemporaryDirectory() as td:
        alt = Path(td) / "drifted.js"
        alt.write_text(drifted, encoding="utf-8")
        if not compare(py, js_side(cases, alt), cases):
            raise SystemExit("self-check FAILED: swapping bankersRound for "
                             "Math.round changed nothing, so these fixtures "
                             "cannot detect the one drift they exist for")
    print("self-check: fixtures do catch a Math.round regression")


if __name__ == "__main__":
    main()
