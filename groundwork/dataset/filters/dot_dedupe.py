"""Drop stacked dots — two detections on ONE object (physically impossible pair).

Two objects can never overlap centres: the closest real pair on an image sits about
one object-diameter apart. So any two dots closer than a small fraction of the
image's own typical spacing (median nearest-neighbour distance) are one object
counted twice — an undertrained checkpoint or a capsule's two halves each
getting a box. Same rule the web dot-editor and eval gallery use to flag dupes;
here it actually removes them. Shared by the runtime counter and count_eval so
the metric measures exactly what ships.

`find_stacked` and `dot_radius` are ALSO ported to JavaScript for the browser
dot-editor (apps/static/editor/editor_mirrors.js). Change either and re-run
`.venv/bin/python scratch/check_editor_mirrors.py`.
"""
from __future__ import annotations
import math


def dot_radius(w: int, h: int) -> float:
    """Overlay dot radius for an image — and the FLOOR of the stacked-dot
    threshold, so it must be derived identically everywhere.

    Single definition on purpose. It previously existed five times (count_eval,
    yolo_counter, annotate, teacher_label, and a Swift port) with three
    different rounding behaviours: Python's banker's round(), Swift's
    half-away-from-zero .rounded(), and the web editor's no rounding at all.
    A 900px-short-side image therefore produced a dedupe floor of 4.8px, 6.0px
    or 5.4px depending on which copy you asked.

    That sentence was written before the Swift copy was actually fixed, and read
    as though it had been — the Python and JavaScript sides were consolidated
    onto this definition while a third port went on computing
    6.0px until 2026-07-30, because the check that was supposed to hold the ports
    equal had no Swift arm. It has one now (source-text only; the box has no
    Swift toolchain). Two lessons, both cheap: a port is not consolidated until
    something can FAIL when it drifts, and a docstring written in the past tense
    is a claim that needs the same evidence as the code."""
    return max(3, round(min(w, h) / 200))


def find_stacked(centers, dot_r: float) -> set[int]:
    """Indices of dots stacked on another dot (keep the first of each pair).

    Threshold: 0.4x the median nearest-neighbour spacing, floored at ~a dot
    radius so a 2-object image (median == the only gap) still behaves."""
    n = len(centers)
    if n < 2:
        return set()
    nn = []
    for i, a in enumerate(centers):
        best = math.inf
        for j, b in enumerate(centers):
            if i != j:
                best = min(best, math.hypot(a[0] - b[0], a[1] - b[1]))
        nn.append(best)
    med = sorted(nn)[n // 2]
    thresh = max(dot_r * 1.2, med * 0.4)
    drop: set[int] = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j not in drop and \
                    math.hypot(centers[i][0] - centers[j][0],
                               centers[i][1] - centers[j][1]) < thresh:
                drop.add(j)
    return drop


def drop_stacked(centers, dot_r: float):
    """(kept_centers, n_dropped) with stacked dots removed."""
    dupes = find_stacked(centers, dot_r)
    if not dupes:
        return list(centers), 0
    return [c for i, c in enumerate(centers) if i not in dupes], len(dupes)
