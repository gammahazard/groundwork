"use strict";
/* Ports of three Python functions, kept here so the coupling is reviewable.
 *
 * These CANNOT import their originals — the editor draws in the browser, the
 * pipeline runs in Python — so the duplication is unavoidable. What is
 * avoidable is duplication nobody can see: scattered through editor.js, a
 * reviewer had no way to know which lines were load-bearing copies of pipeline
 * behaviour, and drift went unnoticed until the editor disagreed with the
 * counter about an image.
 *
 * This file's ONE job is to stay equal to:
 *
 *   boxAt          <- groundwork/dataset/point_to_box.py  box_at()
 *   adaptiveBoxes  <- point_to_box.points_to_boxes() (strategy "adaptive",
 *                     nn_alpha 0.5, min_frac 0.015, max_frac 0.08, fixed 0.05)
 *   boxesFor       <- point_to_box.boxes_for()  — stored extent else default
 *   readingOrder   <- groundwork/yolo_counter.py          _reading_order()
 *   findStacked    <- groundwork/dataset/filters/dot_dedupe.py  find_stacked()
 *   bankersRound   <- Python's built-in round()
 *
 * Signatures deliberately mirror the Python ones (same argument order, same
 * meaning) so the two can be diffed side by side, and so one check can drive
 * both from the same fixtures:
 *
 *     .venv/bin/python scratch/check_editor_mirrors.py
 *
 * Run it after editing this file OR any of the four originals. It compares
 * every box, rank, stacked index and radius, and it self-checks that its
 * fixtures can still catch the rounding drift described below.
 *
 * THE TRAP, hit twice already: Python's round() is half-to-EVEN, JavaScript's
 * Math.round() is half-away-from-zero. round(2.5) is 2 in Python and 3 in JS.
 * Any port of a Python round() must use bankersRound below. Never Math.round.
 */

// Python's round(): ties go to the nearest EVEN integer.
function bankersRound(v){
  const f = Math.floor(v);
  return (v - f === 0.5) ? (f % 2 === 0 ? f : f + 1) : Math.round(v);
}

// Mirror of point_to_box.box_at: centred on the dot with INDEPENDENT half-extents,
// shrunk PER AXIS at the frame edge — never a shifted centre. Clamping the corners
// instead is what made edge dots migrate inward on every editor save->load cycle.
function boxAt(x, y, halfW, halfH, w, h){
  const hx = Math.max(1, Math.min(halfW, x, w - x));
  const hy = Math.max(1, Math.min(halfH, y, h - y));
  return [x - hx, y - hy, x + hx, y + hy];
}

// Mirror of points_to_boxes(strategy="adaptive") — the exact boxes the training
// pipeline writes from these dots, so the editor's "show boxes" is truthful.
function adaptiveBoxes(pts, w, h){
  const ref = Math.min(w, h), lo = 0.015 * ref, hi = 0.08 * ref;
  if (pts.length < 2){
    const half = 0.05 * ref / 2;
    return pts.map(p => boxAt(p[0], p[1], half, half, w, h));
  }
  return pts.map((p, i) => {
    let nn = Infinity;
    for (let j = 0; j < pts.length; j++){ if (j === i) continue;
      const d = Math.hypot(p[0] - pts[j][0], p[1] - pts[j][1]); if (d < nn) nn = d; }
    const half = Math.min(Math.max(0.5 * nn, lo), hi);
    return boxAt(p[0], p[1], half, half, w, h);
  });
}

// Mirror of point_to_box.boxes_for — THE rule for which box a point becomes:
// keep a stored extent, synthesise the adaptive default only for points without
// one. Points are ragged, exactly as labelio produces them: [x,y], [x,y,cls] or
// [x,y,cls,bw,bh] in PIXELS.
//
// This is the pair that matters most in this file. Python writes the label with
// this rule; the editor DRAWS with it. If they drift, the boxes you see are not
// the boxes you save — the worst bug a labelling tool can have.
function boxesFor(pts, w, h){
  const defaults = adaptiveBoxes(pts, w, h);
  return pts.map((p, i) => (p.length > 4 && p[3] && p[4])
    ? boxAt(p[0], p[1], p[3] / 2, p[4] / 2, w, h)
    : defaults[i]);
}

// Mirror of yolo_counter._reading_order: band rows by ~0.7x median box height,
// then top->bottom, left->right. Returns pointIndex -> 1-based number.
// bankersRound, not Math.round: the Python sorts on round(y / rowH).
function readingOrder(pts, boxes){
  const order = new Array(pts.length);
  if (!pts.length) return order;
  const hs = boxes.map(b => b[3] - b[1]).sort((a, b) => a - b);
  const rowH = Math.max(1, 0.7 * hs[Math.floor(hs.length / 2)]);
  const idx = pts.map((_, i) => i).sort((a, b) => {
    const ra = bankersRound(pts[a][1] / rowH), rb = bankersRound(pts[b][1] / rowH);
    return ra !== rb ? ra - rb : pts[a][0] - pts[b][0];
  });
  idx.forEach((pi, k) => order[pi] = k + 1);
  return order;
}

// Mirror of dot_dedupe.find_stacked: indices of dots stacked on another dot
// (the first of each pair is kept). Two objects cannot share a centre, so a pair
// much closer than this image's typical nearest-neighbour gap is one object dotted
// twice. Threshold floors at a dot radius so a 2-dot image still behaves.
function findStacked(pts, dotR){
  const n = pts.length;
  if (n < 2) return [];
  const nn = pts.map((a, i) => {
    let best = Infinity;
    for (let j = 0; j < n; j++){ if (j === i) continue;
      const d = Math.hypot(a[0] - pts[j][0], a[1] - pts[j][1]); if (d < best) best = d; }
    return best;
  });
  const med = [...nn].sort((a, b) => a - b)[Math.floor(n / 2)];
  const thresh = Math.max(dotR * 1.2, med * 0.4);
  const drop = new Set();
  for (let i = 0; i < n; i++){ if (drop.has(i)) continue;
    for (let j = i + 1; j < n; j++){ if (drop.has(j)) continue;
      if (Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < thresh)
        drop.add(j); } }
  return [...drop];
}

// Mirror of dot_dedupe.dot_radius — the overlay dot size AND the floor of the
// stacked threshold above, so both sides must derive it identically.
function dotRadius(w, h){
  return Math.max(3, bankersRound(Math.min(w, h) / 200));
}

// Node (the equivalence check) vs browser (globals via script tag).
if (typeof module !== "undefined" && module.exports)
  module.exports = {bankersRound, boxAt, adaptiveBoxes, boxesFor,
                    readingOrder, findStacked, dotRadius};
