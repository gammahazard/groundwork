"use strict";
/* Editor geometry: where a pointer is, and what it is over.
 *
 * SPLIT OUT OF editor.js (2026-08-02), second cluster. Unlike the ⋯ menu this one
 * genuinely USES the shared state — sixteen reads of `ed` and a write to
 * `edDirty` — which is what makes it the real test of the split rather than a
 * proof that a file loads.
 *
 * THAT WORKS BECAUSE OF A MEASURED FACT, not an assumption: editor.js declares
 * `const ed`, `let edDirty`, `const canvas` and `const ED_COARSE` at top level,
 * and top-level const/let are NOT window properties — window.ed is undefined.
 * They ARE reachable, and writable, from another classic script through the
 * shared global lexical environment. Verified in the live page before any of
 * this moved. The constraint that comes with it: those names resolve at RUNTIME,
 * so nothing here may touch them at module top level, and editor.js must be
 * parsed first. Every symbol below is inside a function; there is no top-level
 * statement in this file at all.
 *
 * boxesFor comes from editor/editor_mirrors.js, whose one job is staying equal to
 * point_to_box.boxes_for in Python — the overlay must draw the box the save will
 * write, or the editor shows one thing and the file records another.
 */

function evtImg(e){
  const rect = canvas.getBoundingClientRect();
  return [(e.clientX - rect.left) / ed.zoom, (e.clientY - rect.top) / ed.zoom];
}
function nearestPt(x, y, rad){
  let best = -1, bd = (rad || ed.r*2)**2;
  ed.pts.forEach((p,i)=>{ const dd=(p[0]-x)**2+(p[1]-y)**2; if(dd<bd){bd=dd;best=i;} });
  return best;
}

/* ---------- box mode ---------- */
// Does this point carry a hand-set extent? Same test as boxes_for / save_points.
const edSized = i => { const p = ed.pts[i]; return p.length > 4 && p[3] && p[4]; };
// 8 handles: 4 corners (size both axes) + 4 edge midpoints (size one).
// [x, y, sx, sy] — sx/sy say which axes that handle drives.
function edHandles(b){
  const [x1, y1, x2, y2] = b, mx = (x1+x2)/2, my = (y1+y2)/2;
  return [[x1,y1,1,1], [x2,y1,1,1], [x1,y2,1,1], [x2,y2,1,1],
          [mx,y1,0,1], [mx,y2,0,1], [x1,my,1,0], [x2,my,1,0]];
}
// Which handle of the selected box is under (x,y)? Tolerance is in SCREEN px so
// it stays grabbable at any zoom, and is fatter on touch.
function edHandleAt(x, y){
  if (ed.sel == null) return null;
  const b = boxesFor(ed.pts, ed.w, ed.h)[ed.sel];
  if (!b) return null;
  const tol = (ED_COARSE ? 18 : 9) / ed.zoom;
  for (const [hx, hy, sx, sy] of edHandles(b))
    if (Math.abs(x-hx) <= tol && Math.abs(y-hy) <= tol) return {sx, sy, base: b};
  return null;
}
// The box under the cursor, smallest first so a small box on top of a big one
// is still selectable.
function edBoxAt(x, y){
  const bx = boxesFor(ed.pts, ed.w, ed.h);
  let best = null, bestArea = Infinity;
  bx.forEach((b, i) => {
    if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) return;
    const a = (b[2]-b[0]) * (b[3]-b[1]);
    if (a < bestArea){ bestArea = a; best = i; }
  });
  return best;
}
const ED_MIN_BOX = 6;      // px — a box below this is a mis-drag, not an object
// Resize from a handle drag. The box is CENTRED on its dot, so the half-extent
// is just the distance from the dot to the cursor; position never moves, which
// is what keeps the count and the dedupe unaffected by sizing work.
//
// `base` is the selected box measured ONCE when the drag started. It must not be
// recomputed per mousemove: adaptiveBoxes is O(n²) in the image's dot count, and
// a full image here is ~400 objects — 160k distance calculations per pointer event
// would make dragging crawl on exactly the dense images this feature is for.
// Caching is safe because the defaults depend only on dot POSITIONS, and sizing
// never moves a dot.
function edResizeTo(x, y, sx, sy, base){
  const p = ed.pts[ed.sel];
  const bw = sx ? Math.max(ED_MIN_BOX, 2*Math.abs(x - p[0])) : (base[2]-base[0]);
  const bh = sy ? Math.max(ED_MIN_BOX, 2*Math.abs(y - p[1])) : (base[3]-base[1]);
  while (p.length < 5) p.push(0);
  p[3] = bw; p[4] = bh;
  edDirty = true;
}
