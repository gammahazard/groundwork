"use strict";
/* The touch dot pad: select a dot, then act on it with a thumb.
 *
 * SPLIT OUT OF editor.js (2026-08-02), fourth cluster — and the first one that
 * MOVES A DOT. edNudge, the pad's ✗ and its 🗑 all write ed.pts and set edDirty
 * from another file, so this is the split that has to be verified by saving and
 * re-reading the label, not by looking at the canvas.
 *
 * WHY IT ALL LOADS AT PARSE TIME, unlike editor_hit.js and editor_draw.js. The
 * four `$("#edPadX").onclick` bindings and the nudge-button loop are top-level
 * statements, because binding a listener IS this module's job. They are safe
 * because #edDotPad is STATIC markup in index.html (checked, not assumed) and
 * because index.html loads editor.js first — but that ordering is now
 * load-bearing rather than incidental, which the other two files did not need.
 *
 * loadTray() calls this through `if (window.edSyncPad) edSyncPad()`. That guard
 * keeps working across the move for a reason worth stating: a function
 * DECLARATION is a window property even though `const ed` is not.
 *
 * Everything here is gated on ED_COARSE (editor.js) — desktop has right-click,
 * hover and pixel precision, so none of it buys a mouse anything.
 */

/* ---------- touch: SELECT a dot, then act on it -------------------------
 *
 * The old model was a mode toggle: ➕ meant a tap added a dot, ➖ meant a tap
 * deleted the nearest one. Two problems on a real phone. A fingertip is much
 * wider than an object at fit zoom, so "tap exactly there" is not a thing you can
 * do — every placement needed a follow-up nudge that the mode had no way to
 * express. And the two destructive-vs-creative actions were separated only by
 * the state of a button off in the toolbar, so a mistimed tap in ➖ deleted a
 * object with no confirmation and no undo.
 *
 * Now: tap SELECTS the nearest dot, and everything you can do to it is an
 * explicit button on a pad you can reach with a thumb — nudge one pixel at a
 * time, delete (with a confirm), ✗ to put it back where it was, ✓ to keep it.
 * Adding is a LONG-PRESS on empty canvas, which cannot happen by accident
 * while panning and puts the new dot straight into the selected state so you
 * can immediately place it exactly.
 *
 * Desktop is deliberately untouched — a mouse has right-click, hover and pixel
 * precision, so none of the above buys it anything. All of this is gated on
 * ED_COARSE, and in box or crop mode the pad never appears at all.
 */
const ED_LONGPRESS_MS = 500;      // long enough not to fire while panning
let edLongPressTimer = null;

function edPadVisible(){
  return ED_COARSE && !ed.showBoxes && !ed.cropping && ed.dotSel != null
         && !!ed.pts[ed.dotSel];
}
function edSyncPad(){
  const pad = $("#edDotPad");
  if (!pad) return;
  const on = edPadVisible();
  pad.hidden = !on;
  if (!on) return;
  const p = ed.pts[ed.dotSel];
  $("#edPadPos").textContent = `${Math.round(p[0])}, ${Math.round(p[1])}`;
  $("#edPadWhat").textContent = "dot";
}
/** Select a dot (or nothing) and refresh both canvas and pad. */
function edSelectDot(i, {added = false} = {}){
  ed.dotSel = i;
  ed.dotAdded = added;
  // Remember where it started so ✗ can put it back. A dot that was just ADDED
  // has no previous position — ✗ removes it instead, which is the only sane
  // reading of "cancel" for something that did not exist a second ago.
  ed.dotOrig = i != null && ed.pts[i] ? ed.pts[i].slice(0, 2) : null;
  redraw(); edSyncPad();
}
function edDeselectDot(){ ed.dotSel = null; ed.dotOrig = null; ed.dotAdded = false;
                          redraw(); edSyncPad(); }

/* Nudge: one IMAGE pixel per press regardless of zoom, so the step means the
 * same thing whatever you are looking at. Held down it repeats via the
 * browser's own click repeat only on keyboards, so press-and-hold is handled
 * explicitly below — dotting an image one careful pixel at a time is the whole
 * point of this control. */
function edNudge(dx, dy){
  if (ed.dotSel == null || !ed.pts[ed.dotSel]) return;
  const p = ed.pts[ed.dotSel];
  p[0] = Math.max(0, Math.min(ed.w, p[0] + dx));
  p[1] = Math.max(0, Math.min(ed.h, p[1] + dy));
  edDirty = true;
  redraw(); edSyncPad();
}
document.querySelectorAll("#edDotPad .edNudge").forEach(b => {
  const dx = +b.dataset.dx, dy = +b.dataset.dy;
  let rep = null, accel = null;
  const stop = () => { clearInterval(rep); clearTimeout(accel); rep = accel = null; };
  const start = e => {
    e.preventDefault();                 // no synthesized click, no text selection
    edNudge(dx, dy);
    // Hold to repeat, but only after a beat: a quick tap must move exactly one
    // pixel or precise placement is impossible.
    accel = setTimeout(() => { rep = setInterval(() => edNudge(dx, dy), 60); }, 400);
  };
  b.addEventListener("pointerdown", start);
  ["pointerup", "pointercancel", "pointerleave"].forEach(ev => b.addEventListener(ev, stop));
});
$("#edPadDel").onclick = () => {
  if (ed.dotSel == null) return;
  // Confirm, because this is the one irreversible action on the pad and the
  // button sits next to two harmless ones.
  if (!confirm("Delete this dot?")) return;
  ed.pts.splice(ed.dotSel, 1);
  edDirty = true;
  edDeselectDot();
};
$("#edPadCancel").onclick = () => {
  if (ed.dotSel == null) return;
  if (ed.dotAdded) ed.pts.splice(ed.dotSel, 1);       // never existed -> remove
  else if (ed.dotOrig){ const p = ed.pts[ed.dotSel];  // existed -> put it back
                        p[0] = ed.dotOrig[0]; p[1] = ed.dotOrig[1]; }
  edDeselectDot();
};
$("#edPadOk").onclick = () => edDeselectDot();
