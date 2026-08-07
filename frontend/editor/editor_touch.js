"use strict";
/* Touch gestures on the canvas: pinch to zoom, drag to pan, tap and long-press.
 *
 * SPLIT OUT OF editor.js (2026-08-02), fifth and last cluster.
 *
 * One-finger drag PANS (native scroll — touch-action allows it), so a tap only
 * counts if the finger neither moved nor lingered. Two fingers pinch-zoom around
 * the midpoint. In crop mode one-finger drag draws the keep-box instead, and in
 * box mode a finger on a handle sizes rather than pans. preventDefault on a
 * handled touchend swallows the browser's synthesized mouse events — without it
 * every tap would also fire mousedown and double-add.
 *
 * A TAP NEVER CHANGES THE COUNT. It selects; the long-press adds; the pad's 🗑
 * removes. (The header this replaces described a "#edMode" toggle for tap
 * add/remove — that button was removed when the select-then-act pad landed, and
 * only a dead CSS rule still mentions it. It is rewritten rather than moved,
 * because a stale comment in a brand-new file reads as deliberate.)
 *
 * IT WRITES A `let` THAT LIVES IN ANOTHER EXTRACTED FILE. edLongPressTimer and
 * ED_LONGPRESS_MS are declared in editor/editor_dotpad.js and all three handlers
 * here clear and reassign the timer. The same shared-lexical-scope rule that lets
 * an extracted file reach editor.js's `ed` also connects two extracted files to
 * each other — so index.html must load editor.js, then editor_dotpad.js, then
 * this. Registering listeners is top-level work, so that order is required, not
 * merely tidy.
 */

function zoomAt(nz, cx, cy){          // zoom keeping the image point under (cx,cy) still
  const wrap = $("#edCanvasWrap"), rect = canvas.getBoundingClientRect();
  const ix = (cx - rect.left) / ed.zoom, iy = (cy - rect.top) / ed.zoom;
  ed.zoom = nz; redraw();
  const wr = wrap.getBoundingClientRect();
  wrap.scrollLeft = ix * nz - (cx - wr.left);
  wrap.scrollTop = iy * nz - (cy - wr.top);
}
const tDist = ts => Math.hypot(ts[0].clientX - ts[1].clientX, ts[0].clientY - ts[1].clientY);
let edTouch = null, edPinch = null;
canvas.addEventListener("touchstart", e => {
  if (e.touches.length === 2){
    edPinch = {d0: tDist(e.touches), z0: ed.zoom};
    edTouch = null;
    e.preventDefault();
  } else if (e.touches.length === 1 && !edPinch){
    const t = e.touches[0];
    edTouch = {x: t.clientX, y: t.clientY, t: Date.now(), moved: false};
    if (ed.showBoxes){
      const [x, y] = evtImg(t);
      const h = edHandleAt(x, y);
      if (h){ ed._rz = h; e.preventDefault(); }  // sizing a box, not panning
    }
    if (ed.cropping){
      const [x, y] = evtImg(t);
      ed._drag = [x, y]; ed.crop = [x, y, x, y]; redraw();
      e.preventDefault();                        // drawing the box, not scrolling
    }
    if (ED_COARSE && !ed.showBoxes && !ed.cropping){
      const [x, y] = evtImg(t);
      // Finger down ON the already-selected dot starts a DRAG of that dot
      // rather than a pan. Only the selected one: otherwise every attempt to
      // scroll a dense image would grab whichever object was under your thumb.
      if (ed.dotSel != null && ed.pts[ed.dotSel]
          && Math.hypot(ed.pts[ed.dotSel][0] - x, ed.pts[ed.dotSel][1] - y) <= ed.r * 4){
        ed._dotDrag = true;
        e.preventDefault();
      } else {
        // Otherwise arm the long-press-to-add. Cancelled by any movement
        // (touchmove) or by lifting early (touchend), so panning never adds.
        clearTimeout(edLongPressTimer);
        edLongPressTimer = setTimeout(() => {
          edLongPressTimer = null;
          if (!edTouch || edTouch.moved) return;
          ed.pts.push([x, y, edPlaceClass()]);
          edDirty = true;
          edSelectDot(ed.pts.length - 1, {added: true});
          if (navigator.vibrate) navigator.vibrate(12);   // it happened under your finger
          edTouch.consumed = true;      // touchend must not also treat this as a tap
        }, ED_LONGPRESS_MS);
      }
    }
  }
}, {passive: false});
canvas.addEventListener("touchmove", e => {
  if (edPinch && e.touches.length === 2){
    e.preventDefault();
    const nz = Math.min(6, Math.max(0.1, edPinch.z0 * tDist(e.touches) / edPinch.d0));
    zoomAt(nz, (e.touches[0].clientX + e.touches[1].clientX) / 2,
               (e.touches[0].clientY + e.touches[1].clientY) / 2);
  } else if (edTouch && e.touches.length === 1){
    const t = e.touches[0];
    if (Math.hypot(t.clientX - edTouch.x, t.clientY - edTouch.y) > 8) edTouch.moved = true;
    if (edTouch.moved && edLongPressTimer){        // panning, not long-pressing
      clearTimeout(edLongPressTimer); edLongPressTimer = null;
    }
    if (ed._dotDrag){                              // dragging the selected dot
      e.preventDefault();
      const [x, y] = evtImg(t);
      const p = ed.pts[ed.dotSel];
      p[0] = Math.max(0, Math.min(ed.w, x));
      p[1] = Math.max(0, Math.min(ed.h, y));
      edDirty = true;
      redraw(); edSyncPad();
      return;
    }
    if (ed._rz){
      e.preventDefault();
      const [x, y] = evtImg(t);
      edResizeTo(x, y, ed._rz.sx, ed._rz.sy, ed._rz.base); redraw();
      return;
    }
    if (ed.cropping && ed._drag){
      e.preventDefault();
      const [x, y] = evtImg(t);
      ed.crop = [ed._drag[0], ed._drag[1], x, y]; redraw();
    }
  }
}, {passive: false});
canvas.addEventListener("touchend", e => {
  clearTimeout(edLongPressTimer); edLongPressTimer = null;
  if (edPinch){ if (!e.touches.length) edPinch = null; return; }
  if (ed._dotDrag){ e.preventDefault(); ed._dotDrag = false; edTouch = null; return; }
  if (ed._rz){ e.preventDefault(); ed._rz = null; edTouch = null; return; }
  if (ed.cropping && ed._drag){ e.preventDefault(); endCropDrag(); edTouch = null; return; }
  const consumed = edTouch && edTouch.consumed;   // the long-press already acted
  const tap = edTouch && !edTouch.moved && Date.now() - edTouch.t < 600;
  edTouch = null;
  if (consumed){ e.preventDefault(); return; }
  if (!tap || !e.changedTouches.length) return;
  e.preventDefault();
  const [x, y] = evtImg(e.changedTouches[0]);
  if (ed.showBoxes){                             // BOX MODE: tap selects, never adds
    ed.sel = edBoxAt(x, y); redraw();
    return;
  }
  // DOT MODE on touch: a tap only ever SELECTS. Adding is the long-press, and
  // removing is the pad's 🗑 — so no single accidental tap can change the count.
  const near = nearestPt(x, y, ed.r * 4);        // fingers are wider than cursors
  if (near >= 0) edSelectDot(near);
  else edDeselectDot();                          // tap empty space = put the pad away
}, {passive: false});
