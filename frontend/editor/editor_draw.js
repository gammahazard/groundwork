"use strict";
/* Editor rendering: everything the canvas shows, plus the chrome that describes it.
 *
 * SPLIT OUT OF editor.js (2026-08-02), third cluster — the largest single job in
 * the file. One function, because one pass over the image is what keeps the layers
 * consistent: photo, then boxes, then dots, then reading-order numbers, then the
 * selection and crop overlays, then the title and the prev/next buttons. Drawing
 * the count in the title from a different pass than the dots is how a title comes
 * to disagree with the picture under it.
 *
 * It READS and never writes: no branch here mutates ed.pts or sets edDirty. That
 * is worth keeping true — redraw is called from la.js and audit_overlay.js as
 * well as from every pointer handler, so a write here would fire at times no
 * caller intends.
 *
 * BOXES COME FROM boxesFor, NOT adaptiveBoxes (editor/editor_mirrors.js, which is
 * kept equal to point_to_box.boxes_for in Python). A point carrying a hand-set
 * extent must draw at that extent, or the overlay shows one box and the save
 * writes another. edSized lives in editor/editor_hit.js and answers which is which.
 *
 * Same lexical-scope rule as editor_hit.js: `ed`, `ctx` and `canvas` are
 * editor.js's top-level consts, reachable across classic scripts but resolved at
 * RUNTIME — so nothing here runs at top level.
 */

function redraw(){
  const z = ed.zoom;
  // Render at the DISPLAY's pixel density, not in CSS pixels. Phone photos are
  // now ~12 MP (3024px wide) and get fitted into ~1200px of viewport, so the
  // canvas both downscales the photo AND used to get upscaled again by a
  // retina screen — objects came out mushy and hard to dot. Backing-store size
  // x dpr + a CSS size in logical px fixes the second half; imageSmoothing
  // 'high' fixes the first. setTransform(dpr) means every drawing call below
  // keeps working in logical pixels, unchanged.
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const cw = Math.round(ed.w * z), chh = Math.round(ed.h * z);
  canvas.width = Math.round(cw * dpr); canvas.height = Math.round(chh * dpr);
  canvas.style.width = cw + "px"; canvas.style.height = chh + "px";
  // NB: assigning width/height resets the context, so configure it AFTER.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(ed.img, 0, 0, cw, chh);
  const r = ed.r * z;
  let order = null;
  if (ed.showBoxes){                                  // the real training boxes, under the dots
    // boxesFor, NOT adaptiveBoxes: a point carrying a hand-set extent must draw
    // at THAT size, or the overlay shows one thing and the save writes another.
    // Same rule as labelio.save_points (point_to_box.boxes_for), mirrored.
    const bx = boxesFor(ed.pts, ed.w, ed.h);
    order = readingOrder(ed.pts, bx);
    ctx.lineWidth = 1;
    bx.forEach((b, i) => {
      // Colour carries the fact, not just decoration: green = still the auto
      // size derived from spacing, yellow = a human set this one. Without it
      // there is no way to see which boxes you have actually fixed.
      ctx.strokeStyle = edSized(i) ? "rgba(255,212,0,0.95)" : "rgba(0,230,120,0.85)";
      ctx.strokeRect(b[0]*z, b[1]*z, (b[2]-b[0])*z, (b[3]-b[1])*z);
    });
    if (ed.sel != null && bx[ed.sel]){
      const b = bx[ed.sel];
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.strokeRect(b[0]*z, b[1]*z, (b[2]-b[0])*z, (b[3]-b[1])*z);
      ctx.fillStyle = "#fff";
      for (const [hx, hy] of edHandles(b))
        ctx.fillRect(hx*z - 4, hy*z - 4, 8, 8);       // 8 screen px, zoom-independent
    }
  }
  // MULTI-CLASS projects colour by class AND label the index next to the dot.
  const multiCls = ed.classes.length > 1;
  ed.pts.forEach((p, i) => {
    ctx.fillStyle = multiCls ? edClassColour(p[2] || 0)
      : "rgba(0,122,255,0.95)";
    ctx.beginPath(); ctx.arc(p[0]*z, p[1]*z, r, 0, 7); ctx.fill();
    if (ed._flag && ed._flag.has(i)){                      // stacked-dot warning ring
      ctx.lineWidth = 3; ctx.strokeStyle = "#ff2d2d";
      ctx.beginPath(); ctx.arc(p[0]*z, p[1]*z, r + 4, 0, 7); ctx.stroke();
    } else {
      ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
    }
  });
  // The selected dot, drawn LAST so nothing overlaps it. A halo plus crosshairs
  // rather than a colour swap: colour already means whole-vs-half here, and on a
  // dense image a recoloured dot is genuinely hard to find again after a nudge.
  // The crosshair arms extend well past the dot so it is locatable at a glance
  // even when your thumb is over that part of the screen.
  if (ed.dotSel != null && ed.pts[ed.dotSel]){
    const p = ed.pts[ed.dotSel], cx = p[0]*z, cy = p[1]*z, R = r + 7;
    ctx.lineWidth = 2; ctx.strokeStyle = "#fff";
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.beginPath(); ctx.arc(cx, cy, R + 1.5, 0, 7); ctx.stroke();   // keeps it visible on white objects
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
    const arm = R + 10;
    ctx.beginPath();
    ctx.moveTo(cx - arm, cy); ctx.lineTo(cx - R - 2, cy);
    ctx.moveTo(cx + R + 2, cy); ctx.lineTo(cx + arm, cy);
    ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy - R - 2);
    ctx.moveTo(cx, cy + R + 2); ctx.lineTo(cx, cy + arm);
    ctx.stroke();
  }
  // The class INDEX beside each dot. Colour is the shortcut; this is the fact —
  // a five-colour legend is not learnable while dotting, and the owner is
  // colour-blind on part of any palette. Skipped in box mode, where the
  // reading-order numbers already occupy that space.
  if (multiCls && !ed.showBoxes){
    const fs = Math.max(9, Math.round(r * 1.6));
    ctx.font = `bold ${fs}px sans-serif`;
    ctx.textBaseline = "middle";
    ed.pts.forEach(p => {
      const t = String(p[2] || 0), x = p[0]*z + r + 2, y = p[1]*z;
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.strokeText(t, x, y);                       // halo, so it reads on any object
      ctx.fillStyle = "#fff"; ctx.fillText(t, x, y);
    });
  }
  if (ed.showBoxes && order){                          // reading-order numbers, on top
    ctx.fillStyle = "rgba(255,235,0,0.97)";
    ctx.font = `${Math.max(9, Math.round(r*1.7))}px sans-serif`;
    ctx.textBaseline = "bottom";
    ed.pts.forEach((p, i) => ctx.fillText(order[i], p[0]*z + r, p[1]*z - r*0.5));
  }
  if (window.laDraw) laDraw(ctx, z);                   // LA crosshair overlay (la.js)
  // Model-vs-label disagreements, drawn LAST so a mark is never under a dot.
  if (window.auditDraw) auditDraw(ctx, z);             // audit_overlay.js
  if (ed.crop){                                        // crop selection: dim outside + yellow box
    const rx = Math.min(ed.crop[0], ed.crop[2])*z, ry = Math.min(ed.crop[1], ed.crop[3])*z;
    const rw = Math.abs(ed.crop[2]-ed.crop[0])*z, rh = Math.abs(ed.crop[3]-ed.crop[1])*z;
    const cwL = ed.w * z, chL = ed.h * z;      // logical px (ctx is dpr-scaled)
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(0, 0, cwL, ry);
    ctx.fillRect(0, ry+rh, cwL, chL-(ry+rh));
    ctx.fillRect(0, ry, rx, rh);
    ctx.fillRect(rx+rw, ry, cwL-(rx+rw), rh);
    ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2; ctx.strokeRect(rx, ry, rw, rh);
  }
  let t = `<b>${ed.stem}</b> — `;
  {
    const c = ed.pts.length;
    t += `dots: ${c}`;
    if (ed.target != null){ const d = c - ed.target; t += ` / target ${ed.target} <span class="${d===0?'ok':'short'}">(${d>0?'+':''}${d})</span>`; }
  }
  if (window.laTitleExtra) t += laTitleExtra();         // "· LA ✛ N" when overlaid
  if (window.auditTitleExtra) t += auditTitleExtra();   // "⚠ model 92 — …"
  $("#edTitle").innerHTML = t;
  $("#edPos").textContent = `${ed.idx + 1} / ${ed.list.length}`;
  $("#edPrev").disabled = ed.idx === 0;
  $("#edNext").disabled = ed.idx === ed.list.length - 1;
}
