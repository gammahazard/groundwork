"use strict";
/* Dot editor overlay — add/remove/adjust the object dots on an image, then save or
 * promote. Loaded AFTER cockpit.js and editor_mirrors.js, sharing their globals:
 *   $, api            (helpers)            — defined in cockpit.js
 *   loadGrid, loadState                    — defined in cockpit.js
 *   adaptiveBoxes, boxesFor, readingOrder, findStacked, dotRadius
 *                                          — defined in editor_mirrors.js,
 *     whose one job is keeping those equal to their Python originals.
 * cockpit.js's loadGrid calls openEditor() (defined here) at click time, so the
 * two files reference each other's globals only after both have loaded.
 *
 * THIS FILE WAS 963 LINES AND WAS SPLIT (2026-08-02). What is left is the image
 * lifecycle — open, load, navigate, crop, save, promote, close — plus the state
 * every other piece reads. Five files came out, each with the one invariant it
 * exists to protect in its own header:
 *
 *   editor_more.js     the ⋯ narrow-screen tools menu (shares no state)
 *   editor_hit.js      pointer geometry: what is under the cursor, and resizing
 *   editor_draw.js     redraw() — the whole canvas, and the title that describes it
 *   editor_dotpad.js   touch: select a dot, then nudge / ◐ / 🗑 / ✗ / ✓ it
 *   editor_touch.js    touch gestures: pinch, pan, tap, long-press-to-add
 *
 * THEY SHARE THIS FILE'S STATE, and the reason that works was measured on
 * 2026-08-02 by injecting a second classic script into the live page:
 *
 *   window.ed        undefined      `const ed` is NOT a window property
 *   window.edDirty   undefined      nor is `let edDirty`
 *   ed               object         but BOTH are reachable lexically from
 *   edDirty          boolean        another classic script, and WRITABLE
 *   openEditor       function       (a function declaration IS on window)
 *
 * The shared global lexical environment carries `const`/`let` across classic
 * scripts even though `window` does not — which is also why loadTray's
 * `if (window.edSyncPad)` guard still works while `window.ed` would not.
 *
 * LOAD ORDER IS LOAD-BEARING, not tidy. Those bindings resolve at RUNTIME, so
 * index.html must load this file first; editor_dotpad.js and editor_touch.js then
 * register listeners at top level, and editor_touch.js writes edLongPressTimer,
 * a `let` declared in editor_dotpad.js — so dotpad must precede touch.
 *
 * VERIFY A CHANGE BY SAVING AND RE-READING THE LABEL, never by looking at the
 * canvas. Dots that render right and write wrong is the failure that costs work,
 * and only a round trip catches it. Three drivers do exactly that and are the
 * regression suite for this directory:
 *   scratch/ui_editor_resize.cjs      box resize -> one line, centre unmoved
 *   scratch/ui_editor_nudge.cjs       one ▲ press -> exactly one image pixel
 *   scratch/ui_editor_longpress.cjs   long press -> exactly one new dot
 * Each snapshots the image it edits and restores it byte-identically. */

/* ⬚ Boxes is a MODE, not just an overlay. With it on you size boxes and cannot
 * add or remove dots; with it off you place dots and cannot touch extents. That
 * separation is deliberate: a full image is ~400 objects, so "drag near a corner to
 * resize" would fire constantly by accident while dotting. One toggle, two
 * non-overlapping jobs, and no way to damage an image by mis-clicking in the
 * wrong one. `sel` is the box being sized; `_rz` is an in-progress handle drag. */
const ed = { collection:null, list:[], idx:0, canDone:false, showBoxes:false,
             stem:null, w:0, h:0, pts:[], target:null, zoom:null, img:null, r:5,
             fitted:null, sig:null,   // fitted = the zoom fitZoom chose; sig = w x h
             sel:null, _rz:null,
             // --- touch dot editing (see the ED_COARSE block further down) ---
             // dotSel is the selected DOT index; ed.sel is the selected BOX in
             // box mode. Two different modes, two different selections, never
             // both live at once — keeping them in one field made a tap in dot
             // mode light up a box handle.
             dotSel:null, dotOrig:null, dotAdded:false, _dotDrag:false,
             // --- multi-class ---
             // classes is the project's vocabulary; cls is the one being placed.
             // A point's third element IS its class index — the format has
             // always carried it (labelio.save_points reads p[2]) and nothing
             // could ever set it, so every one of the first project's 10,247 labels is
             // class 0. Correct for one class, silently wrong for two.
             classes:["object"], cls:0 };
let edDirty = false;   // unsaved dot changes pending?

const canvas = $("#edCanvas"), ctx = canvas.getContext("2d");

// Touch devices get a tap-mode toggle (no right-click to remove with) and the
// help text speaks their language. Desktop behaviour is untouched.
const ED_COARSE = matchMedia("(pointer: coarse)").matches;
function helpHTML(){
  if (ed.cropping)
    return "✂ <b>Crop mode</b>: drag a box around the area to KEEP — everything "
      + "outside is trimmed off. Toggle ✂ Crop off to go back to editing dots.";
  if (ed.showBoxes)
    return "⬚ <b>Box mode</b>: " + (ED_COARSE ? "tap" : "click") + " a box to select it, "
      + "then drag a <b>handle</b> to size it — corners set both, edges set one. "
      + "<b>r</b> = back to auto size. Dots can't be added or removed here; "
      + "toggle ⬚ Boxes off for that. <span style='color:#0e8'>green = auto "
      + "(from spacing)</span> · <span style='color:#ffd400'>yellow = you sized it</span>";
  if (ED_COARSE)
    return "<b>tap a dot</b> to select it, then nudge / ✗ / 🗑 / ✓ on the pad below · "
      + "<b>press and hold</b> empty space to add one · drag a selected dot to move it"
      + " · pinch = zoom · drag = pan";
  return "Left-click add · right-click remove · wheel/± zoom · ← → prev/next image";
}

async function openEditor(collection, stem, canDone, list){
  ed.collection = collection;
  ed.list = (list && list.length) ? list : [stem];
  ed.idx = Math.max(0, ed.list.indexOf(stem));
  ed.canDone = canDone;
  $("#editor").hidden = false;
  edSyncMore();                        // ⋯ only exists on a narrow screen
  await loadTray();
}
/* THE WAIT IS VISIBLE NOW. Opening an image is two round trips — the points
 * call, then the photograph — and until both land the canvas is blank. So the
 * editor opened onto an empty rectangle and the picture appeared with no
 * warning, which reads as broken and then as a jump. */
function edLoading(on){
  const el = document.getElementById("edLoading");
  if (el) el.hidden = !on;
}

async function loadTray(){
  const stem = ed.list[ed.idx];
  edLoading(true);
  const d = await api(`/api/points/${ed.collection}/${stem}`);
  Object.assign(ed, {stem, w:d.w, h:d.h, pts:d.points, target:d.target,
                     earmarked:d.earmarked,
                     classes:(d.classes && d.classes.length) ? d.classes : ["object"]});
  // Clamp: an image from a project with fewer classes than the one last open must
  // not leave `cls` pointing past the end of the vocabulary.
  if (ed.cls >= ed.classes.length) ed.cls = 0;
  ed.cropping = false; ed.crop = null; ed._drag = null; // crop mode never carries across images
  ed.sel = null; ed._rz = null;        // a selected INDEX means nothing on another image
  // Same for the dot selection, and it matters more: the pad would otherwise
  // stay open pointing at whatever dot happens to sit at that index on the NEW
  // image, so a nudge or a 🗑 would silently edit the wrong object.
  ed.dotSel = null; ed.dotOrig = null; ed.dotAdded = false; ed._dotDrag = false;
  if (window.edSyncPad) edSyncPad();
  $("#edCrop").classList.remove("active");
  if (window.laReset) laReset();       // LA crosshairs are per-image (la.js)
  if (window.auditReset) auditReset(); // …and so is the model comparison
  // Mirror of groundwork/dataset/filters/dot_dedupe.dot_radius, EXACTLY — including
  // Python's banker's rounding. This was the unrounded copy of the formula, so
  // the editor flagged stacked dots at a different threshold than the runtime
  // dedupes at. Math.round alone is not enough: it rounds .5 away from zero
  // while Python rounds .5 to even, which differs whenever the short side is
  // 100, 300, 500, 900, 1300... px (900 -> 4 not 5).
  ed.r = dotRadius(d.w, d.h);          // editor_mirrors.js <- dot_dedupe.dot_radius
  // Both destinations stay available at fix time (the phone press is not a
  // commitment): →test always confirms, and →training confirms when it
  // contradicts a 🎯 earmark. Mistakes remain fixable afterward with the
  // 🎯/↩ move buttons on the Training/Test tabs.
  $("#edDone").hidden = !ed.canDone;                         // → training
  $("#edDoneTest").hidden = !ed.canDone;                     // → test set
  edSyncClasses();
  $("#edHelp").innerHTML = helpHTML();
  edDirty = false;
  // If you arrived from "Check labels against the model", the comparison loads
  // itself — ~1s, unlike the LA probe's 30-90s, so a button press would just be
  // a step in the way. Off by default everywhere else (audit_overlay.js).
  if (window.auditAfterOpen) auditAfterOpen();
  ed.img = new Image();
  // Refit on first open, and whenever the next image is a different SIZE — a
  // zoom fitted to a landscape image is wrong for the portrait one after it.
  // Same-sized images keep the zoom you set, which is the deliberate behaviour
  // (you dot a run of images at one magnification).
  const _sig = `${d.w}x${d.h}`;
  ed.img.onload = () => {
    if (ed.zoom == null || ed.sig !== _sig) fitZoom();
    ed.sig = _sig;
    redraw();
    edLoading(false);          // only once something is actually on the canvas
  };
  /* A BROKEN IMAGE MUST NOT SPIN FOREVER. Without this the overlay is the last
   * thing on screen and says "Loading…" about a request that has already
   * failed — the one state worse than a blank canvas, because it never ends. */
  ed.img.onerror = () => {
    edLoading(false);
    const t = document.getElementById("edTitle");
    if (t) t.textContent = `${stem} — image failed to load`;
  };
  ed.img.src = withProject(`/img/${ed.collection}/${stem}?t=${Date.now()}`);
}
/* The class picker exists only where it means something.
 *
 * One class is every project that behaves as this editor always has, and a
 * dropdown with a single option is noise that also implies a choice you do not
 * have. HIDDEN, not disabled — and `ed.cls` is forced to 0, so a one-class
 * project can never write a class index its data.yaml does not declare.
 * */
/* 0 unless this project actually has more than one class — see edSyncClasses. */
function edPlaceClass(){
  return ed.classes.length > 1 ? ed.cls : 0;
}

function edSyncClasses(){
  const sel = $("#edClass");
  if (!sel) return;
  const multi = ed.classes.length > 1;
  sel.hidden = !multi;
  if (!multi){ ed.cls = 0; return; }
  sel.innerHTML = ed.classes
    .map((c, i) => `<option value="${i}">${i}: ${c}</option>`).join("");
  sel.value = String(ed.cls);
  sel.onchange = () => { ed.cls = +sel.value; redraw(); };
}

/* A colour per class, plus the INDEX drawn beside every dot when there is more
 * than one. Both, deliberately: the owner is colour-blind on blue/purple, and
 * more to the point a reader cannot learn a five-colour legend while dotting.
 * The number is the fact; the colour is the shortcut. */
const ED_CLASS_COLOURS = ["rgba(0,122,255,0.95)", "rgba(0,200,120,0.95)",
                          "rgba(255,150,0,0.97)", "rgba(220,80,200,0.95)",
                          "rgba(240,220,60,0.97)", "rgba(120,220,230,0.95)"];
const edClassColour = i => ED_CLASS_COLOURS[i % ED_CLASS_COLOURS.length];

async function gotoTray(delta){
  const ni = ed.idx + delta;
  if (ni < 0 || ni >= ed.list.length) return;         // at an end
  if (edDirty){
    // Never move on silently: choose to save or discard.
    const save = confirm("You have unsaved dots on this image.\n\n"
      + "OK = Save & continue    ·    Cancel = Discard changes & continue");
    if (save && !(await saveDots())) return;           // save failed -> stay put
  }
  ed.idx = ni;
  await loadTray();
}
$("#edPrev").onclick = () => gotoTray(-1);
$("#edNext").onclick = () => gotoTray(1);

/* The ⋯ tools menu lives in editor/editor_more.js — ED_NARROW, edCloseMore and
 * edSyncMore, plus its own listeners. Moved out because it shares no state with
 * anything here; openEditor and the resize handler still call edSyncMore(),
 * which resolves at runtime across classic scripts. */
$("#edHelp").onclick = () => $("#edHelp").classList.toggle("open");
window.addEventListener("keydown", e => {
  if ($("#editor").hidden) return;
  if ((e.key === "r" || e.key === "R") && ed.showBoxes && ed.sel != null){
    // Back to the spacing-derived default: drop the stored extent entirely,
    // rather than guessing a size. boxes_for then re-synthesises it.
    e.preventDefault();
    ed.pts[ed.sel] = ed.pts[ed.sel].slice(0, 3);
    edDirty = true; redraw();
    return;
  }
  if (e.key === "ArrowLeft"){ e.preventDefault(); gotoTray(-1); }
  else if (e.key === "ArrowRight"){ e.preventDefault(); gotoTray(1); }
  else if (e.key === "Escape"){ $("#edClose").click(); }
});
function fitZoom(){
  // Fit the WHOLE image, not just its width. #edCanvasWrap is flex:1 inside a
  // position:fixed inset:0 column, so clientWidth/clientHeight ARE the space
  // available. Fitting width alone meant a portrait phone photo (3024x4032 —
  // which is most of them) always overflowed vertically and had to be scrolled
  // to dot the bottom rows. Never upscale past 1: blowing an image up past native
  // makes objects mushy without showing more.
  const wrap = $("#edCanvasWrap");
  const byW = (wrap.clientWidth  - 20) / ed.w;
  const byH = (wrap.clientHeight - 20) / ed.h;
  // A measured 0 means "no layout yet", not "no room". Taking it literally
  // yields a NEGATIVE zoom and a canvas of nothing — the same shape as the
  // null*1.25 = 0 bug documented on the zoom buttons below. Fall back to the
  // dimension that did measure, then to native.
  const cands = [byW, byH].filter(v => v > 0);
  ed.zoom = cands.length ? Math.min(1, ...cands) : 1;
  ed.fitted = ed.zoom;      // remember, so resize can refit ONLY if untouched
}
/* redraw() — the whole canvas plus the title and nav chrome — lives in
 * editor/editor_draw.js. It only reads `ed`; nothing in it sets edDirty. */
/* Pointer geometry — evtImg, nearestPt, edSized, edHandles, edHandleAt, edBoxAt,
 * ED_MIN_BOX and edResizeTo — lives in editor/editor_hit.js. It reads `ed` and
 * writes `edDirty` from there, which works because those are lexical bindings
 * shared across classic scripts (see this file's header). */
canvas.addEventListener("mousedown", e => {
  if (e.button === 1) return;                    // middle: let it scroll
  const [x, y] = evtImg(e);
  if (ed.cropping){                              // crop mode: left-drag a keep-box
    if (e.button !== 0) return;
    ed._drag = [x, y]; ed.crop = [x, y, x, y]; redraw();
    return;                                      // no dot add/remove while cropping
  }
  if (ed.showBoxes){                             // BOX MODE: size only, never add/remove
    if (e.button !== 0) return;
    const h = edHandleAt(x, y);
    if (h){ ed._rz = h; return; }                // grabbed a handle -> resize
    ed.sel = edBoxAt(x, y);                      // else (re)select, or clear
    redraw();
    return;
  }
  const near = nearestPt(x, y);
  if (e.button === 2){                           // right: remove nearest
    if (near >= 0) ed.pts.splice(near, 1);
  } else if (e.button === 0){                     // left
    if (near < 0) ed.pts.push([x, y, edPlaceClass()]);  // add a dot of the active class
  }
  edDirty = true;
  redraw();
});
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousemove", e => {
  if (ed._rz){                                   // sizing a box
    const [x, y] = evtImg(e);
    edResizeTo(x, y, ed._rz.sx, ed._rz.sy, ed._rz.base); redraw();
    return;
  }
  if (!ed.cropping || !ed._drag) return;
  const [x, y] = evtImg(e);
  ed.crop = [ed._drag[0], ed._drag[1], x, y]; redraw();
});

/* The touch dot pad — tap to select, then nudge / ◐ / 🗑 / ✗ / ✓ — lives in
 * editor/editor_dotpad.js, along with edSelectDot, edDeselectDot, edNudge and
 * edSyncPad. It WRITES ed.pts and edDirty from there. */
/* Canvas touch gestures — pinch zoom, pan, tap-to-select and long-press-to-add,
 * plus zoomAt — live in editor/editor_touch.js. */
$("#edCanvasWrap").addEventListener("wheel", e => {
  if (!e.ctrlKey && !e.shiftKey) return;         // ctrl/shift+wheel zooms; plain wheel scrolls
  e.preventDefault(); ed.zoom = Math.min(6, Math.max(0.1, ed.zoom * (e.deltaY<0?1.15:0.87))); redraw();
}, {passive:false});
// Clamp BOTH ends on both buttons. ed.zoom starts null and fitZoom() only runs
// `if (ed.zoom == null)`, so a click before the image decodes made null*1.25 = 0,
// Math.min(6,0) = 0 — and since 0 == null is false, fitZoom() could never run
// again. The canvas stayed 0px wide for the rest of the session, across every
// image, because closeEditor deliberately keeps ed.zoom. Likeliest on a phone,
// where a 12MP image takes a moment to decode.
const zclamp = z => Math.min(6, Math.max(0.1, Number.isFinite(z) && z > 0 ? z : 1));
$("#edZoomIn").onclick  = () => { ed.zoom = zclamp((ed.zoom || 1) * 1.25); redraw(); };
$("#edZoomOut").onclick = () => { ed.zoom = zclamp((ed.zoom || 1) / 1.25); redraw(); };
async function saveDots(){
  try {
    const res = await fetch(`/api/points/${ed.collection}/${ed.stem}`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({points: ed.pts})});
    if (!res.ok) throw new Error("HTTP " + res.status);
    const r = await res.json();
    edDirty = false;
    $("#edSave").textContent = `Saved ${r.count} ✓`;
    setTimeout(()=>$("#edSave").textContent="Save", 1400);
    return true;
  } catch (e) {
    alert("⚠️ SAVE FAILED — your dots were NOT saved (" + e.message +
          ").\nIs the server still running? Do not close this window; "
          + "restart the server, then click Save again.");
    return false;
  }
}
$("#edSave").onclick = saveDots;
$("#edClear").onclick = () => {
  if (!ed.pts.length) return;
  if (!confirm(`Remove ALL ${ed.pts.length} dots from ${ed.stem}?\n\n`
      + `Nothing is written until you Save — Close discards.`)) return;
  ed.pts = []; edDirty = true; redraw();
};
$("#edDupes").onclick = () => {
  // findStacked is the port of dot_dedupe.find_stacked (editor_mirrors.js) —
  // the same rule the counter uses to actually remove them, so what the editor
  // flags is exactly what would be deduped.
  const drop = findStacked(ed.pts, ed.r);
  if (!drop.length){
    ed._flag = null; redraw();
    alert("✅ No stacked dots — every dot is far enough apart to be its own object.");
    return;
  }
  ed._flag = new Set(drop); redraw();      // show them in red first
  if (confirm(`⚠ Found ${drop.length} dot(s) stacked on top of another — too close to be`
      + ` separate objects (highlighted red). These inflate the count.\n\nRemove the`
      + ` ${drop.length} extra dot(s)? (You still need to Save afterwards.)`)){
    ed.pts = ed.pts.filter((_, i) => !drop.includes(i));
    edDirty = true;
  }
  ed._flag = null; redraw();
};
$("#edBoxes").onclick = () => {
  ed.showBoxes = !ed.showBoxes;
  ed.sel = null; ed._rz = null;        // leaving box mode drops the selection
  if (ed.showBoxes && ed.cropping){    // the two modes are mutually exclusive
    ed.cropping = false; ed.crop = null; $("#edCrop").classList.remove("active");
  }
  $("#edHelp").innerHTML = helpHTML();
  $("#edBoxes").classList.toggle("active", ed.showBoxes);
  edDeselectDot();                     // box mode sizes boxes; dots are not editable there
  redraw();
};
$("#edTarget").onclick = async () => {
  const cur = ed.target == null ? "" : ed.target;
  const v = prompt("True count for this image.\nLeave blank to clear it (unknown — just make every dot sit on an object).", cur);
  if (v === null) return;                            // cancelled
  const t = v.trim();
  const n = t === "" ? null : parseInt(t, 10);
  if (t !== "" && (isNaN(n) || n < 0)) return alert("Enter a whole number, or leave blank to clear.");
  try {
    const r = await fetch(`/api/truth/${ed.stem}`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({count: n})}).then(x => x.json());
    ed.target = r.target;                             // null when cleared
    redraw();
  } catch (e) { alert("Could not update target: " + e.message); }
};
/* "auto-scale to however large the page is" means tracking the page, not just
 * sampling it once. Refit ONLY when the zoom is still the one fitZoom chose —
 * if you have zoomed in to inspect an object, a window resize must not yank you
 * back out. */
let _edFitTimer = null;
window.addEventListener("resize", () => {
  // Crossing the 760px breakpoint (rotating a phone) changes whether ⋯ exists
  // at all — sync it before the early-return below, which only cares about zoom.
  if (!$("#editor").hidden) edSyncMore();
  if ($("#editor").hidden || ed.zoom == null || ed.zoom !== ed.fitted) return;
  clearTimeout(_edFitTimer);
  _edFitTimer = setTimeout(() => { fitZoom(); redraw(); }, 120);
});

// warn before leaving with unsaved dots (covers refresh / tab close)
window.addEventListener("beforeunload", e => { if (edDirty){ e.preventDefault(); e.returnValue = ""; } });
/* After an image is promoted it has LEFT this collection, so it must leave the
 * working list too — and then the list has already advanced: whatever sits at
 * the same index is the next image to fix.
 *
 * WHY NOT JUST CLOSE, which is what this did. Fixing a queue meant open an image,
 * correct it, press Done, get thrown back to the grid, find the next card, tap
 * it, wait for the image. Per image. On a phone, with a queue of fifty, the
 * hunting costs more than the dotting. Staying in the editor makes it
 * fix-fix-fix and closes by itself when the queue is empty, which is the only
 * moment you actually wanted the grid back.
 *
 * The grid and the badges are refreshed either way, so the page behind is
 * correct whenever you do leave. */
async function advanceAfterPromote(){
  edDirty = false;
  ed.list.splice(ed.idx, 1);                     // it is no longer in this queue
  loadGrid("needs_fix", "#fixGrid");             // keep the page behind honest
  loadState();
  if (!ed.list.length){                          // queue drained — that IS done
    closeEditor();
    return;
  }
  if (ed.idx >= ed.list.length) ed.idx = ed.list.length - 1;   // was the last one
  await loadTray();
}

$("#edDone").onclick = async () => {
  if (!await saveDots()) return;                 // don't promote if the save failed
  if (ed.earmarked && !confirm(`${ed.stem} was earmarked 🎯 for the TEST SET `
      + `on Telegram — send it to TRAINING instead?`)) return;
  await api(`/api/promote/${ed.stem}`, {method:"POST"});
  await advanceAfterPromote();
};
$("#edDoneTest").onclick = async () => {
  if (!await saveDots()) return;                 // don't promote if the save failed
  if (!confirm(`Send ${ed.stem} to the TEST SET (holdout)?\n\n`
      + `It won't be trained on — it'll only measure accuracy. `
      + `Good for an image the model got wrong.`)) return;
  const r = await api(`/api/promote_testset/${ed.stem}`, {method:"POST"});
  if (r && r.ok === false) return alert("🔒 " + (r.error || "move refused"));
  await advanceAfterPromote();
};
$("#edClose").onclick = () => {
  if (edDirty && !confirm("You have unsaved dot changes — close without saving?")) return;
  closeEditor();
};
function closeEditor(){ edDirty = false; $("#editor").hidden = true; ed.img = null; }
