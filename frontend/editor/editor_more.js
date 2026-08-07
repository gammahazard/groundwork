"use strict";
/* The editor's ⋯ tools menu — narrow screens only.
 *
 * SPLIT OUT OF editor.js (2026-08-02), which was 963 lines. This cluster went
 * first precisely because it touches NOTHING shared: no `ed`, no `edDirty`, no
 * redraw — only its own DOM ids. That makes it the cheap way to prove the split
 * mechanism before anything riskier moves.
 *
 * Classic script, and load order matters in one direction only: the top-level
 * lines below bind handlers at parse time, so this must come AFTER
 * cockpit/core.js (which defines `$`) and after the markup exists. editor.js
 * CALLS edSyncMore() — from openEditor and from the resize handler — but only at
 * runtime, so it does not care which of the two parsed first.
 */

/* ---------- the ⋯ tools menu (narrow screens only) ----------------------
 * The bar carries fifteen buttons. On a 390px phone they wrapped to four rows
 * and, with the help text, took 39% of the viewport before the image appeared —
 * on the one screen whose entire job is showing you an image. The secondary ones
 * now live behind ⋯; CSS decides whether that wrapper is a menu or nothing at
 * all (display:contents), so there is no second toolbar to keep in sync and no
 * DOM is moved.
 *
 * The button is shown by the same 760px breakpoint the CSS uses, evaluated in
 * JS so `hidden` and the media query cannot disagree. */
const ED_NARROW = () => window.innerWidth <= 760;
function edCloseMore(){
  $("#edMore").classList.remove("open");
  $("#edMoreBtn").setAttribute("aria-expanded", "false");
}
function edSyncMore(){
  const narrow = ED_NARROW();
  $("#edMoreBtn").hidden = !narrow;
  if (!narrow) edCloseMore();          // leaving narrow must not strand it open
}
$("#edMoreBtn").onclick = e => {
  e.stopPropagation();
  const open = $("#edMore").classList.toggle("open");
  $("#edMoreBtn").setAttribute("aria-expanded", String(open));
};
// Any tool you pick closes the menu — leaving it open over the thing you just
// asked for is the classic menu mistake, and here it would cover the image.
$("#edMore").addEventListener("click", e => { if (e.target.closest("button")) edCloseMore(); });
/* Tapping anywhere else closes it, including on the canvas — and that has to be
 * pointerdown, not click. The canvas touchend calls preventDefault() to stop the
 * browser synthesising a mouse click (otherwise every tap would fire the mouse
 * handlers too and double-act, as the touch block below explains). That
 * suppression also eats the click this listener would have needed, so on a phone
 * the menu stayed open over the image. pointerdown fires for mouse and touch
 * alike and is not suppressed. */
document.addEventListener("pointerdown", e => {
  if (!$("#editor").hidden && !e.target.closest("#edMore, #edMoreBtn")) edCloseMore();
});
window.addEventListener("keydown", e => {
  if (e.key === "Escape" && $("#edMore").classList.contains("open")){
    e.stopPropagation(); edCloseMore();
  }
});
