"use strict";
/* ✂ Crop mode — extracted from editor.js (size cap). SHARED-LEXICAL rules:
 * reads/writes `ed`, `edDirty`, `redraw` declared in editor.js, which loads
 * BEFORE this file; editor.js's mouse handlers call endCropDrag() at event
 * time, after this file has loaded. Top-level names here (GRID_SEL,
 * endCropDrag, applyCrop) must stay unique across every classic script —
 * check_frontend #5 enforces it. */

const GRID_SEL = {raw:"#trainGrid", testset:"#testGrid", needs_fix:"#fixGrid"};
function endCropDrag(){
  ed._drag = null;
  let [x1, y1, x2, y2] = ed.crop;
  [x1, x2] = [Math.min(x1, x2), Math.max(x1, x2)];
  [y1, y2] = [Math.min(y1, y2), Math.max(y1, y2)];
  if (x2 - x1 < 10 || y2 - y1 < 10){ ed.crop = null; redraw(); return; }  // stray click
  const outside = ed.pts.filter(p => p[0] < x1 || p[0] > x2 || p[1] < y1 || p[1] > y2).length;
  applyCrop(x1, y1, x2, y2, outside);
}
window.addEventListener("mouseup", () => {
  if (ed._rz){ ed._rz = null; return; }
  if (ed.cropping && ed._drag) endCropDrag();
});
async function applyCrop(x1, y1, x2, y2, outside){
  if (edDirty){
    // The server crops the dots ON DISK — unsaved edits would be silently
    // thrown away, and the removal warning below would count the wrong dots.
    if (!confirm("You have unsaved dot changes. Cropping applies to the SAVED "
        + "dots, so your edits must be saved first.\n\n"
        + "OK = Save & continue cropping · Cancel = abort the crop")){
      ed.crop = null; redraw(); return;
    }
    if (!(await saveDots())){ ed.crop = null; redraw(); return; }
  }
  const warn = outside
    ? `\n\n⚠ ${outside} dot(s) are OUTSIDE the box and will be REMOVED (the count changes).`
    : "";
  if (!confirm(`Crop ${ed.stem} to the selected box?${warn}\n\nThis overwrites the image `
      + `and its dots and can't be undone.`)){ ed.crop = null; redraw(); return; }
  try {
    const res = await fetch(`/api/crop/${ed.collection}/${ed.stem}`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({box:[x1, y1, x2, y2]})});
    if (!res.ok) throw new Error("HTTP " + res.status);
    const r = await res.json();
    ed.cropping = false; ed.crop = null; $("#edCrop").classList.remove("active");
    ed.zoom = null;                              // refit to the new (smaller) image
    await loadTray();
    loadGrid(ed.collection, GRID_SEL[ed.collection] || "#trainGrid"); loadState();
    $("#edSave").textContent = `Cropped ✓ ${r.count}`;
    setTimeout(() => $("#edSave").textContent = "Save", 1600);
  } catch (e) {
    alert("⚠️ Crop failed (" + e.message + "). The image was not changed.");
    ed.crop = null; redraw();
  }
}
$("#edCrop").onclick = () => {
  ed.cropping = !ed.cropping; ed.crop = null; ed._drag = null;
  if (ed.cropping && ed.showBoxes){    // mutually exclusive modes
    ed.showBoxes = false; ed.sel = null; ed._rz = null;
    $("#edBoxes").classList.remove("active");
  }
  $("#edCrop").classList.toggle("active", ed.cropping);
  $("#edHelp").innerHTML = helpHTML();
  edDeselectDot();                     // the pad has no meaning in crop mode
  redraw();
};
