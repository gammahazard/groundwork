"use strict";
/* LocateAnything probe — emergency second opinion for the dot editor.
 * Loaded LAST; shares page globals with cockpit.js ($, api, loadState) and
 * editor.js (ed, edDirty, redraw). editor.js calls the laDraw/laReset/
 * laTitleExtra hooks defined here; cockpit.js's card ✛ calls openLaModal().
 *
 * Flow: openLaModal -> POST /api/la/{collection}/{stem} {desc} -> poll GET
 * /api/la every 2s (the 3B runs 30–90s in its own subprocess) -> crosshairs
 * appear over the current image. Adopt replaces ed.pts (still needs Save);
 * nothing is ever written to disk by LA itself. */

const la = {pts: null, desc: "", timer: null, t0: null, appliedAt: null};

// Starter phrasings — a describe-to-detect prompt works best short and
// physical ("hex nut", "bottle cap"), so these seed that shape.
const LA_PRESETS = ["object", "round object", "small metal part", "screw", "coin"];

/* ---------- hooks editor.js calls ---------- */
window.laReset = () => { la.pts = null; laButtons(); };
window.laTitleExtra = () =>
  la.pts ? ` · <span class="laMark">✛ LA ${la.pts.length}</span>` : "";
window.laDraw = (ctx, z) => {
  if (!la.pts) return;
  const arm = Math.max(4, ed.r * 1.4 * z);
  ctx.lineWidth = Math.max(1.5, ed.r * 0.35 * z);
  ctx.strokeStyle = "rgba(230,60,210,0.95)";           // magenta ✛ (dots are blue)
  for (const [x, y] of la.pts){
    const cx = x * z, cy = y * z;
    ctx.beginPath();
    ctx.moveTo(cx - arm, cy); ctx.lineTo(cx + arm, cy);
    ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy + arm);
    ctx.stroke();
  }
};

/* ---------- toolbar: adopt / dismiss ---------- */
function laButtons(){
  $("#edLAAdopt").hidden = $("#edLADismiss").hidden = !la.pts;
  if (la.pts) $("#edLAAdopt").textContent = `✛→● Adopt LA (${la.pts.length})`;
}
$("#edLAAdopt").onclick = () => {
  if (!la.pts) return;
  if (!confirm(`Replace this image's ${ed.pts.length} dot(s) with LocateAnything's `
      + `${la.pts.length} crosshair(s)?\n\nStill needs Save — Close discards.`)) return;
  ed.pts = la.pts.map(p => [p[0], p[1], 0]);
  edDirty = true;
  la.pts = null; laButtons(); redraw();
};
$("#edLADismiss").onclick = () => { la.pts = null; laButtons(); redraw(); };
$("#edLA").onclick = () => openLaModal();

/* ---------- modal ---------- */
window.openLaModal = () => {
  $("#laChips").innerHTML = LA_PRESETS.map(p =>
    `<button class="chip" data-p="${p}">${p}</button>`).join("");
  document.querySelectorAll("#laChips .chip").forEach(b =>
    b.onclick = () => { $("#laDesc").value = b.dataset.p; });
  $("#laModal").hidden = false;
  pollLA();                       // reflect a probe already running (or its result)
};
function closeLaModal(){ $("#laModal").hidden = true; }
$("#laClose").onclick = closeLaModal;
$("#laModal").addEventListener("click", e => { if (e.target === $("#laModal")) closeLaModal(); });
window.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("#laModal").hidden) closeLaModal();
});

$("#laRun").onclick = async () => {
  const desc = $("#laDesc").value.trim() || "object";
  const r = await api(`/api/la/${ed.collection}/${ed.stem}`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({desc})});
  if (!r.ok){ $("#laStatus").textContent = "⚠️ " + (r.error || "could not start"); return; }
  la.t0 = Date.now();
  pollLA();
};
$("#laCancel").onclick = async () => {
  if (!confirm("Cancel the running probe?")) return;
  await fetch("/api/la", {method: "DELETE"});
  pollLA();
};

function laApplyResult(s){
  const res = s.result;
  if (!res) return;
  if (s.finished === la.appliedAt){          // already applied once — show a recap
    $("#laStatus").textContent =
      `last result: "${res.desc}" → ${res.count} ✛ on ${res.stem}. Run again for a fresh probe.`;
    return;
  }
  la.appliedAt = s.finished;
  if (res.stem !== ed.stem || res.collection !== ed.collection || $("#editor").hidden){
    $("#laStatus").textContent =
      `✅ finished, but the result is for ${res.stem} (${res.count} ✛) — open that image and rerun ✛.`;
    return;
  }
  la.pts = res.points; la.desc = res.desc;
  laButtons(); redraw();
  const diff = res.count - ed.pts.length;
  $("#laStatus").textContent = `✅ "${res.desc}" → ${res.count} crosshair(s) in ${res.seconds}s `
    + `(dots ${ed.pts.length}, ${diff > 0 ? "+" : ""}${diff}). `
    + `Compare on the image; ✛→● Adopt replaces the dots.`;
  closeLaModal();
}

async function pollLA(){
  clearTimeout(la.timer); la.timer = null;
  let s;
  try { s = await api("/api/la"); }
  catch { $("#laStatus").textContent = "⚠️ server unreachable"; return; }
  const running = s.status === "running";
  $("#laRun").disabled = running;
  $("#laCancel").hidden = !running;
  if (running){
    const el = Math.round((Date.now() - (la.t0 || s.started * 1000)) / 1000);
    $("#laStatus").textContent =
      `⏳ probing ${s.stem} for "${s.desc}" — ${el}s (typically 30–90s; fine to close this and keep editing)`;
    la.timer = setTimeout(pollLA, 2000);
  } else if (s.status === "done"){
    laApplyResult(s);
  } else if (s.status === "error"){
    $("#laStatus").textContent = "⚠️ " + (s.error || "failed")
      + (s.log_tail ? " — " + s.log_tail.split("\n").pop() : "");
  } else {
    $("#laStatus").textContent = "";
  }
}
