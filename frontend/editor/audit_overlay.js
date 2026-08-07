"use strict";
/* Where THIS image disagrees with the model — only the disagreements.
 *
 * Loaded last, like la.js, and it borrows that file's shape: a per-image point
 * set, a draw hook editor.js calls, a title extra, and a reset. What it does
 * NOT borrow is LocateAnything itself — LA is a 3B VLM and takes 30-90s, which
 * is why it is an emergency button. This runs the project's SERVING model, the
 * same one /api/count uses for every bot photo: ~1s per image including decoding
 * a 12MP JPEG (measured: 160 images in 138s). So it loads on arrival rather than
 * being something you press and wait for.
 *
 * ONLY THE DISAGREEMENTS, and that is the whole design.
 *
 * The question a flagged image raises is "the model says 92 and I have 90 —
 * WHERE are the two?". Drawing all 92 model dots over your 90 answers it by
 * making you find the odd one out among 182 marks. Drawing the two side by side
 * is worse: locating a 2-dot difference then means eyeball-matching 90
 * near-identical white objects across a gap.
 *
 * So the two sets are PAIRED by nearest neighbour and everything that pairs is
 * left alone. What is left is typically one to four marks:
 *
 *     ⊕  the model saw something here and you have no dot   (missing label?)
 *     ⊖  you have a dot here the model does not see         (false dot? hard object?)
 *
 * NEITHER IS A VERDICT. The model is wrong sometimes — that is what the fix
 * queue is for — so a ⊕ means "look here", not "you missed one".
 *
 * The pairing threshold is 2x the editor's own dot radius (dotRadius(), the
 * same function point_to_box uses), so a dot that is merely a little off-centre
 * still pairs and does not show up as a phantom disagreement.
 */

const audit = {pts: null, model: 0, dots: 0, extra: [], missing: [],
               missConf: [], servedConf: null, modelName: null,
               showAll: false, want: false, busy: false, stem: null};

/* ---------- hooks editor.js calls ---------- */
window.auditReset = () => {
  audit.pts = null; audit.extra = []; audit.missing = []; audit.missConf = [];
  audit.stem = null; audit.showAll = false; audit.modelName = null;
};

window.auditTitleExtra = () => {
  if (!audit.pts) return "";
  const n = audit.extra.length, m = audit.missing.length;
  // WHICH model agreed. The audit runs the PINNED weights, so this is about
  // one specific run — pin a different one and the answer can change.
  const who = audit.modelName ? ` [${audit.modelName}]` : "";
  if (!n && !m)
    return ` · <span class="auditMark ok">✓ model agrees (${audit.model})${who}</span>`;
  /* WHY, not just how many. A ⊖ the model scored just under the served
   * threshold means the label is fine and the threshold cut it — nothing to
   * fix, and loosening the threshold is exactly what lets a fingertip back in.
   * A ⊖ with nothing there at any score means the model is blind to that object,
   * which is a reason to keep the image in training. Same count, opposite
   * response, and telling them apart used to mean a separate run by hand. */
  const seen = audit.missConf.filter(c => c != null);
  const blind = audit.missConf.length - seen.length;
  let why = "";
  if (seen.length) {
    const hi = Math.max(...seen);
    why = ` — ${seen.length} it scored up to ${hi.toFixed(3)}, under the `
        + `${audit.servedConf} it serves at (threshold, not a miss)`;
  }
  if (blind) why += `${why ? "; " : " — "}${blind} it does not see at any score`;
  return ` · <span class="auditMark">⚠${who} model ${audit.model} vs your ${audit.dots}`
       + `${n ? `, ${n} it sees that you do not` : ""}${why}</span>`;
};

/* Drawn AFTER the dots (editor.js calls this from redraw), so a mark is never
 * hidden under one. Ring + sign rather than a filled dot: the image already has
 * filled dots on it and a third filled shape reads as another label. */
window.auditDraw = (ctx, z) => {
  if (!audit.pts) return;
  const R = Math.max(6, ed.r * 2.2 * z);
  ctx.lineWidth = Math.max(2, ed.r * 0.5 * z);
  ctx.font = `bold ${Math.max(11, R * 1.1)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const mark = (pt, colour, glyph) => {
    const [x, y] = [pt[0] * z, pt[1] * z];
    ctx.strokeStyle = colour;
    ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = colour;
    ctx.fillText(glyph, x, y + R * 1.6);
  };
  // Amber for "the model sees one here", cyan for "you have one here it does
  // not". They also carry a SIGN, because the owner is colour-blind and two
  // rings that differ only in hue say nothing.
  if (audit.showAll)
    for (const p of audit.pts) {
      ctx.strokeStyle = "rgba(150,160,175,.55)";
      ctx.beginPath(); ctx.arc(p[0] * z, p[1] * z, R * 0.7, 0, Math.PI * 2); ctx.stroke();
    }
  for (const p of audit.extra) mark(p, "rgba(224,168,62,.95)", "+");
  audit.missing.forEach((p, i) => {
    const c = audit.missConf[i];
    // The score right on the mark: "0.19" next to a ⊖ says "it saw this, the
    // threshold cut it" without a trip back to the title.
    mark(p, "rgba(80,200,220,.95)", c == null ? "−" : `− ${c.toFixed(2)}`);
  });
};

async function auditLoad(force) {
  // BARE `ed`, not `window.ed`. editor.js declares it as a top-level `const`,
  // and in a classic script a top-level const/let does NOT become a window
  // property — so `window.ed` is undefined and this returned instantly, with no
  // request, no error and nothing on screen. (la.js gets this right by
  // referencing `ed` directly; they share the global lexical scope.) Third time
  // this exact scoping rule has cost time in this codebase.
  if (typeof ed === "undefined" || !ed.stem) return;
  if (audit.busy) return;
  if (!force && !audit.want) return;
  if (audit.stem === ed.stem && audit.pts) return;    // already have this image
  audit.busy = true;
  const btn = $("#edAudit");
  if (btn) { btn.disabled = true; btn.textContent = "⊕ asking the model…"; }
  try {
    const r = await apiOrThrow(
      `/api/label_audit/${encodeURIComponent(ed.collection)}/${encodeURIComponent(ed.stem)}`,
      {method: "POST"});
    // THE PAIRING IS THE SERVER'S. It used to be duplicated here, which made it
    // a second rule to keep in step — and it compared against the dots on
    // SCREEN, so an unsaved edit changed what the audit said. An audit is of
    // what is on disk.
    audit.pts = r.points || [];
    audit.model = r.model;
    audit.dots = r.dots;
    audit.stem = ed.stem;
    audit.servedConf = r.served_conf;
    audit.modelName = r.model_name || null;
    audit.extra = (r.extra || []).map(e => [e.x, e.y]);
    audit.missing = (r.missing || []).map(m => [m.x, m.y]);
    // WHY each ⊖ happened, in the same order as audit.missing.
    audit.missConf = (r.missing || []).map(m => m.conf);
    if (typeof redraw === "function") redraw();
    if (typeof edTitle === "function") edTitle();
  } catch (e) {
    audit.pts = null;
    alert(`Could not compare with the model: ${e.message || e}`);
  }
  audit.busy = false;
  if (btn) { btn.disabled = false; btn.textContent = "⊕ Compare"; }
}

/* Arriving from the audit table turns it on for the image you were sent to; the
 * ⊕ button turns it on for any other. */
window.auditWant = () => { audit.want = true; };
window.auditToggleAll = () => {
  audit.showAll = !audit.showAll;
  if (typeof redraw === "function") redraw();
};
window.auditAfterOpen = () => auditLoad(false);

(() => {
  const b = document.querySelector("#edAudit");
  if (b) b.onclick = () => { audit.want = true; auditLoad(true); };
  const a = document.querySelector("#edAuditAll");
  if (a) a.onclick = () => auditToggleAll();
})();
