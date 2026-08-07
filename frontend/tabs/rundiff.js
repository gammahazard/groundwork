"use strict";
/* Run-vs-run per-image diff (the Δ button on the Runs table): joins two runs'
 * saved eval results (/api/runs/{run}/images) on image stem and shows which
 * holdout images got FIXED, which BROKE, and which stayed wrong — the honest
 * answer to "did this run actually help, or just shuffle the errors?".
 * Baseline = the previous evaluated run in the ledger. Reuses #runModal.
 * Loaded after history.js (which wires the button to openRunDiff). */

window.openRunDiff = async run => {
  const runs = await api("/api/runs");                 // newest first
  const idx = runs.findIndex(r => r.run === run);
  let base = null;
  for (let j = idx + 1; j < runs.length; j++){
    if (runs[j].mae != null){ base = runs[j]; break; }
  }
  if (!base) return alert("No older evaluated run to compare against.");
  const fetchEval = async name => {
    const res = await fetch(`/api/runs/${name}/images`);
    return res.ok ? await res.json() : null;
  };
  const [ea, eb] = await Promise.all([fetchEval(base.run), fetchEval(run)]);
  const ta = ea && ea.images, tb = eb && eb.images;
  if (!ta || !tb)
    return alert("One of the runs has no saved per-image eval (pre-gallery run).");

  const A = Object.fromEntries(ta.map(t => [t.stem, t]));
  const B = Object.fromEntries(tb.map(t => [t.stem, t]));
  const stems = [...new Set([...Object.keys(A), ...Object.keys(B)])].sort();
  const rows = [];
  const n = {fixed: 0, broke: 0, improved: 0, worsened: 0, sameWrong: 0, exact: 0, holdoutChanged: 0};
  for (const s of stems){
    const a = A[s], b = B[s];
    if (!a || !b){ n.holdoutChanged++; rows.push({s, a, b, cls: "hchg"}); continue; }
    const da = a.pred - a.gt, db = b.pred - b.gt;
    if (da === 0 && db === 0){ n.exact++; continue; }          // boring: right both times
    let cls;
    if (da !== 0 && db === 0){ n.fixed++; cls = "fixed"; }
    else if (da === 0 && db !== 0){ n.broke++; cls = "broke"; }
    else if (Math.abs(db) < Math.abs(da)){ n.improved++; cls = "improved"; }
    else if (Math.abs(db) > Math.abs(da)){ n.worsened++; cls = "worsened"; }
    else { n.sameWrong++; cls = "sameWrong"; }
    rows.push({s, a, b, da, db, cls});
  }
  const chip = (label, count, cls) => count
    ? `<span class="dfChip ${cls}">${label} <b>${count}</b></span>` : "";
  const fmt = d => `<span class="${d === 0 ? "ok" : "short"}">${d > 0 ? "+" : ""}${d}</span>`;
  const CLS_WORD = {fixed: "✅ fixed", broke: "❌ broke", improved: "improved",
                    worsened: "worsened", sameWrong: "still wrong", hchg: "holdout changed"};
  const body = rows.length ? rows.map(r => {
    if (r.cls === "hchg")
      return `<tr class="df-hchg"><td>${r.s}</td><td colspan="3" class="mMeta">only in ${r.a ? base.run : run} (holdout changed between runs)</td><td>${CLS_WORD[r.cls]}</td></tr>`;
    const url = `${(eb && eb.preview_base) || ""}/${r.s}.png`;
    return `<tr class="df-${r.cls}">
      <td><a href="${url}" target="_blank" rel="noopener">${r.s}</a></td>
      <td>${r.a.gt}</td>
      <td>${r.a.pred} ${fmt(r.da)}</td>
      <td>${r.b.pred} ${fmt(r.db)}</td>
      <td>${CLS_WORD[r.cls]} <span class="mMeta">${r.a.bucket || ""}</span></td>
    </tr>`;
  }).join("") : `<tr><td colspan="5" class="mMeta">every image exact in both runs</td></tr>`;

  $("#runModalBody").innerHTML =
    `<h3 class="modalTitle">Δ ${run} vs ${base.run} — per-image holdout diff</h3>
     <p class="hint">MAE ${fmtScore(base.mae)} → <b>${fmtScore(runs[idx].mae)}</b>. Images exact in BOTH runs are
     hidden (${n.exact}); stem links open the new run's dot overlay.</p>
     <div class="dfChips">${chip("fixed", n.fixed, "fixed")}${chip("broke", n.broke, "broke")}
       ${chip("improved", n.improved, "improved")}${chip("worsened", n.worsened, "worsened")}
       ${chip("still wrong", n.sameWrong, "sameWrong")}${chip("holdout changed", n.holdoutChanged, "hchg")}</div>
     <div class="tblwrap"><table class="runs dfTable"><thead><tr>
       <th>image</th><th>gt</th><th>${base.run}</th><th>${run}</th><th>verdict</th>
     </tr></thead><tbody>${body}</tbody></table></div>`;
  $("#runModal").hidden = false;
  if (typeof showRunModal === "function") showRunModal(run);
};
