"use strict";
/* Duplicate sweeps UI — the 🔍 buttons on the Training/Test set tabs. Runs
 * the full three-way check (train↔test leak + within each bucket), merging
 * both server detectors (image hash + label-geometry re-shoot twins), and
 * renders each flagged pair as side-by-side thumbnails with a per-copy 🗑 so
 * a human eyeballs before anything is deleted. Loaded after buckets.js
 * (uses GRID_OF) and cockpit.js ($, api, loadGrid, loadState). */

const DEDUP_SWEEPS = [
  ["raw", "testset", "train ↔ test",
   "same capture on BOTH sides — a leak: ↩ the test copy or 🗑 the training copy"],
  ["raw", "raw", "within training",
   "same capture twice in training (redundant) — 🗑 one copy"],
  ["testset", "testset", "within test set",
   "same capture twice in the holdout (double-weights its MAE) — 🗑 one copy"],
];

function dupeSide(coll, stem){
  return `<figure class="dupeSide">`
    + `<a href="${withProject(`/img/${coll}/${stem}`)}" target="_blank" rel="noopener">`
    + `<img loading="lazy" decoding="async" src="${withProject(`/img/${coll}/${stem}?w=320`)}"></a>`
    + `<figcaption><b>${stem}</b><br><span class="mMeta">${coll === "raw" ? "training" : coll}</span><br>`
    + `<button class="dupeDel" data-coll="${coll}" data-stem="${stem}">🗑 delete this copy</button>`
    + `</figcaption></figure>`;
}

async function runDedupSweep(el){
  el.textContent = "checking…";
  let html = "";
  for (const [a, b, label, advice] of DEDUP_SWEEPS){
    const r = await api(`/api/dedup?a=${a}&b=${b}`);
    // Merge the two detectors: image hash (re-encodes/copies) + label-geometry
    // twins (re-shoots the hash misses). A pair found by both shows once.
    const seen = new Map();
    for (const p of (r.flagged || [])) seen.set(p.a + "|" + p.b, {...p, how: `image hash ${p.dist}`});
    for (const p of (r.twins || [])){
      const k = p.a + "|" + p.b;
      if (seen.has(k)) seen.get(k).how += ` + geometry ${p.dist}`;
      else seen.set(k, {...p, how: `dot geometry ${p.dist}`});
    }
    const pairs = [...seen.values()];
    if (!pairs.length){ html += `<div class="dedupLine">✅ ${label}: none</div>`; continue; }
    html += `<div class="dedupLine">⚠ <b>${label}</b>: ${pairs.length} pair(s) — ${advice}. `
      + `Eyeball each pair below; delete ONE copy if they're really the same capture.</div>`;
    for (const p of pairs){
      html += `<div class="dupePair">${dupeSide(a, p.a)}${dupeSide(b, p.b)}`
        + `<div class="dupeMeta">${p.how}</div></div>`;
    }
  }
  el.innerHTML = html
    + `<div class="mMeta">image hash: ~0 = same capture, 13+ = different · geometry: &lt;0.8 = same arrangement</div>`;
  el.querySelectorAll(".dupeDel").forEach(btn => btn.onclick = async () => {
    const coll = btn.dataset.coll, stem = btn.dataset.stem;
    if (!confirm(`Delete ${stem} from ${coll === "raw" ? "training" : coll}?\n\n`
        + `The other copy stays. This removes the image + its dots.`)) return;
    const r = await api(`/api/image/${coll}/${stem}`, {method: "DELETE"});
    if (r && r.ok === false && r.error) return alert("🔒 " + r.error);
    runDedupSweep(el);                              // re-verify after the removal
    if (GRID_OF[coll]) loadGrid(coll, GRID_OF[coll]);
    loadState();
  });
}
$("#dedupBtn").onclick = () => runDedupSweep($("#dedupResult"));
$("#dedupTrainBtn").onclick = () => runDedupSweep($("#dedupTrainResult"));
