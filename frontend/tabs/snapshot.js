"use strict";
/* 📸 Dataset-snapshot modal for the Runs tab. Classic script, loaded after
   history.js; shares $, api and the run modal. history.js delegates via
   window.openRunSnapshot(run). */

async function openRunSnapshot(run){
  const res = await fetch(`/api/runs/${run}/snapshot`);
  if (!res.ok){
    $("#runModalBody").innerHTML = `<h3 class="modalTitle">${run} — dataset snapshot</h3>
      <p class="hint">${res.status === 404
        ? "No snapshot — this run predates dataset snapshots (they are written automatically at the end of every retrain from now on)."
        : "Could not load the snapshot."}</p>`;
    showRunModal(run); return;
  }
  const d = await res.json();
  const when = new Date(d.created * 1000).toLocaleString();
  const section = (name, c) => {
    const drift = c.missing.length + c.image_changed.length + c.label_changed.length;
    const li = (label, stems, restorable) => stems.length
      ? `<li>${label}: ${stems.map(s =>
          `<code>${s}</code>${restorable
            ? ` <button class="snapRestore" data-run="${run}" data-stem="${s}">restore</button>` : ""}`
        ).join(", ")}</li>` : "";
    return `<div class="snapSec"><b>${name}</b> — ${c.images} images, `
      + (drift ? `<span class="short">${drift} drifted since</span>` : `<span class="ok">unchanged</span>`)
      + `<ul class="hint">`
      + li("missing now", c.missing, true)
      + li("image edited since", c.image_changed, false)
      + li("labels edited since", c.label_changed, false)
      + `</ul></div>`;
  };
  $("#runModalBody").innerHTML =
    `<h3 class="modalTitle">${run} — dataset snapshot <span class="mMeta">(${when})</span></h3>
     <p class="hint">Exactly what this run trained and tested on. "Drifted" images differ
     from today's dataset; <b>restore</b> brings a now-missing image back byte-exact
     (it never overwrites an existing image).</p>`
    + Object.entries(d.collections).map(([n, c]) => section(n, c)).join("");
  $("#runModalBody").querySelectorAll("button.snapRestore").forEach(b => {
    b.onclick = async () => {
      if (!confirm(`Restore ${b.dataset.stem} exactly as run ${b.dataset.run} saw it?`)) return;
      const r = await api(`/api/runs/${b.dataset.run}/restore/${b.dataset.stem}`, {method: "POST"});
      if (r && r.ok === false) return alert("🔒 " + (r.error || "restore refused"));
      alert(`Restored ${b.dataset.stem} → ${r.collection === "testset" ? "test set" : "training (raw)"}.`);
      openRunSnapshot(b.dataset.run);      // re-render the diff
    };
  });
  showRunModal(run);
}
window.openRunSnapshot = openRunSnapshot;
