"use strict";
/* YOLO tab: train controls (Trainer-first via identity.js's hqTrainWrap),
 * live retrain progress/log, and the Models list + serving pin. Split out
 * of dashboard.js when the Dashboard became pure status. */

async function loadYolo(){
  const r = await api("/api/retrain");
  renderRetrain(r);
  loadModels();
  // Keep the Trainer-watch card + ghost row fresh while parked on this tab
  // (they used to update only via the Dashboard's loadState poll).
  if (window.IS_LAB === false) updateLabWatch();
}

let modelsExpanded = false;               // dashboard model list: collapsed by default
const MODELS_COLLAPSED = 6;               // show the newest N until "Show all"
async function loadModels(){
  const m = await api("/api/models");
  const box = $("#modelList"); box.innerHTML = "";
  const activeW = m.active && m.active.weights;
  if (!m.runs.length){ box.innerHTML = "<p class='hint'>No trained models yet.</p>"; return; }
  // Collapsed: newest N + always keep the currently-served run visible even if old.
  let shown = m.runs;
  if (!modelsExpanded && m.runs.length > MODELS_COLLAPSED){
    shown = m.runs.slice(0, MODELS_COLLAPSED);
    if (activeW && !shown.some(r => r.weights === activeW)){
      const act = m.runs.find(r => r.weights === activeW);
      if (act) shown = [...shown, act];
    }
  }
  for (const r of shown){
    const isActive = activeW ? (r.weights === activeW) : (r === m.runs[0]);
    const row = document.createElement("div"); row.className = "modelRow";
    const when = new Date(r.mtime * 1000).toLocaleString();
    row.innerHTML = `<span class="mName">${r.name}</span>`
      + `<span class="mMeta">imgsz ${r.imgsz} · ${when}</span>`
      + (isActive ? `<span class="mActive">● serving${activeW ? "" : " (newest)"}</span>`
                  : `<button class="mUse">Use this</button>`);
    if (!isActive) row.querySelector(".mUse").onclick = async () => {
      await fetch("/api/model/activate", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({weights: r.weights, imgsz: r.imgsz})});
      alert(`Set ${r.name} (imgsz ${r.imgsz}) as the model to serve.\n\n`
        + `Send /restart to the bot (and refresh here) to load it.`);
      loadState();
    };
    box.appendChild(row);
  }
  if (m.runs.length > MODELS_COLLAPSED){    // toggle to expand/collapse the rest
    const t = document.createElement("button");
    t.className = "modelToggle";
    t.textContent = modelsExpanded
      ? "▴ Show fewer" : `▾ Show all ${m.runs.length} runs`;
    t.onclick = () => { modelsExpanded = !modelsExpanded; loadModels(); };
    box.appendChild(t);
  }
  renderGhostRow();       // re-apply the Trainer ghost row after the rebuild
}
if ($("#modelUnpin")) $("#modelUnpin").onclick = async () => {
  await fetch("/api/model/activate", {method:"DELETE"});
  loadState();
};
if ($("#retrainCancel")) $("#retrainCancel").onclick = async () => {
  if (!confirm("Cancel the running training? The half-finished run is discarded "
    + "(won't appear in Models).")) return;
  await fetch("/api/retrain", {method:"DELETE"});
  pollRetrain();
};
const fmtEta = s => s == null ? "" : s < 90 ? `~${Math.round(s)}s`
  : `~${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`;

// Live progress bar during a retrain: epoch n/250 + current mAP50 + ETA from
// recent epoch pace (runs are fixed-length, so the ETA is honest). During the
// post-training eval sweep the bar sits full with an "evaluating…" note.
function renderProgress(r){
  const bar = $("#retrainBar"), fill = $("#retrainBarFill"), txt = $("#retrainBarText");
  const p = r.progress;
  if (r.status !== "running"){ bar.hidden = true; return; }
  $("#retrainVs").hidden = true;               // main branch re-shows when data exists
  if (p && p.total && p.epoch > 0){
    bar.hidden = false;
    fill.style.width = (100 * p.epoch / p.total).toFixed(1) + "%";
    let t = `${p.run} · epoch ${p.epoch}/${p.total}`;
    if (p.map50 != null) t += ` · mAP50 ${p.map50.toFixed(3)}`;
    if (p.epoch_s != null) t += ` · ${p.epoch_s}s/epoch`;
    if (p.eta_total_s != null) t += ` · ${fmtEta(p.eta_total_s)} until done (incl. eval)`;
    else if (p.eta_s != null) t += ` · ${fmtEta(p.eta_s)} left in training`;
    if (r.gpu) t += ` · GPU ${r.gpu.util}% ${r.gpu.vram_used_gb}/${r.gpu.vram_total_gb}GB`;
    txt.textContent = t;
    const v = p.vs_prev, vs = $("#retrainVs");
    vs.hidden = !v;
    if (v) vs.innerHTML = `<span class="vsChip vs-${v.verdict}">${
      v.verdict === "ahead" ? "📈 ahead of" : v.verdict === "behind" ? "📉 behind" : "≈ similar to"} ${v.run} `
      + `<b>${v.delta > 0 ? "+" : ""}${v.delta.toFixed(3)}</b> mAP @ epoch ${v.epoch}</span>`;
  } else if ((r.step || "").startsWith("evaluating")){
    bar.hidden = false; fill.style.width = "100%";
    txt.textContent = "training finished — sweeping conf×iou on the holdout…";
  } else {
    bar.hidden = false; fill.style.width = "2%";
    txt.textContent = "starting… (split + model warm-up)";
  }
}

function renderRetrain(r){
  window._lastRetrain = r;                    // peek.js reads this on its ticks
  const st = $("#retrainStatus");
  // #retrainBtn is gone — the one Train control (cockpit/runs/train.js) decides for
  // itself what is startable, from /api/train/options. Cancel is still this
  // panel's, because it belongs to the run rather than to starting one.
  const cancel = $("#retrainCancel");
  if (cancel) cancel.hidden = r.status !== "running";
  renderProgress(r);
  if (window.peekTick) peekTick(r);
  // THE LOG PANEL IS run_log.js's NOW. This wrote HQ's own log_tail, which
  // meant the terminal was blank for every run on the worker — and if both wrote
  // it, they would fight on every heartbeat. This file keeps the status line.
  if (r.status === "running"){ st.textContent = "⏳ " + r.step; }
  else if (r.status === "error"){ st.textContent = "⚠️ " + r.step; }
  /* NO "✅ done — count MAE …" LINE (owner, 2026-08-02). It restated a number
   * that is already on screen twice by then — the Models list gains the run with
   * its MAE, and the Runs table has the full row with the previous run to compare
   * against. Three copies of one number is how they get to disagree; the two that
   * survive are the ones attached to the run they describe.
   *
   * The two-size comparison it also carried has the same answer: batch_results
   * lands in Runs as two rows, which is where you would compare them anyway.
   *
   * running/error still speak, because those describe something happening NOW
   * that no other panel is reporting. */
  else st.textContent = "";
  if (r.status === "running" && !pollTimer) pollRetrain();   // resume live updates after a refresh
}
/* The Retrain button lived here. Training is one control now — model, machine
 * and card — in cockpit/runs/train.js, posting to /api/train. This file keeps what
 * is still its own: the live run's status, cancel, and the Models/pin list. */
let pollTimer = null;
function pollRetrain(){
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const r = await api("/api/retrain"); renderRetrain(r);
    if (r.status !== "running"){ clearInterval(pollTimer); pollTimer = null; loadState(); }
  }, 2000);
}
