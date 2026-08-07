"use strict";
/* 🔬 Watch-training modal — a live look inside the running retrain, built
 * ENTIRELY from artifacts ultralytics already writes into the run dir (served
 * by the /outputs static mount): the actual augmented batches fed to the
 * model, the label-geometry plot, per-epoch curves, and (near the end) the
 * first predictions-vs-labels images. Training itself is never touched.
 * Reuses the shared #runModal and history.js's svgLineChart. cockpit.js
 * stashes each /api/retrain poll in window._lastRetrain and calls
 * peekTick(r) — so everything here updates live at the 2s poll cadence. */

let peekOpen = false;
let peekImgsKey = "";           // re-render the image sections only on change

$("#peekBtn").onclick = () => {
  peekOpen = true; peekImgsKey = "";
  $("#runModalBody").innerHTML = "";        // clear whatever the modal held
  renderPeek();
  $("#runModal").hidden = false;
};

/* THE BUTTON FOLLOWS THE RUN, NOT THE MACHINE.
 *
 * peekTick was fed only by HQ's own /api/retrain, so the moment training moved to
 * the worker — which is where it lives now — 🔬 Watch training simply stopped
 * appearing. Same machine-instead-of-run bug run_log.js and run_vis.js already
 * fixed for logs and pictures.
 *
 * Two tickers, one decision. Each stashes what its source last said and calls
 * peekSync(); whichever is RUNNING wins. Letting them each set `.hidden`
 * directly meant the idle one hid a button the busy one had just shown, which is
 * a race that depends on poll order.
 *
 * The lab payload is usable unchanged because api/state.py absolutises
 * progress.peek.base to the worker's /outputs mount — every image URL below is
 * built off that one string.
 */
let peekLocal = null, peekLab = null;

function peekSource(){
  if (peekLocal && peekLocal.status === "running") return peekLocal;
  if (peekLab && peekLab.status === "running") return peekLab;
  return null;
}

function peekSync(){
  const live = peekSource();
  $("#peekBtn").hidden = !live;
  if (!peekOpen) return;
  if ($("#runModal").hidden){ peekOpen = false; return; }   // user closed it
  if (live) renderPeekBody(live, true);
}

window.peekTick = r => { peekLocal = r; peekSync(); };
// Called from identity.js's updateLabWatch with /api/lab_status, which now
// carries the worker's full `progress`.
window.peekLabTick = r => { peekLab = (r && r.ok) ? r : null; peekSync(); };

// Runs-tab post-mortem 🔬: same modal, fed once from /api/runs/{run}/peek
// (the artifacts persist in the run dir), no live ticking.
window.openRunPeek = async run => {
  const res = await fetch(`/api/runs/${run}/peek`);
  if (!res.ok){ alert(`No saved training artifacts for ${run}.`); return; }
  const p = await res.json();
  peekOpen = false; peekImgsKey = "";
  $("#runModalBody").innerHTML = "";
  renderPeekBody({status: "done", step: "", progress: p}, false);
  $("#runModal").hidden = false;
  if (typeof showRunModal === "function") showRunModal(run);  // row highlight
};

const PEEK_STAGES = ["split", "train (mosaic)", "train (real layouts)", "holdout eval"];

function peekStageInfo(r, live){
  const p = r.progress || {};
  const cmn = p.close_mosaic || 0, total = p.total || 0, ep = p.epoch || 0;
  const realPhase = total && cmn && ep > total - cmn;
  const step = r.step || "";
  if (!live)
    return {idx: 4, narr: "Finished run — these are the artifacts saved while it trained "
      + "(batches, curves, validation predictions)."};
  if (step.startsWith("split"))
    return {idx: 0, narr: "Copying images into train/val (image-level, hash-stable) and running the duplicate/leak checks."};
  if (step.startsWith("evaluating"))
    return {idx: 3, narr: "Training finished. Sweeping conf × NMS-IoU across the whole holdout for BOTH checkpoints "
      + "(best.pt and last.pt) — whichever COUNTS better gets promoted, and its tuned thresholds ship with it."};
  if (realPhase)
    return {idx: 2, narr: `Mosaic is OFF for the final ${cmn} epochs — the model now finishes on real, `
      + "un-collaged images, the same layouts it will count in production."};
  return {idx: 1, narr: "Each step: grab a shuffled batch, collage 4 images into 1 (mosaic), jitter the colors (HSV), "
    + "flip/scale, predict every object, and nudge the weights by the error. The collages below are REAL batches "
    + "from this run — literally what the model is looking at."};
}

/* Whichever source is live — the local retrain or the worker's. Reading
 * window._lastRetrain directly meant opening the modal during a RIG run
 * rendered HQ's idle payload: an empty modal with no error. */
function renderPeek(){ renderPeekBody(peekSource() || window._lastRetrain, true); }

function renderPeekBody(r, live){
  if (!r) return;
  const p = r.progress || {};
  const body = $("#runModalBody");
  if (!body.querySelector("#peekWrap")){
    body.innerHTML = `<h3 class="modalTitle">🔬 Inside ${p.run || "the retrain"}${live ? "" : " (finished)"}</h3>
      <div id="peekWrap">
        <div id="peekStages" class="peekStages"></div>
        <p id="peekNarr" class="hint"></p>
        <div id="peekVs" class="peekVs"></div>
        <div id="peekCharts" class="charts"></div>
        <div id="peekImgs"></div>
      </div>`;
  }
  const info = peekStageInfo(r, live);
  $("#peekStages").innerHTML = PEEK_STAGES.map((s, i) =>
    `<span class="pkStage${i < info.idx ? " done" : i === info.idx ? " cur" : ""}">${s}</span>`
  ).join(`<span class="pkArrow">→</span>`);
  $("#peekNarr").textContent = info.narr;

  // Trajectory verdict vs the previous run at the same epoch (live only).
  const v = p.vs_prev;
  $("#peekVs").innerHTML = !live || !v ? "" :
    `<span class="vsChip vs-${v.verdict}">${v.verdict === "ahead" ? "📈 tracking ahead of"
      : v.verdict === "behind" ? "📉 tracking behind" : "≈ similar to"} ${v.run}
     <b>${v.delta > 0 ? "+" : ""}${v.delta.toFixed(3)}</b> mAP @ epoch ${v.epoch}</span>
     <span class="mMeta"> — val-mAP leading indicator; the verdict that counts is the holdout MAE at the end</span>`;

  if (typeof svgLineChart === "function"){
    const mk = (arr, color, title, fmt, ghost, legend) => svgLineChart({
      points: (arr || []).map((y, i) => ({label: String(i + 1), y})),
      color, title, yfmt: fmt, markers: false, ghost, legend});
    const ghost = live ? p.prev_map50_curve : null;
    const lv = live ? " — live" : "";
    $("#peekCharts").innerHTML =
      mk(p.map50_curve, "#3987e5",
         `mAP50 by epoch${lv} (${p.epoch ?? 0}/${p.total ?? "?"})`,
         v2 => v2.toFixed(2), ghost,
         ghost ? `— this run&nbsp; ┄ ${p.prev_run}` : "&nbsp;")
      + mk(p.box_loss_curve, "#e0a83e",
           `box loss${lv} — edge placement (the slow grind)`,
           v2 => v2.toFixed(2), null, "&nbsp;")
      + ((p.cls_loss_curve || []).length ? mk(p.cls_loss_curve, "#b07de0",
           `cls loss${lv} — "is it an object?" (collapses early)`,
           v2 => v2.toFixed(2), null, "&nbsp;") : "")
      + ((p.dfl_loss_curve || []).length ? mk(p.dfl_loss_curve, "#2fbfa0",
           `dfl loss${lv} — box-edge sharpness (fights NMS merges)`,
           v2 => v2.toFixed(2), null, "&nbsp;") : "");
  }

  const pk = p.peek;
  if (!pk) return;
  const cmn = p.close_mosaic || 0, realPhaseNow = live && p.total && cmn
    && (p.epoch || 0) > p.total - cmn;
  const key = JSON.stringify([pk.mosaic_batches, pk.real_batches, pk.labels,
                              pk.val_pairs, pk.stream, pk.timelapse, realPhaseNow]);
  if (key === peekImgsKey) return;        // images unchanged: no reflicker
  peekImgsKey = key;
  const t = Date.now();
  const fig = (src, cap) => `<figure class="pkFig">`
    + `<a href="${src}" target="_blank" rel="noopener"><img loading="lazy" src="${src}?t=${t}"></a>`
    + `<figcaption>${cap}</figcaption></figure>`;
  let html = "";
  // Rotating carousel of the REAL saved batches. ultralytics photographs the
  // first 3 batches (epoch 1, mosaic on) and 3 more the moment mosaic closes;
  // streaming the literal current batch would mean instrumenting training, so
  // we cycle the genuine saved samples — and the close-mosaic set joins the
  // rotation automatically when it appears (~epoch total-close_mosaic).
  pkCarList = [];
  pk.mosaic_batches.forEach(n => pkCarList.push({
    src: `${pk.base}/${n}?t=${t}`, name: n,
    tag: "mosaic collage · saved at epoch 1"}));
  pk.real_batches.forEach(n => pkCarList.push({
    src: `${pk.base}/${n}?t=${t}`, name: n,
    tag: realPhaseNow ? "real layout · it's training on these NOW"
                      : "real layout · saved when mosaic closed"}));
  // Live stream: rolling every-N-steps snapshots from LiveVizTrainer — a
  // genuinely fresh collage every ~minute while training runs.
  (pk.stream || []).forEach(n => {
    const step = parseInt((n.match(/(\d+)/) || [])[1] || 0, 10);
    pkCarList.push({src: `${pk.base}/live/${n}?t=${t}`, name: n,
                    tag: `🔴 live snapshot · step ${step}`});
  });
  if (pkCarList.length){
    html += `<h4 class="pkH">What the model actually sees — cycling the saved batches</h4>
      <p class="hint">4 images collaged per tile, colors jittered, flipped/scaled — how ~100 images behave
      like 1000. Boxes = the labels it learns from. The real-layout set joins the rotation
      automatically once mosaic closes.</p>
      <div class="pkCar">
        <button id="pkCarPrev" title="Previous batch">‹</button>
        <a id="pkCarLink" target="_blank" rel="noopener"><img id="pkCarImg" alt="training batch"></a>
        <button id="pkCarNext" title="Next batch">›</button>
      </div>
      <div id="pkCarCap" class="mMeta pkCarCap"></div>`;
  }
  // "Watch it learn" filmstrip: same holdout images, one frame per periodic
  // checkpoint — drag the slider (or it starts on the final frame).
  const tl = pk.timelapse;
  if (tl && tl.images && Object.keys(tl.images).length){
    html += `<h4 class="pkH">Watch it learn — the same holdout images, every 25 epochs</h4>
      <p class="hint">Each frame is a periodic checkpoint predicting the SAME image
      (fixed conf ${tl.conf} / iou ${tl.iou} so frames are comparable). Drag the slider:
      early epochs miss and double-count; the final frame is the shipped best.pt.</p>`;
    for (const [stem, frames] of Object.entries(tl.images)){
      if (!frames.length) continue;
      const last = frames.length - 1;
      html += `<div class="tlBlock" data-stem="${stem}">
        <div class="tlHead"><b>${stem}</b> · gt ${frames[0].gt} ·
          <span class="tlLabel"></span></div>
        <input type="range" class="tlSlider" min="0" max="${last}" value="${last}"
               data-frames='${JSON.stringify(frames)}' data-base="${pk.base}/live/">
        <img class="tlImg" loading="lazy">
      </div>`;
    }
  }
  // "Where it looks" — feature-activation heatmaps on the probe images.
  const hm = pk.heatmaps;
  if (hm && hm.images && Object.keys(hm.images).length){
    html += `<h4 class="pkH">Where it looks — feature-activation heatmaps</h4>
      <p class="hint">Brightness = feature-vector magnitude at that spot. <b>p3</b> is the
      object-scale detection layer (stride 8) — it should light up ON the objects;
      <b>sppf</b> is the deepest context layer (stride 32). If p3 glows on glare or
      image text, that's the network being tempted by a distractor.</p>
      <div class="pkGrid">`;
    for (const [stem, frames] of Object.entries(hm.images))
      for (const f of frames)
        html += fig(`${pk.base}/live/${f.file}`, `${stem} · ${f.layer}`);
    html += `</div>`;
  }
  if (pk.labels){
    html += `<h4 class="pkH">Label geometry — where objects live across the train split</h4>
      <div class="pkGrid">${fig(`${pk.base}/labels.jpg`, "labels.jpg")}</div>`;
  }
  if (pk.val_pairs.length){
    html += `<h4 class="pkH">Validation: labels (left) vs the model's predictions (right)</h4>
      <div class="pkGrid">${pk.val_pairs.map(n =>
        fig(`${pk.base}/${n.replace("_pred", "_labels")}`, n.replace("_pred", "_labels"))
        + fig(`${pk.base}/${n}`, n)).join("")}</div>`;
  }
  $("#peekImgs").innerHTML = html;
  $("#peekImgs").querySelectorAll(".tlSlider").forEach(sl => {
    const frames = JSON.parse(sl.dataset.frames), base = sl.dataset.base;
    const block = sl.closest(".tlBlock");
    const show = () => {
      const f = frames[+sl.value];
      block.querySelector(".tlImg").src = base + f.file;
      const d = f.pred - f.gt;
      block.querySelector(".tlLabel").innerHTML =
        `${f.epoch == null ? "final (best.pt)" : "epoch " + f.epoch} — pred <b>${f.pred}</b> `
        + `<span class="${d === 0 ? "ok" : "short"}">(${d > 0 ? "+" : ""}${d})</span>`;
    };
    sl.oninput = show;
    show();                                     // start on the final frame
  });
  if (pkCarList.length){
    $("#pkCarPrev").onclick = () => { pkCarShow(-1); pkCarStart(); };
    $("#pkCarNext").onclick = () => { pkCarShow(1); pkCarStart(); };
    // Start on the newest live snapshot when streaming; else the "NOW" set
    // when mosaic has closed; else the beginning.
    pkCarIdx = (pk.stream || []).length ? pkCarList.length - 1
      : realPhaseNow && pk.mosaic_batches.length < pkCarList.length
        ? pk.mosaic_batches.length : 0;
    pkCarShow(0);
    pkCarStart();
  }
}

/* ---------- batch carousel (auto-advances every 4s; self-stops on close) ---- */
let pkCarTimer = null, pkCarIdx = 0, pkCarList = [];

function pkCarStop(){ if (pkCarTimer){ clearInterval(pkCarTimer); pkCarTimer = null; } }

function pkCarShow(delta){
  if (!pkCarList.length) return;
  pkCarIdx = (pkCarIdx + (delta || 0) + pkCarList.length) % pkCarList.length;
  const it = pkCarList[pkCarIdx], img = $("#pkCarImg");
  if (!img) return;
  img.src = it.src;
  $("#pkCarLink").href = it.src;
  $("#pkCarCap").innerHTML = `<b>${pkCarIdx + 1}/${pkCarList.length}</b> · ${it.tag} · ${it.name}`;
}

function pkCarStart(){
  pkCarStop();
  pkCarTimer = setInterval(() => {
    if ($("#runModal").hidden || !$("#pkCarImg")){ pkCarStop(); return; }
    pkCarShow(1);
  }, 4000);
}
