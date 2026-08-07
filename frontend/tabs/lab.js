"use strict";
/* Where THIS project's challenger previews live. The browser used to spell
 * `/outputs/alt/...` itself, which is only correct for the first project — every other
 * project's runs are under its own tree (ProjectPaths.ALT_DIR). The server says
 * which, in /api/lab/runs, so there is one answer and the browser is not a
 * second place that decides where data lives.
 * null means "outside the /outputs mount, not servable" — then say so, rather
 * than falling back to a path that would show ANOTHER project's images. */
let LAB_MEDIA = null;
/* Lab tab: RF-DETR sidecar — results next to (never mixed into) the yolo
   ledger, plus training controls for big-GPU machines (the worker). Reuses
   $, api, svgLineChart and the run modal from cockpit.js/history.js. */

let labPoll = null;

/* Which metric the cross-model chart is showing. Module state, not a URL
 * parameter: it is a way of LOOKING at one tab, not a place you navigate to or
 * would want to share. Persisted for the session so an 8s repaint does not throw
 * you back to MAE mid-read — that repaint is the whole reason it cannot just
 * live in the DOM. */
let LAB_METRIC = "mae";

/* Delegated once, because the picker is rebuilt from innerHTML on every repaint
 * and a handler bound to the button would not survive it. */
if (!window.__labMetricWired){
  window.__labMetricWired = true;
  document.addEventListener("click", e => {
    const b = e.target && e.target.closest && e.target.closest(".metricBtn");
    if (!b) return;
    LAB_METRIC = b.dataset.metric;
    if (window.loadLab) loadLab();
  });
}

/* What a marker says when tapped or hovered. The x-axis is an attempt number,
 * which is the one fact about a point you cannot act on — the run NAME is what
 * you paste into the table, a re-score or a cancel. Misses are given as a count
 * out of the holdout as well as the metric, because "5/68 wrong" is a sentence.
 *
 * Angle brackets stripped rather than escaped: <title> is markup, run names match
 * a [A-Za-z0-9._-] server-side regex so they cannot contain any, and apps/static/
 * has no escaping helper to reach for. */
/* WHICH FAMILY a run name belongs to, from the prefixes the server sends with
 * /api/train/options. registry.for_run is the authority; this is the same
 * answer, longest-prefix-first for the same reason — "deimv2-n" is a prefix of
 * nothing else, but stripping tv28's suffix also yields "deimv2-n", so the
 * longer key has to win or every deimv2n13 run is labelled as the wrong family.
 * Falls back to the run's own first token rather than a made-up word. */
/* A row per challenger that is TRAINING, greyed, for the shared runs table.
 *
 * yolo has had one since 2026-08-02 (identity.js renderGhostRow) — a faded line
 * where the adopted run WILL appear. Challengers had nothing: DEIM trains for
 * 25 minutes, scores, and only then appears, which reads as "did my run start?"
 * and there is nowhere else in this table a live challenger shows up.
 *
 * IT MUST NOT READ AS A RUN THAT EXISTS, the same lesson the yolo ghost carries.
 * The name is real here — a challenger is named at launch, unlike a retrain
 * whose number is assigned at adoption — so the ghosting is carried by the word
 * "training", the "in progress" cell and the greyed row rather than by inventing
 * a label.
 *
 * mae:null and when:now put it first under either sort with no special case in
 * the comparator.
 */
function _labGhostRows(st, ledgerRows) {
  const known = new Set((ledgerRows || []).map(r => r.run));
  /* ACTIVES *AND* FINISHING. A challenger that has stopped training is not in
   * this machine's ledger yet either — it is being scored on the worker, or
   * waiting for the adopt job to carry it home — so leaving `finishing` out meant
   * the run vanished from the table for the five to ten minutes between the
   * last epoch and the row appearing. Same gap the Overview had. */
  const live = [
    ...((st && st.actives) || []).map(a => ({
      run: a.run, card: a.card, state: "training…",
      how: (a.epoch != null && a.total) ? `epoch ${a.epoch}/${a.total}` : "training",
    })),
    ...((st && st.finishing) || []).map(fin => ({
      run: fin.run, card: null,
      // The STATE column follows the phase. It was hardcoded "training…", which
      // sat next to "training done, scoring…" in the same row and contradicted
      // it outright.
      state: fin.phase === "adopting" ? "adopting…"
           : fin.phase === "queued"   ? "queued"
                                      : "scoring…",
      how: fin.phase === "adopting" ? "verdict in, waiting to be adopted"
         : fin.phase === "queued"
             ? "trained — waiting for a free card to score on"
             : "training done, scoring on the holdout",
    })),
  ];
  return live
    .filter(a => a && a.run && !known.has(a.run))
    .map(a => {
      const arch = _labArchOf(a.run, ledgerRows);
      const how = a.how;
      return {mae: null, when: Date.now() / 1000, model: arch,
        html: `<tr class="runGhost" data-run="${a.run}">
          ${runsModelCell(arch, "")}
          <td class="wrap"><b>${a.run}</b><br>
            <span class="mMeta">${how}${a.card != null ? ` on card ${a.card}` : ""}</span></td>
          <td class="wrap"><span class="mMeta">${a.state}</span></td>
          <td></td>
          <td></td>
          <td class="acts"><span class="mMeta">in progress</span></td>
        </tr>`};
    });
}

function _labArchOf(run, ledger) {
  /* TWO SOURCES, AND THE LEDGER IS THE ONE THAT ALWAYS WORKS HERE.
   *
   * TRAIN_OPTS carries the registry's prefixes, which is the authoritative
   * answer — but loadTrain() only runs on the TRAIN tab, and this renders on
   * RESULTS, so it is null exactly where it is needed. Measured: TRAIN_OPTS was
   * false on Results while a run was live.
   *
   * The ledger is already in hand and already knows: every past run carries its
   * own `arch`, so the longest run-name prefix shared with a known run gives the
   * family without another request.
   *
   * BARE NAME, NOT window.TRAIN_OPTS — train.js declares it as a top-level `let`
   * in a classic script, so it lives in the shared global LEXICAL scope and
   * window.TRAIN_OPTS is undefined; a `window.` guard is always falsy. typeof,
   * because a `let` that has not executed yet is in the temporal dead zone and a
   * bare read would throw. */
  const models = (typeof TRAIN_OPTS !== "undefined" && TRAIN_OPTS
                  && TRAIN_OPTS.models) || [];
  let best = null, bestLen = -1;
  for (const m of models)
    for (const p of (m.prefixes || []))
      if (run.startsWith(p) && p.length > bestLen) { best = m.key; bestLen = p.length; }
  if (best) return best;
  // Longest shared prefix against runs whose family is recorded.
  for (const r of (ledger || [])) {
    if (!r || !r.run || !r.arch) continue;
    let n = 0;
    while (n < r.run.length && n < run.length && r.run[n] === run[n]) n++;
    if (n > bestLen) { bestLen = n; best = r.arch; }
  }
  return (bestLen >= 4 && best) ? best : (String(run).split("-")[0] || "challenger");
}

function labRunTip(r, attempt, metric){
  const clean = v => String(v ?? "").replace(/[<>&]/g, "");
  const val = metric.fmt(metric.get(r)) + (metric.unit === "%" ? "%" : "");
  const bits = [`${clean(r.run)} — ${metric.label} ${val}`];
  if (metric.key !== "mae" && r.mae != null) bits.push(`MAE ${r.mae}`);
  if (r.n_misses != null && r.n_images) bits.push(`${r.n_misses}/${r.n_images} images wrong`);
  bits.push(`attempt #${attempt + 1}`);
  if (r.epochs) bits.push(`${r.epochs} epochs`);
  if (r.imgsz || r.resolution) bits.push(`@${r.imgsz || r.resolution}`);
  return bits.join(" \u00b7 ");
}

async function loadLab(){
  // The CHAMPION'S runs come along for the chart. "How do the challengers
  // compare" is a question about the champion, and it was the one model missing
  // from the comparison — you had its MAE as a number in the subtitle and no
  // line to read it against. /api/runs carries the same derived metrics now
  // (web/run_metrics.py serves both endpoints), so it plots identically.
  // Failure here must not take the tab down: challengers still chart alone.
  const [data, st, champ] = await Promise.all([
    api("/api/lab/runs"), api("/api/lab/status"),
    api("/api/runs").catch(() => []),
  ]);
  // This used to hide the challenger controls on HQ and put a link to the worker's
  // SEPARATE cockpit in their place. Both are gone: the worker is a value in the
  // machine dropdown now, so there is nothing to hide and nowhere to send you.
  renderLabStatus(st);
  const box = $("#labRuns");
  LAB_MEDIA = data.media || null;
  const p = data.pinned;
  $("#labPinned").innerHTML = p
    ? `best so far: <b>MAE ${fmtScore(p.mae)}</b> <span class="mMeta">(conf ${p.conf} / iou ${p.iou}, pinned)</span>`
    : "no pinned eval yet";
  /* THE GHOST IS BUILT BEFORE THE EMPTY-LEDGER RETURN, because that is exactly
   * when it matters most: a project's FIRST challenger run has no ledger rows
   * to sit above, and "did my run start?" is the whole question at that moment.
   * Computing it after the return meant the first run of any project showed
   * nothing at all. */
  const _ghosts = _labGhostRows(st, data.runs);
  // No challengers is not "no runs" — the champion's rows live in the same
  // shared table now, so contribute nothing rather than writing over them.
  if (!data.runs.length){
    box.innerHTML = "";
    runsContribute("lab", _ghosts, () => {});
    return;
  }
  /* The cross-model chart lives in cockpit/runs/model_metrics.js — which metric is
   * shown, where "not in contention" sits for it, and how the series are built.
   * This tab only decides WHEN to draw one. */
  let trend = "";
  /* EVERY run in the project, champion included. The ledger spells two fields
   * differently — `model` where the challengers say `arch`, and `finished` where
   * they say `created` — so they are normalised here rather than teaching the
   * chart two vocabularies. Without `created` every yolo run sorts as 0 and the
   * whole family collapses onto attempt #1. */
  const champRuns = (Array.isArray(champ) ? champ : (champ.runs || []))
    .filter(r => r && r.mae != null)
    .map(r => ({...r, arch: r.model || r.arch || "yolov8n",
                created: r.created || r.finished || 0}));
  const scored = data.runs.filter(r => r.mae != null).concat(champRuns);
  if (scored.length >= 2 && !window.IS_LAB && window.svgMultiChart
      && window.modelMetricSeries){
    const metric = modelMetric(LAB_METRIC);
    const {series, maxN, all} = modelMetricSeries(scored, metric, labRunTip);
    if (series.length){
      trend = modelMetricPicker(metric.key)
        + svgMultiChart({series,
            xlabels: Array.from({length: maxN}, (_, i) => `#${i + 1}`),
            // No longer "by model" alone: the champion is on it now, so the
            // title has to say the scope is the whole project or it reads as a
            // challengers-only chart with a stray yolo line.
            title: `${metric.label} per attempt — every model in this project`,
            yfmt: v => metric.fmt(v), zeroFloor: metric.zeroFloor,
            markPerfect: true})
        + modelMetricNote(all, series, metric);
    }
  }

  const rows = data.runs.map(r => {
    // The verdict of the whole licensing programme must NOT be colour alone —
    // the owner is colour-blind, and `class="ok"` is only `color:var(--ok)`.
    // Carry it in words and in the signed delta against the champion.
    const beat = p && r.mae != null && r.mae <= p.mae;
    const d = (p && r.mae != null) ? +(r.mae - p.mae).toFixed(4) : null;
    const verdict = r.mae == null ? ""
      : beat ? ` <span class="mMeta">✓ matches/beats the best</span>`
      : ` <span class="mMeta">▲ ${d > 0 ? "+" : ""}${d} vs best</span>`;
    const mae = r.mae != null
      ? `<b class="${beat ? "ok" : ""}">${fmtScore(r.mae)}</b> <span class="mMeta">(${r.n_misses}/${r.n_images} miss)</span>${verdict}`
      : "<span class='mMeta'>not scored</span>";
    const model = r.arch ?? "rfdetr";
    // conf/iou were their own column; in the shared table they join the details
    // line, because the champion's rows have no equivalent and an empty column
    // on 27 of 81 rows is worse than a suffix on 54.
    const tune = r.conf != null || r.iou != null
      ? ` · conf ${r.conf ?? "?"}/iou ${r.iou ?? "?"}` : "";
    return {mae: r.mae, when: r.created, model, html: `<tr data-run="${r.run}">
      ${runsModelCell(model, r.license)}
      <td class="wrap"><b>${r.run}</b>${(r.arch || "").startsWith("rtmdet") || (r.arch || r.run).startsWith("rfdet") ? ` <span class="lic" title="architecture no longer actively developed here">deprecated</span>` : ""}<br><span class="mMeta">${r.resolution ?? "?"}px${r.num_queries ? ` · ${r.num_queries}q` : ""} · ${r.epochs ?? "?"}ep${tune}</span></td>
      <td class="wrap">${mae}</td>
      <td></td>
      <td></td>
      <td class="acts"><button class="ldet" data-run="${r.run}">detail</button>
          <button class="lsc" data-run="${r.run}">score</button></td>
    </tr>`};
  });
  /* A CHALLENGER THAT IS TRAINING GETS A ROW TOO, greyed, at the top.
   *
   * yolo has had one since 2026-08-02 (identity.js renderGhostRow) — a faded
   * line where the adopted run WILL appear, so a training run is visible
   * instead of the table simply not mentioning it. Challengers had nothing:
   * DEIM would train for 25 minutes, score, and only then appear, which reads
   * as "did my run start?" — and there is no other place in this table that a
   * live challenger shows up.
   *
   * SAME LESSON AS THE YOLO ONE, which this deliberately copies: it must not
   * read as a run that EXISTS. The name is real here (a challenger is named at
   * launch, unlike a retrain whose number is assigned at adoption), so the
   * ghosting is carried by the word "training" and the greyed row rather than
   * by inventing a label.
   *
   * `mae: null, when: Date.now()/1000` puts it first under either sort without
   * needing a special case in the comparator.
   */
  box.innerHTML = trend;                       // the chart stays; the table moved
  runsContribute("lab", _ghosts.concat(rows), box => {
  box.querySelectorAll("button.ldet").forEach(b => { b.onclick = () => labDetail(b.dataset.run); });
  box.querySelectorAll("button.lsc").forEach(b => {
    b.onclick = async () => {
      const r = await api("/api/lab/score", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({run_name: b.dataset.run})});
      alert(r.ok ? "Scoring started — refresh in a few minutes." : "🔒 " + (r.error || "refused"));
    };
  });
  });
}

/* ONE PANEL PER LIVE CHALLENGER.
 *
 * This rendered `st.active`, a single run — so with two challengers training the
 * second had no progress bar and no curve here at all. Fourth place the
 * one-challenger assumption was baked in (dashboard.js, train.js and run_log.js
 * were the others); the server sends `actives` now.
 *
 * Extracted whole rather than looped in place: the body builds one run's html
 * and returns it, so the multi-run case is a map+join and cannot half-render. */
function _labRunHtml(a){
    const pct = a.epoch && a.total ? Math.min(100, Math.round(100*a.epoch/a.total)) : null;
    const bits = [];
    if (a.epoch != null) bits.push(`epoch ${a.epoch}${a.total ? "/" + a.total : ""}`);
    if (a.map50 != null) bits.push(`val mAP50 ${(+a.map50).toFixed(3)}`);
    // D-FINE reports a val LOSS instead — never call it a score
    if (a.val_loss != null) bits.push(`val loss ${(+a.val_loss).toFixed(3)} ↓`);
    if (a.loss != null) bits.push(`loss ${a.loss}`);
    if (a.eta_s) bits.push(`~${Math.max(1, Math.round(a.eta_s/60))} min left`);
    let html = `⚙️ <b>${a.run}</b>${a.remote ? ` <span class="mMeta">on 🧪 Trainer</span>` : ""}`
      + ` is training — ${bits.join(" · ")
          || "this family prints no progress line; watching its files"
          + (a.elapsed_s ? ` (${Math.round(a.elapsed_s / 60)} min in)` : "")}`
      + (pct != null
          ? `<div class="pbar"><div class="pbarTrack"><div class="pbarFill" style="width:${pct}%"></div></div>`
            + `<span class="mMeta">${pct}% · auto-refreshing…</span></div>`
          : ` <span class="mMeta">auto-refreshing…</span>`);
    // Live learning curve — every val pass so far (the "is it still improving?"
    // question, answered while it trains).
    if (a.curve && a.curve.length > 4 && window.svgLineChart){
      // two shapes: mAP families (up = good) and D-FINE's loss (down = good)
      const isLoss = a.curve[0].val_loss != null;
      const last = a.curve[a.curve.length - 1];
      html += svgLineChart({
        points: a.curve.map(c => ({label: String(c.epoch),
                                   y: isLoss ? c.val_loss : c.map50})),
        color: isLoss ? "#e0555f" : "#c77b1e", markers: false, legend: "",
        title: isLoss
          ? `val loss per epoch — LOWER is better (latest ${(+last.val_loss).toFixed(3)})`
          : `val score per pass (latest ${(+last.map50).toFixed(3)})`,
        yfmt: v => v.toFixed(isLoss ? 2 : 2)});
    }
    return `<div class="labRunBlock">${html}</div>`;
}

function renderLabStatus(st){
  const el = $("#labStatus");
  const running = st.actives || (st.active ? [st.active] : []);
  if (running.length){
    el.innerHTML = running.map(_labRunHtml).join("");
    // The "what it sees" panel used to live HERE, inside the challenger view —
    // which meant a yolo retrain never had one, and with a challenger running
    // beside it there were two picture panels for two runs in different places.
    // cockpit/runs/run_vis.js now renders one block per LIVE run next to its log, the
    // same unification run_log.js did for the two terminals.
    if (!labPoll) labPoll = setInterval(() => {
      // Either half — #labStatus is on Train, #labRuns on Results.
      if (window.onRunsTabs && onRunsTabs()) loadLab();
      else { clearInterval(labPoll); labPoll = null; }
    }, 8000);
  } else if (!window.IS_LAB){
    el.innerHTML = "No other model is training right now — start one above; live progress and results appear here.";
    if (labPoll){ clearInterval(labPoll); labPoll = null; }
  } else if (!st.train_unlocked){
    // One message now, not two: "no challenger venv here" and "card too small"
    // were always the same fact — this is the laptop, not the lab. The VRAM
    // probe that used to draw the distinction is gone (it wedged nvidia-smi in
    // D-state under WSL2; see api_lab.py).
    el.innerHTML = `<span class="short">🔒 no environment for other models on this machine — `
      + `pick a model and a card above (setup: altmodels/README.md). `
      + `Scoring checkpoints here still works.</span>`;
    if (labPoll){ clearInterval(labPoll); labPoll = null; }
  } else {
    el.innerHTML = st.yolo_retrain
      ? "⏳ a yolo retrain holds the GPU — alt training will refuse to start until it finishes"
      : "GPU idle — clear to train.";
    if (labPoll){ clearInterval(labPoll); labPoll = null; }
  }
  // The Train button moved to cockpit/runs/train.js, which decides what is startable
  // from /api/train/options. `train_unlocked` is still reported by the API and
  // still true — it just has no button here to disable any more.
}

async function labDetail(run){
  const d = await api(`/api/lab/runs/${run}/detail`);
  let html = `<h3 class="modalTitle">${run}</h3>`;
  if (d.curve && d.curve.length)
    html += svgLineChart({points: d.curve.map(c => ({label: String(c.epoch), y: c.map})),
      color: "#b07de0", markers: false, legend: "",
      // D-FINE logs a LOSS here (down = good); every other family logs mAP.
      // api_lab flags which, so the axis never implies the wrong direction.
      title: `${d.curve_metric === "val loss" ? "val LOSS per epoch (lower is better)"
                : "val EMA mAP50-95 per epoch"} (final ${d.curve[d.curve.length-1].map.toFixed(3)})`,
      yfmt: v => v.toFixed(2)});
  const ce = d.count_eval;
  if (ce){
    const misses = ce.images.filter(t => t.pred !== t.gt)
      .sort((a, b) => Math.abs(b.pred - b.gt) - Math.abs(a.pred - a.gt));
    html += `<p class="hint"><b>MAE ${fmtScore(ce.mae)}</b> at conf ${ce.conf} / iou ${ce.iou} — ${misses.length} missed image(s):</p>`;
    html += misses.map(t => {
      const e = t.pred - t.gt;
      const url = LAB_MEDIA ? `${LAB_MEDIA}/${run}/eval_preview/${t.stem}.png` : null;
      const stem = url
        ? `<a href="${url}" target="_blank" rel="noopener">${t.stem}</a>` : t.stem;
      return `<div class="hint">${stem}
        <span class="${e > 0 ? "over" : "short"}">${e > 0 ? "+" : ""}${e}</span> (${t.pred}/${t.gt}) · ${t.bucket}</div>`;
    }).join("") || "";
  } else {
    html += "<p class='hint'>Not scored yet — press <b>score</b> once training finishes.</p>";
  }
  // The dots, image by image — wrong images FIRST (worst error at the top),
  // each captioned pred/target; correct ones trail with a ✓.
  if (d.previews && d.previews.length && !LAB_MEDIA){
    html += `<p class="hint">${d.previews.length} eval preview(s) exist on disk but `
      + `this project's run tree is outside the served <code>outputs/</code> `
      + `directory, so they cannot be shown here.</p>`;
  } else if (d.previews && d.previews.length){
    const have = new Set(d.previews);
    const items = ce
      ? ce.images.slice().sort((a, b) => Math.abs(b.pred - b.gt) - Math.abs(a.pred - a.gt))
      : d.previews.map(p => ({stem: p.replace(".png", "")}));
    html += `<h4 class="pkH">Eval previews — wrong images first (${d.previews.length} images)</h4>`
      + `<div class="pkGrid">` + items.filter(t => have.has(t.stem + ".png")).map(t => {
        const e = t.pred != null ? t.pred - t.gt : null;
        const bad = e !== null && e !== 0;
        const cap = e === null ? t.stem
          : `${t.stem} — ${t.pred}/${t.gt} ` + (bad
              ? `<b class="${e > 0 ? "over" : "short"}">${e > 0 ? "+" : ""}${e}</b>` : "✓");
        const url = `${LAB_MEDIA}/${run}/eval_preview/${t.stem}.png`;
        return `<figure class="pkFig${bad ? " missFig" : ""}"><a href="${url}" target="_blank" rel="noopener">`
          + `<img loading="lazy" src="${url}"></a><figcaption>${cap}</figcaption></figure>`;
      }).join("") + `</div>`;
  }
  $("#runModalBody").innerHTML = html;
  showRunModal(run);
}

/* The challenger Train controls lived here — arch, res, epochs, batch, and a
 * button that pinned the 3090. All of that is one control now (cockpit/runs/train.js
 * -> /api/train), which also knows WHICH CARDS each model can actually use:
 * DEIM's torch build has no sm_120 kernels, so the 5070 Ti was never an option
 * and this panel could not have told you. What stays here is the challenger
 * ledger, the live status card and the detail modal. */

window.loadLab = loadLab;
/* BOTH buttons, and OPTIONALLY — this was `.addEventListener` on a bare
 * querySelector, so a missing button would be a load-time TypeError that killed
 * the rest of this file silently. */
["runs", "results"].forEach(t =>
  document.querySelector(`nav button[data-tab="${t}"]`)
    ?.addEventListener("click", loadLab));

/* ---- serving engine (champion vs challenger) ----
   Lives on the Challenger tab, not the Dashboard: the Dashboard is pure
   status. Switching writes serving_engine.json and leaves the yolo pin
   untouched, so "go back to yolo" is one click and never a re-pin. */
async function loadEngine(){
  const el = $("#engineCtrls");
  if (!el) return;
  // The engine is PER-PROJECT (core.js appends ?project= to every call when
  // one is open). With no project open there is nothing to ask about — and
  // asking anyway 422s on a fresh instance, seen on the wizard's first load.
  if (!PROJECT) return;
  let d;
  try { d = await api("/api/engine"); } catch (e){ return; }
  el.innerHTML = (d.options || []).map(o => {
    const on = o.engine === d.engine;
    const mae = o.mae != null ? ` · holdout ${fmtScore(o.mae)}` : "";
    return `<button class="engineBtn${on ? " active" : ""}" data-engine="${o.engine}"`
      + ` title="${o.license}${mae}"${on ? " disabled" : ""}>`
      + `${on ? "● serving: " : ""}${o.label}`
      + `<span class="lic">${o.license}</span></button>`;
  }).join("") + `<span class="mMeta" id="engineMsg"></span>`;
  el.querySelectorAll("button[data-engine]").forEach(b => {
    b.onclick = async () => {
      b.disabled = true;
      const r = await api("/api/engine", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({engine: b.dataset.engine})});
      await loadEngine();
      const msg = $("#engineMsg");
      if (msg) msg.textContent = r && r.ok
        ? "switched — /restart the bot to load it"
        : "⚠ " + ((r && r.error) || "switch failed");
    };
  });
}
// The engine switch is on Results now; Train keeps the hook so the panel is
// primed either way.
["runs", "results"].forEach(t =>
  document.querySelector(`nav button[data-tab="${t}"]`)
    ?.addEventListener("click", loadEngine));
loadEngine();
