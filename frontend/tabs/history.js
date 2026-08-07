"use strict";
/* Training-runs history tab: the champion's own ledger, its charts and the
 * per-run modal. Classic script: reuses $, api and loadState from
 * cockpit/core.js, and the chart builders from cockpit/charts.js.
 *
 * The SVG chart primitives used to live here and now do not — see charts.js for
 * why a tab was the wrong home for four other modules' infrastructure. */

function drawRunCharts(runs){
  const chrono = [...runs].reverse();                       // oldest -> newest
  const short = r => r.run.replace("yolov8n-", "r");
  const maePts = chrono.filter(r => r.mae != null).map(r => ({label: short(r), y: r.mae, imgsz: r.imgsz}));
  const mapPts = chrono.filter(r => r.map50 != null).map(r => ({label: short(r), y: r.map50, imgsz: r.imgsz}));
  // Per-bucket MAE across runs — the type lens over time ("is capsule STAYING
  // solved?"). Only runs whose eval carried tags participate.
  const bruns = chrono.filter(r => r.buckets &&
    Object.keys(r.buckets).some(k => k !== "untagged" && r.buckets[k] && r.buckets[k].images));
  const series = Object.keys(BUCKET_TREND_COLORS).map(name => ({
    label: name, color: BUCKET_TREND_COLORS[name],
    pts: bruns.map((r, i) => r.buckets[name] && r.buckets[name].images
      ? {i, y: r.buckets[name].mae} : null).filter(Boolean),
  })).filter(s => s.pts.length);
  // SAY WHOSE RUNS THESE ARE. drawRunCharts is fed the yolo ledger only, so all
  // three describe the champion — but they sit on the same page as a chart that
  // covers every model in the project, and nothing distinguished them. A reader
  // has no way to know one axis is the whole field and the next three are one
  // family. Said once above the group rather than three times in three titles.
  /* HOW THE DATA GREW, on the same x as the scores above it.
   *
   * WHY IT EARNS A CHART. When the holdout GROWS rather than forking, past MAEs
   * stop being strictly comparable — and the fix chosen in PIVOT was to make the
   * incomparability VISIBLE rather than pretend it away. It matters right now:
   * the exam went 65 -> 68 at yolov8n-57, which is the run that scored 0.0. A
   * bare 0.0 cannot say "on a bigger exam"; this line can.
   *
   * TWO SERIES. `images` is the labelled training pool, which train and val
   * are split out of; the pool is the series with the full history.
   *
   * No tips: this answers "how big was the data", not "which run" — and without
   * tips svgMultiChart draws no hit surface, so it does not claim to be
   * interactive. */
  const sizePts = (key) => chrono
    .map((r, i) => r[key] != null ? {i, y: r[key]} : null).filter(Boolean);
  const sizeSeries = [
    {label: "training pool", color: "#3987e5", pts: sizePts("images")},
    {label: "holdout", color: "#c77b1e", pts: sizePts("test_images")},
  ].filter(s => s.pts.length);

  // No scope line here: the "YOLO stats — champion only" heading above this
  // group says it, and saying it twice was two lines of the prose the headings
  // replaced.
  $("#runsCharts").innerHTML =
    svgLineChart({points: maePts, color: "#3987e5",
      title: "Holdout MAE — objects/image (lower better)", yfmt: v => v.toFixed(2)}) +
    svgLineChart({points: mapPts, color: "#199e70",
      title: "mAP50 (higher better)", yfmt: v => v.toFixed(3)}) +
    (series.length ? svgMultiChart({series, xlabels: bruns.map(short),
      title: "Per-bucket MAE — the type lens", yfmt: v => v.toFixed(2)}) : "") +
    "";

  /* DATASET SIZE LIVES IN ITS OWN SECTION, not under "YOLO stats — champion
   * only" (owner, 2026-08-02). How big the training pool and the holdout are is
   * a fact about the PROJECT that every model is measured on; filing it under
   * one family implied the other families were examined on something else.
   *
   * The x axis is still the yolo run sequence, because the yolo ledger is the
   * record that carries image counts per run — that is a limitation of where the
   * numbers come from, not a claim that the data belongs to yolo. */
  const dataEl = $("#dataCharts");
  if (dataEl) dataEl.innerHTML = sizeSeries.length
    ? svgMultiChart({series: sizeSeries, xlabels: chrono.map(short),
        title: "Dataset size — images per run", yfmt: v => String(Math.round(v))})
    : "<p class='hint'>No image counts recorded yet.</p>";
}

const fmtTime = s => s == null ? "—" : s < 60 ? `${Math.round(s)}s`
  : `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`;

// Per-bucket MAE chips under a run's overall MAE. Buckets are an eval lens (tagged
// on the Test-set tab); a blended average can stay flat while one type regresses,
// so we surface each type's own MAE + image count. "untagged" shown last, dimmed.
function bucketChips(buckets){
  if (!buckets) return "";
  const names = Object.keys(buckets).filter(b => buckets[b] && buckets[b].images);
  if (!names.length || (names.length === 1 && names[0] === "untagged")) return "";
  names.sort((a, b) => (a === "untagged") - (b === "untagged") || a.localeCompare(b));
  const chips = names.map(b => {
    const x = buckets[b];
    return `<span class="bchip b-${b}">${b} <b>${x.mae.toFixed(2)}</b>`
      + `<span class="mMeta"> ·${x.images}</span></span>`;
  }).join("");
  return `<div class="bchips">${chips}</div>`;
}

// A run's MAE is only trustworthy (and eligible for "best") if it was scored on
// the frozen holdout AND that holdout was big enough — a low MAE on a tiny testset
// (early experiments) means little. Auto-hides the star until the holdout is real.
const MIN_TRUST_TRAYS = 20;

// The run whose curve/log modal is open — used to highlight its table row.
let modalRun = null;
function markExpanded(){
  document.querySelectorAll("#runsTable tr[data-run]").forEach(tr =>
    tr.classList.toggle("expanded", tr.dataset.run === modalRun));
}
function showRunModal(run){ $("#runModal").hidden = false; modalRun = run; markExpanded(); }
function closeRunModal(){ $("#runModal").hidden = true; modalRun = null; markExpanded(); }
// Wire the modal's close affordances once (the element is static in index.html).
$("#runModalClose").onclick = closeRunModal;
$("#runModal").addEventListener("click", e => { if (e.target === $("#runModal")) closeRunModal(); });
window.addEventListener("keydown", e => { if (e.key === "Escape" && !$("#runModal").hidden) closeRunModal(); });

/* A YOLO RETRAIN THAT IS IN FLIGHT, as a greyed row at the top of the table.
 *
 * The challengers got one first (tabs/lab.js _labGhostRows) and yolo did not,
 * which is backwards: the retrain is the run most people are waiting on. While
 * it trains, scores and is adopted — tens of minutes — the table simply did not
 * mention it, so "is my retrain running?" had to be answered on another tab.
 *
 * "WILL BE CALLED", NOT THE NAME. A retrain has no name until adoption assigns
 * the next number; `next_name` is a prediction. CLAUDE.md records the cost of
 * rendering it as a run that exists — 2026-08-02, the owner hunted through HQ,
 * the worker and the ledger for a yolov8n-60 that had never existed. The model
 * list's ghost row says "will be called" for the same reason and this matches
 * it deliberately.
 *
 * EITHER MACHINE. A retrain runs here or on the Trainer, so both are asked and
 * whichever is running wins; `remote` only changes the wording.
 */
function _yoloGhostRows(local, lab) {
  const pick = (local && local.status === "running") ? {s: local, where: ""}
             : (lab && lab.ok && lab.status === "running")
               ? {s: lab, where: " on 🧪 Trainer"} : null;
  if (!pick) return [];
  const s = pick.s;
  const how = (s.epoch != null && s.total)
    ? `epoch ${s.epoch}/${s.total}` : (s.step || "training");
  const name = s.next_name ? `will be called ${s.next_name}` : "a new run";
  return [{mae: null, when: Date.now() / 1000, model: "yolov8n",
    html: `<tr class="runGhost" data-run="${s.next_name || "yolo-pending"}">
      ${runsModelCell("yolov8n", "AGPL-3.0")}
      <td class="wrap"><b>⏳ ${name}</b><br>
        <span class="mMeta">${how}${pick.where}${
          s.imgsz ? ` · ${s.imgsz}px` : ""}</span></td>
      <td class="wrap"><span class="mMeta">training…</span></td>
      <td></td>
      <td></td>
      <td class="acts"><span class="mMeta">in progress</span></td>
    </tr>`}];
}

async function loadHistory(){
  /* The two retrain statuses come along so the table can show a run in flight.
   * Failure in either must not take the ledger down — a missing ghost is a
   * cosmetic loss, an empty table is not. */
  const [runs, models, rt, lab] = await Promise.all([
    api("/api/runs"), api("/api/models"),
    api("/api/retrain").catch(() => null),
    api("/api/lab_status").catch(() => null),
  ]);
  const _yghosts = _yoloGhostRows(rt, lab);
  drawRunCharts(runs);
  // No yolo runs is not "no runs": contribute nothing and let the shared table
  // still show the challengers. Writing "No runs recorded yet" into #runsTable
  // here would erase them.
  if (!runs.length){
    $("#runsSummary").innerHTML = "";
    // Still contribute the ghost: a project's FIRST retrain has no ledger rows
    // to sit above, and that is exactly when "did it start?" is loudest.
    runsContribute("yolo", _yghosts, () => {});
    return;
  }
  const recByRun = {}; for (const r of runs) recByRun[r.run] = r;   // for the curve drill-down
  // HOLDOUT-SATURATION HONESTY. When the last few evaluated runs all tie on
  // the same MAE, the test set can no longer rank models — every new run
  // "confirms" the tie, and reading the ties as progress is the most
  // expensive analytical mistake this platform knows about. Say it, once,
  // above the table.
  (() => {
    const host = document.querySelector("#runsSatNote");
    if (!host) return;
    const scored = runs.filter(r => r.mae != null).slice(0, 4);
    const sat = scored.length >= 3
      && scored.every(r => r.mae === scored[0].mae);
    host.hidden = !sat;
    if (sat) host.innerHTML = `⚠ <b>The holdout can no longer rank these ` +
      `models</b> — the last ${scored.length} evaluated runs all tie at ` +
      `MAE ${scored[0].mae}. Add harder cases to the test set (new ` +
      `conditions, denser scenes, worse light); until then, differences ` +
      `between runs are noise.`;
  })();

  const wmap = {}; for (const r of models.runs) wmap[r.name] = r;
  const activeW = models.active && models.active.weights;
  const newest = models.runs[0] && models.runs[0].name;

  // Best = lowest MAE among trustworthy runs only (real holdout, big enough).
  const trusted = runs.filter(r => r.mae != null && r.mae_source === "testset"
                                   && (r.test_images ?? 0) >= MIN_TRUST_TRAYS);
  let best = null;
  for (const r of trusted){ if (!best || r.mae < best.mae) best = r; }
  const latest = runs[0];
  const bestTxt = best
    ? `<b>Best MAE ${fmtScore(best.mae)}</b> — ${best.run} <span class="mMeta">(holdout ${best.test_images})</span>`
    : `<b>No trustworthy MAE yet</b> <span class="mMeta">— need a holdout ≥ ${MIN_TRUST_TRAYS} images (now ${latest.test_images ?? "?"})</span>`;
  $("#runsSummary").innerHTML =
    `<span>${runs.length} run${runs.length === 1 ? "" : "s"}</span>`
    + `<span>${bestTxt}</span>`
    + `<span class="mMeta">latest: ${latest.images ?? "?"} train / ${latest.test_images ?? "?"} test</span>`;

  // Rows are CONTRIBUTED to the shared table (cockpit/runs/runs_table.js), not
  // rendered here: the champion and the challengers belong in one ranking.
  // Each carries the keys that table sorts on, plus its finished HTML.
  const rows = [];
  for (let i = 0; i < runs.length; i++){
    const r = runs[i];
    // Δ vs the previous run that actually has a known MAE.
    let prev = null;
    for (let j = i + 1; j < runs.length; j++){ if (runs[j].mae != null){ prev = runs[j].mae; break; } }
    let d = "";
    if (r.mae != null && prev != null){
      const diff = +(r.mae - prev).toFixed(2);
      const cls = diff < 0 ? "ok" : diff > 0 ? "short" : "";
      d = `<span class="${cls}">${diff < 0 ? "↓" : diff > 0 ? "↑" : "="} ${diff > 0 ? "+" : ""}${diff}</span>`;
    }
    const serving = wmap[r.name] && (activeW ? wmap[r.name].weights === activeW : r.name === newest);
    const isBest = best && r.run === best.run;
    // Short date keeps the table narrow; the full timestamp lives in the title.
    const dt = new Date(r.finished * 1000);
    const date = `<span title="${dt.toLocaleString()}">${
      dt.toLocaleDateString([], {month: "short", day: "numeric"})} ${
      dt.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}</span>`;
    // MAE always carries its eval context (source · holdout size) so a low number
    // on a tiny testset reads as weak, not as a win.
    let mae = r.mae != null
      ? `${fmtScore(r.mae)} <span class="mMeta">(${r.mae_source ?? "?"}${r.test_images != null ? " · " + r.test_images : ""})</span>`
      : "<span class='mMeta'>—</span>";
    mae += bucketChips(r.buckets);      // per-type MAE breakdown, if this run has tags
    const data = `${r.images ?? "?"}${r.test_images != null ? ` / ${r.test_images}` : ""}`;
    const noteVal = (r.note || "").replace(/"/g, "&quot;");
    const rowCls = isBest ? "best" : (r.mae == null ? "muted" : "");
    // Everything that used to be 7 skinny columns collapses into one wrapped
    // details line — the table fits on screen with no sideways scrolling.
    const opHonesty = [];
    if (r.ties) opHonesty.push(`${r.ties} tied cells`);
    if (r.grid_edge && r.grid_edge.length) opHonesty.push("winner at grid edge");
    const details = [
      opHonesty.length
        ? `<span title="The served conf/iou is a tie-break, not a clear optimum — see the run's eval.">⚖ ${opHonesty.join(" · ")}</span>`
        : "",
      `${r.imgsz ?? "?"}px`,
      `${r.epochs_trained ?? "?"}${r.epochs_requested ? "/" + r.epochs_requested : ""}ep`,
      fmtTime(r.train_time_s),
      r.peak_vram_gb != null ? `${r.peak_vram_gb}GB` : null,
      data,
      r.map50 != null ? `mAP ${r.map50}` : null,
    ].filter(Boolean).join(" · ");
    // The SERVER says which model this run is of, and what it is licensed
    // under. This used to strip a trailing -N off the run name here, which
    // works only while every run is named <model>-<n> and silently mislabels
    // anything else — a re-scored checkpoint, a hand-named experiment, a second
    // family that does not follow the convention. The licence was hardcoded
    // AGPL-3.0 for the same reason, which would have been wrong for the first
    // non-ultralytics run to enter this ledger.
    const model = r.model || "yolov8n";
    rows.push({mae: r.mae, when: r.finished, model, html:
      `<tr data-run="${r.run}"${rowCls ? ` class="${rowCls}"` : ""}>
      ${runsModelCell(model, r.license || "AGPL-3.0")}
      <td class="wrap"><b>${r.run}</b>${isBest ? ' ⭐' : ""}${serving ? ' <span class="mActive">● serving</span>' : ""}${
        r.ckpt === "last" ? ' <span class="mMeta" title="the holdout scored the final epoch better than ultralytics\' val-fitness pick, so last.pt was promoted into best.pt">ckpt:last</span>' : ""}
        ${r.split_seed ? `<span class="mSeed" title="trained on train/val split seed ${r.split_seed}, NOT the seed-0 split every other run used — its MAE is not directly comparable to theirs">split ${r.split_seed}</span>` : ""}
        <br><span class="mMeta">${date} · ${details}</span></td>
      <td class="wrap">${mae}</td>
      <td>${d}</td>
      <td><input class="note" data-run="${r.run}" value="${noteVal}" placeholder="note…"></td>
      <td class="acts"><button class="cv" data-run="${r.run}">curve</button><button class="lg" data-run="${r.run}">log</button><button class="gal" data-run="${r.run}">dots</button><button class="pk" data-run="${r.run}" title="Inside this run: the augmented batches it trained on, curves, labels plot">🔬</button><button class="snap" data-run="${r.run}" title="Dataset snapshot: exactly what this run trained/tested on, what drifted since, restore missing images">📸</button>${
        r.mae != null ? `<button class="df" data-run="${r.run}" title="Per-image diff vs the previous evaluated run: which images got fixed, which broke">Δ</button>` : ""}${
        wmap[r.name] && !serving ? ` <button class="use" data-run="${r.run}">use</button>` : ""}</td>
    </tr>`});
  }
  // Wiring runs on EVERY render of the shared table (innerHTML discards
  // handlers), so it is a function of the container rather than a one-shot.
  runsContribute("yolo", _yghosts.concat(rows), box => {
  box.querySelectorAll("input.note").forEach(inp => {
    inp.onchange = () => fetch(`/api/runs/${inp.dataset.run}/note`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({note: inp.value})});
  });
  /* OPEN THE MODAL FIRST, THEN LOAD INTO IT.
   *
   * These three used to `await fetch(...)` and only call showRunModal once the
   * response had landed, so the click produced NOTHING until it did. Measured
   * against a stalled endpoint: no modal at 600ms, modal at 3077ms — three
   * seconds in which the only honest reading is that the button is broken. On
   * this network a 25s stall is documented as normal (see cockpit/ui.js), so
   * this is not a hypothetical.
   *
   * loadInto gives the whole three-state contract for free — skeleton, a
   * "taking longer than usual" notice at 5s, and a failure card with Retry.
   * The old code had no failure path at all: a rejected fetch left the handler
   * dead and the button did nothing, forever, with no way to try again.
   */
  box.querySelectorAll("button.cv").forEach(b => {
    b.onclick = () => {
      const run = b.dataset.run;
      showRunModal(run);
      loadInto("#runModalBody", () => curveHtml(run),
               {label: `the curve for ${run}`, kind: "chart", n: 1});
    };
  });
  async function curveHtml(run){
      const res = await fetch(`/api/runs/${run}/curve`);
      let html;
      if (!res.ok){ html = `<h3 class="modalTitle">${run}</h3><p class="hint">No epoch data for this run.</p>`; }
      else {
        const c = (await res.json()).curve.filter(e => e.map50 != null);
        const pts = c.map(e => ({label: String(e.epoch), y: e.map50}));
        const peak = c.reduce((a, e) => e.map50 > a.map50 ? e : a, c[0]);
        // Judged against THIS run's real cap (its epochs_requested) — not a magic
        // number. Runs are fixed-length now (early stopping disabled), so reaching
        // the cap is normal and falling short means cancelled/crashed; only OLD
        // runs could genuinely early-stop on a plateau.
        const cap = (recByRun[run] || {}).epochs_requested;
        const hitCap = cap != null && c.length >= cap;
        const wasted = hitCap ? c.length - peak.epoch : 0;
        const note = hitCap
          ? `trained the full ${cap} epochs (best mAP50 @ ${peak.epoch}`
            + (wasted > cap * 0.3
               ? `) — <b>${wasted} epochs past the best produced nothing</b>; a shorter budget or early stopping would have cost this run nothing.`
               : `).`)
          : `⚠ ended early at epoch ${c.length}${cap ? " of " + cap : ""} (best @ ${peak.epoch}) — early-stop on old runs; on fixed-length runs this means cancelled or crashed.`;
        html = svgLineChart({points: pts, color: "#3987e5", markers: false,
          title: `${run} — mAP50 per epoch (best ${peak.map50.toFixed(3)} @ epoch ${peak.epoch} of ${c.length})`,
          yfmt: v => v.toFixed(2)})
          + `<p class="hint">${note}</p>`;
      }
      return html;
  }

  box.querySelectorAll("button.lg").forEach(b => {
    b.onclick = () => {
      const run = b.dataset.run;
      showRunModal(run);
      // "text", not "chart": a log is lines of text and the hole should be that
      // shape, or the skeleton reserves a space nothing fills.
      loadInto("#runModalBody", () => logInto(run),
               {label: `the log for ${run}`, kind: "text", n: 8});
    };
  });
  /* Returns NOTHING and writes into the container itself — the second of
   * loadInto's two contracts. The log is inserted as textContent on a <pre>
   * rather than as an HTML string, deliberately: a training log is arbitrary
   * process output and assigning it to innerHTML would execute whatever it
   * happens to contain. */
  async function logInto(run){
      const el = $("#runModalBody");
      const res = await fetch(`/api/runs/${run}/log`);
      const body = res.ok ? (await res.json()).log
        : `(no saved log for ${run} — it predates per-run logging)`;
      el.innerHTML = `<h3 class="modalTitle">${run} — training log</h3>`;
      const pre = document.createElement("pre");     // textContent = no HTML injection
      pre.className = "log"; pre.textContent = body;
      el.appendChild(pre);
  }
  box.querySelectorAll("button.gal").forEach(b => {
    b.onclick = () => {
      const run = b.dataset.run;
      showRunModal(run);
      // "card": this fills with a grid of image thumbnails, so the hole is
      // card-shaped and the real gallery lands in the space already reserved.
      loadInto("#runModalBody", () => dotsHtml(run),
               {label: `the dots for ${run}`, kind: "card", n: 6});
    };
  });
  async function dotsHtml(run){
      const res = await fetch(`/api/runs/${run}/images`);
      if (!res.ok){
        // A run with no previews is an ANSWER, not a failure — it returns the
        // explanation as content rather than throwing, so loadInto renders it
        // as the result instead of as a retryable error.
        return `<h3 class="modalTitle">${run} — holdout dots</h3>` +
          `<p class="hint">No eval previews for this run — it was scored before the ` +
          `gallery existed. Retrain (or re-run count_eval) to generate them.</p>`;
      }
      const info = await res.json();
      // Stacked dots can be MASKED by the count (a +1 double-count cancelling a -1
      // miss reads as exact), so surface them separately, not only via worst-first.
      const dupeTrays = info.images.filter(t => t.dupes).sort((a, x) => x.dupes - a.dupes);
      const dupeTotal = info.images.reduce((s, t) => s + (t.dupes || 0), 0);
      const images = info.images.slice()
        .sort((a, x) => Math.abs(x.pred - x.gt) - Math.abs(a.pred - a.gt));   // worst first
      const cells = images.map(t => {
        const d = t.pred - t.gt, cls = d === 0 ? "ok" : d < 0 ? "short" : "over";
        const drop = t.dropped ? ` · <span class="short">${t.dropped} off-image</span>` : "";
        const dup = t.dupes ? ` · <span class="dupeFlag">${t.dupes} stacked</span>` : "";
        const url = `${info.preview_base || ""}/${t.stem}.png`;
        return `<figure class="galCell">` +
          `<a href="${url}" target="_blank" rel="noopener"><img loading="lazy" src="${url}"></a>` +
          `<figcaption><span class="delta ${cls}">${d > 0 ? "+" : ""}${d}</span> ` +
          `pred <b>${t.pred}</b> / gt ${t.gt}${drop}${dup}` +
          `<br><span class="mMeta">${t.stem} · ${t.bucket}</span></figcaption></figure>`;
      }).join("");
      // Red rings only exist for runs scored while the on-image filter was ON
      // (disabled 2026-07-30, see groundwork/serving_filters.py). Explaining
      // them unconditionally sent a reader hunting for rings that can no longer
      // be drawn — so key the sentence off whether this run actually has any.
      const hadDrops = images.some(t => (t.dropped || 0) > 0);
      const legend = `Dots = objects it counted. ` +
        (hadDrops
          ? `<span class="short">Red rings</span> = detected but dropped by the on-image filter ` +
            `(a miss with a red ring = filter ate a real object; no dot = detection miss). `
          : `A miss with no dot = detection miss; the on-image filter is off, so nothing was discarded. `) +
        `<span class="dupeFlag">Magenta rings</span> = dots stacked on another (likely double-counts). ` +
        `Click any image to open full-size.`;
      const dupeSummary = dupeTotal
        ? `<p class="dupeSummary">⚠ Stacked dots: <b>${dupeTotal}</b> across ${dupeTrays.length} image(s) — `
          + dupeTrays.map(t => `${t.stem} (${t.dupes})`).join(", ") + `</p>`
        : `<p class="dupeSummary ok">✅ No stacked dots in this run.</p>`;
      return `<h3 class="modalTitle">${run} — holdout dots (conf ${info.conf} / iou ${info.iou}) · worst first</h3>` +
        `<p class="hint">${legend}</p>${dupeSummary}<div class="galGrid">${cells}</div>`;
  }
  box.querySelectorAll("button.pk").forEach(b => {
    b.onclick = () => { if (window.openRunPeek) openRunPeek(b.dataset.run); };
  });
  box.querySelectorAll("button.snap").forEach(b => {
    b.onclick = () => { if (window.openRunSnapshot) openRunSnapshot(b.dataset.run); };
  });
  box.querySelectorAll("button.df").forEach(b => {
    b.onclick = () => { if (window.openRunDiff) openRunDiff(b.dataset.run); };
  });
  markExpanded();                                // re-apply row highlight after a re-render
  box.querySelectorAll("button.use").forEach(b => {
    b.onclick = async () => {
      const m = wmap[b.dataset.run];
      await fetch("/api/model/activate", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({weights: m.weights, imgsz: m.imgsz})});
      alert(`Set ${b.dataset.run} (imgsz ${m.imgsz}) to serve.\nSend /restart to the bot to load it.`);
      loadHistory(); loadState();
    };
  });
  });
}
window.loadHistory = loadHistory;
