"use strict";
/* WHICH NUMBER the cross-model chart is showing, and the chart that shows it.
 *
 * MAE was the only thing plotted, and on a saturated holdout it is the least
 * legible of the numbers we already have: 0.0154 has to be decoded, and it is a
 * MEAN, so it hides both the worst single image and the direction of the error.
 * Those two turned out to be findings. Every DEIM miss ever measured is +1 and
 * never -1 — it invents objects rather than losing them, which points at duplicate
 * detections on one object rather than missed objects — and that is invisible in
 * any absolute-error number.
 *
 * ONE TABLE, not four charts. Each metric declares how to read a run, how to
 * format it, and where "not in contention" sits for THAT number, because the
 * threshold is not transferable: 0.2 objects/image and 20% of images wrong are
 * different statements. Adding a metric is one entry here and nothing else.
 *
 * Deliberately its own file: lab.js is a tab, this is a chart vocabulary, and
 * the tab was already carrying the whole thing inline.
 */

/* cut:    one-sided. Above this a run is not plotted, because at that error the
 *         model is not competing for anything.
 * absCut: symmetric, for SIGNED metrics — |v| above this is not plotted. A
 *         one-sided cut is useless on a signed number: it would drop every
 *         over-count and keep every under-count, which is not a threshold, it is
 *         a bias of its own.
 * zeroFloor: false lets the axis go negative. Only `bias` needs it, and the
 *         default matters — every other metric here is an absolute error, where
 *         space below zero would be meaningless.
 *
 * fmt MUST tolerate a fractional argument even when the data is integral: the
 * y-axis interpolates four ticks between lo and hi, so an integer metric still
 * asks it to render 3.7833333. `worst image` printed exactly that. */
const MODEL_METRICS = [
  {key: "mae", label: "MAE", unit: "objects/image", cut: 0.2, zeroFloor: true,
   get: r => r.mae,
   fmt: v => (+v).toFixed(3),
   blurb: "mean absolute count error per image — the ranking metric"},

  {key: "missrate", label: "images wrong", unit: "%", cut: 20, zeroFloor: true,
   // A RATE, not a count. The holdout has been 61, 63, 65 and now 68 images, so
   // "5 wrong" means different things across runs and cannot be compared; the
   // share can. This is also the most readable number here — "1.5% of images"
   // needs no decoding, which 0.0154 does.
   get: r => (r.n_images ? 100 * r.n_misses / r.n_images : null),
   fmt: v => (+v).toFixed(1),
   blurb: "share of holdout images counted wrong, at any margin"},

  {key: "worst", label: "worst image", unit: "objects", cut: 10, zeroFloor: true,
   // What MAE hides. A run can average beautifully and still be forty out on one
   // image, and for a technician counting a script that is the number that
   // decides whether the tool is usable.
   get: r => r.worst_image_error,
   // Rounded, because the axis asks for values BETWEEN the integers it plots.
   fmt: v => String(Math.round(v)),
   blurb: "largest error on any single image — the risk a mean hides"},

  {key: "bias", label: "net over/under", unit: "objects", absCut: 25, zeroFloor: false,
   // A SIGNED axis, on purpose: this is a diagnostic, not a ranking. A model at
   // +12 and one at -12 have the same MAE and opposite problems — duplicate
   // detections versus missed objects.
   //
   // Symmetric cut at 25 because it started with none, and centernet's -1657
   // (it lost most of the objects on every dense image, the topk=100 artifact) set
   // an axis running to -1657, on which +1 and -1 are the same pixel — and the
   // whole finding here is that every DEIM miss is +1 and never -1.
   get: r => r.bias,
   fmt: v => (v > 0 ? `+${Math.round(v)}` : String(Math.round(v))),
   blurb: "positive = counts objects that are not there, negative = misses them"},
];

function modelMetric(key){
  return MODEL_METRICS.find(m => m.key === key) || MODEL_METRICS[0];
}

/* ONE definition of "is this run on the chart", used by the series builder AND
 * the note. They have to agree: if the filter and the caption disagree, the
 * caption is claiming things about runs that are plotted. */
function metricInRange(metric, v){
  if (metric.absCut != null) return Math.abs(v) <= metric.absCut;
  if (metric.cut != null) return v <= metric.cut;
  return true;
}

/* The picker. Same shape as the runs table's sort buttons so it reads as the
 * same kind of control, and it carries the metric's blurb as a title rather than
 * a separate legend nobody has room for on a phone. */
function modelMetricPicker(active){
  return `<div class="metricPick">` + MODEL_METRICS.map(m =>
    `<button type="button" class="metricBtn${m.key === active ? " on" : ""}"
       data-metric="${m.key}" title="${m.blurb}">${m.label}</button>`).join("") +
    `</div>`;
}

/* Series for one metric, one line per architecture.
 *
 * ATTEMPT NUMBERS ARE THE FAMILY'S REAL ONES — counted over all its runs and
 * filtered afterwards, never re-indexed over the survivors. Numbering the
 * filtered list made a family whose early attempts are all above the cut show
 * its eighteenth try as "#1" under a chart titled "per attempt". Counting first
 * also makes the gap legible: a family that took eighteen tries to get under the
 * cut starts eighteen columns in.
 */
function modelMetricSeries(runs, metric, tipFor){
  const all = {};
  for (const r of runs){
    const v = metric.get(r);
    if (v == null || !isFinite(v)) continue;
    (all[r.arch || "challenger"] ||= []).push({run: r, v});
  }
  const fams = {}, spans = [];
  for (const [arch, list] of Object.entries(all)){
    const ordered = list.sort((a, b) => (a.run.created || 0) - (b.run.created || 0));
    spans.push(ordered.length);
    const kept = ordered
      .map((o, i) => ({...o, attempt: i}))
      .filter(o => metricInRange(metric, o.v));
    if (kept.length) fams[arch] = kept;
  }
  const series = Object.entries(fams)
    .sort((a, b) => Math.min(...a[1].map(o => o.v)) - Math.min(...b[1].map(o => o.v)))
    .map(([arch, kept]) => ({
      label: arch,
      color: (window.modelColor ? modelColor(arch) : "#7a8699"),
      pts: kept.map(o => ({i: o.attempt, y: o.v,
                           tip: tipFor(o.run, o.attempt, metric)})),
    }));
  return {series, maxN: Math.max(1, ...spans), all};
}

/* What fell off, SUMMARISED BY FAMILY. Listing runs individually read fine at
 * seven and is a wall of text at twenty-five — which is a silent omission
 * wearing a caption. Two different cases, said differently: a family whose BEST
 * is above the cut has no line at all, while a plotted family's earlier attempts
 * being above it is just history the chart is zoomed past. */
function modelMetricNote(all, series, metric){
  if (metric.cut == null && metric.absCut == null) return "";
  const signed = metric.absCut != null;
  const shown = new Set(series.map(s => s.label));
  // "Best" means CLOSEST TO THE CHART for a signed metric, not smallest — on
  // net over/under, -400 is not better than +3, it is worse in the other
  // direction, and reporting a family's minimum would name -1657 as its best.
  const nearest = l => l.reduce((b, o) =>
    (signed ? Math.abs(o.v) < Math.abs(b) : o.v < b) ? o.v : b, l[0].v);
  const suffix = metric.unit === "%" ? "%" : "";
  const absent = Object.entries(all)
    .map(([a, l]) => [a, nearest(l)])
    .filter(([a]) => !shown.has(a))
    .sort((x, y) => (signed ? Math.abs(x[1]) - Math.abs(y[1]) : x[1] - y[1]))
    .map(([a, v]) => ({name: a, best: `${metric.fmt(v)}${suffix}`,
                       // The colour it WOULD have had, from the one palette, so
                       // a model keeps its identity whether or not it is plotted.
                       color: (window.modelColor ? modelColor(a) : "#7a8699")}));
  const earlier = Object.entries(all).filter(([a]) => shown.has(a))
    .reduce((n, [, l]) => n + l.filter(o => !metricInRange(metric, o.v)).length, 0);
  if (!absent.length && !earlier) return "";
  const band = signed ? `±${metric.absCut}` : `0–${metric.cut}`;
  /* A LABELLED LIST, not a sentence with names buried in it. This was one long
   * paragraph and the model names — the only part anyone scans for — were the
   * hardest thing in it to find. Each name is drawn in the colour it WOULD have
   * had on the chart, so a model you were looking for is findable by the same
   * cue the legend uses.
   *
   * Colour is reinforcement, never the message: the name is right there in text,
   * and the score follows it. That matters here specifically — the owner is
   * colour-blind, so anything carried by hue alone is carried by nothing. */
  let msg = `<span class="noteBand">Chart covers ${band} ${metric.unit}.</span> `;
  if (absent.length)
    msg += `<span class="noteLead">Models not shown:</span> `
         + absent.map(a =>
             `<span class="noteModel" style="color:${a.color}">${a.name}</span>`
             + `<span class="noteScore"> ${a.best}</span>`).join(`<span class="noteSep">·</span>`)
         + `<span class="noteWhy"> — nothing they have run lands within `
         + `${band} ${metric.unit}.</span> `;
  if (earlier)
    msg += `<span class="noteWhy">${earlier} earlier attempt`
         + `${earlier > 1 ? "s" : ""} by the models above also sit outside it.</span> `;
  return `<p class="hint chartNote">${msg}`
       + `<span class="noteWhy">Every run is still in the table below.</span></p>`;
}
