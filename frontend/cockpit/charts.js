"use strict";
/* Chart vocabulary: the two SVG builders, their interaction, and the palettes.
 *
 * WHY IT IS NOT IN tabs/history.js ANY MORE. It was, and that made a TAB the home
 * of shared infrastructure for four other modules — cockpit/dashboard.js,
 * cockpit/runs/model_metrics.js, tabs/lab.js and tabs/peek.js all draw with these, and
 * history was simply the file they happened to have been written in.
 *
 * It was also an active hazard rather than untidiness. svgLineChart and
 * svgMultiChart sit adjacent with near-identical bodies — the same
 * `const pad = (hi - lo) * 0.15` line appears in both — and on 2026-08-02 a
 * one-shot string replace aimed at the multi-series version hit the single-series
 * one instead, and the repair then deleted that function's X and Y scale helpers.
 * Two bugs out of one file's shape. They still live together because they share a
 * vocabulary, but edit them by the code that FOLLOWS the shared line.
 *
 * Classic script: these are function DECLARATIONS, so they become window
 * properties. The palettes are `const` and are NOT — modelColor() is the
 * accessor, which is why callers guard with `window.modelColor ? ... : fallback`.
 */
// Small single-series SVG line chart (dark-surface, validated palette colors).
// points: [{label, y}] in chronological order. One axis, no legend (title names it).
function svgLineChart({points, color, title, yfmt, markers = true,
                       ghost = null, legend = "● 960&nbsp; ◆ 1280"}){
  if (!points.length)
    return `<div class="chartCard"><div class="chartTitle">${title}</div>
            <div class="chartEmpty">awaiting first holdout run</div></div>`;
  const W = 340, H = 158, m = {l: 40, r: 14, t: 12, b: 24};
  const iw = W - m.l - m.r, ih = H - m.t - m.b, n = points.length;
  // Ghost = a faded dashed comparison series (e.g. the previous run's curve),
  // indexed by position; the x-domain and y-scale cover BOTH series.
  const N = Math.max(n, ghost ? ghost.length : 0);
  const ys = points.map(p => p.y).concat(ghost || []);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (lo === hi){ lo -= 0.1; hi += 0.1; }
  const pad = (hi - lo) * 0.15; lo = Math.max(0, lo - pad); hi += pad;  // metrics are >=0: axis never dips negative
  const X = i => m.l + (N === 1 ? iw / 2 : iw * i / (N - 1));
  const Y = v => m.t + ih * (1 - (v - lo) / (hi - lo));
  let grid = "", ticks = "";
  for (let k = 0; k <= 3; k++){
    const v = lo + (hi - lo) * k / 3, y = Y(v);
    grid += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" class="grid"/>`;
    ticks += `<text x="${m.l - 6}" y="${y + 3}" class="ytick">${yfmt(v)}</text>`;
  }
  const path = points.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
  let dots = "", xlab = "";
  points.forEach((p, i) => {
    const cx = +X(i).toFixed(1), cy = +Y(p.y).toFixed(1);
    if (markers){
      const tip = `<title>${p.label}: ${yfmt(p.y)} · imgsz ${p.imgsz ?? "?"}</title>`;
      // shape encodes resolution: ◆ diamond = 1280, ○ circle = 960 (not colour-only)
      dots += (p.imgsz >= 1280)
        ? `<polygon points="${cx},${cy-5} ${cx+5},${cy} ${cx},${cy+5} ${cx-5},${cy}" fill="${color}" class="dot">${tip}</polygon>`
        : `<circle cx="${cx}" cy="${cy}" r="4" fill="${color}" class="dot">${tip}</circle>`;
    }
    if (i === 0 || i === n - 1)
      xlab += `<text x="${cx}" y="${H - 6}" class="xtick" text-anchor="${i === 0 ? "start" : "end"}">${p.label}</text>`;
  });
  const last = points[n - 1];
  const vlabel = `<text x="${X(n - 1).toFixed(1)}" y="${(Y(last.y) - 9).toFixed(1)}" class="vlabel" text-anchor="end" fill="${color}">${yfmt(last.y)}</text>`;
  const gpath = ghost && ghost.length
    ? `<path d="${ghost.map((y, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(y).toFixed(1)}`).join("")}"
        fill="none" stroke="#5b6372" stroke-width="1.5" stroke-dasharray="4 3"/>` : "";
  return `<div class="chartCard"><div class="chartTitle">${title}
      <span class="chartLegend">${legend}</span></div>
    <svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${title}">
      ${grid}${ticks}${gpath}
      <path d="${path}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      ${dots}${vlabel}${xlab}
    </svg></div>`;
}

// Multi-series variant of svgLineChart for the per-bucket trend: each series
// plots only the runs where that bucket existed (x = shared run index).
function svgMultiChart({series, xlabels, title, yfmt, zeroFloor = true,
                        markPerfect = false}){
  const W = 340, H = 158, m = {l: 40, r: 14, t: 12, b: 24};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const N = xlabels.length;
  const ys = series.flatMap(s => s.pts.map(p => p.y));
  if (!ys.length)
    return `<div class="chartCard"><div class="chartTitle">${title}</div>
            <div class="chartEmpty">no tagged runs yet</div></div>`;
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (lo === hi){ lo -= 0.1; hi += 0.1; }
  // zeroFloor: absolute-error metrics are >=0, so the axis never dips negative
  // and the space below zero would be meaningless. A SIGNED metric (net
  // over/under count) needs the opposite — its sign is the whole point — so the
  // floor is opt-out rather than a law. NOTE the identical line exists in
  // svgLineChart above; edit them by the code that FOLLOWS, not the line itself.
  const pad = (hi - lo) * 0.15;
  lo = zeroFloor ? Math.max(0, lo - pad) : lo - pad; hi += pad;
  const X = i => m.l + (N === 1 ? iw / 2 : iw * i / (N - 1));
  const Y = v => m.t + ih * (1 - (v - lo) / (hi - lo));
  // A zero line, drawn only when the data crosses it — on a signed chart the
  // difference between +1 and -1 IS the finding, and without it both are just
  // "near the bottom".
  const zline = (lo < 0 && hi > 0)
    ? `<line x1="${m.l}" y1="${Y(0).toFixed(1)}" x2="${W - m.r}" y2="${Y(0).toFixed(1)}"
        stroke="var(--edge,#39404e)" stroke-width="1"/>` : "";
  let grid = "", ticks = "";
  for (let k = 0; k <= 3; k++){
    const v = lo + (hi - lo) * k / 3, y = Y(v);
    grid += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" class="grid"/>`;
    ticks += `<text x="${m.l - 6}" y="${y + 3}" class="ytick">${yfmt(v)}</text>`;
  }
  // Shape and dash pattern carry the series identity, not colour. The palette
  // holds near-duplicate shades (two cyans #3fb8c4/#7fd4dc, two ambers
  // #c77b1e/#d4a63f) and up to 8 lines share one axis, so hue alone could not
  // separate them — and the legend was a coloured dot beside a name, which is
  // the same information again. Each series now has a distinct marker glyph
  // repeated in the legend, plus its own dash, so it reads without colour.
  const MARKS = ["circle", "square", "diamond", "triangle",
                 "circle", "square", "diamond", "triangle"];
  const DASH  = ["", "6 3", "2 3", "8 3 2 3", "1 4", "10 4", "4 2 1 2", "3 3"];
  const GLYPH = {circle: "\u25CF", square: "\u25A0", diamond: "\u25C6", triangle: "\u25B2"};
  /* A HAIRLINE IN THE CARD COLOUR around every marker. Consecutive runs that
   * score the SAME value sit at the same y one column apart, and with a
   * connecting line of the same colour between them the markers merge into one
   * long blob — most visible on deimv2-n, which is square and has six runs tied
   * at 0.0154, so its best result rendered as a rectangle rather than six wins.
   * Stroking each shape in the background colour cuts them apart without adding
   * any new colour to read. */
  const marker = (kind, cx, cy, col) => {
    const r = 3.6;
    const edge = ` stroke="var(--card,#1a1f2b)" stroke-width="0.9"`;
    if (kind === "square")   return `<rect x="${(cx-r).toFixed(1)}" y="${(cy-r).toFixed(1)}" width="${2*r}" height="${2*r}" fill="${col}"${edge}>`;
    if (kind === "diamond")  return `<polygon points="${cx.toFixed(1)},${(cy-r-1).toFixed(1)} ${(cx+r+1).toFixed(1)},${cy.toFixed(1)} ${cx.toFixed(1)},${(cy+r+1).toFixed(1)} ${(cx-r-1).toFixed(1)},${cy.toFixed(1)}" fill="${col}"${edge}>`;
    if (kind === "triangle") return `<polygon points="${cx.toFixed(1)},${(cy-r-1).toFixed(1)} ${(cx+r+1).toFixed(1)},${(cy+r).toFixed(1)} ${(cx-r-1).toFixed(1)},${(cy+r).toFixed(1)}" fill="${col}"${edge}>`;
    return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r}" fill="${col}"${edge}>`;
  };
  const closeTag = k => (k === "circle" ? "</circle>" : k === "square" ? "</rect>" : "</polygon>");
  // Every selectable point, in viewBox coordinates: [x, y, tip, colour].
  // Collected while drawing so the geometry is the SAME arithmetic that placed
  // the markers — recomputing it for the hit test is how a cursor ends up
  // half a pixel off the thing it claims to be on.
  const hits = [];
  const body = series.map((s, si) => {
    const kind = MARKS[si % MARKS.length], dash = DASH[si % DASH.length];
    const path = s.pts.map((p, k) => `${k ? "L" : "M"}${X(p.i).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    // p.tip lets a caller say WHICH run a marker is, not just where it sits on
    // the x-axis. "deimv2-n · #3: 0.0154" tells you nothing you can act on; the
    // run name is what you paste into the table or a re-score. Optional, so the
    // per-bucket and dashboard callers keep the positional form.
    // TWO SHAPES PER POINT: the visible marker (r=3.6) and an invisible circle
    // big enough for a finger. A fingertip is ~44px and these markers are ~7px
    // across, so on a phone the chart was decorative — you could see a point and
    // never touch it. `fill="transparent"` is rgba(0,0,0,0), which still receives
    // pointer events; `fill="none"` would not.
    //
    // <title> is kept for desktop hover, but it does NOT fire on touch, so the
    // hit circle also carries data-tip, which the delegated handler below reads
    // into a readout line under the chart. Same gesture works with a mouse.
    const dots = s.pts.map(p => {
      const tip = p.tip || `${s.label} · ${xlabels[p.i]}: ${yfmt(p.y)}`;
      const cx = X(p.i), cy = Y(p.y);
      // A PERFECT SCORE IS THE POINT OF THE WHOLE EXERCISE, so it should not
      // look like every other marker. Ringed rather than recoloured: the series
      // colour is its identity and a second colour would read as another model,
      // and the shape stays the one in the legend.
      //
      // OPT-IN, because "0 is remarkable" is a property of the METRIC, not of
      // the chart. On the cross-model comparison a zero is the goal and rare. On
      // the per-bucket lens most points are exactly 0 — a tagged bucket usually
      // has no misses — so ringing them ringed nearly every marker on the chart,
      // and emphasis that fires on most of the data is not emphasis.
      // ONE thin ring in --ok, not two thick ones in the series colour. Drawn in
      // the series colour it read as part of the line and had to be big to be
      // seen at all; a distinct hue means it can be small and still register.
      // Green is the semantic choice — a perfect count is the good outcome — and
      // it is reinforcement only: the ring SHAPE is the signal, and the legend
      // spells it out, because the owner is colour-blind and anything carried by
      // hue alone is carried by nothing.
      const perfect = (markPerfect && p.y === 0)
        ? `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="6" fill="none"
             stroke="var(--ok,#2fbf71)" stroke-width="1.1" opacity="0.85"/>`
        : "";
      // NO PER-POINT HIT TARGET. Each point used to get its own r=13 circle,
      // which works until points are closer together than the targets are wide —
      // twenty attempts across a 340-unit axis are ~15 units apart against a
      // 26-unit target, so neighbours overlapped and whichever was painted last
      // silently won. Selection is nearest-point against ONE surface now (see
      // the rect below), which cannot be ambiguous: every position on the chart
      // has exactly one closest point.
      if (p.tip) hits.push([+cx.toFixed(1), +cy.toFixed(1), tip, s.color]);
      return perfect
        + marker(kind, cx, cy, s.color) + `<title>${tip}</title>` + closeTag(kind);
    }).join("");
    return `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2"
      ${dash ? `stroke-dasharray="${dash}"` : ""}
      stroke-linecap="round" stroke-linejoin="round"/>${dots}`;
  }).join("");
  // The ring is explained in the legend, not left as a mystery — an emphasis
  // nobody can decode is just noise. Only shown when a perfect run is actually
  // on the chart, so it does not promise a state the data never reaches.
  const anyPerfect = markPerfect && series.some(s => s.pts.some(p => p.y === 0));
  const legend = series.map((s, si) =>
    `<span style="color:${s.color}">${GLYPH[MARKS[si % MARKS.length]]} ${s.label}</span>`).join(" ")
    + (anyPerfect ? ` <span class="legendPerfect">◎ perfect</span>` : "");
  const xlab = `<text x="${X(0)}" y="${H - 6}" class="xtick" text-anchor="start">${xlabels[0]}</text>`
    + (N > 1 ? `<text x="${X(N - 1)}" y="${H - 6}" class="xtick" text-anchor="end">${xlabels[N - 1]}</text>` : "");
  // The readout only appears once something is picked — an empty bar under every
  // chart is furniture, and the hint text has to be the first thing it says or a
  // phone user has no way to learn the interaction exists.
  const hasTips = series.some(s => s.pts.some(p => p.tip));
  const readout = hasTips
    ? `<div class="chartReadout" data-empty="tap a point for run details">`
      + `tap a point for run details</div>`
    : "";
  /* ONE surface over the plot area, carrying every point. The handler finds the
   * nearest one, so there is no such thing as missing a target — a mouse gets a
   * live readout as it moves, and a thumb gets the point closest to where it
   * landed rather than whichever overlapping circle happened to be on top.
   *
   * touch-action:pan-y keeps vertical scrolling working: the chart is halfway
   * down a long page and trapping the finger to read a tooltip would be a worse
   * bug than the one this fixes. Horizontal movement scrubs.
   *
   * &quot; because the JSON goes in a double-quoted HTML attribute; the browser
   * decodes it on getAttribute. Tips are already stripped of angle brackets by
   * the caller. */
  const surface = hits.length
    ? `<rect class="chartSurface" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}"
         fill="transparent" tabindex="0" role="application"
         aria-label="${title} — arrow keys to step through runs"
         data-pts="${JSON.stringify(hits).replace(/"/g, "&quot;")}"/>
       <circle class="chartCursor" r="9" fill="none" stroke-width="1.8"
         opacity="0" cx="0" cy="0"/>`
    : "";
  return `<div class="chartCard"><div class="chartTitle">${title}
      <span class="chartLegend">${legend}</span></div>
    <svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${title}">
      ${grid}${ticks}${zline}${body}${xlab}${surface}
    </svg>${readout}</div>`;
}

/* ONE delegated listener for every chart on the page, registered once.
 *
 * Delegation rather than per-marker handlers because the charts are rebuilt from
 * innerHTML on an 8s heartbeat — handlers bound to nodes would be dropped on
 * every repaint, and rebinding after each one is the kind of thing that works
 * until someone adds a third chart. document-level survives all of it.
 *
 * pointerup, not click: on iOS a click on an SVG child can be swallowed, and
 * pointerup fires for mouse, touch and pen alike. focus is handled too so the
 * markers are reachable by keyboard, which is what tabindex/role promise. */
if (!window.__chartTapWired){
  window.__chartTapWired = true;

  const ptsOf = el => {
    try { return JSON.parse(el.getAttribute("data-pts") || "[]"); }
    catch (e){ return []; }
  };

  /* Client pixels -> viewBox units via the SVG's own matrix. The obvious
   * shortcut — width ratio times offset — is wrong the moment the element's
   * aspect differs from the viewBox's, which it does here: .chart has a
   * min-height on phones, so preserveAspectRatio letterboxes the drawing and a
   * linear map lands off by the letterbox. getScreenCTM knows about all of it. */
  const toView = (svg, cx, cy) => {
    const m = svg.getScreenCTM();
    if (!m) return null;
    const p = svg.createSVGPoint(); p.x = cx; p.y = cy;
    return p.matrixTransform(m.inverse());
  };

  const pick = (surface, clientX, clientY) => {
    const svg = surface.ownerSVGElement;
    const loc = svg && toView(svg, clientX, clientY);
    if (!loc) return;
    const pts = ptsOf(surface);
    let best = null, bestD = Infinity;
    for (const p of pts){
      const d = (p[0] - loc.x) ** 2 + (p[1] - loc.y) ** 2;
      if (d < bestD){ bestD = d; best = p; }
    }
    // NEAREST WITH NO RADIUS LIMIT, on purpose. The surface only covers the plot
    // area, so anywhere inside it there IS a closest run and answering is more
    // useful than going blank because the finger was 30 units above the line.
    if (!best) return;
    const card = surface.closest(".chartCard");
    const out = card && card.querySelector(".chartReadout");
    const cur = svg.querySelector(".chartCursor");
    if (cur){
      // The marker under the pointer grows into a halo in its own series colour,
      // so you can see WHICH point you are about to read before you read it —
      // which matters most on a phone, where the finger covers the marker.
      cur.setAttribute("cx", best[0]);
      cur.setAttribute("cy", best[1]);
      cur.setAttribute("stroke", best[3]);
      cur.setAttribute("opacity", "1");
    }
    if (out){
      // textContent: the tip carries a run name and apps/static/ has no
      // escaping helper.
      out.textContent = best[2];
      out.classList.add("chartReadoutOn");
    }
  };

  const from = e => {
    const s = e.target && e.target.closest && e.target.closest(".chartSurface");
    if (s) pick(s, e.clientX, e.clientY);
    return s;
  };

  // A mouse tracks live as it moves. A finger only when it is down — hovering
  // is not a thing on touch, and reacting to a stray pointermove during a scroll
  // would rewrite the readout while you are trying to read it.
  document.addEventListener("pointermove", e => {
    if (e.pointerType === "mouse" || e.buttons) from(e);
  }, true);
  document.addEventListener("pointerdown", from, true);

  /* Keyboard: the surface is focusable, and arrows step run by run. The previous
   * version gave every marker a tabindex, which meant sixty tab stops on one
   * chart; this is one stop and then arrows, which is both fewer and the
   * convention for a plot. */
  document.addEventListener("keydown", e => {
    const s = e.target && e.target.classList
      && e.target.classList.contains("chartSurface") ? e.target : null;
    if (!s || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const pts = ptsOf(s);
    if (!pts.length) return;
    // Step in DRAWN order along x, so "right" means later attempt regardless of
    // which series the previous point belonged to.
    const order = pts.map((p, i) => [p[0], i]).sort((a, b) => a[0] - b[0]).map(o => o[1]);
    const cur = s.ownerSVGElement.querySelector(".chartCursor");
    const at = cur ? pts.findIndex(p => String(p[0]) === cur.getAttribute("cx")
                                     && String(p[1]) === cur.getAttribute("cy")) : -1;
    const k = order.indexOf(at);
    const next = pts[order[(k < 0 ? 0 : k + (e.key === "ArrowRight" ? 1 : -1)
                            + order.length) % order.length]];
    if (!next) return;
    e.preventDefault();
    const svg = s.ownerSVGElement, card = s.closest(".chartCard");
    const out = card && card.querySelector(".chartReadout");
    if (cur){
      cur.setAttribute("cx", next[0]); cur.setAttribute("cy", next[1]);
      cur.setAttribute("stroke", next[3]); cur.setAttribute("opacity", "1");
    }
    if (out){ out.textContent = next[2]; out.classList.add("chartReadoutOn"); }
  }, true);
}

const BUCKET_TREND_COLORS = {object: "#3987e5", capsule: "#b07de0", stacked: "#2fbfa0"};

/* ONE model palette, beside the charts that draw it. dashboard.js's families
 * race chart and lab.js's challenger trend both colour by architecture, and a
 * second copy would eventually give the same model two colours on two pages —
 * which is worse than no colour, because it reads as two models.
 *
 * Colour is DECORATION here, never the identity: svgMultiChart gives each series
 * its own marker glyph and dash pattern and repeats the glyph in the legend,
 * because this palette holds near-duplicate shades (two cyans, two ambers) and
 * up to eight lines share one axis. */
const MODEL_COLORS = {
  "yolov8n": "#3987e5", "deimv2-n": "#c77b1e", "deimv2-n-tv28": "#d4a63f",
  "rtmdet-tiny": "#b07de0", "rfdetr-nano": "#5a9e6f", "dfine-small": "#e0555f",
  "yolox-s": "#3fb8c4", "yolox-tiny": "#7fd4dc", "centernet": "#d4a63f",
};
function modelColor(arch){ return MODEL_COLORS[arch] || "#7a8699"; }

