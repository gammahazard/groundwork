"use strict";
/* Shared UI primitives: loading, failure, and motion. ONE implementation.
 *
 * Every panel in this cockpit is fed by an API call, and until now each one
 * either rendered nothing while it waited or wrote its own "⏳ loading…" string.
 * An empty panel is indistinguishable from "nothing here", which is how a dead
 * endpoint looks exactly like an empty dataset.
 *
 * THREE STATES, not two — and the middle one exists because of this system
 * specifically. The network path to the Trainer goes cold and a 25s stall is
 * normal-ish (a VPN path can take seconds to warm before
 * it calls the path dead). So "slow" must not look like "broken":
 *
 *     0 - 5s     skeleton. It is working, nothing to say.
 *     5s+        skeleton + "taking longer than usual" in amber. Still trying.
 *     threw      the failure card, with the reason and a Retry.
 *
 * Amber rather than red for the slow state because --acc red is already the
 * destructive/primary-action colour here; an unreachable lab is a warning, not
 * an error, and it usually resolves itself.
 *
 * MOTION IS OPT-OUT. Everything animated goes through `reveal()`, which honours
 * prefers-reduced-motion — the setting exists for people who get motion sick,
 * and a dashboard that ignores it is not "sleek", it is rude.
 */

/* Respect the OS setting, and re-check it live rather than caching at load —
 * it can be toggled while the page is open. */
const uiReduceMotion = () =>
  matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Fade + rise an element into view. No-op under reduced motion. */
function reveal(el, delay = 0) {
  if (!el) return;
  if (uiReduceMotion()) { el.style.opacity = ""; return; }
  el.classList.remove("uiReveal");
  void el.offsetWidth;                 // restart the animation if it re-renders
  el.style.animationDelay = delay ? `${delay}ms` : "";
  el.classList.add("uiReveal");
}

/** Stagger a reveal across children — used for card grids. Capped, because 400
 *  images x 30ms is a twelve-second wave, which is not "subtle". */
function revealAll(container, step = 40, cap = 12) {
  if (!container) return;
  [...container.children].forEach((c, i) => reveal(c, Math.min(i, cap) * step));
}

const UI_SKELETONS = {
  card:  n => `<div class="uiSkel uiSkelCard"></div>`.repeat(n ?? 6),
  row:   n => `<div class="uiSkel uiSkelRow"></div>`.repeat(n ?? 5),
  tile:  n => `<div class="uiSkel uiSkelTile"></div>`.repeat(n ?? 4),
  text:  n => `<div class="uiSkel uiSkelText"></div>`.repeat(n ?? 3),
  /* A CHART-SHAPED HOLE. Sized by aspect-ratio rather than a fixed height,
   * because .charts is a flex row: two charts share it at ~200px tall, one
   * stretches full-width and is ~500px. A px min-height would be right for
   * exactly one of those and wrong for the other, and both occur on this page.
   * See .uiSkelChart in 09-ui.css. */
  chart: n => `<div class="uiSkel uiSkelChart"></div>`.repeat(n ?? 2),
  /* One line of text, for the one-line summary strips. */
  line:  n => `<div class="uiSkel uiSkelLine"></div>`.repeat(n ?? 1),
};

/**
 * Reserve a container's space with a skeleton, BEFORE its data is asked for.
 *
 * WHY THIS EXISTS SEPARATELY FROM loadInto. loadInto owns one container fed by
 * one call, which is most panels. Models & Runs is not that shape: `loadHistory`
 * makes two fetches and writes three containers from them, `loadLab` makes three
 * and writes three more, and both then contribute rows to a fourth shared table.
 * Wrapping either in loadInto would mean splitting one fetch into three, or
 * lying about which container a failure belongs to.
 *
 * So this is the half of loadInto that those panels can actually use: put the
 * hole in place now, let the existing loader fill it whenever it lands.
 *
 * MEASURED, 2026-08-02, at 1280x900 on the Runs tab — every container went from
 * ZERO height to its full size in one jump as its fetch returned:
 *
 *     #labRuns    0 -> 591      #dataCharts  0 -> 494      #runsCharts 0 -> 201
 *     #runVis     0 -> 206      #retrainLog  0 -> 502      #tCards    20 -> 109
 *
 * That is ~2400px of content appearing under the reader's eyes, and everything
 * below each one is shoved down as it arrives. Note the browser's own CLS metric
 * reports 0.0000 for this and is USELESS here: layout-shift entries are not
 * recorded for content inside a tab that was display:none, so the only honest
 * measurement is the heights themselves.
 *
 * ONLY IF EMPTY. Re-entering the tab, or an 8s heartbeat repaint, must not flash
 * a skeleton over content that is already on screen — that is a worse flicker
 * than the one being fixed. Same guard runsTableLoading() already used.
 */
function uiHold(sel, kind = "chart", n = null) {
  const el = typeof sel === "string" ? $(sel) : sel;
  if (!el || el.children.length || (el.textContent || "").trim()) return;
  /* NO .uiLoading CLASS HERE, unlike loadInto — and that is the point of the
   * separation, not an omission. .uiLoading forces `display:grid`, which is
   * right for a container loadInto owns outright and WRONG for these: #runsCharts
   * and #dataCharts are `.charts`, a flex row, and the class would override it.
   * The loaders here write innerHTML and never remove a class they did not add,
   * so the grid would then survive into the real charts and relayout them.
   *
   * These containers already have the layout their content needs. The skeleton's
   * job is to occupy it, not to impose one — so it is styled to sit correctly in
   * whatever it lands in, and the .uiSkel children are their own marker.
   *
   * `kind` MAY BE A SHAPE, not just one block repeated. Several of these panels
   * are a strip plus a chart plus a caption — #labRuns is a metric picker, a
   * chart and a "what fell off the chart" note — and holding only the chart left
   * the other two to appear around it, 125px of movement the hold was supposed
   * to prevent. Pass [["line",1],["chart",1],["text",2]] to build the hole in
   * that order. */
  el.innerHTML = Array.isArray(kind)
    ? kind.map(part => {
        const [k, count] = Array.isArray(part) ? part : [part, null];
        return (UI_SKELETONS[k] || UI_SKELETONS.text)(count);
      }).join("")
    : (UI_SKELETONS[kind] || UI_SKELETONS.chart)(n);
}

/** Drop a hold that its loader never came back to fill (an empty result). The
 *  loaders write innerHTML themselves, which clears the skeleton with it — this
 *  is only for the paths that legitimately render NOTHING and would otherwise
 *  leave a permanent shimmering hole. */
function uiRelease(sel) {
  const el = typeof sel === "string" ? $(sel) : sel;
  if (!el) return;
  el.classList.remove("uiLoading");
  // ONLY a hold, never real content: a loader that succeeded has already
  // replaced the skeleton, and this must not erase what it wrote.
  if (el.querySelector(".uiSkel")) el.innerHTML = "";
}

function uiFailure(label, err, retryId) {
  const why = (err && (err.message || String(err))) || "no response";
  return `<div class="uiFail" role="alert">
      <div class="uiFailIcon" aria-hidden="true">!</div>
      <div class="uiFailBody">
        <b>Couldn't load ${label}</b>
        <div class="uiFailWhy">${why}</div>
      </div>
      <button class="uiRetry" id="${retryId}">Retry</button>
    </div>`;
}

let _uiRetrySeq = 0;

/**
 * Run `work()` and render its HTML into `sel`, handling all three states.
 *
 *   work    async () => htmlString
 *   label   what failed, in a sentence: "the image grid", "runs"
 *   kind    which skeleton shape (card / row / tile / text)
 *   n       how many skeleton placeholders
 *   slowMs  when to admit it is being slow (default 5s, the owner's number)
 *   after   optional (el) => void, run once the real content is in the DOM —
 *           for attaching handlers, since innerHTML discards them
 */
async function loadInto(sel, work, {label = "this", kind = "card", n = null,
                                    slowMs = 5000, after = null} = {}) {
  const el = typeof sel === "string" ? $(sel) : sel;
  if (!el) return;
  el.innerHTML = (UI_SKELETONS[kind] || UI_SKELETONS.card)(n);
  el.classList.add("uiLoading");
  // The slow notice is ADDED beside the skeleton, never replaces it: the work
  // has not failed, and swapping the view would imply it had.
  const slow = setTimeout(() => {
    if (el.classList.contains("uiLoading"))
      el.insertAdjacentHTML("afterbegin",
        `<div class="uiSlow">taking longer than usual…</div>`);
  }, slowMs);
  try {
    const html = await work();
    clearTimeout(slow);
    el.classList.remove("uiLoading");
    // TWO KINDS OF WORK, and the difference is load-bearing. Most panels build
    // a string and we assign it. The image grids cannot: they create each card
    // with createElement so they can attach per-card handlers (delete, move,
    // bucket select, open-editor) to the element itself, and a string would
    // lose all of them. Those render into `el` themselves and return nothing,
    // so assigning here would erase exactly the work we just waited for.
    if (typeof html === "string") el.innerHTML = html;
    if (after) after(el);
    reveal(el);
  } catch (e) {
    clearTimeout(slow);
    el.classList.remove("uiLoading");
    const id = `uiRetry${++_uiRetrySeq}`;
    el.innerHTML = uiFailure(label, e, id);
    const btn = document.getElementById(id);
    if (btn) btn.onclick = () => loadInto(sel, work, {label, kind, n, slowMs, after});
    reveal(el);
  }
}

/* `api()` resolves on 4xx/5xx — fetch only rejects on network failure — so a
 * 404 would render as "undefined" rather than reaching the failure card. This
 * is the version to use inside loadInto(). */
async function apiOrThrow(url, opts) {
  const r = await fetch(withProject(url), opts);
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail || ""; } catch { /* not json */ }
    throw new Error(detail || `${r.status} ${r.statusText}`);
  }
  return r.json();
}
