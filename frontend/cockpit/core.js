"use strict";
/* Cockpit core: the two helpers every other script leans on ($, api) + tab
 * switching + the tiny Count tab. LOAD ORDER CONTRACT: this file loads first;
 * identity.js, grids.js, dashboard.js follow (dashboard.js bootstraps at its
 * end), then the feature scripts (buckets, editor, history, …). All classic
 * scripts — cross-file references resolve at call time, not load time. */
const $ = s => document.querySelector(s);

/* ONE way to write a model score — MAE or mAP alike — because the same number
 * appears in a dozen panels and they disagreed about how many digits it has.
 * JSON drops trailing zeros, so a PERFECT score rendered as a bare `0` in a
 * column of `0.0154`s: the best number on the page looked like a missing one.
 * Four decimals is what the conf x iou grid resolves to. A number >= 1 (rfdetr's
 * 1.375) keeps its own shape rather than growing meaningless zeros. */
const fmtScore = v => v == null ? "–" : (Math.abs(v) < 1 ? v.toFixed(4) : String(v));

/* ---------- which project am I looking at ----------
 * Lives in the URL (?project=<slug>) so refresh, bookmark and share all keep
 * you where you were — it matters most on a phone, where the tab gets reopened
 * constantly. Null means "the default", which is what every call meant before
 * projects existed, so an unadorned URL behaves exactly as it always has.
 *
 * withProject() is applied inside api() AND inside a fetch wrapper (below), so
 * every request inherits it from ONE place. Image URLs built into src= do not go
 * through either, so they call it explicitly — grep withProject to find them. */
let PROJECT = new URLSearchParams(location.search).get("project") || null;
const withProject = u => {
  // /static and /outputs are StaticFiles mounts — no project, and appending one
  // would just bust their cache keys for nothing.
  if (!PROJECT || /^https?:\/\//.test(u) || u.startsWith("/static/") || u.startsWith("/outputs/"))
    return u;
  // IDEMPOTENT: the wrapper below and api() can both see the same URL, and a
  // caller may have added it already. Appending twice is not merely untidy —
  // FastAPI takes the FIRST value, so a second one is silently ignored and the
  // bug would only appear if the two ever disagreed.
  if (/[?&]project=/.test(u)) return u;
  return u + (u.includes("?") ? "&" : "?") + "project=" + encodeURIComponent(PROJECT);
};

/* EVERY request carries the project, not just the ones that remembered to.
 *
 * api() has always applied withProject, but 22 call sites used raw fetch() —
 * and among them were the writes: saving dots, cropping an image in place,
 * setting a target count, tagging a bucket, deleting an image. With a
 * project open they all wrote to the DEFAULT project instead. Found when a save
 * in a new project 500'd; had the stem existed in the first project it would have
 * silently overwritten its labels, which is the worst outcome this repo has.
 *
 * Fixing 22 call sites fixes today and is wrong again the moment someone adds a
 * 23rd, so the rule lives at the one place every request must pass through.
 * Same-origin app paths only — /static and /outputs are excluded by
 * withProject, and anything absolute is left alone.
 */
const _rawFetch = window.fetch.bind(window);
window.fetch = (input, init) =>
  _rawFetch(typeof input === "string" ? withProject(input) : input, init);

const api = (u, o) => fetch(withProject(u), o).then(r => r.json());

/* ---------- tabs ---------- */
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  // Clicking PROJECTS means "I am done with this one" — it is the landing page,
  // and being on it while still inside a project is a contradiction the user
  // can see (the project's nav group stays listed). Since the merge it is also
  // the machine view, which is why leaving a project and consulting the machine
  // are now the same gesture rather than two.
  if (b.dataset.tab === "projects" && document.body.classList.contains("inProject")
      && window.closeProject)
    return closeProject();
  showTab(b.dataset.tab);
});
/* Tabs that exist WITHOUT a project open. showTab() bounces anything else back
 * to the landing page, so this list and the nav's global group must agree —
 * they are the same statement in two files, and "collect" moving to the global
 * group while this stayed at two entries made Data collection silently
 * unreachable: the button was visible, and clicking it landed you on Projects. */
/* "dashboard" is kept in the list even though the tab is GONE. showTab() maps
 * it to "projects" below, so a bookmark, a browser back-button entry or any
 * caller that still names it lands on the page that absorbed it rather than
 * being bounced somewhere arbitrary. Remove when nothing names it. */
const NAV_GLOBAL_TABS = ["projects", "dashboard", "collect", "guide",
                         "machines", "api", "admin", "account"];
const NAV_MERGED = {dashboard: "projects",
                    // the three grid tabs merged into Images; upload.js
                    // still jumps to "fix" after an upload, and a
                    // bookmark may name any of them.
                    fix: "images", train: "images", testset: "images"};

/* IS EITHER HALF OF THE OLD RUNS TAB ON SCREEN?
 *
 * Models & Runs split into Train (#runs) and Results (#results), and the things
 * that keep a live training run visible do NOT respect that seam: loadYolo
 * feeds the progress bar on one and the model list on the other; loadLab feeds
 * #labStatus on one and #labRuns on the other. Every poller therefore asks this
 * rather than naming a tab — six literals would be six chances to update five
 * of them, on the paths that make a run cancellable. */
const onRunsTabs = () =>
  !!(document.querySelector("#runs.active") || document.querySelector("#results.active"));
window.onRunsTabs = onRunsTabs;
function showTab(name){
  // The machine dashboard was merged INTO Projects, so anything still asking
  // for it gets the page that absorbed it — not a blank tab and not a bounce.
  name = NAV_MERGED[name] || name;
  // A project's tabs vanish when you leave it (11-nav.css). Landing on one
  // would show an empty page with no nav entry lit — fall back to Projects.
  if (!document.body.classList.contains("inProject")
      && !NAV_GLOBAL_TABS.includes(name)) name = "projects";
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === name));
  // THREE COLLECTIONS, ONE TAB. cockpit/images.js owns which pane is showing
  // and loads only that one. Halves stays its own tab — it is an extension,
  // shown only for projects that declare it.
  if (name === "images") loadImages();

  if (name === "collect") loadCollect();
  if (name === "account" && window.loadAccount) loadAccount();
  /* THE API TAB OWNS THE KEYS NOW. loadKeys lives in cockpit/account.js and is
   * called from both — Account no longer shows them, but the function is the
   * one place that knows how to render and revoke a key. */
  if (name === "admin" && window.loadAdmin) loadAdmin();
  if (name === "api") {
    if (window.loadApiDocs) loadApiDocs();
    if (window.loadKeys) loadKeys();
  }
  if (name === "machines" && window.loadMachines) loadMachines();
  // Built once, then it is just a step change — see tabs/guide.js.
  if (name === "guide" && window.renderGuide) renderGuide();
  // Export is its own tab now and asks the ledgers for its own runs, so it no
  // longer has to wait for Results to have been opened first.
  if (name === "export" && window.loadExport) loadExport();
  /* THE LANDING PAGE HAS TWO LOADERS because it answers two things from two
   * endpoints: /api/overview for the machine, /api/projects for the cards.
   * Both fire; each owns its own containers and its own skeleton. */
  if (name === "projects"){ loadOverview(); loadProjects(); }
  // ONE dashboard per SCOPE, and the machine's is the landing page above.
  // "projdash" is whichever project is open. These were once a single tab
  // calling both loaders, which is how opening it inside a project showed
  // project data under a machine heading.
  // HOLD FIRST, THEN FETCH — same order as the Runs tab below, and the reason is
  // the same: loadState writes six containers from one call, so no single
  // loadInto owns them and the hold has to be placed here.
  if (name === "projdash"){
    projdashHold();
    loadState();
    // The Serving controls live here now, beside the readout they change.
    // loadYolo fills the model list, loadEngine the architecture switch.
    loadYolo();
    if (window.loadEngine) loadEngine();
  }
  /* TRAIN — "now, watch it". The ledger half moved to #results below; what is
   * left here is the control and everything live. */
  if (name === "runs"){
    uiHold("#tCards", "tile", 2);           // one box per GPU
    loadTrain();      // the one Train control — model x machine x card
    // loadYolo feeds BOTH tabs: the retrain status/progress/cancel here, and
    // the model list on Results. loadLab likewise — #labStatus is here,
    // #labRuns is there. Neither loader respects the seam, so both tabs call
    // both; the containers exist in the DOM either way (tabs are display:none,
    // not removed), so a write always lands.
    loadYolo();
    loadLab().catch(e => runsFail("lab", e));
  }
  /* RESULTS — "what have I got". The two ledger loaders feed ONE shared table
   * (cockpit/runs/runs_table.js), so a skeleton goes up first and each failure
   * is caught by name: one endpoint being down must not hide the other's runs. */
  if (name === "results"){
    /* RESERVE THE SPACE FIRST, so the panels do not shove each other down as
     * they land. Measured at 1280x900 before this existed: every one of these
     * went from ZERO height to full size in a single jump as its fetch
     * returned — #labRuns 0->591, #dataCharts 0->494, #runsCharts 0->201 —
     * about 2400px of content appearing under the reader, each arrival pushing
     * everything below it down the page.
     *
     * Five fetches across three loaders fill six containers, which is why these
     * are held here rather than each loader calling loadInto: no loader owns a
     * single container, so none of them could.
     *
     * uiHold is a no-op on a container that already has content, so re-entering
     * the tab does not flash a skeleton over runs already on screen. */
    uiHold("#runsSummary", "line", 1);
    uiHold("#runsCharts", "chart", 2);      // MAE + mAP50, sometimes + buckets
    uiHold("#dataCharts", "chart", 1);      // one full-width dataset chart
    uiHold("#labPinned", "line", 1);
    // Not just the chart: #labRuns is the metric picker, the chart, and the
    // "models not shown" note — holding only the middle one let the other two
    // shove it around anyway.
    uiHold("#labRuns", [["line", 1], ["chart", 1], ["text", 2]]);
    runsTableLoading();
    loadYolo();       // the Models/pin list lives on this tab
    /* A FAILED LOADER MUST DROP ITS HOLDS. Both write their containers on the
     * success path only, so without this a dead endpoint leaves a skeleton
     * shimmering forever — which reads as "still loading" and never resolves.
     * runsFail already names the failure in the shared table; this just stops
     * the holes pretending. */
    loadHistory().catch(e => {
      runsFail("yolo", e);
      ["#runsSummary", "#runsCharts", "#dataCharts"].forEach(uiRelease);
    });
    loadLab().catch(e => {
      runsFail("lab", e);
      ["#labPinned", "#labRuns"].forEach(uiRelease);
    });
  }
}

/* ---------- count ---------- */
$("#countForm").onsubmit = async e => {
  e.preventDefault();
  const f = $("#countFile").files[0]; if (!f) return;
  $("#countResult").innerHTML = "⏳ counting…";
  const fd = new FormData(); fd.append("image", f);
  const r = await api("/api/count", {method:"POST", body:fd});
  $("#countResult").innerHTML = `<div class="big">🔢 ${r.count}</div><div class="hint">${r.seconds}s</div><img src="${r.overlay}?t=${Date.now()}">`;
};
