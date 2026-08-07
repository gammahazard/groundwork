"use strict";
/* The projects landing — the shell the pivot exists to produce.
 *
 * One card per project: what it detects, how it is labelled, how much data it
 * has, and how its best run scored. Opening one records the choice and drops
 * you on that project's Dashboard.
 *
 * WHY THE CARD IS RENDERED FROM A SERVER PAYLOAD AND NOTHING IS COMPUTED HERE:
 * every number on it is already a rule in Python — which collections a project
 * has depends on its extensions, objects is a filesystem walk, and "best run" has
 * a tie-break that matters (seven runs score 0.0, so ordering by MAE alone
 * picks whichever the ledger listed first). Re-deriving any of that in JS is
 * how the second implementation starts. See web/api/projects.py.
 *
 * SCOPING IS WIRED (2026-07-31). This used to say it was not, and a comment
 * that states the opposite of the truth is worse than none — the reader trusts
 * it and stops checking. Opening a project sets CURRENT_PROJECT, puts
 * ?project=<slug> in the URL, and core.js's fetch wrapper carries it on EVERY
 * request from one place (the 22 raw fetch() calls that bypassed api() were the
 * bug that made this real). What is left is not "wire the tabs" but the
 * remaining machine-vs-project judgement calls. PIVOT.md's "what is NOT
 * project-scoped" explains the axis (the GPU lock is a machine resource;
 * project scope LABELS the job, it does not enable concurrency); CLAUDE.md
 * lists the specific call sites by name. */

const pjEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

let CURRENT_PROJECT = null;          // slug, once one is opened
let PROJECT_CARDS = {};              // slug -> card payload

/* The card was labelling its counts with the STORAGE keys — "raw", "needs_fix",
 * "testset" — while the nav two inches to its left called the same three things
 * "Training set", "Fix queue" and "Test set". Two vocabularies for one concept,
 * on one screen. The nav's words win, because they are the ones the operator
 * clicks. The five keys are labelio._SUFFIXES; an extension may bring a sixth
 * this map has never heard of, so the fallback de-underscores rather than
 * hiding it — an unlabelled count is still a true count. */
const COLLECTION_LABEL = {
  raw: "training", testset: "holdout", needs_fix: "need fixing",
};
const collectionLabel = k => COLLECTION_LABEL[k] || String(k).replace(/_/g, " ");

const _fmtAgo = ts => {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 90) return `${Math.round(m)} min ago`;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(d)} d ago`;
};

/* The headline is whatever the PROJECT ranks by, named by the server so the
 * browser never has to know which way is better — mAP is better higher and MAE
 * better lower, and a card that labelled one as the other would present the
 * worst run as the best with nothing about it looking wrong.
 *
 * Say what it is OF, too: a number without its denominator is how a 0.0 on 63
 * images gets mistaken for a 0.0 on 65. */
function _bestRunHTML(b) {
  if (!b) return `<div class="pjMetric pjMuted">no scored run yet</div>`;
  const tie = b.tied_on_mae
    ? `<span class="pjTie" title="${b.tied_on_mae} other run(s) score exactly this — the holdout cannot separate them">${b.tied_on_mae} tied</span>`
    : "";
  const val = b.value ?? b.mae;
  // WHERE it was measured, always. count-MAE is the frozen holdout; mAP50 is
  // the val split the model early-stopped against, which is optimistic by
  // construction. Showing them identically would imply they were equally
  // trustworthy.
  const src = b.metric_source && b.metric_source !== "holdout"
    ? ` <span class="pjTie" title="Measured on the val split the model early-stopped against, not the frozen holdout — optimistic by construction">${pjEsc(b.metric_source)}</span>`
    : "";
  const denom = b.metric_source === "holdout"
    ? `${b.test_images ?? "?"} holdout images` : "val split";
  return `<div class="pjMetric">
     <b>${fmtScore(val)}</b> <span class="pjMuted">${pjEsc(b.metric_label || "MAE")}</span>${src} ${tie}
     <div class="pjSub">${pjEsc(b.run)} · ${denom} · ${_fmtAgo(b.finished)}</div>
   </div>`;
}

/* The card's picture. A real image if the project has one, otherwise a drawn
 * placeholder — NOT a missing image and not a grey box. An empty project is a
 * normal state (it is what every project looks like on day one, and Phase 6
 * exists to create them), so it gets a deliberate empty look that says "no images
 * yet" rather than something that reads as a broken thumbnail.
 *
 * ?v=<mtime> is the image's own timestamp, which is what lets the server mark
 * these immutable: the URL changes the moment the image does, so the browser can
 * cache hard and still never show a stale picture. */
function _thumbHTML(p) {
  const t = p.thumb;
  if (!t)
    return `<div class="pjThumb pjThumbEmpty" aria-hidden="true">
              <span class="pjThumbIcon">▨</span><span class="pjThumbNone">no images yet</span>
            </div>`;
  const url = `/img/${encodeURIComponent(t.collection)}/${encodeURIComponent(t.stem)}`
            + `?w=320&v=${t.v}&project=${encodeURIComponent(p.slug)}`;
  return `<div class="pjThumb">
      <img src="${url}" alt="" loading="lazy" decoding="async">
    </div>`;
}

function _cardHTML(p) {
  const cls = p.classes.map(c => `<span class="chip">${pjEsc(c)}</span>`).join("");
  const ext = (p.extensions || []).map(e => `<span class="chip chipExt">${pjEsc(e)}</span>`).join("")
              || `<span class="pjMuted">none</span>`;
  const cols = Object.entries(p.collections || {})
    .map(([k, v]) => `<div><b>${v}</b><label>${pjEsc(collectionLabel(k))}</label></div>`)
    .join("");
  return `<article class="pjCard" data-slug="${pjEsc(p.slug)}">
    ${_thumbHTML(p)}
    <header class="pjHead">
      <div>
        <h3>${pjEsc(p.name)}</h3>
        <div class="pjSlug">${pjEsc(p.slug)} · ${pjEsc(p.labelling)} · ${p.nc} class${p.nc === 1 ? "" : "es"}</div>
      </div>
      <button class="pjOpen" data-slug="${pjEsc(p.slug)}">Open →</button>
    </header>
    <div class="pjChips">${cls}</div>
    <div class="pjCounts">${cols}</div>
    <div class="pjFoot">
      ${_bestRunHTML(p.best_run)}
      <div class="pjSub">${(p.objects ?? 0).toLocaleString()} labelled objects · active ${_fmtAgo(p.last_activity)}</div>
      <div class="pjSub">extensions: ${ext}</div>
    </div>
  </article>`;
}

/* Skeleton -> cards, or a failure card with Retry. The bespoke "loading…" and
 * "could not load projects" strings this used to carry are what ui.js exists to
 * replace: one vocabulary, so a slow panel looks the same everywhere and a
 * broken one is always recoverable without a page reload. */
async function loadProjects() {
  return loadInto("#projectGrid", async () => {
    const d = await apiOrThrow("/api/projects");
    PROJECT_CARDS = {};
    (d.projects || []).forEach(p => { PROJECT_CARDS[p.slug] = p; });
    /* A project whose manifest will not parse is REPORTED, never dropped: a
     * list that quietly omits a project is how you lose one. */
    const broken = (d.broken || []).map(b =>
      `<article class="pjCard pjBroken"><h3>${pjEsc(b.slug)}</h3>
        <div class="pjErr">manifest unreadable — ${pjEsc(b.error)}</div></article>`).join("");
    return (d.projects || []).map(_cardHTML).join("")
      + `<article class="pjCard pjNew" id="pjNew" role="button" tabindex="0">
           <div class="pjPlus">+</div>
           <h3>New project</h3>
           <div class="pjSub">Declare what it detects. You can add data afterwards.</div>
         </article>` + broken;
  }, {
    label: "projects", kind: "card", n: 3,
    // innerHTML discards handlers, so wiring belongs in `after`, not before.
    after: host => {
      host.querySelectorAll(".pjOpen").forEach(b => b.onclick = () => openProject(b.dataset.slug));
      host.querySelectorAll(".pjCard[data-slug]").forEach(c =>
        c.ondblclick = () => openProject(c.dataset.slug));
      const n = host.querySelector("#pjNew");
      if (n) {
        n.onclick = openNewProject;
        n.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openNewProject(); } };
      }
      revealAll(host, 40, 8);        // stagger the cards in
    },
  });
}

/* Creating a project — openNewProject and its modal — lives in
 * cockpit/projects_new.js. It shares no state with this file; loadProjects()
 * below wires the "+ New project" card to it. */

/* Which tabs an EXTENSION brings with it. A project detecting screws has no use
 * for a DIN formulary, an object-cutout compositor, or the concept of half a screw
 * (PIVOT.md) — so those tabs must not merely be empty, they must be absent.
 *
 * The server already refuses: `collection_names(pp)` drops collections behind an
 * extension a project lacks, and /api/bots filters the same way. The CLIENT
 * never did, which was invisible while there was one project that enabled all
 * three. Opening a second project with none of them showed Synthesize and Drug
 * ID anyway — tabs that can only ever be empty, on data that cannot exist. */
const EXTENSION_TABS = {};   // extension tab -> nav id; none ship built-in

function applyExtensionTabs(p) {
  const on = new Set(p && p.extensions ? p.extensions : []);
  for (const [ext, tab] of Object.entries(EXTENSION_TABS)) {
    // NOT scoped to #navPanel any more: a project's tabs live in the
    // .projTabs strip at the top of the content now, and the old scope
    // would have quietly matched nothing — leaving every extension tab
    // visible for every project.
    const btn = document.querySelector(`nav button[data-tab="${tab}"]`);
    if (btn) btn.hidden = !on.has(ext);
  }
  // Which extensions exist decides whether the strip needs a ⋯ at all.
  if (window.projTabsSync) projTabsSync();
}

/* LEAVING a project — the one implementation, called from two places.
 *
 * It used to live only inside the "← all projects" button's handler, and the
 * nav's Projects entry just switched tab. So the landing page could be showing
 * every project while `body.inProject` was still set, the project's nav group
 * still listed, and PROJECT still on every request: clicking Projects LOOKED
 * like leaving and was not. Two controls that read the same must do the same.
 *
 * The URL is cleared too, or a refresh from the landing page drops you back
 * inside the project you just left.
 */
function closeProject() {
  PROJECT = null; CURRENT_PROJECT = null;
  if (window.resetRunsTable) resetRunsTable();
  document.body.classList.remove("inProject");
  delete document.body.dataset.project;
  const u = new URL(location); u.searchParams.delete("project");
  history.pushState({}, "", u);
  const bar = $("#projectBar");
  if (bar) bar.hidden = true;
  showTab("projects");
}

function openProject(slug, push = true) {
  // Leaving a project must drop what belonged to it. The shared runs table is
  // fed by two loaders that CONTRIBUTE rows and keep them; without this, opening
  // a second project showed the first one's 80 runs under the new project's name
  // until its own (empty) ledgers replied — and for the challenger ledger, which
  // has no project dimension yet, they would simply have stayed.
  if (CURRENT_PROJECT && CURRENT_PROJECT !== slug && window.resetRunsTable)
    resetRunsTable();
  CURRENT_PROJECT = slug;
  PROJECT = slug;                       // core.js: every api() call now carries it
  // Same seam as body.lab in identity.js: one class, CSS branches off it.
  // 10-project.css changes SHAPE as well as tint, because colour alone cannot
  // carry the signal here.
  document.body.classList.add("inProject");
  document.body.dataset.project = slug;
  if (push) {
    // Own the URL so refresh / bookmark / share land back here. replaceState on
    // restore, pushState on a click, so Back returns to where you were rather
    // than replaying project switches you never made.
    const u = new URL(location);
    u.searchParams.set("project", slug);
    history.pushState({project: slug}, "", u);
  }
  const p = PROJECT_CARDS[slug];
  // The dashboard heading names the project. index.html claims every id in
  // that section is written by dashboard.js; this one had no writer anywhere,
  // so the heading read "Serving now" for every project you ever opened.
  const title = $("#dashProjectTitle");
  if (title) title.textContent = p ? `${p.name} — serving now` : "Serving now";
  applyExtensionTabs(p);                // a project's tools are its own
  const bar = $("#projectBar");
  if (bar && p) {
    bar.hidden = false;
    // The class list is the one thing here that is EDITABLE, so it is a button
    // rather than text — it used to be the only way to see a project's classes
    // and the only way to change them was hand-editing project.json.
    bar.innerHTML = `<span class="pjOpenName">${pjEsc(p.name)}</span>
      <span class="pjSub"><button id="pjClasses" class="pjClsLink"
        title="Add, rename or remove this project's classes"
        >${pjEsc((p.classes||[]).join(', '))}</button> · ${p.images ?? 0} images${
          (p.collections?.needs_fix)
            ? ` · <b class="pjWaiting">${p.collections.needs_fix} to label</b>` : ""
        }</span>
      <button id="pjBack" class="pjBackBtn">← all projects</button>`;
    $("#pjBack").onclick = () => closeProject();
    const cls = $("#pjClasses");
    if (cls) cls.onclick = () => window.openClassEditor && window.openClassEditor(slug);
  }
  // The retrain/models panels are primed on OPEN, not at boot — boot no longer
  // knows a project (nothing auto-opens one), and priming them from the default
  // project on a page deliberately showing none is the same leak the Dashboard
  // split exists to prevent.
  if (window.loadYolo) loadYolo();
  // ...and land on the PROJECT's dashboard. Landing on the machine's would
  // answer a question nobody just asked by clicking into a project.
  showTab("projdash");
}

/* Where a bare load lands you.
 *
 * ?project=<slug> puts you INSIDE that project rather than on the landing page
 * with a stale header. With NO param the honest answer depends on how many
 * projects there are, and getting this wrong strands the user completely:
 *
 * ?project=<slug> puts you inside it. NOTHING ELSE DOES (owner, 2026-07-31).
 *
 * This used to auto-open when exactly one project existed, on the reasoning
 * that a landing page listing one card is a menu of one, and that outside a
 * project `body.inProject` is unset so the nav is two items and Models & Runs
 * bounces. That reasoning was solving a different problem: the complaint it
 * came from ("I can't even see the yolo run info") was about a tab you were
 * being redirected away from, and the fix for THAT is a landing page that
 * makes opening a project obvious — which it now is, one click on the card.
 *
 * Auto-opening also broke the thing the owner actually asked for: with a
 * project silently open, the global Dashboard showed one project's serving
 * model, image counts and odometer under a heading that claims to describe the
 * machine. You cannot have "the Dashboard is project-free" AND "a project is
 * always open". The card is the door, and you have to walk through it.
 *
 * The URL is still not written by openProject on restore — see openProject.
 */
async function restoreProjectFromURL() {
  const asked = new URLSearchParams(location.search).get("project");
  if (!Object.keys(PROJECT_CARDS).length) {
    try { (await api("/api/projects")).projects.forEach(p => PROJECT_CARDS[p.slug] = p); }
    catch { return false; }
  }
  if (asked && PROJECT_CARDS[asked]) { openProject(asked, false); return true; }
  if (asked) _forgetBadProject(asked);
  return false;
}

/* A ?project= slug that does not exist must be FORGOTTEN, not merely ignored.
 *
 * core.js reads PROJECT out of the URL at load, before anything can check it,
 * and withProject() appends it to every api() call. So a typo'd or renamed slug
 * does not degrade to the landing page — it 404s /api/state and every tab fetch
 * for the rest of the session, because deps.py correctly refuses an unknown
 * project rather than silently serving the default one.
 *
 * Returning false here without clearing PROJECT left exactly that: a landing
 * page that renders (because /api/projects lists all projects and ignores the
 * param) sitting on top of a global that poisons every other request. Found by
 * check_cockpit_live.mjs, which loads ?project=definitely-not-a-project on
 * purpose — the entry-state axis that the inProject bug also lived on.
 *
 * The param is stripped with replaceState, not pushState: the bad URL is not a
 * place worth being able to navigate Back into.
 */
function _forgetBadProject(slug) {
  console.warn(`unknown project "${slug}" — ignoring it`);
  PROJECT = null;
  CURRENT_PROJECT = null;
  const u = new URL(location);
  u.searchParams.delete("project");
  history.replaceState({}, "", u);
}

// Back/forward through project switches
window.addEventListener("popstate", () => {
  const slug = new URLSearchParams(location.search).get("project");
  if (slug && PROJECT_CARDS[slug]) openProject(slug, false);
  else { PROJECT = null; CURRENT_PROJECT = null;
         document.body.classList.remove("inProject");
         delete document.body.dataset.project;
         const b = $("#projectBar"); if (b) b.hidden = true; showTab("projects"); }
});
