"use strict";
/* The Dashboard's headline: what is on this MACHINE, across every project.
 *
 * The Dashboard lives in the global half of the nav, next to Projects — but it
 * described one project, because it was written when there was one and "the
 * dataset" meant the first project's. Opening a second project made that visible: a
 * global tab reporting one project's image count as though it were the total.
 *
 * So the top of the page is now platform-wide (this file) and the rest still
 * describes whichever project is open (dashboard.js). Two questions, two
 * sections, rather than one section quietly answering the wrong one.
 *
 * MODELS, NOT CHAMPION-VS-CHALLENGER. Runs are grouped by model and nothing
 * else. yolov8n is not privileged here — it is simply the model that happens to
 * serve the first project today, and the two-ledger split it comes from is a storage
 * detail (training_history.json vs outputs/alt) that Phase 2 deletes by moving
 * everything under <project>/models/<model>/<run>/. Counting them in two buckets
 * would re-teach the framing the pivot is removing.
 *
 * Every number is summed SERVER-side from the same `_card()` the landing page
 * renders (web/api/projects.py::overview), so a total can never disagree with
 * the cards it totals. Two independent walks of the same directories is how a
 * dashboard starts contradicting the page below it.
 */

const _ovNum = n => (n ?? 0).toLocaleString();

const _ovAgo = ts => {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 90) return `${Math.round(m)} min ago`;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(d)} d ago`;
};

/* MAE is meaningless without saying what it is OF, and across models it is
 * worse than meaningless: the holdout has SATURATED, so several runs tie at
 * 0.0 and a bare "best" implies a winner the exam cannot actually separate. */
function _ovModelRows(models) {
  if (!models || !models.length)
    return `<div class="sbRow"><span class="mMeta">no runs recorded yet</span></div>`;
  return models.map(m => `
    <div class="sbRow">
      <span><b>${_ovEsc(m.model)}</b>
        ${m.license ? `<span class="lic">${_ovEsc(m.license.replace("-2.0", ""))}</span>` : ""}
        <span class="mMeta">· ${m.runs} run${m.runs === 1 ? "" : "s"}</span></span>
      <span class="sbMae">${fmtScore(m.best_mae)}</span>
    </div>`).join("");
}

const _ovEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function loadOverview() {
  return loadInto("#overviewTiles", async () => {
    const d = await apiOrThrow("/api/overview");
    /* WHICH MACHINE, from the platform endpoint. It used to come from
     * /api/state, which the Dashboard no longer calls until a project is open
     * — so on the lab the ember theme and the "edit on HQ only" banner would
     * not appear until someone clicked into a project. That is the one machine
     * where "am I allowed to edit here?" must never be in doubt. */
    window.IS_LAB = !!d.lab;
    applyMachineTheme(window.IS_LAB);
    _ovRenderPanels(d);
    /* COUNTS ARE PLAIN NUMBERS, CARDS ARE THINGS YOU OPEN. These were six
     * gradient cards with borders and shadows, sitting directly above the
     * project cards — so a fact and a destination looked identical. The counts
     * now carry their weight through the type scale, which leaves "is a card"
     * meaning exactly one thing on this page: you can click it. Same treatment
     * as the project Overview's own counts. */
    return `
      <div class="gwCt"><span class="v">${_ovNum(d.projects)}</span><span class="k">projects</span></div>
      <div class="gwCt"><span class="v">${_ovNum(d.images)}</span><span class="k">images</span></div>
      <div class="gwCt"><span class="v">${_ovNum(d.objects)}</span><span class="k">labelled objects</span></div>
      <div class="gwCt"><span class="v">${_ovNum(d.runs)}</span><span class="k">runs</span></div>
      <div class="gwCt"><span class="v">${_ovNum(d.models)}</span><span class="k">models</span></div>
      <div class="gwCt"><span class="v">${d.bots_running}/${d.bots}</span><span class="k">bots running</span></div>`;
  }, {label: "the overview", kind: "tile", n: 6});
}

function _ovRenderPanels(d) {
  const m = $("#overviewModels");
  if (m) m.innerHTML = _ovModelRows(d.per_model);
  const p = $("#overviewProjects");
  if (!p) return;
  /* NO PROJECT ROWS ANY MORE. They existed because the machine dashboard was a
   * separate tab with no cards on it; since the merge the CARDS sit directly
   * below this, so a row list beside them was the same information twice — and
   * the one thing a duplicate list guarantees is that the two disagree
   * eventually.
   * What survives is what the cards do NOT say: when anything last happened,
   * and any project whose manifest cannot be read. A broken manifest must never
   * be the thing that gets dropped in a tidy-up. */
  p.innerHTML =
    (d.last_activity ? `<span class="gwNote">last activity `
        + `<b>${_ovAgo(d.last_activity)}</b></span>` : "")
    + ((d.broken || []).map(b =>
        `<span class="gwNote bad">⚠ ${_ovEsc(b.slug)} — manifest unreadable</span>`).join(""));
}
