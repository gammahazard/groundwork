"use strict";
/* Creating a project — the "+ New project" card's modal.
 *
 * SPLIT OUT OF cockpit/projects.js (2026-08-02). It is the one piece of that file
 * that neither renders a card nor routes between projects: a form, a preview, a
 * POST. It reads no module state, writes none, and is reached only through
 * openNewProject().
 *
 * THE FORM ASKS FOR THE FOUR THINGS A PROJECT CANNOT BE INFERRED WITHOUT and
 * nothing else. No slug field: the id is DERIVED server-side from the name and
 * validated there, because it becomes a directory name and a form must not be
 * able to choose a path. No dataset_root either — a project that could be pointed
 * at another project's data is the worst possible way to lose a dataset. What is
 * shown as you type is a PREVIEW of the server's decision, not an input.
 *
 * Every rule stays in Python (reserved ids, path-ish names, unknown extensions);
 * the catch below shows what the server said rather than re-implementing any of
 * it. See web/api/projects.py::create_project.
 */

/* ---------- create a project ----------------------------------------------
 * The pivot's headline gap: the landing page has said "Phase 6 — not built yet"
 * since it shipped.
 *
 * The earlier plan held that this needed the upload/label flow first: a POST that
 * minted an empty manifest would just be a way to make a broken project. (That
 * was written here as a QUOTE from PIVOT.md; the phrase appears nowhere in that
 * file's history, so it is stated as the argument it is.) True when written,
 * not now: Phase 4 gave every panel an empty state, so a project
 * with no data reads as a coherent empty project rather than a half-drawn page —
 * "Nothing here", a drawn no-images-yet thumbnail, no runs. check_multiproject
 * creates exactly this and asserts every tab is empty.
 *
 * The form asks for the four things a project cannot be inferred without, and
 * nothing else. No slug field: the id is DERIVED server-side from the name and
 * validated there, because it becomes a directory name and a form should not be
 * able to choose a path. No dataset_root either, for the same reason — a project
 * that could be pointed at another project's data is the worst possible way to
 * lose a dataset.
 */
function openNewProject() {
  const body = $("#runModalBody");
  body.innerHTML = `
    <h3 class="modalTitle">New project</h3>
    <form id="pjNewForm" class="pjForm">
      <label>Name
        <input id="pjfName" type="text" required maxlength="40" autocomplete="off"
               placeholder="e.g. Bolt Counter">
        <span class="pjSub" id="pjfSlug">its id is made from the name</span>
      </label>
      <label>What it detects
        <input id="pjfClasses" type="text" required autocomplete="off"
               placeholder="bolt, nut" value="object">
        <span class="pjSub">comma-separated. One class is the simplest case and
          what the original project uses; more makes it a multi-class project.</span>
      </label>
      <label>Labelling
        <select id="pjfLabelling">
          <option value="dots">dots — a centre per object, extent from spacing</option>
          <option value="boxes">boxes — the extent is measured, not derived</option>
        </select>
        <span class="pjSub">Recorded as the project's intent. <b>The editor works
          the same way for both today</b> — place a centre, then size it with
          ⬚&nbsp;Boxes — so this changes what the project SAYS it is, not yet how
          you label it. Dots is how the original project's 160 images were made.</span>
      </label>
      <label>Ranked by
        <select id="pjfMetric">
          <option value="count_mae">count MAE — objects per image, lower is better</option>
          <option value="map50_holdout">mAP50 — detection quality on the frozen holdout, higher is better</option>
        </select>
        <span class="pjSub" id="pjfMetricHint">Which number decides a project's best
          run. Count-MAE collapses an image to one number, which is right for
          counting and <b>meaningless with several classes</b> — three bolts and one
          nut counted as four objects is a perfect score for a model that cannot
          tell them apart. mAP50 here is measured on the <b>frozen holdout</b>, by
          the same validator the trainer uses.</span>
      </label>
      <div class="pjFormErr" id="pjfErr" hidden></div>
      <div class="pjFormActs">
        <button type="button" id="pjfCancel">Cancel</button>
        <button type="submit" id="pjfCreate" class="pjfPrimary">Create project</button>
      </div>
    </form>`;
  $("#runModal").hidden = false;
  const name = $("#pjfName"), slug = $("#pjfSlug");
  // Show the derived id as they type. It is not editable and not submitted —
  // this is a preview of what the server will decide, so the directory name is
  // never a surprise.
  name.oninput = () => {
    const s = name.value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "").replace(/-{2,}/g, "-");
    slug.textContent = s ? `id: ${s}` : "its id is made from the name";
  };
  // Suggest the metric that matches the shape of the project. Not forced — the
  // owner may have a counting project with two classes — but a multi-class
  // project defaulting to count-MAE would be a wrong default that looks fine.
  const classes = $("#pjfClasses"), metric = $("#pjfMetric");
  classes.oninput = () => {
    const n = classes.value.split(",").map(s => s.trim()).filter(Boolean).length;
    if (!metric.dataset.touched) metric.value = n > 1 ? "map50_holdout" : "count_mae";
  };
  metric.onchange = () => { metric.dataset.touched = "1"; };
  name.focus();
  $("#pjfCancel").onclick = () => { $("#runModal").hidden = true; };
  $("#pjNewForm").onsubmit = async e => {
    e.preventDefault();
    const btn = $("#pjfCreate"), err = $("#pjfErr");
    btn.disabled = true; btn.textContent = "creating…";
    err.hidden = true;
    try {
      const r = await apiOrThrow("/api/projects", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          name: name.value.trim(),
          classes: $("#pjfClasses").value.split(",").map(s => s.trim()).filter(Boolean),
          labelling: $("#pjfLabelling").value,
          headline_metric: $("#pjfMetric").value,
        })});
      $("#runModal").hidden = true;
      await loadProjects();
      // Straight into it: creating a project and being left on the list is a
      // step you would immediately take yourself.
      openProject(r.project.slug);
    } catch (ex) {
      // The server owns every rule (reserved ids, path-ish names, unknown
      // extensions), so show what it said rather than re-implementing any of it.
      err.textContent = ex.message || String(ex);
      err.hidden = false;
      btn.disabled = false; btn.textContent = "Create project";
    }
  };
}
