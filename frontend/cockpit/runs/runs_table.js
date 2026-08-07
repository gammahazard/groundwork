"use strict";
/* ONE runs table, fed by two families that share almost no fields.
 *
 * Models & Runs used to stack two tables: the yolo ledger (27 rows) and the
 * challenger ledger (54 rows). Same question — "which model counts best?" —
 * asked twice, in two shapes, so the champion and the thing trying to beat it
 * could never be read side by side.
 *
 * WHY A COORDINATOR AND NOT ONE RENDERER. The payloads share exactly three
 * keys: run, mae, imgsz. /api/runs carries epochs_trained, peak_vram_gb, map50,
 * buckets, note, delta, machine, project; /api/lab/runs carries arch, license,
 * conf, iou, n_misses, num_queries, created. Each family also owns ~8 action
 * handlers (curve, log, dots, peek, snapshot, diff, use / detail, score) that
 * close over its own lookup tables. Rewriting both into one normalised renderer
 * would be a large diff whose only visible gain is the merge itself — and it
 * would put every one of those handlers at risk.
 *
 * So each family keeps building its own <tr> and wiring its own buttons. This
 * module only decides ORDER and owns the shell. A family contributes rows via
 * runsContribute(); the table re-renders and re-wires every time, which makes
 * the two async loaders race-free by construction — whichever arrives second
 * simply triggers another render, and a family that never arrives costs
 * nothing but its own rows.
 *
 * SORTING IS THE POINT, not decoration. CLAUDE.md records that the holdout has
 * SATURATED: runs 44/45/46/49/50 all score exactly 0.0, and the DEIM family
 * spans 0.0154-0.0615. Chronological order answers "what did I run last";
 * best-MAE order answers "what should ship", and it is the only view in which
 * a challenger beating the champion is obvious rather than two tables apart.
 * Newest stays the default because it is what the tab did before.
 *
 * The model column carries the LICENCE, because it is decision-relevant here
 * and not cosmetic: ultralytics is AGPL and may cover the trained weights,
 * which is the open question blocking app distribution (IPHONE.md Part 6). The
 * challengers exist precisely because they are Apache-2.0. A table that ranks
 * them without saying which is which invites picking the one you cannot ship.
 */

/* key -> {rows: [{mae, when, model, html}], wire: (container) => void} */
const _RUN_SOURCES = {};
const _RUN_FAILS = {};                  // key -> reason, for families that threw
let _runsSort = "newest";

/** A family hands over its rows and a function to wire them once in the DOM. */
function runsContribute(key, rows, wire) {
  delete _RUN_FAILS[key];
  _RUN_SOURCES[key] = {rows, wire};
  renderRunsTable();
}

/* A family that failed is NAMED, and the other family's rows still render.
 * A whole-panel failure card would be wrong here: the champion ledger and the
 * challenger ledger are independent endpoints, and losing one is no reason to
 * hide the other. This is the same three-state idea as ui.js, applied where the
 * panel has two independent feeds. */
function runsFail(key, err) {
  _RUN_FAILS[key] = (err && (err.message || String(err))) || "no response";
  renderRunsTable();
}

/* Switching project must DROP what the last one contributed. The two ledger
 * loaders hand rows over and this module keeps them, which is right within a
 * project and wrong across one: opening a second project showed the first's 80
 * runs under the new name until its own ledgers replied, and the challenger
 * ledger has no project dimension yet, so for that half they would have stayed
 * indefinitely. Attributing one project's runs to another is the exact
 * silent-wrong-data failure this repo cares most about. */
function resetRunsTable() {
  for (const k of Object.keys(_RUN_SOURCES)) delete _RUN_SOURCES[k];
  for (const k of Object.keys(_RUN_FAILS)) delete _RUN_FAILS[k];
  const box = $("#runsTable");
  if (box) box.innerHTML = "";
}

/* Skeleton on tab entry, so an empty box never means "no runs" while it is
 * still fetching. Skipped once a family has reported, or reopening the tab
 * would flash a skeleton over rows already on screen. */
function runsTableLoading() {
  if (Object.keys(_RUN_SOURCES).length) return;
  const box = $("#runsTable");
  if (!box) return;
  box.classList.add("uiLoading");
  box.innerHTML = UI_SKELETONS.row(8);
}

/* Ranking rule, stated once. Unscored runs always sink: a missing MAE is not a
 * good one, and floating them to the top of a "best" sort would be a lie. */
function _runsCmp(a, b) {
  if (_runsSort === "mae") {
    const am = a.mae == null, bm = b.mae == null;
    if (am !== bm) return am ? 1 : -1;
    if (!am && a.mae !== b.mae) return a.mae - b.mae;
  }
  return (b.when || 0) - (a.when || 0);
}

function renderRunsTable() {
  const box = $("#runsTable");
  if (!box) return;
  box.classList.remove("uiLoading");
  const srcs = Object.values(_RUN_SOURCES);
  const rows = srcs.flatMap(s => s.rows);
  // Named by what they ARE — two stores, not two ranks. This string only
  // surfaces inside a failure card, which is why the vocabulary sweep
  // missed it: it is assembled from a lookup rather than written inline.
  const LEDGER = {yolo: "the yolov8n run history", lab: "the other models' run history"};
  const fails = Object.entries(_RUN_FAILS).map(([k, why]) =>
    `<div class="uiFail" role="alert">
       <div class="uiFailIcon" aria-hidden="true">!</div>
       <div class="uiFailBody"><b>Couldn't load ${LEDGER[k] || k}</b>
         <div class="uiFailWhy">${why}</div></div>
     </div>`).join("");
  if (!rows.length) {
    box.innerHTML = fails || `<p class='hint'>No runs recorded yet.</p>`;
    return;
  }
  rows.sort(_runsCmp);

  const btn = (mode, label, title) =>
    `<button class="runsSortBtn${_runsSort === mode ? " active" : ""}" `
    + `data-sort="${mode}" title="${title}">${label}</button>`;

  box.innerHTML = fails +
    `<div class="runsSortBar">
       <span class="mMeta">${rows.length} run${rows.length === 1 ? "" : "s"} across `
    + `${new Set(rows.map(r => r.model)).size} model${new Set(rows.map(r => r.model)).size === 1 ? "" : "s"}</span>
       <span class="runsSortGroup" role="group" aria-label="Sort runs">
         ${btn("newest", "newest", "Most recently finished first")}
         ${btn("mae", "best MAE", "Lowest holdout MAE first — the cross-family comparison; unscored runs sink to the bottom")}
       </span>
     </div>
     <div class="tblwrap"><table class="runs"><thead><tr>
       <th>Model</th><th>Run · details</th><th>Holdout MAE</th><th>Δ</th><th>Note</th><th></th>
     </tr></thead><tbody>${rows.map(r => r.html).join("")}</tbody></table></div>`;

  box.querySelectorAll(".runsSortBtn").forEach(b => {
    b.onclick = () => { _runsSort = b.dataset.sort; renderRunsTable(); };
  });
  // Re-wire EVERY family on every render: innerHTML discarded their handlers.
  srcs.forEach(s => { try { s.wire(box); } catch (e) { console.error("runs wire failed", e); } });
  // …then collapse each row's actions behind ⋯ on a phone. AFTER the wiring,
  // because it counts the buttons the families just put there — and a row with
  // none gets no ⋯ at all.
  if (window.runsActsSync) runsActsSync(box);
}

/** The Model cell — name plus licence, used by both families so the column
 *  reads consistently no matter which ledger a row came from. */
function runsModelCell(model, license) {
  const lic = license
    ? `<span class="lic" title="${license}">${license.replace("-2.0", "")}</span>` : "";
  return `<td class="runsModel"><b>${model}</b>${lic ? "<br>" + lic : ""}</td>`;
}
