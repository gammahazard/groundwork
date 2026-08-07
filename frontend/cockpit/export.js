"use strict";
/* Export: turn a run into something you can deploy.
 *
 * ITS OWN TAB. It began as a panel under Results, on the reasoning that picking
 * what to export is the same act as reading which run won. In practice it is a
 * DESTINATION — you arrive meaning to ship something — and it was at the bottom
 * of the longest page in the app. Being its own tab also means it fetches its
 * own run list instead of scraping the ledger's table, which is what made it
 * claim "no runs" whenever Results had not been opened first.
 *
 * THE CAVEAT IS SHOWN AT THE MOMENT OF CHOOSING, not in a footnote. Two of these
 * formats cost something this repo has measured — NCNN is one object worse, and a
 * TensorRT engine only loads on the model of card it was built on — and both are
 * facts you want BEFORE a five-minute job, not after. The server sends them with
 * the format list so there is one copy of each.
 */

const _xEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

let XFORMATS = [];          // formats for the run currently selected
let XUNPROBED = [];         // machines with no probe on record
let _xPoll = null;

/* EXPORT ASKS FOR ITS OWN RUNS.
 *
 * It used to read them out of the Results table's DOM, on the reasoning that
 * whatever the table shows is exactly what can be exported. That held only while
 * Export lived on the Results tab — as its own tab it can be opened FIRST, and
 * then #runsTable has never been filled and the panel says "No runs yet" about a
 * machine with 117 of them. The DOM was never the source of truth; the ledgers
 * are, so ask them.
 *
 * BOTH ledgers, because a run name is a run name: /api/runs is the yolo ledger
 * and /api/lab/runs the challenger one, and the server decides which family a
 * name belongs to (registry.for_run) and therefore which formats it can produce.
 * Listing challengers rather than hiding them is deliberate — selecting one
 * explains why it cannot export yet, which is more use than an absence. */
async function _xRuns() {
  const names = [];
  const push = rows => (rows || []).forEach(r => {
    const n = r && (r.run || r.name);
    if (n && !names.includes(n)) names.push(n);
  });
  const [yolo, alt] = await Promise.all([
    api("/api/runs").catch(() => []),
    api("/api/lab/runs").catch(() => []),
  ]);
  push(Array.isArray(yolo) ? yolo : yolo.runs);
  push(Array.isArray(alt) ? alt : alt.runs);
  return names;
}

/* EVERY CARD ON EVERY MACHINE, not just this one — and that is the point of the
 * control rather than a nicety. A TensorRT engine only loads on the model of GPU
 * it was compiled on, so "which card" IS the question, and two of this fleet's
 * three cards are on the worker. The picker used to be a hardcoded
 * <option>0</option><option>1</option>, which offered a second card on a
 * one-card laptop and could not reach the worker's at all.
 *
 * Choosing a remote card means the build HAPPENS there: the weights go up, the
 * engine is compiled on that GPU and the artifact comes back. Anything else
 * would produce a file that cannot load on the machine it was asked for.
 *
 * MEASURED, NOT ENUMERATED. These come from each machine's last probe. Counting
 * cards calls into the driver, and under WSL2 that crosses /dev/dxg where a
 * blocked thread cannot be killed — never from a page load. */
async function _xLoadCards() {
  const sel = $("#xCard");
  if (!sel) return 0;
  let cards = [], unprobed = [];
  try {
    const d = await apiOrThrow("/api/export/cards");
    cards = d.cards || []; unprobed = d.unprobed || [];
  } catch { /* fall through to the empty case */ }
  /* SHORT LABELS, because a <select> is as wide as its widest option and there
   * is no wrapping inside a native one. The full string — "Groundwork Trainer ·
   * card 0 · NVIDIA GeForce RTX 5070 Ti · 16 GB" — stretched the whole page
   * sideways the moment TensorRT was picked. "NVIDIA GeForce " is noise on
   * every card here, and the machine's first word identifies it. The full name
   * stays in the title attribute for anyone who wants it. */
  const short = n => String(n || "").replace(/^NVIDIA\s+GeForce\s+/i, "")
                                    .replace(/\s+Laptop GPU$/i, " (laptop)");
  sel.innerHTML = cards.length
    ? cards.map(c => {
        const who = c.local ? "This machine"
                            : String(c.machine_name || c.machine).split(/\s+/)[0];
        return `<option value="${_xEsc(c.machine)}:${c.index}" `
          + `title="${_xEsc(c.machine_name)} · card ${c.index} · ${_xEsc(c.name)}`
          + `${c.vram_gb ? " · " + c.vram_gb + " GB" : ""}">`
          + `${_xEsc(who)} · ${_xEsc(short(c.name))}`
          + `${c.vram_gb ? " · " + c.vram_gb + "GB" : ""}</option>`;
      }).join("")
    : `<option value="">no probed cards</option>`;
  sel.disabled = !cards.length;
  XUNPROBED = unprobed;
  return cards.length;
}

/* The tab's entry point: fill the run list, the cards and the formats, then the
 * artifacts already on disk. Order matters only in that formats depend on which
 * run is selected, so the run list is built first. */
async function loadExport() {
  const sel = $("#xRun");
  if (!sel) return;
  const runs = await _xRuns();
  // The empty state HIDES the controls instead of replacing the panel's
  // innerHTML — the old version destroyed #xRun and friends, and the null
  // guard above then returned forever: opening Export before the first run
  // bricked the tab until a full page reload.
  const empty = $("#xEmpty");
  const ctrls = $("#xControls");
  if (!runs.length) {
    if (empty) empty.hidden = false;
    if (ctrls) ctrls.hidden = true;
    return;
  }
  if (empty) empty.hidden = true;
  if (ctrls) ctrls.hidden = false;
  const keep = sel.value;
  sel.innerHTML = runs.map(r => `<option>${_xEsc(r)}</option>`).join("");
  if (keep && runs.includes(keep)) sel.value = keep;
  await _xLoadCards();
  await _xLoadFormats();
  _xLoadArtifacts();
  _xResume();
}

/* AN EXPORT OUTLIVES THE PAGE, so the page has to go and ask.
 *
 * The job is a thread in the SERVER keyed off module state — leaving the tab,
 * switching project or closing the browser does not stop it, and a TensorRT
 * build is long enough that people will. What was missing is the other half:
 * the status only ever rendered from the poller, and the poller only ever
 * started from the Go button. So coming back mid-build showed an idle panel
 * with a live Go button over a job that was still running, and pressing it got
 * a bare 409. The work was never at risk; the page just lied about it.
 *
 * It is also why the clock counts from the server's `started` rather than from
 * when this tab happened to open. */
async function _xResume() {
  let s;
  try { s = await apiOrThrow("/api/export"); } catch { return; }
  if (!s || !s.status || s.status === "idle") return;
  _xRenderStatus(s);
  if (s.status === "running") {
    $("#xGo").disabled = true;
    _xStartPolling();
  }
}

async function _xLoadFormats() {
  const run = $("#xRun").value;
  const fs = $("#xFormat"), note = $("#xNote");
  if (!run) return;
  try {
    const d = await apiOrThrow(`/api/export/formats?run=${encodeURIComponent(run)}`);
    XFORMATS = d.formats || [];
    fs.innerHTML = XFORMATS.map(f =>
      `<option value="${_xEsc(f.key)}">${_xEsc(f.label)}</option>`).join("");
    // The size is the RUN's, not a choice: exporting at a different resolution
    // is a different model, and one the holdout score no longer describes.
    $("#xSize").textContent = `${d.imgsz}px · ${d.family}`;
    /* A FAMILY WITH NO EXPORTER SAYS SO. Selecting a DEIM run used to leave an
     * empty dropdown and a live Go button — a page that looks broken for work
     * that simply is not done. Disable the control and quote the server. */
    const none = !XFORMATS.length;
    fs.disabled = none; $("#xGo").disabled = none;
    if (none) {
      note.innerHTML = `<span class="xCaveat">${_xEsc(d.not_wired
        || "no export formats for this model family yet")}</span>`;
      note.hidden = false;
      $("#xCardWrap").hidden = true;
      return;
    }
    _xShowCaveat();
  } catch (e) {
    note.textContent = `Could not read formats: ${e.message || e}`;
  }
}

function _xShowCaveat() {
  const key = $("#xFormat").value;
  const f = XFORMATS.find(x => x.key === key);
  const note = $("#xNote");
  if (!f) { note.hidden = true; return; }
  const bits = [`<span class="xWhat">${_xEsc(f.what)}</span>`];
  if (f.caveat) bits.push(`<span class="xCaveat">${_xEsc(f.caveat)}</span>`);
  note.innerHTML = bits.join("");
  note.hidden = false;
  // TensorRT is the only one that compiles on a card, so it is the only one
  // that offers a card choice — and the only one refused while training.
  const cardWrap = $("#xCardWrap");
  if (cardWrap) cardWrap.hidden = !f.needs_gpu;
  /* AND IT NEEDS A REAL CARD TO NAME. With no probe on record the select is
   * empty, and `+"" || 0` would have quietly compiled for card 0 — a guess, on
   * the one format where guessing wrong produces an engine that will not load.
   * Ask for the probe instead. */
  const known = !$("#xCard").disabled;
  const blocked = f.needs_gpu && !known;
  $("#xGo").disabled = blocked;
  if (blocked) note.innerHTML += `<span class="xCaveat">No machine here has `
    + `been probed, so no cards are known — and TensorRT has to name the one it `
    + `compiles for. Probe ${XUNPROBED.length ? _xEsc(XUNPROBED.join(", "))
        : "a machine"} on the Machines tab first.</span>`;
}

async function _xLoadArtifacts() {
  const box = $("#xArtifacts");
  if (!box) return;
  try {
    const d = await apiOrThrow("/api/export/artifacts");
    const a = d.artifacts || [];
    box.innerHTML = a.length
      /* THE BADGE SAYS WHAT IT IS. Half these names do not: _openvino_model and
       * _ncnn_model are extensionless directories, and .mlpackage / .engine mean
       * nothing unless you already know them. The server derives the label from
       * the same ext table the formats come from, so the badge and the dropdown
       * can never disagree. */
      ? a.map(x => `<div class="xRow">
          <a class="xName" href="${_xEsc(x.url)}" download>${_xEsc(x.name)}</a>
          <span class="xFmt">${_xEsc(x.format || "")}</span>
          <span class="xSize gwNum">${x.size_mb} MB</span></div>`).join("")
      : `<p class="hint">Nothing exported yet.</p>`;
  } catch { /* the list is a convenience; never break the panel over it */ }
}

function _xRenderStatus(s) {
  const el = $("#xStatus");
  if (!el) return;
  if (s.status === "running") {
    /* A SPINNER AND A CLOCK, not just a sentence. A CoreML or TensorRT build is
     * minutes of silence, and static text reads as a hung button — the same
     * reason the Machines probe says "Asking…". .uiSlow is the app's existing
     * spinner (14-ui.css), so this looks like every other slow thing here and
     * respects prefers-reduced-motion for free.
     *
     * The elapsed seconds come from the server's own `started`, so they survive
     * a refresh: reopening the tab mid-build shows a job 90s in, not one that
     * has apparently just begun. */
    const secs = s.started ? Math.max(0, Math.round(Date.now() / 1000 - s.started)) : 0;
    const where = s.card != null ? " on the chosen card" : "";
    el.innerHTML = `<span class="xRun uiSlow">Converting ${_xEsc(s.run)} to `
      + `${_xEsc(s.format)}${where}… <span class="gwNum">${secs}s</span>`
      + `<span class="xHint"> — CoreML and TensorRT take minutes; you can leave `
      + `this tab.</span></span>`;
  } else if (s.status === "done" && s.path) {
    el.innerHTML = `<span class="xOk">Exported ${_xEsc(s.run)} → `
      + `${_xEsc(s.format)} · ${s.size_mb ?? "?"} MB</span>`;
  } else if (s.status === "error") {
    el.innerHTML = `<span class="xErr">${_xEsc(s.error || "export failed")}</span>`;
  } else {
    el.textContent = "";
  }
}

function _xStartPolling() {
  clearInterval(_xPoll);
  // 2s, not 3: this drives a visible clock now, and a stalled digit is exactly
  // the "is it stuck?" impression the spinner exists to remove.
  _xPoll = setInterval(async () => {
    let s;
    try { s = await apiOrThrow("/api/export"); } catch { return; }
    _xRenderStatus(s);
    if (s.status !== "running") {
      clearInterval(_xPoll); _xPoll = null;
      $("#xGo").disabled = false;
      _xLoadArtifacts();
    }
  }, 2000);
}

(() => {
  const go = $("#xGo");
  if (!go) return;
  $("#xRun").onchange = _xLoadFormats;
  $("#xFormat").onchange = _xShowCaveat;
  go.onclick = async () => {
    const body = {run: $("#xRun").value, format: $("#xFormat").value};
    const cw = $("#xCardWrap");
    if (cw && !cw.hidden) {
      // "<machine>:<index>" — the machine decides WHERE the engine is built,
      // because an engine is only valid on the card it was built on.
      const [mk, ix] = String($("#xCard").value || "").split(":");
      if (mk) body.machine = mk;
      body.card = +ix || 0;
    }
    go.disabled = true;
    $("#xStatus").innerHTML = `<span class="xRun">Starting…</span>`;
    try {
      const r = await fetch("/api/export", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)});
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "could not start");
      _xStartPolling();
    } catch (e) {
      $("#xStatus").innerHTML = `<span class="xErr">${_xEsc(e.message || e)}</span>`;
      go.disabled = false;
    }
  };
})();

window.loadExport = loadExport;
