"use strict";
/* Dashboard: pure status — the serving hero card, data tiles, live Activity
 * (every job on either machine, yolo or challenger) and the champion-vs-
 * challenger Scoreboard. No controls: doing things lives in the YOLO and
 * RTMDet tabs (yolo.js / lab.js). Loads LAST of the cockpit/ modules — the
 * bootstrap at the bottom kicks everything off. */

/* Hold this tab's shape BEFORE its data is asked for.
 *
 * projdash was the one tab with no loading state at all: every other panel goes
 * through loadInto or uiHold, and this one shipped literal "–" and "…"
 * characters in the markup and left them there until the fetch landed. So the
 * project's own front page — the most-opened screen here — was the only place
 * that could not tell you whether it was working, empty, or broken.
 *
 * TWO SHAPES, because the panel has two kinds of placeholder:
 *
 *   containers  Activity / Scoreboard / Odometer / the MAE chart are filled
 *               with innerHTML, so uiHold's skeletons drop straight in. Their
 *               "…" had to come OUT of index.html first — uiHold deliberately
 *               refuses a container that already has content, so the ellipsis
 *               was blocking the very thing meant to replace it.
 *
 *   numbers     the tiles and the serving hero are static markup written to BY
 *               ID. Replacing them with skeletons would delete the ids that
 *               loadState writes to, so those shimmer in place instead —
 *               transparent text over a sheen, keeping their own width.
 */
function projdashHold(){
  uiHold("#activityList", "row", 2);
  uiHold("#scoreboard", "row", 5);
  uiHold("#odometer", "row", 5);
  uiHold("#maeChart", "chart", 1);
  // NAMED, not matched structurally. `.servedMeta b` caught the two bold fields
  // and missed conf/iou, which are plain spans — so those showed a bare "–"
  // beside four shimmering neighbours. It would also have caught #servingNote,
  // which is EMPTY at rest and would shimmer as a stray block forever.
  // This list is exactly what loadState() writes, which is the only thing that
  // makes "pending" true.
  PROJDASH_FIELDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add("uiPending");
  });
}

/* Every id loadState() fills in the hero and the tiles. Kept beside the hold so
 * a field added to one and not the other is visible in a single screen. */
const PROJDASH_FIELDS = ["stServed", "stConf", "stIou", "stServedMae", "stExact",
                         "stTrays", "stTest", "stObjects", "stFix"];

/* Clear it in ONE place, at the end of loadState, rather than at each of the
 * forty sites that write a number — those write .textContent and would each
 * have to remember. A pending marker that outlives its data is worse than none:
 * it says "still loading" over a number that has already arrived. */
function projdashRelease(){
  document.querySelectorAll("#projdash .uiPending")
    .forEach(el => el.classList.remove("uiPending"));
}

async function loadState(){
  /* NO PROJECT OPEN, NO PROJECT DATA. /api/state falls back to the default
   * project when nobody says which — correct for the API, wrong here: the
   * Dashboard would fetch the first project's serving model and image counts and render
   * them under a heading that claims to describe the machine. Hiding the panel
   * in CSS is not enough; the request itself must not happen, or the numbers
   * exist in the page and only a stylesheet stands between them and a reader.
   * Every other caller (grids, editor, yolo, history, upload) is inside a
   * project by construction, so this guard is a no-op for them. */
  if (!CURRENT_PROJECT) return;
  const s = await api("/api/state");
  window.IS_LAB = !!s.lab;      // grids render read-only on the lab mirror
  window._lastState = s;
  // Lab machine announces itself loudly: this UI trains; curation lives at home.
  if (s.lab) applyLabIdentity(s); else applyHqIdentity(s);
  if (!s.lab){    // the lab's hero is the Trainer card (identity.js owns it)
    const sv = s.served || {};
    // The engine switch changes WHICH model answers a photo. The name and the
    // "the bot and the iPhone app count with this model" line used to describe
    // the pinned yolo run either way, so running the challenger made the whole
    // hero card confidently wrong. Say the engine, and mark it when it is not
    // the champion — with a word, not just a colour.
    const exp = !!sv.experimental;
    $("#stServed").textContent = sv.name
      ? `${sv.name} @${sv.imgsz} (${exp ? `${sv.engine} — EXPERIMENTAL`
                                        : sv.pinned ? "pinned" : "newest"})`
      : "none";
    $("#stServedMae").textContent = fmtScore(sv.mae);
    $("#stExact").textContent = sv.exact ? `${sv.exact[0]}/${sv.exact[1]}` : "–";
    $("#stConf").textContent = s.conf; $("#stIou").textContent = s.iou;
    const note = $("#servingNote");
    if (note) note.textContent = exp
      ? ` — ⚠ ${sv.engine} is serving: the bot and this cockpit `
        + `count with it. Exported files are unaffected — they are copies.`
      : " — the bot and this cockpit count with this model. Exported files "
        + "are separate copies and keep working unchanged.";
    const hero = $("#projdash .dashCard.hero");
    if (hero) hero.classList.toggle("experimental", exp);
  }
  const o = s.odometer || {};
  const ch = o.challenger || {};
  const f = o.fleet || {};   // both machines; absent on the lab, or on old servers
  // Challenger runs are most of the Trainer's mileage but were invisible here
  // (the odometer only ever read the yolo ledger). Show the split when this
  // machine has actually trained challengers, and the yolo-only view when not.
  const hasCh = (ch.runs || 0) > 0;
  $("#odometer").innerHTML =
    `<div class="sbRow"><span>model generations</span><span class="sbMae">${
      hasCh ? `${o.total_runs} <span class="mMeta">(${o.runs} yolov8n + ${ch.runs} other)</span>`
            : (o.runs ?? "–")}</span></div>`
    /* GPU TIME ACROSS BOTH MACHINES, not "this machine" — which it never
     * honestly was. Measured 2026-08-02: HQ showed 23.0h under that label while
     * the worker had burned 72.6h, and 11.9h of HQ's own figure was worker-trained
     * yolo it had adopted. The fleet number is what the owner actually wants
     * now that HQ is becoming the only control plane.
     *
     * The breakdown says where the hours went, because "79.3h" alone hides that
     * ~56h of it is the challenger programme on the 3090. And when the worker
     * cannot be reached the total is marked INCOMPLETE rather than quietly
     * shrinking to 23h — a number that silently drops its largest component is
     * the same failure as a cached reading passed off as live. */
    + (f.hours != null
        ? `<div class="sbRow"><span>GPU time, all machines${
             f.complete ? "" : " <span class='mMeta'>(partial)</span>"}</span>`
          + `<span class="sbMae" title="${
               f.complete
                 ? `yolov8n ${o.gpu_hours}h + other models ${ch.hours}h here + ${f.remote_hours}h on other machines. Yolo hours are NOT added across machines: every lab run is adopted into this ledger, so they are already counted once.`
                 : "Not every machine could be reached, so some hours are missing from this total."
             }">${f.hours}h${
               f.complete && f.remote_hours
                 ? ` <span class="mMeta">(${o.total_hours} here + ${f.remote_hours} elsewhere)</span>` : ""
             }</span></div>`
        : `<div class="sbRow"><span>GPU time, this machine</span><span class="sbMae">${
            hasCh ? `${o.total_hours}h <span class="mMeta">(${o.gpu_hours} + ${ch.hours})</span>`
                  : `${o.gpu_hours ?? "–"}h`}</span></div>`)
    + `<div class="sbRow"><span>epochs trained</span><span class="sbMae">${
        ((hasCh ? o.total_epochs : o.epochs) ?? 0).toLocaleString()}</span></div>`
    + (hasCh ? Object.entries(ch.by_arch || {}).map(([a, h]) =>
        `<div class="sbRow"><span class="mMeta">&nbsp;&nbsp;${a}</span>`
        + `<span class="sbMae mMeta">${h}h</span></div>`).join("") : "")
    /* The tooltip must name the SAME hours the kWh was computed from. The server
     * charges the fleet total when it has one, so quoting this machine's 23.0h
     * beside a figure derived from 79.3h would read as an arithmetic error. */
    + (o.power ? `<div class="sbRow" title="ESTIMATED, not metered: ${o.power.watts}W whole-system while training (PSU losses included) x ${(f.complete ? f.hours : o.total_hours) ?? o.gpu_hours}h x $${o.power.rate}/kWh${f.complete ? " across all machines" : ""}. Set GW_KWH_RATE / GW_TRAIN_WATTS to correct."><span>electricity (est.)</span>`
        + `<span class="sbMae">$${o.power.usd} <span class="mMeta">${o.power.kwh} kWh</span></span></div>` : "")
    + `<div class="sbRow"><span>largest count on file</span><span class="sbMae">${o.biggest ?? "–"}</span></div>`;
  $("#stTrays").textContent = s.raw;
  $("#stTest").textContent = s.testset;
  // Thousands-separated, like every other big number on the page — it sits
  // directly under the Overview's "14,997 labelled objects" and the two were
  // formatted differently AND counting different things (that one is every
  // project's train+test, this one is this project's TRAINING set). The label
  // in index.html now says which; the formatting stops the pair looking like a
  // typo before the reader gets to the words.
  $("#stObjects").textContent = (s.objects ?? 0).toLocaleString();
  $("#stFix").textContent = s.needs_fix;
  /* A BADGE HANGS OFF A TAB, AND TABS COME AND GO — so writing one must not be
   * able to take the rest of this function down with it. A badge that lived
   * inside a since-removed button once made this line throw, and everything
   * below it stopped running — including projdashRelease()
   * below, which is what clears the skeleton. The Overview shimmered forever and
   * the console said only "cannot set textContent of null".
   *
   * The numbers above are ids this page owns and always has. These five are
   * decorations on OPTIONAL tabs, so they are written defensively; a missing one
   * is a tab that is not on screen, not an error. */
  const badge = (sel, v) => { const e = $(sel); if (e) e.textContent = v; };
  badge("#fixBadge", s.needs_fix || "");
  badge("#trainBadge", s.raw || "");
  badge("#testBadge", s.testset || "");
  // Every number this function writes is now on screen, so drop the shimmer.
  // renderDashboardExtras below is fire-and-forget and fills the CONTAINERS,
  // whose skeletons it clears itself by writing innerHTML.
  projdashRelease();
  renderDashboardExtras(s);     // fire-and-forget: activity + scoreboard
}

/* Activity: everything training right now, on either machine, either model
   family. Scoreboard: the licensing race — pinned champion vs best challenger. */
/* WHICH CARD, when the job actually told us.
 *
 * Two GPU jobs on one machine read identically without it — that is not a
 * cosmetic gap: on 2026-08-02 a retrain landed on card 0 beside a challenger on
 * card 1 and killed it, and every screen described both as simply "training".
 *
 * NOT DERIVED, and never defaulted to 0. altmodels/gpu.py writes the card into
 * gpu_holder.json from CUDA_VISIBLE_DEVICES precisely so the cockpit does not
 * have to infer it ("WHICH CARD, recorded rather than inferred"), and a second
 * table that guesses is how the run-name/arch label went wrong before. A job
 * that did not record one gets no card shown, which is honest — the yolo retrain
 * does not record one yet, so its rows stay bare until retrain_state carries it.
 */
const cardTag = c => (c === 0 || c > 0) ? ` · <b>card ${c}</b>` : "";

async function renderDashboardExtras(s){
  const acts = [];
  /* WHAT WE DO NOT KNOW IS NOT "NOTHING". Every probe below used to swallow its
   * failure into "no row", and no rows rendered as "😴 all quiet — nothing
   * training". So an unreachable Trainer, or a lab_proxy cache that has not
   * warmed yet, positively asserted that nothing was training — the exact claim
   * it was least entitled to make. Both happened repeatedly on 2026-08-02: the
   * worker spent 15 minutes off the network while the dashboard looked serene.
   *
   * Three states now, not two: rows / still asking / genuinely idle. */
  const unanswered = [];
  if (s.retrain && s.retrain.status === "running")
    acts.push(`🔁 yolo training on this machine — ${s.retrain.step}`
      + cardTag(s.retrain.card));
  // GUARDED, not merely caught. The comment below already knew the lab 400s this
  // endpoint and the try made it harmless — but "harmless" was a failed request
  // per poll cycle in the worker's own console, which is the noise that hides a
  // real one. `s.lab` is the server's own answer, already in the payload, so
  // there is no window.IS_LAB race to lose.
  //
  // ONLY THIS BLOCK. The challenger block below must still run on the lab: it
  // reads /api/lab/status, which is the machine asking about ITSELF, and it is
  // the row that says what the Trainer is training.
  if (!s.lab) {
    try {   // yolo on a worker — HQ watching it
      const l = await api("/api/lab_status");
      // no_workers is a REAL answer: a single-machine install has nobody to
      // ask, and treating that as "still checking" left the activity feed
      // saying ⏳ forever. ok:false without it is lab_proxy's cold cache — "I
      // have not heard from the worker yet" — which is genuinely unanswered.
      if (l && l.no_workers) { /* nothing to wait for */ }
      else if (!l || l.ok === false) unanswered.push("the worker");
      const ph = labCooking(l);
      if (ph === "running")
        acts.push(`🧪 yolo <b>${l.next_name}</b> on Trainer — `
          + (l.epoch != null ? `epoch ${l.epoch}/${l.total}` : (l.step || "training"))
          + (l.eta_s ? ` · ~${Math.max(1, Math.round(l.eta_s/60))} min` : "")
          + cardTag(l.card));
      else if (ph === "adopting")
        acts.push(`🧪 yolo <b>${l.next_name}</b> — verdict in, adopting…`);
    } catch (e){ unanswered.push("the worker"); }
  }
  try {   // challenger, wherever it runs
    const c = await api("/api/lab/status");
    // ONE ROW PER LIVE CHALLENGER. This read `c.active` — a single run — so with
    // two challengers training the second was simply absent from Activity, on
    // the one screen whose job is "what is happening right now". Third place the
    // same assumption was baked in today (train.js and run_log.js were the
    // others); the server sends `actives` now, and `active` is only its first
    // element for older payloads.
    //
    // Name the run, not a hardcoded architecture: this said "RTMDet" for every
    // challenger, so a DEIMv2 run was announced as a deprecated family. The
    // crumb has no arch field (altmodels/gpu.py writes {pid,what,started}), so
    // the run name is the honest label.
    const chals = c.actives || (c.active ? [c.active] : []);
    for (const a of chals) acts.push(`🧪 <b>${a.run}</b> — `
      + (a.epoch != null ? `epoch ${a.epoch}/${a.total ?? "?"}` : "training")
      + (a.eta_s ? ` · ~${Math.max(1, Math.round(a.eta_s/60))} min` : "")
      + (a.remote ? " · on Trainer" : "")
      + cardTag(a.card));
    /* AND THE ONES BETWEEN TRAINING AND THE LEDGER. `actives` only lists runs
     * holding a GPU, so a challenger vanished from here the moment training
     * ended and stayed invisible through scoring (count_eval over the holdout,
     * ~5 min) and adoption (the adopt job, a 5-minute timer). Asked as "how come I
     * don't see the deim run in the ledger, is it still adopting" — the UI had
     * no way to say. yolo has had an "adopting…" row all along; this is the
     * same courtesy for challengers. */
    for (const fin of (c.finishing || []))
      acts.push(`🧪 <b>${fin.run}</b> — `
        + (fin.phase === "adopting" ? "verdict in, adopting…"
         : fin.phase === "queued"
             ? "trained — waiting for a free card to score on"
             : "training done, scoring on the holdout…"));
  } catch (e){ unanswered.push("other-model status"); }
  // Mirror health (HQ only) — ONLY WHEN THERE IS SOMETHING TO DO ABOUT IT.
  //
  // This used to warn on "last sync >= 15 min ago" as well. That reason is gone
  // (owner, 2026-08-02): /api/train now rsyncs the project to the target BEFORE
  // every remote run and REFUSES if the sync fails, so a stale mirror can no
  // longer reach a training run — the thing the warning existed to prevent is
  // structurally impossible. Left in, it fired on every quiet period and became
  // the noise that hides a real row.
  //
  // A FAILED sync still shows, because that is different: it means the next
  // Train press will be refused, backups are not running either, and when
  // the VPN wants a re-auth the fix is a link otherwise buried in the journal.
  // Today proved that half is worth keeping — the worker spent 15 minutes
  // unreachable while everything on screen looked normal.
  if (!s.lab && s.last_sync){
    const mins = Math.round((Date.now() / 1000 - s.last_sync) / 60);
    const m = s.mirror || {};
    if (m.ok === false || m.auth_url){
      let row = `⚠️ <b>dataset mirror is failing</b> — last successful sync `
        + `${mins} min ago. Training will be refused rather than run on data it `
        + `cannot confirm, and backups are not running.`;
      // The whole point: when the VPN wants a re-auth, the fix is a link
      // buried in the journal. Put it on screen as a tap target instead.
      if (m.auth_url)
        row += ` <a class="authFix" href="${m.auth_url}" target="_blank"`
          + ` rel="noopener">🔑 Re-authenticate the VPN →</a>`;
      if (m.error)
        row += `<div class="mMeta syncErr">${m.error.replace(/</g, "&lt;")}</div>`;
      acts.unshift(row);
    }
  }
  // "all quiet" is a CLAIM, and it is only made once every probe has answered.
  $("#activityList").innerHTML = acts.length
    ? acts.map(x => `<div class="actRow">${x}</div>`).join("")
    : unanswered.length
      ? `<div class="actRow actWaiting">⏳ checking ${unanswered.join(" and ")}…</div>`
      : "<div class='actRow'>😴 all quiet — nothing training</div>";
  try {   // the leaderboard: every architecture ever tested, best MAE each
    const [lr, hist] = await Promise.all([api("/api/lab/runs"), api("/api/runs")]);
    const rows = [];
    // ONE definition of "what counts as an attempt", used by the leaderboard
    // count AND the chart below. They used to disagree: the yolo column counted
    // ledger entries (one per training) while the challenger column counted
    // every scored ROW, so re-scorings of one run inflated it — 39 shown against
    // 16 real DEIMv2 trainings, printed side by side as if comparable.
    // reexam-/stg2- are the same checkpoints re-scored on a later holdout.
    const isRescore = (n) => /^(reexam|stg2)-/.test(n);
    // -last/-best/-best_stg1/-best_stg2/-bilinear/-epN are all EXAMS of one
    // training, not separate attempts. The old pattern knew only -last and -epN.
    const basename = (n) => n.replace(/-(last|best_stg1|best_stg2|best|bilinear|ep\d+)$/, "");
    const ymaes = (hist.runs || hist || []).map(r => r.mae).filter(m => m != null);
    if (ymaes.length)
      rows.push({arch: "yolov8n", lic: "AGPL", mae: Math.min(...ymaes), n: ymaes.length});
    const byArch = {};
    for (const r of lr.runs || []){
      if (r.mae == null || isRescore(r.run)) continue;
      const a = r.arch || "unnamed model";
      byArch[a] = byArch[a] || {arch: a, lic: r.license || "Apache-2.0",
                                mae: r.mae, seen: new Set()};
      byArch[a].seen.add(basename(r.run));      // distinct trainings, not exams
      if (r.mae < byArch[a].mae) byArch[a].mae = r.mae;
    }
    rows.push(...Object.values(byArch).map(v => ({...v, n: v.seen.size})));
    rows.sort((x, y) => x.mae - y.mae);
    // The families race chart: MAE per attempt, one line per architecture.
    // EXCLUDED, NOT CLIPPED. This used to do Math.min(mae, 0.5), which draws a
    // 22.03 run as a flat line at exactly 0.5 — a number the model never scored,
    // rendered indistinguishably from one it might have. Runs above the cut are
    // dropped from the series instead, and the title says how many, so the chart
    // never shows a value that did not happen.
    if (window.svgMultiChart){
      const CLAMP = 0.5;
      let dropped = 0;
      const fams = {};
      for (const r of lr.runs || []){
        if (r.mae == null || isRescore(r.run)) continue;
        const a = r.arch || "unnamed model";
        const b = basename(r.run);
        fams[a] = fams[a] || {};
        const cur = fams[a][b];
        if (!cur || r.mae < cur.mae)
          fams[a][b] = {mae: r.mae, created: r.created || 0};
      }
      // ONE palette, defined beside the chart helpers in tabs/history.js — a
      // second copy eventually gives one model two colours on two pages, which
      // reads as two models.
      const pal = a => (window.modelColor ? modelColor(a) : "#7a8699");
      const series = [];
      // /api/runs is newest-first — flip to chronological BEFORE taking the
      // recent 8, else the line shows the oldest runs, backwards (the bug
      // that hid run 43's 0.0)
      const ymaes2 = (hist.runs || hist || [])
        .filter(r => r.mae != null).reverse().slice(-8)
        .filter(r => { if (r.mae > CLAMP){ dropped++; return false; } return true; })
        .map(r => r.mae);
      if (ymaes2.length)
        series.push({label: "yolov8n", color: pal("yolov8n"),
                     pts: ymaes2.map((y, i) => ({i, y}))});
      let maxN = ymaes2.length;
      for (const [a, runs] of Object.entries(fams)){
        const pts = Object.values(runs)
          .sort((x, y) => x.created - y.created)
          .filter(r => { if (r.mae > CLAMP){ dropped++; return false; } return true; })
          .map((r, i) => ({i, y: r.mae}));
        if (pts.length){
          series.push({label: a, color: pal(a), pts});
          maxN = Math.max(maxN, pts.length);
        }
      }
      const el = $("#maeChart");
      if (el && series.length)
        el.innerHTML = svgMultiChart({series,
          xlabels: Array.from({length: maxN}, (_, i) => String(i + 1)),
          title: `holdout MAE per attempt${dropped
            ? ` — ${dropped} run${dropped > 1 ? "s" : ""} above ${CLAMP} not shown`
            : ""}`,
          yfmt: v => v.toFixed(2), markPerfect: true});
    }
    // "trainings", NOT "runs". This count collapses every exam of one training
    // (-last/-best/-epN, and the reexam-/stg2- re-scores) back into the
    // training itself, so it is deliberately smaller than the Overview panel's
    // "every run on this machine" directly above. Two panels on one screen
    // saying "39 runs" and "16 runs" about deimv2-n is a contradiction the
    // reader has to reconcile; two different words is one they can just read.
    $("#scoreboard").innerHTML = rows.length
      ? rows.map(r =>
          `<div class="sbRow"><span>${r.arch}<span class="lic">${r.lic}</span>`
          + ` <span class="mMeta">· ${r.n} training${r.n === 1 ? "" : "s"}</span></span>`
          + `<span class="sbMae">${fmtScore(r.mae)}</span></div>`).join("")
      : "<div class='sbRow'><span class='mMeta'>no scored models yet</span></div>";
  } catch (e){ /* keep last render */ }
}

/* ---------- bootstrap ---------- */
/* ?project=<slug> is restored FIRST, so the dashboard's very first paint is
 * about the right project rather than the default one followed by a flicker.
 * restoreProjectFromURL is async and sets PROJECT before the loads it awaits. */
(async () => {
  /* NOTHING FETCHES UNTIL THERE IS A SESSION. Every call below is gated
   * server-side, so booting before sign-in means a dozen 401s and a page of
   * failure cards behind the login form — and the failure cards would be the
   * first thing seen after signing in, because they were rendered before the
   * session existed. cockpit/auth.js resolves this once /api/me answers; it
   * never rejects, because being signed out is a normal state and not an
   * error, so this simply waits at the door. */
  if (window.whenSignedIn) await whenSignedIn();
  if (window.initNav) initNav();
  // Only ?project= opens one now (see projects.js) — so with no param the
  // landing page IS the destination, whether there are one, two or ten
  // projects. index.html ships with Dashboard active; say otherwise explicitly
  // rather than leaving whatever tab the markup marked.
  const opened = window.restoreProjectFromURL ? await restoreProjectFromURL() : false;
  if (!opened) showTab("projects");
  loadState();    // no-ops until a project is open
  // loadYolo() primes the retrain/models panels, which live inside a project.
  // Skipped outside one: it hits project-scoped endpoints and would prime them
  // with the default project's runs on a page that is deliberately showing none.
  if (opened) loadYolo();
})();
setInterval(() => {
  // On the lab, loadState always runs: the header banner's sync-age line is
  // visible on every tab and would otherwise freeze at its boot value.
  if ($("#projdash").classList.contains("active") || window.IS_LAB) loadState();
  // BOTH halves: the retrain status lives on Train, the model list on Results.
  if (onRunsTabs()) loadYolo();
  // …and what each GPU is doing. Same heartbeat rather than a second timer:
  // two intervals polling the same two endpoints is twice the traffic and two
  // things to reason about when one of them lags.
  // The card panel, the live logs and the picture panels are all on TRAIN,
  // so these stay scoped to it — polling them from Results would be work
  // whose output nobody can see.
  if ($("#runs").classList.contains("active")) {
    if (window.trainBusyTick) trainBusyTick();
    // Same heartbeat, not a second timer: the log panel and the card
    // panel read overlapping state and two timers would race on it.
    // AFTER the log tick, not beside it: runLogTick publishes LIVE_RUNS and
    // run_vis reads it, so running them concurrently would paint pictures
    // for the PREVIOUS tick's runs.
    if (window.runLogTick) runLogTick().then(
      () => window.runVisTick && runVisTick());
  }
}, 8000);
