"use strict";
/* Machine identity: which cockpit is this page? The lab (🧪 Groundwork Trainer,
 * ember theme, read-only data, Sync-now) vs home (Groundwork HQ: Trainer-first
 * train area, lab-watch card, ghost row). dashboard.js's loadState() calls
 * applyLabIdentity / applyHqIdentity on every state poll. */

let lastSyncResult = null;   // {ts, msg} from the last manual "Sync now" (lab only)

/* WHICH MACHINE, and nothing else. Split out of applyLabIdentity on 2026-07-31
 * because the Dashboard stopped fetching /api/state until a project is open —
 * and machine identity is not a project fact. Everything here needs only the
 * boolean, so it can run from loadOverview() the moment the page loads: the
 * ember theme, the "edit on HQ only" banner, the hidden curation tabs and the
 * read-only notices. Getting those late would mean the lab briefly looked like
 * HQ, on the one machine where "am I allowed to edit here?" must never be in
 * doubt. Idempotent — it is called on every overview poll. */
function applyMachineTheme(isLab){
  if (!isLab || document.body.classList.contains("lab")) return;
  document.body.classList.add("lab");   // ember theme (see cockpit.css tail)
  $("#appTitle").textContent = "🧪 Groundwork Trainer";
  document.title = "Groundwork Trainer";
  _buildLabChrome();
}

/* The lab's chrome: banner, hidden curation tabs, read-only notices, Sync now.
 * Built once; all of it is decided by "this is the lab" alone. */
function _buildLabChrome(){
  if (document.getElementById("labBanner")) return;
  {
    const b = document.createElement("div");
    b.id = "labBanner";
    b.innerHTML = "🧪 <b>LAB (the worker)</b> — press Retrain here for fast training. " +
      "<b>Edit labels on the home UI only</b> (edits here get overwritten). " +
      "<span id='labSyncTxt' class='mMeta'></span> " +
      "<button id='labSyncBtn' title='Pull the latest dataset from home right now'>🔄 Sync now</button>";
    document.querySelector("header").appendChild(b);
    // The lab trains; it never curates. Editing tabs hide entirely;
    // Training/Test set stay VISIBLE but read-only (loadGrid checks
    // IS_LAB, and the API refuses writes server-side) — so what the sync
    // delivered can be verified by eye right here.
    // "history" and "rtmruns" are gone (merged into "runs"); harmless to ask
    // for, kept so a rollback to an older index.html still hides them.
    for (const t of ["count", "history", "rtmruns"]){   // worker hides Try it
      const btn = document.querySelector(`nav button[data-tab="${t}"]`);
      if (btn) btn.hidden = true;
    }
    /* THE FIX QUEUE IS A PANE NOW, NOT A TAB. Hiding `nav button[data-tab=fix]`
     * used to remove it; since the three collections merged into Images that
     * button does not exist, and the loop above silently did nothing — the lab
     * would have offered a labelling queue on a read-only mirror. Hide the
     * SEGMENT, and fall back to Training if that is where you were. */
    const fixSeg = document.querySelector('.imgSegBtn[data-pane="fix"]');
    if (fixSeg){
      fixSeg.hidden = true;
      if (fixSeg.classList.contains("active") && window.imagesShow) imagesShow("train");
    }
    /* THE PANES, not the old section ids. #train and #testset were sections and
     * are now #pane-train / #pane-testset — this was getElementById(sec).prepend
     * on null, a TypeError that would have taken the lab's whole chrome with it
     * while looking perfectly fine on HQ. Guarded as well as retargeted: this
     * runs on a machine nobody opens a browser on casually. */
    for (const sec of ["pane-train", "pane-testset"]){
      const host = document.getElementById(sec);
      if (!host) continue;
      const p = document.createElement("p");
      p.className = "provenance";
      p.innerHTML = "🧪 <b>Read-only mirror</b> — synced one-way from HQ so you "
        + "can verify the data that will train here. Click an image to view the "
        + "full image. All editing lives on Groundwork HQ.";
      host.prepend(p);
    }
    // The worker no longer pulls from HQ itself — syncing is an
    // HQ-initiated action (Machines tab), so the worker holds no
    // credential for its HQ. Hide the legacy button if present.
    const sb = $("#labSyncBtn"); if (sb) sb.hidden = true;
  }
}

/* The state-dependent half: what the Trainer LAST trained, and how stale its
 * mirror is. Runs from loadState(), so only inside an open project — which is
 * correct, because both facts are about a project's data. */
function applyLabIdentity(s){
  applyMachineTheme(true);
  // The Trainer serves nothing — its hero card shows what it LAST TRAINED
  // and where it went. (loadState skips the serving fields when IS_LAB.)
  const hero = document.querySelector("#projdash .dashCard.hero");
  /* THE LABEL, NOT AN <h3>. The Overview port replaced the hero's heading with
   * a .gwReadK label span, and this line went on calling .textContent on the
   * result of querySelector("h3") — null here, and a TypeError inside
   * loadState, which would have taken the worker's whole cockpit down while
   * looking fine on HQ. Guarded as well as retargeted: this file runs on a
   * machine that is awkward to open a browser on. */
  const heroLab = hero && hero.querySelector(".gwReadK");
  if (heroLab) heroLab.textContent = "This machine";
  $("#stServed").textContent = "🧪 Trains for HQ";
  const br = (s.retrain && s.retrain.batch_results || [])[0];
  const meta = hero && hero.querySelector(".servedMeta");
  if (meta) meta.innerHTML =
    (br ? `last run here: <b>${br.run}</b> — MAE <b>${fmtScore(br.mae)}</b> · adopted home automatically · ` : "")
    + "serving & curation live on Groundwork HQ";
  if (!window._labBooted){
    window._labBooted = true;
    // "yolo" has not been a tab since the four run/train tabs merged into
    // "runs" — showTab("yolo") fell through to the projects redirect, so the
    // Trainer stopped opening on its own job. Only visible ON the lab, which
    // is why nothing here caught it.
    showTab("runs");            // the Trainer's whole job — open it by default
  }
  const t = $("#labSyncTxt");
  if (s.last_sync){
    const min = Math.max(0, Math.round(Date.now()/1000 - s.last_sync) / 60) | 0;
    // Colour alone can't carry this — the owner is colour-blind and the string
    // was byte-identical at 3 min and 300 min. This is the "am I about to train
    // on yesterday's data?" signal, so the warning has to be in the text. HQ's
    // dashboard already does it this way (cockpit/dashboard.js).
    const stale = min > 10;
    t.textContent = (stale ? "⚠ MIRROR STALE — " : "· ")
      + `data synced ${min < 1 ? "just now" : min + " min ago"}`;
    t.style.color = stale ? "var(--warn)" : "";
  } else {
    t.textContent = "· sync age unknown";
  }
  if (lastSyncResult && Date.now() - lastSyncResult.ts < 60000)
    t.textContent += ` — ${lastSyncResult.msg}`;
}

function applyHqIdentity(s){
  /* This used to build a "🐢 Trainer unreachable? Train on this machine"
   * fallback around #retrainBtn, moving the retrain controls inside a
   * <details>. Both are gone: training is one control with MACHINE as a field
   * (cockpit/runs/train.js), so "train here instead" is picking a different option
   * rather than a second hidden panel — and on HQ it is not even offered, by
   * the owner's call, because the 8GB laptop card is a CLI-only fallback now. */
  updateLabWatch();               // fire-and-forget; renders/removes #labWatch
}

/* HQ-only: watch the Trainer over the network, and keep `lastLab` fresh for the
   ghost row in the Models list.

   THE "🧪 Trainer is cooking" CARD IS GONE (owner, 2026-08-02). It said the same
   thing as the Activity row directly above it — which now also names the CARD and
   sits beside the Cancel button, so it is both more informative and the place you
   would act from. Two panels narrating one run is how they come to disagree, and
   the redundant one had already told the owner a run existed that never did.

   labCooking() stays: the ghost row still uses it to decide whether a run is
   inbound, and the "error" phase is the only place a failed lab run is announced
   at all. */
let lastLab = null;                       // latest /api/lab_status, for the ghost row
/* CHALLENGERS TOO, on the Overview. /api/lab_status is the yolo retrain only —
 * it has no `actives` — so a training DEIM was invisible on the one screen that
 * answers "what is this project doing". The Results table got a ghost row for
 * them; this is the same fact in the place people actually watch.
 *
 * A SECOND FETCH, and a cheap one: on HQ /api/lab/status resolves through
 * lab_proxy, which is cached and refreshed off-thread precisely so a poll does
 * not pay the network round trip. */
let lastActives = [];     // challengers holding a GPU right now
let lastFinishing = [];   // …and those past training, not yet in the ledger
function labCooking(r){
  const recent = r && r.finished && (Date.now()/1000 - r.finished) < 900;
  if (!r || !r.ok) return null;
  if (r.status === "running") return "running";
  if (r.status === "done" && recent && !r.adopted) return "adopting";
  if (r.status === "error" && recent) return "error";
  return null;
}
async function updateLabWatch(){
  // Both in parallel; a failure in either must not blank the other.
  const [r, ls] = await Promise.all([
    api("/api/lab_status"),
    api("/api/lab/status").catch(() => null),
  ]);
  lastLab = r;
  lastActives = (ls && ls.actives) || [];
  lastFinishing = (ls && ls.finishing) || [];
  renderGhostRow();
  // The 🔬 Watch-training button follows the RUN, not the machine. peek.js gates
  // it on HQ's own /api/retrain, so it vanished the day training moved to the
  // worker — the same machine-instead-of-run bug run_log.js and run_vis.js already
  // fixed for logs and pictures.
  if (window.peekLabTick) peekLabTick(r);
  const c = document.getElementById("labWatch");
  const phase = labCooking(r);
  /* NO CARD AT ALL NOW (owner, 2026-08-05). This drew a banner on an error
   * saying "Trainer run hit an error — its log is in the run terminal on
   * Train", which is a card telling you to look at something already on the
   * same screen: the Activity row names the card and sits beside Cancel, and
   * the terminal is directly below it. A notice whose entire content is a
   * pointer at its own neighbour is noise, and it appeared on every poll for
   * as long as the error state lasted.
   *
   * The error is not hidden — it is in the run terminal, which is where a log
   * belongs, and the Activity row still shows the run. */
  if (c) c.remove();
}
/* Faded placeholder above the newest model while the Trainer cooks — where
   the adopted run WILL appear. Re-applied after every loadModels rebuild.

   IT MUST NOT READ AS A RUN THAT EXISTS. `next_name` is the name the NEXT run
   will be given, not a run — but rendered as `⏳ yolov8n-60` in a list of real
   models it looks exactly like one, and on 2026-08-02 the owner went hunting
   through HQ, the worker and the ledger for a yolov8n-60 that had never existed.
   That was made worse by the stale `adopted` flag keeping this row up for 15
   minutes after every finished run (fixed in api/state.py), but the wording was
   misleading on its own merits, so it says "will be called" now. */
function renderGhostRow(){
  const box = $("#modelList");
  if (!box) return;
  const old = document.getElementById("ghostRow");
  if (old) old.remove();
  box.querySelectorAll(".modelRow.ghost[data-alt]").forEach(el => el.remove());

  /* A CHALLENGER ON ITS WAY, in the list you pick a model FROM.
   *
   * This is not the Activity feed restated, which is what I first thought and
   * why this was briefly removed. Activity answers "what is happening on this
   * machine"; THIS list answers "what can I serve" — and a model that is 40
   * epochs into training is about to be an option. Seeing it here is the
   * difference between choosing from what exists and knowing to wait ten
   * minutes for something better. yolo has had exactly this row since
   * 2026-08-02, immediately below.
   *
   * ALL THREE PHASES, because the picker's question does not stop at training:
   * a run that is scoring has no MAE yet, and one that is adopting is a row
   * that is about to appear. Both are reasons not to pin something else.
   */
  const ghosts = [
    ...lastActives.map(a => ({
      run: a.run,
      how: (a.epoch != null && a.total) ? `epoch ${a.epoch}/${a.total}` : "training",
      card: (a.card === 0 || a.card > 0) ? ` · card ${a.card}` : "",
    })),
    ...lastFinishing.map(fin => ({
      run: fin.run, card: "",
      how: fin.phase === "adopting" ? "verdict in, adopting…"
         : fin.phase === "queued"   ? "trained — waiting for a free card"
                                    : "training done, scoring…",
    })),
  ];
  for (const g of ghosts) {
    if (!g.run) continue;
    const row = document.createElement("div");
    row.className = "modelRow ghost";
    row.dataset.alt = g.run;
    // NOT servable yet, and the row says so in words — the greying and the
    // dashed border are not enough on their own for a colour-blind reader, and
    // this list is otherwise entirely made of things you CAN pick.
    row.innerHTML = `<span class="mName">⏳ ${g.run}</span>`
      + `<span class="mMeta"> — ${g.how} on 🧪 Trainer${g.card}`
      + ` · not servable yet</span>`;
    box.prepend(row);
  }

  const phase = labCooking(lastLab);
  if (!phase || phase === "error") return;
  const row = document.createElement("div");
  row.id = "ghostRow";
  row.className = "modelRow ghost";
  const card = (lastLab.card === 0 || lastLab.card > 0)
    ? ` · card ${lastLab.card}` : "";
  const meta = phase === "running"
    ? (lastLab.epoch != null && lastLab.total
        ? `training on 🧪 Trainer — epoch ${lastLab.epoch}/${lastLab.total}${card}`
        : `on 🧪 Trainer — ${lastLab.step || "working"}${card}`)
    : "exam done — adopting…";
  row.innerHTML = `<span class="mName">⏳ will be called ${lastLab.next_name}</span>`
    + `<span class="mMeta"> — ${meta}</span>`;
  box.prepend(row);
}
