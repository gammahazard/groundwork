"use strict";
/* ONE Train control: model, machine, card.
 *
 * It was two panels — "Train — champion (YOLO)" and "Train — challengers" —
 * plus a link that opened a SECOND cockpit on the worker so you could press ITS
 * Retrain. That split taught champion-vs-challenger, which is the framing the
 * pivot removes, and it tied "where you work" to "which machine has the GPU".
 *
 * WHAT IS POSSIBLE IS DECIDED SERVER-SIDE, and this file only renders it.
 * /api/train/options answers model x machine x card with a REASON for every
 * refusal, because the reasons are facts about the machines that a browser
 * cannot know: DEIM's torch build carries no sm_120 kernels, so it cannot use
 * the worker's 5070 Ti at all. A disabled option that says why is a fact; a
 * disabled option that does not is a mystery, and an ENABLED one that fails at
 * the first kernel launch is forty minutes.
 *
 * THE CARD AUTO-SELECTS AND STAYS OVERRIDABLE. Picking a model moves the card to
 * the first one that can run it AND fit in it — the 5070 Ti for yolo, the 3090
 * for a challenger, arrived at from the capability data rather than hardcoded to
 * agree with it. You can still choose the other card where the hardware allows,
 * which is the whole point of two GPUs: yolo on one, DEIM on the other, at once.
 *
 * "STAYS OVERRIDABLE" WAS NOT TRUE UNTIL 2026-08-02, and nothing said so. This
 * paragraph described the intent; the code reset the card to the model's default
 * on every repaint. `_paintCards(preferred)` was the mechanism for keeping a
 * choice and NO caller ever passed it, so rebuilding the <option> list threw the
 * selection away and line "want" put it back to the default. The 8s heartbeat
 * repaints too, so clicking the other card visibly snapped back within seconds.
 * A doc comment is not a test: the browser found this in one pass, reading code
 * did not. `preferCard` now threads through every repaint that is not a model or
 * machine change. */

const _tEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

let TRAIN_OPTS = null;
// What is running where, refreshed by the same heartbeat that paints the cards.
/* `known` is a THIRD state, and its absence was a real bug: on a fresh load
 * TRAIN_BUSY reads "nothing is running" — which is indistinguishable from the
 * truth on an idle box — so reloading the page mid-run showed a live Train
 * button over a training card until the first poll landed. Nothing false
 * started (the server refuses), but the control invited a press it would
 * reject. Unknown is not idle; the button waits for an answer.
 *
 * `splitOk` is the server's is_current() for the machine being targeted: true,
 * false, or null for "not asked / could not tell". Null must not read as
 * stale — a worker that predates the field would otherwise look permanently
 * blocked. */
const TRAIN_BUSY = {yolo: false, challenger: null, challengers: [],
                    known: false, splitOk: null, runningSeed: null};

/* CARDS WITH A CANCEL IN FLIGHT, and this has to live outside the render.
 *
 * _paintBusy rebuilds the card panel with innerHTML on every heartbeat, so a
 * `btn.disabled = true` set by the click handler is gone within seconds — and
 * the run is still in the busy list, because the backend has not stopped it
 * yet. The button therefore came back enabled over a run that WAS being
 * cancelled, which reads as "the click did nothing" and gets clicked again.
 * Reported exactly that way.
 *
 * Keyed by "machine:card" rather than a bare index: two machines can each have
 * a card 1, and cancelling on one must not grey out the other.
 *
 * The value is WHEN, so a cancel that never lands cannot disable the control
 * forever — see CANCEL_SETTLE_MS. */
const TRAIN_CANCELLING = new Map();
const CANCEL_SETTLE_MS = 90_000;

function cancelKey(machineKey, card) { return `${machineKey}:${card}`; }

/* Mark a card as cancelling, or clear it. train_cancel.js owns the call. */
window.trainMarkCancelling = (machineKey, card, on) => {
  const k = cancelKey(machineKey, card);
  if (on) TRAIN_CANCELLING.set(k, Date.now());
  else TRAIN_CANCELLING.delete(k);
  _paintBusy();
};

/* WHEN A START WAS ACCEPTED BUT IS NOT YET VISIBLE.
 *
 * POSTing /api/train returns as soon as the pipeline is launched, which is well
 * before the run shows up in /api/retrain or /api/lab/status. The button was
 * re-enabled the moment the POST returned and _paint() then recomputed the gate
 * from TRAIN_BUSY — still false — so Train sat there live and inviting over a
 * run that was already starting.
 *
 * Nothing terrible follows from a second press: retrain_job.start() takes a
 * cross-process flock and refuses with "a retrain is already running", so two
 * yolo runs cannot both begin. But the user sees an enabled button, presses it,
 * and gets an error for doing the obvious thing — and on the challenger path
 * the refusal costs a round trip to a machine that answers slowly while it
 * trains.
 *
 * So the button stays locked from a successful start until the poll actually
 * SEES the job. Time-boxed, because a start that never appears (the pipeline
 * died in its first second) must not lock the control forever — after that the
 * gate falls back to the real busy state, which is the honest answer. */
let TRAIN_STARTED_AT = 0;
const TRAIN_SETTLE_MS = 30000;

function _tModel() {
  return (TRAIN_OPTS?.models || []).find(m => m.key === $("#tModel")?.value);
}
function _tMachine() {
  return (TRAIN_OPTS?.machines || []).find(m => m.key === $("#tMachine")?.value);
}

/* The card currently selected, or undefined when none is — NEVER "".
 *
 * `+"" === 0`, so handing an empty select straight to _paintCards would read as
 * a deliberate choice of card 0 and pin every model to it on first load. */
const _tCardNow = () => {
  const v = $("#tCard")?.value;
  return v == null || v === "" ? undefined : v;
};

/* Every card on the chosen machine, each enabled only if the chosen model can
 * actually run on it. The <option> carries the reason in its text, because a
 * disabled option has no tooltip on a phone.
 *
 * TWO DIFFERENT WORDS, BECAUSE THEY ARE TWO DIFFERENT THINGS. "unusable" means
 * the venv has no kernels for that card and the run would die at its first
 * launch — the option is disabled. "too small" means it runs correctly and
 * slowly: WSL2 spills the overflow into host RAM rather than raising OOM, so
 * the option stays SELECTABLE. Spilling on purpose is a fair trade when the
 * alternative is an idle card, and only the person pressing the button knows
 * which they want.
 *
 * Measured 2026-08-02: deimv2-n-tv28 needs 17.2 GiB, went to the 16 GB 5070 Ti
 * because it was the first card that could run it, and trained at 2.19 s/it
 * against 0.72-0.94 for the same model on the 24 GB 3090. Nothing said a word.
 * Said in TEXT, never by colour alone. */
function _paintCards(preferred) {
  const sel = $("#tCard"), note = $("#tCardNote");
  const m = _tModel(), mach = _tMachine();
  if (!sel || !m || !mach) return;
  const info = m.machines[mach.key] || {cards: [], default: null};
  sel.innerHTML = info.cards.map(c =>
    `<option value="${c.index}"${c.ok ? "" : " disabled"}>`
    + `${c.index} · ${_tEsc(c.name)} ${c.vram_gb}GB`
    + (c.ok ? (c.fits === false ? " — too small, ~3x slower" : "") : " — unusable")
    + `</option>`).join("")
    || `<option value="">(no cards measured — press Probe on the Machines tab)</option>`;
  const want = [preferred, info.default, ...info.cards.filter(c => c.ok).map(c => c.index)]
    .find(v => v != null && info.cards.some(c => c.index === +v && c.ok));
  if (want != null) sel.value = String(want);
  /* THE NOTE DESCRIBES THE CARD YOU PICKED. It used to list every BLOCKED
   * card's reason regardless of the selection, so choosing the 3090 for DEIM —
   * which works — still showed "card 0: sm_120 is not in .venv-deim's torch
   * build", reading as a complaint about the card you had just chosen. The
   * reason for a card you cannot use belongs on that card's own <option>, where
   * it already is ("0 · RTX 5070 Ti 16GB — unusable"), and here only when the
   * SELECTED one is the unusable one. */
  const chosen = info.cards.find(c => c.index === +sel.value);
  note.textContent =
    chosen && !chosen.ok ? chosen.why
    /* A card that RUNS it but does not FIT it: say what it costs, in full,
     * where the choice is being made. This is the only warning the 3x tax
     * gets — the run itself reports nothing unusual. */
    : chosen && chosen.fits === false ? `${chosen.sm} · ${chosen.tax}`
    /* UNMEASURED IS NOT "FINE". D-FINE and RF-DETR print no memory line, so no
     * run of either has ever recorded a peak and fits() passes them by default.
     * Showing only "sm_120 · 16 GB" would read as a clean bill of health for a
     * question nobody has asked yet, which is the confusion that cost the four
     * hours in the first place. fits() hands back its reason either way. */
    : chosen ? `${chosen.sm} · ${chosen.vram_gb} GB`
               + (chosen.tax ? ` · ${chosen.tax}` : "")
    : "";
}

/* `preferCard` KEEPS A CARD YOU CHOSE. Omit it to let the model's default win.
 *
 * Which is the whole difference between the two: changing the MODEL should move
 * the card (that is the documented auto-select), and everything else — a card
 * change, the 8s heartbeat, a reload — must not. */
function _paint(preferCard) {
  if (!TRAIN_OPTS) return;
  const m = _tModel(), mach = _tMachine();
  const why = $("#tWhy"), btn = $("#tTrain");
  /* A HIDDEN FAMILY IS STILL SAID OUT LOUD. The server drops families whose
   * sidecar venv is installed on no machine — every card would refuse them —
   * but silence turns "where did DEIM go?" into the next question, so the one
   * place someone is looking for it says where it comes from. */
  const ns = TRAIN_OPTS.needs_stack || [];
  $("#tModelNote").textContent = (m
    ? `${m.license}${m.deprecated ? " · deprecated" : ""} · ${m.venv}` : "")
    + (ns.length ? `${m ? " · " : ""}not installed: ${ns.join(", ")}`
                 + ` — Admin → Extras` : "");
  $("#tMachineNote").textContent = mach ? (mach.blocked || mach.what) : "";
  _paintCards(preferCard);

  // ONE reason, the first that applies, in the order you would hit them.
  let block = "";
  /* BEFORE THE FIRST POLL WE DO NOT KNOW. See TRAIN_BUSY.known — a reload
   * during a run showed an enabled Train button over a training card, because
   * "no jobs seen yet" and "no jobs running" were the same value. */
  if (!TRAIN_BUSY.known) block = "checking what is running…";
  /* THE YOLO CASE MOVED, and it is a warning now rather than a block — the
   * reasoning is at the `warn` assignment below. What used to be here claimed a
   * challenger "cannot start while a yolo retrain is on that machine", which was
   * true of an older server and is not true of this one.
   *
   * The danger it described is real: launching a challenger runs a split that
   * rmtree's outputs/dataset/images|labels/{train,val}, the exact directories a
   * running retrain reads, and per-card GPU locks do not help because the
   * collision is on the FILESYSTEM. But lab_ops.train now SKIPS the split when
   * the tree on disk already is what a fresh split would build, and only refuses
   * when it is not. Blocking both cases here forbade the one this box has two
   * GPUs for. */
  /* Checked FIRST: it outranks every other reason, and while it holds the other
   * messages would be describing a machine whose state we have not re-read. */
  if (TRAIN_STARTED_AT) {
    if (TRAIN_BUSY.yolo || (TRAIN_BUSY.challengers || []).length)
      TRAIN_STARTED_AT = 0;                     // it appeared; hand back control
    else if (Date.now() - TRAIN_STARTED_AT > TRAIN_SETTLE_MS)
      TRAIN_STARTED_AT = 0;                     // it never did; stop pretending
    else
      block = "starting… the run is coming up. This unlocks as soon as it "
            + "appears below.";
  }
  if (block) { /* settled above */ }
  else if (mach && !mach.usable) block = mach.blocked || `${mach.name} is not usable`;
  else if (m && mach && !(m.machines[mach.key] || {}).any)
    block = `${m.label} cannot run on any card of ${mach.name}`;
  else if ($("#tCard").value === "" || $("#tCard").selectedOptions[0]?.disabled)
    block = "pick a card this model can use";
  /* A YOLO RETRAIN IS A WARNING, NOT A BLOCK — and it used to be a block whose
   * message said "it will be refused". That is not what the server does.
   * lab_ops.train refuses a challenger under a live retrain ONLY when the
   * dataset split is not current; when it already is, the split is skipped and
   * both runs proceed on separate cards. That case is the entire reason this
   * box has two GPUs, and the client was forbidding it.
   *
   * The client CANNOT decide this: `split.is_current()` compares the on-disk
   * tree against what a fresh split would build, which is a server fact. So say
   * what is true — this may be refused — and let the authority answer. Being
   * stricter than the server is not caution; it is a wrong answer that cannot
   * be appealed. */
  /* A CHALLENGER UNDER A LIVE RETRAIN: say the fact, not a paragraph.
   *
   * The server's rule is exactly one boolean — is the split already what a
   * fresh split would build. If it is, the split is skipped and both runs
   * proceed on separate cards; if it is not, re-splitting would delete the
   * images the retrain is reading, so it refuses. So the UI shows that boolean
   * and matches the same decision, rather than explaining the mechanism every
   * time someone picks a model.
   *
   * NULL IS NOT FALSE. An older Trainer does not send the field; blocking on
   * absent evidence would forbid every challenger there forever. Unknown lets
   * the press through and the server answers — which is where the authority
   * lives anyway. */
  /* THE CONCURRENCY GATE IS A SEPARATE REFUSAL, and the UI used to be silent
   * about it. train_dispatch refuses ANY second GPU job on a machine unless
   * allow_concurrent is passed — that pairing has failed 3 of 4 trials here —
   * while this panel talked only about the SPLIT. So it said "split: ok … will
   * be reused" and the press then returned a 409 about something else
   * entirely. Reported as "it says split ok AND YET DEIM STILL DOESN'T START".
   *
   * The checkbox appears only when something is actually training, because
   * that is the only time the flag means anything. */
  const busyNow = TRAIN_BUSY.yolo || (TRAIN_BUSY.challengers || []).length > 0;
  const cw = $("#tConcWrap");
  if (cw) cw.hidden = !busyNow;
  if (!busyNow && $("#tConc")) $("#tConc").checked = false;

  let note = "";
  if (!block && m && mach && !m.champion && TRAIN_BUSY.yolo) {
    const conc = $("#tConc") && $("#tConc").checked;
    /* A SECOND SEED CANNOT BE BUILT while a retrain trains — one split
     * directory, and rebuilding it deletes what that run is reading. The server
     * refuses with the number; say so before the press. */
    const wantSeed = +($("#tSeed") || {}).value || 0;
    if (TRAIN_BUSY.runningSeed != null && wantSeed !== TRAIN_BUSY.runningSeed)
      block = `the retrain running here is on split seed `
            + `${TRAIN_BUSY.runningSeed}; this asks for ${wantSeed}. Both runs `
            + `share one split, so use seed ${TRAIN_BUSY.runningSeed} or wait.`;
    else if (TRAIN_BUSY.splitOk === false)
      block = "split: not ok — a retrain is running and re-splitting would "
            + "delete the images it is training on. Wait for it to finish.";
    else if (!conc)
      /* SAY WHAT WILL ACTUALLY HAPPEN. The split being reusable is necessary
       * and not sufficient: the server still refuses a second GPU job unless
       * asked. Reporting only the split was the whole bug. */
      note = "split: ok — but a job is already training, and a second GPU job "
           + "is refused by default (that pairing has failed 3 of 4 trials "
           + "here). Tick the box to run it anyway.";
    else
      note = "split: ok, and you have asked to run alongside the other job. "
           + "Two GPU jobs at once has wedged this box twice — DEIM + yolo "
           + "both times.";
  }
  /* A CANCEL MESSAGE IS NOT OVERWRITTEN BY THE HEARTBEAT. This line runs on
   * every repaint and used to wipe "cancelling …" within seconds of the click,
   * which is half of why the control felt dead. While anything is cancelling,
   * whatever train_cancel.js put there stays. */
  if (why && !TRAIN_CANCELLING.size) {
    why.textContent = block || note; why.hidden = !(block || note);
  }
  if (why) why.classList.toggle("tWhyOk", !block && !!note);
  if (btn) btn.disabled = !!block;

  /* THE SIZE LIST IS THE MODEL'S, not a fixed 960/1280/1920. The mmdet trainer
   * takes --scale-factor 2|3 and nothing else, so a 960 pick for yolox mapped to
   * 2 and trained at 1280 — request and run disagreed and nothing said so. And
   * rfdetr's --res must be a multiple of 56, which 960 and 1280 are not. */
  const size = $("#tSize");
  if (m && size) {
    const want = size.dataset.touched && m.sizes.includes(+size.value)
      ? +size.value : m.default_imgsz;
    size.innerHTML = (m.sizes || []).map(v =>
      `<option value="${v}"${v === want ? " selected" : ""}>${v}</option>`).join("");
    size.value = String(want);
  }
  /* THE PLACEHOLDER IS A PROMISE about what a blank field will do, so it must
   * track the model whenever the field is empty.
   *
   * `touched` alone was the wrong test. It is set on the first keystroke and
   * never cleared, so typing 300, clearing it, then switching yolo -> DEIM left
   * "250" sitting in an EMPTY field while the run would actually train 60
   * (lab_ops: `epochs=req.epochs or model.default_epochs or 60`). The only
   * thing on screen saying what would happen was wrong, and blank is exactly
   * when someone is relying on it.
   *
   * Touched still protects a value you TYPED — that is what it is for — but a
   * value you no longer have is not a preference to preserve. */
  const ep = $("#tEpochs");
  if (m && ep && (!ep.dataset.touched || !ep.value))
    ep.placeholder = String(m.default_epochs || "auto");
  /* Same rule for batch, and the number is worth showing rather than "auto":
   * yolo halves above 960 and every other family takes its own default, so
   * "auto" hid the one fact a reader wants — see _batch_for() on the server,
   * which this mirrors. */
  /* EVERY FAMILY TAKES A SEED NOW. This was champion-only, which made the
   * comparison one-sided: yolo could ask "does this survive a different split?"
   * and a challenger could not, so the two were never varied the same way. */
  const sw = $("#tSeedWrap");
  if (sw) sw.hidden = !m;
  const bt = $("#tBatch");
  if (m && bt && (!bt.dataset.touched || !bt.value))
    bt.placeholder = String(m.champion
      ? (+(($("#tSize") || {}).value || m.default_imgsz) <= 960 ? 8 : 4)
      : (m.default_batch || 8));
}

async function loadTrain() {
  try {
    TRAIN_OPTS = await apiOrThrow("/api/train/options");
  } catch (e) {
    const why = $("#tWhy");
    if (why) { why.textContent = `could not load training options: ${e.message || e}`;
               why.hidden = false; }
    return;
  }
  const ms = $("#tModel"), mc = $("#tMachine");
  if (!ms || !mc) return;
  const keepM = ms.value, keepMc = mc.value;
  ms.innerHTML = TRAIN_OPTS.models.map(m =>
    `<option value="${_tEsc(m.key)}">${_tEsc(m.label)}</option>`).join("");
  if (keepM) ms.value = keepM;
  mc.innerHTML = TRAIN_OPTS.machines.map(m =>
    `<option value="${_tEsc(m.key)}"${m.usable ? "" : " disabled"}>`
    + `${_tEsc(m.name)}${m.usable ? "" : " — unavailable"}</option>`).join("");
  mc.value = keepMc && [...mc.options].some(o => o.value === keepMc && !o.disabled)
    ? keepMc
    : (TRAIN_OPTS.machines.find(m => m.usable) || {key: ""}).key;
  // Keeps the card for the same reason it keeps the model and machine above.
  _paint(_tCardNow());
  _paintBusy();
}

/* What each card is doing RIGHT NOW. Two GPUs means "is it free?" is a question
 * per card, and the answer is the difference between starting a second run and
 * queueing behind the first.
 *
 * IT HAS TO POLL. This ran only when the tab opened or the machine changed, so
 * it was a SNAPSHOT: start a run and both cards kept saying "idle" until you
 * navigated away and back. A status panel that does not update is worse than no
 * panel — it does not say "unknown", it says "idle", confidently and wrongly.
 * dashboard.js's existing 8s heartbeat drives it (one heartbeat, not a second
 * timer competing with it), and only while this tab is on screen. */
window.trainBusyTick = () => _paintBusy();

/* The cancel button for one card, in whichever state it is in. A cancel in
 * flight renders as "Cancelling…" and disabled — and because this is called
 * from the render rather than set on the element afterwards, it survives the
 * heartbeat that rebuilds the panel. */
