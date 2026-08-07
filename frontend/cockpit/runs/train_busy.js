"use strict";
/* The per-card live boxes + cancel buttons — extracted from train.js (size
 * cap). SHARED-LEXICAL rules: reads TRAIN_OPTS/TRAIN_BUSY/TRAIN_CANCELLING
 * declared in train.js, which loads BEFORE this file; train.js calls
 * _paintBusy()/trainBusyTick() only from timers and events, which fire after
 * load. Unique top-level names enforced by check_frontend #5. */

/* PROGRESSING vs WEDGED, decided by CPU-ticks MOVEMENT across polls plus log
 * freshness — filesystem//proc facts that work for every family, including
 * the ones whose trainers print nothing. A wedged driver call accumulates no
 * ticks while the process "exists" beautifully; that asymmetry is the whole
 * diagnostic. */
const _liveTicks = new Map();   // run -> {ticks, at, stale}
function _liveHealth(a) {
  const bits = [];
  if (a.elapsed_s != null) {
    const m = Math.round(a.elapsed_s / 60);
    bits.push(m >= 1 ? `${m} min in` : "just started");
  }
  if (a.eta_s != null && a.eta_source)
    bits.push(`~${Math.max(1, Math.round(a.eta_s / 60))} min left (${a.eta_source})`);
  if (a.log_age_s != null && a.log_age_s > 90)
    bits.push(`last output ${Math.round(a.log_age_s / 60)} min ago`);
  if (a.ticks != null && a.run) {
    const prev = _liveTicks.get(a.run);
    const now = Date.now();
    if (prev && a.ticks === prev.ticks && now - prev.at > 20000) {
      if ((a.log_age_s ?? 0) > 90)
        bits.push("⚠ no CPU activity — possibly wedged (a frozen log plus " +
                  "frozen CPU time is the earliest wedge sign)");
    } else if (!prev || a.ticks !== prev.ticks) {
      _liveTicks.set(a.run, {ticks: a.ticks, at: now});
    }
  }
  return bits.length ? " · " + bits.join(" · ") : "";
}

function _cancelBtn(machineKey, cardIndex) {
  const started = TRAIN_CANCELLING.get(cancelKey(machineKey, cardIndex));
  if (started && Date.now() - started > CANCEL_SETTLE_MS) {
    // IT NEVER LANDED. Rather than leave the control dead, hand it back and let
    // the operator try again — a button stuck on "Cancelling…" forever is worse
    // than one that admits it did not work.
    TRAIN_CANCELLING.delete(cancelKey(machineKey, cardIndex));
    return `<button class="trainCardCancel" data-card="${cardIndex}">✕ Cancel</button>`;
  }
  return started
    ? `<button class="trainCardCancel" data-card="${cardIndex}" disabled>Cancelling…</button>`
    : `<button class="trainCardCancel" data-card="${cardIndex}">✕ Cancel</button>`;
}

async function _paintBusy() {
  const box = $("#tCards");
  if (!box || !TRAIN_OPTS) return;
  const mach = _tMachine();
  const cards = (mach && mach.cards) || [];
  // THE CARD INVENTORY MUST NOT GATE THE BUSY FETCH. This returned here when a
  // machine had no measured cards, so TRAIN_BUSY was never populated — and
  // run_log.js and run_vis.js both read TRAIN_BUSY.challenger to find the
  // running challenger. On the WORKER, outputs/machines.json is written by
  // scratch/probe_machines.py over ssh FROM HQ and outputs/ is gitignored, so
  // the worker has never had one: /api/train/options there reports zero cards for
  // both machines. The result was that the Trainer could not see, or cancel,
  // the run it was itself training — the one thing that cockpit exists for.
  //
  // "Which cards exist" and "what is running" are different questions. Only the
  // per-card PANEL needs the first; fetch the busy state first, always.
  let jobs = [];
  try {
    // /api/lab_status is HQ-watching-the-worker and 400s on the lab by design, so
    // only ask when this is not the lab. /api/lab/status is this machine's own
    // challenger and is asked everywhere.
    const [lab, retrain] = await Promise.all([
      api("/api/lab/status").catch(() => ({})),
      window.IS_LAB ? Promise.resolve({})
                    : api("/api/lab_status").catch(() => ({})),
    ]);
    /* WHICH CARD, from the run's own crumb. This used to be hardcoded `card: 1`
     * — true only because challengers happen to pin the 3090. gpu.acquire now
     * records CUDA_VISIBLE_DEVICES, so the panel shows where the job actually
     * is; the 1 remains only for crumbs written before that field existed. */
    /* EVERY live challenger, not one. Two can train at once since per-run
     * dataset trees landed, and this pushed only `lab.active` — so with both
     * cards busy the panel drew one card correctly and the OTHER AS IDLE, which
     * is worse than saying nothing. The server now sends `actives`; `active`
     * is kept as its first element for older payloads. */
    const running = (lab && lab.actives) || (lab && lab.active ? [lab.active] : []);
    /* WHICH MACHINE a job is on, so it is drawn on THAT machine's cards.
     *
     * Every job used to be pushed without one and rendered onto whichever
     * machine happened to be selected — so with the laptop selected, the WORKER's
     * retrain appeared on the laptop's RTX 5060, complete with a Cancel button
     * that would have been sent to the wrong machine. Reported as "it's showing
     * training on my laptop card when this machine is selected lol — that's not
     * true", and it was not.
     *
     * `remote` is set by lab_ops.status when the answer came from the proxy, so
     * it names the Trainer; anything else is this cockpit's own machine. */
    // Prefer the machine the PAYLOAD names; "first non-local" is only the
    // fallback for older servers, and wrong the day a second worker exists.
    const rigKey = (retrain && retrain.machine)
      || (TRAIN_OPTS.machines || []).find(m => !m.local)?.key || "";
    const hereKey = (TRAIN_OPTS.machines || []).find(m => m.local)?.key || "here";
    for (const a of running)
      jobs.push({card: a.card ?? 1, kind: "challenger", what: a.run,
                 machine: a.remote ? rigKey : hereKey,
                 more: (a.epoch != null
                   ? `epoch ${a.epoch}/${a.total ?? "?"}`
                   : "training (this family reports no epoch line)")
                   + _liveHealth(a)});
    TRAIN_BUSY.yolo = !!(retrain && retrain.ok && retrain.status === "running");
    // The split belongs to the machine the run would go to; lab.split_current
    // is already the remote's when this cockpit is watching the Trainer.
    TRAIN_BUSY.splitOk = (lab && "split_current" in lab) ? lab.split_current : null;
    // Which split the running retrain built, so the seed box can warn before
    // the press rather than after the 409.
    TRAIN_BUSY.runningSeed = (lab && lab.split_seed != null) ? lab.split_seed : null;
    TRAIN_BUSY.known = true;          // an answer arrived, whatever it says
    // .challengers is the real answer; .challenger stays as the first so the log
    // and picture panels keep working until they read the list too.
    // {run, card}, not just names: run_log.js labels each terminal with the card
    // it is on, and two challengers are on DIFFERENT cards — a list of bare
    // names would have made both panels claim card 1.
    TRAIN_BUSY.challengers = running.map(a => ({run: a.run, card: a.card ?? 1}));
    TRAIN_BUSY.challenger = running.length ? running[0].run : null;
    // /api/lab_status IS the Trainer's retrain — HQ watching the worker. It is
    // never this machine's, which is what /api/retrain answers.
    if (retrain && retrain.ok && retrain.status === "running")
      jobs.push({card: 0, kind: "yolo", machine: rigKey,
                 what: retrain.next_name || "yolo retrain",
                 more: retrain.epoch != null
                   ? `epoch ${retrain.epoch}/${retrain.total ?? "?"}` : (retrain.step || "")});
  } catch (e) { /* a card panel must never be the thing that breaks the tab */ }
  // The busy state decides whether Train is startable — but this runs on the 8s
  // heartbeat, so it MUST keep the card you picked. Without the argument it
  // re-selected the model's default every tick, which made the other card
  // unselectable in practice: you could click it and watch it snap back.
  _paint(_tCardNow());
  // NOW the inventory matters: with nothing measured there are no per-card boxes
  // to draw. TRAIN_BUSY is already set above, so the log and picture panels keep
  // working on a machine nobody has probed.
  /* A CANCEL IS DONE WHEN THE JOB IS GONE, not when the request returned. The
   * request only asks; the pipeline stops at its next stage boundary. So the
   * mark is cleared here, by the poll that can actually see it happen. */
  for (const [k, started] of [...TRAIN_CANCELLING]) {
    const [mk, ci] = k.split(":");
    if (!mach || mk !== mach.key) continue;     // no/other machine selected
    if (!jobs.some(x => String(x.card) === ci)) TRAIN_CANCELLING.delete(k);
    else if (Date.now() - started > CANCEL_SETTLE_MS) TRAIN_CANCELLING.delete(k);
  }
  if (!cards.length) { box.innerHTML = ""; return; }
  // ONLY THIS MACHINE'S JOBS. Matching on card alone put another machine's run
  // on these cards — see the `machine` tag above.
  const mine = jobs.filter(j => !j.machine || j.machine === mach.key);
  box.innerHTML = cards.map(c => {
    const j = mine.find(x => x.card === c.index);
    return `<div class="trainCard${j ? " busy" : ""}">
      <b>card ${c.index}</b> <span class="mMeta">${_tEsc(c.name)} · ${c.vram_gb}GB</span>
      <span class="trainCardWhat">${j
        ? `${_tEsc(j.what)} — ${_tEsc(j.more)}`
        : "<span class='mMeta'>idle</span>"}</span>${j
        ? _cancelBtn(mach.key, c.index) : ""}
      </div>`;
  }).join("");

  /* Wired AFTER the render, per button, with the job it belongs to captured in
   * the closure. Not one delegated handler reading data- attributes: the run
   * name and progress go into the confirm dialog, and re-deriving them from the
   * DOM at click time is how a dialog ends up naming a run that already ended. */
  box.querySelectorAll(".trainCardCancel").forEach(btn => {
    const idx = +btn.dataset.card;
    const j = mine.find(x => x.card === idx);
    if (!j) return;
    // A button already cancelling stays inert across this repaint.
    if (btn.disabled) return;
    btn.onclick = () => window.trainCancel({
      machineKey: mach.key, machineName: mach.name, local: !!mach.local,
      card: idx, kind: j.kind, run: j.what, more: j.more});
  });
}

(() => {
  const ms = $("#tModel"), mc = $("#tMachine"), cd = $("#tCard"), btn = $("#tTrain");
  if (!ms || !mc || !btn) return;
  // Model and machine changes auto-select the card; a CARD change must keep it.
  ms.onchange = () => { _paint(); };
  mc.onchange = () => { _paint(); _paintBusy(); };
  cd.onchange = () => _paint(_tCardNow());
  const conc = $("#tConc");
  if (conc) conc.onchange = () => _paint(_tCardNow());
  // "touched" so a model change stops overwriting a size you chose on purpose.
  ["tSize", "tEpochs", "tBatch"].forEach(id => {
    const el = $("#" + id);
    if (el) el.addEventListener("input", () => { el.dataset.touched = "1"; });
    if (el) el.addEventListener("change", () => { el.dataset.touched = "1"; });
  });

  btn.onclick = async () => {
    const m = _tModel(), mach = _tMachine();
    if (!m || !mach) return;
    const card = $("#tCard").value === "" ? null : +$("#tCard").value;
    const imgsz = +$("#tSize").value || m.default_imgsz;
    const epochs = $("#tEpochs").value ? +$("#tEpochs").value : null;
    const batch = $("#tBatch").value ? +$("#tBatch").value : null;
    // Both families take it now (lab_ops.TrainReq.split_seed).
    const seed = $("#tSeed").value ? +$("#tSeed").value : 0;
    // A challenger run needs a name; generated here so a retry after a network
    // error cannot start a SECOND run under a new one.
    const run_name = m.champion ? ""
      : `${m.key === "deimv2-n-tv28" ? "deimv2n13" : m.key}-${imgsz}-`
        + Date.now().toString(36).slice(-4);
    const cardName = ($("#tCard").selectedOptions[0] || {}).textContent || card;
    if (!confirm(`Train ${m.label}${run_name ? ` (${run_name})` : ""}?\n\n`
      + `${mach.name} · ${cardName}\n`
      + `${imgsz}px · ${epochs ?? "default"} epochs · batch ${batch ?? "auto"}`
      + `${seed ? ` · split seed ${seed}` : ""}`
      + `\n\n`
      + (mach.local ? "" : "The dataset is synced to that machine first — "
         + "training is refused if the sync fails.\n"))) return;
    btn.disabled = true; btn.textContent = "starting…";
    const why = $("#tWhy");
    try {
      const r = await apiOrThrow("/api/train", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({model: m.key, machine: mach.key, card, imgsz,
                              epochs, batch, run_name,
                              split_seed: seed,
                              allow_concurrent: !!($("#tConc") && $("#tConc").checked)})});
      const s = r.synced;
      const note = s ? (s.sent ? `synced ${s.sent} file(s), then ` : "already in sync, ") : "";
      if (r.ok === false) throw new Error(r.error || "refused");
      // SAY WHAT ACTUALLY STARTED, not what was asked for. retrain_job clamps
      // epochs to a 50-600 sane band, so typing 3 trains 50 — the response has
      // always said so and the UI did not, which is a small silent-wrong-
      // parameter of exactly the kind this repo keeps paying for.
      const got = r.epochs;
      const clamped = got != null && epochs != null && got !== epochs
        ? ` — ${got} epochs, not ${epochs} (the trainer clamps to its sane band)`
        : (got != null ? ` — ${got} epochs` : "");
      if (why) { why.textContent = `▶ ${note}started on ${mach.name} card ${r.card}${clamped}.`;
                 why.hidden = false; }
      if (window.loadYolo) loadYolo();
      if (window.loadLab) loadLab();
      // ACCEPTED. Hold the button until the poll sees it — see TRAIN_STARTED_AT.
      TRAIN_STARTED_AT = Date.now();
      _paintBusy();
    } catch (e) {
      if (why) { why.textContent = `✗ ${e.message || e}`; why.hidden = false; }
      // REFUSED, so nothing is starting and the control must come back at once.
      TRAIN_STARTED_AT = 0;
    }
    btn.textContent = "▶ Train";
    _paint(_tCardNow());
  };
})();
