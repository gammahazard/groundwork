"use strict";
/* ONE log panel, showing whatever is actually training — on either machine.
 *
 * WHAT IT REPLACES. #retrainLog was written only from HQ's own /api/retrain, so
 * once training moved to the worker it was permanently blank: the run you were
 * watching produced pages of output and the terminal under it showed nothing.
 * The fix is not a second terminal for the worker — it is for this one to follow
 * the run instead of the machine.
 *
 * FOUR SOURCES, ONE PANEL, in the order a run can exist:
 *   1. a yolo retrain HERE            -> /api/retrain            (live, age 0)
 *   2. a yolo retrain on the worker      -> /api/lab_status.log_tail (proxied)
 *   3. a challenger on the worker        -> /api/lab/log?run=…       (proxied)
 *   4. nothing running                -> the last finished local run, or hidden
 *
 * IT SAYS HOW STALE IT IS, ALWAYS. Anything from the worker arrives through
 * lab_proxy, which answers instantly from cache and refreshes out of band — so
 * it is typically 8-20s behind and can be minutes behind if the network blips.
 * A terminal that redraws every few seconds LOOKS live, and a log that silently
 * stops updating during a wedge is exactly how tonight's failure hid for 16
 * minutes. So the header carries the machine and the age, and the age is the
 * proxy's own number rather than anything this file invents.
 *
 * NO POLL OF ITS OWN. dashboard.js's 8s heartbeat drives it, the same one that
 * paints the cards — a second timer would just race it.
 */

/* Internal, and DELIBERATELY not on window. A top-level const in a classic
 * script does not become a window property — `window.RUNLOG` is undefined, as
 * `window.ed` was when it silently broke the audit overlay. Nothing outside this
 * file needs it; the note exists so the next reader who checks `window.RUNLOG`
 * in a console and sees undefined does not conclude the panel is broken. */
const RUNLOG = {src: null, at: 0};

function _rlEl() { return document.querySelector("#retrainLog"); }

/* WHICH GPU, not just which index. "card 0" is an index; "5070 Ti 16GB" is the
 * thing you are actually reasoning about when you wonder whether a run will fit
 * — rfdetr-nano died on card 0 today for exactly that reason.
 *
 * TRAIN_OPTS is train.js's cached /api/train/options, which already carries
 * name and vram_gb per card (the Train picker shows them). Reading it here keeps
 * one source; degrading to a bare "card N" when it has not loaded yet is right,
 * because a wrong GPU name is worse than no GPU name. */
/* STRIP ANSI. rfdetr's trainer draws `rich` tables, so its log arrives full of
 * CSI escape sequences and the panel rendered them literally - every number in
 * its epoch tables came out wrapped in visible bracket codes and the block was
 * unreadable. Caught on a screenshot; the DOM measurements said nothing.
 *
 * textContent still writes the body - it is raw process output and must never
 * be parsed as markup. This removes only the escapes themselves.
 *
 * Broad CSI match, not only the colour form: rich also emits cursor moves and
 * line erases, and a stray one leaves an orphan bracket behind. */
const _RL_ANSI = /\u001b\[[0-9;?]*[ -\/]*[@-~]/g;
function _rlClean(t) { return String(t || "").replace(_RL_ANSI, ""); }

function _rlCardName(idx, machineKey) {
  /* Card names come from the RUN'S OWN machine (its status row carries the
   * key), never a hardcoded machine name — the old literal broke the moment
   * the fleet was not the author's. Falls back across every machine block so
   * a missing key degrades to a bare index rather than a blank. */
  if (typeof TRAIN_OPTS === "undefined" || !TRAIN_OPTS) return "";
  const models = TRAIN_OPTS.models || [];
  const blocks = [];
  for (const m of models){
    const per = m.machines || {};
    if (machineKey && per[machineKey]) blocks.push(per[machineKey]);
    else blocks.push(...Object.values(per));
  }
  for (const blk of blocks){
    const c = (blk.cards || []).find(x => x.index === idx);
    if (c) return ` — ${String(c.name).replace(/^NVIDIA GeForce /, "")} ${c.vram_gb}GB`;
  }
  return "";
}

function _rlPaint(blocks) {
  const el = _rlEl();
  if (!el) return;
  if (!blocks.length) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false;
  el.textContent = "";
  for (const [title, body] of blocks) {
    const wrap = document.createElement("div");
    wrap.className = "runLogBlock";
    const h = document.createElement("div");
    h.className = "runLogTitle";
    // textContent for BOTH, never innerHTML: the title carries a run name and
    // the body is raw process output — file paths and exception text, any of
    // which can contain something that parses as markup.
    h.textContent = title;
    const pre = document.createElement("pre");
    pre.className = "log runLogBody";
    pre.textContent = _rlClean(body);
    wrap.append(h, pre);
    el.append(wrap);
    pre.scrollTop = pre.scrollHeight;
  }
}


/* How stale, in words. The distinction that matters is not the exact number but
 * whether it is the normal proxy lag or something has stopped arriving. */
function _rlAge(age) {
  if (age == null) return "";
  if (age < 12) return `${age.toFixed(0)}s behind`;
  if (age < 60) return `${age.toFixed(0)}s behind (proxy lagging)`;
  return `${Math.round(age / 60)}min behind — the worker may not be answering`;
}

window.runLogTick = async function () {
  const el = _rlEl();
  // A TAB IS HIDDEN BY THE `active` CLASS (.tab{display:none}), not the
  // hidden attribute — this guard was never true, so the tick paid for
  // a fetch on every heartbeat while the tab was closed.
  if (!el || !el.closest("section.tab")?.classList.contains("active")) return;

  // BOTH CARDS, NOT THE FIRST ONE FOUND. This used to pick a single source in
  // priority order and return — fine when one card trained, wrong the moment
  // two do: yolo on card 0 won every time and the challenger on card 1 was
  // invisible, with no hint that a second run even existed. Reported from the
  // UI, which is the only place it shows.
  const parts = [];
  // PUBLISHED, not recomputed elsewhere: run_vis.js needs the same answer to
  // "what is training right now", and a second copy of that rule is exactly
  // the drift this repo pays for most. Explicitly on window — a top-level
  // const in a classic script is NOT a window property (see RUNLOG above).
  const live = [];

  // 1. a local run — first-hand, no staleness at all.
  let local = null;
  try { local = await api("/api/retrain"); } catch (e) { /* HQ restarting */ }
  if (local && local.status === "running") {
    // BRACES, deliberately. This `if` was brace-less, so appending a second
    // statement to it silently ran that statement unconditionally — LIVE_RUNS
    // then carried a phantom "this machine" run on every tick, which made
    // run_vis think two runs were training and halve each one's thumbnails.
    parts.push([`this machine · yolov8n${local.next_name ? " → " + local.next_name : ""}`
                + ` · ${local.step || "training"} · live`,
                local.log_tail || "(no output yet)"]);
    live.push({title: `this machine · yolov8n`, run: local.next_name});
  }

  // 2. the worker's yolo run, through the proxy that already carries log_tail.
  // NOT ON THE LAB. /api/lab_status is HQ watching the worker, and state.py answers
  // it with a deliberate 400 there ("the lab doesn't watch itself"). Asking
  // anyway put a failed request in the worker's console every 8 seconds. IS_LAB is
  // undefined until the first state poll, so one early call still slips through
  // — that is the point at which we genuinely do not know yet.
  let lab = null;
  if (!window.IS_LAB) {
    try { lab = await api("/api/lab_status"); } catch (e) { /* ignore */ }
  }
  if (lab && lab.ok && lab.status === "running") {
    const ep = lab.epoch != null ? ` · epoch ${lab.epoch}/${lab.total ?? "?"}` : "";
    // NAME THE MODEL. "card 0" says where, not what — and with a challenger
    // running beside it, which family produced which log is the whole question.
    // next_name is the run this retrain will be adopted as.
    const who = lab.next_name ? `yolov8n → ${lab.next_name}` : "yolov8n";
    parts.push([`card 0${_rlCardName(0)} · ${who} · ${lab.step || "training"}${ep} · `
                + `${_rlAge(lab.age_s)}`, lab.log_tail || "(no output yet)"]);
    live.push({title: `card 0${_rlCardName(0)} · ${who}`, run: lab.next_name});
  }

  // 3. a challenger, which shares the box with the above rather than replacing it.
  /* ONE TERMINAL PER LIVE CHALLENGER. This read TRAIN_BUSY.challenger, a single
   * run, so with two challengers training the second had no log panel at all. */
  const runs = (typeof TRAIN_BUSY !== "undefined")
    ? (TRAIN_BUSY.challengers && TRAIN_BUSY.challengers.length
        ? TRAIN_BUSY.challengers
        : (TRAIN_BUSY.challenger ? [{run: TRAIN_BUSY.challenger, card: 1}] : []))
    : [];
  for (const {run, card} of runs) {
    try {
      const r = await api(`/api/lab/log?run=${encodeURIComponent(run)}`);
      // MATCH ON THE REGISTRY PREFIXES, longest first — the same rule
      // registry.for_run uses server-side. Matching on the model key instead
      // labelled a plain deimv2-n run as "deimv2-n-tv28": tv28's key minus its
      // suffix is also "deimv2-n", and being longer it matched first.
      const fam = (TRAIN_OPTS?.models || [])
        .flatMap(m => (m.prefixes || [m.key]).map(px => [px, m.key]))
        .sort((a, b) => b[0].length - a[0].length)
        .find(([px]) => run.startsWith(px))?.[1]
        || run.split("-").slice(0, 2).join("-");
      parts.push([`card ${card}${_rlCardName(card)} · ${fam} → ${run}`
                  + `${r.remote ? " · " + _rlAge(r.age_s) : " · live"}`,
                  r.tail || "(no output yet — a challenger writes nothing until "
                          + "its first epoch closes)"]);
    } catch (e) { /* a missing challenger log must not blank the panel */ }
    live.push({title: `card ${card}${_rlCardName(card)} · ${run}`, run});
  }
  window.LIVE_RUNS = live;

  if (parts.length) {
    // Newest-relevant first, and each block trimmed so two runs still fit on a
    // phone. The header carries which CARD, because "The worker" alone stopped
    // being enough the moment both of its GPUs could be busy.
    // Fewer lines each when two runs share the panel: the goal is BOTH
    // visible at once, and a body that is mostly scrollbar defeats that.
    const per = parts.length > 1 ? 12 : 40;
    _rlPaint(parts.map(([h, body]) =>
      [h, body.split("\n").slice(-per).join("\n")]));
    RUNLOG.src = parts.length > 1 ? "both" : "one";
    return;
  }

  // 4. nothing running — keep the last LOCAL run's tail as the post-mortem.
  if (local && local.log_tail && (local.status === "done" || local.status === "error")) {
    _rlPaint([[`this machine · ${local.status === "error" ? "⚠ " : "✅ "}`
               + `${local.step || local.status} · finished`, local.log_tail]]);
    RUNLOG.src = "local-finished";
  } else {
    _rlPaint([]);
    RUNLOG.src = null;
  }
};
