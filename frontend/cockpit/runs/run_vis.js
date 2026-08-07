"use strict";
/* WHAT EACH LIVE RUN IS LOOKING AT — pictures, for every model, on either machine.
 *
 * WHAT IT REPLACES. "what it sees" existed twice and worked for neither case you
 * actually watch:
 *   - lab.js drew a #labVisWrap panel only inside the CHALLENGER view, so a yolo
 *     retrain — the thing training most nights — never got one;
 *   - the 🔬 Watch-training button is gated on HQ's own /api/retrain status, so
 *     once training moved to the worker it stayed hidden.
 * Same shape of bug run_log.js already fixed for logs: the panel followed the
 * MACHINE instead of the run. This follows the run.
 *
 * IT SAYS WHICH KIND OF PICTURE. "val predictions during training" and "the
 * augmented batches it trains on" are different things and reading one as the
 * other is misleading — mmdet draws the first, ultralytics the second, and DEIM
 * and the DETR families draw nothing at all while training. The caption comes
 * from the server (lab_ops._VIS_SOURCES), so one table decides it for both
 * machines rather than this file guessing from a filename.
 *
 * NO POLL AND NO RUN DISCOVERY OF ITS OWN. run_log.js already worked out what is
 * training this tick and publishes it as window.LIVE_RUNS; dashboard.js's 8s
 * heartbeat drives both. Rediscovering it here would mean a second copy of
 * "what counts as running" — the drift this repo pays for most.
 */

const RUNVIS = {key: ""};   // last painted image set, so a repaint is a no-op

function _rvEl() { return document.querySelector("#runVis"); }

/* Repaint ONLY when the picture set changes. The heartbeat is 8s and these are
 * <img> tags: rebuilding them every tick restarts every download, so a panel of
 * live batches would flicker permanently and never finish loading one. */
function _rvPaint(blocks) {
  const el = _rvEl();
  if (!el) return;
  const key = JSON.stringify(blocks);
  if (key === RUNVIS.key) return;
  RUNVIS.key = key;
  if (!blocks.length) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false;
  el.textContent = "";
  for (const b of blocks) {
    const wrap = document.createElement("div");
    wrap.className = "runVisBlock";
    const h = document.createElement("div");
    h.className = "runLogTitle";        // same header style as the log blocks
    h.textContent = b.title;
    wrap.append(h);
    // textContent, never innerHTML: the caption can arrive from the worker's
    // payload, and apps/static/ has no escaping helper anywhere.
    const cap = document.createElement("p");
    cap.className = "hint runVisCap";
    cap.textContent = b.caption;
    wrap.append(cap);
    if (b.images.length) {
      const grid = document.createElement("div");
      grid.className = "runVisGrid";
      for (const u of b.images) {
        const a = document.createElement("a");
        a.href = u; a.target = "_blank"; a.rel = "noopener";
        const img = document.createElement("img");
        img.loading = "lazy"; img.src = u;
        a.append(img); grid.append(a);
      }
      wrap.append(grid);
    }
    el.append(wrap);
  }
}

window.runVisTick = async function () {
  const el = _rvEl();
  if (!el || el.closest("section")?.hidden) return;
  const live = window.LIVE_RUNS || [];
  if (!live.length) { _rvPaint([]); return; }

  const blocks = [];
  for (const r of live) {
    if (!r.run) continue;
    let v = null;
    try { v = await api(`/api/lab/vis?run=${encodeURIComponent(r.run)}`); }
    catch (e) { continue; }   // one unreachable run must not blank the others
    const imgs = v.images || [];
    // FEWER EACH when two runs share the panel, for the same reason the log
    // bodies shrink: the point is both visible at once on a phone.
    const per = live.length > 1 ? 4 : 8;
    blocks.push({
      title: `${r.title} · what it sees`,
      caption: imgs.length
        ? (v.what || "")
        : (v.note || "no pictures yet — they appear once training writes some"),
      images: imgs.slice(0, per),
    });
  }
  _rvPaint(blocks);
};
