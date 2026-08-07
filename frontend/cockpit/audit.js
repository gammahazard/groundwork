"use strict";
/* Ask the serving model to check the training labels, and list the disagreements.
 *
 * WHAT IT IS FOR. Every training image's dot count was put there by a human, and
 * nothing has ever re-checked those 160 images — the frozen holdout is verified
 * by hand, the training set is not. Running the model over the same images gives
 * a second opinion, and where the two differ one of them is wrong. The
 * interesting case is when it is the LABEL, because a mislabelled training image
 * teaches the next model that mistake and nothing else in the system will ever
 * mention it.
 *
 * IT IS A SUSPICION, NOT A VERDICT, and the wording says so everywhere. The
 * model is wrong sometimes — that is the whole reason the fix queue exists — so
 * a disagreement means "look at this one". Worst disagreement first, because
 * that ordering is the entire value: the top of the list is where an hour of
 * checking pays.
 *
 * EVERY ROW OPENS THE EDITOR on that image. A list of stems you then have to go
 * and find is a list nobody acts on.
 */

const _aEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

/* STARTS the sweep and then POLLS. It used to be one blocking request for the
 * whole ~138 seconds, which is indistinguishable from a hang — and a reload
 * (which is what you do when something looks hung) started a second sweep on
 * the same card. See web/audit_job.py.
 *
 * The steps are shown as WORDS as well as a bar, because what it is doing
 * changes what waiting means: listing the collection is instant, the first
 * image carries the model load, and the rest is steady. A bar alone sits at 0
 * for five seconds and then leaps, which reads as broken. */
let _auditPoll = null;

async function runLabelAudit(btn, out) {
  btn.disabled = true;
  btn.textContent = "🔎 checking…";
  out.className = "hint auditOut";
  out.innerHTML = `<b>Starting…</b>`;
  try {
    await apiOrThrow("/api/label_audit?collection=raw", {method: "POST"});
  } catch (e) {
    out.className = "hint auditOut short";
    out.innerHTML = `✗ ${_aEsc(e.message || e)}`;
    btn.disabled = false;
    btn.textContent = "🔎 Check labels against the model";
    return;
  }
  clearInterval(_auditPoll);
  _auditPoll = setInterval(() => _auditTick(btn, out), 900);
  _auditTick(btn, out);
}

async function _auditTick(btn, out) {
  let s;
  try {
    s = await apiOrThrow("/api/label_audit");
  } catch (e) {
    return;                       // a dropped poll is not a failed audit
  }
  if (s.status === "running") {
    out.className = "hint auditOut";
    out.innerHTML = _auditProgressHTML(s);
    return;
  }
  clearInterval(_auditPoll);
  _auditPoll = null;
  btn.disabled = false;
  btn.textContent = "🔎 Check labels against the model";
  if (s.status === "error") {
    out.className = "hint auditOut short";
    out.innerHTML = `✗ ${_aEsc(s.error || "the audit failed")}`;
    return;
  }
  if (!s.result) return;
  out.className = "hint auditOut";
  out.innerHTML = _auditHTML(s.result, s.model);
  out.querySelectorAll(".auditOpen").forEach(a => {
    a.onclick = e => {
      e.preventDefault();
      // Arriving from HERE means you want to see the disagreement, so the
      // overlay loads itself (~1s). Opening the same image from the grid does
      // not — most of the time you are just labelling.
      if (window.auditWant) auditWant();
      if (window.openEditor) openEditor("raw", a.dataset.stem);
    };
  });
}

function _auditProgressHTML(s) {
  const pct = s.total ? Math.round(100 * s.done / s.total) : 0;
  const eta = s.eta_s != null && s.eta_s > 0
    ? ` · about ${s.eta_s < 60 ? s.eta_s + "s" : Math.round(s.eta_s / 60) + " min"} left`
    : "";
  // NAME THE MODEL WHILE IT RUNS. The audit compares against whatever is
  // PINNED, so "the model" is ambiguous the moment the pin moves — and two
  // audits of the same image legitimately disagreeing looks like a broken tool
  // unless you can see they used different weights.
  const who = s.model ? ` <span class="mMeta">· ${_aEsc(s.model)}</span>` : "";
  return `<b>Checking labels against the model</b>${who}
    <div class="pbar"><div class="pbarTrack">
      <div class="pbarFill" style="width:${pct}%"></div></div>
      <span class="mMeta">${s.total ? `${s.done} / ${s.total}` : "…"}${eta}</span>
    </div>
    <div class="mMeta auditStep">${_aEsc(s.step || "working…")}</div>`;
}

function _auditHTML(r, model) {
  const bad = r.flagged || [];
  const who = model ? ` <span class="mMeta">(${_aEsc(model)})</span>` : "";
  const head = `<b>${r.agreed}/${r.checked} images agree with the model.</b>${who}`;
  if (!bad.length)
    return head + ` <span class="mMeta">Nothing to look at — every dot count `
      + `matches what the model sees.</span>`
      + (r.errors.length ? `<div class="mMeta">${r.errors.length} unreadable</div>` : "");
  return head
    + ` <b>${bad.length}</b> disagree — one of the two is wrong, and it is worth `
    + `a look at the ones near the top.`
    + `<div class="mMeta">The model is not ground truth; this only says "check `
    + `this image".</div>`
    + `<table class="auditTable"><thead><tr><th>image</th><th>labelled</th>`
    + `<th>model</th><th>difference</th></tr></thead><tbody>`
    + bad.map(f => `<tr>
        <td><a href="#" class="auditOpen" data-stem="${_aEsc(f.stem)}">${_aEsc(f.stem)}</a></td>
        <td>${f.dots}</td><td>${f.model}</td>
        <td class="${f.diff > 0 ? "over" : "short"}">${f.diff > 0 ? "+" : ""}${f.diff}</td>
      </tr>`).join("")
    + `</tbody></table>`
    + (r.errors.length
        ? `<div class="mMeta">${r.errors.length} image(s) could not be read: `
          + r.errors.slice(0, 3).map(e => `${_aEsc(e.stem)} (${_aEsc(e.why)})`).join(", ")
          + `</div>`
        : "");
}

(() => {
  const btn = document.querySelector("#auditBtn");
  if (!btn) return;
  btn.onclick = () => runLabelAudit(btn, document.querySelector("#auditResult"));
})();
