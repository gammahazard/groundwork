"use strict";
/* Machines: where runs can execute, and what each one actually has.
 *
 * ABOUT THE INSTALLATION, not about you — which is why it is a sibling of
 * Account rather than a panel inside it, the same way Data collection is.
 *
 * EVERY CAPABILITY SHOWN HERE IS MEASURED, never asserted. `sm_120` against a
 * venv's compiled arch list is what decides whether a model can run on a card;
 * a hardcoded "compatible cards" list would be wrong the first time a venv is
 * rebuilt on a different CUDA. So the page shows what the machine reported and
 * when, and an unprobed machine says so rather than showing an empty shelf that
 * reads as "no GPUs".
 *
 * THE TWO CREDENTIALS ARE DIFFERENT THINGS and the form says so. The API key is
 * the control plane — probe, dispatch, status — and is minted ON that machine.
 * ssh is the data plane: mirror.py rsyncs the dataset there and adopt_run pulls
 * finished runs back, and neither becomes HTTP, because rsync's incremental
 * delta is why a pre-train sync costs a second instead of re-uploading 7 GB.
 */

const _mEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const _mAgo = ts => {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 90) return `${Math.round(m)} min ago`;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(d)} d ago`;
};

/* A card row. The sm and the VRAM are the two facts that decide everything
 * downstream, so both are shown as figures rather than described. */
/* NAMED _mCardHTML, not _cardHTML: projects.js already owns that name at top
 * level, and these are classic scripts sharing ONE global lexical scope — a
 * duplicate declaration throws and the whole second file never loads. Caught
 * by check_frontend, which exists for exactly this. */
function _mCardHTML(c) {
  return `<div class="mCard">
    <span class="mCardIx">card ${c.index}</span>
    <span class="mCardName">${_mEsc(c.name)}</span>
    <span class="mCardSpec gwNum">${_mEsc(c.sm || "?")} · ${c.vram_gb ?? "?"} GB</span>
  </div>`;
}

function _machineHTML(m) {
  const cards = (m.cards || []).length
    ? (m.cards || []).map(_mCardHTML).join("")
    /* NOT AN EMPTY SHELF. Zero cards is ambiguous — it could mean "no GPUs" or
     * "nobody has asked yet" — and those need completely different actions. The
     * second is by far the more likely for a machine you just added. */
    : `<p class="hint mNoCards">Not probed yet — press <b>Probe</b> to ask it
        what it has. Until then no model can be dispatched to it, because
        nothing knows which cards it holds.</p>`;
  const acts = [
    `<button type="button" class="mProbe" data-key="${_mEsc(m.key)}">Probe</button>`,
    m.builtin ? "" :
      `<button type="button" class="mKill keyKill" data-key="${_mEsc(m.key)}">Remove</button>`,
  ].join("");
  return `<div class="mBox" data-key="${_mEsc(m.key)}">
    <div class="mHead">
      <span class="mName">${_mEsc(m.name)}</span>
      ${m.local ? `<span class="mTag">this machine</span>` : ""}
      ${m.builtin ? `<span class="mTag">built in</span>` : ""}
      ${!m.trains ? `<span class="mTag mTagOff">not a training target</span>` : ""}
      <span class="mSpacer"></span>
      <span class="mActs">${acts}</span>
    </div>
    <div class="mMeta gwNum">${_mEsc(m.url || "local")}
      · measured ${_mAgo(m.measured)}
      · ${m.has_key ? "key stored" : (m.local || m.builtin ? "" : "no API key")}
      ${m.ssh_host ? "· ssh " + _mEsc(m.ssh_host) : ""}</div>
    <div class="mCards">${cards}</div>
    <div class="mMsg" id="mMsg-${_mEsc(m.key)}"></div>
  </div>`;
}

async function loadMachines() {
  return loadInto("#machineList", async () => {
    const d = await apiOrThrow("/api/machines");
    return (d.machines || []).map(_machineHTML).join("");
  }, {
    label: "machines", kind: "card", n: 2,
    after: host => {
      host.querySelectorAll(".mProbe").forEach(b => b.onclick = () => _probe(b.dataset.key, b));
      host.querySelectorAll(".mKill").forEach(b => b.onclick = () => _remove(b.dataset.key));
    },
  });
}

async function _probe(key, btn) {
  const msg = document.getElementById(`mMsg-${key}`);
  const was = btn.textContent;
  btn.disabled = true;
  // SAY IT MAY BE SLOW, because it is: reading each venv's compiled arch list
  // means starting that interpreter and importing torch. Silence for ten seconds
  // reads as a dead button.
  btn.textContent = "Asking…";
  if (msg) msg.textContent = "Reading its cards and venvs — this takes a few seconds.";
  try {
    const r = await fetch(`/api/machines/${encodeURIComponent(key)}/probe`,
                          {method: "POST"});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "probe failed");
    // A machine that is TRAINING answers with its previous cards, flagged. Say
    // that rather than showing the numbers as though they were just measured.
    if (msg) msg.textContent = j.cards_stale
      ? `Cards not re-read: ${j.why || "it is busy"}`
      : `Found ${(j.cards || []).length} card(s).`;
    // UPDATE JUST THIS BOX IN PLACE — loadMachines() re-renders the whole list
    // and flashes a loading skeleton over every machine, which reads as the
    // panel breaking and rebuilding after a 10s wait. Swap only the probed
    // box's card rows and let them fade in; nothing else on screen moves.
    const box = document.querySelector(`.mBox[data-key="${(window.CSS && CSS.escape) ? CSS.escape(key) : key}"]`);
    const cardsEl = box && box.querySelector(".mCards");
    if (cardsEl && !j.cards_stale && (j.cards || []).length) {
      cardsEl.innerHTML = j.cards.map(_mCardHTML).join("");
      cardsEl.classList.remove("mCardsIn");
      void cardsEl.offsetWidth;          // restart the animation on a re-probe
      cardsEl.classList.add("mCardsIn");
    } else if (!box) {
      loadMachines();                    // box gone (rare) — fall back to a full refresh
    }
  } catch (e) {
    if (msg) { msg.textContent = e.message || String(e); msg.classList.add("bad"); }
  } finally {
    btn.disabled = false; btn.textContent = was;
  }
}

async function _remove(key) {
  if (!confirm(`Forget ${key}? Runs already adopted from it stay; it just stops `
             + `being a place you can train.`)) return;
  let r = await fetch(`/api/machines/${encodeURIComponent(key)}`, {method: "DELETE"});
  if (r.status === 409) {
    const j = await r.json();
    /* THE REFUSAL IS THE POINT, so it is quoted rather than summarised: it says
     * either "it is training" or "I could not reach it to check", and those need
     * different decisions from the person reading them. */
    if (!confirm(`${j.detail}\n\nRemove it anyway?`)) return;
    r = await fetch(`/api/machines/${encodeURIComponent(key)}?force=true`,
                    {method: "DELETE"});
  }
  if (!r.ok) alert((await r.json()).detail || "could not remove it");
  loadMachines();
}

/* PAIRING — the one-paste path. The worker's admin mints a GW1.… string on
 * that machine; pasting it here registers, records host keys, pushes HQ's ssh
 * pubkey, dry-run tests the data plane and probes cards, reporting each step.
 * The four-field form below survives as the manual/advanced path. */
(() => {
  const f = document.getElementById("pairForm");
  if (!f) return;
  f.onsubmit = async e => {
    e.preventDefault();
    const msg = $("#pairMsg");
    msg.textContent = "pairing…"; msg.classList.remove("bad");
    try {
      const r = await fetch("/api/machines/pair", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({code: $("#pairCode").value.trim()})});
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "pairing failed");
      const bits = [`registered ${j.key}`];
      if (j.steps && j.steps.host_keys != null)
        bits.push(`${j.steps.host_keys} host key(s) recorded`);
      if (j.steps && j.steps.pubkey_installed) bits.push("ssh key installed");
      if (j.test) bits.push(j.test.ok ? "data plane ✓"
        : `data plane ✗ (${j.test.step}: ${j.test.error})`);
      if (j.probe) bits.push(`${(j.probe.cards || []).length} card(s)`);
      msg.textContent = bits.join(" · ");
      if (j.manual_step) {
        msg.textContent += " — one manual step remains:";
        const pre = document.createElement("pre");
        pre.className = "hint"; pre.textContent = j.manual_step;
        msg.after(pre);
      }
      f.reset();
      loadMachines();
    } catch (err) {
      msg.textContent = err.message || String(err); msg.classList.add("bad");
    }
  };
})();

/* WORKER SIDE: mint the pairing code to paste at an HQ. The card renders only
 * when this machine runs as a worker (the server said so via /api/machine/self
 * — identity.js stashes it on window.SELF_ROLE at boot). */
(async () => {
  const btn = document.getElementById("mintPairBtn");
  if (!btn) return;
  const card = document.getElementById("workerPairCard");
  if (card) {
    try {   // the card renders only on a worker — ask the machine itself
      const me = await fetch("/api/machine/self").then(r => r.ok ? r.json() : null);
      if (me && me.role === "worker") {
        card.hidden = false;
        // Prefill the reachable URL with how THIS BROWSER reached the box —
        // the one address guaranteed to work from the human's side of the
        // network. Self-IP guesses pick docker/VPN interfaces on multi-homed
        // machines; the field stays editable for the cases where the HQ
        // reaches it differently.
        const inp = $("#mintPairUrl");
        if (inp && !inp.value)
          inp.value = location.origin;
      }
    } catch (e) { /* signed-out or unreachable: stay hidden */ }
  }
  btn.onclick = async () => {
    const out = $("#mintPairOut");
    out.textContent = "minting…";
    try {
      const r = await fetch("/api/machine/pairing-code", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({url: $("#mintPairUrl").value.trim() || null})});
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "refused");
      out.textContent = j.code;
      out.title = "paste this on the HQ's Machines tab";
    } catch (err) { out.textContent = err.message || String(err); }
  };
})();

(() => {
  const f = document.getElementById("machineForm");
  if (!f) return;
  f.onsubmit = async e => {
    e.preventDefault();
    const msg = $("#machineFormMsg");
    msg.textContent = ""; msg.classList.remove("bad");
    const body = {
      name: $("#mfName").value.trim(), url: $("#mfUrl").value.trim(),
      api_key: $("#mfKey").value.trim(), ssh_host: $("#mfSsh").value.trim(),
      remote_root: $("#mfRoot").value.trim(),
    };
    try {
      const r = await fetch("/api/machines", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)});
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "could not add it");
      f.reset();
      msg.textContent = `Added ${j.key}. Probe it to find its cards.`;
      loadMachines();
    } catch (err) {
      msg.textContent = err.message || String(err); msg.classList.add("bad");
    }
  };
})();

/* The one-command join card: mint on click, show, copy. */
(() => {
  const mint = document.getElementById("joinMint");
  if (!mint) return;
  const out = document.getElementById("joinOut");
  const copy = document.getElementById("joinCopy");
  mint.onclick = async () => {
    mint.disabled = true;
    try {
      const r = await apiOrThrow("/api/machines/join-token", {method: "POST"});
      out.textContent = r.command;
      out.hidden = false;
      copy.hidden = false;
      copy.onclick = () => navigator.clipboard?.writeText(r.command)
        .then(() => { copy.textContent = "Copied ✓"; });
    } catch (e) {
      out.textContent = e.message || String(e);
      out.hidden = false;
    } finally { mint.disabled = false; }
  };
})();

window.loadMachines = loadMachines;
