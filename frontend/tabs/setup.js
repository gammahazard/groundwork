"use strict";
/* The FIRST-RUN WIZARD. Shown instead of the login form exactly while the
 * instance has zero accounts (auth.js asks /api/setup/status before showing
 * login). State is DERIVED per step from /api/setup/facts — nothing stored,
 * nothing to go stale, and after the claim it is ordinary admin traffic.
 *
 * SHARED-LEXICAL rules: declares only setup-prefixed top-level names; reads
 * loadMe/enterApp from auth.js (loaded earlier); called from auth.js via
 * window.maybeShowSetup. */

async function maybeShowSetup() {
  /* Returns true when the wizard took the screen (auth.js then skips login). */
  try {
    const r = await fetch("/api/setup/status");
    if (!r.ok) return false;
    const s = await r.json();
    if (!s.needed) return false;
    document.body.classList.remove("authUnknown");
    document.body.classList.add("signedOut");
    const sec = document.querySelector("#setupWizard");
    const login = document.querySelector("#login");
    if (login) login.hidden = true;
    if (!sec) return false;
    sec.hidden = false;
    _setupClaimStep(sec, s);
    return true;
  } catch (e) { return false; }
}
window.maybeShowSetup = maybeShowSetup;

function _setupClaimStep(sec, s) {
  sec.querySelector("#suMachine").textContent = s.machine || "this machine";
  const form = sec.querySelector("#suClaimForm");
  const msg = sec.querySelector("#suClaimMsg");
  form.onsubmit = async e => {
    e.preventDefault();
    const u = sec.querySelector("#suUser").value.trim();
    const p1 = sec.querySelector("#suPass").value;
    const p2 = sec.querySelector("#suPass2").value;
    if (p1 !== p2) { msg.textContent = "passwords do not match"; return; }
    msg.textContent = "creating…";
    const r = await fetch("/api/setup/claim", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({username: u, password: p1})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { msg.textContent = j.detail || "refused"; return; }
    msg.textContent = "";
    sec.querySelector("#suStepClaim").hidden = true;
    sec.querySelector("#suStepRest").hidden = false;
    _setupRest(sec);
  };
}

async function _setupRest(sec) {
  /* Post-claim: signed in as the new admin. Render the derived checklist and
   * keep it live while probes/downloads run. */
  const paint = async () => {
    const f = await fetch("/api/setup/facts").then(r => r.json())
      .catch(() => null);
    if (!f) return;
    const gpu = sec.querySelector("#suGpu");
    gpu.innerHTML = f.probed
      ? "✓ " + (f.cards.length
          ? f.cards.map(c => `${c.name} ${c.vram_gb ?? "?"}GB`).join(" · ")
          : (f.train_env
              ? "no GPU found — you can label and manage projects; training "
                + "needs a GPU (add a machine later under Machines)"
              : "no training environment on this machine — labeling and "
                + "projects work now; to train, install the train extra "
                + "(pip install 'groundwork[train]') or add a GPU machine "
                + "under Machines"))
      : "⏳ probing this machine…";
    const pr = sec.querySelector("#suProject");
    if (f.projects.length) {
      pr.innerHTML = `✓ project <b>${f.projects[0]}</b> exists`;
      sec.querySelector("#suProjForm").hidden = true;
    }
    const la = sec.querySelector("#suLaState");
    const st = (f.extras || {}).la || {};
    if (f.la_present) la.textContent = "✓ installed";
    else if (st.state === "running") la.textContent = "⏳ downloading…";
    else if (st.state === "error") {
      la.textContent = "✗ " + (st.error || "failed");
      if (st.needs_token) sec.querySelector("#suHfTokRow").hidden = false;
    }
    return f;
  };
  await paint();
  const t = setInterval(async () => {
    const f = await paint();
    if (f && f.probed && f.projects.length) { /* keep polling for extras */ }
  }, 3000);
  sec.dataset.timer = String(t);

  sec.querySelector("#suProjForm").onsubmit = async e => {
    e.preventDefault();
    const name = sec.querySelector("#suProjName").value.trim();
    const classes = sec.querySelector("#suProjClasses").value.trim()
      .split(",").map(x => x.trim()).filter(Boolean);
    const msg = sec.querySelector("#suProjMsg");
    msg.textContent = "creating…";
    const r = await fetch("/api/projects", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name, classes: classes.length ? classes : ["object"]})});
    const j = await r.json().catch(() => ({}));
    msg.textContent = r.ok ? "" : (j.detail || "refused");
    if (r.ok) paint();
  };

  sec.querySelector("#suLaBtn").onclick = async () => {
    const tok = sec.querySelector("#suHfTok").value.trim();
    const r = await fetch("/api/setup/extras/la", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({accept_license: sec.querySelector("#suLaOk").checked,
                            hf_token: tok || null, remember_token: false})});
    const j = await r.json().catch(() => ({}));
    sec.querySelector("#suLaState").textContent =
      r.ok ? "⏳ downloading…" : (j.detail || "refused");
  };

  sec.querySelector("#suDone").onclick = () => {
    clearInterval(Number(sec.dataset.timer || 0));
    sec.hidden = true;
    // Ordinary signed-in boot from here — same path as a fresh login.
    location.reload();
  };
}
