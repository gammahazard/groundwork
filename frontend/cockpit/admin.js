"use strict";
/* Admin: the accounts on this machine, and the security trail.
 *
 * ITS OWN FILE, and its own tab, because it is the only screen here whose
 * subject is OTHER PEOPLE. Account is you, API is your keys — this is the
 * household, and mixing it into either would put "delete this person" on a page
 * someone opens to change their own password.
 *
 * EVERYTHING HERE IS ADMIN-ONLY SERVER-SIDE. /api/users, /api/users/stats and
 * /api/audit all pass through require_admin, which also refuses API-key auth —
 * so a key in a config file can never read the trail that would show it being
 * used. Hiding the tab for non-admins is a courtesy; the refusal is the server's.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: show anyone else's sessions. An admin can
 * end another account's sessions by deleting it, which is the power that
 * matters; watching where a person is signed in from is a different thing and
 * not one this cockpit needs.
 */

const _adEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const _adWhen = ts => {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 90) return `${Math.round(m)} min ago`;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(d)} d ago`;
};

const _adClock = ts => ts
  ? new Date(ts * 1000).toLocaleString(undefined,
      {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
       second: "2-digit"})
  : "";

/* How each event reads in the trail. A closed vocabulary on the server means
 * this can be a lookup rather than string-munging, and an event this file has
 * not been taught still renders — as its raw name, which is honest. */
const AD_EVENTS = {
  "login.ok":         {label: "signed in",         kind: "ok"},
  "login.fail":       {label: "failed sign-in",     kind: "bad"},
  "login.throttled":  {label: "blocked — too many", kind: "bad"},
  "logout":           {label: "signed out",         kind: ""},
  "password.changed": {label: "password changed",   kind: "warn"},
  "username.changed": {label: "username changed",   kind: "warn"},
  "user.created":     {label: "account created",    kind: "warn"},
  "user.deleted":     {label: "account deleted",    kind: "warn"},
  "key.minted":       {label: "API key created",    kind: "warn"},
  "key.revoked":      {label: "API key revoked",    kind: ""},
  "session.revoked":  {label: "session revoked",    kind: ""},
  "denied":           {label: "refused",            kind: "bad"},
};

async function loadAdmin() {
  window._axState?.();
  await Promise.all([_adUsers(), _adTrail()]);
}

/* ------------------------------------------------------------- accounts --- */

async function _adUsers() {
  const box = document.getElementById("adUsers");
  if (!box) return;
  let d;
  try { d = await apiOrThrow("/api/users/stats"); }
  catch (e) {
    box.innerHTML = `<p class="hint">${_adEsc(e.message || e)}</p>`;
    return;
  }
  const me = (window.currentUser && currentUser() || {}).username;
  const rows = d.users || [];
  box.innerHTML = rows.map(u => {
    const isMe = u.username === me;
    return `<div class="adUser">
      <div class="adUserTop">
        <span class="adName">${_adEsc(u.username)}</span>
        ${u.admin ? `<span class="adTag">admin</span>` : ""}
        ${isMe ? `<span class="adTag adTagMe">you</span>` : ""}
        ${u.must_change ? `<span class="adTag adTagWarn">must change password</span>` : ""}
        <span class="mSpacer"></span>
        ${isMe ? "" : `<button type="button" class="keyKill adKill"
                        data-user="${_adEsc(u.username)}">Remove</button>`}
      </div>
      <div class="adUserMeta gwNum">
        <!-- ACTIVE, then SIGNED IN. Two different facts and the first is the
             one that answers "are they using this" — a 30-day session means
             last_login can be weeks old while the account is in daily use. -->
        active ${_adWhen(u.last_active || u.last_login)}
        · signed in ${_adWhen(u.last_login)}
        ${u.created ? `· joined ${_adWhen(u.created)}` : ""}
        · ${u.sessions} session${u.sessions === 1 ? "" : "s"}
        · ${u.keys} key${u.keys === 1 ? "" : "s"}${u.keys_expired
            ? ` (${u.keys_expired} expired)` : ""}
        ${u.failed_logins
          ? ` · <span class="adBad">${u.failed_logins} failed sign-in${
                u.failed_logins === 1 ? "" : "s"}</span>` : ""}
      </div>
      ${(u.last_device || u.last_ip) ? `<div class="adUserMeta gwNum adQuiet">
        last from ${_adEsc(u.last_device || "unknown")}${
          u.last_ip ? ` · ${_adEsc(u.last_ip)}` : ""}${
          u.devices && u.devices.length > 1
            ? ` · ${u.devices.length} devices seen` : ""}${
          u.key_last_used ? ` · key used ${_adWhen(u.key_last_used)}` : ""}
      </div>` : ""}
      ${_adWork(u)}
    </div>`;
  }).join("") || `<p class="hint">No accounts.</p>`;

  box.querySelectorAll(".adKill").forEach(b => b.onclick = () => _adRemove(b.dataset.user));
}

/* WHAT THIS ACCOUNT ACTUALLY HAS. A row that says only "owns the first project" tells
 * an admin nothing about whether removing it would strand real work — 244
 * images and 42 runs is a different decision from an empty project. The numbers
 * come from the same collections the project page counts, so the two cannot
 * disagree.
 *
 * A <details>, closed. One account is a line; five accounts each with three
 * projects expanded is a page nobody scans. */
function _adWork(u) {
  if (!u.projects || !u.projects.length)
    return `<div class="adUserMeta gwNum adQuiet">owns no projects</div>`;
  const rows = u.projects.map(p => `
    <div class="adProj">
      <span class="adProjName">${_adEsc(p.name || p.slug)}</span>
      <span class="adProjNums gwNum">
        ${p.training} training · ${p.holdout} holdout
        · ${(p.objects || 0).toLocaleString()} objects
        · ${p.runs} run${p.runs === 1 ? "" : "s"}
        ${p.best_mae != null ? `· best ${(+p.best_mae).toFixed(4)}` : ""}
        ${p.needs_fix ? `· <span class="adBad">${p.needs_fix} to fix</span>` : ""}
      </span>
      <span class="adProjWhen gwNum">${p.last_run
        ? "last run " + _adWhen(p.last_run) : "never trained"}</span>
    </div>`).join("");
  return `<details class="adWork">
    <summary class="adWorkSum">${u.projects.length} project${
      u.projects.length === 1 ? "" : "s"} ·
      ${(u.images || 0).toLocaleString()} image${u.images === 1 ? "" : "s"} ·
      ${u.runs || 0} run${u.runs === 1 ? "" : "s"}</summary>
    <div class="adProjs">${rows}</div>
  </details>`;
}

async function _adRemove(username) {
  /* THE CONFIRMATION NAMES WHAT ELSE GOES. Deleting an account also ends its
   * sessions and revokes its keys — server-side, in the same call — and a
   * confirm that says only "delete?" hides the half that is irreversible. */
  if (!confirm(`Remove the account "${username}"?\n\n`
             + `Their sessions end immediately and their API keys are revoked. `
             + `Projects they own stay on disk, but nobody will own them.`)) return;
  try {
    const r = await fetch(`/api/users/${encodeURIComponent(username)}`,
                          {method: "DELETE"});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `could not remove ${username}`);
  } catch (e) {
    alert(e.message || String(e));
  }
  loadAdmin();
}

/* ---------------------------------------------------------------- trail --- */

async function _adTrail() {
  const box = document.getElementById("adTrail");
  const sum = document.getElementById("adSummary");
  if (!box) return;
  let d;
  try { d = await apiOrThrow("/api/audit?n=200"); }
  catch (e) {
    box.innerHTML = `<p class="hint">${_adEsc(e.message || e)}</p>`;
    return;
  }
  const s = d.summary || {};
  if (sum) {
    /* THE FAILURES LEAD, because they are the only number here that can mean
     * something is wrong. A count of successful sign-ins is context for them. */
    const blocked = (d.throttle && d.throttle.blocked || []).length;
    sum.innerHTML = `
      <div class="gwCt"><span class="v ${s.failed_logins ? "adBad" : ""}">${
        s.failed_logins ?? 0}</span><span class="k">failed sign-ins, 24 h</span></div>
      <div class="gwCt"><span class="v">${s.logins ?? 0}</span>
        <span class="k">successful sign-ins</span></div>
      <div class="gwCt"><span class="v">${s.admin_actions ?? 0}</span>
        <span class="k">accounts created or removed</span></div>
      <div class="gwCt"><span class="v ${blocked ? "adBad" : ""}">${blocked}</span>
        <span class="k">currently rate-limited</span></div>`;
  }
  const rows = d.events || [];
  box.innerHTML = rows.length ? rows.map(e => {
    const meta = AD_EVENTS[e.event] || {label: e.event, kind: ""};
    const who = e.actor || e.target || "—";
    return `<div class="adRow${meta.kind ? " ad-" + meta.kind : ""}">
      <span class="adTime gwNum" title="${_adEsc(_adClock(e.ts))}">${_adWhen(e.ts)}</span>
      <span class="adEvent">${_adEsc(meta.label)}</span>
      <span class="adWho">${_adEsc(who)}</span>
      <span class="adDev">${_adEsc(e.device || "")}</span>
      <span class="adFrom gwNum">${_adEsc(e.ip || "")}</span>
      <span class="adDetail">${_adEsc(e.detail || "")}</span>
    </div>`;
  }).join("") : `<p class="hint">Nothing recorded yet.</p>`;
}

/* ------------------------------------------------------------ new account --- */

(() => {
  const f = document.getElementById("adNewUser");
  if (!f) return;
  f.onsubmit = async e => {
    e.preventDefault();
    const msg = document.getElementById("adNewMsg");
    msg.textContent = ""; msg.classList.remove("bad");
    const body = {
      username: document.getElementById("adNewName").value.trim(),
      password: document.getElementById("adNewPw").value,
      admin: document.getElementById("adNewAdmin").checked,
    };
    if (body.password.length < 8) {
      msg.textContent = "at least 8 characters"; msg.classList.add("bad"); return;
    }
    try {
      const r = await fetch("/api/users", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)});
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || "could not create the account");
      f.reset();
      /* SAY WHAT TO DO NEXT. A new account is useless until the person can
       * reach the machine, and on a network that is a separate step nobody
       * guesses — so the success message names it. */
      msg.textContent = `Created ${j.user.username}. They also need to be on your `
                      + `network to reach this cockpit.`;
      loadAdmin();
    } catch (err) {
      msg.textContent = err.message || String(err); msg.classList.add("bad");
    }
  };
})();

/* Extras — the wizard's opt-in downloads, re-runnable here. */
(() => {
  const la = document.getElementById("axLaBtn");
  if (!la) return;
  const msg = $("#axMsg");
  /* EVERY BUTTON IS A ROW OF FACTS, not a hopeful click. Each knows the extras
   * key it starts and the label it wears; paint() below decides from the
   * server's facts whether it is offerable, already done, or mid-install. */
  const BTNS = [
    {el: la, key: "la", label: "⬇ Auto-labeler", done: f => f.la_present,
     ok: () => $("#axLaOk").checked, path: "/api/setup/extras/la",
     body: () => ({accept_license: $("#axLaOk").checked})},
    {el: $("#axStackDeim"), key: "stack:deim", label: "⛏ Enable DEIMv2",
     done: f => (f.stacks || {}).deim?.installed,
     ok: () => $("#axStackOk").checked, path: "/api/setup/extras/stack",
     body: () => ({stack: "deim", accept_license: $("#axStackOk").checked})},
    {el: $("#axStackRtm"), key: "stack:rtmdet",
     label: "⛏ Enable RTMDet · YOLOX · CenterNet",
     done: f => (f.stacks || {}).rtmdet?.installed,
     ok: () => $("#axStackOk").checked, path: "/api/setup/extras/stack",
     body: () => ({stack: "rtmdet", accept_license: $("#axStackOk").checked})},
  ].filter(b => b.el);

  let _poll = null;
  const post = async b => {
    b.el.disabled = true;
    msg.textContent = `starting ${b.key}…`; msg.classList.remove("bad");
    try {
      const r = await fetch(b.path, {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(b.body())});
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || "refused");
      /* THE CARD FOLLOWS THE JOB NOW. It used to say "progress shows here on
       * refresh" and then never change — an install that takes ten minutes
       * (a multi-GB torch download) looked identical to one that had died,
       * and a failure was only visible to someone who thought to reload. */
      msg.textContent = j.already ? `${b.key} is already running`
                                  : `${b.key}: installing…`;
      window._axState();
    } catch (e) {
      msg.textContent = `✗ ${e.message || e}`; msg.classList.add("bad");
      window._axState();
    }
  };
  for (const b of BTNS) b.el.onclick = () => post(b);

  /* State from the same facts endpoint the wizard reads — fetched when the
   * Admin tab actually loads, never at script parse: this file is parsed on
   * the login screen too, and a pre-auth fetch is a guaranteed 401 in the
   * console of every signed-out visitor. */
  window._axState = () => fetch("/api/setup/facts")
    .then(r => r.ok ? r.json() : null).then(f => {
      if (!f) return;
      const ex = f.extras || {}, bits = [];
      let anyRunning = false;
      for (const b of BTNS) {
        const st = (ex[b.key] || {}).state;
        const installed = !!b.done(f);
        const running = st === "running";
        anyRunning = anyRunning || running;
        /* ALREADY INSTALLED READS AS INSTALLED. Re-running is not harmful —
         * install() skips an existing venv — but a live button over a done
         * job is an invitation to sit through a download for nothing. */
        b.el.disabled = installed || running;
        b.el.textContent = installed ? `✓ ${b.label.replace(/^\S+\s/, "")} installed`
                         : running ? "installing…" : b.label;
        b.el.title = installed ? "already installed on this machine"
                   : running ? "installing now — this can take several minutes"
                   : "";
        if (installed) bits.push(`${b.key} ✓`);
        else if (st) bits.push(`${b.key}: ${st}`);
        /* A FAILURE MUST BE READABLE WITHOUT THE SERVER LOG. The state dict
         * carries the tail of the install's own output; the card is where
         * someone is standing when it fails. */
        if (st === "error" && !installed) {
          msg.textContent = `✗ ${b.key}: ${(ex[b.key].error
            || ex[b.key].log_tail || "install failed").toString().slice(-300)}`;
          msg.classList.add("bad");
        }
      }
      $("#axState").textContent = bits.join(" · ");
      /* Poll only while something is running — no timer on an idle tab. */
      clearTimeout(_poll);
      if (anyRunning) _poll = setTimeout(window._axState, 4000);
    }).catch(() => {});
})();

window.loadAdmin = loadAdmin;
