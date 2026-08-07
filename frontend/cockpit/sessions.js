"use strict";
/* Where this account is signed in, and how to sign it out from here.
 *
 * BEHIND THE SESSION COUNT, because that number was already on the Account page
 * and already the question: "3 active sessions" invites "which three?" and had
 * no answer. Making the count the control costs no new surface and puts the
 * detail exactly where the question is asked.
 *
 * THE CURRENT BROWSER CANNOT BE REVOKED HERE. The server refuses it (409) and
 * so does this — but the row is still SHOWN, marked, because a list of your
 * sessions that silently omits the one you are using is a list that does not
 * add up. Sign out is the button for leaving; it is elsewhere and it says so.
 *
 * IDS ARE HASH PREFIXES, not tokens. sessions.listing hands out twelve hex
 * characters of the stored SHA-256 — enough to tell four rows apart, useless as
 * a credential, and revoking still requires being signed in as this account.
 */

const _seEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const _seWhen = ts => {
  if (!ts) return "—";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "active now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 90) return `${Math.round(m)} min ago`;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(d)} d ago`;
};

const _seDays = secs => secs
  ? `${Math.max(0, Math.round(secs / 86400))} d left` : "";

async function openSessions() {
  const modal = document.getElementById("sessModal");
  if (!modal) return;
  modal.hidden = false;
  document.getElementById("sessX")?.focus();
  await _seRender();
}

function closeSessions() {
  const modal = document.getElementById("sessModal");
  if (modal) modal.hidden = true;
}

async function _seRender() {
  const box = document.getElementById("sessList");
  if (!box) return;
  box.innerHTML = `<p class="hint">Loading…</p>`;
  let d;
  try { d = await apiOrThrow("/api/sessions"); }
  catch (e) {
    box.innerHTML = `<p class="hint">Could not read sessions: ${_seEsc(e.message || e)}</p>`;
    return;
  }
  const rows = d.sessions || [];
  const others = rows.filter(r => !r.current).length;
  box.innerHTML = rows.map(r => `
    <div class="sessRow${r.current ? " current" : ""}">
      <span class="sessWho">
        <b>${_seEsc(r.agent || "unknown")}</b>
        <span class="sessId gwNum" title="session id">${_seEsc(r.id)}</span>
      </span>
      <span class="sessWhen">${_seEsc(_seWhen(r.last_seen))}</span>
      <span class="sessMeta gwNum">${_seEsc(r.ip || "")}${
        r.ip && _seDays(r.expires_in) ? " · " : ""}${_seEsc(_seDays(r.expires_in))}</span>
      ${r.current
        ? `<span class="sessHere">this browser</span>`
        : `<button type="button" class="keyKill sessKill"
             data-id="${_seEsc(r.id)}">Revoke</button>`}
    </div>`).join("") || `<p class="hint">No live sessions.</p>`;

  const all = document.getElementById("sessRevokeAll");
  if (all) {
    all.hidden = others === 0;
    all.textContent = `Sign out the other ${others} session${others === 1 ? "" : "s"}`;
  }
  box.querySelectorAll(".sessKill").forEach(b => b.onclick = () => _seRevoke(b.dataset.id));
}

async function _seRevoke(id) {
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`,
                          {method: "DELETE"});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || "could not revoke it");
  } catch (e) {
    alert(e.message || String(e));
  }
  await _seRender();
  if (window.loadAccount) loadAccount();     // the count on the page changes too
}

(() => {
  const modal = document.getElementById("sessModal");
  if (!modal) return;
  document.getElementById("sessX").onclick = closeSessions;
  modal.onclick = e => { if (e.target === modal) closeSessions(); };
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !modal.hidden) closeSessions();
  });
  const all = document.getElementById("sessRevokeAll");
  if (all) all.onclick = async () => {
    if (!confirm("Sign out every other browser signed in as you?\n\n"
               + "This one stays signed in.")) return;
    try {
      const r = await fetch("/api/sessions/revoke-others", {method: "POST"});
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || "could not sign them out");
    } catch (e) { alert(e.message || String(e)); }
    await _seRender();
    if (window.loadAccount) loadAccount();
  };
})();

window.openSessions = openSessions;
