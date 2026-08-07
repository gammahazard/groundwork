"use strict";
/* The API reference: endpoints down the side, one of them open beside it.
 *
 * THE SHAPE IS THE POINT. A reference is not read top to bottom — you arrive
 * knowing roughly what you want ("start a run", "get a run's score") and need
 * to find it in one glance and read it in one screen. So the left rail lists
 * every documented endpoint grouped by job, and the right pane shows exactly
 * one: what it does, the parameters that change the outcome, a command that
 * runs, and what comes back. Nothing is behind a fold except the reference
 * itself.
 *
 * DEEP-LINKABLE, like the Guide, and for the same reason: "see the export
 * endpoint" should be a link you can send. #api/POST-api-export.
 *
 * THE HOST IS SUBSTITUTED, NOT ILLUSTRATED. Examples are meant to be run from
 * ANOTHER machine, and an operator sitting at this box has an origin of
 * localhost:8000 — the one address that does not need a key. /api/machine/self
 * reports the network address; $HOST in the examples becomes that.
 *
 * IT SAYS WHAT IT CANNOT DO. Every endpoint carries `auth`, and "session" means
 * an API key is refused — account and key management go through require_admin,
 * which rejects key auth outright. That asymmetry is the reason a key is safe to
 * paste into a config file, so the reference states it rather than implying it.
 */

let _apiAt = null;          // "METHOD path" of the open endpoint
let _apiHost = null;        // resolved once from /api/machine/self

const _apEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

/* A stable, URL-safe id for an endpoint. Built from method + path so it
 * survives reordering — the same rule the Guide's step keys follow. */
const _apId = e => `${e.method}-${e.path}`.replace(/[^A-Za-z0-9]+/g, "-")
                                          .replace(/^-|-$/g, "");

function _apAll() {
  return (window.API_GROUPS || []).flatMap(g =>
    g.endpoints.map(e => ({...e, group: g.title, groupKey: g.key})));
}

function _apFind(id) {
  return _apAll().find(e => _apId(e) === id) || null;
}

/* $HOST is replaced at RENDER time rather than baked into the content, so the
 * same reference is correct on HQ, on the worker, and on someone else's install. */
function _apSub(text) {
  return String(text ?? "").split("$HOST").join(_apiHost || "http://localhost:8000");
}

function _apRail() {
  return (window.API_GROUPS || []).map(g => `
    <div class="apGroup">
      <div class="apGroupHead">${_apEsc(g.title)}</div>
      ${g.blurb ? `<p class="apGroupBlurb">${_apEsc(g.blurb)}</p>` : ""}
      ${g.endpoints.map(e => `
        <button type="button" class="apItem" data-id="${_apId(e)}">
          <span class="apVerb v-${e.method}">${e.method}</span>
          <span class="apPath">${_apEsc(e.path)}</span>
        </button>`).join("")}
    </div>`).join("");
}

function _apParams(e) {
  if (!e.params || !e.params.length) return "";
  return `<h4 class="apH">Parameters</h4>
    <table class="apParams"><tbody>${e.params.map(([n, t, d]) => `
      <tr><td class="apPName"><code>${_apEsc(n)}</code></td>
          <td class="apPType">${_apEsc(t)}</td>
          <td class="apPDesc">${d}</td></tr>`).join("")}</tbody></table>`;
}

function _apDetail(e) {
  if (!e) return `<p class="hint">Pick an endpoint on the left.</p>`;
  /* AUTH IS A BADGE BECAUSE IT CHANGES WHAT YOU CAN BUILD. "session" is not a
   * detail — it means a script cannot do this at all, ever, by design. */
  const auth = e.auth === "session"
    ? `<span class="apBadge apBadgeNo" title="An API key is refused. Sign in with a password.">session only</span>`
    : `<span class="apBadge">API key</span>`;
  const mut = e.mutates
    ? `<span class="apBadge apBadgeMut" title="Changes data or starts work">writes</span>` : "";
  return `
    <div class="apHead">
      <span class="apVerb v-${e.method}">${e.method}</span>
      <code class="apPathBig">${_apEsc(e.path)}</code>
      ${auth}${mut}
    </div>
    <p class="apWhat">${_apEsc(e.what)}</p>
    ${e.detail ? `<div class="apDetail">${e.detail}</div>` : ""}
    ${_apParams(e)}
    ${e.example ? `<h4 class="apH">Example</h4>
      <div class="apCode"><code>${_apEsc(_apSub(e.example))}</code></div>` : ""}
    ${e.returns ? `<h4 class="apH">Returns</h4>
      <div class="apCode apCodeOut"><code>${_apEsc(_apSub(e.returns))}</code></div>` : ""}
    ${(e.notes || []).length ? `<h4 class="apH">Worth knowing</h4>
      <ul class="apNotes">${e.notes.map(n => `<li>${n}</li>`).join("")}</ul>` : ""}`;
}

function apiShow(id, {push = true} = {}) {
  const e = _apFind(id);
  if (!e) return;
  _apiAt = id;
  const pane = document.getElementById("apDetail");
  if (pane) pane.innerHTML = _apDetail(e);
  document.querySelectorAll("#apRail .apItem").forEach(b =>
    b.classList.toggle("on", b.dataset.id === id));
  if (push) {
    const h = `#api/${id}`;
    if (location.hash !== h) history.replaceState(null, "", h);
  }
  // On a phone the rail and the pane are stacked; jump to what was chosen.
  if (pane && window.matchMedia("(max-width:900px)").matches)
    pane.scrollIntoView({block: "start", behavior: "smooth"});
}

async function loadApiDocs() {
  const host = document.getElementById("apRail");
  if (!host) return;
  if (_apiHost === null) {
    // Same reasoning as the key examples on Account: these commands are for
    // ANOTHER machine, so localhost is the one host that is never useful.
    try { _apiHost = (await api("/api/machine/self")).url || ""; }
    catch { _apiHost = ""; }
  }
  if (!host.dataset.built) {
    host.innerHTML = _apRail();
    host.dataset.built = "1";
    host.querySelectorAll(".apItem").forEach(b =>
      b.onclick = () => apiShow(b.dataset.id));
  }
  const m = (location.hash || "").match(/^#api\/(.+)$/);
  const first = _apAll()[0];
  apiShow(m && _apFind(m[1]) ? m[1] : (first ? _apId(first) : ""), {push: false});
}

window.addEventListener("hashchange", () => {
  const tab = document.getElementById("api");
  if (!tab || !tab.classList.contains("active")) return;
  const m = (location.hash || "").match(/^#api\/(.+)$/);
  if (m && _apFind(m[1])) apiShow(m[1], {push: false});
});

window.loadApiDocs = loadApiDocs;
window.apiShow = apiShow;
