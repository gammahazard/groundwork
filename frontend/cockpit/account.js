"use strict";
/* The Account page: who you are, your password, your API keys.
 *
 * ABOUT YOU, not about the installation. Machines — where runs execute, which
 * cards exist — is a sibling tab for the same reason Data collection is: they
 * are machine-level facts with a different audience and a different frequency.
 * Burying infrastructure inside a settings page is how it stops being found.
 *
 * WHY THE KEY IS SHOWN ONCE AND LOUDLY. web/auth/keys.py stores only a SHA-256,
 * so there is no second chance and no "reveal" button to build. The UI has to be
 * honest about that in the moment rather than apologetic afterwards — hence a
 * full-width mono block with `user-select:all`, which is the difference between
 * copying it on a phone and losing it.
 *
 * RE-AUTHENTICATION IS NOT CEREMONY. Both destructive changes here — password
 * and username — send the current password, because an unattended open tab is
 * otherwise a complete account takeover for anyone who walks up. The server
 * requires it; this just asks for it in the same breath as the change.
 */

const _acEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const _acWhen = ts => {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 90) return `${Math.round(m)} min ago`;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(d)} d ago`;
};

/* Real commands against THIS cockpit, with its own origin filled in.
 *
 * ALWAYS A HEADER, NEVER ?key=. keys.bearer only reads the Authorization header,
 * and the docstring says why: a secret in a URL lands in access logs, in browser
 * history and in the Referer of any outbound link — three places nobody thinks
 * to rotate. Showing the wrong shape here would teach it.
 */
/* SELF_URL went with _keyHowHTML. It held the network address so the curl
 * examples were copy-pasteable from another machine; the API tab resolves that
 * itself now (tabs/api/docs.js). Keeping it here would have been a fetch on
 * every Account load feeding a variable nobody reads. */

/* _keyHowHTML lived here and is GONE (2026-08-05). It rendered the curl
 * examples into #keyHow on this page; both the container and the examples moved
 * to the API tab, where the reference replaced them. What was left was a
 * function with no caller and no element to write to — the kind of thing that
 * survives a year because it still parses.
 */

async function loadAccount() {
  const me = window.currentUser && currentUser();
  const who = $("#acctWho");
  if (who && me) {
    who.innerHTML =
      `<span class="acctName">${_acEsc(me.username)}</span>`
      + (me.admin ? `<span class="acctRole">admin</span>` : "")
      /* THE COUNT IS THE CONTROL. "3 active sessions" was already the
       * question and had no answer; making the number itself open the list
       * costs no new surface and puts the detail where it is asked for. */
      + `<button type="button" class="acctSub acctSessions" id="acctSessions">`
      + `${me.sessions ?? 1} active session`
      + `${(me.sessions ?? 1) === 1 ? "" : "s"}</button>`;
  }
  const sb = $("#acctSessions");
  if (sb && window.openSessions) sb.onclick = openSessions;
  loadKeys();
}

/* ---------------------------------------------------------------- keys --- */
/* The scope vocabulary comes FROM THE SERVER (auth/keys.py owns it), so adding
 * one is a change in a single file. Fetched once; the descriptions are long
 * enough that showing only the selected one keeps the form readable. */
let KEY_SCOPES = null;

async function loadKeyScopes() {
  const sel = $("#keyScope"), what = $("#keyScopeWhat");
  if (!sel) return;
  if (!KEY_SCOPES) {
    try { KEY_SCOPES = await apiOrThrow("/api/keys/scopes"); }
    catch { return; }
  }
  sel.innerHTML = (KEY_SCOPES.scopes || []).map(s =>
    `<option value="${_acEsc(s.key)}"${s.key === KEY_SCOPES.default ? " selected" : ""}>`
    + `${_acEsc(s.label)}</option>`).join("");
  const say = () => {
    const s = (KEY_SCOPES.scopes || []).find(x => x.key === sel.value);
    if (what) what.textContent = s ? s.what : "";
  };
  sel.onchange = say;
  say();
}

async function loadKeys() {
  loadKeyScopes();
  const box = $("#keyList");
  if (!box) return;
  let d;
  try { d = await apiOrThrow("/api/keys"); }
  catch (e) { box.innerHTML = `<p class="hint">Could not load keys: ${_acEsc(e.message || e)}</p>`; return; }
  const rows = d.keys || [];
  if (!rows.length) {
    box.innerHTML = `<p class="hint">No keys yet. A key lets a script or another
      machine act as you — it can train, read and report, but it cannot change
      your password or manage accounts.</p>`;
    return;
  }
  box.innerHTML = rows.map(k => `
    <div class="keyRow${k.expired ? " expired" : ""}">
      <span class="keyId">${_acEsc(k.id)}</span>
      <span class="keyName">${_acEsc(k.name)}</span>
      <!-- WHAT IT MAY DO, on the row. A list of keys that does not say which
           one is the wide-open one is a list you cannot act on. A key minted
           before scopes existed says so, because it IS full and was never
           chosen to be. -->
      <span class="keyScopeTag${k.scope === "full" ? " wide" : ""}"
            title="${k.legacy_scope
              ? "Minted before scopes existed, so it can do everything a key may. Revoke and re-mint to narrow it."
              : ""}">${_acEsc(k.scope || "full")}${k.legacy_scope ? " (legacy)" : ""}</span>
      <span class="keyMeta">created ${_acWhen(k.created)} ·
        used ${_acWhen(k.last_used)}${k.expired ? " · EXPIRED" : ""}</span>
      <button type="button" class="keyKill" data-id="${_acEsc(k.id)}">Revoke</button>
    </div>`).join("");
  box.querySelectorAll(".keyKill").forEach(b => {
    b.onclick = async () => {
      if (!confirm(`Revoke key ${b.dataset.id}? Anything using it stops working `
                 + `immediately.`)) return;
      await fetch(`/api/keys/${encodeURIComponent(b.dataset.id)}`, {method: "DELETE"});
      loadKeys();
    };
  });
}

/* THE ONE-TIME REVEAL. A modal rather than an inline block: the server keeps
 * only a SHA-256, so this string exists exactly once and an inline panel on a
 * long page can be scrolled past and lost. Dismissing it is a deliberate act.
 */
function showKeyModal(d) {
  $("#keyValue").textContent = d.key;
  $("#keyModalMeta").textContent = `Key ${d.id}`;
  const copy = $("#keyCopy");
  copy.textContent = "Copy";
  copy.classList.remove("done");
  $("#keyModal").hidden = false;
  copy.focus();
}

function hideKeyModal() {
  $("#keyModal").hidden = true;
  // Do not leave the secret in the DOM once it is dismissed.
  $("#keyValue").textContent = "";
}

function _wireKeyModal() {
  const m = $("#keyModal");
  if (!m) return;
  $("#keyCopy").onclick = async () => {
    const v = $("#keyValue").textContent;
    const btn = $("#keyCopy");
    /* THREE WAYS, because the first one is unavailable where this cockpit
     * actually runs. `navigator.clipboard` requires a SECURE CONTEXT, and this
     * is plain HTTP on a network IP — secure contexts are https, localhost and
     * nothing else, so on the phone and on 100.x the API is simply undefined
     * and the old code fell straight through to "select it yourself". That is
     * a working fallback and it is not what "Copy" promises.
     *
     * execCommand("copy") is deprecated and still implemented everywhere, and
     * it does NOT require a secure context — which makes it exactly right for
     * the middle case. Selecting the text stays as the last resort. */
    const selectIt = () => {
      const r = document.createRange();
      r.selectNodeContents($("#keyValue"));
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      return sel;
    };
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(v);
      } else {
        const sel = selectIt();
        if (!document.execCommand("copy")) throw new Error("execCommand refused");
        sel.removeAllRanges();
      }
    } catch {
      selectIt();
      btn.textContent = "Selected — press ⌘/Ctrl+C";
      return;
    }
    // Confirmation in a WORD, not a colour swap alone.
    btn.textContent = "Copied";
    btn.classList.add("done");
  };
  $("#keyModalX").onclick = hideKeyModal;
  $("#keyModalDone").onclick = hideKeyModal;
  m.onclick = e => { if (e.target === m) hideKeyModal(); };
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !m.hidden) hideKeyModal();
  });
}

function _wireAccount() {
  _wireKeyModal();
  const kf = $("#keyForm");
  if (kf) kf.onsubmit = async e => {
    e.preventDefault();
    const name = $("#keyName").value.trim();
    const msg = $("#keyMsg");
    msg.textContent = ""; msg.classList.remove("bad");
    try {
      const r = await fetch("/api/keys", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name})});
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Could not create the key.");
      $("#keyName").value = "";
      showKeyModal(d);
      const fold = $("#keyHowFold");
      if (fold) fold.open = true;      // the example belongs beside the key
      loadKeys();
    } catch (err) {
      msg.textContent = err.message || String(err);
      msg.classList.add("bad");
    }
  };

  const pf = $("#pwForm");
  if (pf) pf.onsubmit = async e => {
    e.preventDefault();
    const msg = $("#pwMsg");
    msg.textContent = ""; msg.classList.remove("bad");
    const nw = $("#pwNew").value;
    if (nw !== $("#pwNew2").value) {
      msg.textContent = "The two passwords do not match."; msg.classList.add("bad"); return;
    }
    try {
      const r = await fetch("/api/me/password", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({old_password: $("#pwOld").value, new_password: nw})});
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Could not change it.");
      pf.reset();
      msg.textContent = d.other_sessions_ended
        ? `Password changed. ${d.other_sessions_ended} other session`
          + `${d.other_sessions_ended === 1 ? "" : "s"} signed out.`
        : "Password changed.";
    } catch (err) {
      msg.textContent = err.message || String(err); msg.classList.add("bad");
    }
  };

  const uf = $("#userForm");
  if (uf) uf.onsubmit = async e => {
    e.preventDefault();
    const msg = $("#userMsg");
    msg.textContent = ""; msg.classList.remove("bad");
    if (!confirm("Changing your username also moves every project you own to the "
               + "new name. Continue?")) return;
    try {
      const r = await fetch("/api/me/username", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({new_username: $("#userNew").value.trim(),
                              password: $("#userPw").value})});
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Could not change it.");
      uf.reset();
      msg.textContent = `You are now ${d.user.username}.`;
      location.reload();      // the header, the project list and keys all move
    } catch (err) {
      msg.textContent = err.message || String(err); msg.classList.add("bad");
    }
  };

  const so = $("#signOutBtn");
  if (so) so.onclick = () => window.signOut && signOut();
}

_wireAccount();
window.loadAccount = loadAccount;
// The API tab renders the key list; this is the only code that knows how.
window.loadKeys = loadKeys;
window.showKeyModal = showKeyModal;
