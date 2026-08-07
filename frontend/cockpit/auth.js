"use strict";
/* Signing in, and who you are once you have.
 *
 * LOADS BEFORE EVERYTHING ELSE and gates the boot. dashboard.js's bootstrap
 * fetches /api/overview and /api/projects immediately; with no session those are
 * fourteen 401s and a page of failure cards before the login form appears. So
 * this file owns one promise — whenSignedIn() — and the bootstrap awaits it.
 *
 * THE GATE IS THE SERVER'S. Hiding the app with a body class is a convenience,
 * not the enforcement: web/auth/middleware.py refuses every route and the
 * /outputs mount alike, so a stale tab or a hand-typed URL gets 401 regardless
 * of what this file thinks. That is the same relationship lab_guard has with the
 * hidden curation tabs.
 *
 * ONE PLACE HANDLES 401, for the same reason core.js wraps window.fetch in one
 * place: a session expires while the tab is open, and every panel would
 * otherwise render its own "could not load" card. The wrapper below turns any
 * 401 back into the login view — which is the honest thing to show, because it
 * is the only action that fixes it.
 */

let ME = null;                       // {username, admin, must_change, sessions}

const _authEl = id => document.getElementById(id);

/* Resolves once the user is signed in. Never rejects: an unauthenticated visit
 * is a normal state, not an error, so it simply stays pending until they are. */
let _signedInResolve;
const _signedIn = new Promise(r => { _signedInResolve = r; });
window.whenSignedIn = () => _signedIn;
window.currentUser = () => ME;

/* ---------- 401 anywhere -> the login view -------------------------------
 * Wraps the wrapper core.js already installed, so both rules apply and neither
 * file has to know about the other. Deliberately does NOT touch the login call
 * itself: a wrong password is a 401 that the form must render inline, not a
 * reason to tear the form down and rebuild it. */
(() => {
  const inner = window.fetch;
  window.fetch = async (input, init) => {
    const r = await inner(input, init);
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (r.status === 401 && !url.includes("/api/login") && !url.includes("/api/me"))
      showLogin();
    return r;
  };
})();

/* ---------- the two views ------------------------------------------------ */
function showLogin(msg) {
  // The answer is in: stop withholding a view.
  document.body.classList.remove("authUnknown");
  document.body.classList.add("signedOut");
  if (msg) authError(msg);
  const u = _authEl("loginUser");
  if (u && !u.value) u.focus();
}

function authError(msg) {
  const e = _authEl("loginErr");
  if (!e) return;
  e.textContent = msg || "";
  e.hidden = !msg;
}

/* Crossing the datum: the tick that drew itself on arrival extends into a full
 * baseline, then the app appears behind it. The signature element carries the
 * transition rather than a fade, because the mark IS the idea — a measured
 * baseline, established. Skipped entirely under reduced motion. */
/* `fresh` = someone just typed a password, as opposed to arriving with a session
 * already in the cookie. The two want different destinations and conflating them
 * is why this takes an argument:
 *
 *   fresh          land on Projects. You have just come through the front door;
 *                  the landing page is where you choose where to go.
 *   not fresh      honour ?project= — a reload, a bookmark or a shared link
 *                  should put you back exactly where you were, which is the
 *                  whole reason the project lives in the URL (core.js).
 *
 * A session that expired mid-session is the interesting case: you are bounced to
 * the form, and signing back in counts as fresh — so you land on Projects rather
 * than silently resuming a project you may have stopped caring about. */
function enterApp(fresh) {
  const done = () => {
    if (fresh) {
      // Clearing the param is what makes restoreProjectFromURL find nothing and
      // the bootstrap fall through to showTab("projects") — one mechanism, not
      // a second redirect racing it.
      const u = new URL(location);
      if (u.searchParams.has("project")) {
        u.searchParams.delete("project");
        history.replaceState({}, "", u);
      }
    }
    document.body.classList.remove("authUnknown");
    document.body.classList.remove("signedOut");
    /* THE ADMIN TAB IS REVEALED THE MOMENT WE KNOW WHO YOU ARE, not when the
     * Account page happens to load. It was done in loadAccount, which meant an
     * admin who never opened Account never saw the tab at all — a control that
     * exists only after visiting an unrelated page.
     *
     * A COURTESY, NOT THE ENFORCEMENT: every endpoint behind it goes through
     * require_admin server-side, which also refuses API-key auth. Unhiding this
     * button by hand gets you a page full of 403s. */
    const adminBtn = document.querySelector('nav button[data-tab="admin"]');
    if (adminBtn) adminBtn.hidden = !(ME && ME.admin);
    _signedInResolve(ME);
  };
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const datum = document.querySelector("#login .loginDatum");
  if (reduce || !datum || !datum.animate) return done();
  datum.classList.add("crossing");
  setTimeout(done, 340);
}

/* ---------- who am I ------------------------------------------------------ */
async function loadMe() {
  let r;
  try {
    r = await fetch("/api/me");
  } catch {
    // The server is unreachable, which is NOT "you are signed out" — saying so
    // would send someone to retype a password that was never the problem.
    return showLogin("Can't reach the cockpit. Check it is running, then retry.");
  }
  if (r.status === 401) {
    // Zero-accounts first run: the wizard takes the screen instead of a
    // login form nobody can pass. Any error inside falls through to login.
    if (window.maybeShowSetup && await maybeShowSetup()) return;
    return showLogin();
  }
  if (!r.ok) return showLogin("Sign-in check failed. Try again.");
  ME = await r.json();
  renderIdentity();
  if (ME.must_change) return showChangeStep();
  enterApp(false);
}

/* "logged in as", top right of the header. Small on purpose — it answers a
 * question you ask once a session, and it is a link to the Account page rather
 * than a menu, because there is exactly one thing behind it. */
function renderIdentity() {
  const host = _authEl("whoami");
  if (!host || !ME) return;
  host.hidden = false;
  host.innerHTML =
    `<button id="whoamiBtn" class="whoamiBtn" title="Account settings">`
    + `<span class="whoamiK">signed in</span>`
    + `<span class="whoamiV">${_authEsc(ME.username)}</span></button>`;
  const b = _authEl("whoamiBtn");
  if (b) b.onclick = () => showTab("account");
}

const _authEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

/* ---------- must-change ---------------------------------------------------
 * A bootstrapped account arrives with a password that is in .env, in a README
 * and possibly in a git history. It is not usable until it is replaced, and the
 * flag says so — but the SERVER does not enforce it, deliberately: locking every
 * route behind a flag would mean a half-signed-in state for every endpoint to
 * reason about. The UI insists instead, which is where insistence belongs. */
function showChangeStep() {
  document.body.classList.remove("authUnknown");
  document.body.classList.add("signedOut");
  const wrap = document.querySelector(".loginWrap");
  if (wrap) wrap.classList.add("loginStep2");
  _authEl("loginName").textContent = "Choose a new password";
  _authEl("loginWhat").textContent =
    `${ME.username} is still using the password from setup.`;
  _authEl("loginFields").hidden = true;
  _authEl("changeFields").hidden = false;
  _authEl("loginGo").textContent = "Set password";
  const n = _authEl("newPw");
  if (n) n.focus();
}

/* ---------- submit -------------------------------------------------------- */
function wireLogin() {
  const form = _authEl("loginForm");
  if (!form) return;
  form.onsubmit = async e => {
    e.preventDefault();
    const go = _authEl("loginGo");
    authError("");
    go.disabled = true;

    try {
      if (ME && ME.must_change) {
        const pw = _authEl("newPw").value;
        const again = _authEl("newPw2").value;
        if (pw !== again) throw new Error("The two passwords do not match.");
        if (pw.length < 8) throw new Error("Use at least 8 characters.");
        const r = await fetch("/api/me/password", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({old_password: _authEl("loginPw").dataset.used || "",
                                new_password: pw})});
        if (!r.ok) throw new Error((await r.json()).detail || "Could not set it.");
        ME.must_change = false;
        return enterApp(true);
      }
      const username = _authEl("loginUser").value.trim();
      const password = _authEl("loginPw").value;
      const r = await fetch("/api/login", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username, password})});
      if (!r.ok) {
        // The server says only "wrong username or password" — never which half.
        // Repeating that verbatim keeps the two layers telling one story.
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || "Wrong username or password.");
      }
      ME = (await r.json()).user;
      // Kept only to satisfy the must-change step's old-password field, which
      // the server requires. Never stored anywhere that outlives the page.
      _authEl("loginPw").dataset.used = password;
      renderIdentity();
      if (ME.must_change) { go.disabled = false; return showChangeStep(); }
      enterApp(true);
    } catch (err) {
      authError(err.message || String(err));
    } finally {
      go.disabled = false;
    }
  };
}

async function signOut() {
  try { await fetch("/api/logout", {method: "POST"}); } catch { /* going anyway */ }
  location.reload();          // simplest correct reset: every module re-boots
}
window.signOut = signOut;

/* ---------- how it works, before you are in --------------------------------
 * Rendered from window.GUIDE_STEPS — the Guide tab's own content — rather than
 * a second copy. guide_steps.js loads after this file, which is fine: nothing
 * reads it until the button is pressed, long after every script has run.
 */
function _wireHow() {
  const btn = _authEl("loginHow"), modal = _authEl("howModal");
  if (!btn || !modal) return;
  let mounted = false;

  /* CLOSING IS ANIMATED TOO, which is the half usually skipped: [hidden] removes
   * the element on the spot, so a dialog that glides open would blink out. Add
   * the class, let the animation run, then hide.
   *
   * A TIMER, not animationend. Under prefers-reduced-motion the animation is
   * `none` and animationend NEVER FIRES — the modal would stay on screen
   * forever for exactly the people who asked for less motion. A timeout always
   * arrives. _authEl is re-queried nowhere here; `modal` is stable. */
  let closing = null;
  const close = () => {
    if (closing) return;
    modal.classList.add("howClosing");
    closing = setTimeout(() => {
      modal.hidden = true;
      modal.classList.remove("howClosing");
      closing = null;
      btn.focus();                 // put the caret back where it came from
    }, 180);
  };

  btn.onclick = () => {
    /* THE SAME GUIDE THE TAB SHOWS, mounted a second time from the same
     * GUIDE_STEPS. This used to be a hand-written list of titles and ledes —
     * a second, thinner telling of the same thing, which drifted: it never
     * gained the Export step, and nothing would have said so. */
    if (!mounted && window.mountGuide) {
      mountGuide(_authEl("howSteps"), {
        deepLink: false,           // a dialog must not take the URL
        isActive: () => !modal.hidden && !modal.classList.contains("howClosing"),
      });
      mounted = true;
    }
    if (!mounted) {
      _authEl("howSteps").innerHTML =
        `<p class="hint">The guide could not be loaded.</p>`;
    }
    modal.hidden = false;
    _authEl("howX").focus();
  };

  _authEl("howX").onclick = close;
  modal.onclick = e => { if (e.target === modal) close(); };
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !modal.hidden) close();
  });
}

wireLogin();
_wireHow();
loadMe();
