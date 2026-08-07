"use strict";
/* Data collection — the Telegram bots of the projects you can open.
 *
 * IT USED TO BE EVERY BOT ON THE MACHINE, and that was a deliberate decision
 * (owner, 2026-07-31): "a bot is a thing running on this box, and the question
 * worth a screen is what is collecting right now and where it is putting
 * things — which spans projects by nature." True with one account, wrong with
 * two. Reversed on 2026-08-05 at the owner's request; recorded rather than
 * quietly changed, because the old reasoning is still the reasoning you would
 * reach for.
 *
 * A bot belongs to a project, a project belongs to a person, so a bot belongs
 * to a person. Every control here is gated that way SERVER-side — see
 * web/api/bot_deps.py. Nothing on this page is admin-only except the legacy
 * owner card, which is the point: the person whose project it is sets up their
 * own bot without needing one.
 *
 * EVERY REQUEST NAMES ITS BOT'S PROJECT EXPLICITLY. core.js wraps window.fetch
 * with withProject(), which appends the CURRENTLY OPEN project to any
 * same-origin URL — and this is a machine-level tab reachable while a project
 * is open, so a bot belonging to project A would otherwise be actioned with
 * ?project=B. withProject is idempotent (it leaves a URL that already has the
 * parameter alone), so building it in is what wins. `?project=` is now REQUIRED
 * server-side, so forgetting it is a 422 rather than a wrong answer.
 *
 * TWO SIGNALS, NOT ONE, because they fail apart and the difference is the whole
 * diagnosis:
 *
 *     unit state     what systemd thinks — running / stopped / crashed
 *     last activity  the bot's own log mtime
 *
 * A bot that is "active" with a log that has not moved in three days is polling
 * Telegram happily and receiving nothing: the service is fine and the pipeline
 * is dead. Showing only a coloured dot would hide exactly that case, so the age
 * is always spelled out in words — and the owner is colour-blind, so a dot is
 * never the only carrier anyway.
 *
 * THE TOKEN GOES ONE WAY. It is typed here, POSTed once, verified with Telegram
 * and written to .env by the server. Nothing ever reads one back: the API sends
 * a boolean, the input is cleared either way, and it is type=password so it is
 * not left sitting on screen.
 */

const _cEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const _cAgo = ts => {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 90) return `${Math.round(m)} min ago`;
  if (h < 36) return `${Math.round(h)} h ago`;
  return `${Math.round(d)} d ago`;
};

let COLLECT = {projects: [], roles: [], bots: []};

/* The bot's own project, on the URL, always. See the header.
 *
 * A QUERY-STRING HELPER RATHER THAN A URL BUILDER, deliberately: every path
 * below stays a literal so `scratch/check_frontend.py` can match it against the
 * route table. Factoring the whole URL into one function made all five calls
 * invisible to that check — which is the check that catches a fetch to a route
 * that no longer exists, and it caught this. */
const _cQ = key => `?project=${encodeURIComponent(_cBot(key).project || "")}`;

const _cBot = key => (COLLECT.bots || []).find(b => b.key === key) || {};

/* Health is stated in WORDS as well as carried by the dot. The owner is
 * colour-blind, and more importantly "green" alone cannot express "the service
 * is up but nothing is arriving" — which is the failure this panel exists to
 * make visible. */
function _botHealth(b) {
  const stale = b.last_seen && (Date.now() / 1000 - b.last_seen) > 3 * 86400;
  if (!b.token_set)
    return {cls: "warn", dot: "●", word: "no token yet — paste it below"};
  /* UNCLAIMED IS ITS OWN STATE, and it has to be, because the alternative reads
   * as a broken bot: an installed bot with nobody named answers "not
   * authorized" to everyone, including the person who just set it up. An empty
   * list deliberately does NOT mean "whoever owns the machine". */
  if (!(b.allowed_ids || []).length)
    return {cls: "warn", dot: "●", word: "nobody may use it yet — add your id"};
  if (b.state === "active")
    return stale
      ? {cls: "warn", dot: "●", word: "running, but nothing received in days"}
      : {cls: "ok", dot: "●", word: "running"};
  if (b.state === "backoff")
    return {cls: "bad", dot: "●",
            word: "crashing on start — Probe the token, then check the log"};
  if (!b.installed)
    return {cls: "warn", dot: "●",
            word: COLLECT && COLLECT.process_model === "supervisor"
              ? "not running yet" : "ready to install"};
  if (b.state !== "active")
    return {cls: "bad", dot: "●", word: b.state === "unknown"
              ? "can't tell — systemd did not answer" : `not running (${b.state})`};
  if (stale)
    return {cls: "warn", dot: "●", word: "running, but nothing received in days"};
  return {cls: "ok", dot: "●", word: "running"};
}

/* WHAT TO DO NEXT, as one obvious control per card. A card showing five buttons
 * at once makes you work out which applies; the state already knows, so it
 * decides. Nothing is hidden that is still meaningful — the controls that drop
 * off are the ones that would 400. */
function _botNext(b) {
  const k = _cEsc(b.key);
  if (!b.token_set)
    return `<form class="botTok" data-key="${k}">
        <label class="botTokLab">Paste the token from @BotFather
          <input type="password" class="botTokIn" autocomplete="off"
                 spellcheck="false" placeholder="1234567890:AA…">
        </label>
        <button type="submit">Verify &amp; save</button>
      </form>`;
  if (!(b.allowed_ids || []).length)
    return `<form class="botWho" data-key="${k}">
        <label class="botTokLab">Message the bot <code>/whoami</code>, then paste
          the number it replies with
          <input type="text" class="botWhoIn" inputmode="numeric"
                 autocomplete="off" spellcheck="false" placeholder="123456789">
        </label>
        <button type="submit">Let them in</button>
      </form>`;
  if (!b.installed && b.role)
    return (COLLECT && COLLECT.process_model === "supervisor")
      ? `<button class="botSvc" data-key="${k}" data-act="start">Start ▸</button>`
      : `<button class="botInstall" data-key="${k}">Install as a service →</button>`;
  if (!b.installed && !b.role)
    return `<p class="hint">Registered before roles existed, so there is no
      service to install. Re-register it choosing what it does.</p>`;
  /* reset-failed is offered ONLY when systemd has latched the unit into
   * `failed`, because that is the state where start/stop/restart all refuse and
   * the card would otherwise be a dead end. */
  const stuck = b.state === "failed"
    ? `<button class="botSvc" data-key="${k}" data-act="reset-failed">Clear the error</button>`
    : "";
  return `<button class="botSvc" data-key="${k}" data-act="restart">Restart</button>
          <button class="botSvc" data-key="${k}" data-act="${b.state === "active" ? "stop" : "start"}">
            ${b.state === "active" ? "Stop" : "Start"}</button>${stuck}`;
}

/* Who may talk to it, and the control to change that. Telegram ids are not
 * secrets — you hand yours to a bot by messaging it — so they are shown. */
function _botWhoHTML(b) {
  const ids = b.allowed_ids || [];
  if (!ids.length) return `<span class="short">nobody yet</span>`;
  return `${ids.map(i => `<code>${_cEsc(i)}</code>`).join(" ")}
    <button class="botWhoEdit" data-key="${_cEsc(b.key)}">change</button>`;
}

function _botCardHTML(b) {
  const h = _botHealth(b);
  const k = _cEsc(b.key);
  const opts = COLLECT.projects.map(p =>
    `<option value="${_cEsc(p.slug)}"${p.slug === b.project ? " selected" : ""}>`
    + `${_cEsc(p.name)}</option>`).join("");
  const mine = (COLLECT.projects.find(p => p.slug === b.project) || {}).why;
  return `<article class="botCard" data-key="${k}">
    <header class="botHead">
      <div>
        <h3>${_cEsc(b.name)}</h3>
        <div class="botState ${h.cls}"><span class="botDot" aria-hidden="true">${h.dot}</span>
          ${_cEsc(h.word)}</div>
      </div>
      <button class="botProbe" data-key="${k}"
              title="Ask Telegram whether this token still works">Probe</button>
    </header>
    <p class="botWhat">${_cEsc(b.what)}</p>
    ${mine === "admin" ? `<p class="hint">Someone else's project — visible to you
      as an admin.</p>` : ""}
    <dl class="botFacts">
      <div><dt>feeds</dt><dd><select class="botProj" data-key="${k}">${opts}</select></dd></div>
      <div><dt>who may use it</dt><dd>${_botWhoHTML(b)}</dd></div>
      <div><dt>last activity</dt><dd>${_cAgo(b.last_seen)}</dd></div>
      <div><dt>counts with</dt><dd>the serving model — before one exists
        it still collects (photos land in the fix queue)</dd></div>
      <div><dt>service</dt><dd><code>${_cEsc(b.unit)}</code>${
        b.installed ? "" : ` <span class="mMeta">(not installed)</span>`}</dd></div>
      <div><dt>token</dt><dd>${b.token_set
        ? `<code>${_cEsc(b.token_env)}</code> set
           <button type="button" class="botTokSwap" data-key="${_cEsc(b.key)}"
                   title="Paste a different token — revoked the old bot, made a new one">replace</button>`
        : `<span class="short"><code>${_cEsc(b.token_env)}</code> empty</span>`}</dd></div>
    </dl>
    <div class="botActs">${_botNext(b)}</div>
    <div class="botFoot"><button type="button" class="botRemove mMeta"
        data-key="${_cEsc(b.key)}">remove this bot</button></div>
    <div class="botProbeOut" id="botProbe-${k}"></div>
  </article>`;
}

async function loadCollect() {
  return loadInto("#botList", async () => {
    COLLECT = await apiOrThrow("/api/collect");
    _renderNewBotForm();
    _renderBotSteps(COLLECT.steps || []);
    if (!COLLECT.projects.length)
      return `<p class="hint">You have no projects yet. A bot collects images
        <i>for</i> a project, so there is nothing for one to feed —
        create a project first and it will appear here.</p>`;
    if (!COLLECT.bots.length)
      return `<p class="hint">No bots yet — see “Add a collection bot” below.</p>`;
    return COLLECT.bots.map(_botCardHTML).join("");
  }, {
    label: "the collection bots", kind: "card", n: 2,
    after: host => {
      host.querySelectorAll(".botProbe").forEach(b => b.onclick = () => _probeBot(b));
      host.querySelectorAll(".botTok").forEach(f => f.onsubmit = _saveToken);
      // "replace": swap the actions area for the SAME verified paste form the
      // no-token state uses — a revoked token is set yet useless, and Probe
      // can only report that, not fix it.
      host.querySelectorAll(".botTokSwap").forEach(btn => btn.onclick = () => {
        const card = btn.closest("article");
        const acts = card && card.querySelector(".botActs");
        if (!acts) return;
        acts.innerHTML = `<form class="botTok" data-key="${btn.dataset.key}">
            <label class="botTokLab">Paste the token from @BotFather
              <input type="password" class="botTokIn" autocomplete="off"
                     spellcheck="false" placeholder="1234567890:AA…">
            </label>
            <button type="submit">Verify &amp; save</button>
          </form>`;
        acts.querySelector(".botTok").onsubmit = _saveToken;
        acts.querySelector(".botTokIn").focus();
      });
      host.querySelectorAll(".botWho").forEach(f => f.onsubmit = _saveWho);
      host.querySelectorAll(".botWhoEdit").forEach(b => b.onclick = () => _editWho(b));
      host.querySelectorAll(".botInstall").forEach(b => b.onclick = () => _install(b));
      host.querySelectorAll(".botSvc").forEach(b => b.onclick = () => _service(b));
      host.querySelectorAll(".botProj").forEach(s => s.onchange = () => _moveProject(s));
      host.querySelectorAll(".botRemove").forEach(b => b.onclick = () => _removeBot(b));
      revealAll(host, 50, 6);
    },
  });
}

/* One place to say what happened, on the card it happened to. It doubles as the
 * probe output because a card only ever reports one thing at a time. */
function _say(key, cls, html) {
  const out = $(`#botProbe-${key}`);
  if (!out) return;
  out.className = "botProbeOut" + (cls ? " " + cls : "");
  out.innerHTML = html;
}

async function _probeBot(btn) {
  const key = btn.dataset.key;
  btn.disabled = true; btn.textContent = "…";
  _say(key, "", "asking Telegram…");
  try {
    const r = await apiOrThrow(`/api/bots/${encodeURIComponent(key)}/probe` + _cQ(key),
                               {method: "POST"});
    if (r.ok)
      _say(key, "ok", `✓ token works — this is <b>@${_cEsc(r.username)}</b>`
                    + ` <span class="mMeta">(${_cEsc(r.name)})</span>`);
    else
      _say(key, "short", `✗ ${_cEsc(r.error)}`);
  } catch (e) {
    _say(key, "short", `✗ ${_cEsc(e.message || e)}`);
  }
  btn.disabled = false; btn.textContent = "Probe";
}

/* THE TOKEN. The server verifies it with Telegram BEFORE writing, so a typo is
 * an error message here rather than a service that crash-loops on 401 while
 * three screens agree it is configured. Cleared from the input either way — a
 * rejected token is still a secret. */
async function _saveToken(e) {
  e.preventDefault();
  const form = e.currentTarget, key = form.dataset.key;
  const input = form.querySelector(".botTokIn"), btn = form.querySelector("button");
  const token = input.value.trim();
  btn.disabled = true; btn.textContent = "verifying…";
  _say(key, "", "asking Telegram whether this token works…");
  try {
    const r = await apiOrThrow(`/api/bots/${encodeURIComponent(key)}/token` + _cQ(key),
      {method: "POST", headers: {"Content-Type": "application/json"},
       body: JSON.stringify({token})});
    input.value = "";
    _say(key, "ok", `✓ saved — this is <b>@${_cEsc(r.username)}</b>. ${_cEsc(r.note)}`);
    await loadCollect();
  } catch (ex) {
    input.value = "";
    _say(key, "short", `✗ ${_cEsc(ex.message || ex)}`);
    btn.disabled = false; btn.textContent = "Verify & save";
  }
}

/* WHO MAY USE IT. Sent as a list from the first day — a single id per bot would
 * repeat, one level down, the machine-wide-single-id bug this whole page is
 * fixing, and a workplace has more than one technician. */
async function _saveWho(e) {
  e.preventDefault();
  const form = e.currentTarget, key = form.dataset.key;
  const input = form.querySelector(".botWhoIn"), btn = form.querySelector("button");
  const ids = input.value.split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
  btn.disabled = true; btn.textContent = "saving…";
  try {
    const r = await apiOrThrow(`/api/bots/${encodeURIComponent(key)}/allowed` + _cQ(key),
      {method: "POST", headers: {"Content-Type": "application/json"},
       body: JSON.stringify({telegram_ids: ids})});
    _say(key, "ok", `✓ ${r.telegram_ids.length} person(s) may use it. ${_cEsc(r.note)}`);
    await loadCollect();
  } catch (ex) {
    _say(key, "short", `✗ ${_cEsc(ex.message || ex)}`);
    btn.disabled = false; btn.textContent = "Let them in";
  }
}

/* Changing the list re-uses the same endpoint; prefilled so removing one person
 * is an edit rather than retyping the rest. */
function _editWho(btn) {
  const key = btn.dataset.key, b = _cBot(key);
  const now = (b.allowed_ids || []).join(" ");
  const next = prompt("Telegram ids that may use this bot, separated by spaces."
                    + "\n\nEmpty means nobody — the bot will answer no one.", now);
  if (next === null) return;
  const ids = next.split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
  apiOrThrow(`/api/bots/${encodeURIComponent(key)}/allowed` + _cQ(key),
    {method: "POST", headers: {"Content-Type": "application/json"},
     body: JSON.stringify({telegram_ids: ids})})
    .then(r => { _say(key, "ok", `✓ ${_cEsc(r.note)}`); return loadCollect(); })
    .catch(ex => _say(key, "short", `✗ ${_cEsc(ex.message || ex)}`));
}

async function _install(btn) {
  const key = btn.dataset.key;
  btn.disabled = true; btn.textContent = "installing…";
  _say(key, "", "writing the unit, enabling and starting it…");
  try {
    const r = await apiOrThrow(`/api/bots/${encodeURIComponent(key)}/install` + _cQ(key),
                               {method: "POST"});
    _say(key, "ok", `✓ installed <code>${_cEsc(r.unit)}</code> and started it`);
    await loadCollect();
  } catch (ex) {
    _say(key, "short", `✗ ${_cEsc(ex.message || ex)}`);
    btn.disabled = false; btn.textContent = "Install as a service →";
  }
}

async function _removeBot(btn) {
  const key = btn.dataset.key;
  if (!confirm("Remove this bot?\n\nThe service is stopped and the entry is "
             + "forgotten; the role frees up so it can be set up again from "
             + "scratch. The token stays in .env until overwritten."))
    return;
  btn.disabled = true;
  try {
    // Stop first — the backend's delete is entry-only by design, and a
    // process left polling for a forgotten bot is exactly the gap it warns
    // about in its own response.
    await apiOrThrow(`/api/bots/${encodeURIComponent(key)}/service` + _cQ(key),
      {method: "POST", headers: {"Content-Type": "application/json"},
       body: JSON.stringify({action: "stop"})}).catch(() => {});
    await apiOrThrow(`/api/bots/${encodeURIComponent(key)}` + _cQ(key),
                     {method: "DELETE"});
    loadCollect();
  } catch (ex) {
    _say(key, "bad", ex.message || String(ex));
    btn.disabled = false;
  }
}

async function _service(btn) {
  const key = btn.dataset.key, act = btn.dataset.act;
  btn.disabled = true;
  _say(key, "", `${act === "reset-failed" ? "clearing" : act + "ing"}…`);
  try {
    await apiOrThrow(`/api/bots/${encodeURIComponent(key)}/service` + _cQ(key),
      {method: "POST", headers: {"Content-Type": "application/json"},
       body: JSON.stringify({action: act})});
    // systemd reports the new state a beat later; ask again rather than assert.
    setTimeout(loadCollect, 1200);
  } catch (ex) {
    _say(key, "short", `✗ ${_cEsc(ex.message || ex)}`);
    btn.disabled = false;
  }
}

/* Moving a bot between projects rewrites its unit too (the server does), so the
 * cockpit and the running service cannot disagree about where images land. On
 * failure the <select> is put BACK — leaving it showing a project the move did
 * not make is the kind of quiet lie this page exists to prevent. */
async function _moveProject(sel) {
  const key = sel.dataset.key, want = sel.value;
  const was = _cBot(key).project;
  sel.disabled = true;
  try {
    const r = await apiOrThrow(`/api/bots/${encodeURIComponent(key)}/project` + _cQ(key),
      {method: "POST", headers: {"Content-Type": "application/json"},
       body: JSON.stringify({project: want})});
    _say(key, "ok", `✓ now feeds <b>${_cEsc(want)}</b>`
       + (r.unit_rewritten ? " — its service was rewritten and restarted" : ""));
    await loadCollect();
  } catch (ex) {
    sel.value = was || sel.value;
    sel.disabled = false;
    _say(key, "short", `✗ ${_cEsc(ex.message || ex)}`);
  }
}

/* ------------------------------------------------------------ registering --- */
/* Registering is METADATA — a project, a role, a name. The service name and the
 * token variable are DERIVED by the server from the project and the job, so
 * they cannot collide with another account's; the token itself is pasted on the
 * card afterwards, so it never travels with the rest of a form. */
function _renderNewBotForm() {
  const proj = $("#bfProject"), role = $("#bfRole");
  const form = $("#botForm"), none = $("#bfNone");
  if (!proj || !role || !form) return;

  // Only projects with a job left to fill — the server caps a project at one
  // bot per role, so offering a used one would build a form that can only 409.
  const open = (COLLECT.projects || []).filter(p => (p.roles || []).length);
  if (!open.length) {
    form.hidden = true;
    if (none) {
      none.hidden = false;
      none.innerHTML = (COLLECT.projects || []).length
        ? `Every project you own already has its bot. A project gets one bot per
           job — make another project to add another bot.`
        : `<b>Create a project first.</b> A bot collects images <i>for</i> a
           project, so it needs one to feed. Open the Projects tab, create one,
           then come back and set up your first bot.`;
    }
    return;
  }
  form.hidden = false;
  if (none) none.hidden = true;

  const keepP = proj.value, keepR = role.value;
  proj.innerHTML = open.map(p =>
    `<option value="${_cEsc(p.slug)}">${_cEsc(p.name)}</option>`).join("");
  if (keepP && open.some(p => p.slug === keepP)) proj.value = keepP;
  const paint = () => {
    const p = open.find(x => x.slug === proj.value) || {roles: []};
    role.innerHTML = (p.roles || []).map(r =>
      `<option value="${_cEsc(r.key)}">${_cEsc(r.name)}</option>`).join("");
    if (keepR && [...role.options].some(o => o.value === keepR)) role.value = keepR;
    const r = (p.roles || []).find(x => x.key === role.value);
    const what = $("#bfRoleWhat");
    if (what) what.textContent = r ? r.what : "";
  };
  proj.onchange = paint;
  role.onchange = paint;
  paint();
}

(() => {
  const form = document.querySelector("#botForm");
  if (!form) return;
  form.onsubmit = async e => {
    e.preventDefault();
    const btn = document.querySelector("#bfAdd"), err = document.querySelector("#bfErr");
    btn.disabled = true; btn.textContent = "registering…"; err.hidden = true;
    try {
      // Registering targets a PROJECT and this page is NOT inside one, so the
      // slug goes on the URL explicitly. core.js's withProject() would append
      // whichever project happens to be open, which is not the one the form
      // chose — see this file's header.
      const slug = document.querySelector("#bfProject").value;
      await apiOrThrow(`/api/bots?project=${encodeURIComponent(slug)}`,
        {method: "POST", headers: {"Content-Type": "application/json"},
         body: JSON.stringify({name: document.querySelector("#bfName").value.trim(),
                               role: document.querySelector("#bfRole").value})});
      form.reset();
      await loadCollect();
    } catch (ex) {
      // The server owns every rule; show what it said.
      err.textContent = ex.message || String(ex);
      err.hidden = false;
    }
    btn.disabled = false; btn.textContent = "Register bot";
  };
})();

/* The steps, on the page that performs most of them. Kept as server text rather
 * than markup here so the flow is described in one place — api/bots.py
 * ::setup_steps — instead of drifting between a docstring and a template. */
function _renderBotSteps(steps) {
  const el = $("#botSteps");
  if (!el) return;
  el.innerHTML = (steps || []).map(s =>
    `<li><b>${_cEsc(s.title)}</b>
       <span class="botWhere">${_cEsc(s.where)}</span>
       <div class="botStepBody">${_cEsc(s.body)}</div></li>`).join("");
}
