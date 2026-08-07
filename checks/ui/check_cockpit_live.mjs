/* Drive the running cockpit in REAL browsers and assert it is usable.
 *
 *     npm  --prefix checks/ui install                  # once
 *     npx  playwright install chromium webkit        # once
 *     node checks/ui/check_cockpit_live.mjs          # cockpit up on :8000
 *     COCKPIT=http://<worker>:8000/ node checks/ui/check_cockpit_live.mjs
 *
 * WHY, AND WHY check_frontend.py WAS NOT ENOUGH.
 *
 * check_frontend.py is static: assets resolve, tabs pair with sections, fetched
 * paths exist in the OpenAPI table, every file parses, no top-level collisions.
 * All seven checks passed green while the cockpit was, in practice, unusable.
 *
 * The defect: commit aa26389 added a GATE (`.navGroupProject{display:none}`,
 * shown only under `body.inProject`) and a REDIRECT (showTab sends any project
 * tab back to "projects" when that class is absent). The only thing that set
 * the class was openProject(), reached by a click or by ?project= in the URL.
 * A bare URL therefore set nothing: ten of thirteen tabs invisible, and
 * "Models & Runs" silently bouncing. Nothing was missing and nothing was
 * misspelled — every file, route and id was correct. The COMPOSITION was
 * wrong, and no grep finds that.
 *
 * So this file asserts BEHAVIOUR, and specifically the ENTRY STATES, because
 * that is the axis that broke. A checker that only ever loads one URL would
 * have stayed green through the whole incident.
 *
 * WHY A REAL BROWSER AND NOT A STUB. Three simulations were written before
 * this and all three LIED:
 *   - `eval(src)` in a function scope does not create globals the way a classic
 *     <script> does, so nav.js's top-level `function initNav()` read as
 *     undefined and the burger looked unwired. It was wired.
 *   - a fake `fetch` returning {} made yolo.js throw on `m.runs.length`, which
 *     looked like a crash in the runs renderer. It was the stub.
 *   - jsdom has no layout, so it can confirm a class toggles but never that the
 *     drawer MOVES. The bug report was about movement.
 * Each pointed at healthy code while the real defect sat elsewhere. A check
 * that cannot really run must fail loudly, not answer confidently — hence the
 * hard exits below rather than skips.
 *
 * WebKit is included because the owner reports from an iPhone and WebKit is
 * Safari's engine; Chromium-only would not have been evidence about the device
 * the complaint came from.
 *
 * STILL NOT COVERED: real network conditions, real touch latency, and the
 * owner's actual cache. Clicking every tab on the phone remains the honest
 * test. This replaces the part a human is bad at — noticing that one entry
 * state out of four leaves the page crippled.
 */
import { chromium, webkit, devices } from "playwright";

const BASE = (process.env.COCKPIT || "http://localhost:8000/").replace(/\/?$/, "/");
const bad = [];
const note = (ok, msg) => { console.log(`  ${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) bad.push(msg); };

/* A page's console and uncaught errors are evidence; collect them per page.
 * Favicon 404s are the browser's, not ours, and would drown the signal. */
function watch(pg, sink) {
  pg.on("pageerror", e => sink.push(`uncaught: ${e.message}`));
  pg.on("console", m => { if (m.type() === "error" && !/favicon/i.test(m.text())) sink.push(`console: ${m.text()}`); });
  // A bare "Failed to load resource" is useless; record the URL and status so a
  // failure names the endpoint instead of sending you back to the browser.
  pg.on("response", r => { if (r.status() >= 400 && !/favicon/i.test(r.url()))
    sink.push(`HTTP ${r.status()} ${r.url().replace(BASE, "/")}`); });
}

const state = pg => pg.evaluate(() => ({
  inProject: document.body.classList.contains("inProject"),
  slug: document.body.dataset.project || null,
  // Only tabs the user can actually SEE: a display:none group is not a tab.
  visibleTabs: [...document.querySelectorAll("#navPanel nav button")]
    .filter(b => b.offsetParent !== null).map(b => b.dataset.tab),
  activeSection: [...document.querySelectorAll(".tab")]
    .filter(t => t.classList.contains("active")).map(t => t.id),
}));

async function open(ctx, url, errs) {
  const pg = await ctx.newPage();
  watch(pg, errs);
  await pg.goto(url, {waitUntil: "networkidle"});
  await pg.waitForTimeout(1200);
  return pg;
}

/* ---- entry states: the axis the incident happened on -------------------- */
async function checkEntryStates(browser) {
  const ctx = await browser.newContext({viewport: {width: 1440, height: 900}});
  const errs = [];

  /* 1. bare URL — MUST land on the LANDING PAGE with NO project open.
   *
   * This assertion was the exact opposite until 2026-07-31, and both versions
   * were right for their moment. The original incident was a bare URL that hid
   * ten of thirteen tabs with no way in, and the fix chosen then was to
   * auto-open when only one project existed. The owner has since ruled that the
   * Dashboard must show nothing about any project until one is opened — which
   * cannot coexist with a project always being open. The landing page is now
   * the way in, and it is one click.
   *
   * What still has to hold is what the incident was actually about: you must
   * not be STRANDED. So this asserts the entry state AND that the project tabs
   * become reachable — checkTabs below drives them after opening one. */
  let pg = await open(ctx, BASE, errs);
  let s = await state(pg);
  note(!s.inProject,
       `bare URL stays OUT of a project (got inProject=${s.inProject}, slug=${s.slug})`);
  note(s.activeSection.includes("projects"),
       `bare URL lands on the landing page (active: ${s.activeSection.join(",")})`);
  // ...and the way in is present and clickable, or the above is just "stranded"
  // with a nicer name.
  const cards = await pg.$$eval(".pjCard .pjOpen", b => b.length);
  note(cards > 0, `the landing page offers a project to open (${cards} card(s))`);
  const slug = await pg.$eval(".pjCard .pjOpen", b => b.dataset.slug).catch(() => null);
  await pg.close();

  // 2. explicit ?project=<real> — same place, and the URL wins.
  if (slug) {
    pg = await open(ctx, `${BASE}?project=${slug}`, errs);
    s = await state(pg);
    note(s.inProject && s.slug === slug, `?project=${slug} opens that project (slug=${s.slug})`);
    await pg.close();
  }

  // 3. ?project=<bogus> — must NOT silently pretend it worked. deps.py 404s
  //    unknown slugs server-side for the same reason: a silent fallback means
  //    you are looking at another project's data believing it is this one.
  pg = await open(ctx, `${BASE}?project=definitely-not-a-project`, errs);
  s = await state(pg);
  note(s.slug !== "definitely-not-a-project",
       `unknown ?project= does not claim to have opened it (slug=${s.slug})`);
  await pg.close();

  /* 4. every visible tab must ACTIVATE and render something. A tab that
   *    activates blank looks identical to "no data yet" and gets ignored.
   *
   * INSIDE a project, deliberately. A bare URL now shows two tabs, so loading
   * BASE here would have quietly reduced this from thirteen tabs to two while
   * still printing "ok" for each — a check that got weaker without ever going
   * red, which is the failure mode this whole file exists to catch. */
  pg = await open(ctx, slug ? `${BASE}?project=${slug}` : BASE, errs);
  const tabs = (await state(pg)).visibleTabs;
  note(tabs.length > 3,
       `inside a project the tabs are reachable (${tabs.length} visible: ${tabs.join(",")})`);
  for (const t of tabs) {
    /* RE-ENTER FIRST. Clicking Projects LEAVES the project (2026-07-31 — two
     * controls that read the same must do the same, and "← all projects" always
     * did), which hides every project tab. This loop clicked them in nav order,
     * so the first click stranded the other nine and the run died on a click
     * timeout. Reopening between tabs keeps the coverage instead of skipping
     * the tab that causes it — and asserts the leave behaviour on the way. */
    if (slug && !(await state(pg)).inProject) {
      await pg.click(`.pjCard .pjOpen[data-slug="${slug}"]`);
      await pg.waitForTimeout(1500);
    }
    await pg.click(`#navPanel nav button[data-tab="${t}"]`);
    await pg.waitForTimeout(1800);
    const st = await state(pg);
    const chars = await pg.evaluate(id =>
      (document.getElementById(id)?.textContent || "").replace(/\s+/g, " ").trim().length, t);
    const active = st.activeSection.includes(t);
    note(active && chars >= 40,
         `tab "${t}" activates and renders (${active ? chars + " chars" : "DID NOT ACTIVATE — redirected to " + st.activeSection}`
         + (active ? ")" : ")"));
    // The one tab whose job is to leave: assert it did, rather than only
    // recovering from it above.
    if (t === "projects")
      note(!st.inProject, `clicking "Projects" LEAVES the project (inProject=${st.inProject})`);
  }
  await pg.close();

  note(errs.length === 0, `no uncaught page errors (${errs.length})`);
  [...new Set(errs)].slice(0, 6).forEach(e => bad.push(e));
  await ctx.close();
}

/* ---- the drawer must MOVE, not merely toggle a class -------------------- */
async function checkNav(browser, label, ctxOpts, tap) {
  const ctx = await browser.newContext(ctxOpts);
  const errs = [];
  const pg = await open(ctx, BASE, errs);
  const panelX = () => pg.evaluate(() =>
    Math.round(document.querySelector("#navPanel").getBoundingClientRect().x));
  const before = await panelX();
  tap ? await pg.tap("#navBurger") : await pg.click("#navBurger");
  await pg.waitForTimeout(700);
  const after = await panelX();
  note(before !== after, `${label}: burger MOVES the panel (x ${before} -> ${after})`);
  await pg.close();
  await ctx.close();
}

const cr = await chromium.launch({channel: "chromium"});   // headless-shell lacks libs here
await checkEntryStates(cr);
await checkNav(cr, "chromium desktop 1440", {viewport: {width: 1440, height: 900}}, false);
await checkNav(cr, "chromium iPhone (touch)", {...devices["iPhone 13"]}, true);
await cr.close();

const wk = await webkit.launch();                          // Safari's engine — the owner's phone
await checkNav(wk, "webkit iPhone (touch)", {...devices["iPhone 13"]}, true);
await wk.close();

if (bad.length) {
  console.log("\nCOCKPIT IS BROKEN:");
  bad.forEach(b => console.log("  - " + b));
  process.exit(1);
}
console.log("\nevery entry state is usable and the drawer moves in both engines");
