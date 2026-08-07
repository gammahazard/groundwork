/* THE TWO DASHBOARDS must stay separate: the machine's, and the open project's.
 *
 *     node checks/ui/check_dashboard_scope.mjs      # cockpit up on :8000
 *
 * The owner's rule (2026-07-31): the Dashboard sits in the GLOBAL half of the
 * nav, so it answers "what is on this machine". A serving model, image counts, a
 * scoreboard and an odometer are things you go INTO a project to see.
 *
 * TWO THINGS HAVE TO HOLD, and only the first is obvious:
 *
 *   1. the project dashboard is a TAB in the project nav group, so it is
 *      unreachable outside a project, and
 *   2. it is never FETCHED. Hiding the panel in CSS alone would leave one
 *      project's numbers sitting in the DOM with only a
 *      stylesheet between them and a reader. So this asserts zero /api/state
 *      requests, which is the assertion that actually pins the behaviour: the
 *      CSS half would still pass with the fetch restored.
 *
 * And the inverse, because a check that only proves absence passes just as well
 * on a page that renders nothing at all: opening the project must bring the
 * panel back WITH REAL VALUES in it.
 */
import {chromium} from "playwright";

const BASE = process.env.COCKPIT || "http://127.0.0.1:8000";
const bad = [];
const note = (ok, msg) => { console.log(`  ${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) bad.push(msg); };

const browser = await chromium.launch({channel: "chromium"});
const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
const errs = [];
page.on("pageerror", e => errs.push(String(e).slice(0, 160)));
page.on("console", m => { if (m.type() === "error") errs.push(m.text().slice(0, 160)); });
const stateCalls = [];
page.on("request", r => { if (r.url().includes("/api/state")) stateCalls.push(r.url()); });

const shown = () => page.evaluate(() => {
  const vis = id => { const e = document.getElementById(id); return !!e && e.offsetHeight > 0; };
  const txt = id => (document.getElementById(id) || {}).textContent;
  return {inProject: document.body.classList.contains("inProject"),
          proj: vis("projdash"), machine: vis("dashboard"),
          // The project Dashboard's nav button — hidden outside a project by
          // the same .navGroupProject rule as every other project tab.
          navBtn: (() => { const b = document.querySelector('nav button[data-tab="projdash"]');
                           return !!b && b.offsetParent !== null; })(),
          served: txt("stServed"), imgs: txt("stImages"), mae: txt("stServedMae")};
});

await page.goto(BASE, {waitUntil: "networkidle"});
await page.waitForTimeout(2000);
await page.evaluate(() => showTab("dashboard"));
await page.waitForTimeout(2500);


let s = await shown();
note(!s.inProject, `no project is open on a bare load (inProject=${s.inProject})`);
note(s.machine, `the MACHINE dashboard renders (visible=${s.machine})`);
note(!s.proj, `the PROJECT dashboard is not showing (visible=${s.proj})`);
note(!s.navBtn, `its nav button is hidden outside a project (visible=${s.navBtn})`);
note(stateCalls.length === 0,
     `/api/state was never requested (${stateCalls.length} call(s))`);
// The placeholders index.html ships with. Anything else means data got in.
note(s.served === "–" && s.imgs === "–",
     `project fields still hold their placeholders (served=${JSON.stringify(s.served)}, imgs=${JSON.stringify(s.imgs)})`);

// ---- the inverse: opening one must bring it back, populated ---------------
await page.evaluate(() => showTab("projects"));
await page.waitForTimeout(1200);
await page.click(".pjCard .pjOpen");
await page.waitForTimeout(3500);

s = await shown();
note(s.inProject, `opening a project sets inProject (${s.inProject})`);
note(s.navBtn, `the project Dashboard tab appears in the nav (${s.navBtn})`);
note(s.proj, `opening a project LANDS on it (visible=${s.proj})`);
note(!s.machine, `and the machine dashboard is no longer the active tab (${s.machine})`);
note(stateCalls.length > 0, `/api/state is requested now (${stateCalls.length} call(s))`);
note(s.served !== "–" && /\d/.test(s.imgs || ""),
     `it is populated with real values (served=${JSON.stringify(s.served)}, imgs=${JSON.stringify(s.imgs)})`);
note(/^\d\.\d{4}$|^\d+(\.\d+)?$/.test((s.mae || "").trim()),
     `the MAE is formatted, not a bare 0 (${JSON.stringify(s.mae)})`);

note(errs.length === 0, `no page errors (${errs.length}${errs.length ? ": " + errs[0] : ""})`);

await browser.close();
if (bad.length) {
  console.log("\nTHE DASHBOARD IS SHOWING THE WRONG SCOPE:");
  bad.forEach(b => console.log("  - " + b));
  process.exit(1);
}
console.log("\nthe Dashboard is project-free until a project is opened");
