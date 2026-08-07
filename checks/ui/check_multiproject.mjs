/* The pivot's central promise, tested: one project's data never appears under
 * another's name.
 *
 *     node checks/ui/check_multiproject.mjs      # cockpit up on :8000
 *     GW_UI_PROJECT=<slug>  names the project WITH runs used for steps 5-7
 *                           (default: the first landing card that is not the
 *                           probe — it should be a project that has runs)
 *
 * WHY THIS EXISTS. With only one project, nothing exercises the multi-project
 * path. When a second was finally created by hand, four bugs fell out at once
 * — and every one of them was the same failure, one project showing another's
 * data:
 *
 *   1. with two projects a bare load landed on a Dashboard describing no
 *      project, with the Projects tab never loaded and zero cards
 *   2. extension tabs were never gated client-side, so a project with no
 *      extensions showed every extension tab
 *   3. the runs table kept the previous project's rows across a switch
 *   4. /api/runs filtered with `r.get("project", p.slug) == p.slug`, so every
 *      row predating the project column matched EVERY project — a second
 *      project showed 26 of the first one's 27 runs as its own
 *
 * None of those are visible with one project, and all four would come back
 * silently. The other checks pass green on a cockpit that has this class of
 * bug, because they only ever load one project.
 *
 * THE FIXTURE CREATES AND DELETES A THROWAWAY PROJECT (probe_project.py in
 * this directory, which refuses to name anything but its own reserved slug).
 * Teardown runs in a finally — a check that can leave a stray project behind
 * would be worse than none.
 */
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");
const BASE = (process.env.COCKPIT || "http://localhost:8000/").replace(/\/?$/, "/");
const PY = process.env.GW_PY || join(REPO, ".venv", "bin", "python");
const FIX = join(HERE, "probe_project.py");
const bad = [];
const note = (ok, msg) => { console.log(`  ${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) bad.push(msg); };

const fixture = cmd => execFileSync(PY, [FIX, cmd], {encoding: "utf8"}).trim();

let slug;
try {
  slug = fixture("create");
  console.log(`  fixture: created throwaway project ${slug}`);

  const b = await chromium.launch({channel: "chromium"});
  const pg = await b.newPage({viewport: {width: 1280, height: 1000}});
  const errs = [];
  pg.on("pageerror", e => errs.push("uncaught: " + e.message));
  pg.on("response", r => { if (r.status() >= 400 && !/favicon/.test(r.url()))
    errs.push(`HTTP ${r.status()} ${r.url().replace(BASE, "/")}`); });

  await pg.goto(BASE, {waitUntil: "networkidle"});
  await pg.waitForTimeout(2500);

  // 1. two projects -> the landing, not a stranded Dashboard
  let s = await pg.evaluate(() => ({
    active: [...document.querySelectorAll(".tab")].filter(t => t.classList.contains("active")).map(t => t.id),
    cards: document.querySelectorAll(".pjCard[data-slug]").length,
    slugs: [...document.querySelectorAll(".pjOpen[data-slug]")].map(x => x.dataset.slug),
    inProject: document.body.classList.contains("inProject"),
  }));
  note(s.cards >= 2 && !s.inProject && s.active.includes("projects"),
       `with 2 projects a bare load shows the landing (active=${s.active}, cards=${s.cards})`);

  // The project WITH data, for the switch-back steps. Named, or discovered.
  const FULL = process.env.GW_UI_PROJECT || s.slugs.find(x => x !== slug);
  note(!!FULL, `a second (real) project exists to switch against (${FULL})`);

  // 2. open the probe: its own project, and no tab it was not granted
  await pg.click(`.pjOpen[data-slug="${slug}"]`);
  await pg.waitForTimeout(2500);
  s = await pg.evaluate(() => ({
    slug: document.body.dataset.project,
    tabs: [...document.querySelectorAll("#navPanel nav button")]
      .filter(x => x.checkVisibility({checkVisibilityCSS: true})).map(x => x.dataset.tab),
  }));
  note(s.slug === slug, `opening it sets the project (${s.slug})`);

  /* 3. EVERY project-scoped surface must be empty. This is the promise.
   *
   * "collect" is deliberately NOT on this list: Data collection is a
   * machine-level tab (a bot is a thing running on this box, and that page
   * shows every one of them whichever project it feeds). Another project's
   * bots showing there is the correct answer; what protects isolation is
   * OWNERSHIP — a bot is listed under the project that owns it — which
   * check_bot_ownership.py asserts server-side. */
  for (const [tab, sel] of [["runs", "#runsTable"], ["train", "#trainGrid"],
                            ["testset", "#testGrid"], ["fix", "#fixGrid"]]) {
    await pg.evaluate(n => showTab(n), tab);
    await pg.waitForTimeout(2000);
    const o = await pg.evaluate(x => {
      const e = document.querySelector(x);
      return {rows: e.querySelectorAll("tbody tr").length,
              cards: e.querySelectorAll(".card,.botCard,.pjCard").length};
    }, sel);
    note(o.rows === 0 && o.cards === 0,
         `"${tab}" is empty in the new project (rows=${o.rows} cards=${o.cards})`);
  }

  // 4. the platform overview counts BOTH, and is not one project's numbers
  await pg.evaluate(() => showTab("dashboard"));
  await pg.waitForTimeout(2500);
  const ov = await pg.evaluate(() => ({
    projects: (document.querySelector("#overviewTiles .tile span") || {}).textContent,
    rows: document.querySelectorAll("#overviewProjects .ovProject").length,
  }));
  note(Number(ov.projects) >= 2 && ov.rows >= 2,
       `the Dashboard overview counts every project (tile=${ov.projects}, rows=${ov.rows})`);

  // 5. going back must RESTORE the real project completely
  await pg.evaluate(() => showTab("projects"));
  await pg.waitForTimeout(1200);
  await pg.click(`.pjOpen[data-slug="${FULL}"]`);
  await pg.waitForTimeout(2000);
  await pg.evaluate(() => showTab("runs"));
  await pg.waitForTimeout(3500);
  const back = await pg.evaluate(() => ({
    slug: document.body.dataset.project,
    rows: document.querySelectorAll("#runsTable tbody tr").length,
  }));
  note(back.slug === FULL && back.rows > 0,
       `switching back restores the full project's runs (${back.rows} rows)`);

  // 6. AND BACK AGAIN — the order matters, which is why this step exists.
  // Going landing -> probe -> full never catches a stale runs table, because
  // the probe is opened first and there is nothing to leak yet. The leak needs
  // a project WITH runs opened before one without: the full project's rows are
  // contributed, then the probe opens and must not inherit them. Written after
  // discovering this file passed with resetRunsTable() removed.
  await pg.evaluate(() => showTab("projects"));
  await pg.waitForTimeout(1200);
  await pg.click(`.pjOpen[data-slug="${slug}"]`);
  await pg.waitForTimeout(1500);
  await pg.evaluate(() => showTab("runs"));
  await pg.waitForTimeout(2500);
  const again = await pg.evaluate(() => ({
    slug: document.body.dataset.project,
    rows: document.querySelectorAll("#runsTable tbody tr").length,
  }));
  note(again.slug === slug && again.rows === 0,
       `re-opening the empty project AFTER a full one shows no runs (${again.rows})`);

  // 7. AND NOT EVEN FOR A MOMENT. The step above passes with resetRunsTable()
  // removed, because both ledger loaders reply and overwrite their sources with
  // empty — so by the time anything is measured the table is already correct.
  // What the reset actually prevents is the WINDOW: from opening the project
  // until the ledgers answer, the previous project's rows sit on screen under
  // the new project's name. Over a cold link that is seconds, not frames.
  // Held open deliberately here (the ledgers are delayed) rather than raced, so
  // the assertion is about behaviour and not about timing luck.
  await pg.route("**/api/runs*", async r => { await new Promise(x => setTimeout(x, 2500)); r.continue(); });
  await pg.route("**/api/lab/runs*", async r => { await new Promise(x => setTimeout(x, 2500)); r.continue(); });
  await pg.evaluate(() => showTab("projects"));
  await pg.waitForTimeout(1200);
  await pg.click(`.pjOpen[data-slug="${FULL}"]`);
  await pg.waitForTimeout(4000);                 // let the full project's rows land
  await pg.evaluate(() => showTab("runs"));
  await pg.waitForTimeout(4000);
  const loaded = await pg.evaluate(() => document.querySelectorAll("#runsTable tbody tr").length);
  await pg.evaluate(() => showTab("projects"));
  await pg.waitForTimeout(800);
  await pg.click(`.pjOpen[data-slug="${slug}"]`);
  await pg.evaluate(() => showTab("runs"));
  await pg.waitForTimeout(300);                  // ledgers are still in flight
  const during = await pg.evaluate(() => ({
    slug: document.body.dataset.project,
    rows: document.querySelectorAll("#runsTable tbody tr").length,
  }));
  note(loaded > 0 && during.rows === 0,
       `while its ledgers load, the new project shows none of the old one's `
       + `${loaded} rows (saw ${during.rows})`);
  await pg.unroute("**/api/runs*");
  await pg.unroute("**/api/lab/runs*");

  note(errs.length === 0, `no uncaught errors or failed requests (${errs.length})`);
  [...new Set(errs)].slice(0, 6).forEach(e => bad.push(e));
  await b.close();
} finally {
  try { fixture("destroy"); console.log("  fixture: throwaway project removed"); }
  catch (e) { console.log("  *** FIXTURE TEARDOWN FAILED: " + e.message); bad.push("teardown failed"); }
}

if (bad.length) {
  console.log("\nMULTI-PROJECT ISOLATION IS BROKEN:");
  bad.forEach(x => console.log("  - " + x));
  process.exit(1);
}
console.log("\none project's data never appears under another's name");
