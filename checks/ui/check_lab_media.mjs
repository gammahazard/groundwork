/* Does the Lab tab still SHOW challenger previews, in a real browser?
 *
 *   node checks/ui/check_lab_media.mjs
 *
 * The preview URL used to be spelled in the browser as `/outputs/alt/<run>/...`,
 * which is only correct for one fixed project. It now comes from the server (the `media`
 * field of /api/lab/runs) so there is exactly one place that decides where a
 * project's data lives. That is a good change and also a new way to break the
 * gallery silently: a wrong prefix yields broken <img> tags, not an exception,
 * and a screenshot of a modal full of alt-text looks a lot like a modal.
 *
 * So this asserts on `img.naturalWidth`, which is 0 for an image that failed —
 * the only signal that distinguishes "rendered" from "requested and 404'd".
 *
 * Read-only: it opens the cockpit, clicks Lab, opens one run's detail modal.
 */
import {chromium} from "playwright";

const BASE = process.env.COCKPIT || "http://127.0.0.1:8000";
const fail = [], note = [];

const browser = await chromium.launch();
const page = await browser.newPage({viewport: {width: 1400, height: 1000}});
const errs = [];
page.on("pageerror", e => errs.push(String(e)));
page.on("console", m => { if (m.type() === "error") errs.push("console: " + m.text()); });

const PROJECT = process.env.GW_UI_PROJECT || "";
await page.goto(BASE + (PROJECT ? `/?project=${PROJECT}` : "/"),
                {waitUntil: "networkidle"});

// The Lab tab — by its text, since ids move.
const tab = page.locator("nav a, nav button, .tabs a, .tabs button")
  .filter({hasText: /lab|challenger|models/i}).first();
if (await tab.count()) await tab.click();
else fail.push("no Lab tab found in the nav");
await page.waitForTimeout(2500);

// 1. the runs table must have rows
const rows = await page.locator("tr[data-run]").count();
note.push(`${rows} challenger row(s) in the table`);
if (rows < 50) fail.push(`expected 54 challenger rows, saw ${rows}`);

// 2. LAB_MEDIA must be what the server said, not the old hardcoded guess
// `let` at the top of a classic script does NOT become a window property, so
// `window.LAB_MEDIA` is undefined and a check reading it can never fail. Read
// the binding by bare name (evaluate's function body sees the script's global
// lexical scope) and treat "not declared" as a failure, not a pass.
const media = await page.evaluate(
  () => (typeof LAB_MEDIA === "undefined" ? "«undeclared»" : LAB_MEDIA));
const served = await page.evaluate(async () =>
  (await (await fetch("/api/lab/runs")).json()).media);
note.push(`server media=${served}`);
if (media !== served)
  fail.push(`browser LAB_MEDIA=${media} disagrees with server media=${served}`);
else note.push(`browser LAB_MEDIA agrees: ${media}`);

// 3. open one detail modal and check the images actually DECODED
const det = page.locator("button.ldet").first();
if (!(await det.count())) fail.push("no detail button on any run row");
else {
  await det.click();
  await page.waitForTimeout(4000);
  const imgs = await page.evaluate(() =>
    [...document.querySelectorAll("#runModalBody .pkFig img")]
      .slice(0, 8)
      .map(i => ({src: i.getAttribute("src"), w: i.naturalWidth})));
  note.push(`${imgs.length} preview img(s) sampled`);
  if (!imgs.length) fail.push("detail modal rendered no preview images");
  const broken = imgs.filter(i => !i.w);
  if (broken.length)
    fail.push(`${broken.length}/${imgs.length} previews failed to load, e.g. ${broken[0].src}`);
  else if (imgs.length) note.push(`all sampled previews decoded (first ${imgs[0].w}px wide)`);
  if (imgs.length && !imgs[0].src.startsWith(served))
    fail.push(`preview src ${imgs[0].src} does not start with the server's media prefix`);
}

if (errs.length) fail.push(`page errors: ${errs.slice(0, 3).join(" | ")}`);

await browser.close();
for (const n of note) console.log("  ·", n);
if (fail.length) {
  console.log("\n  FAIL");
  for (const f of fail) console.log("   ✗", f);
  process.exit(1);
}
console.log("\n  ✓ lab previews resolve through the server-supplied media prefix");
