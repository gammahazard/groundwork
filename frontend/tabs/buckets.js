"use strict";
/* Bucket tags (object/capsule/stacked) + chip totals + click-to-filter — for
 * both the Test set (eval lens: per-bucket MAE) and Training set (inventory
 * only) tabs. Loaded right after cockpit.js, whose loadGrid() calls
 * bucketSelect/renderBucketTotals/applyBucketHighlight at render time. */

/* THE VOCABULARY LIVES ON THE PROJECT, not here (/api/bucket-vocab).
 *
 * It used to be a hardcoded array on this line, which is why adding a tag meant
 * editing JavaScript and redeploying. Nothing server-side ever required that:
 * eval_core.bucket_mae groups by whatever tags exist on disk
 * (testset_buckets.bucketize), so a new tag is reported by every FUTURE run the
 * moment an image carries it — the list was only ever a UI affordance.
 *
 * Runs already scored keep the buckets they were scored WITH, which is correct:
 * a past run cannot be re-bucketed without re-scoring it.
 *
 * Tagging never touches the model. It is an INVENTORY count on the training
 * tab, and on the holdout an EVAL LENS — per-bucket MAE, so one object type
 * regressing cannot hide inside a blended average. `stacked-capsule` (owner,
 * 2026-08-01) is the case that motivated it: stacking is the hard case and
 * capsules the hard shape, and an image that is both can regress while tablet
 * stacks improve, with the blended number never moving.
 *
 * The four below are the pre-fetch value AND the manifest's default, so a slow
 * load shows the previous vocabulary rather than an empty dropdown, and a
 * project nobody has edited behaves exactly as before. */
/* One escaper for this file, declared before anything renders. Tag names are
   user text and reach markup from four directions — a class, a data attribute,
   a title and an <option> value. */
const _bkEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

// Minimal until the API answers — the project's real vocabulary replaces
// these on load, so anything listed here is a flash of someone else's tags.
let BUCKET_OPTS = ["untagged", "object"];
let BUCKET_ORDER = ["object"];

async function loadBucketVocab(){
  try {
    // api() applies withProject(), so this asks about the OPEN project rather
    // than the default one — the whole point of a per-project vocabulary.
    const v = await api("/api/bucket-vocab");
    if (!Array.isArray(v.buckets) || !v.buckets.length) return v;
    BUCKET_ORDER = v.buckets.slice();
    // ORPHANS ARE OFFERED TOO. An image can carry a tag the vocabulary no longer
    // lists (renamed away, or edited on another machine); leaving it out of the
    // dropdown means its own <select> cannot show its own value, so opening it
    // would silently retag the image.
    const orphans = Object.keys(v.orphans || {});
    BUCKET_OPTS = ["untagged", ...v.buckets, ...orphans];
    return v;
  } catch (e) { return null; }
}
// Tags work on BOTH the holdout (eval lens -> per-bucket MAE) and the training
// set (inventory only — count_eval never reads tags of non-eval images).
const TAGGABLE = {testset: "#testBucketTotals", raw: "#trainBucketTotals"};
const GRID_OF = {testset: "#testGrid", raw: "#trainGrid"};

function bucketSelect(cur){
  /* THIS IS THE ONE THAT WAS EXPLOITED. `<option value="${o}">` put a raw tag
   * name inside an attribute, so a tag called `x" onmouseover="alert(1)` closed
   * the value and opened a live handler — on 176 <option> elements, one per
   * image, which any other signed-in user would run by hovering. Measured
   * 2026-08-05, before the server refused such names. */
  const c = cur || "untagged";
  const ce = _bkEsc(c);
  const opts = BUCKET_OPTS.map(o => {
    const oe = _bkEsc(o);
    return `<option value="${oe}"${o === c ? " selected" : ""}>${oe}</option>`;
  }).join("");
  return `<select class="bucket b-${ce}" title="Eval bucket — per-type MAE on the Runs tab (does not affect the model)">${opts}</select>`;
}

/* Click a chip to pull that tag's images to the TOP of the grid; click again to
 * send them back. Several tags can be on at once, and the order you clicked
 * them in is the order the groups stack — so "capsule then stacked" reads
 * capsules first, stacked underneath, everything else below that.
 *
 * AN ARRAY, NOT A STRING. This was one tag at a time; the click order IS the
 * layout, so it has to be a sequence rather than a set — which is also why the
 * chips show their position number once more than one is on.
 */
const bucketFilter = {};                  // {raw: ["capsule", "stacked"], …}

function renderBucketTotals(images, buckets, sel, word, collection){
  const el = $(sel); if (!el) return;
  if (!images.length){ el.innerHTML = ""; return; }
  const t = {}, p = {};
  for (const x of images){
    const b = buckets[x.stem] || "untagged";
    t[b] = (t[b] || 0) + 1; p[b] = (p[b] || 0) + x.count;
  }
  // Simple -> hard, so the chips read as a difficulty ramp rather than
  // alphabetically — the PROJECT's order, which is why reordering tags is a
  // real edit rather than cosmetic. Anything not in the vocabulary (an orphan
  // tag, or "untagged") sorts last.
  const order = b => { const i = BUCKET_ORDER.indexOf(b); return i < 0 ? 9 : i; };
  const keys = Object.keys(t).sort((a, b) => order(a) - order(b) || a.localeCompare(b));
  const totalPills = images.reduce((s, x) => s + x.count, 0);
  const on = bucketFilter[collection] || [];
  const chips = keys.map(b => {
    const at = on.indexOf(b);
    // The position number appears only when it MEANS something — with one tag
    // selected there is no stacking order to explain, and a lone "1" reads as
    // a count of something.
    const ord = at >= 0 && on.length > 1
      ? `<span class="bchipOrd">${at + 1}</span>` : "";
    /* ESCAPED, EVERY TIME. A tag name is user text and this puts it inside a
     * class, a data attribute, a title and the body — four ways to break out.
     * The server refuses quotes now, but a renderer that only works because
     * something upstream is careful is one bad release from a stored XSS, and
     * this one already had one: `x" onmouseover="alert(1)` became a live
     * handler on 176 elements. */
    const be = _bkEsc(b);
    return `<span class="bchip b-${be}${at >= 0 ? " active" : ""}" data-b="${be}" `
      + `title="${at >= 0 ? `Click to send ${be} images back`
                          : `Click to pull ${be} images to the top`}">`
      + `${ord}${be} <b>${t[b]}</b><span class="mMeta"> image${t[b]===1?"":"s"} · ${p[b]} objects</span></span>`;
  }).join("");
  // EDIT sits at the END of the chips, so the row still reads as data first and
  // the affordance is where you already are when you notice a tag is missing.
  el.innerHTML = `<span class="bktLead">${images.length} ${word} images · ${totalPills} objects</span>${chips}`
    + `<button class="bchip bchipEdit" data-edit="1" title="Add, rename or reorder this project's image tags">＋ tags</button>`;
  const ed = el.querySelector(".bchip[data-edit]");
  if (ed) ed.onclick = () => window.openTagEditor && window.openTagEditor(collection);
  el.querySelectorAll(".bchip[data-b]").forEach(ch => ch.onclick = () => {
    // TOGGLE, PRESERVING CLICK ORDER: a newly-picked tag joins the END of the
    // list, so it stacks under the ones already chosen. Removing one closes
    // the gap and everything below it moves up a group.
    const cur = bucketFilter[collection] || [];
    const i = cur.indexOf(ch.dataset.b);
    bucketFilter[collection] = i >= 0
      ? cur.filter(x => x !== ch.dataset.b)
      : [...cur, ch.dataset.b];
    renderTotalsFromGrid(collection);          // refresh chip active states
    applyBucketHighlight(collection);
  });
}

// Rebuild the totals row from the CURRENT cards (their tags update in place),
// so chip counts and the active filter stay live after a tag change.
function renderTotalsFromGrid(collection){
  const sel = TAGGABLE[collection]; if (!sel) return;
  const cards = [...document.querySelectorAll(GRID_OF[collection] + " .card")];
  const images = cards.map(c => ({stem: c.dataset.stem, count: +c.dataset.count || 0}));
  const buckets = {};
  cards.forEach(c => { if (c.dataset.bucket && c.dataset.bucket !== "untagged")
    buckets[c.dataset.stem] = c.dataset.bucket; });
  renderBucketTotals(images, buckets, sel,
                     collection === "testset" ? "holdout" : "training", collection);
}

/* Pull the selected tags' images to the top, and animate the move.
 *
 * ORDER, in full:
 *   1. the selected tags, in the order they were CLICKED
 *   2. inside each of those, NEWEST UPLOAD FIRST
 *   3. everything else after, in the grid's original order, untouched
 *
 * "NEWEST" IS THE SERVER'S ORDER REVERSED, not a timestamp comparison.
 * labelio.list_trays sorts by birth time (oldest first) and hands back an
 * `mtime` that is deliberately something else — its own docstring notes an
 * in-place ✂ crop bumps mtime, "acceptable" for cache-busting and wrong for
 * "when did this arrive". So renderGrid stamps each child's position as
 * data-ord and we read that: it is the arrival order the server actually
 * computed, rather than a proxy for it.
 *
 * FLIP (First-Last-Invert-Play) rather than animating layout: the cards are
 * moved in the DOM, then transformed BACK to where they were and released, so
 * the browser animates a compositor-only transform instead of reflowing a grid
 * of 240 images sixty times a second.
 *
 * IT HAS TO CO-EXIST WITH TWO EXISTING RULES on .card (02-runs.css): a
 * `transition: transform .12s` and a `:hover { transform: translateY(-2px) }`.
 * The inline transform written here beats the hover rule (inline always wins),
 * so passing the pointer over a card mid-flight cannot yank it; and .flipping
 * is a two-class selector, so it overrides the .12s without !important.
 */
function applyBucketHighlight(collection){
  const grid = document.querySelector(GRID_OF[collection]);
  if (!grid) return;
  const sel = bucketFilter[collection] || [];
  // EVERY CHILD, not just .card: the training grid carries a "Synthetic —"
  // divider, and leaving it behind while the cards move would strand a heading
  // over images it no longer describes.
  const kids = [...grid.children].filter(el => el.dataset.ord !== undefined);
  if (!kids.length) return;

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const first = still ? null : new Map(kids.map(el => [el, el.getBoundingClientRect()]));

  const active = sel.length > 0;
  kids.forEach(el => {
    if (!el.classList.contains("card")) {
      // The divider labels a grouping that a filter has just broken up, so it
      // is hidden while one is on rather than left asserting something false.
      el.classList.toggle("gridHeadHidden", active);
      return;
    }
    const hit = sel.includes(el.dataset.bucket);
    el.classList.toggle("bdim", active && !hit);
    el.classList.toggle("bmatch", active && hit);
  });

  const ord = el => +el.dataset.ord || 0;
  const rank = el => {
    if (!el.classList.contains("card")) return active ? sel.length : -1;
    const i = sel.indexOf(el.dataset.bucket);
    return i < 0 ? sel.length : i;
  };
  const sorted = kids.slice().sort((a, b) => {
    const ra = rank(a), rb = rank(b);
    if (ra !== rb) return ra - rb;
    // Inside a chosen group: newest arrival first. Everywhere else: exactly the
    // order the server sent, which is also what restores the original layout
    // (divider included) the moment the last chip is switched off.
    return ra < sel.length ? ord(b) - ord(a) : ord(a) - ord(b);
  });
  sorted.forEach(el => grid.appendChild(el));

  if (still) return;
  // INVERT: put everything back where it was, with no transition, then release
  // on the next frame so the browser has a start and an end to interpolate.
  const moved = [];
  for (const el of kids){
    const a = first.get(el), b = el.getBoundingClientRect();
    const dx = a.left - b.left, dy = a.top - b.top;
    if (!dx && !dy) continue;
    el.style.transition = "none";
    el.style.transform = `translate(${dx}px, ${dy}px)`;
    moved.push(el);
  }
  if (!moved.length) return;
  void grid.offsetHeight;                       // flush the inverted positions
  requestAnimationFrame(() => moved.forEach(el => {
    el.classList.add("flipping");
    el.style.transition = "";
    el.style.transform = "";
    // Clean up so hover's own transform works again afterwards. `once` matters:
    // without it a hover translate would re-fire this handler forever.
    el.addEventListener("transitionend", function done(ev){
      if (ev.propertyName !== "transform") return;
      el.classList.remove("flipping");
      el.removeEventListener("transitionend", done);
    });
  }));
}
