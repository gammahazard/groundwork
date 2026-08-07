"use strict";
/* Image grids: the shared renderer behind the Fix queue, Training set and
 * Test set tabs. On a worker (IS_LAB) grids are a read-only mirror: no
 * delete/move/tag/editor — click opens the full image (the API refuses writes
 * server-side too). Bucket tags (BUCKET_OPTS/TAGGABLE/GRID_OF, bucketSelect,
 * renderBucketTotals, renderTotalsFromGrid, applyBucketHighlight) live in
 * buckets.js; the dot editor (openEditor) in editor.js. */

/* What a grid is called when it fails to load, in a sentence: "Couldn't load
 * the fix queue". The tab title is not always the right phrase for that. */
const GRID_LABEL = {
  needs_fix: "the fix queue", raw: "the training set",
  testset: "the test set",
};

/* The tab's entry point: skeleton while it loads, a "taking longer than usual"
 * notice at 5s, and a failure card with Retry if the fetch throws. An image
 * grid is the slowest thing in the cockpit (hundreds of images) and used to
 * render as an empty panel until it was ready — indistinguishable from an
 * empty collection, which is how a dead endpoint looked exactly like no data.
 *
 * `n: 8` because eight skeleton cards fill a laptop viewport without pretending
 * to predict the real count; guessing 160 placeholders would reflow hard. */
async function loadGrid(collection, sel){
  return loadInto(sel, () => renderGrid(collection, sel), {
    label: GRID_LABEL[collection] || "these images", kind: "card", n: 8});
}

async function renderGrid(collection, sel){
  const RO = !!window.IS_LAB;   // worker = read-only mirror: view, never edit
  const images = await apiOrThrow("/api/images/" + collection);
  const grid = $(sel); grid.innerHTML = "";
  if (!images.length){ grid.innerHTML = "<p class='hint'>Nothing here.</p>"; return; }
  // THE VOCABULARY BEFORE THE TAGS. bucketSelect() builds each image's <select>
  // from BUCKET_OPTS, so fetching it after the cards were built would render
  // every dropdown from the stale default list — a project with custom tags
  // would show the old ones until something else forced a re-render.
  if (TAGGABLE[collection]) await loadBucketVocab();
  const buckets = TAGGABLE[collection] ? await api("/api/buckets") : {};
  if (TAGGABLE[collection])
    renderBucketTotals(images, buckets, TAGGABLE[collection],
                       collection === "testset" ? "holdout" : "training", collection);
  const stems = images.map(x => x.stem);
  for (const t of images){
    const card = document.createElement("div"); card.className = "card";
    let delta = "";
    if (t.target != null){
      const d = t.count - t.target, cls = d === 0 ? "ok" : d < 0 ? "short" : "over";
      delta = `<span class="delta ${cls}">${d>0?"+":""}${d}</span>`;
    }
    const tgt = t.target != null ? ` / target ${t.target}` : "";
    const mv = RO ? "" : collection === "raw"
             ? `<button class="mv" title="Send to test set (holdout)">🎯</button>`
             : collection === "testset" ? `<button class="mv" title="Return to training">↩</button>` : "";
    // Card-level LA shortcut: open the editor AND the LocateAnything modal in one
    // click — the rescue path for an image the model messed up (see la.js).
    const laBtn = RO ? "" : `<button class="laShort" title="Second opinion: open the editor + run LocateAnything on this image">✛</button>`;
    // Earmarked-for-test flag (set on the bot's 🎯 Wrong → test). Only meaningful
    // in the fix queue — it's the image's pending destination. Once graduated the
    // earmark is cleared and the Training/Test *tab* is the source of truth, so we
    // don't badge those (avoids a stale flag ever contradicting the image's folder).
    const mark = (collection === "needs_fix" && t.earmarked)
      ? `<span class="earmark" title="Earmarked for the test set — Done sends it to the holdout">🎯 test</span>` : "";
    const bkt = !RO && TAGGABLE[collection]
      ? `<div class="bucketWrap">${bucketSelect(buckets[t.stem])}</div>` : "";
    card.dataset.stem = t.stem;
    card.dataset.count = t.count;
    card.dataset.bucket = buckets[t.stem] || "untagged";
    card.innerHTML = `${RO ? "" : '<button class="del" title="Remove image">🗑</button>'}${laBtn}${mv}${mark}
      <img loading="lazy" decoding="async" width="480"
           src="${withProject(`/img/${collection}/${t.stem}?w=480&v=${t.mtime || 0}`)}">
      <div class="cap">${delta}<b>${t.count}</b>${tgt}<br>${t.stem}</div>${bkt}`;
    card.onclick = RO
      ? () => window.open(withProject(`/img/${collection}/${t.stem}?t=${Date.now()}`), "_blank")
      : () => openEditor(collection, t.stem, collection === "needs_fix", stems);
    if (laBtn) card.querySelector(".laShort").onclick = async e => {
      e.stopPropagation();
      await openEditor(collection, t.stem, collection === "needs_fix", stems);
      if (window.openLaModal) openLaModal();
    };
    if (bkt){
      const sb = card.querySelector(".bucket");
      sb.onclick = e => e.stopPropagation();      // don't open the editor
      sb.onchange = async e => {
        e.stopPropagation();
        await fetch(`/api/buckets/${t.stem}`, {method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({label: sb.value})});
        sb.className = "bucket b-" + sb.value;     // recolor to match new tag
        card.dataset.bucket = sb.value;            // keep filter/totals live
        renderTotalsFromGrid(collection);
        applyBucketHighlight(collection);
      };
    }
    // Surgical removal: taking the one card out (instead of rebuilding the
    // grid) keeps the scroll position where you were working.
    const removeCard = () => {
      card.remove();
      if (!grid.querySelector(".card"))
        grid.innerHTML = "<p class='hint'>Nothing here.</p>";
      if (TAGGABLE[collection]){
        renderTotalsFromGrid(collection); applyBucketHighlight(collection);
      }
      loadState();
    };
    if (mv) card.querySelector(".mv").onclick = async e => {
      e.stopPropagation();
      const method = collection === "raw" ? "POST" : "DELETE";
      const r = await api(`/api/testset/${t.stem}`, {method});
      if (r && r.ok === false) return alert("🔒 " + (r.error || "move refused"));
      removeCard();
    };
    const delBtn = card.querySelector(".del");
    if (delBtn) delBtn.onclick = async e => {
      e.stopPropagation();
      if (!confirm(`Remove ${t.stem} from ${collection}? This deletes its image + dots.`)) return;
      const r = await api(`/api/image/${collection}/${t.stem}`, {method:"DELETE"});
      if (r && r.ok === false && r.error) return alert("🔒 " + r.error);
      removeCard();
    };
    grid.appendChild(card);
  }
  // ARRIVAL ORDER, stamped once. list_images sorts by birth time (oldest
  // first), so a child's index IS when its image arrived — which is what the
  // chip filter means by "newest first", and what restores this exact layout
  // when the last chip is switched off. Deliberately not the `mtime` field: an
  // in-place ✂ crop bumps that, as list_images' own docstring says, so it
  // answers "when was the file last written", not "when did this image
  // arrive".
  [...grid.children].forEach((el, i) => { el.dataset.ord = i; });
  if (TAGGABLE[collection]) applyBucketHighlight(collection);   // restore filter after re-render
}
