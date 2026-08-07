"use strict";
/* The Images tab: three collections, one screen.
 *
 * Fix queue, Training set and Test set were three tabs rendering the SAME grid
 * from the same module (grids.js), with the same cards and the same editor. The
 * only difference is which folder a picture is in — a property of the picture,
 * not a reason for a separate screen. On a phone they were three taps deep in a
 * drawer.
 *
 * WHY PANES AND NOT A REWRITE. Each pane keeps the exact markup it had as a
 * tab, ids included: #fixGrid / #trainGrid / #testGrid, both bucket-total
 * strips, the audit and dedup outputs, and all three badges. Nothing that reads
 * them had to learn anything. The badges matter most — dashboard.js writes them
 * with no null guard, so a missing one is a TypeError that takes the project
 * Overview down with it.
 *
 * LAZY, AND ONLY ONCE. Switching pane loads that collection; re-entering does
 * not re-fetch, because loadGrid rebuilds every card and would throw away the
 * scroll position you were working at. The tab's own entry point loads whatever
 * pane is showing, so arriving and switching behave identically.
 *
 * HALVES IS NOT HERE, deliberately. It is an EXTENSION tab (projects.js
 * EXTENSION_TABS), shown only for projects that declare it — folding a
 * conditional tab into a fixed three-way control would mean a control that
 * sometimes has four buttons and sometimes three, for a reason the reader
 * cannot see.
 */

/* pane -> [collection, grid selector]. The collection names are grids.js's,
 * unchanged: they are what /api/images/<collection> is called. */
const IMG_PANES = {
  fix:     ["needs_fix", "#fixGrid"],
  train:   ["raw", "#trainGrid"],
  testset: ["testset", "#testGrid"],
};

let _imgPane = "train";          // what Training set used to be: the busiest one
const _imgLoaded = new Set();

function imagesShow(pane, {reload = false} = {}) {
  if (!IMG_PANES[pane]) pane = "train";
  _imgPane = pane;
  document.querySelectorAll(".imgPane").forEach(el =>
    el.hidden = el.id !== `pane-${pane}`);
  document.querySelectorAll(".imgSegBtn").forEach(b => {
    const on = b.dataset.pane === pane;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
  });
  const [collection, sel] = IMG_PANES[pane];
  if (reload || !_imgLoaded.has(pane)) {
    _imgLoaded.add(pane);
    loadGrid(collection, sel);
  }
}

/* The tab's entry point. Called by showTab, so arriving at Images and switching
 * pane inside it go through the same path. */
function loadImages() { imagesShow(_imgPane); }

/* A pane whose contents changed underneath us must re-fetch next time it is
 * shown — moving an image to the holdout changes TWO collections, and the one you
 * are not looking at would otherwise still show it. */
function imagesInvalidate(pane) {
  if (pane) _imgLoaded.delete(pane); else _imgLoaded.clear();
}

document.querySelectorAll(".imgSegBtn").forEach(b => {
  b.onclick = () => imagesShow(b.dataset.pane);
});

window.imagesShow = imagesShow;
window.loadImages = loadImages;
window.imagesInvalidate = imagesInvalidate;
