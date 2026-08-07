"use strict";
/* The Guide, as a stepper: one step on screen at a time.
 *
 * It was 166 lines of prose in index.html, all eight steps expanded, which on a
 * phone is a single scroll several screens long. Nobody reads that; they scroll
 * past it looking for the bit they need, and the bit they need has no landmark.
 *
 * ONE STEP AT A TIME, with the rail above showing where you are and how much is
 * left. The content is data (guide_steps.js) rather than markup, so this file
 * can count the steps, address them and render them — and so the prose can be
 * edited without touching the page shell.
 *
 * MOUNTED TWICE, from one definition. The Guide tab is one instance; the login
 * page's "How it works" modal is another, and it shows THE SAME steps with the
 * same pictures. It used to be a hand-written list of titles and ledes in
 * auth.js — a second, thinner telling that could drift out of step with the real
 * one and, being separate, silently did not gain the Export step. Two mounts of
 * one array cannot disagree.
 *
 * INSTANCE STATE, NOT MODULE STATE. The index used to be a module-level `_gAt`
 * and go() addressed #gStage/#gRail by id, which is fine for one stepper and
 * wrong the moment there are two: paging the modal would have moved the tab
 * behind it. Each mount closes over its own index and only ever queries inside
 * its own host.
 *
 * DEEP-LINKABLE, but only the tab. The step lands in the URL hash, so "see step
 * 4" is a link you can send someone and a reload keeps your place. The hash is
 * the step's NAME (#guide/split), not its number: renumbering must not break a
 * link somebody saved. The modal does not touch the hash — it is a glance, not
 * a destination, and hijacking the URL from a dialog is how you lose the page
 * behind it.
 *
 * KEYBOARD AND SWIPE. Left/right arrows move; on a touchscreen a horizontal
 * drag does. Both go through the same go(), so there is one definition of what
 * moving means. Arrows only act while that instance is the one on screen —
 * `isActive` — or the modal and the tab would both page on one keypress.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: hide anything. Every step is reachable from
 * the rail in one tap, and the whole guide is still one page for Ctrl+F — the
 * steps are display-toggled, not built on demand, so the browser's own find
 * still works across all of them.
 */

/* Name, not index — see the header. Falls back to the first step. */
function _gIndexFromHash() {
  const m = (location.hash || "").match(/^#guide\/([a-z-]+)$/);
  if (!m) return 0;
  const i = GUIDE_STEPS.findIndex(s => s.key === m[1]);
  return i < 0 ? 0 : i;
}

function _gMedia(s) {
  if (!s.media) return "";
  /* loading="lazy" matters here: the art is 27-257 KB and only one step is on
   * screen, so the others must not be fetched to sit in a hidden div.
   * decoding="async" keeps a big frame off the main thread. */
  return `<figure class="gFig">
      <img src="${s.media.src}" alt="${s.media.alt}" loading="lazy" decoding="async">
      <figcaption>${s.media.cap}</figcaption>
    </figure>`;
}

function _gStepHTML(s, i) {
  const note = s.note
    ? `<div class="gNote g${s.note.kind === "care" ? "Care" : "Why"}">
         <b>${s.note.head}</b>${s.note.text}</div>` : "";
  return `<article class="gStep" id="gStep-${s.key}" data-i="${i}" ${i ? "hidden" : ""}>
      <p class="gStepNum">Step ${i + 1} of ${GUIDE_STEPS.length}</p>
      <h3>${s.title}</h3>
      <p class="gLede">${s.lede}</p>
      ${_gMedia(s)}
      ${s.body}
      ${note}
    </article>`;
}

/* Build a stepper inside `host`. Returns {go, at} or null if already built.
 *
 * deepLink  write the step to location.hash and read it back (the tab only)
 * isActive  () => is this instance the one the keyboard should drive */
function mountGuide(host, {deepLink = false, isActive = () => true} = {}) {
  if (!host || host.dataset.built) return null;
  host.dataset.built = "1";
  let at = 0;

  host.innerHTML =
    `<nav class="gRail" aria-label="Guide steps">` +
    GUIDE_STEPS.map((s, i) =>
      `<button type="button" data-i="${i}" title="${s.title}">
         <span class="n">${i + 1}</span><span class="t">${s.title}</span>
       </button>`).join("") +
    `</nav>
     <div class="gSteps">${GUIDE_STEPS.map(_gStepHTML).join("")}</div>
     <div class="gNav">
       <button type="button" class="gNavBtn gNavPrev"></button>
       <button type="button" class="gNavBtn gNavNext"></button>
     </div>`;

  function go(i, {push = deepLink} = {}) {
    const n = GUIDE_STEPS.length;
    at = ((i % n) + n) % n;                     // wrap, so the loop is a loop
    host.querySelectorAll(".gSteps .gStep").forEach(el =>
      el.hidden = +el.dataset.i !== at);
    host.querySelectorAll(".gRail button").forEach((b, k) => {
      const on = k === at;
      b.classList.toggle("on", on);
      b.setAttribute("aria-current", on ? "step" : "false");
    });
    const prev = host.querySelector(".gNavPrev"), next = host.querySelector(".gNavNext");
    if (prev) prev.textContent = `← ${GUIDE_STEPS[(at - 1 + n) % n].title}`;
    if (next) next.textContent = `${GUIDE_STEPS[(at + 1) % n].title} →`;
    if (push && deepLink) {
      const h = `#guide/${GUIDE_STEPS[at].key}`;
      if (location.hash !== h) history.replaceState(null, "", h);
    }
    /* Only scroll when the step is off screen — jumping the page on every arrow
     * press is worse than not scrolling at all. Inside a modal, scroll the
     * DIALOG back to its top instead: scrollIntoView there drags the page
     * underneath the overlay, which looks like the modal jumping. */
    if (deepLink) {
      if (host.getBoundingClientRect().top < 0) host.scrollIntoView({block: "start"});
    } else {
      const card = host.closest(".modalCard");
      if (card) card.scrollTop = 0;
    }
  }

  host.querySelectorAll(".gRail button").forEach(b =>
    b.onclick = () => go(+b.dataset.i));
  host.querySelector(".gNavPrev").onclick = () => go(at - 1);
  host.querySelector(".gNavNext").onclick = () => go(at + 1);

  document.addEventListener("keydown", e => {
    if (!isActive()) return;
    // e.target is not always an Element (a synthetic event can target
    // `document`), and .closest would throw and kill the handler.
    if (e.target?.closest?.("input,textarea,select")) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); go(at - 1); }
    if (e.key === "ArrowRight") { e.preventDefault(); go(at + 1); }
  });

  // Swipe. Threshold and the vertical guard matter: without them a normal
  // scroll down the page reads as a sideways flick on the way past.
  let x0 = null, y0 = null;
  const steps = host.querySelector(".gSteps");
  steps.addEventListener("touchstart", e => {
    x0 = e.touches[0].clientX; y0 = e.touches[0].clientY;
  }, {passive: true});
  steps.addEventListener("touchend", e => {
    if (x0 == null) return;
    const dx = e.changedTouches[0].clientX - x0;
    const dy = e.changedTouches[0].clientY - y0;
    x0 = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.6) go(at - Math.sign(dx));
  }, {passive: true});

  go(deepLink ? _gIndexFromHash() : 0, {push: false});
  return {go, at: () => at};
}

let _gTab = null;

function renderGuide() {
  const host = document.getElementById("gStage");
  const inst = mountGuide(host, {
    deepLink: true,
    isActive: () => document.getElementById("guide")?.classList.contains("active"),
  });
  if (inst) _gTab = inst;
}

window.addEventListener("hashchange", () => {
  if (_gTab && document.getElementById("guide")?.classList.contains("active"))
    _gTab.go(_gIndexFromHash(), {push: false});
});

window.renderGuide = renderGuide;
window.mountGuide = mountGuide;
