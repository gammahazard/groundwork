"use strict";
/* The navigation drawer — a burger that slides out, and the tab bar it replaces
 * on small screens.
 *
 * WHY A DRAWER AND NOT A SCROLLING TAB STRIP. There are 13 tabs. On a phone
 * `nav{flex-wrap:wrap}` stacked them into four rows of chips that ate a third of
 * the viewport before any content appeared, on a page whose primary job is
 * showing you an image photo. A drawer costs one tap and gives the whole screen
 * back.
 *
 * Desktop keeps the horizontal bar — there is room, and a drawer would be a tap
 * tax for nothing. Same markup, same handlers; CSS decides which is visible, so
 * there is no second nav to keep in sync.
 *
 * MOTION. The drawer slides, the backdrop fades, and the items stagger in
 * behind it. All of it goes through the prefers-reduced-motion switch in
 * ui.js's stylesheet — under that setting the drawer simply appears.
 *
 * The behaviours that make a drawer usable rather than merely present, none of
 * which are free: it closes on selection, on backdrop tap, on Escape and on
 * resize past the breakpoint (otherwise rotating a phone leaves an open drawer
 * over a desktop layout); it moves focus into the panel on open and back to the
 * burger on close; and while open it blocks background scroll, because a drawer
 * you can scroll the page behind feels broken.
 */

const NAV_BREAKPOINT = 760;             // must match 11-nav.css

let _navOpen = false;
let _navLastFocus = null;

function navIsDrawer() {
  return window.innerWidth <= NAV_BREAKPOINT;
}

function openNav() {
  if (_navOpen) return;
  _navOpen = true;
  _navLastFocus = document.activeElement;
  document.body.classList.add("navOpen");
  const panel = $("#navPanel");
  $("#navBurger").setAttribute("aria-expanded", "true");
  panel.removeAttribute("inert");
  // stagger the items in behind the panel
  if (typeof revealAll === "function") revealAll(panel.querySelector("nav"), 28, 14);
  (panel.querySelector("nav button") || panel).focus({preventScroll: true});
}

function closeNav({restoreFocus = true} = {}) {
  if (!_navOpen) return;
  _navOpen = false;
  document.body.classList.remove("navOpen");
  $("#navBurger").setAttribute("aria-expanded", "false");
  // inert keeps the off-screen panel out of the tab order — without it you can
  // Tab into a drawer that is not on screen, which is a genuinely confusing bug
  // rather than a theoretical accessibility box-tick.
  if (navIsDrawer()) $("#navPanel").setAttribute("inert", "");
  if (restoreFocus && _navLastFocus) _navLastFocus.focus({preventScroll: true});
}

function toggleNav() {
  // On a wide screen the panel is pinned, so the burger COLLAPSES it (and main
  // reclaims the space) rather than opening an overlay nobody needed.
  if (!navIsDrawer()) {
    document.body.classList.toggle("navCollapsed");
    const c = document.body.classList.contains("navCollapsed");
    $("#navBurger").setAttribute("aria-expanded", String(!c));
    return;
  }
  _navOpen ? closeNav() : openNav();
}

/** Re-evaluate on resize/rotate: a drawer left open while the layout goes wide
 *  would sit as a stray overlay on top of the desktop bar. */
function navSyncToViewport() {
  const panel = $("#navPanel");
  if (!panel) return;
  if (navIsDrawer()) {
    if (!_navOpen) panel.setAttribute("inert", "");
  } else {
    panel.removeAttribute("inert");     // desktop: always reachable
    if (_navOpen) closeNav({restoreFocus: false});
  }
}

function initNav() {
  const burger = $("#navBurger");
  if (!burger) return;
  burger.onclick = toggleNav;
  $("#navScrim").onclick = () => closeNav();
  // Selecting a destination closes the drawer — leaving it open over the thing
  // you just asked for is the classic drawer mistake.
  document.querySelectorAll("#navPanel nav button")
    .forEach(b => b.addEventListener("click", () => { if (navIsDrawer()) closeNav(); }));
  window.addEventListener("keydown", e => {
    if (e.key === "Escape" && _navOpen) { e.stopPropagation(); closeNav(); }
  });
  let t = null;
  window.addEventListener("resize", () => {
    clearTimeout(t); t = setTimeout(navSyncToViewport, 120);
  });
  navSyncToViewport();
  // Wide screens start pinned open, so the burger reads as expanded.
  if (!navIsDrawer()) burger.setAttribute("aria-expanded", "true");
}

/* ---- the project strip's ⋯ ---------------------------------------------
 * The four core tabs are always on screen; the EXTENSION tabs (Synthesize,
 * extension tabs) fold behind ⋯ on a phone. Same reasoning as moving the
 * strip out of the drawer: what you use constantly should cost nothing, what
 * you use occasionally can cost a tap.
 *
 * It respects [hidden]. applyExtensionTabs hides a tab the project has not
 * declared, and 01-base.css makes that `display:none !important`, so a project
 * with no extensions gets no ⋯ at all rather than one opening an empty row.
 */
const PROJ_EXT_TABS = [];   // extension-gated tabs; none ship built-in

function projTabsSync(){
  const more = document.getElementById("projTabsMore");
  const strip = document.getElementById("projTabs");
  if (!more || !strip) return;
  const any = PROJ_EXT_TABS.some(t => {
    const b = strip.querySelector(`button[data-tab="${t}"]`);
    return b && !b.hidden;
  });
  // Nothing to reveal, or a wide screen showing them all already.
  more.hidden = !any || !navIsDrawer();
  if (more.hidden) strip.classList.remove("moreOpen");
}

document.addEventListener("click", e => {
  const more = e.target.closest("#projTabsMore");
  const strip = document.getElementById("projTabs");
  if (!strip) return;
  if (more){
    const open = strip.classList.toggle("moreOpen");
    more.setAttribute("aria-expanded", String(open));
    return;
  }
  // Choosing anything closes it — a row left open over the thing you just
  // asked for is the same mistake the drawer avoids.
  if (e.target.closest("#projTabs button")) strip.classList.remove("moreOpen");
});
window.addEventListener("resize", () => projTabsSync());
window.projTabsSync = projTabsSync;
