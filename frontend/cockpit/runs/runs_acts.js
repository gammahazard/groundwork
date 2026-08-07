"use strict";
/* On a phone, a run row's actions become one ⋯ menu.
 *
 * A yolo row carries up to SEVEN buttons — curve, log, dots, 🔬, 📸, Δ, use —
 * and a challenger row two. At 390px that column took most of the width and the
 * thing you actually came to read (which run, what it scored) was squeezed into
 * what was left. Worse, the two families looked completely different for a
 * reason that is not about the runs.
 *
 * THE BUTTONS ARE WRAPPED, NOT REBUILT. They are moved into one <div> so the
 * menu can be absolutely positioned — inline, an open menu grew its row to
 * ~900px and pushed every other run off the screen. Moving a node preserves its
 * listeners (they live on the element, not on its position), so the handlers
 * `runsContribute` wires on every render survive; REBUILDING them would not,
 * and that is how a button ends up doing nothing on the second render.
 *
 * A ROW WITH NO ACTIONS GETS NO BUTTON. An unscored challenger has fewer
 * controls than a champion run, and a ⋯ that opens an empty box is worse than
 * no ⋯ at all.
 */

const RUNS_NARROW = () => window.innerWidth <= 760;

function runsActsSync(box) {
  const root = box || document.querySelector("#runsTable");
  if (!root) return;
  const narrow = RUNS_NARROW();
  root.querySelectorAll("td.acts").forEach(td => {
    const existing = td.querySelector(".actsMore");
    // Count only the row's OWN buttons, never the toggle we may have added.
    const n = [...td.querySelectorAll("button")].filter(b => b !== existing).length;
    if (!narrow || !n) {
      if (existing) existing.remove();
      // Unwrap on the way back to a wide screen, so the cell is exactly what
      // the ledger loaders rendered.
      const m = td.querySelector(".actsMenu");
      if (m) { [...m.childNodes].forEach(x => td.appendChild(x)); m.remove(); }
      td.classList.remove("open", "hasMore");
      return;
    }
    td.classList.add("hasMore");
    if (existing) return;
    // One wrapper the popover can be positioned as. Built from the buttons
    // already in the cell, by MOVING them.
    const menu = document.createElement("div");
    menu.className = "actsMenu";
    [...td.childNodes].forEach(n => menu.appendChild(n));
    td.appendChild(menu);
    const b = document.createElement("button");
    b.className = "actsMore";
    b.type = "button";
    b.setAttribute("aria-label", `${n} action${n === 1 ? "" : "s"} for this run`);
    b.setAttribute("aria-expanded", "false");
    b.textContent = "⋯";
    b.onclick = e => {
      e.stopPropagation();
      const open = !td.classList.contains("open");
      // One at a time: two open popovers on a 390px screen overlap each other.
      root.querySelectorAll("td.acts.open").forEach(o => {
        o.classList.remove("open");
        o.querySelector(".actsMore")?.setAttribute("aria-expanded", "false");
      });
      td.classList.toggle("open", open);
      b.setAttribute("aria-expanded", String(open));
    };
    td.prepend(b);
  });
}

/* Tapping anything inside the menu is an ACTION — close after it, or the
 * popover sits over the modal the action just opened. */
document.addEventListener("click", e => {
  const root = document.querySelector("#runsTable");
  if (!root) return;
  const inMenu = e.target.closest("td.acts.open");
  if (!inMenu || e.target.closest("button.actsMore")) {
    if (!e.target.closest("button.actsMore"))
      root.querySelectorAll("td.acts.open").forEach(o => o.classList.remove("open"));
    return;
  }
  if (e.target.closest("button")) inMenu.classList.remove("open");
});
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  document.querySelectorAll("#runsTable td.acts.open").forEach(o => o.classList.remove("open"));
});
// The breakpoint is evaluated in JS as well as CSS, so `hidden` and the media
// query cannot disagree — the same rule the editor's ⋯ follows.
window.addEventListener("resize", () => runsActsSync());
