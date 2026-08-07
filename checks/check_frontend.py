"""The cockpit SPA has no import checker. This is one.

    .venv/bin/python checks/check_frontend.py

WHY THIS EXISTS. `frontend/` is a plain SPA: index.html lists ~17 `<script
src>` tags and the tab machinery pairs a `data-tab="x"` button with a `<section
id="x">`. Nothing verifies any of it. A renamed file, a typo in a path, or a
section that loses its id all fail the same way — the page still returns 200,
the nav still renders, and one tab is quietly dead. Python has three checks that
catch this class of thing (check_repo_paths, check_editor_mirrors, the OpenAPI
route diff); the front end had none, which is why the standing rule after any
reorganisation is "every tab must be clicked".

Clicking every tab is the honest test and this does not replace it. It replaces
the part a human is bad at: noticing that a file listed in a script tag stopped
existing, or that a fetch() calls a route the backend no longer serves.

SIX CHECKS:

  1. every <script src> and <link href> resolves to a real file on disk
  2. every data-tab button has a matching <section id>, and vice versa
  3. every /api/... path a JS file fetches exists in the FastAPI route table
  4. every .js file parses (node --check)
  5. no two scripts declare the same top-level name
  6. nothing under static/ is orphaned — every .js and .css is actually loaded

Check 3 has the most teeth: it is the frontend half of the OpenAPI route diff
that verified the web/api/ split, and it would have caught a router rename
that index.html did not follow. Check 5 is the subtlest: 18
classic <script> tags share one global scope, so a duplicate top-level `const`
is a TypeError at load and the browser discards the whole second file.

Every one of them is proven to FAIL on a deliberate break, not assumed to.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATIC = ROOT / "frontend"
INDEX = STATIC / "index.html"

# A fetch target is only checkable when it is a literal. `/api/images/${c}` has
# a template hole, so we compare against the route table with the hole widened
# to match any single path segment — the same shape FastAPI's own {param} is.
_TEMPLATE = re.compile(r"\$\{[^}]*\}")


def _routes() -> set[str]:
    """Every path the app serves, from the OpenAPI schema.

    NOT by walking `app.routes`: this FastAPI wraps each included router in an
    `_IncludedRouter` whose own `.path` is `""`, so that walk returns six real
    routes and twelve empty strings — and a filter over it silently matches
    nothing. The first version of this file did exactly that and PASSED, having
    checked zero paths — the same shape as a `grep --include` that silently
    matches nothing.

    The schema is also what was diffed to certify the web/api/ split
    ("66 endpoints, identical"), so it is the surface already trusted here.
    """
    from groundwork.web.app import app
    paths = set(app.openapi().get("paths", {}))
    if not paths:
        raise RuntimeError("OpenAPI schema has no paths — refusing to 'pass'")
    return paths


def _route_matches(called: str, routes: set[str]) -> bool:
    """Does a called path match a registered route, allowing for params?

    JS builds URLs two ways and both have to be understood, or the check cries
    wolf on working code:

        `/api/images/${c}`       template literal -> ${...} is one segment
        "/api/images/" + c       concatenation    -> the literal ENDS in '/'
                                                    and is a prefix of a route

    The second form is why an earlier version reported five dead endpoints that
    were all live. A checker's false positives are as corrosive as its false
    negatives — both teach you to ignore it.
    """
    if called in routes:
        return True
    # a concatenated prefix: some real route must start with it
    if called.endswith("/") and any(r.startswith(called) for r in routes):
        return True
    # ONE direction: build the pattern from the ROUTE ({param} -> one segment)
    # and match the call against it. Matching both ways round was the mistake —
    # it handled `/api/runs/${run}/curve` vs `/api/runs/{run}/curve` but not
    # `/api/images/raw` vs `/api/images/{collection}`, where the JS has a
    # literal exactly where the route has a parameter. From the route, both
    # fall out.
    concrete = _TEMPLATE.sub("PARAM", called)
    for r in routes:
        pat = "^" + re.sub(r"\\\{[^}]*\\\}", "[^/]+", re.escape(r)) + "$"
        if re.match(pat, concrete):
            return True
    return False


def main() -> None:
    bad: list[str] = []
    html = INDEX.read_text(encoding="utf-8")

    # ---- 1. every asset a tag points at exists -----------------------------
    tags = re.findall(r'(?:src|href)="(/static/[^"?]+)', html)
    for t in tags:
        p = STATIC / t.lstrip("/").removeprefix("static/")
        if not p.exists():
            bad.append(f"index.html references {t} — NO SUCH FILE ({p})")
    print(f"  ok    {len(tags)} asset tag(s) resolve" if not bad else "")

    # ---- 2. tab buttons <-> sections --------------------------------------
    buttons = set(re.findall(r'data-tab="([a-zA-Z0-9_-]+)"', html))
    sections = set(re.findall(r'<section id="([a-zA-Z0-9_-]+)"', html))
    for t in sorted(buttons - sections):
        bad.append(f'nav button data-tab="{t}" has no <section id="{t}"> — dead tab')
    # A SECTION REACHED WITHOUT THE NAV IS NOT A DEAD TAB. `login` is shown by
    # body.signedOut, from cockpit/auth.js, and it deliberately has no button:
    # the nav is part of what the sign-in gate is gating, so a way to click to it
    # would be a way around it. Named rather than pattern-matched, so a genuinely
    # orphaned section still fails — this list is the whole of the exemption.
    NOT_TABS = {"login", "setupWizard"}   # login-siblings, shown by auth.js
    for s in sorted(sections - buttons - NOT_TABS):
        bad.append(f'<section id="{s}"> has no nav button — unreachable')
    if buttons and buttons == sections:
        print(f"  ok    {len(buttons)} tab(s) each pair a button with a section")

    # ---- 3. every fetched /api path is a real route ------------------------
    # NO SILENT SKIP. A check that cannot run must say so as a failure, not as
    # silence that reads like a pass — see _routes() for the version of this
    # file that got that wrong.
    try:
        routes = _routes()
    except Exception as e:  # noqa: BLE001
        bad.append(f"could not build the route table ({type(e).__name__}: {e}) "
                   "— cannot verify any API call")
        routes = set()
    if routes:
        called: dict[str, str] = {}
        for js in sorted(STATIC.rglob("*.js")):
            for m in re.findall(r'["`](/(?:api|img)[^"`\s]*)["`]', js.read_text(encoding="utf-8")):
                called.setdefault(m.split("?")[0], js.name)
        missing = [(c, f) for c, f in called.items() if not _route_matches(c, routes)]
        for c, f in sorted(missing):
            bad.append(f"{f} fetches {c} — not in the route table ({len(routes)} routes)")
        print(f"  ok    {len(called)} distinct API path(s) called, all served"
              if not missing else "")

    # ---- 4. every JS file actually parses ----------------------------------
    # `node --check` is a syntax check, not a run — it catches the edit that
    # leaves a stray brace or an unterminated template literal, which in a
    # browser kills the whole file silently and takes every function it defined
    # with it. node is already required for the editor-mirror check on HQ.
    import shutil as _sh
    import subprocess as _sp
    if _sh.which("node"):
        js = sorted(STATIC.rglob("*.js"))
        for f in js:
            r = _sp.run(["node", "--check", str(f)], capture_output=True, text=True)
            if r.returncode:
                first = (r.stderr.strip().splitlines() or ["?"])[0]
                bad.append(f"{f.name} is not valid JavaScript — {first}")
        if not any("not valid JavaScript" in b for b in bad):
            print(f"  ok    {len(js)} script(s) parse")
    else:
        print("  skip  node not found — syntax check not run (HQ has node)")

    # ---- 5. no two scripts declare the same top-level name -----------------
    # 18 classic <script> tags share ONE global scope. Two files declaring the
    # same top-level `const` is a TypeError at load, and the browser then
    # discards the whole second file — every function it defined vanishes and
    # the tabs that call them die, with nothing in the page to say why. The
    # existing files are collision-free by luck rather than by check; a generic
    # name like `esc` in a new file is all it takes.
    decl = re.compile(r"^(?:const|let|var|function|async function)\s+([A-Za-z_$][\w$]*)",
                      re.M)
    owner: dict[str, str] = {}
    for f in sorted(STATIC.rglob("*.js")):
        for name in set(decl.findall(f.read_text(encoding="utf-8"))):
            if name in owner:
                bad.append(f"top-level `{name}` declared in BOTH {owner[name]} and "
                           f"{f.name} — the second file will fail to load entirely")
            else:
                owner[name] = f.name
    if not any("top-level `" in b for b in bad):
        print(f"  ok    {len(owner)} top-level name(s), no collisions across scripts")

    # ---- 6. orphaned JS ----------------------------------------------------
    # CSS too, not just JS: cockpit.css was split into ordered slices, and a
    # slice that stops being linked is invisible — the page still renders, just
    # without those rules. Same failure shape as an unloaded script.
    for ext, how in ((".js", "<script src>"), (".css", "<link href>")):
        listed = {t.split("/")[-1] for t in tags if t.endswith(ext)}
        on_disk = {q.name for q in STATIC.rglob("*" + ext)}
        for orphan in sorted(on_disk - listed):
            bad.append(f"{orphan} exists under static/ but no {how} loads it")
        if listed and not (on_disk - listed):
            print(f"  ok    {len(listed)} {ext.lstrip('.')} file(s) listed, none orphaned")

    if bad:
        print("\nFRONTEND IS BROKEN:")
        for b in bad:
            print("  - " + b)
        raise SystemExit(1)
    print("\nthe cockpit's assets, tabs and API calls all resolve")


if __name__ == "__main__":
    main()
