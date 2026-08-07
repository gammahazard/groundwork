"""The model registry is the ONE description of a family, and it must refuse
rather than misroute.

    .venv/bin/python checks/check_model_registry.py

WHY THIS EXISTS. `groundwork/models/registry.py` replaced six hand-maintained
dispatch tables. The failure mode is on record in the registry's own header:
an unrecognised arch used to FALL THROUGH to another family's predictor, so
scoring one family's run would have loaded its checkpoint with the wrong
loader out of a venv that cannot even train it — a wrong number rather than a
refusal. Everything here defends the properties that stop that class of bug:

  * a run name resolves to its OWN family, most-specific prefix first,
    independent of declaration order;
  * an unknown name resolves to None — never to somebody else's entry;
  * the command line a family builds uses only flags its trainer declares;
  * the default size is one the family can actually train at;
  * every declared predictor exists in altmodels/predictors.

THE MUTATION ARM: part 5 re-sorts the prefix table shortest-first and requires
resolution to go WRONG — proving part 1 reads the longest-first rule rather
than agreeing with today's data, in which no cross-family shadow happens to
exist. This repo's standing lesson: a check that cannot fail verifies nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundwork.models import registry  # noqa: E402

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


def main() -> None:
    # ---- 1. every family's own prefixes come home ---------------------------
    print("1. a run name resolves to its own family, whatever the order")
    n = 0
    for m in registry.MODELS:
        for p in m.prefixes:
            got = registry.for_run(p + "-check-1")
            if got is None or got.name != m.name:
                note(False, f"run {p + '-check-1'!r} resolved to "
                            f"{got.name if got else None!r}, not {m.name!r}")
            else:
                n += 1
    note(n > 0, f"{n} prefix(es) across {len(registry.MODELS)} families all "
                f"resolve to their own entry")
    got = registry.for_run("yolox-tiny-1")
    note(got is not None and got.name == "yolox-tiny",
         "yolox-tiny does not lose to a shorter sibling prefix")
    got = registry.for_run("yolox-s-whatever")
    note(got is not None and got.name == "yolox-s", "and yolox-s still wins its own")

    # ---- 2. unknown means None, never someone else's loader -----------------
    print("\n2. an unknown name is a refusal, not a fall-through")
    note(registry.for_run("no-such-family-run-7") is None,
         "for_run() on an unknown run name is None")
    note(registry.by_name("no-such-arch") is None,
         "by_name() on an unknown arch is None")
    note(registry.by_name(None) is None, "…and on None")
    dupes = {}
    for m in registry.MODELS:
        for p in m.prefixes:
            dupes.setdefault(p, []).append(m.name)
    shared = {p: v for p, v in dupes.items() if len(v) > 1}
    note(not shared,
         f"no prefix is claimed by two families ({shared or 'none'}) — a "
         f"shared prefix would silently shadow one of them")

    # ---- 3. every trainable family's ARGV is one its trainer accepts --------
    #
    # THE BUG THIS EXISTS FOR: the dispatch used to build the command with
    # `if <one family>: --size else: --scale-factor`, so the moment a SECOND
    # entry of that family existed it fell to the else branch and would have
    # launched the trainer with a flag it does not accept. Four trainers spell
    # resolution three ways (--size, --res, --scale-factor) and nothing
    # compared the two sides.
    #
    # Read the flags from the trainer's SOURCE, not by importing it: most live
    # in venvs this interpreter is not, so an import check would skip exactly
    # the families most likely to drift.
    print("\n3. the command line a family builds is one its trainer accepts")
    import ast as _ast
    n_argv = 0
    for m in registry.MODELS:
        if not m.trainer_module or m.name == "yolov8n":
            continue                       # the champion goes through retrain_job
        src = ROOT / (m.trainer_module.replace(".", "/") + ".py")
        if not src.exists():
            note(False, f"{m.name}: trainer module {m.trainer_module} has no "
                        f"file at {src}")
            continue
        accepted = set()
        for node in _ast.walk(_ast.parse(src.read_text())):
            if (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args
                    and isinstance(node.args[0], _ast.Constant)):
                accepted.add(node.args[0].value)
        if not accepted:
            note(False, f"{m.name}: found no add_argument calls in {src} — "
                        f"this check would verify nothing")
            continue
        argv = m.train_argv(imgsz=1280, epochs=1, batch=1, run_name="x")
        unknown = [a for a in argv if a.startswith("--") and a not in accepted]
        if unknown:
            note(False, f"{m.name}: {m.trainer_module} does not accept "
                        f"{', '.join(unknown)} (it takes "
                        f"{', '.join(sorted(a for a in accepted if a.startswith('--')))})")
        else:
            n_argv += 1
    note(n_argv > 0, f"{n_argv} trainer command line(s) use flags their "
                     f"trainer declares")

    # ---- 3b. the default size must be one the family can train at ----------
    #
    # One family's default_imgsz was a resolution it had never trained at and
    # which was not even legal for it (--res must be a multiple of its patch
    # grid). The UI pre-selects default_imgsz, so it was offering an illegal
    # default; and for the scale-factor families an off-menu pick silently
    # trains at a different size than requested.
    print("\n3b. every family's default size is on its own menu")
    n_sizes = 0
    for m in registry.MODELS:
        if not m.trainer_module:
            continue
        if m.default_imgsz not in m.sizes:
            note(False, f"{m.name}: default_imgsz {m.default_imgsz} is not in "
                        f"its offered sizes {m.sizes}")
        else:
            n_sizes += 1
    note(n_sizes > 0, f"{n_sizes} family default size(s) are ones that family offers")

    # ---- 4. every declared predictor is one altmodels can actually build ----
    print("\n4. every declared predictor exists in the predictor registry")
    src = (ROOT / "altmodels" / "predictors" / "__init__.py").read_text()
    missing = [m.name for m in registry.MODELS
               if m.predictor and f'name == "{m.predictor}"' not in src]
    note(not missing,
         f"every predictor is in altmodels/predictors/__init__.py "
         f"({missing or 'all present'})")

    # ---- 5. MUTATION: longest-first is a rule, not a coincidence ------------
    # Current data has no cross-family shadow pair, so part 1 would pass under
    # a first-declared-wins matcher too. Plant one and flip the sort: with a
    # fake short prefix "yolo" in the table, longest-first must still resolve
    # yolov8n runs correctly, and SHORTEST-first must get them wrong — which is
    # what proves the property is live.
    print("\n5. mutation — the prefix table's sort is load-bearing")
    fake = registry.Model(name="fake-family", label="x", predictor="",
                          venv=".venv", license="x", default_imgsz=1280,
                          prefixes=("yolo",))
    saved = registry._BY_PREFIX
    try:
        pairs = list(saved) + [("yolo", fake)]
        registry._BY_PREFIX = tuple(sorted(pairs, key=lambda pm: -len(pm[0])))
        got = registry.for_run("yolov8n-33")
        note(got is not None and got.name == "yolov8n",
             "with a shadowing short prefix planted, longest-first still "
             "resolves the real family")
        registry._BY_PREFIX = tuple(sorted(pairs, key=lambda pm: len(pm[0])))
        got = registry.for_run("yolov8n-33")
        note(got is not None and got.name == "fake-family",
             "sorted shortest-first the SAME lookup goes wrong — so the "
             "longest-first assertion can fail and is testing the rule")
    finally:
        registry._BY_PREFIX = saved

    print()
    if FAIL:
        print("REGISTRY GUARANTEES DO NOT HOLD:")
        for b in FAIL:
            print("  - " + b)
        raise SystemExit(1)
    print("the registry resolves its own families, refuses strangers, and "
          "builds command lines its trainers accept")


if __name__ == "__main__":
    main()
