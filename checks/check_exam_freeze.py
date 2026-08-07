"""A frozen exam must be reproducible EXACTLY, or refuse to be scored.

    .venv/bin/python checks/check_exam_freeze.py

WHY THIS EXISTS. Every MAE is denominated in a holdout, and on the predecessor
that holdout silently changed size seven times. What that cost is recorded
there: a spread that looked like a model difference was partly the exam
growing underneath the runs, and one run's single miss read as two different
scores in two places (1/65 against 1/68 of the same mistake).

`store/exam.py` freezes the image set under a name; `count_eval --exam` scores
that name. The dangerous failure is not a crash — it is scoring 65 of a frozen
68 and reporting it under the exam's name, because that number looks
comparable to the historical one and is not. So the contract being checked
here is:

    every frozen image present and unchanged   -> score it
    ANY image missing or relabelled            -> REFUSE, and say which

THE MUTATION ARM MATTERS MORE THAN THE REST. Asserting "a good exam scores"
and "a broken one refuses" would both pass if the refusal were deleted and the
fixtures happened to line up. So part 4 removes the guard from the source and
asserts the broken exam then goes through — proving these tests read the rule
rather than agreeing with today's arrangement. This repo's standing lesson: a
check that verifies nothing must FAIL, not pass.

READ-ONLY AND GPU-FREE: every fixture is a temp directory, and it calls the
restriction predicate directly rather than running an eval.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from groundwork.dataset.pipeline import count_eval          # noqa: E402
from groundwork.dataset.store import exam as exam_mod       # noqa: E402

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


class _PP:
    """Minimal stand-in for ProjectPaths — exam.load only needs the slug."""
    def __init__(self, slug):
        self.slug = slug


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="exam-check-"))
    exam_mod.EXAM_DIR = tmp          # never touch the real exams/
    pp = _PP("testproj")

    frozen = {"a": {"gt": 10, "sha": "x"}, "b": {"gt": 20, "sha": "y"},
              "c": {"gt": 30, "sha": "z"}}
    (tmp / "testproj-E.json").write_text(json.dumps(
        {"name": "E", "project": "testproj", "n_images": 3, "images": frozen}))

    print("1. an intact holdout scores the exam")
    stems = ["a", "b", "c"]
    gt = {"a": 10, "b": 20, "c": 30}
    got = count_eval._restrict_to_exam(stems, "E", gt, pp, "testset")
    note(got == ["a", "b", "c"], f"all three frozen images returned ({got})")

    print("\n2. a GROWN holdout scores only the exam's images")
    got = count_eval._restrict_to_exam(
        ["a", "b", "c", "d", "e"], "E", {**gt, "d": 1, "e": 2}, pp, "testset")
    note(got == ["a", "b", "c"],
         f"the 2 newer images are excluded, not silently averaged in ({got})")

    print("\n3. THE REFUSALS — a subset under the exam's name is the danger")
    try:
        count_eval._restrict_to_exam(["a", "b"], "E", {"a": 10, "b": 20},
                                     pp, "testset")
        note(False, "a MISSING image must refuse")
    except SystemExit as e:
        note("MISSING" in str(e) and "c" in str(e),
             "a missing image refuses AND names it")

    try:
        count_eval._restrict_to_exam(stems, "E", {"a": 10, "b": 21, "c": 30},
                                     pp, "testset")
        note(False, "a RELABELLED image must refuse")
    except SystemExit as e:
        note("RELABELLED" in str(e) and "20" in str(e) and "21" in str(e),
             "a relabelled image refuses AND shows both counts")

    try:
        count_eval._restrict_to_exam(stems, "E", gt, pp, "val")
        note(False, "--exam with --source val must refuse")
    except SystemExit:
        note(True, "--exam against the val split refuses (an exam is a HOLDOUT)")

    print("\n4. MUTATION — is the refusal live, or would anything pass?")
    import inspect
    src = inspect.getsource(count_eval._restrict_to_exam)
    mutated = src.replace("    if missing or drift:", "    if False:", 1)
    note(mutated != src, "the refusal is findable in the source")
    ns = {"SystemExit": SystemExit, "print": print,
          "exam_mod": exam_mod, "sorted": sorted, "set": set, "len": len}
    # It imports exam_mod inside the function; give the exec'd copy the same name.
    mutated = mutated.replace("    from ..store import exam as exam_mod\n", "")
    exec(compile(mutated, "<mutated>", "exec"), ns)
    try:
        ns["_restrict_to_exam"](["a", "b"], "E", {"a": 10, "b": 20}, pp, "testset")
        slipped = True
    except SystemExit:
        slipped = False
    note(slipped,
         "with the guard removed a 2-of-3 subset IS accepted — so part 3 is "
         "testing the rule, not a coincidence")

    print()
    if FAIL:
        print("FROZEN EXAMS ARE NOT SAFE:")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)
    print("a named exam is scored whole or not at all")


if __name__ == "__main__":
    main()
