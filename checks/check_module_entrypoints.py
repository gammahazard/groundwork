"""Every module a shipped script runs with `python -m` must actually RUN.

WHY. `scripts/install.sh` ended with `exec python -m groundwork.install`, and
that module had no `__main__` block — so the last step of every native install
imported it, executed nothing, printed nothing and exited 0. It looked like a
success and left the machine with no units and no service to survive a reboot.
The console script (`groundwork install`) worked the whole time, which is why
nobody noticed: two doors, one of them a no-op.

compileall cannot see this (the module is perfectly valid), pyflakes cannot see
it (nothing is undefined), and check_import_targets cannot see it (the module
exists). It is only visible if you ask the question this file asks: does the
thing a script invokes have an entry point?

WHAT IT CHECKS. Every `python -m <mod>` / `-m <mod>` in the shell scripts, unit
templates and Python launch strings that ship — the module must resolve inside
this repo AND define a `if __name__ == "__main__":` block (or be a package with
a `__main__.py`). Third-party modules are skipped: pip's entry point is not
ours to verify.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Where a `-m module` can hide: shell installers, unit templates, and the
# argv lists Python builds to launch its own stages.
SEARCH = [
    *(ROOT / "scripts").glob("*.sh"),
    *(ROOT / "deploy").glob("*.sh"),
    *(ROOT / "deploy" / "units").glob("*.tmpl"),
    *(ROOT / "docker").glob("*"),
    *(ROOT / "groundwork").rglob("*.py"),
]

# `-m foo.bar` in a shell line, or "-m", "foo.bar" in a Python argv list.
# The trailing segment must be a real identifier, so documentation placeholders
# (`python -m groundwork.dataset.<name>`) are not mistaken for invocations —
# they end at the dot and would otherwise report a module nobody runs.
_SH = re.compile(r"-m\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?![\w.])")
_PY = re.compile(r'"-m",\s*"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"')


def _is_ours(mod: str) -> bool:
    return mod.split(".")[0] in {"groundwork", "altmodels"}


def _runnable(mod: str) -> tuple[bool, str]:
    """(has an entry point, why not)."""
    parts = mod.split(".")
    pkg_main = ROOT.joinpath(*parts) / "__main__.py"
    if pkg_main.exists():
        return True, ""
    f = ROOT.joinpath(*parts).with_suffix(".py")
    if not f.exists():
        return False, "no such module on disk"
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return False, f"does not parse: {e}"
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        # if __name__ == "__main__":
        src = ast.dump(node.test)
        if "__name__" in src and "__main__" in src:
            return True, ""
    return False, "imports cleanly but has no `if __name__ == \"__main__\"` block"


def main() -> int:
    found: dict[str, set] = {}
    for f in SEARCH:
        if not f.is_file() or f.suffix in (".pyc",):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = f.relative_to(ROOT).as_posix()
        for pat in (_SH, _PY):
            for mod in pat.findall(text):
                if _is_ours(mod):
                    found.setdefault(mod, set()).add(rel)

    if not found:
        print("MODULE ENTRY POINTS: found no `-m` invocations at all — this "
              "check has stopped looking at anything")
        return 1

    bad = []
    for mod in sorted(found):
        ok, why = _runnable(mod)
        if not ok:
            bad.append(f"{mod}: {why}\n      invoked by: "
                       f"{', '.join(sorted(found[mod]))}")

    print(f"  {len(found)} module(s) invoked with -m by shipped scripts")
    if bad:
        print(f"MODULE ENTRY POINTS: {len(bad)} cannot run")
        for x in bad:
            print("    " + x)
        return 1
    print("every module a shipped script runs with -m has an entry point")
    return 0


if __name__ == "__main__":
    sys.exit(main())
