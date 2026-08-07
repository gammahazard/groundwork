"""Every relative import must point at a module that EXISTS.

WHY. compileall proves syntax; pyflakes proves names used are names defined.
NEITHER can see a `from ..yolo_counter import YoloCounter` written inside a
function whose target module moved — it parses, it flake-checks, and it
crashes at CALL TIME, which for a bot supervisor means a crash-loop with the
reason buried. The first real bot start hit exactly that, twice in a row
(a stale sys.path shim, then this).

Walks every ImportFrom — module-level AND function-local — in the shipped
packages, resolves the relative target against the file's own package, and
asserts a module or package exists there. Absolute `groundwork.*` imports are
resolved too; third-party absolutes are ignored (the venv answers for those).

HOW IT CAN FAIL: point any relative import at a module that is not there.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("groundwork", "altmodels")

bad: list[str] = []
checked = 0

def _bound_names(init_path) -> set:
    """Names an __init__ actually BINDS — imports, assignments, defs, classes.

    This was a substring test against the file's text, which is why it missed a
    live crash: `from ...lab_ops import score` passed because the characters
    "score" appear inside `_score_blocked`. autoscore called that name in-process
    on every finished challenger run, so the timer whose whole job is "every run
    comes back scored" raised ImportError instead, and had since the package was
    split. A name test has to bind names.
    """
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
                    if t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        out.update(e.value for e in node.value.elts
                                   if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return out


def _exists(mod_parts: list[str]) -> bool:
    base = ROOT.joinpath(*mod_parts)
    return (base.with_suffix(".py")).exists() or (base / "__init__.py").exists()

for pkg in PACKAGES:
    for f in sorted((ROOT / pkg).rglob("*.py")):
        rel = f.relative_to(ROOT)
        # this file's package parts, e.g. groundwork/serve/runtime_model.py
        # -> ["groundwork", "serve"]
        pkg_parts = list(rel.parts[:-1])
        if rel.name != "__init__.py":
            file_pkg = pkg_parts
        else:
            file_pkg = pkg_parts[:-1] if len(pkg_parts) > 1 else pkg_parts
            file_pkg = pkg_parts  # an __init__ imports relative to its own dir
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{rel}: does not parse: {e}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0:
                mod = node.module or ""
                if not mod.split(".")[0] in PACKAGES:
                    continue
                target = mod.split(".")
            else:
                # from .x / ..x: climb `level` packages from THIS file's package
                up = node.level
                base = pkg_parts[:len(pkg_parts) - (up - 1)] if up > 1 else pkg_parts
                if len(pkg_parts) - (up - 1) < 0:
                    bad.append(f"{rel}:{node.lineno}: relative import climbs "
                               f"above the package root")
                    continue
                target = base + (node.module.split(".") if node.module else [])
            checked += 1
            if _exists(target):
                # The MODULE half exists. When it is a PACKAGE, the imported
                # names must each be a submodule on disk or appear in its
                # __init__ — `from .. import runtime_model` resolving to a
                # package that holds neither is the crash the module check
                # alone cannot see (found live: two counters importing
                # runtime_model from the wrong depth).
                pkg_init = ROOT.joinpath(*target) / "__init__.py"
                if pkg_init.exists():
                    exported = _bound_names(pkg_init)
                    for al in node.names:
                        nm = al.name
                        if nm == "*" or _exists(target + [nm]) or nm in exported:
                            continue
                        bad.append(f"{rel}:{node.lineno}: {nm!r} is neither a "
                                   f"submodule of {'.'.join(target)!r} nor in "
                                   f"its __init__")
                continue
            # `from .pkg import name` where name is an attr of pkg/__init__ —
            # the MODULE half is what must exist; names are pyflakes' job.
            bad.append(f"{rel}:{node.lineno}: import target "
                       f"{'.'.join(target)!r} does not exist on disk")

print(f"  {checked} intra-repo import targets resolved")
if bad:
    print(f"IMPORT TARGETS: {len(bad)} broken")
    for x in bad:
        print("  " + x)
    sys.exit(1)
print("every relative and groundwork.* import points at a real module")
