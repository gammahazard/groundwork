"""No GET handler may block on another machine. It must go through lab_proxy.

THE RULE AND ITS HISTORY. lab_proxy exists because HQ's dashboard polled two
endpoints that each called the worker inline with a 3s timeout; on a cold VPN
path every cycle burned ~6s and the whole cockpit felt stuck. Its header calls
this the third instance of the same bug in this codebase: an expensive external
call sitting on a polled endpoint (nvidia-smi twice, then the VPN link).

It has since happened three more times, which is why this is a check and not a
paragraph:
  4. /api/retrain called nvidia-smi per poll and killed the worker's threadpool.
  5. /api/lab/log proxied with a blocking urlopen.
  6. /api/lab/vis did the same -- and lab.js polls it every 8 SECONDS while a
     challenger trains, which is exactly when the worker is least able to answer.

WHY GET AND NOT "POLLED". Whether a browser polls something is a fact about the
frontend, and tying a server-side rule to it means the rule silently stops
applying when someone adds a timer. Every GET here is a read, every read is
cheap to serve from cache, and lab_proxy answers instantly with an age. So the
rule is simply: GET handlers do not block on the network. POSTs may -- they are
one-shot user actions where a spinner is honest (see train_dispatch._remote).

lab_proxy itself is exempt: it IS the out-of-band fetch.
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "groundwork" / "web"
EXEMPT = {"lab_proxy.py"}
NET_ATTRS = {"urlopen"}
NET_MODULES = {"httpx", "requests"}
NET_METHODS = {"get", "post", "request"}

FAILURES: list[str] = []


def blocking_gets(path: Path) -> list[str]:
    """GET handlers in this file that call the network directly."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        FAILURES.append(f"{path}: will not parse: {e}")
        return []
    out = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        # @router.get(...) / @app.get(...)
        deco = [d.func.attr for d in fn.decorator_list
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)]
        if "get" not in deco:
            continue
        for c in ast.walk(fn):
            if not isinstance(c, ast.Call) or not isinstance(c.func, ast.Attribute):
                continue
            # urllib.request.urlopen(...) in any spelling
            if c.func.attr in NET_ATTRS:
                out.append(f"{fn.name}():{c.lineno} -> {c.func.attr}")
            # httpx.get / requests.post / ...
            elif (c.func.attr in NET_METHODS and isinstance(c.func.value, ast.Name)
                  and c.func.value.id in NET_MODULES):
                out.append(f"{fn.name}():{c.lineno} -> {c.func.value.id}.{c.func.attr}")
    return out


print("  [1] GET handlers under groundwork/web/")
scanned = 0
for f in sorted(WEB.rglob("*.py")):
    if f.name in EXEMPT:
        continue
    scanned += 1
    for hit in blocking_gets(f):
        FAILURES.append(f"{f.relative_to(ROOT)} {hit} — use lab_proxy.get()")
        print(f"      ✗ {f.relative_to(ROOT)} {hit}")
print(f"      scanned {scanned} modules")
if not FAILURES:
    print("      ✓ none block on the network")

print("\n  [2] lab_proxy is still the thing that DOES fetch")
lp = (WEB / "lab_proxy.py").read_text()
if "urlopen" not in lp:
    FAILURES.append("lab_proxy no longer fetches anything — the exemption is "
                    "now hiding a module that does nothing")
else:
    print("      ✓ lab_proxy fetches out of band, as the exemption assumes")

print("\n  [3] prove it can fail")
with tempfile.TemporaryDirectory() as td:
    bad = Path(td) / "regression.py"
    bad.write_text(
        "import urllib.request\n"
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n\n\n"
        "@router.get('/api/thing')\n"
        "def thing():\n"
        "    with urllib.request.urlopen('http://worker:8000/x', timeout=3) as r:\n"
        "        return r.read()\n")
    hits = blocking_gets(bad)
    if hits:
        print(f"      ✓ a reintroduced blocking GET is caught: {hits[0]}")
    else:
        FAILURES.append("the scanner did NOT catch a deliberately blocking GET "
                        "handler — it verifies nothing")
        print("      ✗ not caught")

    # And a POST doing the same must NOT be flagged: train_dispatch._remote is a
    # legitimate one-shot, and a check that forbids it would be wrong.
    okfile = Path(td) / "allowed.py"
    okfile.write_text(
        "import urllib.request\n"
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n\n\n"
        "@router.post('/api/thing')\n"
        "def thing():\n"
        "    with urllib.request.urlopen('http://worker:8000/x', timeout=60) as r:\n"
        "        return r.read()\n")
    if blocking_gets(okfile):
        FAILURES.append("a POST was flagged — the rule is about GETs; POSTs are "
                        "one-shot user actions where blocking is honest")
        print("      ✗ a POST was wrongly flagged")
    else:
        print("      ✓ a blocking POST is correctly left alone")

print()
if FAILURES:
    for f in FAILURES:
        print(f"  ✗ {f}")
    sys.exit(1)
print("  all checks passed")
