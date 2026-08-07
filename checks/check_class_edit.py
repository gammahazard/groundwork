"""Editing a project's classes must never silently re-point existing labels.

A class is an INDEX, not a name: every label line starts with a position in the
class list, so the list is a schema for data already on disk. Append and rename
are safe; REORDER and REMOVE are not, and reorder is the dangerous one because
nothing downstream errors — the split succeeds, training succeeds, and the model
learns bolts as nuts.

THIS CHECK IS BUILT TO BE ABLE TO FAIL, which is the standing rule here. It does
not assert today's answer; it MUTATES the world and requires the answer to move:

  * removal is refused while a label uses the class, and ACCEPTED once the label
    is deleted — so the guard is proven to read the labels rather than to be a
    blanket "no"
  * a class used only by the FROZEN TESTSET is still protected, which a check
    that only wrote raw/ would pass with that half of the guard deleted

Works on a throwaway project and deletes it, so it never touches a real one.

    python3 checks/check_class_edit.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# SIGN IN FIRST, or this checks nothing. The cockpit grew a login gate and this
# file did not: every request 401'd, the very first one ("create the scratch
# project") failed, and the run aborted before a single class rule was
# exercised. It stayed that way silently — a check that cannot reach its subject
# reports a setup error, which reads like an environment problem rather than
# like the invariants going unverified. They are the invariants that stop a
# class edit silently rewriting the meaning of every label on disk, so this is
# the last check that should be dark.
#
# The stores are redirected to a temp dir BEFORE groundwork.web is imported (it
# bootstraps the user store at module level), exactly as check_auth_gate does,
# so the real accounts and the real audit trail are never touched.
import os                                                       # noqa: E402
import tempfile                                                 # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="classcheck-"))
os.environ["GW_AUTH"] = "1"
os.environ.pop("GW_ADMIN_USER", None)
os.environ.pop("GW_ADMIN_PASSWORD", None)

from groundwork.web.auth import audit as _audit                # noqa: E402
from groundwork.web.auth import sessions as _sessions          # noqa: E402
from groundwork.web.auth import users as _users                # noqa: E402
_users.USERS_PATH = _TMP / "users.json"
_sessions.SESSIONS_PATH = _TMP / "sessions.json"
_audit.AUDIT_PATH = _TMP / "audit.jsonl"

from fastapi.testclient import TestClient                       # noqa: E402

from groundwork.web.app import app                                  # noqa: E402
from groundwork import project                                 # noqa: E402
from groundwork.dataset import paths                           # noqa: E402

SLUG = "zz-classcheck"
_users.add("classcheck", "classcheckpass1", admin=False)
c = TestClient(app)
c.post("/api/login", json={"username": "classcheck",
                           "password": "classcheckpass1"})
fails: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


def label(pp, tree: str, stem: str, cls: int) -> Path:
    d = getattr(pp, "RAW_LABELS" if tree == "raw" else "TESTSET_LABELS")
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{stem}.txt"
    f.write_text(f"{cls} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    return f


def cleanup() -> None:
    root = project.PROJECTS_DIR / SLUG
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        p = project.load(SLUG)
    except KeyError:
        return
    ds = paths.for_project(p).root
    if ds.exists() and SLUG in str(ds):
        shutil.rmtree(ds, ignore_errors=True)


cleanup()
r = c.post("/api/projects", json={"name": "zz classcheck",
                                  "classes": ["bolt", "nut"]})
if r.status_code >= 400:
    print(f"could not create the scratch project: {r.status_code} {r.text[:200]}")
    raise SystemExit(2)

try:
    pp = paths.for_project(project.load(SLUG))

    print("\n1. the safe edits are allowed")
    r = c.patch(f"/api/projects/{SLUG}/classes",
                json={"classes": ["bolt", "nut", "washer"]})
    ok("append", r.status_code == 200 and r.json().get("added") == ["washer"],
       f"{r.status_code}")
    r = c.patch(f"/api/projects/{SLUG}/classes",
                json={"classes": ["bolt", "hex-nut", "washer"]})
    ok("rename in place", r.status_code == 200 and r.json().get("renamed"),
       f"{r.status_code}")
    ok("rename kept the index", project.load(SLUG).classes[1] == "hex-nut",
       str(project.load(SLUG).classes))

    print("\n2. reorder is refused, always")
    r = c.patch(f"/api/projects/{SLUG}/classes",
                json={"classes": ["washer", "bolt", "hex-nut"]})
    ok("reorder refused", r.status_code == 409, f"{r.status_code}")
    ok("and says why", "re-point" in r.json().get("detail", "").lower())

    print("\n3. removal follows the LABELS, not a hardcoded answer")
    f = label(pp, "raw", "t1", 2)                     # class 2 = washer, in use
    r = c.patch(f"/api/projects/{SLUG}/classes",
                json={"classes": ["bolt", "hex-nut"], "allow_removal": True})
    ok("refused while a raw label uses it", r.status_code == 409, f"{r.status_code}")

    # THE MUTATION: delete the only label using it. The SAME request must now
    # succeed — if it still fails, the guard is not reading the labels at all.
    f.unlink()
    r = c.patch(f"/api/projects/{SLUG}/classes",
                json={"classes": ["bolt", "hex-nut"], "allow_removal": True})
    ok("ALLOWED once nothing is labelled with it", r.status_code == 200,
       f"{r.status_code} {r.text[:90]}")

    print("\n4. the frozen testset counts too")
    c.patch(f"/api/projects/{SLUG}/classes",
            json={"classes": ["bolt", "hex-nut", "washer"]})
    label(pp, "testset", "e1", 2)                     # only in the holdout
    r = c.patch(f"/api/projects/{SLUG}/classes",
                json={"classes": ["bolt", "hex-nut"], "allow_removal": True})
    ok("a class used ONLY by the testset is still protected",
       r.status_code == 409, f"{r.status_code}")

    print("\n5. removal needs confirming even when it is safe")
    (pp.TESTSET_LABELS / "e1.txt").unlink()
    r = c.patch(f"/api/projects/{SLUG}/classes", json={"classes": ["bolt", "hex-nut"]})
    ok("refused without allow_removal", r.status_code == 409, f"{r.status_code}")

    print("\n6. the read-back reports usage")
    label(pp, "raw", "t2", 0)
    body = c.get(f"/api/projects/{SLUG}/classes").json()
    ok("counts labels per class",
       body["classes"][0]["labels"] == 1 and body["classes"][1]["labels"] == 0,
       str([x["labels"] for x in body["classes"]]))
finally:
    cleanup()

print()
if fails:
    print(f"FAILED {len(fails)}: {', '.join(fails)}")
    raise SystemExit(1)
print("all class-edit guards hold")
