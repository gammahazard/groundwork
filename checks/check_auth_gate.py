"""Nothing under outputs/ reaches an unauthenticated caller. Not a route, not a file.

    .venv/bin/python checks/check_auth_gate.py

PART 2 IS THE WHOLE REASON THIS FILE EXISTS. `groundwork/web/app.py` mounts the
entire outputs/ tree as StaticFiles — gigabytes of source photographs, eval
previews, challenger checkpoints and CoreML exports. A FastAPI dependency guards
ROUTES and would have left every one of those files reachable by anyone who
could open the port. That is not a weaker version of being secured; it is the
same exposure with a login page in front of it. The gate is ASGI middleware
specifically so one rule covers routes and mounts alike, and this asserts the
mount, by fetching a real file.

Part 3 is the hole the allow-list could open by itself: `/static/` is public, so
`/static/../outputs/...` starts with a public prefix and a plain startswith()
would wave it through — after which the router serves it from the outputs mount.

Part 6 is project ownership: a project owned by someone else must 403 AND be
absent from the listing, because a card carries the project's name, its counts
and a thumbnail lifted from its data.

HOW IT CAN FAIL: every assertion is a status code that must CHANGE when the
credential changes — 401 without a cookie, 200 with one. Deleting the middleware
turns parts 2, 3 and 4 red; deleting the ownership check turns 6 red.

SAFE: users and sessions are redirected to a temp directory before the app is
imported, so the real store is never read or written. No GPU, no network.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


# REDIRECT BEFORE groundwork.web IS IMPORTED. It calls users.bootstrap() at module
# level, so importing it first would touch the real store.
TMP = Path(tempfile.mkdtemp(prefix="checkauth-"))
os.environ["GW_AUTH"] = "1"
os.environ.pop("GW_ADMIN_USER", None)
os.environ.pop("GW_ADMIN_PASSWORD", None)

from groundwork.web.auth import audit, sessions, users        # noqa: E402
users.USERS_PATH = TMP / "users.json"
sessions.SESSIONS_PATH = TMP / "sessions.json"
# THE TRAIL TOO. Without this the check's fake sign-ins ("boss", "worker", from
# "testclient") land in the REAL outputs/audit.jsonl — found on 2026-08-05 while
# reading the admin panel and wondering who `boss` was. A security log polluted
# by its own test suite is a log you learn to discount.
audit.AUDIT_PATH = TMP / "audit.jsonl"

from groundwork.config import OUTPUTS_DIR              # noqa: E402
from groundwork import project as project_mod          # noqa: E402

PROBE = OUTPUTS_DIR / ".authcheck.txt"
ADMIN, ADMIN_PW = "boss", "adminpass1"
PLAIN, PLAIN_PW = "worker", "workerpass1"


def main() -> int:
    made_probe = False
    made_project = None
    made_extra: list[str] = []
    try:
        users.add(ADMIN, ADMIN_PW, admin=True)
        users.add(PLAIN, PLAIN_PW)

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        if not PROBE.exists():
            PROBE.write_text("a stand-in for gigabytes of source photographs\n")
            made_probe = True

        from fastapi.testclient import TestClient
        import groundwork.web.app as webui
        anon = TestClient(webui.app)

        # ---- 1. the front door is open, and says you are not signed in ------
        print("\n1. the shell loads so it can ask whether you are signed in")
        note(anon.get("/").status_code == 200, "GET / serves the SPA shell")
        note(anon.get("/api/me").status_code == 401,
             "GET /api/me answers 401 (reachable, so the browser can learn it)")

        # ---- 2. THE MOUNT ---------------------------------------------------
        print("\n2. the outputs/ MOUNT is gated, not just the routes")
        r = anon.get("/outputs/.authcheck.txt")
        note(r.status_code == 401,
             f"GET /outputs/<file> -> {r.status_code} (must be 401, not 200)")
        note("stand-in" not in r.text, "and the file's bytes did not come back")
        note(anon.get("/api/state").status_code == 401, "GET /api/state -> 401")
        note(anon.get("/api/projects").status_code == 401, "GET /api/projects -> 401")
        # `machine` IS DELIBERATELY UNROUTABLE, and this is not paranoia — it is
        # the fix for damage this check already did.
        #
        # An empty body defaults to a real worker machine and model. While
        # MUTATION-TESTING this file (removing AuthGate to prove part 2 could go
        # red) that default reached the real dispatcher, which called
        # mirror.sync_one and rsynced THIS NEAR-EMPTY WORKTREE to the worker with
        # --delete. It erased the worker's raw/, testset/ and needs_fix/ — 958 files
        # — and then started a retrain there that died at "split failed". HQ is
        # the source of truth so a re-mirror restored it, but a check that can
        # reach another machine is a check that can break one.
        #
        # An unknown machine is rejected at argument validation, BEFORE any sync
        # or dispatch, so the assertion keeps its full meaning — the gate runs
        # ahead of the route either way — while a gate-off mutation now gets a
        # 422 instead of a live rsync.
        r = anon.post("/api/train", json={"machine": "nowhere-check-only"})
        note(r.status_code == 401,
             f"POST /api/train -> {r.status_code} (no anonymous training)")

        # ---- 3. the allow-list cannot be walked out of ----------------------
        print("\n3. an open prefix cannot be traversed out of")
        r = anon.get("/static/../outputs/.authcheck.txt")
        note(r.status_code != 200 and "stand-in" not in r.text,
             f"/static/../outputs/<file> -> {r.status_code}, no bytes")

        # ---- 4. with a session, the same URLs work -------------------------
        print("\n4. signing in opens exactly those doors")
        bad = anon.post("/api/login", json={"username": ADMIN, "password": "wrong-one"})
        note(bad.status_code == 401, "a wrong password is refused")
        note("username" not in bad.text.lower() or "wrong username or password" in bad.text,
             "and the message does not say WHICH half was wrong")

        boss = TestClient(webui.app)
        r = boss.post("/api/login", json={"username": ADMIN, "password": ADMIN_PW})
        note(r.status_code == 200, "the right password signs in")
        note(sessions.COOKIE in boss.cookies, "a session cookie was set")
        note(boss.get("/api/me").json().get("username") == ADMIN,
             "/api/me now names the account (the casing bug would 401 here)")
        r = boss.get("/outputs/.authcheck.txt")
        note(r.status_code == 200 and "stand-in" in r.text,
             f"GET /outputs/<file> -> {r.status_code} WITH a session")

        # ---- 5. admin-only routes ------------------------------------------
        print("\n5. managing users is admin-only")
        hand = TestClient(webui.app)
        hand.post("/api/login", json={"username": PLAIN, "password": PLAIN_PW})
        note(hand.get("/api/users").status_code == 403,
             "a non-admin gets 403 on /api/users — 403, not 401 (see routes.py)")
        note(boss.get("/api/users").status_code == 200,
             "an admin gets the list")
        note(all("hash" not in u for u in boss.get("/api/users").json()["users"]),
             "and no digest is in the payload")

        # ---- 6. PROJECT OWNERSHIP -------------------------------------------
        print("\n6. a project belongs to whoever made it")
        r = hand.post("/api/projects", json={"name": "Auth Check Bolts",
                                             "classes": ["bolt"]})
        if r.status_code != 200:
            note(False, f"could not create a project as {PLAIN}: {r.status_code} {r.text[:120]}")
        else:
            made_project = r.json()["project"]["slug"]
            note(r.json()["project"]["owner"] == PLAIN,
                 f"it is owned by its creator ({r.json()['project']['owner']})")
            second = TestClient(webui.app)
            second.post("/api/login", json={"username": ADMIN, "password": ADMIN_PW})
            # The admin CAN see it (documented trade — see deps.readable_by),
            # so ownership is checked with the OTHER direction: make a project as
            # the admin and confirm the plain user is refused.
            r2 = second.post("/api/projects", json={"name": "Auth Check Boss Only",
                                                    "classes": ["thing"]})
            boss_slug = r2.json()["project"]["slug"] if r2.status_code == 200 else None
            if boss_slug:
                got = hand.get(f"/api/state?project={boss_slug}")
                note(got.status_code == 403,
                     f"a non-owner gets 403 on someone else's project ({got.status_code})")
                listed = [p["slug"] for p in hand.get("/api/projects").json()["projects"]]
                note(boss_slug not in listed,
                     "and it is ABSENT from their project list, not merely locked")
                note(made_project in listed, "while their own project is listed")
                import shutil
                shutil.rmtree(project_mod.manifest_path(boss_slug).parent,
                              ignore_errors=True)

        # ---- 6b. two accounts may use the same project NAME -----------------
        #
        # The slug and the name were lockstep, so the second account to want
        # "Bolts" got 409 about a project it cannot see — a dead end AND a small
        # existence oracle. The slug stays globally unique (directory name,
        # ?project=, ledger column, GW_PROJECT); the name does not.
        print("\n6b. the same project name works for two different accounts")
        ra = hand.post("/api/projects", json={"name": "Shared Name Check",
                                              "classes": ["x"]})
        rb = second.post("/api/projects", json={"name": "Shared Name Check",
                                                "classes": ["x"]})
        sa = ra.json().get("project", {}).get("slug")
        sb = rb.json().get("project", {}).get("slug")
        for s in (sa, sb):
            if s:
                made_extra.append(s)
        note(ra.status_code == 200 and rb.status_code == 200,
             f"both accounts created it ({ra.status_code}, {rb.status_code})")
        note(sa and sb and sa != sb, f"with DIFFERENT ids ({sa} vs {sb})")
        note(rb.json().get("project", {}).get("name") == "Shared Name Check",
             "and the second keeps the name that was typed")
        roots = {project_mod.load(s).dataset_root for s in (sa, sb) if s}
        note(len(roots) == 2, "so their datasets are separate directories")
        # ...but one account may not have two of its own, or the landing page
        # shows two identical cards.
        again = hand.post("/api/projects", json={"name": "Shared Name Check",
                                                 "classes": ["x"]})
        note(again.status_code == 409,
             f"the SAME account is refused a duplicate ({again.status_code})")

        # ---- 6c. API KEYS: same rules, one deliberate asymmetry -------------
        print("\n6c. an API key does the work but cannot escalate")
        from groundwork.web.auth import keys as keys_mod
        kid, raw = keys_mod.mint(PLAIN, "check-key")
        H = {"Authorization": f"Bearer {raw}"}
        note(anon.get("/api/projects", headers=H).status_code == 200,
             "a key authenticates for ordinary work")
        note(anon.get("/outputs/.authcheck.txt", headers=H).status_code == 200,
             "...including the outputs mount")
        note(anon.get("/api/projects").status_code == 401,
             "and the SAME client without the header is still refused")
        # The asymmetry: a leaked training key must not become a takeover.
        note(anon.get("/api/users", headers=H).status_code == 403,
             "a key cannot list accounts")
        note(anon.post("/api/keys", headers=H, json={"name": "x"}).status_code == 403,
             "a key cannot mint another key (or revocation means nothing)")
        note(anon.post("/api/me/password", headers=H,
                       json={"old_password": PLAIN_PW,
                             "new_password": "somethingnew1"}).status_code == 403,
             "a key cannot change the password")
        note(anon.post("/api/me/username", headers=H,
                       json={"new_username": "taken-over",
                             "password": PLAIN_PW}).status_code == 403,
             "a key cannot change the username")
        # ...and it is revocable, which is the whole point of storing a hash.
        keys_mod.revoke(kid, PLAIN)
        note(anon.get("/api/projects", headers=H).status_code == 401,
             "a revoked key stops working immediately")
        note(all(raw not in json.dumps(k) for k in keys_mod.listing(PLAIN)),
             "and the key itself never appears in a listing")

        # ---- 7. logging out ends it ----------------------------------------
        print("\n7. logging out actually revokes")
        boss.post("/api/logout")
        note(boss.get("/outputs/.authcheck.txt").status_code == 401,
             "the same cookie no longer opens the mount")
    finally:
        if made_probe:
            PROBE.unlink(missing_ok=True)
        for slug in made_extra:
            import shutil as _s2
            _s2.rmtree(project_mod.manifest_path(slug).parent, ignore_errors=True)
            _s2.rmtree(OUTPUTS_DIR / "projects" / slug, ignore_errors=True)
        if made_project:
            import shutil
            shutil.rmtree(project_mod.manifest_path(made_project).parent,
                          ignore_errors=True)
            shutil.rmtree(OUTPUTS_DIR / "projects" / made_project, ignore_errors=True)
        import shutil as _sh
        _sh.rmtree(TMP, ignore_errors=True)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("PASS — routes AND the outputs/ mount are closed; ownership is enforced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
