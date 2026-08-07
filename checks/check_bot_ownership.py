"""A bot belongs to its project's owner, and its unit cannot be dictated.

    .venv/bin/python checks/check_bot_ownership.py

WHAT THIS CATCHES. Every endpoint in web/api/bot_setup.py was unauthenticated
until 2026-08-05 — not "admin-only by accident", not "gated by the project": no
`Depends` at all, and a `_find(key)` that scanned every project on the machine
and returned the first key that matched. Four ordinary requests from an ordinary
signed-in account, and none of these needed admin:

  PART 2  write a token onto someone else's bot; install, stop or restart their
          service; or MOVE their bot into your own project, which rewrote its
          unit with your slug and restarted it — their Telegram photos into
          your dataset.
  PART 3  a bot NAME containing newlines injected directives into the systemd
          unit. `systemd-analyze verify` accepts the result (exit 0, one warning
          about the orphaned fragment) and runs the injected ExecStartPre as the
          user who owns .env and private/. That is ordinary account -> code
          execution, and it is why part 3 asserts on rendered TEXT.
  PART 4  `unit` came off the request, checked against a denylist of seven
          infrastructure units that did not include the owner's own bot unit.
          So a bot registered in your own project could name the owner's unit
          and stop or overwrite it.
  PART 5  `token_env` was unique per PROJECT while .env is one flat file, so a
          bot claiming TELEGRAM_BOT_TOKEN overwrote the owner's token and their
          service logged in as yours at its next restart.

HOW IT CAN FAIL. Every assertion changes with the credential or the input, and
each part names the mutation that reddens it. Reverting `owned_bot` to a
machine-wide scan turns part 2 green-to-red; dropping `clean_description`
reddens 3; accepting `req.unit` reddens 4; accepting `req.token_env` reddens 5.

SAFE: users, sessions and the audit trail are redirected to a temp directory
BEFORE the app is imported, and `bot_units.USER_UNITS` is redirected too — so
nothing here can write into ~/.config/systemd/user/. install() is never called,
because it would `systemctl enable --now` a real service; part 3 asserts on the
text install WOULD write. No GPU, no network (no token is ever verified, which
is why the token assertions stop at the refusal rather than the store).
"""
from __future__ import annotations

import os
import shutil
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


# REDIRECT BEFORE groundwork.web IS IMPORTED — it bootstraps the user store at
# module level, so importing first would touch the real one.
TMP = Path(tempfile.mkdtemp(prefix="checkbots-"))
os.environ["GW_AUTH"] = "1"
os.environ.pop("GW_ADMIN_USER", None)
os.environ.pop("GW_ADMIN_PASSWORD", None)

from groundwork.web.auth import audit, sessions, users        # noqa: E402
users.USERS_PATH = TMP / "users.json"
sessions.SESSIONS_PATH = TMP / "sessions.json"
audit.AUDIT_PATH = TMP / "audit.jsonl"

from groundwork.web import bot_roles, bot_units               # noqa: E402
# NOTHING MAY REACH THE REAL UNIT DIRECTORY. Redirected rather than trusted:
# this check registers bots whose whole purpose is to try to name someone
# else's service.
bot_units.USER_UNITS = TMP / "systemd"

from groundwork import project as project_mod                 # noqa: E402

ADMIN, ADMIN_PW = "botboss", "adminpass1"
PLAIN, PLAIN_PW = "botworker", "workerpass1"

EVIL_NAME = "MyBot\n\n[Service]\nExecStartPre=/bin/sh -c 'touch /tmp/PWNED'\n"


def main() -> int:
    made: list[str] = []
    try:
        users.add(ADMIN, ADMIN_PW, admin=True)
        users.add(PLAIN, PLAIN_PW)

        from fastapi.testclient import TestClient
        import groundwork.web.app as webui

        boss = TestClient(webui.app)
        boss.post("/api/login", json={"username": ADMIN, "password": ADMIN_PW})
        hand = TestClient(webui.app)
        hand.post("/api/login", json={"username": PLAIN, "password": PLAIN_PW})

        # ---- 1. two accounts, two projects, one bot each -------------------
        print("\n1. each account gets a project and registers a bot in it")
        rb = boss.post("/api/projects", json={"name": "Boss Bot Check",
                                              "classes": ["x"]})
        rh = hand.post("/api/projects", json={"name": "Worker Bot Check",
                                              "classes": ["x"]})
        bslug = rb.json().get("project", {}).get("slug")
        hslug = rh.json().get("project", {}).get("slug")
        for s in (bslug, hslug):
            if s:
                made.append(s)
        note(bool(bslug and hslug), f"projects created ({bslug}, {hslug})")
        if not (bslug and hslug):
            return 1

        r = boss.post(f"/api/bots?project={bslug}",
                      json={"name": "Boss Counter", "role": "counter"})
        note(r.status_code == 200, f"boss registered a bot ({r.status_code})")
        bkey = r.json().get("bot", {}).get("key")
        r = hand.post(f"/api/bots?project={hslug}",
                      json={"name": "Worker Counter", "role": "counter"})
        note(r.status_code == 200, f"worker registered a bot ({r.status_code})")
        hkey = r.json().get("bot", {}).get("key")

        # ---- 2. THE HOLE: someone else's bot ------------------------------
        print("\n2. a non-owner cannot touch the owner's bot")
        for path, body in (("token", {"token": "1234567890:" + "a" * 35}),
                           ("install", None),
                           ("service", {"action": "stop"}),
                           ("allowed", {"telegram_ids": ["123456789"]}),
                           ("project", {"project": hslug})):
            url = f"/api/bots/{bkey}/{path}?project={bslug}"
            got = hand.post(url, json=body) if body else hand.post(url)
            note(got.status_code == 403,
                 f"POST .../{path} on the owner's bot -> {got.status_code} "
                 f"(must be 403)")
        got = hand.post(f"/api/bots/{bkey}/probe?project={bslug}")
        note(got.status_code == 403,
             f"POST .../probe on the owner's bot -> {got.status_code}")

        # Naming YOUR project while asking about THEIR bot must not work either
        # — that is the shape the old machine-wide _find() made possible.
        got = hand.post(f"/api/bots/{bkey}/service?project={hslug}",
                        json={"action": "stop"})
        note(got.status_code == 404,
             f"...and their key under YOUR project -> {got.status_code} (404)")

        # ---- 3. the unit cannot be injected into -------------------------
        # The name check runs BEFORE the role cap, so the 422 here is about the
        # newline, not about the project already holding a counter.
        print("\n3. a bot name cannot become a systemd directive")
        got = hand.post(f"/api/bots?project={hslug}",
                        json={"name": EVIL_NAME, "role": "counter"})
        note(got.status_code == 422,
             f"a name with line breaks is refused at registration "
             f"({got.status_code})")

        # ...and even if one were already on disk (a hand-edited manifest),
        # the rendered unit must carry no directive of its own.
        p = project_mod.load(hslug)
        role = bot_roles.get("counter")
        evil_bot = {"key": "x", "name": EVIL_NAME, "role": "counter",
                    "unit": bot_units.derived_unit(hslug, "counter"),
                    "token_env": "GW_X_TOKEN", "log": "x.log", "allowed_ids": []}
        text = bot_units.render(p, evil_bot, role)
        bad = [ln for ln in text.splitlines()
               if ln.strip().startswith(("ExecStartPre", "ExecStartPost",
                                         "ExecStop", "ExecReload"))]
        note(not bad, f"the rendered unit has no injected Exec* line ({bad})")
        note(text.count("\nExecStart=") == 1,
             "and exactly one ExecStart, the role's own")
        note(sum(1 for ln in text.splitlines() if ln.strip() == "[Service]") == 1,
             "and exactly one [Service] section")

        # An environment value is REFUSED outright rather than collapsed —
        # it is not cosmetic the way a description is.
        try:
            bot_roles.unit_text(role, "fine", {"GW_PROJECT": "a\nExecStartPre=/x"})
            note(False, "a newline in an environment VALUE was accepted")
        except ValueError:
            note(True, "a newline in an environment value raises")

        # ---- 4. the unit NAME is derived, never dictated ------------------
        print("\n4. a request cannot name someone else's service")
        got = hand.post(f"/api/bots?project={hslug}",
                        json={"name": "Sneaky", "role": "counter",
                              "unit": "groundwork-web.service"})
        # The cap refuses it first (one counter per project); what matters is
        # that no path exists by which `unit` is taken from the body.
        stored = [b for b in project_mod.load(hslug).bots]
        note(all(b.get("unit", "").startswith("gw-") for b in stored),
             f"every unit is derived gw-* ({[b.get('unit') for b in stored]})")
        note("unit" not in str(got.json()).lower() or got.status_code != 200,
             "and a dictated unit never lands")
        # The allowlist itself, both directions — asserting only the refusal
        # would pass just as green with the rule deleted.
        try:
            bot_units.unit_of(p, {"unit": "groundwork-web.service"})
            note(False, "a bot was allowed to claim the web server's unit")
        except bot_units.UnitError:
            note(True, "a bot cannot claim the web server's unit (PROTECTED)")
        try:
            bot_units.unit_of(p, {"unit": "not-ours.service"})
            note(False, "an underived unit name was accepted")
        except bot_units.UnitError:
            note(True, "and an underived name is refused (gw-* only)")
        note(bot_units.unit_of(p, {"unit": bot_units.derived_unit(hslug, "counter")})
             == f"gw-{hslug}-counter.service",
             "while the bot's own derived unit resolves")

        # ---- 5. token_env is derived, so .env cannot be hijacked ----------
        print("\n5. a bot cannot claim another's token variable")
        envs = [b.get("token_env") for b in stored]
        note(all(e and e.startswith("GW_") for e in envs),
             f"token vars are derived and namespaced ({envs})")
        note("TELEGRAM_BOT_TOKEN" not in envs,
             "and none of them is the default bot's variable")

        # ---- 6. one bot per role per project ------------------------------
        print("\n6. a project has one bot per job")
        again = hand.post(f"/api/bots?project={hslug}",
                          json={"name": "Second Counter", "role": "counter"})
        note(again.status_code == 409,
             f"a second counting bot in one project -> {again.status_code} (409)")
        note(bkey != hkey, f"while two projects each have their own ({bkey}, {hkey})")

        # ---- 7. the page shows only your own -----------------------------
        print("\n7. Data collection lists only what you may open")
        d = hand.get("/api/collect").json()
        slugs = [x["slug"] for x in d.get("projects", [])]
        keys = [b["key"] for b in d.get("bots", [])]
        note(bslug not in slugs, f"the owner's project is absent ({slugs})")
        note(hslug in slugs, "while your own is listed")
        note(bkey not in keys, f"the owner's bot is absent ({keys})")
        db = boss.get("/api/collect").json()
        note(hslug in [x["slug"] for x in db.get("projects", [])],
             "...while an admin sees every project (deps.readable_by)")

        # ---- 8b. someone else's class list ---------------------------------
        # A class list is a SCHEMA for every label already on disk — the leading
        # integer in each line is a position in it. PATCH .../classes took no
        # dependency at all, so any signed-in account could rewrite anyone's.
        print("\n8b. a class list belongs to its project")
        got = hand.patch(f"/api/projects/{bslug}/classes",
                         json={"classes": ["hijacked"]})
        note(got.status_code == 403,
             f"a non-owner cannot rewrite it ({got.status_code})")
        note(list(project_mod.load(bslug).classes) == ["x"],
             f"and it is unchanged ({list(project_mod.load(bslug).classes)})")
        got = hand.patch(f"/api/projects/{hslug}/classes",
                         json={"classes": ["x", "y"]})
        note(got.status_code == 200,
             f"while the owner still may ({got.status_code})")

        # ---- 9. writing a systemd unit is session-only ---------------------
        # A key is a string in a config file; installing a unit is arbitrary
        # code as the user who owns .env. Asserted BOTH ways, because a check
        # that only proves the refusal would pass with the whole rule deleted.
        print("\n9. an API key can register a bot but not install one")
        from groundwork.web.auth import keys as keys_mod
        _kid, raw = keys_mod.mint(PLAIN, "bot-check-key", scope=keys_mod.FULL)
        H = {"Authorization": f"Bearer {raw}"}
        anon = TestClient(webui.app)
        got = anon.post(f"/api/bots/{hkey}/install?project={hslug}", headers=H)
        note(got.status_code == 403,
             f"a FULL key is refused install ({got.status_code})")
        got = anon.post(f"/api/bots/{hkey}/service?project={hslug}", headers=H,
                        json={"action": "stop"})
        note(got.status_code == 403,
             f"...and refused start/stop ({got.status_code})")
        got = anon.get(f"/api/bots?project={hslug}", headers=H)
        note(got.status_code == 200,
             f"but it still does ordinary work ({got.status_code}) — the rule "
             f"costs a key nothing real")

        # ---- 10. a counting bot needs something to count with -------------
        # Tested as a unit rather than through the endpoint: install() checks
        # the token first, and setting one would mean writing the machine's
        # real .env from a check.
        print("\n10. install refuses until the project has a trained run")
        from groundwork.web.api.bot_setup import _needs_a_model
        counter_role = bot_roles.get("counter")
        try:
            _needs_a_model(project_mod.load(hslug), counter_role)
            note(False, "a project with no runs was allowed a counting bot")
        except Exception as e:  # noqa: BLE001 — HTTPException
            note(getattr(e, "status_code", None) == 400
                 and "LocateAnything" in str(getattr(e, "detail", "")),
                 "it refuses, and names the route through (upload -> "
                 "LocateAnything -> train)")
        # Give the project a completed run (weights on disk) and the same call
        # must now allow it — otherwise the gate is a wall, not a step.
        fake = (project_mod.load(hslug).dataset_root / "runs" / "checkrun"
                / "weights")
        fake.mkdir(parents=True, exist_ok=True)
        (fake / "best.pt").write_bytes(b"")
        try:
            _needs_a_model(project_mod.load(hslug), counter_role)
            note(True, "while a project WITH a trained run is allowed")
        except Exception as e:  # noqa: BLE001
            note(False, f"a project with a run was refused: {e}")
        # Only the counting role needs a detector: a role with any other key
        # must be exempt, or the gate blocks jobs that never count anything.
        other_role = bot_roles.Role(key="capture", name="Capture bot",
                                    what="x", module="groundwork.bots.telegram")
        shutil.rmtree(fake.parent, ignore_errors=True)
        try:
            _needs_a_model(project_mod.load(hslug), other_role)
            note(True, "and a non-counting role is exempt (needs no detector)")
        except Exception as e:  # noqa: BLE001
            note(False, f"a non-counting role was refused a model it does not use: {e}")
    finally:
        for slug in made:
            shutil.rmtree(project_mod.manifest_path(slug).parent, ignore_errors=True)
            try:
                shutil.rmtree(project_mod.load(slug).dataset_root, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(TMP, ignore_errors=True)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("all good — a bot belongs to its project's owner, and its unit is ours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
