"""Which bot a process thinks it is — token, people, project, log.

    .venv/bin/python checks/check_bot_identity.py

WHAT THIS CATCHES, and none of it is hypothetical.

  THE TOKEN. The predecessor's bot read one literal machine-wide token
  variable, so a SECOND bot — installed for a second project, with its own
  token verified and stored — would long-poll Telegram as the FIRST bot. Two
  processes on one getUpdates stream split the messages between them, so one
  person's images would be counted by someone else's process, with someone
  else's model, into someone else's project. Nothing errors. `token_env`
  existed on the bot record the whole time and NOTHING read it. Now
  groundwork/bots/env.py is the ONE place identity resolves, and a unit that
  does not name its token variable is refused, never guessed.

  THE PEOPLE. One machine-wide allowed-id, shared by every bot, is why a
  second person could not have a bot at all: setting their id took the first
  person's away. There is NO machine-wide fallback anymore: a bot that names
  nobody answers NOBODY — never "whoever owns the box", because a new user
  cannot learn their own numeric id until the bot is running to answer
  /whoami, so a fallback would hand their freshly installed bot to someone
  else.

HOW IT CAN FAIL: every row is a resolution that must CHANGE with the
environment. Adding any fallback token variable reddens rows 1 and 6.
Dropping the token shape check reddens row 7. Any machine-wide id fallback
reddens row 4.

WHY SUBPROCESSES: systemd delivers identity as Environment= lines before
exec, so each row is a fresh interpreter with exactly the variables a unit
would set — immune to whether resolution happens at import or at call time.

SAFE: no Telegram, no GPU, no network — nothing is imported but
groundwork.bots.env (stdlib-only, by design). Tokens are fake and never sent
anywhere; the machine's .env is never read (env.py takes os.environ only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []

FAKE_A = "1111111111:" + "A" * 35
FAKE_B = "2222222222:" + "B" * 35

# Printed by the child; the parent compares. `describe` is included because it
# is the startup log line, and the one place a token could leak by accident.
PROBE = (
    "import json;"
    "from groundwork.bots import env as botenv;"
    "tok, var, refusal = botenv.token();"
    "print(json.dumps({'token': tok, 'var': var, 'refusal': refusal,"
    " 'ids': sorted(botenv.allowed_ids()),"
    " 'project': botenv.project_slug(),"
    " 'log': botenv.log_name('bot.log'),"
    " 'describe': botenv.describe(var)}))"
)


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


def resolve(**env) -> dict:
    """Resolve identity in a fresh interpreter under exactly this environment."""
    e = dict(os.environ)
    # Anything a cockpit unit would set starts absent, so a row means what it
    # says — leftovers from the calling shell cannot leak in.
    for k in ("GW_PROJECT", "GW_TOKEN_ENV", "GW_ALLOWED_IDS", "GW_BOT_LOG"):
        e.pop(k, None)
    e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run([sys.executable, "-c", PROBE], env=e, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        return {"error": (r.stderr or "").strip()[-400:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": f"unparseable: {r.stdout[-200:]}"}


def main() -> int:
    print("\n1. a unit that names no token variable is refused, not guessed")
    d = resolve()
    note(bool(d.get("refusal")),
         f"no GW_TOKEN_ENV -> refusal ({(d.get('refusal') or str(d))[:60]}…)")
    note(d.get("token") == "", "and no token resolves — there is no fallback "
                               "variable to guess from")

    print("\n2. a cockpit unit: everything named")
    d = resolve(GW_PROJECT="eli-thing",
                GW_TOKEN_ENV="GW_ELI_THING_COUNTER_TOKEN",
                GW_ELI_THING_COUNTER_TOKEN=FAKE_B,
                GW_ALLOWED_IDS="555555555,666666666")
    note(d.get("token") == FAKE_B, "it uses ITS OWN token, from its own variable")
    note(d.get("var") == "GW_ELI_THING_COUNTER_TOKEN",
         f"and reports which variable ({d.get('var')})")
    note(not d.get("refusal"), f"nothing refuses it ({d.get('refusal')})")
    note(d.get("project") == "eli-thing", "and its own project resolves")
    note(d.get("ids") == [555555555, 666666666],
         f"and its own people, plural ({d.get('ids')})")

    print("\n3. more than one person may use one bot")
    d = resolve(GW_PROJECT="p", GW_TOKEN_ENV="X_TOKEN",
                X_TOKEN=FAKE_B, GW_ALLOWED_IDS="111111111,222222222,333333333")
    note(len(d.get("ids") or []) == 3, f"three ids resolve ({d.get('ids')})")

    print("\n4. THE ROW THAT MATTERS: an unclaimed new bot is nobody's")
    d = resolve(GW_PROJECT="eli-thing", GW_TOKEN_ENV="X_TOKEN", X_TOKEN=FAKE_B)
    note(d.get("ids") == [],
         f"a unit with nobody named inherits NOBODY (got {d.get('ids')}) — "
         f"any machine-wide fallback here would hand a new user's bot to "
         f"whoever owns the box")
    note("NOBODY" in (d.get("describe") or ""),
         "and the startup line says so out loud (unclaimed)")

    print("\n5. no project named means no project, never a guess")
    note(d.get("project") == "eli-thing", "GW_PROJECT set -> that slug")
    d2 = resolve(GW_TOKEN_ENV="X_TOKEN", X_TOKEN=FAKE_B)
    note(d2.get("project") is None,
         "GW_PROJECT unset -> None, which the bot entrypoint refuses to start "
         "on — a bot must never guess whose dataset its images belong to")

    print("\n6. an empty variable is a refusal, not a silent fallback")
    d = resolve(GW_PROJECT="p", GW_TOKEN_ENV="X_TOKEN")
    note(d.get("token") == "" and bool(d.get("refusal")),
         "an unset per-bot variable does NOT fall through to anything")

    print("\n7. the variable name cannot be used to exfiltrate a secret")
    d = resolve(GW_PROJECT="p",
                GW_TOKEN_ENV="GW_ADMIN_PASSWORD",
                GW_ADMIN_PASSWORD="hunter2-not-a-token")
    note(d.get("token") == "" and bool(d.get("refusal")),
         "a variable holding something not shaped like a bot token is "
         "refused — otherwise the bot would POST it to api.telegram.org")
    note("hunter2" not in json.dumps(d),
         "and the value itself appears nowhere in the resolution")

    print("\n8. junk ids are dropped, not crashed on")
    d = resolve(GW_PROJECT="p", GW_TOKEN_ENV="X_TOKEN",
                X_TOKEN=FAKE_B,
                GW_ALLOWED_IDS="123456789, notanid ,,987654321,1")
    note(d.get("ids") == [123456789, 987654321],
         f"only well-formed ids survive ({d.get('ids')})")

    print("\n9. a log name cannot walk the filesystem")
    d = resolve(GW_PROJECT="p", GW_TOKEN_ENV="X_TOKEN", X_TOKEN=FAKE_B,
                GW_BOT_LOG="../../etc/evil.log")
    note(d.get("log") == "bot.log",
         f"a name with separators falls back to the default ({d.get('log')})")
    d = resolve(GW_PROJECT="p", GW_TOKEN_ENV="X_TOKEN", X_TOKEN=FAKE_B,
                GW_BOT_LOG="eli-thing-counter.log")
    note(d.get("log") == "eli-thing-counter.log",
         "while a plain per-bot name is honoured")

    print("\n10. the startup line never carries the token")
    d = resolve(GW_PROJECT="p", GW_TOKEN_ENV="X_TOKEN", X_TOKEN=FAKE_B)
    note(FAKE_B not in (d.get("describe") or "") and FAKE_B.split(":")[1]
         not in (d.get("describe") or ""),
         "describe() names the VARIABLE, not the secret")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("all good — a bot resolves its own token, people and project")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
