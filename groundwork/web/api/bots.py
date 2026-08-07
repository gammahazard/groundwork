"""Which Telegram bots exist, and recording a new one.

Read-only plus registration and one explicit probe. The SETUP half — writing a
token, writing a systemd unit, starting it, moving a bot between projects —
lives in `bot_setup.py`, which also carries the security reasoning for why a
cockpit is allowed to do any of that.

This file used to say "nothing here creates a bot, writes a token, or starts a
service". That was true and is not any more; it is corrected rather than left,
because a docstring asserting the opposite of the truth is worse than none.

ONE BOT PER ROLE PER PROJECT (owner, 2026-08-05). An ordinary project therefore
gets exactly one bot — you have as many bots as you have projects. the first project has
two because it enables an extension, which unlocks a second role; the
roles table is already extension-gated, so the cap needed no new concept.

That cap is also what makes the NAMES derivable, which is the security half.
`key`, `unit`, `token_env` and `log` used to be typed or free-derived from a
display name and checked for uniqueness only WITHIN a project — while systemd
units and .env are machine-global namespaces. So a second account could claim
a hand-written counting-bot unit or `TELEGRAM_BOT_TOKEN` from inside its own project and take
over the owner's bot. With at most one bot per (project, role) and slugs already
globally unique, `<slug>-<role>` is unique by construction: nothing to check,
nothing to collide, and no 409 that would reveal a bot the caller cannot see.
"""
from __future__ import annotations

import re
from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ... import project as project_mod
from ...web import bots as bots_mod
from .. import bot_roles, bot_units, lab_guard
from .bot_deps import owned_bot
from .deps import current_project

router = APIRouter()


@router.get("/api/bots")
def list_bots(p=Depends(current_project)):
    """This project's bots, with live systemd + last-activity state.

    Filtered by project: a bot that feeds another project's dataset is not this
    project's business, and showing it here would imply the images it collects
    land in this one.
    """
    rows = bots_mod.status(p.slug)
    # WHICH process model runs bots here — the card offers "Install as a
    # service" under systemd and a plain Start under the supervisor, and
    # guessing client-side had it showing an install step that does not exist.
    from .. import botsup
    process_model = botsup.mode()
    # A bot behind an extension the project does not enable is not "missing",
    # it is not applicable — drop it rather than showing a dead card.
    rows = [b for b in rows if not b["extension"] or p.has(b["extension"])]
    return {"project": p.slug, "process_model": process_model, "bots": rows, "steps": setup_steps(p.slug)}


class NewBot(BaseModel):
    name: str
    # WHICH JOB. The one field that decides what a service would run, and the
    # reason install() is safe — it is looked up in web/bot_roles.py, a closed
    # set of literal module paths, never interpolated. It is REQUIRED now:
    # every other identifier is derived from it, and an entry with no role was
    # only ever "recorded but not installable", which is a state nothing needs
    # to be able to create on purpose.
    role: str
    what: str = ""


# A display name. Length and control characters are checked SERVER-SIDE because
# this string reaches a systemd unit's Description line — the HTML maxlength is
# a courtesy, not a check, and a newline here was code execution until
# 2026-08-05 (see bot_roles.unit_text).
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
NAME_MAX = 60


@router.post("/api/bots")
def register_bot(req: NewBot, p=Depends(current_project)):
    """Record that this project HAS a bot. Metadata only — never a token.

    Still metadata-only, and that is now a UI decision rather than a security
    one: the token is pasted on the bot's own card afterwards
    (bot_setup.set_token), so it never travels alongside a name and cannot end
    up in a browser's saved form data.

    `role` is the field that matters. It names one of bot_roles.ROLES, which is
    what makes installing a service safe — the ExecStart is a literal in that
    file, so a request chooses WHICH job and nothing else.

    Refused on the lab: its tree is a one-way mirror, and a manifest written
    there is gone at the next sync.
    """
    lab_guard.no_lab_edits()
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(422, "a bot needs a name")
    if len(name) > NAME_MAX or _CTRL.search(name):
        raise HTTPException(422, f"a bot name is plain text, up to {NAME_MAX} "
                                 f"characters — no line breaks")
    what = (req.what or "").strip()
    if len(what) > 200 or _CTRL.search(what):
        raise HTTPException(422, "the description is plain text, up to 200 "
                                 "characters")

    role_key = (req.role or "").strip()
    role = bot_roles.get(role_key)
    if role is None:
        raise HTTPException(422, f"unknown role {role_key!r} — one of "
                                 f"{', '.join(bot_roles.ROLES)}")
    # SERVER-SIDE, not just absent from the form. The UI omitting a choice is a
    # convenience; this is the refusal — same reasoning as lab_guard being in
    # the router rather than a hidden button.
    if not role.offerable:
        raise HTTPException(400, f"{role.name} is not on offer for new bots "
                                 f"right now — counting bots only")
    if role.extension and not p.has(role.extension):
        raise HTTPException(400, f"{p.slug} does not enable the "
                                 f"{role.extension!r} extension")
    # THE CAP. One bot per role per project — so an ordinary project has one
    # bot, and the derived identifiers below cannot collide.
    if any((b.get("role") or "") == role_key for b in p.bots):
        raise HTTPException(409, f"{p.name} already has a {role.name.lower()}. "
                                 f"A project has one bot per job — remove that "
                                 f"one first, or add the bot to another project.")

    key = bot_units.derived_key(p.slug, role_key)
    if any(b.get("key") == key for b in p.bots):
        raise HTTPException(409, f"{p.slug} already has a bot called {key!r}")
    entry = {
        "key": key,
        "name": name,
        "role": role_key,
        # DERIVED, all three. See the module docstring: these are machine-global
        # namespaces, and letting a request choose a value in one of them from
        # inside a project it owns is how a second account reaches the first's.
        "unit": bot_units.derived_unit(p.slug, role_key),
        "token_env": bot_units.derived_token_env(p.slug, role_key),
        "log": bot_units.derived_log(p.slug, role_key),
        # Nobody may talk to it until someone is named. An empty list is
        # "unclaimed", which is deliberately NOT "whoever owns the machine".
        "allowed_ids": [],
        "what": what or "Collects data for this project.",
    }
    project_mod.save(replace(p, bots=tuple(p.bots) + (entry,)))
    print(f"[bots] {p.slug}: registered {key!r} role={role_key} "
          f"unit={entry['unit']} env={entry['token_env']}", flush=True)
    return {"ok": True, "bot": entry}


@router.delete("/api/bots/{key}")
def unregister_bot(key: str, p=Depends(current_project)):
    """Forget a bot. Removes the ENTRY only — never the unit, never the token.

    Deleting a systemd unit or editing .env from here would be the same
    overreach registering avoids; this just stops the project claiming a bot it
    does not have. The response SAYS whether a service is still running, because
    silently leaving one polling Telegram for a bot the cockpit no longer lists
    is the kind of gap this panel exists to close.
    """
    lab_guard.no_lab_edits()
    gone = next((b for b in p.bots if b.get("key") == key), None)
    if gone is None:
        raise HTTPException(404, f"no bot {key!r} in {p.slug}")
    left = tuple(b for b in p.bots if b.get("key") != key)
    project_mod.save(replace(p, bots=left))
    still = bot_units.installed(gone)
    return {"ok": True, "removed": key, "unit": gone.get("unit"),
            "service_still_installed": still,
            "note": (f"{gone.get('unit')} is still installed and may still be "
                     f"running — stop it with systemctl if you meant to retire "
                     f"it.") if still else None}


@router.post("/api/bots/{key}/probe")
def probe_bot(pb=Depends(owned_bot)):
    """Ask Telegram whether this token actually works, and who it belongs to.

    POST rather than GET because it makes an outbound network call to a third
    party; that should not happen because something prefetched a URL.

    Scoped like everything else in bot_setup: this used to take a bare key and
    search every project, so anyone could confirm anyone's token was live.
    """
    _p, bot = pb
    return bots_mod.probe(bot)


def setup_steps(slug: str | None) -> list[dict]:
    """The setup flow, as text the cockpit can render.

    ONE STEP IS STILL YOURS, and always will be: creating a bot is a
    conversation with @BotFather from YOUR Telegram account. There is no API for
    it, by design, because a bot is owned by a human account. Everything after
    that is a button now — this list used to be five manual steps ending in
    "edit a systemd unit", which is what the owner asked to remove.

    `slug` may be None — a signed-in account with no projects yet. The text then
    names no project rather than naming one they cannot open.
    """
    where = f"(default {slug})" if slug else ""
    return [
        {"n": 1, "title": "Create the bot with @BotFather",
         "body": "In Telegram, message @BotFather and send /newbot. Give it a "
                 "display name and a username ending in 'bot'. It replies with "
                 "a token that looks like 1234567890:AA... This is the one step "
                 "no button can do — Telegram has no API for creating a bot.",
         "where": "Telegram"},
        {"n": 2, "title": "Register it here",
         "body": f"'Add a collection bot' below: pick the project it feeds "
                 f"{where}, what it does, and a name. That records the "
                 "bot; nothing runs yet.".replace("  ", " "),
         "where": "this page"},
        {"n": 3, "title": "Paste the token on its card",
         "body": "The card appears with a token field. Paste what BotFather "
                 "gave you and press Verify & save — it is checked with "
                 "Telegram before it is stored, so a typo is an error here "
                 "rather than a service that fails to log in.",
         "where": "this page"},
        {"n": 4, "title": "Say who may use it",
         "body": "Message your new bot /whoami — it replies with your numeric "
                 "Telegram id. Paste that on the card. Until you do, the bot "
                 "answers nobody: an unclaimed bot is not 'owned by whoever "
                 "owns the machine'.",
         "where": "Telegram, then this page"},
        {"n": 5, "title": "Install it as a service",
         "body": "Press Install. The systemd unit is written, reloaded, enabled "
                 "and started, pinned to the project you chose and to the "
                 "people you named. The card turns 'running'.",
         "where": "this page"},
        {"n": 6, "title": "Send it an image",
         "body": "Photograph an image and send it. 'last activity' starts moving "
                 "as soon as the bot receives anything, and the count comes "
                 "back from your project's own model.",
         "where": "Telegram"},
    ]
