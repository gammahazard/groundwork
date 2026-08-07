"""Setting a bot UP: its token, who may use it, its service, which project it feeds.

`api/bots.py` reports what exists. This does the parts that used to be a list of
instructions telling you to go and edit files by hand.

THE SECURITY POSITION, changed deliberately (owner, 2026-07-31). The previous
one was "the cockpit will not write a token or create a service, because it
binds to 0.0.0.0 and is reachable over the network with no auth". That is still
a true description of the risk; the owner's answer is that the network is
theirs, every machine on it is theirs, and the alternative was ssh-ing in to
edit a dotfile every time a project gains a bot. Recorded rather than quietly
reversed, and the rails that make it defensible are:

  A UNIT'S CONTENTS ARE NEVER TAKEN FROM A REQUEST. The ExecStart comes from
  `web/bot_roles.py`, a closed set of literal module paths. So does everything
  else in the file now — see the 2026-08-05 note below, because for a while the
  first sentence was true and the file was still injectable.

  A BOT IS OWNED BY ITS PROJECT'S OWNER. Every endpoint here resolves its bot
  through `bot_deps.owned_bot`, so "may I touch this" is the question
  `deps.readable_by` already answers about the project it lives in. NOT admin —
  the point of the feature is that the person whose project it is can set up
  their own bot without one.

  A TOKEN IS VERIFIED BEFORE IT IS STORED. getMe first; a typo'd or revoked
  token is rejected instead of persisted, so ".env says it is set" and "the bot
  can log in" cannot disagree. Nothing here ever returns a token, and the error
  paths report exception TYPES because an httpx error carries the request URL.

  NOTHING IS DESTRUCTIVE. Installing overwrites one unit file this cockpit
  NAMED — see web/bot_units.py, which no longer lets a request choose the name.
  There is no endpoint that deletes a unit or removes data.

  THE LAB REFUSES ALL OF IT. Its tree is a one-way mirror; a manifest or a unit
  written there is gone at the next sync, and it does not collect data anyway.

WHAT WAS WRONG, 2026-08-05, recorded because a security fix with no account of
what it fixed is the shape this repo rejects. None of this needed admin, and all
of it needed only an ordinary signed-in account:

  * no endpoint here had ANY authorization — `_find` scanned every project and
    returned the first key match, so every bot on the box was everyone's;
  * a bot NAME containing newlines injected directives into the systemd unit
    (`bot_roles.unit_text`), giving code execution as the user who owns .env;
  * `unit` came off the request and the denylist did not include the counting bot.service,
    so another account could stop or hijack the owner's bot;
  * `token_env` was unique per PROJECT while .env is machine-global, so a bot
    registered elsewhere could overwrite the owner's token;
"""
from __future__ import annotations

import re
from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ... import project as project_mod
from .. import bot_roles, bot_units, env_file, lab_guard
from ..auth import audit as audit_mod
from ..auth.routes import refuse_key_auth, require_admin
from .bot_deps import clean_ids, owned_bot, required_project
from .deps import current_user, readable_by

router = APIRouter()

# What Telegram issues: a numeric bot id, a colon, then the secret. Checked
# before we spend a round trip, and — more usefully — so a pasted "Use this
# token to access the HTTP API:" line is rejected with a clear message rather
# than sent to Telegram and reported as "unauthorized".
_TOKEN = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")


def _trail(request: Request, user: str | None, event: str, p, key: str,
           detail: str) -> None:
    """One line in the security trail. Never raises; never carries a token."""
    audit_mod.record(event, actor=user, target=f"{p.slug}/{key}",
                     via=request.scope.get("auth_via") if request else None,
                     ip=(getattr(getattr(request, "client", None), "host", None)
                         if request else None),
                     detail=detail)


def _role_of(bot: dict):
    return bot_roles.get(bot.get("role") or "")


# ------------------------------------------------------------ who may use it ---

class Allowed(BaseModel):
    telegram_ids: list[str] = []


@router.post("/api/bots/{key}/allowed")
def set_allowed(req: Allowed, request: Request,
                pb=Depends(owned_bot), user: str | None = Depends(current_user)):
    """Which Telegram accounts THIS bot answers to.

    A LIST from the first day rather than a single id. A scalar per bot would
    reproduce, one level down, the exact bug this replaces — and a workplace has
    more than one technician, which is the actual product.

    The bot reads its environment at import, so this rewrites the unit,
    daemon-reloads and try-restarts it. Setting it before the service exists is
    fine: the value is on the record and the unit picks it up at install.
    """
    lab_guard.no_lab_edits()
    p, bot = pb
    ids = clean_ids(req.telegram_ids)
    entry = {**bot, "allowed_ids": ids}
    project_mod.save(replace(p, bots=tuple(
        entry if b.get("key") == bot["key"] else b for b in p.bots)))
    rewritten = bot_units.refresh(p, entry, _role_of(bot))
    _trail(request, user, audit_mod.BOT_ALLOWED, p, bot["key"],
           f"{len(ids)} id(s)" + (" — unit restarted" if rewritten else ""))
    print(f"[bots] {p.slug}/{bot['key']}: {len(ids)} allowed id(s)"
          f"{' (unit rewritten)' if rewritten else ''}", flush=True)
    return {"ok": True, "telegram_ids": ids, "unit_rewritten": rewritten,
            "note": "Restarted with the new list." if rewritten
                    else "Saved. It applies when you install the service."}


# ---------------------------------------------------------------- the token ---

class Token(BaseModel):
    token: str


@router.post("/api/bots/{key}/token")
def set_token(req: Token, request: Request,
              pb=Depends(owned_bot), user: str | None = Depends(current_user)):
    """Verify a token with Telegram, then store it in .env under this bot's var.

    VERIFY FIRST, ALWAYS. Storing an unverified token gives you a .env that says
    configured, a panel that says configured, and a service that crash-loops on
    401 — three things agreeing about something false. getMe costs one round
    trip and turns that into an error message at the moment of the mistake.

    The response carries the bot's @username, which is public and is also the
    thing you need to go and message it.
    """
    lab_guard.no_lab_edits()
    p, bot = pb
    var = (bot.get("token_env") or "").strip()
    if not var:
        raise HTTPException(400, f"{bot['key']!r} has no token variable — "
                                 "re-register it")
    _refuse_shared_var(p, bot, var)
    tok = (req.token or "").strip()
    if not _TOKEN.match(tok):
        raise HTTPException(422, "that does not look like a bot token — "
                                 "BotFather gives you 1234567890:AA... (paste "
                                 "just the token, not the whole message)")
    ok, info = _probe_raw(tok)
    if not ok:
        raise HTTPException(400, info)
    try:
        env_file.set_key(var, tok)
    except env_file.EnvError as e:
        raise HTTPException(422, str(e)) from None
    _trail(request, user, audit_mod.BOT_TOKEN, p, bot["key"],
           f"stored in {var} (@{info.get('username')})")
    print(f"[bots] {p.slug}/{bot['key']}: token stored in {var} "
          f"(@{info.get('username')})", flush=True)
    return {"ok": True, "username": info.get("username"),
            "name": info.get("name"), "id": info.get("id"),
            "note": "Stored in .env. Install or restart the service to use it."}


def _refuse_shared_var(p, bot: dict, var: str) -> None:
    """Refuse to write a variable another project's bot also claims.

    .env is ONE flat machine-global file while `register_bot` only ever checked
    `token_env` for uniqueness within a project — so a bot registered in your
    own project claiming TELEGRAM_BOT_TOKEN overwrote the owner's, and their
    service logged in as your bot at its next restart.

    New bots cannot reach this: the variable is derived from the project slug
    and role. It stays for the entries that predate that, and because a manifest
    is hand-editable.
    """
    for slug in project_mod.slugs():
        if slug == p.slug:
            continue
        try:
            other = project_mod.load(slug)
        except Exception:  # noqa: BLE001 — a broken manifest is not this bot
            continue
        for b in other.bots:
            if (b.get("token_env") or "").strip() == var:
                raise HTTPException(
                    409, f"{var} already holds another bot's token. Two bots "
                         f"cannot share one variable — .env has a single value "
                         f"for it, so writing here would take that bot over. "
                         f"Re-register this one to get a variable of its own.")


def _probe_raw(tok: str) -> tuple[bool, dict | str]:
    """getMe for a token we hold in hand (not yet stored). Never echoes it."""
    try:
        import httpx
        d = httpx.get(f"https://api.telegram.org/bot{tok}/getMe", timeout=12).json()
    except Exception as e:  # noqa: BLE001 — an httpx error's URL contains the token
        return False, f"could not reach Telegram ({type(e).__name__})"
    if not d.get("ok"):
        return False, str(d.get("description", "Telegram rejected it"))[:160]
    me = d.get("result", {})
    return True, {"username": me.get("username"), "name": me.get("first_name"),
                  "id": me.get("id")}


# --------------------------------------------------------------- the service ---

@router.post("/api/bots/{key}/install")
def install(request: Request, pb=Depends(owned_bot),
            user: str | None = Depends(current_user)):
    """Write this bot's systemd unit, reload, enable and start it.

    Everything in the unit comes from the role table and validated manifest
    fields — see web/bot_units.py. What the request chooses is which bot, and
    that bot's role was fixed when it was registered.

    REFUSES WITHOUT A TOKEN. A unit that starts, fails to authenticate and
    restarts every 5 seconds forever is worse than no unit: it buries the real
    error in a restart loop and the panel shows "activating" indefinitely.
    """
    lab_guard.no_lab_edits()
    # SESSION ONLY. This writes a unit file and starts it — arbitrary code as
    # the user who owns .env and private/. A key in a config file should not
    # reach that; registering, tokens and probes still work with one.
    refuse_key_auth(request, "install a service")
    p, bot = pb
    role = _role_of(bot)
    if role is None:
        raise HTTPException(400,
            f"{bot['key']!r} has no installable role. It was registered before "
            "roles existed, or its role is not one this build ships — "
            f"re-register it choosing one of: {', '.join(bot_roles.ROLES)}")
    if not env_file.has(bot.get("token_env") or ""):
        raise HTTPException(400, f"set the token first — {bot.get('token_env')} "
                                 "is empty, and the service would restart-loop")
    _needs_a_model(p, role)
    try:
        unit = bot_units.install(p, bot, role)
    except bot_units.UnitError as e:
        raise HTTPException(400 if "will not touch" in str(e) else 500,
                            str(e)) from None
    _trail(request, user, audit_mod.BOT_INSTALLED, p, bot["key"],
           f"{unit} -> {role.module}")
    print(f"[bots] {p.slug}/{bot['key']}: installed {unit} -> {role.module}",
          flush=True)
    return {"ok": True, "unit": unit,
            "path": str(bot_units.USER_UNITS / unit), "exec": role.exec_start}


def _needs_a_model(p, role) -> None:
    """A counting bot is the reward for having a model, not the way to get one.

    Without this the flow is: install, the bot starts with no counter, and every
    photo comes back "this project has no trained model yet" — which reads as a
    broken bot rather than as a step not done. Refusing here names the route
    through instead, and the route is entirely existing machinery.

    Only the COUNTING role. A capture bot collects reference photographs and
    needs no detector at all.
    """
    if role.key != "counter":
        return
    from ...dataset import paths as _paths
    from ...serve import runtime_model
    if runtime_model.list_runs(_paths.for_project(p)):
        return
    raise HTTPException(400,
        f"{p.name} has no trained model yet, so a counting bot would have "
        f"nothing to count with. Upload some images, auto-label them with "
        f"LocateAnything, check them, then train a run — come back and install "
        f"this bot after that.")


class Action(BaseModel):
    action: str


# reset-failed is here because without it a bot that crash-looped is
# UNRECOVERABLE from its own card: systemd's StartLimitBurst latches the unit
# into `failed`, and start/stop/restart all refuse until it is cleared. It
# grants no new authority — the unit was already validated by unit_of.
_ACTIONS = ("start", "stop", "restart", "reset-failed")


@router.post("/api/bots/{key}/service")
def service(req: Action, request: Request, pb=Depends(owned_bot),
            user: str | None = Depends(current_user)):
    """start / stop / restart / reset-failed this bot's unit. Nothing else's."""
    lab_guard.no_lab_edits()
    refuse_key_auth(request, "start or stop a service")
    p, bot = pb
    act = (req.action or "").strip()
    if act not in _ACTIONS:
        raise HTTPException(422, f"action must be one of: {', '.join(_ACTIONS)}")
    try:
        unit = bot_units.unit_of(p, bot)
    except bot_units.UnitError as e:
        raise HTTPException(400, str(e)) from None
    from .. import botsup
    if botsup.mode() == "supervisor":
        # Same verbs, supervisor-parented process. reset-failed maps to a
        # plain start — the supervisor holds no failure latch to clear.
        {"start": botsup.start, "stop": botsup.stop,
         "restart": botsup.restart,
         "reset-failed": botsup.start}[act](p, dict(bot))
    else:
        ok, msg = bot_units.systemctl(act, unit)
        if not ok:
            raise HTTPException(500, msg or f"{act} failed")
    _trail(request, user, audit_mod.BOT_SERVICE, p, bot["key"], f"{act} {unit}")
    return {"ok": True, "action": act, "unit": unit}


# ------------------------------------------------------------- which project ---

class Move(BaseModel):
    project: str


@router.post("/api/bots/{key}/project")
def move_project(req: Move, request: Request, pb=Depends(owned_bot),
                 user: str | None = Depends(current_user)):
    """Re-associate a bot with a different project.

    The bot ENTRY moves between manifests, and the unit is rewritten so
    GW_PROJECT matches — otherwise the cockpit would say the bot feeds
    project B while the running service kept writing into project A, which looks
    right on every screen and is wrong in the only place that counts.

    BOTH ENDS ARE CHECKED. `owned_bot` clears the source; the destination is
    checked here, before either save. Without that, owning the bot was enough to
    push it into somebody else's project — which is an image-injection primitive
    into their training set, not merely a leak. The two saves are sequential, so
    the check has to happen before the first one or a refusal leaves half a move.
    """
    lab_guard.no_lab_edits()
    src, bot = pb
    key = bot["key"]
    dst_slug = (req.project or "").strip()
    if dst_slug == src.slug:
        return {"ok": True, "project": dst_slug, "note": "already there"}
    try:
        dst = project_mod.load(dst_slug)
    except Exception:  # noqa: BLE001
        raise HTTPException(404, f"no project {dst_slug!r}") from None
    if not readable_by(dst, user):
        raise HTTPException(403, f"project {dst_slug!r} belongs to {dst.owner!r} "
                                 f"— you cannot move a bot into it")
    if any(b.get("key") == key or b.get("token_env") == bot.get("token_env")
           for b in dst.bots):
        raise HTTPException(409, f"{dst_slug} already has a bot using {key!r} "
                                 f"or {bot.get('token_env')}")
    role = _role_of(bot)
    ext = bot.get("extension") or getattr(role, "extension", None)
    if ext and not dst.has(ext):
        raise HTTPException(400, f"{dst_slug} does not enable the {ext!r} "
                                 f"extension, so this bot would have nowhere "
                                 f"to write")
    project_mod.save(replace(src, bots=tuple(b for b in src.bots
                                             if b.get("key") != key)))
    project_mod.save(replace(dst, bots=tuple(dst.bots) + (bot,)))
    rewritten = bot_units.refresh(dst, bot, role)
    _trail(request, user, audit_mod.BOT_MOVED, src, key,
           f"-> {dst.slug}" + (" — unit rewritten" if rewritten else ""))
    print(f"[bots] {key}: {src.slug} -> {dst.slug}"
          f"{' (unit rewritten)' if rewritten else ''}", flush=True)
    return {"ok": True, "project": dst.slug, "unit_rewritten": rewritten}
