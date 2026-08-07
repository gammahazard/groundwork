"""What a bot can BE — the closed set of jobs the cockpit will install a service for.

THIS FILE IS THE SECURITY BOUNDARY, and it is the only reason writing a systemd
unit from a web request is defensible at all.

The dangerous version of "install this bot as a service" takes a command from
the request and runs it. This takes a ROLE NAME, looks it up here, and builds
the ExecStart from a module path that is a literal in this file. A request can
choose WHICH of two jobs to run and nothing else — not the interpreter, not the
module, not an argument. Adding a third job is a code change and a commit, which
is exactly the review step a request must never be able to skip.

WHAT A ROLE OWNS:

    module      the `python -m` target. A literal, never interpolated from input.
                Moving one of these breaks a string no import checker sees —
                same trap as web/retrain_job.py::_STAGE.
    extension   the role is only offered to projects that enable it (the
                mechanism ships with zero built-in extensions; it exists so a
                deployment-specific role can be scoped to the projects that
                carry its tooling).
    project_env the unit exports GW_PROJECT so the bot writes into the
                project it was installed FOR. Without it every bot on the box
                would feed one fixed project's dataset, which is this repo's
                most expensive failure mode wearing a new hat.

THE BOUNDARY IS THE FILE, NOT THE ExecStart LINE — corrected 2026-08-05, after
this docstring's claim was demonstrated false. See the comment above unit_text:
a bot NAME containing newlines was enough to append arbitrary directives to a
unit systemd then executed. Everything written into a unit is validated now, and
refused rather than escaped.

The unit template is deliberately close to the hand-written counting-bot unit that
has been running for months — same Restart policy, same WorkingDirectory, same
network-online ordering. A generated unit that behaves differently from the one
you already trust is a second thing to debug.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import ROOT


@dataclass(frozen=True)
class Role:
    key: str
    name: str
    what: str
    module: str                 # LITERAL. Never built from a request.
    extension: str | None = None
    token_hint: str = "BOT_TOKEN"
    # MAY A NEW BOT BE REGISTERED FOR THIS JOB? Separate from `extension`,
    # which asks whether a project has the tooling — this asks whether we are
    # offering the job at all. An existing bot with this role keeps working;
    # only the "add a bot" form stops listing it, so turning a role off is
    # this one field.
    offerable: bool = True

    @property
    def exec_start(self) -> str:
        # The conventional venv when it exists; otherwise the interpreter this
        # service runs on (pip installs, Docker) — a unit pointing at a python
        # that is not there fails on start with the least helpful error systemd
        # can produce.
        py = ROOT / ".venv" / "bin" / "python"
        if not py.exists():
            import sys
            py = sys.executable
        return f"{py} -m {self.module}"


ROLES: dict[str, Role] = {
    "counter": Role(
        key="counter", name="Collection / counting bot",
        what="Photograph a scene. With a trained model it replies with a "
             "count and an overlay to confirm or correct; before one exists, "
             "every photo lands in the fix queue for labeling — either way, "
             "training images arrive continuously.",
        module="groundwork.bots.telegram", token_hint="BOT_TOKEN"),
}


def get(key: str) -> Role | None:
    return ROLES.get((key or "").strip())


def for_project(p) -> list[Role]:
    """The roles this project may register a NEW bot for.

    Two gates, and they answer different questions: `offerable` is whether the
    job is on offer at all, `extension` is whether this project has the tooling
    for it. A role already held by an existing bot is unaffected — `get()` still
    resolves it, so that bot installs, moves and restarts as before.
    """
    return [r for r in ROLES.values()
            if r.offerable and (not r.extension or p.has(r.extension))]


# A UNIT FILE IS LINE-BASED, WHICH IS THE WHOLE VULNERABILITY. Until 2026-08-05
# this template interpolated a description built from `p.name` and `bot["name"]`,
# and `register_bot` checked only that the name was non-empty — no charset, no
# length, no newline check. So a bot named
#
#     MyBot\n\n[Service]\nExecStartPre=/bin/sh -c '…'\n
#
# produced a unit systemd ACCEPTS (`systemd-analyze verify` exits 0; it warns
# about the orphaned fragment and runs the ExecStartPre) executing as the user
# who owns .env and private/. Four ordinary requests, no Telegram account, no
# admin. The docstring above was true of ExecStart and false of the file.
#
# So: the boundary is the FILE, not the ExecStart line. Every value that reaches
# it is checked here, and REFUSED rather than escaped — the same choice
# env_file.py makes, and for the same reason: an escaper is a thing that can
# have a bug, and nothing legitimate needs a control character in a bot name.
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
# No whitespace, no quotes, no '#'. systemd would need quoting for a value with
# spaces, and a value that needs quoting is a value we should not be writing.
_ENV_VAL = re.compile(r"^[A-Za-z0-9_,.:@/=+-]{1,200}$")

DESCRIPTION_MAX = 180


def clean_description(raw: str) -> str:
    """A Description line that cannot become another directive.

    Control characters are COLLAPSED to spaces rather than refused, because
    unlike an environment value this string is cosmetic and a project called
    "Eli's\tPills" should not be unable to have a bot. The newline is the thing
    that mattered and a space is a faithful rendering of it.
    """
    return _CTRL.sub(" ", raw or "")[:DESCRIPTION_MAX].strip() or "Groundwork bot"


def unit_text(role: Role, description: str, env: dict[str, str]) -> str:
    """The systemd unit for this role, with a validated Environment block.

    `env` replaces the old `slug` argument: a bot now carries its project, its
    token variable, its log and its owner ids, and four hand-interpolated lines
    is four chances to write the one that is not checked. One dict, one gate.

    RAISES rather than sanitising. The caller (web/bot_units.py) validates too —
    two independent checks in two modules, deliberately, because this is the file
    whose docstring claims to be the security boundary.

    KillMode is NOT set: unlike the web service, a bot has no detached child to protect,
    and the default (control-group) is what you want for a poller.
    """
    if _CTRL.search(description or ""):
        raise ValueError("refusing a control character in a unit Description")
    for k, v in env.items():
        if not _ENV_KEY.match(k):
            raise ValueError(f"refusing {k!r} as an environment variable name")
        if not _ENV_VAL.match(str(v)):
            raise ValueError(f"refusing the value given for {k} — it must be "
                             f"printable, unquoted and space-free")
    # Sorted so reinstalling an unchanged bot rewrites a byte-identical file,
    # which is what makes `diff` a useful check after an upgrade.
    block = "".join(f"Environment={k}={v}\n" for k, v in sorted(env.items()))
    return f"""# Written by Groundwork (cockpit → Data collection). Safe to edit by hand;
# reinstalling from the cockpit overwrites it.
[Unit]
Description={description}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=simple
WorkingDirectory={ROOT}
{block}ExecStart={role.exec_start}
Restart=always
RestartSec=5
# The counting bot loads a model before it is "started".
TimeoutStartSec=120

[Install]
WantedBy=default.target
"""
