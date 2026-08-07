"""Render + enable the systemd user units — the native deployment's installer.

Templates live in deploy/units/. The web unit ships KillMode=process BAKED IN
(without it, a service restart kills any detached training run — the single
most expensive operational lesson this platform carries). The timers are
RENDERED FROM the scheduler's job table (groundwork/ops/scheduler.py), so an
interval changed there changes systemd and Docker alike.

Idempotent by construction: rendering is deterministic, and a unit whose text
is unchanged is not rewritten — `groundwork install` after `git pull` is a
safe "upgrade units" step whose diff means something.

Values interpolated into units are refused rather than escaped when they look
wrong — the same rule the bot-unit writer enforces, for the same reason: a
unit file is something systemd executes.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .config import ROOT

UNITS_SRC = ROOT / "deploy" / "units"
USER_UNITS = Path.home() / ".config" / "systemd" / "user"

# Same shapes bot_roles enforces: nothing that can smuggle a directive.
# `=` is allowed because whole directives render through here
# (OnCalendar=daily); NEWLINES stay banned — a value containing one would
# append arbitrary directives to a file systemd executes, which is the exact
# injection the bot-unit writer was caught by.
_SAFE_PATH = re.compile(r"^[A-Za-z0-9/._~+=-]+$")


def _render(tmpl: str, subs: dict[str, str]) -> str:
    out = tmpl
    for k, v in subs.items():
        if not _SAFE_PATH.match(v.replace(" ", "")):
            raise SystemExit(f"[install] refusing to write {k}={v!r} into a "
                             f"unit — it does not look like a plain value")
        out = out.replace("{{" + k + "}}", v)
    if "{{" in out:
        missing = sorted(set(re.findall(r"{{(\w+)}}", out)))
        raise SystemExit(f"[install] template placeholders unfilled: {missing}")
    return out


def _write_if_changed(dest: Path, body: str) -> bool:
    if dest.exists() and dest.read_text(encoding="utf-8") == body:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")
    return True


def _systemctl(*args: str) -> tuple[bool, str]:
    r = subprocess.run(["systemctl", "--user", *args],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def install(role: str | None = None, units_only: bool = False) -> int:
    from .ops.scheduler import JOBS
    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable
    subs = {"ROOT": str(ROOT), "PYTHON": py}

    changed = []

    web_tmpl = (UNITS_SRC / "groundwork-web.service.tmpl")
    if not web_tmpl.exists():
        raise SystemExit(f"[install] {web_tmpl} is missing — run from a "
                         f"complete checkout")
    body = _render(web_tmpl.read_text(encoding="utf-8"), subs)
    if role:
        body = body.replace("[Service]", f"[Service]\nEnvironment=GW_ROLE={role}", 1)
    if _write_if_changed(USER_UNITS / "groundwork-web.service", body):
        changed.append("groundwork-web.service")

    job_s = (UNITS_SRC / "groundwork-job.service.tmpl").read_text(encoding="utf-8")
    job_t = (UNITS_SRC / "groundwork-job.timer.tmpl").read_text(encoding="utf-8")
    for job in JOBS:
        # Role-scoped jobs are still INSTALLED — their predicates self-skip at
        # run time, which keeps one source of truth for "should this run".
        name = f"groundwork-{job.name}"
        svc = _render(job_s, {**subs, "JOB": name,
                              "ARGS": " ".join(("-m", *job.argv)),
                              "TIMEOUT": str(job.timeout_s)})
        every = job.every_s
        # One directive per placeholder — _render refuses newlines in values,
        # and that refusal is load-bearing (see the timer template's comment).
        # OnUnitActiveSec for ANY interval — the old `>= a day means daily`
        # branch silently turned the 30-hour backup into a 24-hour one, so the
        # rendered timers and the Docker scheduler disagreed about the schedule
        # the job table states. Persistent catches up a long interval missed
        # while the machine was off.
        s1, s2 = (f"OnUnitActiveSec={every}s",
                  "OnBootSec=120s" if every < 86400 else "Persistent=true")
        tim = _render(job_t, {**subs, "JOB": name,
                              "SCHEDULE_1": s1, "SCHEDULE_2": s2})
        if _write_if_changed(USER_UNITS / f"{name}.service", svc):
            changed.append(f"{name}.service")
        if _write_if_changed(USER_UNITS / f"{name}.timer", tim):
            changed.append(f"{name}.timer")

    if changed:
        ok, msg = _systemctl("daemon-reload")
        if not ok:
            raise SystemExit(f"[install] daemon-reload failed: {msg}")
        print(f"[install] wrote {len(changed)} unit(s): {', '.join(changed)}")
    else:
        print("[install] units already current")

    if not units_only:
        ok, msg = _systemctl("enable", "--now", "groundwork-web.service")
        if not ok:
            raise SystemExit(f"[install] could not enable the web service: {msg}")
        for job in JOBS:
            _systemctl("enable", "--now", f"groundwork-{job.name}.timer")
        # linger: user units die at logout without it — a headless box would
        # stop serving the moment the installing session closed.
        subprocess.run(["loginctl", "enable-linger"], capture_output=True)
        print("[install] enabled. Open the cockpit and the setup wizard "
              "takes it from there.")
    return 0


def main(argv=None) -> int:
    """`python -m groundwork.install` — the form scripts/install.sh execs.

    THIS DID NOT EXIST. The module had no entry point at all, so the native
    installer's final line imported it, ran nothing, printed nothing and
    exited 0: every native install ended looking successful with no units
    written and no service to survive a reboot. The console script
    (`groundwork install`) called install() correctly the whole time, which is
    why this was invisible — the two doors did not do the same thing.
    """
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m groundwork.install",
        description="Write and enable this machine's systemd user units.")
    ap.add_argument("--role", choices=["hq", "worker"], default=None,
                    help="bake GW_ROLE into the web unit")
    ap.add_argument("--units-only", action="store_true",
                    help="render units without enabling or starting anything")
    ap.add_argument("--native", action="store_true",
                    help=argparse.SUPPRESS)   # scripts/install.sh passes it
    a = ap.parse_args(argv)
    return install(role=a.role, units_only=a.units_only)


if __name__ == "__main__":
    raise SystemExit(main())
