"""What `groundwork install` deploys must match the code that defines it.

    .venv/bin/python checks/check_deploy_parity.py

TWO PARITIES, each with a recorded cost behind it:

  KillMode=process in the web unit. systemd stops a unit by killing its whole
  cgroup, and a detached training run still lives in that cgroup — setsid does
  not leave it. Without this line a routine cockpit restart kills the run
  (measured on the predecessor: a 250-epoch run died 24 minutes in, 6 seconds
  after a restart). The installer must ship it BAKED IN, not as an optional
  drop-in someone remembers on the third machine.

  Timer intervals rendered FROM the scheduler's job table. The table
  (groundwork/ops/scheduler.py JOBS) is the single definition of what runs
  when; the systemd timers are only its rendered shape. If the installer
  hardcoded an interval, changing the table would change Docker installs and
  silently NOT native ones — two machines disagreeing about "every 5 minutes"
  is the drift this platform treats as its most expensive failure shape.

TWO BUGS THIS CHECK CAUGHT ARE NOW ASSERTED FIXED, not worked around:
_render once refused the very schedule values install() builds (they carry
'='), and the job template put the unit NAME after -m where a module belongs,
so no rendered job unit could start. The render below goes through the REAL
validator — no widening — and the ExecStart assertion is hard. The validator's
newline refusal is also proven still alive, because '=' being admitted must
never quietly admit the injection the rule exists for.

HOW IT CAN FAIL: part 3 MUTATES the job table and requires the rendered timer
to move with it — an installer that renders today's numbers from anywhere but
the table stays red under the mutation even though every current file looks
right. Deleting KillMode from the template reddens part 1; a job whose module
stops appearing in its rendered ExecStart reddens part 2.

SAFE: everything renders into a temp directory — USER_UNITS is redirected and
systemctl is stubbed before install() runs, so the machine's real user units
are never written, reloaded or enabled. No GPU, no network, no ssh.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


from groundwork import install as install_mod          # noqa: E402
from groundwork.ops import scheduler                   # noqa: E402


def _sandboxed_install(units_dir: Path) -> list[str]:
    """Run install(units_only=True) with units + systemctl sandboxed.

    Returns the systemctl calls it attempted, so the check can assert nothing
    beyond a daemon-reload was ever tried.
    """
    calls: list[str] = []

    def fake_systemctl(*args: str):
        calls.append(" ".join(args))
        return True, ""

    saved_units, saved_ctl = install_mod.USER_UNITS, install_mod._systemctl
    install_mod.USER_UNITS = units_dir
    install_mod._systemctl = fake_systemctl
    try:
        install_mod.install(units_only=True)
    finally:
        install_mod.USER_UNITS = saved_units
        install_mod._systemctl = saved_ctl
    return calls


def main() -> int:
    # The validator must accept the directives install() builds AND still
    # refuse the newline injection it exists for — both arms, so neither can
    # rot silently.
    note(install_mod._render("x {{V}}", {"V": "OnCalendar=daily"}) == "x OnCalendar=daily",
         "_render accepts a whole directive as a value ('=' admitted)")
    try:
        install_mod._render("x {{V}}", {"V": "a\nExecStart=/bin/evil"})
        note(False, "_render REFUSES a value containing a newline")
    except SystemExit:
        note(True, "_render REFUSES a value containing a newline")

    tmp = Path(tempfile.mkdtemp(prefix="checkdeploy-"))
    try:
        # ---- 1. KillMode=process ships in the template AND the render -------
        print("\n1. the web unit protects a detached run from its own restart")
        tmpl = ROOT / "deploy" / "units" / "groundwork-web.service.tmpl"
        note(tmpl.exists(), f"web template exists ({tmpl.name})")
        text = tmpl.read_text(encoding="utf-8") if tmpl.exists() else ""
        note("KillMode=process" in text,
             "KillMode=process is in the template — baked in, not a drop-in")

        calls = _sandboxed_install(tmp)
        rendered = tmp / "groundwork-web.service"
        note(rendered.exists(), "install() rendered groundwork-web.service")
        body = rendered.read_text(encoding="utf-8") if rendered.exists() else ""
        note("KillMode=process" in body, "…and the rendered unit carries it")
        note("{{" not in body, "…with no unfilled placeholder")
        note(all(c.startswith("daemon-reload") for c in calls),
             f"units_only touched systemd only to daemon-reload ({calls})")

        # ---- 2. every job in the table got a service + timer pair -----------
        print("\n2. the timers are the job table's systemd shape")
        for job in scheduler.JOBS:
            svc = tmp / f"groundwork-{job.name}.service"
            tim = tmp / f"groundwork-{job.name}.timer"
            note(svc.exists() and tim.exists(),
                 f"{job.name}: service and timer both rendered")
            if not (svc.exists() and tim.exists()):
                continue
            t = tim.read_text(encoding="utf-8")
            if job.every_s < 86400:
                want = f"OnUnitActiveSec={job.every_s}s"
                note(want in t,
                     f"{job.name}: interval matches the table ({want})")
                note("OnBootSec=" in t,
                     f"{job.name}: has a boot-time first run")
            else:
                note("OnCalendar=" in t,
                     f"{job.name}: a day-or-longer job renders as OnCalendar")
            s = svc.read_text(encoding="utf-8")
            note(f"TimeoutStartSec={job.timeout_s}" in s,
                 f"{job.name}: the table's timeout bounds the unit "
                 f"({job.timeout_s}s) — a wedged job must die, not pile up")
            note(job.argv[0] in s,
                 f"{job.name}: the table's module is in the ExecStart")
            note(f"-m groundwork-{job.name}" not in s,
                 f"{job.name}: ExecStart carries only the MODULE — the unit "
                 f"name after -m is the render bug that made every job unit "
                 f"unable to start")

        # ---- 3. THE MUTATION: the interval must FOLLOW the table ------------
        # install() imports JOBS from the scheduler module at call time, so
        # patching the table and re-rendering proves the number flows from it.
        # An installer with the interval hardcoded passes part 2 today and
        # fails here.
        print("\n3. mutate the table — the rendered timer must move")
        probe = scheduler.JOBS[0]
        mutated = types.SimpleNamespace(**{**probe.__dict__,
                                           "every_s": probe.every_s + 7717})
        saved_jobs = scheduler.JOBS
        tmp2 = Path(tempfile.mkdtemp(prefix="checkdeploy2-"))
        try:
            scheduler.JOBS = (mutated,)
            _sandboxed_install(tmp2)
            t = (tmp2 / f"groundwork-{probe.name}.timer").read_text(encoding="utf-8")
            note(f"OnUnitActiveSec={probe.every_s + 7717}s" in t,
                 f"changing the table changed the render "
                 f"({probe.name}: {probe.every_s}s -> {probe.every_s + 7717}s)")
            note(f"OnUnitActiveSec={probe.every_s}s" not in t,
                 "and the old interval is gone — not merely appended to")
        finally:
            scheduler.JOBS = saved_jobs
            shutil.rmtree(tmp2, ignore_errors=True)

        # ---- 4. idempotence: an unchanged render is not rewritten -----------
        print("\n4. reinstalling an unchanged tree rewrites nothing")
        stamps = {p.name: p.stat().st_mtime_ns for p in tmp.glob("groundwork-*")}
        _sandboxed_install(tmp)
        after = {p.name: p.stat().st_mtime_ns for p in tmp.glob("groundwork-*")}
        note(stamps == after,
             "second install() left every unit byte- and mtime-identical, "
             "so `groundwork install` after `git pull` is a safe upgrade step")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("PASS — units carry KillMode=process, and the timers are rendered "
          "from the scheduler's table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
