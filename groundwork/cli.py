"""The `groundwork` command — install, run, and small operational verbs.

    groundwork install [--role hq|worker] [--units-only]
    groundwork run
    groundwork version

`install` renders the systemd user units from deploy/units/*.tmpl (the
scheduler's job table is the source of truth for the timers) and enables
them. `run` is the no-systemd fallback: foreground web service with the bot
supervisor and the inline scheduler — for trying Groundwork out, or WSL
without a user manager; a training run does not survive Ctrl-C here, and the
command says so.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="groundwork", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ins = sub.add_parser("install", help="install systemd user units + timers")
    ins.add_argument("--role", choices=["hq", "worker"], default=None,
                     help="write GW_ROLE into the unit environment")
    ins.add_argument("--units-only", action="store_true",
                     help="re-render units without enabling anything new")
    ins.add_argument("--native", action="store_true",
                     help=argparse.SUPPRESS)   # scripts/install.sh hand-off

    sub.add_parser("run", help="foreground server: supervisor bots + inline "
                               "scheduler (no systemd needed)")
    sub.add_parser("version", help="print the version")

    args = ap.parse_args(argv)

    if args.cmd == "version":
        try:
            from importlib.metadata import version
            print(version("groundwork"))
        except Exception:  # noqa: BLE001
            print("0.1.0 (source checkout)")
        return 0

    if args.cmd == "run":
        os.environ.setdefault("GW_PROCESS_MODEL", "supervisor")
        os.environ.setdefault("GW_SCHEDULER", "inline")
        print("[groundwork] foreground mode: bots supervised in-process, "
              "scheduler inline. A training run will NOT survive Ctrl-C — "
              "use `groundwork install` (systemd) or Docker for that.")
        from groundwork.web.app import main as web_main
        web_main()
        return 0

    if args.cmd == "install":
        from groundwork.install import install
        return install(role=args.role, units_only=args.units_only)

    return 1


if __name__ == "__main__":
    sys.exit(main())
