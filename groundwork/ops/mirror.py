"""Push EVERY project's dataset to every verified worker. One job, not one per
project or per machine.

    python -m groundwork.ops.mirror                     # all projects x workers
    python -m groundwork.ops.mirror --dry-run           # show, transfer nothing
    python -m groundwork.ops.mirror --project <slug>
    python -m groundwork.ops.mirror --machine <key>

ONE JOB THAT ENUMERATES, NOT A UNIT PER PROJECT. Creating a project (or pairing
a machine) should not mean installing anything — that is a step to forget, and
the failure is silent (the project simply never syncs). The scheduler runs this
every 5 minutes; it asks which projects and which workers exist and does the
cross product. With no workers registered it prints one line and exits 0.

THE PATH RULE, and it is the same one for both shapes. rsync a source WITHOUT a
trailing slash and it transfers the directory itself into the destination, so
`<root>` goes to `<its parent, on the worker>` and lands at the identical
repo-relative path: `outputs/projects/<slug>/dataset` arrives at
`outputs/projects/<slug>/dataset` under the worker's own root.

THE EXCLUDES MUST STAY ANCHORED. `/dataset/images`, never `images/`. Unanchored,
it also matched `raw/images`, `testset/images` and friends, so the mirror
silently skipped the actual data and the worker trained on whatever it had
(cost a day once; docs/machines.md). Because each source is the `dataset`
directory itself, the anchor is the same string for every project.

THE MIRROR IS NOT A BACKUP. It rsyncs with `--delete`, so a worker's copy is a
CLONE: an image deleted on HQ is gone there within one tick. The backup job is
ops/backup.py, and it is a different thing on purpose.

BatchMode=yes is load-bearing (via ssh_identity): an interactive prompt would
leave rsync SITTING THERE waiting for a human while timer runs pile up.
`.last_sync.<machine>` is stamped only on success, so the cockpit's sync-age
warning surfaces a broken path within a tick.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from .. import project as project_mod
from ..config import ROOT
from ..dataset import paths

# See the module docstring. `runs`/`snapshots` are the worker's own to make;
# `data.yaml` and the split dirs are MACHINE-LOCAL — the worker splits for
# itself, and copying ours over would hand it a data.yaml pointing at our paths.
EXCLUDES = ["runs", "snapshots", "data.yaml", "/dataset/images", "/dataset/labels"]

RSYNC_TIMEOUT = 900


def _stat(out: str, label: str) -> int:
    m = re.search(rf"{re.escape(label)}: ([\d,]+)", out)
    return int(m.group(1).replace(",", "")) if m else 0


def _rel(p: Path) -> str:
    """A repo-relative path, or refuse. A dataset_root outside the repo cannot
    be mirrored to "the same place" on another machine, because there is no
    such place — say so rather than inventing a destination."""
    try:
        from ..config import DATA_DIR
        return p.relative_to(DATA_DIR).as_posix()
    except ValueError:
        raise SystemExit(f"{p} is outside {ROOT} — this mirror only carries "
                         f"datasets that live under the install root") from None


def _run_captured(cmd, *, timeout):
    """The default runner: capture text output, bounded. The `run` parameter's
    CONTRACT is exactly `run(cmd, timeout=) -> .returncode/.stdout/.stderr` —
    the web dispatcher injects safe_proc.run, whose signature is that minimal
    shape and nothing more. Passing subprocess.run's richer kwargs here once
    made the seam a lie: every check stubbed `run` with **kwargs, so the first
    real dispatch through safe_proc was the first time the signatures met.
    """
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def sync_one(slug: str, machine_key: str, dry: bool = False,
             run=_run_captured) -> dict:
    """Push one project's dataset to one worker. Returns what happened.

    {ok, slug, machine, sent, deleted, seconds, error} — `sent` counts regular
    files actually transferred, which is the number that answers "did the
    worker have this already?". rsync is incremental, so calling this before
    every remote training run costs a directory listing when nothing changed.

    `run` is injectable because the web dispatcher calls this from a REQUEST
    HANDLER through safe_proc.run (a wedged rsync must not park an anyio worker
    thread forever — the repo's standing subprocess rule); the scheduler's
    oneshot invocation uses the plain default.
    """
    from ..web import ssh_identity
    host, remote_root = ssh_identity.target(machine_key)
    ssh = ssh_identity.ssh_option_string(machine_key)

    p = project_mod.load(slug)
    pp = paths.for_project(p)
    src = pp.DATASET_DIR
    if not src.exists():
        print(f"  {slug} -> {machine_key}: {src} does not exist — nothing to send")
        return {"ok": True, "slug": slug, "machine": machine_key,
                "sent": 0, "deleted": 0, "note": "no dataset directory"}
    rel_parent = _rel(src.parent)
    dest = f"{host}:{remote_root}/{rel_parent}/"
    cmd = ["rsync", "-az", "--delete", "--stats", "--mkpath", "-e", ssh]
    for e in EXCLUDES:
        cmd += ["--exclude", e]
    cmd += [str(src), dest]

    labeled = len(list(pp.RAW_LABELS.glob("*.txt"))) if pp.RAW_LABELS.exists() else 0
    held = len(list(pp.TESTSET_LABELS.glob("*.txt"))) if pp.TESTSET_LABELS.exists() else 0
    # PRINT WHAT IS BEING SENT AND WHERE, before sending it. Silent wrong-data
    # is this repo's most expensive failure and a mirror is how data moves.
    print(f"  {slug} -> {machine_key}: {src} -> {dest}  "
          f"({labeled} training + {held} holdout)")
    if dry:
        print(f"    dry-run: {' '.join(cmd)}")
        return {"ok": True, "slug": slug, "machine": machine_key,
                "sent": 0, "deleted": 0, "dry": True}
    t0 = time.time()
    r = run(cmd, timeout=RSYNC_TIMEOUT)
    if r.returncode != 0:
        err = (r.stderr or "rsync failed").strip()[:400]
        print(f"    FAILED: {err}", file=sys.stderr)
        return {"ok": False, "slug": slug, "machine": machine_key, "error": err}
    sent = _stat(r.stdout, "Number of regular files transferred")
    deleted = r.stdout.count("deleting ")

    # THE MANIFEST TRAVELS TOO, and without it the project could not train
    # remotely at all. `src` is the project's DATASET dir; project.json lives
    # one level above it (outputs/projects/<slug>/project.json), so the rsync
    # above has never carried it. The worker's deps.project_slug refuses a slug
    # with no manifest — the data would arrive and the job could not name it.
    man = project_mod.manifest_path(slug)
    if man.exists():
        mdest = f"{host}:{remote_root}/{_rel(man.parent)}/"
        mr = run(["rsync", "-az", "--mkpath", "-e", ssh, str(man), mdest],
                 timeout=120)
        if mr.returncode != 0:
            # NOT FATAL TO THE DATASET SYNC that already succeeded, but it must
            # be said: a remote run for this project will fail to resolve it.
            print(f"    manifest NOT sent: {(mr.stderr or '').strip()[:200]}",
                  file=sys.stderr)
        else:
            sent += 1
    # Stamped ONLY on success, PER MACHINE — the cockpit reads this to warn
    # that a worker may be training on old data, and one stamp for the fleet
    # would let a healthy path mask a broken one.
    (src / f".last_sync.{machine_key}").write_text(str(int(time.time())),
                                                   encoding="utf-8")
    took = round(time.time() - t0, 1)
    print(f"    ok — {sent} file(s) sent in {took}s")
    return {"ok": True, "slug": slug, "machine": machine_key,
            "sent": sent, "deleted": deleted,
            "seconds": took, "labeled": labeled, "holdout": held}


def main() -> None:
    from ..web import machines as machines_mod
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=None,
                    help="one project slug (default: every project)")
    ap.add_argument("--machine", default=None,
                    help="one machine key (default: every verified worker)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.machine:
        targets = [args.machine]
    else:
        targets = list(machines_mod.workers())
    if not targets:
        print("[mirror] no workers registered — nothing to sync")
        raise SystemExit(0)
    slugs = [args.project] if args.project else project_mod.slugs()
    if not slugs:
        print("[mirror] no projects — nothing to sync")
        raise SystemExit(0)
    print(f"[mirror] {len(slugs)} project(s) -> {', '.join(targets)}")
    ok = True
    for key in targets:
        for s in slugs:
            try:
                ok = sync_one(s, key, args.dry_run).get("ok", False) and ok
            except Exception as e:  # noqa: BLE001 — one bad pair must not stop the rest
                print(f"  {s} -> {key}: {type(e).__name__}: {e}", file=sys.stderr)
                ok = False
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
