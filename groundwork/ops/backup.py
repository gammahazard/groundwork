"""ONE off-machine copy of the hand-made data, on the worker. Nothing else.

    .venv/bin/python -m groundwork.ops.backup              # every project
    .venv/bin/python -m groundwork.ops.backup --dry-run
    .venv/bin/python -m groundwork.ops.backup --project the first project

WHY THIS EXISTS, and why the mirror is not it. `the mirror job` rsyncs the dataset to
the worker with `--delete`, which makes the worker a CLONE, not a backup: delete an image
here and the next five-minute tick faithfully deletes it there. So the 174
hand-dotted training images and the frozen 68-image holdout — work that cannot be
regenerated from anything — effectively existed in one place, with a second copy
standing by to replicate their destruction. None of the six timers was a backup.

WHAT IT COPIES: images and labels for `raw/` (the training source) and
`testset/` (the frozen holdout). That is the irreplaceable set — 125 MB, against
5.4 GB for the whole dataset tree. Everything left out is reproducible:
`pending/` is an inbox, `synthetic/` is generated, the `preview/` dirs are
rendered from the labels, and `images/{train,val}` is a split that any machine
rebuilds from a seed. A backup that also carries those is mostly carrying things
it does not need to, and the bigger it is the more likely it is to be turned off.

TWENTY SNAPSHOTS, AND IT IS NEVER BRIEFLY ZERO. Each run writes a NEW timestamped
directory and prunes beyond the newest 20 only after the new one is verified —
file counts compared against the source, per collection. Pruning first would open
a window where a failed transfer leaves nothing at all, which is the one outcome
a backup exists to prevent.

TWENTY IS CHEAP BECAUSE OF `--link-dest`. Each copy hardlinks unchanged files to
the previous one, so a snapshot costs only what CHANGED since the last — 484
files at 126 MB for the first, and four new images for a tick that finds four new
images. Twenty is therefore roughly one copy plus the deltas, not twenty copies,
and it buys the thing a single copy cannot: a mistake you notice three days later
is still undoable, because the version from before it is still there.

A depth of 20 at 30 hours is about 25 days of history.

IT LIVES OUTSIDE `outputs/dataset` DELIBERATELY. `the mirror job` rsyncs that tree
with `--delete`; a backup stored inside it would be deleted by the mirror as an
extraneous file — the backup destroyed by the thing it exists to survive.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .. import project as project_mod
from ..config import OUTPUTS_DIR
from ..dataset import paths

# The destination is a REGISTERED MACHINE with `backup_target` switched on
# (Machines tab). Host, root and the ssh identity all come from the registry
# through web/ssh_identity — one resolution, shared with the mirror.
def _target(machine_key: str) -> tuple[list[str], str, str]:
    """(ssh_argv, user@host, backups_root) for the chosen machine."""
    from ..web import ssh_identity
    host, root = ssh_identity.target(machine_key)
    return ssh_identity.ssh_argv(machine_key), host, f"{root}/outputs/backup"

# How many timestamped snapshots to keep. Cheap because of --link-dest: the
# marginal cost of one more is the files that changed, not another full copy.
KEEP = 20

# Attempts before giving up, and the gap between them. 5 x 5min = ~25 minutes of
# patience, inside the unit's 45-minute TimeoutStartSec.
TRIES = 5
WAIT_S = 300.0

# (label, ProjectPaths attribute). Images AND labels — a label file without its
# image is not an image, and an image without its labels is unlabelled data.
PARTS = [
    ("raw/images", "RAW_IMAGES"),
    ("raw/labels", "RAW_LABELS"),
    ("testset/images", "TESTSET_IMAGES"),
    ("testset/labels", "TESTSET_LABELS"),
]


def _ssh(t, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    ssh_argv, host, _ = t
    return subprocess.run([*ssh_argv, host, *args],
                          capture_output=True, text=True, timeout=timeout)


def reachable(t) -> tuple[bool, str]:
    """Is the worker answering at all? Asked FIRST, and separately.

    The worker is off, asleep or behind a cold network path often enough that this
    has to be a normal outcome rather than an error — but it must be told apart
    from "answering, and has no backups", because those two demand opposite
    behaviour and produce the same empty output from `ls`.
    """
    try:
        r = _ssh(t, "echo ok", timeout=90)
    except Exception as e:  # noqa: BLE001 — a timeout IS unreachable
        return False, type(e).__name__
    if r.returncode != 0 or "ok" not in r.stdout:
        return False, (r.stderr or "no answer").strip().splitlines()[-1][:120] \
            if (r.stderr or "").strip() else "no answer"
    return True, ""


def _existing(t, slug: str) -> list[str] | None:
    """Backup directory names already on the worker, oldest first. None = COULD NOT ASK.

    The names are sortable by construction (YYYY-MM-DD_HHMM), so lexical order is
    chronological order and no parsing is needed to find the newest.

    None rather than [] when ssh fails, and the caller aborts on it. An empty list
    from a failed connection would read as "no backups yet", which has two
    consequences and both are wrong: the transfer loses its --link-dest and copies
    everything, and the prune sees one snapshot where there are twenty, so nothing
    is ever pruned again. "I could not see it" and "it is not there" are different
    answers — the rule this repo keeps relearning.
    """
    r = _ssh(t, f"ls -1 {t[2]}/{slug} 2>/dev/null; echo __rc$?__")
    if r.returncode != 0 or "__rc" not in r.stdout:
        return None
    body = r.stdout.split("__rc")[0]
    return sorted(x.strip() for x in body.splitlines() if x.strip())


def _remote_counts(t, slug: str, stamp: str) -> dict[str, int]:
    """How many files actually landed, counted ON the worker.

    Counted remotely on purpose: rsync's own summary says what it believes it
    sent, and the question a backup has to answer is what is THERE.
    """
    base = f"{t[2]}/{slug}/{stamp}"
    script = "; ".join(
        f'echo "{lbl} $(ls -1 {base}/{lbl} 2>/dev/null | wc -l)"' for lbl, _ in PARTS)
    r = _ssh(t, script)
    out: dict[str, int] = {}
    for line in r.stdout.splitlines():
        bits = line.split()
        if len(bits) == 2 and bits[1].isdigit():
            out[bits[0]] = int(bits[1])
    return out


def backup_one(slug: str, machine_key: str, dry: bool = False,
               keep: int = KEEP) -> dict:
    """Copy one project's hand-made data to the backup target."""
    t = _target(machine_key)
    p = project_mod.load(slug)
    pp = paths.for_project(p)
    stamp = time.strftime("%Y-%m-%d_%H%M")
    prev = _existing(t, slug)
    if prev is None:
        # NOT a full transfer, and NOT a prune. See _existing's docstring.
        print(f"  {slug}: could not list existing backups — target "
              f"unreachable. Nothing sent, nothing pruned.", file=sys.stderr)
        return {"ok": False, "slug": slug, "error": "backup target unreachable",
                "unreachable": True}
    link = prev[-1] if prev else None

    want = {}
    for lbl, attr in PARTS:
        d = getattr(pp, attr)
        # NOT dotfiles: the remote count is `ls -1`, which hides them, so one
        # .DS_Store on this side made verification fail forever and no backup
        # was ever accepted. Both sides count the same set now.
        want[lbl] = (len([f for f in d.iterdir()
                          if f.is_file() and not f.name.startswith(".")])
                     if d.exists() else 0)
    total = sum(want.values())
    # SAY WHAT IS MOVING AND WHERE BEFORE MOVING IT — the same rule the mirror
    # follows, for the same reason: silent wrong-data is the expensive failure.
    print(f"  {slug}: {total} file(s) -> {t[1]}:{t[2]}/{slug}/{stamp}")
    print(f"    {', '.join(f'{k} {v}' for k, v in want.items())}")
    print(f"    link-dest: {link or '(none — first backup, full transfer)'}")
    if total == 0:
        # Refusing is the safe answer: proceeding would write an empty backup and
        # then delete the previous good one.
        print("    REFUSED: nothing to back up", file=sys.stderr)
        return {"ok": False, "slug": slug, "error": "no source files"}
    if dry:
        return {"ok": True, "slug": slug, "dry": True, "stamp": stamp, "want": want}

    t0 = time.time()
    for lbl, attr in PARTS:
        src = getattr(pp, attr)
        if not src.exists():
            continue
        dest = f"{t[1]}:{t[2]}/{slug}/{stamp}/{lbl}/"
        cmd = ["rsync", "-a", "--mkpath", "-e", " ".join(t[0])]
        if link:
            # RELATIVE to the destination directory, which is what rsync expects.
            cmd += ["--link-dest", f"../../../{link}/{lbl}/"]
        cmd += [f"{src}/", dest]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            err = (r.stderr or "rsync failed").strip()[:400]
            print(f"    FAILED on {lbl}: {err}", file=sys.stderr)
            # The previous backup is NOT touched — a half-written new copy is
            # worthless, but the old good one is not.
            return {"ok": False, "slug": slug, "error": err, "stamp": stamp}

    got = _remote_counts(t, slug, stamp)
    bad = {k: (want[k], got.get(k, 0)) for k, _ in PARTS if got.get(k, 0) != want[k]}
    if bad:
        print(f"    VERIFY FAILED (want vs got): {bad}", file=sys.stderr)
        print("    keeping the previous backup", file=sys.stderr)
        return {"ok": False, "slug": slug, "error": f"count mismatch {bad}",
                "stamp": stamp}

    took = round(time.time() - t0, 1)
    manifest = {"project": slug, "taken": stamp, "epoch": int(time.time()),
                "source": "HQ", "counts": want, "seconds": took,
                "note": "raw + testset images and labels only — the hand-made "
                        "data. Everything else in the dataset is reproducible."}
    _ssh(t, f"cat > {t[2]}/{slug}/{stamp}/MANIFEST.json << 'EOF'\n"
         f"{json.dumps(manifest, indent=1)}\nEOF")

    # ONLY NOW is anything pruned, and only past the newest `keep`. Names are
    # YYYY-MM-DD_HHMM, so lexical order IS chronological and the oldest are the
    # front of the list. The new stamp is skipped explicitly rather than trusted
    # to sort last — a clock that has stepped backwards would otherwise make this
    # delete the copy it just wrote.
    snaps = sorted(set(prev) | {stamp})
    doomed = [d for d in snaps[:-keep] if d != stamp] if len(snaps) > keep else []
    removed = []
    for old in doomed:
        r = _ssh(t, f"rm -rf {t[2]}/{slug}/{old}")
        if r.returncode == 0:
            removed.append(old)
    # STAMPED LOCALLY, and only on success — the same rule mirror.py follows for
    # .last_sync. It makes "when did this last work" answerable from HQ alone,
    # which matters precisely when the worker is the thing that is unreachable.
    try:
        (OUTPUTS_DIR / ".last_backup").write_text(
            json.dumps({"epoch": int(time.time()), "stamp": stamp,
                        "project": slug, "files": total}), encoding="utf-8")
    except OSError:
        pass
    kept = len(snaps) - len(removed)
    print(f"    ok — {total} file(s) verified in {took}s · {kept} snapshot(s) kept"
          + (f", pruned {len(removed)} beyond {keep}" if removed else ""))
    return {"ok": True, "slug": slug, "stamp": stamp, "counts": want,
            "seconds": took, "pruned": removed, "kept": kept}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=None,
                    help="one project slug (default: every project)")
    ap.add_argument("--machine", default=None,
                    help="one machine key (default: every backup_target worker)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", type=int, default=KEEP,
                    help=f"timestamped snapshots to retain (default {KEEP})")
    ap.add_argument("--tries", type=int, default=TRIES,
                    help=f"attempts before giving up (default {TRIES})")
    ap.add_argument("--wait", type=float, default=WAIT_S,
                    help=f"seconds between attempts (default {WAIT_S:g})")
    args = ap.parse_args()
    from ..web import machines as machines_mod
    if args.machine:
        targets = [args.machine]
    else:
        targets = [k for k, r in
                   ((k, machines_mod.registered(k)) for k in machines_mod.workers())
                   if r.get("backup_target")]
    if not targets:
        # OFF BY DEFAULT is the policy: an HQ->worker backup presumes that
        # machine is durable and trusted with the irreplaceable set, which is
        # the operator's call (the Machines tab toggle), not a platform default.
        print("[backup] no machine is marked as a backup target — nothing to do "
              "(enable one on the Machines tab)")
        raise SystemExit(0)
    slugs = [args.project] if args.project else project_mod.slugs()
    print(f"[backup] {len(slugs)} project(s) -> {', '.join(targets)}")

    ok = True
    for key in targets:
        t = _target(key)
        # WAIT FOR THE TARGET RATHER THAN SKIPPING A CYCLE. It may be a desktop
        # on a domestic network: asleep, rebooting, or on a path that takes a
        # minute to come up. One failed attempt would otherwise cost THIRTY
        # HOURS of coverage, which is the difference between a backup and a
        # lottery. Bounded by the unit's TimeoutStartSec, so a machine that is
        # genuinely off still frees the timer rather than holding it.
        up = args.dry_run
        for attempt in range(1, max(1, args.tries) + 1):
            if args.dry_run:
                break
            up, why = reachable(t)
            if up:
                if attempt > 1:
                    print(f"  {key} reachable on attempt {attempt}")
                break
            if attempt == args.tries:
                print(f"[backup] {key} unreachable after {attempt} attempt(s) "
                      f"({why}) — nothing sent, existing snapshots untouched",
                      file=sys.stderr)
                ok = False
                break
            print(f"  {key} unreachable ({why}) — attempt {attempt}/{args.tries}, "
                  f"retrying in {args.wait:g}s", file=sys.stderr)
            time.sleep(args.wait)
        if not up and not args.dry_run:
            continue
        for s in slugs:
            try:
                ok = backup_one(s, key, args.dry_run, args.keep).get("ok", False) and ok
            except Exception as e:  # noqa: BLE001 — one project must not stop the rest
                print(f"  {s}: {type(e).__name__}: {e}", file=sys.stderr)
                ok = False
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
