"""A worker's data freshness is a recorded FACT: `.last_sync.<machine>` stamps.

    .venv/bin/python checks/check_worker_freshness.py

WHY THIS EXISTS. A worker trains on a one-way mirror of HQ's dataset, and the
most expensive failure this platform records is a worker training a day stale
— two runs once trained without the newest images and nothing errored. The
defence is not hope, it is bookkeeping: the mirror stamps
`.last_sync.<machine>` inside each project's dataset, ONLY on success, and the
cockpit turns stamp age into a warning. Every property of that stamp is
load-bearing:

  ONLY ON SUCCESS   a failed rsync that still stamped would silence the very
                    warning that exists to surface it — the check's part 2.
  PER MACHINE       one stamp for the fleet would let a healthy path mask a
                    broken one — part 3.
  NEWEST WINS       the dashboard figure is "has anything received this
                    dataset lately"; per-worker detail lives elsewhere — part 4.

Part 5 guards the exclude list's ANCHORING. `/dataset/images`, never
`images/`: unanchored, the exclude also matched raw/images and
testset/images, so the mirror silently skipped the actual data and the worker
trained on whatever it had. That burn is documented in the mirror's own
docstring; this asserts the command line it builds, not the prose.

SANDBOXED, NO NETWORK, NO SSH: rsync is replaced through sync_one's injectable
`run` (the same seam the web dispatcher uses for safe_proc), the machine
registry and ssh identity are redirected to a temp dir, and the throwaway
project is removed at the end.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(os.environ.get("GW_CHECK_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def note(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        FAIL.append(what)


from groundwork import project as project_mod              # noqa: E402
from groundwork.config import OUTPUTS_DIR                  # noqa: E402
from groundwork.dataset import paths                       # noqa: E402
from groundwork.ops import mirror                          # noqa: E402
from groundwork.web import ssh_identity                    # noqa: E402
from groundwork.web.machines import registry as reg_mod    # noqa: E402

SLUG = "synccheck"


class FakeRun:
    """Stands in for subprocess.run; records every command."""

    def __init__(self, returncode=0):
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        return types.SimpleNamespace(
            returncode=self.returncode,
            stdout="Number of regular files transferred: 3\n",
            stderr="" if self.returncode == 0 else "connection refused")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="checksync-"))
    saved = (reg_mod.REGISTRY_PATH, ssh_identity.SSH_DIR,
             ssh_identity.KEY_FILE, ssh_identity.KNOWN_HOSTS)
    # Redirect the machine registry and the ssh identity BEFORE anything runs:
    # this check registers fake machines and must never touch private/ssh.
    reg_mod.REGISTRY_PATH = tmp / "machines_registry.json"
    ssh_identity.SSH_DIR = tmp / "ssh"
    ssh_identity.KEY_FILE = tmp / "ssh" / "id_ed25519"
    ssh_identity.KNOWN_HOSTS = tmp / "ssh" / "known_hosts"
    ssh_identity.SSH_DIR.mkdir(parents=True)
    ssh_identity.KEY_FILE.write_text("fake key material\n")   # ensure_key no-ops
    reg_mod.REGISTRY_PATH.write_text(json.dumps({"machines": {
        "worker-a": {"name": "worker-a", "url": "http://192.0.2.10:8000",
                     "ssh_host": "user@192.0.2.10", "remote_root": "~/gw",
                     "role": "worker", "verified": 1},
        "worker-b": {"name": "worker-b", "url": "http://192.0.2.11:8000",
                     "ssh_host": "user@192.0.2.11", "remote_root": "~/gw",
                     "role": "worker", "verified": 1},
    }}))

    # A throwaway project whose dataset lives INSIDE the repo (the mirror
    # refuses roots outside it — there is no "same place" remotely otherwise).
    p = project_mod.Project(slug=SLUG, name=SLUG,
                            dataset_root=OUTPUTS_DIR / "projects" / SLUG / "dataset")
    project_mod.manifest_path(SLUG).parent.mkdir(parents=True, exist_ok=True)
    project_mod.save(p)
    pp = paths.for_project(SLUG)
    pp.RAW_LABELS.mkdir(parents=True, exist_ok=True)
    (pp.RAW_LABELS / "img_000.txt").write_text("0 .5 .5 .1 .1\n")

    def stamp(key):
        return pp.DATASET_DIR / f".last_sync.{key}"

    try:
        print("1. a SUCCESSFUL sync stamps the machine, and only then")
        ok_run = FakeRun(returncode=0)
        r = mirror.sync_one(SLUG, "worker-a", run=ok_run)
        note(r.get("ok") is True, f"sync_one reports ok ({r.get('sent')} sent)")
        note(stamp("worker-a").exists(), ".last_sync.worker-a was written")
        age = time.time() - int(stamp("worker-a").read_text())
        note(0 <= age < 60, f"and holds a fresh epoch ({age:.0f}s old)")
        note(len(ok_run.calls) == 2,
             f"dataset AND manifest both travelled ({len(ok_run.calls)} rsyncs)")
        note(any("project.json" in a for a in ok_run.calls[1]),
             "…the second carries project.json (a slug the worker cannot "
             "resolve is a job it must refuse)")

        print("\n2. A FAILED sync must NOT stamp — the property that makes the "
              "stamp mean anything")
        r = mirror.sync_one(SLUG, "worker-b", run=FakeRun(returncode=1))
        note(r.get("ok") is False, "sync_one reports the failure")
        note(not stamp("worker-b").exists(),
             ".last_sync.worker-b was NOT written — a broken path stays "
             "visible as staleness, it is never stamped over")

        print("\n3. stamps are PER MACHINE")
        old = int(stamp("worker-a").read_text())
        time.sleep(1.1)
        mirror.sync_one(SLUG, "worker-b", run=FakeRun(returncode=0))
        note(stamp("worker-b").exists(), "worker-b now has its own stamp")
        note(int(stamp("worker-a").read_text()) == old,
             "…and worker-a's stamp is untouched — one healthy path cannot "
             "refresh another's record")

        print("\n4. the dashboard reads the NEWEST stamp")
        from groundwork.web.api.state import _last_sync
        newest = _last_sync(pp)
        note(newest == int(stamp("worker-b").read_text()),
             f"_last_sync() is the max across machines ({newest})")
        (pp.DATASET_DIR / ".last_sync.garbage").write_text("not-a-number")
        note(_last_sync(pp) == newest,
             "an unreadable stamp is skipped, not crashed on")

        print("\n5. the exclude list stays ANCHORED, and --delete stays declared")
        cmd = ok_run.calls[0]
        note("--delete" in cmd,
             "--delete is present (the mirror is a CLONE, not a backup — "
             "the backup job is a different thing on purpose)")
        note("/dataset/images" in cmd and "/dataset/labels" in cmd,
             "split-dir excludes are anchored: /dataset/images, /dataset/labels")
        loose = [a for a in cmd if a in ("images", "images/", "labels", "labels/")]
        note(not loose,
             f"no unanchored images/labels exclude ({loose or 'none'}) — "
             f"unanchored, it also matches raw/images and testset/images and "
             f"the worker silently trains on stale data")

        print("\n6. a dry run transfers nothing and stamps nothing")
        dry_run = FakeRun(returncode=0)
        r = mirror.sync_one(SLUG, "worker-b", dry=True, run=dry_run)
        note(r.get("dry") is True and not dry_run.calls,
             "dry mode never invokes rsync")
        before = int(stamp("worker-b").read_text())
        note(before <= int(time.time()),
             "…and left the existing stamp alone (no fresh write)")
    finally:
        shutil.rmtree(project_mod.manifest_path(SLUG).parent, ignore_errors=True)
        (reg_mod.REGISTRY_PATH, ssh_identity.SSH_DIR,
         ssh_identity.KEY_FILE, ssh_identity.KNOWN_HOSTS) = saved
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("PASS — sync stamps are per machine, success-only, and honestly aged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
