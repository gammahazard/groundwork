"""Adopt a run trained on a worker into HQ's ledger.

Workers train fast; HQ is the source of truth. Adoption pulls a remote run
home, renames it to the next free local number (both machines would otherwise
mint the same "yolov8n-N"), records it in training_history (so the Runs tab,
charts, gallery, curve and Δ-diff all see it), and stamps a dataset snapshot.
Exam-parity between machines is proven bit-identical, so the adopted
count_eval scores need no re-certification.

    python -m groundwork.dataset.pipeline.adopt_run --project <slug> \
        --machine <key> --run yolov8n-37 [--note "worker run"]
    python -m groundwork.dataset.pipeline.adopt_run --project <slug> --scan

Host, remote root and the ssh identity all come from the machine REGISTRY
through web/ssh_identity — never assumed. The remote layout is derived by
making each local path repo-relative and re-rooting it at the worker's own
install root: both sides run the same layout, so `outputs/projects/b/alt`
names the same place on each.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess

from .. import paths
from . import run_snapshot, training_history


def _next_local_name(pp, remote_name: str = "") -> str:
    """The next free local number IN THE ADOPTED RUN'S OWN FAMILY.

    This hardcoded `yolov8n-`, so adopting a challenger renamed it into the
    champion's family — and the ledger's model and licence columns are
    resolved from the run name, so a DEIM run arrived home recorded as an
    AGPL yolo. The family comes from the registry, which is the one thing
    that knows a name's prefix; an unrecognised name keeps its own stem.
    """
    from ...models import registry
    m = registry.for_run(remote_name)
    stem = (m.prefixes[0] if m and m.prefixes else
            re.sub(r"-?\d+$", "", remote_name) or "run")
    ns = [int(mm.group(1)) for p in pp.RUNS_DIR.glob(f"{stem}-*")
          if (mm := re.fullmatch(rf"{re.escape(stem)}-(\d+)", p.name))]
    return f"{stem}-{max(ns, default=0) + 1}"


def _says_running(state_raw: str) -> bool:
    """Is the worker's retrain_state.json reporting a live run? Unreadable
    state answers TRUE — refusing to adopt costs one five-minute tick, while
    adopting mid-run copies a checkpoint that is still being written."""
    import json as _json
    raw = (state_raw or "").strip()
    if not raw:
        return False
    try:
        return (_json.loads(raw) or {}).get("status") == "running"
    except ValueError:
        print("[adopt-scan] could not parse the worker's retrain state — "
              "assuming a run is live and waiting")
        return True


def _t(machine_key: str) -> tuple[list[str], str, str]:
    """(ssh_argv, user@host, remote_root) for the worker."""
    from ...web import ssh_identity
    host, root = ssh_identity.target(machine_key)
    return ssh_identity.ssh_argv(machine_key), host, root


def _rel(p) -> str:
    # DATA_DIR == ROOT on a native checkout; on a data-dir install it is
    # where the layout actually lives, and ROOT would refuse every path.
    from ...config import DATA_DIR
    return p.relative_to(DATA_DIR).as_posix()


def _ssh(t, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run([*t[0], t[1], *args],
                          capture_output=True, text=True, timeout=timeout)


def _remote_text(t, path: str, timeout: int = 60) -> str | None:
    """cat a file on the worker; None if it isn't there or ssh fails."""
    r = _ssh(t, "cat", path, timeout=timeout)
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def _worker_record(t, remote_run: str) -> dict:
    """The worker's OWN ledger row for this run.

    The worker runs the same pipeline we do, so it already measured this run
    properly: images and objects as its split actually consumed them,
    wall-clock duration, peak VRAM. Adoption used to throw all of that away
    and re-derive it from HQ's live dataset at adoption time, which produced a
    different answer for every field.

    Keyed on the REMOTE name: the worker's row still carries it at this point,
    because the rename to the local name happens later.
    """
    raw = _remote_text(t, f"{t[2]}/outputs/training_history.json")
    if not raw:
        return {}
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    match = [r for r in rows if isinstance(r, dict) and r.get("run") == remote_run]
    return match[-1] if match else {}


def adopt(machine_key: str, remote_run: str, note: str = "", pp=None,
          project: str | None = None) -> str:
    if project is None or pp is None:
        raise ValueError("adopt() needs a project — a run always belongs to one")
    t = _t(machine_key)
    runs_rel = _rel(pp.RUNS_DIR)
    remote_runs = f"{t[2]}/{runs_rel}"
    local = _next_local_name(pp, remote_run)
    dst = pp.RUNS_DIR / local
    dst.mkdir(parents=True, exist_ok=True)
    # Weights, tuning, previews, curve, args — skip the fat epoch checkpoints.
    cmd = ["rsync", "-az", "-e", " ".join(t[0]),
           "--include", "weights/", "--include", "weights/best.pt",
           "--include", "weights/last.pt", "--exclude", "weights/*",
           "--include", "count_eval*.json", "--include", "args.yaml",
           "--include", "results.csv", "--include", "eval_preview*/**",
           # The worker's own dataset manifest — what this run REALLY trained
           # on. Without it we used to regenerate one here from HQ's split
           # dirs, which describes a different dataset entirely.
           "--include", "dataset_manifest.json.gz",
           "--include", "eval_preview*/", "--include", "*.jpg",
           "--include", "*.png", "--exclude", "*",
           f"{t[1]}:{remote_runs}/{remote_run}/", str(dst) + "/"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)   # no empty ghost run dirs
        raise SystemExit(f"[adopt] rsync failed: {r.stderr.strip()[:400]}")
    ce = dst / "count_eval.json"
    if not ce.exists():
        shutil.rmtree(dst, ignore_errors=True)
        raise SystemExit(f"[adopt] {remote_run} has no count_eval.json — score it first")
    # Mark the WORKER copy adopted FIRST — before the ledger write — so a crash
    # or ssh failure anywhere later can only lose THIS adoption, never cause
    # the 5-minute scan to re-adopt a duplicate.
    m = _ssh(t, f"touch {remote_runs}/{remote_run}/.adopted", timeout=30)
    if m.returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        raise SystemExit(f"[adopt] could not mark {remote_run} .adopted on "
                         f"{machine_key} — aborting so the scan can't "
                         f"double-adopt: {m.stderr.strip()[:200]}")
    d = json.loads(ce.read_text())
    prev = None
    for rec in reversed(training_history.load()):
        if rec.get("mae") is not None:
            prev = rec["mae"]
            break
    # Carry the worker's measurements home rather than re-deriving worse ones.
    # Anything it didn't record falls back to the run's own manifest.
    wk = _worker_record(t, remote_run)
    training_history.record_run(
        local, mae=d.get("mae"), prev_mae=prev, mae_source=d.get("source", "?"),
        images=wk.get("images"), objects=wk.get("objects"),
        duration_s=wk.get("duration_s"), peak_vram_gb=wk.get("peak_vram_gb"),
        log_text=_remote_text(t, f"{t[2]}/outputs/training_logs/{remote_run}.log"),
        ckpt="last" if d.get("weights") == "last.pt" else "best",
        machine=wk.get("machine") or machine_key,
        # Carried home like the rest of the worker's measurements. A remote
        # split-seed run is exactly the case that must not arrive looking
        # like a control.
        split_seed=wk.get("split_seed") or 0,
        pp=pp, project=project)
    training_history.set_note(local, note or f"adopted from {machine_key} ({remote_run})")
    # A manifest that came home with the run is the authoritative one — keep it
    # and just make sure its blobs are present locally. Only write a fresh one
    # when the worker had none.
    if run_snapshot.manifest_path(local, pp).exists():
        b = run_snapshot.ensure_blobs(local, pp)
        print(f"[adopt] kept the worker's manifest | blobs +{b['copied']}" +
              (f" | {len(b['missing'])} unrestorable (image gone from raw/)"
               if b["missing"] else ""))
    else:
        try:
            run_snapshot.write_snapshot(local, pp)
        except Exception as e:  # noqa: BLE001 — snapshot is a bonus, not a gate
            print(f"[adopt] snapshot skipped: {e}")
    # Rename the run ON THE WORKER too, so both cockpits show the same name —
    # a run having two identities was confusing the moment it happened.
    if remote_run == local:
        print(f"[adopt] {machine_key}:{remote_run} -> {local} | MAE {d.get('mae')} "
              f"({d.get('source')}) | recorded in the ledger")
        return local
    r2 = _ssh(t, f"mv {remote_runs}/{remote_run} {remote_runs}/{local}",
              timeout=60)
    print(f"[adopt] worker copy renamed {remote_run} -> {local}"
          if r2.returncode == 0 else
          f"[adopt] ⚠ worker rename failed (cosmetic): {r2.stderr.strip()[:120]}")
    print(f"[adopt] {machine_key}:{remote_run} -> {local} | MAE {d.get('mae')} "
          f"({d.get('source')}) | recorded in the ledger")
    return local


def scan(machine_key: str, pp=None, project: str | None = None) -> int:
    """Auto-adopt: bring home every finished worker run not yet adopted.

    A run qualifies when it has a count_eval.json and no .adopted marker, and
    the worker isn't mid-retrain (so we never grab a half-evaluated run). Runs
    from the scheduler every few minutes — pressing Train is all a human ever
    has to do."""
    t = _t(machine_key)
    runs_rel = _rel(pp.RUNS_DIR)
    probe = _ssh(
        t, "bash", "-c",
        f"'cat {t[2]}/outputs/retrain_state.json 2>/dev/null; echo ---; "
        f"for d in {t[2]}/{runs_rel}/*/; do "
        "n=$(basename $d); "
        "[ -f $d/count_eval.json ] && [ ! -f $d/.adopted ] && echo $n; done; exit 0'",
        timeout=60)
    if probe.returncode != 0:
        print(f"[adopt-scan] {machine_key} unreachable: {probe.stderr.strip()[:120]}")
        return 0
    state_raw, _, listing = probe.stdout.partition("---")
    # PARSED, not string-matched. This tested for one exact spacing of
    # '"status": "running"', so a formatting change on the worker would let the
    # scan adopt a half-evaluated run mid-retrain.
    if _says_running(state_raw):
        print(f"[adopt-scan] {machine_key} is mid-retrain — waiting")
        return 0
    n = 0
    for name in listing.split():
        try:
            local = adopt(machine_key, name, pp=pp, project=project)
            _ssh(t, f"touch {t[2]}/{runs_rel}/{local}/.adopted", timeout=30)
            n += 1
        except SystemExit as e:
            print(f"[adopt-scan] {name}: {e}")
    n += _scan_alt(machine_key)
    if n == 0:
        print(f"[adopt-scan] {machine_key}: nothing new to adopt")
    return n


def _scan_alt(machine_key: str) -> int:
    """Bring home finished challenger results — verdicts, curves and previews
    for the home Lab tab. Checkpoints stay on the worker (that's where
    re-scoring happens); no renaming or ledger involved, which is why a
    challenger never appears in training_history.json.

    EVERY PROJECT'S TREE, not just one. The remote path is derived by making
    each ALT_DIR repo-relative: both machines run the same layout, so
    `outputs/projects/b/alt` names the same place on each. A project whose
    dataset_root sits OUTSIDE the repo has no such shared name and is skipped
    rather than guessed at.
    """
    from ...config import ROOT
    from .. import paths as _paths
    t = _t(machine_key)

    targets = []                       # (remote_rel, local_alt_dir)
    for pp in _paths.every_project():
        try:
            from ...config import DATA_DIR
            rel = pp.ALT_DIR.relative_to(DATA_DIR).as_posix()
        except ValueError:
            print(f"[adopt-scan] {pp.slug}: alt dir is outside the install root, skipped")
            continue
        targets.append((rel, pp.ALT_DIR))
    if not targets:
        return 0

    # ONE probe for every tree. The rels come from project slugs, which are
    # `^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$` — no quoting hazard — but they are
    # still passed as a quoted shell list rather than interpolated bare.
    rel_list = " ".join(f"'{rel}'" for rel, _ in targets)
    probe = _ssh(
        t, "bash", "-c",
        f"'for rel in {rel_list}; do for d in {t[2]}/$rel/*/; do "
        "n=$(basename \"$d\"); "
        "[ -f \"$d/count_eval.json\" ] && [ ! -f \"$d/.adopted\" ] "
        "&& echo \"$rel|$n\"; done; done; exit 0'",
        timeout=60)
    if probe.returncode != 0:
        return 0

    by_rel = dict(targets)
    n = 0
    for line in probe.stdout.split():
        rel, _, name = line.partition("|")
        dst_root = by_rel.get(rel)
        if dst_root is None or not name:
            continue                    # a tree we did not ask about
        if name in ("datasets", "pretrain", "parity"):
            continue
        dst = dst_root / name
        dst.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["rsync", "-az", "-e", " ".join(t[0]),
                            "--exclude", "train/",
                            f"{t[1]}:{t[2]}/{rel}/{name}/", str(dst) + "/"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            _ssh(t, f"touch {t[2]}/{rel}/{name}/.adopted", timeout=30)
            print(f"[adopt-scan] alt result home: {rel}/{name}")
            n += 1
    return n


def main() -> None:
    from ...web import machines as machines_mod
    ap = argparse.ArgumentParser(description="Adopt a worker-trained run into the ledger.")
    ap.add_argument("--machine", default=None,
                    help="machine key (default for --scan: every verified worker)")
    ap.add_argument("--run", help="remote run name, e.g. yolov8n-39")
    ap.add_argument("--note", default="")
    ap.add_argument("--scan", action="store_true",
                    help="auto-adopt every finished, un-adopted worker run")
    from ...project import load, slugs
    # A MACHINE job, not a stage: --scan sweeps every project when none is
    # named (the scheduler runs it that way every 5 minutes — requiring a
    # project made every tick exit 2, so auto-adoption never ran). Adopting
    # ONE named run still requires naming whose it is.
    ap.add_argument("--project", default=None, help="project slug "
                    "(--scan default: every project)")
    args = ap.parse_args()
    if args.scan:
        keys = [args.machine] if args.machine else list(machines_mod.workers())
        if not keys:
            print("[adopt-scan] no workers registered — nothing to scan")
            return
        targets = ([load(args.project)] if args.project
                   else [load(s) for s in slugs()])
        for proj in targets:
            pp = paths.for_project(proj)
            print(f"[adopt-scan] project {proj.slug} | {proj.dataset_root}")
            for key in keys:
                scan(key, pp, proj.slug)
    elif args.run:
        if not args.machine:
            ap.error("--machine is required with --run")
        if not args.project:
            ap.error("--project is required with --run")
        proj = load(args.project)
        pp = paths.for_project(proj)
        print(f"[adopt] project {proj.slug} ({proj.name}) | {proj.dataset_root}")
        adopt(args.machine, args.run, args.note, pp, proj.slug)
    else:
        ap.error("--run or --scan required")


if __name__ == "__main__":
    main()
