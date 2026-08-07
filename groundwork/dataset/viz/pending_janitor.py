"""Nightly janitor: discard pending images that were never routed.

Pending = counted on Telegram but never given a ✓/✗/🎯 — staging, not data.
Runs from a systemd user timer at midnight and only touches items older than
--older-than-hours (default 24), so anything staged today survives until at
least tomorrow night: same-evening "I'll route it in the morning" work is
never eaten. Deleting pending can never touch training or test data
(collect.discard removes image + labels + truth/earmark only from pending).

    python -m groundwork.dataset.viz.pending_janitor          # delete >24h
    python -m groundwork.dataset.viz.pending_janitor --older-than-hours 0
"""
from __future__ import annotations
import argparse
import time

from .. import paths
from ..store import collect


def sweep(older_than_hours: float = 24.0, pp=None) -> int:
    cutoff = time.time() - older_than_hours * 3600
    n = 0
    if pp.PENDING_LABELS.exists():
        for lbl in list(pp.PENDING_LABELS.glob("*.txt")):
            # pp, and it is not decoration: the glob above enumerates THIS
            # project's stale pending, and discard() without it deleted from
            # paths.default(). Identical trees for the first project, so it looked
            # correct; for any other project it swept nothing and, on a name
            # collision, would have deleted the first project's sample instead.
            if lbl.stat().st_mtime < cutoff and collect.discard(lbl.stem, pp):
                print(f"[pending-janitor] discarded {lbl.stem}")
                n += 1
    print(f"[pending-janitor] removed {n} stale pending image(s) "
          f"(older than {older_than_hours:g}h)")
    return n


def sweep_overlays(older_than_days: float = 7.0) -> int:
    """Prune the bot's per-count overlay images from outputs/ — multi-MB PNGs
    (+ their downscaled .send.jpg copies) written on EVERY count and never
    referenced again after the Telegram reply. A week's grace keeps recent
    'what did the bot see' checks possible; found at ~1GB/477 files when the
    2026-07-26 audit noticed nothing ever deleted them."""
    from ...config import OUTPUTS_DIR
    cutoff = time.time() - older_than_days * 86400
    n = 0
    for pat in ("*_annotated.png", "*_annotated.png.send.jpg"):
        for f in OUTPUTS_DIR.glob(pat):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                n += 1
    print(f"[pending-janitor] pruned {n} old count overlay file(s) "
          f"(older than {older_than_days:g}d)")
    return n


# EXPLICIT PATTERNS, NEVER A BROAD GLOB. outputs/ root also holds
# training_history.json (the ledger — every run ever, and not reproducible),
# machines.json, machines_registry.json, users.json, api_keys.json,
# sessions.json, retrain_state.json and the gpu locks. What currently SERVES
# moved out of this directory on 2026-08-05 — active_model.json and
# serving_engine.json are per project now, under outputs/serving/<slug>/ — so
# these globs cannot reach them at all (they are non-recursive). Naming them
# here anyway, because the next person to add a pattern needs to know what is
# in the blast radius and what is merely nearby. A janitor that swept by
# age alone would eat the record of everything this repo has done. Each entry
# below names one thing that a one-off script wrote and then abandoned.
ONE_OFF = (
    "tg_*.raw.txt",              # LocateAnything raw model output, 2026-07-19
    "autolabel_full.log",
    "dataset_autolabel.log",
    "autocrop_preview.log",
    "finish_labels.log",
    "label_new.log",
    "relabel_4377.log",
    "webui_test.log",
    "cockpit_test.log",
    "la.log",
    "la_result.json",
    "la_state.json",
    "rimfilter_check_*.png",     # misses sweep_overlays' *_annotated.png pattern
)


def sweep_one_offs(older_than_days: float = 30.0) -> int:
    """Prune debris from one-off scripts that nothing owns.

    Distinct from sweep_overlays, which handles a file the bot writes on EVERY
    count and therefore needs a short grace. These are the opposite: written once
    by a script that was run by hand, never read again, and never cleaned by
    anything. Found at ~50 files dated 2026-07-19 and 07-23 during the
    2026-08-02 outputs audit — small, but the only things in outputs/ that
    nothing writes, nothing reads and nothing prunes.

    THIRTY DAYS, not seven. There is no cost to keeping them and a real cost to
    deleting something a script is still iterating on, so the floor is generous.
    """
    from ...config import OUTPUTS_DIR
    cutoff = time.time() - older_than_days * 86400
    n = 0
    for pat in ONE_OFF:
        for f in OUTPUTS_DIR.glob(pat):
            # Files only, and only at the ROOT — glob does not descend, but say
            # it anyway: a pattern edited later must not start matching a run dir.
            if f.is_file() and f.parent == OUTPUTS_DIR and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                n += 1
    print(f"[pending-janitor] pruned {n} abandoned one-off file(s) "
          f"(older than {older_than_days:g}d)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Discard stale unrouted pending images.")
    ap.add_argument("--older-than-hours", type=float, default=24.0)
    ap.add_argument("--overlay-days", type=float, default=7.0)
    ap.add_argument("--one-off-days", type=float, default=30.0)
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    # This is the the janitor job systemd unit. It only ever touches pending/ — never
    # raw/, never the holdout — but it DELETES, so it says which tree first.
    print(f"[pending-janitor] project {proj.slug} ({proj.name}) | "
          f"{proj.dataset_root}")
    sweep(args.older_than_hours, pp)
    # sweep_overlays prunes the bot's per-count PNGs from outputs/ root, which is
    # machine-global rather than per-project — no pp, deliberately.
    sweep_overlays(args.overlay_days)
    sweep_one_offs(args.one_off_days)


if __name__ == "__main__":
    main()
