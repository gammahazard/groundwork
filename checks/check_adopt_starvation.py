"""A finished run must come home even while the next one trains — and a
half-evaluated one must never.

    .venv/bin/python checks/check_adopt_starvation.py

WHY THIS EXISTS. Adoption used to skip a whole machine whenever its retrain
state said "running". That guard was protecting something real — count_eval.json
exists DURING a champion run's eval, while checkpoint promotion may still
rewrite it — but machine-wide it starved chained runs: back-to-back training
keeps the state "running", the 5-minute tick never sees idle, and a finished
run sits on the worker with nothing anywhere saying why. Measured 2026-08-09:
a run with MAE 0.0 waited behind a 90-second idle gap no tick ever landed in.

The fix is a per-run contract: the pipeline stamps `.complete` when a run is
fully finished (evaluated, snapshotted, recorded, extras done), and the scan
adopts stamped runs mid-retrain. This check pins both halves of the scan side:

  1. THE PROBE — the exact bash script scan() ships (`_probe_cmd`) — run
     against a real fixture tree, not asserted textually. It must list a
     scored run, flag a stamped one, and stay silent about an unscored or
     already-adopted one.
  2. THE DECISION (`_eligible`): mid-retrain, only stamped runs go; idle,
     unstamped runs go too (workers running older code never stamp, and their
     runs must keep the old cadence rather than wait forever).

Also pinned: `_says_running`'s safe side — an UNPARSEABLE state answers
"running", because adopting mid-run costs a checkpoint while waiting costs one
five-minute tick.

The pipeline-side stamping is a one-line `touch` at the end of a real GPU run
and is not exercised here; what this check guarantees is that IF the stamp is
present the run comes home immediately, and if the rule inverts (stamped runs
waiting, unstamped runs adopted mid-retrain) CI fails.

READ-ONLY AND GPU-FREE: the fixture is a temp directory; nothing is started.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Before the import below: _machine_lock creates its lockfile under
# OUTPUTS_DIR, which config resolves from this env at import time — the check
# must never write into a real install's outputs/.
_TMP = tempfile.mkdtemp(prefix="check_adopt_")
os.environ["GW_DATA_DIR"] = _TMP

from groundwork.dataset.pipeline.adopt_run import (_eligible, _machine_lock,  # noqa: E402
                                                   _probe_cmd, _says_running)

FAILS = 0


def expect(cond: bool, what: str) -> None:
    global FAILS
    print(("  ok  " if cond else "  FAIL") + f" {what}")
    if not cond:
        FAILS += 1


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        runs = root / "proj" / "runs"
        # scored + stamped: fully finished, must come home even mid-retrain
        (runs / "a" ).mkdir(parents=True)
        (runs / "a" / "count_eval.json").write_text("{}")
        (runs / "a" / ".complete").touch()
        # scored, no stamp: an old-worker run — home only when idle
        (runs / "b").mkdir()
        (runs / "b" / "count_eval.json").write_text("{}")
        # unscored: mid-training or crashed — never listed at all
        (runs / "c").mkdir()
        # already adopted: never listed again
        (runs / "d").mkdir()
        (runs / "d" / "count_eval.json").write_text("{}")
        (runs / "d" / ".adopted").touch()
        (root / "outputs").mkdir()
        (root / "outputs" / "retrain_state.json").write_text(
            '{"status": "running"}')

        print("[1] the shipped probe script, against a real tree")
        r = subprocess.run(["bash", "-c", _probe_cmd(str(root), "proj/runs")],
                           capture_output=True, text=True, timeout=30)
        expect(r.returncode == 0, "probe exits 0 (scan treats nonzero as unreachable)")
        state_raw, _, listing = r.stdout.partition("---")
        lines = set(listing.split())
        expect(lines == {"a|complete", "b"},
               f"lists the two candidates, stamp flagged (got {sorted(lines)})")
        expect("running" in state_raw, "state rides in front of the listing")

        print("[2] the stamp is read from disk, not assumed")
        (runs / "a" / ".complete").unlink()
        r2 = subprocess.run(["bash", "-c", _probe_cmd(str(root), "proj/runs")],
                            capture_output=True, text=True, timeout=30)
        expect("a|complete" not in r2.stdout and "a" in r2.stdout.split(),
               "unstamping a run drops its flag")

    print("[3] the decision: who goes, who waits")
    listing = "a|complete b"
    expect(_eligible(listing, running=True) == (["a"], ["b"]),
           "mid-retrain: stamped goes, unstamped waits")
    expect(_eligible(listing, running=False) == (["a", "b"], []),
           "idle: everything goes (old workers never stamp)")
    expect(_eligible("", running=True) == ([], []), "empty listing is empty")

    print("[4] unknown is not idle")
    expect(_says_running('{"status": "running"}') is True, "running is running")
    expect(_says_running('{"status": "done"}') is False, "done is not")
    expect(_says_running("") is False, "no state file means idle")
    expect(_says_running("{not json") is True,
           "an unparseable state answers RUNNING — waiting costs a tick, "
           "adopting mid-run costs a checkpoint")

    print("[5] one adopter per machine — two scans must not interleave")
    # Measured 2026-08-09: a manual --scan beside the scheduler's tick adopted
    # the same run twice and the colliding worker-side rename nested one run
    # dir inside another. flock conflicts between open file descriptions, so a
    # second open() in this same process is a faithful stand-in for the
    # scheduler's separate process.
    with _machine_lock("m") as first:
        expect(first is True, "the first adopter takes the machine")
        with _machine_lock("m") as second:
            expect(second is False, "a concurrent adopter is refused, not queued")
        with _machine_lock("other") as other:
            expect(other is True, "a different machine is not blocked")
    with _machine_lock("m") as again:
        expect(again is True, "released, the machine is takeable again")

    print("PASS" if not FAILS else f"{FAILS} FAILURE(S)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
