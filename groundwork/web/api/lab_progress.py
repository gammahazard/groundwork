"""How far along is a training run — one parser per family, and who holds a card.

SPLIT OUT OF api/lab_ops.py (2026-08-02). That file was 709 lines doing two
unrelated jobs: reading progress out of four vendor frameworks' logs, and
launching/scoring/cancelling runs. This is the reading half, and it is the half
with no side effects at all — every function here opens files and returns dicts.

WHY FOUR PARSERS AND NOT ONE MECHANISM. mmengine, DEIMv2, D-FINE and
rfdetr/lightning genuinely record progress differently — a dumped python config
and scalars.json, json-lines with a COCO eval array, a json history file, a CSV.
Unifying them means either patching three vendor codebases or inventing a
lowest-common-denominator that loses the per-family facts the cockpit shows. The
duplication is real and is the honest option.

D-FINE's curve is a val LOSS, lower better, unlike every other family's val mAP.
It is reported as `val_loss` precisely so the UI can never label it a score.

IT TAKES THE ALT DIRS RATHER THAN FINDING THEM. local_active(alt_dirs) is
answering a MACHINE question — "is a card busy right now" — and CLAUDE.md records
that getting the machine-vs-project axis backwards fails in both directions
without looking like a bug. Passing the dirs in keeps that decision at the call
site, where lab_ops._every_alt() names it out loud, instead of burying it in a
module that would otherwise have no reason to know what a project is.

NOTHING HERE MAY SHELL OUT. It is read by /api/lab/status, which the Lab tab
polls; this repo's most expensive service bug is an unbounded subprocess on a
polled endpoint (see safe_proc.py and the /dev/dxg notes in CLAUDE.md). Enforced
by scratch/check_service_subprocess.py and check_no_subprocess_on_poll.py.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path


def _mm_scalars(run_dir: Path):
    scal = sorted(run_dir.glob("train/*/vis_data/scalars.json"))
    return scal[-1] if scal else None


def _mm_max_epochs(text: str) -> int | None:
    """The epoch budget that actually GOVERNS an mmengine run.

    A dumped config contains max_epochs TWICE: once at module level, inherited
    from the base config (300 for rtmdet/yolox/centernet), and once inside
    `train_cfg = dict(...)`, which is what the trainer sets from --epochs and
    what mmengine obeys. Taking the first match reported a 100-epoch run as
    37/300 — a third of the truth, so progress and every ETA derived from it
    were wrong by 3x for all four mmdet families.

    Reads the train_cfg block specifically: after the `train_cfg = dict(` line,
    scan the INDENTED lines that belong to it and stop at the next top-level
    statement. Falls back to the module-level value, which is better than
    nothing when a config is shaped differently.
    """
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if not ln.startswith("train_cfg"):
            continue
        for nxt in lines[i + 1:]:
            if nxt.strip() and not nxt[:1].isspace():
                break                      # left the block
            if (m := re.search(r"max_epochs\s*=\s*(\d+)", nxt)):
                return int(m.group(1))
        break
    m = re.search(r"^max_epochs\s*=\s*(\d+)", text, re.M)
    return int(m.group(1)) if m else None


def _mm_progress(run_dir: Path, with_curve: bool = False) -> dict:
    """Progress from an mmengine work_dir (RTMDet): the dumped config carries
    max_epochs; scalars.json streams train lines (epoch/loss) and val lines
    (coco/bbox_mAP_50). with_curve adds the full val-mAP50 history,
    downsampled — the live chart on the status card."""
    d: dict = {}
    for cfg in run_dir.glob("train/*.py"):
        total = _mm_max_epochs(cfg.read_text())
        if total:
            d["total"] = total
            break
    f = _mm_scalars(run_dir)
    if not f:
        return d
    try:
        if with_curve:
            lines = f.read_text(errors="ignore").splitlines()
        else:
            lines = f.read_bytes()[-65536:].decode(errors="ignore").splitlines()
    except OSError:
        return d
    curve = []
    for ln in lines:
        try:
            j = json.loads(ln)
        except Exception:  # noqa: BLE001 — a tail-truncated first line
            continue
        if "epoch" in j and "loss" in j:
            d["epoch"] = j["epoch"]
            d["loss"] = round(j["loss"], 3)
        if "coco/bbox_mAP_50" in j:
            d["map50"] = j["coco/bbox_mAP_50"]
            curve.append({"epoch": j.get("step"), "map50": j["coco/bbox_mAP_50"]})
    if with_curve and curve:
        step = max(1, len(curve) // 60)          # cap the payload at ~60 points
        d["curve"] = curve[::step] + ([curve[-1]] if len(curve) % step else [])
    return d


def _deim_progress(run_dir: Path) -> dict:
    """Progress from a DEIMv2 run: run_config.yml carries the epoch budget;
    train/log.txt is one json line per epoch with the val AP array."""
    d: dict = {}
    cfg = run_dir / "run_config.yml"
    if cfg.exists():
        if (m := re.search(r"epoches:\s*(\d+)", cfg.read_text())):
            d["total"] = int(m.group(1))
    curve = []
    lg = run_dir / "train" / "log.txt"
    if lg.exists():
        for ln in lg.read_text(errors="ignore").splitlines():
            try:
                j = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if "epoch" in j:
                d["epoch"] = int(j["epoch"]) + 1
                if "train_loss" in j:
                    d["loss"] = round(j["train_loss"], 2)
                ev = j.get("test_coco_eval_bbox")
                if ev:
                    d["map50"] = round(float(ev[0]), 4)   # val AP (0.5:0.95)
                    curve.append({"epoch": int(j["epoch"]), "map50": d["map50"]})
    if curve:
        step = max(1, len(curve) // 60)
        d["curve"] = curve[::step] + ([curve[-1]] if len(curve) % step else [])
    if "epoch" not in d:
        d.update(_deim_first_epoch(run_dir))
    return d


_DEIM_BATCH = re.compile(r"Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)\s*/\s*(\d+)\]")


def _deim_first_epoch(run_dir: Path) -> dict:
    """Batch-level progress from lab.log, for the window before ANY epoch closes.

    train/log.txt above gets one json line per COMPLETED epoch, so for the first
    several minutes of a DEIM run there is nothing in it and the cockpit showed
    "None/None" — a healthy run rendered identically to a dead one. That
    distinction is not cosmetic here: on 2026-08-01 a DEIM run wedged in
    uninterruptible D-state on /dev/dxg and sat at exactly that display for 16
    minutes, and telling the two apart needed an ssh session and CPU-tick
    arithmetic. The information was in lab.log the whole time.

    Reads the LAST matching line and only the tail of the file: DEIM writes a
    batch line every few seconds for the whole run, so this is a large file by
    the end and re-reading it per poll is the kind of cost that belongs nowhere
    near a polled endpoint.
    """
    lg = run_dir / "lab.log"
    if not lg.exists():
        return {}
    try:
        text = lg.read_bytes()[-16384:].decode(errors="ignore")
    except OSError:
        return {}
    last = None
    for m in _DEIM_BATCH.finditer(text):
        last = m
    if not last:
        return {}
    # +1 because DEIM's epochs are 0-based and log.txt above already reports
    # them 1-based — the two sources must not disagree by one.
    return {"epoch": int(last.group(1)) + 1,
            "batch": int(last.group(2)), "batches": int(last.group(3))}


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def _dfine_progress(run_dir: Path) -> dict:
    """Progress from a D-FINE run: train/run_config.json carries the budget,
    train/log.json is one record per epoch. NOTE its curve is a val LOSS
    (lower is better), unlike every other family's val mAP — reported as
    `val_loss` so the UI never labels it a score."""
    d: dict = {}
    cfg = run_dir / "train" / "run_config.json"
    if cfg.exists():
        try:
            d["total"] = int(json.loads(cfg.read_text())["epochs"])
        except Exception:  # noqa: BLE001
            pass
    lg = run_dir / "train" / "log.json"
    if lg.exists():
        try:
            hist = json.loads(lg.read_text())
        except Exception:  # noqa: BLE001
            hist = []          # mid-write (the trainer replaces atomically,
                               # but belt-and-braces): show nothing, not a 500
        if hist:
            d["epoch"] = int(hist[-1]["epoch"]) + 1
            d["loss"] = round(float(hist[-1]["train_loss"]), 2)
            d["val_loss"] = round(float(hist[-1]["val_loss"]), 4)
            step = max(1, len(hist) // 60)          # keep the curve small
            d["curve"] = [{"epoch": h["epoch"],
                           "val_loss": round(float(h["val_loss"]), 4)}
                          for h in hist[::step]]
    return d


def _cpu_ticks(pid) -> int | None:
    """utime+stime from /proc/<pid>/stat — the progressing-vs-wedged fact.

    A training job accumulates CPU ticks; a job wedged in an uninterruptible
    driver call does not. Process EXISTENCE proves nothing (a wedged process
    exists beautifully) — climbing ticks is the discriminator, and it costs
    one file read. The comm field may contain spaces/parens, so fields are
    taken from after the LAST ')' — the documented parse."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = raw[raw.rindex(")") + 2:].split()
        return int(fields[11]) + int(fields[12])   # utime + stime
    except Exception:  # noqa: BLE001
        return None


def _dir_bytes(d: Path) -> int:
    """Bytes under a run's train/ tree — a family-agnostic liveness fact: a
    training run writes SOMETHING (checkpoints, logs, tfevents) and a frozen
    byte count alongside a frozen log is the earliest visible wedge sign."""
    total = 0
    try:
        for f in (d / "train").rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except Exception:  # noqa: BLE001
        pass
    return total


def universal_signals(run_dir: Path, h: dict) -> dict:
    """Signals every family gets, whatever its trainer prints.

    These are FILESYSTEM AND /proc FACTS, not trainer cooperation — which is
    the repo's standing preference and the reason a family that logs nothing
    (some print no memory line, no epoch line) still gets an honest live card
    instead of a blank one.
    """
    out: dict = {}
    if h.get("started"):
        out["elapsed_s"] = int(time.time() - h["started"])
    # newest mtime across the log files this run writes
    newest = None
    for name in ("lab.log", "train/log.txt", "train/log.json"):
        f = run_dir / name
        try:
            t = f.stat().st_mtime
            newest = t if newest is None else max(newest, t)
        except OSError:
            continue
    if newest:
        out["log_age_s"] = max(0, int(time.time() - newest))
    out["ticks"] = _cpu_ticks(h.get("pid"))
    out["train_bytes"] = _dir_bytes(run_dir)
    return out


def similar_seconds(run_dir: Path, alt_dirs) -> int | None:
    """Median train_seconds of FINISHED sibling runs of the same family+size.

    The history-based ETA: a family whose trainer prints no progress at all
    still gets "similar runs took ~N min", because every finished run's
    meta.json records train_seconds. Median, not mean — one degraded-driver
    outlier must not double the estimate."""
    from ...models import registry
    fam = registry.for_run(run_dir.name)
    if not fam:
        return None
    try:
        my_meta = json.loads((run_dir / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        my_meta = {}
    res = my_meta.get("resolution")
    secs = []
    for alt in alt_dirs:
        for m in alt.glob("*/meta.json"):
            if m.parent == run_dir:
                continue
            f2 = registry.for_run(m.parent.name)
            if not f2 or f2.name != fam.name:
                continue
            try:
                meta = json.loads(m.read_text())
            except Exception:  # noqa: BLE001
                continue
            if res and meta.get("resolution") not in (None, res):
                continue
            if meta.get("train_seconds"):
                secs.append(float(meta["train_seconds"]))
    if not secs:
        return None
    secs.sort()
    return int(secs[len(secs) // 2])


def local_actives(alt_dirs) -> list[dict]:
    """EVERY alt run holding a GPU here, with parsed progress.

    A LIST, because two challengers can train at once now (per-run dataset trees,
    2026-08-02). This returned the FIRST crumb it found and stopped, which was
    correct while only one challenger could ever run — and became a real bug the
    moment two could: the cockpit showed one card busy and the other IDLE while
    both were training, because the second run was never reported at all.

    A crumb whose pid is dead (OOM-kill, or os._exit skipping the finally) is
    stale and skipped, so it cannot wedge the Lab tab or the train guard forever.
    """
    out = []
    for alt in alt_dirs:
      for crumb in sorted(alt.glob("*/gpu_holder.json")):
          try:
              h = {**json.loads(crumb.read_text()), "run": crumb.parent.name}
          except Exception:  # noqa: BLE001
              continue
          if not _alive(h.get("pid")):
              continue
          prog: dict = {}
          m = crumb.parent / "train" / "metrics.csv"          # rfdetr/lightning
          if m.exists():
              try:
                  prog["epoch"] = max(int(r["epoch"]) for r in csv.DictReader(open(m))
                                      if r.get("epoch"))
              except Exception:  # noqa: BLE001
                  pass
          elif (crumb.parent / "train" / "log.txt").exists():  # DEIMv2 json-lines
              prog = _deim_progress(crumb.parent)
          elif (crumb.parent / "train" / "log.json").exists():  # D-FINE
              prog = _dfine_progress(crumb.parent)
          else:                                               # mmengine (rtmdet)
              prog = _mm_progress(crumb.parent, with_curve=True)
          if prog.get("epoch") and prog.get("total") and h.get("started"):
              per = (time.time() - h["started"]) / prog["epoch"]
              prog["eta_s"] = int(per * (prog["total"] - prog["epoch"]))
          row = {**h, **prog, **universal_signals(crumb.parent, h)}
          if not row.get("eta_s"):
              sim = similar_seconds(crumb.parent, alt_dirs)
              if sim and h.get("started"):
                  row["similar_s"] = sim
                  row["eta_s"] = max(0, int(sim - (time.time() - h["started"])))
                  row["eta_source"] = "similar runs"
          out.append(row)
    return out


def local_active(alt_dirs) -> dict | None:
    """The FIRST alt run holding a GPU here, or None.

    Kept for the callers that only ask "is anything training" — the train guard
    and train_dispatch's busy check. Anything that DISPLAYS what is running wants
    local_actives(), or it will show a busy card as idle.
    """
    a = local_actives(alt_dirs)
    return a[0] if a else None
