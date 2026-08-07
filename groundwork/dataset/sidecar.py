"""Atomic read/write for the JSON sidecars several processes share.

This repo runs at least four processes against the same files: the web service, the
Telegram bot, the detached retrain pipeline, and the the adopt job/the janitor job timers.
They coordinate through JSON on disk and nothing else, so every sidecar is a
cross-process shared mutable object whether or not its module says so.

Two failure modes were live before this module existed, both found 2026-07-30:

  TORN READ. `Path.write_text` truncates before writing, so a concurrent reader
  can observe an empty or half-written file. Modules that swallow the resulting
  JSONDecodeError silently reset to a default — la_state.json reported "no probe
  running" and would have admitted a second GPU job; synth_provenance losing its
  map would let an image enter the FROZEN holdout while synthetic images built from
  its pixels stayed in training. Modules that do NOT swallow it (collect's truth
  counts) 500 the cockpit's image grid instead.

  SHARED TMP NAME. `training_history._save` and the two job-state writers all
  used `path.with_suffix(".json.tmp")` — one fixed name for every process. A
  truncates the tmp and writes partially; B truncates the same tmp, writes fully
  and os.replace()s it into place; A's remaining bytes then land INSIDE the live
  file, and A's own replace raises. The ledger holds 23 runs of training history
  with no backup, and `load()` swallowing the parse error means the next write
  silently rewrites it with one record.

So: tmp names are per-process, and the replace is atomic. Readers get a default
on absence but the caller decides what a CORRUPT file means — silently resetting
a file that records what is safe to train on is not always the kind thing to do.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any, *, indent: int | None = None,
               sort_keys: bool = False) -> None:
    """Write `data` so no other process can ever observe a partial file.

    The tmp name carries our pid: os.replace is atomic, but two processes
    sharing one tmp path are not, and that is the bug this exists to prevent.

    sort_keys mirrors json.dumps — the stem-keyed sidecars write sorted so the
    file diffs cleanly by hand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=indent, sort_keys=sort_keys),
                       encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)     # never leave a stale tmp behind
        raise


def read_json(path: Path, default: Any = None, *, strict: bool = False) -> Any:
    """Read `path`, returning `default` when it does not exist.

    strict=False (default): a corrupt file also yields `default` — right for
    caches and display-only data, where carrying on beats a 500.
    strict=True: a corrupt file RAISES. Use it when the file records something
    safety-bearing (what has already been trained on, which images may enter the
    holdout), where silently continuing from an empty default is the dangerous
    outcome rather than the safe one.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if strict:
            raise
        return default
