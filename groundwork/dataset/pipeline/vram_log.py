"""What a training configuration ACTUALLY cost, including when it failed.

    {"model": "yolov8n", "imgsz": 960, "batch": 4,
     "peak_gb": 7.8, "ok": false, "card_gb": 8, "at": 1785…}

WHY A SEPARATE STORE. training_history records a completed run — it is the record
of results, and a run that died has no result to record. But the most useful VRAM
fact on this system comes from exactly that run: the one that reached for more
than the card had. `_record_history` is only called on success, so today an OOM
teaches nothing and the configuration stays "unmeasured", which the fit check
correctly treats as ALLOWED. It would be offered again, and fail again.

A FAILED PEAK IS A FLOOR, NOT A REQUIREMENT, and conflating those would be worse
than having no data. `torch.cuda.max_memory_reserved()` after an OOM reports what
the allocator MANAGED to reserve before it ran out — the configuration needs MORE
than that, by an unknown amount. So a failed observation is never read as "this
needs 7.8 GB"; it is read as "this did not fit in an 8 GB card", which is the
only thing it actually proves and is enough to stop offering it there.

A hypervisor fault writes nothing at all, because the process is gone. That case
is unreachable from inside and the fit check stays honest about it: unmeasured
means unmeasured.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ...config import OUTPUTS_DIR

PATH = OUTPUTS_DIR / "vram_observed.jsonl"


def record(model: str, imgsz: int | None, batch: int | None,
           peak_gb: float | None, ok: bool, card_gb: int | None = None) -> None:
    """Append one observation. Never raises — this must not fail a training run."""
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"model": model, "imgsz": imgsz, "batch": batch,
                                "peak_gb": peak_gb, "ok": bool(ok),
                                "card_gb": card_gb, "at": time.time()}) + "\n")
    except Exception:  # noqa: BLE001
        pass


def observations(model: str, imgsz: int | None = None,
                 batch: int | None = None) -> list[dict]:
    try:
        lines = PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if d.get("model") != model:
            continue
        if imgsz is not None and d.get("imgsz") != imgsz:
            continue
        # A LOGGED batch of None means auto-batch: the trainer chose, and we do
        # not know what it chose. Treating that as "matches every batch" let one
        # auto-batch OOM veto explicit batch sizes it never tested.
        if batch is not None and d.get("batch") != batch:
            continue
        out.append(d)
    return out


def known_too_big(model: str, imgsz: int | None, batch: int | None,
                  card_gb: float | None) -> str:
    """Has this exact configuration already FAILED on a card this size? Why.

    Only a card the same size or SMALLER counts as evidence: failing on 8 GB says
    nothing about 24 GB. Returns "" when there is no such evidence, which is the
    common case and must stay permissive.
    """
    if not card_gb:
        return ""
    for d in observations(model, imgsz, batch):
        if d.get("ok"):
            continue
        seen = d.get("card_gb")
        # THIS card must be no bigger than the one that already failed. Written
        # the other way round first, which made an 8 GB failure "prove" that a
        # 24 GB card would fail too — the docstring above said the right thing
        # and the comparison said the opposite.
        if seen and card_gb <= seen:
            when = time.strftime("%Y-%m-%d", time.localtime(d.get("at") or 0))
            return (f"this configuration already failed on a {seen} GB card "
                    f"({when}), reaching {d.get('peak_gb')} GB before it died — "
                    f"it needs more than that, by an unknown amount")
    return ""
