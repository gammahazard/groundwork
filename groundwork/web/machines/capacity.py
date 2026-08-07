"""The MEASURED capability map: cards, venv arch lists, fit, summary."""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass

from ...config import OUTPUTS_DIR
from .model import MACHINES, Machine, check  # noqa: F401

MAP_PATH = OUTPUTS_DIR / "machines.json"

# ------------------------------------------------------------- measured map ---

def _map() -> dict:
    try:
        return json.loads(MAP_PATH.read_text()).get("machines") or {}
    except Exception:  # noqa: BLE001 — no map is "unknown", never "no cards"
        return {}


def cards(key: str) -> list[dict]:
    """This machine's cards, as measured. Empty when nothing has probed it."""
    return list((_map().get(key) or {}).get("cards") or [])


def measured_at(key: str) -> float | None:
    return (_map().get(key) or {}).get("measured")


def venv_info(key: str, venv: str) -> dict:
    """{present, torch, archs} for one venv on one machine, or {} if unmeasured."""
    return dict(((_map().get(key) or {}).get("venvs") or {}).get(venv) or {})


def can_run(key: str, venv: str, card: dict) -> tuple[bool, str]:
    """Can a model needing `venv` train on `card` of machine `key`?

    THE WHOLE POINT of this module, and the answer is one comparison: is the
    card's compute capability in the list the venv's torch was COMPILED for. A
    venv whose newest kernel is sm_90 cannot drive an sm_120 card, and without a
    `compute_XX` PTX entry there is nothing to JIT forward from either.

    "Unknown" is answered as unknown. A machine nobody has probed gets a
    permissive True with a note, because refusing on absent evidence would make
    a fresh checkout unable to train anything — but the UI shows the note.
    """
    info = venv_info(key, venv)
    if not info:
        return True, "not measured yet — press Probe on the Machines tab"
    if not info.get("present"):
        return False, f"{venv} is not installed on {key}"
    archs = info.get("archs") or []
    if not archs:
        return True, f"{venv}'s arch list could not be read"
    sm = card.get("sm")
    if sm and sm not in archs:
        return False, (f"{sm} is not in {venv}'s torch build "
                       f"({info.get('torch','?')}, up to "
                       f"{max(archs, key=lambda a: int(a.split('_')[1]))}) — "
                       f"this card cannot run it")
    return True, ""


# Slack demanded over a family's measured peak before a card counts as roomy.
# The peak is the largest allocation ever SEEN, not a bound — DEIM resamples
# resolution per batch, so seven runs of one config peaked between 13.0 and 17.2
# GiB — and a card matching it exactly would spill on the unlucky batch. It also
# covers the difference between a card's nominal size and what CUDA can actually
# hand out on a card that is also driving a display.
VRAM_HEADROOM_GB = 1.0


def fits(card: dict, need_gb: float | None) -> tuple[bool, str]:
    """Is there ROOM for this family on this card? A softer question than can_run.

    THE TWO ARE DIFFERENT FAILURES and must not be merged. A card whose compute
    capability the venv was never built for cannot run the model at all — the
    first kernel launch dies, so can_run refuses. A card that is merely too SMALL
    runs it perfectly: under WSL2 the driver spills the overflow into host RAM
    over PCIe instead of raising OOM, so the run finishes, scores correctly, and
    is simply slower. Refusing that would be wrong; not knowing about it is what
    actually cost something.

    MEASURED 2026-08-02, which is why this exists at all. deimv2-n-tv28 needs
    17.2 GiB and landed on the 16 GB 5070 Ti, because the card chooser took the
    first card that COULD run it. Seven prior runs of the same model on the 24 GB
    3090 held 0.72-0.94 s/it; that one ran at 2.19 — three times slower on the
    FASTER card, and nothing anywhere reported an error.

    So this returns a NOTE, never a refusal. Callers use it to prefer a card, and
    an explicit choice still wins: spilling on purpose is a reasonable trade when
    the alternative is leaving a card idle.

    Unmeasured is answered as unmeasured, in both directions — a family with no
    recorded peak and a card with no recorded VRAM both pass with a reason,
    rather than being refused on absent evidence.
    """
    if need_gb is None:
        return True, "peak VRAM has never been recorded for this family"
    have = card.get("vram_gb")
    if not have:
        return True, "this card's VRAM is not in machines.json"
    if have >= need_gb + VRAM_HEADROOM_GB:
        return True, ""
    return False, (f"needs ~{need_gb:.1f} GiB at its default size and this card "
                   f"has {have} GB — it will still run, spilling into host RAM "
                   f"over PCIe, measured about 3x slower")


def pick_card(key: str, *, venv: str, need_gb: float | None = None,
              requested: int | None = None,
              busy: "set[int] | frozenset[int]" = frozenset()) -> tuple[int, str]:
    """Choose the GPU for a job: the caller's explicit card, else the best free
    card this venv can drive. Raises ValueError with the reason when the choice
    cannot be honoured — busy explicit card, no such index, or no free card the
    venv's kernels support.

    N CARDS, NOT A FIXED COUNT. Machines register with however many cards the
    probe measured, and everything here iterates that list — a single-card
    laptop and an eight-card server go through the same three questions per
    card: is it free, can this venv drive it (can_run), is there room for the
    family's measured peak (fits). Roomy cards win; a too-small card is still
    chosen over refusing, with the spill note passed back for the log.

    An UNPROBED machine (empty cards list) is answered permissively with card
    0 — refusing on absent evidence would make a fresh install unable to train
    at all — and the note says so.
    """
    cs = cards(key)
    if requested is not None:
        req = int(requested)
        if cs and not any(int(c.get("index", -1)) == req for c in cs):
            raise ValueError(
                f"no card {req} on {key} — measured: "
                + ", ".join(str(c.get("index")) for c in cs))
        if req in busy:
            raise ValueError(f"card {req} is busy — pick another card or wait")
        card = next((c for c in cs if int(c.get("index", -1)) == req), {})
        ok, why = can_run(key, venv, card)
        if not ok:
            raise ValueError(why)
        _, note = fits(card, need_gb)
        return req, note
    if not cs:
        return 0, "this machine has not been probed — assuming card 0"
    usable: list[tuple[bool, int, str]] = []
    refusals: list[str] = []
    for c in cs:
        idx = int(c.get("index", 0))
        if idx in busy:
            refusals.append(f"card {idx}: busy")
            continue
        ok, why = can_run(key, venv, c)
        if not ok:
            refusals.append(f"card {idx}: {why}")
            continue
        roomy, note = fits(c, need_gb)
        usable.append((not roomy, idx, note))     # roomy cards sort first
    if not usable:
        raise ValueError("no usable card is free — " + "; ".join(refusals))
    usable.sort()
    _, idx, note = usable[0]
    return idx, note


def summary() -> list[dict]:
    """Every machine, its cards, and how fresh the measurement is. For the UI."""
    from .registry import all_machines
    from .peaks import can_train
    out = []
    for key, m in all_machines().items():
        ok, why = check(m)
        at = measured_at(key)
        trainable, train_why = can_train(key)
        out.append({
            "key": key, "name": m.name, "what": m.what, "local": m.local,
            # BOTH: the policy flag AND the measured capability. A machine can be
            # marked never-train, and one that is allowed still has to hold a
            # model. `train_why` carries the reason so the UI never has to guess.
            "trains": bool(m.trains) and trainable,
            "train_why": "" if (m.trains and trainable) else
                         (train_why if m.trains else "not offered as a training target"),
            "reachable_name": ok, "why": why,
            "cards": cards(key), "measured": at,
            "age_h": None if not at else round((time.time() - at) / 3600, 1),
        })
    return out


