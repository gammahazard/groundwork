"""GET /api/train/options, POST /api/train, DELETE /api/train."""

from __future__ import annotations

from .model import router, TrainReq, REMOTE_TIMEOUT  # noqa: F401
from .busy import _busy_on, _yolo_busy_on  # noqa: F401
from .cards import _cells, _default_card, _batch_for  # noqa: F401
from .remote import _remote, _remote_ok, _speaks_project, refuse_if_busy  # noqa: F401

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ....ops import mirror as mirror_mod
from .... import project as project_mod
from ....models import registry
from ... import machines as machines_mod
from .. import lab_ops, train as train_api
from ..deps import current_project


@router.get("/api/train/options")
def options(p=Depends(current_project)):
    """Every model x machine x card, with why each one is or is not usable.

    Answered up front so the UI can disable a combination WITH ITS REASON,
    instead of letting you press Train and reading a 400 — and so the browser
    never decides what is trainable.
    """
    # Per MACHINE: one may be current and another old, so each row carries
    # its own project-awareness answer rather than one machine speaking for
    # the fleet.
    ms = []
    remote_answers = {}
    for row in machines_mod.summary():
        if row["local"]:
            r_ok, r_why = True, ""
        else:
            r_ok, r_why = _remote_ok(p, machines_mod.get(row["key"]))
        remote_answers[row["key"]] = (r_ok, r_why)
        ms.append({**row,
                   "usable": row["trains"] and row["reachable_name"] and r_ok,
                   "blocked": ("" if row["trains"] else
                               "not a training target — use the CLI here")
                              or (row["why"] if not row["reachable_name"] else "")
                              or ("" if r_ok else r_why)})
    models = []
    for m in registry.MODELS:
        # NOT TRAINABLE IS NOT A CARD PROBLEM. Five of the nine families in the
        # registry have `trainer_module=None` — they are there so the ledger can
        # NAME a run someone trained by hand, not because the cockpit can start
        # one. Offering them and 400ing on the press is the exact thing this
        # endpoint exists to prevent, so they are excluded here rather than
        # disabled: a permanently-disabled option is clutter, not information.
        if not m.trainer_module:
            continue
        per = {}
        for key in machines_mod.all_machines():
            cells = _cells(m, key)
            per[key] = {"cards": cells, "default": _default_card(m, key),
                        "any": any(c["ok"] for c in cells)}
        models.append({
            "key": m.name, "label": m.label, "license": m.license,
            "deprecated": m.deprecated, "venv": m.venv,
            "default_imgsz": m.default_imgsz, "default_epochs": m.default_epochs,
            # WHAT A BLANK BATCH FIELD WILL ACTUALLY DO. The UI showed "auto",
            # which hid the one fact worth knowing — yolo halves above 960 and
            # every other family takes its own default, and rfdetr's 2 is
            # load-bearing (it asserts batch * grad_accum == 16). Sent so the
            # placeholder does not have to re-derive the rule and get it wrong.
            "default_batch": m.default_batch,
            # The sizes THIS family can train at. Offering one it cannot is how
            # a 960 pick for yolox silently became a 1280 run.
            "sizes": list(m.sizes),
            # THE RUN-NAME PREFIXES, so the UI can say which family a run
            # belongs to without a second copy of the mapping. Matching on the
            # model KEY instead labelled deimv2-n-1280-… as "deimv2-n-tv28",
            # because stripping tv28's suffix also yields "deimv2-n" and the
            # longer key won. registry.prefixes is what decides a run's family
            # everywhere else (registry.for_run); this is the same answer.
            "prefixes": list(m.prefixes),
            "champion": m.name == "yolov8n", "machines": per})
    any_remote_blocked = next(((ok, why) for ok, why in remote_answers.values()
                               if not ok), (True, ""))
    return {"project": p.slug, "machines": ms, "models": models,
            "remote_ok": any_remote_blocked[0], "remote_why": any_remote_blocked[1]}


@router.post("/api/train")
def start(req: TrainReq, p=Depends(current_project)):
    """Start a run. The ONE entry point, whatever the model and wherever it runs.

    Prints project, model, machine and card before dispatching — the same rule
    every stage follows, and it matters more here because "which machine" and
    "which card" are two more dimensions a log line is the only record of.
    """
    m = registry.by_name(req.model)
    if m is None:
        raise HTTPException(422, f"unknown model {req.model!r} — one of "
                                 f"{', '.join(x.name for x in registry.MODELS)}")
    machine = machines_mod.get(req.machine)
    if machine is None:
        raise HTTPException(422, f"unknown machine {req.machine!r} — one of "
                                 f"{', '.join(machines_mod.MACHINES)}")
    if not m.trainer_module:
        raise HTTPException(400, f"{m.label} has no trainer wired — it is in the "
                                 f"registry so the ledger can name runs trained "
                                 f"by hand, not so the cockpit can start one")
    if not machine.trains:
        raise HTTPException(400, f"{machine.name} is not a training target — "
                                 f"use the CLI there if you need it")
    trainable, train_why = machines_mod.can_train(machine.key)
    if not trainable:
        raise HTTPException(400, f"{machine.name}: {train_why}")
    ok, why = machines_mod.check(machine)
    if not ok:
        raise HTTPException(502, why)
    if not machine.local:
        ok, why = _remote_ok(p, machine)
        if not ok:
            raise HTTPException(400, why)

    # WHICH CARD, and can this model use it at all.
    card = req.card if req.card is not None else _default_card(m, machine.key)
    cells = {c["index"]: c for c in _cells(m, machine.key)}
    if card is None:
        raise HTTPException(400, f"no card on {machine.name} can run "
                                 f"{m.label} — {m.venv} is missing or its "
                                 f"torch build has no kernels for these cards")
    if card in cells and not cells[card]["ok"]:
        raise HTTPException(400, cells[card]["why"])

    # AN UNDERSIZED CARD IS SAID OUT LOUD, not refused. _default_card already
    # prefers one that fits, so reaching here means either an explicit choice or
    # no card in the box is big enough — both legitimate. What is NOT acceptable
    # is paying 3x quietly, which is exactly what happened to deimv2-n-tv28 on
    # 2026-08-02: it ran for hours at a third of its usual rate and the only
    # evidence anywhere was an s/it figure nobody was comparing.
    # FIT IS JUDGED AGAINST THE REQUESTED CONFIGURATION, not the family default.
    #
    # registry.peak_vram_gb is ONE number per family, taken at its default size.
    # Judging a 960/batch-4 request by yolov8n's 1280 figure (10.4 GiB) refused
    # a configuration the owner has actually trained on an 8 GB laptop — the
    # ledger shows 1280/batch-4 peaking 5.40-9.20 and 960/batch-8 at 6.33-10.00,
    # so resolution and batch move this by gigabytes and the family number cannot
    # answer for any of them.
    #
    # UNMEASURED IS NOT TOO BIG. There is no 960/batch-4 row at all — those runs
    # predate the ledger — and refusing on absent evidence is the mistake this
    # whole block exists to stop making in the other direction.
    want_batch = _batch_for(m, req)
    seen = machines_mod.measured_peak(m.name, req.imgsz or m.default_imgsz,
                                      want_batch)
    have = (cells.get(card) or {}).get("vram_gb")
    spill = ""
    # A CONFIGURATION THAT HAS ALREADY DIED ON A CARD THIS SIZE. This is the
    # strongest evidence available and it is not a peak: after an OOM
    # max_memory_reserved reports what the allocator MANAGED to take, so the true
    # requirement is higher by an unknown amount. Treated as "did not fit here",
    # never as "needs exactly that much".
    from ....dataset.pipeline import vram_log
    died = vram_log.known_too_big(m.name, req.imgsz or m.default_imgsz,
                                  want_batch, (cells.get(card) or {}).get("vram_gb"))
    if died and not req.allow_spill:
        raise HTTPException(
            409, f"{m.label} at {req.imgsz or m.default_imgsz}px batch "
                 f"{want_batch}: {died}. Use a smaller size or batch, a bigger "
                 f"card, or allow_spill:true to try it anyway.")
    if seen is not None and have:
        roomy, tax = machines_mod.fits({"vram_gb": have}, seen)
        spill = "" if roomy else (
            f"measured {seen:.1f} GiB at {req.imgsz or m.default_imgsz}px "
            f"batch {want_batch}, and this card has {have} GB — {tax.split('—')[-1].strip()}")
    if spill and not req.allow_spill:
        # IT USED TO JUST PRINT THIS AND CARRY ON, and that cost four hours on
        # 2026-08-02: deimv2-n-tv28 needs 17.2 GiB, landed on the 16 GB card, and
        # trained at 2.19 s/it against 0.72-0.94 for seven runs of the same model
        # on the 24 GB card. Nothing errored, because WSL2 spills into host RAM
        # instead of raising OOM — the only evidence anywhere was an s/it figure
        # nobody was comparing.
        #
        # On an 8 GB box the same spill is worse than slow: config.py records the
        # hypervisor faulting and taking the whole machine down.
        #
        # So it is refused by default and opted into by name, exactly like
        # allow_concurrent: spilling deliberately to keep a card busy is a fair
        # trade, and it is a trade someone should make on purpose.
        raise HTTPException(
            409, f"{m.label} does not fit card {card} on {machine.name}: {spill}. "
                 f"Pick a bigger card, or pass allow_spill:true to accept the "
                 f"cost deliberately.")
    if spill:
        print(f"[train] card {card} is undersized for {m.name} and allow_spill "
              f"was set: {spill}", flush=True)

    # REMOTE CARD CHOICE IS WIRED NOW (2026-08-02, owner: "I can't leave a full
    # card unused, these are expensive"). /api/lab/train takes a `card` and turns
    # it into CUDA_VISIBLE_DEVICES, so a challenger is no longer nailed to the
    # 3090. Four families can physically use the worker's 5070 Ti — yolov8n,
    # deimv2-n-tv28, dfine-small, rfdetr-nano — and three of them previously had
    # no route to it at all.
    #
    print(f"[train] {p.slug} | model {m.name} | {machine.key} card {card}",
          flush=True)

    refuse_if_busy(req, m, machine, p.slug)

    # SYNC BEFORE TRAINING, ALWAYS, when the run is going somewhere else.
    #
    # The worker's copy of a dataset is a MIRROR on a 5-minute timer, and the gap
    # between "I labelled an image" and "the worker has it" is exactly when you press
    # Train. On 2026-07-30 the lab was a full day stale — 136 images against HQ's
    # 155 — and two challenger runs trained without the 19 newest, silently, for
    # hours. That is the failure this repo is most expensive at.
    #
    # rsync is incremental, so this costs a directory listing when nothing has
    # changed (measured: 1.1s for the first project's 225 images). A run is 40 minutes.
    # And it REFUSES rather than training on a copy it could not confirm: an
    # unreachable worker means the data might be anything, and "probably fine" is
    # how a day gets lost.
    synced = None
    if not machine.local:
        try:
            # THROUGH safe_proc — this runs in a request handler, and a wedged
            # rsync must not park an anyio worker thread forever (the repo's
            # standing subprocess rule; mirror's own scheduler runs use the
            # plain default).
            from .... import config as _cfg  # noqa: F401 — keeps import graph explicit
            from ... import safe_proc as _sp
            synced = mirror_mod.sync_one(p.slug, machine.key, run=_sp.run)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"could not sync {p.slug} to "
                                     f"{machine.name} ({type(e).__name__}) — "
                                     f"refusing to train on data I cannot "
                                     f"confirm") from None
        if not synced.get("ok"):
            raise HTTPException(
                502, f"sync to {machine.name} failed, so its copy of {p.slug} "
                     f"may be stale — refusing to train. "
                     f"{synced.get('error','')}"[:300])
        print(f"[train] synced {p.slug}: {synced['sent']} file(s) sent, "
              f"{synced.get('images')} training + {synced.get('holdout')} holdout",
              flush=True)

    # WHERE this run went, answered once for all four dispatch paths below.
    # `spill` is None on a card with room, so a caller can tell "fine" from
    # "running, and here is what it is costing you".
    where = {"machine": machine.key, "card": card, "synced": synced,
             "spill": spill or None}

    if m.name == "yolov8n":
        want = req.imgsz or m.default_imgsz
        if want not in m.sizes:
            raise HTTPException(422, f"{m.label} does not train at {want}px — it "
                                     f"offers {', '.join(str(x) for x in m.sizes)}")
        body = {"imgsz": want, "sizes": req.sizes,
                "epochs": req.epochs, "batch": req.batch,
                "split_seed": req.split_seed, "card": card}
        if machine.local:
            return {**train_api.start(train_api.RetrainOpts(**body), p),
                    **where}
        return {**_remote(machine, "/api/retrain", body, p.slug),
                **where}

    if not req.run_name:
        raise HTTPException(422, "a challenger run needs a run_name")
    imgsz = req.imgsz or m.default_imgsz
    if imgsz not in m.sizes:
        raise HTTPException(422, f"{m.label} does not train at {imgsz}px — it "
                                 f"offers {', '.join(str(x) for x in m.sizes)}")
    # scale_factor is the lab's spelling of resolution: 2 -> 1280, 3 -> 1920.
    # THE CARD TRAVELS WITH THE REQUEST. Which cards this model may use was
    # already decided above from machines.json (_cells / can_run), so the worker can
    # honour it without a second capability table of its own — it has no
    # machines.json anyway.
    body = {"arch": m.name, "card": card,
            "scale_factor": 3 if imgsz >= 1920 else 2,
            # BOTH: scale_factor for the mmdet families that speak it, and the
            # real size for everyone else. Sending only the former silently
            # rounded every other family to 1280 or 1920.
            "imgsz": imgsz,
            "epochs": req.epochs or 0, "batch": _batch_for(m, req),
            # Challengers take a split seed too now — the comparison was
            # one-sided while only yolo could vary it.
            "split_seed": req.split_seed,
            "run_name": req.run_name}
    if machine.local:
        return {**lab_ops.train(lab_ops.TrainReq(**body), p),
                **where}
    return {**_remote(machine, "/api/lab/train", body, p.slug),
            **where}


@router.delete("/api/train")
def cancel(machine: str = "here", kind: str = "any", card: int | None = None,
           p=Depends(current_project)):
    """Scrap what is training on `machine` — the counterpart to start().

    ONE IMPLEMENTATION, REACHED TWICE. For a remote machine this forwards the
    same DELETE with machine=here, so the worker runs exactly the code HQ would
    have run locally. The alternative — mapping kinds to /api/retrain and
    /api/lab/train from here — is a second copy of "which endpoint cancels
    what", and this repo's most expensive bugs have all been two tables that
    drifted.

    NEVER 502s FOR "NOTHING TO CANCEL". A Cancel button whose job already
    finished is a normal race, not an error: the caller gets ok:false with a
    reason. It reserves failures for a machine it genuinely could not reach.
    """
    m = machines_mod.get(machine)
    if m is None:
        raise HTTPException(422, f"unknown machine {machine!r} — one of "
                                 f"{', '.join(machines_mod.MACHINES)}")
    if kind not in ("any", "yolo", "challenger"):
        raise HTTPException(422, "kind must be any, yolo or challenger")

    if m.local:
        from ... import cancel_ops
        alt_dirs = lab_ops._every_alt()
        if kind == "yolo":
            r = cancel_ops.yolo()
        elif kind == "challenger":
            r = cancel_ops.challenger(alt_dirs, card=card)
        else:
            r = cancel_ops.anything(alt_dirs, card=card)
        print(f"[cancel] {p.slug} | here | kind={kind} card={card} -> {r}",
              flush=True)
        return {**r, "machine": m.key}

    ok, why = machines_mod.check(m)
    if not ok:
        raise HTTPException(502, why)
    path = f"/api/train?machine=here&kind={kind}"
    if card is not None:
        path += f"&card={int(card)}"
    # Short timeout: cancelling is a few signals and an rmtree. The 60s used for
    # starting covers a split plus a convert, and a Cancel button that appears
    # to hang for a minute is one a user presses again.
    r = _remote(m, path, None, p.slug, method="DELETE", timeout=30.0)
    return {**r, "machine": m.key}
