"""POST/DELETE /api/lab/train + /api/lab/score — launch, cancel, re-score."""

from __future__ import annotations

from .common import (router, MAIN_PY, GPU_BASE_ENV,        # noqa: F401
                     _alt, _every_alt, _trainable_archs,
                     _card_env, held_cards, _local_active, _running_seed,
                     _finishing, _score_blocked, _adoptable,
                     _split_ok)

import json
import fnmatch
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import resource
import subprocess

from ... import safe_proc
import pathlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ... import retrain_job
from ... import machines as machines_mod
from .. import lab_progress
from ..lab import REPO, _NAME
from ..deps import current_project
from ....dataset import paths
from ....dataset.pipeline import split as split_mod
from ....models import registry
from .... import project as project_mod

def _launch(cmd: list[str], log_f: Path, env: dict | None = None,
            name: str = "challenger") -> int:
    """Detached launch for challenger training/scoring.

    start_new_session gives the job its own process group. It does NOT by
    itself survive a web-service restart — systemd's default
    KillMode=control-group signals the whole cgroup, and setsid does not move
    a process out of one. Natively that takes KillMode=process in the web
    unit (deploy/units bakes it in); under Docker the spool mode hands the
    fork to the trainer daemon instead (see web/spawn.py).
    """
    def _raise_nofile():                    # a measured fd-limit lesson
        resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
    from ...spawn import spawn_detached
    pid = spawn_detached(name, cmd, log_f, env=env, cwd=REPO,
                         preexec_fn=_raise_nofile)
    return pid or 0


class TrainReq(BaseModel):
    arch: str = "deimv2-n"    # active challenger; "rtmdet-tiny" kept (deprecated)
    scale_factor: int = 2      # 2 -> 1280px, 3 -> 1920px
    # THE ACTUAL RESOLUTION, when the caller knows it. scale_factor is the
    # mmdet trainer's own spelling and only expresses 1280 or 1920, so deriving
    # the size from it DESTROYED anything else: a request for rfdetr at 1120
    # became scale_factor 2 became `--res 1280`, which its trainer asserts
    # against (must be a multiple of 56) minutes into the job. Families whose
    # sizes happen to be 1280/1920 never noticed.
    imgsz: int | None = None
    epochs: int = 0            # 0 = per-arch default (deim 60 / rtmdet 300)
    batch: int = 0             # 0 = this family's own default (registry)
    run_name: str
    # WHICH GPU. None = pick automatically: a free card this family's venv
    # can drive, roomy ones first (web/machines/capacity.pick_card). An
    # explicit index wins — and is refused with the reason when it cannot
    # be honoured (busy, no such index, venv kernels stop below its sm).
    card: int | None = None
    # WHICH TRAIN/VAL SPLIT. Membership is a seeded hash of the stem, so this is
    # the only way to ask "does this result survive a different split?" of a
    # challenger — yolo has had it since runs 59-65 and challengers had no way
    # to say it at all, which made the comparison one-sided.
    split_seed: int = 0


@router.post("/api/lab/train")
def train(req: TrainReq, p=Depends(current_project)):
    """Fresh convert (split-parity, kills stale symlinks) then the challenger on
    the 3090, detached (see _launch); per-card lock + VRAM guards."""
    if not _NAME.match(req.run_name):
        raise HTTPException(422, "bad run name")
    # Check the venv THIS arch actually needs. The old code demanded
    # .venv-mmdet up front for every request, so a deimv2-n train (the active
    # challenger, and this endpoint's default) was refused on a machine that had
    # .venv-deim but no .venv-mmdet — with a message about RTMDet.
    need = registry.by_name(req.arch) or registry.for_run(req.arch)
    if need is None or not need.trainer_module:
        raise HTTPException(400, f"no trainer wired for arch {req.arch!r}")
    if not need.venv_present:
        raise HTTPException(400, f"{need.venv} missing on this "
                                 f"machine — challengers train on the worker "
                                 f"(docs/models.md)")
    if req.scale_factor not in (2, 3):
        raise HTTPException(422, "scale_factor must be 2 (1280px) or 3 (1920px)")
    # A SECOND CHALLENGER IS FINE WHEN THIS CONVERT CANNOT TOUCH THE LIVE ONE.
    #
    # This was a flat refusal, and it was right at the time: every challenger
    # converted into ONE tree (outputs/alt/datasets/coco_rfdetr) and the convert
    # rmtree's it first, so a second launch destroyed the dataset the first was
    # reading — the dangling-file failure that killed round B's final step.
    #
    # Each run now converts into its OWN tree (web/lab_dataset.py), so the danger
    # is no longer "is something else training" but the narrower "does MY convert
    # write where IT reads". If this run is isolated it writes a directory named
    # after itself and nothing else can be harmed, whatever is running.
    #
    # THE RUNNING RUN'S TREE IS CHECKED ON DISK, NOT IN THE REGISTRY. A job
    # launched before this change reads the shared tree even though its family is
    # perfectly capable of isolation — capability describes the family, existence
    # describes this instance, and only the second one is a fact about the data.
    if (act := _local_active()):
        from ... import lab_dataset as _ds
        mine = registry.by_name(req.arch) or registry.for_run(req.arch)
        if not (mine and mine.isolated_dataset):
            # DEIM: converts into the SHARED tree. Safe only if the live run is
            # not reading it — i.e. it has one of its own.
            live_tree = _ds.tree_for(paths.for_project(p), act["run"])
            if not live_tree.exists():
                return {"ok": False, "error":
                        f"{req.arch} converts into the shared dataset tree and "
                        f"{act['run']} is training out of it — that rebuild would "
                        f"delete data it is reading. Wait for it to finish."}
        print(f"[lab/train] {act['run']} is training; this run converts into its "
              f"own tree — launching alongside it", flush=True)

    # A yolo retrain trains FROM outputs/dataset/images|labels/{train,val}, and
    # the split below rmtree's and rebuilds exactly those directories. Starting a
    # challenger mid-retrain would therefore pull the dataset out from under a
    # running job. This guard did not exist before the split step was added
    # (2026-07-30) because convert alone only touched the separate COCO tree —
    # the split-first fix is what made a concurrent launch dangerous, so the
    # guard ships with it. Per-card GPU locks are NOT enough here: the collision
    # is on the filesystem, not the GPU.
    # A YOLO RETRAIN IS ONLY A BLOCKER IF WE WOULD RE-SPLIT UNDER IT.
    #
    # This used to be a flat refusal, and it was right about the danger: the
    # split below rmtree's images|labels/{train,val}, the very directories a
    # running retrain reads. Per-card GPU locks do not help — the collision is
    # on the filesystem.
    #
    # But membership is a seeded hash of the stem, so when the tree on disk is
    # already what a fresh split would build, the split is pure destruction and
    # recreation of identical files. Skipping it removes the collision entirely,
    # and a challenger can then train on the other card while yolo runs — which
    # is the reason this box has two GPUs. Comparability improves too: both runs
    # then demonstrably saw the same split.
    #
    # If the split is NOT current, we still refuse. Re-splitting under a live
    # retrain is the one thing that cannot be made safe, and training the
    # challenger on a stale tree instead is the silent-wrong-data failure this
    # repo is most expensive at.
    reuse_split = False
    if retrain_job.status().get("status") == "running":
        pp = paths.for_project(p)
        running = _running_seed()
        # TWO RUNS SHARE ONE SPLIT DIRECTORY, so they must want the same split.
        # Asking for a different seed while a retrain trains is not something
        # that can be granted: honouring it means rebuilding the tree the
        # retrain is reading, and ignoring it means training on a split the
        # caller did not ask for and then reporting it as if they had. Refused,
        # with the number, so the choice is theirs.
        if int(req.split_seed or 0) != running:
            return {"ok": False, "error":
                    f"a yolo retrain is running on split seed {running} and "
                    f"this asks for {int(req.split_seed or 0)} — both runs share "
                    f"one split directory, so a second seed cannot be built "
                    f"without deleting the images that retrain is training on. "
                    f"Use seed {running}, or wait for it to finish."}
        fresh, why = split_mod.is_current(pp, val_frac=0.2, seed=running)
        if not fresh:
            return {"ok": False, "error":
                    f"a yolo retrain is running and the split is not current "
                    f"({why}) — re-splitting now would delete the images that "
                    f"run is training on. Wait for it to finish."}
        reuse_split = True
        print(f"[lab/train] yolo retrain in flight; split already current — "
              f"reusing it instead of rebuilding", flush=True)
    # SPLIT FIRST — do not remove this. `convert --match-split` reads the
    # machine-local split dirs (outputs/dataset/images/{train,val}), and
    # the mirror job.service deliberately EXCLUDES those from the mirror, so on the
    # lab they only change when something runs a split here. They therefore go
    # stale the moment HQ gains an image.
    #
    # Burned 2026-07-30: the lab's split dirs were dated 2026-07-29 02:26 (its
    # previous local retrain), so the COCO tree held 109/27 = 136 images while HQ
    # had 155. dfine-s-1280-b and the first attempt at deimv2-n-1280-o both
    # trained WITHOUT the 19 newest images — which were the new capsule data those
    # runs existed to evaluate. Nothing failed; the numbers were just quietly
    # about last night's dataset.
    #
    # Re-splitting is safe and is exactly what --match-split wants: membership is
    # a seeded hash of the stem (split._val_rank), so the same image set yields
    # byte-identical membership on either machine.
    # THIS PROJECT, not the default one. It was hardcoded to DEFAULT_SLUG, which
    # meant a challenger launched for project B split ANOTHER project's dataset and
    # then converted from it — the image-count check below caught the mismatch and
    # refused, so it failed loudly rather than training on the wrong images, but
    # a refusal is not support. This is the line that made a second project
    # untrainable here at all.
    # THE SEED TRAVELS WITH THE SPLIT. Without it this always rebuilt at seed 0,
    # so a challenger asked for on a seed-1 dataset silently trained on a
    # DIFFERENT split from the retrain it was meant to be compared against — and
    # destroyed the seed-1 tree on the way past.
    split_cmd = [MAIN_PY, "-m", "groundwork.dataset.pipeline.split",
                 "--val-frac", "0.2", "--seed", str(int(req.split_seed or 0)),
                 "--project", p.slug]
    # safe_proc: 600s is a real bound only if the child can be killed. A split
    # that wedges would otherwise hold this request thread for the life of the
    # process, and this is the endpoint the Train button calls.
    if reuse_split:
        # Deliberately not run. See the reuse_split comment above: the tree is
        # already byte-for-byte what this would rebuild, and rebuilding it would
        # pull the images out from under a live retrain.
        print("[lab/train] split: SKIPPED (already current)", flush=True)
    else:
        sp = safe_proc.run(split_cmd, cwd=REPO, timeout=600)
        if sp.returncode != 0:
            return {"ok": False, "error": "split failed: "
                    + (sp.stderr or "?").strip()[-200:]}
    # --copy, not symlinks: the tree must survive the user curating images while
    # a run trains. Round I died at epoch 96 when an image was deleted and the
    # dataloader hit a dangling link — ~90 MB of copies buys immunity.
    from ... import lab_dataset
    coco_tree = lab_dataset.tree_for(paths.for_project(p), req.run_name)
    convert_cmd = [str(REPO / ".venv" / "bin" / "python"), "-m",
                   "altmodels.convert", "--match-split", "--copy",
                   # ...and the converter reads the same project's tree. Without
                   # this it globbed paths.default() while writing into project
                   # B's directory — the two halves of one job disagreeing about
                   # whose data it was.
                   "--project", p.slug,
                   "--out", str(coco_tree)]
    convert = safe_proc.run(convert_cmd, cwd=REPO, timeout=600)
    if convert.returncode != 0:
        return {"ok": False, "error": "convert failed: "
                + (convert.stderr or "?").strip()[-200:]}
    # Loud check that the tree really covers the images this machine can see. A
    # silent shortfall is the failure mode above: training succeeds and reports a
    # number about the wrong dataset, which is worse than a crash.
    try:
        from groundwork.dataset.pipeline.split import _stems_with_labels
        expect = len(_stems_with_labels(paths.for_project(p)))
        got = sum(len(json.loads((coco_tree / s /
                                  "_annotations.coco.json").read_text())["images"])
                  for s in ("train", "valid"))
        if got != expect:
            return {"ok": False, "error":
                    f"converted tree has {got} images but this machine sees "
                    f"{expect} — refusing to train on a partial dataset"}
        image_total = expect
    except Exception as e:  # noqa: BLE001 — never block a launch on the check itself
        image_total = None
        print(f"[lab/train] image-count check skipped: {e}", flush=True)
    # WHICH interpreter, WHICH trainer module and HOW MANY epochs by default are
    # facts about the family — registry.py owns them now. The ARGUMENTS are not
    # uniform (DEIM takes --size and an optional --tuning checkpoint, RTMDet
    # takes --scale-factor), and flattening genuinely different CLIs into a data
    # table would be a worse lie than the duplication it removed. So the facts
    # come from the registry and the argv stays here, per family.
    model = registry.by_name(req.arch) or registry.for_run(req.arch)
    if model is None or not model.trainer_module:
        raise HTTPException(400, f"no trainer wired for arch {req.arch!r}")
    if not model.venv_present:
        raise HTTPException(400, f"{model.venv} missing — see {model.trainer_module}")
    try:
        card, card_note = machines_mod.pick_card(
            "here", venv=model.venv, need_gb=model.peak_vram_gb,
            requested=req.card, busy=held_cards())
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    env = {**os.environ, **_card_env(card)}
    print(f"[lab/train] {req.arch} -> card {card}"
          + (f" — {card_note}" if card_note else ""), flush=True)
    # THE COMMAND IS THE MODEL'S OWN (registry.Model.train_argv). This was an
    # `if model.name == "deimv2-n": --size else: --scale-factor`, which is the
    # seventh hand-maintained per-family table and broke the moment a second
    # DEIM entry existed — deimv2-n-tv28 fell to the else branch and would have
    # been launched with a flag its trainer does not accept. Four trainers spell
    # resolution three ways; the registry knows which, and adding a family is an
    # entry rather than a branch here.
    cmd = [str(model.python), "-m", model.trainer_module,
           *model.train_argv(dataset_dir=coco_tree if model.isolated_dataset else None,
                             imgsz=req.imgsz or 640 * req.scale_factor,
                             epochs=req.epochs or model.default_epochs or 60,
                             batch=req.batch or model.default_batch,
                             run_name=req.run_name, alt_dir=_alt(p))]
    if model.predictor == "deim":
        # Keyed on the FAMILY, not the entry name, so both DEIM venvs get the
        # COCO fine-tune. Naming one of them here is how the bug above happened.
        from ....config import DATA_DIR
        tune = DATA_DIR / "vendor" / "DEIMv2" / "ckpts" / "deimv2_n_coco.pth"
        if tune.exists():
            cmd += ["--tuning", str(tune)]
    pid = _launch(cmd, _alt(p) / req.run_name / "lab.log", env=env,
                  name=req.run_name)
    # Report the image count so the caller can see WHICH dataset this run is
    # about — the stale-split bug was invisible precisely because nothing did.
    return {"ok": True, "pid": pid, "images": image_total}


@router.delete("/api/lab/train")
def cancel_challenger():
    """Scrap the challenger run holding a card on THIS machine.

    MACHINE-SCOPED (_every_alt), deliberately, and for the same reason the
    holder scan is: a card is held by whichever project got there first, so
    asking "what is on this card" through one project's tree would refuse to
    cancel a job that is plainly running. Cancelling is also the case where
    getting the scope wrong is worst — the button would report success having
    killed nothing.

    Kept next to train() so the pair is read together: anything train() refuses
    because a card is busy, this is what frees it.
    """
    from ... import cancel_ops
    return cancel_ops.challenger(_every_alt())


class ScoreReq(BaseModel):
    run_name: str


@router.post("/api/lab/score")
def score(req: ScoreReq, p=Depends(current_project)):
    """Run the proven count-eval harness on a trained alt checkpoint."""
    if not _NAME.match(req.run_name):
        raise HTTPException(422, "bad run name")
    meta_f = _alt(p) / req.run_name / "meta.json"
    if not meta_f.exists():
        raise HTTPException(404, "no meta.json — has this run trained?")
    meta = json.loads(meta_f.read_text())
    best = meta.get("best_checkpoint")
    if not best or not Path(best).exists():
        raise HTTPException(404, "no best checkpoint recorded")
    arch = meta.get("arch") or "rfdetr"   # the earliest runs predate the arch field
    # ONE description per family (groundwork/models/registry.py). This used to
    # be a four-branch if/elif that had to agree with three other tables; the
    # comment below records what that cost.
    model = registry.by_name(arch) or registry.for_run(arch)
    if model is None:
        # Loudly, rather than guessing. Until 2026-07-30 an unrecognised arch
        # fell through to the rfdetr predictor, so scoring dfine-s-1280-b would
        # have loaded a D-FINE checkpoint with RF-DETR's loader from a venv that
        # cannot even train it. A wrong number is worse than a refusal.
        raise HTTPException(400, f"no scorer wired for arch {arch!r} — add an "
                                 "entry to groundwork/models/registry.py and a "
                                 "--predictor in altmodels/harness.py")
    ok, why = registry.scorable(model)
    if not ok:
        raise HTTPException(400, why)
    py, predictor = model.python, model.predictor
    # WHICH CARD: a free one this family's venv can drive — the same
    # derivation as a training launch, because scoring loads onto a GPU too.
    try:
        card, _note = machines_mod.pick_card(
            "here", venv=model.venv, need_gb=model.peak_vram_gb,
            busy=held_cards())
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    env = {**os.environ, **_card_env(card)}
    pid = _launch([str(py), "-m", "altmodels.harness",
                   "--predictor", predictor, "--weights", best,
                   "--imgsz", str(meta["resolution"]),
                   "--run-name", req.run_name, "--source", "testset",
                   # A stage subprocess cannot inherit the project — it travels
                   # on the argv or the run silently scores the default one.
                   "--project", p.slug],
                  _alt(p) / req.run_name / "lab.log", env=env)
    return {"ok": True, "pid": pid}
