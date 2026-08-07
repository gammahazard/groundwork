"""DEIMv2 challenger trainer (.venv-deim, the worker 3090, vendor repo ~/DEIMv2).

DEIM's thesis (dense one-to-one matching for fast convergence) targets exactly
the crowded-scene weakness that killed RF-DETR here — that's why it earns the
audition despite being DETR-family. Non-negotiables discovered up front:
  - num_queries/num_top_queries ship at 300: a 406-object capture is UNREPRESENTABLE.
    We raise both to 600 (query embeddings re-init when tuning from COCO).
  - warmup_iter ships at 2000: our whole run is ~1500 iters — unadjusted, the
    LR never finishes warming up. Scaled to 200.
Builds the classic-COCO layout DEIM wants from the existing converted tree,
generates an objects config next to the vendor configs, and runs their train.py.

    ulimit -n 65535
    GW_ALLOW_PARALLEL=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        .venv-deim/bin/python -m altmodels.trainers.deim --epochs 90 \
        --run-name deimv2-n-1280-a [--tuning ckpts/deimv2_n_coco.pth]
"""
from __future__ import annotations
import argparse
import sys
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .. import gpu
from .rtmdet import _vram_guard   # per-card smi row watchdog, reuse

REPO = Path(__file__).resolve().parents[2]
# The challenger run tree. A run belongs to a PROJECT, and the cockpit reads
# it at <project>/alt — so the launcher passes --alt-dir and this default is
# only for a hand-run CLI in a single-project checkout. It also honours
# GW_DATA_DIR, because on a Docker install the repo is the read-only image.
_DATA = Path(os.environ.get("GW_DATA_DIR") or REPO)
ALT = _DATA / "outputs" / "alt"
# LEGACY DEFAULTS ONLY. Both were fixed module constants until 2026-08-02, and
# between them they were the reason DEIM could never train beside another
# challenger: a second launch rebuilt coco_rfdetr out from under a live run, and
# coco_deim was a second shared directory doing the same thing one layer down.
# --dataset-dir now names the source tree per run (web/lab_dataset.py), and the
# classic layout is built INSIDE it. These remain as the fallback for a hand-run
# CLI that names neither.
SRC_DS = ALT / "datasets" / "coco_rfdetr"      # roboflow-style tree (convert.py)
DEIM_DS = ALT / "datasets" / "coco_deim"       # classic layout DEIM wants


def _deim_tree(src: Path) -> Path:
    """Where the classic-COCO layout goes, for a given source tree.

    A SUBDIRECTORY of the source, not a sibling. lab_dataset.reapable() finds a
    run's tree by name (coco_<run>) and would not recognise a sibling called
    coco_deim_<run> — it would parse the owner as "deim_<run>", find no such run,
    and leave it on disk forever. Nested, it is reclaimed with its parent and
    cannot be orphaned.

    Legacy source (the shared coco_rfdetr) keeps the old shared destination, so
    nothing about a hand-run CLI changes.
    """
    return DEIM_DS if src == SRC_DS else src / "_deim_classic"
VENDOR = Path.home() / "DEIMv2"
# THE INTERPRETER THAT LAUNCHED US, not a hardcoded venv.
#
# This was `REPO / ".venv-deim" / "bin" / "python"`, and it silently undid the
# entire point of having a second DEIM environment: `.venv-deim13/bin/python -m
# altmodels.trainers.deim` re-exec'd into `.venv-deim`'s torchrun, so a run
# meant to test torch 2.13 trained on 2.5.1. Both entries produced runs 0.3s
# apart because they WERE the same run — caught 2026-07-31 by training both and
# noticing the timings matched.
PYBIN = Path(sys.executable)

CONFIG_TEMPLATE = """__include__: ['./deimv2_hgnetv2_{variant}_coco.yml']

output_dir: {out}
num_classes: 1
remap_mscoco_category: False

DEIMTransformer:
  num_queries: {queries}

PostProcessor:
  num_top_queries: {queries}

epoches: {epochs}
flat_epoch: {flat}
no_aug_epoch: {no_aug}
warmup_iter: 200          # tiny dataset: default 2000 outlives the whole run
eval_spatial_size: [{size}, {size}]

ema:
  decay: {ema_decay}      # 0.9999 averages ~2700 steps; at 13 steps/epoch that
                          # is ~190 epochs — longer than the whole no-aug phase,
                          # so the shipped (EMA) weights can't see it. Scale it.

optimizer:
  lr: 0.0002              # theirs 0.0008 @ total_batch 32 -> linear to batch 8
  params:
    - params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.0001
    - params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: 0.0001
      weight_decay: 0.
    - params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.

train_dataloader:
  total_batch_size: {batch}
  num_workers: 4
  dataset:
    img_folder: {ds}/images/train/
    ann_file: {ds}/annotations/instances_train.json
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.5}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [{size}, {size}], }}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        name: stop_epoch
        epoch: [4, {flat}, {stop}]
        ops: ['Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
  collate_fn:
    type: BatchImageCollateFunction
    base_size: {size}
    base_size_repeat: 3
    ema_restart_decay: 0.9999
    mixup_epochs: [4, {flat}]
    stop_epoch: {stop}
    copyblend_prob: 0.4
    copyblend_epochs: [4, {flat}]
    area_threshold: 100
    num_objects: 3
    with_expand: True
    expand_ratios: [0.1, 0.25]

val_dataloader:
  total_batch_size: 8
  num_workers: 4
  dataset:
    img_folder: {ds}/images/val/
    ann_file: {ds}/annotations/instances_val.json
    transforms:
      ops:
        - {{type: Resize, size: [{size}, {size}], }}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}

DEIMCriterion:
  matcher:
    change_matcher: True
    iou_order_alpha: 4.0
    matcher_change_epoch: {stop}
"""


def _build_classic_tree(src: Path, dest: Path,
                        include_distill: bool = False,
                        distill_dir: Path | None = None) -> None:
    """roboflow layout -> classic COCO layout, symlinked.
    Self-check: image counts in json vs on disk must match, per split.
    include_distill merges the champion-taught bucket (dataset/distill/) into
    TRAIN only — challenger food, never touching raw/'s hand-verified purity.

    Both paths are ARGUMENTS now. They were module constants, which is what made
    two DEIM runs — or a DEIM run beside any other challenger — impossible: this
    rmtree's `dest`, and every run pointed at the same one."""
    if not (src / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"no converted tree at {src} — run altmodels.convert first")
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "annotations").mkdir(parents=True)
    for split, name in (("train", "train"), ("valid", "val")):
        img_dir = dest / "images" / name
        img_dir.mkdir(parents=True)
        n = 0
        for p in (src / split).iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                (img_dir / p.name).symlink_to(p.resolve())
                n += 1
        ann = json.loads((src / split / "_annotations.coco.json").read_text())
        assert len(ann["images"]) == n, \
            f"{split}: json lists {len(ann['images'])} images, disk has {n}"
        if include_distill and name == "train":
            from PIL import Image
            # NAMED, not derived. The bucket belongs to the PROJECT's dataset
            # (teacher_label writes pp.DATASET_DIR/"distill"), which cannot be
            # inferred from the converted tree — the old repo-level constant was
            # from before projects existed and quietly added nothing on every
            # install. No --distill-dir means no distillation, said out loud.
            di = (distill_dir / "images") if distill_dir else None
            dl = (distill_dir / "labels") if distill_dir else None
            if dl is None:
                print("[deim-ds] --include-distill given without --distill-dir "
                      "— nothing to merge", flush=True)
            img_id = max((im["id"] for im in ann["images"]), default=0)
            box_id = max((a["id"] for a in ann["annotations"]), default=0)
            added = 0
            for lbl in sorted(dl.glob("*.txt")) if dl and dl.exists() else []:
                # NOT `src` — that is this function's SOURCE TREE parameter, and
                # rebinding it here broke the val split that runs after train
                # (it went looking for valid/ under a distill image file).
                img = next((q for q in di.glob(f"{lbl.stem}.*")), None)
                if img is None:
                    continue
                (img_dir / img.name).symlink_to(img.resolve())
                W, H = Image.open(img).size
                img_id += 1
                ann["images"].append({"id": img_id, "file_name": img.name,
                                      "width": W, "height": H})
                for ln in lbl.read_text().splitlines():
                    if not ln.strip():
                        continue
                    _, cx, cy, bw, bh = (float(x) for x in ln.split()[:5])
                    box_id += 1
                    ann["annotations"].append({
                        "id": box_id, "image_id": img_id, "category_id": 0,
                        "bbox": [(cx - bw / 2) * W, (cy - bh / 2) * H,
                                 bw * W, bh * H],
                        "area": bw * W * bh * H, "iscrowd": 0})
                added += 1
                n += 1
            print(f"[deim-ds] +distill: {added} champion-taught image(s) into train")
        # DEIM (remap_mscoco_category: False) uses category_id AS the label:
        # with num_classes=1 only id 0 is legal. Our tree ships id 1 ('object')
        # -> device-side assert in the loss. Remap to 0-based here.
        for c in ann.get("categories", []):
            c["id"] = 0
        for a in ann.get("annotations", []):
            a["category_id"] = 0
        (dest / "annotations" / f"instances_{name}.json").write_text(
            json.dumps(ann), encoding="utf-8")
        print(f"[deim-ds] {name}: {n} images, {len(ann['annotations'])} boxes")


def main() -> None:
    ap = argparse.ArgumentParser(description="DEIMv2 sidecar training.")
    ap.add_argument("--variant", default="n", choices=["atto", "femto", "pico", "n"])
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--queries", type=int, default=1000,
                    help="detection ceiling — a capture can exceed the stock 300")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tuning", type=Path, default=None,
                    help="COCO checkpoint to fine-tune from (-t); query embeds re-init")
    ap.add_argument("--distill-dir", type=Path, default=None,
                    help="the project's distill/ bucket (teacher-labelled "
                         "images), merged into TRAIN with --include-distill")
    ap.add_argument("--include-distill", action="store_true",
                    help="merge champion-taught distill/ images into TRAIN")
    ap.add_argument("--no-aug-epochs", type=int, default=12,
                    help="final clean-image (stage 2) epochs. Vendor's 12 is "
                         "~44k steps on COCO but only ~156 here — raise it")
    ap.add_argument("--ema-decay", type=float, default=0.9999,
                    help="EMA decay; the vendor's 0.9999 averages far longer "
                         "than our whole run (see CONFIG_TEMPLATE note)")
    ap.add_argument("--arch", default="",
                    help="registry family name to record in meta.json "
                         "(default: deimv2-<variant>). The cockpit passes it so "
                         "a run trained in .venv-deim13 is not filed under the "
                         "entry that uses .venv-deim.")
    ap.add_argument("--dataset-dir", type=Path, default=SRC_DS,
                    help="the converted roboflow-layout tree to train from. The "
                         "cockpit passes this run's OWN tree (web/lab_dataset.py) "
                         "so two challengers cannot rebuild each other's data; "
                         "the default is the legacy shared tree for a hand CLI.")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--alt-dir", type=Path, default=ALT,
                    help="where the run directory goes (the cockpit passes "
                         "the open project's alt/ tree)")
    args = ap.parse_args()
    if not VENDOR.exists():
        raise SystemExit(f"vendor repo missing at {VENDOR} (git clone DEIMv2)")

    run_dir = args.alt_dir / args.run_name
    out_dir = run_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with gpu.acquire(run_dir, what=f"train_deim {args.run_name}"):
        # THE LAUNCHER DOES NOT NEED A CUDA CONTEXT, and on WSL2 having one is
        # not free. get_device_properties(0) below initialises CUDA in a process
        # whose remaining job is to write a yaml file and fork torchrun — so a
        # DEIM run holds TWO live contexts on one paravirtualised card. No other
        # family does: rtmdet and rfdetr make the same call, but then train in
        # that same process. Opt-in while it is unproven; see deim_entry.py for
        # the other half, and docs/machines.md for why a spare context on /dev/dxg is
        # worth deleting.
        if os.environ.get("GW_DEIM_CHILD_GUARD") == "1":
            print("[train_deim] launcher stays CUDA-free — VRAM guard and "
                  "sharing strategy run in the torchrun child", flush=True)
        else:
            import torch
            torch.multiprocessing.set_sharing_strategy("file_system")
            total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
            row = int((os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
            _vram_guard(int(total_mib - 900), row)
            print(f"[train_deim] card {torch.cuda.get_device_name(0)} | watchdog "
                  f"{int(total_mib - 900)} MiB (smi row {row})", flush=True)

        deim_ds = _deim_tree(args.dataset_dir)
        print(f"[train_deim] source {args.dataset_dir} -> classic {deim_ds}",
              flush=True)
        _build_classic_tree(args.dataset_dir, deim_ds,
                            include_distill=args.include_distill,
                            distill_dir=args.distill_dir)
        flat = 4 + args.epochs // 2
        # stop_epoch (augmentation off) and no_aug_epoch are the SAME boundary
        # seen from either end — they must move together or stage 2 desyncs
        stop = max(1, args.epochs - args.no_aug_epochs)
        cfg_text = CONFIG_TEMPLATE.format(
            variant=args.variant, out=str(out_dir), ds=str(deim_ds),
            epochs=args.epochs, flat=flat, stop=stop,
            no_aug=args.no_aug_epochs, ema_decay=args.ema_decay,
            batch=args.batch, size=args.size, queries=args.queries)
        cfg_path = VENDOR / "configs" / "deimv2" / f"gw_{args.run_name}.yml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        (run_dir / "run_config.yml").write_text(cfg_text, encoding="utf-8")
        print(f"[train_deim] config -> {cfg_path}")

        # their backbone calls torch.distributed.get_rank() unconditionally —
        # torchrun (even single-process) is mandatory, plain python crashes
        #
        # …and we run the vendor entry through altmodels/deim_entry.py rather
        # than train.py directly, so the torchvision v2 compat shim is applied
        # inside the process that imports their transforms. Without it a
        # .venv-deim13 run dies in the first DataLoader batch and torchrun
        # swallows the traceback into a bare ChildFailedError. The wrapper is a
        # no-op on the old venv, so BOTH entries take the same path — a wrapper
        # used by one and not the other would be a difference between them that
        # is not the venv, which is the only thing they are meant to differ in.
        entry = REPO / "altmodels" / "deim_entry.py"
        cmd = [str(PYBIN.parent / "torchrun"), "--master_port=7777",
               "--nproc_per_node=1", str(entry), "-c", str(cfg_path),
               "--use-amp", "--seed", str(args.seed)]
        if args.tuning:
            cmd += ["-t", str(args.tuning)]
        r = subprocess.run(cmd, cwd=str(VENDOR))
        if r.returncode != 0:
            raise SystemExit(f"[train_deim] train.py exited {r.returncode}")

    # DEIM hoards a checkpoint every few epochs (~30MB each, GBs/run) — keep
    # only the slim faces (best_stg*, last) like the other trainers do.
    for fat in out_dir.glob("checkpoint*.pth"):
        fat.unlink(missing_ok=True)
    ckpts = sorted(out_dir.rglob("*.pth"))
    # DEIM trains in TWO stages and saves a champion for each:
    #   best_stg1 = best val AP during the heavy-augmentation phase
    #   best_stg2 = best val AP during the final no_aug_epoch refinement —
    #               the vendor's INTENDED finished model
    #   last      = final state after stage 2
    # The old `next(c for c in ckpts if "best" in c.name)` picked whichever
    # sorted first — always best_stg1 — so stage 2 was never examined in ANY
    # round (found 2026-07-28). That also explains "the crown lies": last.pth
    # kept beating the "best" because it carried the no-aug refinement and
    # best_stg1 did not. Prefer stg2, and record every face so the exam scripts
    # can score all of them.
    by_name = {c.stem: c for c in ckpts}
    best = (by_name.get("best_stg2") or by_name.get("best_stg1")
            or (ckpts[-1] if ckpts else None))
    # THE FAMILY THIS RUN BELONGS TO. It was always `deimv2-{variant}`, so a
    # deimv2-n-tv28 run recorded itself as `deimv2-n` and the ledger attributed
    # it to the other entry — two different torch builds compared as one family,
    # which is the exact thing a separate registry entry exists to prevent.
    meta = {"run": args.run_name, "arch": args.arch or f"deimv2-{args.variant}",
            "license": "Apache-2.0", "resolution": args.size,
            "epochs": args.epochs, "batch": args.batch,
            "num_queries": args.queries, "seed": args.seed,
            "tuned_from": str(args.tuning) if args.tuning else None,
            "train_seconds": round(time.time() - t0, 1),
            "best_checkpoint": str(best) if best else None,
            "checkpoints": {k: str(v) for k, v in by_name.items()},
            "created": time.time()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[train_deim] done in {meta['train_seconds']}s -> {best}")


if __name__ == "__main__":
    main()
