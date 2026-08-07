"""RTMDet-tiny fine-tune on the converted COCO tree (.venv-mmdet, the worker 3090).

The mmcv build gate that timeboxed RTMDet out of round 1 passes on Ampere
(prebuilt cu121 wheels) — this trainer only ever runs on the 3090. Launch with
the card pinned and the parallel vouch, so it coexists with a cockpit retrain
on the 5070 Ti (per-card locks):

    ulimit -n 65535
    GW_ALLOW_PARALLEL=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        .venv-mmdet/bin/python -m altmodels.trainers.rtmdet --epochs 100 \
        --run-name rtmdet-tiny-1280-a

Config surgery over the stock rtmdet_tiny_8xb32-300e_coco.py: one class, our
COCO tree, every pipeline scale doubled (640 -> 1280 — the objects are tiny),
epoch schedule compressed, lr auto-scaled to the real batch size.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .. import gpu, mmdet_zoo

REPO = Path(__file__).resolve().parents[2]
# The challenger run tree. A run belongs to a PROJECT, and the cockpit reads
# it at <project>/alt — so the launcher passes --alt-dir and this default is
# only for a hand-run CLI in a single-project checkout. It also honours
# GW_DATA_DIR, because on a Docker install the repo is the read-only image.
_DATA = Path(os.environ.get("GW_DATA_DIR") or REPO)
ALT = _DATA / "outputs" / "alt"
DATASET_DEFAULT = ALT / "datasets" / "coco_rfdetr"   # same tree RF-DETR used
PRETRAIN = ALT / "pretrain"
# Which mmdet model this run trains — the zoo keeps the config/ckpt names so
# adding a candidate never touches this file (altmodels/mmdet_zoo.py).
DEFAULT_ARCH = "rtmdet-tiny"

_SCALE_KEYS = ("img_scale", "scale", "crop_size", "size")


def _scale_pipeline(transforms, factor: int) -> None:
    """Double every geometric size in a pipeline in place (640-era -> 1280)."""
    for tr in transforms or []:
        for k in _SCALE_KEYS:
            v = tr.get(k)
            if isinstance(v, (tuple, list)) and len(v) == 2:
                tr[k] = tuple(int(x * factor) for x in v)


def _vram_guard(kill_mib: int, smi_row: int = 0) -> None:
    """Kept as a thin shim so deim.py's existing import keeps working.

    The implementation moved to altmodels/vram_guard.py and no longer shells out
    to nvidia-smi every 20s -- that watchdog wedged in D-state under WSL2 and was
    hanging the driver it was written to protect. `smi_row` is now IGNORED and
    accepted only so callers do not have to change: torch reads the current
    device, which under CUDA_VISIBLE_DEVICES pinning is our card by construction.
    """
    from ..vram_guard import start
    start(kill_mib)


def _pipeline_owner(ds):
    """The dataset level whose `pipeline` mmengine will actually run.

    NOT the same level as `_inner`, and conflating them is what killed
    yolox-tiny with `assert 'mix_results' in results`.

    Two wrappers, opposite answers:
      RepeatDataset        has NO pipeline key -> descend; setting one on the
                           wrapper raises "got an unexpected keyword 'pipeline'".
      MultiImageMixDataset OWNS the mixing pipeline (Mosaic/MixUp) and feeds the
                           inner dataset's output into it as `mix_results`. Put
                           Mosaic on the INNER dataset instead and nothing ever
                           populates mix_results, so every worker asserts.

    So: descend only while this level has no `pipeline` of its own. That is the
    outermost level that owns one, which is exactly what mmengine executes.
    Data paths still go to the leaf via `_inner` — a wrapper has no data_root.
    """
    while "pipeline" not in ds and "dataset" in ds:
        ds = ds.dataset
    return ds


def _dataset(split_dir: Path) -> dict:
    return dict(data_root=str(split_dir) + "/",
                ann_file="_annotations.coco.json",
                data_prefix=dict(img=""),
                metainfo=dict(classes=("object",)))


def _inner(ds):
    """The real CocoDataset inside any wrapper.

    YOLOX (and anything else using Mosaic/MixUp) nests its dataset inside a
    MultiImageMixDataset: the WRAPPER owns the mixing pipeline, the INNER one
    owns data_root/ann_file. Pointing our paths at the wrapper raises
    `MultiImageMixDataset.__init__() got an unexpected keyword 'data_root'`.
    Descend until we find the leaf."""
    while "dataset" in ds:
        ds = ds.dataset
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description="RTMDet-tiny sidecar training.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--scale-factor", type=int, default=2,
                    help="multiply every stock 640-era pipeline size (2 -> 1280)")
    ap.add_argument("--vram-kill-mib", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed = reproducible runs, controlled A/Bs "
                         "(B-vs-C proved unseeded variance is huge)")
    ap.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    ap.add_argument("--arch", default=DEFAULT_ARCH,
                    help="mmdet model from altmodels/mmdet_zoo.py "
                         "(default: the original rtmdet-tiny)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--alt-dir", type=Path, default=ALT,
                    help="where the run directory goes (the cockpit passes "
                         "the open project's alt/ tree)")
    args = ap.parse_args()
    if not (args.dataset_dir / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"no converted dataset at {args.dataset_dir} — run "
                         ".venv/bin/python -m altmodels.convert first")
    spec = mmdet_zoo.get(args.arch)
    config = PRETRAIN / f"{spec['config']}.py"
    if not config.exists():
        raise SystemExit(f"no config at {config} — run: .venv-mmdet/bin/mim "
                         f"download mmdet --config {spec['config']} "
                         f"--dest {PRETRAIN}/")
    ckpt = next(PRETRAIN.glob(spec["ckpt"]), None)
    if not ckpt:
        # NAME THE GLOB AND SHOW WHAT IS THERE. mmdet does not always publish a
        # checkpoint under its config's name — centernet's config is
        # `centernet_r18-dcnv2_8xb16-...` and its weights are
        # `centernet_resnet18_dcnv2_...`, so the download SUCCEEDS and the glob
        # still misses. "no pretrain .pth" sent you looking for a failed
        # download that never happened.
        have = sorted(f.name for f in PRETRAIN.glob("*.pth"))
        raise SystemExit(
            f"no COCO pretrain matching {spec['ckpt']!r} in {PRETRAIN}.\n"
            f"  present: {', '.join(have) or '(none)'}\n"
            f"  fetch:   mim download mmdet --config {spec['config']} "
            f"--dest {PRETRAIN}/\n"
            f"  note:    mmdet sometimes names the weights differently from the "
            f"config — if the download succeeded, fix the 'ckpt' glob in "
            f"altmodels/mmdet_zoo.py")

    run_dir = args.alt_dir / args.run_name
    work_dir = run_dir / "train"
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with gpu.acquire(run_dir, what=f"train_rtmdet {args.run_name}"):
        import torch
        torch.multiprocessing.set_sharing_strategy("file_system")  # low-ulimit boxes
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        kill = args.vram_kill_mib or int(total_mib - 900)
        row = int((os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
        _vram_guard(kill, row)
        torch.cuda.set_per_process_memory_fraction((total_mib - 1900) / total_mib, 0)
        print(f"[train_rtmdet] card {torch.cuda.get_device_name(0)} | allocator "
              f"cap {int(total_mib - 1900)} MiB | watchdog {kill} MiB (smi row {row})",
              flush=True)

        from mmengine.config import Config
        from mmengine.runner import Runner
        cfg = Config.fromfile(str(config))
        E, F = args.epochs, args.scale_factor
        cfg.work_dir = str(work_dir)
        cfg.load_from = str(ckpt)
        cfg.randomness = dict(seed=args.seed, deterministic=False)
        cfg.model.bbox_head.num_classes = 1
        # A capture can exceed 300 objects; the stock test_cfg caps detections at
        # 300/image — the same silent ceiling that sank RF-DETR round 1.
        # DETECTION CEILING. max_per_img alone is NOT enough for every family.
        # CenterNet takes the top-k heatmap peaks FIRST and only then applies
        # max_per_img, so its stock topk=100 is the binding limit and a
        # max_per_img of 1000 is unreachable. centernet-1280-n08010832 scored
        # MAE 22.03 entirely because of it: its maximum prediction across all 65
        # holdout images was exactly 100, and all 14 images with more than 100
        # objects predicted precisely 100 (a 349-object image came back as 100).
        #
        # Images reach ~500 objects here, so both knobs go well clear of that.
        cfg.model.test_cfg.max_per_img = 1000
        if "topk" in cfg.model.test_cfg:
            cfg.model.test_cfg.topk = 1000
        # Our tree + single class, all three loaders (valid doubles as test).
        _inner(cfg.train_dataloader.dataset).update(
            _dataset(args.dataset_dir / "train"))
        for dl in (cfg.val_dataloader, cfg.test_dataloader):
            _inner(dl.dataset).update(_dataset(args.dataset_dir / "valid"))
        val_ann = str(args.dataset_dir / "valid" / "_annotations.coco.json")
        cfg.val_evaluator.ann_file = val_ann
        cfg.test_evaluator.ann_file = val_ann
        # Dense captures: the COCO metric's default 100-detection cap starves
        # recall on 300-object images and misleads best-checkpoint selection.
        cfg.val_evaluator.proposal_nums = (100, 300, 1000)
        cfg.test_evaluator.proposal_nums = (100, 300, 1000)
        # Objects are ~10-25px at the stock 640 — double every pipeline size.
        _scale_pipeline(cfg.train_pipeline, F)
        _scale_pipeline(getattr(cfg, "train_pipeline_stage2", None), F)
        _scale_pipeline(cfg.test_pipeline, F)
        # THROUGH `_inner`, like the dataset update fifteen lines above. Some
        # configs wrap the train set in RepeatDataset (centernet does), and a
        # wrapper has no `pipeline` — setting one there gives
        # "RepeatDataset.__init__() got an unexpected keyword argument
        # 'pipeline'" at build time, minutes in. The unwrapper was already here
        # and already used for the very same object; these two lines just did
        # not go through it.
        _pipeline_owner(cfg.train_dataloader.dataset).pipeline = cfg.train_pipeline
        for dl in (cfg.val_dataloader, cfg.test_dataloader):
            _pipeline_owner(dl.dataset).pipeline = cfg.test_pipeline
        # Batch + honest lr for it (stock schedule assumes 8 GPUs x 32).
        cfg.train_dataloader.batch_size = args.batch
        cfg.train_dataloader.num_workers = 4
        cfg.val_dataloader.batch_size = 2
        cfg.auto_scale_lr = dict(enable=True, base_batch_size=256)
        # Compress the 300-epoch schedule to E: cosine over the back half,
        # mosaic off for the last 20 (the stage-2 pipeline switch).
        switch = max(1, E - 20)
        cfg.train_cfg.update(max_epochs=E, val_interval=10,
                             dynamic_intervals=[(switch, 1)])
        for s in cfg.param_scheduler:
            if s.get("type") == "CosineAnnealingLR":
                s.update(begin=E // 2, end=E, T_max=E - E // 2)
        for h in getattr(cfg, "custom_hooks", []):
            if h.get("type") == "PipelineSwitchHook":
                h["switch_epoch"] = switch
                _scale_pipeline(h.get("switch_pipeline"), F)
        cfg.default_hooks.checkpoint.update(interval=10, max_keep_ckpts=2,
                                            save_best="coco/bbox_mAP")
        # "Watch it see": draw predictions-vs-GT on every 10th val image each
        # val pass -> work_dir/<ts>/vis_data/vis_image/ (the Lab watch view).
        cfg.default_hooks.visualization = dict(type="DetVisualizationHook",
                                               draw=True, interval=10)
        Runner.from_cfg(cfg).train()

    best = next(iter(sorted(work_dir.glob("best_coco_bbox_mAP_epoch_*.pth"))), None) \
        or next(iter(sorted(work_dir.glob("epoch_*.pth"))), None)
    meta = {"run": args.run_name, "arch": args.arch,
            "license": spec["license"], "mmdet_config": spec["config"],
            "resolution": 640 * args.scale_factor, "epochs": E,
            "batch": args.batch, "dataset_dir": str(args.dataset_dir),
            "train_seconds": round(time.time() - t0, 1),
            "best_checkpoint": str(best) if best else None, "created": time.time()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[train_rtmdet] done in {meta['train_seconds']}s -> {best}")
    print(f"[train_rtmdet] next: GW_ALLOW_PARALLEL=1 CUDA_DEVICE_ORDER=PCI_BUS_ID "
          f"CUDA_VISIBLE_DEVICES=1 .venv-mmdet/bin/python -m altmodels.harness "
          f"--predictor rtmdet --weights {best} --imgsz {640 * args.scale_factor} "
          f"--run-name {args.run_name} --restrict-ref")


if __name__ == "__main__":
    main()
