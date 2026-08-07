"""Stage 3: train a single-class YOLO-nano object detector on the split dataset.

Tuned for MANY tiny objects in dense clusters, distilled from ~50 images:
  - high imgsz (1280) so small objects survive downsampling,
  - aggressive mosaic/mixup/HSV to turn ~50 images into 500+ virtual variants,
  - close_mosaic near the end so the model finishes on real (un-collaged) layouts,
  - FIXED length (no early stop) so every run is comparable — see TrainCfg,
  - AMP on to use the Blackwell tensor cores.

    python -m groundwork.dataset.pipeline.train                 # yolov8n @ 1280
    python -m groundwork.dataset.pipeline.train --model yolov12n --imgsz 960
"""
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass

from .. import paths


def _next_run_name(base: str, pp=None) -> str:
    """Explicit run numbering (yolov8n-28, -29, …). ultralytics' own
    auto-increment restarts at the bare name once old run dirs are pruned
    (seen live: a run landed as plain 'yolov8n' after a history cleanup) —
    we continue from the highest existing suffix instead."""
    n = 0
    for p in pp.RUNS_DIR.glob(f"{base}*") if pp.RUNS_DIR.exists() else []:
        m = re.fullmatch(rf"{re.escape(base)}-(\d+)", p.name)
        if m:
            n = max(n, int(m.group(1)))
        elif p.name == base:
            n = max(n, 1)
    return f"{base}-{n + 1}"


@dataclass
class TrainCfg:
    model: str = "yolov8n.pt"     # or "yolov12n.pt" — try both, keep better val
    imgsz: int = 1280
    # Every run trains the SAME number of epochs. Early stopping used to end runs
    # anywhere between epoch 76 and 260 depending on which images the random val
    # split happened to get, and the short ones shipped undertrained weights that
    # emit two offset boxes per object (they clear NMS at IoU<0.5 -> stacked dots).
    epochs: int = 250
    patience: int = 0             # 0 => ultralytics disables early stopping
    batch: float = -1             # -1 => ultralytics auto-picks for the VRAM
    # Periodic checkpoints (epoch25.pt, epoch50.pt, …) — native ultralytics
    # feature, zero effect on training. Fuel for the post-run "watch it learn"
    # probe timelapse (dataset/viz/timelapse.py), which prunes them afterwards.
    save_period: int = 25
    # augmentation (heavy: 50 imgs -> 500+ virtual variants)
    mosaic: float = 1.0
    # Last N epochs train on real layouts. This MUST be reachable: at the old
    # 300/10 no run ever got past epoch 260, so mosaic never actually closed and
    # every shipped model finished on collages.
    close_mosaic: int = 40
    mixup: float = 0.15
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0          # images are top-down; rotation aug optional
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    flipud: float = 0.5           # images have no up/down, so vertical flips are valid


def train(cfg: TrainCfg, pp=None):
    from ultralytics import YOLO   # imported lazily so the rest of the pkg has no hard dep

    if not pp.DATA_YAML.exists():
        raise SystemExit(f"{pp.DATA_YAML} missing — run split first.")

    model = YOLO(cfg.model)
    # Live batch snapshots for the 🔬 Watch modal (see live_trainer.py) —
    # read-only viz via the documented trainer= hook; stock training if the
    # import breaks on a future ultralytics.
    try:
        from ..viz.live_trainer import LiveVizTrainer as _trainer
    except Exception:  # noqa: BLE001
        _trainer = None
    # ultralytics wants an int for an explicit batch count; -1 (auto) or a
    # 0<f<1 VRAM fraction stay float.
    batch = int(cfg.batch) if cfg.batch >= 1 else cfg.batch
    # WRAPPED so a CRASH still reports what it reached. The peak used to be
    # printed after train() returned, so the one run whose VRAM you most want to
    # know — the one that died reaching for it — recorded nothing, and the
    # configuration stayed "unmeasured", which the fit check treats as allowed.
    # A failure that teaches nothing invites the same failure again.
    #
    # A hypervisor fault still writes nothing: the process is gone. This covers
    # OOM and every Python-level death, which is the reachable case.
    results = None
    try:
        results = model.train(
        trainer=_trainer,
        data=str(pp.DATA_YAML),
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        patience=cfg.patience,
        save_period=cfg.save_period,
        batch=batch,
        amp=True,                 # mixed precision -> Blackwell tensor cores
        cache=True,
        # `project=` here is ultralytics' word for "where do run dirs go" — it
        # has nothing to do with groundwork.project. Unfortunate collision;
        # renaming ours would be worse.
        project=str(pp.RUNS_DIR),
        name=_next_run_name(cfg.model.replace(".pt", ""), pp),
        mosaic=cfg.mosaic,
        close_mosaic=cfg.close_mosaic,
        mixup=cfg.mixup,
        hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
        degrees=cfg.degrees, translate=cfg.translate, scale=cfg.scale,
        fliplr=cfg.fliplr, flipud=cfg.flipud,
        )
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_reserved() / 1e9
                ok = results is not None
                print(f"[train] peak VRAM {peak:.2f} GB"
                      + ("" if ok else "  (run FAILED at this point)"),
                      flush=True)
                # RECORDED EVEN ON FAILURE — see vram_log. The ledger only takes
                # completed runs, so without this the single most useful VRAM
                # observation on this system (the run that asked for more than
                # the card had) teaches nothing and the configuration is offered
                # again.
                from . import vram_log
                card_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / 2**30)
                vram_log.record(cfg.model.replace(".pt", ""), cfg.imgsz,
                                int(cfg.batch) if cfg.batch >= 1 else None,
                                round(peak, 2), ok, card_gb)
        except Exception:  # noqa: BLE001
            pass
    print(f"[train] done. weights: {pp.RUNS_DIR}/{cfg.model.replace('.pt','')}/weights/best.pt")
    return results


def main() -> None:
    d = TrainCfg()
    ap = argparse.ArgumentParser(description="Train YOLO-nano object counter.")
    ap.add_argument("--model", default=d.model.replace(".pt", ""),
                    help="yolov8n | yolov12n | path to .pt")
    ap.add_argument("--imgsz", type=int, default=d.imgsz)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch", type=float, default=d.batch)
    from ...project import add_argument, load
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    # SAY WHICH DATASET. A run that trained the wrong data is this repo's most
    # expensive failure and it never announces itself — the lab burned 7 GPU-hours
    # on a day-old tree in silence. The image count comes from the split dirs
    # because that is literally what the trainer is about to read.
    n_train = len(list((pp.LABELS_DIR / "train").glob("*.txt")))
    n_val = len(list((pp.LABELS_DIR / "val").glob("*.txt")))
    print(f"[train] project {proj.slug} ({proj.name}) | classes {list(proj.classes)} "
          f"| {proj.dataset_root} | {n_train + n_val} images ({n_train} train / {n_val} val)")
    model = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    train(TrainCfg(model=model, imgsz=args.imgsz, epochs=args.epochs,
                   batch=args.batch), pp)


if __name__ == "__main__":
    main()
