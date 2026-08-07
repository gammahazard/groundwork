"""RF-DETR-nano fine-tune on the converted COCO tree (.venv-alt, manual, overnight).

Holds the repo's gpu.lock for the WHOLE run (see altmodels.gpu) — kick off only
when the cockpit Dashboard is idle. Effective batch is fixed at 16
(batch x grad-accum), per the low-VRAM guidance; on the 8GB card sweep
resolution upward across nights: 560 (4,4) -> 728 (2,8) -> 896 (2,8; OOM ->
(1,16) -> step back down one 56-multiple).

    .venv-alt/bin/python -m altmodels.trainers.rfdetr --res 560 --batch 4 \
        --grad-accum 4 --epochs 100 --run-name rfdetr-nano-560-a
"""
from __future__ import annotations
import argparse
import os
import json
import time
from pathlib import Path

from .. import gpu

REPO = Path(__file__).resolve().parents[2]
# The challenger run tree. A run belongs to a PROJECT, and the cockpit reads
# it at <project>/alt — so the launcher passes --alt-dir and this default is
# only for a hand-run CLI in a single-project checkout. It also honours
# GW_DATA_DIR, because on a Docker install the repo is the read-only image.
_DATA = Path(os.environ.get("GW_DATA_DIR") or REPO)
ALT = _DATA / "outputs" / "alt"
DATASET_DEFAULT = ALT / "datasets" / "coco_rfdetr"


def _pretrain_for_queries(num_queries: int) -> Path | None:
    """rfdetr's COCO pretrains carry 300 query slots; asking for more breaks
    strict loading. Standard fix: tile the two query tensors (refpoint_embed,
    query_feat) so new slots start as noisy copies of learned ones — they
    diverge during fine-tuning. Cached under outputs/alt/pretrain/."""
    if num_queries <= 300:
        return None
    assert num_queries % 300 == 0, "tile in whole multiples of 300 (600, 900...)"
    out = ALT / "pretrain" / f"rf-detr-nano-q{num_queries}.pth"
    if out.exists():
        return out
    import torch
    src = Path.home() / ".roboflow" / "models" / "rf-detr-nano.pth"
    if not src.exists():
        raise SystemExit(f"pretrain not found at {src} — run a default-config "
                         "RFDETRNano() once to download it")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    reps = num_queries // 300
    # Query tensors are GROUP-MAJOR: (group_detr=13 groups x 300 queries, D).
    # Tile WITHIN each group so the layout stays [g0's 600; g1's 600; ...] —
    # a flat repeat scrambles group boundaries (the loader warned exactly that).
    groups = 13
    for key in ("refpoint_embed.weight", "query_feat.weight"):
        w = sd[key]
        per = w.shape[0] // groups
        t = w.reshape(groups, per, w.shape[1]).repeat(1, reps, 1).clone()
        t[:, per:, :] += torch.randn_like(t[:, per:, :]) * 0.01
        sd[key] = t.reshape(groups * per * reps, w.shape[1])
        print(f"[pretrain-tile] {key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}")
    # The checkpoint's own args still claim 300 — update so the loader slices
    # per-group instead of falling back to a flat slice.
    if "args" in ckpt and hasattr(ckpt["args"], "num_queries"):
        ckpt["args"].num_queries = num_queries
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    return out


def _vram_guard(kill_mib: int) -> None:
    """Same watchdog, same ceiling — now without the nvidia-smi subprocess.

    This file carried a second copy of rtmdet's poller, including its own note
    that row 0 watched the WRONG card on the dual-GPU lab. Both copies are gone;
    torch reads the current device, so there is no row to get wrong.
    """
    from ..vram_guard import start
    start(kill_mib)


def main() -> None:
    ap = argparse.ArgumentParser(description="RF-DETR-nano sidecar training.")
    ap.add_argument("--res", type=int, default=560)
    ap.add_argument("--epochs", type=int, default=100)
    # batch 2 x accum 8 = the SAME effective batch 16, but only 2 images resident
    # on the GPU at once — peak ~5-6GB instead of ~7.9. On this WSL2 laptop,
    # near-max VRAM hangs the Windows display driver (DPC watchdog BSOD,
    # 2026-07-25) — these defaults plus the guards below make that unreachable.
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--vram-kill-mib", type=int, default=None,
                    help="watchdog hard-kills the run past this (default: card total - 900)")
    ap.add_argument("--vram-cap-mib", type=int, default=None,
                    help="CUDA allocator cap — clean OOM instead of creeping toward a "
                         "driver hang (default: card total - 1900). On the 8GB WSL2 "
                         "laptop these resolve to ~6.3/7.3GB; a 24GB card gets "
                         "proportionally more while keeping the same safety margins")
    # 1200, NOT 300. A DETR head can never output more objects than it has
    # query slots, so 300 was a hard ceiling below this dataset's dense images:
    # rfdetr-nano-1120-n08011128 topped out at 297 predictions with three
    # holdout images above it, one off by 53.
    #
    # NOT 1000, which is what the other families use: _pretrain_for_queries
    # tiles the COCO pretrain's query tensors and asserts whole multiples of
    # 300, so the usable values are 600 / 900 / 1200. 1200 is the first at or
    # above the 1000 the rest of the fleet sits at.
    ap.add_argument("--num-queries", type=int, default=1200,
                    help="detection slot ceiling — an image can never count higher than "
                         "this (round 1 capped at 299 on the 349-object capture)")
    ap.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--alt-dir", type=Path, default=ALT,
                    help="where the run directory goes (the cockpit passes "
                         "the open project's alt/ tree)")
    args = ap.parse_args()
    assert args.res % 56 == 0, f"rfdetr resolution must be divisible by 56 (got {args.res})"
    assert args.batch * args.grad_accum == 16, \
        f"batch x grad-accum must equal 16 (got {args.batch}x{args.grad_accum})"
    if not (args.dataset_dir / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"no converted dataset at {args.dataset_dir} — run "
                         ".venv/bin/python -m altmodels.convert first")

    run_dir = args.alt_dir / args.run_name
    out_dir = run_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with gpu.acquire(run_dir, what=f"train_rfdetr {args.run_name}"):
        import torch
        # Dataloader workers pass batches as shared-memory FDs; on boxes with a
        # low open-files ulimit that dies as "received 0 items of ancdata"
        # (killed round 3 on the worker: soft limit 1024 vs the laptop's 1M).
        # file_system sharing sidesteps FD-passing entirely.
        torch.multiprocessing.set_sharing_strategy("file_system")
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        cap = args.vram_cap_mib or int(total_mib - 1900)
        kill = args.vram_kill_mib or int(total_mib - 900)
        _vram_guard(kill)
        torch.cuda.set_per_process_memory_fraction(cap / total_mib, 0)
        print(f"[train_rfdetr] VRAM guards: allocator capped at {cap} MiB, "
              f"watchdog kills at {kill} MiB (card total {total_mib:.0f})",
              flush=True)
        from rfdetr import RFDETRNano
        kw = dict(resolution=args.res, num_queries=args.num_queries)
        if (tiled := _pretrain_for_queries(args.num_queries)):
            kw["pretrain_weights"] = str(tiled)
        model = RFDETRNano(**kw)
        kwargs = dict(dataset_dir=str(args.dataset_dir), epochs=args.epochs,
                      batch_size=args.batch, grad_accum_steps=args.grad_accum,
                      output_dir=str(out_dir))
        try:
            model.train(**kwargs, tensorboard=False, wandb=False)
        except TypeError:              # API drift on the optional loggers
            model.train(**kwargs)
        # INSIDE the gpu block, before the allocator is torn down. RF-DETR
        # printed no memory figure either, so four completed runs left nothing
        # for web/machines.fits to reason with — and this family is one of only
        # three that can physically reach the 16 GB card, which makes "how much
        # does it need" a question with consequences rather than trivia.
        vram = gpu.peak()
        if vram:
            print(f"[train_rfdetr] peak VRAM {vram['peak_vram_gb']} GiB reserved "
                  f"({vram['peak_vram_alloc_gb']} GiB allocated)", flush=True)

    # Lightning drops a ~487MB resume .ckpt every 10 epochs — 3.5GB/run of
    # weight we never resume from. Keep only the slim best_* .pth exports.
    for fat in list(out_dir.rglob("*.ckpt")):
        fat.unlink(missing_ok=True)
    ckpts = sorted(out_dir.rglob("checkpoint*.pth"))
    best = next((c for c in ckpts if "best" in c.name and "ema" in c.name),
                next((c for c in ckpts if "best" in c.name), ckpts[-1] if ckpts else None))
    meta = {**vram, "run": args.run_name, "arch": "rfdetr-nano", "license": "Apache-2.0",
            "resolution": args.res, "epochs": args.epochs, "batch": args.batch,
            "grad_accum": args.grad_accum, "effective_batch": 16,
            "dataset_dir": str(args.dataset_dir),
            "train_seconds": round(time.time() - t0, 1),
            "best_checkpoint": str(best) if best else None,
            "checkpoints": [str(c) for c in ckpts], "created": time.time()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[train_rfdetr] done in {meta['train_seconds']}s -> {best}")
    print(f"[train_rfdetr] next: .venv-alt/bin/python -m altmodels.harness "
          f"--predictor rfdetr --weights {best} --imgsz {args.res} "
          f"--run-name {args.run_name} --restrict-ref")


if __name__ == "__main__":
    main()
