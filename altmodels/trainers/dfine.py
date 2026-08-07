"""D-FINE challenger trainer — HuggingFace transformers, no vendor repo.

Why this one earns a slot (research 2026-07-29):
  - Apache-2.0 for code AND weights, no DINOv3 entanglement.
  - `dfine-small-obj2coco` is pretrained on Objects365 (2M images) THEN COCO.
    Better pretraining is the only lever that has ever moved a challenger here
    (DEIM: 0.077 cold -> 0.03 COCO-finetuned), and we cannot easily get more
    object images — but we can start from a model that has seen 2M more images.
  - It lives in the MAIN .venv (torch 2.13/cu130), so unlike .venv-deim and
    .venv-mmdet it can use the 5070 Ti. First challenger able to run on either
    card.

Deliberately a plain training loop rather than HF Trainer: this project debugs
by reading, and a visible loop beats a framework whose behaviour lives in
callbacks. ~200 lines, one job.

    .venv/bin/python -m altmodels.trainers.dfine --epochs 120 \
        --run-name dfine-s-1280-a [--card 0]
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from PIL import Image

from .. import gpu

REPO = Path(__file__).resolve().parents[2]
ALT = REPO / "outputs" / "alt"
DATASET_DEFAULT = ALT / "datasets" / "coco_rfdetr"
DEFAULT_MODEL = "ustc-community/dfine-small-obj2coco"


class TrayDataset:
    """Our converted COCO tree -> what DFineForObjectDetection wants.

    Boxes go in as NORMALISED cxcywh (the DETR-family convention); the json
    stores absolute xywh. Preprocessing is a SQUASHED resize + /255 — verified
    against the shipped image processor (do_normalize: False), so training and
    serving agree and no ImageNet statistics sneak in.
    """

    def __init__(self, split_dir: Path, size: int, flip: bool = False):
        self.dir, self.size, self.flip = split_dir, size, flip
        ann = json.loads((split_dir / "_annotations.coco.json").read_text())
        self.images = ann["images"]
        self.by_img: dict[int, list] = {}
        for a in ann["annotations"]:
            self.by_img.setdefault(a["image_id"], []).append(a["bbox"])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        # imported here, not stashed on self: a module attribute survives
        # fork-based DataLoader workers but would fail under spawn
        import numpy as np
        import torch
        im = self.images[i]
        pil = Image.open(self.dir / im["file_name"]).convert("RGB")
        W, H = im["width"], im["height"]
        boxes = []
        for x, y, w, h in self.by_img.get(im["id"], []):
            boxes.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
        do_flip = self.flip and random.random() < 0.5
        if do_flip:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[1.0 - b[0], b[1], b[2], b[3]] for b in boxes]
        pil = pil.resize((self.size, self.size), Image.BILINEAR)
        px = torch.from_numpy(np.asarray(pil, dtype=np.uint8).copy())
        px = px.permute(2, 0, 1).float().div_(255.0)
        t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1)
        return px, {"class_labels": torch.zeros(len(boxes), dtype=torch.long),
                    "boxes": t}


def _collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def main() -> None:
    ap = argparse.ArgumentParser(description="D-FINE sidecar training.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF id; *-obj2coco = Objects365 then COCO pretraining")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--queries", type=int, default=1000,
                    help="detection ceiling — a capture can exceed the stock 300")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    if not (args.dataset_dir / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"no converted dataset at {args.dataset_dir} — run "
                         ".venv/bin/python -m altmodels.convert first")

    run_dir = ALT / args.run_name
    out_dir = run_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The cockpit's log tail reads <run>/lab.log (that is where the cockpit's own
    # launcher puts stdout). A CLI launch writing elsewhere leaves the UI's log
    # panel empty, so mirror our progress lines there too.
    lab_log = (run_dir / "lab.log").open("a", buffering=1)

    def say(msg: str) -> None:
        print(msg, flush=True)
        lab_log.write(msg + "\n")

    t0 = time.time()

    with gpu.acquire(run_dir, what=f"train_dfine {args.run_name}"):
        import torch
        from torch.utils.data import DataLoader
        from transformers import DFineForObjectDetection

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        say(f"[dfine] card {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")

        # num_labels=1 + num_queries re-init ONLY the class heads; backbone,
        # encoder and decoder transfer (same shape as DEIM's fine-tune).
        model = DFineForObjectDetection.from_pretrained(
            args.model, num_labels=1, num_queries=args.queries,
            ignore_mismatched_sizes=True).to(dev)

        tr = TrayDataset(args.dataset_dir / "train", args.size, flip=True)
        va = TrayDataset(args.dataset_dir / "valid", args.size)
        dl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                        collate_fn=_collate, num_workers=4, drop_last=True)
        vl = DataLoader(va, batch_size=args.batch, collate_fn=_collate,
                        num_workers=2)
        # written BEFORE the first epoch: the cockpit needs the budget to draw
        # a progress bar, and meta.json only lands when the run finishes
        (out_dir / "run_config.json").write_text(json.dumps(
            {"epochs": args.epochs, "batch": args.batch, "size": args.size,
             "queries": args.queries, "model": args.model}), encoding="utf-8")
        steps = max(1, len(dl)) * args.epochs
        say(f"[dfine] {len(tr)} train / {len(va)} val | {len(dl)} steps/epoch "
            f"| {steps} total steps")

        backbone = [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad]
        rest = [p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": backbone, "lr": args.backbone_lr},
             {"params": rest, "lr": args.lr}], weight_decay=1e-4)
        warm = max(10, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
        scaler = torch.amp.GradScaler(dev)

        best, hist = float("inf"), []
        for ep in range(args.epochs):
            model.train()
            tot = n = 0
            for px, labels in dl:
                px = px.to(dev, non_blocking=True)
                labels = [{k: v.to(dev) for k, v in t.items()} for t in labels]
                with torch.amp.autocast(dev, dtype=torch.float16):
                    loss = model(pixel_values=px, labels=labels).loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += float(loss.detach())
                n += 1
            model.eval()
            vtot = vn = 0
            with torch.no_grad():
                for px, labels in vl:
                    px = px.to(dev)
                    labels = [{k: v.to(dev) for k, v in t.items()} for t in labels]
                    with torch.amp.autocast(dev, dtype=torch.float16):
                        vtot += float(model(pixel_values=px, labels=labels).loss)
                    vn += 1
            trl, vall = tot / max(1, n), vtot / max(1, vn)
            hist.append({"epoch": ep, "train_loss": trl, "val_loss": vall})
            # ATOMIC: the cockpit polls this file every few seconds, and a
            # plain write_text lets a reader catch it truncated mid-write
            # (seen as a JSONDecodeError). Same tmp+replace the retrain job uses.
            tmp = out_dir / "log.json.tmp"
            tmp.write_text(json.dumps(hist), encoding="utf-8")
            os.replace(tmp, out_dir / "log.json")
            say(f"[dfine] epoch {ep + 1}/{args.epochs} "
                f"train {trl:.3f} val {vall:.3f}")
            # Save BOTH faces. Val loss is a proxy the count exam has caught
            # lying before (see "THE CROWN LIES" in the README), so the exam
            # scores last.pt as well and we keep whichever counts better.
            if vall < best:
                best = vall
                model.save_pretrained(out_dir / "best")
            # "last" every 10 epochs, not every one: each save is ~40 MB, and
            # writing that 120 times contributed to the I/O storm that wedged
            # the GPU driver (2026-07-29). Worst case we lose <10 epochs.
            if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
                model.save_pretrained(out_dir / "last")

        # INSIDE the gpu block, while torch is still in scope and before the
        # allocator is torn down. D-FINE printed no memory figure at all, so
        # eight completed runs left no record of what it needs and the card
        # chooser had to guess — it defaulted this family to the 16 GB card on
        # no evidence whatsoever (web/machines.fits).
        vram = gpu.peak()
        if vram:
            say(f"[dfine] peak VRAM {vram['peak_vram_gb']} GiB reserved "
                f"({vram['peak_vram_alloc_gb']} GiB allocated)")

    meta = {**vram, "run": args.run_name, "arch": "dfine-small",
            "license": "Apache-2.0", "hf_model": args.model,
            "resolution": args.size, "epochs": args.epochs,
            "batch": args.batch, "num_queries": args.queries,
            "seed": args.seed, "best_val_loss": round(best, 4),
            "train_seconds": round(time.time() - t0, 1),
            "checkpoints": {"best": str(out_dir / "best"),
                            "last": str(out_dir / "last")},
            "best_checkpoint": str(out_dir / "best"), "created": time.time()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[dfine] done in {meta['train_seconds']}s -> {out_dir}")


if __name__ == "__main__":
    main()
