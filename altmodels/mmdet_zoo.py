"""Which mmdet models the challenger trainer can run, and how to fetch them.

One dict, one lookup — adding a candidate is a four-line entry, not a new
trainer. Everything downstream is already model-agnostic: `trainers.rtmdet`
does config surgery through generic keys (bbox_head.num_classes,
test_cfg.max_per_img, the pipelines), and `predictors/rtmdet_predict`
reads whatever config the run dumped, so it serves any of these unchanged.

Every entry must be Apache-2.0 (weights included) — that is the whole point
of the challenger programme. mmdetection itself is Apache-2.0, and so are the
upstream models below.

Fetch one with:
    .venv-mmdet/bin/mim download mmdet --config <config> --dest outputs/alt/pretrain
CARD CONSTRAINT (measured 2026-07-29): every challenger stack is 3090-ONLY.
.venv-mmdet ships torch 2.1.2+cu121 and .venv-deim torch 2.5.1+cu121 — neither
has sm_120 kernels, so both die on the worker's 5070 Ti with "no kernel image is
available for execution on the device". Only the yolo .venv (torch 2.13+cu130)
can use that card. That is why the cockpit pins CUDA_VISIBLE_DEVICES=1, and why
challengers queue serially instead of using both GPUs. Unlocking the second card
means upgrading a venv's torch — cheap-ish for .venv-deim (pure-PyTorch
deformable attention, no compiled ops) and expensive for .venv-mmdet (mmcv is
compiled against its torch and would need a source rebuild).

NOTE: config names follow the mmdet version installed in .venv-mmdet — this
one still uses the older `8x8` spelling (yolox_s_8x8_300e_coco), not the newer
`8xb8-`. `mim download` prints the full valid list if an entry ever drifts.
"""
from __future__ import annotations

# config  = mmdet config name (also the downloaded .py stem)
# ckpt    = glob for the checkpoint mim drops beside it
# note    = why this one is worth a slot on the GPU
ZOO: dict[str, dict[str, str]] = {
    "rtmdet-tiny": {
        "config": "rtmdet_tiny_8xb32-300e_coco",
        "ckpt": "rtmdet_tiny_8xb32-300e_coco_*.pth",
        "license": "Apache-2.0",
        "note": "first challenger; deprecated after 5 rounds (best 0.08, "
                "unstable 0.08-0.50 band) but kept runnable",
    },
    "yolox-s": {
        "config": "yolox_s_8x8_300e_coco",
        "ckpt": "yolox_s_8x8_300e_coco_*.pth",
        "license": "Apache-2.0",
        "note": "the closest APACHE analogue to the champion: anchor-free, "
                "decoupled head, SimOTA assignment — the same family of ideas "
                "as yolov8 without the AGPL. CNN + NMS, so the CoreML path is "
                "the boring one we already ship.",
    },
    "yolox-tiny": {
        "config": "yolox_tiny_8x8_300e_coco",
        "ckpt": "yolox_tiny_8x8_300e_coco_*.pth",
        "license": "Apache-2.0",
        "note": "yolox-s's little sibling, nearer yolov8n's parameter count "
                "— the fair size-matched comparison if -s wins on capacity",
    },
    "centernet": {
        "config": "centernet_r18-dcnv2_8xb16-crop512-140e_coco",
        # NOT the config name with a suffix. mmdet publishes this one's weights
        # as `centernet_resnet18_dcnv2_140e_coco_*` — a different spelling of the
        # same model, so the glob built from the config name matched nothing and
        # a centernet run would have died on "pretrained checkpoint not found"
        # AFTER the dataset convert. Caught 2026-07-31 by downloading it and
        # checking every zoo glob actually resolves, which is the only way to
        # find this: both strings look right on their own.
        "ckpt": "centernet_resnet18_dcnv2*.pth",
        "license": "Apache-2.0",
        "note": "POINT-based: predicts object centres as heatmap peaks. Our "
                "labels are dots to begin with (boxes are fabricated from "
                "nearest-neighbour spacing in point_to_box.py), so this is the "
                "only candidate that trains on the signal we actually have.",
    },
}


def get(arch: str) -> dict[str, str]:
    if arch not in ZOO:
        raise SystemExit(f"unknown mmdet arch {arch!r} — known: "
                         + ", ".join(sorted(ZOO)))
    return ZOO[arch]
