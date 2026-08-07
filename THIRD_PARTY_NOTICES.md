# Third-party notices

Groundwork is AGPL-3.0 (see [LICENSE](LICENSE)). It builds on the following third-party
work. Licenses are surfaced here and, where a user-facing choice depends on one (model
training, the auto-labeler download), in the cockpit at the point of choice.

## Vendored in this repository

- **IBM Plex** (Sans, Sans Condensed, Mono) — font files vendored at
  [`frontend/fonts/`](frontend/fonts/), © IBM Corp., licensed under the
  **SIL Open Font License 1.1**. The license text lives beside the fonts at
  `frontend/fonts/OFL.txt` and MUST exist in any distribution of this repository —
  if it is missing from your checkout, restoring it is required, not optional.

## Python dependencies (dynamic use, not vendored)

- **ultralytics** — **AGPL-3.0**. The training and export backend for the YOLOv8n family.
  This repository is itself AGPL-3.0, so the combination is license-compatible. Note
  upstream's stated position that the AGPL extends to *trained weights*; Groundwork
  surfaces each model family's license in the UI so that question is visible before you
  distribute a model. The Apache-2.0 challenger families exist precisely as the
  permissive alternative.
- **python-telegram-bot** — **LGPL-3.0**, used dynamically as an ordinary Python
  dependency by the optional bots extra.
- FastAPI, uvicorn, httpx, Pillow, numpy, python-dotenv and friends — MIT/BSD-family
  licenses, used as ordinary dependencies (see `pyproject.toml`).

## Model families

- **D-FINE** — Apache-2.0. The built-in challenger; runs in the main environment via
  Hugging Face `transformers`.
- **DEIMv2** — Apache-2.0. Installed by the challenger-stacks installer, which clones the
  vendor repository `github.com/Intellindust-AI-Lab/DEIMv2` (Apache-2.0) at a pinned
  commit into a sidecar environment. Not vendored here.
- **RTMDet** (via mmdetection/mmengine) — Apache-2.0. Sidecar-stack install; the same
  generic trainer also drives YOLOX and CenterNet configs from the mmdetection model zoo
  (Apache-2.0).
- **RF-DETR** — Apache-2.0. Its package auto-downloads its own pretrained weights on
  first use; nothing is redistributed here.

## LocateAnything-3B (optional auto-labeler)

Licensed under **NVIDIA's license**, not an open-source license. It is **never
redistributed by Groundwork**: the setup wizard offers it as an opt-in download from
Hugging Face, shows the license for consent first, and the download happens on your
machine under your account. Groundwork is fully usable without it — you label by hand or
with your own trained model's pre-labels.

---

If you believe a notice here is missing or wrong, open an issue.
