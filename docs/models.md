# The model registry

`groundwork/models/registry.py` is the one description of a model family — name, license,
venv, trainer module, default size/epochs/batch, run-name prefixes, measured peak VRAM,
card pinning. Everything that dispatches on a family asks it: the Train control, the
scoring path, the exporters, the ledger's run-name recognition. It replaced six
hand-maintained tables, and the failure it exists to prevent is on record: an
unrecognised architecture once fell through to *another family's* checkpoint loader,
which produces a wrong number instead of a refusal. Adding a family is one entry plus a
predictor; a family the registry cannot vouch for is refused loudly
(`scorable()` returns the reason in words the cockpit shows).

## Families

| family | license | environment | default size | default epochs | status |
|---|---|---|---|---|---|
| `yolov8n` | AGPL-3.0 | `.venv` (main) | 1280 (960/1280) | 250 | **the default** — trains via the cockpit's own retrain pipeline |
| `dfine-small` | Apache-2.0 | `.venv` (main) | 1280 (960/1280/1920) | 120 | **built-in challenger** (HF `transformers`) |
| `deimv2-n` | Apache-2.0 | `.venv-deim` (stack) | 1280 (960/1280/1920) | 60 | challenger |
| `deimv2-n-tv28` | Apache-2.0 | `.venv-deim13` (stack) | 1280 (960/1280/1920) | 60 | the same model on a newer torch — a **separate entry**, because runs across a major torchvision version are not comparable, and its own run prefix means the ledger can never quietly compare them |
| `rtmdet-tiny` | Apache-2.0 | `.venv-mmdet` (stack) | 1280 (1280/1920) | 300 | deprecated |
| `yolox-tiny` / `yolox-s` / `centernet` | Apache-2.0 | `.venv-mmdet` (stack) | 1280 (1280/1920) | 300 | deprecated — model-zoo entries riding the generic mmdet trainer via `--arch`, not trainers of their own |
| `rfdetr-nano` | Apache-2.0 | `.venv-alt` (stack) | 1120 (560/672/1120 — its resolution must be a multiple of its 56-px patch grid) | 100 (batch 2) | deprecated |

"Deprecated" means: still recognised — existing runs keep their family, their scores and
their place in the ledger — but not suggested for new runs. Sizes listed are the ones the
family can *actually train at*; offering a size a trainer silently rounds to another is
the same wrong-parameter failure as training on the wrong data, so the registry only
offers real ones (and records that trainers spell resolution three different ways:
`--size`, `--res`, `--scale-factor` — one field per family, not an `if` in a caller).

Every challenger is scored by the **same count-MAE exam** as the champion, from one
shared `eval_core` — a challenger's number and the champion's number mean the same thing
or they mean nothing.

## Why sidecar venvs exist at all

A torch build carries a **fixed set of compiled GPU kernels**. A wheel built for one CUDA
generation simply has no code for a newer card's architecture (`sm_XX`), and the failure
is not an install error — it imports fine and dies at first kernel launch. Different
model families need incompatible torch/toolchain pins, so each family names its venv and
`Model.python` is the only place a name becomes an interpreter.

The consequence the Train control surfaces: **which venv can drive which card is a
measured fact per machine**, probed from each venv's own arch list and recorded — never
assumed from the model's docs. A card whose architecture a venv's kernels do not cover is
refused with that reason.

## Peak VRAM is measured, and it steers card choice

Each family's entry records the **maximum peak GPU memory ever seen** for it, harvested
from what its own trainer logs (different trainers report different quantities — the
allocator's *reserved* vs *allocated* — so cross-family comparison is approximate, and
where both are known the more conservative reserved figure is recorded). Three rules
learned the expensive way:

- **Record the max, not the mean.** Some trainers resample resolution per batch, so the
  peak is stochastic across identical runs; only the largest observed value is safe to
  plan against.
- **Lean high.** Overestimating steers a model to a roomier card it did not strictly
  need — a scheduling cost. Underestimating puts it on a card it *spills* on, and under
  WSL2 that does not raise OOM: the driver pages into host RAM and the run completes,
  correct and ~3× slower, with no error anywhere to notice it by. The Train control
  refuses that by default; `allow_spill` accepts it deliberately.
- **Unmeasured is not too big.** A family or configuration with no recorded figure gets
  a note, not a refusal — refusing on absent evidence is the same mistake pointed the
  other way. Families whose trainers print no memory line at all are marked unmeasured
  in the UI rather than shown a clean bill of health.

Per-run conversions are isolated (each challenger run gets its own converted dataset
tree), which is what allows two challengers to train simultaneously without one launch
deleting the tree another is reading.

## The challenger stacks installer

D-FINE needs nothing: it lives in the main venv and is the built-in challenger. The
others install as **opt-in stacks** (setup wizard → Extras, re-runnable from Admin, or
by CLI), each stack being a sidecar venv with its own torch pin:

- **DEIMv2** — clones the vendor repository `github.com/Intellindust-AI-Lab/DEIMv2`
  pinned at commit `0fff8d4` (Apache-2.0) and builds `.venv-deim` around it. The pin is
  the point: a moving vendor HEAD under a fixed exam is an uncontrolled variable.
- **RTMDet** (mmdetection) — builds `.venv-mmdet`; the same generic trainer also drives
  the YOLOX and CenterNet zoo entries.
- **RF-DETR** — `.venv-alt`; its package auto-downloads its own pretrained weights on
  first use.

A stack that is not installed simply does not appear as trainable — the registry's
`venv_present` check is the whole test, and the Train control says which venv is missing
rather than offering a button that 400s.

## Licensing map

| component | license | note |
|---|---|---|
| `yolov8n` via ultralytics | AGPL-3.0 | same license as this repo; upstream's stated position is that it extends to trained weights — the reason the challenger track exists |
| D-FINE, DEIMv2, RTMDet/mmdet zoo, RF-DETR | Apache-2.0 | the permissive alternatives, scored by the identical exam |
| LocateAnything-3B (auto-labeler) | NVIDIA license | optional wizard download, never redistributed; labels it produces are drafts for human review |

The cockpit shows each family's license next to it, because the right family for you can
be a licensing question before it is an accuracy one.
