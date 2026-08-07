"""The model registry — one description per family, read by everything.

WHAT WAS DUPLICATED. Six places each knew part of "what is a model family":

  web/api/lab_ops.py  _available()   arch -> which venv, to answer "can this
                                     machine train challengers at all?"
  web/api/lab_ops.py  train()        arch -> venv + trainer module + epochs
  web/api/lab_ops.py  score()        arch -> venv + predictor + card pinning
  web/api/lab.py      list_runs()    run-name prefix -> arch + licence
  altmodels/predictors/__init__.py   predictor name -> import
  apps/static/index.html             the trainable-arch <select>

Adding a family meant editing all six and noticing none of them was a checklist.
The failure mode is on record in `score()`'s own comment: until 2026-07-30 an
unrecognised arch fell through to the RF-DETR predictor, so scoring a D-FINE run
would have loaded a D-FINE checkpoint with RF-DETR's loader out of a venv that
cannot even train it — a wrong number rather than a refusal.

PREFIX MATCHING IS LONGEST-FIRST, and that is a fix, not a translation. The old
table in lab.py was ORDER-dependent and said so: `yolox-tiny` had to be listed
before `yolox-s` or the shorter prefix would swallow the longer name. An
ordering constraint maintained by a comment is one edit away from being wrong.
`for_run()` sorts candidates by prefix length, so the most specific match wins
regardless of declaration order and the constraint cannot be violated.

VENVS ARE NAMED, NOT PATHED, so this module stays a description and the caller
resolves. `python()` is the one place that turns a name into an interpreter.

`yolov8n` is in here too even though it is not a challenger. It is a model
family: it has a licence that matters (AGPL, the open question blocking app
distribution — IPHONE.md Part 6), a predictor, and run names to recognise. A
registry that omitted the champion would just be a seventh table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Model:
    """One model family. Frozen: a registry entry is a fact, not a setting."""
    name: str                       # canonical family name; matches the ledger's arch
    label: str                      # how a human should see it
    predictor: str                  # altmodels/predictors key, or "" if not scorable
    venv: str                       # directory name under the repo root
    license: str
    default_imgsz: int
    prefixes: tuple[str, ...]       # run-name prefixes that identify this family
    trainer_module: str | None = None    # None = not trainable from the cockpit
    default_epochs: int | None = None
    # BATCH IS A FAMILY FACT TOO. rfdetr asserts `batch * grad_accum == 16` and
    # defaults grad_accum to 8, so it needs batch 2 — sending the cockpit's
    # generic 8 gave "batch x grad-accum must equal 16 (got 8x8)" two minutes
    # in. One number per family beats a caller remembering each trainer's
    # arithmetic.
    default_batch: int = 8
    # PEAK GPU MEMORY AT default_imgsz — MEASURED, from what each family records.
    # None means no run of it has ever recorded one.
    #
    #   DEIM/DEIMv2  "max mem: 17613"   torch.cuda.max_memory_ALLOCATED, MiB
    #   mmdet        "memory: 5029"     the same call, via mmengine, MB
    #   yolov8n      the LEDGER's peak_vram_gb column, which pipeline/train.py
    #                writes from torch.cuda.max_memory_RESERVED()/1e9
    #
    # TWO DIFFERENT QUANTITIES SIT IN THIS COLUMN, and pretending otherwise was
    # an error here until 2026-08-03. Reserved is what the caching allocator
    # holds; allocated is what is live inside it, so reserved >= allocated
    # always. yolo's number is therefore measured more conservatively than
    # DEIM's and mmdet's, and cross-family comparison of these values is
    # APPROXIMATE. It does not change any card decision today — it makes the
    # non-yolo families' true footprint LARGER than recorded, so every "does not
    # fit" verdict stands and only the margins move.
    #
    # yolov8n's entry was also plain wrong: it said 8.6, harvested from the
    # ultralytics epoch line, which reports INSTANTANEOUS reserved per epoch. The
    # ledger has carried a real per-run peak for 28 runs and its max is 11.19 GB.
    # Parsing a log for a number the ledger already stores is how that happened.
    #
    # WHICH WAY THIS COLUMN LEANS, decided deliberately 2026-08-03. Where both
    # are known (yolov8n, dfine-small, rfdetr-nano — all via altmodels.gpu.peak)
    # the RESERVED figure is recorded, because the two errors are not symmetric:
    #
    #   too HIGH -> steers a model to a roomier card it did not strictly need.
    #               Costs a little scheduling freedom.
    #   too LOW  -> puts a model on a card it SPILLS on, which under WSL2 does
    #               not error, it just runs 3x slower for hours. That is the
    #               original bug this whole field exists to prevent.
    #
    # So it fails safe. The consequence to keep in mind: the six families still
    # recorded from an ALLOCATED source (both DEIMs, both YOLOXs, rtmdet,
    # centernet) are FLOORS, not true peaks — measured on these three, reserved
    # ran 3-4 GiB above allocated. Re-measuring them through gpu.peak() would
    # make the column uniform; until then, treat their margins as optimistic.
    # UNITS ARE GiB THROUGHOUT (2^30), matching machines.json's card sizes.
    #
    # WHY IT IS HERE AND NOT A GUESS. On 2026-08-02 deimv2-n-tv28 trained on the
    # 16 GB card at 2.19 s/it while seven prior runs of the SAME model
    # on a 24 GB card ran at 0.72-0.94. It needs 17.2 GiB; under WSL2 the
    # driver spills the overflow into host RAM over PCIe instead of raising OOM,
    # so the run completes and scores correctly and is simply 3x slower, with no
    # error anywhere to notice it by. web/machines.fits() turns this number into
    # a card preference so that cannot happen silently again.
    #
    # IT IS THE MAX SEEN, NOT A BOUND. DEIM resamples resolution per batch, so
    # seven runs of one config peaked anywhere from 13.0 to 17.2 GiB — the peak
    # is stochastic and only the largest is safe to plan against. Recorded at
    # default_imgsz; a smaller size needs less, which makes this conservative in
    # the harmless direction.
    peak_vram_gb: float | None = None
    deprecated: bool = False

    # HOW THIS TRAINER SPELLS RESOLUTION. Four trainers, three spellings:
    # deim/dfine take `--size <px>`, rfdetr takes `--res <px>`, rtmdet takes
    # `--scale-factor <2|3>` (2 -> 1280, 3 -> 1920). This lived as
    # `if model.name == "deimv2-n": --size else: --scale-factor` in
    # lab_ops.train — the seventh hand-maintained table, and it broke the moment
    # a second DEIM entry existed: deimv2-n-tv28 fell to the else branch and
    # would have been launched with a flag its trainer does not accept.
    size_flag: str = "--size"
    # WHICH SIZES THIS FAMILY CAN ACTUALLY TRAIN AT. Not cosmetic: the mmdet
    # trainer takes `--scale-factor 2|3` and nothing else, so a UI offering 960
    # for yolox mapped it to scale-factor 2 and trained at 1280 — the run says
    # 1280, the request said 960, and nothing ever mentions it. Same silent-
    # wrong-parameter shape as the epochs clamp. The champion's own pipeline
    # only ships 960 and 1280 weights-wise; the free-size families take anything
    # and these are the ones worth offering.
    sizes: tuple[int, ...] = (960, 1280, 1920)
    # Fixed arguments this family always needs. rtmdet's trainer is GENERIC —
    # `--arch` is how yolox and centernet ride it without a trainer of their own
    # (see altmodels/mmdet_zoo.py: "a four-line entry, not a new trainer").
    train_extra: tuple[str, ...] = ()
    # DOES ITS TRAINER TAKE --dataset-dir? True means the run gets its OWN
    # converted COCO tree (web/lab_dataset.py) and can therefore train beside
    # another challenger. False means it still reads the single shared tree, so a
    # second launch would rmtree the data out from under it. NO FAMILY IS FALSE
    # TODAY — DEIM was, until its derived classic tree moved inside the per-run
    # source (_deim_dest), and a False here would now be a real bug: the
    # launcher converts into the per-run tree and only passes it when this is
    # True, so a False family would train on a shared tree nothing wrote.
    isolated_dataset: bool = True

    # Import names this family's trainer needs BEYOND its venv existing, and
    # the extra that installs them. Checked before a launch rather than
    # discovered by a traceback three stages in. Empty for families with their
    # own stack: there a missing package means the stack was built wrong, which
    # venv_present already speaks to.
    requires: tuple[str, ...] = ()
    requires_extra: str = ""

    @property
    def python(self) -> Path:
        """The interpreter for this family's venv.

        The conventional layout is <repo>/<venv>/bin/python. When the MAIN
        venv directory does not exist — a pip install into some other
        environment, or the Docker image, where the running interpreter IS
        the main env — the main family falls back to sys.executable. Sidecar
        venvs never fall back: running a DEIM trainer on the main interpreter
        would import the wrong torch build and fail far less legibly than a
        missing path does.
        """
        p = REPO / self.venv / "bin" / "python"
        if self.venv == ".venv" and not p.exists():
            return Path(sys.executable)
        return p

    def train_argv(self, *, imgsz: int, epochs: int, batch: int,
                   run_name: str, dataset_dir=None, alt_dir=None) -> list[str]:
        """The trainer's command line, minus the interpreter and -m module.

        ONE place builds this. A caller that assembled it itself would be the
        eighth table, and the seventh is what this method replaced.

        dataset_dir names the run's OWN converted COCO tree (web/lab_dataset.py).
        Passing it is what lets two challengers train at once — they used to
        share one directory that each launch rebuilt. Every trainer takes the
        same flag, including DEIM since 2026-08-02; None omits it and the trainer
        falls back to the legacy shared tree, which is what a hand-run CLI wants.
        """
        size = (["--scale-factor", "3" if imgsz >= 1920 else "2"]
                if self.size_flag == "--scale-factor"
                else [self.size_flag, str(imgsz)])
        ds = ["--dataset-dir", str(dataset_dir)] if dataset_dir else []
        # WHERE THE RUN LANDS. Challenger runs are per-project (pp.ALT_DIR) and
        # the trainers defaulted to one flat repo-level alt/, so a cockpit
        # launch wrote its artifacts where the cockpit never reads them.
        alt = ["--alt-dir", str(alt_dir)] if alt_dir else []
        return ["--epochs", str(epochs), "--batch", str(batch),
                *size, *ds, *alt, *self.train_extra, "--run-name", run_name]

    @property
    def venv_present(self) -> bool:
        return self.python.exists()

    def missing_deps(self) -> list[str]:
        """Packages this family needs that its interpreter cannot import.

        A venv EXISTING is not a venv that can train. D-FINE proved it: it runs
        in the main venv, so venv_present was True, and it died at
        `from transformers import DFineForObjectDetection` — after the launcher
        had synced a dataset, converted a COCO tree and taken a card. One
        subprocess against a list that is almost always empty, never on a
        polled path.
        """
        if not self.requires or not self.venv_present:
            return []
        import subprocess
        code = ("import importlib.util as u,sys;"
                "print(' '.join(m for m in sys.argv[1:] "
                "if u.find_spec(m) is None))")
        try:
            r = subprocess.run([str(self.python), "-c", code, *self.requires],
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return []          # could not ask — never block a launch on that
        return (r.stdout or "").split()


# Order here is presentation order only — `for_run` does not depend on it.
MODELS: tuple[Model, ...] = (
    Model(
        name="yolov8n", label="YOLOv8n (default)",
        predictor="ultralytics", venv=".venv",
        # ultralytics is AGPL and it may cover the trained weights. This is the
        # licence question blocking distribution, which is exactly why the
        # challengers below are all Apache-2.0.
        license="AGPL-3.0", default_imgsz=1280,
        prefixes=("yolov8n",), sizes=(960, 1280),
        trainer_module="groundwork.dataset.pipeline.train",
        # 10.4 GiB = the MAX across 28 ledger runs (min 5.4), CONVERTED: the
        # ledger stores 11.19 from `max_memory_reserved()/1e9`, which is a
        # DECIMAL gigabyte, and 11.19 GB is 10.42 GiB. Recording it as "11.2"
        # against a card whose 16 is GiB overstated yolo's appetite by 0.8 GiB.
        # Peak reserved varies a lot run to run — 960 spans 6.33-9.63 and 1280
        # spans 8.03-9.2, overlapping — so only the max is safe to plan against,
        # the same rule DEIM's stochastic peak already forced.
        default_epochs=250, peak_vram_gb=10.4,
    ),
    Model(
        name="deimv2-n", label="DEIMv2-N",
        predictor="deim", venv=".venv-deim",
        license="Apache-2.0", default_imgsz=1280,
        # reexam-/stg2- are diagnostic RE-SCORINGS of DEIM checkpoints, not new
        # models. They are recognised so the leaderboard is complete; the
        # attempts chart filters them out.
        prefixes=("deimv2-n", "reexam-", "stg2-"),
        trainer_module="altmodels.trainers.deim",
        # Isolated since 2026-08-02: --dataset-dir names the source tree and the
        # classic layout is built inside it (_deim_tree), so nothing is shared.
        default_epochs=60, size_flag="--size",
        sizes=(960, 1280, 1920), peak_vram_gb=17.2,
        # The trainer defaults meta's `arch` to deimv2-<variant>; pass it
        # explicitly so BOTH DEIM entries file their runs under themselves.
        train_extra=("--arch", "deimv2-n"),
    ),
    Model(
        # THE SAME MODEL, A DIFFERENT ENVIRONMENT — and therefore a different
        # entry, because it is not comparable to the one above.
        #
        # .venv-deim is torch 2.5.1+cu121 whose compiled kernels stop at sm_90,
        # so DEIM could only ever use the 3090. .venv-deim13 is torch
        # 2.13.0+cu130 and runs on either card; getting there needed a two-hook
        # torchvision compat shim (altmodels/deim_tv_compat.py — the blocker was
        # never CUDA, it was a v2 API rename). Verified training on the 5070 Ti.
        #
        # A SEPARATE ENTRY rather than swapping the venv, because every one of
        # the 54 existing DEIM runs was trained under torchvision 0.20 and a
        # major version can move augmentation behaviour. Its own run-name prefix
        # means the ledger can never quietly compare across the two, which is
        # the mistake this registry exists to prevent. Promote it to the default
        # only after a same-seed comparison agrees.
        name="deimv2-n-tv28", label="DEIMv2-N (torch 2.13 · either card)",
        predictor="deim", venv=".venv-deim13",
        license="Apache-2.0", default_imgsz=1280,
        prefixes=("deimv2n13-",),
        trainer_module="altmodels.trainers.deim",
        # Isolated since 2026-08-02: --dataset-dir names the source tree and the
        # classic layout is built inside it (_deim_tree), so nothing is shared.
        default_epochs=60, size_flag="--size", sizes=(960, 1280, 1920),
        train_extra=("--arch", "deimv2-n-tv28"),
        # THE ONE FAMILY THIS NUMBER CHANGES ANYTHING FOR. Every other model
        # over 16 GiB is sm_90-capped and cannot reach card 0 regardless; this
        # one CAN, which is the entire reason the entry exists, and that is
        # exactly how it ended up spilling on it.
        peak_vram_gb=17.2,
        # NOT pinned to the second card: being able to use card 0 is the entire
        # reason this entry exists.
    ),
    Model(
        name="dfine-small", label="D-FINE small",
        # D-FINE lives in the MAIN venv (HF transformers) — the only family that
        # does. It still pins the second card, because it trained under that
        # pinning and scoring must not steal card 0 from a cockpit retrain.
        predictor="dfine", venv=".venv",
        license="Apache-2.0", default_imgsz=1280,
        prefixes=("dfine-",),
        # WIRED 2026-07-31. altmodels/trainers/dfine.py already existed and had
        # produced 6 runs from the CLI; it was simply never named here, so the
        # cockpit could not start one. Nothing new was written.
        trainer_module="altmodels.trainers.dfine",
        # It runs in the MAIN venv, which is the point of it — but it loads
        # through transformers, which [train] does not install.
        requires=("transformers",), requires_extra="dfine",
        default_epochs=120, size_flag="--size", sizes=(960, 1280, 1920),
        # MEASURED 2026-08-03 (altmodels.gpu.peak, added because this family and
        # rfdetr printed no memory figure at all — eight completed runs left no
        # record of what they need, and fits() was defaulting them to the 16 GB
        # card on no evidence). 11.39 GiB RESERVED / 7.98 GiB allocated.
        peak_vram_gb=11.4,
    ),
    Model(
        name="rtmdet-tiny", label="RTMDet-tiny",
        predictor="rtmdet", venv=".venv-mmdet",
        license="Apache-2.0", default_imgsz=1280,
        prefixes=("rtmdet-tiny",),
        trainer_module="altmodels.trainers.rtmdet",
        default_epochs=300, deprecated=True,
        peak_vram_gb=11.6,
        size_flag="--scale-factor", train_extra=("--arch", "rtmdet-tiny"),
        sizes=(1280, 1920),
    ),
    Model(
        name="rfdetr-nano", label="RF-DETR nano",
        predictor="rfdetr", venv=".venv-alt",
        license="Apache-2.0", default_imgsz=1120,
        prefixes=("rfdetr-nano", "rfdetr"),
        deprecated=True,
        # WIRED 2026-07-31, same story as dfine — the trainer existed and had
        # produced 3 runs. `--res`, not `--size`: three trainers, three
        # spellings, which is exactly why size_flag exists.
        trainer_module="altmodels.trainers.rfdetr",
        # rfdetr's --res must be a multiple of 56 (its patch grid): 560, 672 and
        # 1120 are the three its existing runs used. default_imgsz was 1280,
        # which it has NEVER trained at and which is not even a legal --res —
        # caught by asserting the default is one of the offered sizes. 1120 is
        # the best-conditions run (1120px/600q on the 24 GB card).
        default_epochs=100, size_flag="--res", sizes=(560, 672, 1120),
        default_batch=2,
        # MEASURED 2026-08-03 at 1120/q1200: 12.87 GiB RESERVED / 8.17 allocated.
        # Comfortably under its own allocator cap (card total - 1900 MiB), so
        # this is the model's appetite rather than the guard's ceiling.
        peak_vram_gb=12.9,
    ),
    Model(
        name="yolox-tiny", label="YOLOX-tiny",
        # SCORABLE, via the same predictor rtmdet uses. That predictor is
        # misleadingly NAMED: `predictors/rtmdet_predict` finds the run's own
        # dumped mmengine config and hands it to `init_detector`, so nothing in
        # it is rtmdet-specific and it serves any detector trained through
        # `trainers.rtmdet` — which is exactly how these three ride that trainer
        # via `--arch`. It was left as `predictor=""` because nobody had checked;
        # `mmdet_zoo` said so in prose all along ("reads whatever config the run
        # dumped, so it serves any of these unchanged").
        predictor="rtmdet", venv=".venv-mmdet",
        license="Apache-2.0", default_imgsz=1280,
        prefixes=("yolox-tiny",), deprecated=True,
        # WIRED 2026-07-31 with NO new trainer. rtmdet's is generic and takes
        # `--arch`; altmodels/mmdet_zoo.py already carried this entry ("a
        # four-line entry, not a new trainer") and the predictor reads whatever
        # config the run dumped. `predictor=""` still means it cannot be SCORED
        # from the cockpit — training and scoring are separate wirings, and
        # claiming otherwise here would be the lie the registry exists to stop.
        trainer_module="altmodels.trainers.rtmdet",
        default_epochs=300,
        peak_vram_gb=12.7,
        size_flag="--scale-factor", train_extra=("--arch", "yolox-tiny"),
        sizes=(1280, 1920),
    ),
    Model(
        name="yolox-s", label="YOLOX-S",
        # SCORABLE, via the same predictor rtmdet uses. That predictor is
        # misleadingly NAMED: `predictors/rtmdet_predict` finds the run's own
        # dumped mmengine config and hands it to `init_detector`, so nothing in
        # it is rtmdet-specific and it serves any detector trained through
        # `trainers.rtmdet` — which is exactly how these three ride that trainer
        # via `--arch`. It was left as `predictor=""` because nobody had checked;
        # `mmdet_zoo` said so in prose all along ("reads whatever config the run
        # dumped, so it serves any of these unchanged").
        predictor="rtmdet", venv=".venv-mmdet",
        license="Apache-2.0", default_imgsz=1280,
        prefixes=("yolox-s",), deprecated=True,
        # WIRED 2026-07-31 with NO new trainer. rtmdet's is generic and takes
        # `--arch`; altmodels/mmdet_zoo.py already carried this entry ("a
        # four-line entry, not a new trainer") and the predictor reads whatever
        # config the run dumped. `predictor=""` still means it cannot be SCORED
        # from the cockpit — training and scoring are separate wirings, and
        # claiming otherwise here would be the lie the registry exists to stop.
        trainer_module="altmodels.trainers.rtmdet",
        default_epochs=300,
        peak_vram_gb=17.2,
        size_flag="--scale-factor", train_extra=("--arch", "yolox-s"),
        sizes=(1280, 1920),
    ),
    Model(
        name="centernet", label="CenterNet",
        # SCORABLE, via the same predictor rtmdet uses. That predictor is
        # misleadingly NAMED: `predictors/rtmdet_predict` finds the run's own
        # dumped mmengine config and hands it to `init_detector`, so nothing in
        # it is rtmdet-specific and it serves any detector trained through
        # `trainers.rtmdet` — which is exactly how these three ride that trainer
        # via `--arch`. It was left as `predictor=""` because nobody had checked;
        # `mmdet_zoo` said so in prose all along ("reads whatever config the run
        # dumped, so it serves any of these unchanged").
        predictor="rtmdet", venv=".venv-mmdet",
        license="Apache-2.0", default_imgsz=1280,
        prefixes=("centernet",), deprecated=True,
        # WIRED 2026-07-31 with NO new trainer. rtmdet's is generic and takes
        # `--arch`; altmodels/mmdet_zoo.py already carried this entry ("a
        # four-line entry, not a new trainer") and the predictor reads whatever
        # config the run dumped. `predictor=""` still means it cannot be SCORED
        # from the cockpit — training and scoring are separate wirings, and
        # claiming otherwise here would be the lie the registry exists to stop.
        trainer_module="altmodels.trainers.rtmdet",
        default_epochs=300,
        peak_vram_gb=4.9,
        size_flag="--scale-factor", train_extra=("--arch", "centernet"),
        sizes=(1280, 1920),
    ),
)

_BY_NAME = {m.name: m for m in MODELS}

# Every (prefix, model) pair, LONGEST FIRST. Building it once here is what makes
# "most specific wins" a property of the data rather than of declaration order.
_BY_PREFIX: tuple[tuple[str, Model], ...] = tuple(sorted(
    ((p, m) for m in MODELS for p in m.prefixes),
    key=lambda pm: -len(pm[0])))


def by_name(name: str | None) -> Model | None:
    """Exact family lookup, e.g. the ledger's `arch` column."""
    return _BY_NAME.get(name or "")


def for_run(run_name: str) -> Model | None:
    """Which family a RUN belongs to, from its name.

    Needed because older exams overwrote the trainer's meta.json and lost the
    arch, and `-last` / `-epN` exam directories never had one — the run name is
    the only thing left that knows.
    """
    for prefix, m in _BY_PREFIX:
        if run_name.startswith(prefix):
            return m
    return None


def trainable_here() -> list[Model]:
    """Families this machine can actually train.

    The venv's presence IS the answer, and this is the whole "am I the lab?"
    test: the challenger venvs are only ever installed on the worker. Cheap, and
    it shells out to nothing that can hang.
    """
    return [m for m in MODELS if m.trainer_module and m.venv_present]


def scorable(m: Model) -> tuple[bool, str]:
    """Can this machine score that family, and if not, why not — in words the
    cockpit can show. Refusing loudly is the point: see the module docstring."""
    if not m.predictor:
        return False, f"no predictor is wired for {m.name}"
    if not m.venv_present:
        return False, f"{m.venv} missing — score {m.name} runs on the worker"
    return True, ""
