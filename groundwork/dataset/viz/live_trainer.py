"""LiveVizTrainer — live training-batch snapshots via ultralytics' documented
custom-trainer hook (YOLO.train(trainer=...), see docs "Customizing Trainer").

Stock ultralytics photographs only batches 0/1/2 (+3 at close-mosaic), gated
by `ni in self.plot_idx`. We wrap plot_idx in a list subclass whose membership
ALSO matches every Nth step, and route those periodic hits to
<run>/live/batch_step{ni}.jpg (rolling window) — rendered by ultralytics' own
plot_images, so no custom tensor handling. The 🔬 carousel picks them up and
becomes a genuinely live feed (~one fresh collage per minute here).

Read-only w.r.t. training: no change to data order, augmentation RNG, loss or
weights — only which batches get photographed. Every override body is
exception-proofed; total failure degrades to stock behavior. If a future
ultralytics stops consulting plot_idx, the stream silently disappears and
training is untouched (train.py also guards the import).
"""
from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer

EVERY = 250        # steps between live snapshots (~22 steps/epoch here -> ~1/min)
KEEP = 8           # rolling window of live snapshots kept on disk (~10MB)


class _EveryN(list):
    """plot_idx wrapper: canonical indices (real list content, incl. the
    close-mosaic trio ultralytics .extend()s in) PLUS every-Nth-step hits."""

    def __contains__(self, ni) -> bool:  # noqa: D105
        return list.__contains__(self, ni) or \
            (isinstance(ni, int) and ni > 0 and ni % EVERY == 0)


class LiveVizTrainer(DetectionTrainer):
    def _setup_train(self, *args, **kwargs):
        out = super()._setup_train(*args, **kwargs)
        try:
            self.plot_idx = _EveryN(self.plot_idx)
        except Exception:  # noqa: BLE001 — viz must never touch training
            pass
        return out

    def plot_training_samples(self, batch, ni):
        try:
            # Canonical saves (train_batch0/1/2 + close-mosaic trio) keep stock
            # names/behavior — everything downstream (peek, docs) relies on them.
            if not isinstance(self.plot_idx, _EveryN) or \
                    list.__contains__(self.plot_idx, ni):
                return super().plot_training_samples(batch, ni)
            from ultralytics.utils.plotting import plot_images
            live = self.save_dir / "live"
            live.mkdir(exist_ok=True)
            plot_images(labels=batch, paths=batch["im_file"],
                        fname=live / f"batch_step{ni:06d}.jpg", on_plot=None)
            for old in sorted(live.glob("batch_step*.jpg"))[:-KEEP]:
                old.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — a snapshot failure must never kill a run
            pass
