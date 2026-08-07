"""Fast runtime counter using a trained YOLO model.

Load best.pt once (~0.1 GB VRAM), detect objects in a single forward pass
(~40 ms), and render a POINT per object (box centers). No prompt, no teacher —
just send an image, get a count. Defaults come from count_eval.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image

from ..config import OUTPUTS_DIR, ROOT
from ..la.annotate import save_annotated
from ..la.parse import Detection, Detections
from ..dataset.filters.dot_dedupe import find_stacked, dot_radius
from ..dataset.yolo_export import boxes_to_yolo_lines
from ..dataset.store import collect
from . import serving_filters

# Fallbacks only, for a run that was never evaluated — a scored run's served
# conf/iou come from its count_eval.json via runtime_model.resolve().
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.5
DEFAULT_IMGSZ = 960
DEFAULT_MAX_DET = 2000


def find_latest_weights(pp) -> Path:
    """Newest best.pt under the training runs (dir auto-increments: yolov8n-2, …).

    `pp` says WHOSE runs, and it is required: a counter picking up the newest
    thing in somebody else's tree is the silent-wrong-data failure this repo
    treats as its most expensive.
    """
    runs = pp.RUNS_DIR
    weights = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not weights:
        raise FileNotFoundError(f"no trained weights under {runs} — train first")
    return weights[-1]


def reading_order_idx(centers: list[tuple[float, float]], boxes: list) -> list[int]:
    """INDICES of `centers` in reading order — the primitive; `_reading_order`
    wraps it. Same split as dot_dedupe.find_stacked / drop_stacked, and for the
    same reason: staging needs to reorder an image's BOXES the way its dots were
    reordered, and re-implementing the sort locally would be a second copy of a
    rule that is already mirrored in two languages."""
    if not centers:
        return []
    heights = sorted(y2 - y1 for x1, y1, x2, y2 in boxes)
    row_h = max(1.0, 0.7 * heights[len(heights) // 2])   # ~0.7 * median box height
    return sorted(range(len(centers)),
                  key=lambda i: (round(centers[i][1] / row_h), centers[i][0]))


def _reading_order(centers: list[tuple[float, float]], boxes: list) -> list[tuple[float, float]]:
    """Sort centers top->bottom, left->right, banding rows by median box height.

    Ported to JavaScript for the dot editor (frontend/editor/editor_mirrors.js) —
    note round() here is half-to-EVEN and JS's Math.round is not, which is a
    difference the editor's numbering can actually show. Change either side and
    re-run `python checks/check_editor_mirrors.py`."""
    return [centers[i] for i in reading_order_idx(centers, boxes)]


@dataclass
class CountResult:
    count: int
    overlay_path: str
    seconds: float
    stats: dict
    name: str = ""          # staging id; feedback (✓/✗) routes this sample

    def to_dict(self) -> dict:
        return asdict(self)


class YoloCounter:
    """Owns the single resident YOLO model."""

    # Which architecture this instance IS — set on the class, so anything
    # holding a counter can say so without re-reading the engine switch (which
    # can have been flipped since this one was loaded). runtime_model.ENGINES.
    ENGINE = "yolo"
    EXPERIMENTAL = False        # the champion; challengers override to True

    # WHOSE PROJECT, as a CLASS default and not only an instance attribute.
    # count() is inherited by every challenger, and DeimCounter.__init__ does
    # NOT chain to super() — it sets each attribute itself — so an instance
    # attribute alone turns staging into an AttributeError on the deim path
    # instead of a wrong-project write.
    pp = None

    def __init__(self, pp, weights: str | Path | None = None,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 imgsz: int = DEFAULT_IMGSZ, max_det: int = DEFAULT_MAX_DET,
                 verbose: bool = True):
        from ultralytics import YOLO   # lazy: only the runtime that counts needs it
        from . import runtime_model
        if weights:
            self.weights = Path(weights)
        else:                                   # pinned run, else newest — at its own imgsz
            # WHOSE pin, and whose runs. Both take the project now: a counter
            # built for one project must never resolve to another's weights.
            rw, rimg, rconf, riou = runtime_model.resolve(pp)
            self.weights = rw or find_latest_weights(pp)
            imgsz = rimg
            # serve the conf/iou count_eval tuned for THIS run (falls back to the
            # passed defaults if the run was never evaluated)
            if rconf is not None:
                conf = rconf
            if riou is not None:
                iou = riou
        # WHOSE PROJECT THIS COUNTER SERVES, kept rather than used-and-dropped.
        # A counter that resolves weights for one project and stages feedback
        # into another splits an image/label pair across two datasets — the
        # silent-wrong-data failure this repo treats as its most expensive.
        # Staging needs it, so it lives on the instance.
        self.pp = pp
        self.conf, self.iou, self.imgsz, self.max_det = conf, iou, imgsz, max_det
        t0 = time.time()
        self.model = YOLO(str(self.weights))
        if verbose:
            try:
                shown = self.weights.relative_to(ROOT)
            except ValueError:      # data-dir install: weights live elsewhere
                shown = self.weights
            print(f"[yolo] loaded {shown} in "
                  f"{time.time()-t0:.1f}s | conf={conf} iou={iou} imgsz={imgsz}", flush=True)

    def _detect(self, image: Image.Image) -> list[list[float]]:
        """Raw boxes (xyxy, image pixels) — the ONLY model-specific step.
        Challengers override just this and inherit count()'s filters unchanged,
        which is the lockstep the hard rules demand (see DeimCounter). Which
        filters actually run is groundwork/serve/serving_filters.py — the
        stacked-dot dedupe."""
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf,
                               iou=self.iou, max_det=self.max_det, verbose=False)[0]
        return r.boxes.xyxy.tolist()

    def count(self, image: Image.Image, name: str | None = None,
              out_dir: Path | None = None, stage: bool = True) -> CountResult:
        t0 = time.time()
        image = image.convert("RGB")

        boxes = self._detect(image)
        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
        # Track WHICH detections survive, not just how many. The filter below
        # is index-preserving, so each surviving dot keeps a line back to the box
        # the model actually measured — which is what lets staging record a real
        # extent instead of re-synthesizing a square from dot spacing.
        live = list(range(len(centers)))          # indices into boxes/centers
        # Drop stacked dots — two dots on one object are one detection counted
        # twice, so the pair counts once. Same rule + radius formula as
        # count_eval (serving_filters.DEDUPE is the one switch).
        dot_r = dot_radius(image.width, image.height)
        dupes = find_stacked([centers[i] for i in live], dot_r) \
            if serving_filters.DEDUPE else []
        live = [i for k, i in enumerate(live) if k not in dupes]
        stacked = len(dupes)
        # Number in reading order (top->bottom rows, left->right within a row) so
        # the overlay's 1,2,3… sit next to each other. Band rows by the objects'
        # own median height so slightly-staggered rows still group correctly.
        order = reading_order_idx([centers[i] for i in live], boxes)
        live = [live[k] for k in order]
        pts = [Detection("point", (float(centers[i][0]), float(centers[i][1])))
               for i in live]
        dets = Detections(items=pts)

        name = name or f"yolo_{int(time.time())}"
        odir = out_dir or OUTPUTS_DIR
        # Dots only (no numbers) — cleaner for spotting a missed object; count
        # is in the reply text.
        out = save_annotated(image, dets, f"{name}.png", out_dir=odir, numbered=False)

        # Stage the CROPPED image + predicted labels so a ✓ can promote it to the
        # training set as-is (labels are in cropped-image space, like raw/).
        #
        # THE BOXES ARE THE MODEL'S OWN. This used to throw them away and rebuild
        # squares with points_to_boxes — dot spacing, not object extent — which is
        # why 99.8% of the 10,247 labels in raw/ are pixel-squares: position real,
        # extent synthetic. It cost nothing for counting (one detection per object is
        # all a counter needs) but it is why mAP is meaningless here, and
        # carries tray_crop_pad=0.3 to stop dense images amputating the imprint, and
        # why Phase 0 preserving a hand-drawn box only ever fixed the images a human
        # touched. Every image staged from here carries what the detector measured.
        if stage:
            staged = [Detection(kind="box", coords=tuple(boxes[i]), label="object")
                      for i in live]
            lines = boxes_to_yolo_lines(staged, image.width, image.height)
            # pp=self.pp — staging writes into the project this counter was
            # built for, never anyone else's.
            collect.save_pending(name, image, lines, pp=self.pp)

        return CountResult(
            count=len(pts),
            overlay_path=str(out),
            seconds=round(time.time() - t0, 2),
            stats={"weights": self.weights.name,
                   "conf": self.conf, "iou": self.iou, "imgsz": self.imgsz,
                   "stacked_dropped": stacked,
                   # per-object boxes (image space), for downstream consumers
                   "boxes": [[round(v, 1) for v in b] for b in boxes]},
            name=name,
        )
