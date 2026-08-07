"""Click-to-fix point editor for review. Add/remove object points with the mouse.

You prefer points, so this works in point space: each label is a dot at the object
center. Left-click adds a missed object; right-click removes the nearest dot. On
save it converts points back to YOLO boxes (adaptive nearest-neighbour size) so
the training labels stay valid. Built on PyQt5 directly (matplotlib's event
callbacks don't fire reliably under WSLg); PyQt5 is the same stack labelImg uses.

Reading and writing the dots is labelio's job, not this file's — it used to keep
its own copy that dropped each dot's CLASS, so saving an image rewrote every half
object (class 1) as a whole one. Same rule, two implementations, one of them lossy.

    ./.venv/bin/python -m groundwork.dataset.viz.point_editor          # all images
    ./.venv/bin/python -m groundwork.dataset.viz.point_editor --only IMG_4349
    ./.venv/bin/python -m groundwork.dataset.viz.point_editor --collection halves

Controls:
    left-click   add a point (missed object)
    right-click  remove the nearest point
    h            toggle the nearest point half <-> whole
    + / -        zoom in / out (or Ctrl+mouse-wheel)
    d            save + next image
    a            save + previous image
    s            save now
    q / Esc      save + quit

Half objects draw as a hollow SQUARE, whole objects as a filled circle — shape, not
just colour, so the two never rely on telling shades apart.
"""
from __future__ import annotations
import argparse
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image

from ..store import labelio


class Editor(QtWidgets.QWidget):
    def __init__(self, stems: list[str], collection: str):
        super().__init__()
        self.stems = stems
        self.collection = collection
        self.img_dir, _ = labelio._dirs(collection)
        self.i = 0
        self.zoom = 1.0
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.resize(1200, 950)
        self._load()

    # --- data ---
    def _load(self):
        self.stem = self.stems[self.i]
        pil = Image.open(labelio.image_path(self.collection, self.stem)).convert("RGBA")
        self.w, self.h = pil.size
        # Manual PIL->QPixmap (Pillow 12 dropped ImageQt's PyQt5 path). QImage does
        # NOT copy the buffer, so keep a reference alive on self._buf.
        self._buf = pil.tobytes("raw", "RGBA")
        qimg = QtGui.QImage(self._buf, self.w, self.h, QtGui.QImage.Format_RGBA8888)
        self.qimg = QtGui.QPixmap.fromImage(qimg)
        # [x, y, cls] — cls 1 is a HALF object. This editor used to read the dots
        # itself and drop the class on the floor, so a save rewrote every half as
        # a whole. labelio is the one implementation that round-trips it, and the
        # web editor already used it.
        d = labelio.load_points(self.collection, self.stem)
        self.pts = [list(p) for p in d["points"]]
        self.target = d["target"]                    # user-reported true count, if any
        self.zoom = 1.0
        self._retitle()
        self.update()

    def _retitle(self):
        # Show the target count (from a ✗-Wrong report) + how far off we are.
        tgt = ""
        if self.target is not None:
            d = len(self.pts) - self.target
            tgt = f"  TARGET={self.target} ({'+' if d >= 0 else ''}{d})"
        halves = sum(1 for p in self.pts if len(p) > 2 and p[2])
        self.setWindowTitle(f"[{self.i+1}/{len(self.stems)}] {self.collection}/{self.stem}  "
                            f"count={len(self.pts)}"
                            f"{f' ({halves} half)' if halves else ''}{tgt}  "
                            f"L=add  R=remove  h=half  d/a=next/prev  +/-=zoom  q=quit")

    def save(self):
        n = labelio.save_points(self.collection, self.stem, self.pts)
        halves = sum(1 for p in self.pts if len(p) > 2 and p[2])
        print(f"[edit] saved {self.stem}: {n} objects"
              + (f" ({halves} half)" if halves else ""), flush=True)

    # --- geometry: image<->widget mapping (fit-to-widget * zoom, centered) ---
    def _scale(self) -> float:
        return min(self.width() / self.w, self.height() / self.h) * self.zoom

    def _offset(self, s: float):
        return (self.width() - self.w * s) / 2, (self.height() - self.h * s) / 2

    def _to_img(self, x, y):
        s = self._scale(); ox, oy = self._offset(s)
        return (x - ox) / s, (y - oy) / s

    # --- painting ---
    def paintEvent(self, _ev):
        s = self._scale(); ox, oy = self._offset(s)
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(25, 25, 25))
        target = QtCore.QRectF(ox, oy, self.w * s, self.h * s)
        p.drawPixmap(target, self.qimg, QtCore.QRectF(self.qimg.rect()))
        r = max(3.0, min(self.w, self.h) * s / 140)
        for pt in self.pts:
            x, y = pt[0], pt[1]
            cx, cy = ox + x * s, oy + y * s
            if len(pt) > 2 and pt[2]:            # half object: hollow SQUARE
                p.setPen(QtGui.QPen(QtGui.QColor(255, 170, 0), 2.5))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawRect(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r))
            else:                                # whole object: filled circle
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.5))
                p.setBrush(QtGui.QColor(255, 40, 40))
                p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        p.end()

    # --- input ---
    def mousePressEvent(self, ev):
        ix, iy = self._to_img(ev.x(), ev.y())
        if not (0 <= ix <= self.w and 0 <= iy <= self.h):
            return
        if ev.button() == QtCore.Qt.LeftButton:
            self.pts.append([ix, iy, 0])          # new dots are whole objects
            print(f"[edit] +point ({ix:.0f},{iy:.0f}) -> {len(self.pts)}", flush=True)
        elif ev.button() == QtCore.Qt.RightButton and self.pts:
            self.pts.pop(self._nearest(ix, iy))
            print(f"[edit] -point -> {len(self.pts)}", flush=True)
        self._retitle(); self.update()

    def _nearest(self, ix: float, iy: float) -> int:
        return min(range(len(self.pts)),
                   key=lambda k: (self.pts[k][0]-ix)**2 + (self.pts[k][1]-iy)**2)

    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            self.zoom *= 1.15 if ev.angleDelta().y() > 0 else 1/1.15
            self.zoom = max(1.0, min(8.0, self.zoom)); self.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_D:
            self.save(); self.i = (self.i + 1) % len(self.stems); self._load()
        elif k == QtCore.Qt.Key_A:
            self.save(); self.i = (self.i - 1) % len(self.stems); self._load()
        elif k == QtCore.Qt.Key_S:
            self.save()
        elif k == QtCore.Qt.Key_H and self.pts:
            # Toggle half <-> whole under the cursor. Half objects belong in the
            # halves/ collection; split.py refuses to train on a class-1 label,
            # so this marks them rather than silently letting them through.
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
            j = self._nearest(*self._to_img(pos.x(), pos.y()))
            while len(self.pts[j]) < 3:
                self.pts[j].append(0)
            self.pts[j][2] = 0 if self.pts[j][2] else 1
            print(f"[edit] point {j} -> {'HALF' if self.pts[j][2] else 'whole'}",
                  flush=True)
            self._retitle(); self.update()
        elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.zoom = min(8.0, self.zoom * 1.25); self.update()
        elif k == QtCore.Qt.Key_Minus:
            self.zoom = max(1.0, self.zoom / 1.25); self.update()
        elif k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.save(); self.close()

    def closeEvent(self, ev):
        self.save(); ev.accept()


def main():
    ap = argparse.ArgumentParser(description="Click-to-fix point editor.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="edit just these stems (e.g. IMG_4349 IMG_4350)")
    ap.add_argument("--needs-fix", action="store_true",
                    help="edit the needs_fix pile (✗-Wrong samples) instead of raw/. "
                         "After fixing, run `python -m groundwork.dataset.store.collect` "
                         "to move them into the training set.")
    ap.add_argument("--collection", choices=sorted(labelio.COLLECTION_NAMES),
                    help="which collection to edit (default raw; --needs-fix is "
                         "shorthand for --collection needs_fix)")
    args = ap.parse_args()
    collection = args.collection or ("needs_fix" if args.needs_fix else "raw")
    img_dir, _ = labelio._dirs(collection)
    all_stems = sorted(p.stem for p in img_dir.glob("*.*") if p.suffix.lower() != ".txt")
    stems = [s for s in all_stems if s in args.only] if args.only else all_stems
    if not stems:
        raise SystemExit("no matching images in " + str(img_dir))
    print(f"[edit] {len(stems)} images from {collection}/. "
          f"Left-click add, right-click remove, h half, d/a next/prev, q quit.")
    app = QtWidgets.QApplication(sys.argv)
    ed = Editor(stems, collection)
    ed.show()
    app.exec_()


if __name__ == "__main__":
    main()
