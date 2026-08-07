"""GPU-free unit tests for the pure conversion logic.

    python -m groundwork.dataset.test_export
"""
from __future__ import annotations

from ..la.parse import Detection
from .yolo_export import box_to_yolo_line, boxes_to_yolo_lines
from .point_to_box import points_to_boxes


def test_box_to_yolo_center_and_norm():
    # 100x50 box at top-left of a 1000x1000 image -> center (0.05,0.025), (0.1,0.05)
    d = Detection(kind="box", coords=(0, 0, 100, 50))
    line = box_to_yolo_line(d, 1000, 1000)
    assert line == "0 0.050000 0.025000 0.100000 0.050000", line


def test_box_clamped_to_frame():
    # Box spilling past the right/bottom edge is clamped, not negative-sized.
    d = Detection(kind="box", coords=(900, 900, 1100, 1100))
    line = box_to_yolo_line(d, 1000, 1000)
    cls, cx, cy, w, h = line.split()
    assert 0 <= float(cx) <= 1 and 0 <= float(cy) <= 1
    assert 0 < float(w) <= 0.1 and 0 < float(h) <= 0.1, line


def test_degenerate_box_dropped():
    assert box_to_yolo_line(Detection(kind="box", coords=(10, 10, 10, 10)), 100, 100) is None
    assert boxes_to_yolo_lines([Detection(kind="box", coords=(0, 0, 0, 0))], 100, 100) == []


def test_point_kind_ignored_by_box_export():
    assert box_to_yolo_line(Detection(kind="point", coords=(5, 5)), 100, 100) is None


def test_points_to_boxes_fixed():
    pts = [Detection(kind="point", coords=(500, 500))]
    boxes = points_to_boxes(pts, 1000, 1000, strategy="fixed", fixed_frac=0.1)
    x1, y1, x2, y2 = boxes[0].coords
    assert (x2 - x1, y2 - y1) == (100.0, 100.0)  # 10% of min(W,H)=1000


def test_points_to_boxes_adaptive_dense_smaller_than_sparse():
    dense = [Detection(kind="point", coords=(100, 100)),
             Detection(kind="point", coords=(120, 100))]   # nn dist 20
    sparse = [Detection(kind="point", coords=(100, 100)),
              Detection(kind="point", coords=(400, 100))]  # nn dist 300
    ds = points_to_boxes(dense, 1000, 1000)[0].coords
    sp = points_to_boxes(sparse, 1000, 1000)[0].coords
    dense_w = ds[2] - ds[0]
    sparse_w = sp[2] - sp[0]
    assert dense_w < sparse_w, (dense_w, sparse_w)


def _run():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"  ok {name}")
    print(f"[test_export] {n} passed")


if __name__ == "__main__":
    _run()
