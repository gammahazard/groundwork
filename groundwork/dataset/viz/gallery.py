"""Build a browsable review gallery of the training labels (dots-only overlays).

Renders each raw image with a dot at every LABEL point (the hand-corrected training
data, not fresh predictions), plus an index.html grid to scroll through and verify
the dots. Regenerate anytime after editing labels or adding images.

    ./.venv/bin/python -m groundwork.dataset.viz.gallery
    -> open outputs/dataset/label_review/index.html in a browser
"""
from __future__ import annotations
import html

from PIL import Image

from .. import paths
from ...la.annotate import annotate
from ...la.parse import Detection, Detections

# Per call, not per import — see collect._truth_path for the reasoning.
def _out_dir(pp=None):
    return pp.DATASET_DIR / "label_review"


def _points(stem: str, w: int, h: int, pp=None) -> list[Detection]:
    f = pp.RAW_LABELS / f"{stem}.txt"
    pts = []
    if f.exists():
        for ln in f.read_text().splitlines():
            if ln.strip():
                _, cx, cy, *_ = ln.split()
                pts.append(Detection("point", (float(cx) * w, float(cy) * h)))
    return pts


def build_gallery(pp=None) -> int:
    OUT_DIR = _out_dir(pp)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in pp.RAW_IMAGES.glob("*.jpeg"))
    tiles = []
    for stem in stems:
        img = Image.open(pp.RAW_IMAGES / f"{stem}.jpeg").convert("RGB")
        dets = Detections(items=_points(stem, *img.size, pp=pp))
        annotate(img, dets, numbered=False).save(OUT_DIR / f"{stem}.jpg", quality=85)
        tiles.append((stem, dets.count))

    total = sum(n for _, n in tiles)
    cards = "\n".join(
        f'<figure><a href="{s}.jpg" target="_blank"><img loading="lazy" src="{s}.jpg"></a>'
        f'<figcaption>{html.escape(s)} · <b>{n}</b></figcaption></figure>'
        for s, n in tiles)
    (OUT_DIR / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Label review — {len(tiles)} images</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
 header{{padding:1rem 1.5rem;position:sticky;top:0;background:#0f1115cc;backdrop-filter:blur(6px)}}
 h1{{margin:0;font-size:1.1rem}} .sub{{color:#9aa4b2;font-size:.85rem}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:1rem 1.5rem}}
 figure{{margin:0;background:#171a21;border-radius:10px;overflow:hidden}}
 figure img{{width:100%;display:block}}
 figcaption{{padding:.5rem .7rem;font-size:.85rem;color:#9aa4b2}} figcaption b{{color:#fff;font-size:1rem}}
</style>
<header><h1>Label review — dots per training image</h1>
<div class="sub">{len(tiles)} images · {total} objects total · click an image to open full size · verify every object has a dot</div></header>
<div class="grid">{cards}</div>""", encoding="utf-8")
    return len(tiles)


if __name__ == "__main__":
    import argparse
    from ...project import add_argument, load
    ap = argparse.ArgumentParser(description="Build the label-review contact sheet.")
    add_argument(ap)
    args = ap.parse_args()
    proj = load(args.project)
    pp = paths.for_project(proj)
    print(f"[gallery] project {proj.slug} ({proj.name}) | {proj.dataset_root}")
    n = build_gallery(pp)
    print(f"[gallery] {n} images -> {_out_dir(pp) / 'index.html'}")
