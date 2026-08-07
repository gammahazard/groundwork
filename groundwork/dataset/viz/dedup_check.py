"""Near-duplicate detector across dataset buckets — catch the same image (same capture,
re-photographed) ending up in both training and test.

Buckets are exclusive file-moves, so the *same file* can't be in two places. This
catches the subtler case: two DIFFERENT files that are really the same arrangement
(an IMG_ and a tg_ of one capture). Uses a perceptual hash (dHash) on image structure:
near-identical re-shoots score a tiny Hamming distance; genuinely different captures of
the same subject score far apart, so same-subject/different-capture is NOT flagged. GPU-free.

    python -m groundwork.dataset.viz.dedup_check                 # raw vs testset
    python -m groundwork.dataset.viz.dedup_check --a raw --b testset --threshold 8
"""
from __future__ import annotations
import argparse

import numpy as np
from PIL import Image

from ..store import labelio

# dHash distance interpretation (64-bit): ~0-6 = near-identical (likely same capture);
# 7-12 = similar (eyeball it); 13+ = different arrangement (fine).
_DEFAULT_THRESHOLD = 8


def _dhash(path, size: int = 8) -> int:
    """64-bit difference hash: compare each pixel to its right neighbour."""
    a = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    diff = a[:, :-1] > a[:, 1:]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hashes(collection: str, pp=None) -> dict[str, int]:
    img_dir, _ = labelio._dirs(collection, pp)
    out = {}
    for p in sorted(img_dir.glob("*.*")):
        if p.suffix.lower() == ".txt":
            continue
        try:
            out[p.stem] = _dhash(p)
        except Exception:  # noqa: BLE001
            pass
    return out


def _count_plausible(a_col, a_stem, b_col, b_stem, pp) -> bool:
    """Could these two images be the SAME capture, judging only by dot count?

    A re-shoot keeps the count within a fix or two — label_twins already encodes
    that as `abs(ca-cb) > max(1, 3%)` and skips pairs that fail it. The image
    hash had no such gate, so it flagged tg_1004507388_1879 ~ _1892 at distance
    7 with counts of 20 and 28. The owner confirmed they are different captures,
    and the geometry signal agreed (score 2.07 against a 0.8 threshold); only
    the hash disagreed, because distance 7 sits inside this module's own
    "7-12 = similar, eyeball it" band while the threshold flags <= 8.

    Same rule, both detectors. A fixed camera and lighting setup makes DIFFERENT
    captures of the same subject look alike to a perceptual hash — that is the exact
    false positive this removes, and it matters because a spurious leak warning
    on the train/test boundary is one somebody eventually stops reading.

    KNOWN LIMIT: a genuine same-capture pair where one side is MISLABELLED will now
    be missed by both detectors. That is a labelling bug rather than a leak, and
    catching it is the label audit's job.
    """
    try:
        ca = len(_label_pts(a_col, a_stem, pp))
        cb = len(_label_pts(b_col, b_stem, pp))
    except Exception:  # noqa: BLE001 — no labels: fall back to flagging it
        return True
    if not ca or not cb:
        return True
    return abs(ca - cb) <= max(1, round(0.03 * max(ca, cb)))


def near_duplicates(a: str = "raw", b: str = "testset",
                    threshold: int = _DEFAULT_THRESHOLD,
                    pp=None, check_counts: bool = True) -> list[tuple[str, str, int]]:
    """Near-duplicate pairs (stem_a, stem_b, distance) with distance <= threshold,
    closest first. a != b => cross-bucket (same capture split across two buckets, a leak).
    a == b => within one bucket (the same capture twice), as unordered pairs, no self-match."""
    ha = _hashes(a, pp)
    pairs = []
    if b == a:
        items = sorted(ha.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = _hamming(items[i][1], items[j][1])
                if d <= threshold and (not check_counts or _count_plausible(
                        a, items[i][0], a, items[j][0], pp)):
                    pairs.append((items[i][0], items[j][0], d))
    else:
        hb = _hashes(b, pp)
        for sa, va in ha.items():
            for sb, vb in hb.items():
                d = _hamming(va, vb)
                if d <= threshold and (not check_counts or _count_plausible(
                        a, sa, b, sb, pp)):
                    pairs.append((sa, sb, d))
    return sorted(pairs, key=lambda t: t[2])


def closest(a: str = "raw", b: str = "testset", top: int = 5,
            pp=None) -> list[tuple[str, str, int]]:
    """The `top` most-similar cross-bucket pairs regardless of threshold (for eyeballing).

    `pp` is a PARAMETER, not a global. It used to reference a name that does not
    exist in this scope at all, so every call raised NameError -- which the web
    endpoint turned into a 500, unnoticed because the call above it in the same
    response raised the same way first.
    """
    ha, hb = _hashes(a, pp), _hashes(b, pp)
    pairs = [(sa, sb, _hamming(va, vb)) for sa, va in ha.items() for sb, vb in hb.items()]
    return sorted(pairs, key=lambda t: t[2])[:top]


# --- label-geometry twins -----------------------------------------------------
# dHash sees pixels, so a re-shoot of the same capture (camera moved, different
# crop/compression) can evade it. But the OBJECT ARRANGEMENT is a fingerprint of
# the capture: two different captures essentially never produce the same constellation
# of dots. So for pairs with (near-)equal counts, compare their label dots after
# normalizing away what a re-shoot changes — centroid translation and per-axis
# scale (which also cancels aspect distortion from normalized YOLO coords).
# Same capture -> mean nearest-neighbour distance near 0; different captures -> about
# the typical dot spacing. The raw distance shrinks with density (~1/sqrt(n)),
# so two DIFFERENT dense fills can coincidentally score low (seen: a yellow and
# a pink ~270-object image at 0.096) — the score is density-corrected by sqrt(n):
# random arrangements land ~1.5+, a genuine re-shoot stays well under ~0.5.
# Complements the image hash; both run in the sweep.
_GEOM_THRESHOLD = 0.8           # on the sqrt(n)-corrected score
_GEOM_MIN_DOTS = 5              # tiny sets align trivially after normalization


def _label_pts(collection: str, stem: str, pp=None) -> list[tuple[float, float]]:
    _, lbl_dir = labelio._dirs(collection, pp)
    f = lbl_dir / f"{stem}.txt"
    pts = []
    if f.exists():
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                _cls, cx, cy, *_ = ln.split()
                pts.append((float(cx), float(cy)))
    return pts


def _norm_pts(pts):
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    sx = xs.std() or 1e-6; sy = ys.std() or 1e-6
    return np.stack([(xs - xs.mean()) / sx, (ys - ys.mean()) / sy], axis=1)


def _geom_dist(p, q) -> float:
    """Symmetric mean nearest-neighbour distance between two normalized dot sets."""
    d2 = ((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)
    return float((np.sqrt(d2.min(1)).mean() + np.sqrt(d2.min(0)).mean()) / 2)


def _all_label_pts(collection: str, pp=None) -> dict[str, list[tuple[float, float]]]:
    """Every image's dots in one bucket OF ONE PROJECT.

    Threading pp matters more here than the NameError above did, because this
    code did not crash: labelio._dirs(collection) falls back to paths.default(),
    so a second project's leak check would silently compare the first project's images
    and report a confident "no duplicates". A wrong answer that looks like a
    right one is this repo's most expensive failure shape.
    """
    img_dir, _ = labelio._dirs(collection, pp)
    out = {}
    for p in sorted(img_dir.glob("*.*")):
        if p.suffix.lower() != ".txt":
            pts = _label_pts(collection, p.stem, pp)
            if len(pts) >= _GEOM_MIN_DOTS:
                out[p.stem] = pts
    return out


def label_twins(a: str = "raw", b: str = "testset",
                threshold: float = _GEOM_THRESHOLD,
                pp=None) -> list[tuple[str, str, float]]:
    """Same-capture suspects by dot geometry: (stem_a, stem_b, distance), closest
    first. Only (near-)equal-count pairs are compared (a re-shoot keeps the
    count within a fix or two). a == b sweeps within one bucket, no self-match."""
    pa = _all_label_pts(a, pp)
    pb = pa if b == a else _all_label_pts(b, pp)
    na = {s: _norm_pts(p) for s, p in pa.items()}
    nb = na if b == a else {s: _norm_pts(p) for s, p in pb.items()}
    pairs = []
    stems_a = sorted(pa)
    stems_b = sorted(pb)
    for i, sa in enumerate(stems_a):
        for sb in (stems_b[i + 1:] if b == a else stems_b):
            ca, cb = len(pa[sa]), len(pb[sb])
            if abs(ca - cb) > max(1, round(0.03 * max(ca, cb))):
                continue
            d = _geom_dist(na[sa], nb[sb]) * (min(ca, cb) ** 0.5)
            if d <= threshold:
                pairs.append((sa, sb, round(d, 3)))
    return sorted(pairs, key=lambda t: t[2])


def main() -> None:
    ap = argparse.ArgumentParser(description="Near-duplicate image check across buckets.")
    ap.add_argument("--a", default="raw")
    ap.add_argument("--b", default="testset")
    ap.add_argument("--threshold", type=int, default=_DEFAULT_THRESHOLD)
    ap.add_argument("--top", type=int, default=5, help="also show the N closest pairs")
    # Every stage that can run as a subprocess names its project and PRINTS it
    # before doing work (CLAUDE.md). A leak check that silently examined the
    # wrong project is the worst thing here to be quietly wrong about.
    from ... import project as project_mod
    project_mod.add_argument(ap)
    args = ap.parse_args()
    from .. import paths
    pp = paths.for_project(project_mod.load(getattr(args, "project", None)))
    print(f"[dedup] project {pp.slug} | root {pp.root} | {args.a} vs {args.b}")

    flagged = near_duplicates(args.a, args.b, args.threshold, pp)
    if flagged:
        print(f"⚠ {len(flagged)} likely-duplicate pair(s) between '{args.a}' and "
              f"'{args.b}' (distance <= {args.threshold}) — review these:")
        for sa, sb, d in flagged:
            print(f"    {sa}  ~  {sb}   (distance {d})")
        print("If a pair is the SAME capture, move one out (they must not span train+test).")
    else:
        print(f"✅ no near-duplicates between '{args.a}' and '{args.b}' "
              f"(distance <= {args.threshold}).")
    twins = label_twins(args.a, args.b, pp=pp)
    if twins:
        print(f"⚠ {len(twins)} same-capture suspect(s) by dot GEOMETRY (re-shoots the "
              f"image hash can miss) — eyeball these:")
        for sa, sb, d in twins:
            print(f"    {sa}  ~  {sb}   (corrected score {d}; <0.8 = same arrangement)")
    else:
        print(f"✅ no label-geometry twins between '{args.a}' and '{args.b}'.")
    print(f"\nClosest {args.top} pairs (sanity check — high distance = different images):")
    for sa, sb, d in closest(args.a, args.b, args.top, pp):
        print(f"    {sa}  ~  {sb}   (distance {d})")


if __name__ == "__main__":
    main()
