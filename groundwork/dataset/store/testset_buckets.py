"""Diagnostic buckets for the fixed holdout.

Tag each TEST image by type/difficulty (e.g. tablet / capsule / hard) so count_eval
can report per-bucket MAE instead of one blended number — a single average can stay
flat while capsules regress and tablets improve, cancelling out. These buckets are an
EVALUATION lens ONLY: the model stays single-class "object"; nothing here touches
training, weights, NMS, or the shipped counter. It's a property of the test image,
assigned by you, that the model never sees.

    python -m groundwork.dataset.store.testset_buckets          # list every image + bucket
    python -m groundwork.dataset.store.testset_buckets set IMG_4346 capsule
    python -m groundwork.dataset.store.testset_buckets rm  IMG_4346
"""
from __future__ import annotations
import argparse
import json

from .. import paths, sidecar

# The buckets belong to a project's holdout, so the path is resolved per call
# rather than bound at import (see collect._truth_path for the same reasoning).
def _store(pp=None):
    return pp.DATASET_DIR / "testset_buckets.json"


UNTAGGED = "untagged"


def load(pp=None) -> dict[str, str]:
    # Tolerant on a corrupt file: this is an eval LENS (the model never sees a
    # bucket), so degrading to "untagged" beats 500ing the cockpit's image grid.
    return sidecar.read_json(_store(pp), {})


def save(d: dict[str, str], pp=None) -> None:
    # Atomic: the web service (api/buckets.py) and the CLI both do load -> mutate ->
    # save, so a torn write here would lose every tag at once.
    sidecar.write_json(_store(pp), d, indent=1, sort_keys=True)


def get(stem: str, default: str = UNTAGGED, pp=None) -> str:
    return load(pp).get(stem, default)


def set_bucket(stem: str, label: str, pp=None) -> None:
    d = load(pp)
    d[stem] = label.strip().lower()
    save(d, pp)


def remove(stem: str, pp=None) -> None:
    d = load(pp)
    if d.pop(stem, None) is not None:
        save(d, pp)


def _testset_stems(pp=None) -> list[str]:
    if not pp.TESTSET_LABELS.exists():
        return []
    return sorted(p.stem for p in pp.TESTSET_LABELS.glob("*.txt"))


def bucketize(stems, pp=None) -> dict[str, list[str]]:
    """Group the given stems by their bucket label (untagged ones included)."""
    d = load(pp)
    out: dict[str, list[str]] = {}
    for s in stems:
        out.setdefault(d.get(s, UNTAGGED), []).append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag holdout images into eval buckets.")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("set", help="tag an image")
    s.add_argument("stem"); s.add_argument("label")
    r = sub.add_parser("rm", help="untag an image")
    r.add_argument("stem")
    from ...project import add_argument, load as load_project
    add_argument(ap)
    args = ap.parse_args()
    proj = load_project(args.project)
    pp = paths.for_project(proj)
    print(f"[buckets] project {proj.slug} ({proj.name}) | {proj.dataset_root} | "
          f"{len(_testset_stems(pp))} holdout images")

    if args.cmd == "set":
        set_bucket(args.stem, args.label, pp)
        print(f"{args.stem} -> {args.label.strip().lower()}")
    elif args.cmd == "rm":
        remove(args.stem, pp)
        print(f"{args.stem} -> untagged")
    else:  # list
        stems = _testset_stems(pp)
        if not stems:
            print("no holdout images found (testset/labels is empty)")
            return
        groups = bucketize(stems, pp)
        for bucket in sorted(groups):
            print(f"\n[{bucket}]  ({len(groups[bucket])} images)")
            for st in groups[bucket]:
                print(f"    {st}")


if __name__ == "__main__":
    main()
