"""Challenger STACK manifests — the one place a sidecar install is defined.

A "stack" is everything a challenger family needs that the main venv cannot
hold: a venv with a DIFFERENT torch build, a vendor repo at a pinned commit,
pretrained weights. The predecessor system left four of these undocumented —
each existed only as a directory someone had built by hand — which is exactly
the knowledge this file exists to pin down.

D-FINE is deliberately NOT here: it trains in the MAIN venv with no vendor
repo, which is why it ships as the built-in Apache-licensed challenger.

install() is called by the setup wizard's extras card and by hand:

    python -m groundwork.stacks deim
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, ROOT


@dataclass(frozen=True)
class Stack:
    key: str
    venv: str                      # directory name under the install root
    license: str
    torch_spec: tuple[str, ...]    # pip args for the torch install
    pip_specs: tuple[str, ...]     # everything else
    vendor_repo: str | None = None
    vendor_ref: str | None = None
    vendor_dir: str | None = None  # under DATA_DIR/vendor/
    notes: str = ""
    extra_steps: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


MANIFESTS: dict[str, Stack] = {
    "deim": Stack(
        key="deim", venv=".venv-deim13", license="Apache-2.0",
        torch_spec=("torch", "torchvision",
                    "--index-url", "https://download.pytorch.org/whl/cu126"),
        pip_specs=("pyyaml", "tensorboard", "scipy", "opencv-python-headless",
                   "pycocotools", "faster-coco-eval", "calflops",
                   "transformers"),
        vendor_repo="https://github.com/Intellindust-AI-Lab/DEIMv2.git",
        vendor_ref="0fff8d4",
        vendor_dir="DEIMv2",
        notes="DEIMv2 N-line (HGNetv2 backbone) — the S/M/L/X variants are "
              "DINOv3-backed under Meta's own license and are NOT offered "
              "here. Uses a current-torch venv; the model registry maps the "
              "family to this venv name."),
    "rtmdet": Stack(
        key="rtmdet", venv=".venv-mmdet", license="Apache-2.0",
        torch_spec=("torch==2.1.2", "torchvision==0.16.2",
                    "--index-url", "https://download.pytorch.org/whl/cu121"),
        pip_specs=("openmim",),
        extra_steps=(("-m", "mim", "install", "mmengine"),
                     ("-m", "mim", "install", "mmcv>=2.0.0,<2.2.0"),
                     ("-m", "mim", "install", "mmdet")),
        notes="mmdetection pins an older torch; its kernels stop at sm_90, so "
              "cards newer than that cannot run this stack — the probe "
              "records it and the Train matrix says so rather than failing "
              "at launch."),
}


def install(key: str, log_cb=print) -> str:
    st = MANIFESTS[key]
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        log_cb(f"[stacks:{key}] {msg}")

    venv_dir = ROOT / st.venv
    py = venv_dir / "bin" / "python"
    if not py.exists():
        log(f"creating {st.venv}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                       check=True, capture_output=True, text=True)
    log("installing torch build")
    subprocess.run([str(py), "-m", "pip", "install", "--no-cache-dir",
                    *st.torch_spec], check=True, capture_output=True, text=True)
    if st.pip_specs:
        log("installing stack packages")
        subprocess.run([str(py), "-m", "pip", "install", "--no-cache-dir",
                        *st.pip_specs], check=True, capture_output=True,
                       text=True)
    for step in st.extra_steps:
        log("running " + " ".join(step))
        subprocess.run([str(py), *step], check=True, capture_output=True,
                       text=True)
    if st.vendor_repo:
        vdir = DATA_DIR / "vendor" / (st.vendor_dir or st.key)
        if not vdir.exists():
            log(f"cloning {st.vendor_repo} @ {st.vendor_ref}")
            vdir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", st.vendor_repo, str(vdir)],
                           check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(vdir), "checkout",
                            st.vendor_ref], check=True, capture_output=True,
                           text=True)
        else:
            log(f"vendor repo already present at {vdir}")
    log("done — probe this machine so the new venv's kernels are recorded")
    return "\n".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stack", choices=sorted(MANIFESTS))
    args = ap.parse_args()
    st = MANIFESTS[args.stack]
    print(f"[stacks] {st.key}: license {st.license}. {st.notes}")
    install(args.stack)


if __name__ == "__main__":
    main()
