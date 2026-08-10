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
        # cu128, NOT cu126: the whole point of this venv over the legacy
        # .venv-deim is lifting the sm_90 kernel ceiling (deim_tv_compat:
        # "what unlocks the 5070 Ti"), and cu126 wheels still stop at sm_90 —
        # measured 2026-08-10 on a two-card worker: torch 2.13.0+cu126
        # answered archs sm_50..sm_90, so the Blackwell card stayed refused
        # right after a successful install. cu128 wheels carry sm_120.
        torch_spec=("torch", "torchvision",
                    "--index-url", "https://download.pytorch.org/whl/cu128"),
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

    def run(argv: list[str]) -> None:
        # Output captured, and SHOWN when the step fails. check=True with a
        # bare capture swallowed pip's own words — an rtmdet install died
        # with nothing in the log but a CalledProcessError, and the actual
        # reason (no torch==2.1.2 wheel for this python) was in the captured
        # stderr nobody could see.
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip()[-800:]
            log(f"FAILED ({' '.join(argv[:4])}…):\n{tail}")
            raise subprocess.CalledProcessError(r.returncode, argv)

    from .config import VENVS_DIR
    venv_dir = VENVS_DIR / st.venv
    py = venv_dir / "bin" / "python"
    if not py.exists():
        log(f"creating {st.venv}" +
            ("" if VENVS_DIR == ROOT else f" under {VENVS_DIR}"))
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", str(venv_dir)])
    log("installing torch build")
    run([str(py), "-m", "pip", "install", "--no-cache-dir", *st.torch_spec])
    if st.pip_specs:
        log("installing stack packages")
        run([str(py), "-m", "pip", "install", "--no-cache-dir", *st.pip_specs])
    for step in st.extra_steps:
        log("running " + " ".join(step))
        run([str(py), *step])
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
    # RE-MEASURE, rather than asking a human to. This used to end with "probe
    # this machine so the new venv's kernels are recorded", and nothing made
    # that happen: the next launch read a map measured before the venv existed
    # and refused every card, stating the venv "is not installed" about one
    # that plainly was. Venvs only — no device calls — so it is safe even while
    # the other card trains.
    try:
        from .web.machine_self import record_local_venvs
        got = (record_local_venvs() or {}).get(st.venv) or {}
        archs = got.get("archs") or []
        log(f"recorded {st.venv}: torch {got.get('torch','?')}"
            + (f", kernels up to {max(archs, key=lambda a: int(a.split('_')[1]))}"
               if archs else "") if got.get("present") else
            f"WARNING: {st.venv} still reads as absent after install")
    except Exception as e:  # noqa: BLE001 — the install itself succeeded
        log(f"could not refresh the venv map ({type(e).__name__}: {e}) — "
            f"press Probe on the Machines tab before training this family")
    log("done")
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
