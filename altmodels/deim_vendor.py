"""Where the DEIMv2 vendor repo lives — ONE answer.

`groundwork.stacks` clones it to DATA_DIR/vendor/DEIMv2 — the pinned,
documented home the manifest system writes. The lab's original hand-built
checkout predates that system and sits at ~/DEIMv2. Four modules each
hardcoded the legacy path (trainer, entry shim, predictor, ONNX exporter),
so a stack installed the documented way produced a vendor repo none of them
could find, and training refused with "vendor repo missing" on a machine
where the install had just succeeded.

Resolution prefers the pinned home and falls back to the legacy one, so both
generations of install keep working and the answer lives in one place.
Import-safe under the sidecar venvs: groundwork.config is stdlib-only.
"""
from __future__ import annotations

from pathlib import Path


def vendor_dir() -> Path:
    from groundwork.config import DATA_DIR
    pinned = DATA_DIR / "vendor" / "DEIMv2"
    return pinned if pinned.exists() else Path.home() / "DEIMv2"
