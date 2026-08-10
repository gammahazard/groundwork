"""Where the DEIMv2 vendor repo lives — ONE answer.

`groundwork.stacks` clones it to DATA_DIR/vendor/DEIMv2 — the pinned,
documented home the manifest system writes. The lab's original hand-built
checkout predates that system and sits at ~/DEIMv2. Four modules each
hardcoded the legacy path (trainer, entry shim, predictor, ONNX exporter), so
a stack installed the documented way produced a vendor repo none of them could
find, and training refused "vendor repo missing" on a machine where the
install had just succeeded.

STDLIB ONLY, and deliberately NO `groundwork` import. This module loads inside
the sidecar venvs, which hold a different torch build and none of the main
package's dependencies, and inside `deim_entry.py`, which torchrun executes as
a SCRIPT from the vendor directory. DATA_DIR is resolved the same way
`groundwork.config` resolves it — GW_DATA_DIR, else the repo root — because
that is the contract; importing the module that owns it would tie the vendor
path to an environment that cannot import it.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def vendor_dir() -> Path:
    """The pinned home when it exists, else the legacy hand-built checkout."""
    data = Path(os.environ.get("GW_DATA_DIR") or REPO)
    pinned = data / "vendor" / "DEIMv2"
    return pinned if pinned.exists() else Path.home() / "DEIMv2"
