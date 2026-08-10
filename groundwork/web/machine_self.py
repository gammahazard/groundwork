"""What THIS machine has: its GPUs and which venvs can drive them.

WHY IT EXISTS. `outputs/machines.json` is written by `scratch/probe_machines.py`
over ssh FROM HQ, and machines.py says why: "HQ cannot ask the worker directly,
because the worker's cockpit is behind this branch and has no endpoint that would
answer." This is the answering half. Once every cockpit serves it, registering a
machine is a host plus an API key — no ssh, no manual probe run, and no
newly-added machine reporting zero cards and refusing every model with no hint.

=============================================================================
READING CARDS TOUCHES THE DRIVER, SO IT IS GATED
=============================================================================

`torch.cuda.get_device_properties` crosses the paravirtualised /dev/dxg under
WSL2, exactly like nvidia-smi, and the cost of that is measured and known: a
thread blocked there sits in uninterruptible D-state and CANNOT be killed — not
by SIGKILL, not by systemd — and it has taken this fleet down twice (36 stuck
processes, load 50, ssh timing out; and 149 of them, load average 171).

So three rules, all of which probe_machines.py already follows and this must not
weaken:

  NEVER WHILE TRAINING. If any GPU job is running, cards are not read at all. The
  caller gets `cards_stale` and the reason, never a hang.
  NEVER ON A POLLED PATH. This is opt-in per request (`cards=True`) and cached;
  the cheap identity answer is the default.
  ALWAYS BOUNDED. Through safe_proc, which kills and then ABANDONS a child rather
  than waiting on one that cannot be killed.

VENV INFO IS FREE by comparison — `torch.cuda.get_arch_list()` reads a compiled
constant and touches no device — so it is refreshed even while training. That
matters, because the arch list is what actually decides whether a model can run
on a card, and it is the half most likely to change (a venv gets rebuilt).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

from ..config import ROOT
from . import safe_proc

# The venvs worth asking about — DERIVED from the model registry (every venv a
# family names, plus the main one), so adding a family never means editing a
# second list. A name that does not exist on this box reports present=False.
def _venv_names() -> tuple[str, ...]:
    from ..models import registry
    names = {".venv"} | {m.venv for m in registry.MODELS if m.venv}
    return tuple(sorted(names))

# Reading six venvs is six interpreter starts (~2s). Nothing about a venv changes
# between requests unless someone rebuilds one, so this is cached for long enough
# that a UI can poll the page without re-paying it, and short enough that a
# rebuild shows up without a restart.
_TTL = 300.0
_cache: dict = {}


def _main_pythons() -> list[str]:
    """Interpreters that may hold the MAIN env's torch: the conventional
    ./.venv first, else the interpreter this service runs on — a bare pip
    install or the Docker image, where the running env IS the main env.
    Sidecar venvs never get this fallback; they exist precisely because their
    torch build differs from the main one."""
    conv = ROOT / ".venv" / "bin" / "python"
    out = [str(conv)] if conv.exists() else []
    if str(conv) != sys.executable:
        out.append(sys.executable)
    return out


def main_python() -> str:
    """THE main-env interpreter, for anything that spawns a main-env subprocess.

    The rule above, as one answer instead of a convention: spawners kept
    hardcoding ROOT/.venv/bin/python, which does not exist in the Docker image
    (its venv is /opt/venv) — so every export from a container HQ died at
    spawn with "No such file or directory" (measured 2026-08-09), and a
    challenger launch would have died the same way at its split. bot_roles
    carries its own copy of this fallback for its systemd ExecStart line;
    everything in-process should ask here."""
    return _main_pythons()[0]


def _training_now() -> str | None:
    """What is using a GPU on this box, or None.

    SCANS /proc FOR THE INTERPRETER PLUS THE MODULE — never `pgrep -f`, which
    matches its own command line. That exact footgun has cost 35
    minutes: a `while pgrep -f "count_eval..."` loop waited on ITSELF forever, and
    the progress checks then reported the deadlocked wrapper as "still running".

    BOTH HALVES OF THAT RULE, deliberately. The first cut checked only the
    module substring, so any spectator merely MENTIONING a marker counted as
    training — measured: a week-old `bash … pgrep -f "…ultralytics…"` watcher
    left over on a worker kept its card probe refused indefinitely. A trainer
    is a PYTHON process whose argv carries the module; argv[0] is the
    interpreter, so both are checked.
    """
    for c in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            argv = open(c, "rb").read().decode("utf8", "replace").split("\0")
        except OSError:
            continue
        if "python" not in os.path.basename(argv[0] or ""):
            continue
        rest = "\0".join(argv[1:])
        for marker in ("altmodels.trainers", "ultralytics",
                       "dataset.pipeline.train"):
            if marker in rest:
                return marker
    return None


def _venvs() -> dict:
    """{venv: {present, torch, cuda, archs}} — no device calls, safe any time."""
    out = {}
    for v in _venv_names():
        cands = (_main_pythons() if v == ".venv"
                 else [str(ROOT / v / "bin" / "python")])
        py = next((c for c in cands if Path(c).exists()), None)
        if py is None:
            out[v] = {"present": False}
            continue
        r = safe_proc.run(
            [py, "-c",
             "import torch,json;print(json.dumps({'torch':torch.__version__,"
             "'cuda':torch.version.cuda,'archs':torch.cuda.get_arch_list()}))"],
            timeout=60)
        try:
            out[v] = {"present": True,
                      **json.loads((r.stdout or "").strip().splitlines()[-1])}
        except Exception:  # noqa: BLE001 — a venv that cannot import torch is a
            out[v] = {"present": True, "error": "torch import failed"}   # fact
    return out


def _cards() -> list[dict]:
    """The GPUs, in PCI order. ONLY call when nothing is training — see header.

    CUDA_DEVICE_ORDER=PCI_BUS_ID is not optional: CUDA orders by capability by
    default, so "card 1" silently becomes a different physical GPU than the
    one the lock file names, and two jobs end up on one card believing they
    are on two (the device-order trap — docs/models.md).
    """
    # UNSET CUDA_VISIBLE_DEVICES — do not inherit it. This function answers
    # "what does this MACHINE have", and a service often pins its OWN jobs to
    # one card via CUDA_VISIBLE_DEVICES. Inheriting that pin makes the probe
    # report one card on a multi-card box, and the answer is written straight
    # into machines.json — which decides dispatch — so every family that could
    # only run on a hidden card becomes undispatchable with no error anywhere.
    # What a process was GIVEN is not what the machine HAS.
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    snippet = ("import torch,json;print(json.dumps([{'index':i,"
               "'name':torch.cuda.get_device_properties(i).name,"
               "'sm':'sm_%d%d'%(torch.cuda.get_device_properties(i).major,"
               "torch.cuda.get_device_properties(i).minor),"
               "'vram_gb':round(torch.cuda.get_device_properties(i)"
               ".total_memory/2**30)}"
               "for i in range(torch.cuda.device_count())]))")
    for py in _main_pythons():
        r = safe_proc.run([py, "-c", snippet], timeout=90, env=env)
        try:
            return json.loads((r.stdout or "").strip().splitlines()[-1])
        except Exception:  # noqa: BLE001 — no torch here; try the next one
            continue
    return []


def _host_keys() -> list[str]:
    """This box's sshd public host keys, for the pairing response."""
    out = []
    try:
        from pathlib import Path as _P
        for f in sorted(_P("/etc/ssh").glob("ssh_host_*_key.pub")):
            try:
                out.append(f.read_text(encoding="utf-8").strip())
            except OSError:
                continue
    except Exception:  # noqa: BLE001 — no sshd is a real machine state
        pass
    return out


def describe(cards: bool = False, venvs: bool = False) -> dict:
    """This machine, in the SAME shape probe_machines.py writes.

    Identical on purpose: `machines.cards()` and `machines.venv_info()` read that
    shape already, so an HTTP probe can be written straight into machines.json
    with no translation layer to drift.
    """
    from ..config import MACHINE
    from . import machines as M

    now = time.time()
    hit = _cache.get("d")
    if hit and now - hit["measured"] < _TTL \
            and (hit.get("cards") or not cards) \
            and (hit.get("venvs") or not venvs):
        return hit

    from .. import config as _cfg
    busy = _training_now()
    d = {"machine": MACHINE, "ip": M.self_ip(), "url": M.self_url(),
         # Pairing facts: what this box IS, where its data lives, and its sshd
         # host keys so HQ can record them without trust-on-first-use. Reading
         # /etc/ssh public keys needs no privilege; a box with none (container)
         # returns [] and the wizard shows the manual step instead.
         "role": _cfg.role(), "root": str(ROOT),
         "ssh_host_keys": _host_keys(),
         "cards": [], "busy": bool(busy), "measured": now}
    # SIX INTERPRETER STARTS, ~10 SECONDS. Measured, and the reason this is not
    # simply always included: `import torch` costs seconds per venv, and this
    # repo's most repeated service bug is an expensive call sitting on a path
    # that gets hit casually. Identity is free; the survey is asked for.
    d["venvs"] = _venvs() if venvs else {}
    if not cards:
        d["cards_stale"] = True
        d["why"] = "cards not requested"
    elif busy:
        # NOT AN ERROR, AND NOT AN EMPTY LIST EITHER. Empty would read as "this
        # machine has no GPUs", which is the opposite of the truth — it has one
        # and it is in use. Reuse whatever was last measured and say it is stale.
        d["cards"] = (hit or {}).get("cards", [])
        d["cards_stale"] = True
        d["why"] = (f"a GPU job is running here ({busy}) — reading card "
                    f"properties crosses /dev/dxg and can wedge in D-state, so "
                    f"it is not done while training")
    else:
        d["cards"] = _cards()
        d["cards_stale"] = False
    _cache["d"] = d
    return d
