"""Running an export on ANOTHER machine, and bringing the artifact back.

TWO FORMATS NEED THIS, for opposite reasons:

  DEIM      the weights are not here at all. An adopted DEIM run is a score
            snapshot with no .pth, and leg 1 needs the vendor repo and a DEIM
            venv besides — so the export has to happen where the checkpoint is.
  TensorRT  the weights ARE here, but the artifact is not portable. TensorRT
            compiles kernels for one exact GPU, so an engine is only useful if
            it was built on the card that will run it. Choosing the worker's 3090
            therefore means building on the worker, not here.

So this module is the shared plumbing: run a bounded command over ssh, push a
file, pull a file. Nothing about which model or which format.

THE SSH QUOTING RULE, which cost an hour. ssh does not transmit argv — it JOINS
the remaining arguments with spaces and hands one string to the remote login
shell, which parses it again. So ["ssh", host, "bash", "-lc", script] arrives as

    bash -lc cd /home/mongo/counter1 && ... && .venv-deim/bin/python ...

and `bash -lc` receives only `cd`; every && clause after it runs in the OUTER
shell, still in $HOME. It presented as ".venv-deim/bin/python: No such file or
directory" about an interpreter that demonstrably existed. A single simple
command survives this by luck, which is why the cheap probes all appeared to
work and only the real export failed. The script is quoted as ONE word here.

EVERYTHING IS BOUNDED, through safe_proc. A service must never call
subprocess.run(timeout=) — its recovery path ends in an unbounded wait() that a
D-state child can park forever, which is how this cockpit has twice ended up
serving nothing sync.
"""
from __future__ import annotations

import shlex

from . import safe_proc

# The same options mirror.py uses. BatchMode above all: without it a missing key
# makes rsync/ssh SIT on a password prompt, and a timer run piles up behind it.
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]

_SSH_CMD = "ssh " + " ".join(SSH_OPTS)


def ssh(host: str, script: str, timeout: float) -> safe_proc.Result:
    """Run a bash snippet on `host`. See the module docstring on the quoting."""
    return safe_proc.run(
        ["ssh", *SSH_OPTS, host, "bash -lc " + shlex.quote(script)],
        timeout=timeout)


def push(local: str, host: str, remote: str, timeout: float = 900) -> None:
    """Copy one file up, creating its directory first.

    NO --delete AND NO --delete-excluded. --delete-excluded is BANNED in this
    repo — it wiped the Pi once, venv and weights and exam set together. This
    copies a single file into a directory holding other artifacts.
    """
    d = remote.rsplit("/", 1)[0]
    mk = ssh(host, f"mkdir -p {shlex.quote(d)}", timeout=60)
    if mk.returncode != 0:
        raise RuntimeError(f"could not make {d} on {host}: "
                           f"{(mk.stderr or '').strip()[:200]}")
    r = safe_proc.run(["rsync", "-az", "-e", _SSH_CMD, local,
                       f"{host}:{remote}"], timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"could not send {local} to {host}: "
                           f"{(r.stderr or '').strip()[:300]}")


def pull(host: str, remote: str, local: str, timeout: float = 900) -> None:
    """Copy one artifact back. -r because some formats are DIRECTORIES.

    openvino, ncnn and .mlpackage are directory formats; a plain file copy
    silently fetches nothing for those and the caller then reports a missing
    artifact rather than a missing flag.
    """
    r = safe_proc.run(["rsync", "-azr", "-e", _SSH_CMD,
                       f"{host}:{remote}", local], timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"exported on {host} but could not fetch it: "
                           f"{(r.stderr or '').strip()[:300]}")


def machines_with_ssh() -> list[dict]:
    """Registered machines that can be reached for this, newest answer first.

    A machine with no ssh host is not a candidate: the control plane is HTTP and
    an API key, but moving weights and artifacts is rsync — that is deliberate,
    because rsync's incremental delta is why a pre-train sync costs a second
    instead of re-uploading gigabytes.
    """
    from . import machines
    return [r for r in machines.public_registry()
            if r.get("ssh_host") and not r.get("local")]


# The markers machine_self._training_now() looks for locally. Kept in ONE place
# and read remotely too, so "is that box training" cannot mean two different
# things depending on which side is asking.
TRAIN_MARKERS = ("altmodels.trainers", "ultralytics", "dataset.pipeline.train")


def training_now(host: str) -> str | None:
    """What is using a GPU on `host`, or None. Scans /proc, NEVER nvidia-smi.

    Under WSL2 an nvidia-smi call crosses the paravirtualised /dev/dxg, where a
    blocked thread sits in uninterruptible D-state and cannot be killed by
    anything — this fleet has been taken down that way twice, once at load 50
    with ssh timing out. A cmdline scan touches no device.

    And it is a scan of /proc/*/cmdline rather than `pgrep -f`, which matches its
    OWN command line: a pgrep loop here would see the ssh command carrying the
    marker strings and report the machine busy every single time.

    UNREACHABLE IS NOT IDLE. A failed ssh raises rather than returning None,
    because "I could not ask" must never be read as "nothing is running" — that
    is the same mistake as treating a missing gpu_holder crumb as a finished run.

    THE MARKERS NEVER APPEAR ON A COMMAND LINE, and that is not tidiness. Written
    the obvious way — grep -E 'altmodels.trainers|ultralytics|...' on the remote
    — the pattern IS the command line of the grep, so the scan matches itself and
    every machine reports busy forever. Worse, the local ssh process carries the
    same pattern, so asking the worker made HQ's own _training_now() report a
    phantom trainer too: one machine's question corrupting another machine's
    answer. Measured 2026-08-05, on both boxes at once. The remote emits raw
    cmdlines and the matching happens HERE, in Python, where no process carries
    the words. It is the `pgrep -f matches itself` footgun this repo already
    documents, wearing a different hat.
    """
    r = ssh(host, "for c in /proc/[0-9]*/cmdline; do "
                  "tr '\\0' ' ' < \"$c\" 2>/dev/null; echo; done", timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"could not ask {host} whether it is training: "
                           f"{(r.stderr or '').strip()[:200] or 'ssh failed'}")
    for line in (r.stdout or "").splitlines():
        for m in TRAIN_MARKERS:
            if m in line:
                return m
    return None
