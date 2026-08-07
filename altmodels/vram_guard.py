"""Kill a run before it drives the card into the driver-hang zone — via torch.

WHAT IT REPLACES. rtmdet.py and rfdetr.py each carried a copy of this, and both
polled `nvidia-smi` in a daemon thread every 20 seconds for the entire run,
parsing a row index to find "our" card. deim.py imported rtmdet's copy. So every
challenger run spawned a subprocess every 20s, forever.

WHY THAT WAS BACKWARDS. Under WSL2 nvidia-smi crosses the paravirtualised
/dev/dxg, and while a card is loaded that call can wedge in uninterruptible
D-state -- unkillable, and `subprocess.run(timeout=)` does not bound it (its
recovery path ends in an unbounded wait on a process that can never die). A
guard that polls it every 20s therefore leaks one permanently-stuck process per
poll under exactly the conditions it exists to detect. Measured 2026-08-01:
36 stuck processes and load 50 from the same interface in the cockpit, and a
DEIM run whose watchdog child sat in D-state with the whole tree at zero CPU.
The guard written to stop the driver hanging was hanging the driver.

WHY torch IS THE RIGHT SOURCE. `torch.cuda.mem_get_info()` returns
(free, total) for the CURRENT device using the CUDA context this process has
already created. No subprocess, no /dev/dxg ioctl storm, nothing new to wedge.
It is also DEVICE-WIDE -- it counts other processes' allocations too, which is
what the guard actually wants (a neighbouring job filling the card is precisely
the danger) and what `torch.cuda.memory_allocated()` would NOT give.

AND IT DELETES THE ROW INDEX. The old version needed a `smi_row` argument
because nvidia-smi lists every card and the run had to guess which line was its
own -- rfdetr's comment records that row 0 watched the WRONG card on the dual-GPU
lab. Under CUDA_VISIBLE_DEVICES pinning, torch's current device IS our card by
construction, so the whole class of "watching the wrong GPU" goes away.
"""
from __future__ import annotations

import os
import threading
import time

POLL_S = 20
STRIKES = 2          # two consecutive readings, so one spike does not kill a run


def used_mib() -> int | None:
    """Total MiB in use on THIS process's card, or None if it cannot be read.

    Returns None rather than 0 when CUDA is not up yet: 0 would read as "plenty
    of headroom" and is the wrong kind of wrong for a safety guard.
    """
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            # Do NOT force a context from a watchdog thread — the training code
            # owns when CUDA starts, and racing it is how a cold init gets slower
            # still. Skip this tick; the next one will find it up.
            return None
        free, total = torch.cuda.mem_get_info()
        return int((total - free) / 2**20)
    except Exception:  # noqa: BLE001 — a failed read is not a strike
        return None


def start(kill_mib: int | None = None, label: str = "vram-guard",
          headroom_mib: int = 900) -> None:
    """Watch this card; os._exit(3) if it stays above the ceiling.

    Daemon thread, same as before. The behaviour that matters is unchanged --
    two consecutive over-ceiling readings kill the run -- only the source of the
    number is different, and the difference is that this one cannot wedge.

    kill_mib=None DERIVES the ceiling from the card, on the first successful
    reading, as total - headroom_mib. That exists so a LAUNCHER process does not
    have to create a CUDA context purely to call get_device_properties() and
    compute a number the training process could work out for itself. On WSL2 a
    spare CUDA context is not free: it is a second live context on a
    paravirtualised card (see CLAUDE.md on /dev/dxg).
    """
    def poll() -> None:
        strikes = 0
        ceiling = kill_mib
        while True:
            time.sleep(POLL_S)
            used = used_mib()
            if used is None:
                continue
            if ceiling is None:
                # mem_get_info already returned total to compute `used` — take
                # the ceiling from the same reading rather than a second call.
                try:
                    import torch
                    ceiling = int(torch.cuda.mem_get_info()[1] / 2**20) - headroom_mib
                    print(f"[{label}] ceiling {ceiling} MiB "
                          f"(card total - {headroom_mib})", flush=True)
                except Exception:  # noqa: BLE001
                    continue
            kill_mib_ = ceiling
            strikes = strikes + 1 if used > kill_mib_ else 0
            if strikes >= STRIKES:
                print(f"[{label}] {used} MiB > {ceiling} MiB — killing the run "
                      f"before the driver can hang", flush=True)
                os._exit(3)

    threading.Thread(target=poll, daemon=True).start()
