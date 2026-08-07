"""Is a process alive? ONE answer, for every caller on this box.

There were six copies of this three-line function and they did not agree.
Five caught `OSError` broadly, which swallows `PermissionError` — and a
`PermissionError` from `kill(pid, 0)` means the process EXISTS and belongs to
somebody else, which is the definition of alive. So the same live pid read
BUSY through one import and DEAD through another, depending only on which
module asked. Under Docker (`user:` in compose) and on any box where a
training job runs as a different account, that is not hypothetical.

The direction of the error matters more than the disagreement. Reading a live
trainer as dead is what lets a second job start on a card that is already
working — the failure this fleet has paid for repeatedly. So doubt resolves to
ALIVE: a malformed pid is dead (there is no process to be wrong about), a
missing one is dead, and anything else is alive.

Stdlib only, and no project imports, so the bots and the trainers can use it
without dragging the web tier in.
"""
from __future__ import annotations

import os


def alive(pid) -> bool:
    """True if `pid` names a live process. Nothing / garbage / gone is False;
    a process owned by another user is TRUE, not False."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True          # exists, just not ours — alive is the true answer
    return True
