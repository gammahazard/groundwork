"""The Machine record and the built-in `here`."""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass

from ...config import OUTPUTS_DIR


@dataclass(frozen=True)
class Machine:
    key: str
    name: str
    what: str
    url: str | None = None          # None = local, no HTTP hop
    # What this machine IS in the fleet: "hq" curates and dispatches; a
    # "worker" trains and holds a read-only mirror. Learned at pairing for
    # registered machines; `here` reports its own from config.role().
    role: str = "hq"
    trains: bool = True
    # Wizard progress: a registered machine is dispatchable only once its
    # pairing verified (reachable + key accepted + data plane tested).
    verified: bool = True

    @property
    def local(self) -> bool:
        return self.url is None


MACHINES: dict[str, Machine] = {
    "here": Machine(
        key="here", name="This machine", url=None,
        # POLICY says yes; CAPABILITY decides — see can_train(): whether this
        # box can train is measured per model, because a card has to HOLD a
        # model, not merely be able to run it.
        trains=True,
        what="Where this cockpit runs. Whether it can train is measured per "
             "model — a card has to HOLD a model, not merely be able to run it."),
}


def get(key: str) -> Machine | None:
    return MACHINES.get((key or "").strip())


def here() -> Machine:
    return MACHINES["here"]


def check(m: Machine) -> tuple[bool, str]:
    """Cheap sanity gate before dispatching to a machine — never a network
    round trip (that is what verify/probe are for; this runs on page loads).

    A registered machine that never finished pairing is refused here: an
    unverified entry is a wizard abandoned halfway, and dispatching a dataset
    to it would be acting on a half-checked address.
    """
    if m.local:
        return True, ""
    if not m.verified:
        return False, (f"{m.name} has not finished pairing — resume setup on "
                       f"the Machines tab")
    if not m.url:
        return False, f"{m.name} has no URL recorded"
    return True, ""


