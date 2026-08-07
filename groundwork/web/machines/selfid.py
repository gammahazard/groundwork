"""How this box names ITSELF to the rest of a fleet."""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass

from ...config import OUTPUTS_DIR

# ---------------------------------------------------------- this machine ---

def self_ip() -> str | None:
    """This box's reachable address, or None.

    NO SUBPROCESS. This repo's standing rule is to prefer a fact readable from
    the kernel over one you have to ask a command for — the whole safe_proc
    module exists because a shelled-out call on the wrong path wedged this
    cockpit three times.

    WHY IT IS NEEDED AT ALL: the cockpit shows copy-pasteable curl examples for
    API keys, and those are meant to be run FROM ANOTHER MACHINE. Building them
    from the browser's own origin gives `http://localhost:8000` whenever the
    operator happens to be on the box, which is a command that works nowhere
    else — the one place it is guaranteed to be wrong is the place it is for.

    Resolution order: the default-route trick (a UDP connect() sends nothing
    and names the interface the kernel would route out of — right on LANs and
    VPNs alike), then SIOCGIFADDR over common interface names (a VPN interface
    like tailscale0 first: when present it is usually the address the rest of
    the fleet can reach).
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))   # no packet leaves the box
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        import fcntl
        import struct
        for iface in (b"tailscale0", b"eth0", b"wlan0", b"en0", b"enp0s3"):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    packed = fcntl.ioctl(
                        s.fileno(), 0x8915,                   # SIOCGIFADDR
                        struct.pack("256s", iface[:15]))
                    return socket.inet_ntoa(packed[20:24])
                finally:
                    s.close()
            except OSError:
                continue
    except Exception:  # noqa: BLE001 — an offline box is normal, not an error
        pass
    return None


def vpn_ip() -> str | None:
    """Deprecated alias — the generic self_ip() replaced the VPN-specific probe."""
    return self_ip()


def self_url(port: int | None = None) -> str | None:
    """The URL another machine would use to reach this cockpit."""
    override = (os.environ.get("GW_SELF_URL", "")
                or os.environ.get("GW_PUBLIC_URL", "")).strip()
    if override:
        return override.rstrip("/")
    ip = self_ip()
    if port is None:
        port = int(os.environ.get("GW_PORT", "8000"))
    return f"http://{ip}:{port}" if ip else None


