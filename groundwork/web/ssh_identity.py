"""HQ's OWN ssh identity for the data plane — one keypair, one option set.

Every rsync/ssh this platform runs used to build its own SSH option string and
lean on whatever happened to be in the operator's ~/.ssh. That works on the
machine it was written on and nowhere else: a container has no ~/.ssh, a fresh
install has no key, and three modules each carrying their own `-o` list is how
one of them drifts.

So: a dedicated keypair under private/ssh/ (generated on first use — private/
is already the secrets home: 0700, outside the served tree, and already a
persistent volume in the Docker layout), a known_hosts file WRITTEN FROM THE
PAIRING RESPONSE rather than trusted-on-first-use, and one function that turns
a machine key into the full ssh argv.

BatchMode is load-bearing everywhere: an interactive prompt (password, host
key, VPN re-auth) would hang a timer or a request thread forever. With the
host key delivered at pairing there is nothing legitimate left to prompt for,
so failing fast is correct.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import PRIVATE_DIR

# Under PRIVATE_DIR so the seam applies: with GW_DATA_DIR set (every Docker
# install) the keypair lands on the persistent volume — ROOT would put it in
# the image, where it dies with the container and every pairing breaks.
SSH_DIR = PRIVATE_DIR / "ssh"
KEY_FILE = SSH_DIR / "id_ed25519"
KNOWN_HOSTS = SSH_DIR / "known_hosts"


def ensure_key() -> Path:
    """The HQ keypair, generated on first use. Returns the private key path."""
    if KEY_FILE.exists():
        return KEY_FILE
    SSH_DIR.mkdir(parents=True, exist_ok=True)
    SSH_DIR.chmod(0o700)
    # ssh-keygen writes the pair atomically enough for our purposes and sets
    # 0600 on the private half itself. No passphrase: this key exists so
    # unattended timers can sync; protecting it is private/'s job.
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q",
                    "-C", "groundwork-hq", "-f", str(KEY_FILE)], check=True)
    return KEY_FILE


def public_key() -> str:
    """The public half, for installing on a worker at pairing."""
    ensure_key()
    return (SSH_DIR / "id_ed25519.pub").read_text(encoding="utf-8").strip()


def record_host_keys(host: str, keys: list[str]) -> int:
    """Append a worker's sshd host keys to our known_hosts, keyed by `host`
    (the bare hostname/IP out of user@host). Lines already present are not
    duplicated. The keys arrive in the PAIRING response — a channel already
    authenticated by the pairing code — which is strictly stronger than
    trusting whatever answers the first connection.
    """
    ensure_key()
    bare = host.split("@", 1)[-1].strip()
    have = KNOWN_HOSTS.read_text(encoding="utf-8").splitlines() \
        if KNOWN_HOSTS.exists() else []
    added = 0
    for k in keys:
        k = k.strip()
        if not k or k.startswith("#"):
            continue
        # a host-key line is "<type> <base64> [comment]" — prefix our host
        parts = k.split()
        if len(parts) < 2:
            continue
        line = f"{bare} {parts[0]} {parts[1]}"
        if line not in have:
            have.append(line)
            added += 1
    KNOWN_HOSTS.write_text("\n".join(have) + ("\n" if have else ""),
                           encoding="utf-8")
    KNOWN_HOSTS.chmod(0o600)
    return added


def ssh_argv(machine_key: str, connect_timeout: int = 20) -> list[str]:
    """The full ssh argv prefix for machine `machine_key` — identity, host-key
    policy and keepalives in ONE place. Raises KeyError for a machine with no
    ssh_host recorded."""
    from . import machines as M
    r = M.registered(machine_key)
    host = (r.get("ssh_host") or "").strip()
    if not host:
        raise KeyError(f"machine {machine_key!r} has no ssh_host recorded — "
                       f"finish pairing on the Machines tab")
    ensure_key()
    argv = ["ssh",
            "-i", str(KEY_FILE), "-o", "IdentitiesOnly=yes",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={connect_timeout}",
            "-o", "ConnectionAttempts=2",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]
    port = int(r.get("ssh_port") or 22)
    if port != 22:
        argv += ["-p", str(port)]
    return argv


def ssh_option_string(machine_key: str, connect_timeout: int = 20) -> str:
    """The same thing as one string, for rsync's `-e`."""
    return " ".join(ssh_argv(machine_key, connect_timeout))


def target(machine_key: str) -> tuple[str, str]:
    """(user@host, remote_root) for machine `machine_key` — the two facts every
    data-plane command needs, resolved from the registry in one place."""
    from . import machines as M
    r = M.registered(machine_key)
    host = (r.get("ssh_host") or "").strip()
    root = (r.get("remote_root") or "").strip()
    if not host or not root:
        raise KeyError(f"machine {machine_key!r} has no ssh_host/remote_root — "
                       f"finish pairing on the Machines tab")
    return host, root.rstrip("/")
