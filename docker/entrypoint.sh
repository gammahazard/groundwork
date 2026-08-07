#!/bin/sh
# Groundwork container entrypoint. Everything writable lives under GW_DATA_DIR
# (/data, a host bind mount), so the container itself is disposable: recreate
# it and nothing is lost.
#
# Settings are sourced from a mounted config DIRECTORY, never from a
# bind-mounted file: the cockpit writes .env atomically (temp file + rename),
# and a file bind mount pins the inode, so the rename either fails or leaves
# the container reading a different file than the host sees. Mounting the
# directory keeps the rename an ordinary rename.
set -eu

data="${GW_DATA_DIR:-/data}"

# STARTED AS ROOT (the default): fix the bind mount's ownership, then drop to
# the unprivileged user and re-run this script. Docker creates a missing bind
# source owned by root, so the very first `docker compose up` on a fresh
# checkout lands here with a /data the app user cannot write — chown once,
# from the only uid that can, and everything after runs unprivileged. A
# compose file that sets `user:` skips this branch entirely and keeps its
# explicit choice.
if [ "$(id -u)" = "0" ]; then
    uid="${GW_UID:-1000}"; gid="${GW_GID:-1000}"
    mkdir -p "$data"
    chown "$uid:$gid" "$data"
    for d in outputs private config models venvs jobs home; do
        [ -e "$data/$d" ] || mkdir -p "$data/$d"
        chown "$uid:$gid" "$data/$d"
    done
    exec gosu "$uid:$gid" "$0" "$@"
fi

# The full layout, every start — mkdir -p is idempotent, and creating it here
# rather than in the image means a fresh bind mount is usable immediately:
#   outputs/  runs, datasets, ledgers     private/  credentials — 0700, never served
#   config/   .env and friends            models/   downloaded model weights
#   venvs/    challenger-stack venvs      jobs/     the spool trainerd watches
#   home/     $HOME, so tool caches land on the volume, not in the container
mkdir -p "$data/outputs" "$data/private" "$data/config" "$data/models" \
         "$data/venvs" "$data/jobs" "$data/home"
chmod 700 "$data/private"

if [ -f "$data/config/.env" ]; then
    # python-dotenv PARSES, the shell only evals the quoted exports it emits.
    # `.`-sourcing the cockpit-written file word-splits values differently
    # from the app (and under set -eu a token containing a space, quote or $
    # kills the container at boot). Same parser as the app = same values.
    eval "$(python - "$data/config/.env" <<'PYEOF'
import re, shlex, sys
from dotenv import dotenv_values
for k, v in (dotenv_values(sys.argv[1]) or {}).items():
    if v is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
        print(f"export {k}={shlex.quote(v)}")
PYEOF
)"
fi

# The compose port mapping and healthcheck are fixed at 8000, so inside the
# container the listener must be too — GW_PORT from .env is a native-install
# setting and must not move it (the HOST port is the compose-time GW_PORT).
export GW_PORT=8000

exec "$@"
