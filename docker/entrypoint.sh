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
    set -a                    # every assignment in the file becomes exported env
    . "$data/config/.env"
    set +a
fi

exec "$@"
