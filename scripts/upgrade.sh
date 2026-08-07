#!/usr/bin/env bash
# Groundwork upgrade — deliberately boring: refuse to break a live run, pull,
# rebuild whichever kind of install this box actually has.
set -euo pipefail

say()  { printf '%s\n' "$*"; }
fail() { printf 'upgrade: %s\n' "$*" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) fail "unknown argument: $arg (the only flag is --force)" ;;
    esac
done

# --- refuse while a run is live ----------------------------------------------
# Read host-side, no API call: this works even when the app is down, and
# retrain_state.json is the cross-process contract every trainer writes.
# All layouts are checked — Docker keeps data under docker/data, native under
# outputs/ — because guessing wrong here costs someone a run's GPU hours.
live=""
for state in docker/data/outputs/retrain_state.json \
             data/outputs/retrain_state.json \
             outputs/retrain_state.json; do
    if [ -f "$state" ] && grep -Eq '"status":[[:space:]]*"running"' "$state"; then
        live="$state"
        break
    fi
done
if [ -n "$live" ]; then
    if [ "$FORCE" -eq 1 ]; then
        say "WARNING: $live says a training run is LIVE, and --force was given."
        say "WARNING: upgrading now WILL kill that run — its GPU hours are gone."
    else
        fail "a training run is live ($live) — wait for it to finish, or --force to kill it"
    fi
fi

# --- pull --------------------------------------------------------------------
git pull --ff-only \
    || fail "git pull --ff-only refused — local commits or edits; stash or rebase first"

# --- rebuild -----------------------------------------------------------------
# "Which install is this?" is answered by what EXISTS, not by which tools are
# on the box: the compose file ships with the repo, so its presence proves
# nothing, and docker being installed proves nothing either. A compose stack
# with containers wins; a .venv is the native fallback.
compose_file="${GW_COMPOSE_FILE:-docker/docker-compose.yml}"   # cpu stack: GW_COMPOSE_FILE=docker/docker-compose.cpu.yml
if command -v docker >/dev/null 2>&1 \
        && docker info >/dev/null 2>&1 \
        && [ -n "$(docker compose -f "$compose_file" ps -aq 2>/dev/null)" ]; then
    say "compose stack found — rebuilding and restarting"
    docker compose -f "$compose_file" build
    docker compose -f "$compose_file" up -d
    say "done — jobs still waiting in the spool are picked up by the new trainer"
elif [ -x .venv/bin/python ]; then
    say "native install found — reinstalling and refreshing units"
    .venv/bin/pip install -e ".[train,bots]"
    .venv/bin/groundwork install --units-only
    # Safe on a detached run: the web unit ships KillMode=process for exactly
    # this restart (deploy/units/groundwork-web.service.tmpl explains).
    systemctl --user restart groundwork-web \
        || say "groundwork-web is not installed as a unit — start the app yourself"
    say "done"
else
    fail "nothing to upgrade: no compose containers and no .venv — run scripts/install.sh or docker compose up first"
fi
