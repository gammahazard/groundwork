# Groundwork — ONE image for all three services (app, trainer, scheduler).
#
# python:3.12-slim + pip CUDA wheels, NOT an nvidia/cuda base image: the torch
# wheels carry their own CUDA runtime, so one Dockerfile serves cu128, cu126
# and cpu by build arg, and the host needs only a driver. The torch layer sits
# BEFORE the source COPY so a code change never re-downloads the multi-GB layer.
FROM python:3.12-slim

# tini      PID 1 that reaps children and forwards signals (-g: to the group)
# rsync + openssh-client   dataset sync between paired machines
# git       challenger-stack checkouts into /data/venvs
# curl      the compose healthcheck
# gosu      drop from root to the host uid after chowning /data (entrypoint)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gosu \
        tini rsync openssh-client git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# cu128 (Blackwell cards need it) | cu126 (older cards) | cpu (label/manage
# only — a real channel on the index, installed the same way, just small).
# The versions are the proven pair; override them together if you move.
ARG TORCH_CHANNEL=cu128
ARG TORCH_VERSION=2.11.0
ARG TV_VERSION=0.26.0
RUN pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TV_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"

WORKDIR /app
COPY pyproject.toml ./
COPY groundwork ./groundwork
COPY altmodels ./altmodels
COPY frontend ./frontend
COPY docker/entrypoint.sh ./docker/entrypoint.sh
# deploy/ and scripts/ are not needed to RUN the app — they are needed to
# GROW it: /join.sh serves deploy/join.sh verbatim, and the join bundle tars
# this tree for the enrolling worker (which then wants the unit templates and
# installer). Without them a Docker HQ 500s on the flagship one-command join.
COPY deploy ./deploy
COPY scripts ./scripts

# torch is already satisfied from the channel index above, so the unpinned
# "torch" inside [train] resolves to it — nothing here can pull a generic build.
# [la] is baked in HERE because the runtime user cannot pip into this venv —
# root built it, the app runs unprivileged, and the auto-labeler's accept
# button used to die on `Permission denied: site-packages` in exactly this
# spot. In a container the license click must only ever download weights.
#
# [export] is here for THE SAME REASON, found the same way: ultralytics
# pip-installs a missing exporter mid-export, which in this image fails on that
# identical unwritable site-packages — so an NCNN or OpenVINO export died on
# `No module named 'ncnn'` seconds after the button (measured 2026-08-10). It
# also could not be fixed by installing it into a running container: that lives
# in a layer and is gone at the next rebuild.
RUN pip install --no-cache-dir -e ".[train,bots,la,export]"

# HEADLESS OpenCV. ultralytics depends on opencv-python, whose wheel links
# against X11/Qt shared libraries (libxcb, libGL) that a slim image does not
# carry — so `import cv2` dies with `libxcb.so.1: cannot open shared object
# file` and EVERY training run fails at its first import. This surfaces ONLY in
# a container (a native install has the desktop libs) and only under load, so it
# is invisible until a real run. opencv-python-headless is the same cv2 with no
# GUI deps; the swap is last so nothing reinstalls the GUI build over it. (pip
# check will note ultralytics still names opencv-python — cosmetic; cv2 resolves
# to the headless distribution at runtime, verified by importing both.)
RUN pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true; \
    pip install --no-cache-dir opencv-python-headless

# /data is the only writable place (a host bind mount — see docker-compose.yml).
# HOME lives inside it so pip/model caches survive a container recreate and an
# arbitrary `user:` uid has a home at all. GW_TRAINER=spool because the app
# container must not parent training runs (the trainer service does — that is
# what lets the app be recreated without killing a run). Supervisor process
# model because there is no systemd in a container.
ENV GW_DATA_DIR=/data \
    GW_TRAINER=spool \
    GW_PROCESS_MODEL=supervisor \
    HOME=/data/home \
    YOLO_CONFIG_DIR=/data/home/.config/Ultralytics
# ^ ultralytics writes a settings file on first import and warns (every run) if
#   its config dir is unwritable, falling back to /tmp so it never persists.
#   Pinning it under HOME (which is on the writable /data volume) silences the
#   warning and keeps the setting.

# 1777 on the venvs mountpoint seeds the named volume world-writable on first
# use, so a container running as the HOST uid (not root) can create venvs in it.
RUN mkdir -p /data/venvs \
    && chmod 1777 /data /data/venvs \
    && chmod +x /app/docker/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/app/docker/entrypoint.sh"]
CMD ["python", "-m", "groundwork.web"]
