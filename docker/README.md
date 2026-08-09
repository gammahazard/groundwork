# Groundwork on Docker

The primary way to run Groundwork: one image, three services — `app` (web UI +
bots), `trainer` (parents training runs, so the app can be recreated without
killing one), `scheduler` (background jobs). Compose v2 required.

## Published images

Version tags publish images to ghcr.io (CUDA build plus a `-cpu` variant) —
once the repository is public. Run one without building anything:

```sh
GW_IMAGE=ghcr.io/gammahazard/groundwork:latest \
  docker compose -f docker/docker-compose.yml up -d
```

While the repo is private, build locally as below — the release workflow
skips itself (private ghcr storage is smaller than a CUDA torch image).

## Quickstart (GPU)

1. Preflight — prove Docker can see your GPU at all, before blaming Groundwork:

       docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

   If that prints your card, continue. If not, install the NVIDIA container
   toolkit and repeat — nothing below works until this does. (WSL2 works via
   Docker Desktop's GPU support; the same preflight answers it.)

2. Build and start, from the repo root:

       export GW_UID=$(id -u) GW_GID=$(id -g)
       docker compose -f docker/docker-compose.yml up -d --build

   The first build downloads the CUDA torch wheels — several GB, one time.
   Pre-Blackwell card? Prefix with `TORCH_CHANNEL=cu126`.

3. Open http://localhost:8000. The setup wizard takes it from there: admin
   account, instance name, GPU probe, first project.

No NVIDIA GPU? `docker compose -f docker/docker-compose.cpu.yml up -d --build`
runs the same stack on CPU-only torch: collect, label, curate, manage
accounts — and train, slowly (the trainer service is included; a small model
on a small dataset is genuinely feasible on CPU). Heavy training belongs on
a paired GPU machine.

## Where your data lives

Host paths in a compose file resolve relative to the file, so everything lands
in `docker/data/`: datasets, runs, ledgers, credentials (`private/`, mode
0700), settings (`config/`). It is a bind mount on purpose — ordinary files
you can inspect and back up; the containers are disposable, the directory is
not. Because the containers run as `GW_UID:GW_GID` (default 1000:1000), export
those to your own uid/gid before the first `up` or the files belong to someone
else.

The one exception is `gw-venvs`, a named volume mounted at `/data/venvs`:
challenger model stacks build isolated venvs there (for example the vendored
DEIMv2 checkout, `github.com/Intellindust-AI-Lab/DEIMv2` @ `0fff8d4`) —
tens of thousands of small files that are slow on a bind mount and broken on
NTFS/exFAT hosts.

## Windows

Docker Desktop runs every container in its WSL2 VM, so the *container* is
equally fast no matter which shell typed the command. What differs is where
the `./data` bind mount lives — and that gives you two supported routes:

- **Clone inside WSL** (recommended): the data directory is native Linux
  filesystem — full speed, and still browsable from Explorer at
  `\\wsl$\<distro>\home\<you>\groundwork\docker\data`.
- **Straight from PowerShell**: an NTFS checkout puts the bind mount on the
  Windows↔Linux file bridge, which crawls on the many-small-file reads a
  training run is made of. Add the fast-storage overlay and the data moves
  into a named volume inside the VM instead:

      docker compose -f docker-compose.yml -f docker-compose.windows.yml up

  The trade: `/data` is no longer a folder next to the compose file — copy
  things out with `docker compose cp app:/data/outputs ./outputs-copy`.

Line endings are already handled: `.gitattributes` pins every file a container
executes or parses to LF, so a PowerShell `git clone` cannot corrupt
`entrypoint.sh` regardless of `autocrlf`.

## Settings: the `.env` rule

There are **two** kinds of setting, and they live in different places:

- **App settings** (GW_ROLE, bot tokens, admin bootstrap, GW_AUTH…) go in
  `docker/data/config/.env` (see `.env.example` at the repo root). The
  entrypoint sources it on every container start, and the cockpit writes it
  when you change settings in the UI.
- **Compose variables** — `GW_UID`, `GW_GID`, `GW_PORT` (the *host* port),
  `TORCH_CHANNEL`, `GW_IMAGE` — are read by Compose itself to build the
  command, so set them by `export` or an inline prefix, exactly as the
  quickstart shows. **Do not put these in a repo-root `.env`:** with
  `-f docker/docker-compose.yml`, Compose's project directory is `docker/`, so
  it reads `docker/.env` — a root `.env` is silently ignored. If you prefer a
  file over `export`, put it at `docker/.env` or pass `--env-file <path>`.

**Never bind-mount the `.env` file itself.** The cockpit writes it atomically —
temp file, then `os.replace` — and a file bind mount pins the inode, so the
replace fails or the container silently keeps reading the old file. Mounting
the config *directory* (which `./data:/data` already does) keeps the rename an
ordinary rename. If you take one rule from this page, take this one.

## Upgrading

    scripts/upgrade.sh

It refuses to run while a training run is live (see below), pulls, rebuilds
and restarts. `--force` overrides the refusal, loudly.

## Honest limits

- **Recreating the `trainer` container kills a live training run.** The
  trainer parents every run precisely so the *app* can be recreated freely,
  but a container recreate tears down every process inside it and there is no
  KillMode=process equivalent across that boundary. This is exactly why
  `scripts/upgrade.sh` checks for a live run first. Spooled jobs that have not
  started yet survive fine — they are files in `/data/jobs`.
- The GPU image is large (~13 GB on disk, measured): the torch CUDA wheels
  dominate, and that layer is cached — rebuilds after code changes do not
  re-download it. The `-cpu` image is a fraction of the size.
- `docker compose down -v` deletes the `gw-venvs` volume; challenger stacks
  rebuild on next use, which takes minutes, not data. Your `docker/data/` is
  never touched by compose.
