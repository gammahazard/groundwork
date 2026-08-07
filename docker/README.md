# Groundwork on Docker

The primary way to run Groundwork: one image, three services — `app` (web UI +
bots), `trainer` (parents training runs, so the app can be recreated without
killing one), `scheduler` (background jobs). Compose v2 required.

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
gives you everything except training: collect, label, curate, manage accounts.
Training happens on a paired GPU machine, or on this box once it has a card.

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

## Settings: the `.env` rule

Settings go in `docker/data/config/.env` (see `.env.example` at the repo
root). The entrypoint sources it on every container start, and the cockpit
writes it when you change settings in the UI.

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
- The GPU image is large (roughly 7–9 GB): that is the torch CUDA wheels, and
  the layer is cached — rebuilds after code changes do not re-download it.
- `docker compose down -v` deletes the `gw-venvs` volume; challenger stacks
  rebuild on next use, which takes minutes, not data. Your `docker/data/` is
  never touched by compose.
