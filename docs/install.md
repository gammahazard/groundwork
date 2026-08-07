# Native install

Docker is the recommended path (see [../docker/README.md](../docker/README.md)); this
page is the native one — a venv, systemd user units, and the same first-run wizard.

## Prerequisites

- Linux (or WSL2 — see the notes at the bottom), Python **3.10+**
- An NVIDIA GPU with a working driver, *if* this machine will train. A GPU-less box can
  still run the cockpit, hold the data and label — the Train control will tell you,
  with the reason, what it cannot do.
- `rsync`, `openssh-client` and `git` if this machine will be part of a fleet
- `node` (any recent) only if you want to run the frontend checks

## scripts/install.sh

```sh
git clone https://github.com/gammahazard/groundwork.git
cd groundwork
./scripts/install.sh
```

What it does, in order:

1. **Preflight** — Python version, disk, and whether an NVIDIA driver answers.
2. **Picks the right torch build** by asking the driver for the GPU's compute
   capability, and selects the matching CUDA wheel index (or CPU wheels when there is no
   GPU). This matters: a torch build carries a fixed set of compiled kernels, and a wheel
   without your card's architecture *imports fine and fails at first use*.
3. Creates `.venv` and installs Groundwork with its training extra.
4. Installs `ultralytics` without letting it drag its own torch pin over the one chosen
   in step 2.

Nothing outside the repo directory is touched at this stage.

## `groundwork install`

```sh
.venv/bin/groundwork install [--role hq|worker] [--units-only]
```

Renders the systemd **user** units from `deploy/units/*.tmpl` with your real paths and
enables them (`groundwork/install.py`). Details that are deliberate:

- **The web unit ships `KillMode=process` baked in.** Without it, systemd kills the whole
  cgroup on a service restart — including a detached training run that has been going for
  hours. This is the single most expensive operational lesson this platform carries;
  `setsid` alone does not help, because it does not change a process's cgroup.
- **Timers are rendered from the scheduler's job table** (`groundwork/ops/scheduler.py`),
  so an interval changed there changes systemd and Docker alike — there is no second copy
  of the schedule to drift.
- **`loginctl enable-linger`** is run for you: without linger, user units die at logout,
  and a headless box would stop serving the moment the installing ssh session closed.
- **Idempotent.** Rendering is deterministic and unchanged units are not rewritten, so
  re-running it after `git pull` is the upgrade step, and its diff means something.
- Values interpolated into unit files are *refused* when they look wrong, not escaped —
  a unit file is something systemd executes.

`--role worker` records that this machine is a worker (see
[machines.md](machines.md)); the default is `hq`, and a single-machine install is simply
an HQ that trains locally.

## `groundwork run` (no systemd)

```sh
.venv/bin/groundwork run
```

Foreground everything: the web service, bots under the in-process supervisor
(`GW_PROCESS_MODEL=supervisor`), the scheduler inline (`GW_SCHEDULER=inline`). Right for
trying Groundwork out, or on WSL setups without a user manager — with one stated limit:
**a training run does not survive Ctrl-C here** (the command says so at startup). Use
`groundwork install` or Docker for runs you care about.

## First run

Open `http://<host>:8000`. With zero accounts, the first-run wizard claims the instance:

1. create the admin account (your password, chosen by you — weak or placeholder
   passwords are refused);
2. name the instance/machine;
3. GPU auto-probe (a GPU-less box gets friendly copy, not an error — the probe also runs
   automatically at startup so the Train control is never dead for lack of one);
4. create the first project;
5. optional extras, each behind its own license consent: the LocateAnything-3B
   auto-labeler (a multi-GB download from Hugging Face under NVIDIA's license),
   challenger stacks (DEIM/RTMDet sidecar environments), and a pointer to Telegram bot
   setup.

The wizard stores nothing of its own — it is derived state, open only while
`users.count() == 0`, and latches shut the moment the admin exists.

### Headless bootstrap

Set `GW_ADMIN_USER` and `GW_ADMIN_PASSWORD` (env or `.env`) before first start and the
account is created at boot — the wizard then never appears. This is safe only because of
the emptiness test: it creates the *first* account and is a no-op forever after, so it
cannot be used to reset an existing instance. The account is flagged must-change; change
the password, then remove both variables.

No wizard, no env vars, and locked out? The CLI works from the box itself:

```sh
.venv/bin/python -m groundwork.web.auth.users add <name> --admin
```

## Upgrading

```sh
git pull
.venv/bin/groundwork install --units-only   # re-render units/timers if they changed
systemctl --user restart groundwork-web
```

A web restart does **not** kill a detached training run (that is what
`KillMode=process` is for) — but check the Train tab first anyway: the one thing that can
still break a live run is moving a module a running pipeline has yet to launch as a
stage subprocess.

## If you run under WSL2

Three things worth knowing up front, each one paragraph:

**The GPU path can wedge unkillably.** Every GPU call crosses the paravirtualised
`/dev/dxg` device to the Windows host, and a thread blocked there sits in uninterruptible
D-state — no signal, not even SIGKILL, can reach it until the call returns. This is why
Groundwork never shells out to `nvidia-smi` from a polled endpoint and why card probes
refuse to touch the driver while a job trains (`cards_stale` instead of a hang). If a run
looks frozen, read CPU ticks in `/proc/<pid>/stat` — process existence proves nothing —
and `cat /proc/<pid>/wchan` naming the dxg sync call *is* the diagnosis.

**The VM can vanish and take a run with it.** WSL2 shuts the VM down when no client is
attached; background processes do not hold it open, and neither does an incoming ssh
session. A training log that stops mid-epoch with no traceback and no error is this, not
a GPU fault. Fix: `vmIdleTimeout=-1` under `[wsl2]` in the Windows-side
`C:\Users\<you>\.wslconfig`, then `wsl --shutdown` to apply.

**Windows can reset the GPU under a long kernel.** The display driver's timeout-detection
(TDR) treats a long-running compute kernel as a hang and resets the device. If long
trainings die with device-reset errors, raise `TdrDelay` in the registry
(`HKLM\System\CurrentControlSet\Control\GraphicsDrivers`) — and prefer a card that is not
also driving your display.
