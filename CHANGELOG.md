# Changelog

All notable changes to Groundwork are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/):
**MAJOR** for breaking changes to the HTTP API, on-disk layout, or unit/env
contracts; **MINOR** for backwards-compatible features; **PATCH** for fixes.

## [Unreleased]

### Fixed
- Sidecar stacks (DEIM, RTMDet) install in a container HQ: their venvs now build under `DATA_DIR/venvs` — the volume the Docker image has always shipped world-writable for exactly this — instead of dying at the read-only code tree, and everything that resolves a sidecar interpreter (model registry, machine probe) checks both homes. Native installs keep today's checkout-root layout unchanged, and a container-built venv lives on the data volume, so it survives image rebuilds.
- The DEIMv2 vendor repo is found where the stack installer puts it. Four modules (trainer, entry shim, predictor, ONNX exporter) each hardcoded the legacy `~/DEIMv2`, so a stack installed the documented way (`DATA_DIR/vendor/DEIMv2`) produced a vendor repo none of them could find — training would refuse "vendor repo missing" right after a successful install. One resolver now answers for all four, preferring the pinned home and falling back to the legacy one.
- A failed stack-install step logs the step's own output (pip's actual words) instead of a bare CalledProcessError — an RTMDet install died with the real reason (no torch==2.1.2 wheel for the machine's Python) captured and discarded.
- The DEIM stack installs a cu128 torch instead of cu126. The `.venv-deim13` env exists to lift the legacy build's sm_90 kernel ceiling, but cu126 wheels also stop at sm_90 (measured: a fresh install answered `sm_50..sm_90`, so a Blackwell card stayed refused) — cu128 carries sm_120, so one venv runs both an sm_86 card and a 50-series.

## [0.4.3] - 2026-08-09

### Fixed
- Exports and challenger launches work in a container HQ. Both spawned their subprocess as `ROOT/.venv/bin/python`, which does not exist in the Docker image (its venv is `/opt/venv`) — so every export died at spawn with "No such file or directory", and a challenger launch would have died the same way at its split step. The main-env interpreter is now resolved in one place (`machine_self.main_python`): the conventional `./.venv` when it exists, else the interpreter the service itself runs on — the same rule `machine_self` and the bot units already applied.

## [0.4.2] - 2026-08-09

### Added
- The Trainer states the concurrency limit where someone would reach for it: a disabled "two runs at once (one per card) — coming soon" checkbox, so running two champion runs on different cards of one machine reads as a known limit rather than being discovered as a refusal.

### Fixed
- Chained training runs no longer starve adoption. The pipeline now stamps each run `.complete` once it is fully finished (evaluated, snapshotted, recorded, previews done), and HQ's adopt scan brings stamped runs home even while the worker is already training the next one. Before, the scan skipped the *whole machine* whenever its retrain state said "running" — so back-to-back runs on a multi-GPU box sat finished-but-unadopted until the machine happened to be idle at a 5-minute tick, with the adopt job showing green the whole time (measured 2026-08-09: a finished run with MAE 0.0 waited behind a 90-second idle gap no tick ever landed in). Runs finished by older workers carry no stamp and keep the old cadence — adopted on the first tick that finds the machine idle. Challenger results also come home mid-retrain now; their `count_eval.json` only ever appears after scoring completes, which never overlaps a running retrain.
- The worker-side rename after adoption uses `mv -T`, so a target run name that already exists on the worker (diverged numbering, or the active run's dir now that adoption lands mid-retrain) is a logged refusal instead of silently nesting one run directory inside the other.
- Adoption takes a per-machine lock, so a manual `adopt --scan` (or `--run`) beside the scheduler's own tick can no longer interleave. Two concurrent scans both listed the same un-adopted run before either marked it, and each adopted it — a duplicate ledger row and run dir at HQ, plus a colliding worker-side rename that nested one run directory inside another (measured 2026-08-09). The second adopter now skips with a message instead.

### Known limits
- Two *champion* (yolo) retrains still cannot run concurrently on one machine — the retrain state is one file per machine and `start()` refuses a second, whatever the card count. A retrain plus challenger sidecar runs on other cards is the supported concurrency today; per-card retrain state is future work.

## [0.4.1] - 2026-08-09

### Fixed
- HEIC/HEIF photos work end to end now — an iPhone photo sent as a *file* (uncompressed, not a "photo") was rejected as undecodable. The HEIF opener is registered once for every process (new `pillow-heif` dependency), and the web upload normalizes HEIC to JPEG on store so it also *trains* — YOLO's OpenCV dataloader can't read HEIC, so a stored `.heic` would have broken a later run. (The bot already stored JPEG.)

## [0.4.0] - 2026-08-09

The first public release. The notes below are the most recent round of hardening;
the [README](README.md) is the full picture of what Groundwork does.

### Added
- The join card mints both terminals' commands — the Linux/WSL one and a PowerShell variant (`wsl bash -c …`) for Windows GPU boxes — and machines.md documents the `GW_PORT=<port>` prefix for workers whose 8000 is taken.
- "Keep me signed in" checkbox on login; unchecked, the session ends when the browser closes.
- `scripts/release.sh` — one-command release: stamps the changelog, bumps the version, waits for CI, tags, and publishes the GitHub Release.
- Windows support hardening: `.gitattributes` pins container-bound files to LF, a `docker-compose.windows.yml` overlay gives PowerShell users full-speed storage via a named volume, and the README gains a git-free fetch path (PowerShell downloads the source ZIP itself).

### Fixed
- Re-running the join command now UPGRADES a worker — it used to skip the download whenever the directory already existed, so a worker kept its old code forever and fixes in the hub's bundle never reached a box that had joined once (the bundle excludes data, credentials and the venv, so only code is replaced).
- Probe now finishes an unverified pairing — host keys, ssh echo, rsync dry-run, verified — so a machine whose enroll verification failed is no longer a dead end with no recovering control; the Train tab's options also refetch after a probe instead of showing stale "unavailable" until a page reload.
- An unreadable machines registry raises with the fix in words instead of silently answering "no machines" — a root-owned file made every registered machine vanish from the UI while root-run diagnostics kept seeing them.
- A never-trained instance no longer prints a FileNotFoundError line on every models read.
- A joining worker that announced an unreachable address (WSL NAT, multi-homed) no longer strands the enroll — the hub falls back to the address the join arrived from when only that one answers, and says so; the join command also honours a `GW_SELF_URL=` prefix, persisted like `GW_PORT`.
- The GPU-busy guard counted any process merely mentioning a trainer marker — a leftover watcher's `pgrep` pattern kept a worker's card probe refused indefinitely; it now requires a python interpreter carrying the module, per its own docstring.
- README: the first build's ~10-minute torch download is stated up front, and upgrades say `docker compose up --build` — plain `up` reuses a cached image and silently runs old code.
- Training with exactly one labelled image crashed with a raw dataloader error — the split always reserves one image for validation, so the launch gate now refuses below two, with the reason.
- Docker images ship `deploy/` and `scripts/` — without them a container HQ answered the flagship one-command join with a 500 (`/join.sh` serves `deploy/join.sh` verbatim) and minted source bundles missing the worker's installer and unit templates.
- Join commands minted inside a container advertised the container's own bridge address with no warning — the mint now self-diagnoses (container, loopback) and the Machines card always shows address guidance, loud when the address is known-unreachable.
- Probing a machine updates its GPU list in place instead of re-rendering the whole section.
- Accepting the auto-labeler license now installs its Python dependencies before downloading the weights; it used to fetch weights only, and the first probe died on missing packages.
- Docker images bake the `[la]` extra in — the container venv is root-built and read-only to the app user, so the accept-time pip install died on permissions; in a container the license click now only downloads weights (older images get a clear message instead of pip's permission error).
- transformers pinned back to 4.57.1: the auto-labeler's vendored code cannot load on the 5.x line (measured), and D-FINE imports clean on 4.57.1.

[Unreleased]: https://github.com/gammahazard/groundwork/compare/v0.4.3...HEAD
[0.4.3]: https://github.com/gammahazard/groundwork/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/gammahazard/groundwork/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/gammahazard/groundwork/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/gammahazard/groundwork/releases/tag/v0.4.0
