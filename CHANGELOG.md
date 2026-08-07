# Changelog

All notable changes to Groundwork are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/):
**MAJOR** for breaking changes to the HTTP API, on-disk layout, or unit/env
contracts; **MINOR** for backwards-compatible features; **PATCH** for fixes.

## [Unreleased]

## [0.2.0] - 2026-08-07

The first operationally tested cut: every headline feature was exercised
against real data, real Telegram, real Docker, and a real two-machine fleet —
and everything that broke on first contact is fixed here.

### Added
- **Bot intake mode**: before a project's first trained model, photos sent to
  its bot land straight in the fix queue, unlabeled, with a running count in
  the reply; the same bot starts counting the moment a run exists.
- **Pre-label model picker**: uploads can name which completed run draws the
  draft dots (the chosen run's own tuned thresholds apply; serving untouched).
- **Progressive uploads**: one request per file with a live N/M counter, so a
  big drop never looks stuck. Pre-labelling keeps the single batch request —
  the model loads once.
- **"Try it" tab**: counting a photo in the browser is a core project tab.
- Live-run cards state it plainly when a model family prints no progress line,
  instead of a bare label.
- `checks/check_import_targets.py`: every relative and package-absolute import
  — module-level and function-local — must point at a module that exists;
  pyflakes runs in CI for call-time NameErrors.
- Dependabot (pip + Actions, weekly; pip majors stay individual PRs).
- `release` workflow: every `v*` tag publishes versioned images to ghcr.io
  (`:X.Y.Z` / `:latest`, plus `-cpu` variants). Skips itself while the
  repository is private.
- Compose honors `GW_IMAGE`, so a published package runs with no local build.

### Fixed
- Seven call-time crashes no static pass could see: five undefined names left
  by module splits, the bots package shadowing the python-telegram-bot library
  through a stale path shim, and three lazy imports spelled for a pre-move
  layout — the first real training launch and the first real bot start found
  them all.
- The API-key scope picker actually sends the chosen scope; every key used to
  mint with the default regardless of selection.
- `groundwork install` works on first run: the unit renderer refused its own
  schedule values, and rendered job units carried the unit name where the
  module belongs — no job unit could start.
- First `docker compose up` from a fresh clone: the entrypoint chowns the
  root-created bind mount and drops privileges; the app spools training jobs
  so the trainer container (now also in the CPU stack) actually receives them.
- Cross-machine asks carry a project: the fleet GPU total no longer reads
  `(partial)` against a healthy worker, and the worker-watch card renders —
  naming which worker answered.
- The dataset-sync runner's injectable contract matches `safe_proc` — the
  first remote dispatch was the first time the signatures ever met.
- The single-image auto-label probe takes `--project` like every other stage.
- Uploads refresh the fix-queue grid (the old path targeted a removed pane).

### Changed
- Every third-party GitHub Action is pinned to the commit SHA its version tag
  pointed at; the default workflow token is `contents: read`.
- Guide screenshots removed pending neutral re-shoots; the vocabulary gate now
  fails on any media file not allowlisted by name — pixels can leak what text
  scanning cannot see.
- Fleet copy assumes any number of machines; per-machine and per-card jobs are
  drawn on the machine that owns them.

## [0.1.0] - 2026-08-07

First cut. Everything is new; the highlights:

### Added
- **Projects**: each owns its classes, dataset tree, bots and owner. No
  default project — a fresh instance starts blank, and every pipeline stage
  names its project or refuses.
- **Browser labeling**: the dot editor (with boxes mode), collections with a
  frozen holdout by construction, and an image fix queue.
- **Training**: one Train control across model × machine × card. Card choice
  is measured (venv kernel support, VRAM fit) and every refusal carries its
  reason; concurrent jobs on one machine are opt-in per launch.
- **The exam**: runs ranked by count MAE per image on the frozen holdout,
  with operating-point honesty (tied grid cells, grid-edge winners) and
  holdout-saturation warnings.
- **Model registry**: one entry per family. YOLOv8n default; D-FINE built in;
  DEIM and RTMDet as opt-in sidecar stacks; per-family licensing surfaced.
- **Multi-machine**: paste-one-string worker pairing with one-way
  credentials, pre-train dataset sync, run adoption, per-worker backups.
- **Accounts**: sessions for people, scoped API keys for scripts, scrypt,
  login throttling, per-project ownership, append-only audit trail.
- **First-run wizard**: claims the first admin (open exactly while zero
  accounts exist), probes the GPU, creates the first project, and offers the
  optional extras (LocateAnything-3B auto-labeler, challenger stacks) with
  license consent.
- **Telegram bots**: per-project data-collection bots with strict identity
  (no fallback token, no fallback project).
- **Deployment**: Docker Compose (app / trainer / scheduler, one image) and
  native systemd installs rendered from one job table; `GW_DATA_DIR` keeps
  all state in one place.
- **Checks**: 29 CI-safe invariant checks (most with mutation arms), plus
  manual GPU-smoke and real-browser suites.

[Unreleased]: https://github.com/gammahazard/groundwork/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/gammahazard/groundwork/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gammahazard/groundwork/releases/tag/v0.1.0
