# Changelog

All notable changes to Groundwork are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/):
**MAJOR** for breaking changes to the HTTP API, on-disk layout, or unit/env
contracts; **MINOR** for backwards-compatible features; **PATCH** for fixes.

## [Unreleased]

### Added
- Dependabot configuration for pip and GitHub Actions (weekly, grouped
  minor/patch for pip so majors stay individual PRs).
- `release` workflow: every `v*` tag publishes versioned images to ghcr.io
  (`:X.Y.Z` / `:latest` for the CUDA build, `-cpu` suffixed for the CPU
  build). The job skips itself while the repository is private.
- Compose honors `GW_IMAGE`, so a published package can be run without a
  local build: `GW_IMAGE=ghcr.io/gammahazard/groundwork:0.1.0 docker compose up`.

### Changed
- Every third-party GitHub Action is pinned to the commit SHA its version tag
  pointed at; the default workflow token is `contents: read`.

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

[Unreleased]: https://github.com/gammahazard/groundwork/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gammahazard/groundwork/releases/tag/v0.1.0
