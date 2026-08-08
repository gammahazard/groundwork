# Changelog

All notable changes to Groundwork are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/):
**MAJOR** for breaking changes to the HTTP API, on-disk layout, or unit/env
contracts; **MINOR** for backwards-compatible features; **PATCH** for fixes.

## [Unreleased]

### Added
- "Keep me signed in" checkbox on login; unchecked, the session ends when the browser closes.
- `scripts/release.sh` — one-command release: stamps the changelog, bumps the version, waits for CI, tags, and publishes the GitHub Release.

### Fixed
- Probing a machine updates its GPU list in place instead of re-rendering the whole section.
- Accepting the auto-labeler license now installs its Python dependencies before downloading the weights; it used to fetch weights only, and the first probe died on missing packages.
- transformers pinned back to 4.57.1: the auto-labeler's vendored code cannot load on the 5.x line (measured), and D-FINE imports clean on 4.57.1.

## [0.3.0] - 2026-08-07

Everything in 0.2.0 was tested against real data and a real fleet; 0.3.0 is
what testing the paths 0.2.0 could not reach turned up. The two headliners:
one-command worker join (mint a command on your HQ, paste it on any GPU box,
it enrolls itself), and the GPU Docker stack actually training — proven by
building and running it on a real two-card machine, which is where four
container-only bugs were hiding. Plus a full-repo audit pass, remote
challenger training working end to end for the first time, and the first-run
experience hardened by installing from scratch in a browser.

### Added
- **One-command worker join**: the Machines tab mints a single-use,
  30-minute join command; run on a GPU box it downloads the source bundle
  from the HQ itself (private-repo friendly, version-matched), installs,
  self-configures as a worker — own admin account, key + pairing ticket,
  service started — and announces itself. The HQ runs the same enroll chain
  as the manual paste (register → trust host keys → ticketed key install →
  data-plane test → card probe). New: `/join.sh`, `/api/join/bundle`,
  `/api/machines/join-token`, `/api/machines/join`; `join_worker` CLI.
- `docs/api.md`: the API reference — auth model, scopes, the project rule,
  error conventions, worked curl examples, and a generated catalog of every
  route (100 today) that CI regenerates and diffs so it cannot go stale. The
  in-app API tab gains the machine-pairing family and a bots group.
- README: table of contents, section dividers, and a marked case-study block
  telling the origin deployment's story (owner-approved; the vocabulary gate
  exempts exactly that delimited block and nothing else — mutation-proven).
  GitHub description and topics set for discoverability.
- In-app screenshots in the README and the guide (editor, live training,
  Try-it), shot on a synthetic dataset generated for the purpose — fully
  owned imagery, allowlisted by name in the media gate.

### Fixed
- **The GPU Docker stack could not train — now it does, proven on real
  two-GPU hardware.** Building the compose stack (app + trainer + scheduler)
  and dispatching a run through the spool surfaced four container-only bugs,
  each of which failed every training run and none of which a native install
  shows: ultralytics' `cv2` links against X11/Qt libraries a slim image lacks
  (→ `opencv-python-headless`); the base weights download to the cwd, which
  was the read-only code tree (→ training runs in the writable data dir, a
  no-op on native); the arbitrary container uid has no `/etc/passwd` entry, so
  torch's cache setup died in `getpwuid` (→ the entrypoint adds one); and the
  VRAM fit guard logged **any** mid-training crash as an OOM, so an unrelated
  failure then permanently refused a good config with a misleading "needs more
  VRAM" message (→ a failure is recorded only on an actual CUDA
  out-of-memory). End to end after the fixes: dispatch → trainer container →
  GPU → evaluate → scored weights on disk, and GPU inference in the Try-it
  tab. Docs gained the two-`.env` distinction (a repo-root `.env` is ignored
  with `-f docker/…`) and the real image size (~13 GB).
- **First-run polish, found by installing from scratch in a browser.** A run
  could be *started* with zero labelled images — a doomed pipeline that read
  as a failed run, now a refusal that names the fix queue. The empty holdout
  score rendered as a filled green bar on a project that had never trained;
  "All model dots" was enabled with no model and did nothing; the project bar
  said "0 images" right after an upload (now "N to label" beside it); the
  editor canvas sat flush-left with half the stage black; the setup wizard
  showed "This machine" instead of the hostname; an empty error box painted a
  rule under the first-run form; a chart loading skeleton let a phone scroll
  sideways; and the Try-it tab — a bare file input — was rebuilt to match the
  rest of the cockpit and no longer leaves "counting…" on screen when the
  server refuses.
- **A joined worker now survives a reboot.** The one-command join started the
  cockpit as a detached process and never installed a service, so a paired GPU
  box vanished at its next reboot with nothing on screen to say why. It writes
  and enables the systemd unit (with linger) where there is one, and says which
  it used; the detached path remains the fallback for containers. Two bugs
  under it: `python -m groundwork.install` had **no entry point at all**, so the
  native installer's last line imported it, ran nothing and exited 0 — every
  native install ended looking successful with no units written; and the web
  process never read `.env`, though the unit template said it did, so
  everything the wizard and join persist (GW_PORT, GW_ROLE, GW_SELF_URL) was
  ignored under systemd — measured: a worker joined on 8420 came back on 8000.
  A real environment variable still wins over the file. New
  `check_module_entrypoints` guards the first of those (mutation-proven).
- **Remote challenger training worked end to end for the first time.** Proven
  by dispatching a D-FINE run from an HQ to a paired 3090: it exposed that
  D-FINE could not start anywhere (it runs in the main venv but `transformers`
  ships only in the `[la]` extra — new `[dfine]` extra, and the launcher now
  refuses with the pip line instead of dying at the trainer's first import
  after taking a card); that `autoscore` imported a name `lab_ops` never
  exported, so no challenger run was ever scored; and that four of six
  `lab_proxy` calls sent no `?project=`, so every proxied challenger view
  (status, curve, log, previews) had 422'd silently since the default project
  was removed.
- Remote exports hand-rolled their own ssh options — the one path using
  whatever key the agent offered, with no pinned host keys and no `ssh_port`
  support. They go through the shared identity now. `/join.sh` read its script
  at import, so a missing file took the whole cockpit down at boot; an
  unreachable hub ended a join in a traceback.
- Six copies of `is-this-process-alive` disagreed: five read a process owned by
  another user as DEAD, which under Docker or a differently-owned training job
  is the difference between refusing a second GPU job and starting one. One
  implementation now (`groundwork/procs.py`).
- **Full-repo audit pass**: four parallel read-throughs of every file, then 28
  verified fixes. The ones that would have bitten first: the Export tab bricked
  itself permanently when opened before the first run; the scheduler's adopt and
  clean jobs exited 2 on every tick (auto-adoption had never run); the bot's
  /status, /engine and /model crashed on renamed ledger keys; timelapse and
  heatmap crashed silently after every training run; challenger runs were
  written to a directory the cockpit never reads. Security: worker API keys and
  pairing tickets moved out of the HTTP-served `outputs/` tree, per-run history
  endpoints are scoped to the caller's project (they leaked another project's
  log and accepted notes onto its ledger), a CLI rename no longer orphans
  sessions and keys, the Docker entrypoint parses `.env` instead of sourcing it,
  and guess-list passwords are refused where the docs already claimed. Also: a
  Blackwell card behind an older one got the wrong torch channel on multi-GPU
  boxes; backup verification could never accept a backup once a dotfile existed;
  the review gallery ignored .jpg and .png; supervisor-mode bots never started
  at boot; the run ranking's tie-break read a key nothing writes.
- Data-dir installs (Docker's `/data`, any `GW_DATA_DIR`) could not count,
  sync, adopt, or save a portable project manifest: five call sites resolved
  paths against the install root instead of the data root. Found by running
  the Try-it tab on a data-dir instance; all five now use the data root,
  which equals the install root on a native checkout.

### Security
- transformers bumped to 5.14 (three advisories, two high — RCE-class in
  model-loading paths, all fixed by ≥5.5). The auto-labeler integration needs
  a functional revalidation against 5.x before the next release; it is opt-in
  and loads only weights the user chose, so exposure was limited.

### Known
- The live-run ticker can print "GPU undefined%" for a family with no VRAM
  line; the champion's own job does not always paint its card busy; a
  non-JSON error from /api/count leaves "counting…" on screen. Queued.

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

[Unreleased]: https://github.com/gammahazard/groundwork/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/gammahazard/groundwork/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gammahazard/groundwork/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gammahazard/groundwork/releases/tag/v0.1.0
