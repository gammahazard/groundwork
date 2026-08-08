# Architecture

The map of how a request becomes a labelled image, a training run, a score, and a served
model — and where every file lives. Names below are the real ones; when in doubt, the
module docstrings carry the reasoning in more depth than this page.

## Request flow

```
browser / script / another machine
        │
        ▼
uvicorn  (GW_HOST:GW_PORT, default 0.0.0.0:8000)
        │
        ▼
AuthGate — raw ASGI middleware (groundwork/web/auth/middleware.py)
        │   cookie → session → username        (a person)
        │   Authorization: Bearer gw_… → key   (a script/machine; scope enforced here)
        │   open paths: /  /login  /api/login  /api/me  /static/*  /healthz  /favicon.ico
        │   plus the join door (/join.sh, its bundle, the join POST — each validates its
        │   own single-use token) and setup/claim only while zero accounts exist
        ▼
FastAPI app (groundwork/web/app.py — a thin assembler)
   ├── every router in groundwork/web/api/  (one module per area)
   ├── mount /outputs  → StaticFiles over the outputs tree
   └── mount /static   → the SPA (frontend/, no build step)
```

**Why middleware and not route dependencies:** `/outputs` is gigabytes of photographs,
eval previews and checkpoints served by `StaticFiles`. A dependency guards *routes*; it
would have left every one of those files open to anyone who could reach the port. The
gate wraps the whole app, so one rule covers routes and mounts, and the path is
normalised before matching so `/static/../outputs/…` is judged where it lands.

The gate resolves the caller once and stamps `scope["auth_user"]`; everything downstream
(ownership, admin checks, worker read-only guard, busy gates) asks that, not the cookie.

## The project model

There is **no default project**. A fresh instance has zero; the first exists when someone
creates it. Each project is a manifest at `outputs/projects/<slug>/project.json` carrying
its name, `dataset_root`, classes, image-tag vocabulary (`buckets`), `owner`, labelling
style, headline metric, enabled `extensions`, and registered `bots`.

Every HTTP request that touches project data names its project with a `?project=<slug>`
query parameter. The frontend's fetch wrapper (`frontend/cockpit/core.js`) appends it to
every `/api/` call from the URL's own `?project=`, so refresh, bookmark and share keep the
context and no handler plumbs it by hand. Server-side, `deps.project_slug` makes it
**required**: a request naming no project is a 422 and an unknown slug is a 404 with the
slug in it — never a silent fall-back onto somebody else's data, which is this platform's
named most-expensive failure mode. The same rule holds off HTTP: every stage CLI takes
`--project` and prints project, root and image count before working.

Ownership: `deps.readable_by` — unowned projects are open to every signed-in account
(so an existing install keeps working the day auth is switched on), owned ones only to
their owner, and admins read everything (so deleting an account cannot strand its data).

## Dataset layout and collections

Per project, rooted at its `dataset_root` (`groundwork/dataset/paths.py`):

```
outputs/projects/<slug>/
├── project.json
├── dataset/
│   ├── raw/          images/ labels/ preview/   ← the training source, human-curated
│   ├── pending/      images/ labels/            ← inbox (bot captures, uploads)
│   ├── needs_fix/    images/ labels/            ← the fix queue
│   ├── testset/      images/ labels/            ← the FROZEN holdout
│   ├── images/{train,val}/  labels/{train,val}/ ← derived split (machine-local)
│   ├── data.yaml                                ← written by split
│   ├── test_earmark.json                        ← "this image is destined for the holdout"
│   └── runs/<run>/                              ← champion runs: weights, curves, count_eval.json
└── alt/<run>/                                   ← challenger runs (sibling: a split wipe can never touch it)
```

**Collections are exclusive buckets, and an image *moves* between them** — promote from
the fix queue into `raw/` (training) or `testset/` (holdout), never copy. "Never on both
sides of train/test" is therefore true by construction, not by discipline. The holdout is
frozen: nothing in `testset/` is ever trained on or synthesized from, which is what keeps
every score in the ledger comparable forever.

**The split is a seeded hash, not a list.** An image's train/val membership is
`md5(f"{seed}:{stem}")` rank (`pipeline/split.py`), so it is stable per image: adding
images moves only those images, and any machine rebuilds the identical split from the
seed. `split.is_current()` detects when the tree on disk already *is* what a fresh split
would build — in that case the rebuild (an `rmtree` of the split dirs) is pure destruction
of identical files and is skipped, which is what lets a challenger start beside a live
retrain that is reading those directories. `--val-list` can pin known-hard images into val.

## The training pipeline

One run = `split → train → count_eval → run_snapshot → ledger row` (plus timelapse and
heatmap renders), driven by `groundwork/web/retrain_job/`. Stages run as subprocesses
whose module paths live in the `_STAGE` string table — each is told `--project` explicitly
and announces itself before working.

**A run must survive the service that launched it.** Two process models, chosen by
`GW_TRAINER` (`groundwork/web/spawn.py`):

- `direct` (native default): fork detached (`start_new_session`) and rely on the web
  unit's `KillMode=process` — systemd kills by cgroup, which `setsid` does not leave, so
  the unit setting is the load-bearing half, and `groundwork install` bakes it in.
- `spool` (containers): the app writes `<data-dir>/jobs/<name>.json` and
  `groundwork.ops.trainerd` — its own container, untouched by app upgrades — claims it by
  atomic rename and parents the run. Cancel crosses PID namespaces the same way: a
  `jobs/cancel-<name>` file, honoured by the daemon with the same `killpg` the direct
  path uses.

Either way the launched worker owns `outputs/retrain_state.json`, the cross-process
contract the cockpit, bots and stale-healers read.

**GPU admission is per card.** A job's card becomes `CUDA_VISIBLE_DEVICES` (with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so "card 1" is a stable physical device), and the lock
file is derived from it — `outputs/gpu<N>.lock`, held for the whole run. A GPU-holder
crumb in the run's tree answers "is this card busy" *machine-wide* (it scans every
project's alt tree — scoping it per project would let two projects take one card).
Completion is proven by the trainer's `meta.json`, written last, after the loop returns —
a vanished process or crumb is what a crash looks like too.

## The exam

The product metric is **count MAE (per image)** on the frozen holdout — not mAP. mAP
integrates over thresholds; counting happens at one, so a model can post the best mAP in
the fleet and still be the worst counter (a calibration failure mAP structurally cannot
see). mAP is still recorded, as a secondary signal.

One `eval_core` holds the (conf, iou) sweep grids, shared by the champion's exam
(`pipeline/count_eval.py`) and the challenger harness — the two cannot drift. The winning
cell's conf/iou are written into the run's `count_eval.json` and **served from there** at
runtime; thresholds are never hardcoded. The eval also records what a single number
hides:

- `ties` — how many other grid cells scored the same MAE (on a saturated holdout, many
  cells hit 0.0 and "the winner" is a documented tie-break, not a choice);
- `grid_edge` / `grid_edge_winner` — the winner sat on the sweep boundary, so the score
  is a floor, not the model's best;
- `tuned_on` — where the thresholds came from (the untagged, *served* eval file must be
  holdout-tuned; diagnostic re-scores are tagged so nothing served is touched);
- per-image records with `dupes` counts, and per-tag MAE grouping so one image type
  regressing cannot hide inside a blended average.

Every eval writes a preview image per holdout item. Look at them; aggregate numbers have
hidden every interesting bug this platform has caught.

## Serving

Which run a project serves is a **per-project, per-machine** fact:
`outputs/serving/<slug>/active_model.json` (the pin; absent = serve the newest run, at
that run's own trained size) and `serving_engine.json` (champion vs a challenger engine —
switching engines never disturbs the pin, and an engine with no exported weights is
refused rather than silently served). Deliberately *outside* the dataset tree, because
the dataset tree mirrors to workers with `--delete` and a pin is not a fact about a
dataset.

The ad-hoc counter (`/api/count`), the upload pre-labeller, the label audit and the
Telegram bots all get their model through the same `serve/runtime_model.make_counter`
path — and runtime and exam apply the same filters via one switch module
(`serve/serving_filters.py`, currently `DEDUPE` only). If a filter changes, it changes
for both, or the certified MAE measures a system nobody ships.

## Multi-machine

Machines have **roles**: `hq` curates and dispatches (a single-machine install is an HQ
that trains locally); a `worker` trains against a read-only mirror and refuses every
dataset mutation server-side (`web/lab_guard.py`), because its copy is overwritten by the
next sync tick. `GW_ROLE` sets it; `here` is the only built-in machine.

Everything else is the **registry** (`private/machines_registry.json`, written 0600 —
it holds the API key HQ presents to each worker, which is why it lives in `private/`
and not the served `outputs/` tree; a legacy copy under `outputs/` is auto-migrated
out on first read): key, name, url, `role`, `transport`,
`ssh_host`, `ssh_port`, `remote_root`, `backup_target`, `verified`. A request names a
machine *key*, never a URL — no request supplies a destination. Pairing fills the record;
until its data plane is verified, a machine is "setup incomplete" and never dispatched.
The measured inventory (which cards, which venvs, when probed) lives separately in
`outputs/machines.json`, filled by probes of `GET /api/machine/cards`. Project-free
questions get project-free endpoints: `GET /api/machine/status` answers "is this box
busy" with no slug at all, so a fresh worker with zero projects can still answer it.

Data flows, all initiated by HQ (credentials are one-way — see SECURITY.md):

```
        control plane (HTTP + worker-scoped API key)
  HQ ────────────────────────────────────────────────▶ worker
        POST /api/train (dispatch)   GET /api/machine/status, /api/lab/*

        data plane (ssh with HQ's own keypair, private/ssh/)
  HQ ── mirror: rsync dataset per project×worker, 5 min, --delete ──▶ worker
  HQ ◀── adopt: scan + pull finished, scored runs home, 5 min ───── worker
  HQ ── backup: raw/ + testset/ snapshots, 30 h, opt-in ──────────▶ backup_target
```

The mirror is **not a backup** — `--delete` makes the worker a clone (that is its job:
training must see exactly HQ's data, and a remote train *re-syncs first and refuses if
the sync fails*). The backup job is the backup: hardlinked snapshots of the irreplaceable
set only, pruned only after the new copy verifies. Adoption renames a run to the next
free local number, writes the ledger row and stamps a snapshot; on the worker, an
autoscore job scores any finished-but-unscored run, so `train → score → adopt` completes
with no human in the loop. Full walkthrough: [machines.md](machines.md).

## The scheduler

One job table (`groundwork/ops/scheduler.py`), three execution modes running it
identically: the Docker `scheduler` service, native systemd timers *rendered from the
table* by `groundwork install`, and an in-process thread (`GW_SCHEDULER=inline`) for
`groundwork run`.

| job       | every  | where      | runs when                |
|-----------|--------|------------|--------------------------|
| mirror    | 5 min  | HQ         | ≥1 verified worker       |
| adopt     | 5 min  | HQ         | ≥1 verified worker       |
| backup    | 30 h   | HQ         | a `backup_target` is set |
| clean     | daily  | everywhere | always                   |
| autoscore | 5 min  | worker     | always                   |

Every job runs as a bounded subprocess (a wedged rsync must not take the loop down), and
every outcome is stamped into `outputs/jobs_status.json`:

```json
{"mirror": {"last_ok": 1723370000.0, "last_error": null, "running": false, "at": 1723370300.0}}
```

That file is **the** interface the dashboard reads (`state._mirror_health`) — one
contract for all three modes, instead of scraping journalctl. A fresh single-machine
install runs only `clean`; the role predicates keep everything else quiet instead of
erroring at machines that do not exist.

## Where everything lives

```
<data-dir>/                     GW_DATA_DIR; defaults to the repo root
├── outputs/                    everything served under /outputs (gate-protected)
│   ├── projects/<slug>/        manifest + dataset + runs + alt   (see above)
│   ├── serving/<slug>/         active_model.json, serving_engine.json
│   ├── training_history.json   THE ledger — machine-wide, one row per run, each row
│   │                           carries its project (cross-project questions are real)
│   ├── retrain_state.json      the live run's cross-process contract
│   ├── machines.json           measured inventory (cards, venvs, when)
│   ├── jobs_status.json        scheduler outcomes
│   └── gpu<N>.lock             per-card admission locks
├── jobs/                       spool-mode job files + cancel files (GW_TRAINER=spool)
├── private/                    NEVER mounted, NEVER served — that is a property
│   ├── users.json sessions.json api_keys.json audit.jsonl
│   ├── machines_registry.json  paired machines + their API keys (0600)
│   ├── ssh/id_ed25519(.pub) known_hosts        HQ's data-plane identity
│   └── meta.json               schema stamp for upgrades
└── config/.env                 when GW_DATA_DIR is set; else <repo>/.env
```

Credential stores live under `private/` and not `outputs/` because `outputs/` is a
StaticFiles mount: anything in it is one gate-pass away from being downloadable, and file
permissions cannot protect a file from the process that owns and serves it. A directory
that is simply never mounted can't leak by oversight — the next sensitive file added
there is safe by default.
