# Contributing

Groundwork's conventions are enforced by CI, and each one exists because the codebase has
already paid for its absence. The checks are the contract; this file is the reasoning.

## Commits

Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `build:`, `ci:`).
One concern per commit, so any step reverts alone.

## The 500-line cap — `checks/check_sizecap.py`

No `.py` or `.js` source file over 500 lines, anywhere under `groundwork/`, `frontend/`,
`altmodels/` or `checks/`. Small single-purpose modules are the build style; an informal
limit drifts the moment nobody is looking (the predecessor codebase celebrated a split in
one file's header while five other files sailed past it). A number in CI cannot be argued
with. If your change pushes a file over, split it — and give each extracted piece a header
stating the one invariant it owns.

## Origin-silent vocabulary — `checks/check_vocab.py`

This repo was extracted from a private deployment and carries no trace of it: no original
domain words, machine names, addresses or unit names, in code, comments, docs, or data
keys. The check is zero-tolerance and includes `.md` files. Write in the generic
vocabulary the code uses: image, object, capture, project, worker. (The networking note
in docs/machines.md is the one place allowed to name a specific VPN product, once.)

## Subprocesses in the service — `groundwork/web/safe_proc.py`

**Every subprocess a request handler or long-lived service starts goes through
`safe_proc.run`. Never bare `subprocess.run` in a handler.**

The reason is in safe_proc's header and it generalises beyond any one platform:
`subprocess.run(timeout=)` recovers from a timeout by killing the child and then waiting
on it *without* a timeout. A child wedged in uninterruptible D-state (kernel I/O, a hung
GPU driver call) cannot receive the kill — so the calling thread parks in `wait4()`
forever. FastAPI runs sync endpoints on a bounded thread pool, so each wedged call
permanently eats a worker thread, and the service dies one poll at a time while async
routes still answer — which makes it look like "one slow endpoint", not what it is.
`safe_proc` kills, waits a short grace, then *abandons* the child: a leaked process table
entry is cheap, a worker thread is not.

Enforced by `checks/check_service_subprocess.py` (AST scan plus a behavioural test that
hangs on the old pattern).

## Nothing expensive on polled endpoints

A GET handler must not spawn a process or block on another machine — polling turns any
per-call cost into a sustained load, and it fires most while the system is least able to
absorb it (mid-training). Prefer a fact readable from a file; remote reads go through the
`lab_proxy` cache, which answers instantly with an age. Enforced by
`checks/check_no_subprocess_on_poll.py` (behavioural: makes spawning fatal, then calls the
polled functions) and `checks/check_polled_endpoints.py` (no GET blocks on the network).

## Frontend rules — `checks/check_frontend.py`

`frontend/` is a plain SPA: ~60 classic `<script>` tags and **one shared lexical
environment**. That has consequences the checker exists to catch:

- A duplicate top-level `const`/`let` in two files is a TypeError at load, and the browser
  silently discards the whole second file — one tab quietly dies, everything else works.
- Load order in `index.html` is load-bearing: an extracted file can read and write another
  file's top-level `let` (that is how the editor splits work), but only if its dependency
  loaded first.
- Each extracted file states in its header which invariant it owns and what it shares.

The checker runs six checks: every `<script src>`/`<link href>` resolves to a real file;
every `data-tab` button has a matching `<section id>` and vice versa; every `/api/...`
path fetched in JS exists in the FastAPI route table; every `.js` parses (`node --check`);
no two scripts declare the same top-level name; nothing under `frontend/` is orphaned.
Each check is proven able to FAIL on a deliberate break — a check that cannot fail
verifies nothing.

## `python -m` paths live in strings no import checker sees

The retrain pipeline's stage table (`groundwork/web/retrain_job/state.py`), the bot role
table (`groundwork/web/bot_roles.py`), the scheduler's job table
(`groundwork/ops/scheduler.py`) and the systemd unit templates all name modules as
*string literals*. Moving or renaming a module breaks them with no ImportError anywhere —
the failure arrives later, as a subprocess that does not start. Before moving any module:
grep for its dotted path as a string, then verify by *running* things — the CLI at its new
path, a real pipeline stage — not by reading.

## Dataset CLIs name their project

Every dataset CLI in `groundwork/` takes a required `--project` and prints its project,
its dataset root and its image count before doing work. (Machine-level jobs — the adopt
scan, the janitor — take an OPTIONAL `--project` and iterate every project without one,
because the question they answer is about the box. The challenger trainers under
`altmodels/` are the exception: they are handed an already-converted dataset directory
and never resolve a project themselves.) Silently operating on the wrong data
is this platform's most expensive failure mode, and a stage that announces itself turns a
wrong launch into one readable log line instead of a wrong number three stages later. Keep
the rule when adding a stage.

## Running the checks

Each check runs standalone:

```sh
.venv/bin/python checks/check_vocab.py
.venv/bin/python checks/check_sizecap.py
.venv/bin/python checks/check_frontend.py      # needs node on PATH
...
```

`python checks/run.py` runs the CI-safe set. `checks/gpu/` needs real hardware
and is run manually before a release; `checks/ui/` drives a real browser.

## Releases

Semantic Versioning: MAJOR for breaking changes to the HTTP API, on-disk
layout, or unit/env contracts; MINOR for backwards-compatible features; PATCH
for fixes.

One command does the whole cut:

```sh
scripts/release.sh X.Y.Z            # or --dry-run to preview, changing nothing
```

It refuses on a dirty tree, a non-`main` branch, a version that isn't higher
than the current one, or an empty `[Unreleased]`; then it stamps the changelog,
bumps `pyproject.toml`, commits and pushes, **waits for CI to go green**, tags
`vX.Y.Z`, and creates the GitHub Release from the changelog section. The tag
fires the ghcr publish workflow (a no-op while the repo is private; a real
publish of `ghcr.io/gammahazard/groundwork:X.Y.Z` and `-cpu` once public).

So the only manual work is **keeping `[Unreleased]` stocked as you go** — a line
per change under `### Added` / `### Fixed` / `### Security`. Keep those lines
**short**: one sentence — the change and why it matters, not a post-mortem. The
changelog is the source of truth; the Release page is a copy of it, never the
reverse.
