# checks/

Behavioural guards for the invariants this platform cannot afford to lose.
Run the CI-safe set with:

    python checks/run.py                 # every checks/check_*.py
    python checks/run.py --only vocab    # one of them

Each check is a standalone script that exits 0/1, sandboxes what it touches
(temp dirs, redirected stores, injected subprocess runners), and is written so
it CAN fail — several carry a mutation arm that breaks the rule under test and
requires the failure to be seen. A check that cannot fail verifies nothing.

## The CI-safe set

| Check | The invariant it protects |
|---|---|
| `check_auth_gate` | Nothing under `outputs/` reaches an unauthenticated caller — the gate is ASGI middleware, so it covers the StaticFiles **mount**, not just routes; project ownership 403s AND hides; keys work but cannot escalate. |
| `check_autoscore` | A run is finished only when `meta.json` exists (written last, after the loop) — a checkpoint on disk is not completion; newest first; a live GPU crumb means busy. |
| `check_bot_identity` | A bot resolves its own token variable, people and project from its unit env; no fallback token, no machine-wide owner — an unclaimed bot answers NOBODY; a mis-named variable is refused, never posted to Telegram. |
| `check_bot_ownership` | A bot belongs to its project's owner; unit names/token vars/logs are DERIVED, never taken from a request; a bot name cannot inject systemd directives; keys cannot install or drive services. |
| `check_bot_staging` | A counter stages predictions into the project it was built for; the store refuses to run with no project named (no ambient fallback tree). |
| `check_cancel_targeting` | Cancel kills its own pipeline's process group or refuses — never a kill-by-pattern; an unidentifiable job is left `running` (fail-safe), not marked failed. |
| `check_class_edit` | A class list is a schema for labels on disk: append/rename allowed, reorder refused, removal refused while any label (including the frozen holdout's) uses the class. |
| `check_deim_gloo` | The `GW_DEIM_GLOO` patch still yields a real, initialised gloo process group that satisfies the vendor's `get_rank()` calls — tested against the real function, extracted by AST so a copy cannot rot. |
| `check_deploy_parity` | The web unit template ships `KillMode=process` (a restart must not kill a detached run), and the rendered timers carry exactly the scheduler table's intervals — mutation-tested by editing the table. |
| `check_detection_ceiling` | No model config may cap detections below the densest labelled image (×1.2 headroom) — a capped detector scores as a bad counter with no error anywhere. |
| `check_editor_mirrors` | The JS editor functions (`frontend/editor/editor_mirrors.js`) stay numerically equal to their Python originals — executed under node on shared fixtures, including a banker's-rounding tie that catches the known drift. |
| `check_exam_freeze` | A frozen, named exam is scored whole or not at all: a missing or relabelled image refuses loudly, so a number under an exam's name is always comparable to its history. |
| `check_api_docs` | `docs/api.md`'s route catalog matches the live app — it is generated from `app.openapi()`, so "every route" stays a promise the build enforces rather than one someone has to remember. |
| `check_module_entrypoints` | Every module a shipped script runs with `python -m` actually has an entry point — `groundwork.install` had none, so the native installer's last step imported it, ran nothing and exited 0. |
| `check_import_targets` | Every relative and `groundwork.*` import resolves to a module that exists — including the ones written INSIDE functions, which compileall and pyflakes both miss and which crash at call time. |
| `check_frontend` | The SPA holds together: every asset tag resolves, every tab pairs a button with a section, every fetched `/api` path is a served route, every script parses, no top-level name collisions, nothing orphaned. |
| `check_gpu_gate` | One authority per card: `/api/train` refuses a second GPU job in BOTH directions, treats "could not ask" as busy (unknown ≠ idle), and honours `allow_concurrent` as a deliberate opt-in. |
| `check_janitor_safety` | The pending-queue janitor can only ever delete its own kinds of files — serving pins, ledgers, locks and logs are untouchable. |
| `check_model_registry` | The registry resolves run names to their own family (longest prefix, order-independent), refuses unknowns instead of misrouting, builds trainer command lines from flags the trainer declares, and only names predictors that exist. |
| `check_no_subprocess_on_poll` | The polled retrain status endpoint spawns no process — GPU facts come from the training log; the one-shot GPU sampler latches off permanently after a wedge. |
| `check_polled_endpoints` | No GET handler blocks on another machine — remote reads go through the cached proxy. |
| `check_private_stores` | Credential stores live under `private/`, never inside the HTTP-served `outputs/` tree. |
| `check_repo_paths` | Every `__file__`-derived root still resolves to the repo — a module moved one level deep silently reads/writes a different tree; stray `*/outputs` dirs are the fingerprint of it having happened. |
| `check_retrain_project` | The retrain orchestrator reads/writes only the project named in its state file; no project named is a refusal, not a guess; another project's retrain cannot hide this project's servable runs. |
| `check_service_subprocess` | Long-lived service code shells out only through `safe_proc` (which abandons an unkillable child instead of hanging a worker thread); nvidia-smi appears nowhere on a service or trainer path. |
| `check_serving_project` | Which model serves is per PROJECT: pins, engine switches and weight resolution never cross projects, and a pin cannot name another project's checkpoint. |
| `check_sizecap` | No source file exceeds 500 lines — the build style is small single-purpose modules. |
| `check_split_reuse` | `split.is_current()` agrees with `build_split()` both ways — the rule that lets a launch safely SKIP a destructive re-split, and refuse when the dataset changed. |
| `check_users` | Passwords are salted scrypt digests; usernames unique; renames keep identity; the bootstrap env cannot touch an existing store; the last admin cannot be removed by accident. |
| `check_vocab` | The repo is origin-silent: no identifiers, addresses or domain vocabulary from the project it was extracted from, anywhere including these checks. |
| `check_vram_fit` | Card choice is VRAM-aware: compute capability GATES, measured peak memory STEERS — mutation-tested against a fake machine map so the choice provably follows the number. |
| `check_worker_freshness` | The mirror's `.last_sync.<machine>` stamps are written per machine and only on success, the dashboard reads the newest, and the rsync excludes stay anchored (`/dataset/images`, never `images/`). |

## Not CI

- **`checks/gpu/`** — manual; needs a GPU and a running cockpit.
  `smoke_all_trainers.py` trains every challenger family for two epochs, one
  at a time; `check_vram_guard.py` exercises the VRAM guard on real hardware.
- **`checks/ui/`** — manual; needs a running instance on `:8000` (or
  `COCKPIT=<url>`) plus playwright (`npm --prefix checks/ui install`,
  `npx playwright install chromium webkit` — the live cockpit check
  drives Safari's engine too). Real-browser assertions for entry
  states, dashboard scope, multi-project isolation and the challenger media
  gallery — the class of composition bug no static check sees.
