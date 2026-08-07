# API reference

Everything the cockpit does goes through this HTTP API — the browser holds no
privileges a script can't have. Three ways to read it:

- **This page**: how auth works, the conventions, worked examples, and a
  complete generated catalog of every route.
- **Interactive**: `/docs` on your instance (Swagger UI over the live OpenAPI
  schema), once signed in.
- **In the cockpit**: the API tab — a *curated* walkthrough of the routes that
  matter most, with copy-paste examples templated for your own instance.

## Authentication

Two credentials, one rule each:

- **A session cookie** is a person. Sign in through the UI (or `POST
  /api/login` with `{"username","password"}` and keep the `Set-Cookie`).
- **An API key** (`Authorization: Bearer gw_…`) is a script or another
  machine. Mint keys in the cockpit under **API → keys**. Keys are **scoped**:

  | Scope | May do |
  |---|---|
  | `read` | GET only |
  | `train` *(default)* | GET plus a whitelist of write paths: runs, exports, counting, auto-label, machine pairing |
  | `full` | everything a session can do **except** account/key management and machine registration — those are session-only, so a leaked key can be revoked but never used to mint its successor or exfiltrate datasets to a new machine |

Failed logins are throttled (delays, never lockouts). Every sign-in, refusal
and key change lands in the append-only audit trail (Admin tab).

## The project rule

Almost every route is **project-scoped and the project is never implicit**:
pass `?project=<slug>` on the query string. A missing project is a 422, not a
fallback — no request can silently operate on the wrong dataset. The handful
of machine-level routes (`/api/machine/status`, `/api/machines…`,
`/api/overview`, `/healthz`, `/api/version`) take no project.

## Errors

Refusals are JSON: `{"detail": "…reason in words…"}` with the right status —
`401` unauthenticated, `403` refused (audited), `404` no such thing, `422`
bad request (including a missing `?project=`), `409` conflict. Reasons are
written for humans; if a refusal surprises you, read its `detail` first.

## Worked examples

```sh
BASE=http://localhost:8000
KEY="gw_…"                                    # minted in the cockpit
AUTH="Authorization: Bearer $KEY"

# What is serving, and how good is it?
curl -s "$BASE/api/state?project=widgets" -H "$AUTH" | jq .served

# Add photos to the fix queue (multipart; repeat -F files= per file)
curl -s -X POST "$BASE/api/upload?project=widgets" -H "$AUTH" \
     -F "files=@bench1.jpg" -F "files=@bench2.jpg"

# Count one image with the serving model (returns count + overlay URL)
curl -s -X POST "$BASE/api/count?project=widgets" -H "$AUTH" \
     -F "image=@bench1.jpg"

# Start a training run (model × machine × card; card optional = auto-pick)
curl -s -X POST "$BASE/api/train?project=widgets" -H "$AUTH" \
     -H "Content-Type: application/json" \
     -d '{"model":"yolov8n","machine":"here","epochs":250,"imgsz":960}'

# Watch it
curl -s "$BASE/api/retrain?project=widgets" -H "$AUTH" | jq '.status,.progress.epoch'

# The run ledger, best first
curl -s "$BASE/api/runs?project=widgets" -H "$AUTH" | jq '.[0]'
```

Machine pairing is the one flow that spans two instances: mint a code on the
worker (`POST /api/machine/pairing-code`, admin session), paste it on the
HQ's Machines tab or `POST /api/machines/pair` — registration, host keys,
key install, data-plane test and card probe happen in that one call. See
[machines.md](machines.md).

## Catalog

Generated from the live application — every route, its methods, and the first
line of its handler's docstring. Regenerate with
`python scripts/gen_api_docs.py`; CI fails if this table goes stale.

<!-- catalog:begin (generated — edit scripts/gen_api_docs.py, not this table) -->

### `audit`

| Route | Methods | What it does |
|---|---|---|
| `/api/audit` | GET | Audit Trail |

### `bots`

| Route | Methods | What it does |
|---|---|---|
| `/api/bots` | GET,POST | List Bots |
| `/api/bots/{key}` | DELETE | Unregister Bot |
| `/api/bots/{key}/allowed` | POST | Set Allowed |
| `/api/bots/{key}/install` | POST | Install |
| `/api/bots/{key}/probe` | POST | Probe Bot |
| `/api/bots/{key}/project` | POST | Move Project |
| `/api/bots/{key}/service` | POST | Service |
| `/api/bots/{key}/token` | POST | Set Token |

### `bucket-vocab`

| Route | Methods | What it does |
|---|---|---|
| `/api/bucket-vocab` | GET,PATCH | Get Vocab |

### `buckets`

| Route | Methods | What it does |
|---|---|---|
| `/api/buckets` | GET | Buckets |
| `/api/buckets/{stem}` | POST | Set Bucket |

### `collect`

| Route | Methods | What it does |
|---|---|---|
| `/api/collect` | GET | Collect Overview |

### `count`

| Route | Methods | What it does |
|---|---|---|
| `/api/count` | POST | Count |

### `crop`

| Route | Methods | What it does |
|---|---|---|
| `/api/crop/{collection}/{stem}` | POST | Crop |

### `dedup`

| Route | Methods | What it does |
|---|---|---|
| `/api/dedup` | GET | Dedup |

### `engine`

| Route | Methods | What it does |
|---|---|---|
| `/api/engine` | GET,POST | Engine |

### `export`

| Route | Methods | What it does |
|---|---|---|
| `/api/export` | GET,POST | Start |
| `/api/export/artifacts` | GET | Artifacts |
| `/api/export/cards` | GET | Cards |
| `/api/export/formats` | GET | Formats |

### `healthz`

| Route | Methods | What it does |
|---|---|---|
| `/healthz` | GET | Healthz |

### `image`

| Route | Methods | What it does |
|---|---|---|
| `/api/image/{collection}/{stem}` | DELETE | Delete |

### `images`

| Route | Methods | What it does |
|---|---|---|
| `/api/images/{collection}` | GET | Images |

### `img/{collection}/{stem}`

| Route | Methods | What it does |
|---|---|---|
| `/img/{collection}/{stem}` | GET | Image |

### `join`

| Route | Methods | What it does |
|---|---|---|
| `/api/join/bundle` | GET | Bundle |

### `join.sh`

| Route | Methods | What it does |
|---|---|---|
| `/join.sh` | GET | Join Script |

### `keys`

| Route | Methods | What it does |
|---|---|---|
| `/api/keys` | GET,POST | List Keys |
| `/api/keys/scopes` | GET | Key Scopes |
| `/api/keys/{kid}` | DELETE | Delete Key |

### `la`

| Route | Methods | What it does |
|---|---|---|
| `/api/la` | DELETE,GET | Status |
| `/api/la/{collection}/{stem}` | POST | Start |

### `lab`

| Route | Methods | What it does |
|---|---|---|
| `/api/lab/img` | GET | Lab Img |
| `/api/lab/log` | GET | Log |
| `/api/lab/runs` | GET | Runs |
| `/api/lab/runs/{run}/detail` | GET | Detail |
| `/api/lab/score` | POST | Score |
| `/api/lab/status` | GET | Status |
| `/api/lab/train` | DELETE,POST | Train |
| `/api/lab/vis` | GET | Vis |

### `lab_status`

| Route | Methods | What it does |
|---|---|---|
| `/api/lab_status` | GET | Lab Status |

### `label_audit`

| Route | Methods | What it does |
|---|---|---|
| `/api/label_audit` | GET,POST | Label Audit Start |
| `/api/label_audit/{collection}/{stem}` | POST | Label Audit Points |

### `login`

| Route | Methods | What it does |
|---|---|---|
| `/api/login` | POST | Login |

### `logout`

| Route | Methods | What it does |
|---|---|---|
| `/api/logout` | POST | Logout |

### `machine`

| Route | Methods | What it does |
|---|---|---|
| `/api/machine/cards` | GET | Cards |
| `/api/machine/pair` | POST | Accept Pair |
| `/api/machine/pairing-code` | POST | Mint Pairing Code |
| `/api/machine/self` | GET | Self Info |
| `/api/machine/status` | GET | Machine Status |

### `machines`

| Route | Methods | What it does |
|---|---|---|
| `/api/machines` | GET,POST | List Machines |
| `/api/machines/join` | POST | Join |
| `/api/machines/join-token` | POST | Mint Join Token |
| `/api/machines/pair` | POST | Hq Pair |
| `/api/machines/{key}` | DELETE | Remove Machine |
| `/api/machines/{key}/probe` | POST | Probe |
| `/api/machines/{key}/sync` | POST | Sync Now |
| `/api/machines/{key}/test` | POST | Test Data Plane |

### `me`

| Route | Methods | What it does |
|---|---|---|
| `/api/me` | GET | Me |
| `/api/me/password` | POST | Change Password |
| `/api/me/username` | POST | Change Username |

### `model`

| Route | Methods | What it does |
|---|---|---|
| `/api/model/activate` | DELETE,POST | Activate |

### `models`

| Route | Methods | What it does |
|---|---|---|
| `/api/models` | GET | Models |

### `overview`

| Route | Methods | What it does |
|---|---|---|
| `/api/overview` | GET | Overview |

### `points`

| Route | Methods | What it does |
|---|---|---|
| `/api/points/{collection}/{stem}` | GET,POST | Get Points |

### `projects`

| Route | Methods | What it does |
|---|---|---|
| `/api/projects` | GET,POST | List Projects |
| `/api/projects/{slug}` | GET | Get Project |
| `/api/projects/{slug}/classes` | GET,PATCH | Get Classes |

### `promote`

| Route | Methods | What it does |
|---|---|---|
| `/api/promote/{stem}` | POST | Promote |

### `promote_testset`

| Route | Methods | What it does |
|---|---|---|
| `/api/promote_testset/{stem}` | POST | Promote Testset |

### `retrain`

| Route | Methods | What it does |
|---|---|---|
| `/api/retrain` | DELETE,GET,POST | Start |

### `root`

| Route | Methods | What it does |
|---|---|---|
| `/` | GET | Index |

### `runs`

| Route | Methods | What it does |
|---|---|---|
| `/api/runs` | GET | Runs |
| `/api/runs/{run}` | GET | Run Detail |
| `/api/runs/{run}/curve` | GET | Run Curve |
| `/api/runs/{run}/images` | GET | Run Images |
| `/api/runs/{run}/log` | GET | Run Log |
| `/api/runs/{run}/note` | POST | Set Note |
| `/api/runs/{run}/peek` | GET | Run Peek |
| `/api/runs/{run}/restore/{stem}` | POST | Restore |
| `/api/runs/{run}/snapshot` | GET | Snapshot |

### `sessions`

| Route | Methods | What it does |
|---|---|---|
| `/api/sessions` | GET | List Sessions |
| `/api/sessions/revoke-others` | POST | Revoke Other Sessions |
| `/api/sessions/{sid}` | DELETE | Revoke Session |

### `setup`

| Route | Methods | What it does |
|---|---|---|
| `/api/setup/claim` | POST | Claim |
| `/api/setup/extras/la` | POST | Extras La |
| `/api/setup/extras/stack` | POST | Extras Stack |
| `/api/setup/facts` | GET | Facts |
| `/api/setup/instance` | POST | Instance |
| `/api/setup/status` | GET | Status |

### `state`

| Route | Methods | What it does |
|---|---|---|
| `/api/state` | GET | State |

### `testset`

| Route | Methods | What it does |
|---|---|---|
| `/api/testset/{stem}` | DELETE,POST | To Testset |

### `train`

| Route | Methods | What it does |
|---|---|---|
| `/api/train` | DELETE,POST | Start |
| `/api/train/options` | GET | Options |

### `truth`

| Route | Methods | What it does |
|---|---|---|
| `/api/truth/{stem}` | POST | Set Truth |

### `upload`

| Route | Methods | What it does |
|---|---|---|
| `/api/upload` | POST | Upload |

### `users`

| Route | Methods | What it does |
|---|---|---|
| `/api/users` | GET,POST | List Users |
| `/api/users/stats` | GET | User Stats |
| `/api/users/{username}` | DELETE | Delete User |

### `version`

| Route | Methods | What it does |
|---|---|---|
| `/api/version` | GET | Version |

<!-- catalog:end -->
