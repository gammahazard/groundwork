# Multi-machine training

One cockpit, several GPUs. This page is the operator's guide: what HQ and workers are,
how pairing works, what syncs when, and what to do when ssh says no.

## Concepts

- **HQ** is where people work: the datasets of record, the editor, the ledger, the pins.
  A single-machine install is an HQ that trains locally — nothing here is required until
  you add a second box.
- A **worker** trains and holds a **read-only mirror** of each project's dataset. Every
  dataset-mutating endpoint refuses on a worker, with the reason (edits made there would
  be silently overwritten by the next sync tick — the refusal is the honest version).
  Workers are installed with `GW_ROLE=worker`.
- HQ knows its fleet through the **machine registry**. A training request names a machine
  *key*; the URL, ssh host and root are resolved from the stored record, so no request
  ever supplies a destination. A registered machine that never finished pairing is
  "setup incomplete" and is never dispatched to.

## Adding a worker — one command (recommended)

On the HQ's Machines tab press **Create join command** and run the line it
shows *on the GPU machine, in its Linux terminal* — on a Windows/WSL box that
is the Ubuntu terminal, never PowerShell (the join installs a native Linux
worker):

```sh
curl -fsSL http://<hq>:8000/join.sh | bash -s -- http://<hq>:8000 gwj_…
```

The box downloads Groundwork **from the HQ itself** (no GitHub credential
needed, and always the version the fleet runs), installs into `~/groundwork`
(venv + the right torch wheel channel for its newest card), configures itself
as a worker with its own admin account (random password, printed once), and
announces itself. The HQ registers it, records its sshd host keys, installs
its ssh public key through the same single-use ticket the manual flow uses,
tests the data plane, and probes the cards — the machine appears in the Train
matrix at the end, verified. The join token is single-use and expires in 30
minutes. A box without `sshd` still registers; the output shows the one
manual `authorized_keys` line and the Machines tab holds the resume path.

## Pairing a worker (manual alternative)

On the **worker**:

1. Install Groundwork (`docs/install.md` or Docker) with role `worker`.
2. Its admin mints a **pairing string** from its Machines tab (CLI fallback exists). The
   string bundles what HQ needs to reach it: its URL (editable before minting), a
   `train`-scoped API key minted on the worker, and its sshd host keys.

On **HQ** (admin, signed in — a key cannot register machines):

3. Machines tab → paste the pairing string. HQ **verifies** by calling the worker's
   `/api/machine/self` and diagnoses per status: unreachable, refused, or "no such
   endpoint — that machine runs an older Groundwork".
4. Confirm past the warning that matters: **a registered machine receives this
   instance's full datasets on every sync.** Registering a host is choosing where your
   data goes.
5. **ssh key install.** HQ has its own dedicated data-plane keypair
   (`private/ssh/id_ed25519`, generated on first use). A native worker installs HQ's
   public key itself during pairing; a Docker worker shows you the one manual line to
   append to `authorized_keys`. The worker's host keys from the pairing string are
   written to HQ's `known_hosts` — no trust-on-first-use.
6. **Data-plane test** — `POST /api/machines/{key}/test`: one ssh echo, then an
   `rsync --dry-run` against the recorded remote root. Only success marks the machine
   `verified`.
7. **Probe** — `POST /api/machines/{key}/probe` asks the worker what it has (cards,
   venvs) and records the answer. The machine now appears in the Train control's
   model × machine × card matrix.

## GPUs: any number of cards, chosen per job

A machine has however many GPUs the probe measured — one, two, or eight — and
every card appears in the Train control's card picker for the machines that can
train. Card indexes are PCI order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), so an index
always names the same physical device.

When you don't pick a card, Groundwork picks one: a **free** card whose venv
can actually **drive it** (the card's compute capability must be in the torch
build's kernel list — measured by the probe, never assumed), preferring cards
**roomy** enough for the family's recorded peak VRAM. A card that is merely too
small is still eligible — the job runs, spilling into host RAM, measurably
slower — and the launch log says so. An explicit choice always wins, and is
refused with the specific reason when it can't be honoured: that card is busy,
that index doesn't exist on that machine, or that venv's kernels stop below the
card's architecture.

Two jobs on two cards of one machine is supported (per-card lock files keyed on
`CUDA_VISIBLE_DEVICES`), but concurrent GPU work on one box is opt-in per
launch — driver-level interference is real on some platforms, so the default
keeps one job per machine.

## What syncs, when

| flow | direction | when | what |
|---|---|---|---|
| **mirror** | HQ → worker | every 5 min, and always before a remote train | each project's dataset tree, `rsync --delete` |
| **Sync now** | HQ → worker | on demand (`POST /api/machines/{key}/sync`) | same as mirror, for the "I just labelled and want to train on it *now*" moment |
| **adopt** | worker → HQ | every 5 min (scan) | finished, *scored* runs — renamed to the next local run number, ledger row written, dataset snapshot stamped |
| **backup** | HQ → one designated machine | every 30 h, **off by default** | `raw/` + `testset/` images and labels only — the irreplaceable set |

**THE MIRROR IS NOT A BACKUP.** It rsyncs with `--delete`, so a worker's copy is a
*clone*: an image deleted on HQ is gone from every worker within one tick, faithfully.
That is the mirror doing its job — training must see exactly HQ's data — and it means the
mirror can never save you from a mistake. The backup job is the backup: enable it by
marking one machine as the backup target. It keeps twenty timestamped snapshots that
hardlink unchanged files to each other (twenty snapshots ≈ one copy plus the deltas), and
it prunes old ones only *after* the new copy's file counts verify against the source — an
unreachable target sends nothing and prunes nothing.

Details that keep this honest:

- The mirror excludes `runs/`, `snapshots/`, `data.yaml` and the split directories —
  those are machine-local (a worker splits for itself; HQ's `data.yaml` would point at
  HQ's paths). The excludes are *anchored* paths; an unanchored `images/` once matched
  the real data directories and silently mirrored nothing.
- A remote train **syncs first and refuses if the sync fails** — it will not train on a
  copy it could not confirm. The response says how many files moved and how many
  training/holdout images the worker now has.
- Adoption only pulls runs that already have an eval; on the worker, the autoscore job
  scores any finished-but-unscored run every 5 minutes, so `train → score → adopt`
  completes with nobody watching.
- **`/api/lab_status` is a cache, and it can serve the *previous* run's "done" for a
  couple of minutes after a new launch** — `POST /api/train` returns before the worker's
  status reflects the new run, and `age_s` on the response tells you how old the number
  is. A script watching a remote run to completion should key on the *fact*, not the
  proxy: the ledger gaining the run's row (adoption) is completion; a "done" status is
  not. (Measured 2026-08-08: a watcher broke instantly on the prior run's cached
  "done · adopted" while the new run was at epoch 15.)
- `.last_sync.<machine>` is stamped only on success, so the dashboard's sync-age warning
  surfaces a broken path within a tick.

## The security model

Credentials flow **one way**. HQ holds: a train-scoped API key for each worker (minted on
that worker) and its own ssh private key. The worker holds: HQ's *public* key in
`authorized_keys`, and nothing else — no URL, no key, no route back into HQ. Even
"Sync now" is HQ-initiated for this reason; the old shape (a worker-side resync that
ssh'd back into HQ) meant the worker held a control-plane credential, and that class of
design is gone. A compromised worker yields the datasets it was already sent — which is
exactly why registering a machine is admin-and-session-only, and why the pairing flow
makes you read the exfiltration warning before confirming.

## Networking

Any network where HQ can reach the worker's cockpit port (HTTP) and its sshd (data
plane) works: a LAN, a WireGuard mesh, or a VPN product such as Tailscale — Groundwork
does not care which, it only ever dials addresses you recorded. If a machine has several
interfaces and advertises the wrong one, set `GW_SELF_URL` on it; that value wins over
the interface scan.

When the machines are not in the same building, a mesh VPN (Tailscale, ZeroTier, plain
WireGuard) is the recommended route: every box gets a stable private address and an
encrypted path, and nothing is ever port-forwarded to the public internet. Do not solve
cross-site reachability with a router port-forward — this is a machine that starts GPU
jobs and holds your photos, and the security model above assumes it is never publicly
exposed.

## Troubleshooting

- **The data-plane test shows ssh's own words, verbatim.** "Permission denied
  (publickey)" (key not installed), "No route to host" (wrong address), an rsync "No such
  file or directory" (wrong remote root) all look different in the original and identical
  in a paraphrase — so nothing paraphrases them. Fix what the message names, then test
  again.
- **Probe answers 502 "older Groundwork"** — the worker's code predates
  `/api/machine/cards`. Update the worker, probe again.
- **Probe answers 502 with a 401/403** — the API key is missing or wrong. Mint one *on
  the worker* (`python -m groundwork.web.auth.keys add hq`) and paste it into the
  machine's entry on HQ.
- **A busy worker keeps its previously measured cards.** Reading card properties can
  wedge a driver mid-run, so neither side does it while training; the probe keeps the
  older inventory and flags it `cards_stale` rather than erasing cards that still exist.
- **Mirror failing with a VPN re-authentication demand.** Some VPNs periodically require
  an interactive ssh re-auth that a headless timer can never satisfy. The dashboard's
  mirror-health card extracts the re-auth link from the error text and shows it as
  something you can tap — one human tap, and the next tick succeeds.
- **Unknown is not idle.** Machine changes and deletions are refused while the machine is
  training *or unreachable* — a run on a machine you just deleted would have no URL to
  cancel through and no host to adopt from. For a machine that is genuinely dead,
  deletion takes `force=true`.
