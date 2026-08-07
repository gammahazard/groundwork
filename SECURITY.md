# Security

## The honest posture

Groundwork is designed to run on a **LAN or a private VPN**, reached by people you know.
It is not hardened for the public internet and should not be exposed to it: the cockpit
can start GPU jobs, write systemd units for bots, and serve entire datasets. A login is
one layer, not a reason to port-forward a machine like that.

**Plain HTTP by default.** On the default deployment the session cookie is *not* marked
`Secure` (it would never be sent over http://, locking everyone out). If you terminate TLS
in a reverse proxy, set `GW_TRUST_PROXY=1`: uvicorn then honours the proxy's forwarded
headers (so the login throttle sees real client addresses instead of the proxy's) and the
session cookie is set `Secure`. Do not set it without a trusting proxy in front — the
forwarded headers are client-controlled otherwise.

## Authentication

Two doors, one answer. A **cookie** is a person; a **bearer key** (`gw_…`) is a script or
another machine. `groundwork/web/auth/middleware.py` resolves both to a username once, at
the gate, so ownership checks, admin checks and busy gates apply identically downstream.

- **Sessions** are server-side (revocable), the cookie is `HttpOnly` and `SameSite=lax`.
- **Passwords** are stored as scrypt (memory-hard KDF, per-hash random salt), never
  reversible, never logged.
- **Failed logins are throttled**, keyed on username *and* source address, checked before
  the password is verified (so a blocked caller never costs a scrypt). It delays rather
  than locks out: a hard lockout is a denial of service a stranger can trigger by failing
  your login five times.
- **API keys are scoped**: `read` (GET only), `train` (read plus a whitelist of
  job-starting writes — the default), `full` (everything a session can do except manage
  accounts and keys). The write list is a whitelist, so the next endpoint anyone adds is
  refused to a narrow key until deliberately allowed. Keys are stored hashed (SHA-256);
  the raw value exists once, in the minting response.
- **A key may never manage accounts or keys.** A key lives in a config file and is the
  credential most likely to leak; a session is a person who typed a password minutes ago.
  So a leaked key is revoked, not a takeover — and it cannot mint itself a successor.
- **There is an audit trail**: append-only JSONL of sign-ins and failures, account and key
  changes, and scope denials. It never stores passwords, key values or session tokens.

## The gate covers mounts, not just routes

Authentication is ASGI middleware wrapping the whole app — including the `StaticFiles`
mounts. That is deliberate: `/outputs` serves gigabytes of dataset photographs, eval
previews and model checkpoints, and a FastAPI dependency would have guarded the routes
while leaving every one of those files open. The allow-list of unauthenticated paths is
tiny (the SPA shell, `/api/login`, `/api/me`, `/static/*`, `/healthz`), and paths are
normalised before matching so `/static/../outputs/...` is judged where it lands.
Credential stores live under `private/`, *outside* the served tree entirely — nothing
there is reachable over HTTP as a property, not a rule.

Reads are ownership-checked per project: a signed-in account sees its own projects and
unowned ones; admins read everything (so deleting an account cannot strand its projects).

## Machine pairing: credentials flow one way

- **HQ holds the worker's credentials** — a train-scoped API key minted *on the worker*,
  and HQ's own dedicated ssh identity (generated under `private/ssh/`, installed on the
  worker at pairing). **The worker holds nothing of HQ's.** A compromised worker gets you
  the datasets it was already sent, not the fleet.
- The worker's ssh host keys are delivered in the pairing exchange and written to HQ's
  `known_hosts` — no trust-on-first-use.
- **Machine registration is admin-only and session-only** (no key may do it), because a
  registered machine *receives full datasets* on every sync: registering a host is
  choosing where your data goes, and the UI warns exactly that.
- **Workers refuse dataset mutations server-side.** A worker's copy is a one-way mirror
  from HQ; every dataset-editing endpoint refuses on a worker with the reason, so no UI
  gap or direct call can create silent divergence.

`GW_AUTH=0` disables the gate entirely. It exists as a development opt-out and is stated
loudly at startup; do not run it on anything reachable by others.

## Reporting a vulnerability

Open a private security advisory on the GitHub repository (preferred), or an issue if the
problem is not sensitive. Include reproduction steps and what an attacker gains. There is
no bounty programme; there is a maintainer who takes this seriously.
