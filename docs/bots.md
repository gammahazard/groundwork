# Telegram bots

A bot is the field end of the training loop: someone photographs a scene, the bot answers
with a count, and the buttons under the answer route the image back into the project's
data. No cockpit, no login — a phone camera and a chat.

## What the buttons do

- **✓ Right** — the count was correct; the image and its machine labels are staged as a
  training candidate.
- **🎯 Test set** — earmark this image for the **frozen holdout**: it will measure
  accuracy and never be trained on. (🎯 combines with ✗ for "wrong, *and* destined for
  the holdout".)
- **✗ Wrong** — flag it, then *reply to the photo* with the correct count. The image
  enters the fix queue with your number attached, which is exactly the data a retrain
  needs most.
- **🗑 Discard** — a photo of the floor is a photo of the floor.

Everything staged lands in the project's pending inbox and fix queue; a human pass in
the editor is what promotes an image into training or holdout ([training.md](training.md)
§3–4). Bots also answer `/status` (health), `/restart` (reload), and `/whoami` (your
numeric Telegram id — you will need it below).

## Setup

### 1. BotFather

In Telegram, talk to `@BotFather` → `/newbot` → pick a name and username. You get a
token shaped like `1234567:AA…`. That token *is* the bot — treat it like a password.

### 2. Register it in the cockpit, per project

Data collection tab → your project → add a bot. Bots are **per project**: each one
collects for exactly one project's dataset, and a project holds at most one bot per
role. The role table is closed (a role maps to a literal module path in
`groundwork/web/bot_roles.py`); the shipped role is the **counting bot**.

What registration does, and why it looks the way it does:

- The service name and token variable are **derived, not chosen**:
  `gw-<project>-<role>.service` and `GW_<PROJECT>_<ROLE>_TOKEN`. Free-form names in
  things systemd executes are an injection surface; derived names from an allowlist are
  not.
- The token is **verified with Telegram before it is stored** (a typo'd token fails at
  registration, not silently at 3 a.m.), written atomically into `.env` at mode 0600 —
  and **never readable back out**. The API and UI show only the *variable name*; there
  is no endpoint that returns a token, so no endpoint can grow one by accident.
- Install writes the unit (or starts the supervised process — see below), enables and
  starts it.

### 3. Allow people

A bot answers **only** the Telegram ids listed for it (`GW_ALLOWED_IDS` in its unit).
A bot with an empty list is **unclaimed, and unclaimed answers NOBODY** — deliberately.
There is no machine-wide fallback owner: a freshly installed bot must not belong to
whoever happens to own the box, because the person setting it up cannot even learn their
own id until a bot answers `/whoami`. So the bootstrap is: install → message your bot →
it refuses but `/whoami` tells you your id → add that id in the cockpit → it answers.

## Process models

`GW_PROCESS_MODEL` (auto-detected when unset):

- **systemd** (native installs): each bot is a user unit written by the cockpit;
  systemd restarts it (`Restart=always`).
- **supervisor** (containers, or anywhere without a user systemd): the app process
  itself parents each enabled bot — Popen with restart-and-backoff (5 s → 5 min),
  mirroring the unit semantics closely enough that the status words in the bots panel
  mean the same thing.

The security boundary does not move between the two: what may run is still the closed
role table, and the environment block (`GW_PROJECT`, `GW_TOKEN_ENV`, `GW_ALLOWED_IDS`,
`GW_BOT_LOG`) is built by the same validated builder in both. A bot process refuses to
start rather than guess: no project, no token variable, or a value that is not shaped
like a token each produce a clear refusal — polling Telegram with a guessed or shared
token would silently split another bot's message stream, which is the failure the
refusals exist to prevent.
