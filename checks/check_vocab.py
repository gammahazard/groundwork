#!/usr/bin/env python3
"""The repo is origin-silent: no trace of the project it was extracted from.

Zero-tolerance grep gate, including docs and comments. Two failure classes:

  IDENTITY   the private project's machines, people and addresses
             (pillcounter, traytally, wheatley, tailnet IPs, personal paths)
  DOMAIN     the original domain's vocabulary (pill, tray, pour, pharmacy) —
             banned everywhere including data-artifact keys and API paths,
             by the owner's call: the ledgers and stats must not say "tray".

Word-boundary matching where a word is also a fragment of legitimate words
("stray", "spillover" would false-positive a naive substring scan).

Run: python checks/check_vocab.py            (exit 0 = clean)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (pattern, why). Case-insensitive. Word-ish boundaries where needed.
BANNED: list[tuple[re.Pattern, str]] = [
    (re.compile(r"pillcounter", re.I), "old package name"),
    (re.compile(r"traytally", re.I), "old project name"),
    (re.compile(r"wheatley", re.I), "owner's machine name"),
    (re.compile(r"dual3090", re.I), "owner's machine hostname"),
    (re.compile(r"\bmongo@"), "owner's remote user"),
    (re.compile(r"navraj", re.I), "owner's username"),
    (re.compile(r"100\.79\.157\.\d+"), "owner's tailnet IP"),
    (re.compile(r"100\.120\.32\.\d+"), "owner's tailnet IP"),
    (re.compile(r"100\.84\.201\.\d+"), "owner's tailnet IP"),
    (re.compile(r"\bPILL_[A-Z_]+"), "old env-var prefix"),
    (re.compile(r"pharmac", re.I), "original domain"),
    (re.compile(r"\bpills?\b", re.I), "original domain word"),
    (re.compile(r"\bpour(s|ed|ing)?\b", re.I), "original domain word"),
    # tray/trays as words — NOT substrings ("stray", "betrayal" are fine)
    (re.compile(r"\btrays?\b", re.I), "original domain word"),
    (re.compile(r"Closed for testing", re.I), "private-instance login copy"),
    (re.compile(r"pillweb|pillbot|pillmirror|pilladopt|pillclean|pillstats"
                r"|pillkeepalive|pillbackup|pillautoscore", re.I),
     "old unit names"),
    (re.compile(r"drugid|drug.id", re.I), "stripped extension"),
    (re.compile(r"telegram_allowed_user_id", re.I), "legacy machine-wide id"),
]

# Tailscale may be MENTIONED in networking docs as one VPN option, nowhere
# else — except two generic code literals: the interface name a VPN presents
# (tailscale0, one candidate in the self-IP scan) and the re-auth URL shape the
# mirror-health card turns into a tappable link. Those name a public product's
# mechanics, not this project's origin.
TAILSCALE = re.compile(r"tailscale|tailnet", re.I)
TAILSCALE_OK_DIRS = ("docs/",)
TAILSCALE_OK_LINE = re.compile(r"tailscale0|login\\?\.tailscale\\?\.com")

EXTS = {".py", ".js", ".mjs", ".cjs", ".html", ".css", ".md", ".json",
        ".txt", ".yml", ".yaml", ".svg", ".sh", ".tmpl", ".example", ".toml"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "outputs", "private",
             ".venv", "data"}
SKIP_FILES = {"check_vocab.py"}          # this file names what it bans


def files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SKIP_FILES:
            continue
        yield p


def main() -> int:
    bad = []
    for p in files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            for rx, why in BANNED:
                if rx.search(line):
                    bad.append(f"{rel}:{i}: [{why}] {line.strip()[:90]}")
            if TAILSCALE.search(line) and not rel.startswith(TAILSCALE_OK_DIRS) \
                    and not TAILSCALE_OK_LINE.search(line):
                bad.append(f"{rel}:{i}: [tailscale outside docs/] "
                           f"{line.strip()[:90]}")
    # ---- BINARY MEDIA ARE ALLOWLISTED BY NAME ----------------------------
    # Text scanning cannot see pixels. This gate exists because the origin
    # project's guide screenshots survived every text sweep: one showed the
    # origin domain itself, another had the old product name in its UI copy.
    # A new screenshot or GIF must be added HERE, deliberately, after a human
    # has actually looked at what it shows.
    MEDIA_OK = {
        "frontend/favicon.svg",
        "frontend/guide/verdict.jpg",      # ultralytics loss curves — neutral
        # Reviewed 2026-08-07: shot on a synthetic washers project generated
        # by this repo's own asset session — no third-party or domain imagery.
        "frontend/guide/label-editor.png",
        "frontend/guide/try-it.png",
        "frontend/guide/train-live.png",
    }
    MEDIA_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
                 ".mp4", ".webm", ".avif", ".bmp"}
    for p2 in sorted(ROOT.rglob("*")):
        if not p2.is_file() or p2.suffix.lower() not in MEDIA_EXT:
            continue
        rel2 = p2.relative_to(ROOT).as_posix()
        if rel2.startswith((".git/", "outputs/", "private/", "data/")):
            continue
        if rel2.startswith("frontend/fonts/"):
            continue
        if rel2 not in MEDIA_OK:
            bad.append(f"{rel2}: [unreviewed media file] a human must look at "
                       f"it, then allowlist it in check_vocab.MEDIA_OK")

    if bad:
        print(f"VOCAB GATE: {len(bad)} hit(s) — the repo must be origin-silent")
        for b in bad[:2000]:
            print("  " + b)
        if len(bad) > 2000:
            print(f"  … and {len(bad) - 2000} more")
        return 1
    print("vocab gate: clean — no trace of the origin project")
    return 0


if __name__ == "__main__":
    sys.exit(main())
