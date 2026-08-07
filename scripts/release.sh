#!/usr/bin/env bash
# Groundwork release — one command from a stocked changelog to a published
# GitHub Release, with the steps you cannot afford to forget baked in.
#
#   scripts/release.sh <X.Y.Z>            # cut the release
#   scripts/release.sh <X.Y.Z> --dry-run  # show exactly what it would do, change nothing
#
# What it does, in order, refusing loudly at the first thing that is wrong:
#   1. validates: on main, clean tree, valid SemVer, higher than the current
#      version, tag does not already exist, and [Unreleased] is not empty
#   2. stamps CHANGELOG.md — [Unreleased] becomes [X.Y.Z] - <today>, a fresh
#      empty [Unreleased] is left on top
#   3. bumps pyproject.toml
#   4. commits "release: X.Y.Z" and pushes main
#   5. WAITS for CI to go green on that commit (a tag on a red commit is the
#      mistake this exists to prevent) — aborts if any check fails
#   6. tags vX.Y.Z and pushes the tag (which fires the ghcr publish workflow —
#      a no-op while the repo is private, a real publish once it is public)
#   7. creates the GitHub Release from the changelog section, marked latest
#
# SemVer, decided by what is in [Unreleased]: a new feature -> minor, fixes
# only -> patch, a breaking change to the API / on-disk layout / unit or env
# contracts -> major. The version is explicit on purpose; the script will not
# guess it for you.
set -euo pipefail

die() { printf 'release: %s\n' "$*" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

# --- arguments ---------------------------------------------------------------
DRY=0; VERSION=""
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; exit 0 ;;
        -*)        die "unknown flag: $a" ;;
        *)         [ -z "$VERSION" ] && VERSION="$a" || die "unexpected argument: $a" ;;
    esac
done
[ -n "$VERSION" ] || die "usage: scripts/release.sh <X.Y.Z> [--dry-run]"
printf '%s' "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' \
    || die "version must be X.Y.Z (got '$VERSION')"

# --- preflight ---------------------------------------------------------------
command -v gh >/dev/null 2>&1 || die "the gh CLI is required (creates the Release) — https://cli.github.com"
gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login"
branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "main" ] || die "not on main (on '$branch') — releases are cut from main"
# A real cut demands a clean tree (nothing stray in the release commit); a
# dry run is a preview and reads whatever is on disk, committed or not.
[ "$DRY" = 1 ] || [ -z "$(git status --porcelain)" ] \
    || die "the working tree is dirty — commit or stash first"

cur="$(grep -oE '^version = "[0-9.]+"' pyproject.toml | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')" \
    || die "could not read the current version from pyproject.toml"
[ "$VERSION" != "$cur" ] || die "$VERSION is already the current version"
higher="$(printf '%s\n%s\n' "$cur" "$VERSION" | sort -V | tail -1)"
[ "$higher" = "$VERSION" ] || die "$VERSION is lower than the current $cur"
git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null 2>&1 && die "tag v$VERSION already exists locally"
git fetch -q origin --tags
git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null 2>&1 && die "tag v$VERSION already exists on the remote"

# --- changelog surgery + notes (in temp files; real files touched only later) -
NOTES="$(mktemp)"; NEWCL="$(mktemp)"
trap 'rm -f "$NOTES" "$NEWCL"' EXIT
python3 - "$VERSION" "$NEWCL" "$NOTES" <<'PY' || exit 1
import datetime, re, sys
ver, out_cl, out_notes = sys.argv[1:4]
src = open("CHANGELOG.md", encoding="utf-8").read()
m = re.search(r'(?m)^## \[Unreleased\]\s*\n(.*?)(?=^## \[|\Z)', src, re.S)
if not m:
    sys.exit("release: CHANGELOG.md has no '## [Unreleased]' section")
body = m.group(1).strip()
if not body:
    sys.exit("release: [Unreleased] is empty — add entries there before releasing")
today = datetime.date.today().isoformat()
stamped = (src[:m.start()]
           + f"## [Unreleased]\n\n## [{ver}] - {today}\n\n{body}\n\n"
           + src[m.end():])
open(out_cl, "w", encoding="utf-8").write(stamped)
open(out_notes, "w", encoding="utf-8").write(body + "\n")
print(f"  changelog: [Unreleased] -> [{ver}] - {today}  ({body.count(chr(10)) + 1} lines of notes)")
PY

# --- dry run stops here, having changed nothing -------------------------------
if [ "$DRY" = 1 ]; then
    echo
    echo "=== DRY RUN — nothing was changed ==="
    echo "--- CHANGELOG.md (what would change) ---"
    diff -u CHANGELOG.md "$NEWCL" || true
    echo "--- pyproject.toml version: $cur -> $VERSION ---"
    echo "--- release notes that would be published ---"
    sed 's/^/  | /' "$NOTES"
    echo "--- then: commit 'release: $VERSION' -> push -> wait for CI -> tag v$VERSION -> gh release create ---"
    exit 0
fi

# --- commit + push ------------------------------------------------------------
cp "$NEWCL" CHANGELOG.md
# Only the exact version line, anchored, so nothing else that looks like a
# version is touched.
python3 - "$cur" "$VERSION" <<'PY'
import re, sys
cur, ver = sys.argv[1], sys.argv[2]
p = "pyproject.toml"; s = open(p).read()
s2 = re.sub(rf'(?m)^version = "{re.escape(cur)}"$', f'version = "{ver}"', s, count=1)
assert s2 != s, "version line not found to bump"
open(p, "w").write(s2)
PY
git add CHANGELOG.md pyproject.toml
git commit -q -m "release: $VERSION"
git push -q origin main
sha="$(git rev-parse HEAD)"
echo "  pushed $sha"

# --- wait for CI green on that commit ----------------------------------------
echo "  waiting for CI (up to ~20 min)…"
ok=0
for _ in $(seq 1 60); do
    sleep 20
    json="$(gh run list --commit "$sha" --json name,status,conclusion 2>/dev/null || echo '[]')"
    read -r total pending failed <<EOF
$(printf '%s' "$json" | python3 -c 'import sys,json
d=json.load(sys.stdin)
pend=[r for r in d if r["status"]!="completed"]
fail=[r["name"] for r in d if r["status"]=="completed" and r["conclusion"]!="success"]
print(len(d), len(pend), ",".join(fail) or "-")')
EOF
    [ "${total:-0}" -ge 1 ] || { echo "    …no runs registered yet"; continue; }
    if [ "$failed" != "-" ]; then
        die "CI FAILED ($failed). The release commit is pushed but NOT tagged — fix, push, and re-run scripts/release.sh $VERSION (it will see the version already bumped; revert that commit first if you prefer a clean history)."
    fi
    [ "${pending:-1}" -eq 0 ] && { ok=1; echo "    CI green ($total run(s))"; break; }
    echo "    …$pending run(s) still going"
done
[ "$ok" = 1 ] || die "timed out waiting for CI — check 'gh run list', then tag manually once green"

# --- tag + Release -----------------------------------------------------------
git tag -a "v$VERSION" -m "Groundwork v$VERSION"
git push -q origin "v$VERSION"
gh release create "v$VERSION" --title "Groundwork v$VERSION" --notes-file "$NOTES" --latest --verify-tag
url="$(gh release view "v$VERSION" --json url --jq .url 2>/dev/null || echo '(see the Releases tab)')"
echo
echo "released v$VERSION -> $url"
echo "the ghcr publish workflow fired on the tag (a no-op while the repo is private)."
