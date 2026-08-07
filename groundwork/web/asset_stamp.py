"""Rewrite `?v=` on the SPA's asset tags to a hash of the file's CONTENT.

WHY. index.html carries hand-written cache-busters — `core.js?v=3`, `nav.js?v=1`
— and they only work if a human remembers to bump the number in the same commit
that edits the file. Forget once and the browser keeps serving the previous
bytes with no signal that it is doing so: the page loads, the version number
matches, and you are debugging code the browser is not running.

That is not hypothetical. "The burger menu doesn't open" was reported against a
build whose burger was verified working in four engines — Chromium desktop,
Chromium iPhone, WebKit iPhone, and jsdom — because the reporting browser still
held the previous JS. Hours went into the page instead of the cache.

`Cache-Control: no-cache` on /static (apps/webui.py) fixes the general case by
forcing revalidation. This closes the rest: a URL that CHANGES when the bytes
change cannot serve a stale body even from a cache that ignores revalidation —
back/forward cache, an offline service worker, a corporate proxy, or simply an
entry stored before the no-cache header existed. Belt and braces, because the
failure is silent and the cost of being wrong is an afternoon.

WHY NOT AT BUILD TIME. There is no build step; index.html is served from disk.
Hashing on demand keeps it that way, and the mtime+size cache below means the
common case is a stat() per asset, not a read.

The `?v=` is left alone when the file cannot be found, rather than dropped —
an asset served from somewhere else is not this module's business to break.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Matches the version query on a local asset only: src="/static/…?v=7".
# Anchored on /static/ so a CDN or data: URL is never touched.
_TAG = re.compile(r'((?:src|href)="/static/([^"?]+))\?v=[^"]*"')

# path -> (mtime_ns, size, digest). A file whose mtime and size are unchanged
# cannot have changed in a way we care about, so this makes the hot path a stat.
_CACHE: dict[Path, tuple[int, int, str]] = {}


def _digest(path: Path) -> str | None:
    """Short content hash, or None if the file is missing."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(path)
    if hit and hit[:2] == key:
        return hit[2]
    try:
        h = hashlib.blake2b(path.read_bytes(), digest_size=6).hexdigest()
    except OSError:
        return None
    _CACHE[path] = (*key, h)
    return h


def stamp(html: str, static_root: Path) -> str:
    """Replace every `?v=…` on a /static asset with that file's content hash."""

    def sub(m: re.Match[str]) -> str:
        prefix, rel = m.group(1), m.group(2)
        d = _digest(static_root / rel)
        # Unknown file: leave the author's version string exactly as written.
        return f'{prefix}?v={d}"' if d else m.group(0)

    return _TAG.sub(sub, html)
