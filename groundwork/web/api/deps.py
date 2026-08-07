"""How a REQUEST says which project it is about.

The dataset layer takes a `ProjectPaths` everywhere; this is the HTTP half:
one dependency, injected wherever a route touches project-scoped data.

    @router.get("/api/images/{collection}")
    def images(collection: str, pp: ProjectPaths = Depends(project_paths)):

WHY A QUERY PARAMETER AND NOT A PATH PREFIX. `/api/p/{slug}/images/...` would
be tidier in the abstract and a breaking change to every route at once, in the
same commit as the code that reads it — the sort of change that cannot be
bisected when something goes quiet. `?project=` is a single convention the
frontend's fetch wrapper appends on every call, so no handler ever plumbs it
by hand while the
backend becomes capable. Revisit the path form later if it earns itself.

WHY IT DEFAULTS RATHER THAN REQUIRING. Same reason `--project` defaults on the
stage CLIs: a missing project must resolve to the one that has always been
there, not 422. There is exactly one project today.

The failure mode this guards against is the one this repo keeps paying for —
silently operating on the wrong data. An unknown slug is a 404 with the slug in
it, never a silent fall-back to the default: a typo'd project that quietly
returns the object counter's 160 images is how you would edit the wrong dataset.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Query, Request

from ... import project
from ...dataset import paths
from ...dataset.paths import ProjectPaths
from ..auth import audit as audit_mod
from ..auth import users as users_mod


def project_slug(slug: str = Query(..., alias="project",
                                   description="project slug")) -> str:
    """The requested project's slug, validated. REQUIRED — there is no default
    project, so a request that names none is a 422, not a guess. 404 on an
    unknown one."""
    if slug not in project.slugs():
        raise HTTPException(404, f"no such project {slug!r}")
    return slug


def current_user(request: Request) -> str | None:
    """Who is asking, or None when the gate is off.

    Resolved by web/auth/middleware.py for every request; read from the scope
    rather than re-derived, so there is one implementation of "is this session
    valid". None means the gate is disabled on this machine (the worker), not that
    the caller is anonymous — the middleware refuses anonymous callers itself.
    """
    return request.scope.get("auth_user")


def readable_by(p, user: str | None) -> bool:
    """MAY THIS ACCOUNT OPEN THIS PROJECT? The one rule, in one place.

    Three ways to yes, and the order is the reasoning:

      no gate      `user` is None because auth is disabled on this box (the worker,
                   which serves only machine-to-machine traffic). Nothing to
                   check against.
      unowned      `owner` is None. Every project that predates accounts is in
                   this state, the first project included, so an existing install keeps
                   working the moment auth is switched on rather than locking
                   its operator out of their own data. A project claims an owner
                   when it is created, or by hand.
      yours        the obvious one.

    ADMINS CAN READ EVERYTHING, and that is a deliberate trade rather than an
    oversight. Deleting an account would otherwise strand its projects with no
    route back through the UI — the data would still be on disk and permanently
    invisible. Someone who can create and delete accounts can already appoint
    themselves owner of anything by editing a manifest, so the alternative buys
    no safety and costs a recovery path. It is surfaced, not hidden: the API
    reports WHY a project is readable so the UI can say "visible to you as an
    admin" rather than implying it is yours.
    """
    if user is None:
        return True
    return p.owner in (None, user) or users_mod.is_admin(user)


def why_readable(p, user: str | None) -> str:
    """"owner" | "unowned" | "admin" | "open" — for the UI to be honest with."""
    if user is None:
        return "open"
    if p.owner is None:
        return "unowned"
    return "owner" if p.owner == user else "admin"


def current_project(slug: str = Depends(project_slug),
                    request: Request = None,
                    user: str | None = Depends(current_user)):
    """The requested `Project`, IF this account may open it.

    403 rather than 404 for someone else's project. 404 would hide that it
    exists, which is the stricter choice and the wrong one here: every account
    on this machine belongs to the same household, and a project that answers
    "no such thing" about data the operator can see on disk costs an hour of
    confusion to buy a secret from nobody.

    401 is NOT raised here — the middleware has already refused anonymous
    callers, so reaching this function means there is a session (or the gate is
    off). Keeping 401 out of this layer is what stops a permission error from
    bouncing the browser to a login form that would change nothing.
    """
    try:
        p = project.load(slug)
    except KeyError:
        raise HTTPException(404, f"no such project {slug!r}")
    if not readable_by(p, user):
        # REACHING FOR SOMEONE ELSE'S PROJECT IS THE LINE WORTH RECORDING. Once
        # there is more than one account it is the most useful event in the
        # trail — usually a stale bookmark, occasionally not — and it is the one
        # thing a sign-in log cannot show you.
        audit_mod.record(audit_mod.DENIED, actor=user, target=slug, ok=False,
                         via=(request.scope.get("auth_via") if request else None),
                         ip=(getattr(getattr(request, "client", None), "host", None)
                             if request else None),
                         detail=f"owned by {p.owner!r}")
        raise HTTPException(
            403, f"project {slug!r} belongs to {p.owner!r} — ask them to "
                 f"transfer it, or open one of your own")
    return p


def project_paths(p=Depends(current_project)) -> ProjectPaths:
    """The requested project's dataset layout — the common case."""
    return paths.for_project(p)


def require_extension(name: str):
    """Dependency factory: 404 unless this project enables `name`.

    Extensions are project-scoped tools. Their endpoints exist
    on the app whatever project you ask about, so without this a screws project
    would happily accept a DIN tag or an object-cutout harvest. Gating in the router
    means the UI hiding a tab is a convenience, not the enforcement — the same
    reasoning that makes lab_guard server-side rather than a hidden button.
    """
    def dep(p=Depends(current_project)):
        if not p.has(name):
            raise HTTPException(
                404, f"project {p.slug!r} does not enable the {name!r} extension")
        return p
    return dep
