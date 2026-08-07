"""Who may use this cockpit, and what they may reach once they are in.

THE INVARIANT THIS PACKAGE OWNS: **nothing under `outputs/` is served to an
unauthenticated caller.** Not a route, and not a file.

That second half is the whole reason the gate is ASGI middleware rather than a
FastAPI dependency. `apps/webui.py` mounts the entire `outputs/` tree —
eighteen gigabytes of image photographs, eval previews, challenger checkpoints and
CoreML exports — as StaticFiles. A dependency guards routes and would have left
every one of those files open, which is not a smaller version of being secured;
it is the same exposure with a login page in front of it.

    passwords.py   turning a password into a digest, and back into a yes/no
    users.py       the account store, its rules, and the terminal CLI
    sessions.py    a logged-in browser, and how it stops being one
    middleware.py  the gate. Runs before every route AND every mount
    routes.py      /api/login, /api/me, /api/users

WHAT IS DELIBERATELY NOT HERE. Authorisation for a *project* lives in
`web/api/deps.py`, next to the dependency that already resolves which project a
request is about. Splitting "who are you" from "may you touch this project" is
not tidiness: they answer to different things — one to a session, the other to a
manifest — and a project check inside the login package would be a second place
that decides what `Project.owner` means.

NO SELF-SERVICE SIGNUP, ever. An account comes from a terminal or from an admin,
and from nowhere else. See users.py for the reasoning and for the one decision
the owner reversed.

THE THREAT MODEL HAS NOT CHANGED. This machine is network-only and stays that
way — CLAUDE.md's rule about never funnelling or port-forwarding it is not
softened by there being a password now. Auth is defence in depth against a
device on the LAN, not a licence to expose the box. There is no TLS on this
path, so the session cookie is not marked Secure; over the network that is the
honest setting, and it is another reason the box stays off the public internet.
"""
