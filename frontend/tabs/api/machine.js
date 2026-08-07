"use strict";
/* API reference — Machines & serving. Content only; tabs/api/docs.js renders it.
 * Registered into the shared list by apiGroup() so a new group is a new file
 * plus one <script> tag, and nothing else has to know. */

apiGroup({
  "key": "machine",
  "title": "Machines & serving",
  "blurb": "What this installation is, what it has, and which model answers.",
  "endpoints": [
    {
      "method": "GET",
      "path": "/api/version",
      "auth": "key",
      "what": "What code this machine is running.",
      "detail": "Commit, branch and machine name. Ask both machines and diff to\n          settle \"is the worker up to date\" — the question that gates remote card\n          choice and has been answered by ssh until now.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/version\"",
      "returns": "{\"commit\":\"83c234b…\",\"short\":\"83c234b12\",\"branch\":\"main\",\"machine\":\"Groundwork Trainer\"}"
    },
    {
      "method": "GET",
      "path": "/api/machine/self",
      "auth": "key",
      "what": "This machine's identity and reachable URL.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/machine/self\""
    },
    {
      "method": "GET",
      "path": "/api/machines",
      "auth": "session",
      "what": "Registered machines, their cards and when each was measured.",
      "notes": [
        "Admin only. Registering a machine records an ssh host, and a remote run rsyncs the dataset there — so this is deliberately not something a key can do."
      ]
    },
    {
      "method": "POST",
      "path": "/api/count",
      "auth": "key",
      "mutates": false,
      "what": "Count objects in one image with the serving model. Nothing is saved.",
      "example": "curl -X POST \"$HOST/api/count?project=the first project\" \\\n  -H \"Authorization: Bearer $GW_KEY\" -F \"image=@image.jpg\""
    },
    {
      "method": "GET",
      "path": "/api/state",
      "auth": "key",
      "what": "The project's counts, and which model is serving.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/state?project=the first project\""
    },
    {
      "method": "GET",
      "path": "/api/projects",
      "auth": "key",
      "what": "Every project on this machine.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/projects\""
    }
  ]
});
