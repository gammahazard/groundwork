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
      "path": "/api/machine/pairing-code",
      "auth": "session",
      "what": "On a WORKER: mint the one-time pairing code to paste at the HQ.",
      "detail": "Admin, worker role only. The code carries a fresh train-scope key and\n          this box's sshd host keys; it is valid once, for 15 minutes, and only for\n          /api/machine/pair — holding some other valid key never opens that door."
    },
    {
      "method": "POST",
      "path": "/api/machines/pair",
      "auth": "session",
      "what": "On the HQ: paste a worker's code — register, trust host keys, install the ssh key, test, probe. One call.",
      "example": "curl -X POST \"$HOST/api/machines/pair\" -H \"Content-Type: application/json\" \\\n  --cookie \"$SESSION\" -d '{\"code\":\"GW1.…\"}'"
    },
    {
      "method": "POST",
      "path": "/api/machines/{key}/sync",
      "auth": "session",
      "what": "Push a project's dataset to a worker right now (the mirror job does this every 5 min)."
    },
    {
      "method": "POST",
      "path": "/api/machines/{key}/test",
      "auth": "session",
      "what": "Prove the data plane: one ssh echo + one rsync dry-run, errors verbatim."
    },
    {
      "method": "POST",
      "path": "/api/machines/{key}/probe",
      "auth": "session",
      "what": "Ask a machine what cards and venvs it has; the answer drives the Train matrix."
    },
    {
      "method": "GET",
      "path": "/api/machine/status",
      "auth": "key",
      "what": "Is anything training on this machine — project-free, safe to poll."
    },
    {
      "method": "POST",
      "path": "/api/count",
      "auth": "key",
      "mutates": false,
      "what": "Count objects in one image with the serving model. Nothing is saved.",
      "example": "curl -X POST \"$HOST/api/count?project=widgets\" \\\n  -H \"Authorization: Bearer $GW_KEY\" -F \"image=@image.jpg\""
    },
    {
      "method": "GET",
      "path": "/api/state",
      "auth": "key",
      "what": "The project's counts, and which model is serving.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/state?project=widgets\""
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
