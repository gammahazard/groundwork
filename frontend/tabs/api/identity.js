"use strict";
/* API reference — Identity & keys. Content only; tabs/api/docs.js renders it.
 *
 * WHY THIS GROUP EXISTS. `/api/me` was undocumented, and it is the endpoint
 * every key-based client has to call FIRST: a key is an identity, so "whose key
 * is this" is the question that decides which projects you can see and what
 * your uploads are recorded as — any external client's per-account model
 * is built on it. */

apiGroup({
  "key": "identity",
  "title": "Identity & keys",
  "blurb": "Who a key belongs to, and what that lets it do.",
  "endpoints": [
    {
      "method": "GET",
      "path": "/api/me",
      "auth": "key",
      "what": "Whose credential this is.",
      "detail": "THE FIRST CALL ANY CLIENT SHOULD MAKE. A key is not a password,\n          it is an identity: the gate resolves it to the account that minted\n          it, and every project listing, upload and run afterwards is scoped to\n          THAT account. So this is how a tool discovers who it is acting as —\n          and how it tells \"the key is wrong\" apart from \"the server is down\",\n          which are 401 and a connection error respectively.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/me\"",
      "returns": "{\"username\":\"mongo\",\"admin\":true,\"must_change\":false}",
      "notes": [
        "Works with a session cookie too, which is what the cockpit itself uses.",
        "401 means the key is revoked, expired or never existed — mint a new one rather than retrying."
      ]
    },
    {
      "method": "GET",
      "path": "/api/keys",
      "auth": "session",
      "what": "Your API keys — names, scopes, when each was last used.",
      "detail": "Never the key VALUES. A key is shown once, at the moment it is\n          minted, and stored as a SHA-256 digest; there is nothing to show\n          afterwards and nothing an attacker could read back.",
      "notes": [
        "SESSION ONLY, and this is the asymmetry that makes keys safe to hand out: a key may never manage accounts or keys. A leaked key is therefore revoked, not a takeover — it cannot mint itself a successor. See auth/routes.refuse_key_auth."
      ]
    },
    {
      "method": "POST",
      "path": "/api/keys",
      "auth": "session",
      "what": "Mint a key. The value is returned ONCE and never again.",
      "detail": "Scopes decide what it may do:\n\n            read   every GET, nothing else. A dashboard or a monitor — the\n                   safest thing to leave on a machine you do not control.\n            train  read, plus starting and cancelling runs, exporting,\n                   counting and asking LocateAnything. The default.\n            full   everything a session could do EXCEPT managing accounts and\n                   keys, and except writing a systemd unit.\n\n          The write list is a WHITELIST, not a blacklist of dangerous paths: the\n          next endpoint anyone adds is refused to a narrow key until it is\n          deliberately allowed.",
      "example": "curl -X POST -H \"Content-Type: application/json\" \\\n  -d '{\"name\":\"ci-bot\",\"scope\":\"train\"}' \"$HOST/api/keys\"",
      "notes": [
        "Uploading images and creating projects are outside the `train` whitelist, so a tool that pushes data needs `full`.",
        "Installing or starting a bot's systemd service is session-only even for `full` — writing a unit file is arbitrary code as the user who owns .env, and a string in a config file should not reach that."
      ]
    },
    {
      "method": "DELETE",
      "path": "/api/keys/{id}",
      "auth": "session",
      "mutates": true,
      "what": "Revoke a key immediately.",
      "detail": "Takes effect on the next request that uses it. Because a key can\n          never mint another key, revoking is genuinely the end of it."
    }
  ]
});
