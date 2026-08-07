"use strict";
/* API reference — Data-collection bots. Content only; tabs/api/docs.js renders
 * it. One file per group (see _registry.js). */

apiGroup({
  "key": "bots",
  "title": "Data-collection bots",
  "blurb": "Register, token, permit, run. A bot works from day zero — before "
         + "the project has a model it collects into the fix queue; after, it "
         + "counts with the serving engine.",
  "endpoints": [
    {
      "method": "GET",
      "path": "/api/bots",
      "auth": "key",
      "what": "This project's bots: identity, allowed ids, service state, last activity.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/bots?project=widgets\""
    },
    {
      "method": "POST",
      "path": "/api/bots",
      "auth": "session",
      "what": "Register a bot for this project — metadata only, never a token.",
      "detail": "The unit name and token variable are DERIVED from the project and\n          role against a closed table (web/bot_roles.py), so a request chooses\n          which job runs and nothing else."
    },
    {
      "method": "POST",
      "path": "/api/bots/{key}/token",
      "auth": "session",
      "what": "Paste the @BotFather token. Verified with Telegram before it is stored; never readable back out."
    },
    {
      "method": "POST",
      "path": "/api/bots/{key}/allowed",
      "auth": "session",
      "what": "Who may talk to it — numeric Telegram ids. An unclaimed bot answers nobody and logs who knocked."
    },
    {
      "method": "POST",
      "path": "/api/bots/{key}/service",
      "auth": "session",
      "what": "start / stop / restart the bot process (systemd or the built-in supervisor — detected)."
    }
  ]
});
