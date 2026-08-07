"use strict";
/* API reference — Data. Content only; tabs/api/docs.js renders it.
 * Registered into the shared list by apiGroup() so a new group is a new file
 * plus one <script> tag, and nothing else has to know. */

apiGroup({
  "key": "data",
  "title": "Data",
  "blurb": "Get images in, read and write their labels.",
  "endpoints": [
    {
      "method": "POST",
      "path": "/api/upload",
      "auth": "key",
      "mutates": true,
      "what": "Add images to the fix queue, optionally pre-labelled.",
      "detail": "Multipart. <code>predict=true</code> runs the project's serving\n          model over each image first and saves its objects as a <b>draft</b>.\n          Still the fix queue: model output is a draft, and promoting it\n          unreviewed would train the next model on this one's mistakes. The\n          model is loaded once for the batch, not per file.\n          <br><br>Per-file failures are reported rather than raised — dropping\n          twenty photos in and getting one 400 because one was a .mov tells you\n          nothing about the other nineteen.",
      "params": [
        [
          "files",
          "file[]",
          "One or more images."
        ],
        [
          "predict",
          "bool",
          "Pre-label each one with the serving model."
        ]
      ],
      "example": "curl -X POST \"$HOST/api/upload?project=the first project\" \\\n  -H \"Authorization: Bearer $GW_KEY\" \\\n  -F \"files=@tray1.jpg\" -F \"files=@tray2.jpg\" \\\n  -F \"predict=true\""
    },
    {
      "method": "GET",
      "path": "/api/images/{collection}",
      "auth": "key",
      "what": "List a collection — raw (training), testset (holdout), needs_fix.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \\\n  \"$HOST/api/images/raw?project=<project>\""
    },
    {
      "method": "GET",
      "path": "/api/points/{collection}/{stem}",
      "auth": "key",
      "what": "One image's labels, size and target count.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \\\n  \"$HOST/api/points/raw/IMG_4376?project=the first project\""
    },
    {
      "method": "POST",
      "path": "/api/la/{collection}/{stem}",
      "auth": "key",
      "mutates": true,
      "what": "Ask LocateAnything for a second opinion on one image.",
      "detail": "A large model reads the picture and guesses. 30–90 seconds, and\n          it nearly fills the card — a retrain is refused while a probe runs, and\n          vice versa. It saves nothing itself; the guesses are yours to adopt.",
      "params": [
        [
          "desc",
          "string",
          "What to look for — <code>object</code>, <code>bolt</code>, your own words."
        ]
      ],
      "example": "curl -X POST \"$HOST/api/la/needs_fix/IMG_4376?project=the first project\" \\\n  -H \"Authorization: Bearer $GW_KEY\" \\\n  -H \"Content-Type: application/json\" -d '{\"desc\":\"object\"}'"
    }
  ]
});
