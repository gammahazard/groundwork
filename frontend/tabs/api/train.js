"use strict";
/* API reference — Training. Content only; tabs/api/docs.js renders it.
 * Registered into the shared list by apiGroup() so a new group is a new file
 * plus one <script> tag, and nothing else has to know. */

apiGroup({
  "key": "train",
  "title": "Training",
  "blurb": "Start a run, watch it, stop it. One control covers every model, machine and card — the server refuses combinations that cannot work and says why.",
  "endpoints": [
    {
      "method": "POST",
      "path": "/api/train",
      "auth": "key",
      "mutates": true,
      "what": "Start a training run.",
      "detail": "The one entry point for every model on every machine. It syncs\n          the project's dataset to the target first and <b>refuses if the sync\n          fails</b> rather than training on a copy it could not confirm. A remote\n          run is dispatched over HTTP; the dataset travels by rsync.",
      "params": [
        [
          "model",
          "string",
          "Family key — <code>yolov8n</code>, <code>deimv2-n</code>, <code>deimv2-n-tv28</code>, <code>rtmdet-tiny</code>, <code>yolox-tiny</code>, <code>yolox-s</code>, <code>dfine-small</code>, <code>rfdetr-nano</code>, <code>centernet</code>. Default <code>yolov8n</code>."
        ],
        [
          "machine",
          "string",
          "Where to run it — <code>here</code> or a registered machine's key. Default <code>the worker</code>."
        ],
        [
          "card",
          "int|null",
          "Which GPU. Null lets the server pick the card the model can actually drive."
        ],
        [
          "imgsz",
          "int|null",
          "Training resolution. Enforced per family: yolo 960/1280; DEIM and D-FINE 960/1280/1920; RTMDet, YOLOX, CenterNet 1280/1920; rfdetr 560/672/1120. Blank uses the family default."
        ],
        [
          "epochs",
          "int|null",
          "Blank uses the family default (yolo 250, DEIM 60, RTMDet/YOLOX/CenterNet 300, D-FINE 120, rfdetr 100). Clamped to 50–600 for yolo."
        ],
        [
          "batch",
          "int|null",
          "Blank means yolo 8 at ≤960 and 4 above; every other family its own default. rfdetr must be 2 — it asserts batch × grad_accum == 16."
        ],
        [
          "run_name",
          "string",
          "Required for a challenger, ignored for yolo. Generate it yourself so a retry cannot start a second run under a new name."
        ],
        [
          "sizes",
          "int[]|null",
          "yolo only: queues up to two runs back to back, e.g. <code>[1280, 960]</code>."
        ],
        [
          "split_seed",
          "int",
          "Varies the train/val split. The one knob that asks whether a result survives a different split."
        ]
      ],
      "example": "curl -X POST \"$HOST/api/train?project=the first project\" \\\n  -H \"Authorization: Bearer $GW_KEY\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"model\":\"deimv2-n\",\"machine\":\"the worker\",\"card\":1,\n       \"imgsz\":1280,\"epochs\":60,\"batch\":8,\n       \"run_name\":\"deimv2-n-1280-x1\"}'",
      "returns": "{\"ok\":true,\"card\":1,\"epochs\":60,\"synced\":{\"sent\":0},\"machine\":\"the worker\"}",
      "notes": [
        "A challenger under a live yolo retrain is allowed <b>only if the split is already current</b> — then it is reused, not rebuilt. If it is stale the run is refused, because re-splitting deletes the images the retrain is reading.",
        "Two GPU jobs at once has failed three of four trials on this fleet, and every failure involved yolo. The server permits it; that is not the same as it being wise."
      ]
    },
    {
      "method": "GET",
      "path": "/api/train/options",
      "auth": "key",
      "what": "Every model × machine × card, and why anything is refused.",
      "detail": "What the Train control is built from. Each model carries its\n          default size, epochs and batch, the sizes it accepts, and per machine\n          which cards can run it — with a reason for each that cannot.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \\\n  \"$HOST/api/train/options?project=the first project\""
    },
    {
      "method": "GET",
      "path": "/api/retrain",
      "auth": "key",
      "what": "The yolo retrain's status, step and progress.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/retrain?project=the first project\""
    },
    {
      "method": "GET",
      "path": "/api/lab/status",
      "auth": "key",
      "what": "Challenger runs on this machine — or on the Trainer, when asked from HQ.",
      "detail": "Reports <code>actives</code> (every live challenger, with its\n          card), <code>yolo_retrain</code>, and <code>split_current</code> —\n          the boolean that decides whether a challenger may start while a\n          retrain runs. <code>split_current</code> is computed only while a\n          retrain is running, and is <code>null</code> when not asked or not\n          knowable.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/lab/status?project=the first project\""
    },
    {
      "method": "DELETE",
      "path": "/api/train",
      "auth": "key",
      "mutates": true,
      "what": "Cancel a run.",
      "params": [
        [
          "machine",
          "string",
          "Which machine's run to stop."
        ],
        [
          "kind",
          "string",
          "<code>yolo</code>, <code>challenger</code> or <code>any</code>."
        ],
        [
          "card",
          "int|null",
          "Restrict to one card."
        ]
      ],
      "example": "curl -X DELETE -H \"Authorization: Bearer $GW_KEY\" \\\n  \"$HOST/api/train?machine=the worker&kind=challenger\""
    }
  ]
});
