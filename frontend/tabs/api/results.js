"use strict";
/* API reference — Results. Content only; tabs/api/docs.js renders it.
 * Registered into the shared list by apiGroup() so a new group is a new file
 * plus one <script> tag, and nothing else has to know. */

apiGroup({
  "key": "results",
  "title": "Results",
  "blurb": "What every run scored, and everything known about one of them.",
  "endpoints": [
    {
      "method": "GET",
      "path": "/api/runs/{run}",
      "auth": "key",
      "what": "Everything about a single run — either family.",
      "detail": "Answers for yolo and challenger runs from the same name. Carries\n          the ledger row, an eval summary, what is actually on disk, and which\n          formats it can export to — the last read from the same table the Export\n          tab uses, so it cannot disagree with what pressing the button offers.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \\\n  \"$HOST/api/runs/yolov8n-65?project=the first project\"",
      "returns": "{\"run\":\"yolov8n-65\",\"kind\":\"yolo\",\"model\":\"YOLOv8n (default)\",\n \"on_disk\":true,\n \"eval\":{\"mae\":0.0147,\"conf\":0.3,\"iou\":0.3,\"n\":68,\"misses\":1,\"worst\":1,\n         \"source\":\"testset\",\"imgsz\":960,\"on_tray_filter\":false,\n         \"dupes\":0,\"dropped\":0},\n \"artifacts\":{\"weights\":true,\"count_eval\":true,\"eval_preview\":true},\n \"export_formats\":[\"onnx\",\"torchscript\",\"coreml\",\"tflite\",\"openvino\",\"ncnn\",\"engine\"]}",
      "notes": [
        "<code>misses</code> is how many holdout images it counted wrong, at any margin. <code>worst</code> is the largest error on a single image — the risk a mean hides.",
        "<code>on_tray_filter</code> records the regime the score was measured under. Every count_eval on disk was written before the serving-filter switch existed and says true, while the filter now ships off."
      ]
    },
    {
      "method": "GET",
      "path": "/api/runs",
      "auth": "key",
      "what": "The yolo ledger, newest first, with deltas against the previous run.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/runs?project=the first project\""
    },
    {
      "method": "GET",
      "path": "/api/lab/runs",
      "auth": "key",
      "what": "The challenger ledger.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/lab/runs?project=the first project\""
    },
    {
      "method": "GET",
      "path": "/api/runs/{run}/images",
      "auth": "key",
      "what": "Per-image results for a run — predicted vs true, per holdout image.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \\\n  \"$HOST/api/runs/yolov8n-65/images?project=the first project\""
    },
    {
      "method": "GET",
      "path": "/api/runs/{run}/log",
      "auth": "key",
      "what": "The training log's tail.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/runs/yolov8n-65/log\""
    }
  ]
});
