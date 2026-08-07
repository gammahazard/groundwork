"use strict";
/* API reference — Export. Content only; tabs/api/docs.js renders it.
 * Registered into the shared list by apiGroup() so a new group is a new file
 * plus one <script> tag, and nothing else has to know. */

apiGroup({
  "key": "export",
  "title": "Export",
  "blurb": "Turn a run into a file you can deploy. The conversion happens in the background; poll for it.",
  "endpoints": [
    {
      "method": "GET",
      "path": "/api/export/formats",
      "auth": "key",
      "what": "What a given run can become, with the cost of each.",
      "detail": "Formats are per family: yolo offers ONNX, TorchScript, CoreML,\n          TFLite, OpenVINO, NCNN and TensorRT; DEIM offers ONNX, TorchScript and\n          CoreML. Any other family returns an empty list and a reason.",
      "params": [
        [
          "run",
          "string",
          "The run name."
        ]
      ],
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \\\n  \"$HOST/api/export/formats?run=yolov8n-65&project=the first project\""
    },
    {
      "method": "POST",
      "path": "/api/export",
      "auth": "key",
      "mutates": true,
      "what": "Start an export. Single-flight — a second is refused, not queued.",
      "detail": "The resolution is the run's own and is <b>not</b> a parameter:\n          exporting at another size is a different model, and one the holdout\n          score no longer describes.",
      "params": [
        [
          "run",
          "string",
          "The run to export."
        ],
        [
          "format",
          "string",
          "<code>onnx</code>, <code>torchscript</code>, <code>coreml</code>, <code>tflite</code>, <code>openvino</code>, <code>ncnn</code>, <code>engine</code>."
        ],
        [
          "machine",
          "string|null",
          "TensorRT only — which machine to compile on. An engine only loads on the model of GPU it was built on."
        ],
        [
          "card",
          "int|null",
          "TensorRT only — which physical card."
        ]
      ],
      "example": "curl -X POST \"$HOST/api/export?project=widgets\" \\\n  -H \"Authorization: Bearer $GW_KEY\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"run\":\"yolov8n-65\",\"format\":\"coreml\"}'",
      "notes": [
        "NCNN measured here: 4.3× faster on a Pi and <b>one object worse</b>. Fine for a live preview, not for a count that matters.",
        "TensorRT compiles on the card and is refused while that machine is training. Everything else converts on CPU and can run beside a training job."
      ]
    },
    {
      "method": "GET",
      "path": "/api/export",
      "auth": "key",
      "what": "Progress of the running export, or the last one's result.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/export\"",
      "returns": "{\"status\":\"done\",\"run\":\"yolov8n-65\",\"format\":\"coreml\",\n \"path\":\"…/outputs/exports/yolov8n-65-960.mlpackage\",\"size_mb\":12.4}"
    },
    {
      "method": "GET",
      "path": "/api/export/cards",
      "auth": "key",
      "what": "Every probed card on every machine — for choosing a TensorRT target.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/export/cards\""
    },
    {
      "method": "GET",
      "path": "/api/export/artifacts",
      "auth": "key",
      "what": "What has been exported and is on disk, newest first.",
      "detail": "Each entry carries the format it IS, derived from the same table\n          the dropdown uses — filenames alone do not say, since OpenVINO and NCNN\n          are extensionless directories.",
      "example": "curl -H \"Authorization: Bearer $GW_KEY\" \"$HOST/api/export/artifacts\"\ncurl -H \"Authorization: Bearer $GW_KEY\" -O \"$HOST/outputs/exports/yolov8n-65-960.onnx\""
    }
  ]
});
