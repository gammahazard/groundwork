"""groundwork — local object counting on NVIDIA LocateAnything-3B (8-bit).

Small, single-purpose modules:
  config     — paths, dtype, constants (coord scale, defaults)
  prompts    — prompt templates + phrasing presets
  parse      — raw model text -> pixel-space boxes/points (GPU-free)
  image_ops  — load / crop / resolution math / OOM downscale (GPU-free)
  annotate   — draw detections + index numbers -> PNG (GPU-free)
  model      — load the 8-bit model + processor + tokenizer
  worker     — stateful worker: load ONCE, run detections resident
  infer      — high-level run_detection() with OOM fallback
"""

__version__ = "0.1.0"
