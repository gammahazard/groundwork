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

# Teach PIL to open HEIC/HEIF, once, for every process. iPhones send it when a
# photo goes as a FILE (not a compressed "photo"), and uploads store it verbatim
# (upload._OK_SUFFIX allows .heic/.heif), so a .heic image reaches the training
# and eval SUBPROCESSES, not only the web and bot — and each is its own
# interpreter. Registering here covers every `python -m groundwork.*` at once.
# Guarded: absent pillow-heif, or any failure loading it, this is a silent no-op
# and HEIC is simply unsupported, exactly as before.
try:
    from pillow_heif import register_heif_opener as _register_heif_opener
    _register_heif_opener()
except Exception:  # noqa: BLE001 — no pillow-heif, or a broken native libheif
    pass
