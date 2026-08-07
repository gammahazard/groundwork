"""Load LocateAnything-3B in 8-bit. Single responsibility: return the triple.

Isolated so the (slow, stateful, GPU) load lives in exactly one place and both
app entry points reuse it via worker.py.
"""
from __future__ import annotations
import time

import torch
from transformers import (AutoModel, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig)

from ..config import MODEL_PATH
from .settings import DTYPE


def load_model(model_path: str = MODEL_PATH, verbose: bool = True):
    """Load tokenizer, processor, and the 8-bit model. Returns (model, proc, tok)."""
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=DTYPE,
        device_map={"": 0},
        attn_implementation="sdpa",   # magi/flash not installed -> sdpa
        trust_remote_code=True,
    ).eval()

    if verbose:
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[model] loaded 8-bit in {time.time() - t0:.1f}s | "
              f"VRAM {alloc:.2f} GB alloc / {reserved:.2f} GB reserved", flush=True)
    return model, processor, tokenizer
