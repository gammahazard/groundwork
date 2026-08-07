"""Stateful worker: load the model ONCE, keep it resident, serve detections.

Reloading a quantized 3B on 8 GB is slow, so both the CLI and the playground
share a single Worker instance. This module owns the exact generate() call
signature (from the model card) in one place.
"""
from __future__ import annotations
import time

import torch
from PIL import Image

from .settings import DEVICE, DTYPE
from ..config import (DEFAULT_MAX_NEW_TOKENS,
                     DEFAULT_GENERATION_MODE)
from .model import load_model
from . import prompts
from .parse import parse, Detections


class Worker:
    def __init__(self, verbose: bool = True):
        self.model, self.processor, self.tokenizer = load_model(verbose=verbose)

    @torch.no_grad()
    def generate_raw(self, image: Image.Image, question: str,
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     generation_mode: str = DEFAULT_GENERATION_MODE) -> tuple[str, dict]:
        """Run one generation. Returns (raw_answer, stats)."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                return_tensors="pt").to(DEVICE)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(DTYPE),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            do_sample=False,             # deterministic; greedy loops emit IDENTICAL
                                         # boxes, which parse.py drops as artifacts.
                                         # (A repetition_penalty makes the loop DRIFT
                                         # into near-dups and defeats that — don't add it.)
            verbose=False,
        )
        answer = response[0] if isinstance(response, tuple) else response
        stats = {
            "seconds": round(time.time() - t0, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "n_input_tokens": int(inputs["input_ids"].shape[1]),
        }
        return answer, stats

    def detect(self, image: Image.Image, description: str,
               **kwargs) -> tuple[Detections, dict]:
        """Detect all instances matching a free-form description (box mode)."""
        raw, stats = self.generate_raw(image, prompts.detect_prompt(description),
                                       **kwargs)
        w, h = image.size
        return parse(raw, w, h), stats

    def point(self, image: Image.Image, description: str,
              **kwargs) -> tuple[Detections, dict]:
        """Locate instances as points (often better on dense/touching clusters)."""
        raw, stats = self.generate_raw(image, prompts.point_prompt(description),
                                       **kwargs)
        w, h = image.size
        return parse(raw, w, h), stats
