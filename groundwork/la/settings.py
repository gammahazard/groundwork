"""The auto-labeler's torch-touching settings — kept OUT of config.py.

config is imported by torch-free processes (web assembler, scheduler, trainer
daemon); these two values are the only configuration that requires torch, so
they live with the one subsystem that loads it.
"""
from __future__ import annotations

import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16
