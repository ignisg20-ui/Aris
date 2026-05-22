"""Mixed-precision helpers.

Aris trains in BF16 by default:
* Master weights kept in fp32 (or bf16 with ZeRO-1 + Kahan compensation if memory
  is tight).
* Forward / backward in bf16.
* Optimizer states (Adam moments) in fp32.

BF16 is strictly preferred over fp16 because its exponent range matches fp32 and
removes the need for dynamic loss scaling.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn


def autocast_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


@contextlib.contextmanager
def mixed_precision(dtype: torch.dtype, enabled: bool = True) -> Iterator[None]:
    if dtype == torch.float32 or not enabled or not torch.cuda.is_available():
        yield
        return
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


class BF16Wrapper(nn.Module):
    """Casts forward inputs to bf16 and outputs back to fp32 for the loss.

    Useful for tightly-coupled training loops that want bf16 forward/backward but
    fp32 losses and metrics. Master weights are kept in their original dtype.
    """

    def __init__(self, module: nn.Module, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return self.module(*args, **kwargs)
