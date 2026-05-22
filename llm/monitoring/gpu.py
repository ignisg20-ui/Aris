"""GPU utilization sampling.

Uses NVML when available (``pip install pynvml``), otherwise falls back to
``torch.cuda`` memory stats only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger("aris.gpu")

try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:  # pragma: no cover
    _HAS_NVML = False


@dataclass
class GPUStats:
    device_index: int
    name: str
    util_pct: float
    mem_used_gb: float
    mem_total_gb: float
    temperature_c: float | None


def sample_gpu_stats() -> list[GPUStats]:
    if not torch.cuda.is_available():
        return []
    out: list[GPUStats] = []
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem_used = torch.cuda.memory_allocated(i) / 1e9
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        util = 0.0
        temp = None
        if _HAS_NVML:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                mem_used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception as exc:  # pragma: no cover
                logger.debug("NVML query failed: %s", exc)
        out.append(GPUStats(i, name, util, mem_used, mem_total, temp))
    return out
