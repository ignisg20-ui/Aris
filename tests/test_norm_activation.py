import torch

from llm.model.activations import SwiGLU
from llm.model.rms_norm import RMSNorm


def test_rmsnorm_matches_reference():
    x = torch.randn(2, 4, 16)
    norm = RMSNorm(16, eps=1e-6)
    out = norm(x)
    # Manual RMSNorm.
    rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    ref = (x / rms) * norm.weight
    assert torch.allclose(out, ref, atol=1e-5)


def test_rmsnorm_preserves_dtype():
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    norm = RMSNorm(16)
    out = norm(x)
    assert out.dtype == torch.bfloat16


def test_swiglu_shape_and_finite():
    ff = SwiGLU(32, 64)
    x = torch.randn(3, 8, 32)
    out = ff(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
