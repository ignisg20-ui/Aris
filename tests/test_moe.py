import torch

from llm.model.moe import MoEFFN, Top2Router


def test_router_topk_renormalised():
    router = Top2Router(hidden_size=16, num_experts=8, top_k=2)
    x = torch.randn(32, 16)
    weights, indices, _ = router(x)
    assert weights.shape == (32, 2)
    assert indices.shape == (32, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(32), atol=1e-4)


def test_moe_ffn_forward(moe_config):
    layer = MoEFFN(moe_config).eval()
    x = torch.randn(2, 5, moe_config.hidden_size)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    aux = layer.last_aux
    assert aux is not None
    assert aux.load_balance.dim() == 0
    assert aux.router_z.dim() == 0
