import torch

from llm.model.aris import ArisForCausalLM


def test_aris_forward(tiny_config):
    model = ArisForCausalLM(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(input_ids)
    assert out.logits.shape == (2, 16, tiny_config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_aris_loss(tiny_config):
    model = ArisForCausalLM(tiny_config).train()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_aris_with_moe(moe_config):
    model = ArisForCausalLM(moe_config).train()
    input_ids = torch.randint(0, moe_config.vocab_size, (2, 8))
    out = model(input_ids, labels=input_ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()
