import torch

from llm.inference.sampling import SamplingParams, sample_token


def test_greedy_sampling_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    out = sample_token(logits, SamplingParams(temperature=0.0))
    assert out.item() == 1


def test_top_k_filters():
    torch.manual_seed(42)
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    # top_k=1 forces argmax.
    out = sample_token(logits, SamplingParams(temperature=1.0, top_k=1))
    assert out.item() == 1


def test_repetition_penalty_reduces_repeat_likelihood():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    history = torch.tensor([[2]])
    params = SamplingParams(temperature=0.0, repetition_penalty=2.0)
    out = sample_token(logits, params, token_history=history)
    # Token 2 should NOT be chosen because it was penalised.
    assert out.item() != 2
