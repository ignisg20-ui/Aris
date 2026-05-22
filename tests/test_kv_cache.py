import torch

from llm.model.aris import ArisForCausalLM
from llm.model.kv_cache import KVCache


def test_kv_cache_matches_full_forward(tiny_config):
    model = ArisForCausalLM(tiny_config).eval()
    a = tiny_config.attention
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 12))

    with torch.no_grad():
        full = model(input_ids).logits

        cache = KVCache.allocate(
            num_layers=tiny_config.num_layers,
            batch_size=1,
            num_kv_heads=a.num_kv_heads,
            head_dim=a.head_dim,
            max_seq=20,
            device=input_ids.device,
            dtype=torch.float32,
        )
        # Prefill first 8 tokens.
        _ = model(input_ids[:, :8], kv_cache=cache).logits
        # Step through the remaining 4 tokens one at a time.
        step_logits = []
        for t in range(8, 12):
            step_out = model(input_ids[:, t : t + 1], kv_cache=cache).logits
            step_logits.append(step_out[:, -1])
        incremental = torch.stack(step_logits, dim=1)

    # Compare last 4 positions.
    assert torch.allclose(full[:, 8:], incremental, atol=1e-3)
