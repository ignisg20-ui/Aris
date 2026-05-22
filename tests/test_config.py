from pathlib import Path

import pytest

from llm.model.config import ArisConfig

CONFIG_DIR = Path(__file__).parent.parent / "llm" / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_config_loads(path):
    cfg = ArisConfig.from_yaml(path)
    assert cfg.hidden_size > 0
    assert cfg.num_layers > 0
    assert cfg.vocab_size > 0
    # Param count is reasonable.
    n = cfg.num_parameters()
    assert n > 100_000_000
