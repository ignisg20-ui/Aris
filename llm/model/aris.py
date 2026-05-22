"""Full Aris model assembly.

Two public classes:

* :class:`ArisModel` — embedding + N transformer blocks + final norm. Returns
  hidden states. Useful as a backbone for reward / classification heads.
* :class:`ArisForCausalLM` — adds an LM head and computes the cross-entropy loss
  shifted by 1 (standard causal LM setup). Also returns auxiliary MoE losses
  separately so the trainer can scale them.

Initialization follows the *scaled* small-init scheme used by LLaMA / GPT-NeoX:

    σ_in   = init_std
    σ_out  = init_std / sqrt(2 * num_layers)

so that the residual stream variance stays O(1) regardless of depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .config import ArisConfig
from .kv_cache import KVCache
from .moe import MoEFFN
from .rms_norm import RMSNorm
from .transformer import TransformerBlock


@dataclass
class CausalLMOutput:
    """Forward output for ``ArisForCausalLM``."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class ArisModel(nn.Module):
    """Backbone: token-embed + stacked transformer blocks + final norm."""

    def __init__(self, config: ArisConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        out_std = std / math.sqrt(2 * self.config.num_layers)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                nn.init.zeros_(p)
                continue
            if "embed" in name:
                nn.init.normal_(p, mean=0.0, std=self.config.embedding_init_std)
            elif name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=out_std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_checkpoint: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_checkpoint is None:
            use_checkpoint = self.config.use_gradient_checkpointing

        h = self.embed_tokens(input_ids)

        # Causal additive mask for variable-length / padded batches.
        mask = self._build_attention_mask(attention_mask, h.shape[1], h.dtype, h.device)

        aux_total = torch.zeros((), device=h.device, dtype=torch.float32)
        for layer_idx, block in enumerate(self.layers):
            cache_slab = kv_cache.layers[layer_idx] if kv_cache is not None else None
            cache_off = kv_cache.seq_offset if kv_cache is not None else 0
            ckpt = use_checkpoint and (
                self.config.gradient_checkpointing_layers is None
                or layer_idx < self.config.gradient_checkpointing_layers
            )
            h = block(
                h,
                attention_mask=mask,
                position_ids=position_ids,
                cache=cache_slab,
                cache_offset=cache_off,
                use_checkpoint=ckpt,
            )
            if block.is_moe and isinstance(block.ffn, MoEFFN) and block.ffn.last_aux is not None:
                aux = block.ffn.last_aux
                aux_total = aux_total + aux.total(
                    self.config.moe.load_balance_loss_coef,
                    self.config.moe.router_z_loss_coef,
                )

        if kv_cache is not None:
            kv_cache.advance(input_ids.shape[1])

        return self.norm(h), aux_total

    def _build_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None  # SDPA's ``is_causal`` flag handles the triangular mask.
        # (B, T) -> (B, 1, 1, T) additive mask.
        m = attention_mask.to(dtype=dtype).log_().clamp_min(torch.finfo(dtype).min)
        return m[:, None, None, :]


class ArisForCausalLM(nn.Module):
    """Causal LM head wrapper."""

    def __init__(self, config: ArisConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ArisModel(config)
        if config.tie_word_embeddings:
            self.lm_head = TiedEmbedding(self.model.embed_tokens)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def gradient_checkpointing_enable(self) -> None:
        self.config.use_gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_checkpoint: bool | None = None,
    ) -> CausalLMOutput:
        hidden, aux = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_checkpoint=use_checkpoint,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            # Shift for next-token prediction.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            if aux.requires_grad or aux.item() != 0.0:
                loss = loss + aux
        return CausalLMOutput(logits=logits, loss=loss, aux_loss=aux, hidden_states=hidden)


class TiedEmbedding(nn.Module):
    """LM head that shares weights with the input embedding matrix."""

    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self.embed = embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.embed.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.embed.weight
