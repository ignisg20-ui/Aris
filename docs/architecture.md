# Aris Architecture

Aris is a decoder-only Transformer with optional sparse Mixture-of-Experts FFNs,
designed to scale from 1B to 200B parameters.

## High-level data flow

```
input_ids
   │
   ▼
[Token embedding]──────────────────────────────────┐
   │                                               │
   ▼                                               │
┌─Block 0 ─────────────────────────────────────┐   │
│   RMSNorm → Attention(GQA or MLA) ── residual │   │
│   RMSNorm → SwiGLU FFN or MoE FFN ── residual │   │
└──────────────────────────────────────────────┘   │
   │                                               │
   ▼  (×N blocks)                                  │
[RMSNorm]                                          │
   │                                               │
   ▼                                               │
[LM head]──── tied to embedding (optional) ◄───────┘
   │
   ▼
logits
```

## Module map

| Concern               | Module                                              |
|-----------------------|-----------------------------------------------------|
| Configuration         | `llm.model.config.ArisConfig`                       |
| Embedding             | `llm.model.aris.ArisModel.embed_tokens`             |
| Attention             | `llm.model.attention.{GroupedQueryAttention, MultiHeadLatentAttention}` |
| RoPE                  | `llm.model.rope.RotaryEmbedding`                    |
| FFN (dense)           | `llm.model.activations.SwiGLU`                      |
| FFN (sparse)          | `llm.model.moe.MoEFFN`                              |
| Norm                  | `llm.model.rms_norm.RMSNorm`                        |
| Block                 | `llm.model.transformer.TransformerBlock`            |
| Causal LM             | `llm.model.aris.ArisForCausalLM`                    |
| KV cache              | `llm.model.kv_cache.{KVCache, LayerCache}`          |

## Scaling presets

| Preset      | Params  | Hidden | Layers | KV heads | Experts |
|-------------|---------|--------|--------|----------|---------|
| `1b.yaml`   |  1.2B   | 2048   | 22     | 4        | 1 (dense) |
| `7b.yaml`   |  7.0B   | 4096   | 32     | 8        | 1 (dense) |
| `13b.yaml`  | 13.0B   | 5120   | 40     | 8        | 1 (dense) |
| `70b.yaml`  | 70.0B   | 8192   | 80     | 8 + MLA  | 1 (dense) |
| `200b.yaml` | ~219B   | 8192   | 64     | 8 + MLA  | 64 (top-8) |

The 200B preset activates only ~28B parameters per token thanks to top-8 sparse
routing on every layer.

## Parallelism

* **Tensor parallel (TP)** — Megatron-style sharded matmuls in
  `llm.distributed.tensor_parallel`. The QKV and FFN projections are
  column-parallel; the O and Down projections are row-parallel; embeddings are
  vocab-parallel.
* **Pipeline parallel (PP)** — `llm.distributed.pipeline_parallel.PipelineParallel`
  implements GPipe-style scheduling; for steady-state 1F1B we delegate to
  DeepSpeed (`build_deepspeed_engine`).
* **Data parallel + ZeRO-1** — `llm.distributed.zero.ZeROOptimizer` shards
  optimizer state across DP ranks. For ZeRO-2/3 use DeepSpeed.
* **Mixed precision (BF16)** — `llm.distributed.precision`.
* **Gradient checkpointing** — enabled per-layer via
  `ArisConfig.use_gradient_checkpointing` and `gradient_checkpointing_layers`.

## Inference

* `InferenceEngine.generate` for one-shot.
* `InferenceEngine.generate_stream{,_async}` for streaming.
* `ContinuousBatcher` for in-flight batched serving.
* `SpeculativeDecoder` for distribution-preserving speedups via a small draft
  model.

## Alignment pipeline

```
Pretrained LM ──► SFT ──► Reward Model ──► PPO (RLHF) ──► Aris-chat
                                ▲                ▲
                                │                │
                Constitutional revision ─────────┤
                (SL-CAI: critique → revise)      │
                                                 │
                AI preference labelling (RL-CAI) ┘
```

Each step has a dedicated module under `llm.alignment`.
