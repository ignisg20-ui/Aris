# Aris

> Production-grade decoder-only LLM framework with sparse Mixture-of-Experts,
> Multi-Head Latent Attention, Rotary Positional Embeddings, Flash-Attention,
> KV cache, speculative decoding, BF16 mixed precision, tensor / pipeline /
> ZeRO parallelism, RLHF & Constitutional AI alignment, and a FastAPI serving
> layer with Kubernetes manifests and Prometheus monitoring.

Aris targets the 1B–200B parameter range. The same codebase trains every size
preset; only the YAML config changes.

## Repository layout

```
llm/
  model/         RMSNorm, SwiGLU, RoPE, GQA, MLA, KV cache, MoE, Transformer block
  tokenizer/     BPE wrapper (sentencepiece / tiktoken) + chat template
  training/      Pretraining loop, curriculum, checkpointing, optimizer
  inference/     Generation engine, batching, streaming, speculative decoding
  distributed/   Tensor / pipeline / ZeRO parallelism, BF16, NCCL bootstrap
  alignment/     SFT, reward model, PPO RLHF, Constitutional AI, rejection sampling
  safety/        Refusal policies, harmlessness classifier, constraints, adversarial
  api/           FastAPI server (/generate /chat /train /finetune)
  monitoring/    Prometheus metrics, structured logging, NVML GPU stats
  configs/       Size presets: 1b / 7b / 13b / 70b / 200b
deploy/          Dockerfile, docker-compose, Kubernetes manifests
docs/            Architecture overview + math derivations
tests/           Unit tests (model, attention, moe, sampling)
```

## Quickstart

### Install

```bash
pip install -e .
# optional extras:
pip install -e ".[train]"   # deepspeed, accelerate, wandb
pip install -e ".[flash]"   # flash-attn 2.x (CUDA only)
```

### Build a model from a config

```python
from llm.model import ArisConfig, ArisForCausalLM

cfg = ArisConfig.from_yaml("llm/configs/1b.yaml")
model = ArisForCausalLM(cfg)
print(f"{model.num_parameters/1e9:.2f}B params")
```

### Run pretraining

```bash
# Single node:
python -m llm.training.pretrain \
    --config llm/configs/7b.yaml \
    --data /data/pretrain \
    --out checkpoints/aris-7b \
    --steps 200000 --micro-batch 8 --accum 4

# 8 nodes × 8 GPUs with TP=8 (see deploy/k8s/training-job.yaml):
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_endpoint=... \
    -m llm.training.pretrain --config llm/configs/70b.yaml --tp=8 ...
```

### Serve

```bash
ARIS_MODEL_CONFIG=llm/configs/7b.yaml \
ARIS_CHECKPOINT=checkpoints/aris-7b/latest/model_rank_0000.pt \
aris-serve --host 0.0.0.0 --port 8000
```

Hit it:

```bash
curl -s http://localhost:8000/generate \
    -d '{"prompt":"Hello","sampling":{"max_new_tokens":32}}'

curl -N http://localhost:8000/chat \
    -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

### Built-in chat UI

The server also ships a zero-dependency HTML chat front-end at
[`http://localhost:8000/ui/`](http://localhost:8000/ui/) (root `/` redirects to
it). Streaming tokens via SSE, editable system prompt, live
temperature / top-p / top-k / max-tokens controls, stop generation, new chat
reset, and a `/health` indicator. The assets live in
[`llm/api/web/`](llm/api/web/) and are also bundled inside `aris.exe`.

### Docker / Kubernetes

```bash
# Build images
docker build -t aris:inference -f deploy/Dockerfile --target inference .
docker build -t aris:train     -f deploy/Dockerfile --target train     .

# Local stack with Prometheus + Grafana
docker compose -f deploy/docker-compose.yml up

# Production cluster
kubectl apply -f deploy/k8s/
```

The HPA in `deploy/k8s/hpa.yaml` scales inference replicas on
`aris_tokens_per_second` and CPU utilization.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a high-level diagram and
module map, and the math derivations under [`docs/math/`](docs/math):

* [`attention.md`](docs/math/attention.md) — SDPA, MHA, GQA, RoPE, MLA, Flash.
* [`moe.md`](docs/math/moe.md) — routing, balance/z losses, scaling.
* [`rlhf.md`](docs/math/rlhf.md) — Bradley-Terry RM, PPO, KL control, DPO.
* [`constitutional.md`](docs/math/constitutional.md) — SL-CAI, RL-CAI.
* [`scaling_laws.md`](docs/math/scaling_laws.md) — Kaplan / Chinchilla / MoE / inference scaling.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

Apache-2.0
