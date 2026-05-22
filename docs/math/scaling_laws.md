# Scaling laws

## Kaplan et al. 2020

For sufficiently large compute budgets, language modelling loss obeys a power
law in non-embedding parameters $N$, dataset size $D$, and compute $C$:

$$
\mathcal{L}(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\quad
\mathcal{L}(D) = \left(\frac{D_c}{D}\right)^{\alpha_D},\quad
\mathcal{L}(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}.
$$

Kaplan estimated $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$.

## Chinchilla (Hoffmann et al. 2022)

A joint fit gives:

$$
\mathcal{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},
$$

with $\alpha=0.34$, $\beta=0.28$, $A=406.4$, $B=410.7$, $E=1.69$.

Optimal balance under a compute budget $C \approx 6 N D$ (FLOPs/token ≈ 6N):

$$
N_\mathrm{opt}(C) \propto C^{0.46},\quad D_\mathrm{opt}(C) \propto C^{0.54}.
$$

This gives the famous rule of thumb: **~20 tokens per parameter**.

## Compute estimate per Aris size

| Preset      | Params | Chinchilla tokens | FLOPs (~6ND) |
|-------------|--------|-------------------|---------------|
| 1B          |   1.2B |       24B         |    1.7e20     |
| 7B          |   7.0B |      140B         |    5.9e21     |
| 13B         |  13.0B |      260B         |    2.0e22     |
| 70B         |  70.0B |     1.4T          |    5.9e23     |
| 200B (MoE)  |  ~28B activated | 0.6T (activated) |  ~1.0e23 |

Sparse MoE training operates on the activated-parameter count for compute
estimates, which is a major reason MoE is so attractive at large $N$.

## MoE-specific scaling

DeepSeek-MoE (Dai et al. 2024) and Switch-Transformer scaling laws show that
for fixed activated params, total loss decreases as $\log(E)$ for small $E$ but
saturates beyond a critical $E^*$ (≈ 16–64 for typical LLM workloads). Beyond
$E^*$, expert specialization degrades and load-balancing overhead dominates.
Aris-200B's 128 experts sit in the saturating but still useful regime — to
trade further: cut $E$ and re-invest in `hidden_size`.

## Inference scaling laws

Decoding throughput scales as

$$
\mathrm{tokens/s} \approx \min\!\left(\frac{\mathrm{HBM bandwidth}}{2 N_\mathrm{params,active} \cdot \mathrm{dtype bytes}},\; \frac{\mathrm{compute}}{2 \cdot \mathrm{FLOPs/token}}\right).
$$

For a 7B bf16 model on an H100 (3 TB/s, 989 TFLOPS bf16):
* memory-bound limit: $3\mathrm{e}12 / (2\cdot 7\mathrm{e}9 \cdot 2) \approx 107$ tokens/s/seq.
* compute-bound limit (batch>1): $989\mathrm{e}12 / (2\cdot 7\mathrm{e}9) \approx 70{,}000$ tokens/s.

This is why continuous batching is so impactful — it shifts inference from the
memory-bound to compute-bound regime.
