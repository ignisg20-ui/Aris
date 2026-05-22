# Mixture-of-Experts: math & training stability

## Routing

For each token $x\in\mathbb{R}^h$ the router computes affinities
$g(x) = \mathrm{softmax}(W_r x)\in\Delta^{E-1}$, then selects the top-$k$ experts
$\mathcal{S}(x)\subset\{1,\dots,E\}$:

$$
\mathrm{MoE}(x) = \sum_{e\in\mathcal{S}(x)} \tilde g_e(x)\, \mathrm{Expert}_e(x),
$$

with renormalised gates $\tilde g_e(x) = g_e(x) / \sum_{e'\in\mathcal{S}(x)} g_{e'}(x)$.

## Auxiliary losses

### Load-balancing loss

Let $T$ be the number of tokens in a batch. For each expert $e$ define

$$
f_e = \frac{1}{T k} \sum_t \mathbb{1}[e\in\mathcal{S}(x_t)],\qquad
P_e = \frac{1}{T} \sum_t g_e(x_t).
$$

The Shazeer 2017 / Switch-Transformer load-balance loss is

$$
\mathcal{L}_\mathrm{balance} = E \sum_{e=1}^E f_e \cdot P_e.
$$

When experts are perfectly balanced, $f_e = P_e = 1/E$ and
$\mathcal{L}_\mathrm{balance} = 1$. Any imbalance increases the loss.

### Router-z loss

ST-MoE (Zoph et al. 2022) adds

$$
\mathcal{L}_z = \frac{1}{T}\sum_t \left(\log\sum_e e^{\ell_e(x_t)}\right)^2,
$$

where $\ell_e(x) = (W_r x)_e$ are pre-softmax logits. This stabilises logit
magnitudes — large logits cause router collapse to a single expert. We use
$\lambda_z = 10^{-3}$.

## Token capacity

Capacity per expert: $C = \lceil k T / E \cdot c \rceil$, where $c$ is the
capacity factor (1.0 = exact, 1.25 = 25% slack for noisy routing). Tokens
exceeding capacity are *dropped* (their MoE output is replaced by the residual,
i.e. ``output = x``).

## Activated vs. total parameters

For an MoE FFN with $E$ experts of dense parameter count $P_\mathrm{dense}$, the
*total* params per layer are $E \cdot P_\mathrm{dense}$ but the *activated*
params per token are $k \cdot P_\mathrm{dense}$ (+ router + shared expert).
Aris-200B has $E=128$, $k=8$, giving an activated-to-total ratio of 6.25%.

## Shared expert

We add a single dense expert always evaluated for every token:

$$
\mathrm{MoE}_\mathrm{shared}(x) = \sum_{e\in\mathcal{S}(x)} \tilde g_e(x)\,\mathrm{Expert}_e(x) + \mathrm{Shared}(x).
$$

This is the DeepSeek-MoE design and stabilises early training by ensuring
gradient flow through a non-sparse path.
