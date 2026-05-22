# Attention: mathematical formulation

## Scaled dot-product attention

Given queries $Q\in\mathbb{R}^{T\times d_k}$, keys $K\in\mathbb{R}^{T\times d_k}$,
and values $V\in\mathbb{R}^{T\times d_v}$:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

For causal self-attention we add an additive mask
$M_{ij} = 0$ if $j\le i$ else $-\infty$ before the softmax.

## Multi-head extension

Project $X\in\mathbb{R}^{T\times d_\mathrm{model}}$ into $H$ heads:

$$
Q_h = X W_h^Q,\quad K_h = X W_h^K,\quad V_h = X W_h^V,\quad h=1,\dots,H,
$$

each head has dimension $d_k = d_\mathrm{model}/H$. Concatenate and project:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{Attention}(Q_h, K_h, V_h))_{h=1..H} \, W^O.
$$

## Grouped-Query Attention (GQA)

GQA replaces $H$ KV heads with $H_{kv} < H$. Each KV head is shared by
$g = H/H_{kv}$ query heads. Result:

* Compute is unchanged.
* KV cache size scales by $H_{kv}/H$ — typically 4–8×.

## Rotary Positional Embeddings (RoPE)

Define complex-valued head dimensions $d=d_k$ even, and frequencies
$\theta_k = b^{-2k/d}$, $k=0,\dots,d/2-1$.

The rotation matrix at position $m$ is block-diagonal:

$$
R(m) = \mathrm{diag}\!\left(\begin{pmatrix}\cos m\theta_k & -\sin m\theta_k \\\\
\sin m\theta_k & \cos m\theta_k\end{pmatrix}\right)_{k}.
$$

We apply $q_m = R(m) W^Q x_m$, $k_n = R(n) W^K x_n$ and observe

$$
\langle q_m, k_n\rangle = x_m^\top {W^Q}^\top R(n-m) W^K x_n,
$$

i.e. the inner product depends only on the **relative** offset $n-m$.

### Length extrapolation

For a target length $L' = sL$ with $s>1$ (scaling factor):

* **Linear** (Chen et al. 2023): $m \mapsto m/s$.
* **NTK** (bloc97): $b \mapsto b\cdot s^{d/(d-2)}$.
* **YaRN** (Peng et al. 2023): blend the two on a per-frequency basis using
  ``correction_dim(β_fast)`` and ``correction_dim(β_slow)``.

## Multi-Head Latent Attention (MLA)

DeepSeek-V2 compresses KV via a low-rank latent $c\in\mathbb{R}^{r}$:

$$
c_t = W^{KV}_a x_t,\quad k_t^\mathrm{nope} = W^K_b c_t,\quad v_t = W^V_b c_t.
$$

A small "rope" half of the key is kept separately (per-head shared) so we can
apply RoPE:

$$
k_t^\mathrm{rope} = R(t) W^{KR}_a x_t.
$$

The full key is $k_t = [k_t^\mathrm{nope}; k_t^\mathrm{rope}]$. The KV cache
stores only $c_t$ and $k_t^\mathrm{rope}$, cutting memory by ~5-10× compared to
GQA at 64k+ contexts.

## Flash-Attention

Flash-Attention (Dao 2022) replaces the explicit $S = QK^\top$ matrix with a
**tiled online softmax** computed in SRAM:

1. Split $Q$, $K$, $V$ into blocks of size $B_r$, $B_c$.
2. For each row block, iterate over column blocks; maintain
   $m_i = \max_j s_{ij}$ and $\ell_i = \sum_j e^{s_{ij}-m_i}$ incrementally.
3. The output for block $i$ is $\sum_j e^{s_{ij}-m_i} V_j / \ell_i$.

The recurrence is exact, requires $O(N)$ HBM I/O, and gives 2–4× speedups at
long contexts. PyTorch's ``F.scaled_dot_product_attention`` dispatches to
Flash-Attention 2 on supported hardware automatically.
