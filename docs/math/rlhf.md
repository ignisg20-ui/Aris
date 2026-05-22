# RLHF objective

## Setup

We have:
* A pretrained, SFT-tuned policy $\pi_\theta$.
* A frozen reference policy $\pi_\mathrm{ref}$ (usually the SFT checkpoint).
* A reward model $r_\phi(x,y)$ trained on human preferences.

The RLHF objective is:

$$
\max_\theta\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\!\left[\, r_\phi(x,y) - \beta \, \mathrm{KL}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_\mathrm{ref}(\cdot\mid x)\right) \right].
$$

The KL term is annealed using an **adaptive KL controller**: at the end of each
batch we update

$$
\beta \leftarrow \beta \cdot \exp\!\left(\mathrm{clip}\!\left(\frac{\mathrm{KL}_\mathrm{measured}}{\mathrm{KL}_\mathrm{target}} - 1,\; -0.2,\; 0.2\right) \cdot \frac{n}{N_\mathrm{horizon}}\right).
$$

## Reward model training (Bradley-Terry)

Given preference pairs $(x, y_w, y_l)$, the Bradley-Terry likelihood is:

$$
P(y_w \succ y_l \mid x) = \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right).
$$

Negative log-likelihood gives:

$$
\mathcal{L}_\mathrm{RM} = -\mathbb{E}_{(x,y_w,y_l)} \log\sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right).
$$

## PPO update

Per-token advantage $\hat A_t$ from GAE-$\lambda$ on a value head $V_\psi$:

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t),\quad
\hat A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}.
$$

Importance ratio:

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\mathrm{old}}(a_t\mid s_t)}.
$$

Clipped surrogate loss:

$$
\mathcal{L}^\mathrm{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(\rho_t \hat A_t,\; \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

Total PPO loss:

$$
\mathcal{L} = -\mathcal{L}^\mathrm{CLIP} + c_v\,\mathcal{L}^V - c_e\,\mathcal{H}[\pi_\theta],
$$

with value loss $\mathcal{L}^V = (V_\psi(s) - R)^2$ (clipped in the codebase) and
optional entropy bonus $\mathcal{H}$.

## DPO as a closed-form alternative

For completeness: Direct Preference Optimization (Rafailov et al. 2023) solves
the same KL-regularised reward maximisation problem analytically:

$$
\mathcal{L}_\mathrm{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)} \log\sigma\!\left(\beta \log\frac{\pi_\theta(y_w\mid x)}{\pi_\mathrm{ref}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_\mathrm{ref}(y_l\mid x)}\right).
$$

Aris uses PPO by default; a DPO trainer is a one-file extension.
