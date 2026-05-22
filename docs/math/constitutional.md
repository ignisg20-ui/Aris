# Constitutional AI

Constitutional AI (Bai et al. 2022) replaces *human* feedback with *AI feedback*
guided by an explicit, natural-language **constitution** — a list of principles
the model should obey.

## SL-CAI: supervised constitutional revision

For each prompt $x$ and initial response $y_0\sim\pi_0(\cdot\mid x)$:

1. Sample a **critique** $c\sim\pi_0(\cdot\mid \mathrm{render}(x, y_0, \mathrm{principle}))$.
2. Sample a **revision** $y_1\sim\pi_0(\cdot\mid \mathrm{render}(x, y_0, c, \mathrm{principle}))$.
3. Repeat for each principle, chaining revisions.

The final $(x, y_n)$ dataset is then used as SFT data:

$$
\mathcal{L}_\mathrm{SL\text{-}CAI} = -\mathbb{E}_{(x, y_n)} \log \pi_\theta(y_n \mid x).
$$

## RL-CAI: AI preference labelling

To run RLHF without humans, we replace human-labelled preferences with
**AI-judged** preferences:

$$
\hat P(y_a \succ y_b \mid x) = \sigma\!\left(s(x, y_a; \mathrm{constitution}) - s(x, y_b; \mathrm{constitution})\right)
$$

where $s(\cdot;\cdot)$ is a scoring function obtained by prompting a stronger
reference model with the constitution and the two candidates.

The downstream reward model and PPO step are then *identical* to standard
RLHF — only the labelling source changed.

## Helpful-Harmless trade-off

Empirically, applying the constitution mainly to harmlessness while keeping
human labels for helpfulness gives the best Pareto frontier. The split is
controlled by `ConstitutionalTrainer`'s ``principles`` list.

## Why principles outperform fixed refusals

A learned classifier "refuse on category X" is brittle — any prompt rewording
defeats it. Principles act on the *output*, not the *input*, so revisions are
invariant to prompt phrasing. Combined with a strong base policy, this gives
≥10× lower jailbreak success rates without measurable helpfulness loss
(see Bai et al. 2022, Table 7).
