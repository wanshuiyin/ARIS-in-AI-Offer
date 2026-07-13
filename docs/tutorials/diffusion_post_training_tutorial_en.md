## §0 TL;DR

> 💡 **Diffusion Post-Training in 9 sentences** — one page covering the RL/DPO/Flow-RL family (see §1–§10 for derivations).

1. **Why it's hard**: diffusion generation runs $T$ steps (typically 20–50), with reward given only once at the terminal state $x_0$ — so **sparse terminal reward + long denoising trajectory + credit assignment** combine, giving one extra "trajectory integration" dimension beyond LLM RLHF.

2. **Three main lines**: (i) RL on denoising MDP (DDPO / DPOK — treat $T$-step denoising as an MDP); (ii) Direct reward backprop (DRaFT / AlignProp / ReFL — treat reward as a differentiable loss and backprop through the $T$-step sampler); (iii) Preference optimization (Diffusion-DPO / D3PO / SPO / Diffusion-KTO / MaPO — port the LLM DPO family to diffusion).

3. **DDPO (Black et al. 2024 ICLR, arXiv 2305.13301)**: view denoising as a $T$-step MDP, state $= (x_t, t, c)$, action $= x_{t-1}$, per-trajectory reward $R(x_0, c)$; use REINFORCE or PPO-clip to update $\log p_\theta(x_{t-1} \mid x_t, c)$.

4. **AlignProp (Prabhudesai et al. 2024 ICLR, arXiv 2310.03739)** & **DRaFT (Clark et al. 2024 ICLR, arXiv 2309.17400)**: when reward $R$ is differentiable in $x_0$, directly **backprop** $R(x_0)$ through the $T$-step sampler into $\theta$. **The key engineering problem**: memory is $\mathcal{O}(T)$; DRaFT-K / AlignProp only backprop the final $K$ steps (typically $K \in \{1, 5\}$), and combined with gradient checkpointing the memory drops to $\mathcal{O}(K)$.

5. **Diffusion-DPO (Wallace et al. 2024 CVPR, arXiv 2311.12908)**: replace the LLM DPO's $\log\pi/\pi_\text{ref}$ with diffusion's **per-step ELBO surrogate** — specifically, use $-\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ as one lower-bound term of $\log p_\theta(x_0)$, and assemble it into a DPO contrastive loss over the pair $(y_w, y_l)$.

6. **D3PO (Yang et al. 2024 CVPR, arXiv 2311.13231)**: **completely RM-free** — directly plug human thumbs up/down feedback on generated images into the KL-regularized optimal solution's implicit reward; the derivation parallels DPO but operates on the diffusion **per-step Markov chain**.

7. **SPO (Liang et al. 2024, arXiv 2406.04314)**: observes that different denoising steps have different preferences (high-noise steps learn composition, low-noise steps learn detail), and extends DPO to be **step-aware** — for each $t$, independently sample an in-step pair $(x_{t-1}^w, x_{t-1}^l)$, and weight the loss along the step dimension.

8. **Flow-GRPO (Liu et al. 2025, arXiv 2505.05470)**: the first work bringing GRPO to Flow Matching. Two key tricks: an **ODE→SDE equivalent conversion** that turns the deterministic flow into an explorable stochastic process; and **denoising reduction** — fewer steps for training, full steps for inference. RL-tuned SD3.5-M raised GenEval from 63% to 95%.

9. **Reward hacking is the real boss**: over-saturated colors, monotonous composition, style convergence, high PickScore but ugly to humans — mitigations include reward ensembles (HPSv2 + PickScore + ImageReward + CLIP-Score), KL anchor (Diffusion-DPO's $\beta$), and early stopping on reward plateau. SD3 / FLUX **barely disclose post-training details**, but the community consensus is that SD3.5 Turbo and FLUX.1 dev use a DPO + distillation hybrid.

> ✅ **vs LLM RLHF in one sentence** — LLM RLHF cares about "token-level credit assignment + KL anchor"; diffusion post-training cares about "denoising-step credit assignment + memory blow-up (backprop) or sample blow-up (RL)". Same problem in essence — sparse reward + long trajectory — only the physical meaning of "trajectory" changes.

## §1 Intuition: why diffusion post-training is hard

### 1.1 Single-step vs multi-step generation: the essential difference

LLM rewards are generally also sequence-level, but tokens are discrete, trajectory length $L \sim 10^3$, vocabulary moderate. The diffusion "trajectory" is $T$ steps of denoising, each operating on a continuous high-dimensional tensor $x_t \in \mathbb{R}^{C \times H \times W}$ (SDXL latent is $4\times128\times128 = 65536$ dimensions), with $T$ typically 20–50.

| Dimension | LLM RLHF | Diffusion Post-Training |
| --- | --- | --- |
| Trajectory length | $L$ (response token count) | $T$ (denoising step count, typically 20–50) |
| Single-step action | discrete token | continuous vector in $\mathbb{R}^d$ ($d \sim 10^4$–$10^5$) |
| Reward frequency | usually only at the terminal | usually only at terminal $x_0$ |
| Exploration | sampling temperature / top-p | DDIM is deterministic; "noise injection" needed to explore (DDPO uses stochastic DDPM; Flow-GRPO uses ODE→SDE) |
| Reward source | trained RM (BT) / rule | trained image RM (ImageReward / HPSv2 / PickScore) / rule (object count / OCR) |
| Memory bottleneck | 4 copies (policy + ref + RM + V) | 1 copy UNet/DiT, but **direct backprop** must store $T$-step activations |

### 1.2 The three main lines

- **Line A (RL on denoising MDP)**: treat diffusion as an RL environment, **not requiring reward to be differentiable**. Representatives: DDPO, DPOK, Flow-GRPO.
- **Line B (Direct reward backprop)**: when reward is differentiable in $x_0$, **directly gradient-descend**, akin to treating the reward as a new loss. Representatives: DRaFT, AlignProp, ReFL.
- **Line C (Preference optimization, DPO-style)**: port the LLM DPO family to diffusion, **no longer requiring on-policy sampling**. Representatives: Diffusion-DPO, D3PO, SPO, Diffusion-KTO, MaPO.

```
                              preference / reward signal
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
       reward non-diff           reward diff             preference pairs (offline)
            │                       │                       │
        Line A: RL              Line B: backprop         Line C: DPO family
        DDPO, DPOK,             DRaFT, AlignProp,        Diffusion-DPO,
        Flow-GRPO               ReFL                     D3PO, SPO, KTO, MaPO
            │                       │                       │
        most general            memory-friendly (K-step)   off-policy fast
        but sampling expensive  but requires diff reward   needs preference data
```

### 1.3 One-line intuitions

> 💡 **Core intuitions** —

- Line A treats $T$ denoising steps as an RL trajectory: each step is a stochastic policy output.
- Line B treats reward as a differentiable loss: backprop through $T$ steps, but $O(T)$ memory crushes you.
- Line C bypasses reward entirely: uses the implicit reward of preference pairs $= \beta \log(p_\theta/p_\text{ref})$.

### 1.4 Convention (used throughout)

| Symbol | Meaning |
| --- | --- |
| $x_0$ | clean image (latent or pixel) |
| $x_t,\; t = 0, \dots, T$ | noised sample; $x_T \approx \mathcal{N}(0, I)$ |
| $\epsilon_\theta(x_t, t, c)$ | noise predicted by UNet/DiT (DDPM parameterization) |
| $v_\theta(t, x, c)$ | Flow Matching vector field |
| $c$ | condition (text embedding / class) |
| $p_\theta(x_{t-1} \mid x_t, c)$ | reverse process single-step conditional distribution |
| $R(x_0, c)$ | terminal reward (scalar, from RM or rule) |
| $\pi_\text{ref}$ / $p_\text{ref}$ | reference model (usually the SFT-trained base) |
| $\beta$ | KL/temperature hyperparameter (same meaning as in LLM DPO) |

## §2 RL for Diffusion: DDPO and DPOK

### 2.1 Treating denoising as an MDP (DDPO viewpoint)

Black et al. 2024 ICLR *Training Diffusion Models with Reinforcement Learning* (arXiv 2305.13301) key observation: the DDPM/DDIM reverse process is itself a **finite-horizon MDP**.

Definitions:

- **State**: $s_t = (x_t, t, c)$, with time reversed $t = T, T-1, \dots, 1$
- **Action**: $a_t = x_{t-1}$ (sampled from $p_\theta(\cdot \mid x_t, c)$)
- **Transition**: deterministic — $s_{t-1} = (x_{t-1}, t-1, c)$
- **Reward**: $r_t = 0$ for $t > 1$, $r_1 = R(x_0, c)$ (terminal-only)
- **Policy**: $\pi_\theta(a_t \mid s_t) = p_\theta(x_{t-1} \mid x_t, c)$

Policy gradient theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=1}^{T} \nabla_\theta \log p_\theta(x_{t-1} \mid x_t, c)\, R(x_0, c)\right]$$

**Key**: $\log p_\theta(x_{t-1} \mid x_t, c)$ is Gaussian in DDPM, so the log-prob is analytical and the gradient can be computed directly.

### 2.2 DDPO-SF (Score Function) algorithm

The simplest variant (DDPO-SF, score function estimator):

1. **Sampling phase** — starting from prompt $c$, run $T$ DDPM reverse steps to get a trajectory $\tau = (x_T, x_{T-1}, \dots, x_0)$; compute $R(x_0, c)$.
2. **Update phase** — REINFORCE-style gradient estimator:

$$\hat{g} = \frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log p_\theta(x_{t-1}^{(n)} \mid x_t^{(n)}, c)\, (R^{(n)} - b)$$

where $b$ is a baseline (typically the batch-mean reward).

### 2.3 DDPO-IS (Importance Sampling, PPO-style)

The practical recommended DDPO variant uses **PPO-clip** with per-step importance ratios:

$$\rho_t = \frac{p_\theta(x_{t-1} \mid x_t, c)}{p_{\theta_\text{old}}(x_{t-1} \mid x_t, c)}, \qquad L^\text{CLIP}_t = \min\!\big(\rho_t R, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) R\big)$$

> ⚠️ **Per-step ratio, not trajectory ratio** — diffusion PPO uses **per-step** importance ratios, not the product over $T$ steps. The full-trajectory ratio is the product of $T$ ratios — variance explodes; clipping per-step independently is the only way to stay stable.

### 2.4 DDPO's two reward experiments

Black et al. run DDPO + SD-1.5 on four rewards:

| Reward type | Example | Signal nature |
| --- | --- | --- |
| Compressibility | JPEG file size | rule-based scalar |
| Aesthetic | LAION aesthetic predictor | trained MLP |
| Prompt-image alignment | CLIP-Score / LLaVA judge | VLM-based |
| Object presence | DETR / OWL-ViT count | rule-based |

Across all four rewards, DDPO yields large gains over reward-weighted regression (RWR baseline), and can shift styles (e.g. emoji → oil painting) — proof that RL is genuinely exploring, not just mode-seeking.

### 2.5 DPOK: KL-regularized RL for diffusion

Fan et al. **2023 NeurIPS** *DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models* (arXiv 2305.16381) is contemporaneous with DDPO; the difference is an **explicit KL anchor**:

$$\boxed{\;\max_\theta\; \mathbb{E}_{\tau \sim p_\theta}\!\left[R(x_0, c)\right] - \beta\, \mathbb{E}_c\!\left[\text{KL}\!\big(p_\theta(\cdot \mid c) \,\big\Vert\, p_\text{ref}(\cdot \mid c)\big)\right]\;}$$

The KL term expanded to per-step — **note this is an upper bound, not an identity**:

$$\text{KL}\big(p_\theta(x_0 \mid c) \,\Vert\, p_\text{ref}(x_0 \mid c)\big) \;\le\; \sum_{t=1}^{T} \mathbb{E}\!\left[\text{KL}\!\big(p_\theta(x_{t-1} \mid x_t, c) \,\Vert\, p_\text{ref}(x_{t-1} \mid x_t, c)\big)\right]$$

The right-hand side is the exact chain-rule expansion of the **joint trajectory KL** ($\text{KL}(p_\theta(x_{0:T}\mid c) \Vert p_\text{ref}(x_{0:T}\mid c))$) — this identity holds exactly because $p_\theta$ and $p_\text{ref}$ share the same prior $p(x_T)=\mathcal N(0,I)$. But what DPOK actually wants to regularize is the **terminal marginal** $\text{KL}(p_\theta(x_0\mid c) \Vert p_\text{ref}(x_0\mid c))$; by the data-processing inequality for KL (non-increasing under marginalization), the terminal marginal KL is only an **upper bound** on this per-step sum, not equal to it. DPOK uses this tractable, closed-form upper bound as a surrogate — effective in practice, but not an exact equivalence.

Since the DDPM $p_\theta(x_{t-1} \mid x_t, c)$ is Gaussian, **the KL between two Gaussians has a closed form**, so per-step KL is directly computable. DPOK uses policy gradient + this KL penalty (an upper-bound surrogate), the diffusion analog of RLHF's "$\beta \log(\pi/\pi_\text{ref})$ added to reward".

> 💡 **DPOK vs DDPO** —

- DDPO: pure RL (REINFORCE or PPO-clip), KL is implicit (via ratio clipping).
- DPOK: explicit KL term, matching LLM RLHF's "KL added to reward".
- Empirically, DPOK is more stable on reward-prompt alignment (ImageReward); DDPO is more aggressive on rule-based rewards like compressibility.

### 2.6 DDPO failure modes and mitigations

| Symptom | Cause | Mitigation |
| --- | --- | --- |
| Reward up but FID crashes | over-optimization on RM blind spots | KL penalty / LoRA fine-tune (prevent base drift) |
| Same prompt converges to one composition | mode collapse, policy finds RM high-score mode | reward ensemble / early stop |
| High reward but ugly to humans | RM scale not aligned with humans | weighted multi-RM + human eval calibration |
| Training unstable | per-step ratio accumulates over $T$ steps | per-step PPO-clip $\epsilon = 0.1$ is more stable than the LLM $0.2$ |

## §3 Direct Reward Fine-Tuning: DRaFT / AlignProp / ReFL

### 3.1 Core idea: treat reward as a differentiable loss

If the reward $R(x_0, c)$ is **differentiable** in $x_0$ (true for typical CNN/ViT RMs) and the diffusion sampler is differentiable, then we can **directly backprop to $\theta$**:

$$\theta \leftarrow \theta + \eta\, \nabla_\theta R\!\big(x_0(\theta), c\big), \quad x_0(\theta) = \text{Sample}_\theta^T(c)$$

where $\text{Sample}_\theta^T(c)$ denotes running $T$ reverse steps from $x_T \sim \mathcal{N}(0,I)$ to obtain $x_0$. This treats the entire reverse trajectory as one **giant differentiable computation graph** and optimizes reward end-to-end.

> ✅ **Advantage** — no sampling variance; the gradient signal has far lower variance than REINFORCE-style RL.

> ❌ **Cost** — must store $T$ steps of activations; vanilla implementation uses $\mathcal{O}(T \cdot M_\text{UNet})$ memory, which for SDXL UNet is ≈ hundreds of GB, **completely untrainable**.

### 3.2 DRaFT (Clark et al. 2024 ICLR, arXiv 2309.17400)

*Directly Fine-Tuning Diffusion Models on Differentiable Rewards*. Two core tricks:

**Trick 1: DRaFT-K, only backprop the last $K$ steps.**

The full chain rule (denoise step $\epsilon_\theta(x_t,t)$ both influences the next step $x_{t-1}$ and **directly** depends on $\theta$):

$$\nabla_\theta R(x_0) = \frac{\partial R}{\partial x_0} \cdot \sum_{t=1}^{K}\left(\prod_{s=1}^{t-1} \frac{\partial x_{s-1}}{\partial x_s}\right) \cdot \frac{\partial x_{t-1}}{\partial \theta}\bigg|_{\text{direct}}$$

where $\partial x_{t-1}/\partial\theta|_\text{direct}$ is the partial derivative of step-$t$ on $\theta$ via $\epsilon_\theta(x_t,t)$ directly (not through the indirect $x_t \to x_t$ path), and $\prod_s$ is the Jacobian product propagation during backward. The first $T-K$ steps run under `torch.no_grad()`; only the last $K$ steps retain the graph. K=1 (DRaFT-1) already gives a very strong signal — the last step has the largest direct effect on $x_0$. Autograd automatically sums the $\partial/\partial\theta|_\text{direct}$ across all $K$ steps, so the code only needs `loss.backward()`.

**Trick 2: LoRA fine-tune + high learning rate.**

Only train a LoRA adapter (~1% parameters), with the base UNet frozen. Combined with gradient checkpointing, memory fits within 24 GB on a single GPU.

Pseudo-code:

```python
# DRaFT-K one-step training
x_t = torch.randn(B, C, H, W).to(device)  # x_T
with torch.no_grad():
    for t in range(T-1, K, -1):           # first T-K steps without grad
        x_t = ddim_step(unet_lora, x_t, t, cond)
for t in range(K, 0, -1):                  # last K steps with grad
    x_t = ddim_step(unet_lora, x_t, t, cond)
x_0 = x_t
reward = image_rm(x_0, prompt)             # ImageReward / HPSv2
loss = -reward.mean()                       # note negative sign — we maximize reward
loss.backward()                             # memory O(K)
optimizer.step()
```

> ⚠️ **Is DRaFT-1 equivalent to REINFORCE?** — No. DRaFT-1 is a **true reparameterized gradient** (pathwise estimator); REINFORCE is a score-function estimator. The former has very low variance but requires reward to be differentiable; the latter is general but high variance. $\nabla \log p$ vs $\partial x / \partial \theta$ are two different gradient estimator classes.

### 3.3 AlignProp (Prabhudesai et al. arXiv 2310.03739, 2023-10; ICLR 2024 venue; the arXiv version was later superseded/withdrawn)

*Aligning Text-to-Image Diffusion Models with Reward Backpropagation* — almost parallel to DRaFT (both appeared on arXiv in late 2023), with the same core idea: **reward backprop through denoising**. Differences:

| Dimension | DRaFT | AlignProp |
| --- | --- | --- |
| Truncation | DRaFT-K, last $K$ steps retain grad | randomly selected $K$ steps retain grad (randomized truncated BPTT) |
| Memory optimization | gradient checkpointing | gradient checkpointing + LoRA |
| Recommended reward | HPSv1, PickScore, Aesthetic | ImageReward, HPSv2, PickScore |
| Mode collapse mitigation | simple KL anchor | LoRA scale annealing + early stop |

AlignProp's key contribution is theorizing "why reward backprop works" — proving that under fixed-point assumptions, the truncated BPTT gradient is a biased but low-variance estimator of the true gradient.

### 3.4 ReFL (Xu et al. 2023 NeurIPS, arXiv 2304.05977)

*ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*. This is the **original ImageReward paper**, which also proposes the ReFL (Reward Feedback Learning) algorithm.

ReFL is close to DRaFT in spirit but predates it. It treats reward directly as a loss and **supervises only at one randomly selected intermediate step**, an early implementation of the DRaFT idea:

$$\mathcal{L}_\text{ReFL} = \mathcal{L}_\text{simple} - \lambda \cdot \mathbb{E}_{t' \sim [t_\text{min}, t_\text{max}]}\!\big[R\big(\hat{x}_0(x_{t'}, t')\big)\big]$$

where $\hat{x}_0(x_{t'}, t') = (x_{t'} - \sqrt{1-\bar\alpha_{t'}}\epsilon_\theta(x_{t'}, t'))/\sqrt{\bar\alpha_{t'}}$ is the $x_0$ estimate from a single-step $\epsilon$-prediction (one-step Tweedie unfolding).

**Key difference**: ReFL is "single-step backprop + $L_\text{simple}$ trained jointly", while DRaFT/AlignProp is "multi-step backprop + pure reward loss". ReFL is more stable but reward gains are smaller, because it only sees the one-step $x_0$ estimate rather than the true sampling trajectory.

### 3.5 Three-way comparison

| Algorithm | Backprop strategy | Memory | Reward gain | Stability |
| --- | --- | --- | --- | --- |
| **ReFL** (Xu 2023) | single-step $\hat{x}_0$ + $L_\text{simple}$ hybrid | $\mathcal{O}(1)$ | small | high |
| **DRaFT-K** (Clark 2024) | last $K$ steps BPTT | $\mathcal{O}(K)$ | large | medium (large $K$ prone to over-optimization) |
| **AlignProp** (Prabhudesai 2024) | random $K$ steps BPTT | $\mathcal{O}(K)$ | large | medium |

> 💡 **Memory back-of-the-envelope** — SDXL UNet ~2.6B parameters, one forward in fp16 needs ~6–8 GB activations; $K = 5$ ~ 30–40 GB; $K = T = 50$ > 300 GB, **only feasible with multi-machine sharding**. This is why $K=1$ is the most common in practice — negligible precision loss, engineering-friendly.

### 3.6 Reward hacking in direct reward backprop

Direct backprop is more prone to hacking than RL because the gradient signal is "too precise":

| Symptom | Example |
| --- | --- |
| **Over-saturation** | HPSv2 prefers high contrast → trained images become hyper-saturated |
| **Style monotonicity** | ImageReward training data is biased → all prompts produce the same style |
| **Trypophobia patterns** | some RMs prefer "dense textures", and the model learns trypophobic patterns |
| **Mode collapse** | multiple samples from the same prompt are nearly identical |

**Mitigations**: reward ensemble (mean or min of HPSv2 + PickScore + ImageReward), KL anchor, early stop, small LoRA scale.

## §4 Preference Optimization: the Diffusion-DPO family

### 4.1 Diffusion-DPO (Wallace et al. 2024 CVPR, arXiv 2311.12908)

*Diffusion Model Alignment Using Direct Preference Optimization*. The key challenge in porting LLM DPO to diffusion: **diffusion's $\log p_\theta(x_0 \mid c)$ has no closed form** — must use an ELBO substitute.

#### 4.1.1 Derivation (key steps)

The LLM DPO core is the KL-regularized RL optimal solution:

$$\pi^*(y \mid x) \propto \pi_\text{ref}(y \mid x) \exp\!\big(r(x, y) / \beta\big)$$

Solving for $r = \beta \log(\pi^*/\pi_\text{ref}) + \beta \log Z$ and substituting into Bradley-Terry, the $\log Z$ cancels.

For diffusion, replace "sample $y$" with "sample trajectory $(x_T, \dots, x_0)$"; the optimal form is the same, but $\log p$ is the full-trajectory likelihood:

$$\log p_\theta(x_{0:T} \mid c) = \log p(x_T) + \sum_{t=1}^{T} \log p_\theta(x_{t-1} \mid x_t, c)$$

This trajectory log-likelihood **is analytical** (each term is a Gaussian log-prob). **However**: if every update step requires a full trajectory run, the compute cost explodes.

**Wallace et al.'s trick: use an ELBO surrogate.**

DDPM's $L_\text{simple}$ is one of the (negative) ELBO terms of $\log p_\theta(x_0)$; specifically:

$$-\log p_\theta(x_0 \mid c) \le L_\text{simple}(x_0, c, \theta) = \mathbb{E}_{t, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right] + \text{const}$$

> ⚠️ **$L_\text{simple}$ is not a "tight" ELBO** — the true VLB has a per-timestep SNR-dependent weight $\lambda_t$ (equivalently $\omega(\lambda_t)$) in front of each term; $L_\text{simple}$ **sets this weight to 1**, a deliberate reweighting by Ho et al. 2020 chosen for better perceptual sample quality — it is not a tight bound obtained by directly simplifying the VLB. Treating the unweighted MSE as "the" ELBO is an approximation, not an equality.

Use $-L_\text{simple}$ as a **single-sample estimate** of $\log p_\theta(x_0 \mid c)$ (Jensen's inequality strictly gives a lower bound, but numerically it is usable as a DPO implicit-reward proxy) and substitute into the DPO framework ($\omega(\lambda_t)$ is the SNR-dependent weight; $L_\text{simple}$ is the special case $\omega(\lambda_t)\equiv 1$):

$$\boxed{\;\mathcal{L}_\text{Diff-DPO}(\theta) = -\mathbb{E}_{(x_0^w, x_0^l, c, t, \epsilon)}\log\sigma\!\left(-\beta T\,\omega(\lambda_t)\!\left[\|\epsilon^w - \epsilon_\theta(x_t^w, t, c)\|^2 - \|\epsilon^w - \epsilon_\text{ref}(x_t^w, t, c)\|^2 - \|\epsilon^l - \epsilon_\theta(x_t^l, t, c)\|^2 + \|\epsilon^l - \epsilon_\text{ref}(x_t^l, t, c)\|^2\right]\right)\;}$$

> 💡 **Intuitive reading** — the inside of the sigmoid is "for $y_w$, policy denoises better than ref" minus "for $y_l$, policy denoises better than ref". If the policy is more accurate on $y_w$ and less accurate on $y_l$, the difference is positive and the loss decreases.

#### 4.1.2 Implementation details

- $(x_0^w, x_0^l)$ come from a human-preference pair conditioned on a single prompt $c$ (Pick-a-Pic is the main dataset).
- Each training step **randomly samples $t \in \{1, \dots, T\}$ and $\epsilon \sim \mathcal{N}(0,I)$**, constructing $x_t^w = \sqrt{\bar\alpha_t} x_0^w + \sqrt{1-\bar\alpha_t}\epsilon$ (the same $\epsilon$ is used for both $y_w$ and $y_l$ — paired noise).
- $\pi_\text{ref}$ is a frozen base UNet (usually the original SDXL checkpoint).

> 💡 **Shared noise is a variance-reduction trick, not a mathematical requirement** — $x_t^w$ and $x_t^l$ sharing the same $\epsilon$ (paired noise / common random numbers) reduces gradient variance and makes training more stable; but even with independently sampled $\epsilon^w, \epsilon^l$, the loss remains a valid (higher-variance but unbiased) estimator of the same DPO objective — unpaired noise does not change the scale or comparability of $\beta$, it just makes optimization noisier. Paired noise is recommended in practice, but it is not a correctness requirement.

#### 4.1.3 Results (on SDXL)

- Consistently better than SDXL base on PickScore / HPSv2.
- Training cost is ~1.5–2× SFT (must run UNet twice: policy + ref).
- Simpler than DDPO: fully offline, no sampling needed.

### 4.2 D3PO (Yang et al. 2024 CVPR, arXiv 2311.13231)

*Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model*. Posted to arXiv around the same time as Diffusion-DPO (2023-11); the difference is in **derivation path**:

- **Diffusion-DPO**: KL-regularized RL → ELBO surrogate → DPO loss.
- **D3PO**: directly port the LLM DPO derivation **step-by-step onto the diffusion Markov chain** — each denoising step is treated as an MDP step, using the same "solve for implicit reward + plug into BT" framework.

The final D3PO loss form is nearly the same as Diffusion-DPO's:

$$\mathcal{L}_\text{D3PO}(\theta) = -\mathbb{E}_{(\tau^w, \tau^l)} \log\sigma\!\left(\beta \sum_{t=1}^{T}\!\left[\log\frac{p_\theta(x_{t-1}^w \mid x_t^w, c)}{p_\text{ref}(x_{t-1}^w \mid x_t^w, c)} - \log\frac{p_\theta(x_{t-1}^l \mid x_t^l, c)}{p_\text{ref}(x_{t-1}^l \mid x_t^l, c)}\right]\right)$$

What's needed is a **full trajectory pair** $(\tau^w, \tau^l)$; if the preference pair only has the final image $(x_0^w, x_0^l)$, reconstruct the trajectory using $q(x_{1:T} \mid x_0)$ (with DDPM's forward q-sample).

> 💡 **Diffusion-DPO vs D3PO in practice** — the two are mathematically equivalent (under the ELBO surrogate, D3PO's trajectory log-ratio reduces to Diffusion-DPO's single-step $\epsilon$-distance difference). **In practice**:

- Diffusion-DPO uses single-$t$ estimation (cheaper), D3PO sums over the full trajectory (more accurate but expensive).
- Diffusion-DPO is stable on Pick-a-Pic; D3PO is stable on collected thumbs up/down data.
- Industrial deployments mostly use Diffusion-DPO (simpler compute).

### 4.3 SPO (Liang et al. 2024, arXiv 2406.04314)

*Step-aware Preference Optimization: Aligning Preference with Denoising Performance at Each Step*. **Key observation**: different denoising steps are **responsible for different aspects of the image**:

- High-noise steps ($t \approx T$): determine global structure (composition, object placement).
- Low-noise steps ($t \approx 0$): determine local detail (texture, edges).

Diffusion-DPO's "single-$t$ sampling" treats all steps equally — but human preferences have different "importance" across steps.

#### 4.3.1 The two SPO modifications

**Modification 1: In-step preference** — for the same $x_t$, **independently sample two $x_{t-1}^w, x_{t-1}^l$**, and use a step-wise reward model to judge "which $x_{t-1}$ is better at step $t$".

**Modification 2: Step-aware weighting** — the SPO loss is weighted along the step dimension:

$$\mathcal{L}_\text{SPO}(\theta) = -\mathbb{E}_{t \sim w(t),\; x_t}\!\left[\log\sigma\!\left(\beta\!\log\frac{p_\theta(x_{t-1}^w \mid x_t, c)}{p_\text{ref}(x_{t-1}^w \mid x_t, c)} - \beta\!\log\frac{p_\theta(x_{t-1}^l \mid x_t, c)}{p_\text{ref}(x_{t-1}^l \mid x_t, c)}\right)\right]$$

where $w(t)$ is the step sampling distribution (typically uniform or biased toward middle $t$).

#### 4.3.2 The in-step reward model

To obtain the in-step preference $(x_{t-1}^w, x_{t-1}^l)$, SPO trains a **step-wise reward model** $R(x_{t-1}, x_t, c, t)$ that judges "given $x_t$, is $x_{t-1}$ a good transition at step $t$?" It does not directly score pixel quality of $x_{t-1}$ but rather estimates **whether this transition at step $t$** leads to a high-quality $x_0$.

> ✅ **The key SPO win** — given the same preference data, SPO's effective signal is $\times T$ (each prompt yields a pair at every one of $T$ steps). Empirically SPO outperforms Diffusion-DPO on PickScore / HPSv2 by 1–3 points.

### 4.4 Diffusion-KTO (Li et al. 2024 NeurIPS, arXiv 2404.04465)

*Aligning Diffusion Models by Optimizing Human Utility*. The diffusion version of LLM KTO (Ethayarajh 2024, arXiv 2402.01306).

**LLM KTO core idea**: replace the BT preference model with Kahneman-Tversky prospect theory, requiring only **per-sample binary feedback** (thumbs up/down), **no pairs needed**:

$$L_\text{KTO} = \mathbb{E}_{x, y}\!\left[\lambda_y v\!\big(\beta \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} - z_0(x)\big)\right]$$

where $v(\cdot)$ is the prospect-theoretic value function (thumbs up uses $1 - \sigma(\cdot)$, thumbs down uses $\sigma(\cdot)$), and $z_0$ is the reference utility.

Diffusion-KTO replaces $\log(\pi_\theta/\pi_\text{ref})$ with the Diffusion-DPO $\epsilon$-distance ELBO surrogate:

$$L_\text{Diff-KTO} = \mathbb{E}_{x_0, c, \text{label}}\!\left[\lambda_\text{label}\, v\!\left(\beta T \left[\|\epsilon - \epsilon_\text{ref}\|^2 - \|\epsilon - \epsilon_\theta\|^2\right] - z_0(c)\right)\right]$$

> 💡 **Practical value of Diffusion-KTO** — in industrial settings, binary feedback (like/dislike) vastly outnumbers paired comparisons; KTO lets you use that data directly.

### 4.5 MaPO (Hong et al. 2024, arXiv 2406.06424)

*Margin-aware Preference Optimization for Aligning Diffusion Models without Reference*. **Core idea**: **drop the reference model entirely** — similar in spirit to LLM SimPO.

MaPO loss simultaneously optimizes two things:

1. **Likelihood margin**: $\log p_\theta(x_0^w) - \log p_\theta(x_0^l)$ (estimated via the ELBO surrogate $\|\epsilon - \epsilon_\theta\|^2$).
2. **Likelihood of preferred**: $\log p_\theta(x_0^w)$ itself must be high (to prevent "both sides dropping").

$$\mathcal{L}_\text{MaPO}(\theta) = -\mathbb{E}\!\left[\log\sigma\!\big(\beta(\hat{\ell}_w - \hat{\ell}_l) - \gamma\big) + \alpha \hat{\ell}_w\right]$$

where $\hat{\ell} = -\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$ is the likelihood surrogate, $\gamma$ is the margin, and $\alpha$ is the likelihood-term weight.

**Advantages**:

- **No reference UNet needed**: saves half the memory (from $2 \times M$ to $M$).
- **Solves reference mismatch**: when fine-tuning to a new style (where the reference is far from the target distribution), Diffusion-DPO training collapses, but MaPO is stable.
- **15% faster training** (paper-reported, validated across 5 domains).

### 4.6 DPO family overview

| Method | Needs ref? | Preference type | Memory | When to use |
| --- | --- | --- | --- | --- |
| **Diffusion-DPO** (Wallace 2024) | ✅ | paired | 2× | general alignment |
| **D3PO** (Yang 2024) | ✅ | paired or thumbs | 2× | when no RM available |
| **SPO** (Liang 2024) | ✅ + step-RM | per-step paired | 2× + small step-RM | extracting maximum step-level signal |
| **Diffusion-KTO** (Li 2024) | ✅ | unpaired binary | 2× | large volume of thumbs data |
| **MaPO** (Hong 2024) | ❌ | paired | 1× | style fine-tune / memory-constrained |

## §5 Flow-GRPO: RL for Flow Matching

### 5.1 Why Flow Matching also needs post-training

SD3 / FLUX / Lumina all moved to Flow Matching (Rectified Flow); post-training needs are the same:

- Improve composition benchmarks like GenEval / DPG (color, count, spatial relations).
- Improve OCR / text rendering accuracy.
- Improve prompt-image alignment (VLM judge).

But Flow Matching is a **deterministic ODE** ($\dot x_t = v_\theta(t, x, c)$) — the stochastic transition assumed by DDPO/DPOK does not exist. Directly plugging it into the RL framework fails.

### 5.2 Flow-GRPO's two core tricks

Liu et al. 2025 *Flow-GRPO: Training Flow Matching Models via Online RL* (arXiv 2505.05470) solves two fundamental Flow + RL problems:

#### Trick 1: ODE → SDE equivalent conversion

For the Rectified Flow ODE $\dot x_t = v_\theta(t, x_t, c)$, construct an **equivalent SDE**:

$$dx_t = \big[v_\theta(t, x_t, c) + \tfrac{1}{2}\sigma(t)^2 \nabla_x \log p_t(x_t)\big]\,dt + \sigma(t)\,dW_t$$

**Key property** (Song et al. 2021 score SDE framework): this SDE has **exactly the same marginal** $p_t$ as the original ODE. The difference is that the SDE provides **stochastic exploration** (the $dW_t$ noise), allowing RL to sample different trajectories.

For Flow Matching, $\nabla_x \log p_t = -\epsilon/\sigma_t$ (under a Gaussian path), and the score can be derived from $v_\theta$. Setting $\sigma(t)$ as a schedule (typically $\sigma(t) = \sqrt{1-t}$) gives the SDE sampler used by Flow-GRPO during training.

> 💡 **Physical meaning** — adding $\sigma\,dW$ lets the particle "jitter" out multiple trajectories while keeping the marginal unchanged, so $G$ samples from the same prompt actually differ → group statistics for GRPO are well-defined.

#### Trick 2: Denoising reduction

GRPO needs a group of $G$ trajectories per sample, typically $G = 16$–$32$. Flow Matching inference normally uses 25–50 steps, so **a single training sample needs $\approx 25G$ forwards** — too expensive.

Flow-GRPO uses **fewer steps during training** (e.g. 10 steps); inference still uses 25–50 steps. The SDE is more uniform on the schedule, so the "exploration quality" of few-step training is enough:

$$\text{Training}: T_\text{train} = 10, \quad \text{Inference}: T_\text{infer} = 28$$

Empirically, no quality drop on GenEval / OCR / Aesthetic.

### 5.3 Flow-GRPO advantage computation

Completely parallel to GRPO for LLM — for the same prompt $c$, sample $G$ final images $\{x_0^{(1)}, \dots, x_0^{(G)}\}$, score each with reward $r_i$, and normalize within the group:

$$\hat{A}_i = \frac{r_i - \text{mean}_{j}(r_j)}{\text{std}_j(r_j) + \epsilon}$$

All steps within a trajectory share the same $\hat{A}_i$ (same as per-token sharing in LLM GRPO).

### 5.4 Flow-GRPO loss

Let the SDE Euler step transition log-prob be $\log p_\theta(x_{t-1} \mid x_t, c)$ (Gaussian), with importance ratio $\rho_{i,t} = p_\theta / p_{\theta_\text{old}}$. PPO-clip:

$$L^\text{Flow-GRPO} = \mathbb{E}\!\left[\frac{1}{G}\sum_i\!\frac{1}{T_\text{train}}\!\sum_t \min\!\big(\rho_{i,t}\hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\big) - \beta\, \text{KL}_{i,t}(p_\theta \Vert p_\text{ref})\right]$$

KL still uses the K3 estimator (same as GRPO for LLM).

### 5.5 Geometric meaning of advantage on the vector field

> ✅ **L3 understanding** —

- LLM GRPO's advantage reweights in the token logit space;
- Flow-GRPO's advantage directly reweights the **direction correction** of $v_\theta$ — specifically, when $\hat A_i > 0$, push $v_\theta(t, x_t, c)$ toward the direction $(x_{t-1}^{(i)} - x_t^{(i)})/dt$ that trajectory $\tau_i$ actually took.
- This is a "directional gradient" on the $v_\theta$ space, equivalent to importance-weighted $\epsilon$-prediction reweighting under a Gaussian path.

### 5.6 Flow-GRPO empirical results

The paper reports for SD3.5-M:

| Benchmark | SD3.5-M base | Flow-GRPO |
| --- | --- | --- |
| GenEval overall | 63% | **95%** |
| Visual text rendering | 59% | **92%** |
| Aesthetic (Schuhmann) | 5.8 | 6.1 |

> ⚠️ **GenEval 95% looks too perfect** — the paper does claim this number, but note that GenEval measures rule-verifiable object count / color / spatial-relation tasks, which are inherently RL-friendly (highly regularized rewards). On more subjective benchmarks like PartiPrompt / DPG, gains are 5–10 points, which is more realistic.

## §6 Code Patterns (readable pseudo-code)

### 6.1 DDPO REINFORCE-style update

```python
import torch
import torch.nn.functional as F

def ddpo_step(unet, ref_unet, scheduler, prompts, reward_fn,
              T=20, B=4, lr=1e-5, beta=0.0):
    """
    DDPO-SF one-step training (REINFORCE + optional KL anchor).
    prompts: list of B text prompts
    reward_fn: callable (x0_batch, prompts) -> [B] scalar
    """
    # ── 1. Rollout: sample G=B trajectories ──
    x = torch.randn(B, 4, 64, 64, device=device)
    traj_log_probs = []
    with torch.set_grad_enabled(False):                   # rollout needs no grad
        x_t = x
        for t in reversed(range(T)):
            # predict noise + sample x_{t-1}
            eps_pred = unet(x_t, t, prompts)
            mean, std = scheduler.step_mean_std(x_t, eps_pred, t)
            x_tm1 = mean + std * torch.randn_like(mean)   # stochastic transition
            traj_log_probs.append((mean.detach(), std.detach(), x_tm1.detach()))
            x_t = x_tm1
        x_0 = x_t

    # ── 2. Reward ──
    R = reward_fn(x_0, prompts)                            # [B]
    A = (R - R.mean()) / (R.std() + 1e-8)                  # batch baseline

    # ── 3. Policy gradient: re-forward to get log p_θ ──
    x_t = x.detach()
    loss_pg = 0.0
    for t, (mean_old, std_old, x_tm1) in zip(reversed(range(T)), traj_log_probs):
        eps_pred = unet(x_t, t, prompts)                   # grad needed
        mean, std = scheduler.step_mean_std(x_t, eps_pred, t)
        # t=0 is the deterministic terminal transition under the DDPM convention
        # (std=0, no noise added); dividing by / taking log of std there NaNs/Infs
        # unless guarded.
        std_safe = std.clamp(min=1e-6)
        # Gaussian log-prob
        log_p = -0.5 * (((x_tm1 - mean) / std_safe) ** 2).sum([1, 2, 3])
        log_p -= std_safe.log().sum([1, 2, 3])
        loss_pg = loss_pg - (log_p * A).mean()             # REINFORCE
        if beta > 0:
            ref_eps = ref_unet(x_t, t, prompts).detach()
            mean_ref, std_ref = scheduler.step_mean_std(x_t, ref_eps, t)
            std_ref_safe = std_ref.clamp(min=1e-6)         # same terminal std=0 issue
            # Gaussian-Gaussian KL closed form
            kl = ((mean - mean_ref) ** 2 / (2 * std_ref_safe ** 2)
                  + (std_safe / std_ref_safe) ** 2 / 2
                  - 0.5 - (std_safe / std_ref_safe).log()).sum([1, 2, 3])
            loss_pg = loss_pg + beta * kl.mean()
        x_t = x_tm1.detach()
    return loss_pg
```

> ⚠️ **DDPO implementation footguns** —

- Use `set_grad_enabled(False)` during rollout, then enable grad in the policy-gradient pass — avoids $O(T)$ memory.
- DDPM stochastic transition is critical **for REINFORCE/score-function estimators**: DDIM-eta=0 is deterministic, with no $dW$ dimension and no well-defined log-density to differentiate, so the score-function gradient degenerates — **DDPO (REINFORCE-based) must use DDPM or DDIM-eta=1**; conversely, for the pathwise-gradient methods in §3 (DRaFT/AlignProp), deterministic sampling is actually a prerequisite.
- **The t=0 terminal transition is deterministic (std=0, the Ho et al. 2020 convention)**: if the policy-gradient recompute pass takes log/division on std directly, this step NaNs/Infs; guard with `std.clamp(min=1e-6)` (as in the code above), or special-case t=0 and keep gradient only through the mean term.
- The batch baseline $A = (R - \bar R)/\sigma_R$ is much more stable than no baseline.
- Train LoRA rather than full fine-tune, otherwise the base drifts quickly.

### 6.2 Diffusion-DPO loss

```python
def diffusion_dpo_loss(unet, ref_unet, scheduler,
                       x0_w, x0_l, prompt_embeds, beta=2000.0):
    """
    Diffusion-DPO (Wallace 2024) one-step training.
    x0_w, x0_l: [B, 4, H, W]  preferred / dispreferred latents
    beta: paper uses 2000~5000 — this magnitude depends on implementation conventions
          (mean vs. sum reduction of the MSE; whether T refers to the DDPM training
          schedule or the inference schedule), not a portable, implementation-independent
          "β·T temperature"
    """
    B = x0_w.shape[0]
    t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=x0_w.device)
    noise = torch.randn_like(x0_w)                          # paired noise (variance reduction, not required)

    xt_w = scheduler.add_noise(x0_w, noise, t)
    xt_l = scheduler.add_noise(x0_l, noise, t)

    # ── policy ε-prediction ──
    eps_w = unet(xt_w, t, prompt_embeds)
    eps_l = unet(xt_l, t, prompt_embeds)

    # ── reference ε-prediction (frozen) ──
    with torch.no_grad():
        ref_eps_w = ref_unet(xt_w, t, prompt_embeds)
        ref_eps_l = ref_unet(xt_l, t, prompt_embeds)

    # ── ELBO surrogate: -‖ε - ε_θ‖² is a proxy for log p_θ ──
    err_w_pol = ((noise - eps_w) ** 2).mean([1, 2, 3])      # [B]
    err_w_ref = ((noise - ref_eps_w) ** 2).mean([1, 2, 3])
    err_l_pol = ((noise - eps_l) ** 2).mean([1, 2, 3])
    err_l_ref = ((noise - ref_eps_l) ** 2).mean([1, 2, 3])

    # DPO log-ratio: smaller err = better likelihood
    #   log(π_θ/π_ref)(y_w) ≈ -(err_w_pol - err_w_ref)
    diff_w = -(err_w_pol - err_w_ref)
    diff_l = -(err_l_pol - err_l_ref)

    inner = beta * (diff_w - diff_l)
    loss = -F.logsigmoid(inner).mean()

    with torch.no_grad():
        margin = inner.mean()
        accuracy = (inner > 0).float().mean()
    return loss, {"margin": margin.item(), "acc": accuracy.item()}
```

> ⚠️ **β magnitude note (implementation-dependent, not a clean universal fact)** — Diffusion-DPO's $\beta$ is several orders of magnitude larger than LLM DPO's, and directionally this is because it absorbs the accumulation over trajectory length $T$; but saying "$\beta T$ is the true temperature" glosses over two concrete implementation choices: (1) whether the per-pixel MSE is reduced with `.mean()` or `.sum()` (this tutorial's code uses `.mean([1,2,3])`, which needs a much larger $\beta$ than a sum-reduced loss would); (2) which $T$ is meant — the $\sim$1000-step DDPM training schedule that the DPO loss samples/sums over, not the $\sim$20–50-step inference schedule used elsewhere in this tutorial (§1.4). So $\beta T$ is not a portable, implementation-independent "true temperature" — only informal intuition for why diffusion-DPO's $\beta$ is orders of magnitude larger than LLM DPO's. Paper uses $\beta \in [2000, 5000]$ (under that implementation's mean-reduce + 1000-step convention); LLM DPO uses $\beta \in [0.05, 0.5]$.

### 6.3 AlignProp / DRaFT backprop with checkpointing

```python
def alignprop_step(unet_lora, scheduler, prompts, reward_fn,
                   T=50, K=1, B=4, lr=1e-5):
    """
    DRaFT-K / AlignProp one-step training.
    K: last K steps retain gradient
    Memory O(K); K=1 is single-forward magnitude on SDXL
    """
    x = torch.randn(B, 4, 64, 64, device=device)

    # ── First T - K steps without grad ──
    with torch.no_grad():
        for t in reversed(range(K, T)):
            eps = unet_lora(x, t, prompts)
            x = scheduler.step_ddim(x, eps, t)             # deterministic DDIM

    # ── Last K steps with grad ──
    for t in reversed(range(K)):
        eps = unet_lora(x, t, prompts)                     # gradient ON
        x = scheduler.step_ddim(x, eps, t)

    x_0 = x
    # ── Backprop ──
    reward = reward_fn(x_0, prompts)                       # [B]
    loss = -reward.mean()                                  # max reward = min -reward
    return loss

# Memory analysis:
#   K=1:  ~24 GB on SDXL (single forward + grad)
#   K=5:  ~60 GB
#   K=10: ~120 GB (needs multi-GPU)
#   K=T=50: ~600 GB (completely infeasible)
```

> 💡 **Is K=1 enough?** — Yes. Intuition: the last step $x_1 \to x_0$ has the largest effect on $x_0$ (the variance of the previous 49 steps is compressed), so the 1-step backprop signal already dominates. Clark 2024 empirically confirms $K=1$ and $K=5$ have negligible difference.

### 6.4 SPO step-aware preference loss

```python
def spo_loss(unet, ref_unet, scheduler, step_rm,
             x_t, t, prompt_embeds, beta=500.0):
    """
    SPO (Liang 2024) in-step preference.
    Given x_t and t, independently sample two x_{t-1}, let step_rm judge winner.
    """
    # ── 1. Use policy to sample two x_{t-1} candidates (must no_grad,
    #      otherwise DPO log-prob backprop would flow back into the sampler) ──
    with torch.no_grad():
        eps_sample = unet(x_t, t, prompt_embeds)
        mean_s, std_s = scheduler.step_mean_std(x_t, eps_sample, t)
        noise_a, noise_b = torch.randn_like(mean_s), torch.randn_like(mean_s)
        x_a = mean_s + std_s * noise_a
        x_b = mean_s + std_s * noise_b

    # ── 2. step-wise reward model judges winner ──
    with torch.no_grad():
        r_a = step_rm(x_a, x_t, t, prompt_embeds)          # [B]
        r_b = step_rm(x_b, x_t, t, prompt_embeds)
        winner = (r_a > r_b).long()                        # [B], 1 if a wins
    x_w = torch.where(winner.bool()[:, None, None, None], x_a, x_b).detach()
    x_l = torch.where(winner.bool()[:, None, None, None], x_b, x_a).detach()

    # ── 3. compute log p_θ / log p_ref for x_w, x_l (grad-aware forward) ──
    eps = unet(x_t, t, prompt_embeds)
    mean, std = scheduler.step_mean_std(x_t, eps, t)
    log_p_w = -0.5 * ((x_w - mean) / std).pow(2).sum([1, 2, 3])
    log_p_l = -0.5 * ((x_l - mean) / std).pow(2).sum([1, 2, 3])
    with torch.no_grad():
        ref_eps = ref_unet(x_t, t, prompt_embeds)
        ref_mean, ref_std = scheduler.step_mean_std(x_t, ref_eps, t)
        log_pref_w = -0.5 * ((x_w - ref_mean) / ref_std).pow(2).sum([1, 2, 3])
        log_pref_l = -0.5 * ((x_l - ref_mean) / ref_std).pow(2).sum([1, 2, 3])

    inner = beta * ((log_p_w - log_pref_w) - (log_p_l - log_pref_l))
    return -F.logsigmoid(inner).mean()
```

### 6.5 Flow-GRPO group-relative advantage

```python
def flow_grpo_step(flow_net, ref_flow, prompts, reward_fn,
                   G=16, T_train=10, sigma_fn=lambda t: (1 - t) ** 0.5,
                   eps_clip=0.2, beta=0.04):
    """
    Flow-GRPO one-step training.
    G: sample G trajectories per prompt.
    T_train: SDE step count for training (inference uses 28-50).
    """
    P = len(prompts)
    # Repeat each prompt G times
    prompts_rep = sum([[p] * G for p in prompts], [])      # [P*G]

    # ── 1. SDE rollout: ODE→SDE equivalent conversion ──
    x_t = torch.randn(P * G, 4, 64, 64, device=device)
    log_probs_old = []                                     # for PPO importance ratio
    trajectory = [x_t.clone()]
    with torch.no_grad():
        for i in range(T_train):
            t_now = 1.0 - i / T_train
            t_next = 1.0 - (i + 1) / T_train
            dt = t_next - t_now
            sigma = sigma_fn(t_now)
            v = flow_net(x_t, t_now, prompts_rep)
            # SDE Euler: drift = v + 0.5 σ² ∇log p (PF-ODE → SDE conversion, Song 2021)
            # !!! IMPORTANT: the following drift is a simplified pedagogical placeholder.
            #     Production implementations should re-derive score from the data/noise/velocity
            #     relationship, with the specific Rectified Flow / EDM schedule; refer to Flow-GRPO
            #     paper Eq.(6) + official repo for deployment. The -v/σ here is only illustrative
            #     (guarded with +1e-6, so it does not divide by zero: at t=1, sigma→0, and this
            #     correction term goes to 0 together with sigma² — not NaN — but the formula itself
            #     is not a rigorous derivation).
            #     Note: this placeholder's sigma_fn(t)=sqrt(1-t) gives zero SDE noise at the
            #     trajectory start (t=1, pure noise) and maximal noise right at the trajectory end
            #     (t=0, near-clean data) — the reverse of the usual intuition that exploration noise
            #     should be larger early and smaller near the final output. Anyone implementing a
            #     real Flow-GRPO trainer should re-derive both the score term (paper Eq.6) and the
            #     sign/direction of the noise schedule rather than reusing this illustrative sigma_fn.
            drift = v + 0.5 * sigma ** 2 * (-v / (sigma + 1e-6))  # placeholder, see paper Eq.(6)
            noise = torch.randn_like(x_t)
            x_next = x_t + drift * dt + sigma * noise * abs(dt) ** 0.5
            # Gaussian log-prob (transition)
            mean = x_t + drift * dt
            std = sigma * abs(dt) ** 0.5
            log_p = -0.5 * ((x_next - mean) / std).pow(2).sum([1, 2, 3])
            log_probs_old.append(log_p)
            x_t = x_next
            trajectory.append(x_t.clone())
        x_0 = x_t

    # ── 2. Group-relative advantage ──
    R = reward_fn(x_0, prompts_rep)                        # [P*G]
    R = R.view(P, G)
    mean_R = R.mean(dim=1, keepdim=True)
    std_R = R.std(dim=1, keepdim=True) + 1e-8
    A = ((R - mean_R) / std_R).view(P * G)                 # [P*G]

    # ── 3. PPO-clip loss with KL ──
    loss = 0.0
    x_t = trajectory[0]
    for i in range(T_train):
        t_now = 1.0 - i / T_train
        v = flow_net(x_t, t_now, prompts_rep)              # grad ON
        sigma = sigma_fn(t_now)
        drift = v + 0.5 * sigma ** 2 * (-v / (sigma + 1e-6))
        dt = -1.0 / T_train
        mean = x_t + drift * dt
        std = sigma * abs(dt) ** 0.5
        log_p_new = -0.5 * ((trajectory[i+1] - mean) / std).pow(2).sum([1, 2, 3])
        ratio = (log_p_new - log_probs_old[i]).exp()
        surr1 = ratio * A
        surr2 = ratio.clamp(1 - eps_clip, 1 + eps_clip) * A
        loss = loss - torch.min(surr1, surr2).mean()

        # K3 KL estimator
        with torch.no_grad():
            v_ref = ref_flow(x_t, t_now, prompts_rep)
            drift_ref = v_ref + 0.5 * sigma ** 2 * (-v_ref / (sigma + 1e-6))
            mean_ref = x_t + drift_ref * dt
            log_p_ref = -0.5 * ((trajectory[i+1] - mean_ref) / std).pow(2).sum([1,2,3])
        delta = log_p_ref - log_p_new
        kl_k3 = (delta.exp() - delta - 1)
        loss = loss + beta * kl_k3.mean()

        x_t = trajectory[i + 1].detach()
    return loss
```

### 6.6 Combined reward signal

```python
def combined_reward(images, prompts, weights=None):
    """
    Weighted multi-reward combination — mitigates single-RM hacking.
    """
    weights = weights or {"image_reward": 0.4, "hps_v2": 0.3,
                          "pickscore": 0.2, "clip_score": 0.1}
    rewards = {}
    rewards["image_reward"] = image_reward_model(images, prompts)        # [-1, 4]
    rewards["hps_v2"] = hps_v2(images, prompts)                          # [0, 1]
    rewards["pickscore"] = pickscore(images, prompts)                    # logits
    rewards["clip_score"] = clip_cosine(images, prompts)                 # [-1, 1]

    # ── z-score each individually (different rewards have very different scales) ──
    normed = {k: (v - v.mean()) / (v.std() + 1e-8) for k, v in rewards.items()}

    # ── weighted + length / safety penalty ──
    R = sum(weights[k] * normed[k] for k in weights)

    # NSFW penalty (rule-based)
    nsfw_score = nsfw_detector(images)                                   # [0, 1]
    R = R - 5.0 * nsfw_score

    return R
```

> ⚠️ **Multi-reward practical experience** —

- **Z-score each reward separately**: the scales differ enormously (HPSv2 ~0.25, ImageReward ~1.5, CLIP-Score ~0.3); not normalizing means ImageReward dominates.
- **Min is more stable than mean**: `R = min(normed.values())` forces all RMs to agree on "good", drastically lowering hacking risk (classic reward-ensemble strategy).
- **Keep rule-based safety overrides**: NSFW / political / copyright rewards must not be optimized away by RL.

## §7 Reward Design & failure modes

### 7.1 Reward model choices

| RM | Source | Data | Scale | Preference characteristics |
| --- | --- | --- | --- | --- |
| **CLIP-Score** | OpenAI/LAION | 4B image-text pairs | $[-1, 1]$ cosine | weak text-image alignment signal; prefers literal caption matching |
| **ImageReward** (Xu 2023 NeurIPS) | 137K human pairs | real prompts | $[-1, 4]$ | aesthetic + alignment combined; prefers high contrast |
| **HPSv2** (Wu 2023 arXiv 2306.09341) | 798K human pairs | DiffusionDB-style | $[0, 1]$ | comprehensive human preference; prefers saturated colors |
| **PickScore** (Kirstain 2023 NeurIPS) | Pick-a-Pic 1M pairs | real users | logits | comprehensive; biased toward trained-on-SDXL style |
| **PiCaR** (rule) | OpenAI | counting/OCR | binary | rule-based, hack-proof |

### 7.2 Reward hacking gallery (diffusion-specific)

| Symptom | Visual feature | Cause |
| --- | --- | --- |
| **Over-saturation** | color saturation > 100% | HPSv2 / aesthetic prefers vivid |
| **Center bias** | subject always centered | RM training data is centered-subject heavy |
| **Monotone composition** | different prompts use the same composition | mode collapse to RM high-score mode |
| **Tryphobia-like patterns** | dense dot / hole textures | some RMs prefer "texture richness" |
| **Watermark hallucination** | fake watermarks appear in corners | RM training data contains watermarks → learned "watermark = real photo" |
| **Cartoon shift** | photo-realistic prompts output anime | RM annotators prefer anime |
| **Lighting overcooked** | HDR post-processing exaggerated | aesthetic predictor prefers heavy post-processing |

### 7.3 Step-level vs trajectory-level reward

| Dimension | Trajectory-level | Step-level |
| --- | --- | --- |
| Signal location | only at $x_0$ | at every $t$ |
| Data acquisition | easy (one image) | hard (need step-wise RM or rollout) |
| Learning efficiency | low (sparse) | high (dense) |
| Representative | DDPO, Diffusion-DPO | SPO (Liang 2024) |
| Engineering difficulty | low | high (either train a step-RM or PRM-shepherd-style rollout) |

> 💡 **How to train a step-RM** — SPO uses a binary RM "given $x_t$ at step $t$, is $x_{t-1}$ a good transition?". Training data: run multiple trajectories from the base UNet, and assign per-step rewards by backing out from final-$x_0$ rewards (similar to Math-Shepherd's rollout-based PRM).

### 7.4 Core mechanisms to mitigate reward hacking

1. **Reward ensemble**: take min or mean across multiple RMs (HPSv2 + PickScore + ImageReward is the mainstream combo).
2. **KL anchor**: DPO's $\beta$, DPOK's explicit KL, Flow-GRPO's K3 KL term.
3. **LoRA scale**: full fine-tune drifts fast; LoRA scale caps reward-hacking ceiling.
4. **Early stop on reward plateau**: reward up + FID up = hacking signal.
5. **Composite reward**: rule-based (object count, OCR) + neural RM (aesthetic, alignment) weighted.
6. **Adversarial RM**: include hacking samples as negatives when training RM.

## §8 Production landscape: what SD3 / FLUX actually use

### 8.1 What the public papers / reports actually say

| Model | Post-training? | Disclosed content |
| --- | --- | --- |
| **SD 1.5** | partial community DPO / DDPO LoRA | base is pure LDM; many community fine-tunes |
| **SDXL** | Stability AI did not explicitly describe post-training | base + refiner; Pick-a-Pic + Diffusion-DPO community LoRAs are popular |
| **SDXL Turbo / ADD** | distillation-focused | Adversarial Diffusion Distillation (2311.17042); essentially 1-step distill, not RL post-training |
| **SD3** (Stable Diffusion 3, Esser et al. 2024 ICML) | base uses Rectified Flow + MM-DiT | paper does not disclose post-training; community suspects internal DPO |
| **SD3.5 / SD3.5 Turbo** | distill is present; post-training not disclosed | speculated DPO + distill hybrid |
| **FLUX.1 dev / pro** (Black Forest Labs 2024) | undisclosed | community speculates DPO + distill; pro API is closed-source |
| **DALL-E 3** (OpenAI 2023) | "synthetic recaptioning" (no RLHF described) | synthetic recaptioning (detailed image-caption data augmentation); does not describe an RLHF post-training procedure |
| **Imagen 3** (Google 2024) | undisclosed | internal alignment pipeline |
| **DeepFloyd IF** | no post-training | academic base model |

### 8.2 Did SD3 / FLUX use post-training?

> ⚠️ **Honest answer** — public papers / tech reports **do not explicitly say** RL / DPO post-training was used. But there is indirect evidence:

- SD3 paper (arXiv 2403.03206) "Improving Rectified Flow Transformers" section discusses sampling + reflow, with no mention of reward fine-tuning.
- FLUX has no paper at all; the community infers distillation from the model card (FLUX schnell is a 4-step distilled version).
- Stability AI mentioned in the SD3.5-Large release that it was "fine-tuned with improved aesthetics", which could be SFT rather than RL.
- DALL-E 3's public technical report (Betker et al. 2023, *Improving Image Generation with Better Captions*) has **synthetic recaptioning** as its core contribution — training an image captioner to produce more detailed synthetic descriptions for the training images, and training on a mix of original short captions and synthetic long captions to improve prompt-following. The report does not describe an RLHF (reward model + PPO, etc.) post-training procedure as its mechanism.

**Industry consensus** (from HuggingFace community + Reddit r/StableDiffusion): closed-source large models (FLUX pro, DALL-E 3, Midjourney v6+) have reward-based fine-tuning but the specifics are undisclosed; open-source bases (SD3.5 base, FLUX dev base) have public training pipelines without RL, but Stability AI's internal dev branches may have it.

### 8.3 An industrial-grade pipeline hypothesis

```
                    ┌─────────────────┐
                    │  LDM Pretrain   │  several 100M / B images, $L_simple$
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  SFT on Curated │  high-quality prompt-image pairs
                    │     dataset     │  (Aesthetic > 6.0, no watermark)
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                                  │
   ┌────────▼────────┐              ┌─────────▼────────┐
   │ Diffusion-DPO   │              │  DRaFT / AlignProp│
   │ on Pick-a-Pic    │              │   on multi-RM    │
   └────────┬────────┘              └─────────┬────────┘
            │                                  │
            └────────────────┬─────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Distillation  │  4-step / 1-step turbo
                    │   (ADD / LCM)   │
                    └────────┬────────┘
                             │
                       Production
```

> 💡 **Conclusion** — post-training most likely happens between "SFT → distill"; in terms of method, industry leans Diffusion-DPO (offline, stable, no sampling needed); DDPO/DPOK have large academic impact but limited production deployment.

## §9 vs LLM RLHF comparison

### 9.1 The full table

| Dimension | LLM RLHF (RLHF + DPO + GRPO) | Diffusion Post-Training |
| --- | --- | --- |
| **Trajectory** | $L$ tokens | $T$ denoising steps |
| **Action space** | discrete vocab | continuous $\mathbb{R}^d$ |
| **Reward source** | trained BT-RM / rule (math, code) | trained image RM (HPSv2/PickScore/ImageReward) / rule (count, OCR) |
| **Reward sparsity** | terminal only (end of response) | terminal only ($x_0$) |
| **On-policy cost** | $L$ forwards | $T$ forwards + image RM forward |
| **Offline methods** | DPO / IPO / KTO / SimPO / ORPO | Diffusion-DPO / D3PO / SPO / KTO / MaPO |
| **On-policy methods** | PPO / GRPO / RLOO | DDPO / DPOK / Flow-GRPO |
| **Direct reward backprop** | ❌ (tokens not differentiable) | ✅ (DRaFT / AlignProp / ReFL) |
| **Memory bottleneck** | 4 copies (policy + ref + RM + V) | 1 copy (DPO) / $O(K)$ (DRaFT-K) / $O(T)$ (vanilla backprop) |
| **Typical $\beta$** | $0.05 \sim 0.5$ | $2000 \sim 5000$ (absorbs $T$-fold coefficient) |
| **Typical trajectory length** | $L \sim 10^3$ tokens | $T \sim 20$–$50$ steps |
| **Mode collapse severity** | medium (vocab is large) | **high** (continuous space easily falls into local modes) |
| **Reward hacking difficulty** | medium (depends on RM quality) | **high** (visual RMs are easier to hack than BT-RMs) |

### 9.2 Shared lessons

1. **KL anchor is mandatory**: pure RL without ref policy always reward-hacks (LLMs produce vacuous fluff, diffusion produces over-saturated monotonous compositions).
2. **DPO family >> on-policy RL (engineering-wise)**: no sampling, no value model, offline — true for both LLM and diffusion.
3. **Reward ensemble counters hacking**: min-of-K RMs is the common mitigation across domains.
4. **Group-based advantage**: LLM's GRPO/RLOO and diffusion's Flow-GRPO both bypass the value model via group statistics.

### 9.3 Diffusion-specific differences

- **Diffusion has a "backprop" option**: reward differentiability makes DRaFT/AlignProp viable — LLMs cannot do this because sampling is discrete.
- **Diffusion has "step-aware" preference**: denoising steps have explicit semantics (high-noise: composition, low-noise: detail); SPO exploits this — LLM tokens lack such natural stratification.
- **Diffusion's $\beta$ scale is 1000× larger**: because the ELBO surrogate absorbs the $T$-fold trajectory term.

## §10 25 frequently-asked interview questions

Three difficulty tiers: L1 = multimodal/diffusion-role basics; L2 = research/alignment-focused; L3 = top-lab hardcore.

### L1 must-know (10 questions)

<details>
<summary>Q1. Why does a diffusion model need post-training? Isn't SFT enough?</summary>

- SFT can only imitate positive examples ("what good looks like"), and cannot learn the **contrastive signal** (A is better than B).
- Post-training provides contrastive signals via reward / preference, lifting alignment, aesthetic, prompt-faithfulness all at once.
- Empirically Diffusion-DPO yields +5-10 points on PickScore, far exceeding continued SFT.

Saying only "improves image quality" is shallow; must articulate the "contrastive vs imitative signal" distinction.
</details>

<details>
<summary>Q2. What MDP does DDPO treat diffusion as? State/action/reward definitions?</summary>

- **State**: $s_t = (x_t, t, c)$
- **Action**: $a_t = x_{t-1}$ (sampled from $p_\theta(\cdot \mid x_t, c)$)
- **Transition**: deterministic $s_{t-1} = (x_{t-1}, t-1, c)$
- **Reward**: $r_t = 0$ for $t > 1$, $r_1 = R(x_0, c)$ (terminal-only)

Saying "per-step reward" (wrong, only at the terminal); or not knowing the transition is deterministic (noise is built into the action).
</details>

<details>
<summary>Q3. What does Diffusion-DPO substitute for $\log \pi_\theta(y|x)$?</summary>

- Uses an ELBO surrogate: $-\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$ (DDPM's $L_\text{simple}$) as a proxy for $\log p_\theta(x_0)$.
- This is a (negative) lower-bound term of $\log p_\theta$ — directionally correct.
- Note that $L_\text{simple}$ sets the true VLB's per-timestep weight $\lambda_t$ ($\omega(\lambda_t)$) to 1 — a deliberate reweighting by Ho 2020 for sample quality, not "the tight ELBO on log p_θ" itself; treating it as a proxy for log p_θ is an approximation, not an equality.
- Pairing the noise $\epsilon$ ($y_w$ and $y_l$ share the same $\epsilon$, a variance-reduction trick) makes training more stable.

Saying the closed-form $\log p(x_0)$ is used (wrong, diffusion has no closed form); treating $L_\text{simple}$ as an exact ELBO identity (ignoring the $\omega(\lambda_t)=1$ reweighting); or forgetting that paired noise reduces variance.
</details>

<details>
<summary>Q4. AlignProp and DRaFT core idea? Why is memory $\mathcal{O}(K)$?</summary>

- **Core**: reward $R(x_0)$ is differentiable → directly backprop to $\theta$, skipping RL.
- The $T$-step sampler is a differentiable computation graph; **vanilla backprop must store $T$-step activations** → $\mathcal{O}(T)$.
- **DRaFT-K / AlignProp**: the first $T-K$ steps use `no_grad`, only the last $K$ steps retain gradient — memory compressed to $\mathcal{O}(K)$.
- Typically $K=1$ already gives a strong signal.

Saying K=1 equals REINFORCE (wrong, K=1 is reparameterized gradient with far lower variance than REINFORCE).
</details>

<details>
<summary>Q5. Why is Diffusion-DPO's $\beta$ 1000× larger than LLM DPO's?</summary>

- LLM DPO: $\beta \in [0.05, 0.5]$
- Diffusion-DPO: $\beta \in [2000, 5000]$
- Reason (directional intuition, not an exact formula): the diffusion "trajectory log-likelihood" is the sum of $T$ Gaussian log-probs; the single-step $\epsilon$-distance difference absorbs the $T$-fold coefficient, which explains the order-of-magnitude direction.
- **But "$\beta T$ is the true temperature" is not a clean, implementation-independent fact**: it also depends on whether the MSE is mean-reduced or sum-reduced (differing by a factor of $C \times H \times W$), and which $T$ is meant — the $\sim$1000-step DDPM training schedule vs. the $\sim$20–50-step inference schedule. This tutorial's code uses `.mean([1,2,3])` plus 1000-step DDPM $t$-sampling; a different reduction/T convention gives a different "effective temperature".

Saying "diffusion has more noise so β is larger" (wrong, it's the cumulative effect of trajectory length); or treating "$\beta T$ is the true temperature" as an exact, universal equivalence (ignoring its dependence on mean/sum reduction and the T convention).
</details>

<details>
<summary>Q6. Difference between ImageReward / HPSv2 / PickScore?</summary>

| | ImageReward | HPSv2 | PickScore |
| --- | --- | --- | --- |
| Data scale | 137K pairs | 798K pairs | 1M pairs (Pick-a-Pic) |
| Backbone | BLIP fine-tuned | CLIP fine-tuned | CLIP fine-tuned |
| Scale | $[-1, 4]$ | $[0, 1]$ | logits |
| Preference | aesthetic + alignment | high contrast + alignment | SDXL style |

Industrial best practice: **use an ensemble** (at least two).

Saying they're all the same (wrong, scales and preferences differ substantially).
</details>

<details>
<summary>Q7. Does DDPO sample with DDPM or DDIM? Why?</summary>

- **DDPM** (or DDIM-eta=1) — stochastic transition is needed so that REINFORCE/score-function policy gradients have a well-defined log-density to differentiate.
- DDIM-eta=0 is deterministic (a Dirac-delta transition) with no well-defined log-density — **REINFORCE-style score-function estimators degenerate/fail here**, but that does not mean "the policy gradient is zero": $x_0$ remains a deterministic, differentiable function of $\theta$, so a pathwise/reparameterized gradient (what §3's DRaFT/AlignProp use) still exists and is generally nonzero. DDPO needs stochastic transitions because it chose REINFORCE as its specific gradient estimator — not because the true gradient vanishes under deterministic sampling.
- Analogy: LLM RL's REINFORCE/PPO must use sampling (temperature > 0), cannot use greedy — but the greedy output is still a differentiable function of the parameters; it's only the score-function gradient path that's unavailable.

Saying DDIM is fine (wrong, for REINFORCE eta > 0 is needed); or thinking the true gradient is zero under deterministic sampling (wrong — conflates "REINFORCE fails" with "the true gradient is zero"; see §3's DRaFT/AlignProp pathwise gradient).
</details>

<details>
<summary>Q8. Typical reward-hacking symptoms in diffusion?</summary>

- **Over-saturation** (color saturation explodes) — HPSv2/aesthetic prefers vivid.
- **Center bias** (subject always centered) — RM training data is centered.
- **Monotone composition** (same composition across prompts) — mode collapse.
- **Watermark hallucination** (fake watermarks in corners) — RM training data contains watermarks.
- **Cartoon shift** (photo-realistic prompts produce anime) — RM annotator bias.

Saying just "over-optimization" is non-specific; you must name at least 3 specific visual symptoms.
</details>

<details>
<summary>Q9. Why is Flow-GRPO's ODE→SDE conversion necessary?</summary>

- The Flow Matching ODE $\dot x = v_\theta$ is **deterministic** — given $x_T$, $x_0$ is uniquely determined.
- RL needs a stochastic policy to explore; the ODE has no sampling dimension.
- ODE→SDE adds the $\sigma\, dW$ noise term; **the marginal $p_t$ is unchanged** (Anderson 1982), but each sample path is different → enables exploration.

Not knowing the marginal is preserved (wrong, may think SDE changes the distribution).
</details>

<details>
<summary>Q10. How are $y_w$ and $y_l$ noises handled in Diffusion-DPO training?</summary>

- **Paired noise**: $x_t^w$ and $x_t^l$ use **the same** $\epsilon$ (i.e. $x_t^w = \sqrt{\bar\alpha_t}x_0^w + \sqrt{1-\bar\alpha_t}\epsilon$, with $x_t^l$ similarly using the same $\epsilon$) — this is a **variance-reduction** technique (common random numbers), not a prerequisite for mathematical correctness.
- Without pairing (independently sampled $\epsilon^w, \epsilon^l$), the loss remains an unbiased estimator of the same DPO objective — just with higher variance and less stable optimization; the scale and comparability of $\beta$ is unaffected.
- This is the implementation detail most often overlooked in Diffusion-DPO practice, but it does materially affect training stability.

Not knowing paired noise reduces variance (missing a stability trick); thinking unpaired noise makes the loss mathematically incorrect (wrong — it's just higher variance); or thinking $\epsilon$ refers to $\epsilon_\theta$'s prediction (wrong — here $\epsilon$ is the q-sample noise).
</details>

### L2 advanced (10 questions)

<details>
<summary>Q11. Derive Diffusion-DPO loss (starting from the KL-regularized optimal solution).</summary>

1. KL-regularized objective: $\max_p \mathbb{E}[R] - \beta\, \text{KL}(p \Vert p_\text{ref})$, optimal $p^* \propto p_\text{ref} \exp(R/\beta)$.
2. Solve for implicit reward: $R(x_0, c) = \beta \log(p^*/p_\text{ref}) + \beta \log Z(c)$.
3. Plug into BT: $P(y_w \succ y_l) = \sigma(R_w - R_l)$; $\log Z$ cancels in the difference.
4. Replace $p^* \to p_\theta$; substitute the ELBO surrogate for $\log p_\theta$: $-L_\text{simple} = -\|\epsilon - \epsilon_\theta\|^2$ (at $x_t = q\text{-sample}(x_0, t, \epsilon)$).
5. Expectation over $t \sim U(1, T)$; the loss becomes $-\log\sigma(\beta T [\Delta_w - \Delta_l])$, $\Delta_y = \|\epsilon - \epsilon_\text{ref}\|^2 - \|\epsilon - \epsilon_\theta\|^2$.

Reciting the final formula but unable to explain why $\log Z$ cancels; or not knowing the source of the ELBO surrogate.
</details>

<details>
<summary>Q12. How does AlignProp's $K$-step backprop trade memory against performance?</summary>

- Memory: $\mathcal{O}(K \cdot M_\text{UNet})$. SDXL single forward ~ 8 GB activation; $K=1 \to 24$ GB (incl. weights + grad); $K=5 \to 60$ GB; $K=10 \to 120$ GB.
- Performance: $K=1$ empirically reaches 95% of $K=5$; $K \ge 5$ shows no significant gain on most rewards.
- **Intuition**: the last step $x_1 \to x_0$ has the largest effect on the terminal; the variance from the first 49 steps is compressed.
- Industrial standard is $K=1$ (24 GB, single-GPU trainable).

Saying "the larger K the better" (wrong, performance curve saturates); or not knowing the memory magnitude.
</details>

<details>
<summary>Q13. Difference between REINFORCE and PPO in DDPO? Which is preferred in production?</summary>

- **DDPO-SF (REINFORCE)**: $\hat g = \sum_t \nabla \log p_\theta \cdot (R - b)$, simple but high variance.
- **DDPO-IS (PPO-clip)**: importance ratio $\rho_t = p_\theta/p_{\theta_\text{old}}$, multiple updates per batch, clip $\rho_t$.
- **Per-step ratio**, not trajectory ratio (avoids variance blow-up from multiplying $T$ ratios).
- Production prefers PPO: more stable, better sample efficiency.

Saying "trajectory-level ratio" (wrong, per-step); or not knowing both are DDPO variants.
</details>

<details>
<summary>Q14. How does SPO obtain the in-step preference pair? Why is a step-RM needed?</summary>

- **In-step**: given $x_t$, independently sample two $x_{t-1}^a, x_{t-1}^b$ (use the policy's stochastic transition $p_\theta(\cdot \mid x_t)$ twice).
- Use a **step-wise reward model** $R_\text{step}(x_{t-1}, x_t, c, t)$ to choose the winner.
- step-RM training data: roll out many trajectories from the base UNet; the per-step reward is back-derived from the terminal reward (similar to Math-Shepherd's rollout-based PRM).
- Could use a terminal RM instead, but then rolling out to $x_0$ for scoring is $T$× more expensive.

Not knowing how to obtain the in-step pair (wrong, must sample twice); or not knowing step-RM is SPO-specific.
</details>

<details>
<summary>Q15. Substantive difference between Diffusion-DPO and D3PO?</summary>

- **Derivation path**: Diffusion-DPO uses the ELBO surrogate (single-step $\epsilon$-distance); D3PO uses the full trajectory log-ratio.
- **Mathematical equivalence**: under the ELBO lower bound and expectation over $t$, D3PO's trajectory form reduces to Diffusion-DPO's single-step form.
- **Practical differences**:
  - Diffusion-DPO computes UNet forward once per step (policy + ref), cheap.
  - D3PO strictly requires forward over the full trajectory $T$ times.
- Production mostly uses Diffusion-DPO (cheap + stable).

Saying "completely different" (wrong, they are theoretically equivalent); or not knowing D3PO is also a DPO-family method.
</details>

<details>
<summary>Q16. What is Flow-GRPO's denoising reduction? Why doesn't it degrade?</summary>

- During training, the SDE uses few steps ($T_\text{train} = 10$); inference still uses full steps ($T_\text{infer} = 28$–$50$).
- **Empirically little degradation, because** (note: this is empirical observation + approximation, not strict equivalence):
  - The SDE's **continuous marginal** $p_t$ is independent of step count; but **the discrete sampler's actual distribution** depends on step count — fewer steps yields a higher-discretization-error approximation. Strictly, "same marginal" only holds in the continuous limit.
  - RL learns the direction correction of $v_\theta$; the **direction signal** is loosely coupled to step count (this is empirical observation).
  - No degradation on rule-based rewards like GenEval/OCR; slight but acceptable degradation on more subjective rewards.
- **Saves sampling cost**: each prompt needs $G \cdot T_\text{train}$ forwards → 1/3 the cost.

Not knowing the marginal is invariant (wrong); or thinking train/inference step counts must match.
</details>

<details>
<summary>Q17. How does MaPO remove the reference model? What does the loss look like?</summary>

- Does not use $\log(p_\theta/p_\text{ref})$; uses an **absolute likelihood margin** directly:
$$\mathcal{L}_\text{MaPO} = -\log\sigma\!\big(\beta(\hat\ell_w - \hat\ell_l) - \gamma\big) + \alpha \hat\ell_w$$
- $\hat\ell = -\|\epsilon - \epsilon_\theta\|^2$ is the likelihood surrogate.
- $\gamma$ is the margin (similar to SimPO); the $\alpha\hat\ell_w$ term prevents "both sides dropping".
- Saves half the memory (no ref UNet), trains 15% faster, and solves the reference-mismatch problem (stable when fine-tuning to a stylistically distant target).

Saying just "remove the ref" (wrong — need the likelihood term to prevent degeneration); not knowing about reference mismatch.
</details>

<details>
<summary>Q18. Why is reward-ensemble min better than mean?</summary>

- mean: can still be hacked when one high-scoring RM dominates.
- min: requires all RMs to agree on "good" before the reward is high → hacking must simultaneously fool all RMs, exponentially harder.
- Equivalent to conservative aggregation (Coste 2024 ICLR for LLM); same idea applies to diffusion.
- Cost: reward becomes conservative, gains are smaller.
- Industry often uses `R = mean - k * std` (includes an uncertainty penalty) as a compromise.

Saying only "prevents hacking" without explaining why min; or not knowing this is also an LLM ensemble strategy.
</details>

<details>
<summary>Q19. What unique advantage does Diffusion-KTO have over Diffusion-DPO?</summary>

- Only needs **per-image binary feedback** (thumbs up/down), **no paired comparison required**.
- Industrial scenarios collect far more user reactions (likes/dislikes) than paired comparisons → KTO makes this data usable.
- The prospect-theoretic value function $v(\cdot)$ is asymmetric over positive vs negative feedback (loss aversion).
- No "which is better" annotation cost.

Not knowing KTO is unpaired (wrong — that's the whole point of KTO); or not knowing the prospect-theory origin.
</details>

<details>
<summary>Q20. Why is diffusion's KL anchor trajectory-level rather than token-level?</summary>

- LLM's KL is per-token: $\sum_t \log(\pi_\theta(y_t)/\pi_\text{ref}(y_t))$.
- Diffusion's KL is per-step (per-denoising-step), not per-pixel: $\sum_t \text{KL}(p_\theta(\cdot \mid x_t) \Vert p_\text{ref}(\cdot \mid x_t))$.
- Two-Gaussian KL has a closed form: $\text{KL} = \frac{1}{2}\big[(\mu_\theta - \mu_\text{ref})^2/\sigma^2 + (\sigma_\theta/\sigma_\text{ref})^2 - 1 - 2\log(\sigma_\theta/\sigma_\text{ref})\big]$.
- Pixels are not independent (convolution/attention), so the KL is per-image-level rather than per-pixel.

Saying "per-pixel KL" (wrong, per-step); not knowing the closed-form Gaussian KL.
</details>

### L3 top-lab questions (5 questions)

<details>
<summary>Q21. Derive Diffusion-DPO loss from the reverse ELBO, and explain why the ELBO surrogate works.</summary>

1. DDPM ELBO: $\log p_\theta(x_0) \ge -\sum_{t=2}^T \text{KL}(q(x_{t-1}|x_t,x_0) \Vert p_\theta(x_{t-1}|x_t)) + \log p_\theta(x_0|x_1) - \text{KL}(q(x_T|x_0) \Vert p(x_T))$
2. Simplify (Ho 2020): $-\log p_\theta(x_0) \le L_\text{simple} + C$, $L_\text{simple} = \mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ — **note**: the true VLB has an SNR-dependent weight $\lambda_t$ (i.e. $\omega(\lambda_t)$) in front of each $t$; $L_\text{simple}$ is the reweighted version with $\omega(\lambda_t)$ set to 1 (a deliberate choice by Ho 2020 for sample quality), not a tight bound obtained by directly simplifying the VLB.
3. KL-regularized optimum $p^* \propto p_\text{ref}\exp(R/\beta)$; solve for $R = \beta\log(p^*/p_\text{ref}) + \beta\log Z$.
4. Plug into BT; $\log Z$ cancels.
5. Use the ELBO surrogate for $\log p$: $\log p_\theta(x_0) \approx -L_\text{simple}$ (**note**: this is the negation of the upper bound, used as a single-sample estimate — strictly, it is one term of the lower bound, not $\log p$ itself, but it works numerically as a DPO implicit-reward proxy).
6. Take expectation over $t$ to obtain the final loss.

**Deeper reason the ELBO surrogate works**: DPO's implicit reward is $\beta\log(p_\theta/p_\text{ref})$ — it only depends on **relative** likelihood. The ELBO surrogate's constant term ($C$) cancels between $p_\theta$ and $p_\text{ref}$ (both use the same architecture); only the $-\|\epsilon - \epsilon_\theta\|^2$ difference remains. So even though ELBO isn't a tight bound for $\log p$, **the difference is cancellable**.

Only able to write the final formula but cannot trace the derivation chain; or not knowing constant cancellation is the key.
</details>

<details>
<summary>Q22. Is AlignProp's $\mathcal{O}(K)$ memory truly unavoidable for $K$-step backprop?</summary>

**In theory** it can be bypassed, but it is very expensive in practice:

1. **Gradient checkpointing**: trade activation storage for recompute. Each forward step does not store activations; on backward, redo forward to compute gradient.
   - Memory: from $\mathcal{O}(K \cdot M)$ down to $\mathcal{O}(\sqrt{K} \cdot M)$ + $\mathcal{O}(K \cdot \text{state})$.
   - Cost: backward is 2-3× slower.
2. **Reversible ResNet**: if the UNet has a reversible architecture (i-RevNet style), backward can derive input from output without storing activations.
   - But Stable Diffusion / SDXL UNet is not reversible.
3. **Implicit gradient**: use the implicit function theorem under a fixed-point assumption.
   - Requires the sampler to converge to a fixed point, which diffusion does not satisfy.
4. **Truncated backprop with control variates**: DRaFT-K is already in this direction; theoretically, adding control variates can further reduce variance but not memory.

**Practical answer**: $K=1$ + gradient checkpointing + LoRA is the engineering optimum. The fundamental reason $\mathcal{O}(K)$ is unavoidable is that the sampler is not reversible computation.

Only saying gradient checkpointing is incomplete; not knowing about the reversibility assumption.
</details>

<details>
<summary>Q23. Geometric meaning of advantage on the vector field $v_\theta$ in Flow-GRPO?</summary>

GRPO's advantage acts on the vector field space as follows:

1. **Group statistics**: for the same prompt $c$, sample $G$ SDE trajectories; each yields a different $x_0^{(i)}$; the reward $r_i$ gives the entire trajectory a single advantage $\hat A_i = (r_i - \bar r)/\sigma_r$.
2. **Gradient along trajectory**: $\nabla_\theta L = \sum_t \nabla_\theta \log p_\theta(x_{t-1}^{(i)} \mid x_t^{(i)}) \cdot \hat A_i$. Under a Gaussian transition, $\log p \propto -(x_{t-1} - \mu_\theta)^2/(2\sigma^2)$, so $\nabla_\theta \log p \propto (x_{t-1} - \mu_\theta)\nabla_\theta \mu_\theta / \sigma^2$.
3. **Physical meaning of $\mu_\theta$**: under the Flow Matching SDE, $\mu_\theta = x_t + (v_\theta + \frac{1}{2}\sigma^2 s_\theta) dt$; $\nabla_\theta \mu_\theta \approx dt \cdot \nabla_\theta v_\theta$ (ignoring the score term).
4. **Geometric meaning**: when $\hat A_i > 0$, push $v_\theta(t, x_t^{(i)})$ toward the direction $(x_{t-1}^{(i)} - x_t^{(i)})/dt$ (the direction the trajectory actually took); when advantage $< 0$, push the opposite way.
5. **vs ODE perspective**: equivalent to "group-relative direction reweighting" in vector field space — good trajectories make $v_\theta$ point along their direction at that $(t, x_t)$, bad ones the opposite.

This is "reward-weighted importance sampling" in vector field space: each SDE trajectory is one "proposal direction" for $v_\theta$, and the advantage decides whether to follow it.

Failing to articulate any geometry is 0 points; saying "reweighting" without specifying the space is half credit.
</details>

<details>
<summary>Q24. Did SD3 / FLUX really use RL post-training? How can you tell?</summary>

**Honest answer**: public papers / tech reports **do not explicitly say** RL / DPO was used. But there are clues:

1. **SD3 paper (arXiv 2403.03206)**: discusses Rectified Flow + MM-DiT + reflow only; no reward fine-tuning mentioned.
2. **FLUX**: no paper at all; the model card just says "trained on a large image-text dataset".
3. **DALL-E 3 (OpenAI 2023)**: the public report (Betker et al. 2023) is centered on **synthetic recaptioning** — using an image captioner to produce more detailed synthetic descriptions for training images and mixing them into training to improve prompt-following; it does not describe an RLHF (reward model + PPO, etc.) post-training mechanism.
4. **Industry consensus**: closed-source large models (FLUX pro, DALL-E 3, Midjourney v6+) almost certainly have reward-based fine-tuning, but the method is undisclosed.

**How to test (black-box)**:
- Generate 100 images for the same prompt; low FID-100 / multi-mode diversity → likely RL/DPO (mode collapse signal).
- GenEval high but portrait style monolithic → reward over-optimization signal.
- Excessive response to "vibrant"/"colorful" prompts → traces of HPSv2/aesthetic RM.

**Conclusion**: FLUX likely has internal DPO + distill hybrid; SD3.5 likely has SFT + possibly DPO. But **no public evidence** — the key to this question is to answer "undisclosed but with indirect evidence", and avoid making up technical details.

Saying directly "SD3 uses Diffusion-DPO" is wrong (paper says no such thing); the answer should be "undisclosed but community-inferred with evidence list".
</details>

<details>
<summary>Q25. If you were to design a diffusion post-training pipeline from scratch, how would you choose?</summary>

**Depends on constraints**. A generic recommendation:

**Phase 1: preference data collection**
- Collect paired preference (Pick-a-Pic style): expensive but DPO-ready.
- Collect binary feedback (thumbs up/down): cheap, use Diffusion-KTO.
- Collect rule-based ground truth (GenEval-style prompts + automated verifier): cheap, use Flow-GRPO.

**Phase 2: algorithm choice**
- **Default Diffusion-DPO**: offline, stable, cheap, mature community code (HuggingFace `diffusers` supports it directly).
- **If the base is Flow Matching (SD3/FLUX)**: use Flow-GRPO, with rule-based rewards prioritized.
- **If fine-tuning to a new style / memory constrained**: use MaPO (no ref, saves half the memory).
- **If reward is differentiable and you want to extract maximum signal**: DRaFT-1 + LoRA, paired with HPSv2 + PickScore ensemble.
- **NOT preferred: DDPO**: on-policy sampling is too expensive, engineering complexity is high, no clear advantage over DPO.

**Phase 3: reward design**
- **Multi-RM ensemble** (min or mean - k·std): HPSv2 + PickScore + ImageReward.
- **Add rule-based safety**: NSFW detector hard penalty.
- **Add rule-based alignment**: GenEval automated verifier (object count, OCR).
- **Z-score each RM separately**.

**Phase 4: monitoring and early stop**
- Compute reward + FID-100k every N steps; reward up + FID up = hacking signal.
- KL budget monitoring: stop if $\text{KL}(p_\theta \Vert p_\text{ref}) > K_\text{target}$.
- Human blind A/B (base vs RL) every 1000 steps.

**Phase 5: distillation handoff**
- After post-training, do ADD / LCM distillation to 4-step / 1-step.
- Note: distillation can eliminate some RL gains; consider distill-aware post-training.

Just answering "use Diffusion-DPO" is shallow; the answer should articulate "phase decomposition + multi-reward + monitoring + distill handoff".
</details>

## §A Appendix

### A.1 Key papers (with arXiv IDs)

| Paper | One-line summary | arXiv | Venue |
| --- | --- | --- | --- |
| **DDPO** | treat diffusion as an MDP, train with REINFORCE/PPO | [2305.13301](https://arxiv.org/abs/2305.13301) | ICLR 2024 |
| **DPOK** | KL-regularized RL for diffusion | [2305.16381](https://arxiv.org/abs/2305.16381) | NeurIPS 2023 |
| **DRaFT** | direct reward backprop, last $K$ steps | [2309.17400](https://arxiv.org/abs/2309.17400) | ICLR 2024 |
| **AlignProp** | reward backprop with randomized truncation | [2310.03739](https://arxiv.org/abs/2310.03739) | ICLR 2024 |
| **ImageReward / ReFL** | 137K human pair RM + single-step reward fine-tune | [2304.05977](https://arxiv.org/abs/2304.05977) | NeurIPS 2023 |
| **HPSv2** | 798K human pair RM | [2306.09341](https://arxiv.org/abs/2306.09341) | arXiv 2023 |
| **PickScore (Pick-a-Pic)** | 1M user pairs, CLIP RM | [2305.01569](https://arxiv.org/abs/2305.01569) | NeurIPS 2023 |
| **Diffusion-DPO** | ELBO surrogate + DPO loss | [2311.12908](https://arxiv.org/abs/2311.12908) | CVPR 2024 |
| **D3PO** | trajectory-level DPO for diffusion | [2311.13231](https://arxiv.org/abs/2311.13231) | CVPR 2024 |
| **SPO** | step-aware preference + step-RM | [2406.04314](https://arxiv.org/abs/2406.04314) | arXiv 2024 |
| **Diffusion-KTO** | unpaired binary feedback (KTO for diffusion) | [2404.04465](https://arxiv.org/abs/2404.04465) | NeurIPS 2024 |
| **MaPO** | margin-aware, no ref | [2406.06424](https://arxiv.org/abs/2406.06424) | arXiv 2024 |
| **Flow-GRPO** | GRPO for Flow Matching via ODE→SDE | [2505.05470](https://arxiv.org/abs/2505.05470) | arXiv 2025 |
| **SD3 (Rectified Flow + MM-DiT)** | base model | [2403.03206](https://arxiv.org/abs/2403.03206) | ICML 2024 |
| **Constitutional AI (origin of RLAIF)** | AI feedback replaces human | [2212.08073](https://arxiv.org/abs/2212.08073) | arXiv 2022 |
| **KTO (LLM)** | prospect theory alignment | [2402.01306](https://arxiv.org/abs/2402.01306) | arXiv 2024 |

### A.2 Common reward model resources

- **ImageReward**: https://github.com/THUDM/ImageReward
- **HPSv2**: https://github.com/tgxs002/HPSv2
- **PickScore**: https://github.com/yuvalkirstain/PickScore
- **CLIP**: OpenAI / OpenCLIP, multiple backbones available

### A.3 Open-source training code

- **TRL (HuggingFace)**: `diffusers` + `DPO Trainer` for Diffusion-DPO (most mature)
- **DDPO original repo**: https://github.com/kvablack/ddpo-pytorch
- **AlignProp**: https://github.com/mihirp1998/AlignProp
- **DRaFT (Google research)**: https://github.com/clarkjkr/draft (Clark et al. 2024 ICLR)
- **MaPO**: https://github.com/mapo-t2i/mapo
- **Flow-GRPO**: see paper arXiv 2505.05470 for the official implementation

### A.4 Engineering footgun checklist

| Footgun | Fix |
| --- | --- |
| Diffusion-DPO without paired noise | recommended: share $\epsilon$ for $x_t^w$ and $x_t^l$ (reduces variance, not mathematically required) |
| DDPO with DDIM-eta=0 | REINFORCE-style score-function gradient degenerates; use eta>0 or DDPM (pathwise methods like DRaFT/AlignProp instead require deterministic sampling) |
| AlignProp memory explosion | $K=1$ + gradient checkpoint + LoRA |
| Reward scales not normalized | z-score each RM separately |
| FID crashes after RL | add KL anchor or reward ensemble |
| $\beta$ won't move | Diffusion-DPO uses $\beta \in [2000, 5000]$, not the LLM 0.1 |
| Flow-GRPO trains slowly | use denoising reduction ($T_\text{train} < T_\text{infer}$) |
| MaPO blows up | the $\alpha\hat\ell_w$ term must be large enough to prevent likelihood collapse |
| Step-RM won't train | use rollout-based auto-labeling (Math-Shepherd-style) |
| Reward hacking undetected | simultaneously monitor reward + FID + human blind A/B |

### A.5 Mapping back to §0 TL;DR

| TL;DR point | See section |
| --- | --- |
| 1. Why it's hard | §1 |
| 2. Three main lines | §1.2 |
| 3. DDPO | §2.1–2.4 |
| 4. DRaFT / AlignProp | §3.2–3.3 |
| 5. Diffusion-DPO | §4.1 |
| 6. D3PO | §4.2 |
| 7. SPO | §4.3 |
| 8. Flow-GRPO | §5 |
| 9. Reward hacking | §3.6 + §7 |

> ✅ **Learning checkpoint** —

- Can verbally describe the Diffusion-DPO loss form + paired-noise detail
- Can explain why AlignProp's $K=1$ is enough + memory $\mathcal{O}(K)$
- Can write DDPO's state/action/reward + per-step ratio
- Can explain why Flow-GRPO's ODE→SDE conversion is necessary + denoising reduction
- Know the honest answer about whether SD3/FLUX used RL (publicly undisclosed)
