## §0 TL;DR Cheat Sheet

> 💡 **Diffusion / Flow Distillation in 9 sentences** — compress a 50-1000 NFE teacher into a 1-4 NFE student. One-page interview essentials (full derivations in §1-§9).

1. **Why**: diffusion sampling defaults to 50-1000 NFE, **with network forward passes accounting for >95% of total latency**; the production-deployment threshold is typically ≤ 4 steps (real-time chat / mobile / video generation). This tutorial covers only **few-step / one-step distillation**, not RL post-training.

2. **Trade-off**: fewer steps usually hurt quality — naive uniform-skip DDIM at 4 steps is barely usable. The essence of distillation is **using the teacher's 50-step trajectory/distribution as supervision** to train a student that jumps in one step.

3. **Three major technical routes**: (a) **trajectory matching** (progressive distillation / CM / iCT / sCM / CTM / LCM / TCD) — make the student reproduce the teacher's ODE solution; (b) **distribution matching** (DMD / DMD2 / rCM) — use the score gap as the KL gradient and match two distributions; (c) **adversarial** (ADD / LADD / SDXL-Lightning / FLUX-schnell) — GAN loss + teacher distillation.

4. **Consistency Models** (Song 2023 ICML): learn a consistency function $f_\theta(x_t, t) \to x_0$, **mapping any $x_t$ to the same $x_0$**; the boundary $f_\theta(x_{\sigma_\min}, \sigma_\min) = x_{\sigma_\min}$ is enforced by EDM-style preconditioning; CD (distillation, with teacher) / CT (training, no teacher).

5. **iCT** (Song-Dhariwal 2023): **remove EMA target** + **pseudo-Huber loss** (replacing LPIPS) + lognormal noise schedule + step-count curriculum, bringing CT close to CD quality.

6. **sCM / TrigFlow** (Lu-Song 2024 OpenAI): continuous-time CM with **$x_t = \cos(t) x_0 + \sin(t) z$** (trig parametrization makes EDM precond + PF-ODE + CM same-form); 1.5B ImageNet 512 2-step FID 1.88, **within 10% of the best diffusion**.

7. **DMD** (Yin 2024 CVPR): treat student output as a "fake distribution", then use the **fake score** $s_\text{fake}$ minus the **real score** $s_\text{real}$ as the reverse-KL gradient that pushes the student: $\nabla_\theta \text{KL}(p_\text{fake} \| p_\text{real}) = \mathbb{E}[(s_\text{fake} - s_\text{real}) \cdot \partial G_\theta / \partial \theta]$. **DMD2** (Yin 2024 NeurIPS) drops the regression loss, adds GAN, and supports multi-step students.

8. **ADD / LADD** (Sauer et al. 2023/2024 Stability): teacher score distillation + **DINOv2 / VAE feature discriminator** dual supervision. **SDXL-Turbo** 1-step 1024, **SD3-Turbo** 4-step; **FLUX.1-schnell** also in the LADD family.

9. **LCM-LoRA** (Luo 2023): package Latent CM training as a **LoRA adapter**; ~30 A100·h is enough to let any SD 1.5 / SDXL fine-tune generate in 4 steps **without swapping the base model**. The key enabler for production ecosystem.

## §1 Intuition & why distillation is needed

### 1.1　Sampling cost is diffusion's Achilles' heel

| Model / sampler | Typical NFE | 1024² image latency (A100 fp16) |
|---|---|---|
| DDPM ancestral (1000 step) | 1000 | ~90 s |
| DDIM (50 step) | 50 | ~5 s |
| DPM-Solver++ (20 step) | 20 | ~2 s |
| **EDM Heun (35 step)** | ~35 | ~3.5 s |
| **LCM (4 step)** | 4 | ~0.4 s |
| **SDXL-Turbo (1 step)** | 1 | ~0.1 s |
| **DMD2 / FLUX-schnell (1-4 step)** | 1-4 | 0.1-0.4 s |

**Production typically requires** < 0.5 s (real-time chat) or < 1 s (mobile), and native diffusion is far over budget. Distillation is not an "optional optimization" — it's **a necessary step to deploy diffusion**.

### 1.2　Why naive few-step doesn't work

Change 50-step DDIM to 4-step uniform DDIM: the sampler's $\Delta t$ grows large, and **first-order Euler error $O(\Delta t)$ blows up sharply**, with high-frequency details collapsing and visible noise residue. Even with EDM Heun 2nd-order, 4-step typically has FID > 15 and is far from usable. **Root cause**: the teacher's ODE trajectory is curved (VP/VE path), and 4 steps can only approximate it with a coarse polyline.

> 💡 **Core distillation idea** — not changing the sampler, not lowering precision — but **retraining a student** that learns to "jump directly from any $x_t$ to $x_0$" (CM view) or "match the teacher's output distribution" (DMD view) or "produce images that fool the discriminator" (ADD view). The three views correspond to the three major schools.

### 1.3　Distillation vs accelerated samplers: fundamental difference

| | Accelerated samplers (DDIM / DPM-Solver / Heun) | Distillation (CM / DMD / ADD) |
|---|---|---|
| Change training? | ❌ | ✅ requires a new round of training |
| Change network? | ❌ (same $\epsilon_\theta$) | ✅ separate student network (or fine-tune from teacher) |
| NFE limit | 10-20 (ODE solving accuracy limit) | 1-4 |
| Failure modes | discretization error | mode collapse / saturated colors / lack of diversity |

**Complementary relationship**: production pipelines typically follow **"first pick the sampler type → then distill"** — e.g., SD3 uses RF (Euler-friendly) + LADD to distill to 4 steps; FLUX uses RF + LADD-schnell to distill to 1-4 steps.

### 1.4　Convention used throughout

| Symbol | Meaning |
|---|---|
| $x_0$ | clean data |
| $x_t$ ($t \in [0, T]$ or $\sigma \in [\sigma_\min, \sigma_\max]$) | noised sample |
| $z, \epsilon$ | $\mathcal{N}(0, I)$ noise |
| $\theta$ / $\phi$ | student params / teacher params |
| $f_\theta(x_t, t)$ | CM consistency function (CM output) |
| $G_\theta(z, t)$ | one-step / few-step student generator |
| $s_\theta(x, t) \approx \nabla \log p_t(x)$ | score |
| $D_\psi$ | discriminator (used by ADD/LADD) |
| NFE | Number of Function Evaluations |

> ⚠️ **Time-direction pitfall (disambiguate up front)** — the CM series uses EDM $\sigma$-time ($\sigma_\min = 0.002$, $\sigma_\max = 80$), DDPM uses $t \in [0, T]$, FM uses $t \in [0, 1]$. This tutorial standardizes per section: §2 CM/iCT/sCM use $\sigma$-time; §3 DMD uses $t \in [0, T]$; §4 ADD/LADD use $\sigma$-time (following EDM); §5 Flow family uses $t \in [0, 1]$ ($t=0$ noise / $t=1$ data).

## §2 Consistency Models family

### 2.1　Consistency Models (CM, Song et al. 2023 ICML, arXiv:2303.01469)

**Core definition**: consistency function $f: (x_t, t) \mapsto x_{\sigma_\min}$ is **self-consistent** along PF-ODE trajectories —

$$\boxed{\;f_\theta(x_t, t) = f_\theta(x_{t'}, t')\quad\text{for any } t, t' \in [\sigma_\min, \sigma_\max] \text{ on the same ODE trajectory}\;}$$

This yields one-step generation: $x_0 \approx f_\theta(z \cdot \sigma_\max, \sigma_\max)$ with $z \sim \mathcal{N}(0, I)$.

**Boundary condition**: requires $f_\theta(x, \sigma_\min) = x$ (identity at lowest noise) — enforced via EDM-style preconditioning:

$$f_\theta(x, \sigma) = c_\text{skip}(\sigma)\, x + c_\text{out}(\sigma)\, F_\theta(x, \sigma)$$

with $c_\text{skip}(\sigma_\min) = 1$, $c_\text{out}(\sigma_\min) = 0$. Song 2023's specific values (same form as EDM Karras):

$$c_\text{skip}(\sigma) = \frac{\sigma_\text{data}^2}{(\sigma - \sigma_\min)^2 + \sigma_\text{data}^2},\quad c_\text{out}(\sigma) = \frac{\sigma_\text{data}\,(\sigma - \sigma_\min)}{\sqrt{\sigma_\text{data}^2 + \sigma^2}}$$

**Consistency Loss (core)**: take adjacent noise levels $t_n < t_{n+1}$, and require the student's outputs at $(x_{t_n}, t_n)$ and $(x_{t_{n+1}}, t_{n+1})$ to agree —

$$\boxed{\;\mathcal{L}_\text{CD}(\theta) = \mathbb{E}\left[\lambda(t_n)\, d\!\Big(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat x_{t_n}, t_n)\Big)\right]\;}$$

- $\theta^-$: EMA target (like BYOL, prevents representation collapse)
- $\hat x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) \cdot v_\phi(x_{t_{n+1}}, t_{n+1})$: teacher one-step ODE reverse
- $d$: L2 or LPIPS (CM paper uses LPIPS for ImageNet 64)
- $\lambda(t_n)$: weight, CM paper takes 1

> 💡 **Key difference CD vs CT** — CD (Consistency **Distillation**) uses a pretrained teacher $v_\phi$ to compute $\hat x_{t_n}$; CT (Consistency **Training**) has no teacher and uses $\hat x_{t_n} = x_0 + t_n \epsilon$ (the same noise sample with different noise levels). CD with LPIPS + EMA reaches FID 3.55 on CIFAR-10; CT only reaches 8.7 — until iCT caught up.

### 2.2　Derive the consistency loss from PF-ODE (mandatory derivation)

Consider PF-ODE $\frac{dx}{dt} = v_\phi(x_t, t)$ (teacher). The consistency definition requires self-consistency along trajectories $f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t)$. First-order Taylor expansion (along the trajectory):

$$f_\theta(x_{t+\Delta t}, t+\Delta t) \approx f_\theta(x_t, t) + \Delta t \cdot \frac{d f_\theta}{dt}$$

where $\frac{d f_\theta}{dt} = \partial_t f_\theta + \partial_x f_\theta \cdot v_\phi$. So the **continuous-time consistency loss**:

$$\mathcal{L}_\text{cont} = \mathbb{E}\left\|\frac{d f_\theta}{dt}\right\|^2 = \mathbb{E}\left\|\partial_t f_\theta + (\partial_x f_\theta)\, v_\phi(x_t, t)\right\|^2$$

**Discretization**: use $f_{\theta^-}$ (EMA) as a stop-gradient anchor, $\hat x_{t_n}$ from one teacher ODE step:

$$\mathcal{L}_\text{CD} \approx \mathbb{E}\|f_\theta(x_{t_{n+1}}, t_{n+1}) - f_{\theta^-}(\hat x_{t_n}, t_n)\|^2$$

> ⚠️ **Can't drop the EMA anchor** — if both sides use $\theta$, the loss degenerates to $\|f_\theta - f_\theta\| = 0$ and **the network gets no signal**. EMA provides a "past self" as supervision, similar to BYOL's collapse-prevention mechanism. The iCT paper (§2.3) proves you can **remove EMA** under suitable noise schedule + pseudo-Huber loss — this is one of iCT's core contributions.

### 2.3　iCT / Improved Techniques (Song-Dhariwal 2023, arXiv:2310.14189)

CT (Consistency Training) originally had quality far below CD. iCT improves four things:

| Change | original CT | iCT |
|---|---|---|
| Target | EMA $\theta^- = \tau \theta^- + (1-\tau) \theta$ | **direct stop-grad** (no EMA) |
| Loss | LPIPS | **Pseudo-Huber** $d(a, b) = \sqrt{\lVert a-b \rVert^2 + c^2} - c$ |
| Noise sched | uniform discrete $\sigma_n$ | **Lognormal**: $\log \sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$ |
| Step count | fixed $N$ | **Curriculum**: $N(k) = \lceil N_\min \cdot (N_\max/N_\min)^{k/K} \rceil$ |

**Motivation for pseudo-Huber design**:

- LPIPS introduces a **bias** toward ImageNet pretrained features — at eval FID looks good but there's actual distribution shift
- L2 is sensitive to outliers and unstable to train
- Pseudo-Huber $\sqrt{\|a-b\|^2 + c^2} - c$: for small residuals ≈ $\|a-b\|^2/(2c)$ (L2), for large residuals ≈ $\|a-b\|$ (L1) — **adaptively robust**

**Results**: iCT achieves **1-step FID 2.83 / 2-step FID 2.46** on CIFAR-10 (paper abstract numbers; the deeper iCT-deep reaches 2.51 / 2.24), and **doesn't depend on a teacher** — fully opening the ceiling for from-scratch consistency training.

### 2.4　sCM / TrigFlow (Lu-Song 2024 OpenAI, arXiv:2410.11081)

**Problem**: discrete-time CM has two big ailments — (i) discretization error (larger $N$ is more accurate but slower) and (ii) various hyperparameters (noise schedule / EMA decay / loss curriculum) that are finicky to tune.

**TrigFlow parametrization**: write the forward path in trig form —

$$\boxed{\;x_t = \cos(t)\, x_0 + \sin(t)\, z,\quad t \in [0, \pi/2],\; z \sim \mathcal{N}(0, I)\;}$$

Boundaries: at $t = 0$, $x_t = x_0$ (data); at $t = \pi/2$, $x_t = z$ (standard Gaussian).

**Why trig form?** It is the unique parametrization that **simultaneously simplifies** the following four things (Lu-Song 2024 Theorem 1):

- EDM precond: $D_\theta(x_t, t) = \cos(t)\, x_t - \sin(t)\, F_\theta$, automatically satisfies the boundary
- PF-ODE: $\frac{dx_t}{dt} = -\sin(t) x_0 + \cos(t) z$, clean expression
- CM output: $f_\theta(x_t, t) = \cos(t) x_t - \sin(t) (\sigma_d F_\theta(x_t / \sigma_d, c_\text{noise}(t)))$ ($\sigma_d$ is data std)
- Continuous-time consistency loss: gradient can be written closed-form directly

**Continuous-Time Consistency Loss (sCM core)**: sCM rewrites the continuous-time CM gradient as a **stop-gradient MSE surrogate** (rather than simplifying the target to $r\cdot\mathrm{JVP}$ — that would become a zero-signal self-reference during warmup at $r=0$). The correct form:

$$\mathcal{L}_\text{sCM}(\theta, \phi) = \mathbb{E}_{x, t}\!\left[\frac{e^{w_\phi(t)}}{D}\Big\|F_\theta(x_t/\sigma_d, t) - \operatorname{sg}\!\big(F_{\theta^-}(x_t/\sigma_d, t) + g_{\theta^-}(x_t, t)\big)\Big\|_2^2 - w_\phi(t)\right]$$

where $F_{\theta^-}$ is the EMA / stop-grad copy. **TrigFlow consistency function**: $f_\theta(x_t, t) = \cos t\, x_t - \sin t\, \sigma_d F_\theta(x_t/\sigma_d, t)$. Let $\hat v_t = dx_t/dt$ (in sCT $= \cos t\, z - \sin t\, x_0$; in sCD given by the teacher PF-ODE); the **JVP-rearranged tangent target**:

$$g = -\cos^2(t)\,(\sigma_d F_{\theta^-} - \hat v_t) - r\cos(t)\sin(t)\!\left(x_t + \sigma_d \frac{dF_{\theta^-}}{dt}\right),\quad g \leftarrow \frac{g}{\|g\|_2 + c}.$$

The warmup $r: 0 \to 1$ **only opens the second term**; at $r=0$ the term $-\cos^2(t)(\sigma_d F_{\theta^-} - \hat v_t)$ remains, so it degenerates to velocity / diffusion matching — **not zero loss**.

**Key tricks**:

- **Adaptive double normalization**: normalize both input and output by $\sigma_d$ + $\sigma(t)$ so the network's effective scale doesn't depend on $t$
- **Tangent warmup** (not "turn off all tangents"): $r$ controls the second term $-r\cos t\sin t(\cdots)$, the first term is always active; adaptive weighting $w_\phi(t)$ together with tangent normalization reduces variance
- **JVP via forward-mode autodiff**: PyTorch `torch.func.jvp`, **about 2× faster than computing the Jacobian via backward**

**Results**: 1.5B parameters, ImageNet 512×512 **2-step FID 1.88**, within 10% of the best diffusion baseline — first time CM hits top-tier numbers at large-scale high-res.

### 2.5　CTM / Consistency Trajectory Models (Kim et al. 2024 ICLR, arXiv:2310.02279)

**Problem**: CM can only map $(x_t, t) \to x_{\sigma_\min}$ (trajectory endpoint); cannot jump to intermediate points; step count is fixed.

**CTM's extension**: learn $G(x_t, t, s)$ — jump from $(x_t, t)$ to **any $s < t$**:

$$G_\theta(x_t, t, s) \approx \text{ODE-solver}(x_t, t \to s)$$

- $s = \sigma_\min$ degenerates to CM
- $s = t$ degenerates to identity
- Intermediate $s$ lets the user freely choose NFE: 3-step = $G(z, T, t_1) \to G(\cdot, t_1, t_2) \to G(\cdot, t_2, 0)$

**Loss**: trajectory matching —

$$\mathcal{L}_\text{CTM} = \mathbb{E}\Big[d\big(G_\theta(x_t, t, s),\; \text{ODE-solver}^\text{teacher}(x_t, t \to s)\big)\Big] + \lambda\, \mathcal{L}_\text{score}$$

- First term: trajectory consistency, the student reproduces teacher ODE
- Second term: auxiliary score matching (prevents trivial solutions)

**Results**: CIFAR-10 1-step FID 1.73, ImageNet 64 1.92 — SOTA. **Core contribution**: turns step count from "hard-coded" to "runtime selectable".

### 2.6　LCM / Latent Consistency Models (Luo et al. 2023, arXiv:2310.04378)

**LCM = CM on latent diffusion** (SD 1.5 / SDXL). Three improvements:

1. **Latent space**: do CM in VAE latent ($f=8$), saving $64\times$ compute
2. **CFG distilled into the student**: randomly sample guidance scale $w \in [w_\min, w_\max]$ during training and feed $w$ as an extra condition — $f_\theta(x_t, t, c, w)$. **No double forward needed at inference** for conditional + unconditional
3. **Skipping-Step Distillation**: use a $k$-step skipped teacher (e.g., $k=20$ skipping to 50/20 ≈ 2.5) to accelerate convergence

**Results**: 4-step SD-XL generation, FID close to 50-step SDXL (same base model).

### 2.7　LCM-LoRA (Luo et al. 2023, arXiv:2311.05556)

**Core idea**: the LCM-trained "weight difference" $\Delta \theta = \theta_\text{LCM} - \theta_\text{SD}$ can be parametrized as a LoRA —

$$\Delta W = B A,\quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \in \{8, 16, 32, 64\}$$

Only train $A, B$ (~22M params / SDXL); merge as $W' = W + \alpha B A$.

> ✅ **Ecosystem value of LCM-LoRA** — the SD 1.5 / SDXL ecosystem has tens of thousands of fine-tuned models (DreamShaper / RealisticVision / various character LoRAs). LCM-LoRA **doesn't require retraining each base model**; users just attach LCM-LoRA + their own LoRA and generate in 4 steps. This is why LCM is far more ubiquitous in production than DMD/ADD.

### 2.8　TCD / Trajectory Consistency Distillation (Zheng et al. 2024, arXiv:2402.19159)

**TCD = LCM + trajectory-aware** improvements. Two main contributions:

1. **Trajectory consistency function**: relax the boundary condition from "single $\sigma_\min$" to "any point along the trajectory". Specifically use a **semi-linear consistency function** (derived via exponential integrator) to reduce parametrization error
2. **Strategic stochastic sampling**: during multi-step inference, **explicitly control stochasticity** — add controllable perturbation via $\gamma \in [0, 1]$ to avoid accumulated error skewing the distribution

**Practical effect**: low-NFE (4 step) quality is higher than LCM; **at high NFE (8+ step) it's even more detailed than the teacher itself** (the stochastic sampling adds expressivity).

### 2.9　rCM / Score-Regularized Continuous-Time CM (2025, arXiv:2510.08431)

> 📍 **One of the latest CM works at the time of writing (2025-2026)** — rCM = "Score-Regularized Continuous-Time Consistency Model", arXiv:2510.08431 verified.

**Motivation**: sCM has a quality bottleneck on fine details — the authors attribute it to the **mode-covering nature of forward divergence** (KL(p_data ‖ p_student) tends to cover all modes, blurring details).

**rCM approach**: add a **score distillation regularizer** (reverse-divergence flavor, similar to DMD's KL gradient) on top of the sCM loss, so the student has both **mode-seeking** (sharp details) + mode-covering (diversity).

**Results**: on Cosmos-Predict2, Wan 2.1 (14B), produces 5-second video in 1-4 steps with quality matching DMD2 but better diversity.

## §3 Distribution Matching Distillation (DMD family)

### 3.1　DMD core: reverse-KL via score gap (Yin et al. 2024 CVPR, arXiv:2311.18828)

**Problem view**: student $G_\theta$ maps noise directly to image; we want its **output distribution $p_\text{fake}$ to match the teacher distribution $p_\text{real}$**. Directly optimize the gradient of $\text{KL}(p_\text{fake} \| p_\text{real})$ —

$$\nabla_\theta \text{KL}(p_\text{fake}^\theta \| p_\text{real}) = -\mathbb{E}_{x \sim p_\text{fake}^\theta}\!\left[\big(\nabla_x \log p_\text{real}(x) - \nabla_x \log p_\text{fake}(x)\big) \cdot \frac{\partial G_\theta}{\partial \theta}\right]$$

**Key observation**: $\nabla_x \log p_\text{real}$ is the teacher score $s_\text{real}$ (teacher diffusion model is off-the-shelf); $\nabla_x \log p_\text{fake}$ uses a **fake score model** $s_\text{fake}$ (a small diffusion trained on the student's current outputs).

**DMD Loss** (two losses trained jointly; **strict notation**: $\mu$ is the denoiser / mean predictor, $s_\mu(x_t,t) = (\alpha_t \mu(x_t,t) - x_t)/\sigma_t^2$ is the score derived from the denoiser; the DMD paper uses the denoiser, not the raw score):

$$
\boxed{\;
\begin{aligned}
\nabla_\theta \mathcal{L}_\text{DMD}^G &= \mathbb{E}_{z, t, \epsilon}\!\left[w_t\,\alpha_t\,(s_\text{fake}(x_t, t) - s_\text{real}(x_t, t))^\top\,\tfrac{\partial G_\theta(z)}{\partial\theta}\right] \quad\text{// student, surrogate} \\
\mathcal{L}_\text{fake}(\phi_f) &= \mathbb{E}\!\left[\lambda_t\,\|\mu_{\phi_f}(x_t, t) - \operatorname{sg}(G_\theta(z))\|_2^2\right] \quad\text{// fake denoiser DSM target is the student's current output}
\end{aligned}
\;}
$$

where $x_t = \alpha_t G_\theta(z) + \sigma_t \epsilon$. **Auxiliary regression loss**: DMD v1 also adds $\mathbb{E}\|G_\theta(z) - \text{ODE-solver}^\text{teacher}(z)\|^2$ (teacher-pair supervision) to keep the student from drifting — but this requires pre-generating a large pool of teacher pairs, **expensive and mode-limited**; DMD2 drops this term.

### 3.2　Deriving the DMD gradient from reverse-KL (mandatory derivation)

Let student $G_\theta(z) \mapsto x$ with $z \sim \mathcal{N}(0, I)$. The fake distribution is $p_\text{fake}^\theta(x) = G_\theta \# \mathcal{N}(0, I)$ (push-forward).

Reverse KL:

$$\text{KL}(p_\text{fake} \| p_\text{real}) = \mathbb{E}_{x \sim p_\text{fake}}[\log p_\text{fake}(x) - \log p_\text{real}(x)]$$

Take $\nabla_\theta$:

$$\nabla_\theta \text{KL} = \mathbb{E}_z\!\left[\nabla_\theta \log p_\text{fake}^\theta(G_\theta(z)) - \nabla_\theta \log p_\text{real}(G_\theta(z))\right]$$

Chain rule on the second term: $\nabla_\theta \log p_\text{real}(G_\theta(z)) = \nabla_x \log p_\text{real}(x)\big|_{x=G_\theta(z)} \cdot \partial G_\theta / \partial \theta$.

Expand the first term and use $\mathbb{E}_{p_\text{fake}}[\nabla_\theta \log p_\text{fake}] = 0$ (score-function trick); rearrange:

$$\nabla_\theta \text{KL} = -\mathbb{E}_z\!\left[(\nabla_x \log p_\text{real} - \nabla_x \log p_\text{fake})\big|_{x=G_\theta(z)} \cdot \partial_\theta G_\theta(z)\right]$$

**But $p_\text{real}, p_\text{fake}$ in high dimensions have scores that are discontinuous / non-smooth** — DMD's trick is to **compute scores at all noise levels $t$ for $x_t = G_\theta(z) + \sigma_t \epsilon$**, moving estimation to the smooth $p_t$ — this is why DMD needs both a real diffusion teacher and a fake diffusion (both provide "score at different noise levels").

> 💡 **DMD vs CM fundamental difference** — CM is **trajectory matching** (make the student reproduce the teacher ODE solution); DMD is **distribution matching** (make the two distributions have equal scores everywhere). **CM requires step alignment (noise schedules align), DMD does not** — DMD's student can be any differentiable generator architecture.

### 3.3　DMD2 (Yin et al. 2024 NeurIPS, arXiv:2405.14867)

**Four improvements**:

| Change | DMD | DMD2 |
|---|---|---|
| Regression loss | requires teacher pairs (expensive) | **dropped** |
| GAN | ❌ | **add GAN classifier**: attached to the fake-diffusion denoiser bottleneck, **discriminating on noised real / noised fake** (not clean images) |
| TTUR | 1:1 | fake denoiser **updated ~5× per generator step** (paper default 5:1 on ImageNet) |
| Student | 1-step only | **multi-step backward simulation**: during training, run the current student per the inference schedule to obtain intermediate noisy states, then compute DMD/GAN losses on those states to align training/inference distributions |

**DMD2 total loss** (generator side):

$$\mathcal{L}_\text{DMD2}^G = \underbrace{\mathcal{L}_\text{DMD}^G(\theta)}_{\text{score gap surrogate}} + \lambda_\text{GAN} \cdot \mathcal{L}_\text{adv}^G(\theta)$$

**Discriminator side**: DMD2's D is typically a classifier head on top of the fake-denoiser bottleneck, with input being noised image $x_t = \alpha_t x + \sigma_t\epsilon$ (not clean $x$):

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_\text{data}, t}\!\left[\text{softplus}(-D_\psi(x_t, t))\right] + \mathbb{E}_{z, t}\!\left[\text{softplus}(D_\psi(\hat x_t^{\text{fake}}, t))\right]$$

**Multi-step backward simulation** (key, much stronger than simple unrolling): during training run the current student per a $K$-step inference schedule to obtain intermediate noisy states $x_{t_k}$, then call the student / compute DMD-GAN loss on these states. This ensures the training input distribution = the input distribution seen by the student at step $k$ at inference, **not just naively noised real images**.

**Results**: ImageNet 64 1-step FID **1.28** (DMD v1 was 2.62), **the first time one-step diffusion surpassed GANs**. Production: DMD2-SDXL produces 1-step 1024×1024 megapixel images.

### 3.4　Statistical-physics intuition for the score gap

The reverse-KL "score gap" $s_\text{real} - s_\text{fake}$ physically corresponds to the **force difference between two Gibbs distributions** —

$$s_\text{real} - s_\text{fake} = \nabla_x \log\frac{p_\text{real}}{p_\text{fake}} = -\nabla_x [V_\text{real}(x) - V_\text{fake}(x)]$$

Treating the student as a particle, $s_\text{real} - s_\text{fake}$ is the "force" pushing it from $p_\text{fake}$ toward $p_\text{real}$. **This is the fundamental difference between DMD and GAN** — GANs use a discriminator's binary signal; DMD uses the score gap to give a **dense vector-field signal**, with far higher sample efficiency.

## §4 Adversarial Distillation (ADD / LADD family)

### 4.1　ADD / SDXL-Turbo (Sauer et al. 2023, arXiv:2311.17042)

**Stability AI 2023.11, making SDXL produce 512² images in 1 step**. Two supervisions:

$$\boxed{\;\mathcal{L}_\text{ADD} = \mathcal{L}_\text{adv}^G(\theta, \psi) + \lambda \cdot \mathcal{L}_\text{distill}(\theta, \phi)\;}$$

- **$\mathcal{L}_\text{adv}$**: hinge loss + **DINOv2 vision backbone as discriminator** (not training D from scratch; fix DINOv2 + multiple heads)
- **$\mathcal{L}_\text{distill}$**: discrete form of "score distillation" — MSE between student 1-step output and teacher multi-step output

**The DINOv2 discriminator** is key to ADD:

- Ordinary GANs train D from scratch, **which is unstable for 1-step generators** (severe mode collapse)
- DINOv2 provides "pretrained high-level perceptual features", anchoring the discrimination problem in a **strong semantic space**
- Multiple heads (different layer features) + hinge loss → stable training

> ⚠️ **Actual form of the distillation loss** — the $\mathcal{L}_\text{distill}$ in the ADD paper is a **score-distillation style** loss (estimating targets using the teacher denoiser on the noised student output), **not** a plain pixel-space MSE and **not** a KL. The pedagogical version $\|G_\theta - \text{ODE}\|^2$ shown earlier is an illustrative simplification; see the original paper Eq. (6-7) for details. ADD relies on the GAN loss for mode diversity.

**Results**: SDXL-Turbo at **512×512** does 1-step in ~100ms/image (A100), CLIP score on par with 4-step SDXL; at **1024×1024** quality is limited on SDXL-Turbo and addressed by later **LADD / SD3-Turbo / Lightning**.

### 4.2　LADD / SD3-Turbo (Sauer et al. 2024, arXiv:2403.12015)

**Problem**: ADD computes distill loss in pixel space and uses DINOv2 D — unfriendly to **high resolution (1024+) and latent diffusion**; pixel decoding is expensive and DINOv2's input resolution is limited to 224 / 518.

**LADD = Latent ADD**: move the discriminator directly into latent space —

| | ADD | LADD |
|---|---|---|
| Discriminator backbone | DINOv2 (pixel) | **teacher diffusion's own intermediate layer features** (latent) |
| Distillation | pixel MSE | latent-space distill |
| Resolution scale | limited by DINOv2 | latent any size |
| Application models | SDXL | **SD3 (8B), FLUX (12B)** |

**Discriminator design**: extract the teacher's MM-DiT block and fine-tune it as the D backbone — the rationale being **intermediate features learned during diffusion training already implicitly encode "what a realistic latent looks like"**.

**Results**:

- **SD3-Turbo** = SD3 8B + LADD → 4-step 1024² rivals multi-step SD3
- **FLUX.1-schnell** = FLUX 12B + LADD → 1-4 step 1024² generation (Apache 2.0 open source)

### 4.3　SDXL-Lightning (Lin et al. 2024, arXiv:2402.13929)

ByteDance's open-source SDXL distillation, with **progressive + adversarial** two-pronged design:

- **Progressive (halving)**: starting from teacher multi-step, **halve** the step count per stage ($T \to T/2 \to T/4 \to \dots$), each stage fitting the previous stage's teacher with MSE (in the Salimans-Ho 2022 progressive distillation lineage). Finally selectable at 1/2/4/8-step
- **Adversarial**: end each stage with GAN loss for fidelity
- **Discriminator**: self-trained (not borrowed from an off-the-shelf backbone like ADD/LADD)

**Results**: SDXL 1024² selectable at 1-step / 2-step / 4-step, **open-source LoRA form** (similar to LCM-LoRA), ecosystem-friendly.

### 4.4　ADD/LADD/Lightning comparison

| Method | Discriminator | Distill loss | Application | 1-step quality |
|---|---|---|---|---|
| **ADD** | DINOv2 (pixel) | pixel MSE | SDXL | medium (512²) |
| **LADD** | teacher MM-DiT feat (latent) | latent score-distill | SD3, FLUX | high (1024²) |
| **Lightning** | self-trained CNN | progressive MSE | SDXL | medium-high |
| **DMD2** | self-trained + score gap | reverse-KL via score | SDXL | high (with diversity) |

> 💡 **Production selection cheat sheet** — if base is SD 1.5 / SDXL use **LCM-LoRA** (largest ecosystem) or **SDXL-Lightning** (open-source stable); if base is SD3 / FLUX, **LADD** is the official route; for **GAN-free + score-based** pick **DMD2**; for academic SOTA chasing, pick **sCM / rCM**.

## §5 Flow / Rectified Flow distillation

### 5.1　Rectified Flow + Reflow route (Liu et al. 2022, arXiv:2209.03003)

**Rectified Flow path**: $x_t = (1-t) x_0 + t\, x_1$, $x_0 \sim \mathcal{N}(0, I)$ (noise side), $x_1 \sim p_\text{data}$, target $u_t = x_1 - x_0$ (**constant vector**).

**Reflow algorithm** (reverse via ODE, re-pair, re-train):

1. Train $v_\theta^{(1)}$ on independent pairs $(x_0, x_1) \sim p_0 \otimes p_\text{data}$
2. Use $v_\theta^{(1)}$ to run the ODE and generate coupled pairs $(x_0, x_1^{(1)})$, i.e., $x_1^{(1)} = x_0 + \int_0^1 v_\theta^{(1)}(x_t, t)\, dt$
3. Retrain $v_\theta^{(2)}$ on coupled pairs — **the new trajectory is straighter** (transport-cost-non-increasing theorem)

**Why does reflow make trajectories straight?**

Consider the transport cost $\mathbb{E}[\|x_1 - x_0\|^2]$ as coupling. Independent pairs have large cost; after reflow $(x_0, x_1^{(1)})$ has already been naturally paired by the ODE — it is the "optimal transport" under the current $v_\theta^{(1)}$. Liu 2022 shows: after another round of training the total transport cost is non-increasing (in fact often strictly decreasing), and **"the curve being straight" is equivalent to the vector field not depending on $t$** — $v(t, x) = $ const along the trajectory → 1-step generation.

**InstaFlow (2023, arXiv:2309.06380)**: the first to apply reflow to SD, 1-step generation FID 23.3 (512²).

### 5.2　The "straight-line" limit of reflow

**Ideal limit**: if reflow converges to fully straight, then $v_\theta(t, x) = v_\theta(x)$ (independent of $t$), and 1-step Euler suffices —

$$x_1 = x_0 + 1 \cdot v_\theta(x_0)$$

**In practice**: 1-2 rounds of reflow already make trajectories straight enough to support 4-step Euler matching 50-step quality; full 1-step requires more reflow + adversarial fine-tuning (like SD3-Turbo / FLUX-schnell).

### 5.3　SD3-Turbo / FLUX-schnell = RF + LADD

The actual production stack:

```
SD3 / FLUX (Rectified Flow, ~50-step 1024²)
  │ pretrain
  ↓
LADD distillation (teacher distill + latent discriminator)
  ↓
SD3-Turbo / FLUX-schnell (1-4 step 1024²)
```

**Not pure reflow** — LADD provides adversarial fidelity, more stable than pure reflow for high-res production.

### 5.4　Flow-OPD (arXiv:2605.08063, 2026) — **out-of-scope sidebar**

> 📍 **Scope clarification** — Flow-OPD's main contribution is **multi-reward RL alignment + on-policy specialist distillation**, **not** few-step inference distillation. It appears here because its name contains "Distillation" and it involves flow models; but the core thread of §2-§4 (CM/DMD/ADD compressing 50 steps to 1-4) differs from Flow-OPD's RL-alignment objective.
>
> The companion piece **`diffusion_post_training_tutorial.md`** has a full discussion of RL alignment (Flow-GRPO / Diffusion-DPO / DDPO etc.); Flow-OPD is more accurately placed there.

**Brief idea (sidebar only, see post-training tutorial + original paper for depth)**: use multiple reward-specific teachers (each GRPO fine-tuned on a reward) for on-policy distillation supervision; the student remains few-step at inference while simultaneously gaining multi-reward alignment.

**Loss form** (simplified sketch, refer to original paper):

$$\mathcal{L}_\text{OPD-sketch} = \mathbb{E}_{x \sim \pi_\theta}\!\left[\sum_k w_k(x) \cdot \|v_\theta(x_t, t) - v_{\phi_k}(x_t, t)\|^2\right]$$

with task-aware weighting $w_k$; this is only a structural sketch — **do not claim** that Flow-OPD and DMD are "degeneratively equivalent" mathematically (no reliable basis for that claim; please don't assert it in interviews).

**Paper-reported results**: based on SD 3.5 Medium, GenEval 63 → 92, OCR 59 → 94 (see paper Table for details).

### 5.5　Rectified Diffusion / follow-up work

> 📍 **Brief mention of related work outside main scope, Rectified Diffusion (arXiv:2410.07303)**

Follow-up work (e.g., Rectified Diffusion) challenges whether straightness is necessary — finding that **a straight line is not a necessary condition**; it suffices for the ODE solution space to be sufficiently expressive. This line and sCM's "continuous-time CM" show signs of mathematical convergence.

## §6 CFG distillation

### 6.1　Why CFG must be distilled separately

CFG inference:

$$\tilde\epsilon(x, c) = (1 + w) \epsilon_\theta(x, c) - w\, \epsilon_\theta(x, \emptyset)$$

**Two forwards per step** (conditional + unconditional), doubling latency. So production CFG-aware models need to **distill CFG into a single forward**.

### 6.2　Guidance Distillation (Meng et al. 2023 CVPR, arXiv:2210.03142)

**Stage 1 — Guidance distillation**: train a student $\tilde\epsilon_\theta(x, c, w)$ that takes **guidance scale $w$ as an extra condition**, directly outputting the post-CFG score:

$$\mathcal{L}_\text{guide} = \mathbb{E}\!\left[\|\tilde\epsilon_\theta(x_t, c, w) - \tilde\epsilon^*(x_t, c, w)\|^2\right]$$

where $\tilde\epsilon^*$ is the CFG output obtained by the teacher's two explicit forwards. The student does only one forward.

**Stage 2 — Step distillation**: on top of stage 1, stack progressive distillation, compressing steps from 32 to 4 → 2 → 1.

**LCM's CFG-aware design** is inherited from this — feeding $w$ as a condition is the key to LCM-LoRA.

### 6.3　Step-distillation vs trajectory-distillation differences

| | Step-distillation (Salimans-Ho 2022) | Trajectory-distillation (CM/CTM) |
|---|---|---|
| Goal | distill $N$-step student to $N/2$-step | learn trajectory function $f(x_t, t) \to x_0$ |
| Training stage | multi-stage progressive | single stage |
| Step count | halve each round (32→16→8→4→2→1) | arbitrary (1-step direct training) |
| Boundary condition | none required | requires $f(x_{\sigma_\min}, \sigma_\min) = x$ |
| Teacher | previous-stage student (self-distillation) | original diffusion |

> 💡 **Historical arc** — 2022 progressive distillation was the first to make 4-step diffusion usable; 2023 CM directly enabled 1-step via trajectory function; 2024 sCM / DMD2 pushed 1-step to SOTA. **Idea evolution**: iterative approximation (progressive) → function fitting (CM) → distribution matching (DMD) → trig parametrization (sCM).

## §7 From-scratch PyTorch code

### 7.1　Code 1: Consistency Distillation Loss (CD, base CM)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def edm_precond(F_net, x, sigma, sigma_data=0.5, sigma_min=0.002):
    """EDM-style precond, so boundary f(x, sigma_min) = x is automatically satisfied"""
    c_skip = sigma_data**2 / ((sigma - sigma_min)**2 + sigma_data**2)
    c_out  = sigma_data * (sigma - sigma_min) / torch.sqrt(sigma_data**2 + sigma**2)
    c_in   = 1.0 / torch.sqrt(sigma_data**2 + sigma**2)
    c_noise = 0.25 * torch.log(sigma)
    # Broadcast to [B, 1, 1, 1] (images)
    c_skip = c_skip.view(-1, 1, 1, 1)
    c_out  = c_out.view(-1, 1, 1, 1)
    c_in   = c_in.view(-1, 1, 1, 1)
    F_x = F_net(c_in * x, c_noise)
    return c_skip * x + c_out * F_x

@torch.no_grad()
def teacher_ode_step(x_t1, t1, t0, teacher):
    """teacher one-step Heun (EDM 2nd-order) reverse: t1 -> t0"""
    d1 = (x_t1 - teacher(x_t1, t1)) / t1  # current gradient
    x_euler = x_t1 + (t0 - t1) * d1       # Euler prediction
    d2 = (x_euler - teacher(x_euler, t0)) / t0
    return x_t1 + 0.5 * (t0 - t1) * (d1 + d2)

def consistency_distillation_loss(student, student_ema, teacher,
                                   x_0, sigmas, N=18):
    """
    student / student_ema: same architecture; ema is a stop-grad copy
    teacher: pretrained diffusion (EDM denoiser)
    x_0: clean image batch [B, C, H, W]
    sigmas: noise schedule, **indices in ascending = noise increasing** (sigmas[0]=sigma_min, sigmas[N]=sigma_max)
    
    !!! Pedagogical sketch: we convention sigmas in ascending order so t_{n+1} > t_n is convenient.
        For production refer to EDM official code (karras/edm: typically descending sigmas) + paper Eq. form.
    """
    B = x_0.shape[0]
    # 1) Random adjacent noise level n ~ U{0, N-1}
    n = torch.randint(0, N, (B,), device=x_0.device)
    t_n1 = sigmas[n + 1]    # higher noise (per the above convention: sigmas ascending)
    t_n  = sigmas[n]        # lower noise
    
    # 2) Sample x_{t_{n+1}} = x_0 + t_{n+1} * eps
    eps = torch.randn_like(x_0)
    x_tn1 = x_0 + t_n1.view(-1, 1, 1, 1) * eps
    
    # 3) Teacher one-step ODE reverse to get x_{t_n}
    with torch.no_grad():
        x_tn = teacher_ode_step(x_tn1, t_n1, t_n, teacher)
    
    # 4) student / student_ema both pass through EDM precond
    f_online = edm_precond(student, x_tn1, t_n1)
    with torch.no_grad():
        f_target = edm_precond(student_ema, x_tn, t_n)
    
    # 5) consistency loss (LPIPS / L2; use L2 here)
    loss = F.mse_loss(f_online, f_target)
    return loss

def update_ema(ema_model, model, decay=0.9999):
    """EMA target, BYOL-style"""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
```

### 7.2　Code 2: iCT (no EMA + Pseudo-Huber + Lognormal + Curriculum)

```python
def pseudo_huber(a, b, c=0.00054):
    """Pseudo-Huber loss: sqrt(||a-b||^2 + c^2) - c
    Small residual ≈ L2/2c, large residual ≈ L1. iCT paper c=0.00054 (CIFAR-10)"""
    return torch.sqrt((a - b).pow(2).sum(dim=(1, 2, 3)) + c**2).mean() - c

def lognormal_sigma(B, P_mean=-1.1, P_std=2.0, sigma_min=0.002, sigma_max=80.0):
    """iCT samples sigma from lognormal instead of uniform
    log_sigma ~ N(P_mean, P_std)"""
    log_sigma = torch.randn(B) * P_std + P_mean
    sigma = torch.exp(log_sigma).clamp(sigma_min, sigma_max)
    return sigma

def get_curriculum_N(step, total_steps, N_min=10, N_max=1280, schedule='exp'):
    """Step-count curriculum: N grows from 10 to 1280
    Over K training steps, N(k) = ceil(N_min * (N_max/N_min)^(k/K))"""
    k = step / total_steps
    if schedule == 'exp':
        N = N_min * (N_max / N_min) ** k
    else:
        N = N_min + (N_max - N_min) * k
    return int(math.ceil(N))

def ict_loss(student, x_0, step, total_steps):
    """iCT: no EMA, no LPIPS, no teacher
    consistency loss on (x_0 + sigma_n*eps, x_0 + sigma_{n+1}*eps) with SAME eps"""
    B = x_0.shape[0]
    device = x_0.device
    
    # 1) curriculum N
    N = get_curriculum_N(step, total_steps)
    
    # 2) Pick adjacent sigma_n, sigma_{n+1} (chosen from N+1 lognormal-discretized points)
    # !!! convention: sigmas ascending, so sigmas[n+1] > sigmas[n] (consistent with §2.3 CD code)
    sigmas = lognormal_sigma(N + 1).to(device).sort(descending=False).values
    n_idx = torch.randint(0, N, (B,), device=device)
    t_n1 = sigmas[n_idx + 1]  # higher noise
    t_n  = sigmas[n_idx]      # lower noise
    
    # 3) Key: same epsilon, two different noise levels (no teacher needed)
    eps = torch.randn_like(x_0)
    x_tn1 = x_0 + t_n1.view(-1, 1, 1, 1) * eps
    x_tn  = x_0 + t_n.view(-1, 1, 1, 1) * eps
    
    # 4) Student runs both (no EMA; target uses stop_grad)
    f_online = edm_precond(student, x_tn1, t_n1)
    with torch.no_grad():
        f_target = edm_precond(student, x_tn, t_n)
    
    # 5) Pseudo-Huber
    loss = pseudo_huber(f_online, f_target, c=0.00054)
    return loss
```

### 7.3　Code 3: sCM Continuous-Time Loss (TrigFlow)

```python
import torch.func as tfunc

def trigflow_xt(x_0, z, t):
    """TrigFlow path: x_t = cos(t) x_0 + sin(t) z, t in [0, π/2]"""
    cos_t = torch.cos(t).view(-1, 1, 1, 1)
    sin_t = torch.sin(t).view(-1, 1, 1, 1)
    return cos_t * x_0 + sin_t * z

def scm_loss(F_net, x_0, sigma_data=0.5, r_warmup=0.5):
    """
    Simplified continuous-time CM loss (sCM, Lu-Song 2024).
    F_net: student network F_θ(x_t / σ_d, t)
    r_warmup: NCS warmup ratio (0=pure score, 1=pure CM)
    """
    import math
    B = x_0.shape[0]
    device = x_0.device

    # 1) lognormal t (TrigFlow time t in [0, π/2])
    log_t = torch.randn(B, device=device) * 1.0 - 0.4  # σ ≈ 1, mean shift
    t = torch.sigmoid(log_t) * (math.pi / 2 - 0.001) + 0.001  # avoid boundary

    # 2) Sample x_t
    z = torch.randn_like(x_0)
    x_t = trigflow_xt(x_0, z, t)

    # 3) PF-ODE tangent direction (TrigFlow: dx_t/dt = -sin(t) x_0 + cos(t) z)
    cos_t = torch.cos(t).view(-1, 1, 1, 1)
    sin_t = torch.sin(t).view(-1, 1, 1, 1)
    dxdt = -sin_t * x_0 + cos_t * z

    # 4) F_θ output at (x_t/σ_d, t) + JVP (forward-mode autodiff, faster than backward Jacobian)
    def net_fn(xt_norm, t_):
        return F_net(xt_norm, t_)

    x_t_norm = x_t / sigma_data

    # 4b) Student forward (with grad) to get F_out
    F_out = net_fn(x_t_norm, t)

    # 5) sCM target: stop_grad(F_minus + normalized tangent g)
    # First term (velocity matching) still provides signal at r=0; second term is opened by warmup r.
    # JVP tangent direction = dx/dt (PF-ODE direction); JVP output = dF/dt.
    tangent_x = dxdt / sigma_data        # tangent of (x_t/σ_d) along dx/dt direction
    tangent_t = torch.ones_like(t)       # dt/dt = 1

    with torch.no_grad():
        F_minus, dFdt = tfunc.jvp(
            net_fn,
            (x_t_norm, t),
            (tangent_x, tangent_t),
        )
        # First term: velocity / diffusion matching (signal exists even at r=0)
        g = -(cos_t ** 2) * (sigma_data * F_minus - dxdt)
        # Second term: consistency tangent, opened gradually by warmup (cos·sin factor appears once only)
        g = g - r_warmup * (cos_t * sin_t * x_t + sigma_data * cos_t * sin_t * dFdt)
        # Normalize tangent for stability
        g = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 0.1)
        target = F_minus + g

    # 6) surrogate MSE loss (adaptive w_phi(t) omitted in tutorial code)
    loss = F.mse_loss(F_out, target)
    return loss
```

### 7.4　Code 4: DMD Loss (Distribution Matching via Score Gap)

```python
class DMDTrainer:
    """
    DMD v1 (Yin 2024 CVPR): three networks
    - G_θ: 1-step student generator (z -> x)
    - s_real: pretrained teacher diffusion (frozen)
    - s_fake: fake diffusion, trained on G_θ outputs
    """
    def __init__(self, G, s_fake, s_real_frozen, sigma_data=0.5):
        self.G = G
        self.s_fake = s_fake  # trainable
        self.s_real = s_real_frozen  # frozen
        self.opt_G = torch.optim.AdamW(G.parameters(), lr=1e-5)
        self.opt_f = torch.optim.AdamW(s_fake.parameters(), lr=1e-5)
        self.sigma_data = sigma_data

    def student_loss(self, z):
        """DMD student loss: alpha_t * (s_fake - s_real)^T · ∂G/∂θ
        where s_*(x_t,t) = (alpha_t * mu_*(x_t,t) - x_t) / sigma_t^2 is derived from the denoiser.
        EDM/VE convention has alpha_t = 1; VP/DDPM uses scheduler's alpha_t.
        """
        x = self.G(z)               # G_θ(z), 1-step output
        B = x.shape[0]
        # Random noise level
        sigma = torch.exp(torch.randn(B, device=x.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        alpha = 1.0                     # EDM/VE; for VP use scheduler.alpha(t)
        x_t = alpha * x + sigma * eps   # consistent with paper's x_t = alpha*x + sigma*eps

        with torch.no_grad():
            mu_real = self.s_real(x_t, sigma.squeeze())  # frozen denoiser / mean predictor
            mu_fake = self.s_fake(x_t, sigma.squeeze())  # trainable fake denoiser

            # DMD score gap: s_fake - s_real = α(μ_fake - μ_real)/σ².
            # With DMD weight w_t ∝ σ²/α canceling 1/σ², we get w_t(s_fake - s_real) = α(μ_fake - μ_real).
            # Then apply mean-abs normalization for numerical stability (DMD paper Eq. (8)).
            grad_proxy = alpha * (mu_fake - mu_real)
            grad_proxy = grad_proxy / (
                (x.detach() - mu_real).abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            )

        # surrogate: loss = +(x · grad_proxy.detach()).sum(),
        # backward gives ∇L = grad_proxy · ∂G/∂θ = ∇_θ KL(p_fake‖p_real),
        # optimizer step θ -= η∇L thus minimizes KL.
        loss_G = (x * grad_proxy.detach()).sum(dim=(1, 2, 3)).mean()
        return loss_G

    def fake_score_loss(self, z):
        """fake denoiser uses DSM; target is the student's current output (not -eps/sigma score-head form)"""
        with torch.no_grad():
            x = self.G(z)               # detached student output
        B = x.shape[0]
        sigma = torch.exp(torch.randn(B, device=x.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        alpha = 1.0
        x_t = alpha * x + sigma * eps

        pred_x0 = self.s_fake(x_t, sigma.squeeze())   # denoiser output
        target_x0 = x.detach()                          # student output as DSM target
        loss_f = ((pred_x0 - target_x0) ** 2).flatten(1).mean(1).mean()
        return loss_f

    def step(self, z_batch):
        # 1) update fake score (on G's current output)
        self.opt_f.zero_grad()
        loss_f = self.fake_score_loss(z_batch)
        loss_f.backward()
        self.opt_f.step()
        
        # 2) update G via score gap
        self.opt_G.zero_grad()
        loss_G = self.student_loss(z_batch)
        loss_G.backward()
        self.opt_G.step()
        return loss_G.item(), loss_f.item()
```

### 7.5　Code 5: DMD2 Loss (drop regression + GAN + multi-step)

```python
class DMD2Trainer(DMDTrainer):
    """DMD2: drop regression + add noised-input GAN + multi-step backward simulation.
    In the original paper D is a classifier head on the fake-denoiser bottleneck (shared backbone);
    input is the noised image x_t = alpha*x + sigma*eps. Here we use a standalone D as a pedagogical approximation.
    TTUR: fake denoiser + D updated ~5× per generator step (paper default 5:1 on ImageNet).
    """
    def __init__(self, G, s_fake, s_real_frozen, D, sigma_data=0.5,
                 lambda_gan=1.0, num_steps_train=4, ttur_ratio=5):
        super().__init__(G, s_fake, s_real_frozen, sigma_data)
        self.D = D
        self.opt_D = torch.optim.AdamW(D.parameters(), lr=1e-5)
        self.lambda_gan = lambda_gan
        self.K = num_steps_train
        self.ttur_ratio = ttur_ratio

    def _sample_multistep(self, z, K=None, with_grad=False):
        """Backward simulation: run student per inference schedule, **return each step's clean denoised output**.
        with_grad=True keeps grad along the whole chain (for generator loss); False detaches for D / fake denoiser.
        Returns:
          x_finals: list of [B, ...] clean denoised outputs (including final, len = K)
          x_noised_inputs: list of [B, ...] noisy inputs fed to G at the next step (len = K, first = z)
        In the paper this re-noises per EDM/TrigFlow schedule; here sigma_next = t_next as a placeholder.
        """
        K = K or self.K
        ts = torch.linspace(1.0, 0.0, K + 1, device=z.device)
        x_finals = []
        x_noised_inputs = []
        x_input = z
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            for k in range(K):
                t_k = ts[k].expand(z.shape[0])
                x_noised_inputs.append(x_input)
                x_clean = self.G(x_input, t_k)   # denoised output
                x_finals.append(x_clean)
                if k < K - 1:
                    sigma_next = ts[k + 1]
                    # re-noise clean output to the next timestep's noisy state
                    x_input = x_clean + sigma_next * torch.randn_like(x_clean)
        return x_finals, x_noised_inputs

    def student_loss_dmd2(self, z, real_batch):
        # 1) backward simulation with grad: get each step's clean denoised output
        x_finals, _ = self._sample_multistep(z, K=self.K, with_grad=True)
        # 2) DMD score gap: compute and average over each clean output (DMD2 paper multi-step loss)
        loss_score = sum(self._score_gap_loss(x_c) for x_c in x_finals) / len(x_finals)
        # 3) GAN generator loss: D discriminates on the noised version of the final clean output
        x_final = x_finals[-1]
        sigma = torch.exp(torch.randn(x_final.shape[0], device=x_final.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        x_fake_t = x_final + sigma * torch.randn_like(x_final)
        d_fake_logit = self.D(x_fake_t, sigma.squeeze())
        loss_adv = F.softplus(-d_fake_logit).mean()  # non-saturating
        return loss_score + self.lambda_gan * loss_adv

    def fake_score_loss(self, z):
        """Override: fake denoiser DSM target = student's multi-step backward-simulated outputs.
        DMD2 wants the fake denoiser to learn the whole simulated distribution of the generator, not just 1-step G(z).
        """
        with torch.no_grad():
            x_finals, _ = self._sample_multistep(z, K=self.K, with_grad=False)
        # Do DSM on all K clean outputs
        loss_total = 0.0
        for x_c in x_finals:
            B = x_c.shape[0]
            sigma = torch.exp(torch.randn(B, device=x_c.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
            alpha = 1.0
            x_t = alpha * x_c + sigma * torch.randn_like(x_c)
            pred_x0 = self.s_fake(x_t, sigma.squeeze())
            target_x0 = x_c.detach()
            loss_total = loss_total + ((pred_x0 - target_x0) ** 2).flatten(1).mean(1).mean()
        return loss_total / len(x_finals)

    def step(self, z_batch, real_batch):
        """DMD2 step: each generator update pairs with ttur_ratio fake-denoiser + D updates."""
        for _ in range(self.ttur_ratio):
            self.opt_f.zero_grad()
            loss_f = self.fake_score_loss(z_batch)
            loss_f.backward()
            self.opt_f.step()

            self.opt_D.zero_grad()
            loss_D = self.discriminator_loss(z_batch, real_batch)
            loss_D.backward()
            self.opt_D.step()
        # generator update
        self.opt_G.zero_grad()
        loss_G = self.student_loss_dmd2(z_batch, real_batch)
        loss_G.backward()
        self.opt_G.step()
        return loss_G.item(), loss_f.item(), loss_D.item()

    def _score_gap_loss(self, x_fake):
        # Reuse the denoiser-based score gap from DMDTrainer.student_loss (see §7.4);
        # only difference: input is multi-step student output rather than 1-step.
        B = x_fake.shape[0]
        sigma = torch.exp(torch.randn(B, device=x_fake.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        alpha = 1.0
        x_t = alpha * x_fake + sigma * torch.randn_like(x_fake)
        with torch.no_grad():
            mu_real = self.s_real(x_t, sigma.squeeze())
            mu_fake = self.s_fake(x_t, sigma.squeeze())
            # See §7.4 derivation: w_t(s_fake - s_real) = α(μ_fake - μ_real)
            grad_proxy = alpha * (mu_fake - mu_real)
            grad_proxy = grad_proxy / (
                (x_fake.detach() - mu_real).abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
            )
        return (x_fake * grad_proxy.detach()).sum(dim=(1, 2, 3)).mean()

    def discriminator_loss(self, z, real_batch):
        """D discriminates noised image as real vs student output.
        Softplus / non-saturating loss + shared fake-denoiser backbone (we use a separate D for teaching).
        """
        x_finals, _ = self._sample_multistep(z, with_grad=False)
        x_fake = x_finals[-1]
        B = real_batch.shape[0]
        sigma = torch.exp(torch.randn(B, device=real_batch.device) * 1.6 - 1.0).view(-1, 1, 1, 1)
        x_real_t = real_batch + sigma * torch.randn_like(real_batch)
        x_fake_t = x_fake.detach() + sigma * torch.randn_like(x_fake)
        d_real = self.D(x_real_t, sigma.squeeze())
        d_fake = self.D(x_fake_t, sigma.squeeze())
        return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
```

### 7.6　Code 6: ADD (Adversarial Diffusion Distillation, SDXL-Turbo style)

```python
import torchvision  # for DINOv2 backbone

class ADDTrainer:
    """ADD (Sauer 2023): pretrained DINOv2 backbone as discriminator"""
    def __init__(self, G, teacher_diffusion, sigma_data=0.5,
                 lambda_distill=1.0):
        self.G = G
        self.teacher = teacher_diffusion  # frozen
        # DINOv2 backbone + multiple discriminator heads
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        # multi-layer head: extract features from different DINOv2 blocks, each with a 1x1 conv head
        self.disc_heads = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1024, 1, 1), nn.Flatten())
            for _ in range(4)
        ])
        self.opt_G = torch.optim.AdamW(G.parameters(), lr=1e-5)
        self.opt_D = torch.optim.AdamW(self.disc_heads.parameters(), lr=1e-5)
        self.lambda_distill = lambda_distill

    def get_dino_features(self, x):
        """Extract features from multiple DINOv2 layers"""
        # Simplified: dinov2_vitl14 intermediate layer hooks (real code uses register_forward_hook)
        # Here we return a list of features for each disc head
        x_resized = F.interpolate(x, size=224, mode='bilinear')
        # mock: assume backbone yields [B, 1024, N_patch] sequence, 4 layers
        feats = self.dino.get_intermediate_layers(x_resized, n=4)
        return feats

    def adv_loss_G(self, x_fake):
        feats = self.get_dino_features(x_fake)
        loss = 0
        for feat, head in zip(feats, self.disc_heads):
            logit = head(feat.transpose(1, 2))  # [B, ?]
            loss += -logit.mean()  # non-saturating
        return loss / len(self.disc_heads)

    def adv_loss_D(self, x_real, x_fake):
        feats_real = self.get_dino_features(x_real)
        feats_fake = self.get_dino_features(x_fake.detach())
        loss = 0
        for fr, ff, head in zip(feats_real, feats_fake, self.disc_heads):
            d_r = head(fr.transpose(1, 2))
            d_f = head(ff.transpose(1, 2))
            loss += F.relu(1 - d_r).mean() + F.relu(1 + d_f).mean()
        return loss / len(self.disc_heads)

    def distill_loss(self, z, x_fake):
        """teacher multi-step ODE output as supervision"""
        with torch.no_grad():
            x_teacher = self.teacher_ode_sample(z, steps=4)
        # pixel-level MSE
        return F.mse_loss(x_fake, x_teacher)

    @torch.no_grad()
    def teacher_ode_sample(self, z, steps=4):
        """teacher runs K-step ODE to generate images as student's distillation target"""
        # ... EDM Heun sampler, implementation omitted
        return self.teacher.sample(z, num_steps=steps)

    def step(self, z, x_real):
        # 1) G output (1-step)
        x_fake = self.G(z)
        # 2) D loss
        self.opt_D.zero_grad()
        loss_D = self.adv_loss_D(x_real, x_fake)
        loss_D.backward()
        self.opt_D.step()
        # 3) G loss (adversarial + distill)
        x_fake = self.G(z)  # recompute (D updated)
        self.opt_G.zero_grad()
        loss_adv = self.adv_loss_G(x_fake)
        loss_dist = self.distill_loss(z, x_fake)
        loss_G = loss_adv + self.lambda_distill * loss_dist
        loss_G.backward()
        self.opt_G.step()
        return loss_G.item(), loss_D.item()
```

### 7.7　Code 7: LCM-LoRA attached to SDXL

```python
# Assume a diffusers-style SDXL pipeline is available
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from peft import LoraConfig, get_peft_model

def attach_lcm_lora(sdxl_pipe, lcm_lora_path="latent-consistency/lcm-lora-sdxl"):
    """LCM-LoRA: attach LCM distillation's weight difference as a LoRA.
    In current diffusers (>=0.24) `LCMScheduler`'s teacher-step parameter is `original_inference_steps`
    in the scheduler config; only old community pipelines / early dreamshaper examples use `lcm_origin_steps`.
    """
    # 1) Switch scheduler to LCM-style; teacher-equivalent step count in config
    sdxl_pipe.scheduler = LCMScheduler.from_config(
        sdxl_pipe.scheduler.config,
        original_inference_steps=50,   # current diffusers LCMScheduler API
    )
    # 2) load LCM-LoRA weights
    sdxl_pipe.load_lora_weights(lcm_lora_path)
    # 3) optional: also attach the user's own LoRA (e.g., character LoRA)
    # sdxl_pipe.load_lora_weights("path/to/user_lora", adapter_name="char")
    # sdxl_pipe.set_adapters(["default", "char"], adapter_weights=[1.0, 0.8])
    return sdxl_pipe

# Inference: only 4 steps
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")
pipe = attach_lcm_lora(pipe)
images = pipe(
    prompt="a cat sitting on a chair",
    num_inference_steps=4,         # key: LCM only needs 4 steps
    guidance_scale=0.0,            # HF LCM-LoRA currently recommends 0.0; 1.0-2.0 also OK
).images
```

### 7.8　Code 8: Reflow (Rectified Flow distillation)

```python
@torch.no_grad()
def reflow_generate_pairs(v_net, num_samples, sample_shape, steps=50, device='cuda'):
    """Run current v_θ ODE to generate coupled (x_0, x_1) pairs for reflow retraining.
    sample_shape: tuple, e.g., (D,) for toy data or (C, H, W) for image latents.
    """
    x_0 = torch.randn(num_samples, *sample_shape, device=device)
    x = x_0.clone()
    ts = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(num_samples)
        dt = ts[i + 1] - ts[i]
        x = x + dt * v_net(x, t)
    return x_0, x  # x_1 = ODE(x_0; v_θ), naturally coupled

def reflow_loss(v_net, x_0, x_1, t_dist='uniform'):
    """RF + reflow loss: retrain v_θ^{(k+1)} on coupled (x_0, x_1^{(k)}) pairs"""
    B = x_0.shape[0]
    if t_dist == 'uniform':
        t = torch.rand(B, device=x_0.device)
    else:  # logit-normal (SD3 style)
        t = torch.sigmoid(torch.randn(B, device=x_0.device))

    # Broadcast to any rank: t shape -> (B, 1, 1, ..., 1) aligned with x_0
    t_view = t.view(B, *([1] * (x_0.ndim - 1)))
    x_t = (1 - t_view) * x_0 + t_view * x_1
    # target: u_t = x_1 - x_0 (constant)
    target = x_1 - x_0

    pred = v_net(x_t, t)
    return F.mse_loss(pred, target)

# Full reflow training procedure
def train_with_reflow(v_net, data_loader, num_reflow_rounds=2, device='cuda'):
    """1st round: independent pairs; subsequent rounds: coupled pairs (reflow)"""
    # Round 0: independent pairs (ordinary RF training)
    for batch in data_loader:
        x_1 = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_0 = torch.randn_like(x_1)
        loss = reflow_loss(v_net, x_0, x_1)
        # ... optimizer step

    # Infer sample shape from data_loader (don't rely on custom .dim attribute)
    first_batch = next(iter(data_loader))
    first_x1 = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    sample_shape = tuple(first_x1.shape[1:])  # e.g., (D,) or (C, H, W)

    # Rounds 1, 2, ...: reflow
    for k in range(num_reflow_rounds):
        # 1) Use current v_net to generate coupled pairs
        x_0_pool, x_1_pool = reflow_generate_pairs(
            v_net, num_samples=10_000, sample_shape=sample_shape, device=device
        )
        # 2) Retrain on coupled pairs
        from torch.utils.data import TensorDataset, DataLoader
        coupled_loader = DataLoader(
            TensorDataset(x_0_pool, x_1_pool), batch_size=64, shuffle=True
        )
        for x_0_b, x_1_b in coupled_loader:
            loss = reflow_loss(v_net, x_0_b, x_1_b)
            # ... optimizer step
    return v_net
```

## §8 Production Landscape

### 8.1　Mainstream production few-step models list (2024-2026)

| Model | Distillation method | Base | Step | Resolution | Open source |
|---|---|---|---|---|---|
| **LCM-SDXL / LCM-LoRA** | LCM (consistency on latent) | SDXL | 4-8 | 1024² | ✅ |
| **SDXL-Turbo** | ADD | SDXL | 1 | 512² | ✅ (weights only) |
| **SDXL-Lightning** | progressive + GAN | SDXL | 1/2/4/8 | 1024² | ✅ LoRA |
| **TCD-SDXL** | trajectory CD | SDXL | 4-8 | 1024² | ✅ |
| **DMD2-SDXL** | DMD2 (score gap + GAN) | SDXL | 1/4 | 1024² | ✅ |
| **SD3-Turbo** | LADD | SD3 8B | 4 | 1024² | API only |
| **FLUX.1-schnell** | LADD-style | FLUX 12B | 1-4 | 1024² | ✅ Apache 2.0 |
| **PixArt-LCM / PixArt-α-Lightning** | LCM / Lightning | PixArt-α | 4-8 | 1024² | ✅ |
| **SDXS** | feature alignment + GAN | SDXL | 1 | 512² | ✅ |

> ⚠️ **"Open source" ✅ doesn't mean fully commercially usable** — SDXL-Turbo had a non-commercial early license; FLUX-schnell Apache 2.0 is commercial-OK but FLUX-pro (teacher) is closed-source. Always check the license before production deployment.

### 8.2　Video Distillation status

Video diffusion distillation only started in 2024-2025:

- **AnimateLCM** (Wang et al. 2024): apply LCM to AnimateDiff motion module, 4-step video
- **VideoCrafter-LCM** / **CogVideoX-LCM**: similar applications
- **Hunyuan-Video-Lightning** / **Wan-Lightning**: Lightning-style + temporal-aware D
- **rCM** (arXiv:2510.08431): extend sCM to Wan 2.1 14B / Cosmos-Predict2, 1-4 step 5-second video

> 💡 **Difficulty of video distillation** — for static images D looks at single frames; for video D must look at **temporal coherence** — one approach is D taking a video clip as input (3D conv backbone), another is single-frame D + flow-consistency loss combined. Far less engineering experience than image distillation.

### 8.3　Deployment cheat sheet

| Scenario | Recommendation | Reason |
|---|---|---|
| **Mobile / WebGPU** | SDXL-Turbo / FLUX-schnell 1-step | latency < 200 ms |
| **Server batch (API)** | DMD2-SDXL 4-step / SD3-Turbo | balance of quality + diversity |
| **Secondary development (character LoRA)** | LCM-LoRA / SDXL-Lightning LoRA | doesn't break existing ecosystem |
| **Academic baseline** | sCM / CD / iCT | mathematically clean, reproducible |
| **Real-time video** | rCM / Wan-Lightning | current SOTA |

## §9 Failure modes & selection decisions

### 9.1　Common failure modes

| Phenomenon | Possible cause | Countermeasure |
|---|---|---|
| **Mode collapse** (low output diversity) | 1-step + pure MSE distill; ADD without GAN | add DMD score gap or GAN loss |
| **Saturated colors** (too much red/yellow) | CFG distilled in + too few steps | reduce $w$; use LCM-LoRA 4 step instead of 1 step |
| **High-freq detail blurry** | sCM mode-covering | use rCM with mode-seeking regularizer |
| **Text alignment degenerated** | one-step CFG distill imprecise | use multi-condition $w$ training (LCM style) |
| **EMA collapse** (loss stuck) | EMA decay too high / low | start with 0.9999, monitor spectral norm |
| **Wrong pseudo-Huber c** | too small → L1 dominates (non-smooth); too large → degenerates to L2 | iCT paper $c = 0.00054$ (CIFAR), $c \propto \sqrt{D}$ (D=dim) |
| **JVP NaN** (sCM) | warmup ratio $r$ rises too fast | NCS warmup: $r=0$ for first ~5% steps, then gradually increase |

### 9.2　Selection decision tree

```
Q1: What's the base model?
  ├─ SD 1.5 / SDXL → LCM-LoRA (ecosystem) or SDXL-Lightning
  ├─ SD3 / FLUX → LADD (official SD3-Turbo / FLUX-schnell)
  ├─ DiT / self-trained → sCM (continuous-time) or DMD2
  └─ Pixel-space (CIFAR/ImageNet) → CD / iCT / EDM teacher

Q2: Target NFE?
  ├─ 1-step → DMD2 / ADD / sCM / iCT
  ├─ 2-4 step → LCM / TCD / LADD
  └─ 8 step OK → progressive distillation / EDM Heun sufficient

Q3: Does it need CFG?
  ├─ Yes (text-to-image) → use a CFG-distill-supporting method (LCM / LADD)
  └─ No (unconditional) → CD / DMD usable directly

Q4: Does it need diversity?
  ├─ High (commercial product) → DMD2 / rCM (with reverse-KL / mode-seeking)
  └─ Low (fixed prompt) → ADD / Lightning suffice
```

### 9.3　Evaluation metrics

- **FID** (Fréchet Inception Distance): standard image quality + diversity metric; lower is better
- **CLIP Score** / **CLIPSim**: text-image alignment
- **GenEval** (SD3): structured evaluation of object counts / colors / positions
- **HPSv2 / ImageReward**: human preference scores
- **PRD / Precision-Recall**: measure "fake image quality" and "covering diversity" respectively
- **Step-wise FID**: not just 1-step but 2/4/8-step too

> ⚠️ **The FID pitfall** — FID is **insensitive to mode collapse** — it only computes mean/cov, potentially missing a student that only generates 50% of modes. **Must be paired with Precision-Recall** or IS / Coverage metrics for cross-validation.

## §10 25 frequently-asked interview questions (L1 must-know · L2 intermediate · L3 top lab)

### L1 must-know (potentially asked at any ML / diffusion role)

<details>
<summary>Q1. Why does diffusion need distillation? Can't you just reduce steps?</summary>

- Diffusion sampling has 50-1000 NFE, **network forwards >95% of total latency**, production wants < 1 s real-time

- Direct step reduction (e.g., 50→4) blows up ODE discretization error: 1-step Euler error is $O(\Delta t)$, at 4 steps $\Delta t$ is 12.5× larger, image high-freq details collapse

- The essence of distillation: **retrain a student** that learns to "jump from any $x_t$ directly to $x_0$" (CM) or "match teacher's output distribution" (DMD) or "fool D" (ADD)

Saying only "diffusion is slow" without noting that network forward is the bottleneck; thinking DPM-Solver is enough (10-NFE is its physical limit).

</details>

<details>
<summary>Q2. Write the Consistency Models consistency loss.</summary>

$$\mathcal{L}_\text{CD} = \mathbb{E}\big[d\big(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat x_{t_n}, t_n)\big)\big]$$

- $\theta^-$ is the EMA target

- $\hat x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) v_\phi(x_{t_{n+1}}, t_{n+1})$ from one teacher ODE step

- $d$ = L2 or LPIPS

Confusing EMA target with stop-gradient (the former is learnable, the latter is just stop-grad); forgetting boundary needs EDM precond.

</details>

<details>
<summary>Q3. Differences between CD and CT?</summary>

- **CD (Consistency Distillation)**: has teacher diffusion, uses it for one-step ODE to compute $\hat x_{t_n}$

- **CT (Consistency Training)**: no teacher, uses $\hat x_{t_n} = x_0 + t_n \epsilon$ (same epsilon, different noise levels)

- Original CT had quality far below CD (CIFAR FID 8.7 vs 3.55); iCT brought CT to 2.83 via pseudo-Huber + lognormal sigma + curriculum, **beating CD**

Saying only "CT doesn't need teacher" without iCT improvements; thinking CD is always better than CT (counterexample: iCT).

</details>

<details>
<summary>Q4. What is the core idea of DMD?</summary>

- Treat student $G_\theta(z) \to x$ as a direct generator

- Optimize reverse-KL: $\text{KL}(p_\text{fake}^\theta \| p_\text{real})$, gradient = score gap × ∂G/∂θ

  $$\nabla_\theta \text{KL} = -\mathbb{E}[(s_\text{real} - s_\text{fake}) \cdot \partial G_\theta]$$

- $s_\text{real}$ = teacher diffusion (frozen), $s_\text{fake}$ = fake diffusion trained on $G_\theta$'s outputs

Saying only "DMD is distribution matching" without writing the gradient; not knowing $s_\text{fake}$ is also a diffusion model.

</details>

<details>
<summary>Q5. Fundamental difference between DMD and GAN?</summary>

- GAN uses discriminator for **binary signal** (real/fake), low sample efficiency

- DMD uses **score gap = ∇log(p_real/p_fake)** for **dense vector-field signal**, telling student where each point should move

- Physical intuition: the score gap is the "force" pushing the student from $p_\text{fake}$ toward $p_\text{real}$

- DMD2 actually adds GAN loss as a fidelity auxiliary

Not knowing the physical meaning of score gap; thinking DMD is just a GAN variant.

</details>

<details>
<summary>Q6. Why does ADD (SDXL-Turbo) use DINOv2 as discriminator?</summary>

- Ordinary GAN training D from scratch is **unstable for 1-step generators** (mode collapse / won't train)

- DINOv2 provides "pretrained high-level perceptual features", anchors the discrimination problem in strong semantic space

- Multiple layer heads + hinge loss for training stability

- Also saves D training cost (D backbone frozen, only 1×1 conv heads trained)

Saying only "DINOv2 is useful" without explaining why not from scratch; not knowing heads are multi-layer.

</details>

<details>
<summary>Q7. Core differences between LCM and CM?</summary>

- **Space**: LCM in VAE latent space (64× compute savings); CM in pixel space

- **CFG**: LCM feeds guidance scale $w$ as extra condition $f_\theta(x_t, t, c, w)$, **no double forward at inference**; CM paper doesn't address CFG

- **Skipping-step distillation**: LCM uses $k$-step skipped teacher for faster convergence

Confusing LCM with LCM-LoRA (the latter packages LCM as a LoRA adapter).

</details>

<details>
<summary>Q8. Rectified Flow's reflow algorithm?</summary>

1. Train $v_\theta^{(1)}$ with independent pairs $(x_0, x_1) \sim p_0 \otimes p_\text{data}$

2. Run ODE with $v_\theta^{(1)}$ to generate coupled pairs $(x_0, x_1^{(1)})$

3. Retrain $v_\theta^{(2)}$ on coupled pairs; new trajectory is straighter

- **Transport-cost-non-increasing theorem**: each reflow does not increase total transport cost

- 1-2 rounds of reflow makes 1-step Euler comparable to 50-step

Saying only "reflow makes trajectories straight" without deriving the transport cost monotonicity; forgetting InstaFlow is reflow applied to SD.

</details>

<details>
<summary>Q9. Method differences between SDXL-Turbo and SD3-Turbo / FLUX-schnell?</summary>

- **SDXL-Turbo (ADD)**: DINOv2 pixel-space discriminator + teacher MSE distillation

- **SD3-Turbo / FLUX-schnell (LADD)**: move D to latent space, use teacher MM-DiT intermediate-layer features as D backbone, **supports high resolution + high-param base**

- ADD is limited by DINOv2 input resolution (≤ 518); LADD has no such limit

- FLUX-schnell is the RF version of LADD

Not knowing LADD is the latent version of ADD; thinking FLUX-schnell is ordinary CM distillation.

</details>

<details>
<summary>Q10. Why is LCM-LoRA's ecosystem value large?</summary>

- LCM-trained "weight diff" $\Delta\theta = \theta_\text{LCM} - \theta_\text{SD}$ can be parametrized as LoRA ($r \in [8, 64]$)

- User's existing SD 1.5 / SDXL fine-tunes (DreamShaper / character LoRAs) **don't need retraining**; just attach LCM-LoRA for 4-step generation

- Ecosystem side: SD has tens of thousands of fine-tunes; LCM-LoRA is **the only acceleration solution that doesn't break the existing ecosystem**

- Low training cost (~30 A100 hours / SDXL)

Saying only "LCM-LoRA is the LoRA version of LCM" without ecosystem significance; forgetting LCM-LoRA's training cost is much smaller than LCM.

</details>

### L2 intermediate (research-oriented · need to know diffusion training details)

<details>
<summary>Q11. What are iCT's four improvements over CT? Why can EMA be dropped?</summary>

Four changes:

1. **No EMA**: directly use stop_grad for target

2. **Pseudo-Huber loss**: $\sqrt{\|a-b\|^2 + c^2} - c$ replaces LPIPS, adaptively robust

3. **Lognormal noise schedule**: $\log\sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$ replaces uniform

4. **Step-count curriculum**: $N$ grows from 10 to 1280

**Why EMA can be dropped**: original CT's EMA prevents "the network's output collapsing to trivial $f \equiv 0$ when differentiating against itself". With pseudo-Huber + lognormal sigma, the loss surface is more "convex" (the small-residual region dominates), and stop_grad alone suffices to prevent collapse.

Memorizing the changes without knowing why; thinking EMA is always required (still a misconception).

</details>

<details>
<summary>Q12. Why can the sCM TrigFlow parametrization simultaneously simplify EDM precond / PF-ODE / CM?</summary>

$$x_t = \cos(t) x_0 + \sin(t) z,\; t \in [0, \pi/2]$$

- **EDM precond**: $D_\theta = \cos(t) x_t - \sin(t) F_\theta$, boundary automatically satisfied ($D = x_0$ at $t=0$)

- **PF-ODE**: $dx_t/dt = -\sin(t) x_0 + \cos(t) z$, clean

- **CM**: consistency function $f_\theta = \cos(t) x_t - \sin(t)(\sigma_d F_\theta)$, same form as EDM

- Key: $\cos^2 + \sin^2 = 1$ (variance preservation), and $d\cos/dt = -\sin$ gives a "natural" ODE term

Saying only "sin cos are simple" without explaining why four things "happen to" simplify; not knowing $\sigma^2 + \alpha^2 = 1$ is the VP condition.

</details>

<details>
<summary>Q13. What is sCM's NCS warmup? Why is it needed?</summary>

- NCS = Noise → Consistency → Score (warmup order)

- Early training $r \approx 0$, sCM loss degenerates to standard score matching (learn $F_\theta \approx \epsilon$)

- Gradually increase $r$; consistency term (JVP) takes over

- **Without warmup, directly $r = 1$**: network hasn't learned the score yet, JVP is a noise direction, training NaN

- Similar to "train D first, then alternate" in GAN training: establish base representation first, then increase difficulty

Saying only warmup is a "training trick" without knowing score precedes consistency; not knowing JVP without convergence leads to NaN.

</details>

<details>
<summary>Q14. What did DMD2 change vs DMD? Why are these changes important?</summary>

Three changes:

1. **Drop the regression loss**: DMD v1 required pre-generated teacher pairs (expensive + mode-limited); DMD2 relies purely on score gap + GAN

2. **Add GAN loss**: discriminator sees real data + student outputs, providing high-freq detail supervision

3. **Multi-step student**: at training simulate the $K$-step inference trajectory so the same weights support 1/2/4-step

**Importance**:

- Drop regression → data scale unlocked (no longer dependent on teacher pairs)
- Add GAN → complementary to DMD score gap (score gives distribution-level signal, GAN gives sample-level fidelity)
- Multi-step → production flexibility (same model can switch 1-step / 4-step)

Not knowing why multi-step is needed ("isn't 1-step enough?"); forgetting GAN in DMD2 is auxiliary rather than main loss.

</details>

<details>
<summary>Q15. Two-stage CFG distillation flow?</summary>

**Stage 1 - Guidance distillation** (Meng 2023): train $\tilde\epsilon_\theta(x, c, w)$, feed $w$ as a condition —

$$\mathcal{L}_\text{guide} = \|\tilde\epsilon_\theta(x_t, c, w) - \tilde\epsilon^*(x_t, c, w)\|^2$$

where $\tilde\epsilon^* = (1+w) \epsilon_\theta(x, c) - w \epsilon_\theta(x, \emptyset)$ is the teacher's two-forward CFG output. Student does one forward.

**Stage 2 - Step distillation**: on top of stage 1, stack progressive distillation, compressing 32 → 4/2/1 step. LCM does stage 1 + 2 simultaneously.

Knowing CFG distillation but not writing two stages; not knowing LCM-LoRA's $w$-condition comes from this.

</details>

<details>
<summary>Q16. Role of EDM preconditioning in CM?</summary>

$$f_\theta(x, \sigma) = c_\text{skip}(\sigma) x + c_\text{out}(\sigma) F_\theta(c_\text{in} x, c_\text{noise})$$

Specific values (Song 2023):

$c_\text{skip} = \sigma_d^2 / ((\sigma - \sigma_\min)^2 + \sigma_d^2)$, $c_\text{out} = \sigma_d (\sigma - \sigma_\min) / \sqrt{\sigma_d^2 + \sigma^2}$

**Role**:

1. **Boundary automatically satisfied**: at $\sigma = \sigma_\min$, $c_\text{skip} = 1, c_\text{out} = 0$, so $f(x, \sigma_\min) = x$ (identity)

2. **Unit-variance**: makes $F_\theta$ input/output variance independent of $\sigma$, stabilizing training

Not knowing $c_\text{skip}(\sigma_\min) = 1$ is the key to boundary; confusing EDM precond with score-based reparam.

</details>

<details>
<summary>Q17. If ADD's distillation loss uses pixel MSE rather than score gap, what's the problem?</summary>

- Pixel MSE is **mode-covering** + **blurry**: student output = teacher mean, losing details

- Need to add GAN loss to compensate for high-freq → ADD must have GAN (unlike DMD which can be pure score gap)

- That's why ADD's distill loss is only an "anchor" (preventing severe mode collapse); the main battle is GAN

- Compare DMD: score gap → dense per-pixel gradient, can generate without GAN (DMD2 adds GAN for further gains)

Saying only "MSE is blurry" without why; thinking ADD doesn't need GAN to work.

</details>

<details>
<summary>Q18. What capabilities does CTM add vs CM?</summary>

- CM: only learns $f(x_t, t) \to x_0$ (trajectory endpoint)

- CTM: learns $G(x_t, t, s)$, **any $s < t$ can be jumped to**

- Practical benefits:

  - Runtime-selectable inference step count (CM is fixed)
  - Controllable intermediate states (suitable for image-to-image / inpainting)
  - Training + score-matching auxiliary loss to prevent triviality

- FID: CIFAR 1-step 1.73 / ImageNet 64 1.92 (SOTA)

Saying only "CTM is the trajectory version of CM" without why "any s" is useful; confusing CTM and TCD (the latter is an LCM improvement).

</details>

<details>
<summary>Q19. How to prove reflow's transport-cost monotonicity?</summary>

**Setup**: consider independent pairs $(x_0, x_1) \sim p_0 \otimes p_1$, initial cost $C^{(0)} = \mathbb{E}\|x_1 - x_0\|^2$.

**Reflow**: run ODE with $v_\theta^{(1)}$ to get coupled $(x_0, x_1^{(1)})$, cost $C^{(1)} = \mathbb{E}\|x_1^{(1)} - x_0\|^2$.

**Key observations**:

- $x_1^{(1)} = x_0 + \int_0^1 v_\theta^{(1)}(x_t, t)\, dt$

- In $L^2$, $\|x_1^{(1)} - x_0\| = \|\int v\, dt\| \le \int \|v\|\, dt$ (Cauchy-Schwarz)

- And $v_\theta^{(1)}$'s training objective is to minimize $\mathbb{E}\|v - (x_1 - x_0)\|^2$ → in expectation $\|v\| \approx \|x_1 - x_0\|$

- Rigorous theorem (Liu 2022 Theorem 3.6): $C^{(k+1)} \le C^{(k)}$ (from the OT view, reflow does not increase transport cost)

Intuition: **straight lines are OT solutions** ⇒ repeated reflow pushes toward the OT solution.

Saying only "trajectories become straight" without writing transport cost; not knowing the Cauchy-Schwarz intuition.

</details>

<details>
<summary>Q20. How do you evaluate the distilled student? Is FID alone enough?</summary>

**Why FID isn't enough**:

- FID only computes Inception feature mean + cov, **insensitive to mode collapse** (a student generating 50% of modes may still have low FID)

- Insensitive to high-freq details (Inception backbone pools heavily at 224×224)

**Necessary auxiliary metrics**:

- **Precision / Recall** (Kynkäänniemi 2019): respectively measure "fake image quality" and "covering diversity"

- **CLIP Score**: text-image alignment

- **HPSv2 / ImageReward / PickScore**: human preference

- **Step-wise FID**: look at 1/2/4/8-step, avoid only optimizing for 1-step

- **Mode count / coverage**: directly count how many real clusters the generation covers

Saying only "FID is enough"; forgetting human-preference evaluation is mandatory for production deployment.

</details>

### L3 top-lab questions (research depth · derivation required)

<details>
<summary>Q21. Derive the continuous-time form of the consistency loss from the PF-ODE.</summary>

**PF-ODE**: $dx_t/dt = v_\phi(x_t, t)$ (teacher).

**Consistency definition**: $f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t)$ along the same ODE trajectory.

**First-order Taylor**:

$$f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t) + \Delta t \cdot \frac{d f_\theta}{dt} + O(\Delta t^2)$$

where $\frac{d f_\theta}{dt} = \partial_t f_\theta + (\nabla_x f_\theta)^\top \cdot \dot x_t = \partial_t f_\theta + (\nabla_x f_\theta)^\top v_\phi$ (chain rule + PF-ODE substitution).

**Continuous-time consistency loss**:

$$\mathcal{L}_\text{cont}(\theta) = \mathbb{E}\!\left[\Big\|\partial_t f_\theta(x_t, t) + \nabla_x f_\theta(x_t, t) \cdot v_\phi(x_t, t)\Big\|^2\right]$$

**Discretization** (original CM): use $\hat x_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1}) v_\phi(\cdot)$ as teacher Euler, and $f_{\theta^-}$ as target —

$$\mathcal{L}_\text{CD} \approx \mathbb{E}\|f_\theta(x_{t_{n+1}}, t_{n+1}) - f_{\theta^-}(\hat x_{t_n}, t_n)\|^2$$

Knowing only the discrete loss without deriving the continuous form; confusing $\partial_t$ and $d/dt$ (the former is partial, the latter is total).

</details>

<details>
<summary>Q22. Physical meaning of DMD's two scores? Why must we use fake score and not zero?</summary>

**Physical meaning**:

- $s_\text{real}(x, t) = \nabla_x \log p_\text{real}(x_t)$: the "force" pushing $x_t$ toward real data

- $s_\text{fake}(x, t) = \nabla_x \log p_\text{fake}(x_t)$: score of student's current output distribution

- Difference $s_\text{real} - s_\text{fake} = \nabla_x \log(p_\text{real}/p_\text{fake})$: reverse-KL gradient direction

**Why fake score is necessary**:

- If only $s_\text{real}$ (i.e., $s_\text{fake} \equiv 0$): equivalent to pushing student toward the "modes of $p_\text{real}$" — **mode collapse**

- $s_\text{fake}$ provides the signal "no need to push further to already-covered positions", similar to GAN's D providing contrastive feedback

- Mathematically: $\mathbb{E}_{p_\text{fake}}[s_\text{real} - s_\text{fake}]$ is a Stein discrepancy, the correct distribution-matching signal

**Implementation**:

- $s_\text{real}$ = teacher diffusion (frozen)
- $s_\text{fake}$ = a small diffusion model, **trained with DSM on $G_\theta$'s current outputs**, jointly trained with $G_\theta$

Saying only "DMD uses score" without distinguishing the two roles; not knowing $s_\text{fake}$ needs joint training.

</details>

<details>
<summary>Q23. What's the essence of the scale difference between ADD and LADD? Why can't ADD scale to SD3 8B / FLUX 12B?</summary>

**ADD bottlenecks**:

1. **DINOv2 input resolution**: ADD uses DINOv2 base (518²) as D backbone; above this resolution requires patching / downsampling, limiting 1024² input

2. **Pixel-space distill**: `MSE(G(z), teacher_ode(z))` requires VAE decode, **back-prop through VAE is expensive and unstable**

3. **Discriminator capacity**: DINOv2 ViT-L 1B params is far less than SD3 8B / FLUX 12B's base, D expressiveness insufficient

**LADD's solutions**:

1. **Latent space**: D directly on VAE latent (128×128×16 for SD3), resolution-agnostic

2. **Teacher's own MM-DiT blocks as D backbone**: extract SD3's transformer blocks and fine-tune as D, **capacity automatically matches base scale**

3. **Score distill in latent**: avoids VAE back-prop

**Results**: SDXL-Turbo (2.6B SDXL ADD) at 1024² is the ADD limit; SD3-Turbo (8B LADD) / FLUX-schnell (12B LADD-style) need LADD for stable training.

Saying only "LADD is in latent space" without why ADD can't scale; forgetting DINOv2 resolution is a hard cap.

</details>

<details>
<summary>Q24. Mathematical relationship between Flow-OPD (2026 arXiv:2605.08063) and DMD?</summary>

> 📍 **Clarification**: Flow-OPD is primarily a multi-reward RL alignment paper, somewhat tangential to this tutorial's few-step inference distillation thread; it appears here because the name contains "Distillation". See [diffusion_post_training_tutorial.md](diffusion_post_training_tutorial.md) for detailed discussion.

**DMD**: reverse-KL gradient (on student output distribution), single teacher, single objective = match teacher distribution.

**Flow-OPD**: on-policy distillation with multiple **reward-specific** teachers (each GRPO fine-tuned as a specialist on one reward); it is an **alignment paper** (multi-reward alignment), not an inference distillation paper.

**Saying "DMD reduces to OPD" is wrong**: DMD's reverse-KL and OPD's multi-teacher vector-field weighting have **different mathematical objectives** — one is distribution matching, the other is reward-aware policy supervision. Their priorities differ (**single-teacher distribution match vs multi-reward alignment**) with no reduction relationship. In interviews **don't** say "DMD is a special case of OPD" or vice versa — no reliable mathematical basis.

**Practical significance** (safer phrasing):
- **DMD is more suited to few-step inference** (single objective: match teacher distribution)
- **Flow-OPD is more suited to multi-reward alignment** (multi-reward alignment + on-policy training)
- They solve different problems and are not substitutes; for detailed alignment content see [`diffusion_post_training_tutorial.md`](diffusion_post_training_tutorial.md)

Treating Flow-OPD as just "another inference distillation" is a confusion — its multi-reward / RL nature is central; similarly, don't claim "reduction to DMD" as established math.

</details>

<details>
<summary>Q25. Design a distillation scheme that runs 1024² video in 4 steps + preserves temporal coherence. Give the loss and D design.</summary>

**Setup**:

- Teacher: 50-step video diffusion (e.g., Wan 2.1 14B, Rectified Flow)
- Student: 4-step video generator $G_\theta(z_{1:T}, c)$
- Target: 1024² × 5 sec

**Loss combination** (rCM-style + LADD-style):

$$\mathcal{L}_\text{total} = \underbrace{\mathcal{L}_\text{sCM}^\text{trig}}_{\text{video CM, JVP-based}} + \lambda_1 \cdot \underbrace{\mathcal{L}_\text{score-reg}}_{\text{mode-seeking via score gap}} + \lambda_2 \cdot \underbrace{\mathcal{L}_\text{adv}^\text{video}}_{\text{temporal D}}$$

**Video Discriminator design**:

- **Backbone**: teacher's own 3D MM-DiT block (latent space, avoid VAE decode)

- **Two heads**:
  - **Spatial head**: single-frame latent → real/fake signal (image quality)
  - **Temporal head**: consecutive $k$-frame latent stack → real/fake (motion realism)

- **Optical flow consistency loss** (auxiliary):
  $$\mathcal{L}_\text{flow} = \mathbb{E}\|f_\text{flow}(\hat x_{t}, \hat x_{t+1}) - f_\text{flow}(x_t^\text{real}, x_{t+1}^\text{real})\|$$

**Training tricks**:

- **Multi-stage**: pretrain on static images ($T = 1$) → add temporal D → fine-tune full video
- **Curriculum on T**: train short clips first ($T = 8$ frame) → longer clips ($T = 80$ frame)
- **EMA on G**: prevent student outputs from drifting across steps

**Evaluation**:

- VBench (static quality + dynamic quality across 16 dimensions)
- FVD (Fréchet Video Distance)
- Human comparisons (rCM-style)

**Compared baseline**: rCM is already approaching this on Wan 2.1 14B — production-grade direction, still developing fast in 2026.

Need to fuse "image distillation + temporal supervision + large base"; using a single D on single frames causes motion collapse; using only score-gap without GAN gives blurry details.

</details>

## §A Appendix: References

**Consistency Models family**:

- Song et al. 2023, "Consistency Models", ICML 2023, [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Song & Dhariwal 2023, "Improved Techniques for Training Consistency Models" (iCT), [arXiv:2310.14189](https://arxiv.org/abs/2310.14189)
- Lu & Song 2024, "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models" (sCM / TrigFlow), ICLR 2025, [arXiv:2410.11081](https://arxiv.org/abs/2410.11081)
- Kim et al. 2023, "Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion" (CTM), ICLR 2024, [arXiv:2310.02279](https://arxiv.org/abs/2310.02279)
- Luo et al. 2023, "Latent Consistency Models" (LCM), [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
- Luo et al. 2023, "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module", [arXiv:2311.05556](https://arxiv.org/abs/2311.05556)
- Zheng et al. 2024, "Trajectory Consistency Distillation" (TCD), [arXiv:2402.19159](https://arxiv.org/abs/2402.19159)
- "Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency" (rCM), [arXiv:2510.08431](https://arxiv.org/abs/2510.08431) (rCM acronym verified)

**Distribution Matching Distillation**:

- Yin et al. 2024, "One-step Diffusion with Distribution Matching Distillation" (DMD), CVPR 2024, [arXiv:2311.18828](https://arxiv.org/abs/2311.18828)
- Yin et al. 2024, "Improved Distribution Matching Distillation for Fast Image Synthesis" (DMD2), NeurIPS 2024, [arXiv:2405.14867](https://arxiv.org/abs/2405.14867)

**Adversarial Distillation**:

- Sauer et al. 2023, "Adversarial Diffusion Distillation" (ADD / SDXL-Turbo), [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
- Sauer et al. 2024, "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation" (LADD / SD3-Turbo), [arXiv:2403.12015](https://arxiv.org/abs/2403.12015)
- Lin et al. 2024, "SDXL-Lightning: Progressive Adversarial Diffusion Distillation", [arXiv:2402.13929](https://arxiv.org/abs/2402.13929)

**Flow / Rectified Flow**:

- Liu, Gong & Liu 2022, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023, [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Liu et al. 2023, "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation", ICLR 2024, [arXiv:2309.06380](https://arxiv.org/abs/2309.06380)
- "Flow-OPD: On-Policy Distillation for Flow Matching Models", [arXiv:2605.08063](https://arxiv.org/abs/2605.08063) (Flow-OPD is primarily a multi-reward RL alignment paper, somewhat tangential to this tutorial's few-step inference distillation thread; see `diffusion_post_training_tutorial.md`)

**CFG / Step Distillation**:

- Meng et al. 2023, "On Distillation of Guided Diffusion Models", CVPR 2023, [arXiv:2210.03142](https://arxiv.org/abs/2210.03142)
- Salimans & Ho 2022, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022, [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)

**Foundations**:

- Ho, Jain & Abbeel 2020, "Denoising Diffusion Probabilistic Models", NeurIPS 2020 (DDPM)
- Song et al. 2021, "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021
- Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM), NeurIPS 2022, [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Lipman et al. 2023, "Flow Matching for Generative Modeling", ICLR 2023

**Production models**:

- Stable Diffusion XL: Podell et al. 2024 ICLR
- Stable Diffusion 3: Esser et al. 2024 ICML
- FLUX.1: Black Forest Labs 2024 (technical report)

**Diffusion / Flow Distillation Cheat Sheet** · main references: Song 2023 (CM), Lu-Song 2024 (sCM), Yin 2024 (DMD/DMD2), Sauer 2023/2024 (ADD/LADD), Liu 2022 (RF)
