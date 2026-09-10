## §0 TL;DR Cheat Sheet

> 💡 **One sentence** — the three methods here share the GRPO-style **group sampling** frame (sample $G$ images per prompt, normalise rewards within the group); they part ways at exactly one point: **how the reward enters the gradient**.

1. **Flow-GRPO** (Liu et al. 2025, arXiv 2505.05470): rewrite the ODE as a marginal-preserving **SDE** so that a per-step Gaussian transition density exists, then run **advantage-weighted PPO-clip policy gradient**. Cost: training rollouts must be SDE (inference can still use the ODE), full trajectories stored, a likelihood per step, discretisation bias.
2. **DGPO** (Luo, Hu, Tang 2025, arXiv 2510.08425, ICLR 2026): **drops policy gradient altogether** and generalises DPO from "a pair of samples" to "a pair of sub-groups" — positive group vs negative group inside one sigmoid; the likelihood is replaced by Diffusion-DPO's **DSM difference**. Hence deterministic **ODE** sampling and only clean images + rewards to store. Reported ~20× faster than Flow-GRPO.
3. **DiffusionNFT** (Zheng, Chen et al. 2025, arXiv 2509.16117, ICLR 2026 Oral): RL on the **forward (noising) process** — a single flow-matching regression loss where positives pull toward the target and negatives push away through a **reflected parameterisation** about the old model. No likelihood, no solver dependence, **CFG-free** inference. Reported up to 25× faster than Flow-GRPO on GenEval.
4. None of the three needs a critic; but they are policy gradient / group-preference logistic / reward-weighted two-branch regression respectively — not "three kinds of policy gradient".
5. **old ≠ ref**: old is the behaviour policy at sampling time (the denominator of the importance ratio; NFT's reflection centre), ref is the KL anchor (Flow-GRPO's KL, DGPO's DSM difference). NFT's core loss needs no fixed ref.
6. Keep the number provenances apart: Flow-GRPO self-reports GenEval 0.63→0.95 (CFG base) and PickScore 21.72→23.31 (with KL; 23.41 without); DGPO 0.97; NFT 0.24→0.98 (**0.24 is the CFG-free base**). DGPO's 3.74 vs 3.66 is **UnifiedReward**, not PickScore.
7. Killer questions: **the sign of the reverse SDE** (time decreasing ⇒ drift $v-\tfrac12 g^2 s$), **why DGPO can cancel $\log Z$** (within-group advantages sum to zero ⇒ positive and negative weights have equal total mass), **why NFT's negative branch carries $(1+\beta)$** (reflection about old).

---

## §1 Intuition: the three roads in one sentence each

### 1.1　The shared starting point: LLM GRPO

GRPO (Shao et al. 2024, DeepSeekMath) removes PPO's critic: sample $G$ responses per prompt and use the group mean as the baseline:

$$\hat A_i = \frac{R_i - \operatorname{mean}_j R_j}{\operatorname{std}_j R_j}$$

then PPO-clip. Moving to diffusion, "a response" becomes "an image" and the reward is a rule verifier such as GenEval, OCR accuracy, or a human-preference model such as PickScore. The three methods share the same collection frame: $P$ prompts × $G$ images × scoring × within-group normalisation (the solver, the reward normalisation and how the collector is updated differ). When all rewards in a group coincide, the std is zero and the advantage is set to zero by convention.

### 1.2　The fork: which quantity carries the advantage

| Method | How the reward enters the gradient | Needed at training time | Sampler |
| --- | --- | --- | --- |
| Flow-GRPO | weights the per-step transition $\log p_\theta(x_{k+1}\mid x_k)$ (policy gradient) | full trajectories + per-step old log-prob | training rollouts must be SDE |
| DGPO | reference-relative DSM differences of the positive and negative groups inside one sigmoid (preference learning) | clean images + rewards, re-noised | any (ODE works) |
| DiffusionNFT | weights of two flow-matching regression branches, positive / negative (supervised-learning form) | clean images + rewards, re-noised | any (black-box) |

### 1.3　Why the ODE blocks Flow-GRPO but not the other two

ODE sampling is **not** deterministic as a whole — different initial noise gives different samples. What blocks it is something else: the per-step transition $x_k\to x_{k+1}$ is deterministic (a Dirac), so no density exists and PPO's importance ratio $p_\theta/p_\text{old}$ is undefined. Flow-GRPO's answer is to swap the ODE for an SDE with the same marginals; DGPO's and NFT's answer is to **never use the likelihood of the sampling path at all** — at training time they re-noise clean images with $x_t = (1-t)x_0 + t\epsilon$, i.e. the forward process $q(x_t\mid x_0)$, which knows nothing about the sampler.

> ⚠️ **Don't say "ODEs can't explore"** — the initial noise already explores. What's missing is a computable stochastic per-step density, plus extra exploration at intermediate steps.

---

## §2 Convention (used throughout)

### 2.1　Rectified-flow notation

$$x_t = (1-t)\,x_0 + t\,\epsilon,\qquad t\in[0,1],\quad x_0\sim p_\text{data},\ \epsilon\sim\mathcal N(0,I)$$

- $t=0$ is the clean image, $t=1$ pure noise; **generation runs $t: 1\to 0$**.
- The training pair's target is $v = \epsilon - x_0$; the network learns the conditional expectation $v^*(x_t,t,c) = \mathbb E[\epsilon - x_0 \mid x_t, t, c]$, **not** any single pair's random target.
- The SDE diffusion coefficient is written $g(t)$, kept separate from the forward-noising coefficient $\sigma_\text{path}(t) = t$. Each method has its own $\beta$ with its own meaning — **never compare them across methods**.

### 2.2　The conversion table (memorise)

From $x_t = (1-t)x_0 + t\epsilon$ and $v = \epsilon - x_0$:

$$\hat x_0 = x_t - t\,v_\theta,\qquad \hat\epsilon = x_t + (1-t)\,v_\theta,\qquad s_\theta(x_t) := -\frac{\hat\epsilon}{t}$$

Check $\hat\epsilon$: $x_t + (1-t)v = (1-t)x_0 + t\epsilon + (1-t)\epsilon - (1-t)x_0 = \epsilon$. The score uses the **predicted conditional mean** $\hat\epsilon$, not the true $\epsilon$ (which is unknown). $s_\theta$ equals the true $\nabla_x\log p_t$ only when $v_\theta = v^*$ — that is exactly the “marginal preservation is not inherited strictly” point of §3.2 and Q24.

### 2.3　The three roles a model can play

| Name | Which | Job | Flow-GRPO | DGPO | NFT |
| --- | --- | --- | --- | --- | --- |
| **policy** $\theta$ | being updated | optimised | ✅ | ✅ | ✅ |
| **old** $\theta_\text{old}$ | at sampling time | ratio denominator / reflection centre | ✅ (rollout) | collector (EMA allowed) | ✅ (EMA) |
| **ref** $\theta_\text{ref}$ | frozen | anchor for KL / DSM difference | ✅ (KL term) | ✅ (DSM difference) | ❌ (core loss needs none) |

---

## §3 Flow-GRPO: turn the ODE into an SDE, then run PPO

Liu et al. 2025, *Flow-GRPO: Training Flow Matching Models via Online RL*, arXiv 2505.05470 (Kling / CUHK / Tsinghua).

### 3.1　The marginal-preserving SDE: derivation

**Goal**: an SDE whose marginal $p_t$ at every $t$ equals that of the ODE $\dot x = v$, but whose per-step transition is Gaussian.

The cleanest derivation uses the **forward generation clock** $u = 1 - t$ ($u$ from 0 to 1 as $t$ goes 1 to 0). The ODE in $u$ is $dX_u = -v\,du$. Add a pair of terms that cancel each other:

$$dX_u = \big[-v + \tfrac12 g^2 s\big]\,du + g\,dW_u$$

In the Fokker–Planck equation the $\tfrac12 g^2 s$ in the drift contributes $-\tfrac12 g^2\nabla\cdot(p\,s)$ and the diffusion term contributes $+\tfrac12 g^2\Delta p$; since $p\,s = p\nabla\log p = \nabla p$, they cancel exactly, leaving the ODE's density evolution. So the marginals are unchanged.

Back on the $t$ clock ($dt = -du$):

$$dx_t = \big[v - \tfrac12 g^2 s\big]\,dt + g\,d\bar W_t \qquad (t \text{ decreasing})$$

This is the paper's Eq. 7. Substituting §2.2's $s = -\hat\epsilon/t = -(x_t + (1-t)v)/t$ gives Eq. 8:

$$dx_t = \Big[v + \frac{g^2}{2t}\big(x_t + (1-t)v\big)\Big]dt + g\,d\bar W_t$$

> ⚠️ **The sign depends on the time direction** — with time **increasing** the marginal-preserving drift is $v + \tfrac12 g^2 s$; with time **decreasing** (the reverse SDE used for generation) it is $v - \tfrac12 g^2 s$. The old edition of the post-training tutorial wrote the generation SDE with $+\tfrac12\sigma^2 s$, which is wrong.

### 3.2　Discretisation: use a positive step throughout

Let $h = t_k - t_{k+1} > 0$. Euler–Maruyama:

$$x_{k+1} = x_k - h\Big[v_\theta + \frac{g^2}{2t_k}\big(x_k + (1-t_k)v_\theta\big)\Big] + g\sqrt{h}\,\xi,\qquad \xi\sim\mathcal N(0,I)$$

The paper's Eq. 9 is written with a signed $\Delta t$; copying it verbatim leads to "square root of a negative step" confusion — write it with $h>0$.

**Noise schedule**: $g(t) = a\sqrt{t/(1-t)}$, default $a = 0.7$. The denominator vanishes at $t=1$; the official implementation handles the first step with neighbouring time points rather than computing an infinity.

**Marginal preservation is a continuous-time property**: it requires $v$ and $s$ to correspond to the same family of marginals; learning error, a CFG-modified field, and finite-step discretisation do not automatically inherit strict equality. This is why the denoising reduction of 3.5 is an empirical observation, not a theorem.

### 3.3　Per-step policy, log-prob, ratio

**After Euler discretisation** the per-step transition is an isotropic Gaussian (the exact finite-time transition of a general nonlinear SDE is not): $p_\theta(x_{k+1}\mid x_k) = \mathcal N\big(\mu_\theta(x_k),\ g^2 h\,I\big)$, where $\mu_\theta$ is the line in 3.2 with the noise term removed.

$$\log p_\theta(x_{k+1}\mid x_k) = -\frac{\lVert x_{k+1} - \mu_\theta\rVert^2}{2g^2h} - \frac d2\log(2\pi g^2 h)$$

**summed** over all $d$ latent dimensions. The official code averages the log-prob per dimension, which yields $\rho^{1/d}$ (the $d$-th root of the density ratio), not a constant multiple of the ratio — keep this apart from the paper's density formula.

The denominator of the importance ratio is **old** (the rollout policy), not ref:

$$\rho_k = \exp\big(\log p_\theta(x_{k+1}\mid x_k) - \log p_\text{old}(x_{k+1}\mid x_k)\big)$$

### 3.4　Loss: PPO-clip + closed-form KL

$$\mathcal L_\text{Flow-GRPO} = -\,\mathbb E\Big[\min\big(\rho_k \hat A,\ \operatorname{clip}(\rho_k, 1-\varepsilon_c, 1+\varepsilon_c)\hat A\big)\Big] + \beta_\text{KL}\,\mathbb E\big[\mathrm{KL}(p_\theta\Vert p_\text{ref})\big]$$

The $\min$ stays for negative advantages — no switch to $\max$ (this is standard PPO). $\hat A$ is the within-group normalised reward of the final image, shared by every step of the trajectory.

**The KL is closed-form** — two Gaussians with the same covariance leave only the mean difference:

$$\mathrm{KL}(p_\theta\Vert p_\text{ref}) = \frac{\lVert\mu_\theta - \mu_\text{ref}\rVert^2}{2g^2h}$$

Substituting the drift of 3.2, $\mu_\theta - \mu_\text{ref} = -h\Big(1 + \frac{g^2(1-t)}{2t}\Big)(v_\theta - v_\text{ref})$, hence

$$\mathrm{KL} = \frac h2\Big(\frac1g + \frac{g(1-t)}{2t}\Big)^2\lVert v_\theta - v_\text{ref}\rVert^2$$

The coefficient is non-negative. This is a **conditional per-step KL**; summing along the chain gives the path KL, which is **not** the KL between final-image distributions. The old edition said "KL via the K3 estimator" — the paper gives it in closed form, no K3 needed.

### 3.5　Denoising reduction

Sampling $G$ trajectories of $T$ steps costs $\propto GT$. Flow-GRPO trains with $T_\text{train} = 10$ and evaluates with $T_\text{infer} = 40$; the paper reports "over 4×" speed-up — from a no-KL ablation curve going 40→10 steps, not a universal law. Why fewer training steps don't hurt: the continuous marginal is step-count independent, and RL learns a directional correction of $v_\theta$ that is weakly coupled to the step count — an empirical observation plus an approximation, not a strict equivalence.

### 3.6　Gradient boundaries (a frequent follow-up)

- **Fixed**: the rollout's state sequence, old log-probs, rewards, advantages.
- **Recomputed**: the current policy's log-prob at those states.
- **stop-grad**: the ref outputs.
- "Storing the whole trajectory" stores states, **not** the rollout's backward graph.

Geometric meaning of a positive advantage: push the transition mean $\mu_\theta$ toward the next state $x_{k+1}$ that was actually sampled. Translating that into "which way $v_\theta$ moves" goes through the negative time step and the full drift coefficient — you cannot just say "$v_\theta$ points at the next latent".

### 3.7　Results and costs

On SD3.5-M (each row is a **separately trained** model for that reward):

| Reward | base | Flow-GRPO | $\beta_\text{KL}$ |
| --- | --- | --- | --- |
| GenEval | 0.63 | **0.95** | 0.04 |
| OCR | 0.59 | **0.92** | 0.04 |
| PickScore | 21.72 | **23.31** | 0.01 |

$G = 24$, $a = 0.7$. The cost list: training rollouts must use SDE sampling (evaluation switches the extra noise off and uses the ODE); trajectory buffer $O(PGTd)$; three forward passes (old / policy / ref) per step; two layers of approximation (PPO surrogate + discretisation).

> 💡 **Related work, three lines only** — **DanceGRPO** extends GRPO to image/video and to diffusion/RF backbones; **MixGRPO / Flow-GRPO-Fast** run SDE + GRPO updates only inside a time window and use the ODE elsewhere (so "the SDE must cover the whole chain" is an over-generalisation); **GRPO-Guard** shows the importance-ratio distribution drifts with the timestep, weakening the clip, and fixes it with RatioNorm and gradient re-weighting (see §8).

---

## §4 DGPO: no policy gradient — generalise DPO to groups

Luo, Hu, Tang 2025, *Reinforcing Diffusion Models by Direct Group Preference Optimization*, arXiv 2510.08425, ICLR 2026 (CUHK-SZ / HKUST-GZ).

### 4.1　Motivation

Every cost of Flow-GRPO comes from "needing a stochastic per-step density". DGPO asks: can we keep the advantage signal of group sampling but **replace policy gradient with preference learning**? DPO already shows a pair of samples suffices; replacing "a pair of samples" with "a pair of sub-groups" is DGPO.

### 4.2　From DPO to group DPO

The KL-regularised RL optimum $p^*(x_0\mid c) \propto p_\text{ref}(x_0\mid c)\,e^{R(x_0,c)/\beta}$ inverts into an implicit reward:

$$r_\theta(c, x_0) = \beta\log\frac{p_\theta(x_0\mid c)}{p_\text{ref}(x_0\mid c)} + \beta\log Z(c)$$

DPO puts the difference of two samples' $r_\theta$ into Bradley–Terry, and $\log Z(c)$ cancels in the subtraction. DGPO defines a **group-level reward** as a weighted sum:

$$R_\theta(\mathcal G\mid c) = \sum_{x_0\in\mathcal G} w(x_0)\, r_\theta(c, x_0)$$

Objective:

$$\max_\theta\ \mathbb E\,\log\sigma\big(R_\theta(\mathcal G^+\mid c) - R_\theta(\mathcal G^-\mid c)\big)$$

### 4.3　How the groups are split, and how $\log Z$ cancels

Within the same group of $G$ images for one prompt, normalise $A_i = (R_i - \bar R)/s_R$, then

$$\mathcal G^+ = \{x_0^i: A_i > 0\},\qquad \mathcal G^- = \{x_0^i: A_i \le 0\},\qquad w(x_0^i) = \lvert A_i\rvert$$

**The key step**: because $\sum_i A_i = 0$,

$$\sum_{\mathcal G^+}\lvert A_i\rvert = \sum_{\mathcal G^+} A_i = -\sum_{\mathcal G^-} A_i = \sum_{\mathcal G^-}\lvert A_i\rvert$$

The positive and negative weights have **equal total mass**. Hence $\beta\log Z(c)$ multiplies $\big[\sum_{\mathcal G^+}w - \sum_{\mathcal G^-}w\big] = 0$ and drops out.

> ⚠️ **What balances is weight mass, not head-count** — the two groups usually differ in size (say 1:3). Averaging each side by its head-count breaks the cancellation and changes the objective.

### 4.4　DSM in place of the likelihood

The $\beta\log\frac{p_\theta(x_0\mid c)}{p_\text{ref}(x_0\mid c)}$ inside $r_\theta$ is intractable for diffusion. Follow Diffusion-DPO's route: approximate this terminal log-ratio by the reference-relative path quantity $\beta\,\mathbb E_{q(x_{1:T}\mid x_0)}\log\frac{p_\theta(x_{0:T}\mid c)}{p_\text{ref}(x_{0:T}\mid c)}$ (an approximation, not an identity), and reduce it to a per-timestep **difference of denoising regression errors**. Write

$$d_i = L^\theta_{\text{dsm},i} - L^\text{ref}_{\text{dsm},i}$$

Final loss (compact form of the paper's Eq. 17):

$$\mathcal L_\text{DGPO} = -\,\mathbb E_{t,\epsilon}\log\sigma\Big(-\lambda_t\,\beta\,T\sum_i A_i\, d_i\Big)$$

($A_i$ carries its sign, so "positive group minus negative group" is already inside.) **Sign check**: a preferred sample whose current error drops relative to ref, or a dispreferred one whose current error rises relative to ref, both increase the logit — the whole group shares one logistic gradient coefficient.

Two details:

- DSM is defined on the denoising regression error. Under RF's velocity parameterisation $\lVert\hat x_0 - x_0\rVert^2 = t^2\lVert v_\theta - v\rVert^2$, so the coefficients must follow — you cannot switch the prediction target and still claim the loss is verbatim identical.
- **Jensen's direction**: $-\log\sigma(\mathbb E z) \le \mathbb E[-\log\sigma(z)]$. What is computed at a single timestep is a surrogate; moving the expectation outside is not an exact equality.
- The group **shares $t$ and $\epsilon$**. That is a variance-reduction implementation choice and plays no role in the $\log Z$ proof; with the outer sigmoid nonlinearity, don't recycle the old "changing the noise coupling only changes variance and the objective stays unbiased" line.

### 4.5　Relation to Diffusion-DPO

With $G = 2$ (one positive, one negative, equal weights) DGPO reduces to Diffusion-DPO's form (the scale is absorbed into $\beta$). General DGPO is "**the group difference enters one sigmoid**", not an enumeration of all positive–negative pairs averaged pair by pair.

### 4.6　Why the ODE is fine

At training time the stored clean images are **re-noised** through $q(x_t\mid x_0)$; the likelihood of the sampling path is never computed. So any solver works for collection — the paper uses a 10-step Flow-DPM-Solver. Only clean images, prompts and rewards are stored. A frozen $p_\text{ref}$ is still needed; the collector can be the current policy or an EMA.

### 4.7　Timestep clipping (must know)

Few-step samples carry artefacts (blur etc.), and regression at low-noise timesteps would learn those artefacts as features. So DGPO samples $t$ only from $[t_\min, 1]$. **This truncates the training timestep — it is not PPO's ratio clipping.**

### 4.8　Recipe and results

Paper defaults: $G = 24$, 10-step Flow-DPM-Solver, $\beta = 100$ (tied to the loss-scale convention; not comparable to other methods' $\beta$), collector follows the policy directly for the first 200 steps and then uses EMA decay 0.3.

SD3.5-M (DGPO paper Table 2; Flow-GRPO re-run in the same table):

| Metric | base | Flow-GRPO (as listed in DGPO Table 2) | DGPO |
| --- | --- | --- | --- |
| GenEval | 0.63 | 0.95 | **0.97** |
| OCR | 0.59 | 0.92 | **0.96** |
| PickScore | 21.72 | 23.31 | **23.89** |
| UnifiedReward (preference-trained models) | 3.33 | 3.66 | **3.74** |

"About 20×" is the paper's overall training-efficiency claim; the GenEval curve is separately described as nearly 30× — **not** a 20× faster backward pass.

---

## §5 DiffusionNFT: RL on the forward process

Zheng, Chen, Ye, Wang, Zhang, Jiang, Su, Ermon, Zhu, Liu 2025, *DiffusionNFT: Online Diffusion Reinforcement with Forward Process*, arXiv 2509.16117, ICLR 2026 Oral (Tsinghua / NVIDIA / Stanford).

### 5.1　Origin: NFT for LLMs

Chen et al. 2025, *Bridging Supervised Learning and Reinforcement Learning in Math Reasoning* (arXiv 2505.18116): fine-tuning on positives only (RFT) throws away the information in negatives; NFT decomposes the old policy into a positive and a negative branch and lets negatives enter a supervised loss through an **implicit negative policy**. DiffusionNFT keeps the negative-aware idea — but the extrapolation formula for discrete token probabilities does not carry over to velocities; the diffusion derivation follows.

### 5.2　What "RL on the forward process" means

- **Collection**: any black-box solver (ODE/SDE, any step count) produces $K$ images; score them; **store only the clean $x_0$ and its reward**.
- **Training**: re-sample $t, \epsilon$, build $x_t = (1-t)x_0 + t\epsilon$ and target $v = \epsilon - x_0$, run a weighted flow-matching regression.
- No backprop through the solver, no training along the collection trajectory, no likelihood anywhere.

### 5.3　Reward normalisation and the positive/negative decomposition

Each image's reward is mapped to $[0,1]$:

$$r_i = \tfrac12 + \tfrac12\operatorname{clip}\Big(\frac{R_i - \bar R_c}{Z_c},\ -1,\ 1\Big)$$

$Z_c$ is a normalisation scale (the paper allows the global reward std); it is **not** DPO's partition function. $r_i$ is read as an "optimality probability" — **a per-sample quantity**, not a global success rate.

Modelling-wise, with $\bar r_c = \mathbb E_{\pi_\text{old}}[r]$, the implicit positive and negative policies are

$$\pi^+ = \frac{r\,\pi_\text{old}}{\bar r_c},\qquad \pi^- = \frac{(1-r)\,\pi_\text{old}}{1-\bar r_c}$$

The implementation need not split the data into two piles; the soft weights $r$, $1-r$ do the job.

### 5.4　Decomposition at the noisy state and the improvement direction

At the noisy state $x_t$ the positive share is the posterior

$$\alpha(x_t) = \mathbb E[r(x_0,c)\mid x_t, c] = \bar r_c\,\frac{\pi_t^+(x_t\mid c)}{\pi_t^\text{old}(x_t\mid c)}$$

(neither the per-sample $r_i$, nor a global success rate, nor the noise schedule's $\alpha_t$). The fields of the true positive/negative distributions satisfy

$$v_\text{old} = \alpha\,v^+ + (1-\alpha)\,v^-,\qquad \Delta := \alpha\,(v^+ - v_\text{old}) = (1-\alpha)(v_\text{old} - v^-)$$

$\Delta$ is the "reinforcement guidance" direction. These are the true fields; the **parameterised branches** below get separate notation.

### 5.5　Implicit mixing: why the negative branch carries $(1+\beta)$

One network $v_\theta$ parameterises both branches by **symmetric reflection** about old:

$$v_\theta^+ = v_\text{old} + \beta\,(v_\theta - v_\text{old}),\qquad v_\theta^- = v_\text{old} - \beta\,(v_\theta - v_\text{old})$$

Expanding the second line gives the paper's $(1+\beta)v_\text{old} - \beta v_\theta$. $\beta$ is a mixing parameter.

### 5.6　Loss and gradient (be able to expand this in an interview)

$$\mathcal L(\theta) = \mathbb E_{c,\,x_0\sim\pi_\text{old},\,t,\,\epsilon}\Big[r\,\lVert v_\theta^+ - v\rVert^2 + (1-r)\,\lVert v_\theta^- - v\rVert^2\Big]$$

Both terms are **non-negative MSEs** — the negative feedback comes from the reflected parameterisation, not from a negative weight on an ordinary MSE. With $d = v_\theta - v_\text{old}$ and $e = v_\text{old} - v$, per sample:

$$\ell = \lVert e\rVert^2 + \beta^2\lVert d\rVert^2 + 2\beta(2r-1)\,e^\top d$$

Gradient in $d$: $2\beta^2 d + 2\beta(2r-1)e$. At the start, $v_\theta = v_\text{old}$ ($d=0$), the gradient is $2\beta(2r-1)(v_\text{old} - v)$: a positive ($r=1$) descending along $-\nabla$ **moves toward the target $v$**, a negative ($r=0$) moves away, and $r=\tfrac12$ leaves only a quadratic pull toward old.

### 5.7　Theorem 3.2 and CFG-free inference

The per-sample optimum is $d^* = (2r-1)(v - v_\text{old})/\beta$; taking the conditional expectation with $v_\text{old} = \mathbb E[v\mid x_t]$ and $\mathbb E[r\,v\mid x_t] = \alpha v^+$:

$$v_\theta^* = v_\text{old} + \frac{2}{\beta}\,\Delta$$

The coefficient is $2/\beta$: the positive and negative branches each contribute one share of the same improvement direction. **$1/\beta$ controls the guidance strength** — smaller $\beta$, larger displacement. The theorem assumes ideal distributions and capacity; it does not promise a reward gain for arbitrary $\beta$ or arbitrary finite training.

**CFG re-read**: $v_\text{cfg} = v_\text{uncond} + w(v_\text{cond} - v_\text{uncond})$ is also "extrapolate along a direction" — cond is the positive signal, uncond the negative, and CFG is **offline** reinforcement guidance. NFT makes that online: starting from a model with **only the conditional branch**, collection, training and inference are all CFG-free. The paper's Table 1 multi-reward model reaches its reported numbers after about 1.7k steps and surpasses the CFG baseline.

### 5.8　The practical loss (Eq. 5 alone is not enough)

- **old is an EMA**: $\theta_\text{old} \leftarrow \eta_i\theta_\text{old} + (1-\eta_i)\theta$, frozen within an optimisation phase. NFT needs old; its core loss does **not** need a separately stored, permanently fixed initial ref.
- **Adaptive weighting**: the velocity regression is rewritten as an $x_0$ regression with a stop-grad normalising denominator, $\lVert x_\theta - x_0\rVert^2 / \operatorname{sg}\big(\operatorname{mean}\lvert x_\theta - x_0\rvert\big)$ (inspired by DMD distillation). That is practical loss design, not a verbatim instance of the theorem.
- **The difference from RWR / RFT** is the explicit use of negative feedback: the paper's ablation without the negative branch collapses quickly — an observation about that experiment, not a theorem that "any positive-only diffusion fine-tuning collapses".

### 5.9　Results

- Single reward: SD3.5-M GenEval **0.24 → 0.98** in about 1k steps. **0.24 is the CFG-free base**; the 0.63 usually quoted for Flow-GRPO is the CFG base — the two cannot be compared as baselines under the same CFG setting.
- Efficiency: the paper's §4.3 measure is **wall-clock time** — head-to-head with Flow-GRPO (both at 10 rollout steps), up to about **25×** on GenEval, 3–25× across rewards. It is a comparison of time-to-reach-a-reward on training curves, not per-step throughput, inference speed, or a 25× faster gradient estimator.
- Multi-reward (GenEval / OCR / PickScore / ClipScore / HPSv2.1 trained jointly in stages, 1.7k steps, 40-step collection): GenEval **0.94**, OCR **0.91** — don't fill this row with the single-reward 0.98. It beats SD3.5-L and FLUX.1-Dev on several in-/out-of-domain metrics but **not column by column** (e.g. ClipScore 0.293 vs FLUX's 0.295).

---

## §6 The three side by side

| Dimension | Flow-GRPO | DGPO | DiffusionNFT |
| --- | --- | --- | --- |
| Mathematical form | advantage-weighted policy gradient | group-preference logistic | reward-weighted two-branch regression |
| Sampler | training rollouts must be SDE | any, default ODE (10-step DPM) | any black-box |
| What is stored | full trajectories $O(PGTd)$ | clean images $O(PGd)$ | clean images $O(PGd)$ |
| Needs a likelihood | per-step Gaussian | no (DSM difference) | no |
| old | rollout policy | collector (EMA) | EMA reflection centre |
| ref | KL term | DSM-difference anchor | not needed by the core loss |
| CFG | original config uses CFG; implementations also support CFG-free | paper silent; official implementation defaults to CFG rollouts | CFG-free throughout |
| Main approximation | discretisation + PPO surrogate | path likelihood + DSM + Jensen | no likelihood approximation; model / sampling / optimisation error remain |
| Reported efficiency | baseline | ~20× | up to ~25× |

Storage excludes weights, optimiser state and activations; "one more ref" is not "double the memory".

---

## §7 Code Patterns

Three ~20-line core losses; the full experiments are in [`code/diffusion_online_rl.py`](code/diffusion_online_rl.py). All three share `import math, torch; import torch.nn.functional as F`.

### 7.1　Flow-GRPO: one SDE step + log-prob + ratio

```python
def sde_step(v_theta, x, t, h, a=0.7, xi=None):
    """One reverse-time Euler-Maruyama step, t -> t - h, h > 0.  Returns (x_next, mu, v)."""
    g2 = a**2 * t / (1 - t)                                   # g(t)^2, t in (0,1)
    v = v_theta(x, t)
    eps_hat = x + (1 - t) * v                                 # §2.2
    mu = x - h * (v + g2 / (2 * t) * eps_hat)                 # drift, positive step
    xi = torch.randn_like(x) if xi is None else xi
    return mu + (g2 * h) ** 0.5 * xi, mu, v

def gauss_logp(x_next, mu, g2h):
    d = x_next.numel() // x_next.shape[0]
    return -((x_next - mu) ** 2).flatten(1).sum(1) / (2 * g2h) - 0.5 * d * math.log(2 * math.pi * g2h)

# per step: states x_k, x_{k+1}, old_logp, adv are FIXED from rollout; t_k, h are scalars
g = (0.7**2 * t_k / (1 - t_k)) ** 0.5                          # g(t_k), same a as sde_step
g2h = g**2 * h
_, mu, v_cur = sde_step(policy, x_k, t_k, h, xi=None)          # one policy call: mean for the ratio, v for the KL
logp = gauss_logp(x_k1, mu, g2h)
rho = torch.exp(logp - old_logp)
pg = -torch.min(rho * adv, rho.clamp(1 - eps_c, 1 + eps_c) * adv).mean()
with torch.no_grad():
    v_ref = ref(x_k, t_k)
kl = (h / 2) * (1 / g + g * (1 - t_k) / (2 * t_k)) ** 2 * ((v_cur - v_ref) ** 2).flatten(1).sum(1)
loss = pg + beta_kl * kl.mean()
```

### 7.2　DGPO: within-group DSM differences into one sigmoid

```python
def dgpo_loss(policy, ref, x0_group, c, adv, t, eps, beta, T, lam_t=1.0):
    """x0_group: [G, ...] clean samples of ONE prompt; adv: [G] centred advantages (sum ~ 0).
    Shared t, eps across the group (variance reduction, not needed for the log Z proof)."""
    x_t = (1 - t) * x0_group + t * eps
    v = eps - x0_group
    l_theta = ((policy(x_t, t, c) - v) ** 2).flatten(1).mean(1)            # per-sample DSM
    with torch.no_grad():
        l_ref = ((ref(x_t, t, c) - v) ** 2).flatten(1).mean(1)
    d = l_theta - l_ref                                                     # [G]
    # scale note: the paper's DSM is an x0-space error; this snippet uses per-dim mean
    # velocity error. With D latent dims, d_x0 = D * t**2 * d_here. To keep the paper's
    # objective set lam_t_here = D * t**2 * lam_t_paper; dropping the factor changes the
    # timestep weighting (a constant beta cannot absorb t**2) and needs its own tuning.
    w = adv.abs()                                                           # w = |A|; sum_{+} w == sum_{-} w
    logit = -lam_t * beta * T * (adv * d).sum()                             # sign carried by adv
    return -F.logsigmoid(logit)
```

### 7.3　DiffusionNFT: reflected two-branch regression

```python
def nft_loss(policy, old, x0, c, r, t, eps, beta):
    """x0: [B, ...] clean samples; r: [B] in [0,1] (0.5 + 0.5*clip(normalised reward)).
    t is a shared scalar here; with per-sample t of shape [B], reshape it to [B,1,...] for
    the noising line and keep the [B] form for the network call."""
    x_t = (1 - t) * x0 + t * eps
    v = eps - x0
    with torch.no_grad():
        v_old = old(x_t, t, c)                                # EMA of policy, frozen inside a phase
    d = policy(x_t, t, c) - v_old
    v_pos = v_old + beta * d                                   # (1-beta) v_old + beta v_theta
    v_neg = v_old - beta * d                                   # (1+beta) v_old - beta v_theta
    per = r * ((v_pos - v) ** 2).flatten(1).mean(1) + (1 - r) * ((v_neg - v) ** 2).flatten(1).mean(1)
    return per.mean()
```

---

## §8 Failure modes of online RL

- **Reward hacking — look at three things**: the raw reward (not the group-normalised score, which says nothing about absolute progress across rounds), the images themselves, and diversity. KL, clipping and the negative-branch loss **do not guarantee** its absence.
- **Flow-GRPO's ratio drift** (GRPO-Guard): the importance-ratio distribution shifts with the timestep, so a fixed $\varepsilon_c$ clip stops working at some $t$; RatioNorm / gradient re-weighting are optimisation-level fixes.
- **DGPO's low-noise artefacts**: forget timestep clipping and the model learns the blur of few-step sampling.
- **NFT's $\beta$**: it sets two things at once — the gradient at $d=0$ is proportional to $\beta$, while the ideal optimal displacement is proportional to $1/\beta$; a smaller $\beta$ means a larger guidance target. Updating old too fast is unstable in the paper's experiments (the reflection itself still gives a non-zero gradient with old frozen; what fails is stability, not the reflection).
- **What "out-of-domain" means**: in these papers OOD usually means evaluation metrics on DrawBench that did not take part in training, not an out-of-distribution image domain.

---

## §9 vs GRPO / NFT for LLMs

| | LLM | Diffusion |
| --- | --- | --- |
| Per-step policy | categorical over tokens, with directly computable probabilities | ODE is a Dirac; SDE-ify (Flow-GRPO) or bypass (DGPO / NFT) |
| Sequence length | hundreds to thousands of tokens | 10–40 steps, each a whole latent |
| NFT's negative policy | extrapolation of discrete probabilities | reflection of the velocity about old |
| Reference model | GRPO has a KL-ref | Flow-GRPO has one; NFT's core loss doesn't |
| CFG | no counterpart | NFT reads CFG as offline guidance and makes it online |

Shared lesson: once the group baseline removes the critic, everything hard is about "can the density be computed" — free for LLMs; for diffusion either build one (SDE) or switch to an objective that needs none.

---

## §10 25 high-frequency interview questions

### L1 essentials (10)

<details>

<summary>Q1. What is the real difference between Flow-GRPO, DGPO and DiffusionNFT?</summary>

They share the group-sampling + within-group-normalisation frame. They differ in how the reward enters the gradient: Flow-GRPO weights the per-step transition policy gradient; DGPO puts the reference-relative DSM differences of the positive and negative groups into one sigmoid (preference learning); NFT uses the reward to weight two flow-matching regression branches. None has a critic, but they are not "three kinds of policy gradient".

"They're all GRPO variants" scores zero.

</details>

<details>

<summary>Q2. ODE sampling already has a random seed — why does Flow-GRPO still switch to an SDE?</summary>

The seed gives sample-level randomness; PPO needs the **per-step transition density** $p_\theta(x_{k+1}\mid x_k)$ to form the importance ratio. The ODE's step is a Dirac with no density. After SDE-ification the Euler-discretised step is Gaussian, and intermediate steps gain exploration as a side effect.

"ODEs can't explore" is wrong.

</details>

<details>

<summary>Q3. Can old and ref be used interchangeably?</summary>

No. old is the behaviour policy at sampling time — in Flow-GRPO a snapshot of this round's rollout policy (the ratio's denominator), in NFT the reflection centre, updated by EMA; ref is the frozen anchor — Flow-GRPO's KL, DGPO's DSM difference. NFT's core loss uses the evolving old and needs no fixed initial ref.

</details>

<details>

<summary>Q4. Is NFT's $r$ a success rate?</summary>

No. $r_i = \tfrac12 + \tfrac12\operatorname{clip}((R_i - \bar R_c)/Z_c, -1, 1)$ maps **each sample's** reward into $[0,1]$, read as an optimality probability. The global quantity is $\bar r_c = \mathbb E_{\pi_\text{old}}[r]$; the noisy-state quantity is the posterior $\alpha(x_t)$.

</details>

<details>

<summary>Q5. Write the RF conversions $\hat x_0$, $\hat\epsilon$, score.</summary>

$x_t = (1-t)x_0 + t\epsilon$, $v = \epsilon - x_0$; with the network prediction $v_\theta$: $\hat x_0 = x_t - tv_\theta$, $\hat\epsilon = x_t + (1-t)v_\theta$, $s_\theta := -\hat\epsilon/t$. The score uses the predicted conditional mean and equals the true score only when $v_\theta = v^*$.

</details>

<details>

<summary>Q6. How does DGPO split the groups, and what are the weights?</summary>

Within one prompt's group, $A_i = (R_i - \bar R)/s_R$; $A_i > 0$ goes to $\mathcal G^+$, $A_i \le 0$ to $\mathcal G^-$; $w_i = \lvert A_i\rvert$. Because $\sum A_i = 0$, the two groups have equal total weight.

</details>

<details>

<summary>Q7. What is Flow-GRPO's denoising reduction?</summary>

Train with 10 SDE steps, evaluate with 40. The paper reports "over 4×" speed-up. The justification is empirical: the continuous marginal is step-independent and the directional correction learned by RL is insensitive to the step count — not a strict equivalence.

</details>

<details>

<summary>Q8. What does DiffusionNFT need to store for training?</summary>

Only the clean image $x_0$, its prompt $c$ and its reward. Training re-samples $t,\epsilon$ to build $x_t$. No trajectories, no likelihood, no backprop through the solver.

</details>

<details>

<summary>Q9. Why can DiffusionNFT infer without CFG?</summary>

CFG extrapolates between a cond (positive) and an uncond (negative) model — offline reinforcement guidance. NFT learns, from online positives and negatives, a reward guidance that replaces CFG's role (its direction is not guaranteed to coincide with CFG's), so the model carries its own guidance. Starting from a conditional-only model, it is CFG-free throughout.

</details>

<details>

<summary>Q10. GenEval 0.63, 0.24, 0.95, 0.97, 0.98 — what does each mean?</summary>

0.63: SD3.5-M base **with CFG**; 0.24: **CFG-free** base; 0.95: Flow-GRPO; 0.97: DGPO; 0.98: NFT single-reward at ~1k steps. NFT's multi-reward model is 0.94.

Treating the two as baselines under the same CFG setting scores zero (side by side with the CFG setting labelled is fine — the paper itself does that).

</details>

### L2 advanced (10)

<details>

<summary>Q11. Derive the marginal-preserving SDE and state the sign for the generation direction.</summary>

Forward clock $u = 1-t$: $dX_u = [-v + \tfrac12 g^2 s]du + g\,dW_u$. In Fokker–Planck, $-\tfrac12 g^2\nabla\cdot(ps)$ cancels $+\tfrac12 g^2\Delta p$ (because $ps = \nabla p$), leaving the ODE's density evolution. Back to decreasing $t$: the drift is $v - \tfrac12 g^2 s$. The discrete noise scale is $\sqrt{h}$ with $h = t_k - t_{k+1} > 0$.

Writing $+\tfrac12 g^2 s$ means the time direction was flipped.

</details>

<details>

<summary>Q12. Why is Flow-GRPO's KL closed-form? Write it.</summary>

Policy and ref have **the same covariance** $g^2 h I$ for the per-step transition, so the KL is just the mean difference: $\lVert\mu_\theta - \mu_\text{ref}\rVert^2/(2g^2h)$. Substituting the drift gives $\frac h2\big(\frac1g + \frac{g(1-t)}{2t}\big)^2\lVert v_\theta - v_\text{ref}\rVert^2$. It is a conditional per-step KL, not the KL of final distributions.

</details>

<details>

<summary>Q13. Why can DGPO cancel $\log Z(c)$?</summary>

The implicit reward is $r_\theta = \beta\log(p_\theta/p_\text{ref}) + \beta\log Z(c)$. The group reward is a weighted sum; after subtracting the negative group from the positive, the coefficient of $\log Z$ is $\sum_{\mathcal G^+}w - \sum_{\mathcal G^-}w$; with $w = \lvert A\rvert$ and $\sum A = 0$ that difference is exactly zero. The premise is balanced weight mass, not head-count averaging.

</details>

<details>

<summary>Q14. How does DGPO relate to Diffusion-DPO?</summary>

Same reference-relative DSM surrogate. With $G=2$ DGPO reduces to Diffusion-DPO (scale absorbed into $\beta$). General DGPO replaces the single pair with two $\lvert A\rvert$-weighted sub-groups and puts the group difference into one sigmoid — not a pairwise enumeration.

</details>

<details>

<summary>Q15. What does DGPO's timestep clipping fix, and is it related to PPO's clip?</summary>

Unrelated. Few-step samples carry artefacts; regression at low-noise $t$ would learn them, so $t$ is drawn only from $[t_\min, 1]$. This truncates the training timestep; PPO's clip truncates the importance ratio.

</details>

<details>

<summary>Q16. Write Flow-GRPO's per-step Gaussian log-density and explain "sum over dimensions" vs "average per dimension".</summary>

$\log p = -\lVert x_{k+1} - \mu_\theta\rVert^2/(2g^2h) - \tfrac d2\log(2\pi g^2h)$, summed over the $d$ dimensions to be a density. The official code averages the log-prob per dimension, which yields $\rho^{1/d}$; keep the paper's formula and the implementation convention apart.

</details>

<details>

<summary>Q17. How are DiffusionNFT's $v_\theta^+$ and $v_\theta^-$ defined?</summary>

Symmetric reflection about old: $v_\theta^\pm = v_\text{old} \pm \beta(v_\theta - v_\text{old})$. Expanding the negative branch gives $(1+\beta)v_\text{old} - \beta v_\theta$. $\beta$ is the mixing parameter; $1/\beta$ controls the guidance strength.

</details>

<details>

<summary>Q18. Expand NFT's per-sample loss and state the gradient direction at $v_\theta = v_\text{old}$.</summary>

$d = v_\theta - v_\text{old}$, $e = v_\text{old} - v$: $\ell = \lVert e\rVert^2 + \beta^2\lVert d\rVert^2 + 2\beta(2r-1)e^\top d$. At $d = 0$ the gradient is $2\beta(2r-1)(v_\text{old} - v)$: positives move toward the target, negatives away, and $r = \tfrac12$ leaves only the quadratic pull toward old.

</details>

<details>

<summary>Q19. In Flow-GRPO training, what is fixed and what is recomputed?</summary>

Fixed: the rollout's state sequence, old log-probs, rewards, advantages. Recomputed: the current policy's log-prob at those states. ref outputs are stop-grad. Storing trajectories stores states, not the backward graph.

</details>

<details>

<summary>Q20. What is each method's dependence on CFG?</summary>

None of the three objectives makes CFG a mathematical prerequisite. Flow-GRPO's original SD3 config uses CFG and later implementations also support CFG-free; DGPO's paper is silent, its official implementation defaults to CFG rollouts and conditional predictions for the DSM update; NFT's main recipe is CFG-free in collection, training and inference.

</details>

### L3 top-lab (5)

<details>

<summary>Q21. Prove NFT's optimum is $v_\text{old} + \frac2\beta\Delta$ and explain where the "2" comes from.</summary>

Per sample $d^* = (2r-1)(v - v_\text{old})/\beta$. Take the conditional expectation: $\mathbb E[(2r-1)(v - v_\text{old})\mid x_t] = 2\mathbb E[r(v - v_\text{old})\mid x_t]$ (since $\mathbb E[v\mid x_t] = v_\text{old}$), and with $\mathbb E[rv\mid x_t] = \alpha v^+$, $\mathbb E[r\mid x_t] = \alpha$ this is $2\alpha(v^+ - v_\text{old}) = 2\Delta$. The "2" is the positive and negative branches each contributing one share of the same displacement.

</details>

<details>

<summary>Q22. Does the 25× prove NFT's gradient estimator itself is 25× faster?</summary>

No. It is a **wall-clock training-curve** comparison on one reward (GenEval), both at 10 rollout steps, and it bundles convergence speed, CFG presence, sampling and implementation differences. Other rewards land between 3× and 25×. The multi-reward model's numbers are a separate matter.

</details>

<details>

<summary>Q23. Why does head-count averaging break DGPO's cancellation? Give an example.</summary>

Four images with rewards $[0,1,2,7]$, positive:negative = 1:3. With $w = \lvert A\rvert$ both sides sum to $S$, so the $\log Z$ coefficient is $S - S = 0$. Two ways to break it: (a) keep $\lvert A\rvert$ but average each side by its head-count — the coefficient becomes $S - S/3 \ne 0$ and the cancellation is gone; (b) set all weights to 1 and just sum — the coefficient is $1 - 3 = -2 \ne 0$, so adding the same constant to every implicit reward changes the logit. Experiment C in `code/diffusion_online_rl.py` checks (b).

</details>

<details>

<summary>Q24. Does Flow-GRPO's "marginal preservation" still hold with finite steps, learning error and CFG?</summary>

Not strictly. Marginal preservation is a continuous-time property requiring $v$ and $s$ to correspond to the same family of marginals. Learning error decouples $s_\theta$ from $v_\theta$; a CFG-modified velocity and the substituted score are **not guaranteed to correspond to the same forward-noising marginal family**, so the proof does not carry over directly; finite-step discretisation adds further bias. Hence denoising reduction can only be an empirical observation.

</details>

<details>

<summary>Q25. If you had to ship one of the three, how would you choose?</summary>

Three things: the sampler (only consider Flow-GRPO with an SDE cost budget), storage (trajectories $O(PGTd)$ vs clean images $O(PGd)$), and whether CFG-free is required (NFT gives it for free). DGPO and NFT both use the ODE and store only clean images; DGPO keeps a ref anchor, NFT uses an EMA old plus reflection. All three reported efficiencies are comparisons on specific reward curves — run your own reward.

</details>

---

## §A Appendix

### A.1　Papers

| Method | arXiv | Venue | Code |
| --- | --- | --- | --- |
| Flow-GRPO | [2505.05470](https://arxiv.org/abs/2505.05470) | — | github.com/yifan123/flow_grpo |
| DGPO | [2510.08425](https://arxiv.org/abs/2510.08425) | ICLR 2026 | github.com/Luo-Yihong/DGPO |
| DiffusionNFT | [2509.16117](https://arxiv.org/abs/2509.16117) | ICLR 2026 Oral | github.com/NVlabs/DiffusionNFT |
| NFT (LLM) | [2505.18116](https://arxiv.org/abs/2505.18116) | — | — |
| GRPO | [2402.03300](https://arxiv.org/abs/2402.03300) | — | — |

### A.2　Runnable toy

Script: [`code/diffusion_online_rl.py`](code/diffusion_online_rl.py), pure PyTorch, seconds on CPU.

- **A** marginal preservation: 1-D $x_0\sim\mathcal N(1, 0.25)$ with analytic $v^*$, $s^*$; integrate from $t=0.9$ to $0.1$ and compare the means/variances of ODE / correct SDE / noise-without-score-correction.
- **B** NFT gradient: fixed pair $t=0.5, x_0=0, \epsilon=1$, $\beta=0.5$; assert initial gradients $-1,+1,0$ for $r=1,0,\tfrac12$ and optima $\pm2$ for positive/negative.
- **C** DGPO cancellation: 4-sample group $[0,1,2,7]$; assert equal positive/negative weight mass, invariance to a global reward shift, invariance to a global implicit-score shift; the unit-weight counterexample.

### A.3　Engineering pitfalls

| Symptom | Cause | Fix |
| --- | --- | --- |
| Flow-GRPO NaN when starting at $t=1$ | $g(t) = a\sqrt{t/(1-t)}$ has a zero denominator | handle the first step with neighbouring time points |
| Flow-GRPO clip stops working at some $t$ | ratio distribution drifts with the timestep | RatioNorm / gradient re-weighting (GRPO-Guard) |
| DGPO images go blurry | low-noise $t$ learned the few-step artefacts | timestep clipping $[t_\min, 1]$ |
| DGPO loss doesn't move | all rewards in the group equal, std zero, advantage set to zero | change prompts or enlarge $G$ |
| NFT unstable | $\beta$ too small or old EMA tracking too closely | tune $\beta$, slow the EMA |
| All three “score up, quality flat” | reward and visual quality have decoupled (the raw reward gets hacked too); the group-normalised score has the separate problem of not tracking cross-round progress | raw reward + images + diversity |
