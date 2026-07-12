## §0 TL;DR Cheat Sheet

> 💡 **8 sentences to nail LLM OPD (On-Policy Distillation)** — the hottest "cheap RL" paradigm in 2025-2026 post-training. Get the interview core points in one page (see §1-§9 derivations + §10 25 frequently-asked questions).

1. **OPD = On-Policy Distillation**: the student samples **trajectories** using its own current policy; the teacher provides **per-token supervision** (KL / log-prob / soft label) on the states that the student itself visits. It is **not** a typo for DPO, and **not** Online Preference Distillation — in the LLM context this term refers specifically to "on-policy distillation". Representative papers: MiniLLM (Gu 2024 ICLR, arXiv 2306.08543), GKD (Agarwal 2024 ICLR, arXiv 2306.13649), Thinking Machines blog (Lu 2025-10-27), Qwen3 Technical Report (May 2025, arXiv 2505.09388), Survey (Song & Zheng 2026, arXiv 2604.00626).

2. **Core loss (writeable in one line)**: sample $y \sim \pi_\theta(\cdot|x)$, compute reverse KL per token, defined as $L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta}[\sum_t D_{\text{KL}}(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t}))]$. Note the expectation subscript is $y \sim \pi_\theta$ (**student samples on its own**), and the KL direction is $\pi_\theta \| \pi_T$ (**reverse / mode-seeking**) — these two points are the essential distinction between OPD and vanilla KD. Full formula:
   $$L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]$$

3. **Three distillation paradigms in a nutshell**:
   - **SFT / Hard distillation**: teacher generates $\hat y$, student does cross-entropy on $\hat y$ (off-policy + hard label)
   - **Vanilla KD / Soft distillation (Hinton 2015)**: teacher generates $\hat y$, student matches teacher's soft logits (off-policy + soft label, forward KL, mode-covering)
   - **OPD**: **student generates $y$ itself**, teacher computes logits on $y$ to give a KL signal (on-policy + soft label, reverse KL, mode-seeking)

4. **Why on-policy is crucial**: off-policy distillation suffers from exposure bias — during training the teacher's prefixes are "perfect", but at inference the student encounters its own erroneous prefixes which it has **never seen**, so errors compound (cumulative error roughly $O(L^2)$ with sequence length $L$, see §1.3). OPD aligns the training distribution with the inference distribution, squeezing the compound error from $O(L^2)$ down to $O(L)$.

5. **OPD vs RL (the headline selling point)**: the Thinking Machines blog provides an empirical rule of thumb — RL teaches $O(1)$ bits per trajectory (one outcome reward), OPD teaches $O(N)$ bits (each token gets supervised by the teacher's soft label). On Qwen3-8B-Base + Qwen3-32B-teacher math reasoning experiments, OPD **matches RL's AIME'24 gain, but compute drops to roughly 1/9-1/30**. The Qwen3 Tech Report independently reports OPD ≈ RL in performance but **only 1/10 GPU hours**.

6. **Relation to DPO / GRPO**: (a) **DPO** uses offline preference pairs for closed-form RLHF, **no teacher logits, no student rollouts** — almost orthogonal to OPD; (b) **GRPO** uses group-relative advantage for on-policy RL, **has student rollouts but reward is sparse outcome**; (c) **OPD = GRPO's "dense teacher KL replacing sparse outcome reward"** version. The Survey (Song 2026) gives a unifying view: OPD ≈ "KL-constrained RL with $\beta \to \infty$ and token-level reward from teacher log-prob".

7. **2025-2026 industrial adoption**: Qwen3 (off-policy + on-policy two-stage distillation of small models), DeepSeek-R1 distillation series (off-policy SFT primarily, but follow-ups use OPD), MiMo-V2, Kimi distill series (Gemma 2/3's official distillation method is off-policy soft-target KD on a fixed dataset, with no student self-rollout, so it should not be counted as an OPD adopter — see §5.3). Thinking Machines (Murati's team, 2025) packages OPD as "the cheap RL alternative" route.

8. **Three most-tested footguns**: (a) **Length inflation / truncation collapse** — student rollouts grow longer over training, hitting truncation triggers gradient bias and validation plummets (Demystifying OPD 2026, arXiv 2604.08527); (b) **Reverse KL mode collapse** — student converges to a single mode of the teacher, generation diversity collapses; (c) **Teacher / student gap too large**: the student cannot sample states where the teacher has high probability — note this is not "the KL signal vanishing"; the rare teacher-low-probability tokens the student does sample instead produce numerically extreme penalties ("prefix teach, suffix fade" phenomenon, arXiv 2605.13643, see §7.5).

## §1 Intuition: Why On-Policy Distillation Is Needed

### 1.1　A Brief History of Knowledge Distillation (Hinton → DistilBERT → MiniLLM → OPD)

The core idea of distillation has not changed since 2015: **train the student with the teacher's soft labels**. But for LLM autoregressive generation, there is a hidden assumption that needs to be re-examined — **what distribution does the prefix seen during training come from**.

| Year | Method | Training distribution | KL direction | Applicable |
|---|---|---|---|---|
| 2015 | Hinton soft target | teacher forward on dataset | forward KL $D(\pi_T \,\Vert\, \pi_\theta)$ | classification / detection, single-step prediction |
| 2019 | DistilBERT (Sanh) | teacher logit on dataset | forward KL + MLM cross-entropy | encoder models, single-step prediction |
| 2016 | Seq-level KD (Kim & Rush 2016) | teacher beam-search $\hat y$ | hard token CE on $\hat y$ | NMT, autoregressive but still off-policy |
| 2023 | MiniLLM (Gu, ICLR 2024) | **student rollout** $y \sim \pi_\theta$ | **reverse KL** $D(\pi_\theta \,\Vert\, \pi_T)$ | LLM instruction following |
| 2023 | GKD (Agarwal, ICLR 2024) | mix student rollout + dataset | generalized JSD (forward/reverse interpolation) | LLM seq-to-seq |
| 2024-2026 | Qwen3 / Thinking Machines / Survey | student rollout + token-level teacher logit | primarily reverse KL | production-grade LLM post-training |
| 2025-01 | DeepSeek-R1-Distill official pipeline | teacher offline-generates 800K trajectories | n/a (pure off-policy CE, no KL signal, see §3.4) | more accurately classified as Off-policy SFT KD; only community follow-ups use OPD for further distillation |

The key evolution: **off-policy → on-policy** is the watershed step that upgrades LLM distillation from "imitating the dataset" to "imitating the decision process".

### 1.2　Forward KL vs Reverse KL: mode-covering vs mode-seeking

Vanilla KD (Hinton) uses forward KL:

$$D_{\text{KL}}(\pi_T \,\|\, \pi_\theta) = \sum_y \pi_T(y) \log \frac{\pi_T(y)}{\pi_\theta(y)}$$

The expectation is taken under $\pi_T$; **only places where the teacher assigns high probability contribute to the loss**. If $\pi_T(y) > 0$ but $\pi_\theta(y) = 0$, the loss explodes → the student is forced to **cover every mode of the teacher** (mode-covering / mass-covering), even modes that are unreachable for the student. The result is that the student spreads probability across many tokens, generating "safe but mediocre" answers.

MiniLLM switches to reverse KL:

$$D_{\text{KL}}(\pi_\theta \,\|\, \pi_T) = \sum_y \pi_\theta(y) \log \frac{\pi_\theta(y)}{\pi_T(y)}$$

The expectation is taken under $\pi_\theta$; **only places the student itself samples contribute to the loss**. If $\pi_T(y) = 0$ but $\pi_\theta(y) > 0$, the loss explodes → the student is forced to **avoid tokens the teacher considers impossible** (mode-seeking / zero-forcing). The result is that the student converges to a single high-quality mode of the teacher and generates "sharp and coherent" answers.

> 💡 **Intuitive difference** — Imagine the teacher is a bimodal distribution with "3 equally good answers":

- **Forward KL** makes the student learn a trimodal averaged distribution, where each mode has some probability but none is sharp enough
- **Reverse KL** makes the student lock onto one of the modes (which mode is determined by init + sampling), producing the most confident path closest to the teacher's decision

For LLM generation tasks this is usually a good thing — we want "fluent, confident, correct" output, not "averaged, vague, mediocre" output. This is also the key insight behind MiniLLM being the first to apply reverse KL to LLM distillation.

### 1.3　Exposure bias: the structural flaw of off-policy distillation

Consider an autoregressive generation task with sequence length $L$. Off-policy distillation (including SFT / vanilla KD / seq-level KD) uses prefixes from the teacher or dataset during training; at inference, prefixes come from the student itself.

Assume the per-step error rate (the probability that the student deviates from the teacher on some prefix) is $\epsilon$. Off-policy training only "teaches" the student on teacher prefixes, so at inference once the student takes one wrong step, **subsequent steps enter a state distribution never seen during training** — the student's error rate is no longer $\epsilon$, but $\epsilon' > \epsilon$ (often jumping to 0.5+ random level).

Bagnell 2010 proved for imitation learning: **the cumulative error of off-policy imitation is $O(L^2 \epsilon)$** (compound error), whereas on-policy supervision (such as DAgger) compresses cumulative error to $O(L \epsilon)$. This is precisely OPD's theoretical motivation on LLMs — align the training state distribution with the inference state distribution. DAgger's $O(L\epsilon)$ guarantee is not unconditional: it depends on two prerequisites — the expert must be queryable on any state the learner visits, and the online-learning subroutine used must satisfy a no-regret guarantee.

The Survey (Song & Zheng 2026 arXiv 2604.00626) repeats this result in §3:
> "The exposure bias of off-policy distillation grows roughly quadratically with sequence length; on-policy distillation linearizes the cumulative error."

This theoretical guarantee is the fundamental reason OPD is especially effective on **long sequence + high complexity tasks** like long-CoT, agent, and code.

### 1.4　Bit-rate perspective: dense teacher signal vs sparse RL reward

The Thinking Machines blog (Lu 2025-10) gives a clean information-theoretic intuition:

| Paradigm | Supervision bits per trajectory |
|---|---|
| **RL with outcome reward** | $O(1)$ — one scalar reward (typically $\{0, 1\}$ or [-1, 1] interval) |
| **OPD with per-token teacher KL** | $O(N \log V)$ — $N$ tokens × full distribution over vocab $V$ per token |

In other words, OPD "teaches" the student a whole distribution at every token, while RL only "teaches" a single number at the end of the sequence. On tasks like math reasoning, code, and long-form QA that need dense supervision, OPD's compute efficiency (the FLOPs needed to reach a given accuracy) is significantly higher than RL's — this is efficiency along the FLOPs/compute axis, a different quantity from "how many rollout samples are needed" (sample efficiency); the two need not move together. This also explains why Qwen3 / Thinking Machines both report OPD reaching RL performance with an order of magnitude less compute.

## §2 Precise Definition and Core Formula of OPD

### 2.1　Formalization

Given:
- **prompt distribution** $D$ (e.g., distribution of instructions in the dataset)
- **teacher policy** $\pi_T$ (fixed, frozen, parameter count $\gg$ student)
- **student policy** $\pi_\theta$ (trainable)

The OPD objective (general form, from Survey arXiv 2604.00626 §2.1):

$$\boxed{\;L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D}\,\mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\!\left[\sum_{t=1}^{|y|} D_f\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]\;}$$

where $D_f$ is some $f$-divergence, with common choices:

| Divergence | Formula | Source / preference |
|---|---|---|
| **Reverse KL** | $D(\pi_\theta \,\Vert\, \pi_T) = \sum_v \pi_\theta(v) \log \frac{\pi_\theta(v)}{\pi_T(v)}$ | MiniLLM, Thinking Machines blog, Qwen3 default |
| **Forward KL** | $D(\pi_T \,\Vert\, \pi_\theta)$ (but still sample from $\pi_\theta$) | "Sampled-token OPD", simple and stable |
| **JSD** | $\frac{1}{2}D(\pi_\theta \,\Vert\, M) + \frac{1}{2}D(\pi_T \,\Vert\, M)$, $M = \frac{1}{2}(\pi_\theta + \pi_T)$ | GKD default (symmetric, bounded) |
| **Generalized JSD** | $D_\beta = \beta D(\pi_\theta \,\Vert\, M_\beta) + (1-\beta) D(\pi_T \,\Vert\, M_\beta)$, $M_\beta = \beta \pi_\theta + (1-\beta)\pi_T$ | GKD's $\beta$ interpolation |
| **Total Variation** | $\frac{1}{2}\sum_v \lvert\pi_\theta(v) - \pi_T(v)\rvert$ | rarely used, bounded but non-smooth |

> ⚠️ **The Generalized-JSD endpoints are limits, not values** — plugging in $\beta=0$ directly: $M_0=\pi_T$, so the two terms are $0\cdot D(\pi_\theta\|\pi_T)$ and $1\cdot D(\pi_T\|\pi_T)=0$, i.e. $D_{\beta=0}=0$; likewise $D_{\beta=1}=0$. So strictly speaking $D_\beta$ is identically 0 at both endpoints, and "forward/reverse KL" can only be understood as **one-sided limits**: $\lim_{\beta\to 0^+} D_\beta/\beta = D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)$ (reverse KL), $\lim_{\beta\to 1^-} D_\beta/(1-\beta) = D_{\text{KL}}(\pi_T\,\Vert\,\pi_\theta)$ (forward KL) — note this limiting direction is the opposite of the naive intuition: the $\beta\to 0$ limit is reverse KL, the $\beta\to 1$ limit is forward KL (see the corrected statement in §10 L2-2).

**Key point**: regardless of which $D_f$ is chosen, **the expectation subscript must contain $y \sim \pi_\theta$** — the student samples on its own; this is the definition of "on-policy".

### 2.2　Reverse KL expansion + token-level loss

Expanding reverse KL on each prefix $s_t = (x, y_{<t})$:

$$D_{\text{KL}}\!\big(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t)\big) = \sum_{v=1}^{V} \pi_\theta(v|s_t) \left[\log \pi_\theta(v|s_t) - \log \pi_T(v|s_t)\right]$$

This is the **full KL** ("full-distribution" form), requiring the teacher to do a single teacher-forced forward pass over the whole rollout sequence (causal self-attention computes logits at all positions in parallel), rather than one separate forward per prefix length. Computational cost: $O(B \cdot L \cdot V)$ teacher forward + softmax ($L$ positions share the same forward call, not $L$ independent forwards).

In practice, the **"sampled-token" approximation** is common — only compute on the token $v = y_t$ that the student actually sampled:

$$\hat L_t = \log \pi_\theta(y_t|s_t) - \log \pi_T(y_t|s_t)$$

This is a single-sample Monte-Carlo estimate of reverse KL ($\mathbb{E}_{v \sim \pi_\theta}[\log \pi_\theta(v) - \log \pi_T(v)] \approx \log \pi_\theta(y_t) - \log \pi_T(y_t)$ when $y_t \sim \pi_\theta$). This form is called **per-token reverse KL** and is the version adopted by the Thinking Machines blog and Tinker cookbook implementations.

> ⚠️ **Sampled-token KL ≠ full KL** — sampled-token is an unbiased estimator, but **has high variance**, since it only looks at one token rather than the full vocab distribution. Each form has its preference:

- **Full KL**: high information density, low variance, but requires a vocab-level forward by the teacher at every step (expensive, especially if the teacher is large)
- **Sampled-token KL**: you only `gather` out one number $\log \pi_T(y_t)$ from the result, which saves the **storage/transfer** cost of the $[B,L,V]$ logits tensor — but the teacher still has to run the full $V$-dimensional projection + softmax normalization to get this one precise value, so **the compute (FLOPs) is not actually reduced**; what you get in exchange is high variance, which a control-variate baseline can reduce (see §4.4)

### 2.3　Two implementation routes: full-vocab supervised KL vs REINFORCE

OPD's "theoretical objective" is for the student to minimize the per-state KL on its own state visitation distribution $\rho_{\pi_\theta}$:

$$L_{\text{OPD}}(\theta) = \mathbb{E}_{s \sim \rho_{\pi_\theta}}\!\left[D_{\text{KL}}\!\big(\pi_\theta(\cdot|s)\,\|\,\pi_T(\cdot|s)\big)\right]$$

But $\theta$ appears simultaneously in (a) the inner KL and (b) the **state visitation** $\rho_{\pi_\theta}$. **Production implementations** choose to **decouple** these two $\theta$-dependencies — that is what MiniLLM / GKD / Thinking Machines / Qwen3 actually do:

**Route A: Full-vocab supervised KL + stop-grad rollouts (the textbook-clean path, simplest when teacher full logits are available)**

Treat the rollout as data produced by a "behavior policy" (**stop-grad on $\theta$**), and backprop only through the inner KL:

$$\boxed{\;L_{\text{OPD}}^{\text{A}}(\theta) = \mathbb{E}_{s_t \sim \text{rollout}(\pi_{\theta^-})}\!\left[\sum_t \sum_{v \in V} \pi_\theta(v|s_t) \log\frac{\pi_\theta(v|s_t)}{\pi_T(v|s_t)}\right]\;}$$

Here $\pi_{\theta^-}$ denotes that $\theta$ is stop-grad during the rollout phase (same idea as PPO's old policy); the inner full-vocab sum is **directly differentiable**, autograd handles it, **no REINFORCE needed**. This is the form adopted by the §4.1 code below.

> 💡 **Intuition** — Full-vocab KL, at every student-visited prefix $s_t$, aligns the entire distribution $\pi_\theta(\cdot|s_t)$ with $\pi_T(\cdot|s_t)$, not just the sampled token. Signal density $O(\log V)$, but every step requires one teacher logits forward (expensive).

> ⚠️ **Implementation attribution** — Do not equate Route A with the "Tinker / Thinking Machines blog default". Thinking Machines' open-source implementation (Tinker) actually defaults to **Route B's importance-sampling variant** (sampled-token logprob + negative-KL advantage, corresponding to `train_on_policy.py`'s `loss_fn="importance_sampling"` + `incorporate_kl_penalty`); **MiniLLM** (Gu 2024) is also a sampled-token + REINFORCE-style trajectory PG (with stability tricks like single-step decomposition / length norm / teacher-mixed sampling). Route A's full-vocab autograd is **the pedagogically cleanest form** (one-line PyTorch backprop, zero variance), and is a clean and correct production implementation when teacher full logits are available, but it is **not** the production default of any specific large lab. The two are **equivalent in expectation**, differing in variance / engineering cost — Route A advantages: zero variance, no IS clip needed. Route B advantages: shares sampled-token interface with PPO/GRPO, saves teacher vocab memory. (Here "zero variance" specifically means: given a fixed prefix $s_t$, the inner full-vocab KL is a closed-form sum that no longer needs the Monte-Carlo sampling estimate that Route B1 requires, so it eliminates the sampling-estimation variance for that same state; the training pipeline as a whole still carries cross-batch variance from the student's own sampling during rollout, so it is not "zero variance for the entire training pipeline".)

**Route B: Sampled-token REINFORCE / importance-sampling estimator (Tinker default / MiniLLM trajectory PG form)**

Route B first requires **clarifying two distinct gradient computations** that are often conflated in the literature:

**(B1) Fixed-state inner KL** — assume prefix $s_t$ is given, do REINFORCE estimation only on the inner $D_{\text{KL}}(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t))$:
$$\nabla_\theta D_{\text{KL}}(s_t) = \mathbb{E}_{y_t \sim \pi_\theta(\cdot|s_t)}\!\big[\nabla_\theta \log \pi_\theta(y_t|s_t) \cdot G_t^{\text{detach}}\big],\quad G_t = \log\tfrac{\pi_\theta(y_t|s_t)}{\pi_T(y_t|s_t)}.$$
This is the form used by §4.4's vOPD estimator, **unrelated to state visitation along the trajectory**.

**(B2) Trajectory objective with state visitation** — to strictly retain $\rho_{\pi_\theta}$'s dependence on $\theta$, the policy gradient requires the **return-to-go** (sum of all future KL costs) as the score-function weight:
$$\nabla_\theta\,\mathbb{E}_\tau\!\Big[\sum_t D_{\text{KL}}(s_t)\Big] = \mathbb{E}_\tau\!\Big[\sum_u \nabla_\theta \log\pi_\theta(a_u|s_u)\cdot \underbrace{\sum_{t \ge u} D_{\text{KL}}(s_t)^{\text{detach}}}_{\text{return-to-go}}\Big] + \mathbb{E}_\tau\!\Big[\sum_t \nabla_\theta D_{\text{KL}}(s_t)\Big].$$
Note the score-function weight is not the same-token $G_t$, but **the sum of KLs over all subsequent steps**.

> ⚠️ **Nobody really does the full B2 as-is, but MiniLLM is not a plain semi-gradient**. MiniLLM's (Gu et al. 2024) actual policy gradient uses a **single-step decomposition with a future log-ratio return**: $R_t = \sum_{t' \ge t} r_{t'}$, $r_{t'} = \log\frac{\pi_T(y_{t'}|y_{<t'})}{\pi_\theta(y_{t'}|y_{<t'})}$ — this is essentially a **variance-reduced approximation** of B2 (return-to-go), not a fixed-state KL at just the current state (B1), and not the stop-grad-on-visitation Route A either. It is the full, un-reduced B2 that is essentially impossible to train stably, since return-to-go accumulates across steps, exploding variance — that is the part that is genuinely "nobody does"; MiniLLM sits at a practical middle ground between the two.

**Unified practical rules**:
- **Route A** (pedagogically clear, §4.1): full-vocab KL + stop-grad rollouts, autograd directly, **no REINFORCE needed**. Zero variance, but requires teacher full logits.
- **Route B1** (§4.4 vOPD-style, Tinker / MiniLLM default): sampled-token REINFORCE / IS estimator + control variate. Equivalent in expectation to A, with higher variance but compatible with the sampled-token interface, saves teacher vocab memory, supports logprob-only (gray-box) teachers — note it still needs the teacher to emit $\log \pi_T(y_t|s_t)$; a truly sample-only black-box teacher (no logprobs) needs the dedicated Black-Box OPD method in §6.5.
- You **cannot** directly `.backward()` on the sampled-token $\log(\pi_\theta/\pi_T)$ — that is pathwise gradient + fixed sampled index, dropping the score-function term, and is **not the reverse-KL gradient** (neither MLE nor a KL descent direction)

### 2.4　Relation to KL-constrained RL

Consider RL with a reverse-KL constraint (the standard RLHF objective):

$$\max_\theta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[R(x, y)\big] - \beta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\big]$$

OPD is the special case of this formula when $R \equiv 0$ (no external reward) and $\pi_{\text{ref}} = \pi_T$ (reference replaced with the teacher):

$$\min_\theta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta \,\|\, \pi_T)\big]$$

This means **OPD is RLHF with "pure KL term, teacher as reference"**. From this perspective, GRPO + KL term is equivalent to OPD + GRPO-style group baseline; this is why the Survey classifies OPD as "a special case of KL-constrained RL", and it is also the formal basis for integrating OPD into GRPO in §3.

### 2.5　Forward KL form ("Sampled-Token OPD")

Some works (e.g., parts of Qwen3, TRL GKD trainer) use **forward KL on student samples**:

$$L_{\text{OPD-fwd}}(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_t D_{\text{KL}}\!\big(\pi_T(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t)\big)\right]$$

Note that **the sample comes from $\pi_\theta$, but the KL direction is forward**. Expanded to the token level:

$$L_{\text{OPD-fwd}} = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_t \sum_v \pi_T(v|s_t)\big(\log \pi_T(v|s_t) - \log \pi_\theta(v|s_t)\big)\right]$$

Since $\theta$ appears only in $\log \pi_\theta$ and not in the expectation subscript (although $y \sim \pi_\theta$ still depends on $\theta$, after stop-gradient treatment the $\theta$-dependence of the forward term simplifies), training degenerates into **cross-entropy on the teacher distribution at each student-visited prefix** — formally, this is **soft-label cross-entropy on student rollouts**. This is the engineering-simplest OPD implementation (no REINFORCE / control variate needed), at the cost of losing reverse KL's mode-seeking property.

## §3 The Relationship of OPD with DPO / GRPO / RLHF (Unified View)

### 3.1　Big-picture comparison table

| Dimension | SFT | Vanilla KD (Hinton) | DPO | RLHF + PPO | GRPO | **OPD** |
|---|---|---|---|---|---|---|
| **Training data source** | dataset | dataset | offline preference pair | dataset prompt | dataset prompt | dataset prompt |
| **Who generates $y$** | dataset | teacher | (offline pair) | student rollout | student rollout (group) | **student rollout** |
| **Supervision signal** | hard label | teacher soft logit | binary preference | scalar RM reward | scalar RM reward (group-normalized) | **teacher soft logit (per-token)** |
| **Is on-policy** | no | no | no | yes | yes | **yes** |
| **Needs critic** | n/a | n/a | n/a | yes (value head) | no (group baseline) | no (teacher KL is closed-form value) |
| **Supervision bits per traj** | $O(L \log V)$ | $O(L \log V)$ | $O(1)$ | $O(1)$ | $O(1)$ | **$O(L \log V)$** |
| **Needs teacher** | no | yes | no | no | no | **yes** |
| **Typical KL direction** | n/a (CE) | forward | n/a (closed-form) | reverse (vs $\pi_{\text{ref}}$) | reverse (vs $\pi_{\text{ref}}$) | **reverse (vs $\pi_T$)** |

> 💡 **Three sentences to place OPD** — one line each, memorizable:

- vs **SFT**: OPD trains on student rollouts (SFT trains on teacher/dataset)
- vs **vanilla KD**: OPD is on-policy + reverse KL (vanilla KD is off-policy + forward KL)
- vs **RL (PPO/GRPO)**: OPD uses teacher log-prob as dense reward (RL uses RM as sparse reward)

### 3.2　OPD vs DPO

**DPO** (Rafailov 2023 NeurIPS, arXiv 2305.18290) is closed-form RLHF: plug the optimal solution of KL-regularized RL back into Bradley-Terry, eliminate the partition $\log Z$, and obtain a pairwise preference loss:

$$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\big(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\right]$$

DPO is **completely offline**: no sampling, no teacher, no RM, no critic, only preference pairs.

| Dimension | DPO | OPD |
|---|---|---|
| Data | offline preference pair $(y_w, y_l)$ | online student rollout $y \sim \pi_\theta$ |
| Teacher | none (only $\pi_{\text{ref}}$) | requires stronger teacher $\pi_T$ |
| Supervision | binary preference | per-token soft logit |
| Compute | cheapest (pure forward + backprop) | moderate (student sampling + teacher forward) |
| When to use | have preference dataset, no strong teacher | have strong teacher, want to compress to small student |

**The two can be stacked**: first DPO to align human preferences, then OPD to compress to a small student with a stronger teacher. Qwen3's production pipeline uses a similar recipe; R1-Distill's officially released pipeline is itself pure off-policy SFT (no student rollout / KL signal, see §3.4) and does not stack OPD on top — using OPD to further distill an R1-Distill checkpoint is a community follow-up, not part of DeepSeek's official pipeline.

### 3.3　OPD vs GRPO

**GRPO** (DeepSeekMath 2024, arXiv 2402.03300) replaces PPO's critic with group-normalized advantage:

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf r)}{\text{std}(\mathbf r)}, \quad r_i = R(x, y_i),\; i \in [1..G]$$

$$L_{\text{GRPO}}(\theta) = \mathbb{E}_x \frac{1}{G}\sum_i \frac{1}{|y_i|}\sum_t \min\!\big(\rho_t^i \hat A_i,\, \text{clip}(\rho_t^i, 1{-}\epsilon, 1{+}\epsilon) \hat A_i\big) - \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

where $\rho_t^i = \pi_\theta / \pi_{\theta_{\text{old}}}$ is the importance ratio.

**The relation between OPD and GRPO**: replace GRPO's sparse outcome reward $R(x, y_i)$ with a **trajectory-level reward obtained by summing the per-token teacher KL log-ratio** $r^i = \sum_t \big(\log\pi_T(y_t^i|s_t^i) - \log\pi_\theta^{\text{old}}(y_t^i|s_t^i)\big) = -\sum_t\log\!\frac{\pi_\theta^{\text{old}}(y_t^i|s_t^i)}{\pi_T(y_t^i|s_t^i)}$ (note: this must be a log-ratio, not the LaTeX-prone misreading $-\log\pi_\theta/\pi_T$; and note this is a **single scalar** after summing over all tokens in the trajectory, which after group normalization gets broadcast to every token in that trajectory as the advantage — it does not provide token-level differential credit assignment, see §4.5), and set the reference $\pi_{\text{ref}} \leftarrow \pi_T$; the GRPO loss degenerates to the policy-gradient form of OPD (see §2.3).

> 💡 **Integration in practice** — modern production pipelines often use an **OPD + GRPO hybrid**:

- **outcome reward** from the verifier (math correct / test pass)
- **dense reward** from per-token teacher KL
- Total reward $r_t = \alpha \cdot r_{\text{outcome}} \cdot \mathbb{1}[t = T] + (1-\alpha) \cdot r_{\text{teacher-KL}}$

This gives both the task-aligned signal from outcome and dense supervision from the teacher — it is the standard for small-model post-training in the Qwen3 / DeepSeek series.

### 3.4　OPD vs vanilla RL distillation (R1-Distill)

The DeepSeek-R1-Distill series (Qwen / Llama 1.5B-70B) uses **off-policy distillation**:
1. The R1 teacher generates 800K reasoning trajectories (math + code + general)
2. The small student does token-level cross-entropy SFT
3. **No on-policy rollout, no KL signal**

This is hard-label seq-level KD (the LLM version of Kim & Rush 2016), not OPD. The success of R1-Distill is mainly due to extremely high-quality teacher trajectories (R1 was RL-tuned) + large data volume; **structurally it remains off-policy**.

OPD and R1-Distill are **complementary**:
- R1-Distill trajectory data can serve as OPD's init / SFT warm-start
- Running OPD after R1-Distill can further eliminate exposure bias

Academically / engineeringly this combination is called "**Off-policy SFT + On-policy distillation**", which is the standard two-stage Qwen3 small-model recipe (see §6.1).

## §4 Implementation: Core Code Blocks

> ⚠️ **Pedagogical illustration — for production, refer to the original papers** — This section's code shows the core idea of OPD (per-token reverse KL on student rollouts). **Note**:
>
> - **Full-vocab reverse KL** (summed over the whole vocab) is differentiable, autograd works directly; this is OPD's **Route A pedagogically clean form**, requiring teacher full logits to be available. Route B (sampled-token + REINFORCE / IS, **the default in production implementations like Tinker, MiniLLM**) is equivalent in expectation (see §2.3 callout), with the trade-off being variance vs engineering interface.
> - The sampled-token estimator form $\log\pi_s(y) - \log\pi_t(y)$, when $y$ is a discrete sample, requires **REINFORCE-style policy gradient** (with baseline / control variate) to backprop correctly; you **cannot** call `.backward()` directly, otherwise the gradient flows only through the $\log\pi_s$ term and loses backprop through the sample distribution choice.
> - The OPD + GRPO integration (§4.5) ratio should be **new policy / old behavior policy**, with the teacher used as reward or reference rather than the denominator of the PPO ratio. Cross-check against verl / OpenRLHF / Tinker implementations before production.
>
> The code below annotates which parts are full-vocab (directly autograd-able) vs sampled estimator (REINFORCE needed); for production code we strongly recommend referring to each framework's official release.

### 4.1　Per-token reverse KL (Route A, full-vocab conceptual implementation)

```python
import torch
import torch.nn.functional as F


def per_token_reverse_kl_loss(
    student,        # student model, requires_grad
    teacher,        # teacher model, frozen
    input_ids,      # [B, L]  prompt + student rollout (rollout done externally, stop_grad)
    action_mask,    # [B, L]  1 for student-generated tokens, 0 for prompt / pad
):
    """
    Route A — full-vocab per-token reverse KL on student rollouts (concept-clean form).
    Requires teacher full logits to be available. Route B (Tinker / MiniLLM production default)
    is equivalent in expectation but uses sampled-token + IS / REINFORCE estimator,
    see §2.3 callout and §4.4.

        L(θ) = E_{s_t ~ rollout(π_{θ⁻})} [ Σ_t D_KL(π_θ(·|s_t) || π_T(·|s_t)) ]
             = E_{s_t} [ Σ_t Σ_v  π_θ(v|s_t) · ( log π_θ(v|s_t) − log π_T(v|s_t) ) ]

    Rollout is done externally with stop_grad(θ) (vLLM / TGI returns input_ids);
    this function **only** computes the differentiable full-vocab KL, autograd
    backprops directly, **no REINFORCE needed**.
    """
    # Student full-vocab log-probs (requires_grad on θ)
    s_logits = student(input_ids).logits                   # [B, L, V]
    s_log_probs = F.log_softmax(s_logits[:, :-1], dim=-1)  # [B, L-1, V]
    s_probs = s_log_probs.exp()                            # [B, L-1, V]

    # Teacher full-vocab log-probs (frozen, no grad)
    with torch.no_grad():
        t_logits = teacher(input_ids).logits
        t_log_probs = F.log_softmax(t_logits[:, :-1], dim=-1)  # [B, L-1, V]

    # Full-vocab reverse KL per state s_t: Σ_v π_θ(v) (log π_θ(v) − log π_T(v))
    kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # [B, L-1]

    # Mask to student-generated positions (KL on prompt tokens is meaningless)
    mask = action_mask[:, 1:].float()
    loss = (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0)

    # Diagnostic: at sampled positions, is the student more "confident" than the teacher (monitor over-confidence)
    with torch.no_grad():
        targets = input_ids[:, 1:].unsqueeze(-1)
        s_logp_t = s_log_probs.gather(-1, targets).squeeze(-1)
        t_logp_t = t_log_probs.gather(-1, targets).squeeze(-1)
        overconf_frac = ((s_logp_t > t_logp_t).float() * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {
        "full_vocab_kl": loss.item(),
        "overconf_frac": overconf_frac.item(),
    }
```

A few production-implementation details to note:

- **`action_mask`** must only mask student rollout tokens; KL on prompt tokens is meaningless (the teacher is only conditioning)
- **Use `torch.no_grad()` for the teacher forward**; otherwise memory doubles
- **Full-vocab (Route A) vs sampled-token (Route B)**: the loss above is **Route A** (summed over $V$ tokens, autograd directly backprops). **Route B** (sampled-token + REINFORCE/IS, §4.4) is the production-default form in implementations like Tinker / MiniLLM, sharing the sampled-token interface with PPO/GRPO and supporting logprob-only (gray-box) teachers (a truly sample-only black-box teacher needs the Black-Box OPD method in §6.5); A and B are **equivalent in expectation**. Regardless of which you choose, **you cannot** directly `.backward()` on sampled-token `log π_θ − log π_T` — that is equivalent to taking the pathwise $\nabla\log\pi_\theta(y_t)$ at a fixed sampled index, **losing the score-function term**, and is neither the reverse-KL gradient nor MLE. Route B must use a REINFORCE estimator (detached reward + score-function trick)
- **Batched teacher inference** is typically **asynchronous** in production: spin up a teacher server with vLLM, send the student rollout batch and fetch log-probs; only run student forward + backward locally. This is the standard for Tinker / vLLM-based frameworks
- **Mixed precision**: student in bf16, but compute teacher log-softmax in fp32 (avoid numerical instability)
- **The naive implementation's memory footprint is not small**: this function simultaneously materializes five $[B,L,V]$ BF16 tensors — `s_logits / s_log_probs / s_probs / t_logits / t_log_probs`. With the typical configuration used elsewhere in this tutorial (§4.2/§4.6: `batch_size=64`, `max_new_tokens=512`, vocab 100K-152K), a single tensor is about 6.5-10 GB, so the five together total about **33-50 GB** — already exceeding the activations budget in the §8.2 memory table. Production implementations typically use chunking / fused cross-entropy to avoid materializing five full-vocab tensors at once — see the note in §8.2

#### Shared helpers (used throughout §4.2 - §4.6)

```python
def per_token_logp(model, input_ids):
    """
    Returns log π(y_{t+1}|s_t) per position. Shape [B, L-1]. Grad behavior decided by the caller
    (want grad: call directly; want detached behavior log-prob: wrap in torch.no_grad()).
    """
    logits = model(input_ids).logits[:, :-1]                     # [B, L-1, V]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)                     # [B, L-1, 1]
    return log_probs.gather(-1, targets).squeeze(-1)             # [B, L-1]


def teacher_kl_reward(student, teacher, input_ids, action_mask):
    """
    The teacher-vs-behavior KL used as a trajectory-level reward in OPD-GRPO (computed per
    token then summed over the trajectory — NOT a dense per-token reward, see the
    .sum(dim=-1) at the end of this function):
        r_t = log π_T(y_t|s_t) - log π_θ_old(y_t|s_t)
    Input action_mask shape [B, L] (same as input_ids); internally shifted to [B, L-1].
    All under no_grad; returns scalar reward per trajectory (sum over generated tokens).
    """
    with torch.no_grad():
        s_old = per_token_logp(student, input_ids)               # [B, L-1]
        t_lp  = per_token_logp(teacher, input_ids)               # [B, L-1]
        token_mask = action_mask[:, 1:].float()                  # aligned to [B, L-1]
        per_tok = (t_lp - s_old) * token_mask
    return per_tok.sum(dim=-1)                                   # [B]
```

> ⚠️ **Sanity check before production** — the above is a **pedagogical sketch**: real production (verl / OpenRLHF / Tinker) splits `per_token_logp` into logits-then-gather to save memory, adds `attention_mask` to handle padding, uses a vLLM async server for the teacher rather than same-process forward. Sampler and grad-aware forward are usually separated as well (rollout phase calls `student.generate`, update phase uses grad-aware forward to compute `s_logp_new`). The code below omits these engineering details.

### 4.2　Full OPD training loop (with student rollout)

```python
@torch.no_grad()
def student_rollout(student, prompts, max_new_tokens=512, temperature=1.0):
    """Student samples trajectory on its own (the critical on-policy step)"""
    out = student.generate(
        prompts.input_ids,
        attention_mask=prompts.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        return_dict_in_generate=True,
    )
    full_ids = out.sequences           # [B, L_prompt + L_gen]
    prompt_len = prompts.input_ids.shape[1]
    action_mask = torch.zeros_like(full_ids)
    action_mask[:, prompt_len:] = 1
    # mask out pad
    action_mask = action_mask * (full_ids != student.config.pad_token_id).long()
    return full_ids, action_mask


def opd_train_step(student, teacher, optimizer, batch, kl_coef=1.0):
    """one step of OPD: sample rollout, compute teacher KL, update student"""
    # 1. student samples (on-policy)
    input_ids, action_mask = student_rollout(student, batch.prompts)

    # 2. compute per-token reverse KL with teacher
    loss, stats = per_token_reverse_kl_loss(
        student, teacher, input_ids, action_mask
    )

    # 3. backward
    optimizer.zero_grad()
    (kl_coef * loss).backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    return stats
```

This is the simplest OPD training loop. Production code adds advantage / baseline / clipping in step 2 (see §4.4), and group sampling (GRPO-style, see §4.5) in step 1.

### 4.3　Off-policy KD vs OPD comparison code (one line apart)

```python
# ── Off-policy KD (vanilla, Hinton-style) ──
def off_policy_kd_loss(student, teacher_outputs, dataset_y):
    """
    teacher_outputs computed by forwarding on dataset_y
    student also forwards on dataset_y
    forward KL: teacher → student
    """
    s_logits = student(dataset_y).logits
    t_logits = teacher_outputs                 # already computed
    return F.kl_div(
        F.log_softmax(s_logits, dim=-1),
        F.softmax(t_logits, dim=-1),
        reduction="batchmean",
    )


# ── On-Policy Distillation (OPD) ──
def opd_loss(student, teacher, prompts):
    student_y, mask = student_rollout(student, prompts)         # ← key diff: student samples on its own
    loss, _ = per_token_reverse_kl_loss(student, teacher, student_y, mask)
    return loss
```

**The only key difference**: OPD computes loss on its own sampled trajectory; off-policy computes it on dataset / teacher trajectory. The code differs by only one `student_rollout` call, but the overall benefit of **aligning training distribution with inference distribution** is enormous.

### 4.4　Control Variate Baseline (vOPD-style token-level KL)

> ⚠️ **Use case**: this section is the standard demonstration of Route B (sampled-token REINFORCE / importance sampling). **Route B is the production-default form for Tinker, MiniLLM, etc.**, **equivalent in expectation** to Route A (§4.1, full-vocab autograd); the trade-off is variance vs engineering interface: Route A has zero variance but needs teacher full logits; Route B fits the PPO/GRPO sampled-token interface, supports logprob-only (gray-box) teachers — it still needs the teacher to emit $\log \pi_T(y_t|s_t)$, so it is not a truly sample-only black-box API (that needs the Black-Box OPD method in §6.5) — and saves teacher vocab memory. The specific **closed-form baseline** variance-reduction trick can only be computed when teacher full logits are available; for a gray-box (logprob-only) teacher you must switch to a learned / EMA / per-prompt mean baseline — see the caveat at the end.

> Note: the vOPD ("KL for a KL") control-variate idea is a recurring pattern in the OPD literature (Survey 2026 / Tinker blog etc. all have equivalent discussions). Below gives the **correct detach + sign + estimator** form.

**REINFORCE-form sampled-token estimator**: for a single sampled token $y_t \sim \pi_\theta(\cdot|s_t)$, the unbiased estimate of the *gradient* of reverse KL is

$$\nabla_\theta D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)(s_t) = \mathbb{E}_{y_t \sim \pi_\theta}\!\big[\,\nabla_\theta \log \pi_\theta(y_t|s_t) \cdot \underbrace{(\log \pi_\theta(y_t|s_t) - \log \pi_T(y_t|s_t))}_{\hat r_t,\;\textbf{detached}}\,\big].$$

Note $\hat r_t$ only appears as a **scalar reward** (must be detached); the gradient comes entirely from the outer $\nabla_\theta \log\pi_\theta$. **Control variate** to reduce variance:

$$\hat r_t \leftarrow \hat r_t - B(s_t),\quad B(s_t) = \mathbb{E}_{y_t \sim \pi_\theta}[\hat r_t] = D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)(s_t).$$

$B(s_t)$ is the **full-vocab** KL at that step (with positive sign), computable in closed form from the student forward. Its detached value does not change the unbiasedness of the estimator (because $\mathbb{E}_{y_t}[\nabla \log\pi_\theta \cdot B(s_t)] = B(s_t)\cdot \mathbb{E}[\nabla\log\pi_\theta] = 0$), but significantly reduces variance.

```python
def vopd_token_kl_estimator(student, teacher, input_ids, action_mask):
    """
    Sampled-token REINFORCE estimator of ∇_θ D_KL(π_θ || π_T) with closed-form baseline.

    Returns surrogate loss L_surr, whose ∇L_surr in expectation = ∇ E[D_KL]. Note L_surr
    itself is NOT a numerical estimate of D_KL — for diagnostics use full-vocab D_KL.
    """
    s_logits = student(input_ids).logits[:, :-1]                    # [B, L-1, V]
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    s_probs     = s_log_probs.exp()
    with torch.no_grad():
        t_logits = teacher(input_ids).logits[:, :-1]
        t_log_probs = F.log_softmax(t_logits, dim=-1)

    targets = input_ids[:, 1:].unsqueeze(-1)                        # [B, L-1, 1]
    s_logp_t = s_log_probs.gather(-1, targets).squeeze(-1)          # grad-aware: log π_θ(y_t|s_t)

    # ── reward signal (must detach: scalar weight, not part of grad path) ──
    with torch.no_grad():
        t_logp_t = t_log_probs.gather(-1, targets).squeeze(-1)
        r_hat    = s_logp_t.detach() - t_logp_t                     # scalar per token

        # closed-form baseline B(s_t) = D_KL(π_θ || π_T)(s_t), full vocab
        baseline = (s_probs.detach() * (s_log_probs.detach() - t_log_probs)).sum(-1)
        advantage = r_hat - baseline                                # both detached

    # ── REINFORCE-style surrogate: ∇ = E[ ∇log π_θ(y_t|s_t) · advantage_detached ] = +∇D_KL ──
    mask = action_mask[:, 1:].float()
    surrogate = (s_logp_t * advantage * mask).sum() / mask.sum().clamp_min(1.0)
    #            ↑ Sign note: advantage = (logπ_θ − logπ_T) − B is the KL cost. To minimize D_KL,
    #            the optimizer step is θ ← θ − η·∇L_surr, and ∇L_surr = +E[∇logπ·advantage] = +∇D_KL,
    #            so the surrogate is **positive**; treat surrogate as the loss to .backward().

    # diagnostic (no grad): monitor the actual full-vocab KL
    with torch.no_grad():
        full_kl = (baseline * mask).sum() / mask.sum().clamp_min(1.0)
    return surrogate, {"full_vocab_kl_diag": full_kl.item()}
```

> 💡 **Recurring pitfalls in the code** —
> 1. **`r_hat` must be detached**: it is the reward signal, not part of the loss; if not detached, backward will pull both $\log\pi_T$ and the student log-prob into the gradient, completely different from the paper estimator.
> 2. **`baseline` must be detached**: the control variate cannot backprop; otherwise unbiasedness no longer holds.
> 3. **Sign of the surrogate**: we want to **minimize** $D_{\text{KL}}$. The REINFORCE identity gives $\nabla D_{\text{KL}} = \mathbb{E}[\nabla\log\pi_\theta\cdot \hat r]$, so set `loss = +E[\log\pi_\theta\cdot(\hat r - B).\text{detach}()]`, then `loss.backward()` gives $+\nabla D_{\text{KL}}$, and the optimizer step `θ -= η·∇L` performs KL descent. **Positive sign is correct** — it is easy to get the sign wrong.
> 4. **Route A vs Route B choice**: `full_kl_per_pos` (§4.1 Route A) has zero variance and is one-line autograd, but needs teacher full logits; this section's Route B fits the sampled-token interface (composes naturally with §4.5 PPO clipping), supports logprob-only (gray-box) teachers, saves vocab-size memory. The two are **equivalent in expectation**; pick by engineering constraints.
> 5. **Logprob-only (gray-box) teacher baseline choices**: if you can only query teacher log-prob and not the full vocab (note this is still a gray-box scenario, not a truly sample-only black-box API), the closed-form `B(s_t)` cannot be computed; you can switch to (a) `running mean of r_hat` as baseline; (b) learn a lightweight value head; (c) per-prompt empirical mean. You lose the closed-form advantage but still keep unbiased + significant variance reduction.

### 4.5　Integrating OPD into GRPO (multi-sample group baseline)

GRPO's ratio is **new student policy vs old behavior student policy** (record `s_logp_old` during rollout, compute `s_logp_new` during update); the teacher enters in two independent positions: (i) **reward** via a **trajectory-level scalar reward** obtained by summing the token-level KL reward $r_t^{\text{kl}} = \log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)$ over the trajectory (after group normalization this same advantage is broadcast to every token in the trajectory, so it does not provide token-level differential credit assignment — see `r_kl_dense`'s `.sum().item()` in the code below), weighted with the outcome reward to give the total reward; (ii) **KL regularization** via the explicit $D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)$ constraint on the student to stay close to the teacher — this is the term that is actually dense per-token supervision (closed-form full-vocab, §4.1). **The teacher never enters the denominator of the PPO ratio.**

```python
def opd_grpo_step(student, teacher, batch, G=8, alpha=0.5, kl_coef=0.1, clip=0.2):
    """
    OPD + GRPO hybrid (per-prompt group baseline):
      - sample G rollouts per prompt, record behavior-policy log-prob
      - reward = α * outcome_reward + (1-α) * sum_t [log π_T(y_t|s_t) - log π_θ_old(y_t|s_t)]
      - GRPO group-relative advantage (group-normalized)
      - ratio = exp(s_logp_new - s_logp_old.detach())  ← critical
      - KL penalty term uses student vs teacher (not the PPO ratio)

    Shape conventions:
      input_ids / action_mask:   [B, L]   (prompt + generated)
      per_token_logp output:     [B, L-1] (predict y_{t+1} from prefix up to t)
      token_mask = action_mask[:, 1:]: aligned with per-token logp
    """
    ids_all, action_mask_all, s_old_all, adv_all = [], [], [], []

    for prompt in batch.prompts:
        rollouts = []
        for _ in range(G):
            ids, action_mask = student_rollout(student, prompt)    # [B', L]; B'=1 for single prompt
            with torch.no_grad():
                s_old = per_token_logp(student, ids)               # [B', L-1] behavior log-prob
                t_lp  = per_token_logp(teacher, ids)               # [B', L-1]
            rollouts.append((ids, action_mask, s_old, t_lp))

        # total reward across G rollouts in the group
        rewards = []
        for (ids, action_mask, s_old, t_lp) in rollouts:
            r_outcome = math_verifier(ids)                         # scalar ∈ {0,1}
            token_mask = action_mask[:, 1:].float()                # [B', L-1]
            r_kl_dense = ((t_lp - s_old) * token_mask).sum().item()
            r_total = alpha * r_outcome + (1.0 - alpha) * r_kl_dense
            rewards.append(r_total)
        # Ensure same device/dtype as s_old / s_logp_new, avoid CPU/GPU mismatch
        rewards = torch.tensor(rewards, device=s_old.device, dtype=s_old.dtype)
        adv_per_traj = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # [G]

        for i, (ids, action_mask, s_old, _t_lp) in enumerate(rollouts):
            ids_all.append(ids); action_mask_all.append(action_mask)
            s_old_all.append(s_old)
            adv_all.append(adv_per_traj[i].view(1, 1).expand_as(s_old))     # broadcast to [B', L-1]

    ids         = torch.cat(ids_all, dim=0)                                  # [BG, L]
    action_mask = torch.cat(action_mask_all, dim=0)                          # [BG, L]
    token_mask  = action_mask[:, 1:].float()                                 # [BG, L-1]
    s_logp_old  = torch.cat(s_old_all, dim=0).detach()                       # [BG, L-1]
    A           = torch.cat(adv_all, dim=0).detach()                         # [BG, L-1]

    # ── new student log-prob (this step has grad) ──
    s_logp_new = per_token_logp(student, ids)                                # [BG, L-1]

    # ── PPO clipped ratio: new student vs old student (NOT student vs teacher) ──
    ratio = torch.exp(s_logp_new - s_logp_old)
    pg = torch.min(ratio * A, torch.clamp(ratio, 1 - clip, 1 + clip) * A)
    pg_loss = -(pg * token_mask).sum() / token_mask.sum().clamp_min(1.0)

    # ── KL penalty: student vs teacher (OPD reverse KL, full-vocab closed-form) ──
    # Pass [B, L] action_mask; §4.1 internally shifts it to [B, L-1]
    kl_loss, _ = per_token_reverse_kl_loss(student, teacher, ids, action_mask)

    return pg_loss + kl_coef * kl_loss
```

> 💡 **Critical details** —
> 1. **The ratio denominator is `s_logp_old` (behavior policy)**, computed once during rollout and `.detach()`-ed, unrelated to the teacher. This is identical to vanilla GRPO/PPO.
> 2. **The teacher's two roles do not overlap**: as a reward source providing `r_kl_dense` (a trajectory-level scalar obtained by summing the per-token KL, `.sum().item()`, then fed into advantage and broadcast back to every token via `expand_as` — uniform along the token dimension, not a token-level differential signal); as a KL anchor providing the closed-form reverse-KL penalty (the actual dense per-token supervision, fed directly into the loss).
> 3. **`per_token_reverse_kl_loss` is full-vocab closed-form** (§4.1), not a sampled-token estimator, so the KL term is unbiased and has low variance.
> 4. **Mini-batch multi-step updates**: in real GRPO, a rollout batch undergoes `n_epochs` inner updates, and the ratio is not 1 (critical, otherwise clipping has no effect).

Practical tip: **alpha = 0.5 is an empirical starting point**; alpha → 1 degenerates to pure GRPO (outcome only), alpha → 0 degenerates to pure OPD (KL only). Qwen3 / R1-Distill-style recipes usually stage-wise schedule: early-stage small alpha (pull student toward teacher first), late-stage large alpha (push for outcome SOTA).

### 4.6　Synthetic-data + distillation pipeline (top-level pseudocode)

> 📝 **This is architecture-level pseudocode** (not runnable Python): `sample / sft_train / math_verifier` etc. are placeholders, corresponding in specific frameworks to `torch.utils.data.DataLoader` / TRL `SFTTrainer` / custom reward function. Below only shows the **topology of the three-stage recipe**.

```python
def synthetic_distillation_pipeline(
    teacher,             # large teacher (e.g., R1-671B / Qwen3-32B)
    student,             # small student (e.g., Qwen3-8B-Base)
    prompt_pool,         # massive prompt pool (can be unlabeled instructions)
    n_stage1=800_000,    # off-policy SFT data volume
    n_stage2_steps=5_000,
    n_stage3_steps=1_000,
    verifiable_prompts=None,
):
    # ─── Stage 1: use teacher to generate synthetic trajectories, do off-policy SFT ───
    synthetic_data = []
    for prompt in random_sample(prompt_pool, n_stage1):           # placeholder sampler
        with torch.no_grad():
            y_teacher = teacher.generate(prompt, max_new_tokens=4096)
        synthetic_data.append((prompt, y_teacher))
    run_sft(student, synthetic_data, epochs=2)                    # cross-entropy on (prompt, y_teacher)

    # ─── Stage 2: do OPD on student's own rollouts ───
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    for step in range(n_stage2_steps):
        batch = make_batch(prompt_pool, batch_size=64)
        opd_train_step(student, teacher, optimizer, batch, kl_coef=1.0)

    # ─── Stage 3 (optional): OPD + outcome reward (math/code verifier) ───
    if verifiable_prompts is not None:
        for step in range(n_stage3_steps):
            batch = make_batch(verifiable_prompts, batch_size=64)
            loss  = opd_grpo_step(student, teacher, batch, G=8, alpha=0.5)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    return student
```

This is the standard three-stage form of Qwen3 / Thinking Machines-style recipes (R1-Distill's officially released pipeline corresponds only to Stage 1's off-policy SFT, without the on-policy Stage 2/3 parts — see §3.4). Stage 1 provides cold start (student learns to mimic teacher's style and format), Stage 2 uses OPD to eliminate exposure bias, Stage 3 uses a verifier for task-aligned fine-tuning.

## §5 OPD's Position in the Knowledge Distillation Family

### 5.1　LLM distillation method lineage (from Hinton to OPD)

```

   2015 ─────────── 2019 ─────── 2023 ────── 2024 ─────── 2025/2026
   Hinton          DistilBERT   MiniLLM     GKD          Thinking
   soft target     (Sanh)       (Gu)        (Agarwal)    Machines
                                                          blog
     │              │            │            │            │
     ↓              ↓            ↓            ↓            ↓
   classification   encoder-     first        generalized   "cheap RL"
   forward KL      level KD     reverse KL   JSD + mix     paradigm + large-
                                + on-policy  on/off        scale empirical
                                              policy        results (Qwen3/
                                                            MiMo/Kimi)
```

**Core evolution directions**:
- **off-policy → on-policy** (handles exposure bias)
- **forward KL → reverse KL / JSD** (handles mode-covering)
- **hard label → soft logit** (preserves dark knowledge)
- **single-stage → multi-stage hybrid** (off-policy SFT cold start + on-policy OPD + outcome RL)

### 5.2　One-sentence index of key papers

| Year | Paper | Key contribution |
|---|---|---|
| 2015 | Hinton, Vinyals, Dean — "Distilling the Knowledge in a Neural Network" (arXiv 1503.02531) | Pioneer of soft-target KD + temperature softmax |
| 2016 | Kim & Rush — "Sequence-Level Knowledge Distillation" (EMNLP, arXiv 1606.07947) | Pushed KD to seq-to-seq NMT, proposed seq-level / token-level KD distinction |
| 2019 | Sanh et al. — "DistilBERT" (NeurIPS workshop, arXiv 1910.01108) | Reduced parameters by 40% (retaining about 60% of the original parameter count), retained about 97% of GLUE performance, ~60% faster inference |
| 2020 | Sun et al. — "MobileBERT" (ACL, arXiv 2004.02984) | Task-agnostic KD + progressive distillation |
| 2023 | Gu et al. — "MiniLLM" (ICLR 2024, arXiv 2306.08543) | **First formal application of reverse KL + on-policy to LLMs**; policy gradient optimization of reverse KL |
| 2023 | Agarwal et al. — "GKD: On-Policy Distillation of LM" (ICLR 2024, arXiv 2306.13649) | Unifies forward/reverse KL + on/off-policy data, generalized JSD interpolation; $\lambda$ controls student data fraction |
| 2025 | DeepSeek-R1-Distill (DeepSeek 2025-01, arXiv 2501.12948) | Off-policy SFT distillation on 800K reasoning trajectories, 1.5B-70B family |
| 2025-05 | Qwen3 Tech Report (arXiv 2505.09388) | Production-grade OPD recipe: off-policy SFT cold start + on-policy distillation; saves 10× GPU hours vs RL |
| 2025-10 | Thinking Machines — "On-Policy Distillation" blog (Lu et al.) | Packages OPD as the "cheap RL" route; Qwen3-8B + Qwen3-32B-teacher reproduces RL gain with 9-30× FLOPs savings (direction credible, exact multiplier range [needs-verify]) |
| 2025-11 | Black-Box OPD (arXiv 2511.10643) | No teacher logits required; on-policy distillation using only teacher samples |
| 2026 | Song & Zheng — "A Survey of OPD for LLMs" (arXiv 2604.00626) | Formalizes OPD as $f$-divergence minimization on student rollouts; three-axis taxonomy |
| 2026 | "Rethinking OPD" (arXiv 2604.13016) | Phenomenology + mechanism + recipe: truncation collapse / mode-seeking failure / counterfactual regression |
| 2026 | "vOPD: KL for a KL" (arXiv 2605.07865) | Closed-form control-variate baseline for variance reduction |

### 5.3　OPD's position in Reasoning Model distillation (the R1-Distill era)

The two main lines of reasoning model distillation:

| Line | Representatives | Paradigm |
|---|---|---|
| **Off-policy SFT KD** | DeepSeek-R1-Distill, s1 (Muennighoff 2025), OpenThinker, Gemma 2/3 (soft-target KD on a fixed dataset, no student self-rollout) | teacher generates trajectories / soft labels → student SFT |
| **On-Policy Distillation (OPD)** | Qwen3, MiMo-V2, Thinking Machines | student samples → teacher provides per-token KL |

**When to use which**:
- If teacher trajectories are extremely high quality and the teacher-student capability gap is small → off-policy SFT alone is sufficient
- If the student is much smaller than the teacher (>5×), the task is long horizon, and the inference-time distribution is sensitive → OPD is the more stable choice
- In practice, the two are typically **stacked**: off-policy SFT for cold start, then OPD to eliminate exposure bias

## §6 Engineering Case Studies: OPD in Production

### 6.1　Qwen3 Recipe (arXiv 2505.09388 §3.2)

The post-training of Qwen3 small models (1.7B / 4B / 8B) uses two-stage distillation:

```
Qwen3-235B (teacher, dual /think + /no_think mode)
    │
    │ Stage 1: Off-policy distillation
    │   - teacher generates trajectories (mix of /think and /no_think)
    │   - student SFT on teacher trajectories (cross-entropy)
    │   - goal: basic reasoning + mode switching
    ↓
Qwen3-8B-mid
    │
    │ Stage 2: On-policy distillation (OPD)
    │   - student samples on its own → teacher gives per-token logits
    │   - per-token reverse KL loss
    │   - goal: eliminate exposure bias + push reasoning depth
    ↓
Qwen3-8B-final
```

**Reported results**:
- Saves ~10× GPU hours vs pure RL
- Significant pass@64 improvement on AIME'24 / AIME'25 (showing OPD does not collapse diversity)
- +3-5pp over off-policy SFT on long-CoT tasks

**[needs-verify]** The exact pass@64 numbers and the precise "+3-5pp" measurement protocol should be checked against the original Qwen3 Technical Report §3.2 (arXiv 2505.09388), consistent with the existing needs-verify convention in §A.4.

### 6.2　Thinking Machines Blog (Lu 2025-10-27)

Experimental setting:
- **Student**: Qwen3-8B-Base
- **Teacher**: Qwen3-32B
- **Task**: math reasoning (AIME'24 as primary benchmark)
- **Init**: one SFT round with OpenThoughts data
- **OPD**: implemented with Tinker, per-token reverse KL, no outcome reward

Reported results (blog content):
- Training Qwen3-8B-Base with OPD reaches almost identical AIME'24 performance to RL
- **Total FLOPs saved by ~9-30×**, depending on batch / lr settings
- Emphasizes "**OPD is the cheap alternative to RL**", not a supplement to RL

### 6.3　Tinker Cookbook implementation details

GitHub: `thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation`

Key design points:
- **Environment is designed to be no-reward**: the only supervision is the teacher KL
- **Reverse KL implementation**: `(student_logp - teacher_logp) * mask`
- **Advantage computation**: treat KL as negative reward fed into advantage (`advantage = -kl_penalty_coef * reverse_kl`)
- **LoRA**: rank 128 + lr 1e-4
- **Batch**: groups_per_batch = 64 (GRPO-style grouping)

This "inject KL as reward into advantage" design exactly corresponds to the OPD-GRPO equivalence in §3.3: replace the sparse outcome reward with dense teacher KL reward.

### 6.4　DeepSeek-V4 report (multi-teacher OPD replaces RL)

**[needs-verify]** Details of DeepSeek-V4 and its multi-teacher OPD consolidation come from secondary/inferred information and have not been checked against an original technical report, consistent with the existing needs-verify flag on the same claim (L3-3) in §A.4.

DeepSeek-V4 in the model consolidation stage **completely replaces mixed RL with multi-teacher OPD** — the logits of multiple teachers (specialists: math / code / reasoning / chat) are weighted-ensembled as the OPD target for the student. This is the representative case of extending the OPD paradigm to multi-teacher and specialist consolidation.

### 6.5　Black-Box OPD (no teacher logits required)

**arXiv 2511.10643** "Black-Box On-Policy Distillation": when the teacher is a closed-source API (e.g., GPT-4 / Claude) from which you can only sample but cannot extract logits, how do you do OPD?

Core idea: use **teacher samples as trajectory-level reward**, combined with token-level student log-prob to approximate reverse KL. This extends OPD from "requires teacher logits" to "only needs teacher API". The trade-off is losing per-token dense supervision and falling back to trajectory-level reward — essentially a hybrid between OPD and RL.

## §7 Failure Modes and Mitigation Strategies

### 7.1　Truncation Collapse / Length Inflation

**Phenomenon** (Demystifying OPD 2026, arXiv 2604.08527): during training, the average length of student rollouts keeps growing, eventually hitting max_length truncation; the truncated trajectories **contribute most of the gradient** (because they have many tokens), causing gradient bias and validation performance crash.

**Root cause**: the mode-seeking nature of reverse KL biases the student toward "keep emitting high-teacher-probability tokens", and the teacher naturally favors long output on long-CoT. This forms a **positive feedback loop**.

**Mitigation**:
- **Length normalization**: divide the loss by trajectory length (per-token average, not sum) — this is the default in Tinker / most production implementations
- **Length penalty**: add $-\lambda \cdot L$ to the reward to suppress overly long outputs
- **Max-length much larger than sample max-length**: avoid truncation entering training
- **Early stop on val loss spike**: stop immediately when val crashes

### 7.2　Mode Collapse / Diversity Loss

**Phenomenon**: reverse KL is mode-seeking, so in theory the student should lock onto one mode of the teacher; but on some tasks (e.g., open-ended generation) the student converges to a **single answer template**, losing diversity, with pass@N ($N > 1$) dropping.

**Mitigation**:
- **Use forward KL / JSD instead of reverse KL** (GKD's $\beta$ interpolation)
- **Add entropy bonus** (similar to PPO)
- **Mixed loss**: reverse KL × $\alpha$ + forward KL × $(1-\alpha)$
- **MiniLLM's stabilization trick**: mixed policy $\pi_{\text{mix}} = (1-\alpha)\pi_\theta + \alpha\pi_T$ with $\alpha = 0.2$ to prevent early mode collapse

### 7.3　Reward Hacking (teacher-game)

**Phenomenon**: the student learns to hack — outputs some **tokens that have high teacher probability but are semantically garbage** (e.g., stopword sequences, repeated phrases), with KL loss extremely low but task performance poor.

**Mitigation**:
- **Mix in outcome reward** (the OPD-GRPO hybrid in §4.5)
- **Trajectory-level filter**: filter out trajectories where the teacher itself is uncertain ("teacher-improbable filter")
- **Token diversity regularizer**: penalize overly low token entropy

### 7.4　Prefix Teach, Suffix Fade (local teachability collapse)

**arXiv 2605.13643** reports: in late training, the teacher has almost "nothing left to teach" in the latter half of trajectories — the student's and teacher's log-probs on the suffix are already close, and the loss comes almost entirely from the prefix. This leads to poor quality of the second-half reasoning in long-CoT tasks.

**Mitigation**:
- **Token-level importance weighting**: weight suffix tokens more
- **Teacher upgrade**: use a stronger teacher (multi-teacher ensemble / specialists for different tasks)
- **Curriculum**: start with short trajectories, gradually increase max_length

### 7.5　Teacher / Student Gap too large

**Phenomenon**: when the student is much smaller than the teacher (e.g., 1B vs 70B), the student's samples cannot reach regions of high teacher probability ("student visits states the teacher never thought of"). This is not "the signal vanishing": the sampled-token reverse KL term $\log\pi_\theta(y_t|s_t) - \log\pi_T(y_t|s_t)$ **tends to $+\infty$** (an extreme penalty / numerical instability) as $\pi_T(y_t|s_t)\to 0$, because $\log\pi_T(y_t|s_t)\to -\infty$. The real problem is: (a) the student's own sampling distribution may never cover the teacher's high-probability tokens (a mode-covering deficiency, giving a poorly directed learning signal); (b) whenever it does sample a token the teacher considers extremely unlikely, that step produces a numerically extreme (not vanishing) loss.

**Mitigation**:
- **off-policy SFT warm start**: let the student first imitate the teacher for a while to bring the state distributions closer, then run OPD
- **Temperature scheduling**: high student temperature early in training to enlarge exploration; reduce to 1.0 later
- **Use a closer teacher**: control the teacher-student gap to 4-8× (Qwen3 choosing a 32B teacher → 8B student reflects this consideration)

### 7.6　Catastrophic Forgetting (loss of old capabilities)

**Phenomenon**: OPD is mainly trained on math / reasoning, but the student's general capabilities (chat / safety / instruction-following) degrade.

**Mitigation**:
- **Replay buffer**: retain some general SFT data and mix into OPD training in proportion
- **Regularization to SFT init**: add a $\beta \cdot D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{SFT-init}})$ penalty
- **Multi-task OPD**: distill multiple domain teachers simultaneously (math + chat + code)

## §8 Complexity and Resources

### 8.1　Per-step training cost comparison

| Paradigm | Student forward | Student backward | Teacher forward | Sampling cost | Total wall-clock |
|---|---|---|---|---|---|
| **SFT** | 1× | 1× | 0 | 0 (dataset provides) | 1× |
| **Off-policy KD** | 1× | 1× | 1× (pre-computed cache) | 0 | 1×-1.2× |
| **OPD** | 1× | 1× | **1× per token** | **student rollout ($O(L)$ tokens)** | **2-3×** (depending on teacher size) |
| **RLHF + PPO** | 1× | 1× | 0 | student rollout + RM forward | 2-4× |
| **GRPO** | $G$× (group sample) | 1× | 0 | $G$× student rollout | $G \cdot 1.5$× |

OPD's extra overhead mainly comes from: (1) student rollout (sampling is slower than forward due to KV cache accumulation); (2) teacher forward (teacher is large). But compared to RL, **OPD does not need reward model training + RM forward**, so the total cost is usually **less than a pure RL pipeline**.

### 8.2　Memory

| Component | Memory (8B student + 32B teacher example) |
|---|---|
| Student weights + optimizer | ~60 GB (Adam state + bf16 weight + bf16 grad) |
| Teacher weights (frozen, bf16) | ~64 GB |
| Student activations | ~10-20 GB (depending on batch / seq) |
| Teacher activations (no grad) | ~5-10 GB |
| KV cache (student rollout) | ~5-10 GB |
| **Total** | **~150-170 GB** → does not fit on a single H100 (80GB); needs 2-4 GPUs |

> ⚠️ **This table assumes memory-saving engineering optimizations; running the §4.1 naive code as-is exceeds the activations budget above** — with the typical configuration used elsewhere in this tutorial (`batch_size=64`, `max_new_tokens=512`, vocab 100K-152K), the §4.1 Route A tutorial code simultaneously materializes five $[B,L,V]$ BF16 tensors — `s_logits / s_log_probs / s_probs / t_logits / t_log_probs` — totaling about **33-50 GB**, which already exceeds the table's "Student activations (~10-20 GB) + Teacher activations (~5-10 GB)" budget of 15-30 GB. The table's numbers implicitly assume chunked computation / fused cross-entropy or similar memory-saving optimizations; a naive implementation without these optimizations has significantly higher actual overhead.

**Memory reduction tricks**:
- Run the teacher on a **separate inference server** (vLLM), so the student training machine does not store teacher weights
- Teacher uses **fp8 / int8 quantization** (teacher only forwards, no backprop; precision loss is small)
- Student uses **LoRA** (rank 128) + full teacher weights
- **Async teacher inference**: while the student trains batch $N$, the teacher server computes batch $N+1$

### 8.3　Compute Efficiency

The core data point from the Thinking Machines blog:

| Setting | AIME'24 accuracy | Total FLOPs |
|---|---|---|
| Qwen3-8B-Base + SFT only (OpenThoughts) | ~40% | 1× baseline |
| + RL (GRPO) | ~62% | 10-30× baseline |
| **+ OPD (Qwen3-32B teacher)** | **~62%** | **~1× baseline** |
| + RL trained to match OPD | 62% | ~10× baseline |

**OPD is roughly 9-30× more compute-efficient (measured by Total FLOPs) at the same accuracy** — this is the core number behind OPD's rise. Note this is efficiency along the FLOPs/compute axis, a different quantity from "how many rollout samples are needed" (sample efficiency).

## §9 Comparison and Positioning with Related Methods

### 9.1　Big table (the main interview targets in a cheat sheet)

| Method | Supervision source | Data collection | KL direction | Critic | Use case |
|---|---|---|---|---|---|
| **SFT** | dataset label | offline | n/a (CE) | no | warm start, instruction follow |
| **Hinton KD** | teacher logits | offline (dataset) | forward | no | single-step prediction, classification |
| **Kim-Rush Seq KD** | teacher beam search | offline | n/a (hard) | no | NMT, autoregressive |
| **DistilBERT** | teacher logits + MLM | offline | forward + CE | no | encoder model compression |
| **DPO** | offline preference pair | offline | closed-form | no | no strong teacher, have preference data |
| **PPO RLHF** | learned RM | on-policy rollout | reverse (vs ref) | yes (value head) | have RM, want strict RM-aligned |
| **GRPO** | learned RM / verifier | on-policy rollout (group) | reverse (vs ref) | no (group baseline) | math / code, save critic |
| **MiniLLM** | teacher logits | on-policy student rollout | reverse | no | LLM instruction tuning |
| **GKD** | teacher logits | mix offline + on-policy ($\lambda$) | JSD ($\beta$) | no | flexible, trade off offline/online |
| **OPD** (general) | teacher logits per token | **on-policy student rollout** | **reverse (or JSD)** | no (baseline closed-form) | small student, long-CoT, strong teacher |
| **OPD + GRPO** | teacher logits + outcome verifier | on-policy group | reverse | no | math + dense supervision |
| **R1-Distill** | teacher trajectory | offline | n/a (CE) | no | massive synthetic SFT |

### 9.2　Decision tree: which method for which scenario

```
Do you have a strong teacher model?
├── No
│   ├── have preference dataset → DPO
│   ├── have verifier (math/code) → GRPO
│   └── have RM → PPO RLHF
└── Yes
    ├── teacher-student gap small (< 4×) → off-policy SFT KD is enough
    ├── gap moderate (4-10×) + long CoT task → OPD (first choice)
    ├── gap large (> 10×) + complex task → off-policy SFT warm start + OPD
    ├── also have verifier → OPD + GRPO hybrid (production standard)
    └── teacher only API-accessible → Black-Box OPD
```

## §10 25 Frequently-Asked Interview Questions

### L1 must-know (10 questions, post-training engineer / LLM RL roles)

<details>
<summary><strong>L1-1: What is OPD? Why is it called "On-Policy"?</strong></summary>

**Answer**: OPD = On-Policy Distillation. "On-policy" means **the training data (trajectories) comes from the student's own current policy $\pi_\theta$ sampling**, rather than from teacher / dataset. The teacher only provides per-token supervision (typically reverse KL) on the states the student itself visits. This contrasts with off-policy KD (teacher generates data → student imitates).
</details>

<details>
<summary><strong>L1-2: What are the core differences between OPD and vanilla KD (Hinton)?</strong></summary>

**Answer**: Three points: (1) **Data source** — vanilla KD uses dataset / teacher-generated data, OPD uses student's own rollouts; (2) **KL direction** — vanilla KD uses forward KL (mode-covering), OPD primarily uses reverse KL (mode-seeking); (3) **Problem solved** — vanilla KD solves "compress teacher knowledge", OPD solves "compression + exposure bias" (train/test distribution mismatch in autoregressive generation).
</details>

<details>
<summary><strong>L1-3: Write out OPD's reverse KL loss formula.</strong></summary>

**Answer**:
$$L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]$$
Key points: the expectation subscript is $y \sim \pi_\theta$ (on-policy); the KL direction is $\pi_\theta$ first (reverse / mode-seeking). Sampled-token approximation = $\log \pi_\theta(y_t) - \log \pi_T(y_t)$.
</details>

<details>
<summary><strong>L1-4: What is exposure bias? How does OPD solve it?</strong></summary>

**Answer**: exposure bias = autoregressive models train with prefixes from ground-truth / teacher ("perfect prefixes"), but at inference prefixes come from the model itself (with errors) — train/inference distribution mismatch. Theoretically the cumulative error scales as $O(L^2)$ (Bagnell 2010). OPD computes loss on the student's own rollout trajectories, making training distribution = inference distribution, compressing cumulative error to $O(L)$.
</details>

<details>
<summary><strong>L1-5: What is the difference between reverse KL and forward KL? Why does OPD tend to use reverse?</strong></summary>

**Answer**: forward KL $D(\pi_T \,\|\, \pi_\theta)$ takes the expectation under the teacher, **forcing the student to cover all teacher modes** (mode-covering), prone to blurry averaged outputs; reverse KL $D(\pi_\theta \,\|\, \pi_T)$ takes the expectation under the student, **forcing the student to avoid tokens the teacher considers impossible** (mode-seeking / zero-forcing), prone to sharp confident outputs. LLM generation tasks usually want "fluent + confident", so reverse KL is the first choice.
</details>

<details>
<summary><strong>L1-6: What does OPD need from the teacher? Can OPD be done with only a teacher API (no logits)?</strong></summary>

**Answer**: standard OPD requires the teacher to compute **logits** or **log-probabilities** on every prefix the student visits. If the teacher is a closed-source API (e.g., GPT-4) that returns only samples but not logits, you can use "Black-Box OPD" (arXiv 2511.10643): use teacher samples as trajectory-level reward, combined with student log-prob for approximation — but you lose per-token dense supervision, falling back to trajectory-level, with performance between OPD and RL.
</details>

<details>
<summary><strong>L1-7: What are the core differences between OPD and RL? Why is OPD more sample efficient?</strong></summary>

**Answer**: RL provides one $O(1)$ scalar reward per trajectory (outcome); OPD provides $O(N \log V)$ bits (the entire teacher distribution at each token). So for the same number of rollouts, OPD provides **one to two orders of magnitude more supervision information**. Thinking Machines reports OPD reaches RL performance on math reasoning with 1/9-1/30 of the compute.
</details>

<details>
<summary><strong>L1-8: Does OPD need a critic / value model?</strong></summary>

**Answer**: **No**. Under the semi-gradient (B1, stop-grad on state visitation) approximation this tutorial uses, $D_{\text{KL}}(\pi_\theta(\cdot|s_t) \,\|\, \pi_T(\cdot|s_t))$ is the closed-form conditional expectation of that step's **immediate (single-step) reward**, directly readable from already-computed student and teacher logits, and is used as the zero-bias control variate / baseline for that step's REINFORCE estimator ("KL for a KL", arXiv 2605.07865) — no separate critic training needed, which is one of OPD's engineering advantages over PPO. Strictly speaking, though, it is not the true state-value function of the full trajectory objective $V(s_t) = \mathbb{E}[\sum_{t'\ge t} D_{\text{KL}}(s_{t'})]$ (which would require a return-to-go sum of expected future KL — the full B2 form that §2.3 notes "essentially nobody does"); calling the single-step conditional expectation $V(s_t)$ is a simplified shorthand.
</details>

<details>
<summary><strong>L1-9: What is the relationship between OPD and DPO? Can the two be stacked?</strong></summary>

**Answer**: DPO is offline, binary preference, closed-form RLHF; OPD is online, per-token teacher logit, policy-gradient KD. The two are **almost orthogonal**: DPO does not need a teacher, OPD does not need preference pairs. They can be stacked: first DPO to align human preference (learn "what is a good response"), then OPD to distill to a small student (learn "how to generate").
</details>

<details>
<summary><strong>L1-10: What is the most common failure mode of OPD in production? How to mitigate?</strong></summary>

**Answer**: **Length inflation / truncation collapse** — student rollouts grow longer over training, and after hitting max_length, the truncated trajectories dominate the gradient, causing val to crash. Mitigate: (1) loss per-token average rather than sum; (2) length penalty; (3) max_length 2-4× larger than sample max; (4) monitor val and stop immediately on spike.
</details>

### L2 advanced (10 questions, senior post-training / paper reproduction)

<details>
<summary><strong>L2-1: Derive the policy gradient form of OPD (including the two routes).</strong></summary>

**Answer**: reverse KL under the trajectory expectation:
$$L(\theta) = \mathbb{E}_{s_t \sim \rho_{\pi_\theta}}\!\big[D_{\text{KL}}(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t))\big].$$

$\theta$ appears simultaneously in (a) the **inner KL** ($\pi_\theta(\cdot|s_t)$ itself) and (b) the **state visitation** $\rho_{\pi_\theta}$. This results in **two distinct gradient computations** that are often conflated in the literature:

**(1) REINFORCE estimate of the inner KL** (fixed-state $s_t$):
$$\nabla_\theta D_{\text{KL}}(s_t) = \mathbb{E}_{y_t \sim \pi_\theta(\cdot|s_t)}\!\big[\nabla\log\pi_\theta(y_t|s_t)\cdot G_t^{\text{detach}}\big],\;\; G_t = \log\tfrac{\pi_\theta(y_t|s_t)}{\pi_T(y_t|s_t)}.$$
(Derivation: $\nabla\sum_v \pi_\theta(v)\log\tfrac{\pi_\theta(v)}{\pi_T(v)} = \sum_v\nabla\pi_\theta(v)\cdot(\log\tfrac{\pi_\theta}{\pi_T}+1) = \mathbb{E}_{y}[\nabla\log\pi_\theta(y)\cdot G + \nabla\log\pi_\theta(y)]$; the second term $\mathbb{E}[\nabla\log\pi]=0$.) This is the estimator used by §4.4 vOPD.

**(2) Full policy gradient of the trajectory objective** (including state visitation):
$$\nabla L = \mathbb{E}_\tau\!\Big[\underbrace{\sum_u \nabla\log\pi_\theta(a_u|s_u)\cdot {\textstyle\sum_{t\ge u}} D_{\text{KL}}(s_t)^{\text{detach}}}_{\text{return-to-go score term}} + \underbrace{\sum_t \nabla D_{\text{KL}}(s_t)}_{\text{inner grad}}\Big].$$
Note the score-function weight is **the sum of KLs over all future steps**, not the same-token $G_t$.

**Practical rules in production**:
- **Route A (pedagogically clear)**: stop-grad on $\theta$ during rollout (i.e., $\rho_{\pi_{\theta^-}}$ is fixed), only backprop through the inner full-vocab KL (autograd handles it), **no REINFORCE needed**. Equivalent to a "semi-gradient" approximation. Advantages: zero variance, one-line PyTorch; requires teacher full logits available. ("Zero variance" specifically means the inner full-vocab KL, given a fixed prefix, eliminates the Route B1 sampling-estimation variance; the training pipeline as a whole still carries cross-batch variance from the student's sampling during rollout.)
- **Route B (common in production)**: sampled-token REINFORCE / importance-sampling estimator + control variate (§4.4); shares sampled-token interface with PPO/GRPO, saves teacher vocab memory, supports logprob-only (gray-box) teachers (a truly sample-only black-box teacher needs §6.5's Black-Box OPD). Both MiniLLM (Gu 2024) and Thinking Machines Tinker's open-source implementations follow this route, but they approximate to different degrees: Tinker is closer to the fixed-state estimator in (1), while **MiniLLM actually uses a single-step decomposition with a future log-ratio return** ($R_t=\sum_{t'\ge t} r_{t'}$), a variance-reduced approximation of (2), not a plain semi-gradient. **It is the full, un-reduced (2) that nobody really does** — return-to-go accumulates across steps, variance explodes; production practice sits close to (1) or uses a MiniLLM-style reduced approximation.
- You **cannot** directly `.backward()` on sampled-token $\log(\pi_\theta/\pi_T)$: that is pathwise + fixed index, dropping the score-function term, and is neither the KL gradient nor MLE.

**A and B are equivalent in expectation**; the difference is in variance vs engineering interface trade-off.

> "OPD can wrap around PPO/GRPO" refers to §4.5 where the dense token KL is **used as a reward** fed into GRPO's advantage (the PPO ratio is still new student vs old student); it does **not** mean "the inner KL gradient form is equivalent to PPO" — this is a common conceptual confusion.
</details>

<details>
<summary><strong>L2-2: What are the key differences between GKD and MiniLLM? What do $\lambda$ and $\beta$ control respectively?</strong></summary>

**Answer**: **MiniLLM** (Gu 2024): pure reverse KL on student rollout + REINFORCE optimization + a few stability tricks (mixed policy $\pi_{\text{mix}} = (1-\alpha)\pi_\theta + \alpha\pi_T$, $\alpha = 0.2$ to prevent collapse; length penalty). **GKD** (Agarwal 2024): generalized framework with two hyperparameters: (1) $\lambda \in [0, 1]$ controlling the **student-generated data fraction** ($\lambda=0$ fully off-policy on dataset $\hat y$, $\lambda=1$ fully on-policy on student $y$); (2) $\beta \in [0, 1]$ controlling **the generalized JSD interpolation** — taken strictly as limits, the direction is the opposite of naive intuition: the one-sided limit as $\beta \to 0$ is reverse KL ($D_\beta/\beta \to D_{\text{KL}}(\pi_\theta\|\pi_T)$), and as $\beta \to 1$ is forward KL ($D_\beta/(1-\beta) \to D_{\text{KL}}(\pi_T\|\pi_\theta)$), with $D_\beta$ itself identically 0 at $\beta=0,1$ (see the endpoint note in §2.1). GKD is a superset of MiniLLM.
</details>

<details>
<summary><strong>L2-3: How is OPD integrated into GRPO? Write out the mixed reward and the correct ratio.</strong></summary>

**Answer**: two key things.
**(1) Mixed reward** (per-trajectory scalar, fed into group-relative advantage):
$$R(x,y) = \alpha\cdot R_{\text{outcome}}(x,y) + (1-\alpha)\cdot \sum_t \big(\log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)\big),$$
$\alpha=0$ degenerates to pure OPD, $\alpha=1$ degenerates to pure GRPO, production $\alpha\in[0.3,0.7]$.

**(2) Writing the ratio (most error-prone)**: the GRPO/PPO ratio is
$$\rho_t = \exp\big(\log\pi_\theta^{\text{new}}(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)\big),$$
i.e., **new student vs old behavior student** — **the teacher never enters the ratio denominator**. The teacher appears in two independent positions only: (i) as a reward source (above $r_t^{\text{KL}}$); (ii) as the reference distribution in the explicit KL penalty $D_{\text{KL}}(\pi_\theta\|\pi_T)$ (this term is closed-form differentiable, §4.1 implementation).

Final loss:
$$\mathcal{L} = -\mathbb{E}\!\left[\sum_t \min(\rho_t A_t, \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)\right] + \beta\, D_{\text{KL}}(\pi_\theta\,\|\,\pi_T).$$

Advantage $A$ uses GRPO's group-internal z-score; no critic needed. Code in §4.5.
</details>

<details>
<summary><strong>L2-4: What is vOPD's control variate? Why is it "free"?</strong></summary>

**Answer**: vOPD ("KL for a KL", a control-variate idea frequently cited in the 2026 Survey) is applied to Route B (sampled-token REINFORCE estimator) to reduce variance. For a single sampled $y_t \sim \pi_\theta$, the token-level reward $\hat r_t = \log\pi_\theta(y_t|s_t) - \log\pi_T(y_t|s_t)$ (**must be detached**), and the baseline is $B(s_t) = D_{\text{KL}}(\pi_\theta\,\|\,\pi_T)(s_t)$ (also detached):

$$\nabla_\theta L \approx \mathbb{E}\!\big[\nabla\log\pi_\theta(y_t|s_t)\cdot (\hat r_t - B(s_t))\big].$$

$B(s_t)$ is the conditional expectation of $\hat r_t$ under $y_t \sim \pi_\theta(\cdot|s_t)$; adding it **preserves unbiasedness** (because $\mathbb{E}_y[\nabla\log\pi_\theta(y)\cdot B(s_t)] = B(s_t)\cdot\mathbb{E}[\nabla\log\pi_\theta] = 0$) and **usually significantly reduces variance**. The strict minimum-variance baseline is the score-norm-weighted $\mathbb{E}[\|g\|^2 r]/\mathbb{E}[\|g\|^2]$ (generally not equal to the conditional mean), but the conditional mean works well enough in practice. **"Free"** means $B(s_t)$ is the full-vocab closed-form KL computed synchronously with the student forward (same logits → softmax → sum), requiring no extra critic / inference.

> ⚠️ Two errors that occur over and over in implementation: (a) both `r_hat` and `baseline` must be `.detach()`, otherwise unbiasedness fails and the gradient is distorted; (b) the surrogate sign is **positive**: because $\nabla D_{\text{KL}} = \mathbb{E}[\nabla\log\pi_\theta\cdot\hat r]$, we set `loss = +E[\log\pi_\theta\cdot(\hat r - B).detach()]`, then `∇loss = +∇D_{\text{KL}}`, and the optimizer step `θ -= η∇L` performs KL descent. **A negative sign would cause KL to increase** — code in §4.4.
</details>

<details>
<summary><strong>L2-5: What are the differences between Qwen3's OPD recipe and Thinking Machines blog's implementation?</strong></summary>

**Answer**: **Qwen3** uses two stages of off-policy SFT cold start + on-policy distillation, targeting end-to-end small-model building; **the Thinking Machines blog** emphasizes OPD as the cheap RL alternative, with the core experiment being "reproduce RL's AIME'24 gain with OPD, saving 9-30× FLOPs". On technical details: Qwen3 simultaneously distills /think and /no_think modes (dual-mode logits); Thinking Machines uses Tinker with advantage formally injecting KL as negative reward (OPD-RL perspective). Both are essentially reverse KL on student rollouts; the differences lie in multi-task / multi-mode distillation handling.
</details>

<details>
<summary><strong>L2-6: Trade-off between "sampled-token KL" and "full-vocab KL" in OPD?</strong></summary>

**Answer**: (a) **sampled-token (REINFORCE / importance-sampling, Route B)**: reward uses $G_t = \log \pi_\theta(y_t) - \log \pi_T(y_t)$.detach() (query the teacher for one log-prob only), grad = $\nabla\log\pi_\theta(y_t)\cdot G_t$; cheap but high variance. **Both MiniLLM (Gu 2024) and Thinking Machines Tinker open-source implementations are in this sampled-token PG/IS form** (Tinker `train_on_policy.py` uses `loss_fn="importance_sampling"` + `incorporate_kl_penalty`; MiniLLM uses single-step decomposition + length norm + teacher-mixed sampling and other tricks to stabilize variance). (b) **full-vocab (Route A)**: loss = $\sum_v \pi_\theta(v)(\log \pi_\theta(v) - \log \pi_T(v))$, direct autograd; zero variance but requires teacher full logits + whole-vocab memory; GKD and similar conceptual derivations often use this form as a pedagogical starting point, and it is also the simplest engineering implementation when teacher full logits are available. **Both are equivalent in expectation**, with the trade-off: Route A has zero variance and one-line PyTorch but needs full logits + vocab memory; Route B fits the PPO/GRPO sampled-token interface, supports logprob-only (gray-box) teachers (a truly sample-only black-box teacher needs §6.5's Black-Box OPD), saves vocab memory, but needs a control variate (§4.4) to reduce variance.
</details>

<details>
<summary><strong>L2-7: Why does OPD work better than off-policy KD on long-CoT tasks?</strong></summary>

**Answer**: on long-CoT, the gap between the student's own rollout state distribution and the teacher rollout state distribution is larger (error accumulation $O(L^2)$). With off-policy KD, the student sees only teacher prefixes (smooth, correct) during training; at inference it encounters its own erroneous prefixes that it has never seen — errors compound as $O(L^2)$ (not exponentially, consistent with §1.3/L1-4). OPD trains directly on the student's own erroneous prefixes, with the teacher supervising "how I would recover" — directly training the "error recovery ability".
</details>

<details>
<summary><strong>L2-8: If the student-teacher gap is huge (e.g., 1.5B vs 671B), does OPD fail? How to mitigate?</strong></summary>

**Answer**: **yes, it fails** — the student's samples are very likely to fall in regions of extremely low teacher probability. The failure mechanism here is not "the KL signal vanishing": whenever such a token is sampled, $\log\pi_T(y_t|s_t)\to -\infty$ makes that token's reverse-KL term **tend to $+\infty$** (numerically extreme, unstable gradients); what actually drags down training when the gap is large is that the student's own sampling distribution may never cover the teacher's high-probability tokens, giving a poorly directed learning signal ("the student looks endorsed by the teacher but is actually just rambling"). Mitigate: (1) **off-policy SFT warm start**: first have the student SFT on teacher trajectories to bring state distributions closer, then OPD; (2) **intermediate teacher**: use a medium-sized teacher (e.g., 70B) as a bridge; (3) **curriculum**: start short, gradually lengthen; (4) **temperature scheduling**: high student temperature early to enlarge exploration.
</details>

<details>
<summary><strong>L2-9: Are OPD and R1-Distill the same kind of method?</strong></summary>

**Answer**: **No**. R1-Distill is **off-policy** SFT distillation — the R1 teacher offline generates 800K trajectories, and the student does token-level cross-entropy on this data, **without student rollouts or KL signal**. OPD is on-policy + teacher KL. The two are complementary: R1-Distill can serve as cold-start init for OPD (first imitate teacher style), then OPD eliminates exposure bias.
</details>

<details>
<summary><strong>L2-10: How to diagnose whether OPD training is healthy? What are the key monitoring quantities?</strong></summary>

**Answer**: monitor five things: (1) **per-token reverse KL** — should decrease monotonically, but not hit 0 (hitting 0 = student fully matches teacher, possibly overfitting); (2) **rollout length / truncation rate** — truncation rate < 5%, otherwise length collapse; (3) **token entropy** — decreasing but should not approach 0 (mode collapse); (4) **pass@1 vs pass@N (N>1)** — pass@1 rising while pass@64 falling = diversity collapse; (5) **val accuracy on held-out** — this is the ultimate signal; stop immediately on spike.
</details>

### L3 top labs (5 questions, research / algorithm lead)

<details>
<summary><strong>L3-1: Unify OPD with RLHF / GRPO from a KL-constrained RL perspective.</strong></summary>

**Answer**: KL-constrained RL objective:
$$\max_\theta\, \mathbb{E}_{y \sim \pi_\theta}[R(x, y)] - \beta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\big]$$
- **RLHF / PPO**: $R$ = RM scalar reward, $\pi_{\text{ref}}$ = SFT init, small $\beta$
- **GRPO**: same as RLHF, but advantage uses group-normalization instead of GAE
- **OPD**: $R \equiv 0$ (no external reward), $\pi_{\text{ref}} = \pi_T$ (reference = teacher), $\beta = 1$
- **OPD + GRPO** (production standard): $R$ = outcome verifier + dense teacher KL reward, $\pi_{\text{ref}} = \pi_T$

The Survey (arXiv 2604.00626) collectively calls this "$f$-divergence minimization on student rollouts"; OPD is the $R \equiv 0$ special case, and DPO is the closed-form $R$ + offline special case. **Note that RLHF is not the $\beta \to 0$ special case**: the general KL-constrained RL objective at the common small positive $\beta$ values used in practice (e.g. 0.01-0.1) already IS the standard RLHF/PPO form, not some limiting special case; conversely, $\beta \to 0$ corresponds to dropping the KL regularizer entirely and doing unconstrained pure reward maximization (prone to reward hacking), which contradicts RLHF's core feature of keeping a finite, nonzero KL penalty to stay close to the reference policy.
</details>

<details>
<summary><strong>L3-2: What are the known theoretical convergence results for OPD?</strong></summary>

**Answer**: core known results (as of 2026-05): (1) **Fixed point**: $L_{\text{OPD}} = 0$ iff $\pi_\theta = \pi_T$ on the support of $\pi_\theta$ (reverse KL property, and alignment is only on the support the student visits). So OPD cannot make the student "surpass" the teacher — but it can let the student, within its own capacity, maximally imitate one mode of the teacher; (2) **Convergence**: under convex policy parametrization assumptions, policy gradient converges to a local minimum of reverse KL (Geist & Pietquin 2014 style), but LLM's non-convex parametrization has no global guarantee; (3) **Rethinking OPD** (arXiv 2604.13016) points out that OPD can "surpass the teacher" on reasoning tasks — this seems contradictory, but the explanation is that the teacher logits contain dark knowledge (e.g., self-correction signal), and the student via on-policy training activates capabilities that even the teacher cannot reliably demonstrate. This is the key distinction between OPD and traditional imitation learning.

**[needs-verify]** The "surpasses teacher" phenomenon is reported inconsistently across different papers; the details in the Rethinking OPD original need checking.
</details>

<details>
<summary><strong>L3-3: How is multi-teacher OPD done? What does the DeepSeek-V4 report mean by "OPD replaces mixed RL"?</strong></summary>

**Answer**: multi-teacher OPD weights-ensembles the logits of multiple specialist teachers at each token:
$$\pi_T(v|s_t) = \sum_k w_k(s_t) \cdot \pi_{T_k}(v|s_t)$$
The weights $w_k$ can be fixed (e.g., math teacher weighted high on math tasks), context-dependent (routing-style), or learned. In DeepSeek-V4's model consolidation stage, multi-teacher OPD with math / code / chat / reasoning specialist teachers completely replaces the previous mixed RL (multi-RM weighting). The advantage is **dense supervision at each token**, significantly more sample efficient than the multi-RM mixed RL.

**[needs-verify]** Specific multi-teacher implementation details of DeepSeek-V4 (weight choice, whether token-level routing) need checking in the original tech report; currently public materials are mainly secondary source citations.
</details>

<details>
<summary><strong>L3-4: OPD's "process reward" perspective and its relation to PRM? Can the two be fused?</strong></summary>

**Answer**: viewing the per-token teacher KL of OPD as a "process reward": every token (on sampled-token / behavior-policy rollouts) has a dense reward $r_t = \log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t) = -\log\!\frac{\pi_\theta^{\text{old}}(y_t|s_t)}{\pi_T(y_t|s_t)}$ (consistent with §3 / §4.5 OPD-GRPO; the "old" superscript denotes the behavior-policy log-prob recorded during rollout, avoiding the misreading $-\log\pi_\theta/\pi_T$). This is conceptually consistent with PRM (process reward model, Lightman 2023 "Let's Verify Step by Step") — both are dense rather than sparse supervision. The difference: PRM is step-level (one 0/1 per reasoning step), OPD is token-level (one KL value per token). **Fusion approaches**: (1) step-level reward = $\sum_{t \in \text{step}_k} r_t^{\text{OPD}} + \lambda \cdot r_k^{\text{PRM}}$; (2) use PRM as a trajectory filter (only PRM-high trajectories enter OPD training); (3) use OPD to distill a PRM (train process-level verifier with dense teacher signal). Academically this is active research in 2026.

**[needs-verify]** "OPD + PRM fusion" specific papers and experimental results are not yet complete; the schemes above are reasonable extrapolations from combining multi-source materials.
</details>

<details>
<summary><strong>L3-5: From an information-theoretic perspective, analyze why OPD can be 9-30× more sample efficient than RL.</strong></summary>

**Answer**: consider trajectory $y$ of length $N$, vocab $V$. Per trajectory:

- **RL outcome reward**: 1 scalar, at most $\log_2 V_R$ bits ($V_R$ = reward discretization; binary reward $\log_2 2 = 1$ bit; real-valued about $\log_2 1000 \approx 10$ bits)
- **OPD per-token teacher KL (sampled)**: one $\log \pi_T(y_t)$ value per token, about $\log_2 V \approx 17$ bits (typical vocab 100K-128K); trajectory totals $N \cdot 17$ bits
- **OPD per-token full KL**: full vocab distribution per token; theoretical upper bound $\log_2 V$ bits per token (but actual information depends on teacher distribution entropy)

bit-rate ratio: OPD / RL ≈ $N \cdot 17 / 10 = 1.7N$. On long-CoT tasks ($N = 2K$-$8K$), this ratio is on the order of **3400-13600×** — this provides an information-theoretic explanation for why the 9-30× compute efficiency reported by Thinking Machines has room to exist at all, though the actual efficiency falls far short of this information-theoretic upper bound.

**caveat**: this is an upper-bound argument with two layers of imprecision worth flagging. (1) Equating the continuous scalar $\log \pi_T(y_t)$ directly with carrying $\log_2 V \approx 17$ bits of information is a heuristic analogy — $\log_2 V$ is only an entropy upper bound on the classification outcome of "picking 1 of $V$ discrete candidates", not strictly equal to the information carried by this continuous log-probability value itself; this bit-rate argument should be treated as a heuristic upper bound, not a rigorous information-theoretic derivation. (2) Actual compute efficiency is also affected by gradient noise, teacher-student gap, optimizer, etc., and falls far short of the 3400-13600× theoretical ceiling. OPD's advantage on simple tasks is usually under 10×; only on long-horizon complex tasks does it approach the reported 9-30×.
</details>

## §A Appendix

### A.1　Sanity-check: verifying your OPD implementation via convergence

After implementing OPD, do these three micro-tests to confirm the loss is correct:

```python
# Test 1: student == teacher → loss should be ~0
student.load_state_dict(teacher.state_dict())
loss, _ = per_token_reverse_kl_loss(student, teacher, ids, mask)
assert abs(loss.item()) < 1e-4, f"identical model should give zero KL, got {loss}"

# Test 2: student random init → loss > 0
student = init_random_model(...)
loss, _ = per_token_reverse_kl_loss(student, teacher, ids, mask)
assert loss.item() > 0.5, f"random student should give positive KL, got {loss}"

# Test 3: loss should decrease over training
losses = []
for step in range(100):
    loss = opd_train_step(student, teacher, ...)
    losses.append(loss)
assert losses[-1] < losses[0], "OPD should reduce KL over training"
```

### A.2　Common mistakes and correct practices

| Wrong practice | Symptom | Correct practice |
|---|---|---|
| Compute OPD loss on teacher trajectories | Degenerates to off-policy KD, losing on-policy value | Student must rollout on its own |
| KL direction inverted (forward KL written as reverse KL) | Mode-covering, mediocre outputs | Reverse KL is $\pi_\theta$ first |
| Mask includes prompt tokens in loss | KL signal on prompt tokens is meaningless | `action_mask` only flags student-generated tokens |
| Use sum reduction without length normalization | Length inflation training collapses | per-token average (divide by mask.sum()) |
| Teacher in train mode (dropout on) | Logits unstable, loss noisy | Teacher must be in `.eval()` + `torch.no_grad()` |
| Student rollout uses greedy decode | Trajectory diversity too low; OPD cannot learn robustness | Sampling with temperature ≥ 1.0 |
| Not monitoring truncation rate | Silent failure after hitting max_length | Monitor; raise max_new_tokens if > 5% |

### A.3　Core paper and resource list

**Core OPD papers**:
- MiniLLM (Gu et al. 2023, ICLR 2024) — arXiv 2306.08543
- GKD: On-Policy Distillation of Language Models (Agarwal et al. 2023, ICLR 2024) — arXiv 2306.13649
- A Survey of On-Policy Distillation for LLMs (Song & Zheng 2026) — arXiv 2604.00626
- Rethinking On-Policy Distillation (2026) — arXiv 2604.13016
- KL for a KL (vOPD, 2026) — arXiv 2605.07865
- Black-Box On-Policy Distillation (2026) — arXiv 2511.10643
- Decoupling KL and Trajectories (2026) — arXiv 2605.16826

**Industry tech reports**:
- Qwen3 Technical Report (Qwen Team 2025-05) — arXiv 2505.09388 (§3.2 OPD recipe)
- DeepSeek-R1 (DeepSeek 2025-01) — arXiv 2501.12948 (off-policy distillation series)

**Blogs / code**:
- Thinking Machines Lab — "On-Policy Distillation" blog (Lu et al. 2025-10-27) — `thinkingmachines.ai/blog/on-policy-distillation/`
- Tinker Cookbook — `github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation`
- TRL GKD Trainer — `huggingface.co/docs/trl/gkd_trainer`
- Awesome OPD list — `github.com/thinkwee/AwesomeOPD`, `github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation`

**Related foundations**:
- Hinton, Vinyals, Dean — "Distilling the Knowledge in a Neural Network" (2015) — arXiv 1503.02531
- Sanh et al. — "DistilBERT" (2019) — arXiv 1910.01108
- Kim & Rush — "Sequence-Level Knowledge Distillation" (EMNLP 2016) — arXiv 1606.07947
- Bagnell — "Reinforcement Learning and Imitation Learning" (theoretical foundation for exposure bias)
- DeepSeekMath GRPO (2024) — arXiv 2402.03300
- DPO (Rafailov et al. NeurIPS 2023) — arXiv 2305.18290

### A.4　[needs-verify] index

The following items in this cheat sheet are marked **[needs-verify]**; we recommend cross-checking against the original papers / tech reports before and after interviews:

1. **L3-2 "OPD surpasses teacher"**: the specific setting and magnitude reported by Rethinking OPD (arXiv 2604.13016)
2. **L3-3 DeepSeek-V4 multi-teacher OPD implementation details**: weight choice strategy, whether token-level routing
3. **L3-4 "OPD + PRM fusion"**: as of 2026-05 active research; no single authoritative paper that integrates the two yet
4. **§6.2 Thinking Machines numbers**: "9-30× FLOPs savings" is from blog secondary sources; original blog numbers and specific settings should be checked
5. **§5.2 timeline**: 2025-2026 multiple OPD-related arXiv paper IDs (such as the 2604.* series) are from 2026 Q1-Q2 submissions/preprints; some IDs may be updated to new versions or renumbered after submission
6. **OPD specific adoption details on Qwen3 / MiMo**: most information comes from the Thinking Machines blog and the Qwen3 paper §3.2, but MiMo's distillation specifics need cross-checking in its tech report; **Gemma 2/3 has been reclassified as off-policy soft-target KD (not OPD, see §5.3 / §0 TL;DR item 7)** — this reclassification is based on recollection of how the Gemma 2 technical report describes its method, not a verbatim check, so it is worth re-confirming against the original report
7. **§6.1 Qwen3 specific numbers**: the exact pass@64 (vs pass@1) figures and the precise "+3-5pp" measurement protocol should be checked against the original Qwen3 Technical Report §3.2 (arXiv 2505.09388)

### A.5　Terminology quick reference

| Chinese | English | Meaning |
|---|---|---|
| 在线策略蒸馏 | On-Policy Distillation (OPD) | student does KL distillation on its own rollouts |
| 离线蒸馏 | Off-Policy Distillation | student distills on teacher / dataset data |
| 暴露偏置 | Exposure Bias | autoregressive train/test distribution mismatch |
| 反向 KL | Reverse KL | $D(\pi_\theta \,\mid \, \pi_T)$, mode-seeking |
| 正向 KL | Forward KL | $D(\pi_T \,\mid \, \pi_\theta)$, mode-covering |
| 模式寻找 | Mode-Seeking | lock onto one mode, sharp output |
| 模式覆盖 | Mode-Covering | cover all modes, averaged output |
| 教师强制 | Teacher Forcing | use ground-truth prefix during training |
| 控制变量 | Control Variate | baseline term used for variance reduction |
| 截断坍塌 | Truncation Collapse | rollouts grow longer over training, hitting max_length leads to collapse |
| 局部可教性塌缩 | Local Teachability Collapse | nothing left for teacher to teach on the trajectory's second half |

> ⚠️ **caveat** — OPD as an independent technical term for LLM post-training only became widespread in the **second half of 2025** (Qwen3 + Thinking Machines blog). Before that, the same method had been proposed in MiniLLM (2023) and GKD (2023). So "OPD is a new method" is strictly inaccurate — it is an old method that was renamed and industrialized at scale, benefiting from the reasoning model era's hunger for dense supervision. This historical context is often asked in L3 interviews; please distinguish between "year of method first proposed" and "year of term popularization".
