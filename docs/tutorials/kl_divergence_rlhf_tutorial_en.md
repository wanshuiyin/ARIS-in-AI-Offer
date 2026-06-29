## §0 TL;DR Cheat Sheet

> 💡 **8 sentences to nail KL in RLHF** — one page covering interview essentials (see §1–§8 for derivations).

1. **Definition**: $\text{KL}(p \| q) = \mathbb{E}_{x \sim p}[\log(p(x)/q(x))]$ — non-negative, asymmetric, non-metric. In RLHF, $p = \pi_\theta$, $q = \pi_\text{ref}$, and the role is to anchor the post-RL policy near SFT to prevent reward hacking.

2. **Forward vs Reverse**: convention — $p$ is the data/target, $q_\theta$ is the variational/optimization distribution. **Forward KL** = $\text{KL}(p\|q_\theta)$ (mass-covering); **Reverse KL** = $\text{KL}(q_\theta\|p)$ (mode-seeking). RLHF uses $\text{KL}(\pi_\theta \| \pi_\text{ref})$, which under this convention is **reverse KL** ($\pi_\theta$ is the variational side). Its mode-seeking behavior fits the RL goal — make $\pi_\theta$ pick reward-high modes inside high-density regions of $\pi_\text{ref}$ — and is also engineering-feasible (just estimate from rollout samples). Note that naming varies slightly across communities, but DPO / RLOO / "Rethinking KL" / "Comedy of Estimators" and other 2024-2026 RLHF papers all call $\text{KL}(\pi_\theta \| \pi_\text{ref})$ reverse KL.

3. **Three KL estimators (Schulman 2020 blog `joschu.net/blog/kl-approx.html`)**:
   - **k1** = $\log(\pi_\theta/\pi_\text{ref})$ — unbiased but high variance, can be negative (cannot be read as a "distance").
   - **k2** = $\tfrac{1}{2}(\log(\pi_\theta/\pi_\text{ref}))^2$ — always non-negative but **biased** (second-order Taylor approximation).
   - **k3** = $(\pi_\text{ref}/\pi_\theta) - \log(\pi_\text{ref}/\pi_\theta) - 1$ — **unbiased + non-negative + low variance**, derived from the identity $\mathbb{E}_q[f(\log p/q)]$ with $f(x)=e^x-x-1$.

4. **Two placements**: (a) **In-reward shaping**: $\tilde{r}_t = r_t - \beta \cdot \text{KL}_t$, computed along with advantage / GAE; (b) **In-loss regularization**: $\mathcal{L} = \mathcal{L}_\text{PG} + \beta \cdot \mathbb{E}[\text{KL}]$. **InstructGPT/Anthropic PPO use (a); GRPO uses (b) + k3 estimator**. **The objective is the same, but the gradient path differs** — under a principled estimator (e.g. (a) k1-in-reward or (b) k2-as-loss) the two are gradient-equivalent on-policy; GRPO's (b) k3-as-loss actually carries an $O(\Delta^2)$ first-order Taylor bias (see §3.6 + Rethinking KL 2510.01555). The two placements also respond differently to PPO clip / importance-ratio truncation.

5. **β schedule**: fixed β (simplest), **Adaptive β** (PPO-Penalty original — pulling β based on measured-KL-vs-target), Annealing β (tight early, loose later, like an SFT-to-RL transition). **Too-large β fails to learn** (policy stuck near ref); **too-small β leads to reward hacking** (policy drifts away, with long / sycophantic answers).

6. **Closed-form optimal policy of KL-regularized RL**: solving the BT-style reward objective $\max_\pi \mathbb{E}[r] - \beta \cdot \text{KL}(\pi\|\pi_\text{ref})$ has the unique closed-form solution $\pi^*(y|x) \propto \pi_\text{ref}(y|x) \exp(r(x,y)/\beta)$. **DPO is exactly this inverse plugged into Bradley-Terry**, yielding $r = \beta \log(\pi^*/\pi_\text{ref}) + \beta\log Z$, where the partition function $\log Z$ **cancels** in the pairwise difference.

7. **KL's place in DPO/GRPO/SimPO**: DPO's implicit reward $\hat r_\theta = \beta\log(\pi_\theta/\pi_\text{ref})$ is a sequence-level log-ratio (taking expectation under $y\sim\pi_\theta$ would equal $\beta\cdot$KL, but in DPO training $y$ comes from preference data so this is not KL itself — yet the reference appearing in the denominator still provides implicit anchoring); GRPO puts k3 KL in the loss (note that k3-as-loss yields a first-order gradient approximation, see §3.6); SimPO outright **drops the reference**, so it has **no KL constraint** — making it more sensitive to β/γ/length-norm.

8. **Failure modes**: KL explosion (β too small, importance ratio too large), KL collapse (β too large or entropy too low, policy stuck), **Reward Overoptimization** (Gao 2023 ICML): proxy reward rises monotonically with KL, but gold reward rises then falls (inverted-U) — KL distance is the natural axis for overoptimization.

## §1 KL Fundamentals

### 1.1　Definition, properties, notation

**Definition** (discrete):

$$\boxed{\;\text{KL}(p \| q) \triangleq \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]\;}$$

Conventions: $0 \log 0 = 0$, $0 \log(0/0) = 0$, $p \log(p/0) = +\infty$. Continuous distributions replace $\sum$ with $\int$.

**Core properties** (frequently asked together in interviews):

| Property | One-line statement | Brief proof |
| --- | --- | --- |
| **Non-negative** | $\text{KL} \ge 0$ | Jensen + $-\log$ convex: $\text{KL} = -\mathbb{E}_p \log(q/p) \ge -\log \mathbb{E}_p(q/p) = 0$ |
| **Equality** | $\text{KL} = 0$ iff $p = q$ a.e. | Jensen equality condition |
| **Asymmetric** | $\text{KL}(p,q) \ne \text{KL}(q,p)$ (in KL's asymmetric arguments) | Direct example construction |
| **No triangle inequality** | Not a metric | So "KL distance" is informal/imprecise |
| **Convexity** | $\text{KL}(\cdot,\cdot)$ jointly convex in $(p,q)$ | Log-sum inequality |
| **Chain rule** | Joint KL = marginal KL + expected conditional KL (see display below) | Direct log decomposition |
| **Reparameterization invariance** | Invariant under same transformation $x \mapsto T(x)$ (equality only when $T$ invertible) | Measure change + Jacobian cancellation |

Full chain-rule form:

$$\text{KL}(p(x,y) \,\|\, q(x,y)) = \text{KL}(p(x) \,\|\, q(x)) + \mathbb{E}_{p(x)}\!\big[\text{KL}(p(y|x) \,\|\, q(y|x))\big]$$

> ✅ **Chain rule in RLHF = per-token KL summed** — sequence-level KL is the **expectation** of the sum of trajectory log-ratios: $\text{KL}_\text{seq} = \mathbb{E}_{y\sim\pi_\theta}[\sum_t \log\pi_\theta(y_t|s_t)/\pi_\text{ref}(y_t|s_t)]$. In implementation, computing $\sum_t \log r_t$ on a single rollout is the **single-MC estimator** (the sequence form of k1); to get true expected KL one averages over a batch of rollouts. The **true token-level KL** $D_\text{KL}(\pi_\theta(\cdot|s_t)\|\pi_\text{ref}(\cdot|s_t))$ further requires a full-vocab sum (full-vocab KL) at that prefix rather than a log-ratio only at the sampled token. The two **have the same expectation** but use different estimator forms: sum-over-rollout of sampled log-ratios is the cheap-but-noisy estimator.

### 1.2　Forward KL vs Reverse KL: mass-covering or mode-seeking?

Let $p$ be the data/true distribution and $q_\theta$ the parametric model. The two directions behave completely differently in fitting policies:

| Direction | Form (display below) | Expectation under | Behavior | Classic uses |
| --- | --- | --- | --- | --- |
| **Forward KL** | $\text{KL}(p\,\|\,q_\theta) = \mathbb{E}_{p}[\log(p/q_\theta)]$ | $p$ (data/target) | **mass-covering / mean-seeking**: wherever $p > 0$, $q_\theta$ must be $> 0$, otherwise $\log(p/q) \to \infty$ | MLE / distillation (student covers teacher) |
| **Reverse KL** | $\text{KL}(q_\theta\,\|\,p) = \mathbb{E}_{q_\theta}[\log(q_\theta/p)]$ | $q_\theta$ (variational/optimization) | **mode-seeking / minorization**: wherever $q_\theta > 0$, $p$ must be $> 0$; $q_\theta$ tends to collapse into a single mode | VI, RLHF, GAN-like training |

Notation conventions:

$$\text{Forward KL}: \text{KL}(p \,\|\, q_\theta) = \mathbb{E}_{p}\!\left[\log\frac{p}{q_\theta}\right],\qquad \text{Reverse KL}: \text{KL}(q_\theta \,\|\, p) = \mathbb{E}_{q_\theta}\!\left[\log\frac{q_\theta}{p}\right]$$

Classic picture (bimodal $p$, unimodal $q_\theta$):

- Forward KL fit → $q_\theta$ widens and bridges both modes (**mass-covering**).
- Reverse KL fit → $q_\theta$ picks one mode and shrinks into it (**mode-seeking**).

> ⚠️ **Modern consensus on the naming convention** — Variational inference, DPO, RLOO, "Rethinking KL Regularization in RLHF" (arXiv 2510.01555), "A Comedy of Estimators" (arXiv 2512.21852), and other 2024-2026 RLHF papers all use the unified convention: $\text{KL}(q_\theta\|p)$ is called reverse KL ($q_\theta$ is the variational/optimization side), and $\text{KL}(p\|q_\theta)$ is called forward KL. This tutorial follows that standard convention. A few earlier RL textbooks name it by the sampling distribution — in interviews, simply state the formula to avoid label ambiguity.

**Which one does RLHF use?** Almost all mainstream implementations (InstructGPT / Anthropic PPO / DeepSeekMath GRPO) use $\text{KL}(\pi_\theta \| \pi_\text{ref})$ (**reverse-KL form**: $\pi_\theta$ is the variational side, $\pi_\text{ref}$ is the target). **Fundamental reason**: training already samples from $\pi_\theta$ (rollout), so $\mathbb{E}_{\pi_\theta}[\log \pi_\theta/\pi_\text{ref}]$ is directly estimable from samples; meanwhile reverse KL's mode-seeking behavior matches the RL goal exactly — find the reward-high mode in the high-density region of $\pi_\text{ref}$, rather than "cover all of $\pi_\text{ref}$".

> 💡 Note: if we swapped roles of $\pi_\theta$ and $\pi_\text{ref}$ (forward KL = $\text{KL}(\pi_\text{ref}\|\pi_\theta)$), we would need to sample from $\pi_\text{ref}$, which engineering-wise requires IS, is expensive, and is semantically off-target (we are training $\pi_\theta$, not $\pi_\text{ref}$). So RLHF rarely uses the forward direction.

### 1.3　Relation to other divergences

| Divergence | Definition (display below) | Properties | Notes |
| --- | --- | --- | --- |
| **JS** | Symmetric average of KL to mixture $m = (p+q)/2$ | Symmetric, bounded ($\le \log 2$), square root is a metric | $\sqrt{\text{JS}}$ is a metric (Endres-Schindelin 2003 IEEE TIT 49(7)); the original GAN discriminator loss is equivalent to $2\cdot\text{JSD} - \log 4$; rarely used in RLHF |
| **α-divergence** | $\frac{1}{\alpha(1-\alpha)}(1 - \int p^\alpha q^{1-\alpha})$ | Contains KL as a limit | Unified framework; $\alpha \to 1$ gives forward KL, $\alpha \to 0$ gives reverse KL |
| **Hellinger** | $H^2(p,q) = \tfrac{1}{2}\int(\sqrt{p}-\sqrt{q})^2$ | Symmetric, bounded, $0 \le H^2 \le 1$ | Relation to KL: $H^2 \le \tfrac{1}{2}\text{KL}$ |
| **$\chi^2$** | $\chi^2(p,q) = \int \frac{(p-q)^2}{q}$ | $f$-divergence at $f(t) = (t-1)^2$ | Relation to KL: $\text{KL} \le \log(1 + \chi^2)$ |
| **TV** | $\tfrac{1}{2}\int \lvert p-q\rvert$ | A metric, $\in [0,1]$ | **Pinsker**: $\text{TV} \le \sqrt{\text{KL}/2}$ |

Full forms of four common $f$-divergences:

$$\text{JS}(p, q) = \tfrac{1}{2}\text{KL}(p \,\|\, m) + \tfrac{1}{2}\text{KL}(q \,\|\, m),\quad m = (p+q)/2$$

$$\chi^2(p, q) = \int \frac{(p(x) - q(x))^2}{q(x)}\,dx,\qquad \text{Pinsker: } \mathrm{TV}(p,q) \le \sqrt{\text{KL}(p \,\|\, q) / 2}$$

> 💡 **Why doesn't RLHF use JS / α-divergence?** — Mostly engineering inertia + KL's closed-form advantages: in the PPO/DPO framework, KL admits clean per-token decomposition (chain rule), has a closed-form optimum (softmax-style), and likelihood ratios directly give per-token KL increments — so no extra density-ratio estimator is needed. JS / α-divergence either lack such clean math or require an extra density-ratio estimator.

### 1.4　Why does RLHF add KL?

Put "without KL" and "with KL" objectives side by side:

$$\text{Without KL}:\quad \max_\pi \mathbb{E}_{x,\,y \sim \pi(\cdot|x)}[r(x,y)]$$

$$\text{With KL}:\quad \max_\pi \mathbb{E}_{x,\,y \sim \pi(\cdot|x)}[r(x,y)] - \beta\,\mathbb{E}_x\,\text{KL}\!\big(\pi(\cdot|x)\,\big\|\,\pi_\text{ref}(\cdot|x)\big)$$

**What happens without KL?**

1. **Reward hacking**: the policy finds RM blind spots (longer answers / "As an AI..." boilerplate / sycophancy / format hacks) to gain high RM scores while degrading human-perceived quality.
2. **Linguistic fluency collapses**: the policy outputs token sequences the RM "likes" but humans find unintelligible (in extreme cases, degenerating to repeating a single token / completely ungrammatical).
3. **Distribution shift**: the policy strays far from $\pi_\text{ref}$, to the point that even the RM cannot reliably score it (off-distribution — the more uncertain the RM, the more random the reward).

After adding KL:

- **β provides implicit regularization against RM error**: when the RM is unreliable, KL pulls the policy back to the SFT's known-good distribution.
- **A closed-form optimal policy exists** (derived in §6.1), giving the whole RLHF stack a mathematical foundation.
- **DPO / KTO / GRPO all depend on this KL anchor**: SimPO, which drops the reference, has empirically been observed to be more hyperparameter-sensitive.

## §2 KL Estimators (k1 / k2 / k3): Interview Core

### 2.1　Problem setup

In practice, on every mini-batch we must estimate $\text{KL}(\pi_\theta \| \pi_\text{ref})$. But a full sum $\sum_y \pi_\theta(y) \log \pi_\theta(y)/\pi_\text{ref}(y)$ is infeasible ($y$ is a full response — combinatorial blowup). **Only Monte Carlo is viable**: estimate KL using samples drawn from $\pi_\theta$.

Let $\log r = \log(\pi_\theta(y)/\pi_\text{ref}(y))$ (not the importance ratio, but the policy log-ratio). With samples $y \sim \pi_\theta$, we want to estimate $\mathbb{E}_{\pi_\theta}[\log r] = \text{KL}(\pi_\theta \| \pi_\text{ref})$.

> ⚠️ **Notation unification** — In this section $\log r$ always denotes the **policy log-ratio** $\log \pi_\theta - \log \pi_\text{ref}$ (**not** the PPO importance ratio $\pi_\theta / \pi_\text{old}$). The two have similar form but different meaning: the former measures "how far from reference", the latter measures "how far from the sampling policy".

### 2.2　k1 estimator — "plug definition in directly"

$$\boxed{\;\widehat{\text{KL}}_1 = \log\frac{\pi_\theta(y)}{\pi_\text{ref}(y)} = \log r\;}$$

- **Unbiased**: $\mathbb{E}_{y\sim\pi_\theta}[\log r] = \text{KL}(\pi_\theta\|\pi_\text{ref})$ by definition.
- **Can be negative**: on a single sample, $\log r$ can be positive or negative (the log-ratio has no "non-negative" constraint).
- **High variance**: in the tails, $\log r$ can be very large or very small, especially in regions where $\pi_\theta$ and $\pi_\text{ref}$ have little overlap.

**Problem**: using k1 as a "KL monitoring metric" yields **negative values**; engineering loggers may display a negative KL, confusing newcomers (isn't KL supposed to be non-negative?). This is because **non-negativity of the expectation does not imply non-negativity of every sample**. But as per-token KL in reward shaping it is acceptable (what matters is unbiased mean).

### 2.3　k2 estimator — "$L^2$ form"

$$\boxed{\;\widehat{\text{KL}}_2 = \tfrac{1}{2}\!\left(\log\frac{\pi_\theta(y)}{\pi_\text{ref}(y)}\right)^2 = \tfrac{1}{2}(\log r)^2\;}$$

- **Always non-negative** ✓
- **Biased**: $\mathbb{E}[\tfrac{1}{2}(\log r)^2] \ne \text{KL}$.
- **Small-KL limit**: Taylor expansion $\log r = (r - 1) - \tfrac{1}{2}(r-1)^2 + O((r-1)^3)$; when $\pi_\theta \approx \pi_\text{ref}$, $\log r$ is small, and $\tfrac{1}{2}(\log r)^2$ equals KL to second order (near $p = q$, KL = Fisher-information quadratic form).
- **Variance smaller than k1**: because squaring eliminates sign cancellation, but it is still not optimal.

In practice k2 is mostly used for **monitoring** (providing a "non-negative but biased" visualization metric); it is rarely used for reward shaping.

### 2.4　k3 estimator (Schulman 2020 blog) — **non-negative unbiased value estimator** (but as a loss term, the gradient is biased — see §3.6)

#### 2.4.1　Construction

Consider $f(x) = e^x - x - 1$. From $e^x \ge 1 + x$ (for real $x$), we get $f(x) \ge 0$, with $f(0) = 0$.

Substituting $x = \log(\pi_\text{ref}(y)/\pi_\theta(y)) = -\log r$:

$$f(-\log r) = e^{-\log r} - (-\log r) - 1 = \frac{1}{r} + \log r - 1 = \frac{\pi_\text{ref}(y)}{\pi_\theta(y)} + \log\frac{\pi_\theta(y)}{\pi_\text{ref}(y)} - 1$$

Equivalently (writing it with $\Delta = -\log r = \log(\pi_\text{ref}/\pi_\theta)$): $\frac{\pi_\text{ref}}{\pi_\theta} - \log\frac{\pi_\text{ref}}{\pi_\theta} - 1$ (note $-\log(\pi_\text{ref}/\pi_\theta) = \log(\pi_\theta/\pi_\text{ref})$, the two forms are equivalent).

Equivalently (this is the standard form in Schulman's blog):

$$\boxed{\;\widehat{\text{KL}}_3 = \frac{\pi_\text{ref}(y)}{\pi_\theta(y)} - \log\frac{\pi_\text{ref}(y)}{\pi_\theta(y)} - 1 = e^{\Delta} - \Delta - 1,\quad \Delta = \log\frac{\pi_\text{ref}(y)}{\pi_\theta(y)} = -\log r\;}$$

#### 2.4.2　Three core properties (must-know)

**Property 1 (non-negative)**:

Since $e^\Delta - \Delta - 1 \ge 0$ for all $\Delta \in \mathbb{R}$ (the convex $e^\Delta$ lies above its tangent line $1 + \Delta$ at $\Delta = 0$), $\widehat{\text{KL}}_3 \ge 0$ always holds.

**Property 2 (unbiased)**:

We show $\mathbb{E}_{y \sim \pi_\theta}[\widehat{\text{KL}}_3] = \text{KL}(\pi_\theta\|\pi_\text{ref})$.

$$\mathbb{E}_{\pi_\theta}\!\left[\frac{\pi_\text{ref}}{\pi_\theta}\right] = \sum_y \pi_\theta(y) \cdot \frac{\pi_\text{ref}(y)}{\pi_\theta(y)} = \sum_y \pi_\text{ref}(y) = 1$$

$$\mathbb{E}_{\pi_\theta}\!\left[-\log\frac{\pi_\text{ref}}{\pi_\theta}\right] = \mathbb{E}_{\pi_\theta}\!\left[\log\frac{\pi_\theta}{\pi_\text{ref}}\right] = \text{KL}(\pi_\theta\|\pi_\text{ref})$$

Therefore:

$$\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = 1 + \text{KL}(\pi_\theta\|\pi_\text{ref}) - 1 = \text{KL}(\pi_\theta\|\pi_\text{ref}) \quad \checkmark$$

**Property 3 (variance typically lower than k1)**:

Intuition: $\widehat{\text{KL}}_3$ is $\log r$'s linear term (the $-\log r/\pi_\theta$ part, i.e. the equivalent form of k1, $\mathbb{E}_{\pi_\theta}[-\log(\pi_\text{ref}/\pi_\theta)]$) plus a **mean-1 control variate** $\pi_\text{ref}/\pi_\theta - 1$. The control variate reduces variance: when $r$ is large, $\log r$ is large but $1/r$ is small, and vice versa, so **the two are negatively correlated** and summing them gives lower variance than k1 alone.

Formally: $\widehat{\text{KL}}_3 = \widehat{\text{KL}}_1 + (\frac{\pi_\text{ref}}{\pi_\theta} - 1)$; the added term $(\frac{\pi_\text{ref}}{\pi_\theta} - 1)$ has expectation 0 but is strongly negatively correlated with $\log r$ → variance reduction.

> 💡 **The elegance of k3** — The derivation in Schulman's 2020 blog is equivalent to the above: he **first looks for $f(x) \ge 0$ such that $\mathbb{E}_q f(\log p/q) = \text{KL}$**, then picks $f(x) = e^x - x - 1$ (the simplest non-trivial non-negative differentiable choice). So k3 is not "inspired" — it is the most natural construction under the "non-negative + unbiased" constraint.

#### 2.4.3　Variants and simplifications

Real-world code often writes (this also applies to the PPO ratio $r = \pi_\theta / \pi_\text{old}$, but with different semantics — that is an importance-ratio approximation, not a policy log-ratio):

```python
approx_kl_to_old = ((ratio - 1) - torch.log(ratio.clamp_min(1e-8))).mean()
```

I.e. $\widehat{\text{KL}}_3 \approx (r - 1) - \log r$, which is the equivalent dual of $e^\Delta - \Delta - 1$ above (treat $r = e^{\log r}$ as $e^\Delta$ with sign convention swapped). **TRL / OpenRLHF / verl all default `approx_kl` to this form**.

### 2.5　Comparison table for the three

| Estimator | Form | Unbiased? | Non-negative? | Variance | Typical RLHF use |
| --- | --- | --- | --- | --- | --- |
| **k1** | $\log r$ | ✅ | ❌ | High | InstructGPT PPO reward shaping (per-token KL) |
| **k2** | $\tfrac{1}{2}(\log r)^2$ | ❌ value-estimator biased (second-order approximation of KL) | ✅ | Mid | Often used for monitoring from the value perspective; **from the loss-gradient perspective, k2-as-loss is principled** — on-policy it is gradient-equivalent to k1-in-reward, see §3.6 |
| **k3** | $e^{-\log r} + \log r - 1$ | ✅ (as a **value estimator**) | ✅ | Low | Historical usage in DeepSeekMath GRPO / DAPO (when used **as loss**, its gradient is a biased first-order approximation, see §3.6 + the Rethinking KL paper) |

### 2.6　Code: three estimators + variance simulation

```python
import torch
import torch.nn.functional as F

def k1_estimator(logp_theta, logp_ref):
    """k1: log(π_θ / π_ref)  — unbiased, but can be negative, high variance."""
    return logp_theta - logp_ref           # [B, T]

def k2_estimator(logp_theta, logp_ref):
    """k2: 0.5 (log(π_θ / π_ref))^2  — biased, non-negative, mid variance."""
    return 0.5 * (logp_theta - logp_ref) ** 2

def k3_estimator(logp_theta, logp_ref):
    """k3 (Schulman 2020): exp(log π_ref - log π_θ) + log(π_θ/π_ref) - 1
       = (π_ref/π_θ) - log(π_ref/π_θ) - 1     [letting Δ = log π_ref - log π_θ]

    Value-estimator properties: unbiased, non-negative, low variance.
    ⚠️ Recommended for KL value MONITORING, NOT as a loss term — k3-as-loss
       gives gradient (1 - e^{-Δ})∇logπ_θ = (Δ - ½Δ² + O(Δ³))∇logπ_θ, a
       first-order Taylor approximation of reverse-KL gradient with O(Δ²) bias.
       For principled reverse-KL gradient, use k2-as-loss (gradient-equivalent
       to k1-in-reward on-policy) or k1-in-reward via score function. See §3.6.
    """
    log_r = logp_theta - logp_ref          # log(π_θ/π_ref)
    log_ratio_rev = -log_r                 # Δ = log(π_ref/π_θ)
    return torch.exp(log_ratio_rev) - log_ratio_rev - 1

# Variance comparison: 1D synthetic
def compare_kl_estimators(n_samples=10_000, seed=0):
    """
    Synthetic: π_θ = N(0, 1), π_ref = N(μ, 1). True KL = μ^2 / 2.
    Sample y ~ π_θ; evaluate the three estimators.
    """
    torch.manual_seed(seed)
    mu = 0.5
    true_kl = mu ** 2 / 2.0                        # closed-form for Gaussians w/ same σ

    y = torch.randn(n_samples)                     # y ~ N(0, 1) = π_θ
    # log-pdf of N(0,1) vs N(μ,1) at y (drop common -0.5 log 2π):
    logp_theta = -0.5 * y ** 2
    logp_ref   = -0.5 * (y - mu) ** 2

    k1 = k1_estimator(logp_theta, logp_ref)
    k2 = k2_estimator(logp_theta, logp_ref)
    k3 = k3_estimator(logp_theta, logp_ref)

    print(f"true KL = {true_kl:.4f}")
    for name, vals in [("k1", k1), ("k2", k2), ("k3", k3)]:
        print(f"  {name}: mean={vals.mean().item():+.4f}  var={vals.var().item():.4f}  "
              f"min={vals.min().item():+.3f}  max={vals.max().item():+.3f}")
    # Expected output: k1 mean ≈ 0.125 (unbiased), but min < 0;
    #                  k2 mean > 0.125 (biased upward);
    #                  k3 mean ≈ 0.125 (unbiased), min ≥ 0, var(k3) < var(k1).
```

> ✅ **Running the simulation you will see** —

- k1 mean ≈ 0.125, min around −1.5 (negative possible), variance large.
- k2 mean ≈ 0.18 (biased high), min ≥ 0, mid variance.
- k3 mean ≈ 0.125 (unbiased), min ≥ 0, var **distinctly smaller than** k1.

This matches the takeaway from Schulman's blog: **k3 simultaneously delivers unbiased + non-negative + lower variance**.

## §3 Two placements of KL in RLHF

### 3.1　Option A: In-reward shaping (PPO RLHF / InstructGPT standard)

Bake KL **into per-token reward**, then run normal PPO + GAE:

$$\boxed{\;\tilde{r}_t = \underbrace{\mathbb{1}[t = T] \cdot R(x, y)}_{\text{terminal RM reward}} - \beta \cdot \underbrace{\log\frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_\text{ref}(y_t \mid x, y_{<t})}}_{\text{per-token KL (k1)}}\;}$$

Details to watch:

- **Per-token**: every generated token contributes its log-prob ratio as part of that step's reward.
- **k1 estimator**: this directly uses $\log(\pi_\theta/\pi_\text{ref})$ (k1). Note **a single token's value can be negative**, but as reward shaping only the mean/sum needs to match the KL expectation.
- **Terminal RM reward**: the RM scalar for the entire answer is placed only on the last token (other tokens have RM reward = 0).

After placement, run GAE:

$$\delta_t = \tilde{r}_t + \gamma V(s_{t+1}) - V(s_t),\quad A_t^{\text{GAE}} = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}$$

The KL penalty naturally propagates into policy gradient through the advantage: **at every step the "increase in action probability" is offset by KL's "pull back to reference"**, and the final RL objective is expected $R$ − $\beta$ · KL.

### 3.2　Option B: In-loss regularization (GRPO / DAPO)

Don't put KL in the reward; **make it a separate loss term**:

$$\boxed{\;\mathcal{L}_\text{full}(\theta) = -\underbrace{\mathbb{E}\!\left[\min(\rho_t A_t, \text{clip}(\rho_t, 1\!-\!\epsilon, 1\!+\!\epsilon) A_t)\right]}_{\text{PPO surrogate / GRPO surrogate}} + \beta \cdot \underbrace{\mathbb{E}\!\left[\widehat{\text{KL}}_\text{loss}(\pi_\theta \| \pi_\text{ref})\right]}_{\text{KL loss term}}\;}$$

GRPO's **historical implementation** (DeepSeekMath / DAPO) takes $\widehat{\text{KL}}_\text{loss}$ to be **k3** (per-token):

- Uses k3 estimator (per-token).
- KL **does not enter advantage computation**; it is added to the loss directly as a regularizer.
- Advantage comes from group-internal reward normalization ($\hat{A}_i = (r_i - \bar{r})/\sigma_r$), shared across all tokens.

> ⚠️ **Important caveat**: backpropagating k3 directly as a loss term is **not** the exact reverse-KL gradient — it is a biased first-order approximation (see §3.6 + the 2025 paper "Rethinking KL Regularization in RLHF"). The two more principled on-policy choices are: (1) **k1 in reward** (KL into reward, PPO/GRPO backprops via score function — gradient strictly unbiased); (2) **k2 as loss** ($\tfrac12 (\log r)^2$, $\nabla\mathcal L = \Delta\,\nabla\log\pi_\theta$, **gradient-equivalent** to (1) on-policy — both are the strict reverse-KL gradient). Off-policy still requires IS correction. §3.6 below gives the comparison table.

### 3.3　Mathematically equivalent? Engineering-wise not equivalent

**Math**: both placements optimize the same objective:

$$J(\pi) = \mathbb{E}_{\pi}[R] - \beta \cdot \text{KL}(\pi \| \pi_\text{ref})$$

but the **gradient propagation path differs**:

- Option A: KL as part of reward → through GAE → into advantage → into PPO surrogate gradient.
- Option B: KL as independent loss → direct gradient w.r.t. $\theta$.

**Engineering differences**:

| Dimension | In-reward (Option A) | In-loss (Option B) |
| --- | --- | --- |
| KL estimator | k1 (per-token can be negative, acceptable as reward) | k3 (unbiased + non-negative, more stable) |
| Relation to PPO clip | KL is truncated by clip (importance-ratio truncation cuts KL along with it) | KL is independent, unaffected by clip |
| Advantage interpretation | Advantage contains KL → "net advantage" | Advantage is reward-only → "pure advantage" |
| Hyperparameter sensitivity | β directly affects reward scale → must coordinate with RM scale | β is independent of reward, but KL-vs-PG balance matters |
| Monitoring metrics | Watch token-level KL inside reward | Watch the KL loss curve independently |

> ⚠️ **How PPO clip masks the KL signal** — In Option A, when the importance ratio $\rho_t$ falls outside the clip range ($\rho_t > 1+\epsilon$ with $\tilde A_t > 0$ or $\rho_t < 1-\epsilon$ with $\tilde A_t < 0$), PPO truncates the surrogate into a constant (zero gradient w.r.t. θ), and the KL term in $\tilde r_t$ at that step also becomes ineffective. This means **when the policy drifts too far, PPO clip ironically disables the KL anchor** — this is part of the motivation for Option B (KL in loss is always active).

### 3.4　Practice: which goes where?

| Algorithm | KL placement | Estimator | Source |
| --- | --- | --- | --- |
| InstructGPT PPO | **In-reward** | k1 | Ouyang 2022 NeurIPS |
| Anthropic RLHF | **In-reward** | k1 + adaptive β | Bai 2022 arXiv 2204.05862 |
| GRPO / DeepSeekMath | **In-loss** | k3 | Shao 2024 arXiv 2402.03300 |
| DAPO | **In-loss** (often KL turned off or low-weighted in real configs) | k3 | Yu 2025 arXiv 2503.14476 |
| DPO | **Implicit in-loss** (via $\pi_\text{ref}$ in the log-ratio denominator) | — | Rafailov 2023 NeurIPS |
| SimPO | **No reference, no KL** | — | Meng 2024 NeurIPS |

> 💡 **DAPO field experience** — ByteDance's verl team has reported in engineering write-ups that the KL term is **often turned off or set to a very small β** ($10^{-4}$ order) for large-scale math/code RL training. Reasons: the reward is already rule-based (close to ground truth), there is no RM hacking to worry about, and the KL anchor actually slows down training. This is the extreme "Option B + β ≈ 0" version.

### 3.6　Gradient bias analysis of estimator placement (Rethinking KL Regularization in RLHF)

> 📝 **Key reference**: Kezhao Liu et al., "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization", arXiv 2510.01555 (2025-10-02). This paper systematically distinguishes **value estimation** ("is the KL value itself estimated correctly?") from **gradient optimization** ("after differentiating w.r.t. θ, is the gradient actually the reverse-KL gradient?"), and its conclusions are not fully aligned with the historical GRPO implementation.

#### Setup and three estimator placements

For brevity, let $\Delta_t = \log\!\frac{\pi_\theta(y_t|s_t)}{\pi_\text{ref}(y_t|s_t)}$ (per-token log-ratio, **with θ gradient**), $\hat\Delta_t = \text{stop\_grad}(\Delta_t)$ (detached copy recorded at rollout time). Three common placements:

| Placement | Where KL appears | Per-token form | Gradient contribution to θ |
|---|---|---|---|
| **(P1) k1 in reward** | Enter reward / advantage | $\hat r_t \leftarrow r_t - \beta\,\hat\Delta_t$, then PPO surrogate | $-\beta\cdot \mathbb{E}[\nabla_\theta\log\pi_\theta\cdot \hat\Delta_t]$, **the strict reverse-KL score-function gradient** (on-policy) |
| **(P2) k2 as loss** | Direct loss term | $\mathcal{L}_\text{KL} = \tfrac12 \Delta_t^2$ | $\nabla\mathcal{L}_\text{KL} = \Delta_t\,\nabla_\theta\log\pi_\theta$ (using $\nabla_\theta\Delta_t = \nabla_\theta\log\pi_\theta$). On-policy, $\mathbb{E}_{y\sim\pi_\theta}[\Delta\,\nabla\log\pi_\theta] = \nabla\text{KL}$ — **gradient-equivalent** to (P1), strictly principled |
| **(P3) k3 as loss** | Direct loss term | $\mathcal{L}_\text{KL} = e^{-\Delta_t} + \Delta_t - 1$ | $\nabla\mathcal{L}_\text{KL} = (1 - e^{-\Delta_t})\nabla_\theta\log\pi_\theta$; this is a **first-order Taylor approximation** of the reverse-KL gradient, since $1-e^{-\Delta} = \Delta - \tfrac12\Delta^2 + O(\Delta^3)$, with $O(\Delta^2)$ bias |

#### Why k3 as loss yields a biased gradient (key intuition)

k3 as a **value estimator** (computing the KL numeric value) is unbiased: $\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = \text{KL}(\pi_\theta\|\pi_\text{ref})$ (see §2.4.2 Property 2). But when **backpropagating w.r.t. θ**, autograd computes $\nabla_\theta\widehat{\text{KL}}_3$, **not** $\nabla_\theta\,\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3]$. The latter is given by the score-function trick:

$$\nabla_\theta\,\text{KL}(\pi_\theta\|\pi_\text{ref}) = \mathbb{E}_{y\sim\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(y) \cdot \log(\pi_\theta(y)/\pi_\text{ref}(y))\big] \;+\; \mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta]\,\quad\text{(=0)}$$

= $\mathbb{E}[\nabla\log\pi_\theta \cdot \Delta]$ (**this is the gradient given by (P1)**, equivalent to treating $\Delta$ as a detached reward via the score function).

Meanwhile (P3) `loss.backward()` gives $\nabla_\theta(e^{-\Delta} + \Delta - 1) = (1 - e^{-\Delta})\nabla_\theta\Delta = (1 - 1/r)\nabla_\theta\log\pi_\theta$. Taylor expansion: $1 - 1/r = \Delta - \tfrac12\Delta^2 + O(\Delta^3)$. So (P3) ≈ (P1) only in the $\Delta\to 0$ neighborhood; far from 0 there is $O(\Delta^2)$ bias.

#### Practical rules (combining Rethinking KL + Comedy of Estimators)

| Scenario | Recommended placement | Notes |
|---|---|---|
| **On-policy (rollout-step backprop, no PPO mini-batch multi-step updates)** | **(P1) k1 in reward** or **(P2) k2 as loss** | The two are **gradient-equivalent** on-policy, both being the strict reverse-KL gradient |
| **Off-policy (PPO mini-batch passes over the same data several times)** | (P1) k1 in reward + **IS correction**: $\rho\cdot\Delta$, $\rho = \pi_\theta^\text{new}/\pi_\theta^\text{old}$ | Otherwise the reward mismatches the current policy |
| **Historical GRPO/DeepSeekMath/DAPO implementations** | (P3) k3 as loss | Carries $O(\Delta^2)$ gradient bias; in practice $\Delta$ is small (β anchors tight) and it still trains; but **not theoretically principled** |
| **Value-side monitoring** (looking at how large the KL is) | k3 estimator remains first choice | Unbiased, non-negative, low variance — exactly the three properties from §2.4 |

> 💡 **Summary phrasing** (interview-ready): k3 is an **excellent KL value estimator** (unbiased + non-negative + low variance); but the gradient produced by backpropagating it as a loss is a first-order Taylor approximation of the reverse-KL gradient, with bias. Rethinking KL recommends (P1) or (P2) on-policy; DAPO/GRPO historically use (P3) primarily as engineering convention + because the approximation holds in the small-β regime.

## §4 β schedule

### 4.1　Fixed β — baseline

The most common. $\beta \in [0.01, 0.5]$; **InstructGPT reports β = 0.02 as a reference value** (per-token KL). Pros: simple and reproducible. Cons: KL typically grows from small to large during training; a fixed β may "fail to suppress" late in training or "over-suppress" early on.

### 4.2　Adaptive β — Schulman PPO-Penalty original

The PPO paper (Schulman 2017 arXiv 1707.06347) originally proposed two versions: **Clip** (current mainstream) and **Penalty** (adaptive KL). The Penalty form:

$$\mathcal{L} = \mathbb{E}[r_t A_t] - \beta_k \cdot \text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)$$

**β adaptive rule** (after each epoch, look at measured KL $d$):

- If $d < d_\text{target} / 1.5$: $\beta \leftarrow \beta / 2$ (KL below target, relax)
- If $d > d_\text{target} \times 1.5$: $\beta \leftarrow \beta \times 2$ (KL above target, tighten)
- Otherwise β unchanged

Intuition: treat β as the P term of a PID controller with target = expected KL (e.g. $d_\text{target} = 0.01$). This approach also appeared in InstructGPT and later Anthropic work (Anthropic's helpfulness/harmlessness reports also contain similar adaptive-β descriptions).

### 4.3　β annealing schedule — analogous to learning rate schedule

View β as a time function $\beta(t)$:

- **Tight early, loose late**: $\beta(t) = \beta_0 \cdot \exp(-t / \tau)$. Intuition: early on, the policy is near ref — strengthen anchor to prevent instability; later the policy has learned — relax to let it explore reward.
- **Loose early, tight late**: opposite direction; explore RM signal early, anchor late. Rare.
- **Cosine / linear decay**: reference learning rate schedules.

In RLHF engineering, annealing is uncommon — adaptive β is more robust than schedules (no schedule shape to tune).

### 4.4　β failure modes

| β setting | Symptom | Diagnosis |
| --- | --- | --- |
| **β too large** ($> 1$) | KL ≈ 0, policy stuck near ref, RM reward doesn't rise | Reward curve is flat / `chosen_logp - rejected_logp` doesn't separate |
| **β too small** ($< 0.001$) | KL explodes (runaway), policy increasingly long / sycophantic / repetitive | KL curve climbing, generation length growing, human eval dropping |
| **β jumps** | Loss curve has discontinuities | Adaptive frequency too high / target_kl too strict |

> ⚠️ **Is InstructGPT's β = 0.02 universally right?** — Depends on task. **Math/code RL** typically needs smaller β (DeepSeekMath reports β ≈ 0.04, DAPO frequently uses much smaller β or 0) because reward is close to ground truth. **Helpfulness/safety RL** needs larger β (≥ 0.1) to prevent reward hacking, because neural RMs are easy to hack. That's why "β is not a universal hyperparameter".

### 4.5　Adaptive β code example

```python
class AdaptiveKLController:
    """
    Schulman 2017 PPO-Penalty style adaptive β controller.
    Call update(measured_kl) after every PPO epoch.
    """
    def __init__(self, beta_init=0.02, target_kl=0.01, horizon=10000):
        self.beta = beta_init
        self.target_kl = target_kl
        self.horizon = horizon          # smoothing coefficient (larger = slower)

    def update(self, measured_kl, n_steps):
        # proportional update; clip to prevent extreme jumps
        proportional_error = max(-0.2, min(0.2,
            measured_kl / self.target_kl - 1.0))
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.beta *= mult
        # safety clamp
        self.beta = max(1e-4, min(1.0, self.beta))
        return self.beta
```

Usage:

```python
kl_ctrl = AdaptiveKLController(beta_init=0.02, target_kl=0.01)
for epoch in range(num_epochs):
    # ... rollout and PPO update ...
    measured_kl = compute_mean_kl(policy, ref_policy, batch)   # k3 recommended
    kl_ctrl.update(measured_kl, n_steps=batch_size)
    current_beta = kl_ctrl.beta
```

> 💡 **Analogy: adaptive β vs PID controller** — Above is only a P term (proportional); HuggingFace TRL's implementation also uses only P. You could add I (integral: accumulated error) and D (derivative: rate of change) for full PID, but in practice P alone suffices; adding D tends to oscillate due to noisy KL estimates.

## §5 KL's relation to DPO / GRPO / SimPO / KTO / IPO

### 5.1　DPO: implicit reward = β × sequence log-density ratio

DPO's implicit reward is $\hat{r}_\theta(x, y) = \beta \log(\pi_\theta(y|x) / \pi_\text{ref}(y|x))$, i.e. the sequence-level log-density ratio for the entire $y$:

$$\hat{r}_\theta(x, y) = \beta \sum_t \log\frac{\pi_\theta(y_t | x, y_{<t})}{\pi_\text{ref}(y_t | x, y_{<t})}$$

> ⚠️ **Important distinction: the implicit reward is not KL itself** — $\hat{r}_\theta$ is **a single sequence's pointwise log-ratio**, **not** KL. Only when taking expectation over $y\sim\pi_\theta$ does $\mathbb{E}_{y\sim\pi_\theta}[\hat{r}_\theta(x,y)] = \beta\cdot\text{KL}(\pi_\theta(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x))$ (this is the sequence-level form of the k1 KL estimator). But **during DPO training, $y_w, y_l$ do not come from $\pi_\theta$ — they come from fixed preference data**, so the training-stage $\hat{r}_\theta$ cannot be read as KL; it is just a pairwise log-ratio difference.
>
> **Correct framing**: the dual RLHF interpretation of the DPO loss is to view $\hat{r}_\theta$ as a reward, with the BT model giving probability $\sigma(\hat{r}_w - \hat{r}_l)$; it corresponds to the reward expression obtained by inverting the "KL-regularized RL closed-form optimal policy $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$". $\pi_\theta$ is implicitly KL-constrained under the implicit-RLHF view (because $\pi_\theta/\pi_\text{ref}$ appears in $\hat r_\theta$), but DPO optimizes a pairwise margin rather than an explicit KL margin.

#### 5.1.1　DPO closed-form derivation review

KL-regularized objective:

$$\max_\pi \mathbb{E}_{x,\, y \sim \pi}[r(x, y)] - \beta\, \text{KL}(\pi \| \pi_\text{ref})$$

For a single $x$, using Lagrangian + differentiation (detailed derivation in §6.1):

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right)$$

Inversion:

$$r(x, y) = \beta \log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

Plugging into Bradley-Terry $P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$, the $\beta \log Z$ cancels, giving DPO:

$$\boxed{\;\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\;}$$

#### 5.1.2　KL interpretation of DPO's gradient

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta \mathbb{E}\!\Big[\sigma(\hat{r}_l - \hat{r}_w)\big(\nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta \log\pi_\theta(y_l|x)\big)\Big]$$

**Interpretation**:

- $\sigma(\hat{r}_l - \hat{r}_w)$ is "the current model's confidence in mis-ranking the preference order".
- $\nabla \log\pi(y_w) - \nabla \log\pi(y_l)$ is "raise probability of $y_w$ + lower $y_l$".
- $\beta$ appears twice: once inside the implicit reward $\hat{r}$ (determining the sigmoid's interior), once explicitly as a gradient scale. So **β controls both KL strength and gradient magnitude in DPO** — unlike in RLHF where β controls only one thing.

> ⚠️ **DPO's β hyperparameter pitfall** — Tuning β in DPO is not semantically equivalent to tuning β in RLHF. In PPO, β only affects KL penalty strength; in DPO, β simultaneously affects: (1) implicit-reward scale, (2) gradient magnitude (outer β), (3) sigmoid saturation position (inner β). **Empirically $\beta \in [0.05, 0.5]$ but the optimal differs per task**.

### 5.2　GRPO: k3 KL in loss

DeepSeekMath GRPO uses k3, with KL as a separate loss term:

$$L^\text{GRPO}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\!\Big(\min(\rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1{-}\epsilon, 1{+}\epsilon)\hat{A}_{i,t}) - \beta\,\widehat{\text{KL}}_3^{i,t}\Big)\right]$$

where $\widehat{\text{KL}}_3^{i,t} = e^{-\log r_{i,t}} + \log r_{i,t} - 1$ (per-token k3) and $\log r_{i,t} = \log\pi_\theta(y_{i,t}|\cdot) - \log\pi_\text{ref}(y_{i,t}|\cdot)$.

**Why did GRPO historically pick k3** (note: these are all **value-estimator** properties; they do not directly guarantee gradient correctness — see §3.6):

1. **Non-negative**: the loss-value display is intuitive (k1 values can be negative, making optimizer monitoring curves look odd during debugging — but k1 in reward is harmless).
2. **Value estimator unbiased**: $\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = \text{KL}$ (**unbiased only as a numerical KL estimate; when backpropagated w.r.t. θ, the gradient is a first-order Taylor approximation with $O(\Delta^2)$ bias**).
3. **Value estimator low variance**: smaller variance than k1's value (smoother monitoring curves).
4. **Doesn't enter advantage**: relatively decouples from In-reward placement, easier to diagnose (KL loss and PG loss monitored separately).

> ⚠️ **GRPO's k3-as-loss is not the theoretically principled best choice** — The Rethinking KL paper (arXiv 2510.01555) systematically analyzes that on-policy, (P1) k1-in-reward or (P2) k2-as-loss gives the strict reverse-KL gradient; GRPO/DAPO's actual engineering with tight β and small $\Delta$ makes the first-order approximation error limited — but **in an interview, one should distinguish "value-estimator advantage" from "loss-gradient correctness"**.

### 5.3　SimPO: drops reference, no KL anchor

SimPO redefines the implicit reward as length-normalized log-prob:

$$r_\text{SimPO}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$$

**Key difference**: **no $\pi_\text{ref}$**, hence **no KL term**. Loss:

$$\mathcal{L}_\text{SimPO} = -\mathbb{E}\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi(y_w) - \frac{\beta}{|y_l|}\log\pi(y_l) - \gamma\right)$$

**Consequences**:

- ✅ Saves one reference policy at training time (memory halved).
- ❌ No KL anchor, so the distance from SFT is **completely uncontrolled**.
- Empirically, SimPO outperforms DPO on certain benchmarks (AlpacaEval-2 / Arena-Hard); but generation quality on OOD prompts is **less stable than DPO**. SimPO's design is "length-norm + margin replaces KL anchor" — it assumes "for short prompt-response tasks, length-norm + margin is enough to constrain the policy", which **doesn't hold universally**.

> 💡 **Engineering impact of SimPO's missing KL** — In practice, SimPO-trained models exhibit **worse robustness on repetition patterns, generation length, and prompt-perturbation than DPO**. That's why many production RLHF deployments still use DPO + small β rather than SimPO — the KL anchor has a cost, but it has value.

### 5.4　KL handling in KTO / IPO / ORPO

| Algorithm | KL form | Notes |
| --- | --- | --- |
| **KTO** (Ethayarajh 2024 ICML) | Through reference point $z_0 = \mathbb{E}[\beta\cdot\text{KL}]$ (estimated from batch mismatched pairs, detached) | KL acts implicitly as an "anchor": how far implicit reward deviates from $z_0$ |
| **IPO** (Azar 2024 AISTATS) | Same as DPO ($\log(\pi/\pi_\text{ref})$), but loss changes from sigmoid to squared | Prevents unbounded growth of $\hat{r}$ under deterministic preferences |
| **ORPO** (Hong 2024 EMNLP) | **No reference model** (similar to SimPO), uses odds-ratio instead | Single-stage SFT + preference learning; KL anchor implicitly in SFT loss |

### 5.5　Recent papers (2024-2026)

| Paper | Claim | KL angle |
| --- | --- | --- |
| **DeepSeekMath GRPO** (Shao et al. 2024 arXiv 2402.03300) | Group-relative normalization + k3 KL in loss | First default use of k3 KL in LLM RL |
| **DPO Implicit Reward Models** (Rafailov 2023 NeurIPS) | DPO's implicit reward equals the KL log-ratio | DPO is fundamentally the inverse of KL-regularized optimization |
| **Reward Model Overoptimization Scaling Laws** (Gao, Schulman, Hilton 2023 ICML) | Gold reward shows an inverted-U over KL distance | KL is the natural x-axis for overoptimization |
| **DAPO** (Yu et al. 2025 ByteDance arXiv 2503.14476) | clip-higher + dynamic sampling + token-level loss | KL term often set small or off, with prompt-cleaning dynamic sampling |
| **Cohere DRO / OPO** (various 2024-2025 works) | offline IS-correction RL with KL | Combines IS-correction with KL anchor |
| **"Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization"** (Kezhao Liu et al. 2025-10, arXiv 2510.01555) | Systematically distinguishes KL value estimation from gradient optimization: k3-as-loss is a biased first-order approximation; recommends (1) k1 in reward (strict reverse-KL score-function gradient) or (2) **k2 as loss** (value-estimator biased but on-policy loss gradient strictly equivalent to k1-in-reward / reverse-KL gradient); off-policy needs IS correction | Historical GRPO's "k3 as loss" is engineering convention, not theoretically principled |
| **"A Comedy of Estimators: On KL Regularization in RL Training of LLMs"** (Vedant Shah et al. 2025-12, arXiv 2512.21852, v3 2026-03) | Systematic comparison across many RL algorithms + estimator-placement combinations of k1/k2/k3 estimator bias and gradient bias; analyzes placement-effect | No universally best estimator exists; reward-shaping vs loss-term have different sweet spots |

## §6 Theoretical: optimal policy of KL-regularized RL + reward overoptimization

### 6.1　Closed-form solution of KL-regularized RL (the mathematical basis for DPO)

**Theorem** (closed-form solution of KL-regularized policy optimization):

Consider:

$$\max_\pi J(\pi) = \mathbb{E}_{y \sim \pi(\cdot|x)}[r(x, y)] - \beta\, \text{KL}\!\big(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x)\big)$$

where $\pi$ is a distribution over any $x$, and $\pi_\text{ref}$ is strictly positive ($\pi_\text{ref}(y|x) > 0$ for all $y$).

**Unique optimal solution**:

$$\boxed{\;\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_\text{ref}(y|x)\, \exp\!\left(\frac{r(x, y)}{\beta}\right),\quad Z(x) = \sum_{y'} \pi_\text{ref}(y'|x) \exp\!\left(\frac{r(x, y')}{\beta}\right)\;}$$

**Proof** (Lagrangian):

Fix $x$ and write the objective (general continuous/discrete; discrete shown):

$$\mathcal{L}_x[\pi] = \sum_y \pi(y|x) r(x, y) - \beta \sum_y \pi(y|x) \log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)} - \mu\!\left(\sum_y \pi(y|x) - 1\right)$$

where $\mu$ is the Lagrange multiplier for normalization (sign note: when deriving KKT from max, writing $\sum_y\pi = 1$ as $-\mu(\sum - 1)$ keeps the interior-optimum condition concise).

Differentiating w.r.t. $\pi(y|x)$:

$$\frac{\partial \mathcal{L}_x}{\partial \pi(y|x)} = r(x, y) - \beta \log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)} - \beta - \mu = 0$$

Rearranging:

$$\log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)} = \frac{r(x, y) - \mu - \beta}{\beta}$$

Exponentiating:

$$\pi(y|x) = \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y) - \mu - \beta}{\beta}\right) = \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right) \cdot e^{-(\mu+\beta)/\beta}$$

Plugging into the normalization constraint $\sum_y \pi(y|x) = 1$:

$$e^{(\mu+\beta)/\beta} = \sum_y \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right) = Z(x)$$

So $e^{-(\mu+\beta)/\beta} = 1/Z(x)$, giving:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right) \quad \blacksquare$$

**Uniqueness**: because $J$ is **strictly concave** in $\pi$ (KL is convex, negating gives strictly concave; the reward term is linear), concave optimization has a unique optimum.

### 6.2　Inverting the optimal policy to extract implicit reward

Taking the log of $\pi^*$ from §6.1:

$$\log\pi^*(y|x) = \log\pi_\text{ref}(y|x) + \frac{r(x, y)}{\beta} - \log Z(x)$$

Inverting for $r$:

$$\boxed{\;r(x, y) = \beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta\log Z(x)\;}$$

**This is DPO's implicit reward**. $\beta \log Z(x)$ is the partition function, which **depends only on $x$, not on $y$** — so it **cancels** in the Bradley-Terry difference $r(x, y_w) - r(x, y_l)$.

### 6.3　RL under Bradley-Terry preference is equivalent to DPO

BT model: $P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$.

Plug §6.2 in:

$$r(x, y_w) - r(x, y_l) = \beta \log\frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}$$

Replace $\pi^*$ with learnable $\pi_\theta$, and take NLL over preference data:

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

**This is the DPO formula in §5.1**. So **DPO ≡ Bradley-Terry preference + KL-regularized RL closed form + preference-data NLL**, three pieces unified.

### 6.4　Reward Overoptimization (Gao, Schulman, Hilton 2023 ICML)

#### 6.4.1　Problem statement

Let gold reward $r_g$ (true human-perceived quality) and proxy reward $r_p$ (RM estimate). The RM is trained on finite preference data, so $r_p \ne r_g$.

As RL training pushes KL distance $d = \text{KL}(\pi_\theta \| \pi_\text{ref})$ up:

- $\mathbb{E}_{\pi_\theta}[r_p]$ rises monotonically (the policy learns the RM's preference).
- $\mathbb{E}_{\pi_\theta}[r_g]$ rises then falls (**inverted-U**) — this is reward overoptimization.

#### 6.4.2　Gao 2023's fitted form

Gao 2023 ran large-scale experiments over RM size + KL distance, giving a fitted form for gold reward vs KL distance $d$ (using $\sqrt{d}$ on the x-axis):

$$R_g(d) = d \cdot (\alpha_g - \gamma_g \cdot d) \quad \text{(BoN)}$$

$$R_g(d) = d \cdot (\alpha_g - \beta_g \cdot \log d)\quad \text{(PPO/RL: a log term, not a higher-order polynomial)}$$

where $\alpha_g, \gamma_g, \delta_g$ are coefficients dependent on RM size; larger RM → smaller "higher-order / quadratic" weight → slower overoptimization.

#### 6.4.3　KL is overoptimization's "natural axis"

Whether BoN, PPO, or DPO (via accumulated implicit reward), one can plot training curves as "**KL distance vs gold reward**". Gao 2023 found:

- **Different algorithms (BoN / PPO) exhibit similar over-optimization behavior at the same KL** (the gold curve shape under the same RM is consistent).
- **Larger RM → slower over-optimization** (gold-curve peak shifts right).
- **KL is a one-dimensional progress indicator** — better than step count / reward count for explaining over-optimization.

**Interview takeaway**: in RLHF monitoring, plot `(measured_KL, gold_reward)` to see whether it has entered the descending region — that is the most direct "stop early" signal.

### 6.5　Relation between KL and $\chi^2$ gap

KL and $\chi^2$ are related via the Pinsker family of inequalities:

$$\text{TV} \le \sqrt{\text{KL}/2}\qquad \text{(Pinsker)}$$

$$\text{KL} \le \log(1 + \chi^2)$$

In RLHF, $\chi^2$ is occasionally used as a "sensitive" upper bound on KL: **when $\chi^2$ is large but KL is small, $\pi_\theta$ has heavy-tail deviation from $\pi_\text{ref}$** (high variance but low mean). Some works (Yu et al. 2024 / others) plot KL and $\chi^2$ together when monitoring reward hacking, to track tail behavior.

### 6.6　Sequence-level vs token-level KL

By the **chain rule** in §1.1:

$$\text{KL}(\pi_\theta(\cdot|x) \| \pi_\text{ref}(\cdot|x)) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_t \log\frac{\pi_\theta(y_t | x, y_{<t})}{\pi_\text{ref}(y_t | x, y_{<t})}\right]$$

i.e. **sequence-level KL = expectation of the sum of token-level KLs** (autoregressive chain rule). Note: directly treating $\sum_t \log\pi_\theta(y_t)/\pi_\text{ref}(y_t)$ from a single rollout as KL is the **k1 estimator** (unbiased but high per-rollout variance); the true token-level KL $D(\pi_\theta(\cdot|s_t)\|\pi_\text{ref}(\cdot|s_t))$ would still require a full-vocab forward sum. **The two have the same expectation**; engineering uses the sampled log-ratio because vocab summation is expensive.

But two **implementation tricks** relate to token-level:

1. **Per-token clipping**: occasionally a single token's KL can be very large (rare tokens, long tails); per-token KL clipping prevents one token from dragging the batch total KL away.
2. **Mask on assistant tokens only**: in chat / agent settings, prompt tokens should not enter KL computation (the prompt is identical, policy and ref agree on it exactly, KL = 0; but floating-point noise pollutes it). So the KL mask coincides with the PPO action_mask — **compute KL only on assistant generation tokens**.

```python
# Per-token KL with action mask (chat / agentic RL setting)
def per_token_kl_with_mask(logp_theta, logp_ref, action_mask, estimator="k3"):
    """
    logp_theta, logp_ref: [B, T]   per-token log-prob
    action_mask:          [B, T]   1 = assistant token, 0 = prompt/system/pad
    estimator:            "k1" | "k2" | "k3"
    Returns: mean KL over assistant tokens, scalar.
    """
    log_r = logp_theta - logp_ref
    if estimator == "k1":
        kl_per_tok = log_r
    elif estimator == "k2":
        kl_per_tok = 0.5 * log_r ** 2
    elif estimator == "k3":
        delta = -log_r                        # log(π_ref / π_θ)
        kl_per_tok = torch.exp(delta) - delta - 1.0
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    masked = (kl_per_tok * action_mask).sum()
    n = action_mask.sum().clamp_min(1.0)
    return masked / n
```

## §7 Practice + code

### 7.1　PPO style: In-reward shaping implementation

```python
import torch
import torch.nn.functional as F

def ppo_reward_with_kl(rewards_terminal, rollout_logp_theta, logp_ref,
                       action_mask, beta=0.02):
    """
    PPO / InstructGPT style: KL penalty in reward (k1 estimator).
    rewards_terminal:    [B]      only the last assistant token gets the RM reward
    rollout_logp_theta:  [B, T]   log π_θ_old(y_t | ...), recorded during rollout, DETACHED
    logp_ref:            [B, T]   log π_ref(y_t | ...), reference policy frozen
    action_mask:         [B, T]   1 = assistant token
    Returns: shaped_reward [B, T]  per-token reward with KL penalty baked in.

    ⚠️ Crucial: rollout_logp_theta and logp_ref must be detached scalars (recorded at
    rollout time, or via no_grad forward). The shaped reward is the reward tensor passed
    through PPO / score-function backprop and **must not carry gradient**; otherwise the
    KL term and the PG surrogate will double-backprop and break the score-function
    semantics. Production code typically stores logp_old as fixed data in the rollout
    buffer, then re-forwards at update time to get new_logp.
    """
    B, T = rollout_logp_theta.shape
    # k1 KL per token (signed; mean is unbiased). Both inputs assumed detached.
    kl_per_tok = (rollout_logp_theta.detach() - logp_ref.detach()) * action_mask  # [B, T]
    # spread terminal reward to last assistant token
    last_token_idx = action_mask.cumsum(dim=-1).argmax(dim=-1)  # [B]
    R_per_tok = torch.zeros_like(kl_per_tok)
    R_per_tok[torch.arange(B), last_token_idx] = rewards_terminal
    # combine
    shaped = R_per_tok - beta * kl_per_tok
    return shaped  # the whole tensor is detached / gradient-free, serving as data input to PPO advantage computation
```

### 7.2　GRPO style: In-loss regularization implementation (k3)

```python
def grpo_loss_with_k3_kl(policy, ref_policy, batch, eps_clip=0.2, beta=0.04):
    """
    GRPO / DeepSeekMath style: KL penalty in loss (k3 estimator).
    batch:
      input_ids:     [N, L]    N = sum_b G_b samples
      action_mask:   [N, L]
      old_log_probs: [N, L]
      rewards:       [N]       sequence-level reward
      group_id:      [N]       which prompt
    """
    rewards = batch["rewards"]
    gid = batch["group_id"].long()
    num_groups = int(gid.max().item()) + 1

    # Group-relative advantage (z-score within group)
    counts = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, torch.ones_like(rewards))
    sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, rewards)
    group_mean = sums / counts.clamp_min(1.0)
    diff_sq = (rewards - group_mean[gid]) ** 2
    sq_sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, diff_sq)
    group_std = (sq_sums / counts.clamp_min(1.0)).sqrt()
    A = (rewards - group_mean[gid]) / (group_std[gid] + 1e-8)
    A = A.unsqueeze(-1)                                          # [N, 1] shared per token

    # Forward pass policy + ref
    logits = policy(batch["input_ids"]).logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, tgt).squeeze(-1)
    new_log_probs = F.pad(new_log_probs, (1, 0))                 # [N, L]
    mask = batch["action_mask"].float()

    with torch.no_grad():
        ref_logits = ref_policy(batch["input_ids"]).logits[:, :-1]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_lp = ref_log_probs.gather(-1, tgt).squeeze(-1)
        ref_token_lp = F.pad(ref_token_lp, (1, 0))

    # PPO-Clip surrogate (advantage shared per sample)
    ratio = torch.exp(new_log_probs - batch["old_log_probs"])
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A
    pg_per_tok = torch.min(surr1, surr2)                         # [N, L]

    # k3 KL per token: exp(Δ) - Δ - 1, Δ = log(π_ref / π_θ)
    delta = ref_token_lp - new_log_probs                         # log(π_ref / π_θ)
    kl_per_tok_k3 = torch.exp(delta) - delta - 1.0               # ≥ 0

    # Combine: maximize PG - β·KL  →  minimize -PG + β·KL
    token_obj = pg_per_tok - beta * kl_per_tok_k3
    seq_len = mask.sum(dim=-1).clamp_min(1.0)
    per_seq = (token_obj * mask).sum(dim=-1) / seq_len
    loss = -per_seq.mean()

    with torch.no_grad():
        kl_mean = (kl_per_tok_k3 * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {"kl_k3": kl_mean.item(), "advantage_std": A.std().item()}
```

### 7.3　DPO closed-form loss + implicit-reward monitoring

```python
def dpo_loss_with_implicit_reward_monitor(policy, ref_policy, batch, beta=0.1):
    """
    DPO loss + implicit reward margin (β × pointwise sequence log-density ratio on
    preference data; this is NOT a KL estimator on preference data — only equals
    β·KL when expectation is taken under y~π_θ, which doesn't hold for fixed preference pairs).
    """
    def log_prob_sum(model, ids, mask):
        logits = model(ids).logits[:, :-1]
        logp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:].unsqueeze(-1)
        token_logp = logp.gather(-1, tgt).squeeze(-1)
        token_mask = mask[:, 1:]
        return (token_logp * token_mask).sum(dim=-1)

    pi_w = log_prob_sum(policy, batch["chosen_ids"], batch["chosen_mask"])
    pi_l = log_prob_sum(policy, batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():
        ref_w = log_prob_sum(ref_policy, batch["chosen_ids"], batch["chosen_mask"])
        ref_l = log_prob_sum(ref_policy, batch["rejected_ids"], batch["rejected_mask"])

    # sequence-level log-density ratios on preference data
    log_ratio_w = pi_w - ref_w               # Σ_t log(π_θ/π_ref) on y_w
    log_ratio_l = pi_l - ref_l

    diff = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(diff).mean()

    # implicit rewards (DPO definition: β × pointwise sequence log-ratio, detached for logging)
    chosen_reward = beta * log_ratio_w.detach()
    rejected_reward = beta * log_ratio_l.detach()
    margin = (chosen_reward - rejected_reward).mean()

    # ⚠️ Note: the average log-ratio on preference data is **not** a KL estimator — KL requires y ~ π_θ samples.
    # Here we can only monitor "how much the model deviates from ref on preference data" as a DPO training-health indicator, not as KL.
    avg_pref_log_ratio = ((log_ratio_w.detach() + log_ratio_l.detach()) / 2).mean()
    return loss, {"margin": margin.item(),
                  "avg_pref_log_ratio": avg_pref_log_ratio.item(),
                  "chosen_reward": chosen_reward.mean().item(),
                  "rejected_reward": rejected_reward.mean().item()}
```

### 7.4　Reward overoptimization monitoring

```python
def overoptimization_monitor(policy, ref_policy, gold_reward_fn, prompts,
                             checkpoints_kl, gold_at_kl):
    """
    Track gold reward as a function of measured KL distance during training.
    Call periodically; plot (measured_KL, gold_reward) to visualize the inverted-U.

    Args:
      policy:          current π_θ
      ref_policy:      frozen π_ref
      gold_reward_fn:  callable (text -> float), uses gold RM or human eval
      prompts:         list of held-out prompts (small, e.g. 64)
      checkpoints_kl:  running list of KL values across training
      gold_at_kl:      running list of gold reward values
    """
    measured_kl_total = 0.0
    gold_reward_total = 0.0
    for prompt in prompts:
        # Generate from policy and ref; compute per-token k3 KL on policy generation
        out_policy = policy.generate(prompt, do_sample=True)
        # ... (use generate to get logits, mask, compute k3 KL) ...
        # For simplicity here, just track the average sequence-level KL
        kl_seq = compute_seq_kl_k3(policy, ref_policy, out_policy)
        gold = gold_reward_fn(out_policy)
        measured_kl_total += kl_seq
        gold_reward_total += gold
    avg_kl = measured_kl_total / len(prompts)
    avg_gold = gold_reward_total / len(prompts)
    checkpoints_kl.append(avg_kl)
    gold_at_kl.append(avg_gold)
    # Detection heuristic: if last 5 gold values are decreasing while KL is rising,
    # we are likely in the over-optimization regime.
    if len(gold_at_kl) >= 5:
        recent_gold = gold_at_kl[-5:]
        recent_kl = checkpoints_kl[-5:]
        if all(recent_gold[i] >= recent_gold[i+1] for i in range(4)) \
           and all(recent_kl[i] <= recent_kl[i+1] for i in range(4)):
            print(f"⚠️ Possible reward over-optimization: KL ↑, gold ↓ over 5 checkpoints")
    return avg_kl, avg_gold
```

### 7.5　Engineering checklist (debug checklist)

> ⚠️ **Top-8 KL-related bugs** —

1. **Wrong mask for KL**: KL on prompt tokens should be 0 (same prompt input), but floating-point noise pollutes it — so **mask only assistant tokens**.
2. **k1 shows negative**: a single-sample log-ratio can be negative, but **the expectation is non-negative**. Not a bug — it's an estimator property. Switch to k3 / k2 for more intuitive monitoring.
3. **Wrong β unit**: when reward scale = O(1), β = 0.02; when reward scale = O(100), β must be scaled accordingly. Otherwise the KL anchor fails.
4. **Adaptive β oscillates**: target_kl too strict / mult too large. Clip mult to [0.8, 1.2], reduce update frequency to every 100 steps.
5. **Reference policy not frozen**: forgetting `ref_policy.eval()` + `torch.no_grad()`, ref trains along with policy, KL turns into self-distillation.
6. **Per-token vs sequence-level KL confused**: monitoring sometimes reports per-token, sometimes sequence-level — misreading leads to mis-tuning β. Unify units.
7. **KL term shrinks reward in GRPO**: when reward is binary {0, 1} (math problems) and β = 0.04 with KL ≈ 0.5, the KL penalty already exceeds the mean reward — loss is dominated by KL. **Drop β to 1e-3 or smaller**.
8. **Float overflow in $\pi_\text{ref}/\pi_\theta$**: when the policy diverges from ref by a lot, $\pi_\text{ref}/\pi_\theta$ can be very large and $e^\Delta$ overflows. **Compute k3 in log-space**: `exp(delta) - delta - 1` may still overflow for large $\Delta$, so add `torch.clamp(delta, max=10)` or use a numerically stable form.

## §8 Failure modes: KL Collapse / Runaway / Reward Hacking

### 8.1　KL Collapse (β too large or entropy too low)

**Symptoms**:

- KL curve $\approx 0$, policy equals reference.
- Reward doesn't rise.
- Generation identical to SFT.

**Causes**:

- β too large, KL penalty dominates loss, PG signal is suppressed.
- Policy entropy too low (SFT was already deterministic) → further updates are difficult.

**Fixes**:

- Lower β (adaptive controller should handle this automatically).
- Add entropy bonus ($+ c_e \cdot H[\pi_\theta]$).
- Check PG-vs-KL-loss magnitude ratio: ideal $|\text{PG}| / (\beta \cdot |\text{KL}|) \in [1, 10]$.

### 8.2　KL Runaway (β too small or reward signal too strong)

**Symptoms**:

- KL keeps rising, policy drifts further from ref.
- Generation lengthens, repetition patterns appear, style drifts.
- Gold reward on validation starts dropping (reward overoptimization).

**Causes**:

- β too small, KL anchor fails.
- Reward signal too strong (neural RM produces huge gradient).
- PPO importance ratio frequently outside clip range, KL signal masked by clip (Option A specific).

**Fixes**:

- Raise β (adaptive).
- Add max_kl_budget early stop (when measured KL > target, stop training).
- Use RM ensemble + conservative aggregation (min / mean - std).
- For Option A, consider switching to Option B (KL not masked by PPO clip).

### 8.3　Reward Overoptimization (the inverted-U over KL distance)

**Symptoms**:

- Proxy reward keeps rising.
- Gold reward rises then falls (inverted-U).
- Human eval: model looks good on in-distribution data, but crashes on OOD prompts.

**Causes**: see §6.4 — the RM and gold reward are fundamentally different.

**Fixes**:

1. **KL budget early stop**: cap measured KL before the inverted-U peak (requires gold reward monitoring).
2. **RM ensemble**: multiple seeds of RM, take min or mean - $k\cdot\text{std}$.
3. **Mix RL + DPO**: DPO for 70%, then small-β PPO for the last mile.
4. **Rule-based reward replacing neural RM**: the fundamental fix for math/code tasks.
5. **PRM replacing ORM**: dense reward → bounded per-step over-optimization.

### 8.4　Length bias (DPO-specific, also a form of insufficient KL anchor)

**Symptoms**:

- DPO-trained outputs visibly longer.
- AlpacaEval score high but users find them verbose.

**Cause**: DPO loss is sequence-level log-ratio difference. $y_w$ is typically longer (humans prefer more detailed answers); longer $y_w$ → more negative $\log\pi(y_w)$ → larger log-ratio difference → smaller loss. But this is RM scale, not reasoning quality.

**Fixes**:

- SimPO's length-normalization ($r = (\beta/|y|)\log\pi$).
- Add a length penalty in reward shaping.
- Data curation: make $y_w$ and $y_l$ have similar length distributions.

### 8.5　Reference policy "wrong checkpoint" failure

**Symptoms**: training looks normal but downstream eval is awful.

**Cause**: wrong ref checkpoint (e.g. pretrain base instead of SFT). KL anchors to "language model" rather than "instruction model".

**Fix**: reference must be the immediate-previous-stage SFT checkpoint — never skip levels.

## §9 25 high-frequency interview questions

Split by difficulty into 3 tiers: L1 = any LLM engineering role; L2 = research / alignment teams; L3 = top-lab / DeepSeek-grade hardcore. Each question has an answer summary + common pitfalls. Click to expand.

### L1 must-know (10 questions)

<details>

<summary>Q1. KL divergence's definition and 3 core properties?</summary>

- Definition: $\text{KL}(p\|q) = \mathbb{E}_{x\sim p}[\log(p(x)/q(x))]$
- Non-negative (Jensen); equality ⟺ $p = q$ a.e.
- Asymmetric ($\text{KL}(p\|q) \ne \text{KL}(q\|p)$), not a metric
- Jointly convex, invariant under reparameterization (reparam invariant)

Saying KL is a "distance" — wrong (no triangle inequality); not knowing convexity.

</details>

<details>

<summary>Q2. Why does RLHF need a KL penalty?</summary>

- Prevent reward hacking: the policy finds RM blind spots for high scores but worse human-perceived quality
- Prevent linguistic fluency collapse: without KL, the model may emit "gibberish that RM scores high"
- Prevent distribution shift: when the policy drifts far from SFT, even the RM can't score accurately (OOD)
- Provides a closed-form optimal policy: $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$

Only saying "prevents overfitting" — not specific enough; not knowing about reward hacking.

</details>

<details>

<summary>Q3. What is the k1 estimator? Why can a single sample be negative?</summary>

- k1: $\widehat{\text{KL}}_1 = \log(\pi_\theta(y)/\pi_\text{ref}(y))$ (direct log-ratio)
- **Expectation** $\mathbb{E}_{y\sim\pi_\theta}[\log r] = \text{KL}$ (unbiased), but **a single sample** $\log r$ can be positive or negative
- Reason: at a single sample $y$, $\pi_\theta$ may be larger or smaller than $\pi_\text{ref}$, the log-ratio has no "non-negative" constraint
- Engineering impact: don't panic when k1 monitoring shows negative — that's the estimator's nature

Confusing single estimator with expectation; not knowing expectation non-negative ≠ pointwise non-negative.

</details>

<details>

<summary>Q4. What is the k3 estimator (Schulman 2020) formula? What are its three properties?</summary>

- Formula: $\widehat{\text{KL}}_3 = e^\Delta - \Delta - 1$, $\Delta = \log(\pi_\text{ref}/\pi_\theta) = -\log r$
- Equivalent form: $\widehat{\text{KL}}_3 = (\pi_\text{ref}/\pi_\theta) - \log(\pi_\text{ref}/\pi_\theta) - 1$
- **Unbiased** ($\mathbb{E}_{\pi_\theta} = \text{KL}$, using $\sum_y\pi_\text{ref} = 1$)
- **Non-negative** ($f(x) = e^x - x - 1 \ge 0$)
- **Low variance** (with the expectation-0 control variate $\pi_\text{ref}/\pi_\theta - 1$)

Only memorizing the formula without knowing the derivation; or not knowing it is non-negative.

</details>

<details>

<summary>Q5. What are the two KL placements in RLHF? What's the difference?</summary>

- **In-reward shaping (Option A)**: $\tilde{r}_t = r_t - \beta \cdot \text{KL}_t$ (uses k1), KL enters advantage / GAE. InstructGPT standard.
- **In-loss regularization (Option B)**: $\mathcal{L} = \mathcal{L}_\text{PG} + \beta \cdot \mathbb{E}[\text{KL}]$. GRPO / DAPO historically pick k3 estimator.
- Same objective, but **the gradient path differs**. **Principled gradient** choices: (P1) k1 in reward; (P2) k2 as loss — on-policy these are gradient-equivalent, both being the strict reverse-KL gradient. **GRPO's k3-as-loss** is a first-order Taylor approximation of the reverse-KL gradient, with $O(\Delta^2)$ bias (see §3.6 / Rethinking KL arXiv 2510.01555).
- PPO clip in Option A "masks" out-of-clip KL (advantage's embedded KL gets clipped together); Option B isn't affected.
- Practice: math/code RL historically picks B (k3) because β is small and $\Delta$ is small — bias is negligible; helpfulness / safety mostly uses A (k1); for strictly principled on-policy, P2 (k2-as-loss) is the simplest principled choice.

Saying the two are fully equivalent — same objective but not gradient-equivalent; saying k3-as-loss is principled — it is a first-order approximation, theoretically inexact.

</details>

<details>

<summary>Q6. How do you tune β? What are the symptoms when it's too large / too small?</summary>

- β too large: KL ≈ 0, policy stuck near ref, reward doesn't rise, model essentially equals SFT
- β too small: KL runaway, policy drifts, generation lengthens / repeats / panders, reward hacking
- Starting points: InstructGPT uses β ≈ 0.02 (per-token); GRPO math task β ≈ 0.04 or smaller; DAPO often β ≈ 0
- Methods: fixed / adaptive (pull β based on measured KL vs target) / annealing schedule

Only saying "tune by feel"; not knowing the engineering order of magnitude for β.

</details>

<details>

<summary>Q7. Which one does RLHF use — Forward KL or Reverse KL? Why?</summary>

- RLHF almost always uses $\text{KL}(\pi_\theta \| \pi_\text{ref})$; under the standard VI/RLHF convention (used by DPO/RLOO/Rethinking KL etc.) this is **reverse KL** ($\pi_\theta$ is the variational side), mode-seeking
- Because training samples from $\pi_\theta$ (rollout), and $\mathbb{E}_{\pi_\theta}[\log r]$ is directly estimable from samples
- **Forward KL** $\text{KL}(\pi_\text{ref} \| \pi_\theta)$ requires sampling from $\pi_\text{ref}$, which is engineering-meaningless (we are training $\pi_\theta$, not $\pi_\text{ref}$), and mass-covering doesn't fit the RLHF goal
- Reverse KL's mode-seeking matches the rule perfectly: pick the reward-highest mode inside the high-density region of $\pi_\text{ref}$

Not knowing the naming convention (a few early RL textbooks name by sampling distribution and label reverse as forward; in an interview, giving the formula is safer); or saying forward is better.

</details>

<details>

<summary>Q8. How does DPO's implicit reward relate to KL?</summary>

- DPO implicit reward $\hat{r}_\theta(x, y) = \beta\log(\pi_\theta(y|x)/\pi_\text{ref}(y|x))$, i.e. **β × pointwise sequence log-density ratio**
- ⚠️ **It is not KL itself**: only when taking expectation over $y\sim\pi_\theta$ does $\mathbb{E}_{y\sim\pi_\theta}[\hat r_\theta(x,y)] = \beta\cdot\text{KL}(\pi_\theta(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x))$. **In DPO training, $y_w, y_l$ come from fixed preference data**, so the $\hat r$ here is not a KL estimator nor a KL margin
- During training, maximizing $\hat{r}(y_w) - \hat{r}(y_l)$ ≡ pairwise log-ratio margin (relative confidence on the preference pair), **not** a "KL margin"
- The KL anchor is **implicit**: $\pi_\text{ref}$ appearing in the denominator of $\hat r$ provides anchoring, but no explicit KL term is in the loss
- Dual RL view: DPO is derived from the closed-form optimal policy $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$ of KL-regularized RL; the implicit objective is the same RLHF objective, but the estimator path differs

Saying DPO has no KL (wrong — it's an implicit anchor); reading $\hat r$ directly as KL / KL margin (wrong — requires the y~π_θ expectation condition).

</details>

<details>

<summary>Q9. What is reward overoptimization? Why does it happen?</summary>

- Proxy reward (RM) rises monotonically with KL, but gold reward (human eval) rises then falls (**inverted-U**)
- Gao, Schulman, Hilton 2023 ICML gives the fitted form for KL vs gold-reward
- Cause: RM and gold reward differ (RM is a finite-data learned approximation), policy learns RM's preference but not necessarily true quality
- Mitigations: KL budget early stop / RM ensemble / PRM / mix DPO + PPO

Only saying "overfitting" — too vague; not knowing KL is the axis for over-optimization.

</details>

<details>

<summary>Q10. Why did GRPO historically use k3 instead of k1?</summary>

- **Engineering motivation**: GRPO puts KL in the loss, and k3's **value-estimator** properties (non-negative + value-side unbiased + low variance) make the KL monitoring curve intuitive
- **k1-as-loss is infeasible**: backpropagating $\log r = \log\pi_\theta - \log\pi_\text{ref}$ directly gives $\nabla = \nabla\log\pi_\theta$, which has expectation 0 on-policy — **not** the reverse-KL gradient. k1 must **enter reward / advantage** (as a detached scalar via the score-function trick) to give the strict reverse-KL gradient. Putting k1 as a loss term is bad both because the value can be negative (visually unappealing) and because the gradient is wrong
- **k3-as-loss's true cost**: backpropagating w.r.t. θ gives a **first-order Taylor approximation** of the reverse-KL gradient, $1 - e^{-\Delta} = \Delta - \tfrac12\Delta^2 + O(\Delta^3)$, with $O(\Delta^2)$ bias (see §3.6 + Rethinking KL arXiv 2510.01555)
- **Principled alternative**: on-policy, (P1) k1 in reward or (P2) **k2 as loss** ($\tfrac12\Delta^2$, $\nabla = \Delta\cdot\nabla\log\pi_\theta$, gradient-equivalent to P1) are both the strict reverse-KL gradient
- **Historical DeepSeekMath / DAPO / verl default to k3** as engineering convention; when β is tight and $\Delta$ is small, bias is negligible — but this is empirical rationalization, not theoretical principle

Only saying k3 "satisfies all three" without the gradient-bias caveat; not knowing that (P2) k2-as-loss is the gradient-equivalent alternative to P1.

</details>

### L2 advanced (10 questions)

<details>

<summary>Q11. Derive the unbiasedness of k3.</summary>

We need to show $\mathbb{E}_{y\sim\pi_\theta}[(\pi_\text{ref}/\pi_\theta) - \log(\pi_\text{ref}/\pi_\theta) - 1] = \text{KL}(\pi_\theta\|\pi_\text{ref})$.

1. $\mathbb{E}_{\pi_\theta}[\pi_\text{ref}/\pi_\theta] = \sum_y \pi_\theta(y)\cdot\pi_\text{ref}(y)/\pi_\theta(y) = \sum_y\pi_\text{ref}(y) = 1$
2. $\mathbb{E}_{\pi_\theta}[-\log(\pi_\text{ref}/\pi_\theta)] = \mathbb{E}_{\pi_\theta}[\log(\pi_\theta/\pi_\text{ref})] = \text{KL}(\pi_\theta\|\pi_\text{ref})$
3. Combining: $\mathbb{E}[\widehat{\text{KL}}_3] = 1 + \text{KL} - 1 = \text{KL}$ ✓

Key: use $\sum_y\pi_\text{ref} = 1$. If you forget the $\pi_\text{ref}/\pi_\theta$ term is a control variate with expectation 1, you can't prove unbiasedness.

</details>

<details>

<summary>Q12. Derive the k3 estimator from $f(x) = e^x - x - 1$.</summary>

Constructive proof:

1. $f(x) = e^x - x - 1$ is non-negative on $\mathbb{R}$ (the convex $e^x$ lies above its tangent line $1 + x$ at $x = 0$).
2. Set $x = \log(\pi_\text{ref}(y)/\pi_\theta(y)) = -\log r$:
   - $e^{-\log r} = 1/r = \pi_\text{ref}/\pi_\theta$
   - $-(-\log r) = \log r = \log(\pi_\theta/\pi_\text{ref})$
   - $f(-\log r) = \pi_\text{ref}/\pi_\theta - \log(\pi_\text{ref}/\pi_\theta) - 1 = e^\Delta - \Delta - 1$, $\Delta = -\log r$
3. Expectation: $\mathbb{E}_{\pi_\theta}[f(-\log r)] = \text{KL}$ (proved in Q11).

So k3 = special form of $f(\log p/q)$, which happens to satisfy all three properties "non-negative + unbiased + with control variate".

Just writing the formula without explaining the motivation of $f$; not knowing the convex tangent → non-negative.

</details>

<details>

<summary>Q13. Derive the closed-form optimal policy of KL-regularized RL under BT preference.</summary>

Objective: $\max_\pi \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,\text{KL}(\pi\|\pi_\text{ref})$.

1. Lagrangian: $\sum_y \pi r - \beta\sum_y\pi\log(\pi/\pi_\text{ref}) - \mu(\sum_y\pi - 1)$
2. Differentiate w.r.t. $\pi(y)$ and set $= 0$: $r - \beta(\log(\pi/\pi_\text{ref}) + 1) - \mu = 0$
3. Rearrange: $\log(\pi/\pi_\text{ref}) = (r - \mu - \beta)/\beta$
4. Exponentiate: $\pi(y) = \pi_\text{ref}(y)\exp((r - \mu - \beta)/\beta)$
5. Use $\sum_y\pi = 1$ to solve for $\mu$: $e^{(\mu+\beta)/\beta} = \sum_y\pi_\text{ref}\exp(r/\beta) = Z$
6. Get: $\pi^*(y|x) = \pi_\text{ref}(y|x)\exp(r/\beta)/Z(x)$

Note: $J$ is strictly concave in $\pi$, so the optimum is unique.

Missing the normalization constraint; missing the strict concavity argument; sign flip.

</details>

<details>

<summary>Q14. Derive the DPO loss starting from the closed-form $\pi^*$ in §6.1.</summary>

1. §6.1 gives $\pi^*(y|x) = \pi_\text{ref}\exp(r/\beta)/Z(x)$
2. Inversion: $r(x,y) = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$
3. Bradley-Terry: $P(y_w \succ y_l) = \sigma(r_w - r_l)$
4. Key observation: $\beta\log Z(x)$ doesn't depend on $y$, so **it cancels in $r_w - r_l$**:
   $r_w - r_l = \beta\log(\pi^*(y_w)/\pi_\text{ref}(y_w)) - \beta\log(\pi^*(y_l)/\pi_\text{ref}(y_l))$
5. Replace $\pi^*$ with learnable $\pi_\theta$, NLL over preference data:
   $\mathcal{L}_\text{DPO} = -\mathbb{E}\log\sigma(\beta\log(\pi_\theta(y_w)/\pi_\text{ref}(y_w)) - \beta\log(\pi_\theta(y_l)/\pi_\text{ref}(y_l)))$

Key trick: **$\log Z$ doesn't depend on $y$**, so it cancels in the pairwise difference (this is why DPO doesn't need sampling).

Not explaining why $\log Z$ cancels; not knowing about BT's sigmoid.

</details>

<details>

<summary>Q15. Derive the DPO gradient.</summary>

Let $h_\theta = \beta\log(\pi_\theta(y_w)/\pi_\text{ref}(y_w)) - \beta\log(\pi_\theta(y_l)/\pi_\text{ref}(y_l))$ (implicit reward margin).

$\mathcal{L}_\text{DPO} = -\mathbb{E}\log\sigma(h_\theta)$, $\nabla = -\mathbb{E}\sigma'(h_\theta)/\sigma(h_\theta) \cdot \nabla h_\theta$.

Using $\sigma'(x) = \sigma(x)\sigma(-x)$, $\sigma'(h)/\sigma(h) = \sigma(-h) = \sigma(\hat{r}_l - \hat{r}_w)$ (i.e. "misranking confidence").

$\nabla h_\theta = \beta(\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l))$ ($\pi_\text{ref}$ is constant, derivative w.r.t. $\theta$ is 0).

Combining:

$$\nabla\mathcal{L}_\text{DPO} = -\beta\mathbb{E}[\sigma(\hat{r}_l - \hat{r}_w)(\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l))]$$

Interpretation:

- The coefficient $\sigma(\hat{r}_l - \hat{r}_w)$ is "how much the current model misranks $y_l$ above $y_w$" → hard-example mining
- The rest is "raise $y_w$ probability + lower $y_l$ probability"

Missing a step in the sigmoid' derivation; not knowing $\sigma'(x) = \sigma(x)\sigma(-x)$.

</details>

<details>

<summary>Q16. Why is the k2 estimator biased? In the small-KL limit, by how much?</summary>

Let $r = \pi_\theta/\pi_\text{ref}$, $y \sim \pi_\theta$, $\text{KL}(\pi_\theta\|\pi_\text{ref}) = \mathbb{E}_{\pi_\theta}[\log r]$.

- **k2** is defined as $\mathbb{E}_{\pi_\theta}[\tfrac12(\log r)^2]$, generally $\ne \mathbb{E}[\log r]$ — so **biased**.
- **The correct Taylor expansion** should use $s = \pi_\text{ref}/\pi_\theta = 1/r$ (this is the random variable with $\mathbb{E}_{\pi_\theta}[s] = 1$, so we can do a mean-1 expansion). $\log r = -\log s$, $\log s = (s-1) - \tfrac12 (s-1)^2 + O((s-1)^3)$.
- $\mathbb{E}_{\pi_\theta}[\log r] = -\mathbb{E}[\log s] = -\mathbb{E}[s-1] + \tfrac12\mathbb{E}[(s-1)^2] + O(\cdot) = 0 + \tfrac12\mathrm{Var}(s) + O$. This matches the Fisher second-order expansion.
- $\mathbb{E}_{\pi_\theta}[\tfrac12 (\log r)^2] = \tfrac12\mathbb{E}[(\log s)^2] = \tfrac12\mathbb{E}[(s-1)^2] + O((s-1)^3) = \tfrac12\mathrm{Var}(s) + O$.
- So **in the small-KL limit (i.e. $r\approx 1$)**: $\mathbb{E}[k_2] \approx \mathbb{E}[\log r] = \text{KL}$, second-order equivalent.
- For large KL, k2 systematically deviates from KL (higher-order terms can't be ignored, direction depends on higher moments).

**Key correction**: a previous version wrote "$\mathbb{E}[r-1] = 0$" — wrong, because under $y\sim\pi_\theta$, $\mathbb{E}[r] = \mathbb{E}_{\pi_\theta}[\pi_\theta/\pi_\text{ref}]$ is generally $\ne 1$. $\mathbb{E}[s] = 1$ is the correct identity (this is the standard importance-sampling result).

Only saying "second-order approximation"; not expanding Taylor, or mistakenly using $r$ instead of $s$ for the mean-1 expansion.

</details>

<details>

<summary>Q17. Analogy between adaptive β and PID controllers?</summary>

- Adaptive β (PPO Schulman 2017) only uses the P term: if KL > target → β ↑; if KL < target → β ↓
- Comparison to a PID controller: $u(t) = K_p e + K_i\int e + K_d \dot{e}$
- P term = current-error proportional response
- I term = accumulated error (prevents steady-state error), but KL is stochastic; adding I oscillates
- D term = error derivative (damping), but KL estimates have high variance; D amplifies noise
- Practice: **only the P term** is most robust (TRL default); some frameworks add a small I for long-term convergence

Only saying "P-controller-like" without elaboration; or adding D term without knowing it has problems.

</details>

<details>

<summary>Q18. How to tune β in GRPO with k3 KL when reward is binary (0/1)?</summary>

- Reward = {0, 1}, advantage scale ≈ O(1)
- KL_k3 per-token starts at ≈ 0 (policy equals ref), can reach 0.1 ~ 0.5 during training
- If β = 0.04, KL penalty per-token ≈ 0.04 × 0.3 ≈ 0.012, reasonable vs advantage
- If β = 1, KL penalty per-token ≈ 0.3, **much larger than** advantage, loss dominated by KL
- Experience: math task starts at β = 0.01 ~ 0.04, adjusted by KL curve; DAPO often β = 1e-3 or 0
- Compare RLHF helpfulness: reward scale is large (continuous [-5, 5]) → β can be larger (0.1 ~ 0.5)

Not knowing the coupling between β and reward scale; or mechanically applying β = 0.04.

</details>

<details>

<summary>Q19. SimPO has no reference model — why does it still work? Without the KL anchor, doesn't it reward-hack?</summary>

- SimPO uses **length-normalization** $r = (\beta/|y|)\log\pi(y)$ + **target reward margin** $\gamma$
- No KL anchor, but length-norm prevents the "$y_w$ longer = better" degeneration
- Margin $\gamma$ makes the loss saturate at $r_w - r_l > \gamma$, avoiding unbounded implicit reward
- Empirically SimPO outperforms DPO on some benchmarks (AlpacaEval-2 / Arena-Hard)
- But **the cost of no KL anchor**: poor OOD robustness, sensitive to $\beta, \gamma$ tuning, generation length may still be too large
- In production, DPO + small β is still mainstream; SimPO is a "stronger on some benchmarks but with different tradeoffs" choice

Saying SimPO is strictly better; not knowing length-norm + margin is the KL substitute.

</details>

<details>

<summary>Q20. How are per-token KL and sequence-level KL related?</summary>

- By KL's **chain rule** (conditional decomposition):
  $\text{KL}(\pi_\theta(\cdot|x) \| \pi_\text{ref}(\cdot|x)) = \mathbb{E}_{y\sim\pi_\theta}[\sum_t \log(\pi_\theta(y_t|x,y_{<t})/\pi_\text{ref}(y_t|x,y_{<t}))]$
- I.e. **sequence-level KL = expectation of the sum of token-level KLs**
- Engineering must compute at the token level (vocab summation is fine, sequence summation explodes combinatorially)
- Actual RLHF loss: per-token KL → mask assistant tokens → sum → average per sequence
- Note: on prompt tokens, policy and ref have identical input, so KL should = 0, but floating-point noise pollutes → mask required

Confusing the two; or not knowing why masking only assistant tokens is necessary.

</details>

### L3 top-lab questions (5 questions)

<details>

<summary>Q21. Prove $f(x) = e^x - x - 1 \ge 0$ for all $x \in \mathbb{R}$. Starting from this inequality, derive the non-negativity and unbiasedness of k3. Which property is more critical for "KL as a loss term", and why?</summary>

**Non-negativity proof**:

$f(x) = e^x - x - 1$. $f'(x) = e^x - 1$, $f'(x) = 0$ ⟺ $x = 0$. $f''(x) = e^x > 0$, so $f$ is strictly convex; $x = 0$ is the global minimum, $f(0) = 0$. So $f(x) \ge 0$ for all $x$. ✓

**k3 non-negativity**: Setting $x = \log(\pi_\text{ref}/\pi_\theta) = -\log r$, $\widehat{\text{KL}}_3 = f(-\log r) \ge 0$ always. ✓

**k3 unbiasedness**:

$\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = \mathbb{E}_{\pi_\theta}[e^{-\log r}] + \mathbb{E}_{\pi_\theta}[\log r] - 1 = 1 + \text{KL} - 1 = \text{KL}$ ✓

**Which property is more critical for "KL as a loss term"?**

- **Non-negative + value-side unbiased**: these are good **monitoring/numerical-display** properties — the KL loss curve doesn't show negatives or huge noise. But **the key issue with k3-as-loss is not in the numerical layer**, but in the gradient layer.
- ⚠️ **The gradient when k3 is backpropagated as a loss is a first-order Taylor approximation of the reverse-KL gradient**: $\nabla(e^{-\Delta} + \Delta - 1) = (1-e^{-\Delta})\nabla\Delta = (\Delta - \tfrac12\Delta^2 + O(\Delta^3))\,\nabla\log\pi_\theta$, with $O(\Delta^2)$ bias (see §3.6 + Rethinking KL arXiv 2510.01555).
- **The truly principled loss choice is (P2) k2-as-loss**: $\nabla(\tfrac12\Delta^2) = \Delta\,\nabla\log\pi_\theta$ equals the strict reverse-KL gradient in expectation on-policy. k2's value is not calibrated but its gradient is correct — gradient-equivalent to k1-in-reward.
- **If "non-negative + unbiased + easy to monitor" is required** ⇒ k3 remains the best **value estimator**; but distinguishing "value-side k3 monitor + loss-side k2/k1-in-reward gradient" is a more principled combination than "k3-as-loss".

So k3 is most convenient at the **value-monitoring layer** (non-negative + value-unbiased + low variance). But **for the principled-ness of "as a loss addend", the key is not non-negativity but whether the loss-gradient matches the true reverse-KL gradient** — k3-as-loss's gradient $(1 - e^{-\Delta})\nabla\log\pi_\theta$ is only a first-order Taylor approximation with $O(\Delta^2)$ bias; on-policy, the strictly principled losses are (P2) k2-as-loss or (P1) k1-in-reward. GRPO/DAPO historically used k3 mostly as engineering convention + because the bias is negligible at small β.

**Common mistake**: treating non-negativity as the "key for loss-addend" — that's only the numerical display advantage; what truly determines principled-ness is gradient correctness. The ideal combination is "value-side k3 monitor + loss-side k2/k1-in-reward gradient".

</details>

<details>

<summary>Q22. Derive the "dual equivalence" between DPO's implicit reward and the policy gradient of KL-regularized RL.</summary>

Setup: $\max_\pi \mathbb{E}_\pi[r] - \beta\,\text{KL}(\pi\|\pi_\text{ref})$, closed-form $\pi^*(y|x) = \pi_\text{ref}\exp(r/\beta)/Z$.

**RL view**: when reward $r$ is given, the optimal policy moves towards $\pi^*$ via importance-weighted updates. Score-function policy gradient:

$\nabla_\theta J = \mathbb{E}_{\pi_\theta}[\nabla\log\pi_\theta \cdot (r - \beta\log(\pi_\theta/\pi_\text{ref}))]$

**DPO view**: treat $\pi$ as the learnable $\pi_\theta$, invert BT to get implicit reward $\hat{r} = \beta\log(\pi_\theta/\pi_\text{ref}) + \beta\log Z$, do NLL over preference pairs.

**Equivalence**:

1. The closed-form gives $r = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z$.
2. If $\pi_\theta = \pi^*$ (i.e. RL has converged), then the implicit reward recovers the true $r$ (up to additive $\beta\log Z$, which cancels in BT pairwise).
3. RL gradient $\propto r - \beta\log(\pi_\theta/\pi_\text{ref})$ at convergence $= \beta\log Z(x)$ (constant in $y$) → gradient is 0 (PG stops at $\pi^*$).
4. DPO gradient $\propto \sigma(\hat{r}_l - \hat{r}_w)(\nabla\log\pi(y_w) - \nabla\log\pi(y_l))$, at $\pi_\theta = \pi^*$ corresponds to "BT model fits perfectly" → gradient also vanishes.
5. The fixed points of both views are $\pi^*$, so **RL and DPO are optimizing the same objective from two angles**: RL is forward optimization (directly max $J$), DPO is inverse optimization (fit BT model to preference data).

**Key**: DPO is not a replacement for RL — it is **a different reduction of the same objective**. RL via sampling-based PG, DPO via closed-form + supervised learning. "DPO has no RL" is a misreading.

Only saying "DPO derives from RL closed form" without expanding gradient equivalence; or not knowing both views share the same fixed point.

</details>

<details>

<summary>Q23. From $\sqrt{\text{KL}}$ and the inverted-U gold curve, derive a "safe KL budget" for reward overoptimization.</summary>

Gao 2023's BoN gold reward fit: $R_g(d) = d(\alpha_g - \gamma_g d)$, $d = \sqrt{\text{KL}}$.

Find peak: $dR_g/dd = \alpha_g - 2\gamma_g d = 0$ → $d_\text{peak} = \alpha_g / (2\gamma_g)$.

Corresponding KL distance: $\text{KL}_\text{peak} = d_\text{peak}^2 = \alpha_g^2 / (4\gamma_g^2)$.

**Safe budget**: stop before peak. Typically pick $\text{KL}_\text{stop} = 0.5 \cdot \text{KL}_\text{peak}$ (50% safety margin).

**How to estimate $\alpha_g, \gamma_g$?** Need gold reward signal (human / strong RM ensemble). Run a small pilot at a few KL points and fit $R_g(d)$.

**Larger RM, peak further right**: Gao 2023's scaling law shows $\alpha_g, \gamma_g$ depend on RM size (fit coefficients themselves depend on RM scale and data); larger RM has $\text{KL}_\text{peak}$ further right, with a higher $R_g(\text{peak})$ as well. This is why a larger RM both "rewards the budget" and reaches higher gold.

**PPO slightly more complex than BoN**: $R_g(d) = d(\alpha_g - \beta_g \log d)$; the log term gives a distinct (RL-specific) shape vs BoN.

Application: in practice, cap measured KL at $\text{KL}_\text{peak} \cdot 0.5$ for early stop, i.e. "$\sqrt{\text{KL}}$ < $d_\text{peak}/2$".

Missing the derivative; or not knowing the RM-size-vs-peak scaling.

</details>

<details>

<summary>Q24. If you had to design an RLHF algorithm combining reverse + forward KL, how would you use them?</summary>

**Motivation** (under the standard convention: reverse = $\text{KL}(q\|p)$ mode-seeking; forward = $\text{KL}(p\|q)$ mass-covering):

- **Reverse KL** $\text{KL}(\pi_\theta\|\pi_\text{ref})$ = mode-seeking on $\pi_\theta$ → pick reward-high modes inside high-density regions of ref (matches the rule). **RLHF's default**.
- **Forward KL** $\text{KL}(\pi_\text{ref}\|\pi_\theta)$ = mass-covering on $\pi_\theta$ → make $\pi_\theta$ cover all of ref's possible outputs (prevents mode collapse).

**Problem**: forward KL requires sampling from $\pi_\text{ref}$, which is engineering-meaningless (we are training $\pi_\theta$, not $\pi_\text{ref}$).

**Workaround proposals (theoretical)**:

1. **JSD-style combo**: use $\text{JS}(\pi_\theta\|\pi_\text{ref}) = \tfrac{1}{2}\text{KL}(\pi_\theta\|m) + \tfrac{1}{2}\text{KL}(\pi_\text{ref}\|m)$, $m = (\pi_\theta + \pi_\text{ref})/2$. Symmetric, bounded, captures both mode-seeking and mass-covering. Problem: $m$ has no closed form (mixture distribution), gradient computation is complex.
2. **Importance sampling**: sample from $\pi_\theta$ but reweight to $\pi_\text{ref}$ expectation: $\mathbb{E}_{\pi_\theta}[(\pi_\text{ref}/\pi_\theta)\log(\pi_\text{ref}/\pi_\theta)] = \text{KL}(\pi_\text{ref}\|\pi_\theta)$. Problem: tails have very large $\pi_\text{ref}/\pi_\theta$, variance explodes.
3. **Symmetric KL**: $\text{KL}_\text{sym} = \tfrac{1}{2}(\text{KL}(\pi_\theta\|\pi_\text{ref}) + \text{KL}(\pi_\text{ref}\|\pi_\theta))$, symmetric but still needs the forward-side estimate.
4. **Hybrid penalty**: early training mostly reverse KL (drive policy towards modes), later add small forward-KL term estimated via importance sampling to prevent mode collapse.

**Practical industrial approach (simpler)**:

- Use entropy bonus to replace forward KL's "mass coverage" goal — entropy doesn't need ref sampling.
- Train an ensemble of policies, each mode-seeks a different mode, preserving diversity overall.
- Combine BoN inference + DPO/PPO training: mode-seek during training, source diversity from BoN sampling at inference.

**Summary**: forward KL is "theoretically attractive but engineering-difficult" in RLHF; mainstream approaches use entropy / multiple policies as substitutes.

Just saying "add both" — too naive, must give engineering scheme; not knowing forward KL's sampling barrier.

</details>

<details>

<summary>Q25. How might next-generation RLHF algorithms improve KL regularization? Give 3 directions + tradeoffs.</summary>

**Direction 1: Adaptive KL estimator per token**

- Pick k1 vs k3 per token: small-KL tokens use k1 (unbiased + simple), large-KL tokens use k3 (prevent variance explosion).
- Tradeoff: implementation complexity + hard-to-determine per-token threshold.

**Direction 2: Per-task β controller**

- Different sub-tasks (math / code / dialogue / safety) use different β, dispatched online by a task classifier.
- Tradeoff: requires accurate task labels; cross-task β interference needs study.

**Direction 3: Adversarial KL**

- Don't fix $\pi_\text{ref}$ — train it online (e.g. self-rewarding LM, iterative DPO), but freeze $\pi_\text{ref}$ within each time window.
- Tradeoff: potentially unstable (GAN-like); needs fresh human labels to prevent drift.

**Direction 4: Distributional reward + W2 KL**

- Model reward as a distribution $p(r|x,y)$, replace KL with the Wasserstein-2 metric (geometric).
- Tradeoff: W2 computation is complex, needs sliced approximation; theoretical convergence unclear.

**Direction 5: Hierarchical KL**

- Sentence-level KL + token-level KL combined: sentence-level controls total KL budget, token-level provides fine-grained anchoring.
- Analogy: PRM (process reward) vs ORM (outcome reward).
- Tradeoff: sentence boundaries are not explicit during generation, requires extra annotation or heuristics.

**Direction 6: KL-free but trust-region substitute**

- Drop KL entirely, use PPO clip's trust region for constraint ("clip is enough").
- Already explored in IRL / SimPO.
- Tradeoff: loses the closed-form RL mathematical foundation (no $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$).

Only listing without tradeoffs; not knowing SimPO / IPO have already partly explored the KL-free route.

</details>

## §A Appendix: reference list

Grouped by chapter. This section is a draft; **arXiv IDs and exact venues are not all verified online** — proofreading should re-check via `/arxiv` or codex web_search.

**KL fundamentals and estimators**

- Schulman 2020 blog *Approximating KL Divergence* `http://joschu.net/blog/kl-approx.html` (the canonical source for k1 / k2 / k3 estimators)
- Endres & Schindelin 2003 IEEE TIT 49(7) *A New Metric for Probability Distributions* (proof that $\sqrt{\text{JS}}$ is a metric)
- Pinsker 1964 (original Pinsker inequality $\text{TV} \le \sqrt{\text{KL}/2}$)

**RLHF / PPO**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms* (PPO-Clip + PPO-Penalty adaptive β)
- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, per-token KL in reward)
- Bai et al. 2022 Anthropic arXiv 2204.05862 *Training a Helpful and Harmless Assistant with RLHF* (adaptive β controller)

**DPO family**

- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (closed form + inversion + BT)
- Azar et al. 2024 AISTATS *A General Theoretical Paradigm to Understand Learning from Human Preferences* (IPO, prevents unbounded reward under deterministic preference)
- Ethayarajh et al. 2024 ICML *KTO: Model Alignment as Prospect Theoretic Optimization* (reference point replacing KL)
- Meng et al. 2024 NeurIPS *SimPO: Simple Preference Optimization with a Reference-Free Reward* (drops ref, length-norm + margin)
- Hong et al. 2024 EMNLP *ORPO: Monolithic Preference Optimization without Reference Model* (odds-ratio single-stage)

**GRPO family / RL with k3 KL**

- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (first systematic use of GRPO with k3 + group-relative advantage)
- DeepSeek-AI 2025 arXiv 2501.12948 *DeepSeek-R1* (GRPO + rule-based reward + emergent CoT)
- Yu et al. 2025 ByteDance arXiv 2503.14476 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (clip-higher + dynamic sampling + token-level loss + small β)

**Reward Overoptimization**

- Gao, Schulman, Hilton 2023 ICML *Scaling Laws for Reward Model Overoptimization* (KL distance vs gold reward inverted-U + fitted form)
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*
- Eisenstein et al. 2024 COLM (arXiv 2312.09244, 2023) *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking*

**Critic-free RL (related to KL placement)**

- Ahmadian et al. 2024 ACL *Back to Basics: Revisiting REINFORCE Style Optimization* (RLOO, leave-one-out baseline, KL in reward)
- Li et al. 2024 ICML *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method*

**Two 2024-2026 systematic analyses of KL-in-RLHF (arXiv metadata verified)**

- Kezhao Liu et al., *Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization*, arXiv 2510.01555 (2025-10-02). Systematically distinguishes KL value estimation vs gradient optimization; shows k3-as-loss is a biased first-order approximation; recommends (P1) k1 in reward or (P2) k2 as loss; off-policy needs IS correction. §3.6 of this tutorial is based on this paper.
- Vedant Shah et al., *A Comedy of Estimators: On KL Regularization in RL Training of LLMs*, arXiv 2512.21852 (2025-12-26, v3 2026-03-18). Empirical comparison of k1/k2/k3 across RL algorithms + placement combinations, focusing on estimator bias / gradient bias / placement-effect; conclusion: no universally best estimator.

**Zhihu Chinese resources**

- `https://zhuanlan.zhihu.com/p/1979720260128118305` — *KL deep dive: Forward KL, Reverse KL, KL estimation and applications* (verified, topic consistent with RLHF KL estimators)
- `https://zhuanlan.zhihu.com/p/1892008158626546312` — [needs-verify URL accessibility] topic appears to cover k2-loss vs k3-loss / GRPO off-policy / clip_std; replace if the link is broken at verification time.

**Closely related internal tutorials**

- `docs/tutorials/rlhf_dpo_grpo_ppo_tutorial.md` — RLHF / DPO / GRPO / PPO overview (includes BT + closed form + DPO derivation)
- `docs/tutorials/reasoning_models_tutorial.md` — RL training details of reasoning models
- `docs/tutorials/agentic_rl_tutorial.md` — token mask + KL discussion in the agentic setting
