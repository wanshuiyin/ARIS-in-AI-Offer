## §0 TL;DR Cheat Sheet

> 💡 **Optimizers / LR Schedules in 9 lines** — the interview core on one page (derivations in §1–§11 below).

1. **Why not just plain SGD**: real loss landscapes are **ill-conditioned** (Hessian condition number $\kappa=\lambda_{\max}/\lambda_{\min}$ huge), so plain gradient descent **oscillates** on steep directions and **crawls** on shallow ones. Two cures: **momentum** (accumulate along consistent directions, cancel oscillations) + **per-parameter adaptive step** (diagonal preconditioning, evening out per-coordinate curvature mismatch). The optimizer lineage is "SGD → +momentum → +adaptive → Adam (both)".

2. **SGD / Momentum / Nesterov**: plain SGD $\theta_t=\theta_{t-1}-\eta g_t$; heavy-ball momentum $v_t=\mu v_{t-1}+g_t,\ \theta_t=\theta_{t-1}-\eta v_t$ (damps oscillation, accelerates consistent directions); Nesterov evaluates the gradient at the **look-ahead point** $\theta-\eta\mu v$ (corrects overshoot). Momentum $\approx$ an **exponential moving average** of gradients with window $\sim 1/(1-\mu)$.

3. **AdaGrad → RMSProp → Adam**: AdaGrad's per-coordinate step $\propto 1/\sqrt{\sum g^2}$ (**accumulates all history** → LR monotonically → 0, "can't learn" on long training); RMSProp replaces the sum with an **EMA** (fixes the monotonic decay); Adam = RMSProp's second-moment EMA + **first-moment momentum** + **bias correction**.

4. **Adam bias correction**: $m,v$ start at 0, biased early (toward 0); dividing by $1-\beta^t$ de-biases. Since $\beta_2$ (0.999) is closer to 1 than $\beta_1$ (0.9), $v$ is biased more → **without correction the early step is too LARGE** (with default $\beta$ the first step $\approx\frac{1-\beta_1}{\sqrt{1-\beta_2}}=3.16$× $\eta$), correction returns the first step to $\eta$. Note $\epsilon$ is **outside** the sqrt (PyTorch).

5. **AdamW decoupled weight decay**: **Adam's L2 regularization $\ne$ weight decay**. L2 adds $\lambda\theta$ to the **gradient** → it gets scaled by $1/\sqrt{\hat v}$ → params with large historical gradients get decayed less (decoupling broken). AdamW (Loshchilov & Hutter) **decouples**: subtract $\eta\lambda\theta$ directly from the weights, outside the Adam step. This is the modern default; the two coincide for SGD but not for Adam.

6. **Frontier optimizers**: **Muon** (**orthogonalize** the momentum of 2D weights via Newton-Schulz iterations, hidden 2D layers only, used by Kimi-K2), **Lion** (**sign momentum**, single state → half Adam's memory), **Shampoo** (Kronecker-factored **full-matrix preconditioner**), **SOAP** (run Adam in Shampoo's eigenbasis), **Adafactor** (factored second moment → sublinear memory), **LAMB** (layer-wise adaptive, huge batch), **Sophia** (light second-order, diagonal Hessian).

7. **LR schedule**: **warmup** (early $\hat v$ has high variance + large-batch/Post-LN instability → ramp up linearly); **cosine** (smooth decay to ~0, + warm restarts, **needs the total step count up front**); **inverse-sqrt / Noam** (the original Transformer schedule); **WSD** (warmup-stable-decay, a constant phase + a short final decay → **no need to pre-commit the total step count**, supports continual training and mid-run checkpoints); **one-cycle** (super-convergence).

8. **Weight decay + LR-batch scaling + no-decay groups**: wd does regularization (recently also a "wd $\approx$ effective-LR controller" view); the **linear scaling rule** (batch ×$k$ → LR ×$k$, SGD; Adam commonly $\sqrt k$); **LayerNorm/RMSNorm gains, biases, embeddings get NO wd** (split into decay/no-decay groups).

9. **Gradient clipping + LLM hyperparameters**: clip by **global norm** (preserves direction) vs by value (changes direction), to prevent gradient explosion / loss spikes; LLM pretraining defaults to $\beta_2=\mathbf{0.95}$ (**not** 0.999 — long memory is too sluggish for spikes), $\beta_1=0.9$, $\epsilon=\text{1e-8}$, wd $=0.1$, grad-clip $=1.0$.

## §1 Why we need optimizers beyond SGD

**Plain stochastic gradient descent (SGD)'s fundamental difficulty is that real loss landscapes are almost always "ill-conditioned."** Expand the loss to second order near a minimum; the curvature is described by the Hessian $H$. Define the **condition number** $\kappa=\lambda_{\max}/\lambda_{\min}$ (ratio of $H$'s largest to smallest eigenvalue). Large $\kappa$ means hugely different curvature in different directions — some steep as a canyon wall, some flat as the valley floor. For a quadratic, gradient descent's convergence rate is about $\big(\tfrac{\kappa-1}{\kappa+1}\big)$, slower the larger $\kappa$ is.

Intuitively, plain SGD **oscillates** in an ill-conditioned valley: with a single global $\eta$, to avoid diverging on the steep direction (large $\lambda$) $\eta$ must be tiny ($\eta\lt 2/\lambda_{\max}$); but that tiny $\eta$ then **crawls** on the shallow direction (small $\lambda$). The result is zig-zagging back and forth between the canyon walls while the valley-floor direction you actually want to progress along barely moves. Deep-net loss landscapes routinely have $\kappa\sim 10^4$ or more, so plain SGD is both slow and unstable.

> 💡 **Ill-conditioning = curvature mismatch across directions; one global LR can't please both ends**
> Large-curvature directions need small steps (else oscillation/divergence); small-curvature directions need large steps (else crawling). A single scalar $\eta$ can't satisfy both → this is the root motivation for **momentum** and **per-parameter adaptivity**.

The two complementary cures correspond to the two axes of the optimizer lineage:

- **Momentum**: accumulate a velocity of historical gradients. On oscillating directions the gradient flips sign repeatedly and **cancels** within the velocity; on consistent directions the gradient keeps the same sign and **accumulates, accelerating**. Equivalent to smoothing the update direction, improving GD's convergence constant from $\kappa$ to $\sqrt\kappa$ scale ($\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$). **Note**: $\kappa\to\sqrt\kappa$ is the classic acceleration result for tuned heavy-ball/Nesterov on a **strongly-convex quadratic**; real deep nets have stochastic-gradient noise, non-convexity, and time-varying curvature, so it's an **intuition, not a guarantee**.
- **Per-parameter adaptive step (adaptive / per-parameter LR)**: scale each coordinate's step by its own gradient history — coordinates with persistently large gradients take small steps, persistently small ones take large steps. This acts as a **diagonal preconditioner**, evening out the per-coordinate effective curvature and directly easing the large-$\kappa$ disease.

**The lineage in one line**: SGD (first-order gradient only) → + momentum (add a first-moment EMA) → + adaptivity (add a second-moment EMA) → **Adam = momentum + adaptivity + bias correction**. §2–§5 follow this line; §6 is the frontier (matrix preconditioning / sign / memory-saving); §7–§9 are the accompanying LR schedules, weight decay, and clipping.

> ⚠️ **"adaptive = diagonal preconditioner", not true second-order**
> Adam-like methods only scale each coordinate (diagonal), missing cross-coordinate coupling (off-diagonal curvature). What truly approaches full-matrix preconditioning is Shampoo/SOAP (§6). Calling Adam "approximate Newton" is inaccurate — it's only a **diagonal** preconditioner, and it uses the gradient's second moment, not the Hessian.

## §2 SGD / Momentum / Nesterov

### 2.1　Plain SGD and heavy-ball momentum

Plain SGD looks only at the current mini-batch gradient $g_t$ each step:

$$\theta_t = \theta_{t-1} - \eta\, g_t.$$

**Heavy-ball momentum** (Polyak, 1964) introduces a velocity buffer $v$, giving the update "inertia":

$$\boxed{\;v_t = \mu\, v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\, v_t\;}\qquad (\mu\ \text{typically }0.9).$$

(This is PyTorch's `SGD(momentum=μ)` form: $v$ starts at 0, the first step gives $v_1=g_1$, no $(1-\mu)$ damping.) Why it helps:

- **Oscillating directions** (steep walls): adjacent-step gradients flip sign repeatedly, **partially cancelling** in the exponentially-weighted sum $v_t=\mu v_{t-1}+g_t$ → the zig-zag is suppressed.
- **Consistent directions** (valley floor): adjacent-step gradients share a sign and keep accumulating → velocity grows, **accelerating forward**.

### 2.2　Nesterov accelerated gradient (NAG)

Heavy-ball evaluates the gradient at the **current point** $\theta_{t-1}$. **Nesterov accelerated gradient** instead evaluates at the **look-ahead point** "where the momentum will carry you":

$$v_t = \mu\, v_{t-1} + g\big(\theta_{t-1} - \eta\mu\, v_{t-1}\big), \qquad \theta_t = \theta_{t-1} - \eta\, v_t.$$

Intuition: "peek at where momentum is about to push me, then compute the gradient there" — if you're about to overshoot, the look-ahead gradient gives an early correction, so NAG has a better convergence constant on convex problems. Sutskever et al. (ICML 2013) showed empirically: **carefully designed (Nesterov-style) momentum is crucial for training deep nets**; good momentum + initialization can train deep nets even without an adaptive optimizer.

### 2.3　Momentum ≈ an exponential moving average of gradients

Unrolling the recurrence (with $v_0=0$):

$$v_t = \sum_{i=1}^{t} \mu^{\,t-i}\, g_i.$$

This is a **geometric (exponential) weighted sum** of historical gradients — more recent gradients weigh more, older ones decay as $\mu^{t-i}$. The effective averaging window is about $1/(1-\mu)$ ($\mu=0.9$ → ~10 steps, $\mu=0.99$ → ~100 steps). So momentum does something simple: **average noisy instantaneous gradients into a smoother, more trustworthy descent direction**, amplifying the consistent trend and washing out noise/oscillation.

> 💡 **Momentum's two faces**
> Optimization view: improves the convergence constant for ill-conditioned problems ($\kappa\to\sqrt\kappa$). Statistical view: an EMA that denoises the gradient. Both explain why it's nearly a free default — add $\mu=0.9$ and you get faster + more stable.

## §3 AdaGrad → RMSProp → Adam (the adaptive lineage)

### 3.1　AdaGrad: per-coordinate adaptive, but "learns to death"

AdaGrad (Duchi, Hazan & Singer, JMLR 2011) gives each coordinate an **independent** step size that adapts to its gradient history. Maintain the per-coordinate **cumulative sum of squares** $G_t$:

$$G_t = G_{t-1} + g_t^2 \ (\text{elementwise}), \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t}+\epsilon}\odot g_t.$$

Effect: coordinates with persistently large gradients have large $G$ → small step; sparse/rare features have small $G$ → large step. Very effective on convex optimization and sparse features (early NLP, recommendation). **Fatal flaw**: $G_t$ is a **monotonically increasing** cumulative sum, so the denominator $\sqrt{G_t}$ only grows → the effective learning rate monotonically tends to 0. On a deep net's long training run the step shrinks to nearly 0 partway through and it **can't learn** anymore (premature stagnation).

### 3.2　RMSProp: replace the cumulative sum with an EMA

RMSProp (Hinton, Coursera *Neural Networks for ML*, Lecture 6e, 2012, never formally published) gets straight to the point: the problem is "accumulating all history." Replace the cumulative sum with an **exponential moving average**, so old gradients are forgotten:

$$E[g^2]_t = \rho\, E[g^2]_{t-1} + (1-\rho)\, g_t^2, \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t}+\epsilon}\odot g_t.$$

The EMA denominator no longer grows unboundedly (it tracks the "recent" gradient-square scale), so the **effective LR doesn't decay to 0** — AdaGrad's deadlock is undone. This is the direct precursor of Adam's second-moment term.

### 3.3　Adam = RMSProp + momentum + bias correction

Adam (Kingma & Ba, 2014) merges the two axes: RMSProp's **second-moment EMA** (adaptive step) + heavy-ball's **first-moment EMA** (momentum) + a crucial **bias correction** (§4). One-line mnemonic:

$$\textbf{Adam} = \underbrace{\text{RMSProp}}_{\text{second-moment EMA}} + \underbrace{\text{momentum}}_{\text{first-moment EMA}} + \underbrace{\text{bias correction}}_{\text{remove early bias}}.$$

> ✅ **Each step patches the previous one's hole**
> AdaGrad gave "per-coordinate adaptivity" but dies → RMSProp revives it with an EMA → Adam stacks momentum for denoising + bias correction to stabilize early steps. Telling this "who fixed whom" chain in an interview beats reciting Adam's formula.

## §4 Adam in depth

### 4.1　The full update

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t \qquad (\text{first moment / momentum, EMA}),$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \qquad (\text{second moment / adaptive, EMA}),$$

$$\hat m_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \qquad (\textbf{bias correction}),$$

$$\boxed{\;\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\;}\qquad (\epsilon\ \text{outside the sqrt, PyTorch}).$$

Defaults $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$. $m_0=v_0=0$.

### 4.2　Why bias correction is needed

$m,v$ both initialize at 0. Early on they are **biased** estimates of the true moments — biased toward 0, and the closer $\beta$ is to 1 the longer the bias lasts. Taking the expectation for stationary gradients: $\mathbb{E}[m_t]=(1-\beta_1^t)\,\mathbb{E}[g]$, so $\hat m_t=m_t/(1-\beta_1^t)$ exactly de-biases; $\hat v_t$ likewise.

The key point, also frequently probed deeply: **without correction, the early effective step is not too small, but too large.** Look at the first step ($t=1$):

$$\frac{m_1}{\sqrt{v_1}} = \frac{(1-\beta_1)\,g_1}{\sqrt{(1-\beta_2)}\,\lvert g_1\rvert} = \frac{1-\beta_1}{\sqrt{1-\beta_2}}\,\mathrm{sign}(g_1).$$

Plugging in defaults: $\frac{1-0.9}{\sqrt{1-0.999}}=\frac{0.1}{0.0316}\approx 3.16$. That is, **the uncorrected first step is about 3.16× the target $\eta$** — because $v$ (with $\beta_2$ closer to 1) is pushed toward 0 harder than $m$, the denominator $\sqrt{v}$ is too small, which inflates the step instead. Bias correction restores $\hat m_1=g_1,\ \hat v_1=g_1^2$, returning the first step exactly to $\eta\cdot\mathrm{sign}(g_1)$, of magnitude $\eta$. (§A's [d] verifies this 3.16 numerically.)

> ⚠️ **Don't blurt out "without correction the step is too small"**
> The naive intuition only watches the numerator $m$ biased toward 0 → assumes the step is too small, **missing that the denominator $v$ is biased harder**. For standard $\beta_2=0.999$, the net effect is an early step that's too **large** ($\approx 3.16\eta$), which can cause early instability/divergence — another reason early training pairs with warmup (§7). Getting the direction backwards is a top-lab favorite to catch.

> 💡 **$\epsilon$ outside the sqrt (PyTorch convention)**
> Adam's denominator is $\sqrt{\hat v_t}+\epsilon$, $\epsilon$ **outside** the sqrt — it both prevents division by zero and **caps the step at** $\eta/\epsilon$ (no blow-up to an infinite step when the gradient is tiny). Some implementations (e.g. Adafactor) put it **inside**, $\sqrt{\hat v_t+\epsilon}$, with slightly different numerics. Too small an $\epsilon$ in fp16/bf16 makes the denominator $\approx 0$ and easily overflows.

## §5 AdamW (the most-probed section)

### 5.1　Adam's L2 regularization $\ne$ weight decay

The classic **L2 regularization** approach: add $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss, equivalent to adding $\lambda\theta$ to the gradient:

$$g_t' = g_t + \lambda\,\theta_{t-1}.$$

For plain SGD this is exactly weight decay: $\theta_t=\theta_{t-1}-\eta(g_t+\lambda\theta_{t-1})=(1-\eta\lambda)\theta_{t-1}-\eta g_t$ — each step shrinks the weight by $(1-\eta\lambda)$. **But for Adam, disaster strikes**: this $\lambda\theta$ is stuffed into $g'$, so it flows into $m,v$ and ends up scaled by $1/(\sqrt{\hat v}+\epsilon)$:

$$\theta_t = \theta_{t-1} - \eta\,\frac{\widehat{m'_t}}{\sqrt{\hat v_t}+\epsilon},\qquad m'\ \text{contains the }\lambda\theta\ \text{contribution}.$$

Consequence: **params with large historical gradients ($\hat v$ large) get LESS effective decay**, params with small gradients get more — the decay strength is distorted by $\hat v$, completely divorced from the regularizer's intent of "applying a uniform shrinkage to all weights." This is "L2 and weight decay are no longer equivalent under an adaptive optimizer."

### 5.2　AdamW: decoupled weight decay

AdamW (Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 1711.05101) fixes it minimally — **take weight decay out of the gradient, bypass the adaptive denominator, and apply it directly to the weights**:

$$\boxed{\;\theta_t = \theta_{t-1} - \eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Big)\;}$$

i.e. compute the Adam step on the raw gradient $g$ (**without** $\lambda\theta$), then separately subtract an $\eta\lambda\theta_{t-1}$. Equivalently $\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ — a clean, **$\hat v$-independent** uniform shrinkage (PyTorch's `AdamW` multiplies the decay by lr).

Why it's usually better (more controllable): after decoupling, **all params receive the same relative shrinkage $\lambda$**, the regularization strength is no longer distorted by each param's gradient history, more faithful to "controlling the weight norm"; Loshchilov & Hutter showed empirically that AdamW is **often** stably better than Adam+L2 across tasks, and the optimal LR and wd are easier to tune decoupled (no need to absolutize it to "more generalizing on all tasks," but it is the more controllable, more common default). **Why SGD makes them coincide**: SGD has no $1/\sqrt{\hat v}$ adaptive scaling layer, so $\lambda\theta$ added to the gradient is used as-is, which is exactly uniform shrinkage; once there's an adaptive denominator, adding to the gradient (L2) and shrinking directly (decay) part ways.

> ✅ **The modern default IS AdamW**
> Transformer/LLM training almost always uses AdamW (not Adam). Asked "the difference between Adam and AdamW," the standard answer is "**L2 goes through the $1/\sqrt{\hat v}$ scaling, decoupled decay doesn't**; they coincide for SGD, differ for Adam." Answering only "AdamW adds weight decay" misses the point (Adam can also take a weight_decay, but that's L2).

> ⚠️ **In PyTorch, `Adam(weight_decay=λ)` is L2, not AdamW**
> `torch.optim.Adam(weight_decay=λ)` does **coupled L2** (adds $\lambda\theta$ to the gradient); for decoupled decay you must use `torch.optim.AdamW`. The two produce **different** updates at the same $\lambda$ (§A's [c] shows $\lVert\Delta\rVert\gt 0$ numerically). Getting this API wrong means you never actually used AdamW.

## §6 Frontier optimizers

Remember just **one core idea** for each — being able to state "which part of Adam it changes" in an interview is enough.

### 6.1　Muon: orthogonalize the momentum

**Muon** (Keller Jordan, 2024 tech blog) targets **2D weight matrices**: first compute the update matrix $M$ from momentum, then use a few **Newton-Schulz iterations** to **orthogonalize** it into $O\approx\mathrm{NewtonSchulz}(M)$ (mathematically approximating $UV^\top$ of $M=U\Sigma V^\top$, i.e. the nearest semi-orthogonal matrix, replacing an expensive SVD with iteration), and update the weight with $O$. Intuition: orthogonalization makes the update push **equally on all singular directions**, not dominated by a few large singular values → more balanced updates, faster effective learning. **Only for hidden-layer 2D matrices**; embeddings, the output head, norm gains, biases still go through AdamW (orthogonalization is meaningless for them). Moonshot's Kimi-K2 trains with Muon; for scalability see *Muon is Scalable for LLM Training* (2502.16982).

> 💡 **The original Muon is mostly a blog, with no formal arXiv**
> Muon first appeared as Keller Jordan's tech-blog writeup (no standalone arXiv paper); the one with a formal arXiv is Moonshot's scaling work (2502.16982). Distinguish them when citing — don't invent an arXiv id for the original Muon.

### 6.2　Lion: sign momentum, single state

**Lion** (Chen et al., *Symbolic Discovery of Optimization Algorithms*, 2302.06675) was discovered automatically by **symbolic program search**. The update takes the **sign** of the momentum: $u_t=\mathrm{sign}(\beta_1 m_{t-1}+(1-\beta_1)g_t)$, $\theta_t=\theta_{t-1}-\eta(u_t+\lambda\theta_{t-1})$, then updates the momentum $m_t=\beta_2 m_{t-1}+(1-\beta_2)g_t$. It **stores only 1 state (momentum)** → optimizer memory is **half** Adam's (which stores both $m,v$). The sign makes every coordinate take an equal-magnitude step; usually paired with a smaller LR and a larger wd.

### 6.3　Shampoo / SOAP: full-matrix preconditioning

- **Shampoo** (Gupta, Koren & Singer, 1802.09568): goes beyond Adam's **diagonal** preconditioning to **full-matrix / Kronecker-factored** preconditioning. For a weight matrix it maintains left and right preconditioners $L=\sum GG^\top$, $R=\sum G^\top G$, and updates with $L^{-1/4} G\, R^{-1/4}$ — approximating full-matrix AdaGrad with a Kronecker product, capturing cross-coordinate coupling. Converges faster but each step is expensive (computing matrix inverse-roots).
- **SOAP** (Vyas et al., 2409.11321): runs Adam in **Shampoo's (slowly-changing) eigenbasis** — rotate the gradient into Shampoo's preconditioner's eigenspace, do standard Adam there, rotate back. It connects Shampoo with Adam/Adafactor, more stable with fewer hyperparameters.

### 6.4　Adafactor / LAMB / Sophia

- **Adafactor** (Shazeer & Stern, 1804.04235): **factorize** the second-moment matrix $v\in\mathbb{R}^{m\times n}$ into row and column vectors (rank-1 reconstruction $v\approx rc^\top$) → memory drops from $O(mn)$ to $O(m+n)$ (**sublinear**). T5 used it to train a large model on limited memory.
- **LAMB** (You et al., 1904.00962): on top of Adam's update direction, apply a further **layer-wise adaptation** — scale by each layer's $\lVert\theta\rVert/\lVert\text{update}\rVert$ (trust ratio) (the Adam version of LARS). Stabilizes **very large batch** training (BERT in 76 minutes).
- **Sophia** (Liu et al., 2305.14342): **light second-order**. Uses a diagonal Hessian estimate (Gauss-Newton-Bartlett or Hutchinson) for preconditioning + clipping the update, updating the Hessian only every $k$ steps → converges faster than AdamW with the second-order cost amortized.

## §7 Learning-rate schedules

### 7.1　Warmup: why start slow

**Warmup** linearly ramps the LR from 0 (or very small) **up to the peak** over the first few hundred ~ few thousand steps, then hands off to the subsequent decay. Two motivations:

1. **Adam's early variance is high**: $\hat v$ is estimated from very few samples in the first steps, with **high variance**, making the adaptive step jittery or even too large (the 3.16× effect of §4.2). Use a small LR first to let $m,v$'s statistics "warm up" to reliability, then scale up the step.
2. **Large-batch / Post-LN instability**: large-batch early gradients are large in scale; Post-LN Transformers have naturally large top-layer gradients (see the normalization tutorial, §5) — hitting them with a large LR right away easily diverges. Warmup suppresses early instability.

### 7.2　Cosine annealing (+ warm restarts)

Cosine (Loshchilov & Hutter, *SGDR*, 1608.03983) decays the LR **smoothly with a cosine** from the peak to $\sim\eta_{\min}$:

$$\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\Big(1+\cos\big(\pi\, t/T\big)\Big).$$

Early large LR explores quickly, late small LR refines convergence (less noise, easier to settle into a flatter minimum). SGDR also adds **warm restarts**: periodically pull the LR back to the peak to help escape local solutions. **Cost: you must fix the total step count $T$ in advance** ($\cos$ needs to know the endpoint) — exactly the pain point WSD solves.

### 7.3　Inverse-sqrt / Noam (the original Transformer schedule)

The schedule the original Transformer (Vaswani et al., 1706.03762) used, first a **linear warmup** then a **$1/\sqrt t$ decay**:

$$\eta_t = d_{\text{model}}^{-1/2}\cdot\min\!\Big(t^{-1/2},\ t\cdot t_{\text{warmup}}^{-3/2}\Big).$$

For $t\lt t_{\text{warmup}}$ the second term applies (linear rise), after that the first (decaying as $t^{-1/2}$). No need to pre-fix the total step count; it was the standard for early machine translation / language models.

### 7.4　WSD: Warmup-Stable-Decay

WSD (MiniCPM, Hu et al., 2404.06395) splits the schedule into three phases: **warmup → a long constant phase (stable, at the peak LR) → a short final decay**. The biggest selling point: **no need to pre-commit the total step count** — the constant phase can extend indefinitely, and you append a short decay only when you want to stop (loss drops sharply during the decay phase to a good level). Benefits:

- supports **continual training / adding data** (the constant phase can resume anytime);
- each decay yields a **usable checkpoint**, so you don't train a separate cosine run for each target step count;
- makes it easy to explore "how long to train" under a fixed compute budget, instead of betting on $T$ once and for all.

### 7.5　One-cycle / super-convergence

One-cycle (Smith, *Super-Convergence*, 1708.07120): the whole training is **one cycle** — the LR rises to a **very large** peak then falls to well below the initial value, while the momentum cycles inversely in sync (low momentum when LR is high). In some settings it achieves "super-convergence" (training very fast with a very large LR). Like cosine and WSD it's in the "rise then fall" family, distinguished by the single cycle + the very large peak LR.

> 🎯 **Picking a schedule in one line**
> Fixed budget, chasing peak quality → **cosine** (but pre-fix the step count); uncertain how long to train / want continual training / want mid-run checkpoints → **WSD**; classic Transformer reproduction → **Noam**; want to train super-fast with a very large LR → **one-cycle**. **All of these should be preceded by warmup.**

## §8 Weight decay + LR-batch scaling + no-decay groups

### 8.1　What weight decay does (incl. a modern view)

Classic view: weight decay is **regularization** — each step shrinks the weight toward 0, bounding the weight norm, limiting model complexity, resisting overfitting. **Modern view** (in nets with normalization layers): because LayerNorm/RMSNorm make the weights **scale-invariant** (multiply $W$ by a constant and the normalized output is unchanged), the weight's "absolute size" doesn't directly affect the function, so wd's real role is more like **tuning the effective learning rate / controlling the weight-norm equilibrium** — larger wd, smaller weight norm, relatively larger effective gradient step. This is an important observation in recent training-dynamics research (treating wd as an "effective-LR knob" rather than pure regularization).

### 8.2　Scaling LR with batch size

In large-batch training, LR must scale with the batch, else you waste compute or destabilize:

- **Linear scaling rule** (Goyal et al., 1706.02677): **batch ×$k$ → LR ×$k$** (for large-batch SGD), paired with warmup. Intuition: the mini-batch mean gradient's variance $\propto 1/B$, so enlarging the batch $k$× reduces the noise, letting you safely enlarge the step $k$× to keep "the effective update each sample contributes" roughly constant.
- **Adam's $\sqrt k$ scaling**: Adam's update is approximately normalized by $\sqrt{\hat v}$ (close to sign), so empirically scaling the batch by a **square root** (LR ×$\sqrt k$) is more stable than linear.
- Relatedly, **LARS** (You et al., 1708.03888) / **LAMB** (§6.4) use a per-layer trust ratio to achieve adaptive scaling under very large batches.

> ⚠️ **These are empirical rules, not theorems**
> Linear / $\sqrt{}$ scaling holds only within a certain batch range; beyond a certain point you enter the "diminishing returns" regime (past the critical batch size, more batch no longer speeds up linearly). When discussing scaling rules in an interview, add the "approximate / has a ceiling" caveat.

### 8.3　No-decay parameter groups

**Not all parameters should get weight decay.** Standard practice splits params into two groups:

- **with wd**: each linear / conv layer's weight matrix.
- **without wd (no-decay)**: **LayerNorm/RMSNorm gains $\gamma$, all biases, and (usually) embeddings**.

Reason: these are **scale / shift parameters**, and shrinking them toward 0 directly harms representation — pushing an LN gain toward 0 wipes out that layer's activation scale; a bias shouldn't be regularized at all (it only handles shifts). Codebases like GPT/LLaMA explicitly construct decay / no-decay param groups.

> 💡 **Connection to μP (not re-derived here)**
> μP (maximal update parametrization) makes the **optimal LR width-invariant**, so hyperparameters tuned at small width transfer zero-shot to a large model — another orthogonal thread on "how to pick the LR," details in the μP section of the normalization tutorial (§9), not expanded here.

## §9 Gradient clipping + LLM hyperparameter details

### 9.1　Gradient clipping: global norm vs by value

Occasional **gradient explosions / loss spikes** during training (especially RNNs, long sequences, large LR) can blow the weights away in one step or even produce NaN. Gradient clipping (Pascanu, Mikolov & Bengio, 1211.5063) puts a gate on the gradient:

- **Clip by global norm (most common)**: concatenate all params' gradients into one big vector, compute its $\lVert g\rVert_2$; if it exceeds a threshold $c$, scale the whole thing $g\leftarrow g\cdot c/\lVert g\rVert_2$. **Preserves the direction**, only caps the magnitude.
- **Clip by value**: clamp each component to $[-v, v]$. This **changes the gradient direction** (different components are truncated differently), generally worse than global norm.

Why it stabilizes: occasional huge gradients are clamped within the threshold, the single-step update magnitude is bounded → one bad batch can't crash training. LLM pretraining almost always uses **global-norm clip with threshold 1.0**.

### 9.2　Adam hyperparameter defaults for LLM training

The Adam/AdamW recipe for LLM pretraining differs from the "textbook defaults" in a few key places:

| Hyperparameter | Textbook default | **LLM pretraining common** | Why |
| --- | --- | --- | --- |
| $\beta_1$ | 0.9 | 0.9 | momentum window ~10 steps, enough |
| $\beta_2$ | 0.999 | **0.95** (or 0.99) | 0.999's memory window ~1000 steps is **too long**, sluggish to loss spikes / non-stationarity, easily destabilized after a spike; 0.95 (window ~20 steps) reacts faster and more stably |
| $\epsilon$ | 1e-8 | 1e-8 (sometimes 1e-15) | prevents a zero denominator; too large weakens adaptivity |
| weight decay | 0 ~ 1e-2 | **0.1** | decoupled wd (AdamW), fairly strong |
| grad clip | none | **1.0** (global norm) | suppress loss spikes |

> ⚠️ **Carrying $\beta_2=0.999$ over to LLMs is a classic trap**
> 0.999 is fine on vision / small models; but in large-scale LLM pretraining the gradient statistics are non-stationary with occasional spikes, and 0.999's long memory makes $\hat v$ update too slowly to recover quickly after a loss spike, so the common practice is to drop to **$\beta_2=0.95$**. This is the standard scoring point for "how do you tune Adam for LLMs" in interviews.

## §10 Picking an optimizer + the memory bill

### 10.1　Optimizer-state memory

In mixed-precision training, each **trainable parameter** costs roughly: bf16 weight 2 B + bf16 gradient 2 B + fp32 master 4 B + **optimizer state**. The optimizer state is exactly the watershed between methods:

- **SGD+momentum**: 1 state (the momentum $v$).
- **Adam / AdamW**: **2 states ($m, v$)**, $=8$ B/param in fp32 — often the big chunk, the source of the "save memory" motivation.
- **Lion**: 1 state (momentum) → half of Adam.
- **Adafactor**: factorized $v$ → sublinear, approximately leaving only "param-level" cost.

The memory-saving routes this spawns:

- **8-bit Adam** (Dettmers et al., 2110.02861): **block-quantize** $m,v$ to 8-bit storage (dequantize on use), dropping optimizer-state memory to about $1/4$, with nearly no quality loss.
- **Adafactor**: factorized second moment, sublinear memory (T5).
- **Lion**: one fewer state, directly halved.

### 10.2　When to use which

> 🎯 **The "which optimizer" decision tree**
> - **Transformer / LLM fine-tuning or pretraining**: **AdamW** (safe default, $\beta_2=0.95$, wd 0.1, grad-clip 1.0, with warmup+cosine/WSD).
> - **Classic vision (ResNet, etc.)**: **SGD+momentum** often **generalizes better** and saves memory — much CV SOTA still uses it.
> - **Want extra training speedup / large scale**: **Muon** (2D hidden layers), **Shampoo/SOAP** (full-matrix preconditioning, if you'll pay the per-step compute).
> - **Memory-constrained**: **8-bit Adam** / **Adafactor** / **Lion**.
> - **Very large batch**: **LAMB / LARS** (per-layer trust ratio).

## §11 Engineering comparison + common misconceptions

### 11.1　Optimizer comparison table

| Optimizer | Optimizer state (× params) | Extra compute | Core idea | Typical use |
| --- | --- | --- | --- | --- |
| **SGD+Momentum** | 1× (momentum) | minimal | heavy-ball velocity EMA | classic vision, chasing generalization |
| **Adam / AdamW** | 2× ($m,v$) | low (elementwise) | momentum + diagonal adaptive (+ decoupled wd) | Transformer / LLM default |
| **Lion** | 1× (momentum) | low (take sign) | sign momentum, single state | memory-saving / large batch |
| **Muon** | 1× (momentum) + NS iters | medium (a few small matmuls) | orthogonalize the momentum (Newton-Schulz) | LLM hidden 2D layer speedup |
| **Shampoo** | preconditioners $L,R$ (large) | high (matrix inverse-roots) | Kronecker full-matrix preconditioning | large-scale training speedup |
| **Adafactor** | sublinear (factorized $v$) | low | row/column-factorized second moment | memory-constrained / T5 |

> 💡 **Reading the table**: across the memory axis (SGD/Lion/Muon 1 state, Adam 2 states, Adafactor sublinear, Shampoo heaviest), and the cost axis (diagonal adaptive cheap, full-matrix preconditioning expensive). There is no "optimal optimizer," only "the most suitable for the current model × memory × compute budget."

### 11.2　Common footguns

> ❌ **Footgun 1: treating Adam's L2 as AdamW's decoupled decay**
> `Adam(weight_decay=λ)` is coupled L2 (scaled by $1/\sqrt{\hat v}$), `AdamW` is decoupled decay. The two give **different** updates at the same $\lambda$ (§5, §A [c]).

> ❌ **Footgun 2: forgetting bias correction / getting its direction backwards**
> Dropping $1/(1-\beta^t)$ scrambles the early step; and without correction, at default $\beta$ the first step is too **large** ($\approx 3.16\eta$), not too small (§4.2).

> ❌ **Footgun 3: weight-decaying LayerNorm/RMSNorm gains and biases**
> These scale/shift params shouldn't be shrunk; use a no-decay group (§8.3).

> ❌ **Footgun 4: using $\beta_2=0.999$ for LLM training**
> The long memory is too sluggish for loss spikes and easily destabilizes; switch to **0.95** (§9.2).

> ❌ **Footgun 5: not scaling LR with batch**
> Scale by linear (SGD) / $\sqrt{}$ (Adam) and pair with warmup, else you waste compute or diverge (§8.2).

> ❌ **Footgun 6: using cosine without knowing it needs the total step count up front**
> Cosine needs the endpoint $T$; for mid-run continual training / uncertain step counts use **WSD** (§7.2, §7.4).

> ❌ **Footgun 7: putting Adam's $\epsilon$ inside the sqrt**
> PyTorch is $\sqrt{\hat v}+\epsilon$ ($\epsilon$ outside, also a step-size ceiling); getting the position backwards changes the numerics (§4.2).

## §12 25 high-frequency interview questions

Sorted into three tiers; click for answer points + easy traps. L2/L3 are top-lab deep water (AdamW decoupling, bias-correction direction, $\beta_2$ choice, scaling rules, Muon/Shampoo/Lion/Sophia, etc.).

### L1 must-know

<details>

<summary>Q1. What's the difference between plain SGD and SGD with momentum? What does momentum do?</summary>

- Plain SGD: $\theta\mathrel{-}=\eta g$, looks only at the current gradient each step
- Momentum: $v=\mu v+g,\ \theta\mathrel{-}=\eta v$, an accumulated (exponentially-weighted) velocity of historical gradients
- Consistent directions accumulate/accelerate, oscillating directions cancel by sign-flip; effective averaging window $\sim 1/(1-\mu)$

Calling momentum just "a larger learning rate," unable to say "accumulate along consistent directions, cancel on oscillating ones."

</details>

<details>

<summary>Q2. Write Adam's full update.</summary>

- $m=\beta_1 m+(1-\beta_1)g$ (first-moment EMA), $v=\beta_2 v+(1-\beta_2)g^2$ (second-moment EMA)
- Bias correction $\hat m=m/(1-\beta_1^t)$, $\hat v=v/(1-\beta_2^t)$
- $\theta\mathrel{-}=\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$, $\epsilon$ **outside** the sqrt (PyTorch)

Dropping bias correction; or putting $\epsilon$ inside the sqrt.

</details>

<details>

<summary>Q3. What is warmup for?</summary>

- Early in training **ramp the LR linearly** from 0/very-small to the peak (a few hundred ~ few thousand steps)
- Early Adam's $\hat v$ uses few samples, high variance → adaptive step unstable/too large; large-batch and Post-LN early gradients are large
- Warmup suppresses early instability, avoiding immediate divergence

Treating warmup as black magic, unable to state the "early high variance / large gradients" mechanism.

</details>

<details>

<summary>Q4. What is AdaGrad? Why does it "stop learning" on long training?</summary>

- Per-coordinate accumulation of **all** historical squared gradients $G=\sum g^2$, step $\propto\eta/\sqrt{G}$
- $G$ monotonically grows → effective LR monotonically → 0 → after long training the step is nearly 0, stagnation
- Good for convex / sparse features, bad for deep-net long training

Not knowing it's "accumulating all history" that decays the LR to 0.

</details>

<details>

<summary>Q5. What did RMSProp fix in AdaGrad?</summary>

- Replace the "cumulative sum" with an **EMA**: $E[g^2]=\rho E[g^2]+(1-\rho)g^2$
- The EMA forgets old gradients → the denominator no longer grows unboundedly → LR doesn't decay to 0
- This is the source of Adam's second-moment term

Saying only "RMSProp is faster," unable to state "EMA replaces the cumulative sum, fixing the LR decay."

</details>

<details>

<summary>Q6. Why use Adam instead of plain SGD?</summary>

- Per-parameter adaptive step (**diagonal preconditioning** by $\sqrt{\hat v}$) + momentum, more robust to ill-conditioning / mismatched per-coordinate scale
- Fast convergence, less LR-sensitive, nearly tuning-free, suits Transformers
- Cost: 2× optimizer-state memory, sometimes slightly worse generalization than SGD

Saying only "Adam converges fast," unable to answer "adaptive = diagonal preconditioner"; or calling it "approximate Newton" (it's only diagonal).

</details>

<details>

<summary>Q7. What do weight decay / L2 regularization intuitively do? Same for SGD and Adam?</summary>

- Each step shrinks the weight toward 0 → bounds the weight norm, controls complexity
- **SGD**: L2 and weight decay coincide ($\theta\mathrel{-}=\eta(g+\lambda\theta)=(1-\eta\lambda)\theta-\eta g$)
- **Adam**: the two are **not** equivalent (L2 is scaled by $1/\sqrt{\hat v}$), use AdamW, see Q11

Thinking L2 and weight decay are always the same thing (untrue for adaptive optimizers).

</details>

<details>

<summary>Q8. What is cosine annealing? Why decay the LR?</summary>

- $\eta_t=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi t/T))$, smoothly down from the peak to $\sim 0$
- Early large LR explores fast, late small LR refines convergence, reduces noise, helps settle into a flatter minimum
- SGDR adds warm restarts (periodically back to the peak)

Not knowing it requires **fixing the total step count $T$ in advance**; or unable to say "why decay."

</details>

<details>

<summary>Q9. The difference between Nesterov and heavy-ball momentum?</summary>

- heavy-ball: evaluate the gradient at the **current point** $\theta$, $v=\mu v+g(\theta)$
- Nesterov: evaluate the gradient at the **look-ahead point** $\theta-\eta\mu v$ "where momentum will carry you"
- The look-ahead foresees overshoot and corrects it early, a better convergence constant (Sutskever 2013 stressed its importance for deep nets)

Unable to state that NAG's gradient is evaluated at the look-ahead point, not the current point.

</details>

<details>

<summary>Q10. Why is momentum ≈ an EMA of gradients?</summary>

- Unrolling $v_t=\sum_{i\le t}\mu^{t-i}g_i$ — an exponentially-weighted (geometric) sum of historical gradients
- Equivalent to a moving average with window $\sim 1/(1-\mu)$ ($\mu=0.9\to\sim 10$ steps)
- So momentum is "averaging out noise, keeping the consistent trend"

Treating momentum only as an "inertia" metaphor, unable to write the exponentially-weighted sum.

</details>

### L2 advanced

<details>

<summary>Q11. AdamW vs Adam+L2: what exactly does decoupling decouple? (most frequent)</summary>

- Adam+L2: adds $\lambda\theta$ to the **gradient** → scaled by $m,v$'s $1/\sqrt{\hat v}$ → params with large historical gradients get decayed **less** (decoupling broken)
- AdamW: subtracts $\eta\lambda\theta$ from the weights **directly**, outside the Adam step (bypassing the adaptive denominator): $\theta\mathrel{-}=\eta(\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta)$
- After decoupling all params get the same shrinkage, generalizes better; coincide for SGD, differ for Adam

Saying "AdamW is just Adam plus weight decay," unable to state "L2 goes through the $\sqrt{\hat v}$ scaling" as the key difference.

</details>

<details>

<summary>Q12. Why does Adam need bias correction? What if you don't? Is the direction too large or too small?</summary>

- $m,v$ init at 0, biased toward 0 early, the closer $\beta$ to 1 the longer the bias
- Since $\beta_2$(0.999) is closer to 1 than $\beta_1$(0.9), $v$ is biased more → denominator $\sqrt{v}$ too small → without correction the early step is **too large** (default $\beta$ gives a first step $\approx\frac{1-\beta_1}{\sqrt{1-\beta_2}}=3.16$× $\eta$)
- Correction $\div(1-\beta^t)$ pulls the first step back to $\eta$

Blurting out "without correction the step is too small" — for standard $\beta_2=0.999$ it's exactly the opposite, too **large** (watching only the numerator, missing the denominator).

</details>

<details>

<summary>Q13. Cosine vs WSD, the trade-offs?</summary>

- Cosine: smoothly down to $\sim 0$, good quality, but **must fix the total step count $T$ in advance**, hard to do mid-run continual training / data addition
- WSD: warmup → long constant → short final decay; **no pre-committed step count**, the constant phase extends arbitrarily, each decay yields a usable checkpoint
- WSD suits continual training / uncertain budget; cosine suits a fixed-budget one-shot run

Not knowing cosine needs $T$ in advance, while WSD's selling point is exactly not needing to pre-commit.

</details>

<details>

<summary>Q14. Why does LLM pretraining use $\beta_2=0.95$ instead of 0.999?</summary>

- $\beta_2=0.999$'s effective memory window $\sim 1/(1-\beta_2)=1000$ steps, too long, sluggish to loss spikes
- LLM training gradients are non-stationary / spiky, 0.95 (window $\sim 20$ steps) reacts faster and more stably
- Pairing $\beta_1=0.9,\ \epsilon=\text{1e-8},\ \text{wd}=0.1,\ \text{grad-clip}=1.0$ is a common LLM recipe

Copying vision / small-model 0.999, hard to recover from a loss spike.

</details>

<details>

<summary>Q15. What is the linear scaling rule? Why does Adam commonly use $\sqrt{}$ scaling?</summary>

- Goyal: large-batch SGD, batch ×$k$ → LR ×$k$ (keep the effective per-sample update), paired with warmup
- Intuition: mini-batch mean gradient variance $\propto 1/B$, scale LR linearly to maintain SNR / step size
- Adam's update is approximately normalized by $\sqrt{\hat v}$ → empirically $\sqrt k$ (square-root) scaling is more stable

Not scaling LR with large batch (wasting compute); or forcing linear $k$ on Adam too; forgetting diminishing returns past the critical batch size.

</details>

<details>

<summary>Q16. Gradient clipping: global norm vs by value, why does it stabilize?</summary>

- Global norm: if $\lVert g\rVert\gt c$, scale the whole thing $g\leftarrow g\cdot c/\lVert g\rVert$, **preserves the direction**, only caps magnitude
- By value: clamp each component to $[-v,v]$, **changes the direction**
- Clamps occasional gradient explosions / spikes → prevents loss blow-up, prevents NaN (Pascanu 2013); LLMs commonly use global-norm with threshold 1.0

Saying "just clip," unable to distinguish global norm (preserves direction) from by-value (changes direction).

</details>

<details>

<summary>Q17. No-decay parameter groups: which params get no weight decay? Why?</summary>

- LayerNorm/RMSNorm gains $\gamma$, all biases, and (usually) embeddings get no wd
- They're scale / shift params, shrinking them toward 0 crushes representation (e.g. LN gain→0 wipes out the activation scale)
- Standard practice: split params into decay / no-decay groups (GPT/LLaMA code all do this)

Blindly applying uniform wd to all params, decaying norm gains / biases too.

</details>

<details>

<summary>Q18. AdaGrad → RMSProp → Adam, what did each step fix in the previous?</summary>

- AdaGrad: accumulates **all** $g^2$ → LR monotonically → 0 (dies on long training)
- RMSProp: switches to an **EMA** → fixes the LR decay (doesn't die)
- Adam: RMSProp + **first-moment momentum** + **bias correction** → fast and stable

Able to recite the three names but unable to connect them into the "each fixed which flaw of the previous" chain.

</details>

<details>

<summary>Q19. Where does $\epsilon$ go in Adam, and what does it do?</summary>

- PyTorch: $\theta\mathrel{-}=\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$, $\epsilon$ **outside** the sqrt
- Role: prevents a zero denominator, **and caps the step at** $\eta/\epsilon$ (no blow-up when the gradient is tiny)
- Some implementations / Adafactor put it inside $\sqrt{\hat v+\epsilon}$, slightly different numerics; LLMs commonly 1e-8, sometimes 1e-15

Getting $\epsilon$'s position backwards; or thinking it only "prevents division by zero," ignoring that it caps the max step.

</details>

<details>

<summary>Q20. Why does Adam sometimes generalize worse than SGD? Does Adam have convergence issues?</summary>

- Adaptive methods tend to converge to different (sometimes "sharper") minima, generalizing worse than SGD+M on some vision tasks
- **AMSGrad** (Reddi 2018) gives a counterexample where Adam doesn't converge: the EMA of $v$ can let the effective LR rise again, breaking convergence; the fix takes the historical max of $v$
- In practice: Transformers still prefer AdamW; CV classic nets often do better with SGD+M

Absolutizing "Adam is always best"; not knowing Adam has a convergence counterexample (AMSGrad).

</details>

### L3 advanced

<details>

<summary>Q21. What is Muon? Why orthogonalize the momentum, and why only for 2D layers?</summary>

- Orthogonalize a 2D weight's momentum $M$, $O\approx\mathrm{NewtonSchulz}(M)$ ($\approx$ the $UV^\top$ of SVD, a few Newton-Schulz iterations replacing the SVD)
- Intuition: make the update push "equally" on all singular directions, not dominated by a few large singular values → more balanced updates, faster training
- Only hidden 2D matrices; embeddings / output head / scalars still use AdamW; Kimi-K2 uses it; the original Muon is a blog (no standalone arXiv)

Saying "Muon orthogonalizes all params" (wrong, only hidden 2D layers); or not knowing it replaces the SVD with Newton-Schulz.

</details>

<details>

<summary>Q22. What do Shampoo and SOAP do? How do they essentially differ from Adam's "adaptivity"?</summary>

- Shampoo: **full-matrix / Kronecker-factored** preconditioning — maintain left/right preconditioners $L=\sum GG^\top,\ R=\sum G^\top G$, update with $L^{-1/4}GR^{-1/4}$
- Closer to full-matrix AdaGrad than Adam's **diagonal** preconditioning (captures coordinate coupling), converges faster but each step is expensive (matrix inverse-roots)
- SOAP: run Adam in Shampoo's (slowly-changing) **eigenbasis** → connects Shampoo with Adam, more stable, fewer hyperparameters to tune

Treating Shampoo as "another diagonal adaptive," missing "full-matrix / Kronecker preconditioning."

</details>

<details>

<summary>Q23. Lion's update and memory? Where did it come from?</summary>

- Lion = sign of momentum: $u=\mathrm{sign}(\beta_1 m+(1-\beta_1)g)$, $\theta\mathrel{-}=\eta(u+\lambda\theta)$, then $m=\beta_2 m+(1-\beta_2)g$
- Stores only **1 state (momentum)** → optimizer memory is **half** Adam's ($m,v$, two)
- Discovered by symbolic regression / program search (Chen 2023); the sign makes coordinates take equal-magnitude steps (usually paired with smaller LR, larger wd)

Thinking Lion also stores a second moment; or not knowing it's "sign momentum, single state."

</details>

<details>

<summary>Q24. How do you save optimizer state under memory constraints?</summary>

- Adam stores $m,v=2\times$ params (fp32 $\sim 8$ B/param), plus the fp32 master is the big chunk
- **8-bit Adam** (Dettmers 2021): block-quantize $m,v$ to 8-bit → optimizer state down to about $1/4$
- **Adafactor**: row/column-factorize the second moment $v$ (rank-1 reconstruction) → sublinear memory (T5); **Lion**: 1 state, halved

Knowing only "use Adam," unable to answer 8-bit / Adafactor / Lion when asked about memory constraints.

</details>

<details>

<summary>Q25. A few frontier / theory points: Sophia, weight-decay-as-effective-LR, μP's LR transfer.</summary>

- **Sophia** (Liu 2023): light second-order, uses a diagonal Hessian estimate for preconditioning + clipping, updating the Hessian every $k$ steps → faster than AdamW
- **New view of weight decay**: in nets with normalization (scale-invariant), wd mainly tunes the "effective learning rate / weight-norm equilibrium," rather than traditional regularization
- **μP**: scale init / LR / output multiplier by fan_in → the optimal LR is width-invariant, tune on a small model and zero-shot transfer to a large one (see the normalization tutorial §9, not re-derived)

Treating these as black magic; especially not knowing "wd $\approx$ effective-LR controller" and "μP makes the LR transferable across width."

</details>

## §A Appendix: sanity check

This tutorial's from-scratch implementations should satisfy the following key invariants (verifiable with a short pure-PyTorch script, a few seconds on CPU):

1. **[a] from-scratch SGD+momentum == `torch.optim.SGD`**: with $v_t=\mu v_{t-1}+g_t,\ \theta\mathrel{-}=\eta v_t$ (PyTorch convention: $v$ init 0, no damping), after a few steps the two are elementwise equal within float error (`atol≈1e-5`).
2. **[b] from-scratch Adam == `torch.optim.Adam`**: $\hat m=m/(1-\beta_1^t),\ \hat v=v/(1-\beta_2^t),\ \theta\mathrel{-}=\eta\hat m/(\sqrt{\hat v}+\epsilon)$ ($\epsilon$ outside the sqrt) should match PyTorch elementwise.
3. **[c] AdamW $\ne$ Adam+L2, and from-scratch decoupled decay == `torch.optim.AdamW`**: at the same `weight_decay=λ`, `torch.optim.AdamW` (decoupled) and `torch.optim.Adam(weight_decay=λ)` (L2) produce **different** updates ($\lVert\Delta\rVert\gt 0$); the from-scratch "Adam step first, then subtract $\eta\lambda\theta$" decoupled implementation should match `AdamW`. This is the core of §5.
4. **[d] bias correction returns the first step to $\eta$**: at $t=1$ the **uncorrected** effective step $\approx 3.16\eta$ ($=\frac{1-\beta_1}{\sqrt{1-\beta_2}}$, because $v$ is pushed toward 0 harder by $\beta_2$ than $m$ → denominator too small → step **too large**), corrected it returns to $\eta$. Verify the corrected/uncorrected ratio $=\frac{\sqrt{1-\beta_2}}{1-\beta_1}\approx 0.316$.
5. **[e] cosine-with-warmup schedule shape**: $\eta(0)\approx 0$, the peak $\eta_{\max}$ is hit at the warmup boundary, decays to $\sim\eta_{\min}$ at the end, and is monotonically non-increasing after warmup.
6. **[f] momentum accelerates an ill-conditioned quadratic**: on a quadratic with large condition number $\kappa$, using a **small** $\eta$ inside the stability bound for $N$ steps, GD+momentum's final loss is markedly lower than plain GD's (momentum accumulates and accelerates on the shallow direction).

Below are a few illustrative snippets (CPU-runnable, English comments, with shape/numeric notes; they demonstrate several of the invariants — the full 6 of [a]–[f] are in the standalone script below).

**[a]/[b] SGD+momentum and Adam from scratch, aligned with `torch.optim`:**

```python
import torch

def sgd_momentum_from_scratch(params, grads, lr=0.1, mu=0.9, steps=5):
    """heavy-ball: v = mu*v + g; theta -= lr*v. v init 0 (first step v=g), aligned with PyTorch SGD."""
    theta = params.clone()                      # [d] params
    v = torch.zeros_like(theta)                 # [d] momentum buffer
    for t in range(steps):
        g = grads[t]                            # [d] step-t gradient (a fixed sequence here, for alignment)
        v = mu * v + g                          # velocity = momentum*old velocity + current gradient
        theta = theta - lr * v                  # update along the velocity
    return theta

def adam_from_scratch(grads, theta0, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, steps=5):
    """EMA of m,v + bias correction + eps outside the sqrt, aligned elementwise with PyTorch Adam."""
    theta = theta0.clone()
    m = torch.zeros_like(theta)                 # first moment [d]
    v = torch.zeros_like(theta)                 # second moment [d]
    for t in range(1, steps + 1):
        g = grads[t - 1]                        # [d]
        m = b1 * m + (1 - b1) * g               # momentum EMA
        v = b2 * v + (1 - b2) * g * g           # squared-gradient EMA
        m_hat = m / (1 - b1 ** t)               # bias correction (remove early bias toward 0)
        v_hat = v / (1 - b2 ** t)
        theta = theta - lr * m_hat / (v_hat.sqrt() + eps)   # eps outside the sqrt
    return theta

# Align with torch.optim (feed the same fixed gradient sequence to both):
torch.manual_seed(0)
d, steps = 8, 5
theta0 = torch.randn(d)
grads = [torch.randn(d) for _ in range(steps)]   # fixed gradient sequence, ensures comparability

# --- SGD ---
p = theta0.clone().requires_grad_(True)
opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
for t in range(steps):
    opt.zero_grad(); p.grad = grads[t].clone(); opt.step()
ref_sgd = p.detach()
mine_sgd = sgd_momentum_from_scratch(theta0, grads, lr=0.1, mu=0.9, steps=steps)
assert torch.allclose(mine_sgd, ref_sgd, atol=1e-6)    # from-scratch == torch.optim.SGD
```

**[d] bias correction: uncorrected first step too large (at default $\beta$ $\approx 3.16\eta$), corrected $=\eta$:**

```python
import torch

b1, b2, eps = 0.9, 0.999, 1e-8
g1 = torch.tensor([2.0])                          # any nonzero first-step gradient
m1 = (1 - b1) * g1                                # uncorrected first moment
v1 = (1 - b2) * g1 * g1                           # uncorrected second moment
uncorrected = (m1 / (v1.sqrt() + eps)).item()     # uncorrected effective step / lr
corrected = ((m1 / (1 - b1)) / ((v1 / (1 - b2)).sqrt() + eps)).item()  # after correction
ratio = (1 - b1) / (1 - b2) ** 0.5                # = 3.162..., the theoretical first-step inflation
# invariant: uncorrected ≈ ratio ≈ 3.16 (too large), corrected ≈ 1.0 (= lr scale)
assert abs(uncorrected - ratio) < 1e-3
assert abs(corrected - 1.0) < 1e-3
```

**[c] AdamW (decoupled decay) $\ne$ Adam+L2 (coupled), from-scratch decoupled == `torch.optim.AdamW`:**

```python
import torch

def adamw_decoupled_from_scratch(grads, theta0, lr=0.1, b1=0.9, b2=0.999,
                                 eps=1e-8, wd=0.05, steps=5):
    """Decoupled: do the Adam step on the raw gradient g, then subtract lr*wd*theta directly (bypassing the adaptive denom)."""
    theta = theta0.clone()
    m = torch.zeros_like(theta); v = torch.zeros_like(theta)
    for t in range(1, steps + 1):
        g = grads[t - 1]                          # raw gradient without wd
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
        theta = theta - lr * (m_hat / (v_hat.sqrt() + eps) + wd * theta)  # decoupled decay
    return theta

torch.manual_seed(0)
d, steps, wd = 8, 5, 0.05
theta0 = torch.randn(d)
grads = [torch.randn(d) for _ in range(steps)]

# torch.optim.AdamW (decoupled) vs torch.optim.Adam(weight_decay=) (L2/coupled)
def run(opt_cls, **kw):
    p = theta0.clone().requires_grad_(True)
    opt = opt_cls([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd, **kw)
    for t in range(steps):
        opt.zero_grad(); p.grad = grads[t].clone(); opt.step()
    return p.detach()

ref_adamw = run(torch.optim.AdamW)
ref_adam_l2 = run(torch.optim.Adam)
assert (ref_adamw - ref_adam_l2).norm() > 1e-4         # AdamW != Adam+L2 (differ at the same wd)
mine = adamw_decoupled_from_scratch(grads, theta0, wd=wd, steps=steps)
assert torch.allclose(mine, ref_adamw, atol=1e-6)      # from-scratch decoupled == torch.optim.AdamW
```

**[e] cosine-with-warmup schedule (check first/peak/end + monotonicity):**

```python
import math

def cosine_warmup_lr(step, warmup, total, lr_max, lr_min=0.0):
    """Linear warmup to lr_max, then cosine decay to lr_min. Needs the total step count `total` up front."""
    if step < warmup:
        return lr_max * step / warmup                       # linear rise (step=0 -> ~0)
    progress = (step - warmup) / (total - warmup)           # 0 -> 1
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

warmup, total, lr_max = 100, 1000, 1.0
lrs = [cosine_warmup_lr(t, warmup, total, lr_max) for t in range(total + 1)]
assert lrs[0] < 1e-9                                        # start ~0
assert abs(lrs[warmup] - lr_max) < 1e-9                     # peak hit at the warmup boundary
assert lrs[-1] < 1e-6                                       # decays to ~lr_min(=0) at the end
assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(warmup, total))  # monotonically non-increasing after warmup
```

The full runnable script is at [`code/optimizer_lr_schedule.py`](code/optimizer_lr_schedule.py) (pure PyTorch, a few seconds on CPU, 6 `assert`s covering [a]–[f]). The **real output** of running `python code/optimizer_lr_schedule.py`:

```text
[a] SGD+momentum from scratch vs torch.optim.SGD: max|Δ| = 1.19e-07  OK
[b] Adam from scratch vs torch.optim.Adam: max|Δ| = 1.19e-07  OK
[c] AdamW vs Adam+L2 (same wd=0.1): max|Δ| = 2.13e-01 (DIFFER, decoupled≠coupled); from-scratch decoupled vs AdamW = 2.16e-07 (MATCH)  OK
[d] bias correction at step 1: corrected/uncorrected = 0.316 (= √(1-β2)/(1-β1) ≈ 0.316); without it the under-estimated v̂ makes the first step ~3.2× too large  OK
[e] cosine+warmup lr: t=0 -> 0.000, t=warmup -> 1.000 (peak), t=end -> 0.000 (min); linear warmup midpoint -> 0.500  OK
[f] ill-conditioned quadratic (κ=20) after 100 steps: GD loss = 4.04e-02  vs  GD+momentum loss = 1.46e-05  (momentum lower)  OK

all optimizer / LR-schedule sanity checks passed ✓
```

> ✅ **Reading the numbers**
> - **[a]/[b]** from-scratch SGD/Adam match `torch.optim` numerically (max$|\Delta|$=1.2e-7), implementation correct.
> - **[c]** **AdamW $\ne$ Adam+L2**: at the same wd the two differ by $0.21$ (decoupled≠coupled), while the from-scratch decoupled implementation matches `AdamW` (2.2e-7) — the core of §5, executable proof.
> - **[d]** bias-correction direction: corrected/uncorrected $=0.316=\frac{\sqrt{1-\beta_2}}{1-\beta_1}$, i.e. the **uncorrected first step is about $3.2\times$ too large** (§4.2, not too small) — a top-lab favorite direction question.
> - **[e]** cosine-warmup first($\to0$)/peak($\to1$)/end($\to0$)/warmup-midpoint($\to0.5$) shape correct (§7.2).
> - **[f]** on the ill-conditioned quadratic ($\kappa=20$) GD+momentum's loss (1.5e-5) is ~2700× lower than plain GD's (4e-2) — momentum accumulates and accelerates on the shallow direction (§2).

## 📚 References

- **Adam** — Kingma & Ba, *Adam: A Method for Stochastic Optimization*, arXiv 1412.6980 (2014), ICLR 2015.
- **AdamW (Decoupled Weight Decay)** — Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, arXiv 1711.05101 (2017), ICLR 2019.
- **AdaGrad** — Duchi, Hazan & Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*, JMLR 12 (2011) (**no arXiv id**, cited by JMLR venue).
- **RMSProp** — Hinton, *Neural Networks for Machine Learning*, Coursera Lecture 6e (2012) (course lecture, **no formal paper / arXiv id**).
- **Nesterov accelerated gradient** — Nesterov, *A method of solving a convex programming problem with convergence rate $O(1/k^2)$*, Soviet Math. Doklady 27 (1983) (**no arXiv id**).
- **Momentum for deep nets** — Sutskever, Martens, Dahl & Hinton, *On the importance of initialization and momentum in deep learning*, ICML 2013 (PMLR proceedings; **no standalone arXiv id**, cited by venue).
- **AMSGrad (Adam convergence)** — Reddi, Kale & Kumar, *On the Convergence of Adam and Beyond*, ICLR 2018 (OpenReview; no confirmed standalone arXiv id, cited by venue).
- **SGDR / cosine annealing** — Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, arXiv 1608.03983 (2016), ICLR 2017.
- **Transformer / Noam schedule** — Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762 (2017), NeurIPS 2017.
- **Linear scaling rule** — Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, arXiv 1706.02677 (2017).
- **LARS** — You, Gitman & Ginsburg, *Large Batch Training of Convolutional Networks*, arXiv 1708.03888 (2017).
- **LAMB** — You et al., *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes*, arXiv 1904.00962 (2019), ICLR 2020.
- **Lion** — Chen et al., *Symbolic Discovery of Optimization Algorithms*, arXiv 2302.06675 (2023), NeurIPS 2023.
- **Shampoo** — Gupta, Koren & Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization*, arXiv 1802.09568 (2018), ICML 2018.
- **SOAP** — Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam*, arXiv 2409.11321 (2024).
- **Adafactor** — Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, arXiv 1804.04235 (2018), ICML 2018.
- **Sophia** — Liu et al., *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*, arXiv 2305.14342 (2023), ICLR 2024.
- **One-cycle / Super-Convergence** — Smith & Topin, *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*, arXiv 1708.07120 (2017).
- **WSD / MiniCPM** — Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, arXiv 2404.06395 (2024), COLM 2024.
- **8-bit Adam** — Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv 2110.02861 (2021), ICLR 2022.
- **Gradient clipping** — Pascanu, Mikolov & Bengio, *On the difficulty of training Recurrent Neural Networks*, arXiv 1211.5063 (2012), ICML 2013.
- **Muon (scaling)** — Liu et al. (Moonshot AI), *Muon is Scalable for LLM Training*, arXiv 2502.16982 (2025).
- **Muon (original writeup)** — Keller Jordan et al., *Muon: An optimizer for the hidden layers of neural networks*, 2024 tech blog (**no standalone arXiv paper**, cited as a blog writeup).
