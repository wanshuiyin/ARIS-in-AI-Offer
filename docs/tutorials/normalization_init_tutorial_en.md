## §0 TL;DR Cheat Sheet

> 💡 **Normalization / Residual / Init in 9 lines** — the interview core on one page (derivations in §1–§11 below).

1. **Why normalize**: deep nets amplify/shrink activations layer by layer, so variance diverges or collapses exponentially as $g^L$; normalization pulls each layer's activations back to a controlled scale, and the key benefit is a **smoother loss landscape** (smaller gradient Lipschitz / β-smoothness, Santurkar 2018), which lets you use a larger learning rate and stack deeper — **NOT** the old "reduce internal covariate shift" story.

2. **BatchNorm**: normalize each channel along the **batch (+spatial) dim**, $\hat x=\frac{x-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$; **training uses batch stats + maintains running mean/var, inference uses running stats** → train$\ne$eval, and it depends heavily on batch size (small batch / variable-length sequences / RL / online all break).

3. **LayerNorm**: normalize along the **feature dim, per-token** (fully decoupled from the batch), $y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$; works at batch=1, works for variable-length sequences, train$=$eval — which is why Transformers / RNNs use it instead of BN.

4. **RMSNorm**: drop re-centering, rescale by RMS only, $\bar x=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot\gamma$ (no mean, no $\beta$); the claim is that **re-scaling invariance matters more than re-centering**, saving one reduction — the de-facto default after LLaMA.

5. **Pre-LN vs Post-LN**: Post-LN (original Transformer, LN **after** the residual add) has a slightly higher quality ceiling but **needs LR warmup and is unstable when deep**; Pre-LN (LN **inside** the residual branch) has a clean identity gradient path → stable, warmup-free, but the **residual-stream magnitude grows ∝ $\sqrt L$** with depth, deep layers get diluted, and you must add a final LN.

6. **Placement variants**: DeepNorm (up-scale residual $\alpha x_l$ + down-scale init → train 1000 layers), Sandwich / double LN (Gemma2, one before and one after the branch), **QK-Norm** (normalize Q, K before the dot product to keep attention logits from exploding).

7. **Residual + scaling**: the Jacobian $I+\partial F/\partial x$ of $y=x+F(x)$ gives a **gradient highway** (the identity term keeps the gradient from vanishing); scaling tricks push the branch's starting point toward identity — $1/\sqrt N$ depth scaling, LayerScale (learnable per-channel $\lambda$), ReZero (learnable scalar init 0), SkipInit.

8. **Initialization**: the core goal is **variance preservation** — Xavier (tanh, $\text{Var}(W)=\frac{2}{n_\text{in}+n_\text{out}}$), Kaiming (ReLU, $\text{Var}(W)=\frac{2}{n_\text{in}}$, that 2 compensating the half-variance ReLU kills); residual nets then down-scale by depth, **GPT-2 multiplies the residual-projection weights by $\frac{1}{\sqrt{2N}}$**; Fixup trains deep residual nets norm-free on init alone.

9. **μP / norm-free / engineering**: μP makes the **optimal hyperparameters (especially LR) width-invariant** → tune at small width, zero-shot transfer to the large model; Fixup / NFNets / DyT explore dropping normalization (research direction, not settled); engineering-wise watch the final LN, $\epsilon$ placement, fp32 reduction, fused kernels.

## §1 Why normalize + residuals

**The root disease of deep nets is "scale getting out of control with depth."** Pass a signal through $L$ linear layers; if each layer multiplies the activation variance by a gain $g$, then after $L$ layers the variance becomes $g^L\cdot\text{Var}(\text{input})$. $g\gt 1$ explodes exponentially, $g\lt 1$ vanishes exponentially — only the **critical point** $g=1$ is stable, and random initialization almost never hits it exactly. Backprop is the same: the gradient also multiplies by some gain layer by layer, so you get either exploding gradients (loss → NaN) or vanishing gradients (deep layers can't learn). This was the core obstacle to "training nets deeper than ten or twenty layers" before 2015.

> 💡 **$g^L$ is a simplified model, don't take it as the whole story**
> $g^L$ is a **scalar / mean-field approximation** — real per-layer gain is data-dependent and involves nonlinearity / attention / normalization, and full signal-propagation stability also concerns the Jacobian spectrum / dynamical isometry, mean drift, cross-layer correlation, and parameter drift during training. But $g^L$ captures the most essential "exponential scale blow-up" and is the minimal entry model for understanding normalization / residual / init. If a top interview digs in, point out it's a simplification — stability $\ne$ just a scalar variance product.

Two structural cures attack this disease (plus **init / parametrization**, §8–§9, as a third axis) — the three are **conceptually separable but engineering-coupled**: norm controls local scale, residual improves the gradient path, init sets step-0 criticality; modern deep-net stability is the **joint design** of all three (see §5.3 Pre-LN residual-stream growth, §8 residual scaling, §10 norm-free relying on init+scaling):

- **Normalization**: explicitly pull activations back to a fixed scale in the forward pass (zero mean, unit variance, then re-learn the needed scale via $\gamma,\beta$), directly cutting the $g^L$ product.
- **Residual / skip connection**: write each layer as $y=x+F(x)$, opening an **identity highway** for the gradient (§7), so the gradient flows to the shallow layers undiminished even if it bypasses $F$.

The two together (Pre-LN Transformer = residual + LayerNorm + good init) are what make "tens to thousands of layers" routine.

### 1.1　The internal-covariate-shift narrative — and its refutation

The motivation in the original BatchNorm paper (Ioffe & Szegedy, 2015) was **internal covariate shift (ICS)**: as the earlier layers' parameters update during training, each layer's **input distribution** keeps drifting, and the later layers have to keep re-adapting to this moving distribution, slowing training; BN fixes each layer's input mean/variance to "stabilize the distribution," so training is faster. This story is intuitive and widely repeated — **but it is most likely NOT the real reason BN works.**

Santurkar et al. (*How Does Batch Normalization Help Optimization?*, 2018, arXiv 1805.11604) dismantled this narrative with two experiments:

1. **Deliberately create ICS, BN still works**: inject time-varying random noise **after** the BN layer (artificially amplifying distribution drift, clearly increasing ICS), and a BN net's training speed / accuracy is barely affected. If BN's value were really in "removing ICS," it should have collapsed here.
2. **What BN really changes is the optimization landscape**: they show BN makes the loss **smoother** in the parameters — the Lipschitz constants of the loss and gradient shrink (better β-smoothness), so the gradient is more predictable and stable. A smoother landscape means you can safely take larger steps (larger LR) without overshooting / oscillating, and that is the mechanism behind BN's faster convergence and large-LR tolerance.

> ⚠️ **Stop reciting "BN reduces covariate shift" in interviews**
> This is an old narrative that's been strongly questioned empirically and is no longer seen as the main cause. The safer statement: normalization's main benefit is **a smoother loss landscape and better-conditioned (well-conditioned) gradients** (with side benefits like supporting larger LR, implicit regularization, scale invariance), which is what lets you train deeper; "stabilizing each layer's input distribution" is at best a surface effect, not the accepted causal mechanism. Treating covariate shift as "the proven sole cause" is a classic trap (§11, Q21).

> 💡 **One-line mental model** — normalization isn't "aligning distributions," it's **paving a flatter road** for the optimizer (small Lipschitz); residuals aren't "adding features," they're **building a traffic-free highway** for the gradient (identity Jacobian). Both exist to make "very deep" numerically feasible.

## §2 BatchNorm

### 2.1　Formula: along the batch dim, per-channel

Given a mini-batch $\mathcal{B}=\{x_1,\dots,x_m\}$, compute statistics **independently for each channel / feature $c$**:

$$\mu_c = \frac{1}{m}\sum_{i=1}^m x_{i,c}, \qquad \sigma_c^2 = \frac{1}{m}\sum_{i=1}^m (x_{i,c}-\mu_c)^2,$$

$$\hat x_{i,c} = \frac{x_{i,c}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}, \qquad y_{i,c} = \gamma_c\,\hat x_{i,c} + \beta_c.$$

The key is **which dim the statistics are taken over**: for $[N, C]$ fully-connected features, along $N$ (batch), one $(\mu,\sigma)$ per $C$; for $[N, C, H, W]$ convolutional features, along $(N, H, W)$, one per channel $C$. So BN's statistics are **"cross-sample"** — a single sample's normalized result **depends on the other samples in the batch**, which is the root of all its trouble. $\gamma_c,\beta_c$ are per-channel learnable scale / shift, letting the net recover "non-zero-mean / non-unit-variance" expressivity (forced standardization would otherwise limit it).

### 2.2　train vs eval: running stats and momentum

At inference you often **get one sample at a time**, with no batch to compute over; and you want **deterministic** outputs (same input → same result every time), which can't depend on the other samples in the batch. So during training BN **also maintains a running (moving-average) statistic**, and uses it at inference:

$$\hat\mu \leftarrow (1-\rho)\,\hat\mu + \rho\,\mu_\mathcal{B}, \qquad \hat\sigma^2 \leftarrow (1-\rho)\,\hat\sigma^2 + \rho\,\sigma_\mathcal{B}^2,$$

where $\rho$ is the momentum (PyTorch default 0.1). So:

- **Training (train mode)**: normalize with the **current batch**'s $\mu_\mathcal{B},\sigma_\mathcal{B}^2$, and update the running stats along the way.
- **Inference (eval mode)**: normalize with the frozen running $\hat\mu,\hat\sigma^2$, **no longer looking at the batch**.

> ⚠️ **train$\ne$eval is a hard constraint of BN (frequent exam point)**
> BN is one of the few layers whose training and inference take different compute paths. Forgetting `model.eval()` makes inference use batch stats → results jitter with batch content and aren't reproducible; BN layers also misbehave numerically on tiny eval batches. This contrasts sharply with LayerNorm / RMSNorm — those are structurally identical at train and eval.

### 2.3　Why BN doesn't suit sequences / small batch / RL

BN's "cross-sample statistics" break down directly in many settings:

- **Variable-length sequences (NLP / speech)**: samples have different lengths, padding folds meaningless 0s into the batch statistics; token distributions also differ across positions, so "taking the per-position mean over the batch" requires length alignment and is semantically dubious. Transformers / RNNs therefore almost never use BN.
- **Small batch**: $\mu_\mathcal{B},\sigma_\mathcal{B}^2$ are estimated from $m$ samples; small $m$ means noisy estimates (variance $\propto 1/m$) and jittery normalization. Detection / segmentation / large-image tasks can only fit batch=2~4 due to memory, where BN degrades badly (this is GroupNorm's motivation, §6). At the extreme $m=1$ the variance is 0 and it simply can't be done.
- **RL / online / streaming**: data is **non-stationary** and **strongly correlated** (adjacent states in one trajectory are highly correlated), so batch statistics are neither independent nor stable as the policy drifts; plus RL switches train/eval behavior frequently, easily decoupling BN's running stats from the actual distribution. RL commonly uses LayerNorm or no normalization.

> ❌ **"BN is the default normalization" is CV-era inertia**
> BN is still good on CNN + large-batch classification; but the moment you enter sequence modeling / small batch / RL, BN's cross-sample coupling + train$\ne$eval become a burden. BN is essentially extinct in the modern LLM stack, where LayerNorm / RMSNorm dominate.

## §3 LayerNorm

### 3.1　Formula: along the feature dim, per-token

LayerNorm (Ba, Kiros & Hinton, 2016, arXiv 1607.06450) moves the statistics dim **from batch to feature**. For a **single token / single sample** hidden vector $x\in\mathbb{R}^d$:

$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i-\mu)^2,$$

$$\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}, \qquad y = \gamma\odot\hat x + \beta, \quad \gamma,\beta\in\mathbb{R}^d.$$

For a $[B, T, d]$ Transformer tensor, LN takes $(\mu,\sigma)$ along the **last dim $d$** — one set of statistics per $(b,t)$ position, **never across samples, never across tokens**. $\gamma,\beta$ are $d$-dim per-channel affine (elementwise affine).

### 3.2　Why it fits Transformers / RNNs

LN's statistics are computed entirely **within a single token's features**, so all of BN's pain points vanish at once:

- **No batch coupling**: works at batch=1 (the feature dim always has $d$ numbers), samples don't affect each other.
- **Variable-length friendly**: each position normalizes independently; ragged lengths and padding don't affect a valid token's statistics.
- **train$=$eval**: no running stats, training and inference are the **same deterministic function** — no mode switching, no decoupling.
- **Online / streaming / RL friendly**: compute one token as it arrives, naturally fitting autoregressive decode.

> 💡 **BN along batch, LN along feature (one-line distinction)**
> Same $[N, C]$ tensor: BN slices it vertically (fix the channel, compute statistics across samples), LN slices it horizontally (fix the sample, compute statistics across channels). BN asks "what scale is this feature across the whole batch," LN asks "what scale are this sample's features internally." The former introduces cross-sample dependence, the latter doesn't — that's the entire reason LN suits sequences.

> ⚠️ **LN's $\gamma,\beta$ are not optional**
> Forcing standardization to zero-mean unit-variance **cuts expressivity** (before a sigmoid, say, you may need a non-zero mean); $\gamma,\beta$ let the net **re-learn** the scale / shift. Dropping the affine (`elementwise_affine=False`) works on some tasks but is kept by default. Note RMSNorm keeps only $\gamma$ and drops $\beta$ (§4).

## §4 RMSNorm

### 4.1　Re-scale only, no re-center

RMSNorm (Zhang & Sennrich, 2019, arXiv 1910.07467) starts from a hypothesis: **LayerNorm's success comes mainly from re-scaling invariance (scaling the vector to a fixed scale), not from re-centering (subtracting the mean).** If so, just drop the mean-subtraction and rescale by the **root mean square (RMS)** only:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}, \qquad \boxed{\;\bar x = \frac{x}{\text{RMS}(x)}\odot\gamma\;}$$

Versus LayerNorm, RMSNorm removes two things: **(1) no mean-subtraction $\mu$** (one fewer reduction), **(2) usually no $\beta$ shift** (only $\gamma$ scaling). Essentially LN both shifts and scales $x$, whereas RMSNorm only scales.

### 4.2　Why re-scaling beats re-centering + the cost

Intuition: what the net really fears is **activation scale getting out of control** (the $g^L$ business), and the scale is dominated by $\lVert x\rVert$ / RMS, with little to do with the mean. Zhang & Sennrich's ablation shows: keeping **only re-scaling and dropping re-centering** in LN barely hurts most tasks; conversely keeping only re-centering hurts a lot. So RMSNorm captures LN's main payoff with fewer operations.

Cost-wise RMSNorm is cheaper than LN:

- **One fewer mean reduction**: LN computes $\mu$ then $\sigma^2$ (two passes / one pass with compensation), RMSNorm does a single sum-of-squares pass.
- **Stores no $\beta$**, one fewer add.
- These are memory-bound elementwise + reduction ops; the savings accumulate into noticeable throughput gains on long sequences / large models.

> ✅ **The de-facto default after LLaMA**
> LLaMA, Qwen, Mistral, Gemma and other recent LLMs essentially all use RMSNorm (with Pre-LN placement). Asked "normalization in modern LLMs," the standard answer is **Pre-RMSNorm**; adding "because re-scaling is the main benefit and re-centering is dispensable" nails it.

> ⚠️ **RMSNorm has no re-centering / no bias**
> Using LN code directly as RMSNorm is wrong: RMSNorm **doesn't subtract the mean** (so it's **not** invariant to a global shift of the input — adding a constant $c$ changes the output, whereas LN doesn't), and has **no $\beta$**. One direct consequence: the activation mean after RMSNorm isn't forced to zero, so be careful if downstream code assumes zero mean.

## §5 Pre-LN vs Post-LN (the most important section)

This is the most deeply probed point in normalization: **which side of the residual does LN go on?** The two placements behave very differently.

### 5.1　The two placements

Denote a sublayer (attention or FFN) as $\text{Sublayer}(\cdot)$:

$$\textbf{Post-LN (original Transformer, Vaswani 2017)}:\quad x_{l+1} = \text{LN}\big(x_l + \text{Sublayer}(x_l)\big),$$

$$\textbf{Pre-LN (mainstream modern LLM)}:\quad x_{l+1} = x_l + \text{Sublayer}\big(\text{LN}(x_l)\big).$$

The difference looks like merely moving LN, but it changes the **purity of the residual path**:

- **Post-LN**: LN wraps the entire flow "after the residual add" — the residual path is **repeatedly squeezed by LN**, with no clean identity channel.
- **Pre-LN**: LN acts only on the branch input entering the sublayer, so the residual trunk $x_l + (\cdots)$ is a **pure identity add** — the gradient flows down the trunk undiminished.

### 5.2　The gradient-magnitude argument (Xiong et al. 2020)

Xiong et al. (*On Layer Normalization in the Transformer Architecture*, 2020, arXiv 2002.04745) did a gradient-scale analysis at initialization and gave a clean conclusion:

- **Post-LN**: at init, the parameter gradient norm **near the output layers** is about $\Theta\!\big(d\sqrt{\ln d}\big)$, **independent of depth $L$** — i.e. the top layers naturally have large gradients while the gradient scale is severely imbalanced across layers. Hitting it with a large LR straight away lets these big gradients blow training up, so you **must use learning-rate warmup** (ramp the LR up from 0 over the first few thousand steps) to suppress it, and deep stacks still often diverge.
- **Pre-LN**: each layer's parameter gradient norm is about $O\!\big(d\sqrt{(\ln d)/L}\big)$ — **decaying with depth $L$, bounded, and balanced across layers**. So optimization is well-behaved, **you can use a large LR with no warmup**, and deep stacks are stable.

> ✅ **Grasp Pre vs Post in one line**
> Pre-LN leaves the residual a **clean identity gradient path** (gradient bounded as $\sim 1/\sqrt L$, balanced across layers) → stable, warmup-free; Post-LN piles the gradient scale at the top (large, independent of $L$) → needs warmup, unstable when deep. That's why GPT/LLaMA are uniformly Pre-LN.
>
> **Note**: "Pre-LN is warmup-free" specifically means it removes the warmup that Post-LN needs because of **gradient imbalance**; in practice modern Pre-LN LLMs (GPT-3/LLaMA) **still use warmup** — for a **separate** reason (Adam's early-step $\hat v$ variance, large batch, etc., unrelated to LN placement; see the optimizer tutorial §7.1). Don't read "Pre-LN" as "needs no warmup at all."

### 5.3　Pre-LN's cost: the residual stream inflates with depth

Pre-LN isn't free. Look at the residual trunk: $x_{l+1} = x_l + F_l\big(\text{LN}(x_l)\big)$. Each branch output $F_l(\text{LN}(\cdot))$ is roughly $O(1)$ in magnitude (its input is LN-normalized), so the variance **accumulates** layer by layer:

$$\text{Var}(x_l) \approx \text{Var}(x_0) + \sum_{j\lt l}\text{Var}\big(F_j\big) \;\propto\; l \quad\Longrightarrow\quad \text{std}(x_l)\propto\sqrt{l}.$$

The residual-stream magnitude grows as $\sqrt l$ with depth. Consequence: each branch's internal LN divides by this ever-larger $\text{std}(x_l)$, so the **relative contribution of deep sublayers gets diluted** — the $O(1)$ update added is increasingly negligible against a trunk already at the $\sqrt l$ scale, the net's "effective depth" saturates, and in the extreme it degenerates into "the identity trunk dominates, deep layers nearly idle" (some call it representation / identity collapse). This is also why a Pre-LN net **must add a final LN at the end** (pull the inflated residual stream back to a normal scale before the output head), and why DeepNorm / Sandwich and other variants that try to get the best of both ends emerged.

> ⚠️ **Pre-LN must have a final LN (frequent footgun)**
> In a Pre-LN structure, after the last layer the residual stream is already at the $\sqrt L$ scale and isn't closed off by any LN. Skipping the final LN and feeding the LM head directly makes the output distribution blow up and training quality drop noticeably. GPT-2/3 and LLaMA all add this final LN after the transformer stack and before the output projection — don't miss it.

### 5.4　Why Post-LN's quality ceiling is sometimes higher

If Pre-LN is so stable, why mention Post-LN? Because **when Post-LN can train stably (sufficient warmup + tuning), its final quality is often slightly better**. The reason is the flip side of §5.3: Post-LN re-normalizes the residual stream at every layer, so deep contributions aren't diluted and "effective depth" is more fully used. The cost is fragile training (narrow window, dependent on warmup, prone to collapse when deep). So the engineering trade-off is: **for stability / scalability → Pre-LN (modern LLM default); to squeeze quality at a controllable scale → Post-LN or a "fixed-up Post-LN" like DeepNorm.**

## §6 Placement variants

Around "where to put LN / what to normalize," the community evolved a batch of variants that interviews like to name:

### 6.1　DeepNorm: fixing Post-LN up to 1000 layers

DeepNorm (Wang et al., *DeepNet*, 2022, arXiv 2203.00555) is an **improved Post-LN** aiming for both Post-LN's quality and Pre-LN's stability. Two moves:

$$x_{l+1} = \text{LN}\big(\alpha\,x_l + \text{Sublayer}(x_l)\big), \qquad \text{and shrink the sublayer-weight init by } \beta.$$

For an $N$-layer single stack, take $\alpha=(2N)^{1/4}\gt 1$ (**up-scale the residual trunk**) and a gain $\beta=(8N)^{-1/4}\lt 1$ to **down-scale the sublayer init** (encoder-decoder has different formulas). Intuition: up-scaling $x_l$ lets the residual trunk dominate at the add (close to Pre-LN's identity stability), while down-scaling the sublayer init bounds each step's "model update magnitude" **theoretically** so it doesn't explode with depth. The result is Post-LN placement + a controlled update magnitude → stable training up to **1000 layers**. Mnemonic: **DeepNorm = up-scale residual + down-scale init, a Post-LN**.

### 6.2　Sandwich / double LN: one before and one after the branch

Gemma2 and others use **sandwich norm (double LN)**: place **a normalization both before and after** each sublayer's residual branch:

$$x_{l+1} = x_l + \text{PostNorm}\Big(\text{Sublayer}\big(\text{PreNorm}(x_l)\big)\Big).$$

PreNorm feeds the sublayer a normalized input like Pre-LN (stable gradients), and PostNorm then caps the magnitude of the sublayer **output** (preventing it from pouring unbounded into the residual stream, easing the §5.3 inflation). The cost is one extra LN's compute. Think of it as "Pre-LN's stability + an extra gate on the branch output."

### 6.3　QK-Norm: normalize Q, K to stabilize attention logits

QK-Norm (Henry et al., *Query-Key Normalization*, 2020, arXiv 2010.04245) targets **attention-logit explosion**: during training $q^\top k$ can drift to extreme magnitudes, pushing softmax into saturation (near one-hot), with vanishing gradients / diverging training (worse for large models and long training, also called attention entropy collapse). The fix is minimal — **normalize each head's $q,k$ before the dot product** (L2 or RMS/LayerNorm), then compute logits:

$$\tilde q = \text{Norm}(q),\quad \tilde k = \text{Norm}(k), \qquad \text{logits} = \frac{\tilde q^\top \tilde k}{\tau}\ (\tau\ \text{learnable temperature}).$$

After normalization $q,k$ have clamped norms, so logits no longer explode with scale → stable attention. Gemma2, Chameleon, ViT-22B all use QK-Norm; it's become a standard stabilizer for large-scale training.

### 6.4　GroupNorm and WeightNorm

- **GroupNorm** (Wu & He, 2018, arXiv 1803.08494): split channels into $G$ groups and normalize **within each group of channels (+spatial) per sample** — **batch-independent**. It unifies a family of norms: $G=1$ degenerates to LayerNorm (all channels in one group), $G=C$ degenerates to InstanceNorm (one channel per group). Its main arena is **small-batch vision** (detection / segmentation, where BN collapses and GN stays stable at batch=2~4).
- **WeightNorm** (Salimans & Kingma, 2016, arXiv 1602.07868): instead of normalizing activations, **reparametrize the weights** — split the weight vector into a direction and a magnitude $w = g\,\frac{v}{\lVert v\rVert}$, learning a scalar magnitude $g$ and a direction $v$ separately. It decouples weight scale from direction and speeds optimization, with **no batch dependence and no inference overhead** (can fold back into a plain weight), but is less stable than BN and was later overshadowed by BN/LN.

> 💡 **Place the four norms in one line**
> BN along batch, LN along feature, GN along group (batch-independent, bridging LN and IN), WN normalizes the **weights** rather than activations. Asked "what normalization for small-batch detection" answer GroupNorm; "modern LLM" answer RMSNorm; "stabilize attention logits" answer QK-Norm.

## §7 Residual connections + residual scaling

### 7.1　Why residuals work: three views

The residual block $y = x + F(x)$ (He et al., *ResNet*, 2015, arXiv 1512.03385) can train very deep, with three complementary explanations:

1. **Identity mapping / gradient highway**: the Jacobian $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$. That $I$ guarantees — even if $\frac{\partial F}{\partial x}$ vanishes with depth, the gradient still flows back undiminished through the identity term. Chain $L$ residual blocks and $\frac{\partial x_L}{\partial x_0}=\prod_{l}\big(I+\frac{\partial F_l}{\partial x}\big)$ expands to contain a **pure identity term** (plus higher-order cross terms), so the gradient has **at least** one non-decaying path. This directly breaks the $g^L$ vanishing.
2. **Easier optimization / learning the residual is easier than learning the mapping**: if the ideal mapping is near identity, having $F$ fit the "residual" (the delta) is far easier than having a whole layer fit identity — initialized near 0 it's already near identity, so optimization starts from a good point.
3. **Ensemble of shallow paths (Veit et al., 2016, arXiv 1605.06431)**: an $L$-block residual net is equivalent in the forward pass to an ensemble of $2^L$ paths of different depths (each block either "takes $F$" or "takes identity"), in which **short paths dominate** and the effective gradient comes mainly from these shallow paths — so a deep residual net "behaves like an ensemble of many shallow nets," and optimization is naturally easier.

### 7.2　The residual-stream view

Mechanistic interpretability has a unifying picture: the residual trunk is a **shared communication bus (residual stream)** running through the whole net. Each sublayer **reads from the bus** (pulling the current state out via LN), computes an update, and **adds the update back to the bus**. Information is **preserved along the bus by identity by default**, and each layer only decides "what to add on top." This view explains a lot: why the Pre-LN residual stream inflates with depth (§5.3, everyone writes to the same bus), why residual scaling (below) helps (controlling how strongly each layer writes to the bus), and why semantics can be "read out" at intermediate layers.

### 7.3　Residual scaling: push the branch's start toward identity

A hazard of deep residual nets (§5.3): branch outputs pour unchecked into the residual stream, inflating its variance linearly and destabilizing early training. A family of tricks multiplies the residual branch by **a small scale**, so training **starts near identity**:

- **$1/\sqrt N$ depth scaling**: multiply each branch by $1/\sqrt N$ ($N$=#layers), so the total variance after $N$ layers stays $O(1)$ ($N\times(1/\sqrt N)^2=1$) and the residual stream doesn't blow up with depth. GPT-2's init trick (§8.4) is exactly the init-time version of this idea.
- **LayerScale** (CaiT, Touvron et al., 2021, arXiv 2103.17239): multiply each branch by a **learnable per-channel diagonal** $\text{diag}(\lambda)$, with $\lambda$ initialized to a very small value (e.g. $10^{-4}\sim10^{-6}$): $x_{l+1}=x_l+\text{diag}(\lambda)\,F(x_l)$. The start is nearly identity, and the net then **learns per-channel** how much each branch should contribute, markedly stabilizing deep ViTs.
- **ReZero** (Bachlechner et al., 2020, arXiv 2003.04887): more aggressive — multiply each branch by a **learnable scalar** $\alpha$ **initialized to 0**: $x_{l+1}=x_l+\alpha\,F(x_l)$. At $t=0$ the net is **exactly identity** (dynamical isometry, each layer's input-output Jacobian is $I$), gradients propagate perfectly, and it can train very deep norm-free; during training $\alpha$ grows on its own.
- **SkipInit** (De & Smith, 2020, arXiv 2002.10444): replace the BN on the residual branch with a **learnable scalar (init 0)** and still train deep residual nets — thereby **empirically showing that BN's main role in residual nets is to shrink the branch at init and bias the net toward the identity function** (not "reduce ICS"). This corroborates the §1.1 debunk.

> 💡 **The unifying intuition of residual scaling**
> LayerScale / ReZero / SkipInit / $1/\sqrt N$ all do the same thing: **make the residual branch near 0 at init so the net starts from identity**, giving well-behaved gradients and stable training, then let the net learn how strongly to write into the residual stream. "Init close to identity" is an underlying thread for training ultra-deep nets (and the core of §8 init).

## §8 Initialization

Normalization and residuals treat symptoms; **initialization treats the root** — it determines whether the net's step-0 variance / gradient scale is well-behaved. (Note: init only guarantees a **well-behaved starting point**; stability throughout training also relies on parametrization / LR / optimizer / residual scaling / norm placement together — see §1's "three-axis joint design.")

### 8.1　Goal: variance preservation

We want the forward activation variance and the backward gradient variance **roughly conserved layer by layer**, so $g^L$ doesn't carry them away. For a weight $W\in\mathbb{R}^{n_\text{out}\times n_\text{in}}$ and a linear layer $z=Wx$ with iid-variance input components:

$$\text{Var}(z_j) = n_\text{in}\,\text{Var}(W)\,\text{Var}(x).$$

Requiring $\text{Var}(z)=\text{Var}(x)$ (forward conservation) needs $n_\text{in}\text{Var}(W)=1$; requiring backward gradient-variance conservation needs $n_\text{out}\text{Var}(W)=1$. The two generally can't hold at once, hence different compromises.

### 8.2　Xavier / Glorot: a compromise for tanh / linear

Xavier/Glorot (Glorot & Bengio, AISTATS 2010) targets **symmetric, near-linear** activations (tanh, linear), taking a compromise between the forward and backward conditions:

$$\text{Var}(W) = \frac{2}{n_\text{in}+n_\text{out}} \quad\Longrightarrow\quad W\sim U\!\Big[-\sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}},\ \sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}}\Big]\ \text{or the corresponding Gaussian}.$$

The premise is that the activation is near-linear around 0 (tanh satisfies it, ReLU doesn't), so it's too small for ReLU nets.

### 8.3　Kaiming / He: restoring that factor 2 for ReLU

Kaiming/He (He et al., 2015, arXiv 1502.01852) points out: **ReLU zeros the negative half, killing about half the variance** — $\mathbb{E}[\text{ReLU}(z)^2]=\tfrac12\text{Var}(z)$ (for zero-mean symmetric $z$). To still conserve after ReLU, restore a factor 2:

$$\tfrac12\,n_\text{in}\,\text{Var}(W)=1 \quad\Longrightarrow\quad \boxed{\;\text{Var}(W)=\frac{2}{n_\text{in}}\;}\ (\text{fan\_in mode},\ \text{std}=\sqrt{2/n_\text{in}}).$$

This 2 is the only substantive difference between Xavier and Kaiming, yet it decides whether a deep ReLU net (VGG / ResNet scale) can train: feed ReLU with Xavier's $1/n_\text{in}$, each layer's signal energy is halved by ReLU and never restored, so after $L$ layers the second moment $\mathbb{E}[y^2]\approx (1/2)^L$ **vanishes exponentially**; only Kaiming conserves (the §A [e] check: post-ReLU second moment $\mathbb{E}[y^2]\approx1$ with Kaiming, $\approx 0.5$ and halving per layer with Xavier-for-ReLU). The conserved quantity here is the **second moment $\mathbb{E}[y^2]$** (the signal energy fed to the next layer), not $\text{Var}(y)$ — ReLU makes the output mean non-zero, so $\text{Var}(y)=1-1/\pi\approx0.68$.

> ⚠️ **Xavier vs Kaiming isn't "a different formula," it's "whether you restore ReLU's factor 2"**
> tanh / linear use Xavier ($\frac{2}{n_\text{in}+n_\text{out}}$); ReLU / LeakyReLU use Kaiming ($\frac{2}{n_\text{in}}$, LeakyReLU further adjusts the gain by the negative slope). Using Xavier on a ReLU net is systematically too small → activations / gradients vanish when deep.

### 8.4　Depth-aware down-scaling for residual nets + GPT-2's $1/\sqrt{2N}$

Even with Xavier/Kaiming at every layer, the residual stream still accumulates and inflates (§5.3). So deep residual nets, beyond per-layer init, also **down-scale the residual branch by depth**:

- **initialize the branch's last layer to 0** (branch start outputs 0 → block start is identity, same idea as ReZero / Fixup);
- or multiply the whole branch by $1/\sqrt L$.

**GPT-2's classic trick**: at init, **multiply the residual-projection layers'** weights (attention's output projection + FFN's down projection — the two matrices that "write into the residual stream") **by $\frac{1}{\sqrt{2N}}$**, $N$=#layers. Why $2N$ and not $N$? Because each transformer layer writes to the residual stream **twice** (attn once, FFN once), $N$ layers accumulate $2N$ times; scaling each write by $1/\sqrt{2N}$ keeps the residual-stream variance $O(1)$ after $2N$ adds. Implementations like nanoGPT apply this scaling to `c_proj`-type weights (the §A [f] check: unscaled variance grows linearly with depth, bounded after $1/\sqrt{2N}$ scaling).

### 8.5　Fixup: training deep residual nets norm-free on init alone

Fixup (Zhang et al., 2019, arXiv 1901.09321) pushes "control variance with init" to the extreme — **using no normalization layer at all**, training a deep residual net that rivals a BN-ResNet on carefully designed initialization alone. Three moves:

1. **initialize each residual branch's last layer to 0** (branch start outputs 0 → block is identity, residual-stream initial variance doesn't blow up);
2. **multiply the branch's other-layer weights by an extra $L^{-1/(2m-2)}$** down-scale ($m$=layers per branch, $L$=#blocks), offsetting the depth accumulation;
3. add a few **learnable scalar biases / multipliers** to compensate the affine degrees of freedom that were removed.

The significance: **normalization is not a necessary condition for training deep nets** — solve the variance / gradient scale problem once at init and you can drop BN/LN. This leads directly to the normalizer-free route of §10. (The Transformer-side counterpart is T-Fixup, which argues that with good init you can drop warmup / drop LN — see references, arXiv id to be verified.)

## §9 μP (maximal update parametrization)

### 9.1　Problem: the optimal LR drifts with width under standard parametrization

Standard parametrization (SP, i.e. Xavier/Kaiming + a single LR for the whole net) has a hidden flaw: **when the model width $n$ changes, the scales of each layer's activations and parameter updates become mismatched**, so the **optimal hyperparameters (especially the learning rate) drift with width**. The consequence is very real — the best LR you painstakingly tuned on a small model is no longer optimal on a large one, and you'd have to re-tune every time you widen; while one tuning run on a large model is hugely expensive in compute.

The root cause (the limit analysis in Yang et al., *Tensor Programs V*, 2022, arXiv 2203.03466): under SP, taking width $n\to\infty$, either activations / updates blow up with $n$ (forcing a smaller LR), or you enter the **lazy / kernel regime** (features barely update, i.e. no feature learning). Neither limit is the "still updates features stably and substantially as it widens" that we want.

### 9.2　μP: make hyperparameters width-invariant → zero-shot transfer

μP (maximal update parametrization) is the **unique** scaling scheme that maintains "maximal and stable feature learning" at any width: it writes the **init variance, learning rate, and output multiplier** as functions of width (fan_in), scaling each layer separately, so that —

- each layer's activations and **feature-update magnitude** stay $\Theta(1)$ across width (neither blowing up nor degenerating);
- the **optimal learning rate becomes width-invariant**.

Typical scaling (under Adam, schematically): hidden-layer LR $\propto 1/\text{fan\_in}$, hidden-layer init variance $\propto 1/\text{fan\_in}$, output logits times $1/\text{fan\_in}$; input / output layers use **different** scaling from hidden layers. Realized as **μTransfer**: tune LR / warmup / init and other hyperparameters on a **small-width proxy model**, then **zero-shot transfer directly** to a target model tens-to-hundreds of times larger, saving the astronomical compute of re-tuning on the big model (used at GPT-3 scale, Cerebras-GPT, MiniCPM, etc.).

> 💡 **μP interview soundbite**
> Under SP the "optimal LR drifts with width," so params tuned on a small model can't be used directly on a large one; μP re-scales init / LR / output multiplier by fan_in so that **each layer's feature update is $\Theta(1)$ across width and the optimal LR is width-invariant** → tune at small width, zero-shot transfer at large width (μTransfer). In a line: **μP is the parametrization that makes hyperparameters transferable across width.**

## §10 Norm-free and the frontier

Normalization layers bring train/eval differences (BN), cross-device sync (multi-GPU BN), extra reductions, and other hassles, so people keep asking: **can we go without normalization?** This is a research direction, **not settled**, but the ideas are instructive.

### 10.1　Fixup / NFNets: replacing normalization with init + explicit variance control

- **Fixup** (§8.5): train deep residual nets on init alone (branch last layer to 0 + depth down-scale), with no normalization layer, approaching BN-ResNet on ImageNet. Proof that norm is not required.
- **NFNets** (Brock et al., 2021, arXiv 2102.06171): Normalizer-Free Networks, systematically removing BN. Three pieces — **Scaled Weight Standardization** (standardize weights rather than activations) + **analytically designed scaled residual blocks** $x_{l+1}=x_l+\alpha\,F(x_l/\beta_l)$ (track / control each layer's variance exactly with analytic $\alpha,\beta_l$) + **Adaptive Gradient Clipping (AGC)** (clip gradients adaptively by parameter norm, recovering BN's large-batch stability). The result **surpasses EfficientNet on ImageNet with no normalization layer at all**, showing BN's benefits (scale control + regularization + large-batch stability) can each be restored by explicit means.

### 10.2　DyT (Dynamic Tanh): replacing LN with a learnable tanh

DyT (*Transformers without Normalization*, 2025, arXiv 2503.10622, arXiv id to be verified) comes from an observation: **a trained LayerNorm's input-output curve looks like a squashed $\tanh$** (near-linear for middle values, S-shaped saturation for outliers). Since LN's effect approximates an elementwise squashing, just **drop the mean / variance reduction and learn a tanh directly**:

$$\text{DyT}(x) = \gamma\odot\tanh(\alpha\,x) + \beta,$$

where $\alpha$ is a **learnable scalar** (controlling the input scale, corresponding to LN's "divide by std"), and $\gamma,\beta$ are per-channel affine. The paper reports DyT replacing LN/RMSNorm across ViT / LLM / diffusion with comparable results, and **removes the normalization's statistic computation** (no more per-token reduction).

> 🎯 **Frontier, honest positioning**
> Fixup / NFNets / DyT together convey one message: **normalization is not theoretically a necessary condition for training deep nets** — its core roles (control variance, smooth the landscape, squash outliers) can be replaced by "good init / explicit variance control / learnable squashing." But **LN/RMSNorm remain the safe default in current production systems** (robust, plug-and-play, mature ecosystem). When discussing these in interviews, mark them as "research directions / promising," don't say "already replaced normalization."

## §11 Engineering practice + complexity comparison + common misconceptions

### 11.1　Comparison of the four norms

| Norm | Normalization dim | batch-dependent | train$\ne$eval | learnable params | relative cost |
| --- | --- | --- | --- | --- | --- |
| **BatchNorm** | along batch (+spatial), per-channel | yes | **yes** (running stats at inference) | $\gamma,\beta$ | medium (two reductions + maintain running stats + multi-GPU sync) |
| **LayerNorm** | along feature dim, per-token | no | no | $\gamma,\beta$ | medium (mean + variance, two reductions) |
| **RMSNorm** | along feature-dim RMS, per-token | no | no | $\gamma$ (no $\beta$) | low (one sum-of-squares reduction, no mean, no shift) |
| **GroupNorm** | within-group channels (+spatial), per sample | no | no | $\gamma,\beta$ | medium (within-group reduction, batch-independent) |

### 11.2　Engineering details

- **Pre-LN: don't miss the final LN** (§5.3): there must be a closing LN after the transformer stack and before the LM head, or the inflated residual stream goes straight into the output head and quality drops.
- **The $\epsilon$ placement**: the convention is $\sqrt{\sigma^2+\epsilon}$ ($\epsilon$ **inside** the sqrt, as in PyTorch), not $\sqrt{\sigma^2}+\epsilon$. Too small an $\epsilon$ in fp16 triggers division-by-zero / overflow when $\sigma^2\approx0$; RMSNorm's $\epsilon$ is likewise inside $\sqrt{\text{mean}(x^2)+\epsilon}$.
- **fp16/bf16 numerics**: the normalization reduction (sum / sum-of-squares) **must accumulate in fp32**, even if the activations are bf16 — sum-of-squares easily overflows / loses significant bits in low precision. Standard implementations upcast the norm internals to fp32 then downcast.
- **fused kernels**: LN/RMSNorm are **memory-bound** (read activations → reduce → write back), tiny in FLOPs but heavy in memory traffic; a fused kernel (Apex `FusedLayerNorm`, Triton, FlashNorm, etc.) merges read-compute-write into one kernel, saving HBM round-trips. RMSNorm is naturally cheaper since it has only one reduction and no mean.
- **inference cost**: by FLOPs normalization is negligible, but it **breaks operator fusion + carries a reduction**, accumulating into noticeable latency in large-model per-token decode; this is one reason RMSNorm (one fewer reduction) is favored on the inference side.
- **BN's multi-GPU footgun**: under data parallelism each GPU sees only part of the batch, so BN stats are "local-batch"; getting global stats requires SyncBatchNorm (cross-GPU communication, slow). LN/RMSNorm don't have this issue (per-token, no sync needed).

### 11.3　Common footguns

> ❌ **Footgun 1: treating covariate shift as the sole/proven reason normalization works**
> Empirically refuted by Santurkar 2018 (§1.1). The correct attribution is "smoother loss landscape / better-conditioned gradients." This is the most classic interview trap.

> ❌ **Footgun 2: Pre-LN forgetting the final LN**
> The Pre-LN residual stream inflates as $\sqrt L$ with depth with no closing LN at the end; missing the final LN and feeding the output head directly drops quality (§5.3).

> ❌ **Footgun 3: using LN code as RMSNorm**
> RMSNorm **doesn't subtract the mean, has no $\beta$**, and is **not** invariant to a global input shift; copying LN's mean-subtraction / bias logic is wrong (§4.2).

> ❌ **Footgun 4: using BatchNorm on RL / variable-length sequences / small batch**
> Cross-sample statistics + train$\ne$eval break down across these settings; switch to LayerNorm / GroupNorm (§2.3).

> ❌ **Footgun 5: using Xavier init on a ReLU net**
> Missing ReLU's factor-2 compensation, activations / gradients decay exponentially when deep; use Kaiming for ReLU (§8.3).

## §12 25 high-frequency interview questions

Sorted into three tiers; click for answer points + easy traps. L2/L3 are top-lab deep water (Pre/Post gradient argument, Kaiming derivation, μP, DeepNorm, norm-free, QK-Norm, the covariate-shift debunk, etc.).

### L1 must-know

<details>

<summary>Q1. Why do deep nets need normalization? What does it actually solve?</summary>

- Deep-net activation / gradient variance explodes or vanishes as $g^L$, only the critical point $g=1$ is stable, and random init rarely hits it
- Normalization pulls each layer's activations back to a fixed scale, cutting the $g^L$ product → larger LR, stack deeper
- Modern correct attribution: a smoother loss landscape, better-conditioned gradients (Santurkar 2018), **not** "reducing covariate shift"

Saying only "speeds up convergence," missing the variance-scale / loss-landscape layer; or still reciting covariate shift.

</details>

<details>

<summary>Q2. Write the BatchNorm formula. Which dim does it compute statistics over?</summary>

- Per-channel along batch (conv adds spatial $H,W$) for $\mu_c,\sigma_c^2$, $\hat x=\frac{x-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$, then $\gamma_c\hat x+\beta_c$
- Statistics are **cross-sample** → a sample's normalization depends on the other samples in the batch
- $[N,C]$ along $N$; $[N,C,H,W]$ along $(N,H,W)$, one per channel

Saying BN is along the feature dim (that's LN); or forgetting its cross-sample nature.

</details>

<details>

<summary>Q3. How do BatchNorm training and inference (eval) differ? Why?</summary>

- Training normalizes with the **current batch stats** and updates running mean/var with momentum
- Inference uses the frozen **running stats** (not the batch), ensuring determinism + single-sample use
- So BN is a train$\ne$eval layer; forgetting `eval()` makes inference results jitter with the batch

Not knowing about running stats / momentum; or thinking inference also uses batch stats.

</details>

<details>

<summary>Q4. Write the LayerNorm formula. How does its dim differ from BN's?</summary>

- For a single-token vector $x\in\mathbb{R}^d$ along the **feature dim $d$** for $\mu,\sigma$, $y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$
- $[B,T,d]$ along the last dim, one set of stats per $(b,t)$, **never across samples or tokens**
- BN along batch (cross-sample), LN along feature (within-sample) — that's the essential difference

Saying LN is also cross-batch; or failing to answer "per-token, decoupled from the batch."

</details>

<details>

<summary>Q5. What is RMSNorm? How does it differ from LayerNorm?</summary>

- Rescale by RMS only: $\bar x=\frac{x}{\sqrt{\frac1d\sum x_i^2+\epsilon}}\odot\gamma$
- Removes two things vs LN: **no mean-subtraction** (no re-centering), **no $\beta$** shift
- Claim: re-scaling is LN's main benefit, re-centering is dispensable → cheaper, the default after LLaMA

Saying only "RMSNorm is faster," missing the "drop mean-subtraction + drop bias" two removals.

</details>

<details>

<summary>Q6. Where does LayerNorm go in a Transformer? What are the two placements?</summary>

- Post-LN: $x_{l+1}=\text{LN}(x_l+\text{Sublayer}(x_l))$, LN **after** the residual add (original Transformer)
- Pre-LN: $x_{l+1}=x_l+\text{Sublayer}(\text{LN}(x_l))$, LN **inside** the residual branch (mainstream modern LLM)
- Modern LLMs use Pre-LN + a final LN

Knowing only "there's an LN," unable to say which side of the residual or name the two placements.

</details>

<details>

<summary>Q7. Why do Transformers / RNNs use LayerNorm instead of BatchNorm?</summary>

- Variable-length sequences + padding make "per-position stats over the batch" semantically dubious and hard to align
- Batch coupling: BN's one sample depends on others in the batch; sequence tasks often run small batch / batch=1
- LN is per-token, batch-decoupled, train$=$eval, naturally fitting sequences / autoregressive decode

Saying only "LN is customary," missing batch coupling + variable length + train/eval as concrete reasons.

</details>

<details>

<summary>Q8. Why do residual connections help train very deep networks?</summary>

- $y=x+F(x)$ Jacobian $I+\partial F/\partial x$; that $I$ gives the gradient an **identity highway**, breaking $g^L$ vanishing
- Learning the residual is easier than learning the whole mapping (start from a good point when the ideal is near identity)
- Equivalent to an ensemble of $2^L$ paths, short paths dominate, easier optimization (Veit 2016)

Saying only "prevents vanishing gradients," missing the Jacobian identity term / residual-is-easier / shallow-path-ensemble views.

</details>

<details>

<summary>Q9. What are $\gamma,\beta$ (scale/shift) in normalization for? Can you drop them?</summary>

- Forced standardization to zero-mean unit-variance cuts expressivity; $\gamma,\beta$ let the net **re-learn** the scale / shift
- $\gamma$ scales, $\beta$ shifts, per-channel
- RMSNorm keeps only $\gamma$ and drops $\beta$; dropping all affine works on some tasks but is kept by default

Thinking $\gamma,\beta$ are optional; or not knowing RMSNorm drops $\beta$.

</details>

<details>

<summary>Q10. The difference between Xavier and Kaiming init? Which activation does each suit?</summary>

- Xavier: $\text{Var}(W)=\frac{2}{n_\text{in}+n_\text{out}}$, for **tanh / linear** (symmetric near-linear activations)
- Kaiming: $\text{Var}(W)=\frac{2}{n_\text{in}}$, for **ReLU**, the 2 compensating the half-variance ReLU kills
- Using Xavier on a ReLU net is too small → activations / gradients vanish when deep

Reciting only the two formulas, unable to say "the difference is whether you restore ReLU's factor 2."

</details>

### L2 advanced

<details>

<summary>Q11. Pre-LN vs Post-LN: behavioral differences + the gradient argument.</summary>

- Post-LN needs **LR warmup**, unstable when deep, but a slightly higher quality ceiling; Pre-LN is stable, warmup-free, but the residual stream inflates with depth and needs a final LN
- Xiong 2020: Post-LN top-layer gradient $\Theta(d\sqrt{\ln d})$ (large independent of $L$, imbalanced across layers) → must warmup
- Pre-LN per-layer gradient $O(d\sqrt{(\ln d)/L})$ (decays with $L$, bounded, balanced) → large LR, warmup-free

Saying only "Pre-LN is more stable," missing the quantitative top-layer-large / decays-with-depth argument.

</details>

<details>

<summary>Q12. Why is dropping re-centering in RMSNorm fine? Which matters more, re-scaling or re-centering?</summary>

- What the net fears is activation **scale** getting out of control ($g^L$), and the scale is dominated by RMS / norm, little to do with the mean
- Ablation: keeping only re-scaling in LN barely hurts, keeping only re-centering hurts a lot → re-scaling dominates
- So RMSNorm captures the main payoff with fewer ops

Saying only "saves the mean computation," missing the "re-scaling is the main benefit" justification.

</details>

<details>

<summary>Q13. Why doesn't BatchNorm suit variable-length sequences / small batch / RL?</summary>

- Variable length: padding pollutes stats + different positions have different distributions, "per-position stats over the batch" is semantically poor
- Small batch: estimation noise $\propto1/m$, small $m$ jitters normalization, $m=1$ fails outright
- RL: non-stationary + strongly correlated data + frequent train/eval switching, running stats decouple from reality

Saying vaguely "BN doesn't work well," missing the failure mechanism for each of the three settings.

</details>

<details>

<summary>Q14. Derive the Kaiming init variance. Why is it $2/n_\text{in}$?</summary>

- Linear layer $\text{Var}(z)=n_\text{in}\text{Var}(W)\text{Var}(x)$, forward conservation needs $n_\text{in}\text{Var}(W)=1$
- ReLU zeros the negative half: $\mathbb{E}[\text{ReLU}(z)^2]=\frac12\text{Var}(z)$, killing half the variance
- Restore the factor 2: $\frac12 n_\text{in}\text{Var}(W)=1\Rightarrow\text{Var}(W)=\frac{2}{n_\text{in}}$

Unable to write $\text{Var}(z)=n_\text{in}\text{Var}(W)\text{Var}(x)$, or not knowing the 2 comes from ReLU's halving.

</details>

<details>

<summary>Q15. Why do residuals break vanishing gradients? Give the backprop expression.</summary>

- $L$ blocks chained: $\frac{\partial x_L}{\partial x_0}=\prod_l(I+\frac{\partial F_l}{\partial x})$
- Expansion contains a **pure identity term** (plus higher-order cross terms) → at least one non-decaying gradient path
- Even if $\partial F/\partial x\to0$, the identity term still carries the gradient back to shallow layers undiminished

Saying only "added a skip," unable to write why the identity term in the Jacobian product saves the gradient.

</details>

<details>

<summary>Q16. What does residual scaling ($1/\sqrt N$ / LayerScale / ReZero) solve? How is it done?</summary>

- Solves: deep residual branches pouring unchecked into the residual stream → linear variance inflation, early instability
- $1/\sqrt N$: multiply branch by $1/\sqrt N$ so $N$-layer accumulation stays $O(1)$; LayerScale: learnable per-channel $\lambda$ (tiny init); ReZero: learnable scalar $\alpha$ init 0 (exactly identity start)
- Unifying intuition: make the branch start near 0, the net starts from identity, well-behaved gradients

Remembering only the names, unable to state the shared "make the residual branch start ≈ identity" mechanism.

</details>

<details>

<summary>Q17. What is GPT-2 multiplying the residual-projection weights by $1/\sqrt{2N}$ for? Why $2N$ not $N$?</summary>

- Controls residual-stream variance: scale each write into the residual stream by $1/\sqrt{2N}$ so the accumulation stays $O(1)$
- $2N$: each transformer layer writes to the residual stream **twice** (attn output proj + FFN down proj), $N$ layers total $2N$
- Acts on the init of the attn/FFN output-projection (residual-projection) weights

Thinking it's once per layer ($N$); or not knowing it acts on the two specific "residual-projection" matrices.

</details>

<details>

<summary>Q18. What is QK-Norm? What problem does it solve?</summary>

- Normalize each head's $q,k$ (L2/RMS) before the dot product, then compute logits
- Solves attention-logit explosion with scale → softmax saturation / entropy collapse / diverging training (worse for large models)
- After normalization $q,k$ norms are clamped, logits don't explode; used in Gemma2 / Chameleon / ViT-22B

Saying only "normalize QK," missing that it guards against logit explosion / softmax saturation.

</details>

<details>

<summary>Q19. How does DeepNorm train to 1000 layers? How does it relate to Pre/Post-LN?</summary>

- It's an **improved Post-LN**: $x_{l+1}=\text{LN}(\alpha x_l+\text{Sublayer}(x_l))$, $\alpha=(2N)^{1/4}\gt1$ up-scales the residual
- Also a gain $\beta=(8N)^{-1/4}\lt1$ to **down-scale the sublayer init**, bounding each step's model update
- up-scale residual (near Pre-LN stability) + down-scale init (bounded updates) → Post-LN quality + stable to 1000 layers

Treating DeepNorm as Pre-LN; or remembering only $\alpha$ and not "also shrink the init."

</details>

<details>

<summary>Q20. How do GroupNorm / InstanceNorm relate to BN/LN? When to use GN?</summary>

- GN splits channels into $G$ groups, normalizing within each group per sample, **batch-independent**
- $G=1$ degenerates to LayerNorm, $G=C$ to InstanceNorm (GN bridges the two)
- Use: **small-batch vision** (detection / segmentation batch=2~4, BN collapses, GN stable)

Not knowing GN uses $G$ to bridge LN and IN; or unable to answer small-batch detection as the typical setting.

</details>

### L3 advanced

<details>

<summary>Q21. Is internal covariate shift the real reason BatchNorm works?</summary>

- No (most likely). This is the original paper's narrative, empirically refuted by Santurkar 2018
- Evidence 1: injecting noise after BN to **deliberately increase** ICS barely affects training
- Evidence 2: BN actually shrinks the loss / gradient Lipschitz (smoother landscape) → supports large LR
- SkipInit further shows BN's role in residual nets is to shrink the branch at init, biasing toward identity

Still treating covariate shift as the accepted cause; or not knowing there are experiments that directly falsify it.

</details>

<details>

<summary>Q22. What is μP? Why does the optimal LR drift with width under standard parametrization?</summary>

- Under SP, changing width mismatches each layer's activation / update scale; as width $n\to\infty$ either it blows up or enters the lazy/kernel regime
- So the optimal LR varies with width, and params tuned on a small model can't be used directly on a large one
- μP scales init variance / LR / output multiplier by fan_in so that **feature updates are $\Theta(1)$ across width and the optimal LR is width-invariant** → μTransfer (tune small-width, zero-shot transfer to large-width)

Knowing only "μP transfers hyperparameters," unable to explain why SP drifts / what μP scales.

</details>

<details>

<summary>Q23. How do you train deep residual nets without any normalization layer? (Fixup / NFNets)</summary>

- Fixup: **branch last layer init 0** (identity block start) + multiply other layers by $L^{-1/(2m-2)}$ down-scale + add learnable scalar biases/multipliers
- NFNets: Scaled Weight Standardization + analytic scaled residual block $x+\alpha F(x/\beta)$ + Adaptive Gradient Clipping
- Common point: restore "variance control + large-batch stability" via init / explicit means, showing norm is not required

Saying only "tune init," unable to give "last layer to 0 + depth down-scale" or NFNets' three pieces concretely.

</details>

<details>

<summary>Q24. What is DyT (Dynamic Tanh)? Is normalization a necessary condition for training?</summary>

- Observation: a trained LN's input-output curve looks like a squashed $\tanh$ (near-linear in the middle, saturating for outliers)
- DyT: $\gamma\odot\tanh(\alpha x)+\beta$, a learnable scalar $\alpha$ replacing "divide by std," **dropping the mean/variance reduction**
- Implication: normalization is theoretically not necessary (its role can be replaced by squashing / init / explicit variance control), but LN/RMSNorm remain the safe production default

Saying DyT "already replaced normalization" (overstated); or unable to state it comes from the "LN≈tanh" observation.

</details>

<details>

<summary>Q25. Given a very deep network (hundreds to thousands of layers), how do you combine init / normalization / residual to train it stably?</summary>

- **Residuals** are mandatory (identity gradient path), push the branch start toward identity: last layer init 0 or LayerScale/ReZero, multiply by depth $1/\sqrt N$ (GPT-2's $1/\sqrt{2N}$)
- **Normalization** Pre-LN/RMSNorm (warmup-free, stable) + **don't miss the final LN**; for Post-LN quality use DeepNorm (up-scale residual + down-scale init, up to 1000 layers)
- **init** pick Xavier/Kaiming by activation for variance preservation, depth-aware down-scaling; for norm-free go Fixup/NFNets
- **hyperparameters** tune with μP at small width then transfer; add QK-Norm if attention is unstable; fp32 reduction throughout
- The core thread: **start the net from a well-behaved "near-identity + variance-conserving" point**

Reporting a single trick; or missing the combined punch of final LN / residual scaling / depth-aware init and the "identity start" thread.

</details>

## §A Appendix: sanity check

This tutorial's code has a minimal runnable version in [`docs/tutorials/code/normalization.py`](code/normalization.py) (pure PyTorch, a few seconds on CPU, 6 `assert`s covering [a]–[f]). It should satisfy the following key invariants:

1. **[a] LayerNorm from scratch == `nn.LayerNorm`**: with population variance (`unbiased=False`, divide by $d$), $\epsilon$ inside the sqrt, after affine it should match PyTorch elementwise within float error (`atol≈1e-5`).
2. **[b] RMSNorm from scratch == `nn.RMSNorm`, and RMSNorm's shift-invariance differs from LN's**: add a constant $c$ to the whole input, and **LayerNorm's output barely changes** (it subtracts the mean → re-centering invariant), while **RMSNorm's output changes** (it only re-scales, no mean-subtraction) — exactly the executable check of §4's "RMSNorm has no re-centering."
3. **[c] BatchNorm train$\ne$eval**: train mode uses batch stats (output per-feature mean ≈ 0) and pushes the running mean away from 0; switch to eval and it uses running stats, with a clearly different output for the same input — verifying §2.2's dual path.
4. **[d] Post-LN piles parameter gradients at the top (top-heavy), Pre-LN is balanced across depth**: build a stack of 48 identical residual blocks (with a final linear head — to avoid the artifact where a loss acting directly on the Post-LN normalized output "squashes gradients to 0"), do one forward+backward, and measure each block's Linear weight-grad **top/bottom ratio** (last/first). Pre-LN $\approx0.40$ (balanced, slightly bottom-heavy), Post-LN $\approx2.35$ ($\gt1$, gradient piled near the output) — exactly the executable check of §5.2 Xiong's "Post-LN top-layer gradients large → needs warmup." **Use the top/bottom ratio rather than the absolute gradient, because it's not confounded by the loss choice / output normalization** (a subtle experiment-design pitfall).
5. **[e] Kaiming preserves the second moment, Xavier-for-ReLU decays**: a unit-second-moment input ($\mathbb{E}[x^2]=1$) through `Linear+ReLU`, measuring the **post-ReLU second moment $\mathbb{E}[y^2]$** (the signal energy fed to the next layer, the quantity He's derivation actually propagates): with Kaiming ($\sqrt{2/n_\text{in}}$) $\mathbb{E}[y^2]\approx1$ (conserved), with Xavier-for-ReLU ($\sqrt{1/n_\text{in}}$) $\approx0.5$ (halving per layer → vanishing when deep) — verifying §8.3's factor 2. Note we measure $\mathbb{E}[y^2]$, not $\text{Var}(y)$: ReLU makes the output mean non-zero, so $\text{Var}(y)=1-1/\pi\approx0.68$, while the layer-conserved / propagated quantity is the second moment $\mathbb{E}[y^2]$.
6. **[f] GPT-2 residual scaling $1/\sqrt{2N}$ bounds the residual-stream variance**: accumulate $N$ $O(1)$ updates into the residual stream; unscaled, the variance **grows linearly** with depth, while multiplying by $1/\sqrt{2N}$ keeps it **bounded** — verifying §8.4.

Below are a few illustrative snippets (CPU-runnable, English comments, logic identical to the script above).

**(a) LayerNorm + RMSNorm from scratch, aligned with PyTorch:**

```python
import torch
import torch.nn as nn

def layernorm_from_scratch(x, weight, bias, eps=1e-5):
    """Normalize along the last dim. x: [..., d]. Population variance (unbiased=False), eps inside the sqrt."""
    mean = x.mean(dim=-1, keepdim=True)                       # [..., 1] mean
    var  = x.var(dim=-1, unbiased=False, keepdim=True)        # [..., 1] population var (/d), matches torch
    return (x - mean) / torch.sqrt(var + eps) * weight + bias # standardize + affine

def rmsnorm_from_scratch(x, weight, eps=1e-6):
    """Re-scale by RMS only: no mean-subtraction, no bias. x: [..., d]."""
    ms = x.pow(2).mean(dim=-1, keepdim=True)                  # [..., 1] mean of squares
    return x / torch.sqrt(ms + eps) * weight                  # divide by RMS, then per-channel scale

d = 64
x = torch.randn(8, d)                                         # [B, d]
ln  = nn.LayerNorm(d, eps=1e-5)                              # default affine weight=1, bias=0
assert torch.allclose(layernorm_from_scratch(x, ln.weight, ln.bias), ln(x), atol=1e-5)

rms = nn.RMSNorm(d, eps=1e-6)                                # needs torch>=2.4
assert torch.allclose(rmsnorm_from_scratch(x, rms.weight), rms(x), atol=1e-5)

# RMSNorm is NOT invariant to a global shift, LayerNorm is:
c = 5.0
assert (ln(x + c) - ln(x)).abs().max() < 1e-4                              # LN re-centering invariant
assert (rmsnorm_from_scratch(x + c, rms.weight) - rmsnorm_from_scratch(x, rms.weight)).abs().max() > 1e-2
```

**(b) A deep residual stack, comparing Pre-LN vs Post-LN gradient norms across depth:**

```python
import torch
import torch.nn as nn

class ResidualStack(nn.Module):
    """Deep residual stack + final linear head. Pre-LN: h=h+Linear(LN(h)); Post-LN: h=LN(h+Linear(h)).
       The head matters: otherwise the Post-LN output is LN-normalized and loss=mean(out^2) is ~constant,
       squashing all gradients to 0 (a confound)."""
    def __init__(self, depth, d, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.lns  = nn.ModuleList(nn.LayerNorm(d) for _ in range(depth))
        self.lins = nn.ModuleList(nn.Linear(d, d) for _ in range(depth))
        self.head = nn.Linear(d, d)                           # final projection, removes the output-normalization confound

    def forward(self, h):                                     # h: [B, d]
        for ln, lin in zip(self.lns, self.lins):
            h = h + lin(ln(h)) if self.pre_ln else ln(h + lin(h))
        return self.head(h)

def block_grad_topheavy(depth, d, pre_ln):
    """One forward+backward, return the last/first block weight-grad-norm ratio (>1 = gradient piled at the top).
       This ratio is robust to the loss choice; exactly Xiong 2020's claim."""
    torch.manual_seed(0)                                      # same init for both placements, fair comparison
    stack = ResidualStack(depth, d, pre_ln=pre_ln)
    stack(torch.randn(16, d)).pow(2).mean().backward()       # scalar loss
    gn = [lin.weight.grad.norm().item() for lin in stack.lins]
    return gn[-1] / gn[0]

r_pre  = block_grad_topheavy(48, 64, pre_ln=True)            # expect ≈0.40 (balanced)
r_post = block_grad_topheavy(48, 64, pre_ln=False)          # expect ≈2.35 (top-heavy, >1)
# Post-LN top/bottom ratio >1: gradient piled near the output -> must warmup. Pre-LN balanced across depth.
```

**(c) Kaiming variance preservation (vs misusing Xavier for ReLU):**

```python
import torch
import torch.nn.functional as F

fan_in, fan_out, N = 512, 512, 4096
x = torch.randn(N, fan_in)                                   # E[x^2]=1 (unit second moment) [N, fan_in]
W_kaiming = torch.randn(fan_out, fan_in) * (2.0 / fan_in) ** 0.5   # He: std=sqrt(2/fan_in)
W_xavier  = torch.randn(fan_out, fan_in) * (1.0 / fan_in) ** 0.5   # misusing Xavier for ReLU: std=sqrt(1/fan_in)
# measure the post-ReLU second moment E[y^2] (signal energy fed to the next layer, the quantity He propagates), not Var(y)
ms_kaiming = F.relu(x @ W_kaiming.t()).pow(2).mean()        # ReLU halves, 2/fan_in restores -> E[y^2]≈1
ms_xavier  = F.relu(x @ W_xavier.t()).pow(2).mean()         # no factor 2 -> E[y^2]≈0.5, halving per layer
# expect E[y^2]_kaiming≈1 (conserved), E[y^2]_xavier≈0.5 (vanishes exponentially when deep).
# Note: Var(y)≈0.68 (ReLU makes the mean non-zero); the layer-conserved quantity is the second moment E[y^2], not Var.
```

The real output of running `python docs/tutorials/code/normalization.py` (CPU, pure PyTorch, with the summary of the six [a]–[f] `assert`s):

```text
[a] LayerNorm from scratch vs nn.LayerNorm: max|Δ| = 2.38e-07  OK
[b] RMSNorm vs nn.RMSNorm: max|Δ| = 2.38e-07; mean-shift LN |Δ|=9.54e-07 (~0, re-centers) vs RMS |Δ|=3.05e+00 (>0, only re-scales)  OK
[c] BatchNorm train!=eval: max|Δ| = 8.37e+00; running_mean moved 0.701 from 0; train per-feature mean = 8.15e-08 (~0)  OK
[d] per-block weight-grad top/bottom ratio (last/first over 48 blocks): Pre-LN=0.40 (balanced)  Post-LN=2.35 (top-heavy, >1)  -> Post-LN piles gradient near the output, needs warmup  OK
[e] post-ReLU second moment E[y^2] (input E[x^2]=1): Kaiming = 1.000 (~1, preserved)  Xavier = 0.499 (~0.5, halves per layer)  OK
[f] residual-stream var growth over 50 blocks: unscaled ×51.1  vs  1/sqrt(2N)-scaled ×1.50  OK

all normalization / residual / init sanity checks passed ✓
```

> ✅ **Reading the numbers**
> - **[a]/[b]** from-scratch LN/RMSNorm match PyTorch (max$|\Delta|$=2.4e-7); in the shift test LN barely changes (9.5e-7) while RMSNorm changes clearly (3.05), confirming §4 "RMSNorm only re-scales, no re-center."
> - **[c]** BatchNorm running_mean pushed from 0 to 0.70, train-mode per-feature mean ≈ 0, and train≠eval, confirming §2.2's dual path.
> - **[d]** block weight-grad top/bottom ratio (last/first): Pre-LN $0.40$ (balanced), Post-LN $2.35$ ($\gt1$, gradient piled at the top), confirming §5.2 "Post-LN top-layer gradients large → needs warmup" (the head removes the output-normalization confound).
> - **[e]** post-ReLU second moment Kaiming $\mathbb{E}[y^2]=1.00$ (conserved), Xavier-for-ReLU $=0.50$ (halving per layer), confirming §8.3's factor 2.
> - **[f]** residual-stream variance unscaled ×51 after 50 layers (linear growth), bounded ×1.5 after $1/\sqrt{2N}$ scaling, confirming §8.4's GPT-2 trick.

## 📚 References

- **BatchNorm** — Ioffe & Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, arXiv 1502.03167 (2015), ICML 2015.
- **LayerNorm** — Ba, Kiros & Hinton, *Layer Normalization*, arXiv 1607.06450 (2016).
- **RMSNorm** — Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv 1910.07467 (2019), NeurIPS 2019.
- **How Does BN Help Optimization (covariate-shift debunk)** — Santurkar et al., *How Does Batch Normalization Help Optimization?*, arXiv 1805.11604 (2018), NeurIPS 2018.
- **Pre-LN vs Post-LN (gradient argument)** — Xiong et al., *On Layer Normalization in the Transformer Architecture*, arXiv 2002.04745 (2020), ICML 2020.
- **DeepNorm / DeepNet** — Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*, arXiv 2203.00555 (2022).
- **Kaiming init** — He et al., *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, arXiv 1502.01852 (2015), ICCV 2015.
- **Xavier / Glorot init** — Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010 (PMLR proceedings, **no arXiv id**, citation to be verified).
- **ResNet (residual learning)** — He et al., *Deep Residual Learning for Image Recognition*, arXiv 1512.03385 (2015), CVPR 2016.
- **Residuals as ensembles** — Veit et al., *Residual Networks Behave Like Ensembles of Relatively Shallow Networks*, arXiv 1605.06431 (2016), NeurIPS 2016.
- **Fixup** — Zhang et al., *Fixup Initialization: Residual Learning Without Normalization*, arXiv 1901.09321 (2019), ICLR 2019.
- **ReZero** — Bachlechner et al., *ReZero is All You Need: Fast Convergence at Large Depth*, arXiv 2003.04887 (2020), UAI 2021.
- **LayerScale / CaiT** — Touvron et al., *Going deeper with Image Transformers*, arXiv 2103.17239 (2021), ICCV 2021.
- **SkipInit** — De & Smith, *Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks*, arXiv 2002.10444 (2020), NeurIPS 2020.
- **μP / μTransfer** — Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, arXiv 2203.03466 (2022), NeurIPS 2021.
- **NFNets (normalizer-free)** — Brock et al., *High-Performance Large-Scale Image Recognition Without Normalization*, arXiv 2102.06171 (2021), ICML 2021.
- **GroupNorm** — Wu & He, *Group Normalization*, arXiv 1803.08494 (2018), ECCV 2018.
- **WeightNorm** — Salimans & Kingma, *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks*, arXiv 1602.07868 (2016), NeurIPS 2016.
- **QK-Norm** — Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245 (2020), EMNLP 2020 Findings.
- **DyT (Dynamic Tanh)** — Zhu et al., *Transformers without Normalization*, 2025 (CVPR 2025; arXiv 2503.10622, **arXiv id subject to citation verification**).
- **T-Fixup** — Huang et al., *Improving Transformer Optimization Through Better Initialization*, ICML 2020 (PMLR proceedings; no widely-used standalone arXiv id, cited by venue).
- **GPT-2 (residual-projection $1/\sqrt{2N}$ init)** — Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report (2019) (**no arXiv id**).
