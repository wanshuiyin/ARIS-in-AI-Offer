## §0 TL;DR Cheat Sheet

> 💡 **8 sentences to nail MoE** — get the core points of 2026 fall recruiting season in one page (see §1-§9 derivations).

1. **Core idea**: replace a single FFN with $N$ experts + a router; each token goes through only $k \ll N$ experts, **total parameters go up, active parameters stay the same** (sparse activation). Compute is roughly $k/N$× of dense, but memory / GPU memory follows total parameters.

2. **Routing formula** (Token-Choice top-k): $g_i(x) = \text{softmax}(W_g x)_i$, select $\mathcal{T}_k(x) = \text{TopK}_i\, g_i(x)$, output $y = \sum_{i \in \mathcal{T}_k(x)} g_i(x) \cdot E_i(x)$. The gate probabilities act as **soft weights** multiplied onto expert outputs, making the router differentiable during backprop.

3. **Historical lineage (5 must-memorize papers)**: Shazeer 2017 (first MoE layer that worked in deep learning) → GShard 2020 (top-2 + capacity factor + all-to-all) → Switch Transformer 2021 (top-1 minimalist, aux loss + load balance) → Mixtral 8x7B/8x22B 2024 (first major open-source MoE, top-2) → DeepSeek-V3 2024 (671B/37B, **aux-loss-free**, fine-grained + 1 shared).

4. **DeepSeek line (2026 interview hotspot)**: DeepSeekMoE 2024 proposed **fine-grained experts** (split into smaller experts; $mN$ small experts, select $mK$) + **shared experts** (a few experts seen by all tokens, absorbing common knowledge). V2 uses MLA + DeepSeekMoE; V3 pushes routed experts to 256 + 1 shared, **drops aux loss**, replacing it with online updates to expert biases.

5. **Aux-loss-free balance**: add a per-expert bias $b_i$ to the router score, **used only for top-k selection**, not in gradients (also not in the final gate weight); each step updates $b_i$ in the direction of "actual load - expected load" (over-loaded experts get their bias reduced). **Does not break sparse gradients, does not introduce interfering gradients.**

6. **Capacity and token dropping**: each expert's capacity is $C = \lceil \alpha \cdot T \cdot k / N \rceil$ ($\alpha$ is the capacity factor, commonly 1.0-1.25). Tokens beyond capacity go through **residual bypass** (skip the expert entirely, original residual passes through), or are dropped. This is a core engineering detail in the Switch Transformer paper.

7. **Parallelism**: MoE almost always requires **Expert Parallelism (EP)** — placing different experts on different GPUs, doing **all-to-all** (dispatch) after token routing → expert compute → all-to-all (combine). DeepEP and DualPipe are the two engineering weapons DeepSeek-V3 uses on H800 to overlap EP communication and computation.

8. **Common bugs**: ① routing collapse (some experts get picked by all tokens, others starve); ② router logit overflow in fp16; ③ at inference time, **all experts must be loaded into GPU memory** — only active parameters are sparse; **memory still follows total parameters**; ④ at small batch sizes, EP communication saturates and throughput is worse than dense.

## §1 Intuition: Sparse Activation = Trading "Compute" for "Parameter Capacity"

Dense FFN looks like this: every token passes through the same $D \times 4D \to 4D \times D$ large matrix, with all parameters participating in computation.

MoE replaces an FFN with $N$ independent experts (each still essentially an FFN); the router decides each token **only goes through $k$ of them**:

```

                    Token x  [B, L, D]
                          │
                          ↓
                  Router W_g ∈ R^{D × N}        (gate)
                          │
                          ↓
                  scores  g(x) ∈ R^N
                          │
                  TopK + softmax-renormalize
                          │
              ┌─────┬─────┼─────┬─────┐
              ↓     ↓     ↓     ↓     ↓
            E_1   E_2   ...   E_{N-1}  E_N    (each expert = FFN)
              │     │           │     │
              └─ × g_1 ─ ... ─ × g_N ─┘        (only k of the g_i ≠ 0)
                          │
                          ↓
                  Sum → output y [B, L, D]
```

Why is this a good idea? A three-level answer:

- **Capacity argument**: model capacity mainly comes from parameter count, and generalization from data × capacity. Sparse activation lets total parameters be pushed to thousands of B while per-token compute remains like a 30B dense model — **parameter capacity ≫ compute**.
- **Specialization argument**: in training, different experts naturally divide up labor — code / math / multilingual / general semantics. Switch Transformer showed t-SNE visualizations; DeepSeekMoE intensifies this specialization with fine-grained experts.
- **Inference efficiency argument**: fixed active parameters $\approx$ fixed FLOPs, comparable to dense; but parameter capacity is much larger — this is the fundamental reason Mixtral 8x7B with 13B active beats LLaMA-2 70B.

> ⚠️ **MoE is not a free lunch** — sparse activation reduces **FLOPs** but **not memory** (at inference, all experts must reside on GPU); EP all-to-all communication can easily eat half the time during training; load imbalance can leave most GPUs idle. §3-§7 unpacks these engineering realities in full.

## §2 Routing: From Top-k Token-Choice to Expert Choice to Loss-Free

### 2.1　Token-Choice Top-k (GShard / Switch / Mixtral / DeepSeek)

The most classic and most-tested. The gate computes the affinity per (token, expert) pair, and the token picks the top-k experts:

$$\boxed{\;s(x) = W_g x \in \mathbb{R}^N, \quad g(x) = \text{softmax}(s(x)), \quad \mathcal{T}_k(x) = \text{TopK}_i\, g_i(x)\;}$$

Output (routing-weight-renormalized version, default in Mixtral / DeepSeek):

$$y = \sum_{i \in \mathcal{T}_k(x)} \tilde{g}_i(x) \cdot E_i(x), \quad \tilde{g}_i(x) = \frac{g_i(x)}{\sum_{j \in \mathcal{T}_k(x)} g_j(x)}$$

> 💡 **Renormalize vs use raw softmax probability** — both engineering implementations have been seen: Switch uses $g_i$ directly (when top-1, $g_i$ is multiplied directly); Mixtral / DeepSeek renormalize over top-k so weights sum to 1. Renormalization keeps the output scale stable, not depending on probability mass that "leaks out" of top-k in softmax.

**Some variants' $k$**:

| Model | $k$ | Notes |
| --- | :-: | --- |
| Switch Transformer | 1 | Minimalist; shown to work |
| GShard / Mixtral / Qwen3-MoE | 2 | Mainstream; 2 experts are enough to "ensemble" |
| DeepSeek-V2 / V3 | 6 / 8 (routed) + 1 shared | Combined with fine-grained splitting |
| Llama 4 Scout | 1 (only 1 routed) + 1 shared | 16 experts, 1 shared |
| Llama 4 Maverick | 1 routed + 1 shared | 128 experts |

### 2.2　Auxiliary Load Balancing Loss (Switch formula, must-memorize)

Problem: pure top-k has no mechanism to **balance** the use of all experts — it can easily collapse into "only 1-2 experts used".

Switch Transformer introduces a differentiable load balancing loss:

$$\boxed{\;\mathcal{L}_\text{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i\;}$$

where

- $f_i = \dfrac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{\arg\max_j s_j(x_t) = i\}$ — the fraction of tokens assigned to expert $i$ (non-differentiable, sample-level estimate)
- $P_i = \dfrac{1}{T} \sum_{t=1}^{T} g_i(x_t)$ — expert $i$'s average gate probability (differentiable, softmax output)
- $\alpha$ is the loss weight (commonly $10^{-2}$)
- $N$ is the number of experts; $T$ is the number of tokens in the batch

**Key points / common confusions**:

1. $f_i$ provides the "actual frequency"; $P_i$ provides the "gradient path". Multiplied, $\sum_i f_i P_i$ is largest when both concentrate on the same group of experts and smallest when uniformly distributed ($= 1/N$, multiplied by $N$ = 1).
2. This **encourages a uniform distribution**, not a hard constraint. Extreme collapse is penalized, but moderate day-to-day imbalance is not heavily penalized.
3. The Switch top-1 formula is exactly this; for top-k, change $f_i$ to "top-k hit frequency" (see GShard paper for the more precise form).
4. **Too large $\alpha$ interferes with the main task gradient** — this is the root motivation for DeepSeek to switch to aux-loss-free.

### 2.3　Expert Choice Routing (Zhou et al. 2022, NeurIPS)

Reverse thinking: instead of letting tokens pick experts, let **experts pick tokens**. Each expert picks the top-$M$ tokens by capacity, $M = (T \cdot k) / N$ (the average number of tokens each expert gets).

$$s(x) \in \mathbb{R}^{T \times N}, \quad \mathcal{T}_M^{(i)} = \text{TopM}_{t}\, s_{t,i}, \quad y_t = \sum_{i: t \in \mathcal{T}_M^{(i)}} g_{t,i} \cdot E_i(x_t)$$

**Advantages**:

- **Naturally balanced**: each expert strictly picks $M$ tokens, no aux loss needed
- **No token dropping**: theoretically no "overflow" (capacity is hard-set)

**Disadvantages / limitations**:

- **Not causal**: the assignment of the $t$-th token depends on the scores of all $T$ tokens — for decoder inference, **token-by-token generation is impossible**, requiring a batch-level global view (autoregressive unfriendly)
- The number of experts each token actually activates **is not fixed** (could be 0, could be selected by many)
- Mainly used in encoders (e.g., T5, BERT-style) or vision

> ⚠️ **Interview trap** — asked "Expert Choice solves routing collapse, so why do Mixtral / DeepSeek still use token-choice?" Answer: autoregressive decoder + the need for each token to strictly use $k$ experts makes expert-choice unusable; DeepSeek-V3 switches to aux-loss-free to do balance **within the token-choice framework**.

### 2.4　Auxiliary-Loss-Free Balance (DeepSeek-V3 trademark, must-know)

DeepSeek (Wang et al., arXiv 2408.15664, 2024)'s proposal, fully adopted by V3. Core in one sentence: **add a bias term $b_i$ to each expert, used only for top-k selection, not in the gradient**.

Specifically, the original score is $s_i(x)$; in top-k selection we use the **biased score**:

$$\tilde{s}_i(x) = s_i(x) + b_i$$

$$\mathcal{T}_k(x) = \text{TopK}_i\, \tilde{s}_i(x)$$

But the **final gating weight** still uses the softmax (or sigmoid then renormalize, DeepSeek-V3 uses sigmoid) of the original $s_i$:

$$g_i(x) = \frac{\sigma(s_i(x))}{\sum_{j \in \mathcal{T}_k(x)} \sigma(s_j(x))}, \quad i \in \mathcal{T}_k(x)$$

$b_i$ update rule (once per step, **out-of-graph**, non-gradient update):

$$b_i \leftarrow b_i + u \cdot \text{sign}(\bar{c}_i - c_i)$$

- $c_i$ = actual number of tokens received by expert $i$ in the current step
- $\bar{c}_i = T k / N$ = expected number of tokens per expert under uniform distribution
- If $c_i > \bar{c}_i$ (overloaded) → $b_i$ decreases → harder to pick next time
- If $c_i < \bar{c}_i$ (underloaded) → $b_i$ increases → easier to pick next time
- $u$ is the fixed step size (V3 reports it very small, order $10^{-3}$)

> ✅ **Why this is a non-trivial win** — the problem with aux loss is that it injects "balance" as a **gradient signal** into the router → interfering with the main task gradient; for balance, the model may sacrifice quality. Loss-free turns balance into an **offline control signal** (PID-flavored sign update), completely outside the computation graph — the router's gradient is still only for "language-modeling loss", but expert selection is nudged by $b_i$. **Does not break the semantics of sparse routing, does not pollute the main gradient, does not introduce an $\alpha$ to tune.**

### 2.5　Noisy Top-k (Shazeer 2017 original scheme, now rarely used)

Shazeer et al. 2017 paper adds learnable Gaussian noise to the score:

$$s_i(x) = (W_g x)_i + \text{StandardNormal}() \cdot \text{Softplus}((W_\text{noise} x)_i)$$

Noise "softens" the top-k selection to avoid collapse. Later replaced by GShard / Switch with the more explicit load-balance loss — but the concept ("top-k is non-differentiable → needs some stochastic softening") still has pedagogical value.

## §3 Implementation Details: Core 60-Line PyTorch

Minimal runnable implementation: token-choice top-k MoE layer + Switch aux loss. Without considering EP, for single-GPU pedagogical use.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Each expert is a standard FFN (SwiGLU-style is also fine; here we use GELU for simplicity)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class MoELayer(nn.Module):
    """
    Token-choice top-k MoE layer.
    Input x: [B, L, D]
    Output y: [B, L, D], aux_loss: scalar (Switch load-balance loss)
    """
    def __init__(self, d_model, d_ff, num_experts, top_k=2, capacity_factor=1.25, aux_loss_coef=0.01):
        super().__init__()
        assert top_k <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef

        # Router: D -> N, no bias (neither Switch nor Mixtral uses bias)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # N experts
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        B, L, D = x.shape
        T = B * L
        x_flat = x.view(T, D)                              # [T, D]

        # ----- 1. Router computes score + top-k -----
        logits = self.router(x_flat)                       # [T, N]
        probs = F.softmax(logits, dim=-1)                  # [T, N]
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # [T, k], [T, k]
        # Mixtral / DeepSeek style: renormalize over top-k
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)  # [T, k]

        # ----- 2. Aux load-balance loss (Switch formula) -----
        # f_i = fraction of tokens assigned to expert i (counted top_k times per token, consistent with top_k)
        with torch.no_grad():
            one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()  # [T, k, N]
            f = one_hot.sum(dim=(0, 1)) / (T * self.top_k)                       # [N]
        P = probs.mean(dim=0)                                                    # [N], differentiable
        aux_loss = self.aux_loss_coef * self.num_experts * (f * P).sum()

        # ----- 3. Capacity computation + token dispatch -----
        # Consistent with the main-text formula: ceil(capacity_factor * T * top_k / N_E), avoid truncating to 0 at small batch
        import math as _math
        capacity = max(1, _math.ceil(self.capacity_factor * T * self.top_k / self.num_experts))
        # Expand [T, k] into [T*k] of (token_idx, expert_idx) pairs
        flat_expert = top_idx.view(-1)                     # [T*k]
        flat_weight = top_probs.view(-1)                   # [T*k]
        flat_token = torch.arange(T, device=x.device).repeat_interleave(self.top_k)  # [T*k]

        # ----- 4. Expert forward (capacity-aware) -----
        y_flat = torch.zeros_like(x_flat)                  # [T, D]
        for e in range(self.num_experts):
            mask_e = (flat_expert == e)                    # [T*k]
            if mask_e.sum() == 0:
                continue
            tok_e = flat_token[mask_e]                     # global indices of tokens for expert e
            w_e   = flat_weight[mask_e]                    # corresponding gate weights
            # Capacity truncation: tokens exceeding capacity are dropped (Switch residual bypass)
            if tok_e.numel() > capacity:
                tok_e = tok_e[:capacity]
                w_e   = w_e[:capacity]
            inp_e = x_flat[tok_e]                          # [≤cap, D]
            out_e = self.experts[e](inp_e)                 # [≤cap, D]
            # Scatter-add by token idx (a token can be picked by multiple experts, hence add)
            y_flat.index_add_(0, tok_e, out_e * w_e.unsqueeze(-1))

        return y_flat.view(B, L, D), aux_loss
```

> ⚠️ **Key difference from production implementations** — the pedagogical version uses a Python `for e in range(num_experts)` to call each expert serially, but **production (Megatron / vLLM) all use grouped GEMM or fused kernels**, batching computations across all experts; otherwise, with large $N$, the GPU spends most time launching kernels. This is the truly hard layer of MoE.

### 3.1　Aux-Loss-Free Bias Update (DeepSeek-V3 style)

Remove the aux loss in §3 and switch to expert bias updates:

```python
class MoEAuxFree(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2, bias_step=1e-3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_step = bias_step
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # b_i: non-parameter, not in the gradient graph (buffer)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        B, L, D = x.shape
        T = B * L
        x_flat = x.view(T, D)

        # Router score (V3 uses sigmoid rather than softmax, but the top-k logic is the same)
        logits = self.router(x_flat)                       # [T, N]
        # ★ Top-k uses biased score, but gating weight uses original score
        biased_logits = logits + self.expert_bias.unsqueeze(0)
        _, top_idx = biased_logits.topk(self.top_k, dim=-1)         # [T, k]
        # Take the original score at the selected positions, sigmoid then renormalize
        gate_raw = torch.sigmoid(logits).gather(-1, top_idx)        # [T, k]
        top_weights = gate_raw / (gate_raw.sum(dim=-1, keepdim=True) + 1e-9)

        # ----- Online expert_bias update (no grad, paper's sign update) -----
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()  # [T, k, N]
                c = one_hot.sum(dim=(0, 1))                          # [N], actual load
                c_bar = T * self.top_k / self.num_experts            # expected load (scalar)
                # Overloaded -> decrease bias; underloaded -> increase bias
                self.expert_bias -= self.bias_step * torch.sign(c - c_bar)

        # Dispatch + expert forward (same as §3, omitted here)
        y_flat = torch.zeros_like(x_flat)
        flat_expert = top_idx.view(-1)
        flat_weight = top_weights.view(-1)
        flat_token  = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        for e in range(self.num_experts):
            mask = (flat_expert == e)
            if mask.sum() == 0: continue
            tok_e = flat_token[mask]
            w_e   = flat_weight[mask]
            out_e = self.experts[e](x_flat[tok_e]) * w_e.unsqueeze(-1)
            y_flat.index_add_(0, tok_e, out_e)
        return y_flat.view(B, L, D)
```

Notes:

1. `expert_bias` is a buffer rather than a parameter; **it is not in the gradient graph and is not stepped by the optimizer** — it is only modified by the "sign update" at the end of `forward`.
2. **biased_logits is used only for selecting top-k; the gating weight comes from the original logits** (this is the point the V3 paper repeatedly emphasizes). If even the gating weight used biased values, that would pollute the main gradient path with the control signal, defeating the benefit of aux-loss-free.
3. V3 actually uses sigmoid + 256 experts + 9 selected (8 routed + 1 shared, see §6); here it is simplified to softmax + top-k to demonstrate the principle.

### 3.2　Fine-Grained + Shared Expert Layer (DeepSeekMoE style)

```python
class DeepSeekMoELayer(nn.Module):
    """
    Fine-grained + shared expert: 
      - Total N routed experts (split into smaller experts, each m times smaller than the standard expert)
      - Plus N_shared shared experts (every token goes through them)
      - Each token is routed to K routed experts (K = m × baseline_top_k)
    Paper (Dai et al. 2024, arXiv 2401.06066): split N original experts into mN,
    each expert's d_ff shrunk to d_ff/m, routing count raised from K to mK; total compute / parameter count unchanged,
    but combination count C(mN, mK) >> C(N, K) -> specialization granularity improved.
    """
    def __init__(self, d_model, d_ff_per_expert, num_routed_experts, num_shared_experts, top_k):
        super().__init__()
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)
        # Fine-grained routed experts (each m times smaller than baseline)
        self.routed_experts = nn.ModuleList(
            [Expert(d_model, d_ff_per_expert) for _ in range(num_routed_experts)]
        )
        # Shared experts: always active, not routed
        self.shared_experts = nn.ModuleList(
            [Expert(d_model, d_ff_per_expert) for _ in range(num_shared_experts)]
        )

    def forward(self, x):
        # 1. Shared expert: directly accumulate, no routing
        shared_out = sum(e(x) for e in self.shared_experts)

        # 2. Routed top-k (simplified here, aux loss / capacity omitted)
        B, L, D = x.shape
        T = B * L
        x_flat = x.view(T, D)
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + 1e-9)

        routed_out = torch.zeros_like(x_flat)
        flat_e = top_i.view(-1)
        flat_w = top_p.view(-1)
        flat_t = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        for e in range(self.num_routed):
            mask = (flat_e == e)
            if mask.sum() == 0: continue
            tok = flat_t[mask]
            w   = flat_w[mask]
            routed_out.index_add_(0, tok, self.routed_experts[e](x_flat[tok]) * w.unsqueeze(-1))

        return shared_out + routed_out.view(B, L, D)
```

> 💡 **"Combinatorial explosion" intuition of fine-grained** — standard 8 choose 2 → 28 combinations. Split into 64 choose 16 (each expert shrunk to 1/8, routing count ×8), the combinations become $\binom{64}{16} \approx 4.9 \times 10^{14}$. Under the same parameter/compute budget, **the "expert combination identity" each token can express is exponentially richer** — this is one of DeepSeekMoE's core arguments.

## §4 Capacity, Token Dropping, Routing Collapse

### 4.1　Capacity Factor (must-memorize engineering detail)

It is impossible for an expert to accept arbitrarily many tokens — GPU memory + EP communication both need static buffers. So each expert is pre-set with a capacity $C$:

$$\boxed{\;C = \left\lceil \alpha \cdot \frac{T \cdot k}{N} \right\rceil\;}$$

- $T$: total tokens in the batch
- $k$: number of experts each token goes through
- $N$: total number of experts
- $\alpha$: capacity factor (commonly 1.0, 1.25, 2.0)

**$\alpha = 1$**: just enough under uniform distribution, but any slight imbalance causes drops; **$\alpha = 1.25$** (Switch / GShard default): leaves 25% headroom buffer for imbalance; **$\alpha > 2$**: basically no drops but wastes memory.

### 4.2　Token Dropping vs Residual Bypass

When some expert has filled its $C$ tokens, what happens to the next token? **Two routes**:

| Scheme | Behavior | Who uses |
| --- | --- | --- |
| **Token Drop** | Drop directly, output 0 | GShard early; usually ↓ training stability |
| **Residual Bypass** | Expert output is 0, but **residual $x$ naturally passes through via layer norm + skip connection** | Switch Transformer / Mixtral / DeepSeek default |

Benefit of residual bypass: dropped tokens **are not turned to 0** — on the residual path, they still carry their own representation, just that this layer "is not processed by any expert", equivalent to degenerating to an identity layer. The entire network can still learn.

> ⚠️ **Must-test pitfall** — interviews often ask "why does MoE not catastrophically break". Answer: residual bypass + multi-layer stacking, **single-layer drop is not fatal** — the next layer's router can still pick a suitable expert for that token.

### 4.3　Routing Collapse (💣 classic bug)

If the router has no balance mechanism, in early training some experts get accidentally over-selected → their parameters get updated more → they are more likely to be selected next time → the strong get stronger → **a few experts dominate, the rest starve**. This is routing collapse.

Diagnostic signals:

- Monitor expert load (tokens received per expert per step) → severely uneven
- Monitor aux loss → high or rising
- Validation loss plateau appears early

Fixes:

1. **Aux loss** (Switch formula, §2.2)
2. **Expert dropout** / **Z-loss** (entropy regularizer on router logits)
3. **Expert Choice** (§2.3, hard constraint of fixed tokens per expert)
4. **Aux-loss-free bias** (§2.4, DeepSeek-V3)
5. **Moderately enlarge capacity factor** (let collapse trigger drops, indirectly punishing concentration)

### 4.4　How does Top-k's "hard" selection backprop gradients?

Classic question: $\text{TopK}$ is a non-differentiable discrete operation.

Answer: **top-k only decides which path is taken (discrete), and does not participate in the gradient**; the softmax weights $g_i(x)$ enter the final output $y = \sum_i g_i E_i(x)$, and differentiating with respect to $g_i$ is continuous (standard softmax chain rule). So the router's $W_g$ is differentiable (note the **full Jacobian** induced by the softmax / renorm coupling, not just the diagonal):

Let $s = W_g x$ (logits), $p = \text{softmax}(s)$; after top-k selection $\mathcal{T}_k(x)$, do renormalize $g_i = p_i / Z$ ($Z = \sum_{j \in \mathcal{T}_k} p_j$):

$$\frac{\partial g_i}{\partial s_l} = \mathbf{1}[i \in \mathcal{T}_k]\cdot\left(\frac{\partial p_i / \partial s_l}{Z} - \frac{p_i}{Z^2}\cdot \mathbf{1}[l \in \mathcal{T}_k] \cdot \frac{\partial Z}{\partial s_l}\right)$$

where $\partial p_i/\partial s_l = p_i(\delta_{il} - p_l)$ is the softmax Jacobian (including the cross term $-p_i p_l$, not just the diagonal $p_i(1-p_i)$). Aggregate:

$$\frac{\partial \mathcal{L}}{\partial W_g} = \sum_{i \in \mathcal{T}_k(x)}\sum_{l} \frac{\partial \mathcal{L}}{\partial g_i} \cdot \frac{\partial g_i}{\partial s_l} \cdot \frac{\partial s_l}{\partial W_g}$$

For $l \notin \mathcal{T}_k$ (excluded by top-k), $\partial g_i/\partial s_l$ is still nonzero via $p_l$ (the global coupling of softmax), but this term does not affect the output in forward, and its contribution in backward only enters indirectly via $\partial p_i / \partial s_l = -p_i p_l$. **In practice**, only experts within top-k receive dominant gradient signal; experts outside top-k cannot get the training push "I should be selected" for a long time — this is a chicken-and-egg: not selected → not updated → never selected. Aux loss / bias is exactly to break this cycle.

## §5 Classic Model Lineage: From Shazeer to DeepSeek-V3

Arranged by time and "engineering threshold" in 7 rows (must-memorize timeline + key contributions):

| Year | Paper / Model | Core contribution | arXiv |
| --- | --- | --- | --- |
| 2017 | **Shazeer et al., Outrageously Large NN** | First deep MoE layer (LSTM side-mounted), noisy top-k, load-balance loss prototype | 1701.06538 |
| 2020 | **GShard (Lepikhin et al.)** | Top-2 routing, capacity factor, **automatic sharding / all-to-all**, 600B multilingual NMT | 2006.16668 |
| 2021 | **Switch Transformer (Fedus, Zoph, Shazeer)** | Top-1 minimalist, standard aux loss formula, 1.6T-parameter T5-MoE | 2101.03961 |
| 2022 | **Expert Choice (Zhou et al., NeurIPS)** | Expert-choice routing, hard balance, no aux loss needed (but autoregressive unfriendly) | 2202.09368 |
| 2024.1 | **DeepSeekMoE (Dai et al.)** | **Fine-grained experts + shared experts**, validated at 16B / 145B | 2401.06066 |
| 2024.1 | **Mixtral 8x7B (Jiang et al.)** | First major open-source MoE, 8 experts top-2, 13B active beats LLaMA-2 70B | 2401.04088 |
| 2024.4 | **Mixtral 8x22B** | 141B total / 39B active, 64k context | (Mistral blog) |
| 2024.5 | **DeepSeek-V2 (DeepSeek-AI)** | 236B / 21B, **MLA + DeepSeekMoE**, KV cache reduced 93.3% | 2405.04434 |
| 2024.8 | **Loss-Free Balance (Wang et al.)** | Aux-loss-free bias update scheme | 2408.15664 |
| 2024.12 | **DeepSeek-V3** | **671B / 37B**, 256 routed + 1 shared expert, aux-loss-free, MTP, DualPipe | 2412.19437 |
| 2025.4 | **Llama 4 (Meta)** | Scout (17B/16E), Maverick (17B/128E), Behemoth (288B/16E) | (Meta blog) |
| 2025.5 | **Qwen3 MoE** | Qwen3-235B-A22B (22B active) | 2505.09388 |

### 5.1　Horizontal comparison of mainstream open-source MoE (frequently asked in 2026 interviews)

| Model | Total params | Active / token | Routed experts | Top-k | Shared expert | Aux scheme | Released |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 | 0 | aux loss | 2023.12 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 | 0 | aux loss | 2024.4 |
| Qwen2-57B-A14B | 57B | 14B | 64 | 8 | 1 | aux loss | 2024.6 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | 2 | aux loss + device-level | 2024.5 |
| **DeepSeek-V3** | **671B** | **37B** | **256** | **8** | **1** | **Aux-loss-free + sequence-aux** | 2024.12 |
| Llama 4 Scout | 109B | 17B | 16 | 1 | 1 | undisclosed | 2025.4 |
| Llama 4 Maverick | 400B | 17B | 128 | 1 | 1 | undisclosed | 2025.4 |
| Qwen3-235B-A22B | 235B | 22B | 128 | 8 | 0 | (loss-free variant) | 2025.5 |

> 💡 **How to read this table** — note the "total / active" ratio (Mixtral $\approx 4\times$, V3 $\approx 18\times$, Maverick $\approx 24\times$). The higher this ratio, the bigger the "trade sparsity for capacity" leverage, but also the more dependent on routing quality. V3 pushes this ratio to the extreme and still works, relying on fine-grained + shared + loss-free.

## §6 DeepSeek-V3 Full Picture (must-know focus)

DeepSeek-V3 (DeepSeek-AI, arXiv 2412.19437, 2024.12) is the highest-frequency MoE system in the 2026 interview season. Unpacking its design points:

### 6.1　Architecture level

```
                  DeepSeek-V3 MoE Layer
                 ────────────────────────
       x  [B, L, D=7168]
         │
         ├──────► Shared Expert (1, d_ff ≈ 2048)        →  shared_out
         │         (always active, not routed)
         │
         └──────► Router (D → 256)
                   │
                   ├ score s_i(x) + b_i  (bias only used for top-k selection)
                   │
                   ↓
                Top-8 (select 8 of 256)
                   │
                   │  actual gating weight: sigmoid(s_i) / Σ sigmoid(s_j)
                   ↓
        ┌────────┬────────┬────────┬────────┐
        ↓        ↓        ↓        ↓        ↓
      E_a      E_b      E_c     ...      E_h         (8 routed experts)
        │        │        │        │        │
        └─ × g ─┴─ × g ─┴─ × g ─┴─ × g ─┘
                          │
                          ↓
                routed_out  +  shared_out  →  y
```

Key numbers:

- **Total** 671B parameters
- **Active per token** 37B parameters
- **MoE layer**: 256 routed experts + 1 shared, each token picks **9 experts** (8 routed + 1 shared)
- **Attention** uses **MLA** (Multi-head Latent Attention), compressing KV to a low-rank latent → KV cache drops ~93%
- **MTP** (Multi-Token Prediction) as an auxiliary objective; can do speculative decode at inference

### 6.2　Auxiliary-Loss-Free Balance (see §2.4 for details)

The V3 paper documents two things combined:

1. **Per-expert bias** $b_i$ (core) — see §2.4
2. **Sequence-wise auxiliary loss** (backup) — a very small weight auxiliary loss to prevent extreme imbalance within a single sequence (not the traditional aux loss across the batch)

The second point is often overlooked: V3 is not "completely without aux loss" but **mainly relies on bias for balance, with a sequence-level minimal aux as backup**. In interviews, saying "V3 has no aux loss at all" is inaccurate.

### 6.3　Node-Limited Routing

V3 training runs 64-way EP across 8 nodes. Naive top-8 routing might scatter a token's 8 experts across all 8 nodes → severe all-to-all communication. V3 adds a hard constraint: **each token routes to at most 4 nodes** and at most 3 experts per node. This is algorithm × system joint optimization.

### 6.4　System: DualPipe + DeepEP

- **DualPipe** (github: deepseek-ai/DualPipe): bidirectional pipeline parallelism, overlapping forward and backward compute/communication to near-zero bubble.
- **DeepEP** (github: deepseek-ai/DeepEP): a communication library specifically optimized for MoE all-to-all, supporting FP8 dispatch + asymmetric inter/intra-node bandwidth.

These two open-source libraries are the engineering foundation behind V3's $5.6M USD MoE training cost on 2048 H800 GPUs.

## §7 Complexity, Memory, Inference Accounting

### 7.1　Training-time FLOPs / parameter comparison

Set the dense baseline to use $D$ hidden, $d_\text{ff} = 4D$, $L$ layers, $T$ tokens. FFN FLOPs (dense):

$$\text{FLOPs}_\text{FFN}^\text{dense} \approx 2 \cdot T \cdot L \cdot D \cdot d_\text{ff} \cdot 2 = 16 T L D^2$$

MoE: replace the FFN with $N$ experts, each with $d_\text{ff}$, each token going through $k$. **Compute only counts activated experts**:

$$\text{FLOPs}_\text{FFN}^\text{MoE} \approx 16 T L D^2 \cdot \frac{k}{N} \cdot \frac{N \cdot d_\text{ff}^\text{expert}}{4D}$$

If $d_\text{ff}^\text{expert} = 4D$ ("each expert equals a complete FFN", Mixtral style), $\text{FLOPs} \propto k$, **independent of $N$** (typical Mixtral 8x7B $k=2$ → compute $\approx$ 2 FFNs).

Parameter count grows linearly with $N$:

$$\text{Params}_\text{FFN}^\text{MoE} = N \cdot 8D^2 + D \cdot N \;(\text{router}) \approx 8 N D^2$$

**Conclusion**: MoE has "parameter capacity $N\times$, compute $k\times$" — exactly why Mixtral 8x7B (47B total / 13B active) outperforms a same-active 13B dense model.

### 7.2　Inference memory (must-know!)

**Common interview trap question: at inference, is MoE memory accounted for by "active parameters" or "total parameters"?**

Answer: **almost by total parameters** — all experts must **reside on GPU**, because the next token could route to any expert. You cannot "load one expert on demand", because PCIe / NVLink loading latency is far higher than expert compute latency.

Specifically, single-GPU inference of a 671B model (FP8 / FP16):

| Model | Storage (FP16) | Storage (FP8) | Fit on a single card? |
| --- | :-: | :-: | :-: |
| LLaMA-2 70B (dense) | 140 GB | 70 GB | H100 80G single card with FP8 OK |
| Mixtral 8x22B (141B) | 282 GB | 141 GB | Single card no, needs 2× H100 / 8×A100 |
| DeepSeek-V3 (671B) | 1342 GB | 671 GB | **8× H100 80G = 640 GB still a bit short**; actual deployment needs **2 nodes 16× H100 80G** or **8× H200 141G** (≈1128 GB) to fit FP8 weights + KV cache |

**Bandwidth benefit**: each token only reads `active_param / total_param` fraction of weights for matmul → **memory bandwidth bottleneck alleviated** (inference is mostly memory-bound). 671B / 37B → each token reads ~5% of weights — this is the source of V3's inference throughput matching dense 30B on H800.

### 7.3　KV Cache

MoE does not directly affect KV cache size — KV is related to attention, not FFN sparsity. But V2/V3 also introduced **MLA** that compresses KV to a low-rank latent, so V3's KV cache is ~5% of LLaMA-3 70B. **In interviews, distinguish "MoE reduces FFN memory (FP8 OK), MLA reduces KV memory"** — these are two independent optimization lines.

## §8 EP / TP / PP / DP: MoE Parallelism's 4-Dimensional Mesh

Dense LLM training commonly uses DP + TP + PP (data / tensor / pipeline) three-dimensional parallelism. MoE must add a 4th dimension: **Expert Parallelism (EP)**.

### 8.1　Expert Parallelism

Slice $N$ experts across different GPUs (e.g., 64 EP, 4 experts per GPU). Forward flow:

```

   Step 1.  Token x [local batch] -> Router -> top-k expert IDs
   Step 2.  ★ all-to-all dispatch:
            Send tokens by expert ID to the corresponding GPU
            (tokens received per GPU ≤ capacity)
   Step 3.  Local expert forward
   Step 4.  ★ all-to-all combine:
            Send expert outputs back to the originating GPU by original token ID
   Step 5.  Gate weighted sum -> next layer
```

Two **all-to-all** operations are the soul of MoE and also its pain point: communication volume $\propto T \cdot k \cdot D$, easily eating half the time.

### 8.2　EP × DP × TP × PP

| Dimension | Splits | Communication |
| --- | --- | --- |
| **DP** | Batch | all-reduce (gradient) |
| **TP** | Hidden / heads (within a layer) | all-reduce (in-layer) |
| **PP** | Layers (different stages) | point-to-point (cross-stage) |
| **EP** | Expert | all-to-all (2 per MoE layer) |

Practical combination (DeepSeek-V3 training):

- 16-way PP × 64-way EP × ZeRO-1 DP
- Single H800 cluster: 8 GPU per node × 256 nodes = 2048 GPUs
- DualPipe makes PP bubbles near zero; DeepEP overlaps EP all-to-all with expert compute

### 8.3　EP communication accounting

Let each token average $k$ experts, $D$ hidden, $G$ EP group size, $T$ tokens per step:

$$\text{all-to-all volume per layer} \approx 2 \cdot T \cdot k \cdot D \cdot \frac{(G-1)}{G}$$

($\times 2$ for dispatch + combine. $(G-1)/G$ because 1/G of tokens hit locally and don't need communication.)

For V3: $T \sim$ few-M tokens/step, $k=8$, $D=7168$, $G=64$ → per-layer communication ~ TB-level. This is why DeepEP / NVLink-aware kernels are so critical.

## §9 Comparison with Related Methods

### 9.1　MoE vs Dense (same FLOPs / same parameters)

| Perspective | Dense | MoE |
| --- | --- | --- |
| Same active params | More stable | Lower capacity ceiling |
| Same total params | Higher compute cost | **MoE more compute-efficient, larger capacity** |
| Training stability | High | Medium (routing collapse / EP comm instability) |
| Inference memory | Total ≈ active | **By total (all experts must reside)** |
| Inference latency | memory-bound | **Memory bandwidth advantage** (active part small) |
| Deployment difficulty | Low | High (needs EP / multi-card) |
| Fine-tuning | Simple | **Complex** (see §9.3 ESFT) |

### 9.2　MoE vs Mixture-of-Depths (MoD)

MoD (Raposo et al. 2024) takes a different approach: **at each layer, pick top-k tokens to go through the whole layer; other tokens skip the entire layer** (sparsify along depth). Compared to MoE which sparsifies along width (each token picks part of experts).

| Dimension | MoE | MoD |
| --- | --- | --- |
| Sparsity direction | width (FFN experts) | depth (whole layer) |
| Compute per token | Fixed (top-k experts) | Dynamic (whether each layer is traversed) |
| Router decision | per-layer per-token | per-layer pick top-k tokens |
| Training stability | Medium | Newer, weaker engineering |

Not mutually exclusive — theoretically can be used together; some works after 2025 (e.g., Llama 4 details, certain Qwen3 experiments) started mixing them.

### 9.3　MoE fine-tuning: ESFT and its peers

Standard LoRA has several pitfalls on MoE:

- LoRA added to $W_q/W_k/W_v$ is at the attention level, **does not touch experts**
- Added to each expert → $N$× LoRA parameters, with some experts not trained (not routed)

**ESFT** (Expert-Specialized Fine-Tuning, Wang et al. arXiv 2407.01906):

- First forward with task data, count router scores
- Find the **task-most-relevant top-k experts** (per layer), freeze the rest
- Train only these "task experts"; **memory ↓ 90%, time ↓ 30%, performance ≈ full-parameter fine-tune**

> 💡 **Interview bonus** — why this is more suitable for MoE than LoRA in essence: MoE is already specialized, and fine-tuning should only specialize further, not stir up general experts. This is the natural extension of "sparse pretrain → sparse fine-tune".

### 9.4　MoE vs MQA/GQA

Completely orthogonal:

- **MQA/GQA** reduces KV cache (attention side)
- **MoE** reduces FFN compute (FFN side)

Can be used together (V2/V3 = MLA + MoE; LLaMA-3 = GQA + dense; theoretically GQA + MoE is also legal).

## §10 25 Frequently-Asked Interview Questions

Split into L1 (10 must-know) / L2 (10 advanced) / L3 (5 top labs). Click to expand each question for answer points + pitfalls.

### L1 must-know (any ML engineering role will ask)

<details>

<summary>Q1. What is MoE doing? Why use sparse activation?</summary>

- Replace dense FFN with $N$ independent experts + router; each token only goes through $k \ll N$
- **Total parameters ↑ (capacity) but active parameters unchanged (FLOPs)** — trade parameters for compute efficiency
- Storage-feasible at training, controllable compute at inference
- Key: MoE reduces FLOPs, but **does not reduce inference memory** (all experts must reside)

Pitfall: thinking of MoE as an ensemble. MoE is **selecting experts on a single forward path**, completely different from "train several models and vote".

</details>

<details>

<summary>Q2. What is the Top-k token-choice MoE routing formula?</summary>

- Router: $s(x) = W_g x \in \mathbb{R}^N$, $g(x) = \text{softmax}(s(x))$
- Select $\mathcal{T}_k(x) = \text{TopK}_i\, g_i$, output $y = \sum_{i \in \mathcal{T}_k} \tilde{g}_i E_i(x)$
- Mixtral/DeepSeek: **renormalize** over top-k ($\tilde{g}_i = g_i / \sum_{j \in \mathcal{T}_k} g_j$) so weights sum to 1
- Router has no bias (standard practice)

Pitfall: just saying "pick the top few experts" without writing the formula; forgetting renormalize; writing the Q/K/V formula instead (attention formula, confused).

</details>

<details>

<summary>Q3. What is the Switch Transformer aux load-balance loss?</summary>

- $\mathcal{L}_\text{aux} = \alpha \cdot N \sum_i f_i P_i$
- $f_i$ = fraction of tokens received by expert $i$ (non-differentiable, but has numerical value)
- $P_i$ = average gate probability of expert $i$ (differentiable)
- The smaller the product, the more uniform the distribution (uniform gives $\sum_i f_i P_i = 1/N$)

Pitfall: just saying "encourage uniformity" without writing the formula; only writing $\sum P_i^2$ (this is an entropy regularizer, not the Switch formula).

</details>

<details>

<summary>Q4. What is the capacity factor? Why is it needed?</summary>

- $C = \lceil \alpha \cdot Tk/N \rceil$, the capacity upper bound for each expert
- $\alpha = 1.25$ is the Switch / GShard default (25% buffer)
- Tokens beyond capacity → **residual bypass** (expert output 0, residual passes through)
- Without a capacity limit, you cannot statically buffer; EP communication cannot be scheduled

Pitfall: just saying "prevent OOM"; not knowing residual bypass and assuming drop means setting to 0.

</details>

<details>

<summary>Q5. What is MoE inference memory accounted for by?</summary>

- **Total parameters, not active parameters** — all experts must reside on GPU
- Because the next token could route to any expert; on-demand loading latency is unacceptable
- 671B MoE FP8 ≈ 671 GB; **8× H100 80G = 640 GB still a bit short**; actual deployment commonly uses 16× H100 80G (2 nodes) or 8× H200 141G
- The real saving is in memory bandwidth (each token only reads the active part of weights)

Pitfall: thinking MoE memory is counted by active, so "Mixtral 13B runs on a single card" — wrong; Mixtral has 47B total parameters, doesn't fit on a 24G card.

</details>

<details>

<summary>Q6. DeepSeek-V3 total parameters / active parameters / expert count?</summary>

- **Total parameters 671B, active 37B / token**
- **256 routed experts + 1 shared expert**
- Top-8 routed + 1 shared = 9 experts per token
- Attention uses **MLA**, FFN uses DeepSeekMoE
- arXiv: 2412.19437, 2024.12

Pitfall: mixing V3 with V2 (236B/21B) numbers; saying V3 has "some number of experts" but misremembering 256.

</details>

<details>

<summary>Q7. Does Mixtral 8x7B really have 7B × 8 = 56B parameters?</summary>

- **No**. The 8x7B naming only indicates "8 experts, each at the 7B scale"
- **Actual total params 46.7B** — because attention / norm / embedding are shared across experts, not replicated per expert
- Active parameters 12.9B (not 14B, because the router gate is a sparse weighted sum)

Pitfall: misled by the name, computing $8 \times 7 = 56$ directly.

</details>

<details>

<summary>Q8. Why does MoE training often have routing collapse? How to cure?</summary>

- Without a balance mechanism, the strong get stronger: selected more → trained more → more likely to be selected
- Treatment: **aux loss (Switch)** / Expert Choice / Aux-loss-free bias (V3) / Z-loss / expert dropout / suitably increase capacity factor
- Monitor: load per expert + aux loss value + validation loss

Pitfall: just saying "add aux loss" without giving the formula; not knowing about hard-constraint methods like expert-choice.

</details>

<details>

<summary>Q9. What parallelism does MoE training need?</summary>

- **DP (data) + TP (tensor) + PP (pipeline) + EP (expert)**
- EP is a new dimension: split $N$ experts across GPUs
- Each MoE layer needs **2 all-to-alls** (dispatch + combine)
- DeepSeek-V3: 16-way PP × 64-way EP × ZeRO-1 DP, across 2048 H800 GPUs

Pitfall: forgetting EP; thinking dense DP+TP+PP is enough.

</details>

<details>

<summary>Q10. What are the benefits of MoE on the inference side?</summary>

- At the same active parameters, **memory bandwidth advantage**: each token reads only ~active/total fraction of weights
- Inference mostly token-by-token is memory-bound; reduced bandwidth → throughput ↑
- 671B / 37B → each token reads ~5% of weights → comparable latency to ~30B dense

Pitfall: just saying "less compute" — but memory cannot accommodate (all experts loaded), so single-card deployment is still hard.

</details>

### L2 advanced (research-oriented roles)

<details>

<summary>Q11. What is Expert Choice routing? Why can autoregressive decoders not use it?</summary>

- Reverse: each expert picks top-$M$ tokens ($M = Tk/N$)
- **Naturally balanced**: each expert strictly picks $M$ tokens, no aux loss needed
- Does not drop tokens (capacity is hard-set)
- **Cannot be autoregressive**: the routing of the $t$-th token depends on the scores of the entire batch (including future tokens) → cannot generate token-by-token
- Used in encoder / vision / BERT-style; decoders use token-choice
- Zhou et al. 2022 NeurIPS, arXiv 2202.09368

Pitfall: thinking it can directly replace Mixtral / DeepSeek's token-choice; forgetting the causal restriction.

</details>

<details>

<summary>Q12. How does the discrete top-k operation backprop gradients?</summary>

- Top-k itself is **non-differentiable and does not participate in the gradient**
- But the gate weight $g_i$ of the $k$ selected experts (after softmax) **enters the output weighted sum**, and is differentiable in $g_i$
- The router weights of unselected experts don't get gradient signal in this step (chicken-and-egg → aux loss / bias is needed to break the loop)
- Similar to hard attention, but because of weighted summation, the gradient signal can still reach the router

Pitfall: thinking "top-k requires Gumbel-softmax to backprop" — actually the standard approach is to use softmax + top-k selection directly, and the gate weight is the gradient path.

</details>

<details>

<summary>Q13. What does fine-grained expert in DeepSeekMoE mean?</summary>

- Split $N$ baseline experts into $mN$ smaller experts (each expert's $d_\text{ff}$ shrunk to $d_\text{ff}/m$)
- Routing count raised from $K$ to $mK$
- **Total compute / total parameters unchanged**, but combinations $\binom{mN}{mK} \gg \binom{N}{K}$
- The "expert combination identity" per token is exponentially richer → specialization improves
- Dai et al. 2024, arXiv 2401.06066

Pitfall: thinking fine-grained = increase the number of experts (partly right) but missing the key "each expert is smaller, routing count is increased proportionally" that keeps the compute constant.

</details>

<details>

<summary>Q14. What is a shared expert? Why is it needed?</summary>

- An expert that is **always active, not routed** (every token goes through it)
- Absorbs "general knowledge" (language / common sense), allowing routed experts to focus on specialization
- DeepSeekMoE / Llama 4 both have it; Mixtral / Switch do not
- DeepSeek-V3: 1 shared expert + 256 routed (8 selected)

Pitfall: thinking the shared expert is an ensemble; or not knowing it bypasses the router (the router does not vote for the shared expert).

</details>

<details>

<summary>Q15. How does DeepSeek-V3 achieve aux-loss-free balance?</summary>

- One bias $b_i$ per expert; top-k selection uses the **biased score** $s_i + b_i$
- But the **gating weight uses the original $s_i$** (keeps the main gradient path clean)
- Update each step by sign: $b_i \leftarrow b_i - u \cdot \text{sign}(c_i - \bar{c}_i)$
- $b_i$ is a buffer, not in the gradient graph, not updated by the optimizer
- Backup with a very small sequence-level aux loss (preventing extreme imbalance within a single sequence)
- Wang et al. arXiv 2408.15664; V3 applies it in 2412.19437

Pitfall: saying "V3 has no aux loss at all" — inaccurate (there's a sequence-wise backup); or using biased score for the gating weight (this would pollute the main gradient, violating the design intent).

</details>

<details>

<summary>Q16. What is MoE training's all-to-all? How to compute communication volume?</summary>

- Each MoE layer does 2 all-to-alls: dispatch (send tokens by expert) + combine (receive back by token)
- Communication per layer ≈ $2 T k D (G-1)/G$ ($G$ is the EP group size)
- Usually 30-50% of training time; DeepEP / DualPipe are engineering weapons to overlap communication with computation
- Larger $G$ → more sparse → comm fraction ↑

Pitfall: just saying "use NCCL for communication"; not understanding that comm scales linearly with $T \cdot k \cdot D$; not knowing DualPipe / DeepEP.

</details>

<details>

<summary>Q17. What problems arise with LoRA fine-tuning on MoE? How does ESFT solve it?</summary>

- LoRA added to attention layers → does not touch experts; added to all experts → $N$× LoRA parameters, and some experts are not trained
- **ESFT** (arXiv 2407.01906): first forward to count router scores → pick the top-$k$ task-relevant experts → only train these
- Performance ≈ full-parameter fine-tune; memory ↓ 90%, time ↓ 30%

Pitfall: not knowing ESFT; thinking "MoE = big model, LoRA solves everything".

</details>

<details>

<summary>Q18. MoE inference loads all experts into memory; where does the benefit of sparse activation go?</summary>

- The benefit transfers to **memory bandwidth** — each token only reads the active fraction of weights
- LLM decode is memory-bound (not compute-bound), so bandwidth reduction translates directly to throughput
- 671B/37B model: each token reads ~5% of weights → throughput ≈ 30B dense
- But memory is still budgeted by total parameters: a single H100 80G **cannot run V3**, needs 8 cards minimum

Pitfall: thinking sparse inference can "load 1 expert on demand" — hardware latency forbids this; or thinking the throughput gain comes from FLOPs savings (actually it comes more from bandwidth).

</details>

<details>

<summary>Q19. Difference between MoE and MoD (Mixture-of-Depths)?</summary>

- MoE sparsifies along **width**: each token picks a subset of experts
- MoD sparsifies along **depth**: at each layer, pick top-$k$ tokens to go through the whole layer; others skip
- Compute per token: MoE fixed (top-k experts), MoD dynamic (depending on how many layers accept it)
- Not mutually exclusive, can be stacked

Pitfall: thinking MoD = MoE's alias; or not knowing one can sparsify along depth.

</details>

<details>

<summary>Q20. Why are Mixtral 8x7B's actual total parameters 46.7B and not 56B?</summary>

- 8 experts are only in the **FFN layer** — attention / norm / embedding / unembedding **are all shared, counted once**
- Set Mistral-7B's dense total params $\approx 7.24\text{B}$ (FFN about $4.8\text{B} \approx 2/3$, shared = attention+embed+norm $\approx 2.4\text{B} \approx 1/3$)
- Mixtral 8x7B total params = **shared part + 8 × FFN** = $2.4 + 8 \times 4.8 \approx 2.4 + 38.4 \approx 40.8\text{B}$; plus Mistral-7B is slightly larger than LLaMA-2-7B and Mixtral adds router weights, official **46.7B**
- Active parameters (per token, top-2) = **shared part + 2 × FFN** = $2.4 + 2 \times 4.8 \approx 12\text{B}$ (paper reports 12.9B / 13B active)

Pitfall: computing $8 \times 7 = 56$ as total params (multiplying attention/embed by 8 too); or computing active as $2 \times 7 = 14$, forgetting shared is counted once.

</details>

### L3 top labs / research directions

<details>

<summary>Q21. Why does DeepSeek-V3's aux-loss-free not break sparse routing semantics?</summary>

- **Key design**: bias $b_i$ **is used only for top-k selection**; it does not enter the gating weight, let alone the gradient graph
- This means: the router's gradient signal **comes only from language-modeling loss**, completely different from traditional aux loss that "treats balance as an extra gradient"
- Analogy with control systems: bias is an outer-loop controller (adjusting by load feedback), the main gradient is an inner-loop optimizer (optimizing expressiveness by task loss) — the two loops are **decoupled**
- Compared to aux loss: the aux loss gradient pulls router weights toward "uniform distribution", which may conflict with "task-optimal routing"
- Loss-free lets the router **stay sparse + specialize under the main task**; bias is just post-hoc correction
- One detail: that bias **does not enter the gating weight** is crucial — otherwise it would secretly influence the gradient through the backward path, making the loss-free name unjustified

Pitfall: treating bias as a learnable parameter fed to the optimizer; thinking loss-free means "adding nothing" (actually there's a sign update and a sequence-aux backup).

</details>

<details>

<summary>Q22. Why does fine-grained + shared outperform same-total-params dense? Theory / empirics?</summary>

- **Capacity argument**: the routed combination count of $mN$ small experts $\binom{mN}{mK}$ greatly exceeds $\binom{N}{K}$; the "expert combination identity" per token is exponentially richer, approaching the limit of "an independent small subnet per token"
- **Specialization argument**: shared experts absorb common knowledge, freeing routed experts from the "both-and" burden, more likely to specialize
- **Compute argument**: at the same active FLOPs, dense must stuff capacity into active parameters; MoE puts capacity into dormant experts and only picks a few when active — total parameters can be pushed to 10x+ without increasing FLOPs
- **Empirics**: DeepSeekMoE 16B vs LLaMA-2 7B: similar FLOPs, close or surpassing on benchmarks (paper Table 3); DeepSeekMoE 145B vs DeepSeek 67B dense: equivalent performance, but 145B MoE uses only ~28% of training FLOPs
- **Failure mode**: if routing isn't good enough (collapse), fine-grained instead degrades (a few small experts dominate, equivalent to dense but less efficient)
- **2025-2026 empirical trends**: V3 pushes this line to the extreme (256 routed), proving fine-grained advantages are significant under 1T-level token training

Pitfall: just saying "more parameters work" — without unpacking "combination capacity ≠ single-parameter capacity"; ignoring that routing quality is the prerequisite for this line to work.

</details>

<details>

<summary>Q23. Can MoE inference latency really compete with dense active? Where's the bottleneck?</summary>

- **Theory**: each token only reads active-fraction weights; under bandwidth-bound, throughput should be proportional to active params
- **Actual caveats**:
  1. **Routing overhead**: router compute + top-k selection + scatter/gather has its own latency; under long context, this is also amortized but significant
  2. **Uneven expert load**: routing is uneven within a single batch → some experts queue up, wave is wasted
  3. **EP communication** (multi-card inference): dispatch / combine all-to-all takes a large fraction
  4. **Memory bandwidth ≠ memory size**: each token reads 5% of weights, but which 5% (which experts) is token-dependent; cache hit rate is poor → actual bandwidth utilization below theoretical upper bound
- **Mixtral 8x7B on single H100 80G**: bandwidth utilization ~70-80% (similar to dense 70B utilization); latency close to dense 13B
- **V3 671B**: under 8-card deployment, inference latency is slightly higher than dense 30B (multi-card communication non-negligible)
- **Conclusion**: theoretically yes; engineering has a 5-30% gap depending on batch size / context length / multi-card topology

Pitfall: just saying "theoretically the same", not knowing about routing / EP comm / cache hit and other hidden costs; or conversely saying "MoE inference is necessarily slow", which is also not fully correct (small batch / single GPU, Mixtral is close to dense 13B).

</details>

<details>

<summary>Q24. What stability problems does MoE training have? How to mitigate?</summary>

- **Router logit blow-up**: in early training, router weights swing wildly; in fp16, logits can overflow → **Z-loss** $\mathcal{L}_z = \beta \sum_t (\log \sum_i e^{s_i(x_t)})^2$ penalizes logit magnitude (proposed in ST-MoE 2022)
- **Routing collapse**: solutions in §4.3 (aux loss / bias / capacity factor / expert dropout)
- **Expert representation drift**: representations between experts drift during training (router co-shapes), potentially causing fine-tune with small data to overfit some experts
- **Aux loss interference**: large $\alpha$ → loss pulls router toward uniform, hurting performance; small $\alpha$ → balance insufficient. **This is the root motivation for loss-free**
- **bf16 vs fp16**: bf16's dynamic range is close to fp32, more suitable for routers; in fp16 router logits easily overflow, requiring z-loss + softmax stabilization
- **Router noise large at small batch**: per-step $f_i$ estimation is noisy, aux loss unstable. In production, micro-batch grad accumulation is commonly used to increase effective batch

Pitfall: just answering "add aux loss" — ignoring z-loss, router numerical stability, bf16 importance, small-batch statistical estimation problems.

</details>

<details>

<summary>Q25. If you were to design the next-generation MoE, what would you change?</summary>

This is an open-ended question with no standard answer. Some directions of interest to 2026 frontier labs + your judging angles:

- **Finer experts**: DeepSeek pushed experts to 256; further to 1024 / 4096? Bottleneck is routing collapse + EP communication; needs new balance methods
- **Dynamic top-k**: each token picks a different number of experts (easy tokens use 1, hard tokens use 4). Combined with MoD's idea, "sparsify width and depth simultaneously"
- **Differentiable routing**: Sinkhorn / hash-based / learned permutation, making top-k partially differentiable, helping the router converge faster
- **Expert sharing across layers**: share part of the expert pool across layers (like ALBERT's cross-layer sharing), further compressing total parameters
- **Better MoE post-training**: what comes after ESFT? At SFT/RLHF stages, the router doesn't necessarily preserve pretrain specialization; needs dedicated alignment schemes
- **MoE × long context**: at 1M context, per-layer routing decisions are multiplied by 1M; the router itself may become a bottleneck → routing reuse / hierarchical routing
- **MoE-as-a-Service**: 671B cannot run on a single card; how can small/medium companies deploy? expert offload / streaming weights / disaggregated expert servers
- **Theoretical analysis**: rigorous quantification of fine-grained vs dense capacity? implicit regularization of routing? convergence guarantee of aux-loss-free? Currently all rarely studied

When answering, **first state one specific direction**, give a clear motivation + technical sketch + possible failure mode; this scores better than vague "I'd add more experts".

</details>

## §A Appendix: Reference List (by time)

Sorted by appearance. Note: **all arXiv IDs have been cross-validated**.

1. **Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.** (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR. arXiv:1701.06538.
2. **Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., Chen, Z.** (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. arXiv:2006.16668.
3. **Fedus, W., Zoph, B., Shazeer, N.** (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR. arXiv:2101.03961.
4. **Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A., Chen, Z., Le, Q., Laudon, J.** (2022). Mixture-of-Experts with Expert Choice Routing. NeurIPS. arXiv:2202.09368.
5. **Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., Fedus, W.** (2022). ST-MoE: Designing Stable and Transferable Sparse Expert Models. arXiv:2202.08906. (Source of Z-loss)
6. **Dai, D., Deng, C., Zhao, C., Xu, R., et al.** (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. ACL. arXiv:2401.06066.
7. **Jiang, A. Q., Sablayrolles, A., et al.** (2024). Mixtral of Experts. arXiv:2401.04088.
8. **DeepSeek-AI** (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.
9. **Wang, Z., Chen, D., Dai, D., Xu, R., Li, Z., et al.** (2024). Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning (ESFT). arXiv:2407.01906.
10. **Wang, L., Gao, H., Zhao, C., Sun, X., Dai, D.** (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. arXiv:2408.15664.
11. **DeepSeek-AI** (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
12. **Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P., Santoro, A.** (2024). Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models. arXiv:2404.02258.
13. **Meta AI** (2025). Llama 4: Scout, Maverick, Behemoth (blog announcement, April 5, 2025).
14. **Qwen Team** (2025). Qwen3 Technical Report. arXiv:2505.09388.
15. **DeepSeek-AI / DualPipe / DeepEP** (2025). Open-source releases at github.com/deepseek-ai/{DualPipe,DeepEP}.

> 💡 **Top-5 most-frequent papers asked in 2026 fall recruiting** (by frequency in interviews): DeepSeek-V3 (2412.19437) > Mixtral (2401.04088) > Switch Transformer (2101.03961) > DeepSeekMoE (2401.06066) > Loss-Free Balance (2408.15664). Mastering these 5 covers 95% of MoE questions.
