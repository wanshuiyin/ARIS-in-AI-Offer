## §0 TL;DR Cheat Sheet

> 💡 **LLM Quantization in 8 sentences** — one page covering interview essentials (see §2–§11 for derivations).

1. **Affine quantization formula**: $q = \mathrm{round}(x / s) + z$, dequantize $\hat{x} = s\,(q - z)$. Symmetric quantization $z = 0$; asymmetric quantization $z$ aligns the zero-point to an integer.

2. **Three granularity levels** (smaller scale-sharing range → higher precision, larger overhead): per-tensor → per-channel → per-group (every $g$ elements in a row/column share a scale, $g = 32 / 64 / 128$ are the mainstream choices).

3. **LLM quantization pain points**: activations have **systematic per-channel outliers** (a few channels have magnitudes $50\text{-}100\times$ larger than average); uniform quantization inevitably collapses. Weight distribution is relatively flat; weight-only quantization (GPTQ / AWQ) is inherently easier than weight+activation.

4. **GPTQ (Frantar 2023 ICLR)**: OBS-derived optimal weight update formula $\delta_{\mathbf{w}} = -\dfrac{w_q - \mathrm{quant}(w_q)}{[H^{-1}]_{qq}}\,[H^{-1}]_{q,:}$, columnwise quantization + Cholesky acceleration + 128-column blocks; 4-bit weight nearly lossless.

5. **AWQ (Lin 2024 MLSys)**: observes "1% salient weights drive most loss"; selects salient channels by activation magnitude; **per-channel scale $s_c$ under $w \to w\cdot s_c$ / $x \to x/s_c$ is mathematically equivalent but reduces quantization error**; grid search $s_c = \mathrm{mean}(|x_c|)^\alpha$.

6. **SmoothQuant (Xiao 2023 ICML)**: **migrates activation outliers to weights**—$Y = (X \mathrm{diag}(s)^{-1})(\mathrm{diag}(s)\,W)$, mathematically equivalent but $X / s$ is much smoother, enabling W8A8.

7. **Low-precision float families**: FP8 (E4M3/E5M2, Hopper), MX (OCP MXFP8/MXFP6/MXFP4, 32-elem block + E8M0 shared exp), NVFP4 (Blackwell B100/B200, FP4 E2M1 + per-16-elem FP8 E4M3 scale + per-tensor FP32 scale). Blackwell tensor cores natively support FP4 matmul.

8. **KV cache quant**: K uses **per-channel** (K's outliers are stable along the channel dim), V uses **per-token** (V outliers vary along the token dim)—the basic design of KIVI / KVQuant. QServe further co-designs the entire W4A8KV4 quantization at the SM89/SM90 kernel level.

## §1 Intuition: why LLMs need quantization, and why it's so hard

### 1.1 The essence of quantization

Maps high-precision floats (FP16 / BF16) to low-bitwidth integers (INT8 / INT4) or low-precision floats (FP8 / FP4), trading off:

- **Memory**: FP16 → INT4 directly $4\times$ compression (70B model from 140 GB → 35 GB, fits on a single card for inference).

- **Memory bandwidth**: the decode phase is memory-bound (reading all weights once per token); halving bitwidth halves latency.

- **Compute**: INT8 / FP8 / FP4 tensor core throughput is $2\text{-}8\times$ higher than FP16 (more dramatic on Hopper / Blackwell).

Cost: **precision loss**. For LLMs, loss mainly comes from two sources—activation outliers and weight boundary effects.

### 1.2 Why is LLM quantization harder than CNN?

CNN-era INT8 PTQ (NVIDIA TensorRT 2017 stack) was almost free: CNN activation distributions are approximately Gaussian; per-tensor calibration suffices. Applying the same method to OPT-66B / LLaMA-70B drops 5-10 PPL points. Reasons:

- **Outlier scaling phenomenon** (Dettmers 2022, LLM.int8()): for models ≥ 6.7B, a few channels (about 0.1%-1%) have activation magnitudes significantly higher than other channels ($50\text{-}100\times$), and **these channels are stably the same set across different tokens / samples**.

- **Outliers steal the scale**: per-tensor scale is determined by max. A few outliers push the scale way up, so most "normal" channels after quantization fall into a narrow integer range (only ±10 / ±127 under INT8), dramatically reducing effective bit width.

- **Per-channel weight easy, per-channel activation hard**: the activation's channel dim (hidden dim, $D$) is the matrix multiplication's inner-product dim (K dim); per-channel scale along the K dim means it can't directly fuse into GEMM; weight's output-channel dim (N dim) is naturally per-channel feasible (no inner products between output dims).

### 1.3 Three mainstream approaches

| Approach | What is quantized | Representative methods | Key trick |
|---|---|---|---|
| **Weight-only PTQ** | W4/W8, A stays FP16 | GPTQ, AWQ, QuIP, GGUF Q4_K | weights are easy to quantize; use calibration data to find optimal quantization error compensation |
| **Weight + Activation PTQ** | W8A8 | SmoothQuant, ZeroQuant, FP8 | must handle activation outliers (migration / rotation) |
| **Weight + Act + KV (low bit)** | W4A8KV4 / W4A4 | QuaRot, QServe, SpinQuant | Hadamard / learned rotation flattens outliers in all directions |

> ⚠️ **decode vs prefill difference** — Decode is memory-bound (KV cache + weight reads dominate); low-bit weight + KV directly reduces latency. Prefill is compute-bound (attention $L^2$ + large-batch GEMM); W4A4 / FP8 only matter here; pure weight-only quantization saves almost no time on prefill (and may even slow down due to dequant overhead).

## §2 Quantization Math Foundations

### 2.1 Affine quantization (uniform / linear quantization)

Maps real $x \in [\alpha, \beta]$ to integer $q \in [Q_\min, Q_\max]$ (e.g., INT8: $[-128, 127]$, INT4: $[-8, 7]$).

$$\boxed{\;q = \mathrm{clamp}\!\left(\mathrm{round}(x / s) + z,\; Q_\min,\; Q_\max\right),\quad \hat{x} = s\,(q - z)\;}$$

- **Scale** $s = (\beta - \alpha) / (Q_\max - Q_\min)$, **zero-point** $z = \mathrm{round}(Q_\min - \alpha/s)$.

- **Symmetric quantization**: $\alpha = -\beta$, forces $z = 0$, dequantization simplifies to $\hat{x} = s\cdot q$. Advantage: no zero-point addition in GEMM; disadvantage: wastes 1 bit of expressive range when distribution is asymmetric.

- **Asymmetric quantization**: keeps $z$, can precisely cover any $[\alpha, \beta]$. Cost: when GEMM expands $W\cdot X$, there are 4 extra terms (cross-zero-point terms); production typically uses "subtract zero-point first, then GEMM" to zero out cross terms.

### 2.2 Quantization error and SNR

Rounding error $\epsilon = x - \hat{x}$ is approximately uniform over $[-s/2, s/2]$, variance $\mathbb{E}[\epsilon^2] = s^2/12$. Signal $\mathrm{Var}(x) = \sigma^2$; quantization SNR:

$$\mathrm{SNR} = 10 \log_{10} \frac{\sigma^2}{s^2/12} = 10 \log_{10}(12) + 20 \log_{10}\frac{\sigma}{s}$$

When INT8 is applied to $\mathcal{N}(0, 1)$ with $\beta = 3\sigma$ truncation, each bit reduction lowers SNR by ~6 dB (removing a bit doubles the quant step → SNR decreases by $20\log_{10} 2 \approx 6$ dB). But for LLMs the actual measure is PPL / task metrics, far more complex than SNR.

### 2.3 Granularity

| Granularity | Scale-sharing range | Memory overhead | Precision |
|---|---|---|---|
| **per-tensor** | one $s$ for the whole tensor | $O(1)$ | worst (one outlier breaks everything) |
| **per-channel** | weight along output channel (row) / activation along hidden (column) | $O(N)$ | medium |
| **per-group** | every $g$ elements share an $s$ (typically grouped along input/K dim) | $O(NK/g)$ | high ($g = 32 / 64 / 128$) |
| **per-token (act only)** | one $s$ per token (one row) of activation | $O(L)$ | high, but computed each step |

> 💡 **Per-group is the industry standard for W4 quantization** — GPTQ / AWQ / GGUF Q4_K default to group_size = 128: on the input dim, every 128 weights share a single scale. Storage overhead (symmetric quantization, no zero-point): W4 + group128 (INT4 quant + FP16 scale per 128 weights) ≈ $4 + 16/128 = 4.125$ bits/weight; group32 ≈ $4 + 16/32 = 4.5$ bits/weight, with higher precision. For asymmetric quantization (an extra INT4 zero-point stored per group), the formula becomes $4 + (16+4)/g$: group128 ≈ 4.156 bits/weight, group32 ≈ 4.625 bits/weight.

### 2.4 Rounding modes

- **Round-to-nearest, ties to even** (RNE): IEEE 754 default, unbiased. LLM quantization typically defaults to this.

- **Stochastic rounding**: $\mathrm{round}(x/s + u)$, $u \sim \mathcal{U}[-0.5, 0.5)$. Unbiased in expectation, suitable for QAT backprop.

- **AdaRound** (Nagel 2020 ICML): treats the rounding decision as a $\{0, 1\}$ binary optimization variable; learned per-layer via a proxy loss ("should we round up or down?"); significantly improves PTQ over RTN.

### 2.5 Concise runnable code: asymmetric INT8 per-channel q/dq

```python
import torch

def quantize_per_channel_asym(x: torch.Tensor, n_bits: int = 8, channel_dim: int = 0):
    """
    Asymmetric per-channel quantization.

    Args:
        x: float tensor, e.g. weight [out_features, in_features]
        n_bits: target bit width (e.g. 8 -> INT8)
        channel_dim: which dim to share scale on (typically 0 for weight rows)

    Returns:
        q_int: int quantized tensor (still stored as int32 here for simplicity)
        scale: [N] float scale per channel
        zero_point: [N] int zero-point per channel
    """
    q_min = -(1 << (n_bits - 1))           # -128 for INT8
    q_max = (1 << (n_bits - 1)) - 1        # 127  for INT8

    # Reduce over all dims except channel_dim
    reduce_dims = [d for d in range(x.dim()) if d != channel_dim]
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)

    # Avoid degenerate (zero range) channels
    eps = 1e-8
    scale = (x_max - x_min).clamp(min=eps) / (q_max - q_min)
    zero_point = (q_min - x_min / scale).round().clamp(q_min, q_max)

    q = (x / scale + zero_point).round().clamp(q_min, q_max).to(torch.int32)
    return q, scale, zero_point.to(torch.int32)


def dequantize_per_channel(q_int: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """ x_hat = scale * (q - zero_point)   (per channel) """
    return scale * (q_int.to(scale.dtype) - zero_point.to(scale.dtype))


# Sanity check
if __name__ == "__main__":
    W = torch.randn(64, 1024) * 0.1
    W[3, :] *= 10.0   # one outlier channel
    q, s, z = quantize_per_channel_asym(W, n_bits=8, channel_dim=0)
    W_hat = dequantize_per_channel(q, s, z)
    err = (W - W_hat).abs().max().item()
    print(f"max abs err = {err:.4e} (should be ≤ max_scale/2)")
```

> ⚠️ **Early PyTorch's `round` is banker's rounding (RNE), inconsistent with CUDA / TensorRT** — when deploying across backends, must use the same rounding. At a genuine tie point ($x/s$ exactly at an integer + 0.5), the two rounding conventions produce integers $q$ that differ by exactly 1 quantization level, which after dequantization is a full quantization step $s$ (1 LSB) — not 0.5 ULP.

## §3 The Outlier Problem in LLMs

### 3.1 Empirical observation (Dettmers 2022, LLM.int8())

Once model scale exceeds 6.7B, in each layer's attention input / FFN input activation, **a few hidden channels' magnitudes are $50\text{-}100\times$ those of other channels**, and:

- **Stability**: the same channel is an outlier in all tokens and all samples (not random).

- **Concentration**: about 0.1%-1% of channels contribute almost all the maximums.

- **Consequence**: nearly all PPL loss from per-tensor INT8 comes from these channels; per-channel weight + per-tensor activation INT8 works for < 6.7B models but collapses on OPT-13B / 66B.

### 3.2 LLM.int8() — the first deployable 8-bit LLM

Dettmers 2022 (NeurIPS) core idea: **Mixed-precision decomposition**. At each layer, split the activation into two paths:

- **Outlier path**: pick outlier channels (threshold $\alpha = 6.0$, i.e., channel-max $> 6$); keep the corresponding weight columns and activation columns **in FP16** for matmul.

- **Normal path**: the remaining 99% of channels use INT8 vector-wise (per-row activation × per-column weight) matmul.

- **Combine**: sum results of both paths.

Mathematically equivalent to splitting the matrix:

$$Y = X W = \underbrace{X_O W_O}_{\text{FP16, outlier cols}} + \underbrace{X_N W_N}_{\text{INT8, rest}}$$

Result: OPT-175B INT8 inference is nearly lossless in PPL, but the outlier path's FP16 GEMM is a throughput bottleneck (~10-15% latency), and the outlier mask must be detected every forward step. This motivated subsequent work (SmoothQuant / AWQ) to switch to "kill the outliers instead of bypassing them".

### 3.3 Different outlier morphologies (the targets of GPTQ / AWQ / SmoothQuant)

| Outlier type | Stable along which dim | Solution |
|---|---|---|
| **Activation channel outlier** (hidden dim) | input channel (K dim) | SmoothQuant (migrate to weight), AWQ (per-channel scale protection) |
| **Token outlier** (a few tokens have large rows) | sequence dim (L dim) | per-token activation quant (zeroquant / SmoothQuant default) |
| **Weight outlier** (a few weights are large) | output dim (N dim) | per-channel weight quant absorbs it |
| **KV cache K-channel outlier** | K's head_dim | per-channel K quant (KIVI / KVQuant) |

## §4 GPTQ: Optimal Weight Quantization Based on OBS (must derive)

GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023) is the industry standard for weight-only PTQ. Its mathematical foundation is **Optimal Brain Surgeon (OBS)** (Hassibi & Stork, NeurIPS 1992), generalizing "how to minimize loss increment after removing/modifying a weight" to "how to update remaining weights to compensate after quantizing one weight".

### 4.1 Problem setup

For a single linear layer output $Y = X W$, $X \in \mathbb{R}^{B\times K}$ from calibration set, $W \in \mathbb{R}^{K \times N}$. Quantization goal: find $\hat{W}$ (each element is INT4 / INT3) minimizing:

$$\min_{\hat{W}} \|X W - X \hat{W}\|_F^2$$

This is a layer-wise reconstruction objective w.r.t. $\hat{W}$. Notes:

- This is **layer-wise**, not end-to-end loss (PTQ assumption: good layer-wise reconstruction → small end-to-end loss increment; basically holds for linear / weakly-nonlinear architectures).

- Since GEMM is independent along the $N$ dim (each column $w_j$ has its own minimization problem), splitting $W$ column-wise is common. The following derivation is for **one column** $w \in \mathbb{R}^K$.

### 4.2 Second-order Taylor expansion

Let $L(w) = \frac{1}{2}\|Xw - Xw^*\|^2$ ($w^*$ is the FP16 original weight, $w$ is to be optimized). Expand at $w = w^*$:

$$L(w^* + \delta) = \underbrace{L(w^*)}_{= 0} + \nabla L(w^*)^\top \delta + \frac{1}{2}\delta^\top H \delta + O(\|\delta\|^3)$$

Since $L$ has global minimum at $w^*$, $\nabla L(w^*) = 0$. Hessian:

$$H = X^\top X \in \mathbb{R}^{K\times K}$$

Note $H$ is independent of $w^*$ (computed once from calibration data, reused across columns) and independent of column index $j$ (every column shares the same $H$). So:

$$L(w^* + \delta) \approx \frac{1}{2}\delta^\top H \delta$$

### 4.3 OBS: optimal $\delta$ after fixing one coordinate to a target value

OBS key question: **forcibly setting the $q$-th component $w_q$ of $w$ to a target value $w_q^{\mathrm{target}}$** (in our case $w_q^{\mathrm{target}} = \mathrm{Quant}(w_q)$); how should other coordinates adjust to minimize $\delta^\top H \delta / 2$?

Constraint $e_q^\top \delta = w_q^{\mathrm{target}} - w_q^* := c_q$ (where $e_q$ is the $q$-th standard basis). Use Lagrangian:

$$\mathcal{L}(\delta, \lambda) = \frac{1}{2}\delta^\top H \delta - \lambda (e_q^\top \delta - c_q)$$

Differentiate: $\nabla_\delta \mathcal{L} = H\delta - \lambda e_q = 0 \Rightarrow \delta = \lambda H^{-1} e_q$.

Plug into constraint $e_q^\top \delta = c_q$:

$$\lambda \cdot e_q^\top H^{-1} e_q = c_q \;\Rightarrow\; \lambda = \frac{c_q}{[H^{-1}]_{qq}}$$

So:

$$\boxed{\;\delta^* = \frac{w_q^{\mathrm{target}} - w_q^*}{[H^{-1}]_{qq}}\; H^{-1} e_q\;}$$

i.e., the $j$-th component of $\delta^*$ = $\dfrac{c_q}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{jq}$. **All other coordinates' optimal compensation comes from $H^{-1}$'s $q$-th column**.

Minimum loss increment:

$$\Delta L^* = \frac{1}{2}\delta^{*\top} H \delta^* = \frac{1}{2}\cdot \frac{c_q^2}{[H^{-1}]_{qq}}$$

> ✅ **GPTQ's optimal weight update (must know)** — after quantizing the $q$-th column, all remaining unquantized columns $j > q$ are updated once by:
$$w_j \leftarrow w_j - \frac{w_q - \mathrm{Quant}(w_q)}{[H^{-1}]_{qq}}\,[H^{-1}]_{jq}$$
then quantize column $q+1$. This is the mathematical formula for GPTQ "iterative columnwise quantization".

### 4.4 Engineering acceleration: Cholesky solves $H^{-1}$ submatrix

Directly computing $H^{-1}$'s column per $q$ costs $O(K^2)$; GPTQ uses Cholesky decomposition $H^{-1} = U^\top U$ ($U$ upper triangular, equivalently $H^{-1} = L L^\top$ if taking lower triangular $L = U^\top$); when sweeping to column $q$, only the $U$ submatrix is needed, and updates can be vectorized.

GPTQ's actual implementation blocks the $K$ columns into **block size = 128**; within each block, quantize and update columnwise, and between blocks apply a lazy-batched sync update once (Cholesky decomposition based), reducing memory traffic. Overall complexity: $O(MK^2 + K^3 + NK^2)$ per layer ($M$ = number of calibration samples, $K$ = input dim, $N$ = output dim; $MK^2$ comes from computing the Hessian $H=X^\top X$, $K^3$ from the Cholesky decomposition, $NK^2$ from propagating the update across all $N$ columns sharing $H^{-1}$); on a 7B model with a single A100, ~30 minutes to fully quantize.

> 💡 **block size ≠ group_size** — the "block size = 128" here (update-batching granularity, a pure engineering speedup that determines how many columns' updates are lazily merged before being written back at once) is a separate concept from the **group_size** mentioned in §2.3 (the quantization scale's sharing granularity) — they just happen to both default to 128 in common implementations, which makes them easy to conflate.

### 4.5 GPTQ-style pseudocode (must-know)

```python
import torch

@torch.no_grad()
def gptq_quantize_layer(
    W: torch.Tensor,             # [N, K]  weight (one linear layer)
    X: torch.Tensor,             # [B, K]  calibration input to this layer
    n_bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
):
    """
    Direct (non-batched) GPTQ for ONE linear weight matrix.

    Hessian H = X^T X is shared across rows of W.
    We quantize columns of W one-by-one (so we walk along K).
    After quantizing column q, propagate the residual error
    along H^-1 to all yet-unquantized columns j > q.

    NOTE: for clarity this directly updates ALL remaining columns
    W[:, q+1:] after every single column (mathematically identical to
    §4.3's OBS update). The real GPTQ implementation instead delays
    the update within a `block_size`-wide window (lazy batched update):
    it only updates columns inside the current block after each step,
    and applies the full cross-block update once per block — same
    result, far fewer memory passes. `block_size` here is an update-
    batching granularity, distinct from `group_size` (the quantization
    scale granularity above), even though both commonly default to 128.
    """
    N, K = W.shape
    device = W.device
    H = X.t() @ X                                              # [K, K]
    # Damping: avoid singular H. Replace zero diag with mean diag * damp_percent.
    mean_diag = torch.mean(torch.diag(H))
    diag = torch.arange(K, device=device)
    H[diag, diag] += damp_percent * mean_diag

    # Cholesky on H^-1: GPTQ uses upper-triangular inverse Cholesky factor.
    Hinv = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)

    Q = torch.zeros_like(W)                                    # quantized W
    W = W.clone()                                              # we will mutate

    q_min = -(1 << (n_bits - 1))
    q_max = (1 << (n_bits - 1)) - 1

    for q in range(K):
        # Per-group scale: when entering a new group, recompute scale on W[:, q:q+group]
        if q % group_size == 0:
            end = min(q + group_size, K)
            w_grp = W[:, q:end]
            s = w_grp.abs().amax(dim=1, keepdim=True) / q_max
            s = s.clamp(min=1e-8)                              # [N, 1]

        w = W[:, q:q+1]                                        # [N, 1]   current column
        # Quantize this column (symmetric, per-row scale `s`)
        q_int = (w / s).round().clamp(q_min, q_max)
        w_q = q_int * s                                        # [N, 1]   dequantized
        Q[:, q:q+1] = w_q

        # OBS-derived weight update on all columns to the right
        err = (w - w_q) / Hinv[q, q]                           # [N, 1]
        W[:, q+1:] -= err @ Hinv[q:q+1, q+1:]                  # [N, K-q-1]

    return Q
```

> ⚠️ **GPTQ engineering pitfalls** — (1) Calibration data amount: typical 128 samples × 2048 tokens; too few makes Hessian ill-conditioned; (2) Damp cannot be skipped, $H$ often has zero diag (some input channels are constantly 0 on the calib set); (3) Activation reorder (`act_order=True`, quantize in descending order of $\mathrm{diag}(H)$) significantly improves W3 / W2 precision but adds dequant indexing overhead at inference; AutoGPTQ defaults to off.

### 4.6 GPTQ vs early round-to-nearest (RTN)

|  | RTN | GPTQ |
|---|---|---|
| Error compensation | None (each weight rounds independently) | Yes (OBS formula propagates error backward) |
| Calibration | Not needed | Needed (128-512 samples) |
| 4-bit on LLaMA-7B | PPL +1.5 | PPL +0.1 |
| 3-bit on LLaMA-7B | PPL +14 (collapse) | PPL +0.7 |
| Time | Seconds | Single card 30-60 min (7B) |

## §5 AWQ: Activation-Aware Weight Quantization

AWQ (Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han, MLSys 2024) core insight: **1% "salient" weights determine nearly all the quantization loss**—these salient weights have large input activation magnitudes. So **don't quantize all weights independently**; first scale up salient channels (before quantization), equivalently scale down the corresponding input, and final quantization error decreases.

### 5.1 Identification of salient channels

Not based on weight magnitudes themselves, but on **magnitudes of corresponding activation channels**:

$$\text{salience}(c) = \mathrm{mean}_{x \sim \mathrm{calib}}\,|x_c|$$

Sort channels by salience; top 1% are "salient channels"; corresponding weight columns $w_{\cdot, c}$ are "salient weights".

### 5.2 Per-channel scale equivalent transform (math equivalence ≠ quantization error equivalence)

Consider $Y = X W$, $X \in \mathbb{R}^{B\times K}$, $W \in \mathbb{R}^{K\times N}$. For each input channel $c$, introduce positive scale $s_c > 0$:

$$Y = (X / S) (S \cdot W) = \tilde{X} \tilde{W}$$

where $S = \mathrm{diag}(s_1, \ldots, s_K)$, $\tilde{X}_{:, c} = X_{:, c} / s_c$, $\tilde{W}_{c, :} = s_c \cdot W_{c, :}$. **Exactly equal under FP16**. But after quantization:

- $\tilde{W} = S W$: salient rows $w_{c, :}$ are multiplied by $s_c$ (grow). **Note**: the protection does not come from "per-channel scale getting smaller, quantization getting finer" — all channels in a group share the same $\Delta$, and amplifying a salient channel can actually push $\Delta$ (the group's absmax) higher. The real protection comes from the inverse transform: after $w \to w\cdot s_c$ (matched by activation $x \to x/s_c$), the salient channel's effective error, measured back on the original scale, is reduced by roughly $1/s_c$, as long as $s_c$ isn't so large that $\Delta$ grows by more than $s_c$ itself.

- $\tilde{X} = X / S$: activation is not quantized (weight-only PTQ doesn't touch activation); but if downstream has activation quant, its error also drops.

**Question**: how to pick $s_c$? Too large makes salient rows stand out and steals the scale; too small can't protect. AWQ gives a grid search:

$$s_c = \mathrm{mean}(|x_c|)^\alpha,\quad \alpha \in \{0.0, 0.1, \ldots, 1.0\}$$

Independently grid search $\alpha$ per layer, minimizing layer-wise MSE $\|Y_{\mathrm{fp}} - Y_{\mathrm{quant}}\|^2$. $\alpha = 0$ reduces to RTN (no scale); $\alpha = 1$ directly uses mean act as scale; typical optimum $\alpha \in [0.4, 0.7]$.

### 5.3 Difference from GPTQ

| Item | GPTQ | AWQ |
|---|---|---|
| Data dependency | Hessian $X^\top X$ (needs calibration) | $\text{mean}(\lvert x \rvert)$ per channel (also needs calibration) |
| Optimization target | Optimal error compensation for all weights | Scale for salient channels |
| Quantization flow | iterative, columnwise + OBS update | one-shot scaling + RTN |
| Time | 30-60 min (7B) | 5-15 min (7B) |
| Inference dequant | May need act reordering | No extra overhead (scale can be absorbed into LayerNorm / W) |
| Compatibility with W reordering | Weak | Strong (scale is elementwise, doesn't break GEMM structure) |

> 💡 **AWQ's engineering friendliness** — Per-channel $s_c$ can be **pre-merged into upstream LayerNorm / RMSNorm weights**: in $\mathrm{LN}(x) \cdot \gamma$, $\gamma \leftarrow \gamma / s$, downstream $w \leftarrow s \cdot w$; at runtime, **no extra elementwise operations**. This is one reason AWQ is easier to deploy than SmoothQuant (SmoothQuant can also be merged, but if LN is followed by cat / residual, it can't). **Note**: this equivalence requires the LN to have no bias (or the bias must also be divided by $s$) — standard LayerNorm with a bias ($\mathrm{LN}(x)=\gamma\odot\mathrm{norm}(x)+\beta$, used by OPT/BLOOM) needs $\beta \leftarrow \beta / s$ as well, not just $\gamma$, to preserve equivalence; RMSNorm is unaffected since it has no $\beta$.

### 5.4 AWQ-style per-channel scale search code

```python
import torch

@torch.no_grad()
def awq_search_scale(
    W: torch.Tensor,             # [N, K]   FP16 weight
    X: torch.Tensor,             # [B, K]   calibration activations (post-LayerNorm)
    n_bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
):
    """
    Search per-channel scale s in [0, 1] that minimizes layer-wise MSE
    of dequantized W·X relative to FP16 W·X.

    s_c = mean(|x_c|) ** alpha,   alpha in {0, 1/n_grid, ..., 1}
    """
    device = W.device
    x_mean = X.abs().mean(dim=0).clamp(min=1e-5)              # [K]

    # Y_fp is the FP16 reference output (compute once)
    Y_fp = X @ W.t()                                          # [B, N]

    q_min = -(1 << (n_bits - 1))
    q_max = (1 << (n_bits - 1)) - 1

    best_alpha, best_err = None, float("inf")
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = x_mean.pow(alpha)
        # Normalize s so that its arithmetic mean is 1: keeps scale of W stable.
        s = s / s.mean()
        s = s.clamp(min=1e-4)

        # Apply: W' = W · diag(s),  X' = X / s
        Wp = W * s.unsqueeze(0)                               # [N, K]
        # Group-wise symmetric quantize Wp along K dim
        Wq = _group_quant_dequant(Wp, n_bits, group_size, dim=-1, q_min=q_min, q_max=q_max)
        # Equivalent activation rescaling: divide each input channel by s
        Xp = X / s.unsqueeze(0)
        Y_q = Xp @ Wq.t()

        err = (Y_fp - Y_q).pow(2).mean().item()
        if err < best_err:
            best_err, best_alpha, best_s = err, alpha, s

    return best_alpha, best_s, best_err


def _group_quant_dequant(W, n_bits, g, dim, q_min, q_max):
    """ Symmetric per-group quant-dequant along last dim (assumed K). Assumes K % g == 0. """
    N, K = W.shape
    assert K % g == 0, f"K={K} must be divisible by group_size={g}; pad W or pick g | K."
    Wg = W.view(N, K // g, g)                                # [N, K/g, g]
    s = Wg.abs().amax(dim=-1, keepdim=True) / q_max
    s = s.clamp(min=1e-8)
    Wq = (Wg / s).round().clamp(q_min, q_max) * s
    return Wq.view(N, K)
```

## §6 SmoothQuant: Migrating Activation Outliers to Weights (W8A8)

SmoothQuant (Xiao, Lin, Seznec, Wu, Demouth, Han, ICML 2023) solves the core pain point of quantizing weights + activations together (W8A8): activation outliers cause per-tensor INT8 to collapse.

### 6.1 Core math

For $Y = X W$, introduce a **diagonal smoothing matrix** $S = \mathrm{diag}(s_1, \ldots, s_K)$, $s_c > 0$:

$$Y = (X S^{-1})(S W) = \hat{X} \hat{W}$$

- $\hat{X}_{:,c} = X_{:,c} / s_c$: activation's outlier channels are flattened.

- $\hat{W}_{c,:} = s_c \cdot W_{c,:}$: weight's corresponding rows are amplified (absorbing outlier magnitudes).

Mathematically **exactly equivalent** (elementwise equal under FP16); but after quantization:

- $\hat{X}$'s max drops significantly; per-tensor INT8 effective bit width improves.

- $\hat{W}$, because weights are **inherently flat-distributed** and **can be per-channel quantized** (along output dim, orthogonal to $S$'s K dim), a few rows being amplified is harmless.

### 6.2 Migration strength $\alpha$ (key hyperparameter)

Optimal $s_c$ should balance "how much activation is smoothed" and "how much weight is amplified":

$$\boxed{\;s_c = \dfrac{\max(|X_{:, c}|)^\alpha}{\max(|W_{c, :}|)^{1 - \alpha}}\;}$$

- $\alpha = 0$: $s_c = 1/\max|W_c|$. $\hat X = X \cdot \max|W_c|$ (activation amplified, quantization harder), $\hat W = W / \max|W_c|$ (weight normalized, quantization easier). **Burden entirely on activation**.

- $\alpha = 1$: $s_c = \max|X_c|$. $\hat X = X / \max|X_c|$ (activation flattened, quantization easier), $\hat W = W \cdot \max|X_c|$ (weight amplified, quantization harder). **Burden entirely on weight**.

- $\alpha = 0.5$ (default): $s_c = \sqrt{\max|X_c| / \max|W_c|}$, split half and half.

SmoothQuant paper scans $\alpha \in [0.3, 0.7]$ on OPT / BLOOM; typically 0.5 works.

### 6.3 Equivalent transform doesn't break GEMM (must know)

>  ✅ **Why SmoothQuant migration doesn't break GEMM** — equivalent transform $Y = X W = (XS^{-1})(SW)$, **$S$ is diagonal (per-channel scale), doing elementwise rescale on columns of $X$ / rows of $W$**:

- Mathematical equivalence: diagonal matrices act as channelwise multiplication, independent of GEMM's inner product order; final output $Y$ is elementwise equal under FP16.

- Engineering: $S^{-1}$ can be fused into **upstream LayerNorm weight** ($\gamma \leftarrow \gamma / s$), $S$ fused into **current weight** ($W \leftarrow SW$, offline one-time); at inference, **no elementwise overhead**. If the upstream LN has a bias ($\beta$), it must also be updated as $\beta \leftarrow \beta / s$ to preserve equivalence; RMSNorm has no $\beta$ so it isn't affected; if the LN output also feeds other downstream consumers that aren't correspondingly rescaled, those need to be handled too.

- Key to not breaking: $S$ is diagonal; rescale is independent per channel along the K dim. If $S$ were dense / rotation matrix, explicit matmul would be needed; QuaRot / SpinQuant below take that direction, at greater cost.

### 6.4 SmoothQuant pseudocode

```python
import torch

@torch.no_grad()
def compute_smooth_scale(
    X: torch.Tensor,             # [B*L, K]  per-token flattened activations (calibration)
    W: torch.Tensor,             # [N, K]    weight
    alpha: float = 0.5,
):
    """
    SmoothQuant migration scale (per input channel c):
        s_c = max(|x_c|)^alpha / max(|w_c|)^(1-alpha)
    """
    x_max = X.abs().amax(dim=0)                             # [K]   per channel
    w_max = W.abs().amax(dim=0)                             # [K]   per input ch of W
    s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
    return s                                                # [K]


@torch.no_grad()
def apply_smoothing(
    W: torch.Tensor,
    s: torch.Tensor,
    prev_ln_weight: torch.Tensor,
    prev_ln_bias: torch.Tensor = None,   # pass this if the upstream LN has a bias (e.g. OPT/BLOOM)
):
    """
    Fuse smoothing into upstream LayerNorm weight (+ bias) and current layer W:
        gamma_new = gamma / s   (so output of LN becomes x / s)
        beta_new  = beta / s    (REQUIRED if LN has bias: LN(x) = gamma*norm(x) + beta;
                                  dividing gamma alone gives (gamma/s)*norm(x)+beta,
                                  which is NOT (gamma*norm(x)+beta)/s unless beta is
                                  also divided by s)
        W_new     = W * s       (broadcasts over output dim of W)
    After this, FP16 forward is identical, but X and W are reshaped
    such that simple per-tensor / per-channel quant works well.
    RMSNorm has no bias (beta), so prev_ln_bias is simply omitted there.
    """
    prev_ln_weight.div_(s)                                  # in-place modify γ
    if prev_ln_bias is not None:
        prev_ln_bias.div_(s)                                # in-place modify β (bias-bearing LN only)
    W.mul_(s.unsqueeze(0))                                  # [N, K] broadcasts s along K dim
    return W
```

### 6.5 SmoothQuant applicability

- ✅ Decoder-only LLMs (OPT, BLOOM, LLaMA family): W8A8 nearly lossless (PPL +0.1).

- ✅ FFN and attention input projection (K, V, Q): smoothing can merge with upstream LN.

- ⚠️ Out projection and down projection: upstream is not LN (is residual / attention output); smoothing requires explicit elementwise op; SmoothQuant defaults to skipping the smoothing preprocessing for these two layers — but that does not mean keeping FP16 input: these layers still get quantized to INT8 (typically simple per-tensor / per-token dynamic quantization, without offline outlier-migration), they just don't get the extra precision protection that smoothing provides.

- ⚠️ INT4 activation: W4A4 alone needs SmoothQuant + QuaRot / SpinQuant rotation.

## §7 Rotation Methods: QuIP / QuaRot / SpinQuant

SmoothQuant uses **diagonal** (per-channel) scale to suppress outliers; but outliers still exist in some channel subspaces. **Rotation methods** use a random / learned **orthogonal matrix** $R$ to "scatter" outliers along the hidden dim, making the distribution closer to Gaussian.

### 7.1 QuIP (Chee et al. 2023, NeurIPS)

Core: use a random **incoherence-inducing matrix** $U$ (e.g., random Hadamard or Householder) to rotate weights, making quantization friendlier.

- **Hadamard transform**: $H \in \mathbb{R}^{d \times d}$, $H_{ij} \in \{+1, -1\} / \sqrt{d}$, orthogonal matrix.

- Key property: after random sign flip, Hadamard transform provably reduces weight incoherence (ratio of max column $\ell_2$ norm to Frobenius norm) to $O(\sqrt{\log d / d})$ level.

- $W' = U W V^\top$ ($U, V$ orthogonal); quantizing $W'$ has smaller loss than $W$ (because incoherent). **The full equivalence relation** is $Wx = U^\top W'(Vx)$: the input must first be rotated by $V$, then $W'$ computes, then the output must be rotated back by $U^\top$ to recover $Wx$ — it is not simply "$W'$ is FP16-equivalent to $W$". In practice $V$ must be absorbed into the previous layer's output weights and $U^\top$ into the next layer's input weights to achieve zero extra runtime overhead.

Cost: at inference, need to keep $U, V$ matmul (one dense rotation). Hadamard transform has a fast algorithm ($O(d \log d)$), but still slower than SmoothQuant's diagonal fuse.

### 7.2 QuaRot (Ashkboos et al. 2024 NeurIPS)

Extends Hadamard to the full LLM stack: **weight + activation + KV cache all INT4**. Core:

- Before each residual stream enters a transformer block, **multiply by a Hadamard $H$**.

- Hadamard is orthogonal; the pure normalization part (without $\gamma$) commutes with $H$ ($\mathrm{rms}(Hx) = \mathrm{rms}(x)$), but $\mathrm{diag}(\gamma)$ generally does not commute with the dense $H$, so passing the $\gamma$-weighted RMSNorm output through $H$ doesn't strictly hold. QuaRot's actual fix is to first fuse $\gamma$ **offline into the next layer's weights** ($W \leftarrow \mathrm{diag}(\gamma) W$), turning RMSNorm into a pure, $\gamma$-free normalization that then commutes exactly with $H$; "online Hadamard" is only used at the few spots where the rotation can't be fused offline (e.g. after a nonlinearity, or the KV cache) — it is not the mechanism that resolves the $\gamma$-noncommutativity itself.

- After rotation, activation has a more Gaussian distribution; **outliers scattered across all dims**; INT4 activation quantization error drops significantly.

Result: LLaMA-2 70B under W4A4KV4 has PPL increment ~ +0.5 (vs SmoothQuant's +5). Cost: 1-2 Hadamard matmuls per block (fast Hadamard transform is cheap on H100).

### 7.3 SpinQuant (Liu et al. 2024 → ICLR 2025, Meta)

Replaces QuaRot's random Hadamard with **learned rotation matrices** $R_1, R_2, R_3, R_4$ acting on residual stream / attention input / FFN input / KV cache. Objective: layer-wise output MSE.

- $R_i \in SO(d)$ (special orthogonal group), optimized via Cayley parameterization or stochastic gradient on Stiefel manifold.

- ~0.5 PPL improvement over QuaRot, but training time increases (~1 GPU hour per model to learn R).

> 💡 **Rotation methods vs SmoothQuant** — Smoothing solves "channel-dim outliers"; rotation solves "channel-subspace outliers". Rotation is more general but engineering cost is higher (dense matmul can't fuse into LN, needs online compute or explicit kernel). For LLaMA-3 / Qwen-2 deployment, W4A8KV4 mainstream is still SmoothQuant + GPTQ; W4A4 needs QuaRot / SpinQuant level rotation.

## §8 Low-Precision Floats: FP8 / MX / NVFP4

Low-precision floats aren't new (FP16 / BF16 are widespread); what's new is **FP8 (E4M3 / E5M2)** with native tensor core support on Hopper, and **MX / NVFP4** as block-scaled floats on Blackwell.

### 8.1 IEEE 754-style float encoding

A float $x = (-1)^s \cdot (1 + m) \cdot 2^{e - \text{bias}}$ (normal) or $x = (-1)^s \cdot m \cdot 2^{1 - \text{bias}}$ (subnormal).

| Format | Sign | Exp bits | Mantissa bits | Bias | Max | Min normal |
|---|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 127 | $\sim 3.4\times 10^{38}$ | $\sim 1.2\times 10^{-38}$ |
| FP16 | 1 | 5 | 10 | 15 | 65504 | $\sim 6.1\times 10^{-5}$ |
| BF16 | 1 | 8 | 7 | 127 | $\sim 3.4\times 10^{38}$ | $\sim 1.2\times 10^{-38}$ |
| **FP8 E4M3** | 1 | 4 | 3 | 7 | 448 (no Inf) | $\sim 1.5\times 10^{-2}$ |
| **FP8 E5M2** | 1 | 5 | 2 | 15 | 57344 | $\sim 6.1\times 10^{-5}$ |
| **FP4 E2M1** | 1 | 2 | 1 | 1 | 6 | 1 |

> ✅ **E4M3 vs E5M2 forward/backward choice** — NVIDIA Transformer Engine default:

- **Forward (activation, weight)**: use **E4M3**—smaller dynamic range but higher resolution (mantissa 3 bits). Activation after layer-wise scale falls within $[-448, 448]$; mantissa precision matters more.

- **Backward (gradient)**: use **E5M2**—larger dynamic range but lower resolution (mantissa 2 bits, similar to FP16). Gradient magnitudes span multiple orders; need large dynamic range.

Note E4M3 has no Inf, only NaN (max normal is 448); E5M2 has Inf and NaN (similar to FP16). This is intentional hardware design distinction.

### 8.2 FP8 E4M3 bit-level encoding code

```python
def fp8_e4m3_encode(x: float) -> int:
    """
    Encode a float into FP8 E4M3 8-bit pattern (returned as int 0..255).
    Format: 1 sign | 4 exponent | 3 mantissa, bias = 7, no Inf, NaN = S.1111.111
    Subnormals: exp = 0, value = (-1)^s * (mantissa/8) * 2^(1-7) = (-1)^s * (m/8) * 2^-6
    Max normal: S.1111.110 -> 448.0
    """
    import math
    if math.isnan(x):
        return 0b0_1111_111
    sign = 0 if x >= 0 else 1
    ax = abs(x)
    if ax == 0:
        return sign << 7
    if ax >= 448.0:
        return (sign << 7) | 0b1111_110                    # saturate to max normal

    # Decompose ax = m * 2^e with m in [1, 2)
    e = int(math.floor(math.log2(ax)))
    m = ax / (2 ** e)                                       # m in [1, 2)

    # Adjust to FP8 E4M3 representation
    biased_e = e + 7                                        # bias = 7

    if biased_e <= 0:
        # Subnormal: shift mantissa right by (1 - biased_e), set exp = 0
        shift = 1 - biased_e
        m_int = int(round((m * 2 ** -shift) * 8))           # 3-bit mantissa
        exp_bits = 0
    else:
        m_int = int(round((m - 1.0) * 8))                   # 3-bit mantissa
        if m_int == 8:                                      # mantissa overflow
            m_int = 0
            biased_e += 1
        if biased_e > 15:                                   # exceeds max exp (biased_e==15 with
                                                             # mantissa 0-6 is still legal: 256-448)
            return (sign << 7) | 0b1111_110                 # saturate
        exp_bits = biased_e

    return (sign << 7) | (exp_bits << 3) | m_int


def fp8_e4m3_decode(b: int) -> float:
    """ Decode an 8-bit FP8 E4M3 pattern (int 0..255) back to a Python float. """
    sign = -1.0 if (b >> 7) & 1 else 1.0
    exp_bits = (b >> 3) & 0b1111
    m_bits = b & 0b111

    if exp_bits == 0b1111 and m_bits == 0b111:
        return float("nan")
    if exp_bits == 0:                                        # subnormal
        return sign * (m_bits / 8.0) * (2 ** -6)
    return sign * (1.0 + m_bits / 8.0) * (2 ** (exp_bits - 7))


# Sanity check: round-trip 448 (max normal)
assert abs(fp8_e4m3_decode(fp8_e4m3_encode(448.0)) - 448.0) < 1e-6
```

### 8.3 MX format (OCP / Microsoft 2023)

OCP (Open Compute Project) MX (Microscaling) spec: groups 32 elements into a block sharing an **8-bit shared scale** (E8M0 format, i.e., power-of-two scale); each element within the block is encoded as FP4/FP6/FP8.

| MX format | Element type | Block size | Shared scale | Total bits/element |
|---|---|---|---|---|
| MXFP8 | FP8 (E5M2 or E4M3) | 32 | E8M0 | $8 + 8/32 = 8.25$ |
| MXFP6 | FP6 (E3M2 or E2M3) | 32 | E8M0 | $6 + 8/32 = 6.25$ |
| MXFP4 | FP4 (E2M1) | 32 | E8M0 | $4 + 8/32 = 4.25$ |
| MXINT8 | INT8 | 32 | E8M0 | $8.25$ |

E8M0 is a 1-byte pure-exponent (no mantissa, no sign) power-of-two scale: $s = 2^{e - 127}$, $e \in [0, 254]$ (255 = 0xFF is reserved for NaN, so E8M0 cannot represent code point 255 other than as NaN, and there is no exact zero scale). Dequant with this scale is a bit shift (cheapest hardware op).

### 8.4 NVFP4 (Blackwell 2025 NVIDIA)

NVFP4 is NVIDIA's FP4 format on Blackwell (B100 / B200 / GB200); differences from OCP MXFP4:

- **Element**: FP4 E2M1 (same as MX).

- **Block size**: **16** (not 32), finer granularity.

- **Per-block scale**: **FP8 E4M3** (instead of E8M0), preserving mantissa; scale itself has higher precision.

- **Per-tensor scale**: an additional global FP32 scale (block scale is FP8, limited to $\pm 448$; per-tensor scale extends dynamic range).

Total bits/element: $4 + 8/16 + \text{negligible per-tensor} \approx 4.5$ bits. Blackwell tensor cores natively support NVFP4 × NVFP4 matmul; throughput on B200 is claimed to be $\sim 8\times$ FP16.

> ⚠️ **NVFP4 ≠ MXFP4** — Industry often confuses "FP4". NVFP4 (block=16, FP8 E4M3 scale, +FP32 tensor scale) is NVIDIA Blackwell proprietary; OCP MXFP4 (block=32, E8M0 scale) is an open spec, partly supported by AMD MI350 / Intel Gaudi 3. The two have different numerical precision and hardware paths.

### 8.5 Using FP8 in large model training (Transformer Engine)

NVIDIA Transformer Engine (TE) standard practice for FP8 training of LLMs on Hopper / Blackwell:

- **Maintain two amax histories per GEMM** (forward / backward), update scale every 8-16 steps.

- **Delayed scaling**: use previous window's amax for scale, avoiding blocking the current step.

- **Hybrid precision**: linear weights and grad accumulators stay in FP32 / BF16; only GEMM inputs cast to FP8. Optimizer state (Adam $m, v$) stays in FP32.

- **Loss scaling**: similar to FP16, but since E5M2 dynamic range is close to FP16, loss scale can be small or omitted.

LLaMA-3 / DeepSeek-V3 series FP8 training on Hopper has ~1.5-2$\times$ throughput improvement over BF16.

## §9 KV Cache Quantization

In LLM inference decode phase, per-sample KV cache memory = $L_\text{ctx} \cdot 2 \cdot n_\text{layers} \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$. LLaMA-2-70B (80 layers, GQA $H_\text{kv}=8$, $d_\text{head}=128$, FP16, $L_\text{ctx}=4096$):

$$4096 \times 2 \times 80 \times 8 \times 128 \times 2\text{B} = 1.34\text{ GB / sample}$$

Batch 64 → 86 GB (one A100 80GB can't fit; KV quantization required).

### 9.1 Key observations of KIVI / KVQuant

Empirically:

- **K cache** outliers **stably appear** along the **head_dim** (channel) dimension (related to RoPE encoding phases; some freq bands have large magnitudes).

- **V cache** outliers appear along the **token** dimension (sequence), and each token is independent.

So optimal granularity:

- **K**: **per-channel** quant (one scale per head_dim).

- **V**: **per-token** quant (one scale per sequence position).

### 9.2 KIVI (Liu et al. 2024 ICML)

KIVI = "K per-channel + V per-token" + INT2 quant, paired with sliding window outlier residual. Pipeline:

1. When K cache updates, compute scale along the head_dim dimension (**per-channel**), quantize to INT2/INT4.

2. When V cache updates, compute scale along the token dimension (**per-token**), quantize to INT2/INT4.

3. Keep the most recent $W$ tokens in FP16 (sliding window), to prevent quant noise from dominating recent attention.

LLaMA-2-7B INT2 KV: PPL increment ~ 0.5; KV cache itself FP16→INT2 has theoretical $8\times$ compression, minus scale / outlier residual overhead, KIVI paper reports peak memory (incl. weight + activation) reduced by ~$2.35\text{-}2.6\times$; batch size can be scaled up ~$4\times$.

### 9.3 KVQuant (Hooper et al. 2024, NeurIPS)

Further analysis:

- Per-channel K isn't enough; **Pre-RoPE quant** is more stable than post-RoPE (RoPE introduces phase mixing, disrupting channel dim structure).

- V uses per-token + non-uniform quant (density-aware).

Result: LLaMA-2-70B INT4 KV PPL increment ~ 0.04.

### 9.4 QServe / QoQ (Lin et al. 2024 MLSys 2025)

QServe introduces **W4A8KV4** full-stack quantization + custom GPU kernels. Key engineering points:

- **W4A8 GEMM**: weight INT4, activation INT8. Dequant path: each weight expanded to INT8 in register via lookup table, then INT8×INT8 matmul, no FP16 dequant overhead.

- **KV4 attention**: K and V both INT4, mixed-precision dot product with INT8 query.

- **QoQ (quattuor-octo-quattuor)**: 4+8+4 naming, 4-bit weight, 8-bit activation, 4-bit KV.

QServe's paper primarily benchmarks on A100 / L40S (with some comparisons including A6000), not H100; it reports a 1.2-3.5$\times$ throughput improvement over vanilla TensorRT-LLM FP16 (e.g. letting a cheaper GPU like L40S match/beat more expensive hardware running TensorRT-LLM FP16 is one of the paper's core selling points).

### 9.5 KV cache quant code illustration (per-channel K, per-token V)

```python
import torch

@torch.no_grad()
def quantize_kv_cache(
    K: torch.Tensor,             # [B, H_kv, L, d_head]
    V: torch.Tensor,             # [B, H_kv, L, d_head]
    n_bits: int = 4,
):
    """
    K: per-channel (head_dim) symmetric quant
    V: per-token   (sequence) symmetric quant

    Returns int8-stored tensors plus scales.
    For real deployment, pack two INT4 values into one INT8 byte.
    """
    q_min = -(1 << (n_bits - 1))
    q_max = (1 << (n_bits - 1)) - 1

    # ---- K: per-channel along the last dim (d_head). Same scale across L, H_kv per-batch. ----
    # Common choice: per (batch, head, channel) scale, reduce over L only.
    s_K = K.abs().amax(dim=2, keepdim=True) / q_max         # [B, H_kv, 1, d_head]
    s_K = s_K.clamp(min=1e-8)
    K_q = (K / s_K).round().clamp(q_min, q_max).to(torch.int8)

    # ---- V: per-token, scale per (batch, head, token) reducing over d_head ----
    s_V = V.abs().amax(dim=-1, keepdim=True) / q_max        # [B, H_kv, L, 1]
    s_V = s_V.clamp(min=1e-8)
    V_q = (V / s_V).round().clamp(q_min, q_max).to(torch.int8)

    return K_q, s_K, V_q, s_V


def dequantize_kv(K_q, s_K, V_q, s_V, dtype=torch.float16):
    K = K_q.to(dtype) * s_K.to(dtype)
    V = V_q.to(dtype) * s_V.to(dtype)
    return K, V
```

> ⚠️ **Pre-RoPE or post-RoPE quantization?** — Academic consensus: **quantize K pre-RoPE** (KVQuant's claim). Reason: RoPE is rotation in frequency bands; it "scatters" channel-dim outliers to other dims, breaking per-channel scale stability. Pre-RoPE, each head_dim's outliers are fixed channels; post-RoPE they vary per token. But pre-RoPE quantization needs in-kernel dequant + then RoPE in the attention kernel, hard to fuse engineering-wise; compromise: post-RoPE but with finer group_size (e.g., 32).

## §10 QAT and Training-Time Quantization

PTQ (Post-Training Quantization) doesn't touch weights; QAT (Quantization-Aware Training) simulates quantization during training or finetuning so the model adapts.

### 10.1 STE (Straight-Through Estimator)

Round / clamp are mathematically non-differentiable (round's derivative is 0 almost everywhere); backprop has no signal. **STE** approximates the quant-dequant function $\mathrm{QDQ}(x) = s\,(\mathrm{clamp}(\mathrm{round}(x/s), Q_\min, Q_\max))$ (symmetric quant example) with gradient:

$$\frac{\partial \mathrm{QDQ}(x)}{\partial x} \;\overset{\text{STE}}{:=}\; \mathbf{1}\!\left[\,s\,Q_\min \le x \le s\,Q_\max\,\right]$$

i.e., "use quantized value forward, pass-through gradient in clipping range backward (saturated regions have zero gradient)". This is the basis for LSQ / DoReFa / PACT and other QAT methods.

### 10.2 LLM-QAT (Liu et al. 2023)

- Use **self-distilled data** (teacher is the FP16 model itself, generating sequences) for QAT, avoiding extra training data.

- Simulate INT4 weight quant in each forward step; backward uses STE.

- Suitable for W4A8 / W4A4 finetune; PPL approaches FP16 after a few thousand steps.

Cost: QAT is 100-1000$\times$ slower than PTQ. In production, PTQ (GPTQ + AWQ) is already good enough; QAT is mainly for < 4-bit (W2A4 / W1.58 ternary, etc.).

### 10.3 FP8 Training (Transformer Engine)

See §8.5. FP8 training is a special case of QAT: training uses FP8 GEMM throughout, scale updated periodically via amax history, loss / opt state stay FP32.

### 10.4 BitNet b1.58 / b2

Recently (Ma et al. 2024) Microsoft released **BitNet b1.58**: weights are ternary $\{-1, 0, +1\}$ ($\log_2 3 \approx 1.58$ bits), activations INT8. Requires from-scratch QAT training (can't be PTQ-converted); 3B scale matches FP16 LLaMA. This is currently the lowest-bit production-ready LLM quantization scheme.

## §11 Frameworks and Ecosystem Comparison

| Framework | Supported methods | Inference backend | Typical use |
|---|---|---|---|
| **bitsandbytes** | LLM.int8(), NF4, FP4 | PyTorch + Triton | HuggingFace transformers integration; essential for QLoRA finetune |
| **AutoGPTQ** | GPTQ (W4, W3, W2) | ExLlama / Marlin kernels | 4-bit inference ~2× FP16 throughput |
| **AutoAWQ** | AWQ (W4) | GEMM kernel | Slightly faster than GPTQ with comparable precision |
| **llama.cpp / GGUF** | Q4_K, Q5_K, Q6_K, Q8_0, Q3_K, IQ2_XXS... | CPU + GPU + Metal + ROCm | First choice for edge inference |
| **TensorRT-LLM** | INT8 SmoothQuant, FP8, W4A8, NVFP4 | NVIDIA fused kernel | First choice for production serving (Hopper / Blackwell) |
| **vLLM** | GPTQ, AWQ, FP8, INT8 SmoothQuant | PagedAttention + custom kernel | First choice for multi-user serving |
| **SGLang** | GPTQ, AWQ, FP8, W4A8 KV | radix-tree + custom kernel | latency-sensitive serving |
| **Transformer Engine** | FP8 training + inference | H100 / B100 cuBLAS | First choice for FP8 training |

> 💡 **Marlin kernel** — Frantar 2024's W4A16 GEMM kernel, designed for Ampere / Ada / Hopper; 4-bit weight + FP16 activation is 1.5-2$\times$ faster than FP16 cuBLAS for batch 1-32. vLLM / SGLang default W4 path.

## §12 25 Frequently-Asked Interview Questions

Curated by codex (gpt-5.5 xhigh) from a top-lab interviewer's perspective, divided into 3 tiers by difficulty. Click each for answer points + common pitfalls.

### L1 Must-Know (any ML engineer position will ask)

<details>

<summary>Q1. The quant / dequant formula for affine quantization?</summary>

- Quant: $q = \mathrm{clamp}(\mathrm{round}(x/s) + z,\; Q_\min,\; Q_\max)$

- Dequant: $\hat{x} = s\,(q - z)$

- Symmetric quant $z = 0$, dequant simplifies to $\hat{x} = s\cdot q$

Reverse round and clamp order, or forget zero-point subtraction.

</details>

<details>

<summary>Q2. Trade-off between symmetric vs asymmetric quantization?</summary>

- Symmetric: $z = 0$, GEMM implementation simple (no cross-zero terms), but wastes 1 bit when distribution is skewed

- Asymmetric: precisely covers any $[\alpha, \beta]$, GEMM needs to fuse out zero-point terms

- LLM weights are approximately zero-mean → symmetric is fine; activations can be skewed (e.g., ReLU output is non-negative) → asymmetric is better

Say "asymmetric is always more precise so always better"; forgets the engineering cost of GEMM cross terms.

</details>

<details>

<summary>Q3. Differences among per-tensor / per-channel / per-group?</summary>

- per-tensor: one scale for the whole matrix, overhead $O(1)$, worst precision

- per-channel: weight has one scale per row along output dim (or activation along hidden dim)

- per-group: one scale per $g$ weights, $g = 32 / 64 / 128$, highest precision

- Storage impact: W4 + group128 ≈ 4.125 bits/weight; group32 ≈ 4.5 bits/weight

Confuse "per-channel along which dim"—weight is safe along output channel (GEMM K dim independent); activation along hidden (K dim) can't fuse directly into GEMM.

</details>

<details>

<summary>Q4. Why is LLM quantization harder than CNN?</summary>

- LLMs ≥ 6.7B show systematic activation outliers (0.1%-1% channels $50\text{-}100\times$ magnitude)

- Outliers stable across different tokens / samples (not noise, but structure)

- per-tensor INT8 OK for small models, drops 5-10 PPL points on large models

- This is the pain point that LLM.int8() / SmoothQuant / AWQ all attack

Say "LLM quantization is same as CNN" or "just larger scale".

</details>

<details>

<summary>Q5. Differences among INT8 / INT4 / FP8?</summary>

- INT8: 8-bit integer $[-128, 127]$, with scale to represent reals

- INT4: 4-bit integer $[-8, 7]$, must use group quant + fine calibration

- FP8 E4M3: 1S/4E/3M, dynamic range $\pm 448$, used for forward

- FP8 E5M2: 1S/5E/2M, dynamic range similar to FP16, used for backward

Treat FP8 as INT8; forget E4M3 has no Inf, only NaN.

</details>

<details>

<summary>Q6. What is GPTQ? How is it better than RTN?</summary>

- GPTQ (Frantar 2023) = efficient OBQ version on LLMs, OBS-based layer-wise PTQ

- Uses Hessian $H = X^\top X$ information; after quantizing column $q$, **updates remaining columns** to compensate

- W4 LLaMA-7B: RTN PPL +1.5; GPTQ +0.1

- Time cost: single card 30-60 min/7B

Only say "GPTQ is 4-bit quantization"; doesn't mention OBS error propagation.

</details>

<details>

<summary>Q7. Core idea of AWQ?</summary>

- 1% salient weights (channels with large input activation) determine most of the quantization loss

- Introduce per-input-channel scale $s_c$, do equivalent transform $W \to s W$, $X \to X/s$

- Grid search $\alpha \in [0, 1]$, $s_c = \mathrm{mean}(|x_c|)^\alpha$

- Faster than GPTQ; works well with Marlin / W4A16 kernel

Only say "AWQ is faster than GPTQ"; doesn't mention activation-aware scale equivalent transform math.

</details>

<details>

<summary>Q8. What does SmoothQuant solve? Why can't W8A8 be quantized directly?</summary>

- Direct W8A8 collapses partly because activation per-tensor scale is dominated by outliers

- SmoothQuant: $Y = (X/S)(SW)$, $S$ diagonal matrix, **equivalent transform**

- $X/S$ is smoothed; $SW$ is still per-channel quantizable

- $s_c = \max|X_c|^\alpha / \max|W_c|^{1-\alpha}$, $\alpha = 0.5$ default

Only say "SmoothQuant is W8A8"; doesn't explain outlier migration math.

</details>

<details>

<summary>Q9. How much memory does KV cache take? How to compute?</summary>

- Formula: $L_\text{ctx} \cdot 2 \cdot n_\text{layers} \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$

- LLaMA-2-70B FP16 4K ctx: $4096 \times 2 \times 80 \times 8 \times 128 \times 2 \approx 1.34$ GB / sample

- Batch 64 → 86 GB; needs KV4 / KV8 to fit

- MQA / GQA makes $H_\text{kv} \ll H$ (70B uses GQA G=8; otherwise vanilla MHA would be 10 GB / sample)

Only say "KV cache is large"; can't compute concrete numbers.

</details>

<details>

<summary>Q10. Difference between bitsandbytes' NF4 and INT4?</summary>

- INT4: uniform quantization, 16 evenly-spaced levels

- NF4 (Normal Float 4): non-uniform, 16 levels chosen by standard normal quantiles

- NF4 assumes weight $\sim \mathcal{N}(0, \sigma^2)$ after normalizing to $[-1, 1]$, expected to be better than INT4

- QLoRA (Dettmers 2023) uses NF4 + double quantization

Say NF4 is "non-integer so slower"—wrong. It's lookup-table-based dequant, speed comparable to INT4.

</details>

### L2 Advanced (research-oriented positions)

<details>

<summary>Q11. How is GPTQ's optimal weight update formula derived from OBS?</summary>

- Second-order Taylor: $L(w^* + \delta) \approx \frac{1}{2}\delta^\top H \delta$ ($H = X^\top X$)

- Constraint $e_q^\top \delta = c_q := \mathrm{Quant}(w_q^*) - w_q^*$ (note sign: $c_q$ is quantized minus original)

- Lagrangian + KKT gives $\delta^* = \lambda H^{-1} e_q$, $\lambda = c_q / [H^{-1}]_{qq}$

- So remaining columns update $w_j \mathrel{+}= (c_q / [H^{-1}]_{qq}) \cdot [H^{-1}]_{jq}$ (equivalent to §4.3's $-(w_q-\mathrm{Quant}(w_q))/[H^{-1}]_{qq}\cdot[H^{-1}]_{jq}$ form)

Memorize formula without derivation; or conflate the dual identity of $H = X^\top X$ — it is both the true Hessian with respect to $w$ and, at the same time, the input Gram matrix; these are the same object, not two competing concepts where one is "wrong"; or forget to propagate error to remaining columns after quantizing column $q$.

</details>

<details>

<summary>Q12. Why doesn't SmoothQuant migration break GEMM? Prove mathematically.</summary>

- $S = \mathrm{diag}(s_1, \ldots, s_K)$ diagonal matrix

- $Y = X W = X S^{-1} \cdot S W = \hat{X} \hat{W}$ — matrix multiplication associativity + absorbing diagonal matrix

- Equivalent: $\hat{X}_{:, c} = X_{:, c} / s_c$, $\hat{W}_{c, :} = s_c W_{c, :}$ (channelwise rescale, doesn't change K dim inner-product structure)

- Engineering: $S^{-1}$ fused into upstream LN weight; $SW$ offline merged once; runtime zero overhead

Only say "diagonal matrix can be fused"; doesn't write out algebraic equivalence of $X S^{-1} \cdot S W$.

</details>

<details>

<summary>Q13. How to choose FP8 E4M3 vs E5M2 for forward / backward? Why?</summary>

- **Forward (W, A)**: E4M3 — 4E/3M, dynamic range $\pm 448$ covers layer-scaled weight/activation; **mantissa has 1 more bit, higher precision**

- **Backward (gradient)**: E5M2 — 5E/2M, same dynamic range as FP16; **gradient magnitudes span $10^{-8}$ to $10^4$ need large dynamic range**

- E4M3 has no Inf only NaN (max normal = 448); E5M2 has Inf + NaN

- NVIDIA Transformer Engine defaults to this assignment

Reverse them (use E4M3 for FP backward) → overflow (gradients often > 448).

</details>

<details>

<summary>Q14. Which is faster: AWQ vs GPTQ? How different is precision?</summary>

- Quantization time: AWQ faster (one grid search $\alpha$, 5-15 min/7B); GPTQ slower (Hessian + Cholesky iteration, 30-60 min/7B)

- Precision: virtually tied on W4 (LLaMA-7B both < +0.2 PPL)

- Inference: AWQ's scale can merge into LN weight, runtime zero overhead; GPTQ with act_order=True has reorder index overhead

- Engineering: AWQ pairs well with Marlin W4A16 kernel; vLLM default W4 path uses AWQ

Say "GPTQ is always more accurate"—wrong, tied on W4; say "AWQ doesn't need calibration"—wrong, needs mean(|x|) per channel.

</details>

<details>

<summary>Q15. INT8 quantization GEMM has cross zero-point terms; how to eliminate?</summary>

- $\hat{x}_a = s_a (q_a - z_a)$, $\hat{x}_b = s_b (q_b - z_b)$

- $\hat{x}_a \hat{x}_b = s_a s_b (q_a q_b - z_a q_b - z_b q_a + z_a z_b)$

- 4 terms after expansion; usually pre-set **weight symmetric quant $z_W = 0$** to eliminate two terms

- Remaining $- z_a \cdot q_b$ term can be **pre-computed with one reduce sum** (per-batch only), subtracted once at inference

Only say "symmetric quantization"; doesn't explain how to handle activation asymmetric cross terms.

</details>

<details>

<summary>Q16. PTQ vs QAT? When to use QAT?</summary>

- PTQ: post-training calibration + closed-form quantization (GPTQ, AWQ, SmoothQuant)

- QAT: simulated quantization during training or finetuning (STE backward)

- LLM industry status: W8 / W4 PTQ already sufficient (< 0.2 PPL loss), QAT not needed

- W2 / 1.58-bit BitNet must be from-scratch QAT; finetune W4A4 often also does QAT

Say "QAT is always more accurate so always use it"—cost is 100-1000$\times$ PTQ, unnecessary for W8/W4.

</details>

<details>

<summary>Q17. Why use different granularities for K and V in KV cache quantization?</summary>

- K outliers stable along **head_dim** dim (specific channels are large), so K uses **per-channel quant**

- V outliers vary along **token** dim (each token has its own magnitude), so V uses **per-token quant**

- KIVI / KVQuant both have this design

- Pre-RoPE quant of K is more stable (post-RoPE scatters channel outliers to different freq bands)

Say "K and V treated same per-tensor"—this is exactly why early KV cache quantization collapsed.

</details>

<details>

<summary>Q18. NVFP4 vs MXFP4?</summary>

- Both are FP4 E2M1 element type (1S/2E/1M)

- **MXFP4** (OCP): block size **32**, shared scale **E8M0** (8-bit pure exponent, i.e., $2^{e-127}$)

- **NVFP4** (NVIDIA Blackwell): block size **16**, shared scale **FP8 E4M3** (with mantissa), **additional per-tensor FP32 scale**

- NVFP4 has finer granularity, higher scale precision, but storage overhead is also larger ($4 + 8/16 \approx 4.5$ bits)

- Blackwell tensor cores natively support NVFP4; MXFP4 needs AMD MI350 / Intel Gaudi 3

Conflate them or say "FP4 is just INT4 with sign"—wrong.

</details>

<details>

<summary>Q19. How does LLM.int8() mixed-precision decomposition work?</summary>

- Each layer splits activation into two paths: outlier path (channel-max > 6, kept in FP16) + normal path (rest, INT8 vector-wise quant)

- Math: $Y = X_O W_O + X_N W_N$, two paths independent GEMM then sum

- Outlier mask detected each forward step (can't be pre-baked)

- First deployable OPT-175B INT8 inference scheme, PPL almost lossless

- Drawback: FP16 outlier path is throughput bottleneck (~10-15% latency); subsequent SmoothQuant / AWQ take "kill outliers" route

Say "LLM.int8() is pure INT8"—wrong, it's mixed-precision.

</details>

<details>

<summary>Q20. How is STE (Straight-Through Estimator) used? Why does it work?</summary>

- Round function has derivative 0 or undefined almost everywhere; backward has no signal

- STE: $\partial \mathrm{Round}(x) / \partial x := 1$ (within clamp range), 0 outside

- Intuition: forward uses quantized value (discrete); backward treats as identity (pass gradient through)

- Biased estimator but works in practice; LSQ (Esser 2020) further learns scale; PACT learns clamp threshold

- Doesn't work when: quantization too aggressive (W2 from scratch); gradient direction deviates significantly from true gradient; needs BinaryConnect-type specialized methods

Say STE is an unbiased estimator—wrong, biased but useful.

</details>

### L3 Advanced variants (top labs / systems direction)

<details>

<summary>Q21. Why can QuaRot / SpinQuant's Hadamard rotation eliminate outliers?</summary>

- Any orthogonal matrix $R$ transforms vector $x$ to $Rx$; $\|Rx\|_2 = \|x\|_2$ unchanged, but $\max|x|$ can drop significantly

- Hadamard $H \in \{+1, -1\}^{d\times d} / \sqrt{d}$: each channel becomes a $\pm$-equally-weighted average of all channels, **concentrated outliers scattered across all dimensions**

- Math: if $x$ has $k \ll d$ outliers, $\ell_\infty$ norm of $Hx$ is approximately $\sqrt{k/d} \cdot \max|x|$ (incoherence property). **Note**: this requires $H$ to be a Hadamard matrix with random sign flips ($HD$), and holds with high probability, not deterministically — a fixed, non-randomized Hadamard matrix can align adversarially with $x$ (e.g. $x$ proportional to one of its rows) and fail to spread the outliers at all; the precise concentration bound also needs an extra $\sqrt{\log d}$ factor: $\|Hx\|_\infty \lesssim \sqrt{k/d}\cdot\max|x|\cdot\sqrt{\log d}$.

- QuaRot passes through RMSNorm (uses online Hadamard to bypass the issue that $\gamma$ doesn't go through $H$); SpinQuant learns $R$ instead of random

- W4A4 LLaMA-2-70B: QuaRot PPL +0.5, significantly better than SmoothQuant's +5

Say "rotation is just PCA dimension reduction"—wrong, orthogonal transform preserves norm, no reduction.

</details>

<details>

<summary>Q22. Why does QServe / QoQ (W4A8KV4) need custom kernels?</summary>

- W4 weight + A8 activation GEMM has no direct path in stock cuBLAS / cuDNN

- QServe kernel: each W4 weight dequant to INT8 in register (lookup table), then INT8×INT8 matmul

- Key optimization: dequant and Tensor Core MMA (`mma.sync`, a fixed hardware instruction with no generic dequant logic built in) are overlapped in time within the same kernel via **software pipelining**, entirely in registers — the next tile's dequant runs concurrently with the current tile's MMA, rather than being fused into a single warp instruction — **avoiding writing/reading an FP16 intermediate buffer**

- KV4 attention: K, V both INT4, mixed-precision dot with INT8 query, needs dequant path inside attention kernel

- End-to-end: QServe's paper primarily benchmarks on A100 / L40S (not H100), 1.2-3.5$\times$ faster than FP16 TensorRT-LLM

Only say "W4A8 is faster than FP16"; doesn't explain why kernel-level co-design is needed (stock GEMM doesn't support W4 input).

</details>

<details>

<summary>Q23. What are amax history / delayed scaling in FP8 training?</summary>

- Each GEMM's input / output maintains an amax history (max abs of recent N steps, typical N = 16)

- Scale computed from history max ($s = \max\text{history} / 448$ for E4M3), ensuring next window doesn't overflow

- "Delayed": use **previous** window's amax to compute **current** window's scale, avoiding blocking forward waiting for amax to be computed

- Cast at GEMM entry: FP32 → FP8 with scale; GEMM output accumulates FP32 then casts out

- LLaMA-3 / DeepSeek-V3 FP8 training ~1.5-2$\times$ throughput over BF16

Treat delayed scaling as same as loss scaling—loss scaling is for backward path anti-underflow; delayed scaling is for per-GEMM forward/backward amax usage.

</details>

<details>

<summary>Q24. Why is NVFP4's per-tensor + per-block FP8 scale two-layer structure necessary?</summary>

- FP4 E2M1 max normal = 6, extremely narrow dynamic range

- Single-layer per-block FP8 scale (E4M3, max 448): within a single block can represent $\pm 6 \times 448 \approx \pm 2700$; but across blocks, still limited by FP8 scale's own range

- LLM activation amax often > 2700 (outlier channels appear at $10^4$ level); per-block FP8 scale isn't enough

- **Additional per-tensor FP32 scale**: pulls the whole tensor into FP8 scale's reasonable dynamic range; equivalent to coarse global tuning + fine block-wise tuning

- Analogy: FP32 uses mantissa+exp for big range; NVFP4 uses FP4 mantissa + FP8 block exp + FP32 tensor exp, three-layer hierarchy

Only say "NVFP4 = FP4 + scale"—wrong, it's FP4 + per-block FP8 + per-tensor FP32 three layers.

</details>

<details>

<summary>Q25. Design a W4 quantization scheme for an unknown LLM; how would you proceed?</summary>

- **Step 1**: Run layer-wise activation profiling; check if ≥ 6.7B has systematic outliers; if so, weight-only must pair with AWQ scale or GPTQ Hessian

- **Step 2**: Pick PTQ method:

  - Simple deployment: AWQ (W4) + Marlin kernel, best precision-engineering trade-off

  - Maximum precision: GPTQ (W4) + act_order, +0.5-1% throughput cost but < 0.1 PPL

  - W4A8 / W4A4: must add SmoothQuant / QuaRot

- **Step 3**: Calibration data choice:

  - General LLM: 128 samples × 2048 tokens from C4 / WikiText

  - Task-specific: use in-domain data; PPL difference is significant (math task on math data is 2-3 PPL better than web data)

- **Step 4**: Pick group_size:

  - W4 group128 default (4.125 bits/weight, good precision)

  - W3 / W2 must use group32 or smaller

- **Step 5**: Validation: run PPL + downstream task (MMLU / GSM8K); PPL diff < 0.2 + task drop < 1% counts as PASS

Only say "use GPTQ 4-bit"—doesn't explain how to choose calibration / group_size / kernel backend.

</details>

## §A Appendix: Complete Engineering Reference

### A.1 Major paper reference list

| Method | Paper | Key contribution |
|---|---|---|
| LLM.int8() | Dettmers et al., NeurIPS 2022 | Mixed-precision INT8 decomposition for outlier |
| GPTQ | Frantar et al., ICLR 2023 | OBS-based W4 PTQ for LLM |
| SmoothQuant | Xiao et al., ICML 2023 | Migrate activation outlier to weight (W8A8) |
| AWQ | Lin et al., MLSys 2024 | Activation-aware per-channel scale (W4) |
| QuIP | Chee et al., NeurIPS 2023 | Random Hadamard for weight incoherence |
| QuaRot | Ashkboos et al., NeurIPS 2024 | Full Hadamard rotation, W4A4KV4 |
| SpinQuant | Liu et al., ICLR 2025 (Meta) | Learned rotations $R_1$-$R_4$ |
| OmniQuant | Shao et al., ICLR 2024 | Learnable equivalent transforms |
| KIVI | Liu et al., ICML 2024 | per-channel K + per-token V, INT2 KV |
| KVQuant | Hooper et al., NeurIPS 2024 | Pre-RoPE quant K, non-uniform V |
| QServe / QoQ | Lin et al., MLSys 2025 | W4A8KV4 GPU kernel co-design |
| LLM-QAT | Liu et al., 2023 | Self-distillation QAT for W4 |
| BitNet b1.58 | Ma et al., 2024 | Ternary weight LLM from scratch |
| FP8 Training | Micikevicius et al., 2022 | E4M3 forward / E5M2 backward |
| MX formats | OCP / Microsoft, 2023 | Block-scaled FP4/6/8 with E8M0 |
| NVFP4 | NVIDIA Blackwell, 2025 | FP4 E2M1 + FP8 E4M3 block + FP32 tensor scale |

### A.2 At-a-glance: which quantization scheme to pick

```

  ┌─────────────────────────┐
  │ Can accept PPL +0.5?    │
  └────────┬────────────────┘
           │
     ┌─────┴─────┐
     │ Yes        │  No
     ↓           ↓
 [W4 weight-only]   [INT8 / FP8 W+A quant]
 GPTQ / AWQ          SmoothQuant W8A8
 + Marlin kernel     + per-channel W
 typical PPL: +0.1   PPL: +0.05

  ┌─────────────────────────┐
  │ Doing INT4 activation?  │
  └────────┬────────────────┘
           │
     ┌─────┴─────┐
     │ Yes        │  No
     ↓           ↓
 [W4A4]            [W4A8KV4 / QoQ]
 QuaRot / SpinQuant  AWQ + KV4
 + Hadamard rotation use QServe kernel
 PPL +0.5 (70B)      PPL +0.2
```

### A.3 Quantization quick-reference card

| Task | Recommendation | Framework |
|---|---|---|
| Single-card LLaMA-2-70B inference | AWQ W4 + Marlin (vLLM / SGLang) | AutoAWQ |
| Multi-card LLaMA-3-405B inference | SmoothQuant + GPTQ W4A8 + KV8 | TensorRT-LLM |
| Edge (Apple Silicon / CPU) | GGUF Q4_K_M / Q5_K_S | llama.cpp |
| Training fp8 LLM | TE FP8 (E4M3/E5M2) + amax history | Transformer Engine |
| QLoRA finetune | NF4 + double quant + LoRA | bitsandbytes + peft |
| Max throughput H100/B200 serving | QoQ W4A8KV4 / NVFP4 | QServe / TensorRT-LLM |
| Edge < 100 MB model | BitNet b1.58 (1.58-bit) from scratch | custom / bitnet.cpp |

### A.4 Sanity check checklist

Before deploying any quantized model, must run:

- [ ] **PPL on calibration domain**: pre-quant vs post-quant diff < 0.2 = OK

- [ ] **PPL on out-of-domain** (important!): diff < 0.5 = OK; > 1 means re-choose calibration data

- [ ] **MMLU / GSM8K and other downstream tasks**: drop < 1% = OK

- [ ] **Long context PPL** (best test of KV cache quant): 4K / 8K / 32K three-tier comparison

- [ ] **Subjective generation quality**: blind eval 50 prompt outputs by humans; should not be significantly worse than FP16

- [ ] **Throughput / TTFT (Time To First Token)**: measured vs theoretical numbers diff > 30% means kernel not fully optimized

- [ ] **Peak memory**: measured vs nominal $\text{bits} \cdot \text{params} / 8 + \text{KV cache} + \text{activations}$

**Quantization Quick Reference** · Main references: Dettmers 2022 (LLM.int8()), Frantar 2023 (GPTQ), Xiao 2023 (SmoothQuant), Lin 2024 (AWQ), Ashkboos 2024 (QuaRot), Lin 2025 (QServe). Last updated: 2026-05.
