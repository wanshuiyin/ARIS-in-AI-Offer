## §0 TL;DR Cheat Sheet

> 💡 **8 sentences to nail Long Context** — get the interview core points in one page (see §2-§9 for full derivations).

1. **RoPE**: apply position-$m$-dependent 2D rotations on each pair of dimensions $(2i, 2i+1)$, with $\theta_i = 10000^{-2i/d}$. $q_m^\top k_n$ depends only on the **relative position** $m-n$ (not on absolute $m, n$ separately), and requires no trainable parameters.

2. **PI (Position Interpolation, Chen 2023)**: divide all $\theta_i$ by $s = L_\text{new}/L_\text{train}$ (equivalent to compressing the absolute position $m$ to $m/s$). **Damages high frequencies** (the phase resolution of early dimensions is compressed), but is simple to implement.

3. **NTK-aware (bloc97 2023)**: change the base; new base $b' = b \cdot s^{d/(d-2)}$. **Low-frequency dimensions are heavily compressed while high-frequency dimensions are almost unchanged**, so zero-shot extrapolation is better than PI.

4. **YaRN (Peng 2023)**: NTK-by-parts (segment-wise frequency handling) + temperature scaling (the fitted formula $\sqrt{1/t} \approx 0.1\ln s + 1$, i.e., $t \approx 1/(0.1\ln s + 1)^2$) + attention scale. The three components respectively solve: handle high/low frequencies separately, dilute the softmax, and compensate for post-extrapolation attention entropy inflation.

5. **LongRoPE (Ding 2024 ICML)**: evolutionary search for an independent scaling factor $\lambda_i$ per dimension, plus a short-context "rescue", pushing context to 2M tokens.

6. **MLA (DeepSeek-V2)**: $\mathbf{c}_t^{KV} = W_\text{DKV}\mathbf{h}_t$ compresses K/V to a latent of dimension $d_c \ll N_h d_h$; RoPE must be **decoupled** — keep a separate $d_h^R$-dimensional RoPE key (shared across heads), otherwise the rotation matrix cannot be "absorbed" into $W_\text{UK}$.

7. **Streaming + Sink (Xiao 2024 ICLR)**: keep the first 4 tokens (attention sink, the softmax "trash bin") + a sliding window; tokens outside the window are dropped, but the sink cannot be dropped, otherwise PPL blows up.

8. **System**: Ring Attention / Context Parallel chunk K/V across devices; FlashAttention 2/3 blocks + online softmax; Mistral SWA reduces each layer's receptive field from $L$ to $W$ (multi-layer stacking still sees far).

## §1 Why Long Context Is Hard — One-Paragraph Intuition

Pushing a Transformer to 100K-2M token context is hard because **three things happen at once**:

- **Position-encoding extrapolation**: training only saw $m \in [0, 4096)$, but inference uses $m = 100{,}000$, and the model must know this is "very far" rather than numerically broken. **RoPE by default does not extrapolate**: unseen rotation phases make the relative-position signal of $q_m^\top k_n$ fail.

- **KV cache memory**: in autoregressive decode, $\text{cache} \propto L \cdot n_\text{layers} \cdot 2 \cdot N_h \cdot d_h \cdot \text{bytes}$. For LLaMA-2-7B (32 layers, $N_h=32, d_h=128$, fp16, MHA), one 4K context $\approx 2.1$ GB, one 100K context $\approx 52$ GB, doesn't fit on a single card. MQA/GQA reduces $N_h \to G$; MLA reduces $N_h d_h \to d_c$.

- **Attention's intrinsic $O(L^2)$**: at $L=100\text{K}$, $L^2 = 10^{10}$, and the score matrix doesn't fit. Two routes: **algorithmic sparsification/linearization** (sliding window, sparse attention, linear attention) or **system-level partitioning** (Ring Attention, Context Parallelism, FlashAttention blocking).

> ⚠️ **One-sentence way to tell extension methods apart** — RoPE family (PI / NTK / YaRN / LongRoPE) solves "position encoding extrapolation"; MLA / MQA / GQA solves "KV cache memory"; FlashAttention / Ring / SWA / Sink solves "attention time and memory". **The three are orthogonal**, and production-grade long-context models (e.g., DeepSeek-V2, Qwen2.5-1M, Llama-3.1-405B) typically use all three classes simultaneously.

## §2 RoPE — Rotary Position Embedding

### 2.1 Complex-number perspective derivation

**Goal**: find a position-dependent transformation $f(\mathbf{x}, m)$ for query/key, such that the inner product $\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle$ depends only on the **relative position** $m - n$ (and on the content $\mathbf{q}, \mathbf{k}$ themselves), no longer on absolute $m, n$.

Group $\mathbf{x} \in \mathbb{R}^d$ into adjacent pairs to form $d/2$ complex numbers: $\mathbf{x} \leftrightarrow [x_0 + ix_1, x_2 + ix_3, \dots] \in \mathbb{C}^{d/2}$. Define

$$f(\mathbf{x}, m) = \mathbf{x} \odot e^{im\boldsymbol\theta}, \quad e^{im\boldsymbol\theta}_i = e^{im\theta_i}, \quad \theta_i = b^{-2i/d}\ (b = 10000)$$

where $\odot$ is element-wise complex multiplication. By complex multiplication:

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle_\mathbb{R} = \mathrm{Re}\!\left[(\mathbf{q} \odot e^{im\boldsymbol\theta})^* (\mathbf{k} \odot e^{in\boldsymbol\theta})\right] = \mathrm{Re}\!\left[\sum_{i=0}^{d/2-1} \bar{q_i} k_i \cdot e^{i(n-m)\theta_i}\right]$$

Depends only on $n - m$ (and on $\bar{q_i}k_i$, i.e., the content term); **the absolute position term cancels** — this is the fundamental reason RoPE gives relative positions.

> ✅ **Geometric intuition** — think of each pair of dimensions $(x_{2i}, x_{2i+1})$ as a vector in a 2D plane; RoPE rotates each 2D subspace by angle $m\theta_i$ (different $i$ rotate at different rates). After rotating both query and key and taking the inner product, the **relative angle** is preserved, the absolute direction cancels.

### 2.2 Real-matrix form

On each pair of dimensions, this is a 2D rotation matrix:

$$R_{\theta_i, m} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$$

Viewing $\mathbf{x}$ as a concatenation of $d/2$ 2D vectors, the overall $R_m = \mathrm{blkdiag}(R_{\theta_0, m}, \dots, R_{\theta_{d/2-1}, m})$. Then:

$$\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{n-m} \mathbf{k}$$

The last step uses $R_m^\top R_n = R_{n-m}$ (additivity of 2D rotations). **The relative position $n - m$ is explicitly encoded into the inner product**.

### 2.3 Why $\theta_i = 10000^{-2i/d}$ (frequency distribution)

Treat $\theta_i$ as angular velocity. Larger dimension $i$ means smaller $\theta_i$ and **slower** rotation (low frequency); smaller dimension $i$ (close to 0) means $\theta_i$ close to 1 and **faster** rotation (high frequency).

- **High-frequency dimensions**: short period ($2\pi/\theta_i$ short), phase is sensitive to position changes — encodes fine-grained local relative positions
- **Low-frequency dimensions**: long period (maximum $2\pi \cdot 10000$), phase changes slowly with position — encodes coarse long-range positions

This **geometric-progression frequency distribution** matches Vaswani 2017 sinusoidal PE (not a coincidence: sinusoidal PE also uses $10000^{-2i/d}$), letting the model resolve positions at multiple time scales simultaneously.

> 💡 **Wavelength vs training context** — the wavelength of dimension $i$ is $\lambda_i = 2\pi b^{2i/d}$. When $\lambda_i$ **exceeds the training length** $L$, that dimension has not seen a complete period during training — this is the key observation behind NTK-by-parts: phase interpolation on low-frequency dimensions is risky (extrapolation enters unseen regions), while high-frequency dimensions are safe.

### 2.4 RoPE code from scratch

> 📝 **Note** — all subsequent Python code blocks share this block's import preamble (`import torch`, etc.); when copying and running a later block in isolation, add `import torch` yourself.

```python
import torch

def precompute_rope_cache(seq_len: int, dim: int, base: float = 10000.0, device=None):
    """
    Returns cos / sin tensors, shape [seq_len, dim/2]; pairs of dimensions share a rotation angle.
    dim must be even (RoPE rotates pairs of adjacent dimensions).
    """
    assert dim % 2 == 0, "RoPE dim must be even"
    half = dim // 2
    # θ_i = base^{-2i/dim}, i = 0, 1, ..., dim/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()              # [L]
    freqs = torch.outer(pos, inv_freq)                              # [L, dim/2]
    return freqs.cos(), freqs.sin()                                 # [L, dim/2] each

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   [..., L, dim]              (Q or K)
    cos: [L, dim/2]  sin: [L, dim/2]
    Real-form implementation: split x into two halves x1, x2 corresponding to the real/imag parts
    of complex numbers, do 2D rotation.
    Convention (HuggingFace LLaMA style): pair = (x[..., :half], x[..., half:])
        rather than (x[..., 0::2], x[..., 1::2]).
    Mathematically equivalent (depending on convention; the two are merely a different permutation).
    """
    x1, x2 = x.chunk(2, dim=-1)                                     # each [..., L, dim/2]
    # Rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)

# Full pipeline example ——————————————————————————————————————
def rope_attention(Q, K, V, cos, sin, mask=None):
    """
    Q, K, V: [B, H, L, d_head]
    cos, sin: [L, d_head/2]  (broadcastable)
    """
    Q = apply_rope(Q, cos, sin)
    K = apply_rope(K, cos, sin)
    scores = (Q @ K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ V
```

> ⚠️ **Complex vs real implementation differences** — Meta's official LLaMA repo uses the **complex view** (`torch.view_as_complex`); HuggingFace transformers uses the **real chunk form** (the code above). HF's "front half / back half" convention is **merely a permutation** of the original paper's "even/odd interleaved" convention; the final attention output is **mathematically equivalent**. **But the RoPE cache precomputation and your pairing choice must be consistent** — mixing them causes rotations to act on wrong dimensions, with near-random results. This bug has actually appeared across HuggingFace `LlamaRotaryEmbedding` version changes.

## §3 Recap of Naive Position Encodings (comparison baseline)

| Method | Form | Relative position? | Extrapolation | Where used |
| --- | --- | --- | --- | --- |
| **Sinusoidal absolute** (Vaswani 2017) | $\mathrm{PE}_{m, 2i} = \sin(m / 10000^{2i/d})$, added to input | No (absolute) | Poor (model has not seen extrapolation region) | original Transformer |
| **Learned absolute** | Treat position as a token, look up an embedding table | No | **Very poor** (embedding table fixed length) | BERT, GPT-2 |
| **Relative bias** (T5) | A learned bias added to logits (bucketed by relative distance) | Yes | Moderate (saturates outside buckets) | T5 |
| **ALiBi** (Press 2022) | $\text{score}_{ij} - m_h \cdot \lvert i - j \rvert$, head-dependent slope $m_h$ | Yes | **Good** (linear bias extrapolates naturally) | BLOOM, MPT |
| **RoPE** (Su 2021/2024) | $q_m \to q_m e^{im\boldsymbol\theta}$ rotation | Yes | Moderate (default); with NTK/YaRN can push to 100K-2M | LLaMA-1/2/3, Mistral, Qwen, DeepSeek |
| **NoPE** (Kazemnejad 2023) | No position encoding at all | Indirectly via causal mask | Surprisingly OK (decoder-only small-model setting) | research curiosity |

> 💡 **Why the community converged on RoPE** — three points in one: (1) no trainable parameters (vs learned absolute), (2) explicit relative position (vs sinusoidal), (3) simple implementation and multi-head compatible (each head rotates independently). ALiBi extrapolates better but is slightly less expressive (only monotonic distance decay); RoPE lets the model learn complex position-content coupling itself.

## §4 PI — Position Interpolation (the simplest RoPE extrapolation)

### 4.1 Motivation

In training, $m \in [0, L_\text{train})$; at inference, $m \in [0, L_\text{new})$ with $L_\text{new} > L_\text{train}$. **RoPE naively extrapolating crashes**: when $m \theta_i$ exceeds the phase range seen during training (in particular when $m\theta_i$ on low-frequency dimensions approaches $2\pi$), the phase enters an unseen region and attention behavior becomes unpredictable.

PI (Chen et al., Meta, 2023, "Extending Context Window of LLMs via Position Interpolation"): **don't extrapolate, interpolate**. Linearly compress $m \in [0, L_\text{new})$ to $[0, L_\text{train})$.

### 4.2 Form

Let the scaling factor $s = L_\text{new} / L_\text{train}$. Replace the absolute position $m$ with $m / s$:

$$\text{PI:}\quad f(\mathbf{x}, m) = \mathbf{x} \odot e^{i (m/s) \boldsymbol\theta}$$

Equivalently (the more common implementation): keep $m$ unchanged and replace all $\theta_i$ with $\theta_i / s$. The two statements are **fully equivalent**.

### 4.3 Side effect: high frequencies are damaged

On low-frequency dimensions, $m\theta_i$ is originally $\ll 2\pi$ within training length (it has not completed one period), so compressing to $m\theta_i / s$ is still in a reasonable range. **The problem is on high frequencies**: high-frequency dimensions have $\theta_i \approx 1$, and during training $m\theta_i$ already rotates freely in $[0, L_\text{train}]$; compressing to $1/s$ **drops the relative-position resolution by $s\times$** — originally the phase difference between adjacent tokens was $\theta_i$ (near 1 rad), now only $\theta_i/s$, and the model's ability to distinguish "1 token apart vs 2 tokens apart" degrades.

> ⚠️ **Must fine-tune to recover** — when used zero-shot in the original paper, PI causes PPL to worsen; about 1000 fine-tuning steps essentially recovers and stably extends to 32K context.

### 4.4 PI code

```python
def precompute_rope_cache_pi(seq_len: int, dim: int,
                              base: float = 10000.0,
                              scale: float = 1.0,        # s = L_new / L_train
                              device=None):
    """PI: divide θ_i by s (equivalent to compressing m to m/s)"""
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    inv_freq = inv_freq / scale                         # ← PI's key line
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()
```

## §5 NTK-aware RoPE — Base-Swap Approach That Preserves High Frequencies

### 5.1 Origin and intuition

PI also flattens high-frequency dimensions, which the community considered too crude. **bloc97 / jquesnelle** proposed NTK-aware scaling ("NTK-Aware Scaled RoPE") in a LocalLLaMA reddit post in July 2023; the name comes from the "high frequency vs low frequency" perspective in Neural Tangent Kernel theory: neural networks learn high-frequency signals slowly, so damaging high frequencies harms the model more than damaging low frequencies.

**Core idea**: **change the base** instead of uniform scaling — let high-frequency dimensions remain almost unchanged (protecting fine-grained position), let low-frequency dimensions be heavily compressed (these dimensions had not seen complete periods during training, so the impact is small).

### 5.2 Derivation: what base change compresses low frequencies to $1/s$?

Recall RoPE frequency $\theta_i = b^{-2i/d}$.

- **Highest frequency** ($i = 0$): $\theta_0 = 1$
- **Lowest frequency** ($i = d/2 - 1$): $\theta_{d/2-1} = b^{-(d-2)/d} \approx b^{-1}$ (for large $d$)

PI divides all $\theta_i$ by $s$, equivalent to discounting the position resolution of all dimensions by factor $s$.

NTK-aware: change the base from $b$ to $b'$, such that **the lowest frequency** $\theta$ is compressed to $1/s$ and **the highest frequency** $\theta$ is almost unchanged.

Let $b' = b \cdot \alpha$. The new lowest frequency is

$$\theta'_{d/2-1} = (b')^{-(d-2)/d} = b^{-(d-2)/d} \cdot \alpha^{-(d-2)/d}$$

To make $\theta'_{d/2-1} = \theta_{d/2-1} / s$, we need

$$\alpha^{-(d-2)/d} = 1/s \quad\Longrightarrow\quad \alpha = s^{d/(d-2)}$$

so

$$\boxed{\;b' = b \cdot s^{\,d/(d-2)}\;}$$

**Verify the highest frequency**: $\theta'_0 = (b')^0 = 1 = \theta_0$, completely unchanged. $\checkmark$

**Asymptotics**: on dimension $i$, $\theta'_i / \theta_i = \alpha^{-2i/d} = s^{-2i/(d-2)}$. At $i = 0$ the ratio is 1 (unchanged); at $i = d/2-1$ the ratio is $1/s$ (heavily compressed). The compression ratio **exponentially transitions** from high to low frequencies — this is the geometric meaning of "NTK-aware".

### 5.3 Comparison with PI

| Dimension | PI | NTK-aware |
| --- | --- | --- |
| **Highest frequency ($i=0$) scaling** | $1/s$ (broken) | **$1$** (preserved) |
| **Lowest frequency scaling** | $1/s$ | $1/s$ |
| **Middle dimensions** | uniformly $1/s$ (linear) | $s^{-2i/(d-2)}$ (exponential transition) |
| **Zero-shot PPL** ($s=4$ on LLaMA-7B) | greatly worsens | close to original PPL |
| **Need fine-tuning** | yes | no (zero-shot usable) |

### 5.4 NTK-aware code

```python
def precompute_rope_cache_ntk(seq_len: int, dim: int,
                               base: float = 10000.0,
                               scale: float = 1.0,        # s = L_new / L_train
                               device=None):
    """
    NTK-aware: change base b' = b * s^{d/(d-2)}
    - Highest frequency (i=0) θ unchanged;
    - Lowest frequency (i=d/2-1) θ compressed to 1/s;
    - Middle dimensions transition exponentially with i.
    """
    new_base = base * (scale ** (dim / (dim - 2)))
    half = dim // 2
    inv_freq = 1.0 / (new_base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()
```

> ⚠️ **NTK-aware limitation** — at larger expansion ratios ($s \ge 8$), the lowest-frequency dimensions get compressed too aggressively and performance degrades. This motivates NTK-by-parts, which **handles different frequency bands separately** — and that is exactly the starting point of YaRN.

## §6 YaRN — Yet another RoPE extensioN

### 6.1 Overview

Peng et al. 2023 ("YaRN: Efficient Context Window Extension of Large Language Models") systematizes the NTK-aware idea, splitting it into three relatively independent components:

1. **NTK-by-parts**: split dimensions into three bands by wavelength and handle them separately
2. **Temperature scaling**: apply a global temperature to logits before softmax
3. **Attention scale** (an alternative implementation equivalent to temperature): scale Q/K norms in sync

We derive each below.

### 6.2 NTK-by-parts — segment-wise frequency handling

Let $L_\text{train}$ be the training context length. The wavelength of dimension $i$ is $\lambda_i = 2\pi / \theta_i = 2\pi b^{2i/d}$. Define the ratio

$$r_i = \frac{L_\text{train}}{\lambda_i} = \frac{L_\text{train} \cdot \theta_i}{2\pi}$$

$r_i$ is the number of revolutions dimension $i$ makes within the training length. Split dimensions into three bands:

| Band | Condition | Meaning | Treatment |
| --- | --- | --- | --- |
| **High frequency** | $r_i \ge \beta$ ($\beta=32$) | Within training, $\ge 32$ revolutions, relative positions fully sampled | **No scaling** ($\theta'_i = \theta_i$) |
| **Mid frequency** | $\alpha < r_i < \beta$ ($\alpha=1$) | Partial sampling | **Linear interpolation** (PI applied locally) |
| **Low frequency** | $r_i \le \alpha$ | Within training, < 1 revolution; position encoding has not seen a full period | **Fully scaled to $1/s$** (PI behavior) |

Formally: for dimension $i$, define a ramp function

$$\gamma(r_i) = \mathrm{clip}\!\left(\frac{r_i - \alpha}{\beta - \alpha},\; 0,\; 1\right) \in [0, 1]$$

The new frequency is an interpolation between NTK-aware and PI:

$$\theta'_i = (1 - \gamma(r_i)) \cdot \frac{\theta_i}{s} + \gamma(r_i) \cdot \theta_i$$

- $r_i \ge \beta$ (high frequency): $\gamma = 1$, $\theta'_i = \theta_i$ (unchanged)
- $r_i \le \alpha$ (low frequency): $\gamma = 0$, $\theta'_i = \theta_i / s$ (PI fully scaled)
- Middle: smooth transition

> 💡 **Reason for the three-band split** — high-frequency dimensions have completed many revolutions during training, so during extrapolation, **as long as the phase doesn't jump**, they can keep working (periodicity of rotations); low-frequency dimensions have not completed one revolution during training, so the "extrapolation region" is fully unseen data for the model, and we must **interpolate** into the training-seen phase range. Middle frequencies get in-between handling.

### 6.3 Temperature Scaling — attention entropy compensation

**Problem**: after extending context, the effective statistics of softmax input change — the same query now faces $L_\text{new} \gg L_\text{train}$ keys, making the attention distribution **flatter** (higher entropy) and the effective signal diluted.

**Solution**: divide logits by temperature $t$ before softmax ($t < 1$ sharpens the distribution to compensate for dilution):

$$\mathrm{Attention} = \mathrm{softmax}\!\left(\frac{QK^\top}{t \sqrt{d_k}}\right) V$$

The YaRN paper's fitted formula (from empirical ablations):

$$\boxed{\;\sqrt{1/t} \approx 0.1 \ln s + 1 \quad\Longleftrightarrow\quad t \approx \frac{1}{(0.1 \ln s + 1)^2}\;}$$

For example, $s = 8$ ($L_\text{new} = 32\text{K}$ from $L_\text{train} = 4\text{K}$): $\sqrt{1/t} \approx 0.1 \cdot 2.08 + 1 \approx 1.21$, $t \approx 0.68$.

### 6.4 Attention Scale — equivalent alternative implementation of Temperature

Directly modifying softmax temperature requires changing the attention kernel. Equivalent practice: **multiply** the norms of query and key by $\sqrt{1/t} = (0.1\ln s + 1) > 1$ (when $t < 1$ this is an amplification), so $QK^\top$ is naturally amplified by factor $1/t$, and softmax sees the same logits as if divided by $t$.

YaRN implements this by multiplying the scaling factor directly into the RoPE cache:

$$\text{cos}'_m = \cos(m \theta'_i) \cdot \sqrt{1/t}, \quad \text{sin}'_m = \sin(m \theta'_i) \cdot \sqrt{1/t}$$

Note this only affects the RoPE part, but **the overall effect is equivalent to amplifying query/key norms by $\sqrt{1/t}$** (when $t < 1$ this factor $> 1$) — provided the Q/K norms are dominated by the post-RoPE part. In practice YaRN's attention scale implementation simply multiplies the cos/sin cache by $\sqrt{1/t}$. **This is equivalent to changing the temperature without modifying the attention kernel**.

### 6.5 What does each of YaRN's three components solve (a must-ask L3 question)

| Component | Problem solved | What happens without it |
| --- | --- | --- |
| **NTK-by-parts** | High frequencies should be preserved, low frequencies should be interpolated, mid frequencies need a smooth transition | Using NTK-aware globally, large expansion ratios cause low-frequency collapse |
| **Temperature scaling** | After context lengthens, softmax distribution is diluted | Attention entropy too high, long-range signal drowned |
| **Attention scale (implementation-layer)** | Realize temperature without modifying softmax kernel | Need to rewrite the FlashAttention kernel |

YaRN paper shows: just 400 fine-tuning steps push LLaMA-2-7B from 4K to 128K ($s = 32$), outperforming PI and NTK-aware.

### 6.6 YaRN code (NTK-by-parts + temperature)

```python
import math

def precompute_rope_cache_yarn(
    seq_len: int, dim: int,
    base: float = 10000.0,
    scale: float = 1.0,            # s = L_new / L_train
    original_max_pos: int = 4096,  # L_train
    alpha: float = 1.0,            # ramp lower bound (revolutions)
    beta: float = 32.0,            # ramp upper bound (revolutions)
    device=None,
):
    """
    YaRN: NTK-by-parts + temperature scaling (implemented as attention scale).
    - High-frequency dims (r_i ≥ β): no scaling
    - Low-frequency dims (r_i ≤ α): PI-style full scaling
    - Mid dims (α < r_i < β): smooth transition
    """
    half = dim // 2
    i = torch.arange(0, half, device=device).float()                 # [half]
    inv_freq = 1.0 / (base ** (i / half))                            # θ_i
    wavelen = 2.0 * math.pi / inv_freq                               # λ_i
    r = original_max_pos / wavelen                                   # r_i = L_train / λ_i

    gamma = torch.clamp((r - alpha) / (beta - alpha), 0.0, 1.0)      # ramp ∈ [0,1]
    inv_freq_pi   = inv_freq / scale                                  # PI full scaling
    inv_freq_ntk  = inv_freq                                          # NTK unscaled (high freq)
    inv_freq_yarn = (1.0 - gamma) * inv_freq_pi + gamma * inv_freq_ntk

    # Temperature scaling (implemented as attention scale baked into cos/sin)
    # YaRN empirical formula:  sqrt(1/t) ≈ 0.1 ln(s) + 1
    # Goal: amplify effective QK^T by 1/t (equivalent to softmax temperature t<1 → sharper).
    # Implementation: multiply Q and K norms by sqrt(1/t), then QK^T is naturally multiplied by 1/t.
    # Because RoPE rotates via cos/sin, multiplying sqrt(1/t) into cos/sin suffices.
    sqrt_inv_t = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0
    attn_scale = sqrt_inv_t                                           # ← multiplied into cos/sin to amplify Q/K norm

    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq_yarn)
    return freqs.cos() * attn_scale, freqs.sin() * attn_scale
```

> ⚠️ **YaRN attention scale side effect** — Q/K norms are **amplified** by $\sqrt{1/t} > 1$ (not shrunk), but V is not amplified in sync. In a multi-layer transformer, this is equivalent to changing the effective temperature of each layer's attention, and gradient scales in backprop also differ. In practice, fine-tuning is needed to stabilize (YaRN paper uses ≈ 400 steps).

## §7 LongRoPE — Evolutionary Search + Short-Context Rescue

Ding et al., ICML 2024 (Microsoft) asks further: **can the optimal scaling factor of each dimension be searched independently**, rather than using a single ramp function?

### 7.1 Key observations

1. **Dimensions differ greatly in sensitivity to extension length** (one ramp function is not necessarily optimal).
2. **Ultra-long-context models actually degrade on short contexts (≤ $L_\text{train}$)** — because the RoPE cache has been changed and the original training distribution is disturbed.

### 7.2 Three-stage scheme

| Stage | What |
| --- | --- |
| **Stage 1: Evolution search (256K)** | Each RoPE dimension scaled independently by $\lambda_i$; evolutionary search for the $\{\lambda_i\}$ giving the lowest long-context PPL |
| **Stage 2: Fine-tune at 256K** | Brief fine-tuning (≈ 400 steps) with the searched $\{\lambda_i\}$ |
| **Stage 3: Re-search at 2M + short-context rescue** | Further search up to 2M; maintain two scaling sets — short context uses $\{\lambda_i^\text{short}\}$ (close to 1), long context uses $\{\lambda_i^\text{long}\}$ |

### 7.3 Search space

Each dimension $i$ has $\lambda_i \in [1, s_\text{max}]$ ($s_\text{max} = L_\text{new}/L_\text{train}$); new frequency $\theta'_i = \theta_i / \lambda_i$.

Search objective:

$$\min_{\{\lambda_i\}} \mathrm{PPL}\!\left(M; \theta'_i = \theta_i / \lambda_i\right) \quad \text{on a long-context validation set}$$

Evolutionary algorithm (CMA-ES or similar) maintains a population, iterating to select the best. The paper reports convergence in a few hundred generations.

### 7.4 Comparison with YaRN

| Method | Scaling granularity | Fine-tune requirement | Max context |
| --- | --- | --- | --- |
| PI | All dimensions same $1/s$ | yes (≥ 1000 steps) | 32K |
| NTK-aware | Gradual (single param $\alpha$) | no (zero-shot) | 16K |
| YaRN | Three-band ramp (fixed $\alpha, \beta$) | yes (≈ 400 steps) | 128K |
| LongRoPE | **Per-dim independent** | yes (≈ 400 steps) | **2M** |

> 💡 **Significance of short-context rescue** — directly applying long-context scaling makes the model worse on short contexts (e.g., 1K-4K, covering most real use cases). LongRoPE switches the scaling table at inference based on the actual length of the current batch; this dual-table design is a common trick in production-grade long-context models (DeepSeek-V2 / Qwen2.5 / Llama-3.1 also have similar dual-table designs).

## §8 ABF and NoPE — Two "Non-Mainstream" Extensions

### 8.1 ABF — Adjusted Base Frequency (Xiong et al., "Effective Long-Context Scaling of Foundation Models", arXiv:2309.16039, NAACL 2024, Meta)

The most naive "base change" — simply change the RoPE base from 10000 to something larger (e.g., 500000). Equivalent to uniform NTK-aware scaling across all dimensions, but without considering the ramp.

- **Pros**: simplest, a 1-line config change.
- **Cons**: $\theta_0 = (b')^0 = 1$ is also unchanged (consistent with NTK-aware, highest frequency is precisely preserved), but ABF's $b'$ is picked by empirical guess (e.g., $10^6$), **not calibrated to the training length ratio** — unlike NTK-aware where the lowest frequency is precisely compressed to $1/s$. Result: the compression strength on low-frequency dimensions is entirely by feel.
- **Where used**: CodeLlama uses $b = 10^6$ to extend to 16K; Llama-3 / Llama-3.1 continue using large bases combined with more refined RoPE scaling.

### 8.2 NoPE — No Position Encoding

Kazemnejad et al. 2023 ("The Impact of Positional Encoding on Length Generalization in Transformers"): **decoder-only models without position encoding can still learn position information via the causal mask alone**.

Observation: the causal attention mask already breaks symmetry under permutation (position $i$ cannot see position $j > i$), which itself encodes **order**. On small models / short context, NoPE even extrapolates better.

> ⚠️ **NoPE limitation** — only applies to decoder-only + causal mask. Encoder-only (BERT-like) without causal mask degenerates to bag-of-words after removing position encoding. NoPE has not been broadly validated on large models. **Remember it as an interesting research finding**, not an industry default.

## §9 MLA — Multi-Head Latent Attention (DeepSeek-V2 May 2024)

### 9.1 Motivation

GQA compresses KV cache from $2 N_h d_h$ to $2 G d_h$ (per-token, per-layer), but $G$ must be at least 4-8 to maintain quality. **Can we compress more aggressively?** MLA compresses KV cache to a low-rank latent, theoretically reaching $d_c \ll N_h d_h$ without losing much performance.

### 9.2 Naive low-rank K/V — first-step derivation

Define a compression matrix $W_\text{DKV} \in \mathbb{R}^{d_c \times d_\text{model}}$, projecting each token's hidden state $\mathbf{h}_t \in \mathbb{R}^{d_\text{model}}$ onto a KV latent:

$$\boxed{\;\mathbf{c}_t^{KV} = W_\text{DKV}\, \mathbf{h}_t \in \mathbb{R}^{d_c}\;}$$

Each head's K, V is recovered from this latent via an **up-projection**:

$$\mathbf{k}_t^{(h)} = W_\text{UK}^{(h)}\, \mathbf{c}_t^{KV}, \qquad \mathbf{v}_t^{(h)} = W_\text{UV}^{(h)}\, \mathbf{c}_t^{KV}$$

where $W_\text{UK}^{(h)}, W_\text{UV}^{(h)} \in \mathbb{R}^{d_h \times d_c}$.

**Key: cache stores only $\mathbf{c}_t^{KV}$** ($d_c$-dimensional, plus a small decoupled-RoPE component $d_r$, see §9.6), not $\mathbf{k}, \mathbf{v}$ themselves. Per-token-per-layer cache drops from $2 N_h d_h$ to $d_c+d_r$. DeepSeek-V2 picks $d_c = 4 d_h$ (vs $2 N_h d_h$ for $N_h = 128$), giving **≈ 57× full-cache compression** (precise accounting in §9.6 / the §12.3 table; counting only the bare latent $d_c$ over-estimates it as ≈ 64×).

### 9.3 Absorbing trick — avoid explicit up-projection

Naive approach: each attention computes $\mathbf{k}_t^{(h)} = W_\text{UK}^{(h)} \mathbf{c}_t^{KV}$ from $\mathbf{c}_t^{KV}$, then computes $\mathbf{q}_t^{(h)\top} \mathbf{k}_t^{(h)}$. This equals:

$$\mathbf{q}_t^{(h)\top} \mathbf{k}_t^{(h)} = \mathbf{q}_t^{(h)\top} (W_\text{UK}^{(h)} \mathbf{c}_t^{KV}) = (W_\text{UK}^{(h)\top} \mathbf{q}_t^{(h)})^\top \mathbf{c}_t^{KV}$$

Let $\tilde{\mathbf{q}}_t^{(h)} := W_\text{UK}^{(h)\top} \mathbf{q}_t^{(h)}$; then the attention score becomes $\tilde{\mathbf{q}}_t^{(h)\top} \mathbf{c}_s^{KV}$ — **inner product with the latent cache directly**, no need to compute $\mathbf{k}_s^{(h)}$. Similarly, $W_\text{UV}^{(h)}$ can be absorbed into the left-multiplication of the output projection $W_O$. This is **MLA's absorbing trick**: at training time, the two steps are explicit; at inference, the up-projection matrices are absorbed into the query/output projections, **reading the cache and the matmul done in one step**.

### 9.4 Why RoPE must be decoupled (the most critical L3 question)

**Problem**: what if RoPE is added? Traditional RoPE multiplies directly onto $\mathbf{q}, \mathbf{k}$:

$$\mathbf{q}_t^{(h)} \leftarrow R_t\, \mathbf{q}_t^{(h)}, \qquad \mathbf{k}_t^{(h)} \leftarrow R_t\, \mathbf{k}_t^{(h)} = R_t\, W_\text{UK}^{(h)}\, \mathbf{c}_t^{KV}$$

But $R_t$ is **position-dependent** — different rotation matrices for different cache tokens $t$. If we still want to use the absorbing trick and absorb $R_t W_\text{UK}^{(h)}$ into the query side, this becomes

$$\mathbf{q}_t^{(h)\top} \mathbf{k}_s^{(h)} = \mathbf{q}_t^{(h)\top} (R_s\, W_\text{UK}^{(h)}\, \mathbf{c}_s^{KV})$$

Here $R_s$ differs per cache position $s$ — **no fixed matrix can be absorbed into the query projection**. In other words:

$$(W_\text{UK}^{(h)\top} R_s^\top)\, \mathbf{q}_t^{(h)} \quad \text{with } R_s \text{ varying in } s$$

Insisting on preserving RoPE while doing absorbing is equivalent to **per-position query projection**, destroying all cache-friendliness of the absorbing trick; the cache would need to store post-RoPE K again (returning to $N_h d_h$ size).

### 9.5 MLA's decoupling solution — shared RoPE key + non-RoPE main body

DeepSeek-V2's solution: **split K into two parts**:

1. **Non-RoPE main body**: obtained via up-projection from the latent, dimension $d_h$, participates in absorbing.
2. **RoPE part**: a separate key of dimension $d_h^R$ (usually 64), **shared across all heads**, with RoPE applied independently, not participating in absorbing.

Formally (DeepSeek-V2 paper Eq. 5-11):

$$
\begin{aligned}
\mathbf{c}_t^{KV} &= W_\text{DKV}\, \mathbf{h}_t \in \mathbb{R}^{d_c} \\
\mathbf{k}_t^{C,(h)} &= W_\text{UK}^{(h)}\, \mathbf{c}_t^{KV} \in \mathbb{R}^{d_h} \quad\text{(non-RoPE main body, per head)}\\
\mathbf{k}_t^{R} &= \mathrm{RoPE}(W_\text{KR}\, \mathbf{h}_t) \in \mathbb{R}^{d_h^R} \quad\text{(shared RoPE key, all heads share)}\\
\mathbf{k}_t^{(h)} &= [\mathbf{k}_t^{C,(h)}\; ;\; \mathbf{k}_t^{R}] \in \mathbb{R}^{d_h + d_h^R}
\end{aligned}
$$

The query side is similarly split into two halves:

$$
\begin{aligned}
\mathbf{c}_t^{Q} &= W_\text{DQ}\, \mathbf{h}_t \in \mathbb{R}^{d_c'} \\
\mathbf{q}_t^{C,(h)} &= W_\text{UQ}^{(h)}\, \mathbf{c}_t^{Q} \in \mathbb{R}^{d_h} \quad\text{(non-RoPE, paired with } \mathbf{k}_t^{C,(h)} \text{)}\\
\mathbf{q}_t^{R,(h)} &= \mathrm{RoPE}(W_\text{QR}^{(h)}\, \mathbf{c}_t^{Q}) \in \mathbb{R}^{d_h^R} \quad\text{(per-head RoPE query, paired with } \mathbf{k}_t^{R} \text{)}\\
\mathbf{q}_t^{(h)} &= [\mathbf{q}_t^{C,(h)}\; ;\; \mathbf{q}_t^{R,(h)}]
\end{aligned}
$$

Attention score (same head):

$$\mathbf{q}_t^{(h)\top} \mathbf{k}_s^{(h)} = \underbrace{\mathbf{q}_t^{C,(h)\top}\, \mathbf{k}_s^{C,(h)}}_{\text{absorbed into q projection}} + \underbrace{\mathbf{q}_t^{R,(h)\top}\, \mathbf{k}_s^{R}}_{\text{RoPE part, computed directly}}$$

In the first term, $\mathbf{k}_s^{C,(h)} = W_\text{UK}^{(h)} \mathbf{c}_s^{KV}$, and per §9.3 absorbing trick, $W_\text{UK}^{(h)}$ is absorbed into the query side. In the second term, the RoPE key is shared across all heads, and the cache stores only one copy of $\mathbf{k}_s^R$.

### 9.6 MLA KV cache total size

Per token per layer:

$$\boxed{\;\text{MLA cache} = \underbrace{d_c}_{\mathbf{c}^{KV}} + \underbrace{d_h^R}_{\mathbf{k}^R} \quad \text{vs} \quad \text{MHA cache} = 2 N_h d_h\;}$$

DeepSeek-V2 numbers ($N_h = 128, d_h = 128, d_c = 512, d_h^R = 64$):

- MHA: $2 \cdot 128 \cdot 128 = 32{,}768$ elements / token / layer
- MLA: $512 + 64 = 576$ elements / token / layer
- Compression ratio **57×** (vs MHA); MLA's total KV cache is also about 4× smaller than GQA-8.

### 9.7 MLA simplified code

```python
import torch
import torch.nn as nn
# Reuse apply_rope from §2.4 (omitted).

class MultiHeadLatentAttention(nn.Module):
    """
    Simplified MLA: training version (absorbing trick can be added at inference).
    Per-token-per-layer cache: c_kv [d_c] + k_R [d_h_R]
    """
    def __init__(self, d_model: int, n_heads: int,
                 d_c: int = 512, d_h: int = 128, d_h_R: int = 64,
                 d_c_q: int = 1536):
        super().__init__()
        self.n_heads, self.d_h, self.d_h_R = n_heads, d_h, d_h_R

        # Down-projection to latent
        self.W_DKV = nn.Linear(d_model, d_c,  bias=False)
        self.W_DQ  = nn.Linear(d_model, d_c_q, bias=False)

        # Up-projection (non-RoPE main body)
        self.W_UK = nn.Linear(d_c,   n_heads * d_h,   bias=False)
        self.W_UV = nn.Linear(d_c,   n_heads * d_h,   bias=False)
        self.W_UQ = nn.Linear(d_c_q, n_heads * d_h,   bias=False)

        # RoPE-decoupled part
        self.W_KR = nn.Linear(d_model, d_h_R,            bias=False)  # shared across heads
        self.W_QR = nn.Linear(d_c_q,   n_heads * d_h_R,  bias=False)  # per head

        self.W_O  = nn.Linear(n_heads * d_h, d_model, bias=False)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor = None):
        # h: [B, L, d_model]; cos/sin for RoPE: [L, d_h_R/2]
        B, L, _ = h.shape
        H, dH, dHR = self.n_heads, self.d_h, self.d_h_R

        # ----- KV path -----
        c_kv = self.W_DKV(h)                                              # [B, L, d_c]
        k_C  = self.W_UK(c_kv).view(B, L, H, dH).transpose(1, 2)          # [B, H, L, d_h]
        v    = self.W_UV(c_kv).view(B, L, H, dH).transpose(1, 2)          # [B, H, L, d_h]

        k_R  = self.W_KR(h)                                                # [B, L, d_h_R]  (shared)
        k_R  = apply_rope(k_R, cos, sin)                                   # shared RoPE
        # Broadcast to H heads for concatenation
        k_R_per_head = k_R.unsqueeze(1).expand(B, H, L, dHR)                # [B, H, L, d_h_R]
        k = torch.cat([k_C, k_R_per_head], dim=-1)                          # [B, H, L, d_h+d_h_R]

        # ----- Q path -----
        c_q  = self.W_DQ(h)                                                 # [B, L, d_c_q]
        q_C  = self.W_UQ(c_q).view(B, L, H, dH).transpose(1, 2)             # [B, H, L, d_h]
        q_R  = self.W_QR(c_q).view(B, L, H, dHR).transpose(1, 2)            # [B, H, L, d_h_R]
        q_R  = apply_rope(q_R, cos, sin)                                    # per-head RoPE
        q = torch.cat([q_C, q_R], dim=-1)                                   # [B, H, L, d_h+d_h_R]

        # ----- Attention -----
        scores = (q @ k.transpose(-2, -1)) / ((dH + dHR) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, L, H * dH)   # [B, L, H*d_h]
        return self.W_O(out)                                                # [B, L, d_model]
```

> ⚠️ **Common misconception** — "MLA is just further compression of GQA" — inaccurate. GQA compresses along the head dimension (multiple Q heads share a K/V head); MLA is low-rank compression along the hidden dimension ($N_h d_h \to d_c$) + shared RoPE. GQA still applies RoPE independently per KV head; MLA must decouple RoPE to preserve the absorbing trick.

### 9.8 Training cost

MLA introduces extra down-projection / up-projection, so **training FLOPs slightly increase** (DeepSeek-V2 reports ≈ 2% increase), in exchange for tens of times smaller KV cache at inference — a "slightly more expensive training, much cheaper inference" trade-off.

## §10 Sliding Window and Streaming Attention

### 10.1 Sliding Window Attention (Mistral 2023)

Each query attends only to the previous $W$ keys ($W$ = window size; Mistral-7B uses $W = 4096$).

- **Complexity**: drops from $O(L^2 d)$ to $O(L W d)$, linear in long sequences.
- **Receptive field via multi-layer stacking**: layer 1 sees $W$; layer 2's each position sees the previous $W$ (where each token sees its own previous $W$), giving effective receptive field $2W$; after $\ell$ layers, the receptive field is $\ell W$. **So 32 layers × 4096 window ≈ 131K effective receptive field**.

```python
def sliding_window_mask(L: int, W: int, device=None) -> torch.Tensor:
    """
    L: sequence length; W: window size (including self)
    Returns [L, L] bool mask, True=visible.
    Position i sees j ∈ [max(0, i-W+1), i] (causal + sliding window)
    """
    idx_q = torch.arange(L, device=device).unsqueeze(1)   # [L, 1]
    idx_k = torch.arange(L, device=device).unsqueeze(0)   # [1, L]
    causal     = idx_k <= idx_q
    in_window  = idx_k > (idx_q - W)
    return causal & in_window

# Example L=8, W=4:
# row 0: [T F F F F F F F]
# row 1: [T T F F F F F F]
# row 2: [T T T F F F F F]
# row 3: [T T T T F F F F]
# row 4: [F T T T T F F F]    ← token 0 is pushed out of the window
# row 5: [F F T T T T F F]
# row 6: [F F F T T T T F]
# row 7: [F F F F T T T T]
```

> 💡 **Practical significance of SWA** — Mistral-7B trained at length 8K can handle 32K+ context at inference with SWA (each layer sees only 4K locally; multi-layer stacking sees globally), with memory/compute scaling linearly. But pure SWA's **long-range exact retrieval** (e.g., needle-in-haystack far away) is weak — this is exactly why StreamingLLM adds attention sink.

### 10.2 StreamingLLM — Attention Sink + Sliding Window

Xiao et al. ICLR 2024 ("Efficient Streaming Language Models with Attention Sinks") makes a key inference-time finding:

**During LLM decode, softmax forces attention weights to sum to 1, but the query may actually have "nothing it wants to attend to". The model then dumps most of the weight onto the first 1-4 tokens (especially `<bos>`), forming an attention sink.** These tokens carry no informational content, but their KV cache **cannot be discarded** — once removed, softmax loses its "trash bin" and the remaining tokens' attention distribution is forcibly rebalanced, blowing up PPL.

StreamingLLM inference strategy:

1. **Always keep** the KV cache of the first $S$ tokens ($S = 4$ empirical) as the sink.
2. **Sliding window** keeping the KV cache of the most recent $W$ tokens.
3. Tokens outside the window and outside the sink have their KV **directly discarded**.

The total KV cache size is $S + W$, decoupled from the sequence length $L$, achieving **true streaming** generation.

### 10.3 StreamingLLM inference loop code

The below is **pedagogical**, focused on control flow. The production implementation (HuggingFace `streaming_llm` / the authors' `streaming-llm` repo) has two critical details, noted in the comments.

```python
@torch.no_grad()
def streaming_decode(model, input_ids, max_new_tokens,
                     sink_size=4, window_size=2044):
    """
    Pedagogical streaming inference: sink + sliding window.
    Total cache = sink_size + window_size, independent of generation length.

    Critical details (production code MUST do):
    (a) Cache stores K *before* RoPE (i.e., W_K @ h, unrotated) AND records each token's "logical position".
        At each forward, re-apply RoPE on sink / recent K according to their *new* logical positions in the current cache.
        Otherwise, trimming + position shift causes the rotation angles in the cache to mismatch the new logical positions.
    (b) Sink positions are fixed at [0, S), recent window positions are fixed at [S, S+W),
        and new tokens take S+W (the cache capacity upper bound) as their logical position.
        This way, the "max relative position" the model sees is always ≤ S+W, never touching the RoPE training upper bound.
    """
    device = input_ids.device
    B = input_ids.size(0)
    total = sink_size + window_size

    # ----- 1) Prefill -----
    # past_kv_pre[i] = (k_pre, v) where k_pre = W_K @ h, NOT RoPE-applied
    past_kv_pre = _prefill_unrotated(model, input_ids)            # implementation details omitted

    # If the prompt already exceeds sink+window, trim (sink segment + most recent window segment)
    def trim_unrotated(past_kv_pre):
        new_past = []
        for (k_pre, v) in past_kv_pre:
            if k_pre.size(-2) <= total:
                new_past.append((k_pre, v));  continue
            sink = (k_pre[..., :sink_size, :], v[..., :sink_size, :])
            recent = (k_pre[..., -window_size:, :], v[..., -window_size:, :])
            new_past.append(
                (torch.cat([sink[0], recent[0]], dim=-2),
                 torch.cat([sink[1], recent[1]], dim=-2))
            )
        return new_past

    past_kv_pre = trim_unrotated(past_kv_pre)
    # logits from the last prefill step
    next_token = _last_logits(model, past_kv_pre).argmax(-1, keepdim=True)
    generated = [next_token]

    # ----- 2) Autoregressive decode -----
    for step in range(max_new_tokens - 1):
        cur_len = past_kv_pre[0][0].size(-2)                       # current number of tokens in cache
        # Assign "logical positions" to each cache token; note: when prompt is very short and cur_len < sink_size,
        # all tokens are treated as sink (no recent window).
        if cur_len <= sink_size:
            cache_pos = torch.arange(cur_len, device=device)        # [cur_len]
        else:
            cache_pos = torch.cat([
                torch.arange(sink_size, device=device),             # sink segment: [0..S)
                torch.arange(sink_size, cur_len, device=device),    # window segment: [S..cur_len)
            ])                                                       # length = cur_len
        new_pos = torch.tensor([cur_len], device=device)             # new token's logical position

        # Re-apply RoPE on cache K_pre (by cache_pos) and on the new token (by new_pos).
        out = model(next_token, past_kv_pre=past_kv_pre,
                    cache_pos=cache_pos, new_pos=new_pos, use_cache=True)
        past_kv_pre = trim_unrotated(out.past_kv_pre)
        next_token = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_token)

    return torch.cat(generated, dim=-1)
```

> ⚠️ **Directly trimming the post-RoPE K cache is wrong** — a common bug: directly trim HF's default K cache (already RoPE-applied) as above and feed new tokens with logical position ids, and you get a self-contradictory relative position (cache K is rotated with original absolute positions, but the new query is rotated with logical positions). **Correct approach**: keep the unrotated K (`W_K @ h`, not multiplied by cos/sin) and re-apply RoPE each step by current logical position; or use the author repo's `enable_streaming_llm()` patch, which modifies the attention layer to accept "position-shift" form rotation.

> ⚠️ **StreamingLLM does not increase the model's effective context** — it allows the model to **stream forever** without blowing memory, but what it actually sees is still the tokens within sink + window range. The discarded middle content **really is invisible**. For long-context retrieval, you still need YaRN / LongRoPE / SSM or actual context extension.

### 10.4 Lost-in-the-Middle (Liu 2023)

Liu et al. 2023 ("Lost in the Middle: How Language Models Use Long Contexts") empirically observes: **long-context models pay much more attention to the head and tail of the prompt than the middle**, making "middle content harder to retrieve".

- **U-shaped curve**: placing key information at different positions in the prompt yields a U-shaped retrieval accuracy curve (high at ends, low in middle).
- **Reasons**: in causal-LM training distribution, the first token has the broadest influence (attention sink shared root); the last token is the direct precursor of next-token prediction. Middle content is "squeezed" by both ends.
- **Mitigation**: (a) put important information at the start or end of the prompt; (b) recurrent retrieval (chain prompts); (c) increase mid-segment weight during training (position-aware loss weighting).

> 💡 **Interview takeaway** — this is not "position-encoding extrapolation failure" — the model **has effectively learned** long context, but the attention distribution has a preference. Different from what RoPE/YaRN solve.

## §11 System-Level Long Context — Ring / CP / FlashAttn

### 11.1 Ring Attention (Liu et al. 2023)

Cut the sequence into $P$ chunks on $P$ GPUs; each chunk holds its own Q/K/V chunk. Attention is realized via **K/V chunks passed in a ring around the GPUs**:

```
GPU 0: holds Q0, K0, V0  ←→  GPU 1: holds Q1, K1, V1  ←→  ...  ←→  GPU P-1
            │                       │
            └─ pass K1, V1 to GPU 0, while GPU 0 passes K0, V0 to GPU P-1
            (after P-1 ring iterations, every GPU has seen all K, V)
```

- **At each round**, each GPU does local attention with its currently-held K/V chunk on its local Q chunk, accumulating partial output.
- **Communication overlaps with computation** (the next K/V is being passed while the current attention is being computed).
- Per-GPU communication: $O(L \cdot d)$ (each GPU sends/receives $P-1$ chunks); per-GPU computation: $O(L^2 d / P)$.

**Key effect**: each GPU only needs to hold $L/P$ of K/V, and **effective context scales linearly with the number of GPUs** — theoretically 8 GPUs × 128K per card = 1M context.

### 11.2 Context Parallelism (Megatron 2024)

Megatron-Core's Context Parallel (CP) is an engineering-grade version of Ring Attention, integrated into existing tensor/pipeline parallelism. Main engineering points:

- Uses fused all-to-all comms combined with FlashAttention blocking
- Handles load imbalance between chunks under causal mask (front chunks compute less attention, back chunks compute more, requiring load balancing)
- Compatible with ZeRO-3

### 11.3 FlashAttention 2/3 and long context

FlashAttention v1 (Dao 2022) core is IO-aware exact attention, but v1's loop structure is unbalanced on long sequences.

- **v2 (Dao 2023)**: swaps inner/outer loops (Q-outer, KV-inner), better warp-level parallelism, 2× throughput on long sequences.
- **v3 (Dao 2024)**: targets H100, uses WGMMA / TMA / FP8 asynchronous pipeline.

In long-context scenarios, FlashAttention is **the default in almost all training / inference stacks** (avoiding materializing the $L \times L$ score matrix).

### 11.4 Differential Attention (optional, Microsoft 2024)

Ye et al. 2024 ("Differential Transformer") proposes that each attention head uses **two independent Q/K projections and takes the difference**:

$$\mathrm{Diff} = \mathrm{softmax}(Q_1 K_1^\top / \sqrt{d}) - \lambda \cdot \mathrm{softmax}(Q_2 K_2^\top / \sqrt{d})$$

- **Intuition**: the first term learns "signal", the second learns "noise"; the difference is sharper.
- **Effect**: significant improvement on long-context needle-in-haystack tasks over vanilla attention.
- **Cost**: each head computes an extra set of Q/K projections, but the paper **halves the head count** so total params / FLOPs roughly **match** a standard Transformer (≈0 overhead, not +50%).

> 💡 **Whether to use** — Differential Attention is a new direction from late 2024; industry adoption is still low (DeepSeek-V3 does not use it, Llama-3 does not either), but it is interesting research. When asked about "new long-context directions" in interviews, you can mention it.

## §12 Complexity and Memory Summary Table

### 12.1 KV cache per-token-per-layer size (attention variants)

| Method | KV cache size (elements) | vs MHA ($N_h=128, d_h=128, G=8, d_c=512, d_h^R=64$) |
| --- | --- | --- |
| **MHA** | $2 N_h d_h$ | 32,768 (baseline 1×) |
| **MQA** | $2 d_h$ | 256 (128×) |
| **GQA-8** | $2 G d_h$ | 2,048 (16×) |
| **MLA** | $d_c + d_h^R$ | 576 (57×) |

### 12.2 Total KV cache occupancy (per-sample-per-layer, affected by "window" mechanisms like SWA / Streaming)

| Method | Total cache size (elements) | vs vanilla cache (under the same attention variant) |
| --- | --- | --- |
| **Vanilla (full sequence)** | $L \cdot 2 N_h d_h$ | baseline 1× |
| **SWA (window=W)** | $W \cdot 2 N_h d_h$ (each layer only sees the most recent W tokens) | $W/L \times$ |
| **Streaming (sink+win)** | $(S + W) \cdot 2 N_h d_h$ (constant, decoupled from L) | $(S{+}W)/L \times$ |

Note: SWA / Streaming and GQA / MLA are **orthogonal** — multiplying them together gives the actual cache size in production stacks.

### 12.3 Attention time and memory

| Method | Time per token (decode) | Memory peak (prefill) |
| --- | --- | --- |
| Vanilla MHA | $O(L \cdot N_h d_h)$ | $O(L^2)$ scores |
| FlashAttention | $O(L \cdot N_h d_h)$ | $O(L)$ (no intermediate scores) |
| Sliding Window | $O(W \cdot N_h d_h)$ | $O(L \cdot W)$ |
| Streaming (S+W) | $O((S+W) \cdot N_h d_h)$ | $O((S+W)^2)$ |
| Ring (P GPU) | $O(L \cdot N_h d_h / P)$ per GPU | $O(L^2 / P)$ per GPU |
| MLA | $O(L \cdot N_h (d_c + d_h^R))$ | + projection overhead |

## §13 Overall Comparison and Selection Decision Tree

```
Q: I want to push context from 4K to N tokens, N=?
│
├── N ≤ 16K, zero-shot, cannot fine-tune
│    └── NTK-aware (1-line config, increase base)
│
├── N ≤ 32K, can do limited fine-tuning (~1000 steps)
│    └── PI (simple and stable) or YaRN (better)
│
├── 32K < N ≤ 128K, fine-tune budget < 500 steps
│    └── YaRN (NTK-by-parts + temperature)
│
├── N > 128K (256K-2M)
│    └── LongRoPE (per-dim independent search + short-context rescue)
│
└── Streaming generation (unlimited length, no long-range retrieval needed)
     └── StreamingLLM (sink + sliding window)

Q: KV cache memory unmanageable?
│
├── Want to preserve quality, compress moderately
│    └── GQA (LLaMA-2/3, Mistral)
│
├── Want extreme compression, accept retraining
│    └── MLA (DeepSeek-V2/V3): cache cut 50×, RoPE must be decoupled
│
└── Inference server side
     └── Combine with PagedAttention (vLLM) for cache pagination

Q: Attention infeasible (L^2 too large)?
│
├── Single-card inference
│    └── FlashAttention 2/3 (exact, must install)
│
├── Multi-card training / inference
│    └── Ring Attention / Context Parallelism (chunk K/V ring pass)
│
└── Don't need long-range exact retrieval, only local dependency
     └── Sliding Window Attention (Mistral style)
```

## §14 25 Frequently-Asked Interview Questions

Split into L1 (must-know) / L2 (advanced) / L3 (top labs) tiers. Each question expands to answer points and pitfalls.

### L1 must-know (any long-context-related role)

<details>

<summary>Q1. What is the core formula of RoPE? Why does it give "relative positions"?</summary>

- 2D rotation per pair of adjacent dimensions: $f(\mathbf{x}, m) = \mathbf{x} \odot e^{im\boldsymbol\theta}$ (complex view), $\theta_i = 10000^{-2i/d}$
- $\langle f(\mathbf{q}, m), f(\mathbf{k}, n)\rangle = \mathrm{Re}\!\sum_i \bar{q}_i k_i\, e^{i(n-m)\theta_i}$, **depends only on $n-m$**
- Key: additivity of rotation matrices $R_m^\top R_n = R_{n-m}$

Pitfall: only saying "RoPE encodes relative position" without being able to derive.

</details>

<details>

<summary>Q2. Why is the RoPE frequency $10000^{-2i/d}$?</summary>

- Follows the geometric-progression frequency distribution of Vaswani 2017 sinusoidal
- High-frequency dimensions (small $i$) have short periods, encoding fine-grained local positions; low-frequency dimensions (large $i$) have long periods, encoding coarse long-range positions
- Resolves positions at multiple time scales simultaneously

Pitfall: just saying "so different dimensions see different positions", without pointing out the geometric progression and high/low frequency meaning.

</details>

<details>

<summary>Q3. Why can naive RoPE not extrapolate directly?</summary>

- Training has $m \in [0, L_\text{train})$, and on low-frequency dimensions $m\theta_i$ is far less than $2\pi$
- Inference with $m > L_\text{train}$, low-frequency dimensions enter unseen phase regions
- The model has not learned attention behavior for these regions → PPL blows up / context collapses

Pitfall: saying "RoPE periodicity makes extrapolation OK" — wrong. Periodicity only holds within a dimension; what's being extrapolated across context length is the "position → phase" mapping, and the model has never seen $m\theta_i$ outside the training range combinations.

</details>

<details>

<summary>Q4. How does PI (Position Interpolation) work? What is the side effect?</summary>

- Divide all $\theta_i$ by $s = L_\text{new}/L_\text{train}$ (or equivalently compress $m$ to $m/s$)
- Side effect: **high-frequency dimensions are damaged** — high frequency originally resolves fine-grained positions during training, but now resolution is compressed by $s\times$
- Must fine-tune (≥ 1000 steps) to recover

Pitfall: assuming "interpolation is lossless".

</details>

<details>

<summary>Q5. What is the core difference between NTK-aware and PI?</summary>

- **PI**: all dimensions divided by $s$ (high frequencies damaged)
- **NTK-aware**: change base $b' = b \cdot s^{d/(d-2)}$, so the highest frequency is almost unchanged and the lowest is compressed to $1/s$
- **NTK-aware is zero-shot usable** (no fine-tuning needed); PI must be fine-tuned

Pitfall: saying "NTK and PI are no different".

</details>

<details>

<summary>Q6. Difference between ALiBi and RoPE? Which extrapolates better?</summary>

- ALiBi: add $-m_h |i-j|$ distance bias to logits, head-dependent slope, **no Q/K rotation**
- RoPE: encode position via Q/K rotation, **no explicit bias**
- Extrapolation: ALiBi is better (linear bias extrapolates naturally), but is less expressive (only monotonic distance decay)
- Industry choice: RoPE combined with YaRN/LongRoPE is more common (expressivity + extensibility)

Pitfall: treating RoPE and ALiBi as the same type (one is score-shift, the other is Q/K transformation).

</details>

<details>

<summary>Q7. How to compute KV cache memory?</summary>

- Formula: $L_\text{ctx} \cdot n_\text{layers} \cdot 2 \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$
- The $2$ is because both K and V are stored; MQA has $H_\text{kv} = 1$; GQA has $H_\text{kv} = G$; MLA replaces it with $d_c + d_h^R$ (no longer 2× separately)
- For LLaMA-2-7B (32 layers, $N_h=32, d_h=128$, fp16, MHA), 4K context $\approx 2.1$ GB / sample; 100K $\approx 52$ GB / sample
- LLaMA-2-70B uses GQA-8 ($H_\text{kv}=8$, 80 layers, $d_h=128$); 4K $\approx 1.25$ GB / sample — GQA hugely compresses

Pitfall: forgetting $n_\text{layers}$; or treating the $2$ as a head factor.

</details>

<details>

<summary>Q8. What does MQA / GQA reduce?</summary>

- **KV cache memory** + **memory bandwidth** (during decode, K/V cache must be read from HBM each step)
- Also reduces K/V projection parameters and computation
- **Does not reduce Q projection**; Q head count remains unchanged

Pitfall: mistakenly saying "GQA reduces Q heads".

</details>

<details>

<summary>Q9. How does Sliding Window Attention let the model see far?</summary>

- Each layer sees only $W$, but stacked multi-layer: at layer $\ell$, each position's receptive field is $\ell \cdot W$
- Mistral-7B: 32 layers × 4K window ≈ 131K effective receptive field
- But **long-range exact retrieval** capability is weak (information must propagate via multi-layer "tunnel")

Pitfall: assuming "within window means only $W$ tokens visible" — wrong; that's true for only one layer.

</details>

<details>

<summary>Q10. What is Attention Sink?</summary>

- During LLM decode, the first 1-4 tokens (especially `<bos>`) receive abnormally high attention, even when content is irrelevant
- Intuition: softmax forces weights to sum to 1, and the model needs a "trash bin" to absorb probability mass
- Engineering use: StreamingLLM permanently keeps the first $S$ tokens' KV cache + sliding window

Pitfall: thinking attention sink is BOS / CLS tokens' "semantically normal" attention — wrong; sinks typically appear on all queries, **independent of content**.

</details>

### L2 advanced (research-oriented roles)

<details>

<summary>Q11. How to derive $b' = b \cdot s^{d/(d-2)}$ in NTK-aware?</summary>

- Let $b' = b \cdot \alpha$
- Highest frequency $\theta_0 = (b')^0 = 1$, unaffected by $\alpha$ ✓
- Lowest frequency $\theta'_{d/2-1} = b^{-(d-2)/d} \cdot \alpha^{-(d-2)/d}$
- Requiring $\theta'_{d/2-1} = \theta_{d/2-1}/s$ → $\alpha^{-(d-2)/d} = 1/s$ → $\alpha = s^{d/(d-2)}$

Pitfall: just memorizing the formula without being able to derive.

</details>

<details>

<summary>Q12. What does each of YaRN's three components solve?</summary>

- **NTK-by-parts**: handle high/mid/low frequencies in separate bands; finer than NTK-aware's single-parameter ramp
- **Temperature scaling**: after context lengthens, softmax distribution flattens; lower temperature $t < 1$ sharpens it
- **Attention scale (implementation-layer)**: implement temperature $1/t$ as Q/K norm scaling (equivalent to multiplying into cos/sin cache), without modifying the attention kernel

Pitfall: just saying "YaRN is an improved NTK-aware" without decomposing.

</details>

<details>

<summary>Q13. Where does YaRN's temperature formula $\sqrt{1/t} \approx 0.1 \ln s + 1$ come from?</summary>

- This is an **empirical fit formula**, not a closed-form derivation
- Based on experimental measurements of attention entropy under different expansion ratios $s$
- Idea: the larger the expansion ratio, the lower the temperature needed (sharper distribution) to compensate for dilution

Pitfall: treating it as a "strictly derived optimal temperature" — wrong; the YaRN paper makes clear it is an empirical fit.

</details>

<details>

<summary>Q14. What are the two RoPE real-form pairings?</summary>

- **Even-odd interleaved**: $(x_0, x_1), (x_2, x_3), \dots$ (original RoFormer paper)
- **Front half / back half**: $(x_0, x_{d/2}), (x_1, x_{d/2+1}), \dots$ (HuggingFace LLaMA implementation)
- Mathematically just a dimension permutation; **equivalent** for the final inner product
- But the RoPE cache precomputation and the pairing must be **consistent**; mixing them causes rotations to act on wrong dimensions

Pitfall: not knowing that HF and Meta's official implementations have this difference.

</details>

<details>

<summary>Q15. Core difference between LongRoPE and YaRN?</summary>

- **YaRN**: wavelength-based fixed ramp function; all dimensions follow the same rule
- **LongRoPE**: independent scaling factor $\lambda_i$ per dimension, **evolutionary algorithm** search
- LongRoPE also introduces short-context rescue (separate scaling table for short context)
- Max context: YaRN 128K vs LongRoPE 2M

Pitfall: saying LongRoPE "is no different from YaRN".

</details>

<details>

<summary>Q16. How does Mistral-7B compute the effective receptive field with SWA + multi-layer stacking?</summary>

- Single-layer receptive field $W = 4096$
- After $\ell$ layers, theoretical receptive field is $\ell W$; 32 layers × 4096 = 131K
- But actual "information propagation" is sparse — long-range tokens must propagate through multiple layers, equivalent to a deep pipeline
- Empirically Mistral performs well within 32K, decaying further out

Pitfall: assuming SWA directly looks at 4K as a hard upper bound.

</details>

<details>

<summary>Q17. Why does StreamingLLM use "logical positions" rather than absolute positions for position ids?</summary>

- If using absolute positions: in the cache, sink is at [0,4), the most recent window is at [L-W, L), and the new token is at L
- But $L$ can grow infinitely; RoPE hasn't seen $m > L_\text{train}$, so PPL blows up
- **Logical positions**: sink uses [0, S), within window uses [S, S+W), new token uses S+W
- This way RoPE is always within the training-seen range → streaming generation can be unlimited

Pitfall: saying "absolute position is correct" — wrong; absolute positions hit RoPE's extrapolation upper bound.

</details>

<details>

<summary>Q18. Communication and computation of Ring Attention?</summary>

- $P$ cards, each card holding sequence length $L/P$ of Q/K/V
- Ring-pass K/V chunks; after $P-1$ rounds, every card has seen all K/V
- Per-card communication: $O(L \cdot d)$ (send/receive K/V each)
- Per-card computation: $O(L^2 d / P)$
- **Communication and computation overlap**: next round of K/V is being passed while the current round's attention is being computed

Pitfall: saying "Ring Attention is just chunked attention" — missing the ring communication key point.

</details>

<details>

<summary>Q19. What is Lost in the Middle? Is it the same problem as position-encoding extrapolation?</summary>

- Phenomenon: in long context, the model attends more to head/tail tokens than middle (U-shaped curve)
- Cause: causal-LM training distribution favors head/tail (attention sink shared root + next-token direct precursor)
- **Not a position-encoding extrapolation problem** — it's an attention distribution preference problem
- Even with perfect position encoding extrapolation, this preference exists

Pitfall: confusing it with RoPE extrapolation.

</details>

<details>

<summary>Q20. Relation between ABF and NTK-aware?</summary>

- ABF (Adjusted Base Frequency): directly increase the RoPE base (e.g., 10000 → 500000), all dimensions sync base change
- NTK-aware: change base $b' = b \cdot s^{d/(d-2)}$, **formally identical to ABF** (both increase base)
- Difference is **why this change is made**: NTK-aware has a mathematical derivation (preserve highest frequency + compress lowest to $1/s$); ABF is an empirical choice
- CodeLlama uses ABF (base=$10^6$); LLaMA-3 also greatly increases the base and combines with RoPE scaling

Pitfall: saying "ABF and NTK-aware are completely unrelated" — wrong; the formulas are isomorphic, only the motivation differs.

</details>

### L3 top-lab questions (DeepSeek / Anthropic / OpenAI / Google)

<details>

<summary>Q21. Why does NTK-aware base scaling precisely preserve high frequencies?</summary>

- High frequency corresponds to $i = 0$, $\theta_0 = b^{-0} = 1$, **independent of $b$**
- After base change $b \to b' = b \cdot \alpha$, $\theta'_0 = (b')^0 = 1$, still 1
- Middle dimensions $\theta'_i / \theta_i = \alpha^{-2i/d}$, exponentially transitioning from 1 ($i=0$) to $1/s$ ($i=d/2-1$)
- **Geometric meaning**: base change is "shearing" in log-frequency space (high frequencies anchored, low frequencies compressed by $\log s$ amount)

Pitfall: just saying "NTK does not change high frequencies" — without explaining why base change has this effect automatically.

</details>

<details>

<summary>Q22. After RoPE is decoupled in MLA, how is absolute position information injected into the K/V latent up-projection part?</summary>

- **Key answer: it is not injected**. MLA's non-RoPE main body $\mathbf{k}_t^{C,(h)} = W_\text{UK}^{(h)} \mathbf{c}_t^{KV}$ has no position encoding at all
- The position signal is **provided only by the shared RoPE key** $\mathbf{k}_t^R = \mathrm{RoPE}(W_\text{KR} \mathbf{h}_t)$
- The attention score is additively decomposed: $\mathbf{q}_t^{C\top} \mathbf{k}_s^C$ (content) + $\mathbf{q}_t^{R\top} \mathbf{k}_s^R$ (position)
- This is what "decoupling" means: the content path and the position path are **independent**, not polluting the absorbing trick

Pitfall: assuming MLA absorbs RoPE into the latent — wrong.

</details>

<details>

<summary>Q23. Why can MLA not simply "apply RoPE after the up-projection"? Which step cannot be computed?</summary>

- Assume the cache stores $\mathbf{c}_s^{KV}$, and at attention time computes $\mathbf{k}_s^{(h)} = R_s\, W_\text{UK}^{(h)}\, \mathbf{c}_s^{KV}$
- To absorb: the query inner product becomes $\mathbf{q}_t^{(h)\top} (R_s W_\text{UK}^{(h)} \mathbf{c}_s^{KV}) = (W_\text{UK}^{(h)\top} R_s^\top \mathbf{q}_t^{(h)})^\top \mathbf{c}_s^{KV}$
- Here $R_s$ is **position-$s$-dependent** rotation — each cache position $s$ corresponds to a different $R_s$
- Cannot absorb a fixed matrix into the query projection; **absorbing must be per-position**
- Equivalent to computing $W_\text{UK}^{(h)\top} R_s^\top$ matmul per query × per cache position — **O(L) matmuls**, more expensive than directly materializing K
- So "applying RoPE after up-projection and absorbing" is computationally worse than not decoupling, **completely defeating the absorbing trick**

Pitfall: just saying "RoPE is position-dependent" — not enough; you need to state the key point that **the constancy needed for absorb is broken**.

</details>

<details>

<summary>Q24. What is the implementation-layer difference between YaRN's attention scale and directly changing the softmax temperature?</summary>

- **Direct temperature change**: divide logits by $t$ in the attention kernel, requiring modification of fused kernels like FlashAttention
- **Attention scale**: multiply $\sqrt{1/t}$ into the RoPE cos/sin cache, equivalent to **amplifying** Q/K norms by $\sqrt{1/t}$ ($\sqrt{1/t} > 1$ when $t < 1$); $QK^\top$ naturally amplified by $1/t$
- The two are **mathematically equivalent** (provided Q/K norms come mainly from the post-RoPE part)
- Engineering advantage: **no attention kernel modification at all**, only the RoPE precomputation
- This is a major selling point of YaRN being "infrastructure-friendly"

Pitfall: saying "the two are the same thing" — mathematically equivalent but engineering significance differs.

</details>

<details>

<summary>Q25. Design a 1M-context, streaming-generation, single-card-inference scheme.</summary>

Reference Qwen2.5-1M / DeepSeek-V3 ideas:

- **Position encoding**: YaRN / LongRoPE to push RoPE to 1M (per-dim scaling search)
- **KV cache compression**: MLA (cut cache 50×) to fit the 1M cache "latent" on a single card
- **Attention algorithm**: FlashAttention 3 + Ring Attention (if multi-card) or Sliding Window combined with sink (if streaming)
- **Inference optimization**: vLLM PagedAttention for cache pagination; speculative decoding to speed up decode; chunked prefill (feed prompt in batches to avoid OOM)
- **Training**: must actually fine-tune on long-context data (≥ 1000 steps); zero-shot RoPE modification alone is not enough

Full stack: MLA + YaRN/LongRoPE + FlashAttn3 + (Ring/CP if multi-card) + StreamingLLM(if streaming) + vLLM inference.

Pitfalls:
- Only mentioning one method (e.g., just YaRN) — not complete
- Not distinguishing "extending context" and "compressing cache" as two independent dimensions
- Forgetting "must fine-tune"

</details>

## §A Appendix: Implementation Points Checklist

### A.1 RoPE engineering implementation points

- **Pairing consistency**: even-odd interleaved vs front/back half must be consistent with the cache precomputation
- **Half-dim note**: cos/sin cache shape is $[L, d/2]$; when applying, broadcast to $[L, d]$ or multiply on the two halves separately
- **dtype handling**: compute cos/sin in fp32 then cast to dtype, avoiding rotation angle cumulative error in fp16/bf16
- **For YaRN**: the cos/sin cache has already been multiplied by $1/\sqrt{t}$; do not scale again inside attention
- **For MLA**: the RoPE part and non-RoPE part of query / key must be concatenated (typically RoPE last); attention scale uses $\sqrt{d_h + d_h^R}$, not $\sqrt{d_h}$

### A.2 Long-context fine-tune key hyperparameters (YaRN experience)

- Training tokens: ≈ 1 B tokens (≈ 400-1000 steps) significantly improves PPL
- Data: must contain **real long context** (books / arxiv / code repos); not just concatenated short documents
- Learning rate: typically $1\times 10^{-5}$ to $5\times 10^{-5}$, one order of magnitude smaller than pretrain
- Don't freeze: all layers participate in fine-tuning; freezing attention layers performs significantly worse
- Eval: PPL on long contexts, Needle-in-Haystack retrieval accuracy

### A.3 StreamingLLM deployment checklist

- Sink size $S = 4$ is empirically optimal (first 4 tokens)
- Window size $W$ choice: throughput vs quality trade-off; common $W \in [1024, 4096]$
- Position IDs must use **logical positions** rather than absolute positions
- Compatible with RoPE / YaRN; per §10.3, the cache should store K *before* RoPE, and at each forward re-apply RoPE based on the **current logical position** of each token in sink / window (some vendor implementations equivalently treat the sink segment as "fixed rotation + attention key index shift", with approximate effect)

### A.4 Quick reference table

| Context | Recommended scheme | KV cache optimization |
| --- | --- | --- |
| 4K-16K | RoPE + ABF / NTK-aware (zero-shot) | GQA |
| 16K-32K | PI / YaRN + fine-tune | GQA |
| 32K-128K | YaRN + fine-tune | GQA / MLA |
| 128K-2M | LongRoPE + fine-tune | MLA + Ring/CP |
| Streaming generation | StreamingLLM (sink + window) | Any; cache is constant size |

**Long Context Quick Reference** · Main references: Su et al. 2021/2024 (RoPE/RoFormer, Neurocomputing), Chen et al. 2023 (PI, arXiv:2306.15595, Meta), bloc97 / jquesnelle 2023 (NTK-aware, LocalLLaMA community), Peng et al. 2023 (YaRN, arXiv:2309.00071), Ding et al. 2024 (LongRoPE, ICML 2024, Microsoft), DeepSeek-AI 2024 (DeepSeek-V2, arXiv:2405.04434), Jiang et al. 2023 (Mistral 7B, arXiv:2310.06825), Xiao et al. 2024 (StreamingLLM, ICLR 2024), Nelson F. Liu et al. (Lost in the Middle, arXiv:2307.03172, arXiv 2023 / TACL 2024), Hao Liu et al. 2023 (Ring Attention, arXiv:2310.01889), Dao et al. 2022-2024 (FlashAttention 1/2/3)
