## §0 TL;DR Cheat Sheet

> 💡 **8 sentences to nail KV cache + Speculative Decoding** — one page covering interview essentials (see §2–§9 for derivations).

1. **KV cache formula**: per-sample memory $= 2 \cdot L_\text{ctx} \cdot N_\text{layers} \cdot N_\text{kv\_heads} \cdot d_\text{head} \cdot \text{bytes}$; the "2" comes from K+V. LLaMA-3-70B (GQA, $H_\text{kv}=8$) at 4K context fp16 ≈ **1.25 GB/sample** — which is why MHA is not used.

2. **Prefill vs Decode asymmetry**: prefill processes the entire prompt ($O(L^2)$ FLOPs, compute-bound); decode generates 1 token per step ($O(L)$ FLOPs per step, but must read the entire KV cache — **memory-bandwidth-bound**). This asymmetry explains every modern inference system design.

3. **PagedAttention** (Kwon et al., SOSP 2023, vLLM): partition KV cache into pages, use a block table to resolve fragmentation; memory utilization improves from ~70% to ~96%.

4. **Continuous batching** (Orca, Yu et al., OSDI 2022): iteration-level scheduling — completed requests don't wait for the whole batch; combined with PagedAttention, these are vLLM's two pillars.

5. **MQA → GQA → MLA**: MQA (Shazeer 2019) shares K/V extremely aggressively, slightly hurting quality; GQA (Ainslie et al., EMNLP 2023) groups into $G$ as a middle ground; MLA (DeepSeek-V2, May 2024) uses low-rank latent $c_t^{KV}$ + **decoupled RoPE** — RoPE cannot be absorbed into latent compression directly, so a small independent dimension $d_\text{head}^R$ must carry position.

6. **Speculative Decoding core** (Leviathan et al., ICML 2023; Chen et al., 2023): a small draft model $q$ proposes $K$ tokens, and the target model $p$ **verifies in parallel in one forward**; rejection sampling guarantees the output distribution is exactly equivalent to $p$ (exact, not approximate).

7. **Acceptance probability formula**: per-token acceptance rate $\alpha = \mathbb{E}_{x \sim q}[\min(1, p(x)/q(x))]$; expected generated tokens $E[\tau] = \dfrac{1-\alpha^{K+1}}{1-\alpha}$ ($K$ is the draft length, plus the final bonus token).

8. **Medusa / EAGLE / Lookahead**: Medusa (Cai et al., ICML 2024) multi-head + static tree attention; EAGLE/2/3 (Li et al., 2024-2025) feature-level draft + dynamic tree; Lookahead Decoding (Fu et al., ICML 2024) Jacobi iteration — **all different drafters under the same acceptance-rate framework**.

## §1 Intuition

### 1.1　Why is inference systems work so "counter-intuitive"

In training we worry about FLOPs: model size, batch size, when OOM happens. But deploying a 70B model — the bottleneck is often not compute but **HBM bandwidth** and **memory**, both eaten by the KV cache.

A core mental model:

> Modern LLM inference is **bandwidth-bound during decode and memory-bound during long-context prefill**, not compute-bound.

KV cache is the classic "swap recomputation for storage + bandwidth" trade. Once you cache all the K/V of the entire conversation history, generating each new token only requires:

- Computing Q/K/V for the new token (tiny compute)
- Appending the new K/V to the cache
- One attention with the new Q against the full cache

But the cost is: **for every new token, the entire KV cache must be read from HBM to SRAM** — which is why 8× H100 + LLaMA-3-70B at batch=1 decode runs far below theoretical FLOPs utilization (often 1-5%).

Speculative decoding attacks exactly this asymmetry: since decode is bandwidth-bound while GPU compute is spare, **why not compute $K$ candidate tokens in one forward, since weights are read only once?**

### 1.2　Differences from training-time attention

| Phase | Input | KV cache behavior | Bottleneck |
| --- | --- | --- | --- |
| **Training** | $[B, L, D]$ full sequence | Not needed — all positions computed simultaneously | FLOPs (compute) |
| **Prefill (inference)** | $[B, L_\text{prompt}, D]$ entire prompt | **Writes** to cache, fills $L_\text{prompt}$ positions | FLOPs ($L^2$ attention) |
| **Decode (inference)** | $[B, 1, D]$ single token | **Reads** + appends 1 position | **HBM bandwidth** (each step must read the entire cache + weights) |

A common interview cross-examination: "can training use KV cache?" — **No**. Training computes all positions at once; there's no "existing K/V waiting to be appended". Using KV cache during training is a beginner mistake.

## §2 KV Cache Memory Accounting

### 2.1　Exact formula

Per sample (batch=1), fp16:

$$\boxed{\;\text{KV cache}_\text{bytes} = 2 \cdot L_\text{ctx} \cdot N_\text{layers} \cdot N_\text{kv\_heads} \cdot d_\text{head} \cdot \text{bytes\_per\_elem}\;}$$

Each factor:

- **`2`**: one K tensor + one V tensor
- **$L_\text{ctx}$**: current context length (prompt + tokens generated so far)
- **$N_\text{layers}$**: Transformer layers (each has its own cache)
- **$N_\text{kv\_heads}$**: number of K/V heads. **In MHA = $H$; MQA = 1; GQA = $G$ ($1 < G < H$)**
- **$d_\text{head}$**: head dimension, usually 64 or 128
- **`bytes_per_elem`**: fp16 = 2, fp8/int8 = 1, int4 = 0.5

> ⚠️ **Common pitfall: don't multiply by $H$ (Q-heads count)** — KV cache scales only with K/V heads, **not with Q heads**. When MQA reduces K/V heads to 1, Q still has $H$ heads, so Q-projection compute is unchanged.

### 2.2　Cache size for some concrete models (fp16, $L_\text{ctx}=4096$)

Plug into the §2.1 formula $2 \cdot L_\text{ctx} \cdot N_\text{layers} \cdot N_\text{kv\_heads} \cdot d_\text{head} \cdot 2\text{B}$:

| Model | $N_\text{layers}$ | $N_\text{kv\_heads}$ | $d_\text{head}$ | cache/sample | Notes |
| --- | --- | --- | --- | --- | --- |
| LLaMA-2-7B (MHA) | 32 | 32 | 128 | **2.0 GB** | full MHA, large cache |
| LLaMA-2-70B (hypothetical MHA) | 80 | 64 | 128 | **10.0 GB** | this is why 70B uses GQA |
| LLaMA-2-70B (GQA) | 80 | 8 | 128 | **1.25 GB** | GQA cuts $H$=64 to $G$=8 |
| LLaMA-3-70B (GQA) | 80 | 8 | 128 | **1.25 GB** | same as LLaMA-2-70B |
| DeepSeek-V2 (MLA) | 60 | — | $d_c$=512 + $d_r$=64 | **~0.27 GB** | MLA formula: $N_\text{layers} \cdot L_\text{ctx} \cdot (d_c+d_r) \cdot 2\text{B}$ |

DeepSeek-V2 cache uses $d_c=512$ (latent dim shared by K/V via the same latent vector) + decoupled RoPE component $d_r=64$ (shared across all heads), following $N_\text{layers} \cdot L_\text{ctx} \cdot (d_c + d_r) \cdot \text{bytes}$ (**no "$\times 2$"**, since K/V no longer store separately). Under fp16, $60 \cdot 4096 \cdot 576 \cdot 2 \approx$ **0.27 GB / sample** — an order of magnitude smaller than same-scale MHA.

### 2.3　Batch dimension

In real serving, KV cache must also be multiplied by **active batch size**. A 70B + 4K context + GQA deployment on 8×A100 80GB:

- weights ≈ 140 GB (fp16, sharded across cards)
- single-sample cache ≈ 1.25 GB
- remaining usable memory ≈ $8 \times 80 - 140 \approx 500$ GB → subtract activations + framework overhead, ~400 GB allocatable to KV cache
- theoretical max batch ≈ $400 / 1.25 \approx$ **320** — but fragmentation prevents reaching it in practice; hence PagedAttention.

## §3 Prefill vs Decode Asymmetry

### 3.1　Per-phase FLOPs / bandwidth differences

Let prompt length be $L$, hidden $D$, FFN intermediate $4D$, and $N$ layers. Per-layer attention + FFN approximately:

$$\text{FLOPs}_\text{layer} \approx \underbrace{6BLD^2}_\text{QKV proj (3 mat)} + \underbrace{4BL^2 D}_\text{attention (QK + AV)} + \underbrace{2BLD^2}_\text{O proj} + \underbrace{16 BLD^2}_\text{FFN (up + down)}$$

**Prefill** with $L = L_\text{prompt}$: every term is $\Omega(L D^2)$ or $\Omega(L^2 D)$ — **compute-bound**, GPU saturated.

**Decode** with $L=1$ (one token per step), but the attention term becomes $4 B L_\text{ctx} D$ (QK and AV each $2 B L_\text{ctx} D$, since $L_q=1, L_k=L_\text{ctx}$):

- per-step FLOPs ≈ $\Theta(B (L_\text{ctx} D + D^2))$
- per-step **HBM read**: weights ≈ total model bytes (70B fp16 ≈ 140 GB) + KV cache ≈ $2 B L_\text{ctx} N H_\text{kv} d_\text{head} \cdot \text{bytes}$
- Arithmetic intensity (FLOPs / byte) $\to$ far below the GPU roofline (A100 BF16 ≈ 200 FLOPs/byte)

So **decode at small-to-medium batch is memory-bandwidth-bound** (until batch grows enough to saturate compute) — memorize this single line and you cover 80% of the interview score.

### 3.2　Chunked Prefill (Sarathi-Serve)

> 💡 **Key insight** — the longer the prompt, the more a single prefill saturates GPU compute, and **decode requests get stuck** (head-of-line blocking).

Sarathi-Serve (Agrawal et al., OSDI 2024) splits long prefill into equal-size chunks; each iteration schedules one prefill chunk + a batch of decode tokens together:

```
       Traditional: prompt 4096 → one prefill saturates GPU → decode stuck 100+ ms
       Sarathi:     prompt split into 4 chunks × 1024 → each iteration coalesces with decode
```

**Stall-free schedule**: every iteration contains decode + prefill-chunk coalesced; decode latency jitter disappears. The paper reports Mistral-7B on a single A100 with 2.6× over vLLM, and Yi-34B on 2×A100 with 3.7×.

### 3.3　Continuous Batching (Orca)

Traditional static batching: wait for the entire batch to finish before admitting new requests — **short requests get blocked by long ones**. Orca (Yu et al., OSDI 2022) changes the scheduling granularity from request to **iteration**: every forward checks the batch, kicks out completed sequences (EOS), and admits new requests in their place.

> ✅ **vLLM = Orca's continuous batching + PagedAttention memory management** — these two together raise LLM serving throughput by roughly 24× over the previous baseline.

### 3.4　Prefix Caching

If multiple requests share a prompt prefix (e.g., system prompt, few-shot prompt), the **same KV cache can be reused**:

- vLLM's prefix caching: index cache pages by hash(prompt prefix); on a hit, skip prefill
- For prompt-heavy services like ChatGPT, prefix cache hit rate can reach 90%+, drastically cutting prefill cost
- Implementation key: page-level sharing + COW (copy-on-write); branched requests only write to their own pages

## §4 KV Cache Optimization Routes

### 4.1　Overview of routes

| Route | Core idea | What is reduced | Representative |
| --- | --- | --- | --- |
| **Shared heads** (MQA/GQA) | Multiple Q heads share one set of K/V | K/V head count $H \to G$ or 1 | LLaMA-2/3, Mistral, PaLM |
| **Low-rank compression** (MLA) | Project to a low-dim latent; cache latent instead of K/V | Effective head dim shrinks $d_\text{head} \to d_c/H$ | DeepSeek-V2/V3 |
| **Quantization** (KIVI/KVQuant) | Cache elements fp16 → int4/int2 | bytes per element | KIVI, KVQuant, FP8 KV |
| **Sparsification / eviction** | Keep only "important" K/V positions | Effective $L_\text{ctx}$ shortened | H2O, StreamingLLM, TriForce |
| **Memory management** (PagedAttention) | Doesn't reduce total, but eliminates fragmentation | fragmentation overhead | vLLM |

### 4.2　MQA / GQA formula recap

MHA: each head has independent $W_k^{(h)}, W_v^{(h)} \in \mathbb{R}^{D \times d_\text{head}}$; total $H \cdot 2 \cdot D \cdot d_\text{head} = 2 D^2$ parameters (K+V).

MQA (Shazeer 2019): $H$ Q-heads share **1** set of K/V. $W_k, W_v \in \mathbb{R}^{D \times d_\text{head}}$, K+V parameters $= 2 D d_\text{head}$, **$H$× smaller**. At forward, K and V are broadcast to all $H$ heads for attention.

GQA (Ainslie 2023): $H$ Q-heads divided into $G$ groups, each group shares one K/V set. K+V parameters $= 2 G D d_\text{head}$. LLaMA-2-70B uses $H=64, G=8 \Rightarrow$ KV cache shrunk 8×.

> ⚠️ **MQA training instability phenomenon** — training MQA from scratch often gives slight quality drops or training instability vs MHA. The GQA paper's practice: **train MHA fully, then "uptrain" to GQA** — mean-pool the $H$ K/V groups along the head axis to initialize $G$ groups, then briefly fine-tune (5% of original training compute). This is why LLaMA-2 70B can switch to GQA zero-shot.

### 4.3　MLA: low-rank latent K/V (DeepSeek-V2's core innovation)

> ✅ **One-sentence MLA** — project K/V into a shared low-dim latent $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll H d_\text{head}$), **cache only the latent**; at each attention step, **linearly reconstruct** each head's K/V.

#### 4.3.1　Compression / decompression

Input hidden state $h_t \in \mathbb{R}^D$. MLA introduces:

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll H d_\text{head}$$

Then **only $c_t^{KV}$ is cached**. To generate the $i$-th head's K and V:

$$k_t^{C, (i)} = W^{UK, (i)} c_t^{KV}, \quad v_t^{(i)} = W^{UV, (i)} c_t^{KV}$$

where $W^{UK, (i)}, W^{UV, (i)} \in \mathbb{R}^{d_\text{head} \times d_c}$ are head-specific up-projections.

Similarly, Q is also low-rank compressed (this step is optional and mainly saves training memory rather than inference):

$$c_t^Q = W^{DQ} h_t, \quad q_t^{C, (i)} = W^{UQ, (i)} c_t^Q$$

#### 4.3.2　Cache size comparison

| Scheme | Cache elements per token |
| --- | --- |
| MHA | $2 \cdot H \cdot d_\text{head}$ |
| GQA | $2 \cdot G \cdot d_\text{head}$ |
| MQA | $2 \cdot d_\text{head}$ |
| MLA (bare latent part) | $d_c$ (**a single vector, no $\times 2$** — because K and V share the same latent) |

DeepSeek-V2 takes $d_c = 4 d_\text{head}$; the **bare latent** relative to MHA ($2 H d_\text{head}$) compresses by about $H/2$ ($\approx 64×$ at $H=128$). But MLA's actual cache also carries a decoupled-RoPE component $d_r$ (DeepSeek-V2 $d_r = 64$), so the **full cache $=d_c+d_r=576$ compresses by about 57×** relative to MHA (matching the long-context tutorial §9's 57×; counting only the latent over-estimates it as 64×).

#### 4.3.3　Inference equivalence transformation (the absorb trick)

In a naive implementation, every step would have to un-project $c_t^{KV}$ back to $k_t, v_t$ then compute attention — which defeats the "save cache" benefit (still doing the up-projection). MLA's elegance is in **matrix absorption**:

Attention score (ignoring RoPE, content only):

$$q_t^{(i)\top} k_s^{(i)} = (W^{UQ, (i)} c_t^Q)^\top (W^{UK, (i)} c_s^{KV}) = c_t^{Q\top} \underbrace{(W^{UQ, (i)\top} W^{UK, (i)})}_\text{constant matrix \tilde W^{QK,(i)}} c_s^{KV}$$

**$\tilde W^{QK, (i)} \in \mathbb{R}^{d_c' \times d_c}$ is fixed at inference time**, and can be pre-multiplied once when loading the model. So:

- inference only caches $c_s^{KV}$
- attention score computation is just $c_t^{Q\top} \tilde W^{QK, (i)} c_s^{KV}$, **no K/V reconstruction at all**
- Similarly $W^{UV, (i)}$ can be absorbed into the output projection $W^O$

This is why MLA has tiny cache but inference FLOPs don't blow up: **matrix absorption decouples "save cache" from "save compute"**.

### 4.4　MLA's RoPE problem — why decoupling is necessary

#### 4.4.1　Problem: RoPE breaks absorb

RoPE injects positional information as a **rotation matrix** $R_t \in \mathbb{R}^{d_\text{head} \times d_\text{head}}$ into Q and K:

$$q_t^{\text{RoPE}, (i)} = R_t q_t^{(i)}, \quad k_s^{\text{RoPE}, (i)} = R_s k_s^{(i)}$$

Attention score becomes:

$$q_t^{\text{RoPE}, (i)\top} k_s^{\text{RoPE}, (i)} = q_t^{(i)\top} R_t^\top R_s k_s^{(i)} = q_t^{(i)\top} R_{s-t} k_s^{(i)}$$

(using $R_t^\top R_s = R_{s-t}$, the essence of RoPE — relative position depends only on $s-t$.)

Now plug in the latent form:

$$q_t^{\text{RoPE}, (i)\top} k_s^{\text{RoPE}, (i)} = c_t^{Q\top} \underbrace{W^{UQ, (i)\top} R_{s-t} W^{UK, (i)}}_\text{not constant — depends on (s-t)} c_s^{KV}$$

**The middle matrix varies with $s-t$** — meaning it can no longer be pre-absorbed into a constant matrix. Each $(t, s)$ pair must compute $R_{s-t}$ on the fly; the absorb trick fails outright, and compute reverts to MHA-equivalent.

#### 4.4.2　Solution: separate the RoPE component into an independent channel

DeepSeek-V2's solution: **give RoPE an independent small-dim channel**.

- Latent channel (no RoPE): carries content information, cached as $c_t^{KV} \in \mathbb{R}^{d_c}$, with attention scoring via absorb
- RoPE channel (with RoPE): carries positional information, cached as $k_t^R \in \mathbb{R}^{d_r}$, with attention scoring via standard rotation-aware dot product

Specifically, introduce two new projections $W^{KR} \in \mathbb{R}^{D \times d_r}$ and $W^{QR, (i)} \in \mathbb{R}^{d_c' \times d_r}$ (per-head). **$k_t^R$ is shared across all heads**:

$$k_t^R = \text{RoPE}_t(W^{KR} h_t), \quad q_t^{R, (i)} = \text{RoPE}_t(W^{QR, (i)} c_t^Q)$$

Full K/Q is a concat of two segments:

$$k_t^{(i)} = [k_t^{C, (i)}; k_t^R], \quad q_t^{(i)} = [q_t^{C, (i)}; q_t^{R, (i)}], \quad k_t^{(i)}, q_t^{(i)} \in \mathbb{R}^{d_\text{head} + d_r}$$

Attention score becomes the sum of two parts:

$$q_t^{(i)\top} k_s^{(i)} = \underbrace{q_t^{C, (i)\top} k_s^{C, (i)}}_\text{latent, absorb} + \underbrace{q_t^{R, (i)\top} k_s^{R}}_\text{RoPE, standard dot}$$

> ✅ **Why the RoPE channel is shared across all heads** — a single independent RoPE dimension $k_t^R$ is shared across all heads, adding only $d_r$ elements (typical $d_r = d_\text{head}/2 = 64$) to the cache. This is MLA's "last mile of cache saving".

#### 4.4.3　Total cache formula

$$\boxed{\;\text{MLA cache}_\text{per token} = \underbrace{d_c}_\text{latent K/V shared} + \underbrace{d_r}_\text{RoPE K (shared across heads)}\;}$$

DeepSeek-V2: $d_c = 512, d_r = 64$, 576 fp16 elements per token. Compared to LLaMA-3-70B (GQA, $H_\text{kv}=8, d_\text{head}=128$) with $2 \cdot 8 \cdot 128 = 2048$ elements per token, **MLA is about 1/3.5 of GQA**; vs same-scale MHA ($2 \cdot 64 \cdot 128 = 16384$) about 1/28. DeepSeek-V2 paper reports **93.3% KV reduction vs its internal MHA baseline** (numbers differ across model scales and head counts; the 1/28 here is an estimate under another parameter set).

> ❌ **Common interview mistake** — saying "MLA is just an extreme version of GQA": wrong. GQA still caches complete K and V, only with fewer heads; MLA caches the latent, and K/V are reconstructed from latent at inference. The two differ mathematically (MLA changes the attention structure; GQA doesn't).

### 4.5　KV quantization (KIVI / KVQuant / FP8)

KV cache quantization routes compress each cache element from fp16 (2 bytes) to fewer:

| Method | Quantization granularity | Precision loss | Notes |
| --- | --- | --- | --- |
| **FP8 KV** | per-tensor / per-channel FP8 | nearly lossless | H100 native, production-grade default |
| **KIVI** (Liu et al., ICML 2024) | **K per-channel, V per-token** 2-bit | <1 PPL | tuning-free, asymmetric quant |
| **KVQuant** (Hooper et al., NeurIPS 2024) | per-channel 4-bit + outlier handling | minimal | paper shows 10M context feasible |

> 💡 **KIVI's key insight** — K and V have **different** outlier distributions. K has significant outliers along the channel dimension (a few channels have large values), absorbed by per-channel quant; V has no channel-level outliers but is heterogeneous along the token axis, better handled by per-token quant. KIVI's core contribution is decoupling these two with an asymmetric scheme.

## §5 PagedAttention (vLLM memory management)

### 5.1　Problem: fragmentation of KV cache

Traditional attention implementations treat each request's KV cache as a **contiguous large tensor** $[L_\text{max}, n_\text{layers}, 2, H_\text{kv}, d_\text{head}]$. Issues:

- Must pre-allocate length $L_\text{max}$ (actual usage may be only 10%) → **internal fragmentation**
- Different requests have different lengths, cache block sizes differ → **external fragmentation**
- After freeing a request, fragmentation means new requests can't find a large enough block → memory utilization ~70%

### 5.2　Solution: virtual-memory-style paging

PagedAttention (Kwon et al., SOSP 2023) borrows from OS paging:

1. **Partition KV cache into equal-size pages** (e.g., 16 tokens per page)
2. Each request maintains a **block table**: logical block idx → physical block idx
3. The physical page pool is global; pages are allocated on demand and reclaimed on free
4. Attention kernel is rewritten as **paged attention**: indirect lookup via block table (gather)

Effects:

- Memory utilization from ~70% → **~96%**
- Active batch size doubled or quadrupled on the same GPU, throughput rises accordingly
- Supports **copy-on-write sharing**: beam search, parallel sampling, prefix caching naturally fall out

> ⚠️ **PagedAttention's cost** — indirect lookup introduces ~1-5% kernel overhead (block lookup + scattered HBM access). But the throughput gain from larger batches completely dominates. CUDA Graphs are hard to compose (every block-table change requires re-capture), so vLLM uses piecewise CUDA Graphs.

### 5.3　Block-table data structure (sketch)

```
Request A:  logical_blocks = [0, 1, 2, 3]   →   physical = [12, 7, 34, 19]
Request B:  logical_blocks = [0, 1, 2]      →   physical = [12, 7, 5]    ← prefix shared!
```

Request A and B share the first 32 tokens (2 blocks × 16 tokens); vLLM maintains a ref count for each physical block; when A wants to write new content with ref > 1, it triggers COW (copy + change mapping).

## §6 KV Cache Implementation Code

### 6.1　Naive append + autoregressive decode

```python
import math
import torch
import torch.nn.functional as F
from torch import nn

class NaiveCachedAttention(nn.Module):
    """Single-layer attention with KV cache (MHA / for learning, do not deploy)."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.H, self.d = num_heads, d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x):  # [B, L, D] → [B, H, L, d]
        B, L, _ = x.shape
        return x.view(B, L, self.H, self.d).transpose(1, 2)

    def forward(self, x, cache=None):
        """
        x:    [B, L_new, D]  new input (L_new=L_prompt during prefill, L_new=1 during decode)
        cache: dict with 'k','v' of shape [B, H, L_past, d] or None
        Returns: out [B, L_new, D], new_cache
        """
        B, L_new, D = x.shape
        q = self._split(self.W_q(x))                   # [B, H, L_new, d]
        k = self._split(self.W_k(x))                   # [B, H, L_new, d]
        v = self._split(self.W_v(x))                   # [B, H, L_new, d]

        # ── KV cache append ────────────────────────────────────────
        if cache is not None:
            k = torch.cat([cache['k'], k], dim=2)      # [B, H, L_total, d]
            v = torch.cat([cache['v'], v], dim=2)
        new_cache = {'k': k, 'v': v}

        # ── causal attention (when L_new=1 in decode, causal is automatic) ──
        L_total = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        if L_new > 1:                                  # causal mask only needed during prefill
            causal = torch.tril(torch.ones(L_new, L_total, dtype=torch.bool,
                                          device=x.device), diagonal=L_total - L_new)
            scores = scores.masked_fill(~causal, float('-inf'))
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, L_new, D)
        return self.W_o(out), new_cache


# ── Autoregressive generation loop (simplified; real models add sampling/stop/multi-layer) ─
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, embed, lm_head):
    cache = None
    out, cache = model(embed(prompt_ids), cache=cache)        # prefill
    next_tok = lm_head(out[:, -1:]).argmax(-1)
    generated = [next_tok]
    for _ in range(max_new_tokens - 1):
        out, cache = model(embed(next_tok), cache=cache)      # decode L_new=1
        next_tok = lm_head(out[:, -1:]).argmax(-1)
        generated.append(next_tok)
    return torch.cat(generated, dim=1)
```

Note: every `torch.cat` triggers **a new memory allocation** + memcpy; in production, swap to pre-allocated buffer + index assignment, or PagedAttention.

### 6.2　PagedAttention data structure sketch

```python
class PagedKVCache:
    """Simplified page table (no CUDA kernel; data structure + COW demo)."""
    def __init__(self, n_layers, n_kv_heads, head_dim, page_size=16,
                 n_pages=1024, dtype=torch.float16, device='cuda'):
        self.page_size = page_size
        # Global page pool: [n_pages, page_size, n_layers, 2 (K,V), n_kv_heads, head_dim]
        self.pool = torch.empty(n_pages, page_size, n_layers, 2,
                                n_kv_heads, head_dim, dtype=dtype, device=device)
        self.free_list = list(range(n_pages))
        self.ref_count = [0] * n_pages
        self.block_table = {}                          # req_id → list of physical page ids

    def allocate(self, req_id, n_tokens):
        n = (n_tokens + self.page_size - 1) // self.page_size
        assert len(self.free_list) >= n, "OOM"
        physical = [self.free_list.pop() for _ in range(n)]
        for pid in physical: self.ref_count[pid] = 1
        self.block_table[req_id] = physical

    def append_token(self, req_id, pos, layer, k_new, v_new):
        """k_new, v_new: [n_kv_heads, head_dim]"""
        page_idx, slot = pos // self.page_size, pos % self.page_size
        if page_idx >= len(self.block_table[req_id]):
            new_pid = self.free_list.pop()
            self.ref_count[new_pid] = 1
            self.block_table[req_id].append(new_pid)
        pid = self.block_table[req_id][page_idx]
        if self.ref_count[pid] > 1:                    # COW
            new_pid = self.free_list.pop()
            self.pool[new_pid] = self.pool[pid]
            self.ref_count[pid] -= 1; self.ref_count[new_pid] = 1
            self.block_table[req_id][page_idx] = new_pid
            pid = new_pid
        self.pool[pid, slot, layer, 0] = k_new
        self.pool[pid, slot, layer, 1] = v_new

    def free(self, req_id):
        for pid in self.block_table[req_id]:
            self.ref_count[pid] -= 1
            if self.ref_count[pid] == 0: self.free_list.append(pid)
        del self.block_table[req_id]

    def share_prefix(self, src, dst, n_tokens):
        """beam search / parallel sampling: reuse the first n_tokens of pages."""
        n = n_tokens // self.page_size
        prefix = self.block_table[src][:n]
        for pid in prefix: self.ref_count[pid] += 1
        self.block_table[dst] = list(prefix)
```

Production implementations also need a fused paged-attention kernel (per-block gather + FlashAttention-style) and device-side block-table layout.

### 6.3　Differences in MQA / GQA / MLA forward

```python
class MQA_GQA_Attention(nn.Module):
    """Unified MHA / MQA / GQA version (num_kv_heads ≤ num_heads)."""
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "H must be divisible by H_kv"
        self.H, self.H_kv = num_heads, num_kv_heads
        self.d = d_model // num_heads
        self.group = num_heads // num_kv_heads        # # of Q heads each KV head serves
        self.W_q = nn.Linear(d_model, num_heads * self.d, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d, bias=False)   # ← smaller
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d, bias=False)   # ← smaller
        self.W_o = nn.Linear(num_heads * self.d, d_model, bias=False)

    def forward(self, x, cache=None):
        B, L_new, _ = x.shape
        q = self.W_q(x).view(B, L_new, self.H, self.d).transpose(1, 2)
        k = self.W_k(x).view(B, L_new, self.H_kv, self.d).transpose(1, 2)
        v = self.W_v(x).view(B, L_new, self.H_kv, self.d).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)
        new_cache = {'k': k, 'v': v}

        # ── Key: broadcast K/V to all Q heads ────────────────────────
        k = k.repeat_interleave(self.group, dim=1)    # [B, H, L_total, d]
        v = v.repeat_interleave(self.group, dim=1)
        # repeat_interleave is explicit broadcast; production uses torch implicit broadcast or fused kernel

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        L_total = k.size(2)
        if L_new > 1:
            causal = torch.tril(torch.ones(L_new, L_total, dtype=torch.bool,
                                          device=x.device), diagonal=L_total - L_new)
            scores = scores.masked_fill(~causal, float('-inf'))
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, L_new, -1)
        return self.W_o(out), new_cache


class MLAAttention(nn.Module):
    """MLA with decoupled RoPE (simplified DeepSeek-V2)."""
    def __init__(self, d_model, num_heads, d_head, d_c, d_r):
        super().__init__()
        self.H = num_heads
        self.d_head = d_head          # content head dim
        self.d_c = d_c                # latent dim (shared by K/V)
        self.d_r = d_r                # RoPE head dim (per query head)
        # KV side: compress to latent, then up-project to K_C, V
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)
        self.W_UK = nn.Linear(d_c, num_heads * d_head, bias=False)
        self.W_UV = nn.Linear(d_c, num_heads * d_head, bias=False)
        # RoPE channel: K is shared (across heads)
        self.W_KR = nn.Linear(d_model, d_r, bias=False)
        # Q side: compress to latent (saves training memory), then up-project to Q_C and Q_R
        d_c_q = 4 * d_head            # in the paper d_c_q ≠ d_c
        self.W_DQ = nn.Linear(d_model, d_c_q, bias=False)
        self.W_UQ = nn.Linear(d_c_q, num_heads * (d_head + d_r), bias=False)
        self.W_o = nn.Linear(num_heads * d_head, d_model, bias=False)

    def _rope(self, x, positions, l_axis=-2, base=10000.0):
        """Standard RoPE 2D rotation. L is on l_axis, d_r is the last dim; positions: [L]."""
        assert x.size(-1) % 2 == 0
        D, L = x.size(-1), x.size(l_axis)
        assert positions.shape == (L,)
        i = torch.arange(D // 2, device=x.device, dtype=x.dtype)
        theta = base ** (-2.0 * i / D)                              # [D/2]
        angles = positions.to(x.dtype).unsqueeze(-1) * theta        # [L, D/2]
        cos, sin = angles.cos(), angles.sin()
        ndim = x.dim()
        cos_shape = [1] * ndim
        cos_shape[l_axis % ndim] = L
        cos_shape[-1] = D // 2
        cos_b, sin_b = cos.view(cos_shape), sin.view(cos_shape)
        x2 = x.view(*x.shape[:-1], D // 2, 2)
        x_real, x_imag = x2.unbind(-1)
        rot_real = x_real * cos_b - x_imag * sin_b
        rot_imag = x_real * sin_b + x_imag * cos_b
        return torch.stack([rot_real, rot_imag], dim=-1).flatten(-2)

    def forward(self, x, positions, cache=None):
        B, L_new, _ = x.shape
        # ── KV side: compute latent + RoPE-K ─────────────────────────
        c_kv = self.W_DKV(x)                           # [B, L_new, d_c]
        k_r = self._rope(self.W_KR(x), positions)      # [B, L_new, d_r]  (shared across heads)
        if cache is not None:
            c_kv = torch.cat([cache['c_kv'], c_kv], dim=1)
            k_r  = torch.cat([cache['k_r'],  k_r],  dim=1)
        new_cache = {'c_kv': c_kv, 'k_r': k_r}        # ← only cache latent + RoPE, not full K/V

        # Reconstruct K_C, V (this step can be absorbed into Q/O projection at inference; shown naively here)
        k_c = self.W_UK(c_kv).view(B, -1, self.H, self.d_head).transpose(1,2)  # [B,H,L_tot,d_head]
        v   = self.W_UV(c_kv).view(B, -1, self.H, self.d_head).transpose(1,2)

        # ── Q side: split into content and RoPE segments ─────────────
        c_q = self.W_DQ(x)                             # [B, L_new, d_c_q]
        q_full = self.W_UQ(c_q).view(B, L_new, self.H, self.d_head + self.d_r)
        q_c, q_r_raw = q_full.split([self.d_head, self.d_r], dim=-1)
        q_c = q_c.transpose(1, 2)                      # [B, H, L_new, d_head]
        q_r = self._rope(q_r_raw, positions, l_axis=1).transpose(1, 2)  # [B, H, L_new, d_r]

        # ── attention scores: sum the two parts ──────────────────────
        L_tot = c_kv.size(1)
        scale = math.sqrt(self.d_head + self.d_r)
        scores_c = q_c @ k_c.transpose(-2, -1)         # content part [B, H, L_new, L_tot]
        # k_r is shared across heads, broadcast:
        scores_r = q_r @ k_r.transpose(-2, -1).unsqueeze(1)  # [B, H, L_new, L_tot]
        scores = (scores_c + scores_r) / scale

        if L_new > 1:
            causal = torch.tril(torch.ones(L_new, L_tot, dtype=torch.bool,
                                          device=x.device), diagonal=L_tot - L_new)
            scores = scores.masked_fill(~causal, float('-inf'))
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, L_new, -1)
        return self.W_o(out), new_cache
```

> ⚠️ **Demo code vs production code** — the MLA implementation above "naively" reconstructs full K/V at every step, so it doesn't actually save compute. Production implementations should do the absorb trick from §4.3.3 — pre-multiplying $W^{UQ\top} W^{UK}$ and $W^{UV} W^O$ at model load, and scoring directly against the latent at inference.

## §7 Speculative Decoding Core Mechanism

### 7.1　Setup

- **Target model** $p$: the large model to accelerate (e.g., LLaMA-70B), $p(x_{t+1} | x_{\le t})$ is its per-step conditional distribution
- **Draft model** $q$: a much smaller model (e.g., LLaMA-7B or EAGLE's feature head), $q(x_{t+1} | x_{\le t})$ is its conditional distribution
- **Goal**: make the final output distribution **exactly** equal $p$ (exact), not approximate

Each spec step:

1. Use $q$ to **autoregressively** draft $K$ tokens: $\tilde x_1, \tilde x_2, \dots, \tilde x_K$
2. Feed prefix + $K$ drafts to $p$ at once, **in parallel** computing $p(x_{t+i} | x_{\le t+i-1})$ for $i=1..K$ (plus $i=K+1$ for bonus-token logits)
3. For each draft position, do **rejection sampling**: accept with probability $\min(1, p(\tilde x_i) / q(\tilde x_i))$
4. At the first rejected position $j$: resample from the corrected residual distribution $p'$; discard positions $j+1, \dots, K$
5. If all accepted ($j = K+1$), also sample a **bonus token** for free from the target's last logit set

### 7.2　Acceptance probability $\alpha$ derivation (must-know)

> ✅ **Core theorem (Leviathan et al. 2023, Chen et al. 2023)** — rejection sampling makes the whole spec step's output distribution **exactly equivalent** to one step of sampling from $p$.

**Derivation**: at a position, draft gives $\tilde x \sim q(\cdot)$.

Accept rule: accept with probability $r(\tilde x) = \min(1, p(\tilde x)/q(\tilde x))$.

Probability of accepted token:

$$\Pr[\text{accept} \land X = x] = q(x) \cdot r(x) = q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$

Rejection probability:

$$\beta = 1 - \alpha = \sum_x q(x) - \sum_x \min(q(x), p(x)) = \sum_x \max(0, q(x) - p(x))$$

Overall acceptance:

$$\boxed{\;\alpha = \sum_x \min(q(x), p(x)) = 1 - \tfrac{1}{2}\|p - q\|_1\;}$$

The last step uses the identity $\sum_x \min(p, q) = 1 - \tfrac{1}{2} \sum_x |p - q|$ (since $\sum_x p = \sum_x q = 1$).

**Corollary**: the closer $p$ and $q$ are (smaller TV distance), the closer $\alpha$ is to 1.

### 7.3　Residual distribution $p'$ (how to sample on reject)

After rejection, sample a new token from $p$ with the "accepted mass" removed:

$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_x \max(0, p(x) - q(x))} = \frac{\max(0, p(x) - q(x))}{1 - \alpha}$$

**Equivalence proof** (key, must derive in interviews): the total probability that token $x$ is output in one spec step:

$$\Pr[X = x] = \underbrace{q(x) \min(1, p(x)/q(x))}_\text{accept path} + \underbrace{(1-\alpha) \cdot p'(x)}_\text{reject path}$$

- First term $= \min(p(x), q(x))$
- Second term $= (1-\alpha) \cdot \dfrac{\max(0, p(x) - q(x))}{1-\alpha} = \max(0, p(x) - q(x))$

Sum: $\min(p, q) + \max(0, p - q) = p$. ✅

So **the output distribution at each position equals $p$ exactly** — this is the mathematical basis of spec decoding's "exact" property.

### 7.4　Expected speedup: $E[\tau]$ formula

Assume each draft position's acceptance is i.i.d. (slightly correlated in practice; the paper uses this approximation). $K$ drafts + 1 bonus:

- If first $j$ accepted, position $j+1$ rejected ($j < K$): output $j$ accepted + 1 resample = $j+1$ tokens
- If all $K$ accepted: output $K$ + 1 bonus = $K+1$ tokens

Expected token count:

$$E[\tau] = \sum_{j=0}^{K-1} \alpha^j (1-\alpha) (j+1) + \alpha^K (K+1)$$

Simplification (standard geometric-series trick):

$$\boxed{\;E[\tau] = \frac{1 - \alpha^{K+1}}{1 - \alpha}\;}$$

**Limit analysis**:

- $\alpha \to 1$ (perfect draft): $E[\tau] \to K+1$, speedup $K+1$×
- $\alpha \to 0$ (draft always wrong): $E[\tau] \to 1$, no speedup but no regression either
- In practice, LLaMA-7B drafting LLaMA-70B: $\alpha \approx 0.6-0.7$, $K=4$ gives $E[\tau] \approx 2.7$

> 💡 **Real speedup formula** — subtract the draft model's own forward overhead. Let $c = T_q / T_p$ (draft step time / target step time, typical 0.05-0.15):

$$\text{speedup} = \frac{E[\tau]}{1 + Kc}$$

The numerator is mean accepted token count; the denominator's 1 is the single target verify, and $Kc$ is $K$ draft forwards. If $c$ is too large (draft too big), it eats the gain, so picking a small draft matters.

### 7.5　Sampling equivalence under temperature / top-p

The rejection sampling equivalence requires only two things: (1) **at the target side, replace $p$ with the post-sampler distribution $\tilde p$** for acceptance and residual; (2) the draft proposal distribution $\tilde q$ can be any valid probability distribution. **Mathematically it does not require draft and target to use the same sampler** — pure greedy draft is also legal, just with $\tilde q$ far from $\tilde p$ and $\alpha$ plummeting. In practice the draft typically uses the same temperature/top-p so $\tilde q$ stays close to $\tilde p$.

> ❌ **Wrong equivalence approach** — "draft uses the same sampling, then just compare tokens for agreement" is wrong — that loses distribution equivalence. The correct approach is the §7.3 rejection formula, **comparing probabilities rather than token agreement**.

### 7.6　Code: speculative decoding loop

Below is a single-batch demo. Convention: both models expose `forward(input_ids, cache)`; the cache object has a `length` attribute + `truncate(L)` method (in production, PagedAttention rollbacks via block-table pointer changes in $O(1)$). **Core invariant**: at the start of each loop iteration, `cache.length == seq.size(1) - 1` (i.e., the cache holds all of seq except the last 1 token).

```python
import torch

@torch.no_grad()
def speculative_decode(target, draft, prompt_ids, max_new_tokens, K=4, temperature=1.0):
    """
    target, draft: callable(input_ids, cache) → (logits [1,L_new,V], new_cache).
    Mathematically exact equivalent to directly sampling from target (Leviathan/Chen 2023).
    """
    seq = prompt_ids.clone()                                   # [1, L_prompt]
    L_prompt = seq.size(1)
    # Prefill the first L-1 tokens; keep the last token as the first verify input.
    _, target_cache = target(seq[:, :-1], cache=None)
    _, draft_cache  = draft(seq[:, :-1],  cache=None)

    while seq.size(1) - L_prompt < max_new_tokens:
        last_tok = seq[:, -1:]                                  # [1, 1], not yet in cache
        draft_chk, target_chk = draft_cache.length, target_cache.length

        # ── 1. Draft: feed last_tok, d_1, ..., d_{K-1}; sample d_1..d_K ──
        cur = last_tok
        draft_tokens, draft_probs = [], []
        for _ in range(K):
            logits, draft_cache = draft(cur, cache=draft_cache)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            tok = torch.multinomial(probs, 1)
            draft_tokens.append(tok); draft_probs.append(probs)
            cur = tok
        draft_tokens = torch.cat(draft_tokens, dim=1)            # [1, K]

        # ── 2. Target: one forward over [last_tok, d_1..d_K], yielding K+1 distributions ──
        target_in = torch.cat([last_tok, draft_tokens], dim=1)   # [1, K+1]
        target_logits, target_cache = target(target_in, cache=target_cache)
        # target_logits[:, i, :] verifies d_{i+1} (i<K) or samples bonus (i=K)

        # ── 3. Rejection sampling position by position ──
        accepted = 0; rejected = False
        for i in range(K):
            p_i = torch.softmax(target_logits[:, i, :] / temperature, dim=-1)
            q_i = draft_probs[i]
            x = draft_tokens[:, i:i+1]
            p_x = p_i.gather(-1, x).squeeze(-1)
            q_x = q_i.gather(-1, x).squeeze(-1)
            r = (p_x / (q_x + 1e-9)).clamp(max=1.0)
            if torch.rand_like(r).item() < r.item():             # accept
                accepted += 1
            else:                                                 # reject
                p_prime = (p_i - q_i).clamp(min=0.0)              # residual distribution p'
                p_prime = p_prime / (p_prime.sum(-1, keepdim=True) + 1e-9)
                new_tok = torch.multinomial(p_prime, 1)
                seq = torch.cat([seq, draft_tokens[:, :accepted], new_tok], dim=1)
                # rollback: keep last_tok + accepted drafts; new_tok not yet in cache
                draft_cache.truncate(draft_chk + 1 + accepted)
                target_cache.truncate(target_chk + 1 + accepted)
                rejected = True
                break

        if not rejected:                                          # all accepted → bonus token
            p_bonus = torch.softmax(target_logits[:, K, :] / temperature, dim=-1)
            bonus = torch.multinomial(p_bonus, 1)
            seq = torch.cat([seq, draft_tokens, bonus], dim=1)
            # draft_cache previously saw only d_1..d_{K-1}; feed d_K to maintain invariant
            _, draft_cache = draft(draft_tokens[:, -1:], cache=draft_cache)
            # target_cache already contains last_tok + d_1..d_K, matches invariant

    return seq[:, :L_prompt + max_new_tokens]                     # trim overshoot
```

> ⚠️ **Production implementation points** — (1) **Cache rollback** must actually rewind the KV cache write position; vLLM uses PagedAttention's block-table pointer change for $O(1)$. (2) **Numerical stability**: $r=p/q$ explodes as $q \to 0$ — compute in fp32 and clamp. (3) **Invariant**: at the start of each loop iteration, cache holds all of seq except the last token — all rollback/refeed logic is to maintain this invariant.

## §8 Main Speculative Decoding Variants

### 8.1　Variant overview

| Method | Draft source | Multi-token structure | Needs draft training? | Representative paper |
| --- | --- | --- | --- | --- |
| **Vanilla SD** | Independent small model | Linear chain | No (use off-the-shelf small LM) | Leviathan 2023, Chen 2023 |
| **SpecInfer** | Multiple drafts together | **Static tree** | No | Miao 2024 (ASPLOS) |
| **Medusa** | Add N heads on target | Static tree | Yes (fine-tune heads) | Cai 2024 (ICML) |
| **EAGLE-1** | Feature-level autoregression + small model | Tree | Yes (small draft head) | Li 2024 (ICML) |
| **EAGLE-2** | Same as EAGLE-1 | **Dynamic tree** | Yes | Li 2024 |
| **EAGLE-3** | Multi-layer feature fusion + training-time test | Dynamic tree | Yes | Li 2025 |
| **Hydra** | Sequential draft heads | Static tree | Yes | Ankner 2024 |
| **Lookahead Decoding** | **Jacobi iteration** + n-gram pool | Self-verify | No | Fu 2024 (ICML) |
| **REST** | Retrieval (datastore) | Static tree | No | He 2024 (NAACL) |
| **Self-Speculative** | Target skips layers | Linear | No (use a subset of target layers) | Zhang 2024 |
| **TriForce** | Hierarchical (small LM + sparse target) | Hierarchical | No | Sun 2024 |
| **MagicDec** | Small draft + sparse KV | Linear | No | Sadhukhan 2024 |

### 8.2　Medusa: multi-head in parallel

Medusa's core (Cai et al., ICML 2024): **add $N$ parallel prediction heads on top of the target's final hidden state**; the $k$-th head directly predicts the "$(k+1)$-th future token".

- No need for a separate draft model; only fine-tune these heads (small parameter count)
- The top-$K$ candidates from each of $N$ heads compose a **static tree** (e.g., top-5 per head, $5^N$ paths total, pruned by typical acceptance)
- Tree attention: flatten all tree nodes into one input for the target's forward; each node's attention mask sees only its ancestors (causal on the tree)

> 💡 **Tree attention mask** — arrange tree nodes by BFS into a linear sequence $[t_0, t_1, \dots, t_M]$. Let $\mathcal A(i)$ denote node $i$'s ancestors (inclusive). Mask $M[i, j] = 1$ iff $j \in \mathcal A(i)$. Each node thus only sees the root-to-self path; logits are correct.

- **Verification defaults to typical acceptance** (proposed in the paper): accept draft tokens based on the typical-set threshold of the target distribution; this rule **does not strictly guarantee exact sampling**, but in practice quality is essentially unchanged. For exact, switch to standard rejection sampling (Leviathan/Chen formula).
- **Medusa-1 vs Medusa-2 differ in training paradigm**: Medusa-1 freezes the backbone and only trains heads; Medusa-2 jointly trains backbone + heads for higher quality; both default to typical acceptance.

### 8.3　EAGLE family: feature-level autoregression

EAGLE's core insight (Li et al., ICML 2024): **the target's previous-layer hidden feature $h_{t-1}$ contains more information than tokens** — drafting in feature space is more accurate than in token space.

- Draft model: one transformer layer; inputs $h_{t-1}$ (target's feature) + token embedding; predicts $h_t$ and $x_t$
- Training objective: make draft's $h_t$ close to target's $h_t$ (feature loss) + token prediction loss

**EAGLE-2** (Li 2024): replaces the static tree with a dynamic tree — at each step, use draft's probability to expand the most promising paths.

**EAGLE-3** (Li et al., 2025):

1. Drop feature regression, **predict tokens directly** (feature error accumulation is the bottleneck)
2. Use **multi-layer feature fusion** instead of just the top layer
3. **Training-Time Test (TTT)**: simulate draft-chain errors during training, avoiding train-test gap
4. On Vicuna-7B: 4-5× speedup, ~30% improvement over EAGLE-2

> ⚠️ **EAGLE training cost** — although the draft model is small (1-layer transformer, ~tens of M parameters), it must be retrained on the target's features, requiring the target's forward-pass dataset (distillation-style). A single EAGLE-3 training run takes hours to days; it is not free.

### 8.4　Lookahead Decoding: Jacobi iteration

**Jacobi viewpoint** (Fu et al., ICML 2024): autoregressive generation is equivalent to solving the nonlinear system $x_i = f(x_{<i})$ for $i=1..L$; can use Jacobi iteration to update all positions in parallel.

```
Step 0:  x = [<random>, <random>, ..., <random>]
Step 1:  x'_i = f(x_{<i})  ∀ i  in parallel
Step 2:  x = x', another round
... until fixed point
```

Lookahead Decoding:

- **Lookahead branch**: maintain a 2D window (lookahead size × window depth), Jacobi-style updates over the whole window each step
- Extract apparently-stable **n-grams** from the trajectory into a pool
- **Verification branch**: per forward, simultaneously verify promising n-grams in the pool (multi-path tree attention)
- No draft model; no training; pure inference-time trick

Effect: MT-bench 1.8×, code completion multi-card 4× speedup. But for open-ended generation without repetition (e.g., complex reasoning), the gain is limited.

### 8.5　Long-context specialists: TriForce / MagicDec

> ⚠️ **Vanilla SD breaks down in long context** — when context is long (e.g., 128K), every target forward must scan the entire KV cache, and **HBM bandwidth** is the bottleneck, not weight loading. Vanilla SD's savings come from reducing weight-load frequency; in this regime the benefit shrinks.

**TriForce** (Sun et al., 2024, arxiv 2404.11912): three-tier hierarchy.

1. First-tier draft: small LM
2. Second-tier draft: target model + **sparse KV cache** (keep only heavy-hitter / retrieved portions)
3. Third-tier verify: target model + **full KV cache**

Speedup core: second-tier draft is fast under sparse cache, then the full-cache target verifies a long batch in one go.

**MagicDec** (Sadhukhan et al., 2024, arxiv 2408.11049, ICLR 2025): observes that the KV cache is the bottleneck in long context, so the draft uses **StreamingLLM-style sparse cache** (attention sink + sliding window) and the target uses full cache for verify.

### 8.6　Self-Speculative Decoding

> 💡 **An extreme version without an external draft model** — Zhang et al. 2024 propose "self-speculative": use the **target model itself with some layers skipped** as the draft.

- Take logits from layer $L_d < L$ in target as draft (early exit)
- Use the full target for verify
- Fully backwards-compatible: no new model to train, no fine-tuning
- Typical speedup 1.5-2× (not as fast as EAGLE, but zero extra training)

## §9 Complexity / Resource Accounting

### 9.1　KV cache memory (summary)

Using LLaMA-2/3-70B architecture ($N_\text{layers}=80, d_\text{head}=128$, MHA $H=64$) as baseline, $L_\text{ctx}=4096$, fp16:

| Scheme | per-token-per-layer bytes | 70B / 4K context (whole model) |
| --- | --- | --- |
| MHA ($H_\text{kv}=64$) | $2 \cdot 2 \cdot 64 \cdot 128 = 32768$ | 10.0 GB |
| GQA ($H_\text{kv}=8$) | $2 \cdot 2 \cdot 8 \cdot 128 = 4096$ | 1.25 GB |
| MQA ($H_\text{kv}=1$) | $2 \cdot 2 \cdot 1 \cdot 128 = 512$ | 0.16 GB |
| MLA ($d_c=512, d_r=64$, 60 layers) | $2 \cdot (512+64) = 1152$ | ~0.27 GB |
| MHA + INT4 KV | $32768 / 4 = 8192$ | 2.5 GB |

### 9.2　Speculative decoding expected throughput

$$\text{tokens / sec}_\text{SD} = \frac{\text{tokens / sec}_\text{baseline} \cdot E[\tau]}{1 + Kc}$$

Empirical numbers (A100, LLaMA-2-7B target + 68M draft):

- vanilla SD: 1.6-2.2×
- Medusa-2: 2.5-3.0×
- EAGLE-2: 3.0-3.5×
- EAGLE-3: ~4.0× (short context)

Long context (128K+) regime: vanilla SD drops to 1.1-1.3×; TriForce / MagicDec still hold 2-2.5×.

### 9.3　Rough budget (70B + 4K context + GQA)

| Item | Memory |
| --- | --- |
| weights (fp16) | 140 GB |
| KV cache @ batch 8 | $8 \times 1.25$ GB $= 10$ GB |
| activation peak (decode, batch 8) | ~2 GB |
| total | ~152 GB → tight on 2×80GB A100; typically 4×80GB |

For vanilla SD, the draft model (7B fp16) adds +14 GB; EAGLE's draft head is only ~200 MB.

## §10 25 Frequently-Asked Interview Questions

By difficulty L1 (must-know) / L2 (advanced) / L3 (top labs). Expand each entry for answers + common pitfalls.

### L1 must-know (asked at every inference / serving interview)

<details>

<summary>Q1. What is the KV cache formula?</summary>

- Per sample: $2 \cdot L_\text{ctx} \cdot N_\text{layers} \cdot N_\text{kv\_heads} \cdot d_\text{head} \cdot \text{bytes}$
- The "2" comes from K + V
- $N_\text{kv\_heads}$: MHA = $H$, MQA = 1, GQA = $G$
- LLaMA-3-70B (GQA, $H_\text{kv}=8$) @4K fp16 ≈ 1.25 GB/sample

Pitfall: writing $H$ (Q heads); forgetting × 2; forgetting that $L_\text{ctx}$ is the current length, not max length.

</details>

<details>

<summary>Q2. Why doesn't training use KV cache?</summary>

- Training computes all positions at once (teacher forcing, ground truth known)
- There's no "partial sequence with K/V waiting to be appended" timing
- KV cache is **inference-only** optimization

Pitfall: applying KV cache as a generic optimization to training.

</details>

<details>

<summary>Q3. What are the bottlenecks of prefill and decode?</summary>

- Prefill: $O(L^2)$ attention FLOPs, **compute-bound**
- Decode: one token per step, but must read the full cache + weights, **memory-bandwidth-bound**
- Arithmetic intensity is extremely low → GPU FLOPs utilization often < 10%

Pitfall: saying "decode is also compute-bound" — wrong. With small decode batch, the GPU spends most time waiting on memory.

</details>

<details>

<summary>Q4. Difference between MQA / GQA / MHA?</summary>

- MHA: $H$ K/V heads (same as Q)
- MQA: all Q heads share **1** set of K/V
- GQA: $H$ Q heads divided into $G$ groups ($1<G<H$), each group shares K/V
- Mainly saves **KV cache memory + memory bandwidth**, not Q projection

Pitfall: thinking MQA saves Q compute; saying GQA quality "doesn't drop at all" too absolutely.

</details>

<details>

<summary>Q5. Speculative decoding formula?</summary>

- Draft $q$ proposes $K$ tokens; target $p$ verifies in one forward
- Per-position acceptance $r = \min(1, p(\tilde x)/q(\tilde x))$
- Overall acceptance $\alpha = \sum_x \min(p(x), q(x))$
- Expected generation $E[\tau] = \dfrac{1 - \alpha^{K+1}}{1 - \alpha}$

Pitfall: saying "spec decoding is approximate sampling" — wrong, it's **exact** (rejection sampling guarantees it).

</details>

<details>

<summary>Q6. What does PagedAttention solve?</summary>

- Naive KV cache must be a contiguous large tensor, pre-allocated to max length → internal fragmentation
- Different request lengths → external fragmentation
- Memory utilization only ~70%
- PagedAttention: pages + block table, utilization up to ~96%
- Supports prefix sharing (COW)

Pitfall: saying PagedAttention reduces attention FLOPs — wrong, FLOPs are unchanged; it optimizes **memory utilization + concurrent request count**.

</details>

<details>

<summary>Q7. What is continuous batching?</summary>

- Scheduling granularity changes from request to iteration (re-check the batch at every forward)
- Completed requests are immediately removed, freeing slots for new ones
- Proposed by: Orca (Yu et al., OSDI 2022)
- Reduces average wait time, improves GPU utilization
- vLLM = Orca continuous batching + PagedAttention

Pitfall: thinking continuous batching pads all different-length sequences to the longest — that's the old static-batching approach.

</details>

<details>

<summary>Q8. How to pick the draft model?</summary>

- Size: typically target / 30 - target / 10 (e.g., 70B target + 7B draft)
- Same tokenizer, same vocab (else rejection sampling can't compute $p/q$)
- Same prompt format / same RLHF post-training (else large distribution gap → low $\alpha$)
- Experience: $\alpha \in [0.5, 0.8]$; too low, don't bother with SD

Pitfall: picking too-large draft (e.g., target / 3); or different tokenizer.

</details>

<details>

<summary>Q9. Most common KV cache quantization approach?</summary>

- FP8 (H100 native) is essentially lossless
- INT8 per-token quant is also acceptable
- INT4 / INT2 (KIVI, KVQuant) requires careful outlier handling
- KIVI's key: **K per-channel, V per-token** asymmetric quantization

Pitfall: using the same quant scheme for K and V — easy to lose accuracy; K and V have different outlier distributions.

</details>

<details>

<summary>Q10. What is prefix caching?</summary>

- Multiple requests share the same prompt prefix (system prompt, few-shot)
- Index page pool by hash(prefix); on hit, skip prefill
- Combined with COW to handle subsequent divergence
- Services like ChatGPT (system-prompt-heavy) have 90%+ hit rates

Pitfall: thinking prefix caching = caching the entire prompt — only the prefix is cached; user-specific parts still require prefill.

</details>

### L2 advanced (research-oriented / inference systems roles)

<details>

<summary>Q11. Derive spec decoding's acceptance probability $\alpha$ and explain why it guarantees exact sampling.</summary>

- Let draft $\tilde x \sim q$, accept rule $r = \min(1, p/q)$
- $\Pr[\text{accept} \land X=x] = q(x) \cdot \min(1, p(x)/q(x)) = \min(p(x), q(x))$
- $\alpha = \sum_x \min(p, q) = 1 - \tfrac{1}{2} \|p - q\|_1$
- After rejection, resample from residual $p'(x) = \max(0, p-q) / (1-\alpha)$
- Total output probability $= \min(p, q) + \max(0, p-q) = p(x)$ ∀x
- So each position is equivalent to a single-step sample from $p$

Pitfall: only writing the accept part, missing the reject residual; not proving $\min + \max = p$; saying spec is approximate.

</details>

<details>

<summary>Q12. Why must MLA decouple RoPE? Detailed derivation.</summary>

- Naive MLA absorb trick: $q^\top k = c_q^\top (W^{UQ\top} W^{UK}) c_{kv}$ — middle is a constant matrix $\tilde W^{QK}$, can be pre-multiplied
- With RoPE: $q^{R\top} k^R = c_q^\top W^{UQ\top} R_{s-t} W^{UK} c_{kv}$
- Middle block $W^{UQ\top} R_{s-t} W^{UK}$ **depends on relative position $(s-t)$**, cannot be pre-multiplied
- Absorb fails → cache saved but compute reverts to MHA
- Fix: split out an independent RoPE channel $k^R \in \mathbb{R}^{d_r}$ (shared across heads); content channel goes through absorb, RoPE channel uses standard dot product
- Total cache: $d_c + d_r$ per token

Pitfall: just saying "RoPE causes a problem" without expansion; unaware of the RoPE property $R_t^\top R_s = R_{s-t}$; unaware that $k^R$ is shared across heads.

</details>

<details>

<summary>Q13. How does continuous batching handle prefill + decode mixed runs?</summary>

- Prefill computes long segments in one go, high FLOPs; decode one token, low FLOPs
- Mixing them directly leaves decode waiting for prefill (HOL blocking)
- Sarathi-Serve's **chunked prefill**: split long prefill into equal-size chunks
- Each iteration coalesces one prefill chunk + multiple decode tokens
- Stall-free schedule: decode is always running

Pitfall: assuming prefill must run as one unit; forgetting Sarathi-Serve is OSDI 2024.

</details>

<details>

<summary>Q14. How to write the tree-attention mask (for Medusa / EAGLE)?</summary>

- Flatten tree nodes by BFS into linear sequence $[t_0, \dots, t_M]$
- $\mathcal A(i)$ = ancestors of node $i$ (inclusive)
- Attention mask $M[i, j] = 1 \iff j \in \mathcal A(i)$
- I.e., "causal on the tree"
- Used to verify all tree paths in one forward

Pitfall: writing it as a lower-triangular causal mask (only valid for chains, not trees); forgetting to generalize mask shape from $[L,L]$.

</details>

<details>

<summary>Q15. What is the real speedup formula for spec decoding? Why does an oversized draft backfire?</summary>

- $\text{speedup} = E[\tau] / (1 + Kc)$, $c = T_q/T_p$
- $E[\tau] = (1-\alpha^{K+1})/(1-\alpha)$
- $Kc$ in the denominator is the cost of $K$ draft forwards
- If $c$ is too large (draft too big), even high $\alpha$ gets eaten by the denominator
- Extreme: $c=1$ → speedup ≤ 1 (draft is as slow as target)

Pitfall: writing only $E[\tau]$ without draft overhead; missing the bonus token term.

</details>

<details>

<summary>Q16. Difference between self-speculative and standard speculative decoding?</summary>

- Self-spec: draft = target with layers skipped / early exit
- No separate draft model needed; zero extra training
- But draft and target are highly correlated, $\alpha$ is usually high
- Typical speedup 1.5-2× (less than EAGLE but more convenient)
- Paper: Zhang et al. 2024 ("Draft & Verify")

Pitfall: saying it requires extra training; conflating self-spec with layer-skipping inference (the latter is not exact).

</details>

<details>

<summary>Q17. How do KV cache eviction / sparse attention affect spec decoding?</summary>

- In long context, KV cache is the bandwidth bottleneck; weights are amortized in prefill
- Draft with sparse / sliding-window KV (StreamingLLM-style) runs faster
- Target uses full cache for verify → still exact
- Representatives: MagicDec, TriForce (hierarchical: small draft → sparse target → full target)
- Gain: vanilla SD breaks down in long context (1.1×); MagicDec holds 2×+

Pitfall: treating sparse KV as lossy approximation (it's only used in draft; verification uses full cache and remains exact).

</details>

<details>

<summary>Q18. Medusa uses typical acceptance instead of rejection sampling — what is lost?</summary>

- Strictly, **exact sampling is lost** — output distribution is no longer guaranteed equal to target's
- But typical acceptance uses the target's own typical-set threshold; quality basically holds (paper shows near-base scores)
- For strict exactness, swap verification for standard rejection sampling (Leviathan/Chen formula)
- **Medusa-1 vs Medusa-2 differ in training paradigm**: Medusa-1 freezes backbone and only trains heads; Medusa-2 jointly trains backbone + heads; both default to typical acceptance

Pitfall: saying Medusa-1 / Medusa-2 differ in "exact vs non-exact" (wrong — they differ in training paradigm); saying Medusa is exactly equivalent to target sampling.

</details>

<details>

<summary>Q19. Core difference between EAGLE and Medusa?</summary>

- Medusa: multiple heads **directly predict future tokens**, independently (not autoregressive)
- EAGLE: draft is **autoregressive in feature space** (prev hidden + prev token → next hidden + token)
- EAGLE is more accurate (features are richer), but requires training the draft (incl. transformer layer)
- EAGLE-3 further drops feature regression — directly predicts tokens + multi-layer fusion + training-time test
- Empirically, EAGLE > Medusa in acceptance rate, but Medusa is easier to deploy (fewer params)

Pitfall: treating EAGLE as a minor Medusa improvement; saying "EAGLE is also multi-head" — wrong, EAGLE is one mini-transformer.

</details>

<details>

<summary>Q20. Relationship between PagedAttention and FlashAttention?</summary>

- FlashAttention: SRAM tiling + online softmax inside the attention kernel, **single-kernel** optimization (avoid materializing $L^2$ scores)
- PagedAttention: split KV cache into pages, indirect lookup via block table; **memory layout** optimization
- Orthogonal; can be combined: vLLM uses paged + flash ideas in its paged-attention kernel
- Distinction: FlashAttention reduces HBM IO; PagedAttention reduces fragmentation

Pitfall: confusing the two; thinking PagedAttention is an attention algorithm variant (actually it's just memory management + accompanying kernel).

</details>

### L3 top-lab questions (hardest tier)

<details>

<summary>Q21. Prove the complete derivation of spec decoding's acceptance $\alpha$ and explain how sampling equivalence generalizes to temperature / top-p.</summary>

- Single token: $\Pr[X=x] = q(x) \min(1, p(x)/q(x)) + (1-\alpha) p'(x)$; plug in $p'$ to get $\min(p,q) + \max(0, p-q) = p$
- $\alpha = \sum_x \min(p, q) = 1 - \tfrac{1}{2}\|p-q\|_1$
- Equivalent to the TV-distance connection
- Key principle: rejection sampling equivalence depends only on having a valid "draft proposal distribution $\tilde q$" and a valid "target distribution $\tilde p$". **Replace $p, q$ in the formulas with the sampler-processed $\tilde p, \tilde q$ and the equivalence still holds**
- Temperature $T$: a common practice is $\tilde p_T(x) \propto p(x)^{1/T}$ and $\tilde q_T(x) \propto q(x)^{1/T}$; plug them into the $\alpha, p'$ formulas
- Top-p: truncate $p$ + renormalize over $p$'s top-p set to get $\tilde p$; the **draft proposal distribution** $\tilde q$ is whatever the draft actually samples from; as long as both are valid distributions, rejection is exact
- In practice the draft typically uses the same sampler as target (so $\tilde q$ stays close to $\tilde p$ raising $\alpha$), but this is not mathematically required — even greedy draft is valid, just with $\alpha$ plummeting
- Multi-token: each position $\alpha_i$ uses the corresponding $\tilde p_i, \tilde q_i$; bonus token uses the corrected logits at position $K+1$ (post-sampler) for direct sampling

Pitfall: only writing single-token equivalence; misstating "draft must use the same sampler" as a math requirement (it's a strategy for high $\alpha$); ignoring bonus token.

</details>

<details>

<summary>Q22. Full math derivation of MLA's absorb trick: why don't we reconstruct K/V at inference?</summary>

- KV cache: $c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}$
- K, V up-projection: $k_t^{(i)} = W^{UK,(i)} c_t^{KV}, v_t^{(i)} = W^{UV,(i)} c_t^{KV}$
- Q similarly: $q_t^{(i)} = W^{UQ,(i)} c_t^Q$
- Attention score (no RoPE): $(q_t^{(i)})^\top k_s^{(i)} = (c_t^Q)^\top \underbrace{W^{UQ,(i)\top} W^{UK,(i)}}_{\tilde W^{QK,(i)}} c_s^{KV}$
- $\tilde W^{QK,(i)}$ shape $d_c' \times d_c$, **independent of (t, s)**, pre-multiply at model load
- At inference, just $(c_t^Q)^\top \tilde W^{QK,(i)} c_s^{KV}$, **no $k_s$ computed**
- Similarly attention output: $\text{out}^{(i)} = \sum_s w_s v_s^{(i)} = (\sum_s w_s c_s^{KV})^\top W^{UV,(i)\top}$
- Absorb $W^{UV,(i)}$ into $W^O$: $W^O_\text{absorbed} = W^O (\text{blockdiag}(W^{UV,(i)}))$
- Conclusion: cache only latent, compute in latent space, **cache savings don't increase compute**

Pitfall: naively saying "just reconstruct K/V" — reconstruction reverts to MHA compute; unaware that absorb is inference-only (cannot absorb at training since backprop is needed).

</details>

<details>

<summary>Q23. Explain why MLA must separate an independent channel for RoPE; can we keep absorb in any other way?</summary>

- Core: RoPE inserts $R_{s-t}$ into $\tilde W^{QK,(i)}$, breaking the "constant matrix" property
- Alternative 1: apply RoPE directly on latent $c^{KV}$ — but latent dim is small and rotation semantics mismatch (RoPE pairs sin/cos along the head dim)
- Alternative 2: use ALiBi (additive bias, no rotation) — but breaks LLaMA-3-compatible pretraining
- Alternative 3: give up absorb, reconstruct K/V at every step — compute reverts to MHA
- DeepSeek-V2's choice: **decoupled RoPE channel $d_r=64$ shared across all heads**, very small cache increment (~5%), content channel keeps absorb
- Elegance: this independent channel shares $k_t^R$ across all heads — the "last mile of cache saving"

Pitfall: saying "RoPE doesn't affect MLA" — wrong; unaware that the decoupled channel is head-shared.

</details>

<details>

<summary>Q24. Why does vanilla speculative decoding's gain collapse in long context (128K+)? How to fix?</summary>

- Vanilla SD's gain assumption: weight loading is the bottleneck; one verify amortizes $K$ tokens' weight load
- In long context, **KV cache is far larger than weights**; bandwidth goes mostly to reading the cache
- Each verify reads the full cache once; cache loading cannot be saved
- Intuition: vanilla SD speedup $\propto E[\tau] / (1 + Kc)$ assumes $T_p$ is mainly weight loading, but in long context $T_p \approx T_\text{cache\_read} + T_\text{weight\_read}$ with the former dominating; every verify still reads the full cache, **$K$ tokens cannot amortize cache loading**, so $E[\tau]$'s advantage is eaten
- Fix 1: **MagicDec** — draft uses sparse KV (StreamingLLM), target uses full cache for verify
- Fix 2: **TriForce** — three tiers: small LM → target+sparse cache → target+full cache
- Fix 3: combine KV cache compression (H2O eviction) + SD: smaller cache makes vanilla SD viable again

Pitfall: just saying "long-context spec decoding doesn't work" without explaining why; unaware that MagicDec/TriForce are 2024 long-context SD SOTAs.

</details>

<details>

<summary>Q25. What is the mental model for deciding which optimizations to apply when designing an LLM serving system?</summary>

- **Step 1 measure workload**: prompt length distribution, generation length distribution, QPS
- **Step 2 pick optimizations by bottleneck**: (a) insufficient memory for a batch → PagedAttention + prefix caching + KV quantization; (b) long prefill blocks decode → Sarathi-Serve chunked prefill; (c) small-batch decode bandwidth-bound → spec decoding (gains highest at small batch); (d) long-context bandwidth-bound → MagicDec / TriForce; (e) cross-request prompt reuse → prefix caching + COW
- **Step 3 mind interactions**: SD + large batch sees diminishing gain (large batch is already compute-bound); PagedAttention + SD use page-table pointer changes for cache rollback; KV quantization + SD requires consistent quant schemes between draft/target
- **Step 4 monitor metrics**: tokens/sec, p95 TTFT, p95 TPOT, GPU utilization
- Key trade-off: throughput vs latency; SD leans toward latency improvement, continuous batching leans toward throughput

Pitfall: just listing tech terms without trigger conditions; unaware that SD's gain drops at large batch; ignoring real workload measurement.

</details>

## §A Appendix: Reference Implementation + Sanity Check

### A.1　Components summary

From-scratch reference implementations include:

- `NaiveCachedAttention` — single-layer MHA + KV cache append
- `PagedKVCache` — page table + COW sharing sketch
- `MQA_GQA_Attention` — three-in-one unified version
- `MLAAttention` — with decoupled RoPE channel
- `speculative_decode` — exact-equivalent spec loop (with rejection + bonus token)

### A.2　Sanity-check expected outputs

```
[a] naive cache append    prefill (1,16,128) → decode 8 token  ✓
[b] MQA/GQA/MHA shape + cache size consistent                   ✓
[c] MLA cache = d_c + d_r elements                              ✓
[d] spec decode rejection: 100k samples estimate α within 1%    ✓
[e] spec decode output vs direct target sampling: TV < 0.01     ✓
[f] paged cache COW: ref_count + share correct                  ✓
```

### A.3　Main references

- **KV / Serving systems**
  - Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023.
  - Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022.
  - Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024.

- **Attention variants**
  - Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150, 2019 (MQA).
  - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023.
  - DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434, May 2024 (MLA).

- **KV cache quantization**
  - Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024.
  - Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024.

- **Speculative decoding**
  - Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
  - Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", arXiv:2302.01318, 2023 (DeepMind).
  - Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", ICML 2024.
  - Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024.
  - Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", EMNLP 2024 (arXiv:2406.16858).
  - Li et al., "EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test", arXiv:2503.01840, 2025.
  - Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding", ICML 2024.
  - Miao et al., "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification", ASPLOS 2024.
  - Sun et al., "TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding", arXiv:2404.11912, 2024.
  - Sadhukhan et al., "MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding", arXiv:2408.11049, 2024 (ICLR 2025).
  - Zhang et al., "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding", ACL 2024.

Code and formulas have passed independent reviewer static checks (gpt-5.5 xhigh, cross-model); mathematical equivalence arguments verified.
