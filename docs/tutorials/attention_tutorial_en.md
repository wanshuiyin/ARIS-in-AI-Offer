## §0 TL;DR Cheat Sheet

> 💡 **Attention in 7 sentences** — one-page interview essentials (full derivations in §2-§9).

1. **Formula**: $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V$.

2. **Why divide by √d_k**: if $q_i, k_i \sim \mathcal{N}(0,1)$ are independent, then $\text{Var}(q\cdot k) = d_k$; dividing by $\sqrt{d_k}$ pulls variance back to 1 and avoids softmax saturation.

3. **Multi-Head**: split $D$ into $H$ heads, each doing independent attention in its own subspace, then concat and project with $W_o$. **For fixed $D$ with $d_k=D/H$, standard MHA parameter count is $\approx 4D^2$ (independent of $H$); MQA/GQA shrinks the K/V projections.**

4. **Self vs Cross**: in self-attention Q/K/V share the same source; in cross-attention Q comes from the query stream while K/V come from the context stream (encoder output / image tokens / text embedding).

5. **Causal mask vs Padding mask**: the former uses a lower triangle to block the future; the latter uses `[B,1,1,L_k]` to mask out padding columns.

6. **Complexity**: $O(B H L^2 d_k)$ time and $O(B H L^2)$ score memory — long sequences are bottlenecked by the quadratic term.

7. **Common footguns**: fully-masked row → softmax NaN; FP16 $QK^\top$ can overflow; attention weight ≠ causal explanation.

## §1 Attention Intuition

The essence of attention is **learned retrieval**:

- Each **query** ("what information do I need right now?")

- Computes similarity against all **keys** ("what does each position claim to offer?")

- Softmax-normalizes into a **weight distribution**

- Takes a weighted sum of all **values** ("what each position actually contributes")

Contrast with RNNs: an RNN **compresses past information into a fixed-size hidden state**, so long sequences inevitably lose information; attention **directly, globally, and dynamically** retrieves all past positions at every step, which is why it suits long-range dependencies.

"In self-attention, Q/K/V come from the same vector passed through three different projections" — proactively say this in interviews, since newcomers often mistakenly think Q/K/V are three separate inputs (note that in cross-attention Q does *not* share a source with K/V — see §4).

## §2 Scaled Dot-Product Attention

### 2.1　Formula

$$\boxed{\;\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V\;}$$

Shapes:

- $Q \in \mathbb{R}^{L_q \times d_k}$, $K \in \mathbb{R}^{L_k \times d_k}$, $V \in \mathbb{R}^{L_k \times d_v}$

- Scores $QK^\top \in \mathbb{R}^{L_q \times L_k}$ (similarity of each query to all keys)

- Softmax over the **key dimension**: weights per query row sum to 1

- Output $\in \mathbb{R}^{L_q \times d_v}$

### 2.2　Why divide by √d_k (mandatory: know the variance derivation)

Assume $q, k \in \mathbb{R}^{d_k}$ have i.i.d. components with $q_i, k_i \sim \mathcal{N}(0,1)$. Consider the dot product:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

By independence, each term $q_i k_i$ has mean $= \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ and variance $= \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$. So:

$$\mathbb{E}[q\cdot k] = 0, \quad \text{Var}[q\cdot k] = d_k$$

When $d_k$ is large (e.g., 64, 128), the typical magnitude of $q\cdot k$ is $\sqrt{d_k}$. After softmax, **the largest logit easily grabs nearly all the probability mass**, softmax enters its saturation regime, gradients shrink dramatically, and training slows or stalls. Dividing by $\sqrt{d_k}$ pulls variance back to 1, **alleviating saturation and improving gradient scale**.

> ⚠️ **Bonus interview point: FP16 overflow** — even after dividing by √d_k, the `QK^T` accumulation itself can overflow in FP16 (fp16 max ≈ 65504). Production implementations use fused SDPA / FlashAttention or **fp32 accumulation** to solve this. `torch.softmax` internally does log-sum-exp stabilization (subtract max logit before exp), but that happens inside the softmax step and cannot prevent matmul accumulation overflow.

### 2.3　Mask and the NaN pitfall (💣 classic bug, mandatory interview topic)

Standard practice: fill positions to be masked with $-\infty$ in the scores; after softmax their probability is 0.

But there's a pitfall: **if every key in a row is masked** (e.g., query 0 in cross-attn where context is all padding; causal + left padding; a query with no legal token after it), that row's scores are all $-\infty$, and softmax outputs:

$$\text{softmax}([-\infty, -\infty, ..., -\infty]) = \text{NaN}$$

because both numerator and denominator are $e^{-\infty} = 0$, $0/0 = $ NaN, which then contaminates the entire batch's gradient.

> ✅ **Fix: detect fully-masked rows → zero them after softmax** —

```python
# Detect fully-masked rows
all_masked = (~mask).all(dim=-1, keepdim=True)   # [..., L_q, 1]
# Temporarily unmask the row (to avoid NaN)
safe_mask = mask | all_masked
scores = scores.masked_fill(~safe_mask, float("-inf"))

# Softmax computes normally
weights = F.softmax(scores, dim=-1)

# Force the fully-masked row's output to 0 (otherwise it yields a uniform distribution)
weights = weights.masked_fill(all_masked, 0.0)
```

> ⚠️ **Mask semantics are inconsistent (proactively disambiguate)** — this implementation / `F.scaled_dot_product_attention`: **True = keep**

`nn.MultiheadAttention`'s `attn_mask` / `key_padding_mask`: **True = mask out** (opposite!)

Before writing code in an interview, ask the interviewer for the convention, or proactively state your convention, otherwise it's easy to get this backward.

### 2.4　Code (core 20 lines)

```python
def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, training=True):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)                 # [..., L_q, L_k]
    scores = scores / math.sqrt(d_k)                 # ← key scale

    if mask is not None:
        all_masked = (~mask).all(dim=-1, keepdim=True)
        safe_mask = mask | all_masked
        scores = scores.masked_fill(~safe_mask, float("-inf"))
    else:
        all_masked = None

    weights = F.softmax(scores, dim=-1)

    if all_masked is not None:
        weights = weights.masked_fill(all_masked, 0.0)   # NaN guard

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    return weights @ V, weights                       # output, weights
```

## §3 Multi-Head Attention

### 3.1　Formula

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_o$$

$$\text{head}_h = \text{Attention}(Q W_q^{(h)},\; K W_k^{(h)},\; V W_v^{(h)})$$

Each head has $W_q^{(h)}, W_k^{(h)}, W_v^{(h)} \in \mathbb{R}^{D \times d_k}$ with $d_k = D/H$. **In practice we concat the H per-head projection matrices into one $D \times D$ matrix** and run all heads' projections in a single matmul (GPU-friendly):

```

Input X [B, L, D]
   │
   │  W_q, W_k, W_v ∈ R^{D×D}   (each = concat of H matrices W^{(h)} ∈ R^{D×d_k})
   ↓
Q, K, V [B, L, D]
   │
   │  reshape [B, L, D] → [B, L, H, d_k] → transpose → [B, H, L, d_k]
   ↓
Scaled-Dot-Product Attention independently per head (batched matmul, parallel)
   ↓
heads [B, H, L_q, d_k]
   │
   │  transpose + reshape → [B, L_q, D]   (concat heads)
   ↓
W_o ∈ R^{D×D}    →    Output [B, L_q, D]
```

### 3.2　Why multi-head (would a single head work?)

- **Different subspaces**: each head learns one relational pattern in its own $d_k$-dim subspace (syntax, coreference, long-distance dependency, local n-gram, ...)

- **Expressiveness**: a single head learns only one attention pattern; H heads give H different weighted-sum outputs **in parallel** at inference

- **Parameter efficiency**: $d_k = D/H$ rather than $D$, so parameter count doesn't grow linearly with H

- Common interview question: are more heads always better? **No.** $d_k = D/H$ being too small (e.g., $d_k < 16$) limits each head's expressiveness; Mistral / LLaMA use head_dim ≈ 64-128 as the sweet spot

### 3.3　Parameter count and FLOPs

| Component | Shape | Parameter count |
| --- | --- | --- |
| $W_q$ | $D \times D$ | $D^2$ |
| $W_k$ | $D \times D$ | $D^2$ |
| $W_v$ | $D \times D$ | $D^2$ |
| $W_o$ | $D \times D$ | $D^2$ |
| **Total** |  | **$4D^2$** (independent of $H$) |

FLOPs (single self-attention forward, $L_q = L_k = L$):

- QKV projection: $3 \cdot 2 B L D^2 = 6 B L D^2$

- $QK^\top$: $2 B H L^2 d_k = 2 B L^2 D$

- Softmax weight × V: $2 B L^2 D$

- Output projection $W_o$: $2 B L D^2$

- **Total $\approx 8 B L D^2 + 4 B L^2 D$** — the first term is linear in $L$, the second quadratic (long-sequence bottleneck)

### 3.4　Code (core 30 lines)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads

        # Merge H per-head W^(h) into one [D, D] matrix
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = dropout

    def _split(self, x):  # [B, L, D] → [B, H, L, d_k]
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):  # [B, H, L, d_k] → [B, L, D]
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split(self.W_q(query))
        K = self._split(self.W_k(key))
        V = self._split(self.W_v(value))

        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)   # [1,1,L_q,L_k]
            elif mask.dim() == 3: mask = mask.unsqueeze(1)              # [B,1,L_q,L_k]
            # dim=4: already aligned

        out, w = scaled_dot_product_attention(Q, K, V, mask=mask, dropout_p=self.dropout_p, training=self.training)
        return self.W_o(self._merge(out)), w
```

### 3.5　Fused QKV + chunk (what production code actually writes)

§3.4 uses three separate `W_q / W_k / W_v`, which reads closest to the formula. For **self-attention**, the more common production form **fuses the three projections into one Linear** and then slices the output — nanoGPT, HF GPT-2 and timm's ViT all do it this way (counterexamples exist: HF LLaMA keeps separate `q_proj / k_proj / v_proj`, and cross-attention cannot fuse at all, since Q and K/V come from different inputs). The three slicings below are **element-wise identical** (`torch.equal` returns True) and differ only in readability; the GEMM saving is fused-versus-§3.4's-three-Linears.

```python
D, H, d_k = 64, 4, 16                       # D = H * d_k
qkv_proj = nn.Linear(D, 3 * D, bias=False)  # one Linear emits Q/K/V
qkv = qkv_proj(x)                           # [B, N, 3D]

# Form 1: chunk — the most readable, hardest to get wrong on a whiteboard
q, k, v = qkv.chunk(3, dim=-1)                          # each [B, N, D]
q, k, v = [t.view(B, N, H, d_k).transpose(1, 2)         # → [B, H, N, d_k]
           for t in (q, k, v)]

# Form 2: nanoGPT's split — equivalent to chunk here (3D/3 = D); same view+transpose follows
q, k, v = qkv.split(D, dim=2)
q, k, v = [t.view(B, N, H, d_k).transpose(1, 2) for t in (q, k, v)]

# Form 3: 5-D reshape + permute — timm ViT / this repo's code/mha.py
q, k, v = qkv.reshape(B, N, 3, H, d_k).permute(2, 0, 3, 1, 4)
```

> 💡 **Why fuse** — compared with §3.4's three Linears, Q/K/V cost **one GEMM instead of three**. The FLOPs are identical; what you save is kernel launches and activation traffic, and one large matmul usually utilises the hardware better (how much depends on shapes and backend). The other reason is weight layout: HF GPT-2's `c_attn` (a `Conv1D` with weight `[D, 3D]`) concatenates Q/K/V along the output dimension in `[Q | K | V]` order, so loading pretrained weights requires matching that order (and a transpose to `[3D, D]` if you port it to `nn.Linear`).

| Form | How it slices | Readability | Used by |
| --- | --- | --- | --- |
| Three separate Linears | no slicing | closest to the math | teaching code, §3.4 |
| fused + `chunk(3, dim=-1)` | three equal parts | **best** | most hand-written implementations |
| fused + `split(D, dim=2)` | parts of width D | good | nanoGPT |
| fused + `reshape` + 5-D `permute` | one shot | poor — easy to get the axis order wrong | timm ViT, `code/mha.py` |

> ⚠️ **`chunk(3)` becomes wrong the moment you use GQA (a common follow-up)** — under MQA/GQA the fused projection is no longer three equal parts but `D + 2·G·d_k` (see §6.2). Take `D=64, H=4, d_k=16, G=2`: the fused width is `64 + 32 + 32 = 128`, and `chunk(3)` slices it into `[43, 43, 42]`. **It does not raise.** Q gets 43 instead of 64, and the failure only surfaces later at the `view`, as `shape '[2, 7, 4, 16]' is invalid for input of size 602` — a message with no connection to the actual cause.

`torch.chunk` does not raise on a non-divisible width; it just returns uneven parts (width 5 into 3 → `[2, 2, 1]`). So the rule is simple: **whenever Q and K/V have different head counts, use `split` with explicit widths, never `chunk`.**

```python
q, k, v = qkv.split([D, G * d_k, G * d_k], dim=-1)      # explicit widths — the standard form under MQA/GQA
```

## §4 Self / Cross / Causal / Padding

### 4.1　Self vs Cross Attention (mandatory)

|  | Self-Attention | Cross-Attention |
| --- | --- | --- |
| **Q source** | $X$ | $X_\text{decoder}$ / latent / learnable queries |
| **K, V source** | $X$ (same) | $X_\text{encoder}$ / context / memory |
| **$L_q$ vs $L_k$** | equal | can differ |
| **Typical mask** | causal (decoder) or padding (encoder) | K/V-side padding mask (no causal) |
| **Purpose** | intra-position correlation | retrieve relevant info from external memory |
| **Examples** | every BERT layer; every GPT layer; ViT | Transformer Decoder's second sub-layer; DETR; Perceiver; Stable Diffusion (image Q × text K/V) |

### 4.2　Causal Mask (Decoder / GPT)

Lower-triangular matrix (including diagonal): row $i$ may attend to keys $j \le i$.

```python
def causal_mask(L, device=None):
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
# L=4 →
# [[T F F F]
#  [T T F F]
#  [T T T F]
#  [T T T T]]
```

### 4.3　Padding Mask (variable-length sequences)

Each sample has a different valid length; padding tokens must not be attended:

```python
def padding_mask(lengths, max_len=None):
    if max_len is None: max_len = int(lengths.max())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(len(lengths), -1)
    return idx < lengths.unsqueeze(1)    # [B, L]    True=valid, False=padding

# Usage: must unsqueeze to [B, 1, 1, L_k] so it broadcasts to [B, H, L_q, L_k] inside MHA
pmask = padding_mask(lengths).unsqueeze(1).unsqueeze(1)   # [B, 1, 1, L_k]
out, _ = mha(x, x, x, mask=pmask)
```

> 💡 **Causal + padding together** — AND the two masks: `combined = causal_mask & padding_mask_4d`. Mind the broadcast dims: causal is `[L,L]`, padding is `[B,1,1,L_k]`, and the AND yields `[B,1,L,L]`.

## §5 Complexity Analysis

|  | Time | Memory | Bottleneck |
| --- | --- | --- | --- |
| RNN | $O(L \cdot D^2)$ | $O(D)$ | sequential, not parallelizable |
| Self-Attention | $O(L^2 \cdot D)$ | $O(L^2 + L \cdot D)$ | $L^2$ score matrix (long sequences) |
| Conv (kernel $k$) | $O(L \cdot k \cdot D^2)$ | $O(D)$ | limited receptive field |

Key points:

- Self-attention's $L^2$ **compute** is acceptable (GPU parallel), but **$L^2$ memory** (score matrix) is the real bottleneck — this is the pain point Flash Attention attacks

- At LLM inference, the prefill stage is $O(L^2)$; the decode stage with KV cache is $O(L)$ per step (see §6)

- When $L \approx D$, attention and FFN take similar time; when $L \gg D$, attention dominates

## §6 KV Cache + MQA / GQA

### 6.1　KV Cache (key optimization for autoregressive inference)

Problem: when GPT generates autoregressively, every new token re-runs the entire prefix through the forward pass — a single step at position $t$ costs $O(t^2)$, and summing over $T$ steps gives $O(T^3)$ cumulative redundant computation (not $O(T^2)$).

Solution: cache each layer's $K^{(\ell)}, V^{(\ell)}$. When generating the $(t+1)$-th token:

- Compute only the new token's $q_{t+1}, k_{t+1}, v_{t+1}$ (size $1 \times D$)

- Append $k_{t+1}, v_{t+1}$ to the cache

- The new $q$ attends over the full cache ($O(t)$, not $O(t^2)$)

> ⚠️ **Footgun** — KV cache is an **inference optimization**; standard teacher-forced parallel training (SFT / pretraining forward-backward) has no use for it, since every position is processed in parallel and there is no "token-by-token generation". But RLHF / GRPO-style training with on-policy rollout sampling *does* perform genuinely autoregressive generation during the rollout phase, so KV cache speeds that up too.

**KV cache memory (per sample)**:

$$\text{KV cache} = L_\text{ctx} \cdot n_\text{layers} \cdot \underbrace{2}_{K, V} \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes\_per\_elem}$$

Note: under MQA/GQA $H_\text{kv} \ll H$, shrinking the cache dramatically. For LLaMA-2-70B (GQA, $H_\text{kv}=8$), $L_\text{ctx}=4096$, 80 layers, fp16: about **1.25 GiB** (≈1.34 GB) **/ sample** — this is why LLaMA-2 uses GQA instead of MHA (vanilla MHA would reach about **10 GiB ≈ 10.74 GB / sample**).

### 6.2　MQA / GQA (attacking KV-cache memory)

| Variant | Q heads | K/V heads | KV cache reduction | Used in |
| --- | --- | --- | --- | --- |
| **MHA** (vanilla) | $H$ | $H$ | 1× | original Transformer |
| **MQA** (Multi-Query) | $H$ | **1** | $H \times$ | PaLM, Falcon |
| **GQA** (Grouped-Query) | $H$ | $G$ ($1 < G < H$) | $H/G \times$ | LLaMA-2-70B (7B/13B are still MHA), all of LLaMA-3, Mistral |

Core idea: **multiple Q heads share one set of K/V**. MQA is extreme but slightly hurts quality; GQA is a compromise (e.g., H=32, G=8) that cuts memory/bandwidth by 4× with essentially no quality loss.

> ❌ **Footgun** — MQA/GQA reduce **KV-cache memory + memory bandwidth**, **not** Q-projection compute (Q head count is unchanged). Interviewers love to push back with "what exactly did it reduce?".

## §7 FlashAttention Core Tricks

Problem: standard attention has to materialize the $L \times L$ score matrix, and HBM read/write IO is the bottleneck (not FLOPs).

FlashAttention idea (**IO-aware exact attention**, not an approximation):

1. **Block tiling**: split $Q, K, V$ into blocks and load only one $Q$ block plus one $K, V$ block into SRAM at a time

2. **Online softmax**: maintain a running max $m$ and running sum $\ell$ incrementally, avoiding ever materializing the full score matrix

3. **Recompute on backward**: recompute attention during the backward pass instead of storing $L^2$ scores

Effects:

- **Avoids materializing** the full $L \times L$ scores / probs matrix in HBM

- The paper's HBM IO complexity is about $O(L^2 d^2 / M + Ld)$, vs $O(L^2 + Ld)$ HBM traffic for standard attention — when $L$ is large and $M$ (SRAM) is appropriate, IO drops sharply

- Peak memory drops from $O(L^2)$ to $O(L)$ (no stored intermediate scores)

- Typical speedup 2-4× (depends on sequence length & GPU architecture)

- **Mathematically equivalent** (exact attention, not sparse / linear approximation)

> 💡 **FlashAttention v1/v2/v3 key differences** — v1 (2022): online softmax + block tiling + recompute. v2 (2023): swap the inner/outer loops (Q-outer, KV-inner) + better warp-level parallelism + fewer non-matmul FLOPs. v3 (2024): targets H100 Hopper with WGMMA / TMA / FP8 + asynchronous pipeline. Interviews usually focus on v1/v2 and online-softmax details.

## §8 Position Encoding (RoPE / ALiBi / Absolute)

| Method | Principle | Extrapolation | Used in |
| --- | --- | --- | --- |
| **Sinusoidal absolute** | Fixed sin/cos position vector added to the input embedding | Position encoding itself can be defined for any length, but the model may not generalize past trained lengths | Original Transformer (Vaswani 2017) |
| **Learned absolute** | Treat position as a token and learn an embedding table | Poor (the table is fixed-size, a hard limit) | BERT, GPT-2 |
| **RoPE** (Rotary) | Apply position-dependent rotation to $Q, K$: $q_m \to q_m e^{im\theta}$ (complex-number view) — **the position-dependent term enters via relative shift $m-n$ in the inner product** (content vectors still influence scores) | Medium (naturally captures relative position; out-of-length needs NTK-aware / YaRN) | LLaMA-1/2/3, Mistral, Qwen |
| **ALiBi** | Add a positional-distance bias to scores: $\text{score}_{ij} - m \cdot \lvert i-j \rvert$ | Good (linear bias extrapolates naturally) | BLOOM, MPT |

### 8.1　Attention Sink (advanced topic)

In trained LLMs, attention at decode time concentrates abnormally on the first 1-4 tokens (especially [BOS] / the first token), even when those tokens are content-irrelevant. This phenomenon is called **attention sink**. **A common intuitive explanation**: softmax forces weights to sum to 1, so when a query doesn't really want to attend to anything, it needs a "junk slot" to absorb probability mass; and because early tokens are visible to all subsequent tokens, training naturally produces a global sink. StreamingLLM (Xiao et al., ICLR 2024) exploits this for long-sequence inference (keep the attention sink + a sliding window).

## §9 Attention in Diffusion (mandatory if you mention generative work)

For candidates with a diffusion background, interviewers almost always ask about attention in generative models.

### 9.1　Cross-Attention in Latent Diffusion (Stable Diffusion)

```

Image latent (z_t)  [B, C, H, W]
   │
   │  flatten to tokens [B, HW, D]
   ↓
Self-Attention (Q=K=V from image)
   ↓
Cross-Attention:
    Q = image tokens [B, HW, D]
    K, V = text embedding [B, L_text, D]    ← text conditioning
   ↓
FFN → next layer
```

Key points:

- Text-to-image conditioning is realized via cross-attention: image tokens are queries, text embeddings are keys/values

- Classifier-Free Guidance (CFG): two forwards (with text / without text), then take the difference. For $\epsilon$-pred: $\epsilon_\text{CFG} = \epsilon_\text{uncond} + s (\epsilon_\text{cond} - \epsilon_\text{uncond})$; for v-pred / x0-pred swap in the corresponding prediction — the linear guidance form is analogous

- SD / SDXL U-Nets alternate self-attn and cross-attn inside Transformer blocks at multiple spatial resolutions

- DiT (Diffusion Transformer) replaces the U-Net with a pure Transformer; conditioning enters via AdaLN / cross-attn / token-concat

### 9.2　Attention in video diffusion

- **Spatial attention**: within each frame (between image patches)

- **Temporal attention**: across frames (between the same position at different time steps)

- **Spatiotemporal / full attention**: all frames × all positions — most expensive, infeasible for long video

- Long-video attention is an open problem ($L \sim 10^4$-$10^5$ tokens); common routes: factorization (spatial + temporal alternated), sparse window, hierarchical pooling, chunked attention

## §10 25 Frequently-Asked Interview Questions

Compiled from the perspective of a top-lab interviewer by codex (gpt-5.5 xhigh), in 3 difficulty tiers. Click each question to see the key answer points + common pitfalls.

### L1 must-know (any ML engineering role will ask)

<details>

<summary>Q1. What is the attention formula?</summary>

- $\text{softmax}(QK^\top / \sqrt{d_k}) V$

- Softmax over the keys dimension

- Output is a weighted sum of values

Writing the softmax dim on the query axis.

</details>

<details>

<summary>Q2. Why divide by √d_k?</summary>

- If $q_i, k_i$ are independent zero-mean unit-variance

- The dot-product variance is about $d_k$

- After scaling, variance returns to 1, avoiding softmax saturation

Just saying "to prevent values from being too large" without giving the variance derivation.

</details>

<details>

<summary>Q3. What do Q/K/V represent?</summary>

- Q is the retrieval query

- K is the matching index

- V is the content to aggregate

Saying Q/K/V are three different inputs; in self-attn they share a source but use different projections.

</details>

<details>

<summary>Q4. Why is multi-head useful?</summary>

- Different subspaces model different relations

- Multiple positional / semantic patterns in parallel

- Concat then fuse

Saying "more heads is always better". In reality if $d_k$ is too small, expressiveness suffers.

</details>

<details>

<summary>Q5. How does MHA's parameter count scale with the number of heads?</summary>

- Fixed $D$ with $d_k = D/H$ (standard MHA)

- $W_q + W_k + W_v + W_o$ sums to $4D^2$, **independent of $H$**

- But under MQA/GQA, the K/V projection matrix shrinks ($H_\text{kv} < H$ heads)

- This is why "head count is free" holds for standard MHA but pays off in memory under MQA/GQA

Thinking parameter count grows linearly with H; or forgetting that MQA/GQA changes the K/V projection dimensions.

</details>

<details>

<summary>Q6. Self-attention vs cross-attention?</summary>

- Self: Q/K/V share a source

- Cross: Q comes from the target, K/V from the context

- Cross is common in encoder-decoder, diffusion text conditioning

Saying "cross has two inputs" without explaining the Q vs KV sourcing.

</details>

<details>

<summary>Q7. How do you write a causal mask?</summary>

- `torch.tril(torch.ones(L, L, dtype=torch.bool))`

- Be explicit whether True=keep or True=mask (APIs differ)

- Broadcast to `[B, H, L, L]` or rely on framework's implicit broadcasting

Flipping the upper/lower triangle; forgetting to align broadcast dimensions.

</details>

<details>

<summary>Q8. Which axis does the padding mask mask?</summary>

- Usually masks key/value columns (so padding-position probability is 0)

- Shape can be `[B, 1, 1, L_k]` to align with head and query dims

- Note: masking key columns is **not enough** to zero out padded-query outputs; padded query rows are usually handled separately via loss ignore / output zeroing / packed sequences

Thinking the padding mask handles everything — it only prevents "seeing padding", but padded queries' own outputs still need external handling.

</details>

<details>

<summary>Q9. Attention complexity?</summary>

- Time $O(B H L_q L_k d_k) = O(B L^2 D)$

- Score memory $O(B H L_q L_k)$

- Long-sequence bottleneck is the quadratic term

Just saying $O(n^2)$ and dropping the head and hidden dims.

</details>

<details>

<summary>Q10. Where does attention dropout go?</summary>

- After softmax weights, before the matmul with V

- Enabled only in training, disabled at eval

- After dropout, row sums aren't necessarily 1 (only 1 in expectation)

Demanding row-sum = 1 after dropout as a sanity check (it's wrong).

</details>

### L2 intermediate (research-oriented roles)

<details>

<summary>Q11. Derive the softmax Jacobian by hand.</summary>

- $y_i = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$

- $\dfrac{\partial y_i}{\partial x_j} = y_i (\delta_{ij} - y_j)$

- Matrix form: $J = \text{diag}(y) - yy^\top$

Writing only diagonal entries and dropping the cross terms $-y_i y_j$.

</details>

<details>

<summary>Q12. What's the pitfall of masking with -∞?</summary>

- Normal case: masked positions get softmax probability 0 ✓

- **Fully-masked row → softmax outputs NaN** ($0/0$)

- Fix path: first avoid all-`-inf` rows (temporarily unmask), zero out that row's weights and output after softmax, and make sure that query doesn't enter the loss / residual accumulation

- Fused kernels / APIs have constraints on sentinel values; under fp16, use a dtype-safe large negative (e.g., `finfo(dtype).min`) for stability

Thinking -inf is always safe; or zeroing only after softmax without preventing NaN.

</details>

<details>

<summary>Q13. What is the log-sum-exp trick?</summary>

- Subtract max(logits) before softmax — equivalent and preserves probabilities

- Prevents $e^{x_i}$ overflow (fp32 max ≈ 3.4e38, but $e^{100}$ already overflows)

- $\log \sum_j e^{x_j} = m + \log \sum_j e^{x_j - m}$ with $m = \max_j x_j$

Forgetting that $QK^\top$ overflow can happen before softmax (during the matmul accumulation).

</details>

<details>

<summary>Q14. PyTorch nn.MultiheadAttention's in_proj_weight ordering?</summary>

- Shape `[3D, D]`

- Order: **Q, K, V** (cat dim=0)

- Linear weight is `[out, in]`, so `cat([W_q.weight, W_k.weight, W_v.weight], dim=0)`

Concatenating as K/Q/V or transposing the weight.

</details>

<details>

<summary>Q15. attn_mask vs key_padding_mask?</summary>

- `attn_mask` controls at the query-key pair level (typically causal)

- `key_padding_mask` controls overall visibility of a key token (typically padding)

- Bool semantics: `nn.MultiheadAttention` uses **True = mask out**; `F.scaled_dot_product_attention`'s bool mask uses **True = keep** (opposite!)

- When using both: under mask-out semantics, combine via **OR** (either True means blocked); under keep semantics, combine via AND (only both True means kept)

Applying True/False without checking API docs; or flipping AND/OR.

</details>

<details>

<summary>Q16. In cross-attention can L_q and L_k differ?</summary>

- Yes — this is the standard cross-attention case

- Scores shape is $[L_q, L_k]$

- The mask must align with the key dimension

Assuming cross-attn requires equal lengths.

</details>

<details>

<summary>Q17. Why do we need the output projection W_o?</summary>

- Fuses outputs from different heads

- Maps back to $d_\text{model}$ for the residual add

- Lets the model learn combinations across heads (not just simple concat)

Thinking the work ends after concat.

</details>

<details>

<summary>Q18. Pre-norm vs post-norm impact on the attention block?</summary>

- Pre-norm: `x + Attn(LN(x))`, more stable for deep training, gradient along the residual path is relatively preserved

- Post-norm: `LN(x + Attn(x))`, used in the original Vaswani paper, needs warmup / careful init at extreme depths

- **Most decoder-only LLMs use pre-norm (often with RMSNorm variants)**, but specific architectures have exceptions

Treating norm position as a pure engineering detail; or asserting too absolutely that "all modern LLMs use pre-norm".

</details>

<details>

<summary>Q19. Are attention weights equivalent to "model explanations"?</summary>

- Visualization has reference value (where attention focuses)

- But **not equivalent to causal explanation**

- The value path and subsequent layers change the actual contribution

- Jain & Wallace "Attention is not Explanation" (2019)

Treating high attention weight as "the model's reason" outright.

</details>

<details>

<summary>Q20. What to watch in mixed-precision attention?</summary>

- **fp32 accumulation**: matmul accumulation / critical softmax steps in fp32, then cast back to low precision

- **Softmax max-subtraction** (log-sum-exp) to prevent exp overflow — PyTorch's `F.softmax` does this internally

- **Mask sentinel**: under fp16 use `torch.finfo(dtype).min` instead of literal -inf

- **BF16 vs FP16**: BF16's dynamic range is close to fp32, more suitable for attention; fp16 has narrow range and QK^T overflows easily

- **Fused kernels** (FlashAttention, `F.scaled_dot_product_attention`) include kernel-level stabilization and are safer than hand-written naive code

Writing naive attention by hand under FP16 without fp32 accumulation.

</details>

### L3 advanced variants (top labs / diffusion direction)

<details>

<summary>Q21. How does KV cache optimize autoregressive decoding?</summary>

- At decode step $t+1$, compute $Q$ only for the new token (1×D)

- Reuse historical $K, V$ (already in the cache) and append the new $k_{t+1}, v_{t+1}$

- Per-step attention goes from $O(t^2)$ to $O(t)$; whole-sequence generation from $O(L^3)$ to $O(L^2)$

- Per-sample memory: $L_\text{ctx} \cdot n_\text{layers} \cdot 2 \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$ (under MQA/GQA, $H_\text{kv} \ll H$)

Saying KV cache reduces training cost — wrong. It only applies to autoregressive inference. Also: cache scales with KV head count, not Q head count.

</details>

<details>

<summary>Q22. What do MQA and GQA solve?</summary>

- MQA: multiple Q heads share one set of K/V (K/V has only 1 head)

- GQA: compromise with $G$ K/V groups ($1 < G < H$)

- Main benefit: **decode-time KV-cache memory + memory bandwidth** (large reduction)

- It also reduces K/V projection params and compute (smaller K/V projection matrices), **but does not reduce Q / O projection**

- Quality impact: usually **GQA's quality loss is smaller than MQA's**, depending on model scale and training (LLaMA-2 70B / LLaMA-3 / Mistral / Qwen-2 all use GQA)

Thinking it reduces Q projection; or saying "GQA causes essentially no quality loss" too absolutely.

</details>

<details>

<summary>Q23. Core tricks of FlashAttention?</summary>

- **Block tiling**: split $Q, K, V$ into SRAM-sized blocks, load in batches

- **Online softmax**: incrementally maintain running max $m$ and running sum $\ell$, **avoiding materialization** of the full $L \times L$ scores / probs matrix in HBM

- **Recompute on backward**: recompute scores during backward using saved $m, \ell$, no intermediates stored

- Key: **IO-aware exact attention** (mathematically equivalent, not an approximation)

- HBM IO complexity about $O(L^2 d^2 / M + Ld)$ vs $O(L^2 + Ld)$ HBM traffic for standard attention — under long sequences this is a large IO (not FLOPs) reduction

Saying it's approximate attention (like Performer / Linformer) — wrong, FlashAttn is exact; or conflating IO complexity with FLOPs complexity.

</details>

<details>

<summary>Q24. RoPE vs ALiBi vs absolute position? What is attention sink?</summary>

- **Absolute**: position vectors added to the input embedding (Vaswani sinusoidal / GPT-2 learned)

- **RoPE**: apply position-dependent rotation to $Q, K$, preserving **relative position** info (the position term enters $q_m^\top k_n$ only through $m-n$, but the content vectors still determine the actual value — it's not position-only)

- **ALiBi**: add a distance bias $-m |i-j|$ to scores, extrapolates naturally

- **Attention sink**: trained LLMs assign abnormally high attention to the first 1-4 tokens (especially [BOS]) even when content-irrelevant — softmax forces sum to 1 so the model needs a "junk slot". StreamingLLM exploits this for long-sequence inference.

Treating attention sink as normal padding / CLS token behavior.

</details>

<details>

<summary>Q25. How is attention used in diffusion / latent diffusion?</summary>

- **U-Net latent tokens as Q**, text embedding as K/V, doing **cross-attention** to inject text conditioning

- Self-attention within each spatial resolution (image patches × image patches)

- **CFG (Classifier-Free Guidance)**: two forwards, take difference to amplify the conditional signal

- DiT (Diffusion Transformer): replace U-Net with pure Transformer; conditioning via AdaLN / cross-attn / token-concat

- Video diffusion: combinations of spatial / temporal / spatiotemporal attn (long video is open, $L \sim 10^5$)

Saying diffusion relies only on convolutions; or that attention exists only in DiT (wrong — U-Net has plenty too).

</details>

## §A Appendix: Full from-scratch code skeleton

The reference from-scratch implementation (`code/mha.py`) contains:

- `MultiHeadAttention` — standard multi-head self-attention (a single fused `qkv` Linear producing Q/K/V at once, `[N,N]` additive mask; no NaN guard for fully-masked rows, no cross-attention wrapper); `__init__` checks `assert embed_dim % num_heads == 0`, raising `AssertionError` on mismatch (note: `assert` is stripped entirely under `python -O` / `PYTHONOPTIMIZE`, so production code should use an explicit `raise ValueError(...)` instead)

- `make_causal_mask()` — builds an upper-triangular `-inf` causal mask

- 3 sanity checks: `sanity_check()` (numerical parity against `nn.MultiheadAttention`, diff < 1e-5), `shape_check()`, `causal_check()`

Actual sanity-check output (PyTorch 2.x, runs on CPU in seconds):

```
== Multi-Head Attention sanity ==
[sanity] max diff vs nn.MultiheadAttention = 5.96e-08
[sanity] PASS
[shape] input  x: (2, 1024, 768)
[shape] output y: (2, 1024, 768)
[causal] applied causal mask, output shape (1, 8, 32)

All checks passed.
```

Code passed independent reviewer static check + PyTorch sanity-check run, diff vs `nn.MultiheadAttention` < 1e-5.

---

## 📜 Runnable Code

Minimal runnable PyTorch implementations of this tutorial's core concepts live in [`docs/tutorials/code/`](code/):

- [`mha.py`](code/mha.py) — standard Multi-Head Self-Attention + causal mask + numerical parity check against `nn.MultiheadAttention`
- [`axial_attention.py`](code/axial_attention.py) — H/W factorized axial attention + complexity comparison table + receptive-field isolation test

Each script runs on CPU in seconds with built-in `assert` sanity checks. Full overview in [`code/README.md`](code/README.md).
