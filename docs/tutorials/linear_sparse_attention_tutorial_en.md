## §0 TL;DR Cheat Sheet

> 💡 **9 sentences to nail efficient attention / SSM / sparse attention** — one page covering the interview essentials (see §1–§11 for derivations).

1. **The quadratic bottleneck**: softmax attention trains in $O(L^2 d)$ time, and at decode the KV cache grows linearly with context ($O(L)$ memory, $O(L)$ compute per step). There are three escape routes: **linear attention**, **SSM / Mamba**, **sparse attention**.

2. **Linear attention**: use a kernel feature map $\phi$ to replace softmax with $\phi(q)^\top\phi(k)$, then lean on the **associativity of matrix multiplication** to compute $\phi(Q)\big(\phi(K)^\top V\big)$, dropping $O(L^2)$ to $O(L)$. It is equivalent to an RNN **whose hidden state is a matrix** $S_t\in\mathbb{R}^{d_k\times d_v}$: $S_t = S_{t-1} + \phi(k_t) v_t^\top$.

3. **SSM / Mamba**: the continuous state space $h_t = A h_{t-1} + B x_t,\; y_t = C h_t$ becomes a recurrence after discretization ($\Delta$ + zero-order hold). **Selective SSM (S6)** makes $B, C, \Delta$ vary with the input (data-dependent gating), breaking LTI in exchange for content selection; paired with a hardware-aware selective scan.

4. **Mamba-2 / SSD**: State Space Duality — a selective SSM is **equivalent to a kind of structured masked linear attention** (via a 1-semiseparable matrix). This lets SSMs train efficiently on tensor cores with a matmul-dense **chunkwise** algorithm.

5. **Delta rule (DeltaNet)**: the update is a **rewrite** rather than a pure accumulation — $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, where the $(I-\beta k k^\top)$ term first **erases the old association** before writing the new value (online least squares / error correction). Gated DeltaNet then adds a scalar/diagonal decay gate $\alpha_t$ for forgetting.

6. **Chunkwise parallel**: split the sequence into chunks, run quadratic parallel attention **within a chunk** and pass an $O(1)$ recurrent state **across chunks**. This is the **core bridge** that lets linear attention / SSM / DeltaNet both infer in $O(L)$ and train in parallel with matmuls.

7. **Sparse attention (trainable)**: keep softmax but attend only to a subset. **NSA** (DeepSeek, natively trainable + hardware-aligned, three branches: compress + select + sliding window); **MoBA** (Moonshot, MoE-style routing of a query to its top-k key blocks); **Lightning Attention** (MiniMax-01, IO-aware linear); **DeepSeek-V3.2 DSA** (a lightweight lightning indexer scores and selects tokens).

8. **The inference trump card**: linear / SSM keep a **fixed-size** recurrent state ($O(1)$ in sequence length) → constant memory + $O(1)$ per-token decode; softmax's KV cache grows linearly with length; sparse reduces KV access/compute but still stores KV.

9. **Hybrid architectures**: pure linear/SSM have **weak recall (associative recall / in-context copy)** — a recall–throughput trade-off. The fix is to **interleave a few attention layers** among many linear/SSM layers (full / sliding / gated attention, the exact type and ratio differ by model): Jamba, Hymba, Qwen3-Next, Kimi-Linear, MiniMax-01 all take this route, balancing throughput and recall.

## §1 Intuition: softmax attention's quadratic bottleneck and three escape routes

Recall standard attention: $\text{Attn}(Q,K,V) = \text{softmax}\!\big(QK^\top/\sqrt{d}\big)V$, $Q,K,V\in\mathbb{R}^{L\times d}$. Two costs:

- **Training / prefill**: you must compute the $L\times L$ score matrix, costing $O(L^2 d)$ time and $O(L^2)$ memory (FlashAttention squeezes memory to $O(L)$, but **time is still $O(L^2 d)$** — the FLOPs don't change).
- **Autoregressive decode**: with a KV cache each step is $O(L)$, but **the cache itself grows linearly with context**: at step $t$ you store $t$ key/value pairs. In long context the KV cache's memory and access bandwidth become the real bottleneck.

The root of the quadratic cost is that softmax does a **nonlinear normalization** of query–key similarity; $\text{softmax}(QK^\top)$ cannot be factored into "compress K/V first, then interact with Q," so the $L\times L$ matrix must be materialized explicitly. The three escape routes all essentially ask "can we avoid materializing this $L\times L$ matrix":

> 💡 **A mental model for the three escape routes**
> - **(a) Linear attention**: drop softmax and approximate it with a kernel feature map $\phi$, so $\text{Attn}\approx \phi(Q)\big(\phi(K)^\top V\big)$ — leaning on **associativity** to first compress K and V into a small $d_k\times d_v$ matrix, then interact with Q. Time $O(L d^2)$, decode state $O(d_k d_v)$ **independent of $L$**. The cost: you lose softmax normalization, and recall weakens.
> - **(b) SSM / Mamba**: model the sequence as the recurrence of a **linear dynamical system** $h_t = A h_{t-1}+B x_t$ — naturally $O(L)$ with a fixed state; selective SSM makes the transition parameters data-dependent, restoring content selection.
> - **(c) Sparse attention**: keep softmax, but each query attends only to a **subset** (block / window / top-k), cutting $L^2$ to $L\cdot s$ ($s\ll L$). It can be a fixed pattern, or a **trainable** content-dependent selection.

(a) and (b) are mathematically very closely related (§4's SSD will pierce this veil: linear attention's recurrence ≈ a kind of SSM); (c) is a different family — it **does not give up softmax**, only gives up "full connectivity." Below, §2–§6 follow the (a)(b) "sub-quadratic recurrence" main line, §7 covers (c) sparse, and §8–§11 cover inference gains, hybrid architectures, engineering, and complexity.

## §2 Linear attention

### 2.1　From softmax to a kernel feature map

Write out a single query's output (ignoring $\sqrt{d}$, with $\text{sim}(q,k)$ denoting a similarity kernel):

$$o_i = \frac{\sum_{j} \text{sim}(q_i, k_j)\, v_j}{\sum_{j}\text{sim}(q_i, k_j)}, \qquad \text{softmax is } \text{sim}(q,k)=\exp(q^\top k).$$

**The key step of linear attention**: assume the similarity can be written as the inner product of a feature map $\text{sim}(q,k) = \phi(q)^\top \phi(k)$, where $\phi:\mathbb{R}^{d}\to\mathbb{R}^{d_k}$ (the feature map, $d_k$ is the feature dimension). Substituting:

$$o_i = \frac{\sum_j \big(\phi(q_i)^\top\phi(k_j)\big) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top\big(\sum_j \phi(k_j) v_j^\top\big)}{\phi(q_i)^\top\big(\sum_j \phi(k_j)\big)}.$$

The key is in the numerator: $\sum_j \phi(k_j)v_j^\top$ is a $\mathbf{d_k\times d_v}$ matrix, **independent of the query** — you can compute it once over all $j$ and reuse it for every query. This is using **associativity** to reorder $\big(\phi(Q)\phi(K)^\top\big)V$ into $\phi(Q)\big(\phi(K)^\top V\big)$:

$$\underbrace{\big(\phi(Q)\phi(K)^\top\big)V}_{O(L^2 d_v)\ :\ \text{compute the }L\times L\text{ first}} \;=\; \underbrace{\phi(Q)\big(\phi(K)^\top V\big)}_{O(L d_k d_v)\ :\ \text{compute the }d_k\times d_v\text{ first}}.$$

> ✅ **Associativity is where all the magic lives**
> When $L \gg d$, $O(Ld_kd_v)\ll O(L^2 d)$. softmax cannot be factored this way, because $\exp(q^\top k)$ is not a finite-dimensional inner product $\phi(q)^\top\phi(k)$ (it takes infinite-dimensional RFF to approximate it, see Performer).

Common choices of $\phi$:

- **$\phi(x)=\text{elu}(x)+1$** (Katharopoulos 2020, "Transformers are RNNs"): guarantees non-negativity, simple.
- **Performer (FAVOR+)**: uses random features $\phi(x)=\exp(\omega^\top x - \|x\|^2/2)/\sqrt{m}$ and other **positive random features** to unbiasedly estimate $\exp(q^\top k)$, approximating the true softmax in accuracy (at the cost of needing a large feature dimension $m$).
- **RetNet / GLA**: essentially $\phi=\text{id}$ (or with normalization), putting the emphasis on **decay / gating** rather than the feature map (see 2.4).

### 2.2　Recurrent form: linear attention = an RNN whose hidden state is a matrix

Make the above **causal** ($o_i$ only sees $j\le i$) and define a cumulative state. In the causal case the numerator's sum becomes a prefix sum:

$$S_i = \sum_{j\le i}\phi(k_j)v_j^\top \in\mathbb{R}^{d_k\times d_v}, \qquad z_i=\sum_{j\le i}\phi(k_j)\in\mathbb{R}^{d_k}.$$

This gives the **recurrent form**:

$$\boxed{\;S_t = S_{t-1} + \phi(k_t)\,v_t^\top, \qquad o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}\;}$$

This is the interview punchline: **"linear attention = an RNN whose hidden state is a matrix $S\in\mathbb{R}^{d_k\times d_v}$."** Compared with a plain RNN's vector hidden state, here the state is **a matrix accumulated from outer products** — it stores a "$\phi(k)\to v$" associative memory (a.k.a. fast weight).

> ⚠️ **The state is a matrix, not a vector (a frequent follow-up)**
> Linear attention's recurrent state is $S\in\mathbb{R}^{d_k\times d_v}$ (an accumulation of outer products $\phi(k)v^\top$). Each decode step is $O(1)$ in sequence length, but $O(d_k d_v)$ in state size — both the update and the read are proportional to this matrix's size. If you say "$O(1)$ state," add the caveat "$O(1)$ is with respect to sequence length."

Two views, the same computation:

| View | Formula | Complexity | Suited for |
| --- | --- | --- | --- |
| **Parallel / quadratic** | $o = \big(\phi(Q)\phi(K)^\top \odot M\big)V$ ($M$ is the causal mask) | $O(L^2 d)$ | short sequences / intuition |
| **Recurrent / linear** | $S_t=S_{t-1}+\phi(k_t)v_t^\top,\ o_t=\phi(q_t)^\top S_t$ | $O(Ld_kd_v)$, state $O(d_kd_v)$ | decode / long sequences |

Note: the **naive parallel form** of causal linear attention still materializes $L\times L$ (because the mask is elementwise), so what is actually used at training time is the **chunkwise** compromise of §6 — neither materializing $L^2$ nor giving up matmuls.

### 2.3　The cost of losing softmax normalization

softmax has two implicit benefits that linear attention throws away:

1. **The normalization denominator $\phi(q)^\top z_t$ is unstable**: softmax's denominator is always positive and well-defined ($\sum\exp\ge 1$), but $\phi(q)^\top z_t$ can be very small or even near 0 (if $\phi$ does not guarantee strong positivity), causing exploding outputs / numerical instability. Many implementations simply **drop the denominator** (unnormalized linear attention) and use §2.4's decay / gating + extra normalization to stabilize the magnitude.

2. **Lack of "sharp selection" ability → weak recall**: softmax can concentrate probability mass highly onto one or two of the most relevant keys (near one-hot retrieval); but linear attention's $\phi(q)^\top\phi(k)$ is a finite-dimensional inner product, **unable to express arbitrarily sharp selection**. When a task needs to "precisely copy some token from the context" (associative recall / in-context copy), linear attention's fixed-size state suffers a **capacity conflict**: all the $\phi(k)v^\top$ pile up in the same $S$ and interfere with each other, getting blurrier the longer it gets.

> ❌ **"Linear attention is $O(L)$ so it's strictly better" — wrong**
> It is systematically weaker than softmax on **long-context retrieval / exact copy** tasks; this is a **capacity bottleneck** of the fixed-size state (information is compressed into the $d_k\times d_v$ matrix, and once over capacity it overwrites itself), not an implementation problem. This is exactly the core pain point that DeltaNet (rewrite-style update), gating (selective forgetting), and hybrid architectures (keeping a few full attention layers to rescue recall) later set out to solve.

### 2.4　Gating / decay: RetNet and GLA

Pure accumulation $S_t=S_{t-1}+\phi(k_t)v_t^\top$ has a problem: **old information never decays**, and the state gets polluted by unbounded stacking. Adding a decay helps a lot:

- **RetNet (Retentive Network)**: introduces a **scalar exponential decay** $\gamma\in(0,1)$: $S_t = \gamma S_{t-1} + k_t v_t^\top$. It has three equivalent forms — **parallel** (training, like attention with a decay mask), **recurrent** (decode, $O(1)$ state), and **chunkwise** (long-sequence training, intra-chunk parallel + inter-chunk recurrence). $\gamma$ is usually **fixed per head** (not data-dependent).
- **GLA (Gated Linear Attention)**: upgrades the scalar decay to a **data-dependent diagonal gate** $G_t=\text{diag}(\alpha_t)$, with $\alpha_t\in(0,1)^{d_k}$ computed from the input: $S_t = G_t\,S_{t-1} + k_t v_t^\top$ (per-channel forgetting). GLA gives a hardware-efficient chunkwise form, and is the representative work unifying "gated RNN" and "linear attention."

You can view this line as a spectrum: **pure linear attention (no decay) → RetNet (scalar decay) → GLA (data-dependent diagonal gate) → Gated DeltaNet (gating + rewrite-style update, see §5)** — all in the $O(L)$ tier, differing in how the state "forgets" and "writes."

## §3 State space models (SSM) and Mamba

### 3.1　Continuous SSM and discretization

State space models come from control theory. A **continuous-time** SSM maps an input signal $x(t)$ to an output $y(t)$, with a hidden state $h(t)\in\mathbb{R}^{N}$ in between:

$$h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t),$$

$A\in\mathbb{R}^{N\times N}$ (state transition), $B\in\mathbb{R}^{N\times 1}$, $C\in\mathbb{R}^{1\times N}$. To use it on a discrete token sequence, you must **discretize**: introduce a step size $\Delta$ and use the **zero-order hold (ZOH)** to discretize the continuous system into a recurrence:

$$\bar A = \exp(\Delta A), \qquad \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\,\Delta B \approx \Delta B,$$

$$\boxed{\;h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t\;}$$

This step turns a "continuous dynamical system" into a **linear recurrence** — formally the same as an RNN, but the transition matrix $\bar A$ is structured. $\Delta$ controls "how fast the system reacts": large $\Delta$ ≈ weighting the current input more, fast updates; small $\Delta$ ≈ more persistent state memory.

### 3.2　HiPPO initialization intuition

If $A$ is initialized randomly, this recurrence cannot learn long-range dependencies. **HiPPO (High-order Polynomial Projection Operators)** gives a special structured $A$ matrix that makes the hidden state $h_t$ the **optimal polynomial-projection coefficients of the historical input** — i.e. $h_t$ is mathematically the result of "compressing the entire past signal with a set of orthogonal polynomials (e.g. Legendre)." Intuition: HiPPO lets a fixed-size state remember history in a "principled, minimal-information-loss" way, the key prior that makes S4 / Mamba work on long sequences. For interviews you only need to state clearly "HiPPO = a structured $A$ initialization that makes the state optimally compress history."

### 3.3　Selective SSM (S6 / Mamba): make the parameters data-dependent

Classic S4 is **LTI (linear time-invariant)**: $\bar A, \bar B, C$ are the same for all $t$ and independent of the input. LTI has a fundamental flaw — it **cannot do content selection** (content-based reasoning): the same transition matrix processes every token, with no way to "remember more when it sees an important token and skip filler." This is why S4 underperforms Transformers on language modeling.

**Mamba's core innovation (Selective SSM, S6 for short)**: make $B, C, \Delta$ **vary with the input $x_t$** (data-dependent / input-dependent), achieving **selectivity** — deciding what to remember and what to ignore based on content.

$$B_t = \text{Linear}_B(x_t),\quad C_t = \text{Linear}_C(x_t),\quad \Delta_t = \text{softplus}\big(\text{Linear}_\Delta(x_t)\big),$$

$$h_t = \bar A_t\,h_{t-1} + \bar B_t\,x_t,\qquad y_t = C_t\,h_t, \qquad \bar A_t=\exp(\Delta_t A).$$

> ⚠️ **Which exactly is data-dependent? (very easy to get wrong)**
> In Mamba, **$B, C, \Delta$ are data-dependent, while $A$ itself is still a structured, input-independent fixed parameter** (one learned $A$ per channel). $\Delta_t$ (data-dependent) turns the "effective decay rate" into something data-dependent via $\bar A_t=\exp(\Delta_t A)$ — so **selectivity is injected mainly through $\Delta$ and $B,C$, not by directly making $A$ vary with the input**. Saying "Mamba makes $A$ data-dependent" is inaccurate; you should point out that $A$ is structured and fixed, and $\Delta$ is the data-dependent gate.

**The cost**: once $\bar A_t$ varies with time, the system is no longer LTI, and **the global convolution (FFT) form that S4 relies on for efficient training breaks** — because the convolution kernel $\bar C\bar A^k \bar B$ is no longer fixed. Mamba's solution is a **hardware-aware selective scan**: write this time-varying recurrence as a GPU kernel, using a parallel prefix scan (parallel scan / associative scan) to parallelize over the sequence dimension, keeping the state in SRAM and writing back to HBM only when necessary (kernel fusion, avoiding materializing the $L\times N$ intermediate state). This is the origin of Mamba's two keywords "selective + hardware-aware."

### 3.4　The full picture of a Mamba block

A complete Mamba block is not just the SSM; it wraps in gating and convolution: input → linear projection expansion → a **short causal 1D convolution** (local mixing) → SiLU → **selective SSM** → a **gating branch** (elementwise multiply by SiLU(z)) → project back. The short convolution handles local patterns, the SSM handles long range, and the gate handles selective gating. Remembering "Mamba ≠ pure SSM, but a combined block of conv + selective SSM + gate" shows more understanding than just memorizing the SSM formula.

## §4 Mamba-2 and State Space Duality (SSD)

### 4.1　The core duality: SSM ⇔ structured masked attention

Mamba-1's selective scan is a hand-written CUDA kernel that **does not go through the tensor-core matmul path**, so although its asymptotic complexity is low, its constant is large, and it has no overwhelming speed advantage over the highly optimized FlashAttention. Mamba-2's paper, *Transformers are SSMs*, proposes **State Space Duality (SSD)**, connecting SSM and attention mathematically.

Expand the selective SSM recurrence (with $h_0=0$, scalar $\bar A$ case):

$$y_t = \sum_{s\le t} C_t\Big(\prod_{r=s+1}^{t}\bar A_r\Big)\bar B_s\, x_s = \sum_{s\le t} \underbrace{C_t\Big(\prod_{r=s+1}^{t}\bar A_r\Big)\bar B_s}_{\textstyle M_{ts}}\, x_s.$$

Write it in matrix form $y = M x$, where $M$ is a **lower-triangular matrix**, $M_{ts}=C_t\big(\prod_{r=s+1}^t \bar A_r\big)\bar B_s$ (0 when $t \lt s$). This $M$ looks exactly like a **causal attention matrix with decay**: $C_t$ is like a query, $\bar B_s$ is like a key, and the product $\prod\bar A$ in the middle is the "positional decay."

> ✅ **SSD in one sentence**
> **A selective SSM is equivalent to structured masked attention using a 1-semiseparable matrix $M$.** "Semiseparable" means that any sub-block of $M$'s lower triangle has rank $\le 1$ (determined by the product structure of $C, \bar A, \bar B$), which is exactly the algebraic reason the SSM recurrence can be computed in $O(L)$; and treating $M$ as a matrix and multiplying $x$ directly lets you use matmuls. The two computation paths (linear recurrence vs quadratic matmul) compute the **same $M$** — this is the duality.

> ⚠️ **Don't overclaim**
> SSD says "SSM ≡ **a structured linear attention with a specific decay mask**," **not** "SSM = ordinary softmax attention." $M$ has no softmax and is semiseparable-structured. This equivalence holds at the level of **linear attention / masked linear attention**, and the wording must be precise.

### 4.2　The benefit: training SSMs with chunkwise matmuls

With SSD, Mamba-2 can train with §6's **chunkwise / block-decomposition** algorithm:

- **Intra-chunk (diagonal blocks)**: directly materialize the small $M$ sub-block and compute quadratic attention with matmuls (tensor core) — the block is small, so $L_{\text{chunk}}^2$ is acceptable.
- **Inter-chunk (off-diagonal blocks)**: because of the semiseparable (rank-1) structure, crossing blocks only needs to pass a **low-rank state** (precisely the SSM's hidden state $h$), processed in batch with matmuls.

The effect: Mamba-2 trains several times faster than Mamba-1 (saturating tensor cores), and because the state dimension $N$ can be made larger (matmuls are cheap), its expressiveness is also stronger. Mamba-2 also conveniently simplifies the structure ($A$ degenerates to a **scalar times the identity** $\bar A_t = a_t I$, i.e. one scalar decay per step), making the SSD derivation and implementation cleaner.

> 💡 **Interview punchline**: Mamba-1 = hardware-aware selective **scan** (a custom kernel that doesn't use tensor cores); Mamba-2 = uses **SSD** to rewrite the same computation in a **matmul-heavy chunkwise** form (saturating tensor cores), training faster with a larger state. The two are algorithmically equivalent, differing in "whether you implement with a scan or with matmuls."

## §5 Delta rule: DeltaNet and Gated DeltaNet

### 5.1　From "accumulate" to "rewrite"

Linear attention's state update is **pure accumulation**: $S_t = S_{t-1} + v_t k_t^\top$ (here $\phi(k)$ is abbreviated as $k$). The problem, as said before — when the same key recurs, or different keys interfere, information only **stacks up and pollutes**, with no "update / overwrite" mechanism.

**The delta rule (from fast-weight / online learning)** uses a different formulation: view the state $S$ as a linear map $k\mapsto S k$ (retrieve a value with a key), and for each incoming $(k_t,v_t)$ we want $S$'s "prediction" at $k_t$, $S_{t-1}k_t$, to approach the target $v_t$. Do one step of online gradient descent / least-squares correction:

$$S_t = S_{t-1} - \beta_t\big(\underbrace{S_{t-1}k_t - v_t}_{\text{prediction error}}\big)k_t^\top = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top.$$

$$\boxed{\;S_t = S_{t-1}\big(I - \beta_t\, k_t k_t^\top\big) + \beta_t\, v_t k_t^\top\;}$$

(assuming the normalization $\|k_t\|=1$). $\beta_t\in(0,1]$ is the **write strength / learning rate** (which can also be data-dependent).

> ✅ **$(I-\beta k k^\top)$ is DeltaNet's soul**
> This projection term first **erases the old association** of the state in the $k_t$ direction (right-multiplying $S_{t-1}$ by $(I-\beta kk^\top)$ attenuates the component in the $k_t$ direction), then writes in the new $\beta v_t k_t^\top$. This is "**rewrite / overwrite**" rather than "accumulate."
> Compare pure linear attention: $S_t = S_{t-1} + v_t k_t^\top$ (without that projection term = always add, never rewrite).

Intuitively, look at the extreme cases:

- $\beta_t = 1$ and $k_t$ orthogonal to all historical keys: $(I-k_tk_t^\top)$ leaves history untouched, equivalent to "first clear the $k_t$ direction (already empty) then write" — degenerating to one write of ordinary linear attention.
- $k_t$ recurs: the delta rule **overwrites** the value written at $k_t$ last time, whereas pure linear attention adds the two values together (a wrong accumulation). This is exactly why DeltaNet is significantly stronger than ordinary linear attention on associative recall.

### 5.2　Gated DeltaNet: add a forgetting gate

DeltaNet's delta rule solves "precise rewriting," but lacks a **global forgetting** mechanism (old information fading out as a whole over time). **Gated DeltaNet** combines a GLA-style decay gate with the delta rule:

$$S_t = \alpha_t\, S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,$$

where $\alpha_t\in(0,1)$ (scalar or diagonal, data-dependent) is the **decay / forgetting gate**. In one sentence: **$\alpha_t$ controls "how much history to forget overall," and $\beta_t$ controls "how strongly to rewrite in the $k_t$ direction."** The two mechanisms are orthogonal and complementary — gating handles continuous fading, the delta rule handles precise overwriting. This makes Gated DeltaNet stronger on both long-context recall and state utilization, and is why recent hybrid architectures like Qwen3-Next and Kimi-Linear (whose KDA is a variant) pick it for the linear layer.

### 5.3　How to parallelize: the WY representation + chunkwise

The delta rule has a product of $(I-\beta kk^\top)$ terms, which looks strongly sequential and hard to parallelize. But the product over $T$ steps within a chunk, $\prod_t (I-\beta_t k_t k_t^\top)$, can be expanded into a compact $I - W Y^\top$ form using the **WY representation of a product of Householder matrices** (a classic trick from QR decomposition in numerical linear algebra), so that intra-chunk you batch with matmuls and inter-chunk you pass the state. This is the key to DeltaNet-family chunkwise parallel training over sequence length (one of the main engineering contributions of Yang et al. 2024). For interviews just remember "DeltaNet's intra-chunk parallelism relies on the WY / Householder representation," without memorizing the expansion.

## §6 Chunkwise parallel: how to parallelize at training time

### 6.1　Why it is needed

Linear attention / SSM / DeltaNet have two faces:

- **The recurrent form** is good for **inference**: $O(1)$ state, $O(1)$ per token, but **strongly sequential** ($S_t$ depends on $S_{t-1}$), so at training it runs the full $L$ steps token by token with very low GPU utilization.
- **The parallel / quadratic form** is good for **training**: it can use matmuls, but in the causal case it must materialize $L\times L$, blowing up memory on long sequences.

**Chunkwise parallel** is the compromise bridge between the two, the core trick that makes this whole family **actually trainable**. The idea: split the sequence into $L/C$ chunks of size $C$, **parallelize within a chunk and recur across chunks**.

### 6.2　Two stages: intra-chunk (parallel within a chunk) + inter-chunk (recurrence across chunks)

Take ungated linear attention as the example ($S_t=S_{t-1}+k_tv_t^\top$, state $S\in\mathbb{R}^{d_k\times d_v}$, here with $\phi=\text{id}$, so $k$ is $\phi(k)$). Let the query/key/value of the $c$-th chunk be $Q_c,K_c,V_c\in\mathbb{R}^{C\times d}$, and the state entering this chunk be $S_c$ (= $\sum k v^\top$ over all preceding chunks). The output of the $i$-th query in the chunk splits into two parts:

$$o_i = \underbrace{q_i^\top S_c}_{\textbf{inter: contribution of past chunks}} + \underbrace{\sum_{j\le i,\, j\in c} (q_i^\top k_j)\, v_j}_{\textbf{intra: causal attention within this chunk}}.$$

In chunk-level matrix form:

$$O_c = \underbrace{Q_c\, S_c}_{[C\times d_k][d_k\times d_v]=[C\times d_v]} \;+\; \underbrace{\big(\,(Q_c K_c^\top)\odot M\,\big)\,V_c}_{\text{intra-chunk quadratic, } M\text{ is the } C\times C\text{ causal mask}},$$

after the chunk ends, **update the cross-chunk state**:

$$S_{c+1} = S_c + K_c^\top V_c \quad (\in\mathbb{R}^{d_k\times d_v}).$$

> 💡 **Three matmuls to understand chunkwise**
> - **inter-chunk** `Q_c @ S_c`: history is compressed into a $d_k\times d_v$ matrix $S_c$, read once by all queries in this chunk (this part is **linear** in $L$).
> - **intra-chunk** `((Q_c K_c^T) ⊙ M) @ V_c`: quadratic attention only within the $C\times C$ small block ($C$ is a constant, e.g. 64/128, so **the total cost is linear** in $L$).
> - **state update** `S_c + K_c^T V_c`: accumulate this chunk's information into the state, passed to the next chunk.

Complexity: $O\!\big(\tfrac{L}{C}\cdot(Cd_kd_v + C^2 d)\big)=O(Ld_kd_v + LCd)$, which is $O(L)$ for fixed $C$ and is all matmuls (saturating tensor cores). **The chunk size $C$ is the key knob**: too small $C$ → inter-chunk recurrence dominates, low parallelism; too large $C$ → the intra-chunk quadratic term $C^2$ gets expensive. Note that this section's state update $S_{c+1}=S_c+K_c^\top V_c$ **holds only for the ungated additive version**; with gating / decay you must fold in the corresponding factors — GLA / Mamba-2 multiply $S_c$ and the intra-chunk mask by the cumulative decay $\prod\alpha$, and **DeltaNet additionally carries the $(I-\beta_t k_tk_t^\top)$ rewrite term inside the chunk** (expanded with §5.3's WY representation). The skeleton (parallel within a chunk + pass the state across chunks) is exactly the same; only the intra/cross-chunk operators change from pure-additive to "decay × rewrite."

### 6.3　Code: recurrent vs chunkwise, the outputs should match

```python
import torch

def linear_attn_recurrent(Q, K, V):
    """Token-by-token recurrence. Q,K,V: [L, d]  ->  O: [L, d]
       State S: [d_k, d_v] matrix (here d_k=d_v=d)."""
    L, d = Q.shape
    S = torch.zeros(d, d, dtype=Q.dtype)          # [d_k, d_v] matrix state
    O = torch.zeros(L, d, dtype=Q.dtype)
    for t in range(L):
        S = S + torch.outer(K[t], V[t])           # S += k_t v_t^T   accumulate outer product
        O[t] = Q[t] @ S                           # o_t = q_t^T S    [d]
    return O

def linear_attn_chunkwise(Q, K, V, C=4):
    """Intra-chunk parallel + inter-chunk recurrence. Mathematically identical to recurrent."""
    L, d = Q.shape
    S = torch.zeros(d, d, dtype=Q.dtype)          # cross-chunk state [d_k, d_v]
    O = torch.zeros(L, d, dtype=Q.dtype)
    tri = torch.tril(torch.ones(C, C, dtype=Q.dtype))  # in-chunk causal mask [C, C]
    for s in range(0, L, C):
        e = min(s + C, L)
        Qc, Kc, Vc = Q[s:e], K[s:e], V[s:e]       # [c, d]
        c = e - s
        m = tri[:c, :c]
        inter = Qc @ S                            # history-chunk contribution [c, d]
        intra = ((Qc @ Kc.t()) * m) @ Vc          # in-chunk causal attention [c, d]
        O[s:e] = inter + intra
        S = S + Kc.t() @ Vc                        # update state [d_k, d_v]
    return O

if __name__ == "__main__":
    torch.manual_seed(0)
    L, d = 13, 6
    Q, K, V = torch.randn(L, d), torch.randn(L, d), torch.randn(L, d)
    o1 = linear_attn_recurrent(Q, K, V)
    o2 = linear_attn_chunkwise(Q, K, V, C=4)
    print("max |Δ| =", (o1 - o2).abs().max().item())   # expect ~1e-6 (pure float error)
    assert torch.allclose(o1, o2, atol=1e-5)
```

The two paths (recurrent / chunkwise) compute the same causal linear attention, and the outputs should match within floating-point error — this is the golden sanity check for verifying implementation correctness (see §A).

## §7 Sparse attention (trainable block-sparse)

### 7.1　A different family: keep softmax, attend only to a subset

The preceding §2–§6 are "give up softmax for a recurrence." Sparse attention takes another path: **leave softmax alone, but each query only looks at a subset** $S(i)\subseteq\{1..L\}$, cutting $L^2$ to $L\cdot |S(i)|$. The early ones (2019–2020) used **fixed patterns**:

- **Longformer / BigBird**: a sliding window (local) + a few global tokens + (BigBird adds) random tokens, a fixed sparse graph, $O(L)$. The problems: the pattern is human-designed and may not match the data; and many kernels are IO-unfriendly on modern GPUs.

The frontier of the last two years (2025) is **trainable / content-dependent + hardware-aligned** sparse attention — letting the model **learn for itself** which blocks to look at, with patterns designed so GPUs can run them efficiently.

### 7.2　NSA (Native Sparse Attention, DeepSeek)

**NSA's two keywords: natively trainable + hardware-aligned.** Rather than slapping a sparse mask onto a dense model after the fact, it **uses the sparse structure from pretraining onward**, end-to-end learnable. Each query's output is a **gated sum of three branches**:

> 💡 **NSA's three branches (a frequent exam point, must memorize all)**
> - **(1) Compressed (compressed, coarse-grained)**: **compress** distant key/value into a few "representative tokens" per block (intra-block aggregation), letting the query see a coarse global picture at very low cost.
> - **(2) Selected (selected, fine-grained top-k blocks)**: using the block importance computed by the compressed branch, **select the top-k most relevant key blocks** and do full (fine-grained) attention on them — this is the workhorse for "exact retrieval."
> - **(3) Sliding window**: fixedly attend to the most recent tokens, ensuring **local context** is not lost (local patterns matter a lot for language modeling).
> Each of the three branches produces an output, fused by a **learned gate**.

NSA's "hardware-aligned" means: the block size and selection granularity are designed in a GPU-friendly way for memory access / tensor cores (contiguous block access, avoiding the inefficiency of random gather), so it is not only theoretically $O(Ls)$ but **delivers real wall-clock speedup over full attention on long sequences in practice** (both training and decode accelerate) — this is where it differs from many methods that are "theoretically sparse but actually slower."

### 7.3　MoBA (Mixture of Block Attention, Moonshot)

**MoBA brings MoE's "routing" idea to attention**: split key/value into blocks, and each query, like an MoE router, **routes only to the top-k most relevant key blocks**, doing softmax attention on those k blocks and skipping the rest entirely. "Which blocks are relevant" is scored by the query against each block's representative (e.g. the mean of the keys in the block). The benefits:

- routing is **learnable and content-dependent** (not a fixed pattern);
- block-granularity selection is hardware-friendly (blocks are contiguous);
- it is designed so you can **seamlessly switch between full attention and MoBA** (the same weights, mixable during training), easing migration of existing models.

A one-sentence comparison: **NSA = the fixed fusion of three branches compress + select + window; MoBA = MoE-style routing assigning a query to its top-k key blocks**. Both are representatives of 2025's "trainable sparse," with a shared idea (content-dependently selecting blocks) but different structures.

### 7.4　Lightning Attention (MiniMax-01) and DeepSeek-V3.2 DSA

- **Lightning Attention (MiniMax-01)**: strictly speaking it is an **IO-aware linear attention** implementation (not sparse softmax), using FlashAttention-like tiling + chunkwise to make linear attention IO-efficient, paired with a hybrid architecture (interleaving a few softmax layers) to support million-scale context. It is placed here because MiniMax-01 is a representative large-scale deployment of "linear + a few full attention layers" (see §9).
- **DeepSeek-V3.2's DSA (DeepSeek Sparse Attention) / lightning indexer**: a **lightweight indexer (lightning indexer)** quickly scores, for each query, "which historical tokens to attend to," then runs attention only on the selected tokens. The core is using a **cheap scoring network** to do token-level selection, lowering the cost of long-context attention while trying not to drop quality.

> ⚠️ **Don't classify Lightning Attention as "sparse softmax"**
> Lightning Attention is essentially an efficient (IO-aware) implementation of **linear attention**, belonging to §2's family; DSA / NSA / MoBA are the "keep softmax, select a subset" sparse attentions. Conflating them in an interview gives you away — the distinguishing point is "did it give up softmax, does it have a fixed-size recurrent state."

### 7.5　Fixed pattern vs trainable sparse (a one-sentence comparison)

| Dimension | Fixed pattern (the Longformer/BigBird era) | Trainable sparse (NSA / MoBA, 2025) |
| --- | --- | --- |
| Sparse structure | human-designed (window + global + random) | the model **learns** it (content-dependent block selection) |
| Training | usually slapped on at fine-tuning time | **natively** sparse from pretraining (NSA) |
| Hardware | mostly IO-unfriendly | **hardware-aligned** (contiguous blocks, tensor-core-friendly) |
| Quality | usable on long documents, with quality loss | close to full attention, with real speedup |

## §8 KV cache and inference: the trump card of linear/SSM

### 8.1　The inference profiles of the three families

In the autoregressive decode phase (emitting tokens one by one), the three families have starkly different resource profiles:

| Method | decode per step | "memory" growth with context $L$ | per-step access |
| --- | --- | --- | --- |
| **softmax + KV cache** | $O(L\,d)$ (new q against all historical k/v) | KV cache **grows linearly $O(L)$** | read the whole KV cache (bandwidth bottleneck) |
| **linear attention / SSM** | $O(d_k d_v)$ (update + read a fixed state) | **fixed-size state $O(d_k d_v)$, independent of $L$** | read/write only that small state |
| **sparse (NSA/MoBA)** | $O(s\,d)$ (only the selected $s$ tokens/blocks) | still must **store all KV** ($O(L)$), but only **accesses** a subset | read a subset (less than full) |

> ✅ **This is the biggest selling point of linear / SSM**
> softmax's KV cache **inflates linearly** with context — a 128K-context, 70B-model KV cache can reach tens of GB, the number-one memory / bandwidth killer of long-context inference. Linear attention / SSM compress history into a **fixed-size** recurrent state ($d_k\times d_v$ or the SSM's $N$-dim), so **memory is independent of context length and per-token decode is $O(1)$** — their hardest advantage in long-context / high-throughput serving.

### 8.2　KV cache memory comparison (intuitive numbers)

softmax KV cache (per sample, consistent with the attention sheet):

$$\text{KV cache} = L \cdot n_{\text{layers}} \cdot 2 \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot \text{bytes}.$$

It **grows linearly** with $L$. Whereas linear attention / SSM's "state memory" is:

$$\text{state} = n_{\text{layers}} \cdot (\text{per-layer state size}) ,\quad \text{per layer} \approx d_k\, d_v\ (\text{linear}) \ \text{or}\ N\, d\ (\text{SSM}),$$

**without $L$**. So even if context grows from 4K to 1M, the state memory of linear / SSM layers stays untouched — the cost is the recall capacity ceiling of §2.3. Sparse attention does **not** save on this dimension (it still stores all KV); what it saves is **compute / access** (only computing a subset), which is the essential divide between it and linear/SSM.

> ⚠️ **In hybrid architectures, don't forget the full attention layers' KV cache**
> Those few full attention layers in a hybrid (§9) **still have a KV cache that grows with $L$** — so a hybrid model is not "fully constant memory," but "most layers constant + a few layers linear." This is often overlooked when computing a long-context memory budget.

## §9 Hybrid architectures

### 9.1　The recall–throughput trade-off: why pure linear/SSM is not enough

Pure linear / SSM models have high throughput and a fixed state, but **weak associative recall**: after stuffing a fact into a fixed-size state, retrieving it **precisely** much later is limited by state capacity (information overwritten / interfered). Several studies (e.g. the Based, Zoology series) point out a **recall–memory (throughput) Pareto frontier**: a smaller state is faster but recalls worse. softmax attention sits at the "high recall, low throughput" end of this frontier, pure SSM at the "high throughput, low recall" end.

**The fix is surprisingly simple and crude: among many linear / SSM layers, interleave a few full softmax attention layers.** Those few full attention layers handle "exact retrieval / in-context copy," and the rest of the linear/SSM layers handle "cheaply processing long-range context." A small number of full attention layers can restore recall to near a pure Transformer, while the overall complexity and KV cache are still dominated by the linear/SSM layers (close to linear).

> ✅ **Why hybrid wins**
> Under a fixed budget, a hybrid gets the benefits of both ends at once: **linear/SSM layers → long-context throughput and memory (constant state)**; **a few full attention layers → recall / exact retrieval**. Empirically a very small full-attention ratio (commonly 1:5 to 1:7, i.e. 1 full layer every 6–8 layers) can approach the quality of pure attention while cutting a large chunk of KV cache. This is why a great many industrial long-context models of 2024–2025 chose hybrid.

### 9.2　Representative hybrid models and ratios

| Model | linear / SSM component | full attention component | rough ratio / characteristics |
| --- | --- | --- | --- |
| **Jamba** | Mamba layers | Transformer layers + MoE | Mamba:Attention about 7:1, inserting 1 attention layer per block; the first large-scale Mamba-Transformer hybrid |
| **Zamba** | Mamba backbone | **shares** one global attention block (reused in multiple places) | uses a single shared attention to save parameters |
| **Hymba** | SSM head | attention head | runs the attention head and SSM head **in parallel within the same layer** (head-level mixing, not layer-level interleaving) |
| **Qwen3-Next** | **Gated DeltaNet** layers | output-gated full attention layers | a large proportion of Gated DeltaNet + a few gated attention layers, emphasizing long-context efficiency |
| **Kimi-Linear** | **KDA** (Kimi Delta Attention, a Gated DeltaNet variant) | periodic full attention (MLA) | linear:full about 3:1, with long-context throughput and KV cache greatly optimized |
| **MiniMax-01** | **Lightning Attention** (linear) | softmax attention | a large proportion of lightning + periodic softmax, supporting million-scale context |

> 💡 **Hymba's head-level mixing vs the others' layer-level interleaving**
> Most hybrids interleave at the **layer level** (some layers are SSM, some are attention); **Hymba mixes within a layer**, placing attention heads and SSM heads **in parallel** in the same layer (fused / parallel heads). These are two different mixing granularities, and pointing out "layer-level vs head-level" in an interview earns points.

### 9.3　Recent trends

A clear 2024→2025 trend: **the linear layer upgrades from "pure-additive linear attention / Mamba" to "gating + rewrite-style update" (Gated DeltaNet / KDA)**, because the latter has stronger recall and needs fewer full attention layers; meanwhile those few full attention layers often adopt GQA / MLA / output gating and other KV-saving techniques. In other words, both ends of the hybrid are getting stronger: the linear end approaches attention's recall via the delta rule + gating, and the attention end compresses the KV cache via MLA/GQA.

## §10 Engineering practice and common misconceptions

- **Linear attention is no free lunch**: its weak recall is the **capacity essence of a fixed-size state**, not a bug. On tasks needing exact long-range retrieval (multi-hop QA, long code completion, needle-in-a-haystack), pure linear/SSM drops points — which is the fundamental reason hybrids exist.
- **"Linear = $O(N)$ so it must be faster and better" is wrong**: (1) low asymptotic complexity ≠ actually faster — the constant and access pattern matter, and after FlashAttention's heavy optimization, softmax is often **faster** on short-to-medium sequences; linear/SSM's advantage shows only at very long sequences. (2) faster ≠ better — the recall shortcoming is independent of speed.
- **The chunk size must be tuned**: too small $C$ → low parallelism, inter-chunk recurrence dominates; too large → the intra-chunk $C^2$ term gets expensive + memory grows. Typically $C\in[64,256]$, tuned by sequence length / hardware.
- **No softmax normalization → numerical stability must be managed separately**: pure accumulation state grows unboundedly / drifts in magnitude. In practice you stabilize with **decay gates ($\gamma,\alpha_t$)**, **normalization on the state or output** (RMSNorm / L2), and **bounded activations for $\beta,\Delta$ (sigmoid/softplus)**. Directly carrying over softmax attention's implementation habits easily blows up.
- **The sparse hardware alignment is crucial**: theoretically sparse ≠ actually faster. Random gather and non-contiguous access can make a sparse kernel slower than dense — which is why NSA / MoBA emphasize "contiguous blocks, tensor-core-friendly." When evaluating a sparse method **always look at real wall-clock / end-to-end throughput**, not just FLOPs.
- **The relationship with RoPE**: linear attention / SSM don't apply RoPE directly the way softmax attention does — their "positional information" comes from the recurrent structure itself (the decay $\gamma^{t-s}$, $\Delta$, and the gate $\alpha$ provide an implicit relative position / temporal decay). In hybrid architectures, **full attention layers use RoPE, and linear/SSM layers use their own decay mechanisms**; forcing RoPE into linear attention's $\phi(q),\phi(k)$ takes care (it interacts with the decay term).
- **Distinguish "linear attention" from "sub-quadratic but still attention"**: FlashAttention is an IO optimization of **exact softmax** (still $O(L^2)$ FLOPs, only saving memory), **not** linear attention; Performer/Linformer are the sub-quadratic approximations. Linear attention / SSM **replace attention with a fixed-state recurrence**, fundamentally changing the compute paradigm. Interviewees often conflate these three classes.

> ⚠️ **Biggest footgun #1: taking asymptotic complexity as actual performance**
> "My method is $O(L)$, so it must be faster/better than the $O(L^2)$ Transformer" — wrong twice: actual speed depends on the constant and access (FlashAttention is very strong), and quality depends on recall (linear has a shortcoming). **Any efficient-attention claim must provide real wall-clock + long-context recall benchmarks**, or it doesn't hold up.

> ❌ **Biggest footgun #2: thinking SSD / SSM "is" attention, or linear attention "≈" softmax**
> SSD is the precise equivalence "SSM ≡ **structured masked (semiseparable) linear attention**," **not** softmax attention; linear attention is a **finite-dimensional approximation** of softmax, **fundamentally limited** on sharp selection / recall. Overclaiming these equivalences / approximations into "identical to / lossless replacement for softmax" is a classic mistake that gets probed to the bottom.

## §11 Complexity and resources

The table below compares the five families on "training time / decode per step / KV·state memory / recall / length extrapolation / parallelizable" ($L$=sequence length, $d$=hidden dim, $d_k d_v$=linear state, $N$=SSM state dim, $s$=number selected in sparse, $C$=chunk). Absolute values / norms in the table use $\lvert\cdot\rvert$, $\lVert\cdot\rVert$.

| Method | training time | decode per step | KV / state memory (with $L$) | recall | length extrapolation | parallel training |
| --- | --- | --- | --- | --- | --- | --- |
| **softmax (Flash)** | $O(L^2 d)$ | $O(L d)$ | KV cache $O(L)$ linear growth | **strong** | RoPE+YaRN moderate | yes (matmul) |
| **linear attention** | $O(L d_k d_v)$ (chunkwise) | $O(d_k d_v)$ | state $O(d_k d_v)$, **independent of $L$** | weak (capacity bottleneck) | good (decay extrapolates naturally) | yes (chunkwise matmul) |
| **SSM / Mamba-2** | $O(L\,N d)$ (SSD chunkwise) | $O(N d)$ | state $O(N d)$, **independent of $L$** | medium (stronger than pure linear, weaker than softmax) | good | yes (SSD matmul) |
| **DeltaNet / Gated** | $O(L d_k d_v)$ (WY chunkwise) | $O(d_k d_v)$ | state $O(d_k d_v)$, **independent of $L$** | medium-strong (rewrite-style update lifts recall) | good | yes (WY chunkwise) |
| **sparse (NSA)** | $O(L\,s\,d)$ ($s\ll L$) | $O(s\,d)$ (still stores full KV) | KV $O(L)$ (stores all, accesses a subset) | **strong** (keeps softmax) | depends on window/selection | yes (block matmul, hardware-aligned) |

Key reading points:

- **State memory independent of $L$** is achieved only by the three families linear / SSM / DeltaNet; sparse still stores $O(L)$ KV (what it saves is compute/access).
- **recall** rough ordering (**empirical**, varying by model scale / training recipe / benchmark, not a theorem; in particular the relative positions of DeltaNet/Gated and Mamba-2 depend heavily on the specific setup): softmax ≈ sparse(NSA) > DeltaNet/Gated > Mamba-2 > pure linear attention. Note NSA's "≈ softmax" is **softmax-over-selected** (it keeps softmax, but computes it only over the selected blocks/tokens) — when the selector hits, it approaches full attention's recall, but it is **not exact full softmax**, and in an extreme needle-in-a-haystack it can still miss if it picks the wrong block. This is exactly the motivation for hybrids to combine softmax's recall with linear's throughput.
- **All trainable in parallel**: all five families can train with parallel matmuls (linear/SSM/DeltaNet via chunkwise, sparse via blocks), differing in the constant and access friendliness.
- **decode**: linear/SSM/DeltaNet are truly $O(1)$ in sequence length; softmax is $O(L)$; sparse is $O(s)$ but must maintain the full KV.

## §12 25 high-frequency interview questions

Sorted into three tiers. Click to expand for answer points + pitfalls. L2/L3 are the top-lab deep end (SSD, the delta rule, chunkwise, NSA's three branches, hybrid, etc.).

### L1 must-know

<details>

<summary>Q1. What are softmax attention's two quadratic/linear bottlenecks?</summary>

- training/prefill: materializes the $L\times L$ score matrix, time $O(L^2 d)$ (FlashAttn saves memory but not FLOPs)
- decode: the KV cache **grows linearly** with context ($O(L)$ memory + $O(L)$ access per step)
- three escape routes: linear attention, SSM/Mamba, sparse attention

Saying only "$O(n^2)$," not distinguishing the two different bottlenecks "quadratic training time" and "linear decode KV cache."

</details>

<details>

<summary>Q2. Why can linear attention achieve $O(L)$?</summary>

- use a feature map $\phi$ to write $\text{sim}(q,k)$ as the inner product $\phi(q)^\top\phi(k)$
- lean on **associativity** to reorder $\big(\phi(Q)\phi(K)^\top\big)V$ into $\phi(Q)\big(\phi(K)^\top V\big)$
- compress K, V into a $d_k\times d_v$ small matrix first, then interact with Q, avoiding materializing $L\times L$

Saying only "drop softmax," unable to articulate the "associativity + compress KV first" mechanism; or not knowing that softmax cannot be factored this way (needs infinite dimensions).

</details>

<details>

<summary>Q3. How do you understand "linear attention is an RNN"? What shape is its state?</summary>

- causal linear attention can be written as the recurrence $S_t=S_{t-1}+\phi(k_t)v_t^\top$, $o_t=\phi(q_t)^\top S_t$
- the state $S\in\mathbb{R}^{d_k\times d_v}$ is a **matrix** (an associative memory accumulated from outer products)
- unlike a plain RNN's vector hidden state

Saying the state is a vector (wrong, it's a matrix); or not knowing it and the parallel form are two faces of the same computation.

</details>

<details>

<summary>Q4. What is the SSM's basic recurrence? Why discretize?</summary>

- continuous: $h'(t)=Ah(t)+Bx(t),\ y(t)=Ch(t)$
- discrete (ZOH, step size $\Delta$): $h_t=\bar A h_{t-1}+\bar B x_t,\ y_t=Ch_t$, $\bar A=\exp(\Delta A)$
- a token sequence is discrete, so you must discretize the continuous dynamical system into a recurrence to use it

Unable to write the continuous→discrete step; or not knowing that $\Delta$ is the step size / controls reaction speed.

</details>

<details>

<summary>Q5. What is Mamba's key innovation over S4?</summary>

- Selective SSM (S6): make $B,C,\Delta$ **data-dependent** (varying with the input)
- thereby gaining **content selection** (remember more of what matters, skip filler), filling LTI's shortcoming
- the cost is no longer being LTI, the FFT convolution breaks, replaced by a hardware-aware selective scan

Saying "Mamba makes $A$ data-dependent" (inaccurate; $A$ is structured and fixed, it's $\Delta,B,C$ that are data-dependent).

</details>

<details>

<summary>Q6. What is the biggest inference advantage of linear attention / SSM?</summary>

- keeping a **fixed-size** recurrent state (independent of context length $L$)
- constant memory + $O(1)$ per-token decode, very favorable for long-context / high-throughput serving
- versus softmax's KV cache inflating linearly with $L$

Saying only "faster," unable to articulate the concrete mechanism "fixed state → constant memory + $O(1)$ decode."

</details>

<details>

<summary>Q7. The basic idea of sparse attention? The essential difference from linear attention?</summary>

- sparse: **keep softmax**, each query attends only to a subset (window/block/top-k), $L^2\to Ls$
- linear: **give up softmax**, replace attention with a fixed-state recurrence
- difference: sparse still stores all KV ($O(L)$ memory), strong recall; linear has constant state, weak recall

Conflating the two, or thinking sparse can also reduce the KV cache to constant (wrong, sparse still stores the full KV).

</details>

<details>

<summary>Q8. What is a hybrid architecture? Why is it needed?</summary>

- **interleave a few full softmax attention layers** among many linear/SSM layers
- because pure linear/SSM have **weak recall (associative recall / exact copy)**
- a few full attention layers fill recall, the rest of the linear layers keep throughput, getting both ends

Saying only "mix them," unable to articulate the core motivation "the recall–throughput trade-off."

</details>

<details>

<summary>Q9. Does FlashAttention count as linear attention?</summary>

- **No**. FlashAttention is an IO optimization of **exact softmax** (block tiling + online softmax)
- it cuts memory from $O(L^2)$ to $O(L)$, but the **FLOPs are still $O(L^2)$**, no change to the compute paradigm
- linear attention / SSM are the ones that "replace attention with a fixed-state recurrence"

Treating FlashAttention as "linear/sub-quadratic attention" — it is an engineering optimization of exact attention, and its complexity class is unchanged.

</details>

<details>

<summary>Q10. Why might linear attention have worse recall than a Transformer?</summary>

- history is compressed into a **fixed-size** state $S\in\mathbb{R}^{d_k\times d_v}$ with limited capacity
- many $\phi(k)v^\top$ pile up and interfere, so exact retrieval gets "blurry" on long sequences
- softmax can do near one-hot sharp selection, which the finite-dimensional $\phi(q)^\top\phi(k)$ cannot express

Attributing weak recall to "bad implementation / poor tuning" rather than the capacity essence of the fixed state.

</details>

### L2 advanced

<details>

<summary>Q11. Write linear attention's recurrence, and explain the two problems from dropping softmax normalization.</summary>

- recurrence: $S_t=S_{t-1}+\phi(k_t)v_t^\top,\ o_t=\phi(q_t)^\top S_t / (\phi(q_t)^\top z_t)$, $z_t=\sum_{j\le t}\phi(k_j)$
- problem one: the normalization denominator $\phi(q)^\top z_t$ can be very small / near 0 → numerical instability, so many implementations simply drop the denominator + add extra normalization
- problem two: lacks sharp selection ability → weak recall (capacity bottleneck)

Only writing the unnormalized version, unable to answer the two concrete consequences "unstable denominator + weak selection ability."

</details>

<details>

<summary>Q12. Which quantities exactly are data-dependent in Mamba? Is $A$? Why is selectivity mainly via $\Delta$?</summary>

- the data-dependent ones are $B, C, \Delta$; **$A$ is a structured, input-independent fixed parameter** (one learned $A$ per channel)
- $\Delta_t$ is data-dependent and turns the "effective decay rate" into something data-dependent via $\bar A_t=\exp(\Delta_t A)$ — this is the main source of selectivity
- $B_t, C_t$ are data-dependent, controlling "what is written / read out"

Saying "Mamba makes $A$ vary with the input" (wrong); or not knowing that $\Delta$ modulates the decay indirectly via $\exp(\Delta A)$.

</details>

<details>

<summary>Q13. Explain State Space Duality (SSD). Does it say SSM equals softmax attention?</summary>

- expand the selective SSM recurrence into $y=Mx$, where $M$ is a lower-triangular matrix, $M_{ts}=C_t(\prod_{r}\bar A_r)\bar B_s$
- $M$ is a **1-semiseparable matrix** (lower-triangular sub-blocks have rank $\le 1$), equivalent to a kind of **structured masked (linear) attention**
- **not** softmax attention — $M$ has no softmax and is semiseparable-structured; this is an equivalence at the "linear/masked attention" level

Overclaiming "SSM = softmax attention" (wrong, it's structured masked linear attention, with no softmax).

</details>

<details>

<summary>Q14. What engineering benefit does Mamba-2 get from SSD? The implementation difference from Mamba-1?</summary>

- Mamba-1: a hardware-aware selective **scan** (a custom kernel that **doesn't use tensor cores**)
- Mamba-2: uses SSD to rewrite the same computation as **matmul-heavy chunkwise** (materialize the small $M$ intra-chunk, pass a low-rank state inter-chunk), **saturating tensor cores**
- benefit: trains several times faster, the state dim $N$ can be made larger (matmuls are cheap) → stronger expressiveness

Saying only "Mamba-2 is faster," unable to articulate the causal chain "SSD → chunkwise matmul → tensor core."

</details>

<details>

<summary>Q15. Derive the delta rule, and explain the role of the $(I-\beta k k^\top)$ term.</summary>

- view $S$ as the map $k\mapsto Sk$, do one step of online least squares / gradient descent on the error $(S_{t-1}k_t-v_t)$
- get $S_t=S_{t-1}-\beta_t(S_{t-1}k_t-v_t)k_t^\top=S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$
- $(I-\beta kk^\top)$ first **erases the old association in the $k_t$ direction** then writes the new value → "rewrite/overwrite" rather than "accumulate"

Only memorizing the formula, unable to articulate that it's error correction / online least squares, or unable to answer that the projection term = "delete then write."

</details>

<details>

<summary>Q16. The essential difference between DeltaNet and ordinary linear attention? Why does DeltaNet have stronger recall?</summary>

- ordinary linear: $S_t=S_{t-1}+v_tk_t^\top$, pure-additive — the same key recurring causes **wrong accumulation** of the value
- DeltaNet: $S_t=S_{t-1}(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$, **overwrites** the old association
- the rewrite-style update avoids association interference, so associative recall is significantly stronger

Treating DeltaNet as "linear attention with decay added" (wrong, the key is the rewrite-style projection term, not just decay).

</details>

<details>

<summary>Q17. What does Gated DeltaNet add on top of DeltaNet? What do $\alpha$ and $\beta$ each control?</summary>

- adds a data-dependent **decay/forgetting gate** $\alpha_t\in(0,1)$: $S_t=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$
- $\alpha_t$ controls "how much history to forget overall" (continuous fading)
- $\beta_t$ controls "how strongly to rewrite in the $k_t$ direction" (precise overwriting); the two are orthogonal and complementary

Conflating $\alpha$ and $\beta$, or not knowing that gating (forgetting) and delta (rewrite) are two independent mechanisms.

</details>

<details>

<summary>Q18. Explain chunkwise parallel in detail: what do intra-chunk and inter-chunk each do?</summary>

- split into chunks of size $C$, parallelize within a chunk and recur across chunks
- **intra-chunk**: $C\times C$ quadratic causal attention within the chunk ($((Q_cK_c^\top)\odot M)V_c$)
- **inter-chunk**: history compressed into a state $S_c$, this chunk's queries read $Q_cS_c$ once; update $S_{c+1}=S_c+K_c^\top V_c$ at the chunk's end
- for fixed $C$ the total complexity is $O(L)$ and all matmuls (getting both $O(L)$ inference + parallel training)

Saying only "chunked compute," unable to articulate the two stages "intra-chunk quadratic + inter-chunk recurrent state," or not knowing this is the key to parallel training.

</details>

<details>

<summary>Q19. What are NSA's three branches? Why emphasize "natively trainable + hardware-aligned"?</summary>

- three branches: **compress** (distant blocks compressed into coarse tokens) + **select** (top-k most relevant key blocks for fine-grained attention) + **sliding window** (recent tokens to keep local), gated fusion
- natively trainable: sparse structure used from pretraining, end-to-end learnable (not slapping on a mask after the fact)
- hardware-aligned: contiguous blocks, tensor-core-friendly → real wall-clock speedup (not just fewer theoretical FLOPs)

Only remembering "top-k sparse," unable to recite all three branches; or not knowing "hardware-aligned" is for real speedup.

</details>

<details>

<summary>Q20. MoBA and NSA are both trainable sparse — how do their ideas differ and agree?</summary>

- same: both **content-dependently select blocks**, both save over full attention, both hardware-friendly
- different: **MoBA** = an MoE-style router, routing each query to its top-k key blocks (skipping the rest); **NSA** = the fixed fusion of three branches compress+select+window
- MoBA is designed to switch seamlessly with full attention, easing migration of existing models

Saying the two are the same, or not knowing MoBA borrows MoE routing and NSA is a three-branch structure.

</details>

### L3 advanced

<details>

<summary>Q21. RetNet, Mamba, GLA, DeltaNet are all $O(L)$ recurrences — how do they each "forget/write" the state?</summary>

- **RetNet**: $S_t=\gamma S_{t-1}+k_tv_t^\top$, **scalar fixed decay** $\gamma$ (not data-dependent), pure-additive write
- **GLA**: $S_t=\text{diag}(\alpha_t)S_{t-1}+k_tv_t^\top$, **data-dependent diagonal gate** forgetting, pure-additive write
- **Mamba(S6)**: SSM recurrence $h_t=\bar A_t h_{t-1}+\bar B_t x_t$, $\bar A_t=\exp(\Delta_t A)$ data-dependent decay
- **DeltaNet**: $S_t=S_{t-1}(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$, **rewrite-style** write (delete old + write new); the Gated version adds $\alpha_t$ forgetting

Only able to memorize each formula, unable to abstract the unified framework "decay (scalar/diagonal/data-dependent) × write (additive/rewrite)."

</details>

<details>

<summary>Q22. Why can't a selective SSM train with the FFT global convolution like S4? How does Mamba solve it?</summary>

- S4 is LTI, the convolution kernel $\bar C\bar A^k\bar B$ is fixed, can be written as a global convolution computed in $O(L\log L)$ with FFT
- the selective SSM's $\bar A_t,\bar B_t,C_t$ **vary with time** → the convolution kernel is no longer fixed → the FFT convolution breaks
- Mamba's solution: a hardware-aware **selective scan** (parallel prefix scan + kernel fusion, keep the state in SRAM, don't materialize the $L\times N$ intermediate state); Mamba-2 further uses SSD chunkwise matmuls

Unable to answer "time-varying → convolution kernel not fixed → FFT breaks," or not knowing scan/SSD are the alternatives.

</details>

<details>

<summary>Q23. The delta rule's $\prod_t(I-\beta_t k_tk_t^\top)$ looks strongly sequential — how do you train it in parallel over sequence length?</summary>

- the Householder product over $T$ steps within a chunk can be written in the compact $I-WY^\top$ form using the **WY representation** (a classic numerical-linear-algebra QR trick)
- so you batch with matmuls intra-chunk and pass the state inter-chunk (chunkwise)
- this is the key engineering contribution that lets the DeltaNet family train efficiently on long sequences

Saying only "hard to parallelize" or "use chunkwise," unable to answer the concrete mechanism "WY / Householder representation."

</details>

<details>

<summary>Q24. In a hybrid architecture, how many full attention layers should you insert and where? Is a pure hybrid fully constant memory?</summary>

- empirically: a very small full-attention ratio (commonly 1:5 to 1:7, 1 full layer every 6–8 layers) can restore recall to near a pure Transformer
- insertion comes in two kinds, **layer-level interleaving** (some layers are attention) and **head-level parallel** (Hymba: attention heads + SSM heads within the same layer)
- **not fully constant memory**: those few full attention layers still have a KV cache that grows with $L$, so overall it's "most layers constant + a few layers linear"

Saying "hybrid has no KV cache / fully constant memory at all" (wrong, full attention layers still have $O(L)$ KV).

</details>

<details>

<summary>Q25. Given a 200K long-context + high-concurrency serving scenario, how would you trade off among softmax / linear-SSM / sparse / hybrid?</summary>

- pure softmax: best recall but the KV cache explodes with $L$ (200K × high concurrency → memory/bandwidth infeasible)
- pure linear/SSM: constant state, best throughput and memory, but recall (needle-in-a-haystack / multi-hop) may drop points
- sparse (NSA/DSA): keeps softmax recall, saves compute/access, but still stores the full KV (no memory saving)
- **hybrid (recommended)**: most layers linear/SSM to compress KV, a few full attention (with MLA/GQA + optionally stacking sparse) to keep recall — the mainstream answer for 2025 industrial long-context; the final choice depends on the strength of the recall requirement, the memory budget, and the acceptable quality loss, validated with a real long-context benchmark

Reporting only a single option without trading off; or ignoring key constraints like "sparse doesn't save KV memory" and "a hybrid still has some linear KV."

</details>

## §A Appendix: sanity check

This tutorial's code (§6's `linear_attn_recurrent` / `linear_attn_chunkwise`, plus the delta-rule and block-sparse snippets below) should satisfy the following key invariants. The full runnable script is at [`code/linear_sparse_attention.py`](code/linear_sparse_attention.py) (pure PyTorch, a few seconds on CPU, 6 asserts) — **the real run output is attached at the end of this section**.

1. **chunkwise == recurrent (the core equivalence)**: the output of `linear_attn_chunkwise(Q,K,V,C)` should match `linear_attn_recurrent(Q,K,V)` within floating-point error (`atol≈1e-5`) for any valid chunk size $C$ (e.g. 1/4/7/the full length $L$). This verifies that "intra-chunk quadratic + inter-chunk recurrence" really is equivalent to the token-by-token recurrence — the cornerstone of chunkwise correctness.

2. **chunk-size invariance**: sweeping $C$ from 1 to $L$, the output is unchanged (only the computation path changes). $C=1$ degenerates to the pure recurrence, $C=L$ degenerates to single-block quadratic attention, and any $C$ in between should match.

3. **linear attention state shape**: the recurrent state $S$ is always an $\mathbb{R}^{d_k\times d_v}$ matrix (in this demo $d_k=d_v=d$), independent of sequence length $L$; one decode step only updates/reads this fixed-size matrix.

4. **the delta rule degenerates to additive linear attention**: in the delta-rule recurrence $S_t=S_{t-1}(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$, if you **drop the rewrite term** (i.e. replace $(I-\beta k_tk_t^\top)$ with $I$) and set $\beta=1$, you fall back to ordinary linear attention $S_t=S_{t-1}+v_tk_t^\top$. This verifies that "the delta rule's soul is precisely that $(I-\beta kk^\top)$ projection term" — pull it out, and DeltaNet collapses back to §2's additive linear attention.

5. **block-sparse mask behavior**: the top-k block-sparse mask below should satisfy — each query row keeps exactly $k$ key blocks (the rest $-\infty$), masked columns have weight 0 after softmax, and each row's weights sum to 1 (normalized within the selected blocks).

Below are two illustrative code snippets, the minimal delta-rule recurrence and the block-sparse top-k mask (runnable on CPU, with shape comments):

```python
import torch
import torch.nn.functional as F

# ---- code snippet 2: minimal delta-rule recurrence ----
def delta_rule_recurrent(K, V, beta, use_delta=True):
    """S_t = S_{t-1}(I - beta k k^T) + beta v k^T.
       K,V: [L, d] (k assumed normalized), beta: [L] write strength -> state S: [d, d].
       use_delta=False with beta=1 falls back to plain linear attention S += v k^T (see §A.4)."""
    L, d = K.shape
    S = torch.zeros(d, d, dtype=K.dtype)          # [d_k, d_v] matrix state
    states = []
    I = torch.eye(d, dtype=K.dtype)
    for t in range(L):
        k = K[t]                                   # [d]
        if use_delta:
            S = S @ (I - beta[t] * torch.outer(k, k))  # erase old association in k direction first
        S = S + beta[t] * torch.outer(V[t], k)     # write new v k^T  (note the v k^T order)
        states.append(S.clone())
    return torch.stack(states)                     # [L, d, d]

# ---- code snippet 3: block-sparse top-k attention mask (the NSA / MoBA "select" idea) ----
def block_topk_attention(Q, K, V, block=4, topk=2):
    """Each query only attends its top-k most relevant key blocks (in-block softmax).
       Q,K,V: [L, d] -> out: [L, d]. block: block size, topk: how many blocks to select."""
    L, d = Q.shape
    nblk = (L + block - 1) // block
    scores = (Q @ K.t()) / (d ** 0.5)              # [Lq, Lk] full scores (used for both select + final attn)
    # block representative score: mean over REAL columns only (ignore padding, so L not divisible by block is also correct) -> [Lq, nblk]
    pad = nblk * block - L
    sc  = F.pad(scores, (0, pad), value=0.0).view(L, nblk, block)         # pad with 0
    cnt = F.pad(torch.ones_like(scores), (0, pad)).view(L, nblk, block)   # real-column count (padding=0)
    blk_score = sc.sum(-1) / cnt.sum(-1).clamp(min=1)     # [Lq, nblk] mean over real columns only
    # select top-k blocks -> block-level keep mask
    topk = min(topk, nblk)
    sel = blk_score.topk(topk, dim=-1).indices             # [Lq, topk]
    keep_blk = torch.zeros(L, nblk, dtype=torch.bool)
    keep_blk.scatter_(1, sel, True)                        # [Lq, nblk] True=keep this block
    # expand back to token-level mask
    keep = keep_blk.repeat_interleave(block, dim=1)[:, :L] # [Lq, Lk] True=keep
    masked = scores.masked_fill(~keep, float("-inf"))
    w = F.softmax(masked, dim=-1)                          # normalized within selected blocks, each row sums to 1
    return w @ V, keep                                     # [L, d], mask

if __name__ == "__main__":
    torch.manual_seed(0)
    L, d = 12, 8
    # delta-rule: dropping the rewrite term + beta=1 should fall back to additive linear attention
    K = F.normalize(torch.randn(L, d), dim=-1)
    V = torch.randn(L, d)
    beta = torch.ones(L)
    S_delta = delta_rule_recurrent(K, V, beta, use_delta=True)[-1]
    S_lin   = delta_rule_recurrent(K, V, beta, use_delta=False)[-1]  # additive version
    # Note: the two are strictly equal iff the erase term S_{t-1}k_t=0 at every step; "all k pairwise-orthogonal (and beta=1)" is the cleanest sufficient condition.
    #       In general the rewrite term S_{t-1}(beta k k^T) is nonzero -> a difference appears (exactly DeltaNet overwriting old associations)
    print("delta vs additive final-state diff:", (S_delta - S_lin).norm().item())
    # block-sparse: each row keeps exactly topk blocks
    Q = torch.randn(L, d)
    out, keep = block_topk_attention(Q, K, V, block=4, topk=2)
    nblk = (L + 3) // 4
    per_row_blocks = keep.view(L, nblk, 4).any(dim=-1).sum(dim=-1)   # blocks kept per row
    print("blocks kept per query (expect=2):", per_row_blocks.tolist())
```

The **real output** of running `python code/linear_sparse_attention.py` (CPU, pure PyTorch):

```text
[a] chunkwise == recurrent, C in [1, 4, 7, 13]: max |Δ| = 3.81e-06  OK
[b] chunk-size invariance over C in [1, 4, 7, 13]: all equal = True  OK
[c] state shape over all 13 steps = {(6, 6)} (expect {(6, 6)}), L-independent  OK
[d] delta(use_delta=False, beta=1) == additive S+=vk^T: |Δ| = 0.00e+00  OK
[e] overwrite term: non-orthogonal ||Δ|| = 8.315e+00 (>0), orthonormal |Δ| = 3.58e-07 (~0)  OK
[f] L=12: blocks kept/query = [2] (expect [2]), rows sum to 1 = True  OK
[f] L=13: blocks kept/query = [2] (expect [2]), rows sum to 1 = True  OK

all linear / sparse attention sanity checks passed ✓
```

> ✅ **Reading the numbers**
> - **[a]** chunkwise matches the token-by-token recurrence within floating-point error (max $|\Delta|\approx 3.8\mathrm{e}{-6}$) — the golden validation of chunkwise correctness, holding for $C=1/4/7/L$.
> - **[d]** dropping the rewrite term $(I-\beta kk^\top)$ with $\beta=1$ **strictly** falls back to additive linear attention ($|\Delta|=0$, §A.4).
> - **[e]** the rewrite term really is acting: under non-orthogonal keys, delta and additive differ by $8.3$ (nonzero), and under orthogonal keys + $\beta=1$ they are strictly equal ($3.6\mathrm{e}{-7}\approx0$) — confirming "they are equal only when $S_{t-1}k_t=0$ each step, and pairwise orthogonality is a sufficient condition."
> - **[f]** block-sparse keeps exactly $\text{topk}=2$ blocks per query and has softmax row sums of 1 whether $L$ divides (12) or **does not divide** (13) the block (the masked block-mean makes non-divisible lengths correct too).

## 📚 References

- **Linear Attention / Transformers are RNNs** — Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, arXiv 2006.16236 (2020), ICML 2020.
- **Performer (FAVOR+)** — Choromanski et al., *Rethinking Attention with Performers*, arXiv 2009.14794 (2020), ICLR 2021.
- **RetNet** — Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, arXiv 2307.08621 (2023).
- **GLA (Gated Linear Attention)** — Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*, arXiv 2312.06635 (2023), ICML 2024.
- **Mamba** — Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv 2312.00752 (2023), COLM 2024.
- **Mamba-2 / SSD** — Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, arXiv 2405.21060 (2024), ICML 2024.
- **DeltaNet** — Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, arXiv 2406.06484 (2024), NeurIPS 2024.
- **Gated DeltaNet** — Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv 2412.06464 (2024), ICLR 2025.
- **NSA (Native Sparse Attention)** — Yuan et al. (DeepSeek), *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, arXiv 2502.11089 (2025).
- **MoBA (Mixture of Block Attention)** — Lu et al. (Moonshot AI), *MoBA: Mixture of Block Attention for Long-Context LLMs*, arXiv 2502.13189 (2025).
- **Lightning Attention / MiniMax-01** — MiniMax, *MiniMax-01: Scaling Foundation Models with Lightning Attention*, arXiv 2501.08313 (2025).
- **Jamba** — Lieber et al., *Jamba: A Hybrid Transformer-Mamba Language Model*, arXiv 2403.19887 (2024).
- **Hymba** — Dong et al., *Hymba: A Hybrid-head Architecture for Small Language Models*, arXiv 2411.13676 (2024).
- **Kimi-Linear** — Kimi Team, *Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv 2510.26692 (2025).
- **HiPPO** — Gu et al., *HiPPO: Recurrent Memory with Optimal Polynomial Projections*, arXiv 2008.07669 (2020), NeurIPS 2020.
- **S4** — Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces*, arXiv 2111.00396 (2021), ICLR 2022.
- **DeepSeek-V3.2 / DSA** — DeepSeek-AI, *DeepSeek-V3.2* technical report / model card — DeepSeek Sparse Attention (DSA, lightning indexer), 2025 (per the official technical report / model card; no standalone arXiv paper).
- **Zamba** — Glorioso et al., *Zamba: A Compact 7B SSM Hybrid Model*, arXiv 2405.16712 (2024).
- **Qwen3-Next** — Qwen Team, *Qwen3-Next* (Gated DeltaNet + gated full attention hybrid), 2025 (technical blog / model report; no standalone arXiv paper).
- **Longformer** — Beltagy et al., *Longformer: The Long-Document Transformer*, arXiv 2004.05150 (2020).
- **BigBird** — Zaheer et al., *Big Bird: Transformers for Longer Sequences*, arXiv 2007.14062 (2020), NeurIPS 2020.
