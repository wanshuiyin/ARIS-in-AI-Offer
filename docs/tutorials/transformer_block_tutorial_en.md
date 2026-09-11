## §0 The Basic Parts of a Block

A Transformer block organizes attention and FFN with residual connections. In the serial structure the FFN reads the residual stream after the attention update; in the parallel structure the two have no ordering dependency. The decoder layer of the original encoder-decoder also contains a cross-attention. Normalization, KV sharing, and model structure each have their own choices.

- **A block's definition contains only its own sublayers**: final norm, local/global interleaving, and dense/MoE layer placement are decided by the stack-assembly code and belong to no block.
- **GPT-2 is already Pre-LN**: from GPT-2-style to Llama-style, what actually changed is the normalization type, the positional encoding's owning level, the FFN structure, and the KV-head configuration.
- **MHA, MQA, GQA, and MLA are coexisting design points**: the historical order is MHA→MQA→GQA, the compression-strength order is MHA→GQA→MQA, and MLA is an independent low-rank latent branch.
- **The KV cache mainly tracks $n_\text{kv}$, not $n_q$**: MQA/GQA do not reduce the Q projection's compute.
- **Dense FFN and MoE are two choices on the capacity axis**: not a "next generation inevitably replaces the previous" ordering.
- **The only structural difference between cross-attn and self-attn is where Q/K/V come from**: everything else — scaled dot-product, multi-head split, softmax — is identical.

## §1 A Modern Block: Forward, Parameters, and Code

### 1.1　Assemble a Llama-3-like block, then swap in MoE

A concrete, complete, runnable modern decoder-only block looks like this:

1. **Normalization**: Pre-RMSNorm, twice — once before attention, once before the FFN.
2. **Attention**: causal self-attn, RoPE applied internally to Q/K, $n_q$ query heads grouped to share $n_\text{kv}$ KV heads — that is GQA, degenerating to MHA when $n_\text{kv}=n_q$; K is cached after rotation, V is never rotated, no bias.
3. **FFN**: SwiGLU gated FFN, $h \approx 8d/3$ rounded to a hardware-friendly multiple, no bias.
4. **Single-block assembly**: $x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x))$, $x=x+\text{SwiGLU}(\text{RMSNorm}(x))$ — this is the complete definition of **one block**; repeat it $N$ times to get the whole decoder stack.
5. **Stack-level closure**: after all $N$ blocks, apply one **final RMSNorm** before the LM head: $x_\text{out} = \text{RMSNorm}\big(\text{Block}_N(\cdots\text{Block}_1(x_0))\big)$. The final norm belongs to stack assembly, not to any block in steps 1–4.

Strictly this is a "Llama-3-like core block": exactly reproducing a given checkpoint also requires matching head count, KV-head count, intermediate size, RoPE base, bias settings, and dtype behavior. Compute the parameter count once with the teaching sizes $d=96$, $n_q=8$, $n_\text{kv}=2$, gated hidden $=256$: the GQA attention part has $2d^2(1+n_\text{kv}/n_q)=23{,}040$ matrix parameters, the SwiGLU FFN part has $3dh=73{,}728$ matrix parameters, and each of the two RMSNorms has only $d=96$ per-channel scale parameters. A single block totals about $23{,}040+73{,}728=96{,}768$ matrix parameters, with the FFN taking the vast majority. This is also why the FFN hidden size affects total parameters more than the KV-head count.

**Swap the FFN for MoE and touch nothing else**:

$$x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x)), \qquad x=x+\text{MoEFFN}(\text{RMSNorm}(x))$$

`MoEFFN`'s external **main-activation tensor** interface is still `[B,T,d] -> [B,T,d]`, and not a single line of `ModernDecoderBlock` changes; §1.4 [D09] verifies this directly. What changes are two other things: training usually returns extra router/load-balancing quantities, and distributed deployment usually introduces expert parallelism.

> ✅ **Which fields describe a block's recipe** — normalization topology, i.e. Post/Pre-serial/Pre-parallel/branch pre+post; normalization type, LayerNorm or RMSNorm; positional-encoding scheme and its owning level, learned absolute outside the stack and RoPE inside attention; attention variant MHA/MQA/GQA/MLA, local window or not, QK-Norm or not, attention scale/temperature; FFN variant dense/gated/MoE and intermediate size; bias, dropout, norm epsilon, residual scaling such as LayerScale. This covers the coarse-grained fields common interviews need; finer-grained values such as head dim, RoPE base, and the exact activation implementation are not on this list.

### 1.2　End-to-end forward: the complete data flow of a modern decoder-only block

Take the teaching sizes $d=96$, $n_q=8$, $n_\text{kv}=2$, gated hidden $=256$. The complete forward data flow of a modern decoder-only block is:

$$x \in [B,T,d] \;\xrightarrow{\text{RMSNorm}}\; h_1 \;\xrightarrow{\text{Q/K/V projection + reshape}}\; Q\in[B,n_q,T,d_h],\;K,V\in[B,n_\text{kv},T,d_h] \;\xrightarrow{\text{RoPE}}\; Q',K' \;\xrightarrow{\text{GQA attention}}\; O\in[B,T,d] \;\xrightarrow{\text{residual add}}\; u = x + O$$

$$u \;\xrightarrow{\text{RMSNorm}}\; h_2 \;\xrightarrow{\text{gate/up projection}}\; \text{SiLU}(\text{gate}(h_2))\odot\text{up}(h_2) \;\xrightarrow{\text{down projection}}\; F\in[B,T,d] \;\xrightarrow{\text{residual add}}\; y = u + F$$

Repeat this block $N$ times and then apply one stack-level final RMSNorm to get the full decoder-only backbone: $y_\text{final} = \text{RMSNorm}\big(\text{Block}_N(\cdots(\text{Block}_1(x_0)))\big)$.

This data flow takes different concrete forms under three execution modes:

| Execution mode | Q/K/V lengths | mask | KV cache | Typical scenario |
| --- | --- | --- | --- | --- |
| Training / full-forward | Q, K, V all span the full $T$ | one-shot lower-triangular causal mask $[T,T]$ | not used | training, teacher-forcing evaluation, scoring a full prompt in one pass |
| Prefill (first inference step) | Q, K, V all span the full prompt length $S$ | causal mask $[S,S]$ | written to cache (K and V, $n_\text{kv}$ heads each, length $S$) | first forward after receiving the user prompt, building the cache for subsequent decode |
| One-token decode (step-by-step inference) | Q has length $1$ (new token); K, V grow from cached history length $t$ to $t{+}1$ | with a dynamic cache and no padding/packing, the new token sees all of $0..t$ (whole row True); static cache/left padding/packed sequences still need extra masking (see §9 Q25) | read old cache + append new K/V | every step of autoregressive generation |

The three modes share the same RMSNorm→RoPE→attention→residual code path, differing only in the actual Q/K/V lengths, the mask shape, and whether the cache is read/written. That is why §1.4 [D04] can use one implementation to verify that full-forward and cached decoding agree numerically; if the two paths used different code, that test would lose its cross-validation meaning.

### 1.3　Parameter-count and KV-cache formula accounting

**FFN parameters**, ignoring bias: dense (ReLU/GELU) is $8d^2$ over two matrices; gated (SwiGLU/GeGLU) is $3dh$ over three matrices. Setting them equal gives $h=8d/3$ — a **common default computed from parameter matching**, not an architectural law. Real Llama-family implementations usually start from $4d$, multiply by $2/3$, then round to hardware-friendly multiples (with an optional multiplier), so the intermediate size can deviate noticeably from $8d/3$.

**Attention parameters**, ignoring bias: MHA's four Q/K/V/O projections total $4d^2$; GQA gives $2d^2(1+n_\text{kv}/n_q)$, degenerating back to $4d^2$ at $n_\text{kv}=n_q$ and reaching MQA's minimum at $n_\text{kv}=1$. Both formulas assume $d=n_qd_h$, identical head dim for Q/K/V, a $d\times d$ output projection, and no bias — adjust them accordingly when these assumptions fail.

**Packed projection**: fusing the three Q/K/V Linears, or SwiGLU's gate/up Linears, into one big Linear is a kernel-level equivalent rearrangement — with row-wise concatenated weights the results should agree at the corresponding precision, changing no architectural semantics.

**KV cache**: the table below gives **element counts** per token per layer, not bytes. Full memory usage further multiplies by batch size $B$, sequence length $L$, layer count $N_\text{layer}$, and bytes per element — usually 2 bytes for fp16/bf16, 1 byte for fp8.

| Variant | Cache size (elements/token/layer) |
| --- | --- |
| MHA | $2 n_q d_h$ |
| GQA | $2 n_\text{kv} d_h$ |
| MQA | $2 d_h$ |
| MLA | $d_c + d_h^R$ (latent dim + shared RoPE component; no "$\times 2$" because K/V share one latent) |

### 1.4　From-scratch implementation and executable verification: `code/transformer_block.py`

The full runnable script is [`code/transformer_block.py`](code/transformer_block.py) (pure PyTorch; all 9 demos [D01]–[D09] finish in seconds on CPU). Four entry classes correspond to this tutorial's four "block eras": `VanillaEncoderLayer2017`, `VanillaSeq2SeqDecoderLayer2017` (§2), `GPT2StyleDecoderLayer` (§3), `ModernDecoderBlock` (§4/§1.1, with `ffn=` swappable among `DenseFFN`/`GatedFFN`/`MoEFFN`). The body shows the three most central snippets; the remaining demos are standalone functions in the script.

**[D02] Residual-topology test** (zero out the sublayer output: Pre-LN must return exactly $x$, Post-LN must return $N(x) \ne x$):

```python
def zero_sublayer(_x):
    return torch.zeros_like(_x)

post_ln_out = ln(x + zero_sublayer(x))       # y = N(x + F(x)) -> y = N(x)
pre_ln_out  = x + zero_sublayer(ln(x))       # y = x + F(N(x)) -> y = x

assert torch.allclose(pre_ln_out, x, atol=1e-6)          # Pre-LN: exact identity
assert torch.allclose(post_ln_out, ln(x), atol=1e-6)     # Post-LN: exactly N(x)
assert not torch.allclose(post_ln_out, x, atol=1e-4)     # and generally != x
```

**[D04] Full forward vs cached decoding** (under eval() with no dropout, the full-sequence forward and token-by-token incremental decoding must agree numerically — jointly checking the causal mask, RoPE position offset, and cache append):

```python
causal = build_causal_mask(torch.arange(T), torch.arange(T))
out_full = block(x, cos, sin, mask=causal)                  # compute the whole sequence at once

cache = KVCache()
outs = []
for t in range(T):
    q_pos, k_pos = torch.tensor([t]), torch.arange(t + 1)
    mask_t = build_causal_mask(q_pos, k_pos)                 # new token sees all of 0..t
    outs.append(block(x[:, t:t+1, :], cos, sin, mask=mask_t,
                       kv_cache=cache, start_pos=t))
out_cached = torch.cat(outs, dim=1)
assert torch.allclose(out_full, out_cached, atol=1e-4)
assert cache.k.shape[1] == n_kv_heads                        # cache head count never grows to n_q_heads
```

**[D08] Parallel vs sequential dependency** (use a forward hook to capture the FFN's actual input tensor and **exactly match** it against an independently recomputed expected tensor — not the fragile "two random numbers are unequal"):

```python
captured = {}
seq_blk.ffn.register_forward_hook(lambda m, i, o: captured.setdefault("x", i[0]))
seq_blk(x, causal)
a, _ = seq_blk.attn(seq_blk.norm1(x), mask=causal)
assert torch.allclose(captured["x"], seq_blk.norm2(x + a), atol=1e-5)   # sequential: FFN sees the "updated" stream

captured.clear()
par_blk.ffn.register_forward_hook(lambda m, i, o: captured.setdefault("x", i[0]))
par_blk(x, causal)
assert torch.allclose(captured["x"], par_blk.norm(x), atol=1e-6)       # parallel: FFN sees the "original" stream
```

**Key invariants the script must satisfy** (one per [D01]–[D09], described here by "verification intent"; the exact assertions are as implemented in the script):

1. **[D01] Shape contract**: all four entry classes preserve the `[B,T,d]` residual width; cross-attn still correctly returns length-$T$ output when $T\ne S$, and must verify that memory is actually read (changing memory changes the output), not just the shape.
2. **[D02] Residual-topology test**: with sublayer outputs zeroed, Pre-LN-style blocks return exactly $x$; Post-LN-style blocks return $N(x)$, generally not equal to $x$; the two sublayers' LNs must be compared separately — one $N(x)$ must not be used to certify itself.
3. **[D03] No causal leakage**: perturbing only tokens after position $t{+}1$ must leave outputs at positions $0..t$ untouched; confirm this invariant is actually contributed by the attention branch.
4. **[D04] Full forward vs cached decoding**: under eval() with no dropout, the full-sequence forward and token-by-token incremental decoding agree numerically at all positions, and the cache head dimension stays $n_\text{kv}$ throughout — the most direct way to hunt causal-mask / RoPE-offset / cache-ordering bugs, though not the only test that must cover a nonzero attention-branch contribution.
5. **[D05] MHA/MQA/GQA degeneration**: at $n_\text{kv}=n_q$ the general implementation matches an independently hand-written plain-MHA reference exactly; $n_\text{kv}=1$ (MQA) and intermediate values (e.g. $n_\text{kv}=2$) must each be compared against an independent `repeat_interleave` reference — testing only the $n_\text{kv}=n_q$ degenerate case is not enough.
6. **[D06] Parameter formulas**: at $d=96$, dense FFN (hidden $=4d$) and gated FFN (hidden $=8d/3$) both have exactly 73,728 matrix parameters; GQA/MHA attention matrix parameters equal the $2d^2(1+n_\text{kv}/n_q)$ and $4d^2$ formulas exactly; illegal configurations where $d_\text{model}$ is not divisible by the head count must fail at construction time.
7. **[D07] Packed-projection numerical equivalence**: packed QKV (and packed gate+up) with identical weights match three (or two) separate Linears numerically (numerically equivalent within tolerance, not necessarily bitwise identical).
8. **[D08] Parallel vs sequential**: forward hooks capture both attention's and the FFN's actual inputs; the sequential block matches $N_2(x+\text{Attn}(N_1(x)))$ exactly, and both branches of the parallel block match $N(x)$ exactly.
9. **[D09] Backward smoke test**: all dense-block parameter gradients are finite; the MoE-swapped block must iterate over all experts not unused in this batch and check nonzero gradients, while router/attention/norm and all other parameters also receive gradients normally.

The **actual output** of running `python3 code/transformer_block.py` (D10-D15 are targeted regression tests, covering constructor argument validation, KV-cache consistency, cross-attention combination restrictions, mask safety and broadcasting, default causal behavior, and RoPE-cache device/dtype handling):

```text
[D01] shapes (enc/dec2017/gpt2/modern) = [torch.Size([2, 7, 96]), torch.Size([2, 7, 96]), torch.Size([2, 7, 96]), torch.Size([2, 7, 96])] (expect 4x (2,7,96)); cross-attn ran with Tq=7 != S=5; changing memory changes cross-attn output (memory is actually read, not ignored) = True  PASS
[D02] formula-level: Pre-LN(F=0)==x: True; Post-LN(F=0)==N(x): True (and != x: True); class-level: zeroed ModernDecoderBlock==x: True, zeroed Post-LN encoder layer != x: True, == ln2(ln1(x)) exactly: True  PASS
[D03] perturbing tokens after t=3: positions 0..3 unchanged = True, positions 4..5 changed = True; perturbing position 0 changes positions 1..5 (attention actually mixes forward, not a dead branch) = True  PASS
[D04] full-forward vs token-by-token cached decode: max|Δ| = 4.77e-07 (tol 1e-04, numerically equivalent, NOT bit-identical); cache CONTENT vs independently projected+rotated K/V: max|ΔK|=3.58e-07, max|ΔV|=4.77e-07; final cache shape head_dim=2 (expect 2) len=6 (expect 6)  PASS
[D05] GQA(n_kv=n_q) == independent reference MHA: max|Δ|=0.00e+00; GQA(n_kv=2, group=4) == independent repeat_interleave reference: max|Δ|=8.94e-08; MQA == same reference (n_kv=1): max|Δ|=8.94e-08, cache head_dim=1 (expect 1), output shape=(2, 5, 96), per-query-head outputs distinct (not collapsed) = True  PASS
[D06] FFN params: dense(4d)=73728, gated(8d/3)=73728 (both expect 73728); attn params: GQA=23040 (formula 2d^2(1+n_kv/n_q)=23040), MHA=36864 (formula 4d^2=36864)  PASS
[D07] packed QKV vs separate: max|Δ|=4.77e-07; packed gate+up vs separate: max|Δ|=0.00e+00  PASS
[D08] sequential: attn input == norm1(x): True, FFN input == norm2(x+attn_out) (updated stream): True (and != norm2(x) alone: True); parallel: attn input == norm(x): True, FFN input == norm(x): True, attn and FFN inputs are the literal SAME tensor: True  PASS
[D09] dense block: all grads finite = True, all non-zero = True; MoE tokens_per_expert=[1, 0, 2, 0] (<4 tokens, so >=1 expert unused by pigeonhole); touched experts non-zero+finite grad, untouched experts no/zero grad: True; router gate non-zero+finite grad: True; rest-of-block (attn+norms) non-zero+finite grad: True  PASS
[D10] d_model % n_q_heads != 0 raises ValueError: True; n_q_heads % n_kv_heads != 0 raises ValueError: True; RoPE + odd head_dim raises ValueError: True; MoEFFN(top_k=2) raises NotImplementedError (no silent top-1 downgrade): True  PASS
[D11] inconsistent start_pos vs non-empty cache raises ValueError: True; consistent start_pos does not raise: True; cache.reset() clears state: True; append() with mismatched batch/shape raises ValueError: True  PASS
[D12] cross-attention + RoPE raises ValueError: True; cross-attention + KV cache raises ValueError: True; plain cross-attention (no RoPE/cache) still works, shape=(1, 4, 32): True  PASS
[D13] fully-masked query row produces finite output (no NaN): True; key_padding_mask correctly isolated per-batch-sample when B==n_kv_heads (2): max|Δ| vs per-sample reference=1.79e-07 -> True; same check when B==Tq (3): max|Δ|=1.64e-07 -> True  PASS
[D14] GPT2StyleDecoderLayer: omitted causal_mask == explicit causal_mask: True, default is genuinely causal (no future->past leakage): True; ModernDecoderBlock: omitted causal_mask == explicit causal_mask: True  PASS
[D15] build_rope_cache(dtype=float64) produces float64 cos/sin: True; apply_rope auto-casts a mismatched (float32) cache to a float64 input's dtype (no crash, correct output dtype): True; matches a natively-float64-built cache: True (max|Δ|=0.00e+00)  PASS

all transformer block sanity checks passed ✓
```

## §2 The Original Encoder-Decoder

The original Transformer (Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762, 2017, NeurIPS 2017) is an **encoder-decoder** architecture: the encoder encodes the source sequence into a set of representations, and the decoder autoregressively generates the target sequence conditioned on them. Positional encoding uses fixed sinusoidal PE, added directly to the token embeddings **before** entering the encoder/decoder stacks; it belongs to no block and is input processing outside the blocks. The paper's original configuration is $d_\text{model}=512$, $d_\text{ff}=2048$ (4× expansion), 8 heads, 6 encoder + 6 decoder layers; this tutorial demonstrates with smaller sizes such as $d=96$, which is a **teaching configuration**, not the paper's — do not conflate the two.

The encoder layer and the decoder layer differ in both the number and the kinds of sublayers:

| | Encoder layer | Decoder layer |
| --- | --- | --- |
| Sublayer 1 | Self-attn (bidirectional, no causal mask, only an optional padding mask) | Masked self-attn (causal) |
| Sublayer 2 | FFN | Cross-attn (Q from this layer's output, length $T$; K/V from encoder output, length $S$) |
| Sublayer 3 | — | FFN |
| Norm topology | Post-LN per sublayer: $y=N(x+F(x))$ | Post-LN per sublayer, three times |

The decoder's three sublayers run in the fixed order masked self-attn → cross-attn → FFN, each wrapped in its own Post-LN. Cross-attn's Q, K, V sources are asymmetric: Q comes from the decoder's own in-progress stream, length $T$; K and V come from the encoder output, length $S$, and $T$ and $S$ can differ.

**Walking one concrete shape through**, with teaching sizes batch $B=2$, source length $S=5$, target length $T=7$, $d=96$: the encoder input is `[B, S, d]` and remains `[B, S, d]` after several encoder layers, passed as memory to every decoder layer; the decoder input is `[B, T, d]`, and in the masked self-attn stage Q/K/V are all computed internally from `[B, T, d]`, with a causal mask of size $T\times T$; in the cross-attn stage Q is still `[B, T, d]` but K, V come from the encoder memory with shape `[B, S, d]`, so the attention score matrix is $T\times S$ rather than square, with softmax along the $S$ dimension; after the output projection it returns to `[B, T, d]`, matching the self-attn sublayer's output shape, so it can proceed into the FFN sublayer. **$T \ne S$ does not affect this data path at all** — this is what "cross-attn decouples Q length from K/V length" concretely looks like.

`VanillaEncoderLayer2017` (2 sublayers) and `VanillaSeq2SeqDecoderLayer2017` (3 sublayers) are two independent classes in §1's code. Writing them as one "generic block class + parameter switches" is workable engineering but hides the structural fact that the decoder has a whole extra cross-attn sublayer compared with the encoder.

## §3 The Three Architecture Families and GPT-2→Llama

### 3.0　encoder-only / decoder-only / encoder-decoder

| Family | Representatives | Sublayer composition | Attention direction | Typical training objective | Typical use |
| --- | --- | --- | --- | --- | --- |
| Encoder-only | BERT, RoBERTa | per layer: self-attn + FFN (no causal mask) | bidirectional (sees the whole sequence) | denoising objectives such as masked language modeling (MLM) | representation learning, classification, retrieval embeddings |
| Encoder-decoder | original Transformer, T5 | encoder layer: self-attn + FFN; decoder layer: masked self-attn + cross-attn + FFN | encoder bidirectional; decoder unidirectional and conditioned on the encoder | conditional generation (translation, summarization) | two-sequence conditional generation tasks |
| Decoder-only | GPT, Llama, Qwen | per layer: causal self-attn + FFN (no cross-attn) | unidirectional (causal) | autoregressive language modeling | general-purpose generative LLMs |

The core difference among the three is not "how many parameters" but that attention visibility, sublayer composition, and training objective change together as a bundle.

### 3.1　How decoder-only differs from the original decoder layer

**At the single-layer structural level**, a decoder-only block really is the original decoder layer with cross-attn removed: masked self-attn → cross-attn → FFN, **three** sublayers, becomes causal self-attn → FFN, **two** sublayers.

**At the whole-model level** an entire encoder/memory path is missing as well: source/prompt and target go from "two-stream conditional generation" to "single-stream causal modeling", with prompt and generated content concatenated into one causal sequence. What changes is the information-flow topology and the sequence factorization, not that the task was replaced by a different problem.

Some multimodal/retrieval-augmented models reintroduce cross-attention on a decoder-only backbone; every "decoder-only has no cross-attn" statement in this tutorial is restricted to the **pure decoder-only block**, i.e. the two sublayers causal self-attn + FFN.

### 3.2　GPT-2-style to Llama-style

GPT-2 (Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report, 2019, no arXiv id) is already Pre-LN, and puts a final LayerNorm, denoted `ln_f`, at the end of the stack. The complete GPT-2 block recipe:

$$x = x + \text{Attn}\big(\text{LN}(x)\big), \qquad x = x + \text{FFN}\big(\text{LN}(x)\big), \qquad \text{(GELU dense FFN, MHA, bias=True, no cross-attn)}$$

The learned absolute positional embedding is added to the embedding and sits **outside** the stack, the same as the original sinusoidal PE; neither lives inside a block. RoPE, adopted by the LLaMA family, is the opposite: it lives **inside attention**, acting only on the Q/K dot product. RoPE was first proposed by Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2104.09864, 2021; LLaMA is a prominent adopter, not the proposer. Positional information is carried by an external embedding in GPT-2-style and by rotation inside attention in Llama-style; the two belong to different levels, so this is not a "seamless swap" between members of the same block. What actually changed is this:

| Dimension | GPT-2-style | Llama-style |
| --- | --- | --- |
| Norm topology | Pre-LN (already) | Pre-LN (unchanged) |
| Norm type | LayerNorm | RMSNorm |
| Positional encoding | learned absolute (outside the stack, added to the embedding) | RoPE (inside attention, acting only on Q/K) |
| FFN | GELU dense (2 matrices) | SwiGLU gated (3 matrices) |
| Attention | MHA | MHA or GQA (per the specific checkpoint's spec) |
| bias | bias in both Linear and LayerNorm | usually all bias removed |
| dropout | used in training | usually removed or made tiny in large-scale pretraining |
| final norm | yes (`ln_f`, LayerNorm) | yes (RMSNorm) |

## §4 RMSNorm, RoPE, SwiGLU, and Model Composition

Llama-style — the recipe of Llama 2 and Llama 3 — fills each component with the concrete implementations below; attention is written as MHA first:

$$x = x + \text{Attn}_{\text{RoPE}}\big(\text{RMSNorm}(x)\big), \qquad x = x + \text{SwiGLU}\big(\text{RMSNorm}(x)\big)$$

- **RMSNorm**: re-scale only, no mean-centering, no bias: $y = \dfrac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\odot w$. It often reaches quality comparable to LayerNorm.
- **RoPE**: acts on Q and K inside attention; V is never rotated. Standard implementations usually cache the **already-rotated** K, and incremental decoding rotates new tokens at their true positions. The unrotated-K-cache variant is in §9 Q12.
- **SwiGLU**: $\text{down}\big(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)\big)$, a three-matrix gated FFN; GeGLU is the same structure with a GELU gate (Shazeer, *GLU Variants Improve Transformer*, arXiv 2002.05202, 2020). At similar budgets gated FFNs often deliver better quality — an empirical regularity, not a law.
- **bias, dropout**: Q/K/V/O and the three FFN matrices usually drop all bias; large-scale pretraining commonly removes dropout or reduces it drastically.
- **final norm**: one RMSNorm at the end of the stack, closing off the inflated residual stream. Under the independent-increment approximation the residual stream's **variance** grows roughly as $O(l)$; it is the standard deviation/RMS that goes as $\sqrt l$. Feeding the output head without this closure usually degrades training quality noticeably. It is a common configuration of the standard recipe, not mathematically indispensable.
- **Numerical-precision boundary**: computations sensitive to numeric range — RMS/variance reductions, softmax, final logits — are commonly done at higher precision such as fp32 and cast back to the bf16/fp16 activation dtype. This is part of assembly correctness, not quantization math.

Where a design decision lives determines who assembles it:

| Level | Typical decisions | Owned by |
| --- | --- | --- |
| Inside the operator | QKV projections, RoPE, QK-Norm, head sharing (MHA/GQA/MQA/MLA), gated FFN | the attention/FFN module itself |
| Single block | serial/parallel dependency graph, norm placement (Pre/Post/branch pre+post), residual add, branch scaling/gating | the block's `forward` |
| Stack scheduling | final norm, local/global interleaving, which layers are dense vs. MoE | the stack-assembly code, not any single block |
| Model shell | token embedding, external absolute PE, weight tying, LM head (incl. soft-cap) | the whole model's input/output layers |
| Execution/adaptation overlay | KV cache, FlashAttention, packed projection, quantization, LoRA | deployment/fine-tuning execution strategy; leaves the four levels above architecturally unchanged |

Weight tying, embedding scaling, the LM head's soft-cap, and the global dropout switch are part of the "complete LM recipe" but live outside the block — the token embedding and LM head belong to no Transformer block:

| Periphery field | One-sentence description |
| --- | --- |
| Weight tying | the input token-embedding matrix and the LM-head projection matrix share one set of parameters, saving $Vd$ parameters |
| Embedding scaling | some implementations multiply the token embedding by a constant related to $\sqrt{d}$ before the first block |
| LM head soft-cap | some models (e.g. the Gemma family) also apply a §7.2-style bounded transform to the final vocabulary logits — mechanically the same tool as the attention-logit soft-cap, applied to a different tensor |
| Global dropout switch | whether dropout is on, and the dropout rate, are usually model-wide training configuration; but the dropout **operation itself** typically executes inside attention weights, FFN outputs, and residual branches — "who configures it" and "where the operator actually runs" are two different things; do not lump them into "a property outside the block" |

None of these four appears in the block classes of these examples: `ModernDecoderBlock`'s `forward` receives only the residual stream `x` and never knows whether embedding scaling happened outside or whether the output head is tied.

## §5 KV Sharing and MLA

Every step of autoregressive generation appends the new token's K, V to the cache and attends over the whole cache; the cache size directly determines how many requests one GPU can serve concurrently and how long a context it can hold. GQA/MQA reduce the KV-head count and simultaneously change the K/V projections' own parameter count and compute: the KV cache mainly affects inference memory/bandwidth, while K/V projection parameters are a fixed cost that also exists in training.

### 5.1　MHA / MQA / GQA: different compression ratios on the same axis

MHA is the baseline. **MQA (Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv 1911.02150, 2019) predates GQA (Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023)**, and GQA is the quality-memory compromise between MHA and MQA. **MLA is DeepSeek's separate low-rank latent-compression branch**, not "reduce $n_\text{kv}$ once more". The four are coexisting design points on the "KV memory" design axis, not a historical ladder every model climbed in order.

| Variant | Q heads | KV heads | Relation |
| --- | --- | --- | --- |
| MHA | $n_q$ | $n_q$ | baseline, $n_\text{kv}=n_q$ |
| MQA | $n_q$ | $1$ | all Q heads share one set of K/V |
| GQA | $n_q$ | $n_\text{kv}$ ($1 \lt n_\text{kv} \lt n_q$, with $n_q \bmod n_\text{kv}=0$) | each group of $n_q/n_\text{kv}$ Q heads shares one set of K/V |

- KV-cache size scales only with $n_\text{kv}$, **independent of $n_q$** — MQA/GQA reduce the K/V projections' own parameters and compute, and also reduce KV-cache memory/bandwidth; but the Q projection's compute does not change with $n_\text{kv}$, so do not flatten this into "only saves memory/bandwidth".
- **Standard implementations usually cache K after RoPE is applied; V is never rotated**; in incremental decoding, the new token's Q and K get RoPE at the true position offset, and the already-rotated K in the historical cache is never recomputed.
- GQA/MQA caches **store only $n_\text{kv}$ heads**: broadcasting K/V to the $n_q$ Q heads may happen only inside the attention computation, via view/reshape/grouped einsum, and the broadcast tensor must never be written back to the cache. §1.4 [D05] turns this into an executable assertion.

With vanilla MHA, LLaMA-2-70B would need about 10 GiB of KV cache per sample at a 4096 context; switching to GQA ($n_\text{kv}=8$) drops it to about 1.25 GiB — one of the important reasons many large models adopt GQA over MHA.

### 5.2　MLA: the low-rank latent branch

MLA (DeepSeek-V2, May 2024, arXiv 2405.04434) compresses along the hidden dimension via **low-rank projection**: each token's K/V is squeezed into a latent vector with $d_c \ll n_q d_h$. It must decouple RoPE into a separately shared small-dimension component to preserve the "absorb the up-projection matrix into the query side" inference speedup. MLA's cache format is "store the latent $c_t^{KV}$ + one shared RoPE key", not "store fewer K/V heads".

GQA compresses the **head count**, MQA is GQA's extreme case ($n_\text{kv}=1$), and MLA compresses **each token's representation dimension**. The three are not three points on one scale but two different compression ideas.

## §6 MoE FFN

MoE replaces the FFN with $N$ experts plus one router, and each token passes through only $k \ll N$ experts: total parameters go up, per-token activated parameters stay flat.

- **The block's external main-activation tensor interface is unchanged**: dense or MoE, it is `[B, T, d] -> [B, T, d]` and the residual add stays in place — the direct reason why swapping `ffn` for `MoEFFN` changes not a line of `ModernDecoderBlock`. But training usually returns extra router/load-balancing auxiliary losses or statistics, and distributed deployment usually introduces expert parallelism and all-to-all communication: an unchanged tensor shape does not mean unchanged training return values or system deployment.
- **What changes is inside the FFN**: a router, expert selection, possibly a shared expert, and training-time auxiliary balancing losses/bias updates are added. **Total parameters and per-token activated parameters must be reported separately.**
- **Dense and MoE are two choices on the capacity axis**: representative models such as Llama 1–3, the original Mistral 7B, and Gemma 1–3 use dense FFN; Mixtral, Llama 4, and DeepSeek-V2/V3 use MoE, trading the engineering complexity of communication/routing/load-balancing for larger sparse parameter capacity.

Mixtral (Jiang et al., *Mixtral of Experts*, arXiv 2401.04088, 2024) activates about 12.9B parameters per token and matches or beats Llama 2 70B on most benchmarks reported in the paper. Total parameters are far above 13B, and per-token compute is close to a dense model in the low tens of billions, but not strictly FLOPs-equivalent. DeepSeek-V3's specific load-balancing mechanism comes from Wang, Gao, Zhao, Sun & Dai (arXiv 2408.15664, 2024).

The full routing formulas, capacity factor, load-balancing loss, and token dropping are in moe_tutorial.md §2–§4.

## §7 Residual Structure, Attention, and Execution

Residual connections, normalization, and branch scaling determine a block's computation structure. Numerical control inside attention adjusts scale, and local or global attention determines the visible range. FlashAttention, quantization, and LoRA serve execution optimization or fine-tuning. Most of these mechanisms can be combined; the concrete configuration is influenced by training stability, kernel support, and compatibility with existing checkpoints.

### 7.1　Residual topology, norm placement, and branch scaling

$A$ and $M$ are the attention and FFN sublayers, and $N_1,N_2,N_A,N_M,N_1^{\text{pre}},N_1^{\text{post}},N_2^{\text{pre}},N_2^{\text{post}}$ are all normalization operators. The four topologies are all written at "whole block" granularity:

$$\text{Post-LN}:\quad u = N_1\big(x + A(x)\big),\quad y = N_2\big(u + M(u)\big)$$

Vaswani et al. 2017's original Transformer uses this one: the normalization wraps the residual sum, repeatedly squeezing the trunk.

$$\text{Pre-LN serial}:\quad u = x + A\big(N_1(x)\big),\quad y = u + M\big(N_2(u)\big)$$

Explicitly adopted at least by GPT-2 and later the common form in modern LLMs: the trunk keeps a pure identity-add path, and the FFN reads the updated stream $u$.

$$\text{Pre-LN parallel}:\quad y = x + A\big(N_A(x)\big) + M\big(N_M(x)\big)$$

GPT-J and PaLM use this one: the ordering dependency is removed, and attention and FFN read the same $x$.

$$\text{branch pre+post}:\quad u = x + N_1^{\text{post}}\big(A(N_1^{\text{pre}}(x))\big),\quad y = u + N_2^{\text{post}}\big(M(N_2^{\text{pre}}(u))\big)$$

Gemma-2/3 use this one: the second normalization wraps only the branch output, and the trunk stays pure addition — unlike Post-LN.

Norm placement and serial/parallel dependency are two dimensions, not one axis; §1.4 [D02]/[D08] gives executable positive and negative proofs of all four topologies. **branch pre+post** means one normalization before and after each sublayer branch, in single-sublayer form $y = x + N_\text{post}\big(F(N_\text{pre}(x))\big)$. Gemma-2 uses this topology together with attention-logit/final-logit soft-capping; Gemma-3 keeps this norm topology but replaces soft-capping with QK-Norm.

**Parallel** is defined as "no intra-layer ordering dependency"; it does not require attn and FFN to share one set of norm parameters: two independent normalization modules also satisfy the definition as long as both read the original $x$. GPT-J (Wang & Komatsuzaki, EleutherAI open-source release, 2021, no formal paper/arXiv id) implements it with a single shared LayerNorm; PaLM (Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311, 2022) continues the same idea.

**Residual/branch scaling** is another independent, optional assembly position: multiply the residual branch by a fixed or learnable scale coefficient. **LayerScale** multiplies each branch output by a per-channel, learnable, small-initialized coefficient; **DeepNorm** multiplies the residual branch by a depth-dependent fixed amplification factor, paired with a specific initialization so deep networks train more stably. Whether to scale, whether the scale is a fixed constant or a learnable parameter, and which branch it acts on are chosen separately from norm placement, and can also be used together with it.

### 7.2　attention scale, QK-Norm, logit soft-cap

Three mechanisms act at different positions on the same computation chain:

- **attention scale**: the standard $1/\sqrt{d_h}$ or a learnable temperature, controlling the overall magnitude of the softmax input — the most basic numerical control, present in almost every implementation.
- **QK-Norm** (Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245, 2020, EMNLP 2020 Findings): **before** the dot product, L2-normalize each head's Q and K separately, and **replace** the standard fixed $1/\sqrt{d_h}$ with a learnable scale/temperature; later models' implementation details may differ.
- **logit soft-capping**: **after** the dot product, apply a bounded transform like $c\tanh(z/c)$ to attention scores or final vocabulary logits, reining in the magnitude of scores/logits already computed.

The three act at different positions and control different quantities, **and cannot be stacked blindly** — enabling them together requires re-checking the overall scale, not simple addition.

### 7.3　local / sliding / global attention

This is the design axis of attention's **connectivity structure / visible range (connectivity pattern)** — not an "attention backend", and separate from positional-encoding schemes. Mistral (Jiang et al., *Mistral 7B*, arXiv 2310.06825, 2023) uses fixed-window sliding-window attention; Gemma-2 alternates local-window and global attention 1:1 across layers; Gemma-3 is closer to a periodic pattern of "several local layers then one global layer", about 5:1, not strictly one local layer then one global layer.

None of these change the block's outer residual interface; they only change "how far attention can see" inside. And the placement of "which layers are local, which are global" is itself a stack-assembly decision, not a single block's private attribute. Full sliding-window/StreamingLLM mechanics are in long_context_rope_yarn_mla_tutorial.md §10.

### 7.4　FlashAttention, quantization, and LoRA

These three mechanisms stack on top of an already-chosen architecture to accelerate it or adapt it to a deployment environment; they are not architectural components on the same level as GQA/MoE requiring a mutually exclusive choice:

- **FlashAttention**: changes only exact attention's computation order, IO scheduling, and memory-access pattern; it does not change attention's mathematical result and defines no new connectivity graph, so it stacks on MHA/GQA/MQA/MLA and any local/global pattern.
- **Quantization**: represents weights/activations in lower-precision numeric formats, stackable with any architectural choice above, at the cost of quantization error.
- **LoRA**: adds a pair of low-rank matrices beside trained weights for parameter-efficient fine-tuning, without changing the block's forward architecture.

## §8 Model Comparison and Common Wrong Answers

### 8.1　Block-recipe comparison table for mainstream models

Conventions: "QK-Norm" refers strictly to explicit per-head Q/K normalization before the dot product (MLA's internal latent-projection normalization does not count); "branch pre+post" means $y=x+N_\text{post}(F(N_\text{pre}(x)))$, not the same as original Post-LN; the Attention column lists only the KV-head sharing scheme, with local/sliding/global patterns in the Notes column; final norm and local/global placement belong to stack assembly, and this table presents each model's overall recipe; **none** of these representative models' canonical recipes uses ALiBi as the primary positional scheme.

| Model | Norm topology | Norm type | Positional scheme | Attention | FFN/experts | Parallel attn+FFN | QK-Norm | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | Pre-LN | RMSNorm | RoPE | MHA for 7B/13B; GQA for 70B | SwiGLU, dense | no | no | do not write the whole Llama-2 family uniformly as GQA |
| Llama-3 | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU, dense | no | no | "Llama-3-like" code still needs the specific checkpoint config for exact reproduction |
| Qwen2/2.5 | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU; dense or MoE depending on checkpoint | no | usually no | text models only; do not write the whole Qwen family uniformly as dense or uniformly as MoE |
| Mistral | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU, dense | no | no | the original Mistral dense recipe; also uses sliding-window attention; Mixtral is a separate MoE recipe |
| DeepSeek-V2 | Pre-LN | RMSNorm | decoupled RoPE within MLA | MLA | SwiGLU-style; first layer dense, then mostly DeepSeekMoE | no | no | MLA has internal latent projection normalization, but that is not headwise QK-Norm |
| DeepSeek-V3 | Pre-LN | RMSNorm | decoupled RoPE within MLA | MLA | SwiGLU-style; first 3 layers dense, then mostly DeepSeekMoE | no | no | block backbone keeps MLA+MoE; training/load-balancing improvements must not be miswritten as a new attention type |
| GPT-J | Parallel Pre-LN | LayerNorm | partial RoPE | MHA | GELU, dense | yes | no | core form $x+A(N(x))+M(N(x))$; GELU falls outside the ReLU/GeGLU/SwiGLU trichotomy |
| PaLM | Parallel Pre-LN | LayerNorm, bias-free recipe | RoPE | MQA | SwiGLU, dense | yes | no | parallel attn/MLP use a common normalized residual input |
| Gemma-1 | Pre-LN | RMSNorm | RoPE | MQA for 2B, MHA for 7B | GeGLU, dense | no | no | do not write the whole Gemma-1 series uniformly as GQA |
| Gemma-2 | branch pre+post | RMSNorm | RoPE | GQA | GeGLU, dense | no | no | 1:1 local/global attention alternation across layers, with attention-logit and final-logit soft-capping; the post norm sits inside the branch, before the residual add |
| Gemma-3 | branch pre+post | RMSNorm | RoPE | 1B: MQA; 4B/12B/27B: GQA (per the checkpoint config's `num_key_value_heads`) | GeGLU, dense | no | yes | local/global attention pattern with roughly a 5:1 period (several local layers then one global layer, not strict 1:1 alternation); QK-Norm is one of the key differences from Gemma-2 |
| GPT-2 | Pre-LN | LayerNorm | learned absolute positional embedding | MHA | GELU, dense | no | no | serial attn→MLP per layer, final LayerNorm at the end of the stack |
| GPT-3 | Pre-LN | LayerNorm | learned absolute positional embedding | MHA | GELU, dense | no | no | continues the GPT-2-style block, with dense and locally banded sparse attention patterns varying across layers |

**Three regularities readable from this table**: (1) "Norm topology" and "norm type" almost always come bound together — Pre-LN with RMSNorm, Parallel Pre-LN with LayerNorm, branch pre+post also with RMSNorm; (2) all four of MHA/MQA/GQA/MLA appear among this same batch of representative models, supporting §5's "coexisting design points, not a historical ladder"; (3) "Parallel attn+FFN" is "yes" only in the GPT-J and PaLM rows — niche but genuinely in use, not an abandoned dead end.

The modern recipe has not fully converged: Pre-RMSNorm + RoPE + gated FFN + inference-friendly GQA is only the more common combination; Post-LN variants, LayerNorm, MHA, MQA, parallel residual, dense FFN, and MoE are all still used in production.

### 8.2　The twelve claims most likely to backfire

High-frequency wrong answers and the more accurate claims:

| Common claim | More accurate claim |
| --- | --- |
| Modern LLMs are what replaced Post-LN with Pre-LN | GPT-2 is already Pre-LN; this step happened earlier (§3.2) |
| Decoder-only is just "the decoder with cross-attn removed" | Correct at the single-layer structural level; at the whole-model level also add — the encoder/memory path disappears too, and two-stream conditional generation becomes single-stream causal modeling (§3.1) |
| The residual stream's variance inflates $\propto\sqrt l$ with depth | Variance $\propto l$ (linear); it is the standard deviation/RMS that goes $\propto\sqrt l$ — under the independent-increment assumption (§4, §9 Q11) |
| A Pre-LN stack must have a final norm or the model cannot train well | A common configuration of the standard recipe, not a mathematical necessity; and it is a stack-level operation, not a block field (§1.1, §9 Q11) |
| MHA→MQA→GQA→MLA is one evolution chain | Timeline MHA→MQA→GQA, compression strength MHA→GQA→MQA; MLA is an independent latent branch — there is more than one ordering dimension (§5, §9 Q27) |
| MLA is just GQA with harsher compression | MLA compresses the hidden dimension and needs decoupled RoPE; GQA compresses the head count/sharing relation — the two have different constraints (§5.2, §9) |
| SwiGLU's hidden size must equal $8d/3$ | A common default computed from parameter matching; real models round to hardware-friendly multiples and deviate noticeably (§4, §1.3) |
| Dense FFN will sooner or later be replaced by MoE | An independent capacity axis, not a mandatory stage; Llama 1–3/original Mistral/Gemma 1–3 remain dense, Mixtral/Llama 4 have gone MoE (§6) |
| "Sandwich norm" and original Post-LN are the same thing | The more accurate name is branch pre+post; the second norm acts only on the branch output, not on the residual sum (§7.1) |
| Pre-LN needs no warmup at all | Pre-LN only removes the hard warmup requirement caused by Post-LN's gradient imbalance; modern Pre-LN large models still commonly use warmup (with separate Adam-related motivations, see normalization_init_tutorial.md §5.2) |
| MQA/GQA reduce the Q projection's compute | They only reduce K/V projection parameters/compute and KV-cache memory/bandwidth; the Q-side head count and compute are unchanged (§5.1) |
| Parallel residual requires attn/FFN to share the same norm parameters | Parallel is defined as "no ordering dependency"; two independent norm modules also qualify as long as both read the original $x$ (§7.1, §1.4 [D08]) |

## §9 30 High-Frequency Interview Questions

Three difficulty tiers; expand each for the answer key (the 10 questions that provide a discrimination action end with a pitfall note). L2/L3 are top-lab deep water (topology forks, assembly-axis independence, system correctness, cross-model recipe comparison).

### L1 Must-Know

<details>

<summary>Q1. How do you write the complete forward for the four topologies: Post-LN, Pre-LN serial, Pre-LN parallel, branch pre+post?</summary>

- Post-LN: $u=N_1(x+A(x))$, $y=N_2(u+M(u))$ — normalization wraps each residual sum
- Pre-LN serial: $u=x+A(N_1(x))$, $y=u+M(N_2(u))$ — the residual trunk is pure identity addition; the FFN reads $u$ (the stream after the attention update)
- Pre-LN parallel (GPT-J/PaLM): $y=x+A(N_A(x))+M(N_M(x))$ — both sublayers read the same $x$, no ordering dependency
- branch pre+post (Gemma-2/3): $u=x+N_1^{\text{post}}(A(N_1^{\text{pre}}(x)))$, $y=u+N_2^{\text{post}}(M(N_2^{\text{pre}}(u)))$ — one norm before and after each branch; the residual trunk stays pure addition
- A Pre-LN stack usually ends with one **stack-level** final norm, which is not part of any block formula above (§0, §1.1)

</details>

<details>

<summary>Q2. What is the fundamental difference between the original Transformer decoder and a decoder-only LLM block?</summary>

- The original decoder has causal self-attn, cross-attn, and FFN per layer; pure decoder-only deletes cross-attn and the entire encoder/memory path with it, turning two-stream conditional generation into single-stream causal modeling
- Boundary: multimodal/RAG models may insert extra cross-attn (§3.1)

</details>

<details>

<summary>Q3. What actually changed from the GPT-2-style block to the Llama-style block?</summary>

- Both are **already Pre-LN** (GPT-2 is not Post-LN)
- LayerNorm→RMSNorm; learned absolute PE (outside the stack)→RoPE (inside attention, acting only on Q/K; first proposed by Su et al. 2021's RoFormer — LLaMA is an adopter); GELU dense FFN→SwiGLU gated FFN; MHA→MHA/GQA per the specific spec; most bias/dropout removed
- Both have a final norm, only the type differs (LayerNorm vs RMSNorm) (§3.2)

</details>

<details>

<summary>Q4. Why does attention divide by $\sqrt{d_k}$?</summary>

- Dividing by $\sqrt{d_k}$ pulls the attention logits' variance back to $O(1)$, preventing premature softmax saturation
- Derivation convention: if Q/K components are independent, zero-mean, unit-variance, then $\text{Var}(q\cdot k)=d_k$

You must be able to supply $\text{Var}(q\cdot k)=d_k$.

</details>

<details>

<summary>Q5. How do LayerNorm, RMSNorm, and BatchNorm differ? Why do Transformers usually not use BatchNorm?</summary>

- LayerNorm: normalizes over the feature dimension per sample, per position (subtract mean, divide by standard deviation) — per-sample independent, never across the batch
- RMSNorm: re-scale only (divide by the root mean square), no mean-centering — the mean-subtraction step is dropped
- BatchNorm: normalizes each feature dimension across the batch; its statistics (running mean/var) depend on other samples in the batch
- Transformer inputs are variable-length sequences, so cross-position/cross-sample batch statistics are unstable; inference is often variable-length, single-sample, token-by-token autoregressive decoding, where BatchNorm's running statistics mismatch the training distribution; LayerNorm/RMSNorm compute per sample independently, naturally fitting variable-length sequences and autoregressive decoding

Pitfall: saying only "BatchNorm works poorly in NLP" without naming the batch-statistics dependency and the variable-length/autoregressive-decoding mismatch as the concrete reason.

</details>

<details>

<summary>Q6. How do tensor shapes change through full attention? Why are multiple heads and $W_O$ needed?</summary>

- Input $x\in[B,T,d]$ → after Q/K/V projection, reshape to $[B,n_\text{head},T,d_h]$ ($d=n_\text{head}\cdot d_h$)
- score $=QK^\top/\sqrt{d_h}\in[B,n_\text{head},T,T]$ → add mask → softmax (along the last dim) → multiply with V to get $[B,n_\text{head},T,d_h]$
- Concat heads back to $[B,T,d]$, then the output projection $W_O\in\mathbb{R}^{d\times d}$ gives the final output $[B,T,d]$
- Multiple heads let the model learn different attention patterns in parallel subspaces; $W_O$ re-mixes and projects the heads' outputs — without $W_O$, head outputs are mere concatenation, and information across heads cannot recombine

Pitfall: omitting the reshape/concat steps; failing to say $W_O$'s role is "mixing multi-head information" rather than just "dimension alignment".

</details>

<details>

<summary>Q7. What is the key difference between self-attention and cross-attention?</summary>

- self-attention: Q, K, V come from the same sequence
- cross-attention: Q comes from the current stream (length $T$), K and V from another sequence/memory (length $S$); $T$ and $S$ are generally unequal
- The original Transformer decoder uses cross-attn to condition the target sequence on the source encoding; the pure decoder-only LLM block structurally has **no** cross-attn (§2, §3.1; full mask/softmax formulas in attention_tutorial.md §4.1)

Pitfall: thinking cross-attention is just "self-attention plus a mask"; failing to note that Q length and K/V length can differ.

</details>

<details>

<summary>Q8. Which design axis of attention does local/sliding-window attention belong to?</summary>

- It is the design axis of attention's **connectivity structure / visible range (connectivity pattern)** — not an "attention backend" (FlashAttention is the execution backend) and not a positional-encoding scheme
- Mistral uses a fixed sliding window; Gemma-2 alternates local/global 1:1 across layers; Gemma-3 is closer to a periodic pattern of several local layers then one global layer (about 5:1, not strict 1:1)
- Broadly combinable with any positional-encoding scheme, but extrapolation length, window boundaries, and compatibility with specific kernel implementations still need separate verification — not unconditionally "plug and play" (§7.3)

Pitfall: calling it an "attention backend", or conflating it with positional-encoding schemes like RoPE/ALiBi; assuming "broadly orthogonal" means "swap freely with no verification".

</details>

<details>

<summary>Q9. Do weight tying, embedding scaling, and the LM head belong inside the block or in the periphery recipe?</summary>

- All belong to the "model shell" level and **to no Transformer block** — the token embedding and LM head are the whole model's input/output layers, not components of any block
- Weight tying: input embedding and LM head share one parameter matrix; embedding scaling: token embedding multiplied by a constant related to $\sqrt{d}$; LM head soft-cap: some models also apply a §7.2-style bounded transform to the final logits (§4)
- Mixing them into the block definition when asked "what does a block look like" easily makes the interviewer doubt your grasp of the "layer vs model" boundary

</details>

<details>

<summary>Q10. What is SwiGLU's complete formula, and what is the FFN's role?</summary>

- The FFN is **position-wise**: it applies the same nonlinear transform independently to each position's vector, never mixing across positions (cross-position mixing is attention's job)
- SwiGLU formula: $\text{down}(\text{SiLU}(\text{gate}(x))\odot\text{up}(x))$ — gate and up are two independent linear projections to the intermediate dimension $h$, multiplied elementwise, then projected back to $d$ by down
- The SiLU on the gate branch acts as a gate, deciding "how much passes" per dimension of the up branch — the key difference from a dense FFN (two matrices, one activation)
- Within the block, the FFN plays the "per-position feature transformation / information storage" role, complementary to attention's "cross-position information routing"

Pitfall: remembering only the count "three matrices" without the gate branch's gating semantics and the FFN's position-wise nature.

</details>

### L2 Intermediate

<details>

<summary>Q11. Why is Pre-LN easier to train? Why does the stack usually end with a final norm?</summary>

- ① Pre-LN provides an identity gradient path and improves the optimization landscape, but does not guarantee lossless total gradients
- ② Under the independent-increment approximation, $\text{Var}=O(l)$ and RMS $=O(\sqrt l)$; actual growth depends on initialization and residual correlations
- ③ The final norm is the common stack-level closure; alternative scaling/initialization schemes can omit it (derivation in normalization_init_tutorial.md §5.3)

Note: Pre-LN does not guarantee final quality always beats a well-tuned Post-LN — when Post-LN trains stably, deep-layer contributions are not diluted and the ceiling is sometimes higher.

Pitfall: writing the variance growth as $\propto\sqrt l$ (it is the standard deviation/RMS that goes as $\sqrt l$; variance itself grows linearly as $O(l)$); calling the final norm "mathematically required".

</details>

<details>

<summary>Q12. Where should RoPE be inserted, and what goes in the KV cache?</summary>

- Standard path: RoPE acts only on Q/K; the cache stores K already rotated at each position; **V is never rotated**; GQA/MQA store only $n_\text{kv}$ heads
- Note: caching unrotated K is also legal, but positions must be saved and rotation applied on read
- MLA uses a different latent cache contract (§5.2; full rotation-matrix derivation in long_context_rope_yarn_mla_tutorial.md §2)

</details>

<details>

<summary>Q13. How do parameter counts and KV cache change across MHA, MQA, GQA?</summary>

- MHA parameters $4d^2$; GQA parameters $2d^2(1+n_\text{kv}/n_q)$, degenerating back to MHA at $n_\text{kv}=n_q$ (§1.3 [D06])
- KV cache: MHA $2n_q d_h$, GQA $2n_\text{kv}d_h$, MQA $2d_h$ elements/token/layer
- Conventions: $d=n_qd_h$, identical Q/K/V head dim, $W_O$ is $d\times d$, bias ignored

State the conventions first, then convert element counts to bytes.

</details>

<details>

<summary>Q14. How do you compute a dense Transformer block's parameter count and FLOPs?</summary>

- Parameters: attention about $4d^2$ (or GQA's $2d^2(1+n_\text{kv}/n_q)$) + FFN about $8d^2$ (dense) or $3dh$ (gated) + the two norms' per-channel parameters ($O(d)$, asymptotically lower-order than the matrix parameters) — per layer the FFN usually dominates (about 76% in §1.1's concrete example)
- FLOPs: attention's QKVO projections are $O(Td^2)$, score-matrix computation + weighted sum are $O(T^2d)$ ($T$ is the sequence length), and the FFN is $O(Tdh)\sim O(Td^2)$
- At small $T$ the projections dominate (the $O(Td^2)$ terms); at large $T$ attention's quadratic $O(T^2d)$ term gradually dominates — the direct reason attention compute becomes the bottleneck in long-context settings

Pitfall: memorizing only the parameter formulas without saying which of $O(Td^2)$ and $O(T^2d)$ dominates at which sequence lengths.

</details>

<details>

<summary>Q15. Why is SwiGLU's hidden size often close to $8d/3$?</summary>

- Ignoring bias, a dense FFN (two matrices) has $8d^2$ parameters ($4d$ expansion); a gated FFN (gate/up/down, three matrices) has $3dh$
- Setting them equal: $3dh=8d^2 \Rightarrow h=8d/3$ — a **common default computed from parameter matching**, not an architectural law that "gated FFN hidden size must equal $8d/3$"
- Real Llama-family implementations usually start from $4d$, multiply by $2/3$, then round to hardware-friendly multiples (with an optional multiplier); the intermediate size can deviate noticeably from $8d/3$ (§4; §1.3 [D06] verifies 73,728 parameters on both sides at $d=96$)

</details>

<details>

<summary>Q16. Are serial attn→FFN and parallel attn+FFN equivalent?</summary>

- No: in serial Pre-LN the FFN reads the residual stream **after** the attention update ($\text{FFN}(N_2(x+\text{Attn}(N_1(x))))$); in parallel, FFN and attention read the **same** normalized original input ($x$), independent of each other
- The test should not be "do the two outputs happen to differ" (fragile) but reconstructing the FFN's actual input tensor for an exact match (§1.4 [D08] captures the input with a forward hook + compares against an independent recomputation)
- The parallel structure removes this dependency chain and can cut some synchronization overhead during training (§7.1)

</details>

<details>

<summary>Q17. What problems do attention scale, QK-Norm, and logit soft-cap each solve?</summary>

- scale controls dot-product magnitude (standard $1/\sqrt{d_h}$ or a learnable temperature)
- original QK-Norm normalizes Q/K **before** the dot product and **replaces** the fixed scale with a learnable temperature
- soft-cap bounds scores/logits **after** the dot product (bounded transforms like $c\tanh(z/c)$)
- Combining them requires re-calibrating the scale; specific model implementations may differ (§7.2; full argument in normalization_init_tutorial.md §6.3)

Do not omit that original QK-Norm replaces the fixed scale.

</details>

<details>

<summary>Q18. How does a packed QKV projection differ from three separate Linears?</summary>

- Packed and separate Linears are mathematically equivalent; with equivalent weights they agree within reasonable tolerance for the given dtype (§1.4 [D07])
- Packed usually reduces kernel launches, but whether it is faster depends on sizes, quantization, and parallel layout
- Same for SwiGLU's gate/up

</details>

<details>

<summary>Q19. What is the essential difference between MLA and GQA?</summary>

- GQA compresses the **KV-head count**; MLA compresses KV into a low-dimensional latent ($d_c \ll n_qd_h$)
- MLA's cache is latent + shared RoPE key, not "fewer heads of ordinary K/V"
- To keep the absorbing trick, MLA must decouple RoPE; GQA has no such problem (§5.2; full derivation in long_context_rope_yarn_mla_tutorial.md §9)

</details>

<details>

<summary>Q20. Why do prefill and decode have different bottlenecks? How does the KV cache change the complexity?</summary>

- Prefill processes the whole prompt with high weight reuse, so it is usually more compute-bound; decode processes one token per step but must read the weights and the growing KV, so at small batch it is more bandwidth-bound
- What the KV cache saves is recomputing the historical K/V projections: at fixed model width, recomputing the K/V projections for the first $t$ positions grows linearly with length, while with a cache each step projects only the new token; $O(t^2)$ describes the cost of recomputing attention over the whole sequence, not the cost of the K/V projections ($t$ is the current position; full bottleneck analysis in llm_inference_serving_tutorial.md)

</details>

<details>

<summary>Q21. Which layer do FlashAttention, PagedAttention, and GQA/MQA each optimize?</summary>

- FlashAttention: an IO-aware exact-attention kernel, optimizing "how to compute" — tiling/recompute to reduce HBM access, changing neither attention's mathematical result nor the model architecture (§7.4)
- PagedAttention: a KV-cache memory-management mechanism, optimizing "how to store" — a page table maps logical KV to non-contiguous physical memory blocks, reducing fragmentation; serving-system level
- GQA/MQA: model-architecture choices, optimizing "how much to store" — fewer KV heads shrink the cache itself
- The three are optimizations at different levels and must not be conflated: the first two are execution/system engineering, the last is an architectural decision fixed before training; full mechanics in attention_tutorial.md (FlashAttention) and llm_inference_serving_tutorial.md (PagedAttention)

</details>

<details>

<summary>Q22. Why are bias and dropout gradually removed in modern large-model pretraining?</summary>

- Large-scale, low-epoch pretraining commonly drops bias/dropout: bias's benefit is limited; massive data already provides strong regularization, and dropout can even slow convergence
- This is an empirical recipe at that training scale, not a universal theorem (GPT-2-style keeps both; see the §3.2 bridge table)

</details>

<details>

<summary>Q23. What is inaccurate about the name "sandwich norm"? How should the formula be written?</summary>

- "Sandwich norm" is an ambiguous colloquialism; the more accurate phrasing is "one normalization before and after each sublayer branch" (branch pre+post normalization)
- Exact formula: $y=x+N_\text{post}(F(N_\text{pre}(x)))$ — the second normalization acts only on the **branch output**, not on the residual sum
- This differs from original Post-LN's $N(x+F(x))$: Post-LN normalizes the whole summed stream, while branch pre+post closes off only the branch output, leaving the residual trunk pure addition (§7.1; Gemma-2/3 use this topology; full variant genealogy in normalization_init_tutorial.md §6.2)

</details>

<details>

<summary>Q24. How do MLA's and GQA's KV cache contracts differ?</summary>

- GQA's cache is $n_\text{kv}$ heads of complete K/V (each of dimension $d_h$) — essentially still "multiple heads of ordinary K/V", just fewer than $n_q$
- MLA's cache is one shared latent $c_t^{KV}$ (dimension $d_c$) plus one shared decoupled RoPE key (dimension $d_h^R$) — not "fewer heads of ordinary K/V", and K, V are no longer stored independently. The full absorbing trick, why RoPE cannot be absorbed directly, and the decoupling scheme's algebra are in long_context_rope_yarn_mla_tutorial.md §9, not repeated here

</details>

<details>

<summary>Q25. When hand-writing a KV-cached incremental-decoding causal self-attention, which correctness pits are easiest to miss?</summary>

- ① The standard cache stores rotated K; the alternative contract must save positions (see Q12)
- ② The cache always keeps $n_\text{kv}$ heads; broadcast results must never be written back
- ③ Only under a dynamic dense causal setup with no padding/packing is the single-token mask all True
- ④ Full-forward and cached decode must agree, but mask broadcasting and cache validation still need separate tests (§1.4 [D04])

</details>

<details>

<summary>Q26. Where does GQA's $n_q \bmod n_\text{kv} = 0$ constraint come from? Does it still hold in MLA?</summary>

- Standard uniform GQA requires $n_q \bmod n_\text{kv}=0$: the $n_q$ Q heads split evenly into $n_\text{kv}$ groups of $n_q/n_\text{kv}$ each
- MLA has no $n_\text{kv}$ grouping variable, so the constraint does not apply (§5)
- Note: non-uniform head→KV mappings are definable in principle but are not a common model/kernel contract

</details>

<details>

<summary>Q27. Why is "MHA→MQA→GQA→MLA is one evolution chain" wrong? What is the correct way to organize the four?</summary>

- Historical order: MHA→MQA→GQA (MQA predates GQA)
- Compression strength: MHA→GQA→MQA — MHA ($n_\text{kv}=n_q$) and MQA ($n_\text{kv}=1$) are the two endpoints of the same uniform-GQA compression-ratio axis
- MLA compresses the latent (hidden) dimension, is not on the KV-head-count axis, and should not be appended to its end (§5)

</details>

### L3 Advanced

<details>

<summary>Q28. Given a Llama-3-like dense GQA block, how do you swap in MoE without changing the block interface? Which parts of the contract change, and which do not?</summary>

- Replace only the FFN slot: the attention slot (GQA, RoPE) and normalization slot (Pre-RMSNorm) stay untouched
- Three contract layers: ① **main-activation shape unchanged** — `[B,T,d] -> [B,T,d]`, residual add stays in place; ② **the training API may add** router/load-balancing quantities; ③ **distributed systems usually add** expert parallel/all-to-all; report total and activated parameters separately
- The backward pass only requires experts hit in this batch to receive gradients (§1.4 [D09]; full routing/capacity math in moe_tutorial.md §2–§4)

</details>

<details>

<summary>Q29. What systematic cost differences do parallel residual (GPT-J/PaLM) and serial Pre-LN have in training/inference?</summary>

- Dependency chain: parallel allows attn/FFN to overlap; serial forces attn→FFN
- Quality: both can train well; no unconditional winner
- Norm overhead: shared-norm parallel (e.g. GPT-J) saves one norm; two-norm parallel only removes the dependency without reducing norm count (§7.1)

</details>

<details>

<summary>Q30. Given a new model's technical report, which block fields would you check first to determine its recipe?</summary>

- Block: ① norm topology/type; ② positional encoding and its level; ③ attention: $n_q$/$n_\text{kv}$ or MLA latent/cache, QK-Norm, window; ④ FFN: dense/gated/MoE, activation, intermediate size, total/activated parameters; ⑤ bias/dropout/residual scaling
- Stack: final norm, local/global, dense/MoE layer placement (checked separately, never mixed into block fields)
- Fine details: head dims, scale/temperature, mask, cache layout

</details>

## 📚 References

- **Transformer (original encoder-decoder)** — Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762 (2017), NeurIPS 2017.
- **GPT-2 (decoder-only, Pre-LN, learned absolute PE)** — Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report (2019) (**no arXiv id**).
- **GPT-3** — Brown et al., *Language Models are Few-Shot Learners*, arXiv 2005.14165 (2020), NeurIPS 2020.
- **GPT-J (parallel residual)** — Wang & Komatsuzaki, *GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model*, EleutherAI open-source release (2021) (**no formal paper/arXiv id**).
- **PaLM (parallel residual, MQA)** — Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311 (2022).
- **RoPE** — Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2104.09864 (2021). The LLaMA family is a prominent adopter, not the proposer.
- **Llama 1** — Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, arXiv 2302.13971 (2023).
- **Llama 2** — Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288 (2023).
- **Llama 3** — Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783 (2024).
- **Qwen2** — Yang et al. (Qwen Team), *Qwen2 Technical Report*, arXiv 2407.10671 (2024).
- **Qwen2.5** — Qwen Team, *Qwen2.5 Technical Report*, arXiv 2412.15115 (2024).
- **Mistral 7B (sliding-window attention)** — Jiang et al., *Mistral 7B*, arXiv 2310.06825 (2023).
- **Gemma (1)** — Gemma Team (Google DeepMind), *Gemma: Open Models Based on Gemini Research and Technology*, arXiv 2403.08295 (2024).
- **Gemma 2 (branch pre+post norm, soft-capping)** — Gemma Team (Google DeepMind), *Gemma 2: Improving Open Language Models at a Practical Size*, arXiv 2408.00118 (2024).
- **Gemma 3 (QK-Norm)** — Gemma Team (Google DeepMind), *Gemma 3 Technical Report*, arXiv 2503.19786 (2025).
- **MQA** — Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv 1911.02150 (2019).
- **GQA** — Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv 2305.13245, EMNLP 2023.
- **MLA / DeepSeek-V2** — DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv 2405.04434 (2024).
- **DeepSeek-V3** — DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437 (2024).
- **DeepSeekMoE (fine-grained + shared experts)** — Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, arXiv 2401.06066 (2024), ACL 2024.
- **Auxiliary-Loss-Free Balance** — Wang, Gao, Zhao, Sun & Dai, *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, arXiv 2408.15664 (2024).
- **Mixtral of Experts** — Jiang et al., *Mixtral of Experts*, arXiv 2401.04088 (2024).
- **Pre-LN vs Post-LN gradient argument** — Xiong et al., *On Layer Normalization in the Transformer Architecture*, arXiv 2002.04745 (2020), ICML 2020.
- **DeepNorm / DeepNet** — Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*, arXiv 2203.00555 (2022).
- **RMSNorm** — Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv 1910.07467 (2019), NeurIPS 2019.
- **QK-Norm** — Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245 (2020), EMNLP 2020 Findings.
- **GELU** — Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, arXiv 1606.08415 (2016).
- **GLU Variants / SwiGLU** — Shazeer, *GLU Variants Improve Transformer*, arXiv 2002.05202 (2020). Proposed and systematically evaluated these Transformer FFN variants; the original GLU and the SiLU/Swish base ideas each have earlier sources.
- **Sibling tutorials in this series** — [attention_tutorial.md](attention_tutorial.md), [normalization_init_tutorial.md](normalization_init_tutorial.md), [moe_tutorial.md](moe_tutorial.md), [long_context_rope_yarn_mla_tutorial.md](long_context_rope_yarn_mla_tutorial.md), [linear_sparse_attention_tutorial.md](linear_sparse_attention_tutorial.md), [quantization_tutorial.md](quantization_tutorial.md), [lora_peft_tutorial.md](lora_peft_tutorial.md), [llm_inference_serving_tutorial.md](llm_inference_serving_tutorial.md)
