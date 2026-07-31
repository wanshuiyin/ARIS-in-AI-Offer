## §0 Mental Model + TL;DR Cheat Sheet

**A Transformer block is not one formula but a set of layered assembly rules.** Most interview confusions ("Pre-LN is the next generation of Post-LN", "MQA→GQA→MLA is one upgrade chain", "decoder-only is just the decoder with a part removed") come from compressing these layered rules into a single linear timeline. Start with the mental model: the residual stream $x_l$ passes through two or three **sublayers** in turn (attention, FFN, plus a cross-attn in the original encoder-decoder), with **serial** dependency by default — the FFN reads the residual stream after the attention update (parallel is the exception, see below):

```text
residual stream x_l
  │
  ├─▶ [norm slot: Pre / Post / branch pre+post] → Attention slot (self-/cross-attention,
  │     MHA/GQA/MQA/MLA, ±RoPE, ±QK-Norm, global/local window) → [norm slot] ──▶ ADD back to residual stream
  │     yielding the intermediate state u = x_l + Attn(N(x_l))
  │
  u ─▶ [norm slot] → FFN slot (Dense GELU / Gated SwiGLU / MoE) → [norm slot] ──▶ ADD back to residual stream
  │     (serial default: FFN reads u, not x_l; parallel variant: FFN and attention both read the same N(x_l), not u)
  │
residual stream x_{l+1}
```

**The slots are not strictly orthogonal independent axes; they live at different levels: first locate a design decision by level, then analyze within a level mostly independently. The couplings that need separate checks are the MLA↔RoPE split, the norm-topology↔norm-type binding, the joint scale of QK-Norm/attention scale/soft-cap (the full scale constraint is in §7.2), and GQA↔projection parameter count.** This master rule is stated only once here; later sections state the concrete coupling facts directly. Replace "orthogonal axes" with a layered map:

| Level | Typical decisions | Owned by |
| --- | --- | --- |
| Inside the operator | QKV projections, RoPE, QK-Norm, head sharing (MHA/GQA/MQA/MLA), gated FFN | the attention/FFN module itself |
| Single block | serial/parallel dependency graph, norm placement (Pre/Post/branch pre+post), residual add, branch scaling/gating | the block's `forward` |
| Stack scheduling | final norm, local/global interleaving, which layers are dense vs. MoE | the stack-assembly code, not any single block |
| Model shell | token embedding, external absolute PE, weight tying, LM head (incl. soft-cap) | the whole model's input/output layers |
| Execution/adaptation overlay | KV cache, FlashAttention, packed projection, quantization, LoRA | deployment/fine-tuning execution strategy; leaves the four levels above architecturally unchanged |

Every later judgment of "does this choice belong inside the block, at stack level, or in the model shell" is located by this table, never by a vague "some slot".

> 💡 **Four residual-topology formulas, one screen to memorize, all stated at "whole block" granularity** (§1–§7 below are all concretizations of these four; $A$, $M$ are the attention and FFN sublayers, and $N_1,N_2,N_A,N_M,N_1^{\text{pre}},N_1^{\text{post}},N_2^{\text{pre}},N_2^{\text{post}}$ are all normalization operators)
>
> $$\text{Post-LN (Vaswani et al. 2017 original Transformer)}:\quad u = N_1\big(x + A(x)\big),\quad y = N_2\big(u + M(u)\big)$$
> $$\text{Pre-LN serial (explicitly adopted at least by GPT-2, later the common form in modern LLMs)}:\quad u = x + A\big(N_1(x)\big),\quad y = u + M\big(N_2(u)\big)$$
> $$\text{Pre-LN parallel (GPT-J / PaLM)}:\quad y = x + A\big(N_A(x)\big) + M\big(N_M(x)\big)$$
> $$\text{branch pre+post (Gemma-2/3)}:\quad u = x + N_1^{\text{post}}\big(A(N_1^{\text{pre}}(x))\big),\quad y = u + N_2^{\text{post}}\big(M(N_2^{\text{pre}}(u))\big)$$
>
> Post-LN wraps the normalization around the residual sum, repeatedly squeezing the trunk. Pre-LN serial leaves the trunk a pure identity-add path; the FFN reads the updated stream $u$. Pre-LN parallel removes the ordering dependency; attention and FFN read the same $x$ (the precise definition of parallel is in §7.1.2). In branch pre+post, the second normalization wraps only the branch output; the trunk stays pure addition — unlike Post-LN. Note: norm placement (Post/Pre/branch pre+post) and serial/parallel dependency are two dimensions, not one axis; §1 [D02]/[D08] gives executable positive and negative proofs of all four topologies.

**The Transformer Block in 12 sentences**:

1. **Block = residual stream + sublayers assembled by level**: the internal variants of attention and FFN are independent choices within the same level; encoder-decoder has one more cross-attn sublayer than decoder-only.
2. **Decoder-only: per layer it is "remove cross-attn", but the whole model also loses the encoder/memory path**: 3 sublayers (masked self-attn, cross-attn, FFN) become 2 (causal self-attn, FFN); a complete answer adds one more sentence — two-sequence conditional generation becomes single-sequence causal modeling (§3).
3. **GPT-2 is already Pre-LN**: from GPT-2-style to Llama-style, what actually changed is LayerNorm→RMSNorm, learned absolute PE→RoPE (moved inside attention), GELU dense FFN→SwiGLU gated FFN, MHA→GQA where applicable, plus removing most bias/dropout (§3.2, §4).
4. **A Pre-LN stack usually ends with a final norm**: under the independent-increment assumption the residual stream's **variance** grows roughly linearly as $O(l)$ (it is the standard deviation/RMS that goes as $\sqrt l$); feeding the output head without this closure usually degrades training quality noticeably. It is a common configuration of the standard recipe, not mathematically indispensable (§9 Q11; derivation in normalization_init_tutorial.md §5.3).
5. **MHA→MQA→GQA→MLA is not one evolution chain**: the historical order is MHA→MQA→GQA, the compression-strength order is MHA→GQA→MQA; MLA is DeepSeek's independent low-rank latent branch and does not sit at the end of the KV-head-count axis (§5).
6. **Dense FFN → MoE is another independent "capacity axis"**: not an inevitable "next-generation replacement" — representative models such as Llama 1–3, the original Mistral, and Gemma 1–3 still mostly use dense FFN, while Mixtral, Llama 4, and others have adopted MoE (§6).
7. **QK-Norm, logit soft-cap, branch pre+post norm, parallel residual, and local/sliding attention can mostly be combined as needed**: what needs a separate check is the joint scale when QK-Norm and soft-cap are used together (§7.2).
8. **FFN parameter matching is computed, not a law**: dense FFN ($8d^2$) and gated FFN ($3dh$) have equal parameters at $h=8d/3$; real models round to hardware-friendly multiples and can deviate noticeably (§4, §1 [D06]).
9. **KV cache tracks $n_\text{kv}$, not $n_q$**: MQA/GQA reduce K/V projection parameters, compute, and cache memory/bandwidth, but do not reduce the Q projection's compute (§5; full memory accounting in attention_tutorial.md §6 and kv_cache_speculative_decoding_tutorial.md §2).
10. **The modern recipe has not fully converged**: Pre-RMSNorm + RoPE + gated FFN + inference-friendly GQA is only the "more common" combination; Post-LN variants, LayerNorm, MHA, MQA, parallel residual, dense FFN, and MoE are all still used in production (§8).
11. **The only structural difference between cross-attention and self-attention is where Q/K/V come from**: in self-attn, Q, K, V all come from the same stream; in cross-attn, Q comes from the current stream while K/V come from another sequence whose length can differ — everything else (scaled dot-product, multi-head split, softmax) is identical, needing no separate math (§2).
12. **"Packed projection" is only a kernel-level equivalent rearrangement**: fusing the three Q/K/V Linears (or SwiGLU's gate/up) into one big Linear is purely an implementation optimization; with row-wise concatenated weights the results should agree at the corresponding precision, changing no architectural semantics (§1 [D07]).

> 🎯 **Scope of this tutorial: assembly only — no re-derivation of the math inside each part**
> The softmax-attention derivation, FlashAttention's IO complexity, the Xiong/DeepNorm gradient arguments, the full MoE routing/capacity math, RoPE rotation matrices and YaRN/NTK/MLA latent algebra, linear-attention derivations, quantization error analysis, LoRA's low-rank formulas — each part's own math has dedicated coverage in attention_tutorial.md, normalization_init_tutorial.md, moe_tutorial.md, long_context_rope_yarn_mla_tutorial.md, linear_sparse_attention_tutorial.md, quantization_tutorial.md, and lora_peft_tutorial.md; prefill/decode system bottlenecks and KV-cache paging/scheduling are in llm_inference_serving_tutorial.md. This tutorial only cares **how these parts assemble into a block, and how the assembly evolved from the 2017 version to the modern one**; for any deep math or systems question it gives a 1-2 sentence conclusion plus a link, never a repeated derivation.

## §1 Capstone: Assembly, End-to-End Forward, Parameter Accounting, and Executable Verification

### 1.1　Assemble a Llama-3-like block, then swap in MoE

This is the first body section after the §0 mental model, and it gives the answer directly: what a concrete, complete, runnable modern decoder-only block looks like. Below we actually assemble every slot with the "more common modern combination"; the history and alternatives for each concrete choice (RMSNorm, RoPE, GQA, SwiGLU, etc.) are in §2–§7, and you can follow this example without reading those first. (Strictly this is a "Llama-3-like core block": exactly reproducing a given checkpoint also requires matching head count, KV-head count, intermediate size, RoPE base, bias settings, and dtype behavior.)

1. **Normalization slot (inside the block)**: Pre-RMSNorm, twice (once before attention, once before FFN).
2. **Attention slot**: causal self-attn, RoPE applied internally to Q/K, $n_q$ query heads grouped to share $n_\text{kv}$ KV heads (GQA; degenerates to MHA when $n_\text{kv}=n_q$, see §1.4 [D05]), K cached after rotation, V never rotated, no bias.
3. **FFN slot**: SwiGLU gated FFN, $h \approx 8d/3$ rounded to a hardware-friendly multiple, no bias.
4. **Single-block assembly**: $x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x))$, $x=x+\text{SwiGLU}(\text{RMSNorm}(x))$ — this is the complete definition of **one block**; repeat it $N$ times to get the whole decoder stack.
5. **Stack-level closure (belongs to no single block)**: after all $N$ blocks, apply one **final RMSNorm** before the LM head: $x_\text{out} = \text{RMSNorm}\big(\text{Block}_N(\cdots\text{Block}_1(x_0))\big)$. The final norm is a stack-scheduling operation, not a component of any block in steps 1–4 (see the §0 layered map and §4's block-vs-periphery boundary discussion).

Now compute this block's parameter count concretely with the teaching sizes from the §1.4 code ($d=96$, $n_q=8$, $n_\text{kv}=2$, gated hidden $=256$): the GQA attention part has $2d^2(1+n_\text{kv}/n_q)=23{,}040$ matrix parameters, the SwiGLU FFN part has $3dh=73{,}728$, and each of the two RMSNorms has only $d=96$ per-channel scale parameters — a single block totals about $23{,}040+73{,}728=96{,}768$ matrix parameters, with the FFN taking the vast majority. This is why "how to choose the FFN hidden size" (§4, §1.3) affects total parameters more than "how many KV heads to use".

**Next, swap the FFN slot for MoE and touch nothing else**:

$$x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x)), \qquad x=x+\text{MoEFFN}(\text{RMSNorm}(x))$$

`MoEFFN`'s external **main-activation tensor** interface is still `[B,T,d] -> [B,T,d]`, and not a single line of `ModernDecoderBlock` changes (verified directly in §1.4 [D09]). What changes are the other two contract layers: training usually returns extra router/load-balancing quantities, and distributed deployment usually introduces expert parallelism. The full three-layer contract breakdown is in §6 and §9 Q28.

> ✅ **Capstone summary: which fields describe a block's recipe**
> Normalization topology (Post/Pre-serial/Pre-parallel/branch pre+post) + normalization type (LayerNorm/RMSNorm) + positional-encoding scheme and its level (learned absolute outside the stack / RoPE inside attention) + attention variant (MHA/MQA/GQA/MLA, local window or not, QK-Norm or not, attention scale/temperature) + FFN variant (dense/gated/MoE, intermediate size) + bias/dropout/norm epsilon/residual scaling (e.g. LayerScale, see §7.1.3). This covers the coarse-grained fields common interviews need; finer-grained values (head dim, RoPE base, exact activation implementation, etc.) are not on this list. **Final norm, local/global interleaving, and dense/MoE layer placement are stack-scheduling decisions (step 5) — do not mix them into the definition of a single block.**

### 1.2　End-to-end forward walkthrough: the complete data flow of a modern decoder-only block

Here is the walkthrough of the complete data flow of a modern decoder-only block — an equally detailed walkthrough of the historical 2017 encoder-decoder version is in §2 (that is "where we came from" history; reading it or not does not affect understanding this block now). Using the teaching sizes from §1.1 ($d=96$, $n_q=8$, $n_\text{kv}=2$, gated hidden $=256$), the full tensor flow is:

$$x \in [B,T,d] \;\xrightarrow{\text{RMSNorm}}\; h_1 \;\xrightarrow{\text{Q/K/V projection + reshape}}\; Q\in[B,n_q,T,d_h],\;K,V\in[B,n_\text{kv},T,d_h] \;\xrightarrow{\text{RoPE}}\; Q',K' \;\xrightarrow{\text{GQA attention}}\; O\in[B,T,d] \;\xrightarrow{\text{residual add}}\; u = x + O$$

$$u \;\xrightarrow{\text{RMSNorm}}\; h_2 \;\xrightarrow{\text{gate/up projection}}\; \text{SiLU}(\text{gate}(h_2))\odot\text{up}(h_2) \;\xrightarrow{\text{down projection}}\; F\in[B,T,d] \;\xrightarrow{\text{residual add}}\; y = u + F$$

This is the complete forward pass of a single block; repeat this block $N$ times and then apply one stack-level final RMSNorm (§1.1 step 5) to get the full decoder-only backbone: $y_\text{final} = \text{RMSNorm}\big(\text{Block}_N(\cdots(\text{Block}_1(x_0)))\big)$.

This data flow takes different concrete forms under three execution modes:

| Execution mode | Q/K/V lengths | mask | KV cache | Typical scenario |
| --- | --- | --- | --- | --- |
| Training / full-forward | Q, K, V all span the full $T$ | one-shot lower-triangular causal mask $[T,T]$ | not used | training, teacher-forcing evaluation, scoring a full prompt in one pass |
| Prefill (first inference step) | Q, K, V all span the full prompt length $S$ | causal mask $[S,S]$ | written to cache (K and V, $n_\text{kv}$ heads each, length $S$) | first forward after receiving the user prompt, building the cache for subsequent decode |
| One-token decode (step-by-step inference) | Q has length $1$ (new token); K, V grow from cached history length $t$ to $t{+}1$ | with a dynamic cache and no padding/packing, the new token sees all of $0..t$ (whole row True); static cache/left padding/packed sequences still need extra masking (see §9 Q25) | read old cache + append new K/V | every step of autoregressive generation |

The three modes share the same RMSNorm→RoPE→attention→residual code path, differing only in the actual Q/K/V lengths, the mask shape, and whether the cache is read/written — which is exactly why §1.4 [D04] uses one implementation to verify that full-forward and cached decoding agree numerically (if the two paths used different code, that test would lose its cross-validation meaning). The system-level performance bottlenecks of prefill/decode (arithmetic intensity, batching/scheduling) are out of scope here; see llm_inference_serving_tutorial.md.

### 1.3　Parameter-count and KV-cache formula accounting

**FFN parameters** (ignoring bias): dense (ReLU/GELU, two matrices) $8d^2$; gated (SwiGLU/GeGLU, three matrices) $3dh$. Setting them equal gives $h=8d/3$ — a **common default computed from parameter matching**, not an architectural law; real Llama-family implementations usually start from $4d$, multiply by $2/3$, then round to hardware-friendly multiples (with an optional multiplier), so the intermediate size can deviate noticeably from $8d/3$. §1.4 [D06] verifies exactly 73,728 matrix parameters on both sides with $d=96$, dense hidden $=384$ ($4d$), gated hidden $=256$ ($8d/3$).

**Attention parameters** (ignoring bias): MHA's four Q/K/V/O projections total $4d^2$; GQA gives $2d^2(1+n_\text{kv}/n_q)$ (degenerating back to $4d^2$ at $n_\text{kv}=n_q$, and reaching MQA's minimum at $n_\text{kv}=1$). §1.4 [D06] checks both formulas against real `.numel()`. Both formulas assume $d=n_qd_h$, identical head dim for Q/K/V, a $d\times d$ output projection, and no bias — adjust accordingly when these assumptions fail.

**KV cache** (the table gives **element counts** per token per layer, not bytes; full memory usage further multiplies by batch size $B$, sequence length $L$, layer count $N_\text{layer}$, and bytes per element — usually 2 bytes for fp16/bf16, 1 byte for fp8; full cross-model GB numbers in kv_cache_speculative_decoding_tutorial.md §2.2):

| Variant | Cache size (elements/token/layer) |
| --- | --- |
| MHA | $2 n_q d_h$ |
| GQA | $2 n_\text{kv} d_h$ |
| MQA | $2 d_h$ |
| MLA | $d_c + d_h^R$ (latent dim + shared RoPE component; no "$\times 2$" because K/V share one latent) |

### 1.4　From-scratch implementation and executable verification: `code/transformer_block.py`

The full runnable script is [`code/transformer_block.py`](code/transformer_block.py) (pure PyTorch; all 9 demos [D01]–[D09] finish in seconds on CPU). Four entry classes correspond to this tutorial's four "block eras": `VanillaEncoderLayer2017`, `VanillaSeq2SeqDecoderLayer2017` (§2), `GPT2StyleDecoderLayer` (§3), `ModernDecoderBlock` (§4/§1.1, with `ffn=` swappable among `DenseFFN`/`GatedFFN`/`MoEFFN`). The body shows only the three most central snippets; the remaining demos are standalone functions in the script and are not reproduced here.

**[D02] Residual-topology test** (the executable version of §0's four formulas: zero out the sublayer output — Pre-LN must return exactly $x$, Post-LN must return $N(x) \ne x$):

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

The **actual output** of running `python3 code/transformer_block.py` (re-run and backfilled after the code fixes; D10-D15 are targeted regression tests added during code review, covering constructor argument validation, KV-cache consistency, cross-attention combination restrictions, mask safety and broadcasting, default causal behavior, and RoPE-cache device/dtype handling):

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

> 🧭 **Where to go after this example**
> Above is a concrete, runnable modern decoder-only block; if you want the history, alternatives, and trade-off rationale behind each design choice — RMSNorm, RoPE, GQA, SwiGLU, norm topology, … — continue to §2–§7. You do not need to read those sections first to fully understand the block above.

## §2 The 2017 seq2seq Transformer

The original Transformer (Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762, 2017, NeurIPS 2017) is an **encoder-decoder** architecture: the encoder encodes the source sequence into a set of representations, and the decoder autoregressively generates the target sequence conditioned on them. Positional encoding uses fixed sinusoidal PE, added directly to the token embeddings **before** entering the encoder/decoder stacks — it belongs to no block; it is input processing outside the blocks (full derivation in long_context_rope_yarn_mla_tutorial.md §2/§3). The paper's original configuration is $d_\text{model}=512$, $d_\text{ff}=2048$ (4× expansion), 8 heads, 6 encoder + 6 decoder layers; this tutorial demonstrates with smaller sizes (e.g. $d=96$), which is a **teaching configuration**, not the paper's — do not conflate the two.

**The encoder layer and decoder layer are two different classes, not two usages of one "two-sublayer" template** — they differ in both the number and the kinds of sublayers:

| | Encoder layer | Decoder layer |
| --- | --- | --- |
| Sublayer 1 | Self-attn (bidirectional, no causal mask, only an optional padding mask) | Masked self-attn (causal) |
| Sublayer 2 | FFN | Cross-attn (Q from this layer's output, length $T$; K/V from encoder output, length $S$) |
| Sublayer 3 | — | FFN |
| Norm topology | Post-LN per sublayer: $y=N(x+F(x))$ | Post-LN per sublayer, three times |

The decoder's three sublayers run in the fixed order masked self-attn → cross-attn → FFN, each wrapped in its own Post-LN. Cross-attn's Q, K, V sources are asymmetric: Q comes from the decoder's own in-progress stream (length $T$), K and V come from the encoder output (length $S$), and $T$ and $S$ can differ — exactly the shape separation explicitly verified in §1's code [D01] (formula details and mask handling for self-attn vs cross-attn are in attention_tutorial.md §4.1, not re-derived here).

**Walking one concrete shape through** (the teaching sizes used by §1's code [D01]: batch $B=2$, source length $S=5$, target length $T=7$, $d=96$): the encoder input is `[B, S, d]` and remains `[B, S, d]` after several encoder layers, passed as memory to every decoder layer; the decoder input is `[B, T, d]`, and in the masked self-attn stage Q/K/V are all computed internally from `[B, T, d]` (causal mask of size $T\times T$); in the cross-attn stage Q is still `[B, T, d]` but K, V come from the encoder memory with shape `[B, S, d]` — the attention score matrix is $T\times S$ (not square), with softmax along the $S$ dimension; after the output projection it returns to `[B, T, d]`, matching the self-attn sublayer's output shape, so it can proceed into the FFN sublayer. **$T \ne S$ does not affect this data path at all** — this is what "cross-attn decouples Q length from K/V length" concretely looks like.

> ⚠️ **Two classes, not two call modes of one class**
> `VanillaEncoderLayer2017` (2 sublayers) and `VanillaSeq2SeqDecoderLayer2017` (3 sublayers) are two independent classes in §1's code. Writing them as one "generic block class + parameter switches" is workable engineering but hides the structural fact that the decoder has a whole extra cross-attn sublayer — when hand-writing in an interview, listing each class's sublayers clearly first scores better than coding straight away.

## §3 The Three-Family Map + the Decoder-Only Fork + the GPT-2-Style Bridge

### 3.0　The three-family map: encoder-only / decoder-only / encoder-decoder

The original Transformer is encoder-decoder; from here on this tutorial focuses entirely on **decoder-only** (the mainstream choice of GPT/Llama/Qwen-family LLMs). Interviews often ask for a side-by-side comparison of the three families, so here is a compressed map first, without going into encoder-only's internal pretraining mechanics:

| Family | Representatives | Sublayer composition | Attention direction | Typical training objective | Typical use |
| --- | --- | --- | --- | --- | --- |
| Encoder-only | BERT, RoBERTa | per layer: self-attn + FFN (no causal mask) | bidirectional (sees the whole sequence) | denoising objectives such as masked language modeling (MLM) | representation learning, classification, retrieval embeddings |
| Encoder-decoder | original Transformer, T5 | encoder layer: self-attn + FFN; decoder layer: masked self-attn + cross-attn + FFN | encoder bidirectional; decoder unidirectional and conditioned on the encoder | conditional generation (translation, summarization) | two-sequence conditional generation tasks |
| Decoder-only | GPT, Llama, Qwen | per layer: causal self-attn + FFN (no cross-attn) | unidirectional (causal) | autoregressive language modeling | general-purpose generative LLMs |

The core difference among the three is not "how many parameters" but that attention visibility, sublayer composition, and training objective change together as a bundle. This tutorial expands the full assembly detail of only the decoder-only branch — covering §1's capstone and §3–§9 (§2 is the 2017 encoder-decoder historical background, discussing the encoder-decoder architecture itself, and does not count toward the decoder-only scope); encoder-only's bidirectional masked modeling and encoder-decoder's full training objectives are out of scope — consult each model's original paper when needed.

### 3.1　Why "just remove cross-attn" is not enough

The decoder-only block used by modern LLMs (GPT, Llama, Qwen, …) is, on its face, "the original decoder layer with cross-attn removed". **Read this claim at two levels**:

- **At the single-layer structural level, it is correct**: the original decoder is masked self-attn → cross-attn → FFN (**three** sublayers); the decoder-only block is causal self-attn → FFN (**two** sublayers). What is missing is exactly the cross-attn sublayer; the layer-by-layer comparison holds.
- **At the whole-model level, it is incomplete**: saying only "cross-attn removed" suggests one component was deleted and everything else stayed. In fact the entire encoder/memory path also disappeared — source/prompt and target go from "two-stream conditional generation" (encode source, then generate target conditioned on it) to "single-stream causal modeling" (prompt and generated content concatenated into one causal sequence). What changes is the information-flow topology and the sequence factorization, not that the task was replaced by a different problem.

So the more complete answer is: "per layer, yes, cross-attn is removed; but at the whole-model level you must also say the encoder/memory path disappears with it, and two-stream conditional generation becomes single-stream causal modeling" — more accurate than either "a part was removed" or, at the other extreme, "it is a totally different architecture".

Boundary: some multimodal/retrieval-augmented models reintroduce cross-attention on a decoder-only backbone; every "decoder-only has no cross-attn" statement in this tutorial is restricted to the **pure decoder-only block** (two sublayers: causal self-attn + FFN).

> 🎯 **How to answer this fundamental-difference question in an interview**
> Answer by level: at the single-layer level it is "remove cross-attn" (3 sublayers become 2); at the whole-model level add the information-flow topology change — "two sequences (source/target) become one causal sequence"; then add "only the pure decoder-only block lacks cross-attn; some multimodal/retrieval-augmented models reintroduce it" — two levels of accuracy beyond "a part was removed".

### 3.2　The GPT-2-style bridge: what actually changed from GPT-2 to Llama

**Common misconception**: modern LLMs are what replaced Post-LN with Pre-LN. **Fact**: GPT-2 (Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report, 2019, no arXiv id) is already Pre-LN, and puts a final LayerNorm (denoted `ln_f`) at the end of the stack — why a Pre-LN stack usually needs this closure, and variants without a final norm, are covered together in §9 Q11 and normalization_init_tutorial.md §5.3. The complete GPT-2 block recipe:

$$x = x + \text{Attn}\big(\text{LN}(x)\big), \qquad x = x + \text{FFN}\big(\text{LN}(x)\big), \qquad \text{(GELU dense FFN, MHA, bias=True, no cross-attn)}$$

The learned absolute positional embedding is still added to the embedding, **outside** the stack — the same "owning level" as the original sinusoidal PE; neither lives inside a block. RoPE, adopted by the LLaMA family (first proposed by Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2104.09864, 2021; LLaMA is a prominent adopter, not the proposer), is the opposite: it lives **inside attention**, acting only on the Q/K dot product — viewing them as a "seamless swap" between members of the same block is inaccurate. Positional information is carried by an external embedding in GPT-2-style and by rotation inside attention in Llama-style; the two belong to different levels (full derivation in long_context_rope_yarn_mla_tutorial.md §2).

The GPT-2-style → Llama-style bridge table (this is what actually changed):

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

> ✅ **Skeleton of the standard "what changed from GPT-2 to Llama" answer**
> First state "both are already Pre-LN" (blocking the common wrong answer that Pre-LN is a modern invention), then compare item by item: norm type → positional-encoding level → FFN structure → attention head configuration → bias/dropout, and close with "both have a final norm, only the type differs".

## §4 The Llama-Style Core Recipe

On top of the GPT-2-style bridge, Llama-style (Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288, 2023; Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783, 2024) fills each slot with the concrete implementations below. This section fixes MHA to keep the data path clear; GQA is expanded separately in §5:

$$x = x + \text{Attn}_{\text{RoPE}}\big(\text{RMSNorm}(x)\big), \qquad x = x + \text{SwiGLU}\big(\text{RMSNorm}(x)\big)$$

- **RMSNorm**: re-scale only, no mean-centering, no bias: $y = \dfrac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\odot w$. (Zhang & Sennrich's argument is in normalization_init_tutorial.md §4: it often matches LayerNorm's quality — not a universal "never loses" result.)
- **RoPE**: acts on Q and K inside attention; V is never rotated. Standard implementations usually cache the **already-rotated** K, and incremental decoding rotates new tokens at their true positions. The unrotated-K-cache variant is in §9 Q12, MLA's different cache contract in §5.2; full derivations (complex-number view, frequency choice, YaRN/NTK) in long_context_rope_yarn_mla_tutorial.md §2/§5/§6.
- **SwiGLU**: $\text{down}\big(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)\big)$, a three-matrix gated FFN; GeGLU is the same structure with a GELU gate (Shazeer, *GLU Variants Improve Transformer*, arXiv 2002.05202, 2020). At similar budgets, gated FFNs often deliver better quality — an empirical regularity, not a law. The parameter-matching arithmetic is in §1.3 [D06].
- **bias, dropout**: Q/K/V/O and the three FFN matrices usually drop all bias; large-scale pretraining commonly removes dropout or reduces it drastically.
- **final norm**: one RMSNorm at the end of the stack, closing off the inflated residual stream.
- **Numerical-precision boundary**: computations sensitive to numeric range — RMS/variance reductions, softmax, final logits — are commonly done at higher precision (e.g. fp32) and cast back to the activation dtype (bf16/fp16). This is part of assembly correctness, not quantization math; the full low-precision error analysis is in quantization_tutorial.md.

> ⚠️ **Inside the block vs. the "block-periphery recipe" — do not mix them**
> Weight tying (input embedding sharing parameters with the LM head), embedding scaling, the LM head's soft-cap, and the global dropout switch are part of the "complete LM recipe" but **strictly live outside the block** — the token embedding and LM head belong to no Transformer block. Mixing them into the block definition when asked "what does a block look like" makes the interviewer doubt you can separate "one block" from "the whole model".

The exact fields of the "block-periphery recipe", item by item:

| Periphery field | One-sentence description |
| --- | --- |
| Weight tying | the input token-embedding matrix and the LM-head projection matrix share one set of parameters, saving $Vd$ parameters |
| Embedding scaling | some implementations multiply the token embedding by a constant related to $\sqrt{d}$ before the first block |
| LM head soft-cap | some models (e.g. the Gemma family) also apply a §7.2-style bounded transform to the final vocabulary logits — mechanically the same tool as the attention-logit soft-cap, applied to a different tensor |
| Global dropout switch | whether dropout is on, and the dropout rate, are usually model-wide training configuration; but the dropout **operation itself** typically executes inside attention weights, FFN outputs, and residual branches — "who configures it" and "where the operator actually runs" are two different things; do not lump them into "a property outside the block" |

None of these four appears in any block class described in §1/§8 — `ModernDecoderBlock`'s `forward` receives only the residual stream `x` and never knows whether embedding scaling happened outside or whether the output head is tied. Locate them by the §0 layered map: these four belong to the "model shell" level, while final norm, local/global interleaving, and dense/MoE placement belong to "stack scheduling" — do not lump them together as "stuff outside the block".

## §5 The Inference-Memory Axis: MHA → MQA/GQA, with MLA as an Independent Branch

**§5's main problem is KV-cache memory**: every step of autoregressive generation appends the new token's K, V to the cache and attends over the whole cache; the cache size directly determines how many requests one GPU can serve concurrently and how long a context it can hold. GQA/MQA reduce the KV-head count and simultaneously change the K/V projections' own parameter count and compute (the $2d^2(1+n_\text{kv}/n_q)$ formula below is the evidence): the KV cache mainly affects inference memory/bandwidth, while K/V projection parameters are a fixed cost that also exists in training.

### 5.1　MHA / MQA / GQA: different compression ratios on the same axis

The historical order must be set straight:

> ⚠️ **Not one line MHA→GQA→MQA→MLA**
> MHA is the baseline; **MQA (Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv 1911.02150, 2019) predates GQA (Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023)**, and GQA is the quality-memory compromise between MHA and MQA. **MLA is DeepSeek's separate low-rank latent-compression branch**, not "reduce $n_\text{kv}$ once more". The four are coexisting design points on the "KV memory" design axis, not a historical ladder every model climbed in order.

| Variant | Q heads | KV heads | Relation |
| --- | --- | --- | --- |
| MHA | $n_q$ | $n_q$ | baseline, $n_\text{kv}=n_q$ |
| MQA | $n_q$ | $1$ | all Q heads share one set of K/V |
| GQA | $n_q$ | $n_\text{kv}$ ($1 \lt n_\text{kv} \lt n_q$, with $n_q \bmod n_\text{kv}=0$) | each group of $n_q/n_\text{kv}$ Q heads shares one set of K/V |

Assembly contract (full memory accounting, per-token-per-layer formulas, and concrete per-model GB numbers are in attention_tutorial.md §6 and kv_cache_speculative_decoding_tutorial.md §2, not re-derived here):

- KV-cache size scales only with $n_\text{kv}$, **independent of $n_q$** — MQA/GQA reduce the K/V projections' own parameters and compute, and also reduce KV-cache memory/bandwidth; but the Q projection's compute does not change with $n_\text{kv}$, so do not flatten this into "only saves memory/bandwidth".
- **Standard implementations usually cache K after RoPE is applied; V is never rotated**; in incremental decoding, the new token's Q and K get RoPE at the true position offset, and the already-rotated K in the historical cache is never recomputed (the unrotated-K-cache variant is in §9 Q12).
- GQA/MQA caches **store only $n_\text{kv}$ heads**; broadcasting K/V to the $n_q$ Q heads may happen only inside the attention computation (via view/reshape/grouped einsum) — **never write the broadcast tensor back to the cache**. This is the easiest pit to fall into in hand-written implementations, and §1 [D05] turns it directly into an executable assertion.

To feel this axis's real payoff: LLaMA-2-70B with vanilla MHA would need about 10 GB of KV cache per sample at a 4096 context; switching to GQA ($n_\text{kv}=8$) drops it to about 1.25 GB — one of the important reasons many large models adopt GQA over MHA.

### 5.2　MLA: the independent low-rank latent branch

**MLA (DeepSeek-V2, May 2024, arXiv 2405.04434) is an independent branch, not "a more aggressive GQA"**: GQA compresses along the head dimension (multiple Q heads share KV heads), while MLA compresses along the hidden dimension via **low-rank projection** (squeezing each token's K/V into a latent vector with $d_c \ll n_q d_h$), and it must decouple RoPE into a separately shared small-dimension component to preserve the "absorb the up-projection into the query side" inference speedup — full derivations (the absorbing trick, why RoPE cannot be absorbed directly, the decoupling scheme, the cache-total formula) are in long_context_rope_yarn_mla_tutorial.md §9. Record only the conclusion here: MLA's cache contract is "store the latent $c_t^{KV}$ + one shared RoPE key", not "store fewer K/V heads".

> 💡 **One sentence on what each compresses**
> GQA compresses the **head count** (multiple Q heads crowd onto one K/V group); MQA is GQA's extreme case ($n_\text{kv}=1$); MLA compresses **each token's representation dimension** (K/V jointly projected into one low-rank latent) and inherently must solve RoPE's position dependence. The three are not three points on one scale but two different compression ideas.

## §6 The Capacity Axis: Dense FFN → MoE

**§6's main problem is model capacity**: MoE replaces the FFN with $N$ experts + one router, and each token passes through only $k \ll N$ experts — total parameters go up, per-token activated parameters stay flat. The full routing formulas (token-choice top-k, expert-choice, DeepSeek's aux-loss-free bias update), capacity factor, load-balancing loss, and token dropping are in moe_tutorial.md §2–§4; here we cover only the "assembly contract":

- **The block's external main-activation tensor interface is unchanged**: dense or MoE, the FFN is `[B, T, d] -> [B, T, d]` and the residual add stays in place — the direct reason why, in §1 [D09], "swap `ffn` for `MoEFFN` and not a line of `ModernDecoderBlock` changes". But the training contract usually returns extra router/load-balancing auxiliary losses or statistics, and the system contract usually introduces expert parallelism and all-to-all communication — "tensor-shape contract unchanged" does not mean "training/system contracts unchanged" (the three-layer breakdown is in §9 Q28).
- **What changes is inside the FFN**: a router, expert selection (possibly plus a shared expert), and training-time auxiliary balancing losses/bias updates are added; **total parameters and per-token activated parameters must be reported separately** — the single most-probed follow-up in interviews.

The payoff intuition on this axis: Mixtral (Jiang et al., *Mixtral of Experts*, arXiv 2401.04088, 2024) activates about 12.9B parameters per token and matches or beats Llama 2 70B on most benchmarks reported in the paper. Total parameters are far above 13B; per-token compute is close to a dense model in the low tens of billions, but not strictly FLOPs-equivalent. The full "why sparse activation is a good idea" argument is in moe_tutorial.md §1.

> ⚠️ **Dense → MoE is not every modern block's "next step"**
> Representative models such as Llama 1–3, the original Mistral 7B, and Gemma 1–3 use dense FFN; Mixtral, Llama 4, and DeepSeek-V2/V3 use MoE — the latter trade the engineering complexity of communication/routing/load-balancing for larger sparse parameter capacity. V3's specific load-balancing mechanism comes from Wang, Gao, Zhao, Sun & Dai (arXiv 2408.15664, 2024); do not treat that mechanism paper as the main citation for "why V2/V3 chose MoE". Fusing Dense→MoE with §5's KV-cache axis into one "evolution chain" is the most common way to lose points here.

## §7 Numerical Stabilization / Connectivity Structure / Execution Overlay

This section reclassifies the previously scattered "stabilization tricks" into four categories using §0's layered map: ① residual dependency and norm/scaling assembly (the "single block" level); ② attention numerical-stability control (the "inside the operator" level, controlling numeric scale); ③ attention connectivity structure (also "inside the operator", controlling the visible range rather than numerics); ④ execution/adaptation overlay (the "execution/adaptation overlay" level, pointers only).

### 7.1　Residual dependency, norm placement, and branch scaling/gating

#### 7.1.1　Branch pre+post norm (often misnamed "sandwich norm")

The more accurate description is one normalization **before and after each sublayer branch**, with the second norm acting only on the branch output, not on the residual sum:

$$y = x + N_\text{post}\big(F(N_\text{pre}(x))\big)$$

This is not the same as original Post-LN's $N(x+F(x))$ — Post-LN wraps the normalization "after the residual add", while branch pre+post's second normalization wraps only "the branch output", leaving the residual trunk $x+(\cdots)$ pure addition. Gemma-2 uses this topology together with attention-logit/final-logit soft-capping; Gemma-3 keeps the norm topology but replaces soft-capping with QK-Norm (full variant genealogy in normalization_init_tutorial.md §6.2).

#### 7.1.2　Parallel residual (GPT-J, PaLM)

The core formula was given in §0: $y = x + \text{Attn}(N_A(x)) + \text{MLP}(N_M(x))$. **"Parallel" is defined as "no intra-layer ordering dependency"; it does not require attn and FFN to share one set of norm parameters** — two independent normalization modules also satisfy the definition as long as both read the original $x$ (§1 [D08] writes positive and negative proofs under this wider definition). GPT-J (Wang & Komatsuzaki, *GPT-J-6B*, EleutherAI open-source release, 2021, no formal paper/arXiv id) implements it with a single shared LayerNorm; PaLM (Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311, 2022) continues the same idea.

#### 7.1.3　Residual / branch scaling and gating: another independent assembly position

Some recipes decide not only "where to put the norm" but also multiply the residual branch by a fixed or learnable scale — e.g. **LayerScale** (multiply each branch output by a per-channel, learnable, small-initialized coefficient) or **DeepNorm** (multiply the residual branch by a depth-dependent fixed amplification factor, paired with a specific initialization so deep networks train more stably). This is another independent, optional assembly position: whether to scale, whether the scale is a fixed constant or a learnable parameter, and which branch it acts on — the full initialization and gradient arguments are in normalization_init_tutorial.md §7 (DeepNorm) and related sections. Here we record only the assembly contract: **residual/branch scaling and norm placement are two dimensions that can be chosen separately or used together** — do not merge them into one thing.

### 7.2　Attention numerical-stability control: attention scale, QK-Norm, logit soft-cap

Three mechanisms act at different positions on the same computation chain, all at the "inside the operator" level:

- **attention scale**: the standard $1/\sqrt{d_h}$ (or learnable temperature) scaling of the dot product controls the overall magnitude of the softmax input — the most basic numerical control, present in almost every implementation (derivation in §9 Q4).
- **QK-Norm** (Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245, 2020, EMNLP 2020 Findings): **before** the dot product, L2-normalize each head's Q and K separately, and **replace** the standard fixed $1/\sqrt{d_h}$ with a learnable scale/temperature — later models' implementation details may differ.
- **logit soft-capping** applies a bounded transform like $c\tanh(z/c)$ to attention scores or final vocabulary logits **after** the dot product, reining in the magnitude of scores/logits already computed.

Interviewees often blur the three into "all stabilize numerics", but they act at different positions and control different quantities, **and cannot be stacked blindly** — enabling them together requires re-checking the overall scale, not simple addition; the full "why attention logits blow up" argument is in normalization_init_tutorial.md §6.3.

### 7.3　Attention connectivity structure: local / sliding / global

This is the design axis of attention's **connectivity structure / visible range (connectivity pattern)** — not an "attention backend", and separate from positional-encoding schemes. FlashAttention is the typical execution backend: it changes exact attention's computation order and IO scheduling but defines no new connectivity graph or block topology (full IO-complexity derivation in attention_tutorial.md). Mistral (Jiang et al., *Mistral 7B*, arXiv 2310.06825, 2023) uses fixed-window sliding-window attention; Gemma-2 alternates local-window and global attention 1:1 across layers; Gemma-3 is closer to a periodic pattern of "several local layers then one global layer" (about 5:1, not strictly alternating). None of these change the block's outer residual interface; they only change "how far attention can see" inside — but the placement of "which layers are local, which are global" is itself a **stack-scheduling** decision (see the §0 layered map), not a single block's private attribute. Full sliding-window/StreamingLLM mechanics are in long_context_rope_yarn_mla_tutorial.md §10.

### 7.4　Execution / adaptation overlay: FlashAttention, quantization, LoRA (pointers only)

All three belong to the "execution/adaptation overlay" level of §0's layered map — not architectural slots on the same level as GQA/MoE requiring a mutually exclusive choice, but strategies stacked on top of an already-chosen architecture to accelerate it or adapt it to a deployment environment:

- **FlashAttention**: changes only exact attention's computation order, IO scheduling, and memory-access pattern; it does not change attention's mathematical result and defines no new connectivity graph — stackable on MHA/GQA/MQA/MLA and any local/global pattern. Full IO-aware complexity derivation in attention_tutorial.md.
- **Quantization**: represents weights/activations in lower-precision numeric formats — an overlay at the numeric-representation level, stackable with any architectural choice above, at the cost of quantization error; full error analysis in quantization_tutorial.md.
- **LoRA**: adds a pair of low-rank matrices beside trained weights for parameter-efficient fine-tuning — a training/adaptation-stage overlay that does not change the block's forward architecture; full low-rank formulas in lora_peft_tutorial.md.

None of the three should be treated as an architectural slot "on the same level as GQA and MoE, pick one of three" — they are independent dimensions stackable on almost any architectural combination.

### 7.5　Summary: how these mechanisms combine

The mechanism categories above often co-occur in one model in practice (e.g. Gemma-3 uses branch pre+post norm + QK-Norm + local/global interleaved attention simultaneously). A common checking order: first fix attention's visible range (global / local / sliding, §7.3), then the numerical stabilization of Q/K or logits (QK-Norm or soft-cap, §7.2), and last the norm topology and residual scaling (Pre-LN / branch pre+post / parallel, §7.1). Concrete values are further constrained by engineering factors — training stability, kernel support, compatibility with existing checkpoints.

> 🎯 **This section's interview answering principle**
> For any "new stabilization trick", first ask yourself two things: which tensor does it act on (Q/K? attention score? residual stream?), and does it change the block's input/output shape contract (usually not). Hanging the new term on these two questions makes the answer far more solid than merely listing "what it is".

## §8 Model Recipe Atlas

### 8.1　Block-recipe comparison table for mainstream models

Conventions: "QK-Norm" refers strictly to explicit per-head Q/K normalization before the dot product (MLA's internal latent-projection normalization does not count); "branch pre+post" means $y=x+N_\text{post}(F(N_\text{pre}(x)))$, not the same as original Post-LN; the Attention column lists only the KV-head sharing scheme, with local/sliding/global patterns in the Notes column; final norm and local/global placement belong to the stack-scheduling level (§0), and this table presents each model's overall recipe; **none** of these representative models' canonical recipes uses ALiBi as the primary positional scheme.

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

### 8.2　The twelve claims most likely to backfire

The high-frequency wrong answers from the whole tutorial, collected into one table for a final pre-interview sweep:

| Common claim | More accurate claim |
| --- | --- |
| Modern LLMs are what replaced Post-LN with Pre-LN | GPT-2 is already Pre-LN; this step happened earlier (§3.2) |
| Decoder-only is just "the decoder with cross-attn removed" | Correct at the single-layer structural level; at the whole-model level also add — the encoder/memory path disappears too, and two-stream conditional generation becomes single-stream causal modeling (§3.1) |
| The residual stream's variance inflates $\propto\sqrt l$ with depth | Variance $\propto l$ (linear); it is the standard deviation/RMS that goes $\propto\sqrt l$ — under the independent-increment assumption (§0, §9 Q11) |
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

Three difficulty tiers; expand each for the answer key (the 10 questions that provide a discrimination action end with a pitfall note). L2/L3 are top-lab deep water (topology forks, assembly-axis independence, system correctness, cross-model recipe comparison). When answering, **do not repeat the deep math derivations from the sibling tutorials**; organize as "structural judgment → key formula → common mistake".

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
- The KV cache reduces per-step recomputation of historical K/V from $O(t^2)$ to $O(t)$ ($t$ is the current position; full bottleneck analysis in llm_inference_serving_tutorial.md)

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
