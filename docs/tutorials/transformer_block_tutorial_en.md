## §0 Mental Model + TL;DR Cheat Sheet

**A Transformer block isn't a single formula — it's a set of layered assembly rules.** Most interview confusion ("Pre-LN is the next generation after Post-LN", "MQA→GQA→MLA is one upgrade chain", "decoder-only is just decoder with a piece removed") comes from compressing this layered rule set into a single linear timeline. Start with the mental model: the residual stream $x_l$ passes through two or three **sublayers** in sequence (attention, FFN; the original encoder-decoder has one more, cross-attn), defaulting to a **sequential** dependency — the FFN reads the residual stream *after* attention has updated it (parallel is the exception, see below):

```text
residual stream x_l
  │
  ├─▶ [norm slot: Pre／Post／branch pre+post] → Attention slot (self-/cross-attention,
  │     MHA/GQA/MQA/MLA, ±RoPE, ±QK-Norm, global/local window) → [norm slot] ──▶ ADD back to residual stream
  │     giving intermediate state u = x_l + Attn(N(x_l))
  │
  u ─▶ [norm slot] → FFN slot (Dense GELU / Gated SwiGLU / MoE) → [norm slot] ──▶ ADD back to residual stream
  │     (sequential default: FFN reads u, not x_l; parallel variant: FFN and attention both read the same N(x_l), not u)
  │
residual stream x_{l+1}
```

**The slots aren't strictly mathematically orthogonal — they sit at different levels; choices within a level can mostly be analyzed independently, but across levels there is real coupling.** This is the key difference between this tutorial and simplified "three orthogonal axes" framings. A few concrete counterexamples: the latent representation MLA chooses in turn constrains how RoPE gets split apart; norm topology and norm operator show up clearly bundled together in practice (Pre-LN with RMSNorm, branch pre+post with RMSNorm are common pairings, not arbitrary combinations); QK-Norm, attention scale, and soft-cap cannot be stacked mindlessly — the scale needs to be re-checked; and the claim "the KV axis doesn't touch total model parameter count" doesn't hold up either — GQA's parameter formula clearly shows it does change the projection parameters. A layered map is closer to the truth than "orthogonal axes":

| Level | Typical decisions | Who owns it |
| --- | --- | --- |
| Inside the operator | QKV projection, RoPE, QK-Norm, head sharing (MHA/GQA/MQA/MLA), gated FFN | The attention/FFN module itself |
| Single block | serial/parallel dependency graph, norm placement (Pre/Post/branch pre+post), residual add, branch scaling/gating | The block's `forward` |
| Stack scheduling | final norm, local/global alternation, which layers are dense vs. MoE | The stack assembly code, not any one block |
| Model shell | token embedding, external absolute PE, weight tying, LM head (incl. soft-cap) | The whole model's input/output layers |
| Execution/adaptation overlay | KV cache, FlashAttention, packed projection, quantization, LoRA | Deployment/fine-tuning-stage execution strategy — does not change the architectural semantics of the four levels above |

From here on, any judgment about whether a choice belongs "inside the block, at the stack level, or in the model shell" is anchored to this table, rather than being lumped together as vaguely "some slot."

> 💡 **Four residual-topology formulas, memorize on one screen, all unified at "full block" granularity** (§1–§7 below are all concrete instantiations of these four formulas; $A$ and $M$ are the attention and FFN sublayers respectively, and $N_1,N_2,N_A,N_M,N_1^{\text{pre}},N_1^{\text{post}},N_2^{\text{pre}},N_2^{\text{post}}$ are all normalization operators)
>
> $$\text{Post-LN (Vaswani et al. 2017, the original Transformer)}:\quad u = N_1\big(x + A(x)\big),\quad y = N_2\big(u + M(u)\big)$$
> $$\text{Pre-LN sequential (adopted at least as early as GPT-2, later the common form in modern LLMs)}:\quad u = x + A\big(N_1(x)\big),\quad y = u + M\big(N_2(u)\big)$$
> $$\text{Pre-LN parallel (GPT-J / PaLM)}:\quad y = x + A\big(N_A(x)\big) + M\big(N_M(x)\big)$$
> $$\text{branch pre+post (Gemma-2/3)}:\quad u = x + N_1^{\text{post}}\big(A(N_1^{\text{pre}}(x))\big),\quad y = u + N_2^{\text{post}}\big(M(N_2^{\text{pre}}(u))\big)$$
>
> The key difference between the four isn't as simple as "the LN moved position": in Post-LN the residual backbone itself is repeatedly squeezed by LN; Pre-LN sequential leaves the residual backbone a pure identity-addition channel, but the FFN reads the stream $u$ **after** attention has updated it; Pre-LN parallel removes even that ordering dependency — attention and FFN read the **same** $x$ ($N_A,N_M$ can share parameters or be two independent modules; "no ordering dependency" is what defines parallel, not sharing the norm); branch pre+post places one normalization before and one after each sublayer branch, with the second normalization applied only to the branch output, so the residual backbone $x+(\cdots)$, $u+(\cdots)$ is still a pure addition — completely unlike Post-LN, which wraps normalization around "after the addition." All four topologies have executable proof-and-counterproof code in [D02]/[D08] in §7. **Where the norm is placed (Post/Pre/branch pre+post) and whether the serial/parallel dependency is decoupled are two dimensions that can be discussed separately — don't merge them into one axis.**

**12 sentences to nail the Transformer block**:

1. **Block = residual stream + hierarchically assembled sublayers**: the internal variants of attention and FFN are each independent choices within the same level; encoder-decoder has one more sublayer than decoder-only — cross-attn.
2. **decoder-only: at the single-layer level, "cross-attn removed" is correct, but that's not enough at the whole-model level**: the original decoder layer has three sublayers — masked self-attn, cross-attn, FFN; a modern LLM decoder-only block usually has only two — causal self-attn, FFN. This step, "removing cross-attn," is fine on its own; but at the full-model level you also need to add: the encoder/memory path disappears along with it, and dual-sequence conditional generation becomes single-sequence causal modeling — a change in information-flow topology and sequence factorization (§2).
3. **GPT-2 is already Pre-LN**: going from GPT-2-style to Llama-style, what actually changes is LayerNorm→RMSNorm, learned absolute PE→RoPE (moved inside attention), GELU dense FFN→SwiGLU gated FFN, MHA→GQA where applicable, plus dropping most bias/dropout (§2.2, §3).
4. **Pre-LN stacks usually end with a final norm**: under a simplified independent-increment assumption, the residual stream's **variance** grows roughly **linearly** with depth ($O(l)$), so it's the standard deviation/RMS/typical magnitude that grows roughly as $\sqrt l$; the real growth rate also depends on initialization, residual scaling, and branch correlation. Feeding straight into the output head without normalizing the accumulated residual stream typically hurts training quality noticeably — this is a common configuration in standard recipes like GPT-2/Llama, but not a mathematically indispensable condition for every Pre-LN architecture (§2.2; derivation in normalization_init_tutorial.md §5.3).
5. **MHA→MQA→GQA→MLA is not one evolutionary chain**: by historical timeline it is MHA→MQA→GQA (MQA genuinely predates GQA); but by KV-head compression strength, MHA and MQA are the two endpoints of the same GQA compression-ratio axis; MLA is an independent low-rank latent branch from the DeepSeek line and should not be tacked onto the end of that axis — there are multiple distinct ordering dimensions, not just one timeline (§4).
6. **Dense FFN → MoE is another, independent "capacity axis"**: it is not "the inevitable next generation" — representative models like Llama 1–3, the original Mistral, and Gemma 1–3 still mostly use dense FFN, while Mixtral, Llama 4, and others have adopted MoE (§5).
7. **QK-Norm, logit soft-cap, branch pre+post norm, parallel residual, and local/sliding attention can mostly be combined as needed**, but not stacked mindlessly — using QK-Norm and soft-cap together requires re-checking the scale, and norm topology and norm operator show up clearly bundled together in practice (§6).
8. **FFN parameter matching is computed, not a law**: setting dense FFN ($8d^2$) and gated FFN ($3dh$) parameter counts equal gives $h=8d/3$; real models also round to hardware-friendly multiples, so not every Llama-family model strictly equals this number (§3, §7 [D06]).
9. **KV cache mainly tracks $n_\text{kv}$, not $n_q$**: MQA/GQA reduce the parameters, compute, and cache memory/bandwidth of the K/V projection, but do not reduce the compute of the Q projection (§4; full memory accounting in attention_tutorial.md §6 and kv_cache_speculative_decoding_tutorial.md §2).
10. **Modern recipes haven't fully converged**: Pre-RMSNorm + RoPE + gated FFN + inference-friendly GQA is just the "more common" combination; Post-LN variants, LayerNorm, MHA, MQA, parallel residual, dense FFN, and MoE are all still used in production (§8 model recipe atlas).
11. **The only structural difference between cross-attention and self-attention is where Q/K/V come from**: in self-attn, Q, K, V all come from the same stream; in cross-attn, Q comes from the current stream while K/V come from another sequence, which can have a different length — apart from that, scaled dot-product, multi-head splitting, and softmax are exactly the same; no separate math is needed (§1).
12. **"Packed projection" is just a kernel-level equivalent rearrangement**: concatenating the three Linears for Q/K/V (or SwiGLU's gate/up) into one large Linear is purely an implementation optimization — after row-concatenating the weights, the results should match at the given precision, and no architectural semantics change. When you hit this kind of question, first confirm whether it's asking about engineering implementation or algorithmic design (§7 [D07]).

> 🎯 **Scope of this tutorial: only "assembly," not re-deriving the math inside each part**
> The derivation of the softmax-attention formula, FlashAttention's IO complexity, the gradient argument behind Xiong/DeepNorm, the full math of MoE routing/capacity, the RoPE rotation matrix and YaRN/NTK/MLA latent algebra, linear-attention derivations, quantization error derivations, LoRA's low-rank formula — each of these parts' own math has a dedicated section in attention_tutorial.md, normalization_init_tutorial.md, moe_tutorial.md, long_context_rope_yarn_mla_tutorial.md, linear_sparse_attention_tutorial.md, quantization_tutorial.md, and lora_peft_tutorial.md; prefill/decode system-performance bottlenecks and KV-cache paging/scheduling are in llm_inference_serving_tutorial.md. This tutorial only cares about **how these parts get assembled into a block, and how assembly evolved from the 2017 version all the way to the modern version** — whenever a deep-math or systems question comes up, it gets a 1–2 sentence conclusion plus a link, not a repeated derivation.

## §1 The 2017 seq2seq Transformer

The original Transformer (Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762, 2017, NeurIPS 2017) is an **encoder-decoder** architecture: the encoder encodes the source sequence into a set of representations, and the decoder autoregressively generates the target sequence conditioned on those representations. Positional encoding uses fixed sinusoidal PE, added directly to the token embedding **before** entering the encoder/decoder stacks — it doesn't belong to any block, it's input processing that sits outside the block (full derivation in long_context_rope_yarn_mla_tutorial.md §2/§3). The paper's original configuration is $d_\text{model}=512$, $d_\text{ff}=2048$ (4× expansion), 8 heads, 6 encoder layers + 6 decoder layers; this tutorial's demonstrations use smaller sizes (e.g. $d=96$), which is a **teaching configuration**, not the paper's configuration — the two should not be conflated.

**The encoder layer and decoder layer are two different classes, not two ways of calling the same "two-sublayer" template** — they differ in both the number and the kind of sublayers:

| | Encoder layer | Decoder layer |
| --- | --- | --- |
| Sublayer 1 | Self-attn (bidirectional, no causal mask, optional padding mask only) | Masked self-attn (causal) |
| Sublayer 2 | FFN | Cross-attn (Q from this layer's output, length $T$; K/V from encoder output, length $S$) |
| Sublayer 3 | — | FFN |
| Norm topology | Post-LN per sublayer: $y=N(x+F(x))$ | Post-LN per sublayer, three times |

The decoder's three sublayers have a fixed order — masked self-attn → cross-attn → FFN — with each sublayer wrapped in its own Post-LN. Cross-attn's Q, K, V sources are asymmetric: Q comes from the stream the decoder is itself currently generating (length $T$), while K, V come from the encoder output (length $S$); $T$ and $S$ are generally not equal — this is exactly the shape separation explicitly verified in §7's code [D01] (formula details and mask handling for self-attn vs. cross-attn are in attention_tutorial.md §4.1 and are not re-derived here).

**Walking through a concrete shape** (matching the teaching sizes used in §7's code [D01]: batch $B=2$, source sequence length $S=5$, target sequence length $T=7$, $d=96$): the encoder input is `[B, S, d]`; after several encoder layers it's still `[B, S, d]`, passed as memory to every decoder layer; the decoder input is `[B, T, d]`; in the masked self-attn stage, Q/K/V are all computed internally as `[B, T, d]` (causal mask size $T\times T$); in the cross-attn stage, Q is still `[B, T, d]`, but K, V come from the encoder memory with shape `[B, S, d]` — the attention score matrix is $T\times S$ (not square), and softmax is taken along the $S$ dimension; after the output projection it's restored to `[B, T, d]`, matching the self-attn sublayer's output shape so it can be fed into the FFN sublayer. **$T \ne S$ has zero effect on this data path** — this is exactly what "cross-attn decouples Q length from K/V length" looks like in concrete terms.

> ⚠️ **Two classes, not two calling conventions of one class**
> `VanillaEncoderLayer2017` (2 sublayers) and `VanillaSeq2SeqDecoderLayer2017` (3 sublayers) are two separate classes in §7's code. Writing them as one "generic block class + parameter switches" is feasible in engineering terms, but it obscures the structural fact that the decoder has one entire extra sublayer — cross-attn — compared to the encoder. When whiteboard-coding this live in an interview, laying out each class's sublayer list clearly scores better than jumping straight into code.

## §2 A map of the three families + the decoder-only fork + the GPT-2-style bridge

### 2.0　A map of the three families: encoder-only / decoder-only / encoder-decoder

The original Transformer is encoder-decoder; from here on, this tutorial focuses entirely on **decoder-only** (the mainstream choice for GPT/Llama/Qwen-family LLMs), but interviews often ask for a side-by-side comparison of all three families, so here's a compressed map first, without going deep into encoder-only's internal pretraining mechanics:

| Family | Representatives | Sublayer composition | Attention direction | Typical training objective | Typical use |
| --- | --- | --- | --- | --- | --- |
| Encoder-only | BERT, RoBERTa | Each layer: self-attn + FFN (no causal mask) | Bidirectional (sees the whole sequence) | Denoising objectives such as masked language modeling (MLM) | Representation learning, classification, retrieval embeddings |
| Encoder-decoder | Original Transformer, T5 | Encoder layer: self-attn + FFN; decoder layer: masked self-attn + cross-attn + FFN | Encoder bidirectional, decoder unidirectional and conditioned on the encoder | Conditional generation (translation, summarization) | Dual-sequence conditional generation tasks |
| Decoder-only | GPT, Llama, Qwen | Each layer: causal self-attn + FFN (no cross-attn) | Unidirectional (causal) | Autoregressive language modeling | General-purpose generative LLM |

The core difference among the three isn't "how many parameters" — it's that attention's visible direction, sublayer composition, and training objective all change together as a bundle. This tutorial only expands the full assembly details for the decoder-only branch (§2–§9); encoder-only's bidirectional masked-modeling mechanics and encoder-decoder's full training objective are out of scope here — refer to each model's original paper when needed.

### 2.1　Why saying only "cross-attn removed" isn't enough

The decoder-only block used by modern LLMs (GPT, Llama, Qwen, ...) can be viewed literally as "the original decoder layer with cross-attn removed." **Look at this claim on two levels**:

- **At the single-layer structural level, this claim is correct**: the original decoder is masked self-attn → cross-attn → FFN (**three** sublayers); the decoder-only block is causal self-attn → FFN (**two** sublayers). What's missing is exactly the cross-attn sublayer — a layer-by-layer comparison holds up fine.
- **At the full-model level, this claim is incomplete**: saying only "cross-attn removed" makes it easy to assume just one component was deleted and everything else stayed the same. In reality the entire encoder/memory path disappears along with it — source/prompt and target go from "dual-stream conditional generation" (source encoded, then target generated conditioned on it) to "single-stream causal modeling" (prompt and generated content concatenated into one causal sequence). What changes is the information-flow topology and the sequence factorization, not that the task got swapped for a different problem.

So a more complete answer is: "at the single-layer level it is indeed cross-attn removed, but at the full-model level you also need to spell out that the encoder/memory path disappears with it, and dual-stream conditional generation becomes single-stream causal modeling" — this is more accurate than either just saying "part of it was removed" or, at the other extreme, framing it as "a completely different architecture."

One boundary case worth adding: not every decoder-only LLM is entirely free of cross-attention — some multimodal or retrieval-augmented models reintroduce cross-attention on top of a pure decoder-only backbone (e.g., letting the text stream attend to an image encoder's output, or to retrieved document representations). Every "decoder-only has no cross-attn" statement in this tutorial and below is scoped to a **pure decoder-only block** (causal self-attn + FFN, two sublayers) and does not cover these hybrid architectures.

> 🎯 **How to answer this fundamental-difference question in an interview**
> Answer it in layers: at the single-layer structural level it's "cross-attn removed" (3 sublayers become 2); at the full-model level, add the information-flow topology change — "source/target as two sequences becomes one causal sequence"; then add "only a pure decoder-only block has no cross-attn; some multimodal/retrieval-augmented models reintroduce it" — this is two levels more accurate than just saying "part of it was removed."

### 2.2　The GPT-2-style bridge: what actually changed from GPT-2 to Llama

**Common misconception**: it was modern LLMs that switched Post-LN to Pre-LN. **Fact**: GPT-2 (Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report, 2019, no arXiv id) is already Pre-LN, and it puts a final LayerNorm (denoted `ln_f`) at the end of the stack — this is a common configuration in standard Pre-LN recipes like GPT-2/Llama: without adding this layer, the distribution of the inflated Pre-LN residual stream would feed straight into the output head, and training quality typically drops noticeably; but this isn't a mathematically indispensable condition for every Pre-LN architecture — variants without a traditional final norm can be constructed via residual scaling, special initialization, or a normalized output head (derivation in normalization_init_tutorial.md §5.3, not repeated here). The full GPT-2 block recipe:

$$x = x + \text{Attn}\big(\text{LN}(x)\big), \qquad x = x + \text{FFN}\big(\text{LN}(x)\big), \qquad \text{(GELU dense FFN, MHA, bias=True, no cross-attn)}$$

Learned absolute positional embeddings are still added to the embedding, **outside** the stack — this matches the "level it belongs to" for the original sinusoidal PE; neither belongs inside a block. RoPE, adopted by the LLaMA family (originally proposed by Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2104.09864, 2021; LLaMA is a major adopter, not the originator), is the opposite — it sits **inside attention**, acting only on the Q, K dot-product computation. Treating these as a "seamless swap" between members of the same block is inaccurate: positional information is carried by an external embedding in GPT-2-style and by a rotation inside attention in Llama-style — the two live at different levels (full derivation in long_context_rope_yarn_mla_tutorial.md §2).

The GPT-2-style → Llama-style bridge table (this is where the real changes are):

| Dimension | GPT-2-style | Llama-style |
| --- | --- | --- |
| Norm topology | Pre-LN (already) | Pre-LN (unchanged) |
| Norm type | LayerNorm | RMSNorm |
| Positional encoding | learned absolute (outside the stack, added to embedding) | RoPE (inside attention, acts only on Q/K) |
| FFN | GELU dense (2 matrices) | SwiGLU gated (3 matrices) |
| Attention | MHA | MHA or GQA (per checkpoint spec) |
| bias | Linear and LayerNorm both have bias | Usually all bias removed |
| dropout | Used during training | Usually removed or reduced to near-zero for large-scale pretraining |
| final norm | Present (`ln_f`, LayerNorm) | Present (RMSNorm) |

> ✅ **The standard-answer skeleton for "what changed from GPT-2 to Llama"**
> First state that "both are already Pre-LN" (this blocks the common wrong answer "Pre-LN is a modern invention"), then compare five items in order — norm type → the level positional encoding belongs to → FFN structure → attention head configuration → bias/dropout — and finish with the note that both sides have a final norm, just of different types.

## §3 The Llama-style core recipe

Building on the GPT-2-style bridge, Llama-style (Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288, 2023; Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783, 2024) swaps each slot for the following concrete implementation. This section fixes MHA first to keep the data path clear; GQA is expanded separately in §4:

$$x = x + \text{Attn}_{\text{RoPE}}\big(\text{RMSNorm}(x)\big), \qquad x = x + \text{SwiGLU}\big(\text{RMSNorm}(x)\big)$$

- **RMSNorm**: only re-scales, does no mean-centering, no bias — the full formula and the argument for "why dropping centering can still often keep comparable quality" are in normalization_init_tutorial.md §4 (Zhang & Sennrich's theoretical motivation and experimental results show this **often** matches LayerNorm's quality, not a universal "never hurts" claim); here we only record the conclusion: $y = \dfrac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\odot w$.
- **RoPE**: acts on Q, K inside attention; V is never rotated. Standard implementations usually cache the **already-rotated** K — during incremental decoding the new token is rotated at its true position and can be used directly in the dot product. If an implementation instead chooses to cache the unrotated K, it must additionally keep each position and rotate at read time using the correct position — it cannot treat historical K as if at the current position (this is mathematically just as correct, only slower, hence not the common choice; MLA's cache contract is entirely different again, see §4.2). Full complex-number-view derivation, frequency choice, and YaRN/NTK extensions are in long_context_rope_yarn_mla_tutorial.md §2/§5/§6; here we take only the "assembly contract."
- **SwiGLU**: $\text{down}\big(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)\big)$, a three-matrix gated FFN; GeGLU has the same structure with a GELU gate (Shazeer, *GLU Variants Improve Transformer*, arXiv 2002.05202, 2020). At comparable parameter count/compute budget, gated FFN **tends** to give better quality — this is an empirical regularity rather than a strictly proven law; this tutorial does not revisit that argument, and only uses its parameter-matching arithmetic (§7.3 [D06]).
- **bias, dropout**: bias is usually removed entirely from Q/K/V/O and the three FFN matrices; large-scale pretraining commonly removes or drastically reduces dropout.
- **final norm**: one RMSNorm at the end of the stack, normalizing the inflated residual stream before it hits the output head.
- **Numerical-precision boundary**: computations sensitive to numeric range — RMS/variance reduction, softmax, final logits — are commonly done in higher precision (e.g. fp32) in real implementations, then cast back to the activation dtype (bf16/fp16). This is part of getting the assembly right, not quantization math; full low-precision error analysis is in quantization_tutorial.md.

> ⚠️ **Inside the block vs. the "block-external recipe" — don't conflate them**
> Weight tying (input embedding and LM head sharing parameters), embedding scaling, LM head soft-cap, and the global dropout on/off switch are all part of the "full LM recipe," but **strictly speaking they sit outside the block** — token embedding and the LM head don't belong to any Transformer block. If, when asked "what does a block look like," you fold these into the block definition, it makes the interviewer doubt whether you've separated "one block layer" from "the whole model."

Spelling out exactly which fields the "block-external recipe" contains:

| External field | One-line description |
| --- | --- |
| Weight tying | The input token-embedding matrix and the LM-head projection matrix share the same parameters, saving a $Vd$-sized set of parameters |
| Embedding scaling | Some implementations multiply the token embedding by a constant related to $\sqrt{d}$ before feeding it into the first block |
| LM head soft-cap | Some models (e.g. the Gemma family) also apply a §6.2-style bounded transform to the final vocabulary logits — mechanistically the same class of tool as attention-logit soft-cap, applied to a different tensor |
| Global dropout switch | Whether dropout is enabled, and the dropout rate, are usually training hyperparameters unified across the whole model; but the dropout **operation itself** typically executes inside attention weights, FFN outputs, and residual branches — "who configures it" and "where the operator actually runs" are two different things; don't lump them together as "a property outside the block" |

None of these four items **appear** in any block class described in §7/§8 — `ModernDecoderBlock`'s `forward` only receives the residual stream `x`, and never knows whether embedding scaling happened outside, or whether the output head is tied.

This table lines up with §0's layered map: weight tying, embedding scaling, and the LM head belong to the "model shell" level; final norm, local/global alternation, and the dense/MoE layer layout are cross-layer schedules that belong to the "stack scheduling" level — these cross-layer schedules belong neither to any single block nor to the model shell. These are three distinct levels — don't lump them together as "stuff outside the block."

## §4 The inference-memory axis: MHA → MQA/GQA, with MLA as an independent branch

**The problem this axis primarily solves is KV-cache memory**: during autoregressive generation, every step must append the new token's K, V into the cache and then run attention over the whole cache — the cache's size directly determines how many requests a single GPU can serve concurrently and how long a context it can sustain. This axis and §5's "capacity axis" and §6's stabilization mechanisms can mostly be discussed separately at the analysis level, but they are not strictly independent — while GQA/MQA reduce the number of KV heads, they also change the parameter count and compute of the K/V projection itself (the $2d^2(1+n_\text{kv}/n_q)$ formula below is the evidence). "KV memory" and "total model parameter count" are not two completely unrelated things — it's just that their magnitude and mechanism of impact differ: KV cache mainly affects inference-time memory/bandwidth, while K/V projection parameters are a fixed overhead that exists during training too.

### 4.1　MHA / MQA / GQA: different compression ratios on the same axis

The historical order needs to be clarified:

> ⚠️ **Not one line MHA→GQA→MQA→MLA**
> Correct order: MHA is the baseline; **MQA (Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv 1911.02150, 2019) appeared before GQA**; GQA (Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023) is a quality/memory trade-off between MHA and MQA; **MLA is a separate low-rank latent-compression branch from the DeepSeek line**, not "shrinking $n_\text{kv}$ one step further." It's better to think of the four as coexisting design points on the single "KV memory" design axis, rather than historical rungs that every model climbs through in sequence.

| Variant | Q heads | KV heads | Relationship |
| --- | --- | --- | --- |
| MHA | $n_q$ | $n_q$ | Baseline, $n_\text{kv}=n_q$ |
| MQA | $n_q$ | $1$ | All Q heads share the same K/V |
| GQA | $n_q$ | $n_\text{kv}$ ($1 \lt n_\text{kv} \lt n_q$, and $n_q \bmod n_\text{kv}=0$) | Each group of $n_q/n_\text{kv}$ Q heads shares one K/V |

The assembly contract (full memory accounting, the per-token-per-layer formula, and concrete model GB figures are in attention_tutorial.md §6 and kv_cache_speculative_decoding_tutorial.md §2, not re-derived here):

- KV-cache size scales only with $n_\text{kv}$, **independent of $n_q$** — MQA/GQA reduce the K/V projection's own parameters and compute, and also reduce KV-cache memory/bandwidth; but the Q projection's compute doesn't change with $n_\text{kv}$, so it's not accurate to say broadly that this "only saves memory/bandwidth."
- **Standard implementations usually cache K after applying RoPE; V is never rotated.** During incremental decoding, the new token's Q, K have RoPE applied at their true position offset, and the already-rotated K in the historical cache is never recomputed — if an implementation instead chooses to cache unrotated K, it must additionally keep position and rotate at read time using the correct position, otherwise all historical keys get treated as the same position (mathematically feasible but slower, hence not the common choice).
- GQA/MQA's cache **stores only $n_\text{kv}$ heads**; broadcasting K/V to $n_q$ Q heads can only happen inside the attention computation (via view/reshape/grouped einsum) — **the broadcasted tensor must never be written back to the cache**. This is the easiest pitfall to hit in a hand-written implementation; §7 [D05] turns this directly into an executable assertion.

A concrete example of this axis's real payoff: for LLaMA-2-70B with vanilla MHA, a single sample's KV cache at a 4096 context is about 10 GB; switching to GQA ($n_\text{kv}=8$) brings it down to about 1.25 GB — this shows why GQA is very attractive for serving 70B-scale models, and is one important reason many large models adopt GQA over MHA (the final choice for any given model is also shaped by quality, training compatibility, serving-system factors, and is not decided by this single factor alone; precise accounting is in attention_tutorial.md §6.1 and kv_cache_speculative_decoding_tutorial.md §2.2, not re-derived here).

### 4.2　MLA: an independent low-rank latent branch

**MLA (DeepSeek-V2, May 2024, arXiv 2405.04434) is an independent branch, not "a harsher version of GQA"**: GQA compresses along the head dimension (multiple Q heads share a K/V head), while MLA compresses via a **low-rank projection** along the hidden dimension (compressing each token's K/V into a latent vector of size $d_c \ll n_q d_h$), and it must separately decouple RoPE into a shared, small-dimensional component in order to preserve the "absorb the up-projection matrix into the query side" inference-acceleration trick — the full derivation (the absorbing trick, why RoPE can't be absorbed directly, the decoupling scheme, and the total-cache formula) is in long_context_rope_yarn_mla_tutorial.md §9; here we only record the conclusion: MLA's cache contract is "store the latent $c_t^{KV}$ plus one shared RoPE key," not "store fewer K/V heads."

> 💡 **A one-line way to tell what each of the three compresses**
> GQA compresses **head count** (multiple Q heads squeezed onto one K/V group); MQA is GQA's extreme case ($n_\text{kv}=1$); MLA compresses **each token's representation dimension** (K/V projected as a whole into a low-rank latent), and naturally has to solve RoPE's position-dependence problem along the way. The three aren't three points on the same scale — they're two different compression ideas.

## §5 The capacity axis: Dense FFN → MoE

**The problem this axis solves is model capacity**, and it can mostly be analyzed separately from §4's memory axis and §6's stabilization mechanisms, but again, they are not strictly orthogonal — MoE's external-facing **main activation tensor** interface is indeed unchanged ($[B,T,d]\to[B,T,d]$), but the training contract additionally has to pass along router loss/statistics, and the systems deployment contract also involves expert parallelism and all-to-all communication — none of which is captured by "the interface is completely unchanged" (details below and in §7 [D09]). MoE replaces the FFN with $N$ experts plus a router, and each token only goes through $k \ll N$ experts — total parameters rise while activated parameters per token stay the same. The full routing formulas (token-choice top-k, expert-choice, DeepSeek's aux-loss-free bias update), capacity factor, load-balancing loss, and token dropping are in moe_tutorial.md §2–§4; here we only cover the "assembly contract":

- **The block's external-facing main activation tensor interface is unchanged**: whether the FFN is dense or MoE, it's `[B, T, d] -> [B, T, d]`, and the residual-add position doesn't change — this is the direct reason §7 [D09] can swap `ffn` for `MoEFFN` without changing a single line of `ModernDecoderBlock` code; but during training, the return value usually gains an extra router/load-balancing-related auxiliary loss or statistic, and distributed deployment usually needs to introduce expert parallelism and all-to-all communication — "the tensor-shape contract is unchanged" and "the training/systems contract is unchanged" are two different things (§9 has a dedicated three-layer contract breakdown).
- **What changes is inside the FFN**: a router and expert selection (possibly with a shared expert) are added, along with an auxiliary balancing loss/bias update during training; **total parameter count and per-token activated parameter count must be reported separately** — this is the line most likely to get followed up on in an interview.

The intuition behind this axis's payoff: Mixtral (Jiang et al., *Mixtral of Experts*, arXiv 2401.04088, 2024) activates about 12.9B parameters per token, and matches or beats Llama 2 70B on most/all of the benchmarks reported in its paper — its total parameters are far more than 13B, and its per-token compute scale is roughly comparable to a dense model with more than ten billion parameters, though it is not strictly FLOPs-equivalent (the full three-layer argument for "why sparse activation is a good idea" is in moe_tutorial.md §1).

> ⚠️ **Dense → MoE is not the "next step" for every modern block**
> Representative models like Llama 1–3, the original Mistral 7B, and Gemma 1–3 use dense FFN; but Mixtral, Llama 4, and others have adopted MoE. DeepSeek-V2 (DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv 2405.04434, 2024) and DeepSeek-V3 (DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437, 2024) chose MoE to trade the engineering complexity of communication/routing/load-balancing for much larger sparse parameter capacity; the specific load-balancing mechanism V3 adopts comes separately from Wang, Gao, Zhao, Sun & Dai, *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* (arXiv 2408.15664, 2024), and this mechanism paper shouldn't be treated as the primary citation for "why V2/V3 chose MoE" itself. It's not a road every model "eventually has to take" — mixing Dense→MoE together with §4's KV-cache axis into one single "evolutionary chain" is the most common point where credit is lost in this part.

## §6 Numerical stabilization / connectivity structure / execution overlays

This section reorganizes the previously scattered "stabilization tricks" into four categories per §0's layered map: ① residual dependency and norm/scaling assembly (the "single block" level); ② attention numerical-stability control (the "inside the operator" level, controlling numeric scale); ③ attention connectivity structure (also the "inside the operator" level, but controlling visibility range rather than numerics); ④ execution/adaptation overlays (the "execution/adaptation overlay" level, pointer-only). These four categories can mostly be analyzed independently, but they can't be stacked arbitrarily and unconditionally — real constraints often exist within the same category (e.g., using QK-Norm and soft-cap together requires re-checking the scale).

### 6.1　Residual dependency, norm placement, and branch scaling/gating

#### 6.1.1　Branch pre+post norm (often colloquially mislabeled "sandwich norm")

The more accurate way to write it is: **one normalization before and one after** each sublayer branch, where the second norm acts only on the branch output, not on the residual sum:

$$y = x + N_\text{post}\big(F(N_\text{pre}(x))\big)$$

This is not the same thing as the original Post-LN's $N(x+F(x))$ — Post-LN's normalization wraps around "after the residual addition," while branch pre+post's second normalization wraps only around the "branch output," so the residual backbone $x+(\cdots)$ is still a pure addition. Gemma-2 uses this topology together with attention-logit/final-logit soft-capping; Gemma-3 keeps this norm topology but replaces soft-capping with QK-Norm (full variant lineage in normalization_init_tutorial.md §6.2).

#### 6.1.2　Parallel residual (GPT-J, PaLM)

The core formula was already given in §0: $y = x + \text{Attn}(N_A(x)) + \text{MLP}(N_M(x))$ — attention and FFN have no within-layer ordering dependency, both reading the same normalized input. GPT-J (Wang & Komatsuzaki, *GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model*, EleutherAI open-source release, 2021, no formal paper/arXiv id) implements this formula with a single shared LayerNorm; PaLM (Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311, 2022) follows the same idea. **"Parallel" describes the structural relationship of "no within-layer ordering dependency" — it does not require the code to make attn and FFN share the same norm parameters**; using two independent normalization modules, as long as both read the original $x$, equally satisfies the definition of parallel (the proof-and-counterproof in §7 [D08] is written against this broader definition).

#### 6.1.3　Residual/branch scaling and gating: another independent assembly slot

Some recipes decide not just "where to put the norm" but also multiply the residual branch by a fixed or learnable scaling coefficient — for example **LayerScale** (multiplying each branch output by a per-channel, learnable, small-initial-value coefficient) or **DeepNorm** (multiplying the residual branch by a fixed, depth-dependent amplification coefficient, paired with a specific initialization to make very deep networks easier to train stably). This is another independent, optional assembly slot: whether to scale at all, whether the scale is a fixed constant or a learnable parameter, and which branch it acts on — the full initialization and gradient argument is in normalization_init_tutorial.md §7 (DeepNorm) and related sections; this tutorial only records the assembly contract: **residual/branch scaling and norm placement are two dimensions that can be chosen separately, or used together** — don't merge them into a single thing.

### 6.2　Attention numerical-stability control: attention scale, QK-Norm, logit soft-cap

Three mechanisms act at different points along the same computational chain, all belonging to the "inside the operator" level:

- **Attention scale**: the standard $1/\sqrt{d_h}$ (or learnable temperature) scaling of the dot product, controlling the overall magnitude of the softmax input — the most basic numerical control, present in almost every implementation (derivation in §9 Q4).
- **QK-Norm** (Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245, 2020, EMNLP 2020 Findings) is originally defined as doing L2 normalization on each head's Q, K separately **before** the dot product, and **replacing** the standard fixed $1/\sqrt{d_h}$ scaling with a learnable scale/temperature parameter — the exact implementation details in later models may differ (e.g., whether a fixed scale is retained, or whether it's additionally paired with RMSNorm-style normalization).
- **Logit soft-capping** applies a bounded transform resembling $c\tanh(z/c)$ **after** the dot product, to the attention score or the final vocabulary logits, reining in the magnitude of scores/logits that have already been computed.

Interviewees often blur these three together as "all just stabilizing the numbers," but the position they act at and the quantity they control are completely different, and **they also can't be stacked together mindlessly** — enabling several at once requires re-checking the overall scale rather than simply adding them up; the full argument for "why attention logits explode" is in normalization_init_tutorial.md §6.3.

### 6.3　Attention connectivity structure: local / sliding / global

This is a design axis for attention's **connectivity structure/visibility range**, not an "attention backend," and it's also a separate matter from the positional-encoding scheme — FlashAttention is the typical execution backend: it changes the computation order and IO scheduling of exact attention, without defining a new connectivity graph or block topology (full IO-complexity derivation in attention_tutorial.md). Mistral (Jiang et al., *Mistral 7B*, arXiv 2310.06825, 2023) uses fixed-window sliding-window attention; Gemma-2 alternates local-window and global attention 1:1 across layers; Gemma-3 is closer to a periodic pattern of "several local layers followed by one global layer" (roughly 5:1, not a strict one-local-one-global alternation). None of these change the residual interface at the outer level of the block — they only change how far attention can see internally; but the layout itself of "which layers use local, which use global" is a decision at the **stack-scheduling level** (see §0's layered map), not a private attribute of a single block — full sliding-window/StreamingLLM mechanics are in long_context_rope_yarn_mla_tutorial.md §10.

### 6.4　Execution/adaptation overlays: FlashAttention, quantization, LoRA (pointer only)

These three mechanisms all belong to the "execution/adaptation overlay" level in §0's layered map. They are not architectural slots on the same level as GQA/MoE requiring a mutually exclusive choice — they are strategies layered on top of an already-chosen architecture, used to accelerate it or adapt it to a deployment environment:

- **FlashAttention**: only changes the computation order, IO scheduling, and memory-access pattern of exact attention; it doesn't change attention's mathematical result, nor does it define a new connectivity graph — it can be layered on top of MHA/GQA/MQA/MLA, on any local/global pattern. Full IO-aware complexity derivation in attention_tutorial.md.
- **Quantization**: represents weights/activations in a lower-precision numeric format; it's an overlay at the numeric-representation level, layerable on top of any architectural choice above, at the cost of quantization error; full error analysis in quantization_tutorial.md.
- **LoRA**: adds a pair of low-rank matrices alongside already-trained weights for parameter-efficient fine-tuning; it's an overlay at the training/adaptation stage, and doesn't change the block's forward architecture; full low-rank formulas in lora_peft_tutorial.md.

None of these three should be treated as "an architectural slot on par with GQA or MoE that you must choose exactly one of" — they are independent dimensions that can be layered on top of almost any architectural combination.

### 6.5　Summary: how these categories of mechanisms combine

The categories of mechanisms above can mostly be analyzed independently, and in practice often show up together in the same model (e.g., Gemma-3 uses branch pre+post norm + QK-Norm + alternating local/global attention all at once). A common order to check them in: first pin down attention's visibility range (global / local / sliding, §6.3), then pin down the numerical-stabilization method for Q/K or logits (QK-Norm or soft-cap, §6.2 — using both together requires re-checking the scale, not mindless stacking), and finally pin down the norm topology and residual scaling (Pre-LN / branch pre+post / parallel, §6.1). These dimensions can mostly be combined independently, but they aren't mathematically strictly unconstrained orthogonal axes — the specific values within a given model are often further constrained by engineering factors like training stability, kernel support, and compatibility with existing checkpoints. This is also why the same column in §8's model recipe table shows multiple values, yet the combination space is much narrower than "completely free."

> 🎯 **The interview-answering principle for this section**
> Whenever a "new stabilization trick" comes up, ask yourself three things: which tensor does it act on (Q/K? attention scores? the residual stream?); does it change the block's input/output shape contract (usually not); and is it roughly independent of §4/§5's two main axes (usually roughly independent, but check whether there's real coupling). Using these three questions to classify any new term will yield a much more solid answer than simply listing "what is this."

## §7 Capstone: assembly, end-to-end forward, parameter accounting, and executable verification

### 7.1　Assembling a Llama-3-like block, then swapping in MoE

Actually assembling the previous sections' slots into a "more common modern combination" (more precisely a "Llama-3-like core block" — not a claim of precisely reproducing any specific checkpoint, unless its head count, KV-head count, intermediate size, RoPE base, bias settings, and dtype behavior are all matched too):

1. **Norm slots (inside the block)**: Pre-RMSNorm, twice (once before attention, once before FFN).
2. **Attention slot**: causal self-attn, RoPE applied internally to Q/K, $n_q$ query heads grouped to share $n_\text{kv}$ KV heads (GQA; degenerates to MHA when $n_\text{kv}=n_q$, see §7.4 [D05]), K enters the cache after rotation, V is never rotated, no bias.
3. **FFN slot**: SwiGLU gated FFN, $h \approx 8d/3$ rounded to a hardware-friendly multiple, no bias.
4. **Assembling a single block**: $x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x))$, $x=x+\text{SwiGLU}(\text{RMSNorm}(x))$ — this is the complete definition of **one block**; repeat it $N$ times to get the whole decoder stack.
5. **Stack-level wrap-up (not part of any single block)**: after all $N$ blocks are stacked, one more **final RMSNorm** is applied before feeding into the LM head: $x_\text{out} = \text{RMSNorm}\big(\text{Block}_N(\cdots\text{Block}_1(x_0))\big)$. Final norm is a stack-scheduling-level operation, not a component of any block from steps 1–4 (see §0's layered map and §3's discussion of the block-vs.-external boundary).

Working out this block's parameter count concretely using §7.4's code teaching sizes ($d=96$, $n_q=8$, $n_\text{kv}=2$, gated hidden $=256$): the GQA attention part has $2d^2(1+n_\text{kv}/n_q)=23{,}040$ matrix parameters, the SwiGLU FFN part has $3dh=73{,}728$ matrix parameters, and the two RMSNorms each have only $d=96$ per-channel scaling parameters — a single block's matrix parameters total roughly $23{,}040+73{,}728=96{,}768$, with FFN taking the vast majority. This is also why "how to choose the FFN hidden size" (§3, §7.3) has a bigger effect on total parameter count than "how many KV heads to use."

**Next, swap the FFN slot for MoE, and leave everything else untouched**:

$$x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x)), \qquad x=x+\text{MoEFFN}(\text{RMSNorm}(x))$$

`MoEFFN`'s external-facing **main activation tensor** interface is still `[B,T,d] -> [B,T,d]` — `ModernDecoderBlock`'s code doesn't need to be aware at all of whether the internals are dense matrix multiplication or router + expert selection; this is also the property §7.4 [D09] directly verifies. But this only demonstrates §5's point that "the main-activation-tensor interface is unchanged" — it doesn't mean this swap is completely unrelated to the training/systems contract: training usually needs to return additional router/load-balancing-related quantities, and distributed deployment usually needs to introduce expert parallelism (see the three-layer contract breakdown in §5 and §9 for details).

> ✅ **Capstone summary: which fields describe a block's recipe**
> Norm topology (Post/Pre-serial/Pre-parallel/branch pre+post) + norm type (LayerNorm/RMSNorm) + positional-encoding scheme and the level it belongs to (learned absolute outside the stack / RoPE inside attention) + attention variant (MHA/MQA/GQA/MLA, local window or not, QK-Norm or not, attention scale/temperature) + FFN variant (dense/gated/MoE, intermediate size) + the choices on bias, dropout, norm epsilon, and residual scaling/gating (e.g. LayerScale, see §6.1.3) — these cover the coarse-grained recipe fields needed for common interviews, but **this is not the block's entire design space**: at minimum it still leaves out the specific values of $d_h/d_q/d_v$ and head count, the RoPE base/rotary fraction/context-extension scheme, the precise approximate implementation of the activation function, and whether attention and FFN share certain parameters, among other finer-grained choices. **Final norm, local/global alternation, and the dense/MoE layer layout do not belong on this block-field checklist** — they are stack-scheduling-level decisions (step 5); don't mix them into a single block's definition.

### 7.2　End-to-end forward walkthrough: the modern decoder-only block's complete data flow

§1 gave a complete walkthrough for the 2017 encoder-decoder's decoder, but the modern decoder-only block has been missing an equally detailed data-flow explanation — filling that in here, walking through the complete tensor flow using §7.1's teaching sizes ($d=96$, $n_q=8$, $n_\text{kv}=2$, gated hidden $=256$):

$$x \in [B,T,d] \;\xrightarrow{\text{RMSNorm}}\; h_1 \;\xrightarrow{\text{Q/K/V projection + reshape}}\; Q\in[B,n_q,T,d_h],\;K,V\in[B,n_\text{kv},T,d_h] \;\xrightarrow{\text{RoPE}}\; Q',K' \;\xrightarrow{\text{GQA attention}}\; O\in[B,T,d] \;\xrightarrow{\text{residual add}}\; u = x + O$$

$$u \;\xrightarrow{\text{RMSNorm}}\; h_2 \;\xrightarrow{\text{gate/up projection}}\; \text{SiLU}(\text{gate}(h_2))\odot\text{up}(h_2) \;\xrightarrow{\text{down projection}}\; F\in[B,T,d] \;\xrightarrow{\text{residual add}}\; y = u + F$$

This is the complete forward pass for a single block; repeat this block $N$ times, then apply one stack-level final RMSNorm on top (§7.1 step 5), and that's the complete decoder-only backbone: $y_\text{final} = \text{RMSNorm}\big(\text{Block}_N(\cdots(\text{Block}_1(x_0)))\big)$.

This data flow takes different concrete forms across three execution modes:

| Execution mode | Q/K/V length | Mask | KV cache | Typical scenario |
| --- | --- | --- | --- | --- |
| Training / full-forward | Q, K, V are all the full length $T$ | A one-shot lower-triangular causal mask $[T,T]$ | Not used | Training, teacher-forcing evaluation, scoring a complete prompt in one shot |
| Prefill (inference's first step) | Q, K, V are all the full prompt length $S$ | Causal mask $[S,S]$ | Written into the cache (K, V each with $n_\text{kv}$ heads, length $S$) | The first forward pass after receiving the user's prompt, building the cache for subsequent decode |
| One-token decode (inference step by step) | Q length $1$ (the new token); K, V grow from the cache's historical length $t$ to $t{+}1$ | The new token can see all of $0..t$, degenerating to "the whole row is True" (though static-cache/left-padding/packed-sequence scenarios may still need extra masking, see §9) | Read the old cache + append new K/V | Every step of autoregressive generation |

All three modes share the same RMSNorm→RoPE→attention→residual code path, differing only in the actual length of Q/K/V, the shape of the mask, and whether the cache is read/written — this is also why §7.4 [D04] uses a single implementation to verify that full-forward and cached decoding agree numerically (if the two paths used different code, this test would lose its value as cross-validation). The systems-level performance bottlenecks specific to prefill/decode (arithmetic intensity, batch scheduling) are out of scope here; see llm_inference_serving_tutorial.md.

### 7.3　Parameter-count and KV-cache formula accounting

**FFN parameter count** (ignoring bias): dense (ReLU/GELU, two matrices) is $8d^2$; gated (SwiGLU/GeGLU, three matrices) is $3dh$. Setting the two equal solves to $h=8d/3$ — this is a **common default derived from parameter-count matching**, not an architectural law; real Llama-family implementations typically start from $4d$, multiply by $2/3$, then round to a hardware-friendly multiple (plus an optional multiplier), so the intermediate size may deviate noticeably from $8d/3$. §7.4 [D06] verifies that with $d=96$, dense hidden $=384$ ($4d$) and gated hidden $=256$ ($8d/3$) both come out to exactly 73,728 matrix parameters.

**Attention parameter count** (ignoring bias): MHA's four projections Q/K/V/O total $4d^2$; GQA is $2d^2(1+n_\text{kv}/n_q)$ (degenerating back to $4d^2$ when $n_\text{kv}=n_q$, and to MQA's minimal value when $n_\text{kv}=1$). §7.4 [D06] checks both formulas against real `.numel()` counts. Both formulas hold only under the assumptions $d=n_qd_h$, Q/K/V using the same head dim, an output projection of size $d\times d$, and bias ignored — when these assumptions don't hold, the formulas need to be adjusted accordingly.

**KV cache** (the table below gives **element counts** per token, per layer — not byte counts; the full memory footprint also needs to be multiplied by batch size $B$, sequence length $L$, layer count $N_\text{layer}$, and bytes per element — typically 2 bytes for fp16/bf16, 1 byte for fp8; full cross-model GB figures are in kv_cache_speculative_decoding_tutorial.md §2.2):

| Variant | Cache size (elements/token/layer) |
| --- | --- |
| MHA | $2 n_q d_h$ |
| GQA | $2 n_\text{kv} d_h$ |
| MQA | $2 d_h$ |
| MLA | $d_c + d_h^R$ (latent dimension + shared RoPE component, no "$\times 2$" since K/V share the same latent) |

### 7.4　From-scratch implementation and executable verification: `code/transformer_block.py`

The complete runnable script is at [`code/transformer_block.py`](code/transformer_block.py) (pure PyTorch, all 9 demos [D01]–[D09] run in seconds on CPU). Four entry-point classes correspond to this tutorial's four "block eras": `VanillaEncoderLayer2017`, `VanillaSeq2SeqDecoderLayer2017` (§1), `GPT2StyleDecoderLayer` (§2), `ModernDecoderBlock` (§3/§7.1, whose `ffn=` can be swapped for `DenseFFN`/`GatedFFN`/`MoEFFN`). The main text only shows the three most essential pieces of code; the rest of the demos exist as standalone functions in the script and aren't pasted here.

**[D02] Residual-topology test** (an executable version of §0's four formulas: zero out the sublayer output — Pre-LN should strictly output $x$, Post-LN should output $N(x) \ne x$):

```python
def zero_sublayer(_x):
    return torch.zeros_like(_x)

post_ln_out = ln(x + zero_sublayer(x))       # y = N(x + F(x)) -> y = N(x)
pre_ln_out  = x + zero_sublayer(ln(x))       # y = x + F(N(x)) -> y = x

assert torch.allclose(pre_ln_out, x, atol=1e-6)          # Pre-LN: exact identity mapping
assert torch.allclose(post_ln_out, ln(x), atol=1e-6)     # Post-LN: strictly equals N(x)
assert not torch.allclose(post_ln_out, x, atol=1e-4)     # and generally not equal to x
```

**[D04] Full forward vs. cached decoding** (under eval() + no dropout, the full-sequence forward pass and the token-by-token incremental decode must agree numerically — this simultaneously checks the correctness of causal mask, RoPE position offset, and cache append):

```python
causal = build_causal_mask(torch.arange(T), torch.arange(T))
out_full = block(x, cos, sin, mask=causal)                  # compute the whole sequence in one shot

cache = KVCache()
outs = []
for t in range(T):
    q_pos, k_pos = torch.tensor([t]), torch.arange(t + 1)
    mask_t = build_causal_mask(q_pos, k_pos)                 # the new token sees all of 0..t
    outs.append(block(x[:, t:t+1, :], cos, sin, mask=mask_t,
                       kv_cache=cache, start_pos=t))
out_cached = torch.cat(outs, dim=1)
assert torch.allclose(out_full, out_cached, atol=1e-4)
assert cache.k.shape[1] == n_kv_heads                        # cache head count never grows to n_q_heads
```

**[D08] Parallel vs. sequential dependency** (use a forward hook to capture the FFN's actual input tensor, and do an **exact match** against an "independently recomputed expected tensor" — rather than the fragile check "two random numbers aren't equal"):

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

**Key invariants the script should satisfy** (one per [D01]–[D09], described here by "verification intent" — see the script itself for the concrete assertion implementations):

1. **[D01] Shape contract**: all four entry classes preserve the `[B,T,d]` residual width; cross-attn, when $T\ne S$, still correctly returns output of length $T$, and should verify that memory was actually read (changing memory changes the output), not just verify the shape.
2. **[D02] Residual-topology test**: after zeroing the sublayer output, Pre-LN-style blocks strictly output $x$; Post-LN-style blocks output $N(x)$, generally not equal to $x$; each sublayer's own LN should be compared separately — you can't use the same $N(x)$ to prove itself.
3. **[D03] No causal leakage**: perturbing only tokens after position $t{+}1$ must leave the output at positions $0..t$ completely unchanged; this invariant should be confirmed to actually be contributed by the attention branch.
4. **[D04] Full forward vs. cached decoding**: under eval() + no dropout, the full-sequence forward pass and token-by-token incremental decode agree numerically at every position, and the cache's head dimension stays at $n_\text{kv}$ throughout — this is the most direct way to catch causal-mask/RoPE-offset/cache-order bugs, but it's not the only test needed to cover the attention branch's nonzero contribution.
5. **[D05] MHA/MQA/GQA degeneration**: the general implementation at $n_\text{kv}=n_q$ matches an independently hand-written plain MHA reference exactly; both $n_\text{kv}=1$ (MQA) and intermediate values (e.g. $n_\text{kv}=2$) should each have their own independent `repeat_interleave` reference comparison — testing only the degenerate case $n_\text{kv}=n_q$ isn't enough.
6. **[D06] Parameter formulas**: at $d=96$, dense FFN (hidden $=4d$) and gated FFN (hidden $=8d/3$) matrix parameters both come out to exactly 73,728; GQA/MHA attention matrix parameters match the $2d^2(1+n_\text{kv}/n_q)$ and $4d^2$ formulas exactly; illegal configurations where $d_\text{model}$ doesn't divide the head count should raise an error at construction time.
7. **[D07] Packed-projection numerical equivalence**: packed QKV (and packed gate+up) produce numerically consistent outputs compared to three (or two) separate Linears using the same weights (numerically equivalent within tolerance, but not necessarily bitwise identical).
8. **[D08] Parallel vs. sequential**: a forward hook simultaneously captures attention's and FFN's real inputs; the sequential block exactly equals $N_2(x+\text{Attn}(N_1(x)))$, and the parallel block's two branches both exactly equal $N(x)$.
9. **[D09] Backward smoke test**: all parameters in a dense block have finite gradients; for an MoE-swapped block, all parameters belonging to experts other than those "unused in this batch" should be checked for nonzero gradients, and the remaining parameters (router/attention/norm etc.) should also receive gradients normally.

> ✅ **Each of the three code snippets nails down one high-frequency misconception**
> [D02]: residual-topology differences aren't "an abstract description" — zeroing out the sublayer gets you the exact $x$ or $N(x)$; [D04]: full-forward and cached decoding must agree numerically at every position — this is the most direct way to catch causal-mask/RoPE-offset/cache-order bugs, but it isn't the sole proof of correctness; [D08]: judging whether "parallel is equivalent to sequential" requires reconstructing the **actual input tensor each of attention and FFN really receives** and matching it exactly, rather than simply asserting that two outputs are unequal.

The **actual output** of running `python3 code/transformer_block.py` (rerun and filled back in after the code fixes were completed; D10–D15 are targeted regression tests added during the code-review stage, corresponding respectively to constructor argument validation, KV-cache consistency, cross-attention combination restrictions, mask safety and broadcasting, default causal behavior, and RoPE cache device/dtype handling):

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

## §8 Model Recipe Atlas

### 8.1　Mainstream model block-recipe comparison table

A note on conventions: in the table below, "QK-Norm" refers specifically to explicitly normalizing each head's Q/K before the attention dot product, and does not count the low-rank latent-projection normalization inside DeepSeek MLA as QK-Norm; "branch pre+post" refers to $y=x+N_\text{post}(F(N_\text{pre}(x)))$, not equivalent to the original Post-LN; the table below only describes MHA/MQA/GQA's KV-head sharing scheme, with local/sliding/global attention patterns written separately in the remarks column; final norm and local/global layout belong to the stack-scheduling level (see §0's layered map) — the table presents them as part of the model's overall recipe; this does not imply that they are fields of an individual block; **none of these representative models' canonical recipes use ALiBi as their primary positional scheme**.

| Model | Norm topology | Norm type | Positional scheme | Attention | FFN/experts | Parallel attn+FFN | QK-Norm | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | Pre-LN | RMSNorm | RoPE | 7B/13B are MHA; 70B is GQA | SwiGLU, dense | No | No | The whole Llama-2 family shouldn't be uniformly written as GQA |
| Llama-3 | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU, dense | No | No | "Llama-3-like" code still needs to match the specific checkpoint configuration to count as a precise reproduction |
| Qwen2/2.5 | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU; dense or MoE depending on checkpoint | No | Usually no | Scoped to the text model; the whole Qwen family shouldn't be uniformly written as dense or uniformly as MoE |
| Mistral | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU, dense | No | No | Refers to the original Mistral dense recipe; also uses sliding-window attention; Mixtral is a separate MoE recipe |
| DeepSeek-V2 | Pre-LN | RMSNorm | decoupled RoPE within MLA | MLA | SwiGLU-style; first layer dense, mostly DeepSeekMoE thereafter | No | No | MLA has internal latent-projection normalization, but that is not the same as headwise QK-Norm |
| DeepSeek-V3 | Pre-LN | RMSNorm | decoupled RoPE within MLA | MLA | SwiGLU-style; first 3 layers dense, mostly DeepSeekMoE thereafter | No | No | The block backbone continues to use MLA+MoE; training/load-balancing improvements shouldn't be mistakenly written up as a new attention type |
| GPT-J | Parallel Pre-LN | LayerNorm | partial RoPE | MHA | GELU, dense | Yes | No | Core form $x+A(N(x))+M(N(x))$; GELU doesn't fall under the ReLU/GeGLU/SwiGLU three-way split |
| PaLM | Parallel Pre-LN | LayerNorm, bias-free recipe | RoPE | MQA | SwiGLU, dense | Yes | No | The parallel attn/MLP use a shared normalized residual input |
| Gemma-1 | Pre-LN | RMSNorm | RoPE | 2B is MQA, 7B is MHA | GeGLU, dense | No | No | The whole Gemma-1 lineup shouldn't be uniformly written as GQA |
| Gemma-2 | branch pre+post | RMSNorm | RoPE | GQA | GeGLU, dense | No | No | 1:1 alternation of local/global attention across layers, using attention-logit and final-logit soft-capping; post norm sits inside the branch, before the residual add |
| Gemma-3 | branch pre+post | RMSNorm | RoPE | 1B is MQA (single KV head, multiple query heads); 4B/12B/27B are GQA (an independent judgment call at roughly 85% confidence — ultimately best confirmed against the specific config's `num_key_value_heads`) | GeGLU, dense | No | Yes | Local/global attention pattern with a roughly 5:1 period (several local layers followed by a global layer, not a strict 1:1 alternation); QK-Norm is one key difference from Gemma-2 |
| GPT-2 | Pre-LN | LayerNorm | learned absolute positional embedding | MHA | GELU, dense | No | No | Each layer sequential attn→MLP, with a final LayerNorm at the end of the stack |
| GPT-3 | Pre-LN | LayerNorm | learned absolute positional embedding | MHA | GELU, dense | No | No | Continues the GPT-2-style block, with variation between dense and locally banded sparse attention patterns across layers |

**Three regularities readable from this table** (mapped against §0's layered map): (1) the "Norm topology" and "Norm type" columns almost always show up bundled together — Pre-LN with RMSNorm, Parallel Pre-LN with LayerNorm, branch pre+post also with RMSNorm — showing that even though these two dimensions can be discussed separately in analysis, the combination space in practice is much narrower than the theoretical space; (2) all four values MHA/MQA/GQA/MLA in the "Attention" column can be found among this same set of representative models, corroborating §4's judgment that "these are coexisting design points, not a historical staircase"; (3) only the GPT-J and PaLM rows say "yes" in the "Parallel attn+FFN" column, with every other model using sequential — parallel is a niche but genuinely-in-use variant, not a historically obsolete dead end.

### 8.2　Twelve claims most likely to trip you up

Collecting the high-frequency wrong answers that appeared throughout this tutorial into one table, for a final pre-interview scan:

| Common claim | More accurate version |
| --- | --- |
| Modern LLMs were the ones that switched Post-LN to Pre-LN | GPT-2 is already Pre-LN — this step happened earlier (§2.2) |
| decoder-only is just "decoder with cross-attn removed" | Correct at the single-layer structural level; at the full-model level you also need to add: the encoder/memory path disappears along with it, and dual-stream conditional generation becomes single-stream causal modeling (§2.1) |
| The residual stream's variance inflates with depth as $\propto\sqrt l$ | Under a simplified independent-increment assumption, the variance grows roughly **linearly** as $O(l)$; it's the standard deviation/RMS that grows roughly as $\sqrt l$; the real growth rate also depends on initialization and branch correlation (§0, §2.2) |
| The end of a Pre-LN stack must have a final norm, or the model can't possibly train well | Final norm is a common configuration in standard recipes like GPT-2/Llama; not closing it off typically hurts training quality noticeably, but it isn't a mathematically indispensable condition for every Pre-LN architecture; it's also a stack-level operation, not a field of any single block (§2.2, §7.1) |
| MHA→MQA→GQA→MLA is one evolutionary chain | By historical timeline it genuinely is MHA→MQA→GQA (MQA predates GQA), but that's just one of several ordering dimensions; MLA is an independent low-rank latent branch and shouldn't be tacked onto the end of the same compression-ratio axis — the reason it's "not one chain" is that there's more than one ordering dimension, not that the historical fact "MQA predates GQA" is itself wrong (§4, §9) |
| MLA is just a harsher compression of GQA | MLA compresses the hidden dimension and requires decoupling RoPE; GQA compresses head count/sharing relationships — the constraints on the two are entirely different (§4.2, §9) |
| SwiGLU's hidden size must equal $8d/3$ | This is a common default derived from parameter-count matching; real models round to hardware-friendly multiples and can deviate noticeably (§3, §7.3) |
| Dense FFN will inevitably be replaced by MoE sooner or later | This is an independent capacity axis, not an obligatory stage — representative models like Llama 1–3, the original Mistral, and Gemma 1–3 still use dense today, while Mixtral, Llama 4, and others have adopted MoE (§5) |
| "Sandwich norm" is the same thing as the original Post-LN | The more accurate term is branch pre+post; the second norm acts only on the branch output, not on the residual sum (§6.1) |
| Pre-LN needs no warmup at all | Pre-LN only removes the hard warmup requirement caused by Post-LN's "gradient imbalance"; modern large Pre-LN models still commonly use warmup (for additional, Adam-related reasons too — see normalization_init_tutorial.md §5.2) |
| MQA/GQA reduce the compute of the Q projection | They only reduce the K/V projection's parameters/compute and KV-cache memory/bandwidth; the Q side's head count and compute are unchanged (§4.1) |
| Parallel residual requires attn/FFN to share the same norm parameters | Parallel is defined by "no ordering dependency"; using two independent norm modules also satisfies it, as long as both read the original $x$ (§6.1, §7.4 [D08]) |

## §9 30 High-frequency interview questions

Split into three difficulty tiers; expand each to see the answer's key points plus common pitfalls. L2/L3 are advanced topics commonly discussed in top-tier lab interviews (topology bifurcation, independence of assembly axes, systems correctness, cross-model recipe comparison). When answering, **don't repeat the deep math derivations from sibling tutorials** — organizing around "structural judgment → key formula → common mistakes" is enough.

### L1 — Must-know questions

<details>

<summary>Q1. How do you write out the complete forward pass for each of the four topologies — Post-LN, Pre-LN sequential, Pre-LN parallel, and branch pre+post?</summary>

- Post-LN: $u=N_1(x+A(x))$, $y=N_2(u+M(u))$ — normalization wraps around after every residual addition
- Pre-LN sequential: $u=x+A(N_1(x))$, $y=u+M(N_2(u))$ — the residual backbone is a pure identity addition; the FFN reads $u$ (the stream after attention has updated it)
- Pre-LN parallel (GPT-J/PaLM): $y=x+A(N_A(x))+M(N_M(x))$ — the two sublayers read the same $x$, with no ordering dependency
- branch pre+post (Gemma-2/3): $u=x+N_1^{\text{post}}(A(N_1^{\text{pre}}(x)))$, $y=u+N_2^{\text{post}}(M(N_2^{\text{pre}}(u)))$ — one norm before and one after each branch, with the residual backbone still a pure addition
- The end of a Pre-LN stack usually has one **stack-level** final norm, which is not part of any of the block formulas above by itself (§0, §7.1)

Only remembering three of the four and missing branch pre+post; or treating final norm as one term inside some block formula.

</details>

<details>

<summary>Q2. What is the fundamental difference between the original Transformer decoder and a decoder-only LLM block?</summary>

- At the single-layer structural level: it genuinely is "cross-attn removed" (3 sublayers → 2 sublayers) — this part of the claim is fine
- At the full-model level: you still need to add that the encoder/memory path disappears along with it, and dual-sequence conditional generation becomes single-sequence causal modeling — this is a change in information-flow topology and sequence factorization
- Only a pure decoder-only block lacks cross-attn; some multimodal/retrieval-augmented LLMs reintroduce cross-attention on top of the backbone (§2.1)

Stopping at "cross-attn was removed" without adding the full-model-level information-flow change; or, at the other extreme, claiming this is an entirely different problem setup, which obscures the fact that the single-layer structure genuinely does have one component removed.

</details>

<details>

<summary>Q3. What actually changed going from a GPT-2-style block to a Llama-style block?</summary>

- Both **are already Pre-LN** (GPT-2 is not Post-LN)
- LayerNorm→RMSNorm; learned absolute PE (outside the stack)→RoPE (inside attention, acting only on Q/K, originally proposed by Su et al. 2021's RoFormer, with LLaMA as an adopter); GELU dense FFN→SwiGLU gated FFN; MHA→MHA/GQA depending on spec; most bias/dropout removed
- Both sides have a final norm, just of different types (LayerNorm vs. RMSNorm) (§2.2)

Leading with "modern LLMs were the ones that switched Post-LN to Pre-LN" — GPT-2 is already Pre-LN, and this wrong answer will get you immediately docked on follow-up.

</details>

<details>

<summary>Q4. Why does attention divide by $\sqrt{d_k}$?</summary>

- Assuming Q, K's components are independent, zero-mean, unit-variance, the dot product $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ is a sum of $d_k$ independent product terms; variances add, giving roughly $d_k$
- Without scaling, the dot product's magnitude grows with $d_k$; before entering softmax, some logits will be far larger than others, pushing softmax toward one-hot and saturating the gradient at those positions
- Dividing by $\sqrt{d_k}$ pulls the dot product's variance back to $O(1)$ magnitude — a concrete means of "controlling the softmax input's scale," not an arbitrarily chosen constant

Only reciting the conclusion "divide by $\sqrt{d_k}$ to prevent vanishing/saturating gradients," without being able to derive the step where independent components give a dot-product variance $\approx d_k$.

</details>

<details>

<summary>Q5. What's the difference between LayerNorm, RMSNorm, and BatchNorm? Why does the Transformer usually not use BatchNorm?</summary>

- LayerNorm: normalizes over the feature dimension for each sample, each position (subtract the mean, divide by the standard deviation), independently per sample, not across the batch
- RMSNorm: only re-scales (divides by the root-mean-square), skipping mean-centering
- BatchNorm: normalizes across the batch for the same feature dimension; its statistics (running mean/var) depend on other samples within the batch
- The Transformer's input is variable-length sequences, so statistics across different positions/samples within a batch are unstable; at inference time it's also often variable-length, single-sample, token-by-token autoregressive decoding — BatchNorm's running statistics don't match the training distribution in this setting; LayerNorm/RMSNorm compute independently per sample, naturally fitting variable-length sequences and autoregressive decoding

Only saying "BatchNorm doesn't work well in NLP" without being able to give the specific reason — dependence on batch statistics, and the mismatch with variable-length sequences/autoregressive decoding.

</details>

<details>

<summary>Q6. How does the tensor shape change through complete attention? Why are multiple heads and $W_O$ needed?</summary>

- Input $x\in[B,T,d]$ → after Q/K/V projection, reshaped to $[B,n_\text{head},T,d_h]$ ($d=n_\text{head}\cdot d_h$)
- score $=QK^\top/\sqrt{d_h}\in[B,n_\text{head},T,T]$ → add mask → softmax (along the last dimension) → multiply with V to get $[B,n_\text{head},T,d_h]$
- Concatenate the heads back to $[B,T,d]$, then pass through the output projection $W_O\in\mathbb{R}^{d\times d}$ to get the final output $[B,T,d]$
- Multiple heads let the model learn different attention patterns in parallel across different subspaces; $W_O$ remixes and re-projects the outputs of the different heads — without $W_O$, each head's output would just be a simple concatenation, with no way for information from different heads to combine

Missing the reshape/concat steps; unable to explain that $W_O$'s role is "mixing information across heads," not just "dimension alignment."

</details>

<details>

<summary>Q7. What's the key difference between self-attention and cross-attention?</summary>

- self-attention: Q, K, V all come from the same sequence
- cross-attention: Q comes from the current stream (length $T$), K, V come from another sequence/memory (length $S$); $T$ and $S$ are generally not equal
- The original Transformer decoder uses cross-attn to condition the target sequence on the source encoding; a pure decoder-only LLM block structurally has **no** cross-attn (§1, §2.1; full mask/softmax formulas in attention_tutorial.md §4.1)

Thinking cross-attention is just "self-attention plus a mask"; unable to state that Q's length and K/V's length can differ.

</details>

<details>

<summary>Q8. Which design axis of attention do local/sliding window attention belong to?</summary>

- It's a design axis for attention's **connectivity structure/visibility range**, not an "attention backend" (FlashAttention is the execution backend), and it's also not a positional-encoding scheme
- Mistral uses a fixed sliding window; Gemma-2 alternates local/global 1:1 across layers; Gemma-3 is closer to a periodic pattern of several local layers followed by a global layer (roughly 5:1, not a strict 1:1)
- It can broadly be combined with any positional-encoding scheme, but extrapolation length, window boundaries, and compatibility with specific kernel implementations still need to be verified separately — it's not unconditionally "plug and play" (§6.3)

Calling it an "attention backend," or conflating it with positional-encoding schemes like RoPE/ALiBi; assuming "broadly orthogonal" means "swap it freely without any verification needed."

</details>

<details>

<summary>Q9. Do weight tying, embedding scaling, and the LM head belong inside the block or to the external recipe?</summary>

- All belong to the "model shell" level, **not to any Transformer block** — token embedding and the LM head are the whole model's input/output layers, not components of any single block
- Weight tying: the input embedding and the LM head share the same parameter matrix; embedding scaling: token embedding multiplied by a constant related to $\sqrt{d}$; LM head soft-cap: some models also apply a §6.2-style bounded transform to the final logits (§3)
- If, when asked "what does a block look like" in an interview, you fold these into the block definition, it makes the interviewer doubt your understanding of the "layer vs. model" boundary

Describing embedding/LM head as part of some block, conflating the boundary between "one block layer" and "the whole LM."

</details>

<details>

<summary>Q10. What's SwiGLU's complete formula, and what role does the FFN play?</summary>

- The FFN is **position-wise**: it applies the same nonlinear transformation independently to the vector at each sequence position, without mixing information across positions (mixing information across positions is attention's job)
- SwiGLU formula: $\text{down}(\text{SiLU}(\text{gate}(x))\odot\text{up}(x))$ — gate and up are two independent linear projections into the intermediate dimension $h$, multiplied element-wise, then projected back to $d$ by down
- The SiLU on the gate branch acts as a gate, deciding "how much" of each dimension of the up branch passes through — this is the key difference from a dense FFN (two matrices, a single activation function)
- Within the whole block, the FFN plays the role of "per-position feature transformation/information storage," complementing attention's role of "cross-position information routing"

Only remembering the number "three matrices," without being able to state the gating semantics of the gate branch or the FFN's position-wise property.

</details>

### L2 — Advanced questions

<details>

<summary>Q11. Why is Pre-LN easier to train? Why does the end of the stack usually need a final norm?</summary>

- The residual backbone is a pure identity addition, so there's an identity gradient path that bypasses the sublayer and normalization Jacobians, generally improving the optimization conditions of deep networks — but this doesn't guarantee the total gradient never decays or cancels out; "gradients flow losslessly all the way through" is too strong a claim
- The cost: under a simplified independent-increment assumption, the residual stream's **variance** grows roughly **linearly** with depth ($O(l)$), so it's the standard deviation/RMS/typical magnitude that grows roughly as $\sqrt l$; the real growth rate also depends on initialization, residual scaling, and branch correlation
- Feeding straight into the output head without normalizing the accumulated residual stream typically makes the output distribution deviate noticeably from the scale seen during training, and training quality commonly drops noticeably; standard recipes like GPT-2/Llama therefore configure a final norm, but this isn't a mathematically indispensable condition for every Pre-LN architecture — variants without a traditional final norm can also be constructed via residual scaling/special initialization/a normalized output head
- **There's no guarantee final quality is always superior to a well-tuned Post-LN** — when Post-LN trains stably, depth's contribution isn't diluted, and its ceiling is sometimes higher

Writing the variance growth as $\propto\sqrt l$ (it should be the standard deviation/RMS that is $\sqrt l$, while variance itself grows linearly as $O(l)$); calling final norm "mathematically mandatory."

</details>

<details>

<summary>Q12. Where should RoPE be inserted, and what does the KV cache store?</summary>

- RoPE is inserted inside attention, acting only on Q, K — **V is never rotated**
- Standard implementations usually cache the **already-rotated** K; if an implementation instead chooses to cache unrotated K, it must additionally retain its position and rotate it at the correct position when used — it cannot treat all historical K as if at the current position (this is mathematically just as correct, only slower, hence not the common choice)
- What MLA caches isn't even ordinary complete K — it's a latent plus a decoupled, shared RoPE key component (§4.2)
- Under GQA/MQA the cache stores only $n_\text{kv}$ KV heads; it must not physically expand them to $n_q$ heads before storage (§4; full rotation-matrix derivation in long_context_rope_yarn_mla_tutorial.md §2)

Claiming "caching already-rotated K" is the only mathematically correct implementation; not knowing that MLA's cache contents are entirely different from ordinary K/V.

</details>

<details>

<summary>Q13. How do parameter count and KV cache change across MHA, MQA, and GQA?</summary>

- For the formulas to hold, the assumptions need to be stated explicitly: $d=n_qd_h$, Q/K/V use the same head dim, the output projection is $d\times d$, bias is ignored
- Attention parameters: MHA is $4d^2$; GQA is $2d^2(1+n_\text{kv}/n_q)$; it degenerates back to the MHA formula when $n_\text{kv}=n_q$ (§7.3 [D06])
- KV cache: MHA is $2n_q d_h$/token/layer; GQA is $2n_\text{kv}d_h$; MQA ($n_\text{kv}=1$) is $2d_h$ — these are **element counts**, not byte counts; the full memory footprint still needs to be multiplied by $B\times L\times N_\text{layer}\times$ bytes per element
- GQA genuinely reduces the K/V projection's parameters and compute, but does not reduce the Q projection's compute — it's not as simple as "only saves memory/bandwidth"

Applying the formulas directly without stating their assumptions; treating the KV-cache element-count formula as the actual byte count; assuming GQA has zero effect on any compute at all.

</details>

<details>

<summary>Q14. How do you compute a dense Transformer block's parameter count and FLOPs?</summary>

- Parameter count: attention is roughly $4d^2$ (or GQA's $2d^2(1+n_\text{kv}/n_q)$) + FFN is roughly $8d^2$ (dense) or $3dh$ (gated) + the two norms' per-channel parameters ($O(d)$, asymptotically lower-order and much smaller than the matrix-parameter terms) — within a single layer, FFN parameters usually dominate (in §7.1's concrete example, FFN accounts for about 76%)
- FLOPs: attention's QKVO projections are $O(Td^2)$; the score-matrix computation plus weighted sum is $O(T^2d)$ ($T$ being sequence length); FFN is $O(Tdh)\sim O(Td^2)$
- When sequence length $T$ is small, the projection term ($O(Td^2)$) dominates; when $T$ is large, attention's quadratic term $O(T^2d)$ gradually takes over — this is also the direct reason attention compute becomes the bottleneck in long-context scenarios

Only remembering the parameter-count formulas, without being able to say which of the two FLOPs terms, $O(Td^2)$ and $O(T^2d)$, dominates at what sequence length.

</details>

<details>

<summary>Q15. Why is SwiGLU's hidden size often close to $8d/3$?</summary>

- Ignoring bias, a dense FFN (two matrices) has $8d^2$ parameters ($4d$ expansion); a gated FFN (gate/up/down, three matrices) has $3dh$ parameters
- Setting the two equal: $3dh=8d^2 \Rightarrow h=8d/3$ — this is a **common default derived from parameter-count matching**, not an architectural law that "a gated FFN's hidden size must equal $8d/3$"
- Real Llama-family implementations typically start from $4d$, multiply by $2/3$, then round to a hardware-friendly multiple (plus an optional multiplier); the intermediate size can deviate noticeably from $8d/3$ (§3, §7.3 [D06] verifies with $d=96$ that both sides give 73,728 parameters)

Claiming $8d/3$ means "every Llama model's hidden size strictly equals this value"; unable to say this is a default derived from parameter matching, not a hard architectural constraint.

</details>

<details>

<summary>Q16. Are sequential attn→FFN and parallel attn+FFN equivalent?</summary>

- Not equivalent: in sequential Pre-LN, the FFN reads the residual stream **after** attention has updated it ($\text{FFN}(N_2(x+\text{Attn}(N_1(x))))$); in parallel, FFN and attention read the **same** normalized original input ($x$), independent of each other
- The judging method should not be "whether the two outputs happen to be unequal" (fragile) — it should reconstruct the actual input tensor the FFN really receives and match it exactly (§7.4 [D08] uses a forward hook to capture the input plus an independent recomputation for comparison)
- The parallel structure removes this dependency chain, which can reduce some synchronization overhead during training (§6.1)

Only answering "they should be different, I guess" without giving an executable judging method; or using the weak check "outputs are unequal" as a stand-in for a structural proof.

</details>

<details>

<summary>Q17. What problem does each of attention scale, QK-Norm, and logit soft-cap solve?</summary>

- Attention scale: the standard $1/\sqrt{d_h}$ (or a learnable temperature) scaling of the dot product, controlling the overall magnitude of the softmax input — the most basic numerical control
- QK-Norm: L2-normalizes each head's Q, K separately **before** the dot product, and **replaces** the standard fixed $1/\sqrt{d_h}$ scaling with a learnable scale/temperature parameter — this is the complete version of Henry et al.'s original definition; the specific implementation details in later models may differ
- logit soft-cap: applies a $c\tanh(z/c)$-style bounded transform **after** the dot product, to the score or the final vocabulary logits, reining in scores/logits that have already been computed
- The three act at different positions, and **cannot be stacked mindlessly without re-checking the scale** — Gemma-2 uses soft-capping, Gemma-3 switches to QK-Norm and removes soft-cap; it's not simple stacking (§6.2; full attention-entropy-collapse argument in normalization_init_tutorial.md §6.3)

Describing QK-Norm as "just normalizing Q/K," missing that it simultaneously replaces the fixed-scale component — a key part of it; treating QK-Norm and soft-cap as two names for the same thing.

</details>

<details>

<summary>Q18. What's the difference between packed QKV projection and three separate Linears?</summary>

- Mathematically equivalent: concatenating three independent weight matrices row-wise into one large matrix, then splitting the output back along the original dimensions, gives results that should match within reasonable tolerance for the given dtype once equivalent weights are copied over (not the stronger claim of "bit-identical") (§7.4 [D07])
- The difference is at the kernel level: the packed version is usually one large matrix multiply, reducing kernel-launch overhead; but packed **isn't always faster** — the actual payoff depends on matrix size, quantization format, and parallel/sharding layout
- SwiGLU's gate/up projections can likewise be packed — this is a classic engineering interview point of "same math, different implementation"

Describing numerical equivalence as "bit-identical"; assuming packed is always faster without considering the effects of specific size/quantization/parallel layout.

</details>

<details>

<summary>Q19. What's the essential difference between MLA and GQA?</summary>

- GQA compresses along the axis of **KV-head count/sharing relationship**: multiple Q heads share one K/V head, with the compression ratio determined by $n_\text{kv}/n_q$
- MLA compresses the KV's **content representation** via a low-dimensional latent bottleneck: each token's K/V is compressed into a latent vector of size $d_c \ll n_qd_h$ — note that MLA's cache usually contains not just the latent but also a decoupled, shared RoPE key, so "the entire K/V is projected into just one latent" isn't quite precise
- If RoPE is used and the absorbing trick is to be preserved, MLA must decouple RoPE into an independent, shared component; GQA doesn't need to deal with this problem (§4; full derivation in long_context_rope_yarn_mla_tutorial.md §9)

Describing GQA's compression as "shrinking $d_h$" (it actually shrinks the KV-head count, not the head dim itself); assuming MLA's cache has only the latent as its single item.

</details>

<details>

<summary>Q20. Why do prefill and decode have different performance bottlenecks? How does KV cache change the complexity?</summary>

- This is an inference-serving-level question, not a block-assembly-level question — during prefill, the entire prompt is computed in one shot, weights get reused many times, and arithmetic intensity is usually higher, making it easy to saturate compute; during decode, only 1 new token is computed per step, yet the whole layer's weights and the ever-growing KV must be re-read, so at small batch sizes it often falls into a bandwidth-bottlenecked regime
- KV cache turns "recomputing all historical K/V" into "computing only the new token's K/V + reusing the cache," lowering decode-stage per-step attention complexity from $O(t^2)$ to $O(t)$ ($t$ being the current position)
- The full prefill/decode bottleneck analysis, batch scheduling, and memory management are in llm_inference_serving_tutorial.md, not re-expanded here (§7.2)

Answering this as if it were a block-internal design question, ignoring that it's essentially a systems/scheduling-level question; not knowing which tutorial to check for the full derivation.

</details>

<details>

<summary>Q21. Which level do FlashAttention, PagedAttention, and GQA/MQA each optimize?</summary>

- FlashAttention: an IO-aware exact-attention kernel; it optimizes "how to compute" — using tiling/recompute to reduce HBM access, without changing attention's mathematical result or the model architecture (§6.4)
- PagedAttention: a KV-cache memory-management mechanism; it optimizes "how to store" — using a page table to map logical KV onto non-contiguous blocks of physical memory, reducing fragmentation; it belongs to the serving-systems level
- GQA/MQA: an architecture-level choice; it optimizes "how much to store" — reducing the number of KV heads, thereby reducing the size of the cache itself
- The three are optimizations at different levels and shouldn't be lumped together: the first two are engineering techniques at the execution/systems level, while the last is an architectural decision that must be fixed before training; full mechanisms in attention_tutorial.md (FlashAttention) and llm_inference_serving_tutorial.md (PagedAttention)

Blurring all three together as "techniques that all save memory," without distinguishing which level each acts on — algorithm execution, memory management, or model architecture.

</details>

<details>

<summary>Q22. Why have bias and dropout gradually been dropped in modern large-model pretraining?</summary>

- bias: in large-scale training, removing bias usually doesn't hurt performance and also saves a small number of parameters and compute, becoming a common default starting from the Llama family
- dropout: under large-scale single-epoch pretraining, the regularization effect provided by the data volume itself is already sufficient, and heavy dropout can instead slow convergence; this is an empirical engineering default, not something mathematically proven to be "necessary to remove"
- GPT-2-style still retains both; starting from Llama-style they're substantially trimmed — this is one of the change items explicitly listed in §2.2's bridge table

Answering "theory proves removing them is better" — in reality this is an empirical default under large-scale pretraining, not a strictly proven universal conclusion.

</details>

<details>

<summary>Q23. Where is the term "sandwich norm" inaccurate? How should the formula be written?</summary>

- "Sandwich norm" is an ambiguous colloquial name; the more accurate term is "one normalization before and one after each sublayer branch" (branch pre+post normalization)
- Accurate formula: $y=x+N_\text{post}(F(N_\text{pre}(x)))$ — the second normalization acts only on the **branch output**, not on the residual sum
- This is a different formula from the original Post-LN's $N(x+F(x))$: Post-LN normalizes the entire stream after addition, while branch pre+post only normalizes/stabilizes the branch output, with the residual backbone remaining a pure addition (§6.1; Gemma-2/3 adopt this topology, full variant lineage in normalization_init_tutorial.md §6.2)

Writing branch pre+post's formula the same as the original Post-LN's $N(x+F(x))$, missing the key difference of "what gets normalized."

</details>

<details>

<summary>Q24. What's different about the KV-cache contract between MLA and GQA?</summary>

- GQA caches $n_\text{kv}$ full K/V head pairs (each of dimension $d_h$) — essentially still "multiple heads of ordinary K/V," just fewer heads than $n_q$
- MLA's cache is one shared latent $c_t^{KV}$ (dimension $d_c$) plus one shared, decoupled RoPE key (dimension $d_h^R$) — not "fewer copies of ordinary K/V"; K and V are no longer stored independently either. The full absorbing trick, why RoPE can't be absorbed directly, and the algebraic derivation of the decoupling scheme are in long_context_rope_yarn_mla_tutorial.md §9, not repeated here

Imagining MLA's cache as "GQA with $n_\text{kv}=1$," ignoring that what it stores is a low-rank latent + decoupled RoPE key, not ordinary K/V.

</details>

<details>

<summary>Q25. When hand-writing an incremental-decoding causal self-attention with KV-cache support, which correctness pitfalls are easiest to miss?</summary>

- The K in the cache must be the one already RoPE-rotated in the standard implementation (if choosing to cache unrotated K, position must additionally be retained, see Q12); the new token's Q, K need the correct absolute position offset
- Under GQA/MQA, the cache's head dimension must always remain $n_\text{kv}$ — the broadcasted tensor must never be written back into the cache on any call
- The single-token causal mask, in the common scenario of "dense causal attention + dynamic cache + no padding/packing," genuinely is "sees all of 0..t" (constantly True), but scenarios like static cache, left padding, sliding window, and packed sequences may still need extra masking — it can't be generalized across the board
- Numerical agreement between full-forward and cached decoding is one of the most important regression tests, able to catch common bugs like causal-mask/RoPE-offset/cache-order issues, but this test is strong without being sufficient — it cannot replace separate tests for scenarios like mask broadcasting and cache-consistency checks (§7.4 [D04])

Believing that "the single-token mask is constantly True" holds in every scenario; treating full-forward vs. cached-decoding agreement as the sole and sufficient proof of correctness.

</details>

<details>

<summary>Q26. Where does GQA's constraint $n_q \bmod n_\text{kv} = 0$ come from? Does this constraint still hold for MLA?</summary>

- GQA's constraint comes from the definition of standard uniform GQA and common kernel layouts: $n_q$ Q heads are split evenly into $n_\text{kv}$ groups of $n_q/n_\text{kv}$ each, which requires $n_q\bmod n_\text{kv}=0$
- But this is not a mathematically unavoidable law — a non-uniform Q-head-to-KV-head mapping could in principle be defined; it's just that most model configurations and efficient kernel implementations assume equally sized groups
- MLA usually has no $n_\text{kv}$ grouping variable at all (it's a low-rank latent compression, not head grouping) — this divisibility constraint **does not apply to MLA**, which is also another direct piece of evidence for the judgment that "MLA is not a further compression of GQA" (§4)

Treating this constraint as a mathematical law universal to every KV-compression method; not knowing that non-uniform head sharing can in principle be defined, it's just uncommon in engineering practice.

</details>

<details>

<summary>Q27. Why is the claim "MHA→MQA→GQA→MLA is one evolutionary chain" wrong? How should the relationship among these four be organized correctly?</summary>

- By historical timeline, MHA→MQA→GQA genuinely is a valid timeline (MQA predates GQA — this historical fact is entirely correct on its own, and can't be used to refute this timeline)
- But by KV-head compression strength, the correct order is actually MHA→GQA→MQA — compression degree is not the same as historical order: MHA is the endpoint $n_\text{kv}=n_q$, MQA is the other endpoint $n_\text{kv}=1$, GQA is any value in between; MHA and MQA can be seen as the two endpoints of the same uniform-GQA compression-ratio axis
- MLA is an independent low-rank latent-cache route, compressing the hidden dimension rather than head count, and shouldn't be tacked onto the end of the "KV-head count" axis
- The real reason "the four are not one chain" is that **multiple distinct ordering dimensions exist** (historical order vs. compression strength vs. compression mechanism), not that there's only one timeline; the historical fact "MQA predates GQA" is entirely correct in itself and cannot be used as an argument against "this is one chain" — what actually needs refuting is the act of "forcing all four into one line along some single dimension" (§4)

**This is the focus of this fix — the original draft contained a genuine argumentative error**: the original draft used "MQA appeared before GQA" as evidence against "MHA→MQA→GQA→MLA is one chain," but this timeline itself is entirely correct, so using it as a rebuttal is self-contradictory — the correct rebuttal point is that "the compression-strength order (MHA→GQA→MQA) and the historical order (MHA→MQA→GQA) are not the same axis, and MLA is yet a completely different third axis"; the problem isn't with the fact "MQA predates GQA" itself.

</details>

### L3 — Expert-level questions

<details>

<summary>Q28. Given a Llama-3-like dense GQA block, how do you swap in MoE without changing the block's interface? Where does the block's contract change, and where doesn't it?</summary>

- Only the FFN slot is replaced: swap `SwiGLU FFN` for a combined `router + N expert FFNs` module; the attention slot (GQA, RoPE) and the norm slot (Pre-RMSNorm) all stay unchanged
- Three layers of contract need to be distinguished: ① **the tensor-shape contract is unchanged** — the main activation tensor is still `[B,T,d] -> [B,T,d]`, the residual-add position is unchanged, and swapping to `ffn=MoEFFN` in `ModernDecoderBlock` requires no other code changes; ② **the training return value may change** — an extra router loss or load-balancing statistic usually needs to be returned/passed along; ③ **the distributed-systems topology usually changes** — expert parallelism, all-to-all communication, and total parameter count vs. activated parameter count need to be reported separately
- During backward, only the experts actually routed to within this batch should receive gradients — you can't require every expert to have a nonzero gradient on every batch (full routing/capacity math in moe_tutorial.md §2–§4)
- Stack-level configuration like final norm and local/global layer layout is unrelated to this swap and doesn't need to change along with it

Assuming that switching to MoE is fine as long as "the tensor shape stays the same," ignoring that the training-return-value layer and the distributed-systems-topology layer usually need to change too; or, conversely, assuming the attention/norm slots need to be redesigned.

</details>

<details>

<summary>Q29. What systematic cost differences do parallel residual (GPT-J/PaLM) and sequential Pre-LN each have in training/inference?</summary>

- Dependency chain: parallel removes the dependency that "FFN must wait for attention to finish updating the residual stream," theoretically opening more room for overlapping compute/communication; in the sequential structure, FFN is strictly ordered after attention
- Quality: both topologies can be trained well in practice, and there's no evidence that one is absolutely superior at equal compute/tuning effort — this is an engineering trade-off, not a one-directional advantage
- Normalization overhead: if the parallel version uses a single shared norm (as in GPT-J), it performs one fewer normalization computation than the sequential version; a parallel implementation using two independent norms has the same number of normalization computations as the sequential version, just without the ordering dependency (§6.1)

Broadly claiming "parallel is always faster" or "parallel always has worse quality," without being able to specify which particular aspect — the dependency chain vs. the number of normalizations — causes the difference.

</details>

<details>

<summary>Q30. Given a new model's technical report, which fields inside the block would you focus on checking to determine its recipe?</summary>

- Norm topology (Post-LN / Pre-LN sequential / Pre-LN parallel / branch pre+post) + norm type (LayerNorm/RMSNorm)
- The positional-encoding scheme and the level it belongs to (learned absolute position outside the stack, or RoPE inside attention; whether there's a YaRN/NTK-style extension)
- Attention variant: $n_q$, $n_\text{kv}$ (MHA/MQA/GQA), whether QK-Norm is used, whether local/sliding window is used — **you cannot judge whether it's MLA using only $n_q,n_\text{kv}$**; MLA needs a separate check on the field "KV-cache representation/whether there's a low-rank latent structure"
- FFN variant: dense vs. gated vs. MoE, what the activation function is, and — if MoE — total parameters vs. activated parameters
- The choices on bias and dropout
- Final norm and local/global layer layout belong to **stack-level fields** — these should be checked separately, not lumped together with the block-internal fields above
- Checking item by item covers most block-recipe questions, but there is no single precise coverage percentage — new models may still have fields this tutorial doesn't cover (such as whether Q/K/V head dimensions match, attention scale/temperature, mask/pattern details, cache layout, and so on)

Drawing a conclusion from looking at just one field, "did it use GQA or MHA"; judging MLA using only $n_q,n_\text{kv}$ (MLA's core feature is its low-rank latent structure, not head count); also counting final norm/local-global layout as "block fields" and tallying them together; claiming some fixed percentage of coverage.

</details>

## 📚 References

- **Transformer (original encoder-decoder)** — Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762 (2017), NeurIPS 2017.
- **GPT-2 (decoder-only, Pre-LN, learned absolute PE)** — Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report (2019) (**no arXiv id**).
- **GPT-3** — Brown et al., *Language Models are Few-Shot Learners*, arXiv 2005.14165 (2020), NeurIPS 2020.
- **GPT-J (parallel residual)** — Wang & Komatsuzaki, *GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model*, EleutherAI open-source release (2021) (**no formal paper/arXiv id**).
- **PaLM (parallel residual, MQA)** — Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311 (2022).
- **RoPE** — Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2104.09864 (2021). The LLaMA family is a major adopter, not the originator.
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
- **Pre-LN vs. Post-LN gradient argument** — Xiong et al., *On Layer Normalization in the Transformer Architecture*, arXiv 2002.04745 (2020), ICML 2020.
- **DeepNorm / DeepNet** — Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*, arXiv 2203.00555 (2022).
- **RMSNorm** — Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv 1910.07467 (2019), NeurIPS 2019.
- **QK-Norm** — Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245 (2020), EMNLP 2020 Findings.
- **GELU** — Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, arXiv 1606.08415 (2016).
- **GLU Variants / SwiGLU** — Shazeer, *GLU Variants Improve Transformer*, arXiv 2002.05202 (2020). Proposed and systematically evaluated these Transformer FFN variants; the original GLU and the underlying idea of SiLU/Swish each have earlier origins.
