## §0 Overview

A decoder-only causal LM predicts the next token from the preceding text. The pretraining pipeline is responsible for corpus processing, supervision-signal construction, parameter updates, and training-state recovery.

1. Training compute is approximately $C\approx6ND$, where $N$ is the parameter scale of the main matrix multiplies, $D$ is the number of tokens presented and supervised, and $C$ is counted in FLOPs.
2. MoE's per-token compute is estimated with active parameters; the aggregate size of the model state and of checkpoints is counted with total parameters.
3. Kaplan and Chinchilla give different budget-allocation exponents, and the difference involves the fitting method, the training horizon, and the learning-rate schedule.
4. A small model can keep lowering loss on more new, related data, trading extra training compute for a lower deployment cost.
5. Near-duplicate documents are first grouped into dedup clusters, and whole clusters are then assigned to train/val/test, reducing the validation bias caused by duplication across splits.
6. Document-independent packing needs a block-diagonal causal mask, plus separate handling of cross-document target labels.
7. The loss of each update is the NLL summed over all valid targets, divided by the total number of valid targets.
8. The number of input tokens of one update equals the product of sequences per DP replica, sequence length, accumulation steps, and DP world size.

```text
raw documents → corpus factory → raw-document shards
raw-document shards → tokenizer → token-level sampling → chunk/truncate → tokenized shards
tokenized shards → sampler → packing / labels / masks → parameter update
parameter update → scheduler / token counting → checkpoint / resume
```

## §1 Training budget and scaling laws

### 1.1　$C\approx 6ND$: where the coefficient comes from, and what it ignores

Training compute for a decoder-only dense Transformer is commonly approximated as

$$C \approx 6ND$$

This approximation was first systematically laid out by Kaplan et al. (2020). The coefficient comes from: one multiply-add counts as 2 FLOPs; for each token the forward pass runs through all the main matrix multiplies — QKV projections, the attention output projection, FFN up/down projections — costing approximately $2N$ FLOPs, so the forward pass over $D$ tokens totals $\approx 2ND$. The backward pass computes both activation gradients and weight gradients, roughly twice the forward cost, i.e. $\approx 4ND$. Forward plus backward $\approx 6ND$; the $6$ is likewise an approximate coefficient, more rigorously written $k\approx 6$.

**Parameter-counting convention**: $N$ is the compute-bearing, non-embedding parameter count. The embedding lookup is an index operation rather than a matrix multiply, and mechanically counting embedding parameters into $N$ systematically overestimates that portion of the compute. Whether and how the output vocabulary projection is counted separately varies across papers and implementations, so when reading any specific FLOPs estimate, first confirm its parameter-counting convention.

The terms $6ND$ ignores and simplifies fall into three categories. **Sequence-length term**: attention-score compute grows as $L_{\text{seq}}^2$ within each sequence, while $N$ itself does not vary with sequence length; at fixed total training tokens $D$ the number of sequences is approximately $D/L_{\text{seq}}$, so the total attention cost grows approximately as $D\times L_{\text{seq}}$, not $D\times L_{\text{seq}}^2$. The longer the sequences, the less this term can be neglected relative to the other linear terms in $6ND$. **Model and training details**: the vocabulary projection is a substantial output-layer matrix multiply when $V$ is large; non-matmul operators such as norm/softmax are not counted; activation recomputation re-runs the forward pass to save memory, pushing the effective backward multiplier above 4; MoE has two parameter conventions, active and total (§1.2); and communities differ on whether "1 FLOP" counts a multiply-add once or twice. **Hardware efficiency**: $6ND$ does not equal the FLOPs the GPU actually executes, and cannot directly predict wall-clock time.

A rough GPU-days estimate additionally needs the GPU count, peak FLOPs, and MFU:

$$t\approx C/(n_{GPU}\times\text{peak FLOPs}\times MFU)$$

Communication overhead, recomputation, and data stalls all make this an estimate rather than a precise prediction.

### 1.2　MoE: active parameters for per-token compute, total for memory

MoE's per-token compute is estimated with **active parameters** — the expert parameters that actually participate in that token's forward and backward computation. Model capacity, memory footprint, and checkpoint file size are counted with **total parameters**, which corresponds to the aggregate model-state size. Actual per-GPU memory also depends on the sharding scheme (see Q25).

For a 671B/37B-style model (public configurations like DeepSeek-V3): the $N$ in $6ND$ uses the 37B scale to estimate the budget, while checkpoint storage is counted at 671B. Routing, load balancing, and the aux-loss derivation are in `moe_tutorial.md`.

### 1.3　Kaplan (2020) vs Chinchilla (2022): the exponents must not be conflated

The two foundational papers give different optimal-allocation exponents.

**Kaplan et al. (2020)**:

$$N_{opt}\propto C^{0.73},\qquad D_{opt}\propto C^{0.27}$$

When the compute budget grows, Kaplan's conclusion is to allocate more to parameters than to data.

**Hoffmann et al. (2022, "Chinchilla")**: the paper cross-validates with multiple estimation methods and gives no single set of "exact exponents". Different estimation methods yield a family of close but not identical results, roughly **0.50/0.50**, **0.49/0.51**, and **0.46/0.54**. The most frequently cited parametric loss model is written as

$$L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

with fitted values around $\alpha\approx 0.34,\ \beta\approx 0.28$. $E$ is the fitted asymptotic loss floor, roughly comprising the entropy of the data itself plus the approximation error of that model family; the fit cannot uniquely decompose the two. $A/N^\alpha$ is the price paid for parameters not being large enough, $B/D^\beta$ the price for data not being plentiful enough.

Constrained optimization of $L(N,D)$ under the compute budget $C=kND$ ($k=6$), eliminating variables via Lagrange multipliers, yields the analytic solution:

$$N_* = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}\left(\frac{C}{k}\right)^{\frac{\beta}{\alpha+\beta}},\qquad D_* = \left(\frac{\beta B}{\alpha A}\right)^{\frac{1}{\alpha+\beta}}\left(\frac{C}{k}\right)^{\frac{\alpha}{\alpha+\beta}}$$

Substituting the rounded $\alpha=0.34,\beta=0.28$: $\dfrac{\beta}{\alpha+\beta}=\dfrac{0.28}{0.62}\approx 0.452$, $\dfrac{\alpha}{\alpha+\beta}=\dfrac{0.34}{0.62}\approx 0.548$, i.e.

$$N_*\propto C^{0.452},\qquad D_*\propto C^{0.548}$$

Computed with the paper's full-precision $\alpha,\beta$, the commonly reported values are about 0.46/0.54. [D06] in §6 turns this pair of exponents into an executable fitting unit test.

"About 0.50/0.50" is the coarse-grained empirical conclusion that parameters and tokens scale roughly proportionally; "about 0.46/0.54" is the specific exponent fitted by the parametric loss model. "About 20 training tokens per parameter" is an empirical approximation, not a universal constant derivable from $\alpha,\beta$ alone. $D_*/N_*$ also depends on the constants $A,B$ and on $k$; when $\alpha\ne\beta$, $D_*/N_*\propto (C/k)^{(\alpha-\beta)/(\alpha+\beta)}$ drifts slowly with $C$. Holding $A,B,\alpha,\beta,k$ fixed and keeping this fitted model, the ratio grows by about 1.56x per 100x compute at that exponent, and this derivation is mathematically sound. It cannot be extrapolated into the empirical claim that the true optimal ratio reliably grows 1.56x per 100x compute: Chinchilla's two other independent estimation methods (IsoFLOP curve fitting, parameter/data ratio fitting) give an approximately constant ~0.5/0.5; when $\alpha,\beta$ are close, fitting error strongly affects the exponent $(\alpha-\beta)/(\alpha+\beta)$; and $A,B,k$ need not stay constant across data distributions, architectures, and training recipes spanning orders of magnitude of compute — while the derivation assumes exactly that they are all fixed.

The technical reasons for the Kaplan-Chinchilla divergence involve datasets, experimental coverage, loss-fitting methods, and training-horizon design. The key difference is the training schedule: Kaplan heavily used losses at intermediate points of single training curves, whose LR-schedule endpoint was not configured for that budget; Chinchilla configured a matched training schedule for each predetermined token horizon — e.g. cosine-decay length aligned to the target token count — before comparing.

### 1.4　Over-training small models and deployment cost

Overfitting means training loss falling while validation loss rises. A model of fixed size $N$ fed additional new, non-repeated, same- or related-distribution data usually still lowers loss, which does not meet that definition. What actually happens: as $D$ keeps growing at fixed $N$, the final compute $C=kND$ grows in lockstep, and you arrive at a larger $C$; relative to that $C$'s compute-optimal frontier point $(N_*,D_*)$, the configuration $(N,D)$ has a smaller $N$ and a larger $D$. This configuration is called **over-trained** — an economic judgment relative to the compute-optimal frontier.

Industry deliberately trains many small models far past the Chinchilla-optimal ratio, e.g. choosing a smaller model at equal capability and feeding it more tokens. The justification is deployment-side inference cost: memory, latency, per-token serving cost. One-off extra training compute buys cheaper inference over the model's entire lifecycle — training-compute-optimal and lifecycle-cost-optimal are two different objectives.

### 1.5　Choosing $N,D$ for a given budget

1. First pin down the optimization objective. Training-compute-optimal applies the $N_*,D_*$ analytic solution directly; inference-aware optimal first caps a smaller $N$ by the deployment budget, so at fixed $C$ the $D$ grows accordingly — deliberately over-training a small model.
2. Long-context training brings in the $L_{\text{seq}}^2$ attention term that $6ND$ ignores; budget for it separately.
3. MoE models use active parameters for the compute budget and total parameters when reporting storage and memory budgets.
4. Clarify the accounting for $D$: the number of tokens presented during training, or the deduplicated effective data volume. A highly repetitive corpus makes the effective data volume fall below the nominal $D$, which is also the concern of the post-Chinchilla line of data-constrained scaling work.

## §2 The corpus factory: from web snapshots to training shards

The corpus pipeline decides whether training runs stably and correctly, and its impact often exceeds that of model architecture and hyperparameters.

### 2.1　The eleven-step pipeline overview + lineage

| Step | What it does | Key pitfall |
| --- | --- | --- |
| 1. Raw snapshot + license records | Save crawl time, source, license metadata | copyright/license traceability |
| 2. Extraction/encoding normalization | Web main-content extraction, removing nav bars/ads, Unicode normalization | often affects quality more than downstream classifiers (§2.2) |
| 3. Language identification | Route by language | code-mixed text and short text have low confidence |
| 4. Basic quality filtering | Length/symbol-ratio/repeated-n-gram/perplexity filtering | one-size-fits-all thresholds wrongly delete long-tail but legitimate text |
| 5. Exact dedup | Fully duplicated document/substring dedup | catches only byte-level duplicates, misses near-duplicates |
| 6. Near dedup | MinHash/LSH for near-duplicates | similarity ≠ semantic similarity (§2.3) |
| 7. Privacy/safety processing | PII redaction, illegal-content filtering | needs a dedicated process independent of quality filtering |
| 8. Benchmark decontamination | n-gram overlap detection against eval sets | a different problem from train-val leakage (§2.6) |
| 9. **Cluster-level data split** | **Assign splits by whole dedup clusters from step 6** | **dedup must be complete before this step** (§2.5) |
| 10. Mixture sampling (document-level quotas) | Decide per-domain document contributions by domain weights | document-level weight ≠ token-level weight (§2.7) |
| 11. Deterministic sharding | Emit reproducible raw-document shard files | must be versioned together with lineage (§2.8) |

**Two stages**: all 11 steps in the table are in the raw-document stage. Step 10's mixture sampling is a **document-level** quota deciding how many documents each domain contributes; step 11 emits raw-document shards. The tokenizer runs after that, turning raw-document shards into a token stream. The tokenized stage does token-level mixture sampling and long-document chunk/truncate, writes tokenized training shards, and hands them to the data-stream/sampler layer of §2.10. After the split, train and val/test are processed separately. Mixture sampling happens once at the document level and once at the token level, and the two do not mean the same thing.

**Data lineage**: every training sample should ideally be traceable to its source dataset or URL, snapshot/crawl version, extraction and filtering pipeline version, the quality filters and thresholds it hit, its dedup cluster ID, its split assignment, and the final shard file plus offset it landed in. These records are the only basis, months later, for investigating "did some output memorize training data" or "was some benchmark contaminated".

### 2.2　Extraction and quality filtering

Web main-content extraction (stripping nav bars, ads, footer templates, comment-section noise) and encoding/Unicode normalization sit at the very top of the pipeline. Upstream extraction errors systematically press large amounts of structural noise — navigation text, repeated templates, mojibake — into every document, and their impact often exceeds fine-tuning of downstream quality classifiers. This is an empirical observation, indirectly supported by the C4 corpus audit (Dodge et al.) and the RefinedWeb/FineWeb results (Penedo et al.), not a universal ordering that extraction always outranks classifiers. The training corpus's normalization rules must match those used when the tokenizer was trained, otherwise online/offline segmentation drift appears. Normalization options and byte fallback are in `tokenization_tutorial.md` §3–§4.

Language identification usually routes documents with a lightweight classifier, e.g. a fastText-style n-gram model; code-switched text and short text get low-confidence judgments. Basic quality filtering stacks several heuristics — line-length distribution, symbol/letter ratio, repeated-n-gram share — and/or a small auxiliary language model's perplexity score; in essence it uses cheap signals to screen out the worst batch, not fine-grained semantic judgment.

### 2.3　Exact dedup vs near dedup

**Exact dedup** is byte-level, hash-level full matching: whole-document dedup, or suffix-array-based cross-document duplicate-substring dedup. It catches only identical content, missing near-duplicates like "one word changed in the title".

**Near dedup** (MinHash + LSH) has four steps: represent documents as shingle (n-gram) sets; estimate pairwise Jaccard similarity with MinHash (Broder, 1997); use LSH bucketing for candidate recall; verify candidate pairs with exact Jaccard.

MinHash estimates the Jaccard similarity of shingle sets. Two documents discussing the same concept with entirely different wording can have very low Jaccard similarity; two documents dominated by templated text (nav bars, license notices, boilerplate code) but topically unrelated can have high Jaccard similarity. MinHash/LSH catches **surface-level duplication**: mirror sites, near-verbatim copies, templated documents.

LSH banding only does candidate recall. Under the idealized assumption that MinHash rows and hash functions are approximately independent, the candidate probability is $P=1-(1-s^r)^b$, with $b$ bands of $r$ rows each. Bucketing only guarantees that candidate probability rises monotonically with similarity; it does not guarantee a candidate is a true near-duplicate.

Embedding-threshold semantic dedup can wrongly delete legitimate "same topic, different expression" text, and this over-deletion hits low-resource languages, dialects, and templated technical documents unevenly, because those texts' own embedding distributions are more concentrated. It and near dedup address different levels of duplication.

### 2.4　Privacy/safety processing + benchmark decontamination

Privacy and safety processing is a process independent of "quality level": PII redaction (emails, phone numbers, keys) and illegal/harmful content filtering use dedicated detection rules and thresholds, and are not tuned together with quality filters.

**Benchmark decontamination** detects n-gram overlap between the training corpus and target eval sets, removing or flagging overlapping documents, so that eval answers are not seen during training. **Exact matching at the 13-gram scale** is the concrete practice explicitly adopted in the GPT-3 paper (Brown et al., 2020) and widely borrowed; teams differ in the n-gram lengths they actually use, so it should not be treated as an industry-wide standard.

### 2.5　Dedup before split: cluster-level split

Run near dedup first, partitioning all documents into connected components (dedup clusters), then assign **whole clusters** atomically to train/val/test; documents in the same cluster must never be split across splits. What this relies on is the **transitive closure**: even if document A was only directly compared with B, and B only with C, with A and C never directly compared, as long as A~B and B~C, A and C must land in the same split. [D01] in §6 turns this rule into an executable assertion with union-find.

If you split first and dedup later, near-duplicate content (mirror sites, reposted and rewritten articles, different instances of templated documents) can easily land one copy in train and another in val. The model has "seen" a near-verbatim copy of a validation sample during training, so validation loss is **spuriously low** — not because the model generalizes well, but because it has effectively memorized a rewrite of the validation sample. This misleads early stopping, model selection, and any downstream quality reporting based on that validation loss. For a systematic discussion of dedup's effect on downstream model quality, see Lee et al., 2021.

### 2.6　Train-val leakage, benchmark contamination, and cross-document context

| Type | Definition | Root cause | Response |
| --- | --- | --- | --- |
| train-val leakage | near-duplicate content crosses splits | split before dedup | cluster-level split (§2.5) |
| benchmark contamination | training corpus contains eval-set content | no decontamination check | n-gram overlap detection (§2.4) |
| packed-sequence cross-document history context | after packing, document B can read document A's tokens | causal mask allows reading cross-document history | **usually not data leakage but a sequence-semantics choice** (§3.4) |

The third is **cross-sample context coupling**: the later document reads earlier tokens, not future labels — a modeling choice at the level of training dynamics and statistical independence, not information leakage in the evaluation sense. Whether to sever it with a block-diagonal mask depends on whether you require the modeling assumption that every document must be a statistically independent sample.

### 2.7　Mixture sampling: domain weights need not equal token weights

Mixture sampling has at least three different "sampling units", and the same domain-weight configuration produces **different** final token distributions under each:

- **Sample by document**: first select a document pool by domain weights, then draw documents uniformly or weighted from the pool, taking whole documents;
- **Sample by token**: cut fixed-length blocks directly from the concatenated token stream, decoupled from document boundaries;
- **Sample domain first, then document**: first pick a domain by domain weights, then pick a document within it.

The differences come from domains having different **average document lengths**: a domain with longer average documents contributes more tokens at the same document-count weight. The relationship between domain weights and token weights splits into two cases. When configured as document counts or document weights but used directly as token weights, the two are usually unequal, and this is the source of the bias. When the configuration itself is defined as token-level target shares, the two can be equal in expectation.

Whichever sampling unit is used, the actual token share must be measured and reported separately after sampling, not just the configured values. For a systematic approach to domain reweighting, see DoReMi (Xie et al., 2023).

This also ties into the accounting for $D$ in §1: the $D$ in scaling laws usually means the tokens **presented** to the model during training, i.e. the total number of times tokens flow through the optimizer, and is not guaranteed to equal the corpus's unique token total. If some domain is repeatedly oversampled (multiple epochs), or the corpus still contains many residual near-duplicates, the nominal $D$ is inflated above the effective data volume, and the excess does not carry the same statistical information as new data. For a systematic discussion of how repeated data affects effective $D$, see Muennighoff et al. (2023).

### 2.8　Deterministic sharding + the tokenizer interface

**Deterministic sharding**: given the same corpus, the same mixture configuration, and the same random seed, "which document goes into which shard file, at which position" must be reproducible. This property directly underpins deterministic resume, where a restart must pinpoint exactly where consumption stopped; it also underpins spike triage, where reproducing the specific batch that caused a spike depends on exactly reconstructing the data consumed at the time.

The corpus factory's output toward the tokenizer is a small, frozen contract: a version-locked tokenizer artifact (with its hash), fixed BOS/EOS/PAD IDs, fixed normalization rules, byte-fallback behavior, and a bytes/token ratio for budget estimation. How the tokenizer itself is trained (BPE/WordPiece/Unigram) and SentencePiece framework details are in `tokenization_tutorial.md` §2–§5.

### 2.9　Long-document chunking / truncation

A single document's raw length often exceeds the training context length, sometimes by many multiples — a book or a long code repository can far exceed a window of a few thousand to tens of thousands of tokens. Such documents are **chunked** before entering packing, with four decision points:

- **Whether to overlap**: non-overlapping chunking is simplest, at the cost of hard-cutting the context at every chunk boundary; overlapping chunking keeps an overlap region between adjacent chunks (e.g. tens to hundreds of tokens), mitigating the missing context at boundaries, at the cost of the same original text being shared by multiple chunks.
- **Whether the overlap region counts toward loss twice**: for the overlapping tokens shared by two adjacent chunks, if both chunks count loss, that content is effectively oversampled, distorting its true weight in the total $D$ and in $B_{target}$. Usually the loss should be kept only in the chunk that sees the longer preceding context, and the other chunk should mark it as pure context.
- **Whether to insert EOS at artificial chunk boundaries**: if each chunk should be treated as an independent sample, insert EOS/BOS boundaries at the cuts and assign new doc-ids, so the downstream block-diagonal mask and boundary loss-mask rules can recognize "these are actually two artificially cut segments of the same document"; without boundary markers, multiple chunks are treated as a continuous continuation of the same document.
- **How position ids / doc-ids carry over**: if each chunk is an independent sample, each chunk should have its own doc-id, and whether to reset position ids follows §3.4; if chunks are logically still the continuation of one document, the doc-id should stay consistent across chunks and position ids should be numbered continuously rather than restarting from 0 per chunk. This choice must stay consistent with how a "document" is defined at training and inference time, otherwise you get inconsistencies like treating chunks as independent samples in training but assuming positional continuity at inference.

None of these decisions has a universal default, but they must be recorded explicitly in the lineage. If a piece of training data has been chunked, its lineage should trace back to which chunk of which original document it is, whether it overlaps with adjacent chunks, and whether the overlap region counts toward loss. Without those records, it is impossible to determine afterward whether an anomalous loss weight on some content was caused by the chunking policy.

### 2.10　The data-stream / sampler layer

After tokenized training shards are written to disk, what actually reads them at training time is a **data-stream / sampler layer**, and both gradient accumulation and deterministic resume depend on this layer's state definitions:

- **Shard shuffle**: whether the shard-file read order is shuffled before each epoch or each traversal, and the random seed and state used for the shuffle;
- **Domain-sampling RNG**: if the mixture is sampled dynamically at training time rather than pre-mixed offline, you need a domain-selection random number generator independent of other randomness sources (dropout, data shuffling), whose state must also be capturable by checkpoints;
- **With / without replacement**: whether documents or shards within a domain may be resampled. Multi-epoch training, or data insufficient to support the compute-optimal $D$, usually requires sampling with replacement, and the repetition itself must then be recorded, because it directly affects the "nominal $D$ vs effective data volume" accounting;
- **Mutually exclusive DP-rank sharding**: the shard/sample sets read by each DP rank must be pairwise disjoint, otherwise the same data is counted by multiple ranks simultaneously, which is equivalent to implicitly double-counting $D$;
- **Worker/prefetch state**: multi-process/multi-thread dataloaders typically prefetch several batches into a queue, and that queue's own state — what has been prefetched, what remains unconsumed — is part of the training state. If resume restores only the "which sample have I read" cursor without correctly handling data already dequeued but not yet consumed, that data is skipped or consumed twice;
- **Cursor position**: all the randomness and ordering decisions above must ultimately converge to a cursor that can be saved and restored precisely, usually "which byte or sample offset in which shard file". The cursor's own precision determines how exact the "identical sample sequence" tier can be.

### 2.11　After the cluster split: where train and val/test diverge

The cluster-level split settles only one thing — which dedup clusters belong to which split; once the split is done, train and val/test follow two completely different downstream processes.

**train** continues through the full §2.7–§2.10 process: mixture sampling (with possibly dynamically adjusted domain weights), shuffling, with-replacement resampling when needed, and all the state of the data-stream/sampler layer.

**val/test must be version-frozen**: once fixed, the same val/test set keeps its sample content, size, and scoring protocol unchanged for the entire training lifecycle, and across multiple training runs when comparing fairly. It is not affected by the dynamically adjusted mixture, nor diluted or reweighted by resampling. Per-domain reporting requires an independent, comparable validation set for each domain, not a subset that tracks the train mixture.

If val/test also needs to cover multiple domains, the within-domain sampling should be determined once at the split stage and frozen, rather than dynamically resampled each epoch as train is. Otherwise validation losses across checkpoints are no longer measured with the same yardstick.

## §3 Objective-to-tensor: how the token stream becomes supervision

### 3.1　Teacher forcing + shifted labels

Standard next-token training uses teacher forcing: given a token sequence $x_1,\dots,x_T$, the **label** at position $i$ is $x_{i+1}$, the sequence shifted right by one. The model outputs the predictive distribution $p(x_{i+1}\mid x_{\le i})$ at position $i$. The loss is the NLL over all **valid** target positions; the exact normalization formula is in §3.5. The last position of a sequence has no next token, so its label is marked `IGNORE`, using the sentinel value $-100$ per the PyTorch convention.

### 3.2　Causal mask + EOS/BOS boundary rules

A plain causal mask $M[q,k]=\mathbb{1}[k\le q]$ only guarantees "cannot see the future". Around document boundaries there are two supervision rules:

- **Keep** the supervision "document's last content token → EOS" — the direct signal from which the model learns when to stop;
- **Mask (per configuration)** the supervision "previous document's EOS → next document's first token (or BOS)" — the label at that position is merely an artifact of concatenation, not any meaningful generation continuation.

Both rules can be produced automatically by **one and the same** document-boundary rule. Treat EOS as the last token of its document, with the same id as the rest of that document in the doc-id array, then construct labels with the rule "a label is valid iff the current position and the next position belong to the same document". Content-end token to EOS naturally shares a document, so the rule says valid; EOS to the next document's first token naturally crosses documents, so the rule says invalid. No separate special-case logic for EOS is needed, and [D04] in §6 demonstrates this.

When assembling chat templates, using both the template's built-in BOS and the tokenizer's `add_special_tokens` easily inserts BOS twice and pollutes the supervision signal; details are in `tokenization_tutorial.md` §5.5.

### 3.3　Padding: the loss mask is mandatory; whether the key mask is depends on layout

Padding involves two distinct concerns: the **key padding mask** prevents query positions from attending to padding keys; the **loss mask** excludes positions whose target is padding from the loss, without which gradients are computed against meaningless filler targets.

- **The loss mask is always mandatory** — without it, the loss is guaranteed to be polluted by filler targets;
- **The key mask can be omitted under strict causal + right padding** — real query positions are naturally smaller than the padding start, so the causal mask ($k\le q$) already blocks those future padding keys, and omitting it does not change real tokens' outputs;
- **The key mask is mandatory under left padding, holes inside sequences, or other layouts** — the causal condition alone cannot exclude padding keys, and real queries can easily read meaningless content.

A general-purpose implementation constructs both; only omit the key mask, with justification, when you can confirm you are strictly in the "right padding + causal" special case and are willing to maintain that implicit assumption long-term.

Many tokenizers have no dedicated PAD token and simply set `pad_id` to `eos_id`, which is not necessarily wrong. The accident happens when downstream code indiscriminately builds masks via `input_ids == pad_id`: this also masks out **every real EOS position** in the corpus as padding. The supervision for "where documents should end" is systematically removed from the training signal, significantly weakening the model's stopping ability, and is especially fatal when training from scratch. Masks must be built from the **actual padding positions** — each sequence's valid length `valid_len`, or an explicit padding flag carried with the batch — never from equality with some token ID. [D05] in §6 turns this mask construction into executable assertions.

### 3.4　Document packing: attention, labels, and positions

Packing multiple documents into one training sequence is a common way to raise effective batch utilization. A plain causal mask only prevents reading the future; it does not prevent a later document from reading an earlier document's history tokens. Many systems deliberately treat the packed token stream as one **EOS-separated continuous stream** with a plain causal mask.

If the semantics require "each document is an independent sample", you need a **block-diagonal causal mask**:

$$M[q,k] = (k\le q)\ \wedge\ (\text{docid}[q]=\text{docid}[k])$$

Whether to reset position IDs within each document depends on the position-encoding scheme and training semantics; it is not a universal rule:

- **Absolute position embeddings**: without a reset, later documents start from a huge position number that does not belong to them, so usually reset;
- **Standard RoPE**: depends only on the relative position difference between query and key. Once the block-diagonal mask restricts attention to within documents, adding the same offset to a whole document does not, in ideal math, change the relative phases within the document, so attention is not polluted merely because numbering did not start from 0;
- **Variants and engineering considerations**: RoPE length-extrapolation/scaling variants, numerical behavior at finite precision, and the desire to train every document strictly from position 0 to align with the position distribution at inference — these still lead most implementations to reset.

Blocking attention does not automatically mask cross-document prediction labels. After adding the block-diagonal mask, if labels are not handled separately, document A's last position is still, by default, asked to predict document B's first token. That is a cross-document target through and through, and it must be excluded separately by the boundary loss mask of §3.2.

**Numerical example** (tokens = `[11,12,13,21,22]`, doc IDs = `[0,0,0,1,1]`): under independent-document semantics the shifted labels are `[12,13,IGNORE,22,IGNORE]`; the block mask asserts `M[3,2]=False` (position 3 belongs to document 1, position 2 to document 0 — blocked for crossing documents even though $k\le q$), `M[4,0]=False`, `M[4,3]=True`; under continuous-stream semantics the same position has `M[3,2]=True`, a legitimate configuration choice.

**Functional test**: construct $QK^\top=0$, so softmax degenerates to uniform attention over the allowed keys; set document 0's three positions' values to `[100,100,100]` and document 1's two positions' values to `[0,0]`. Under the plain causal mask, position 3 (document 1's first token) outputs the mean over allowed keys `[0,1,2,3]` $= (100+100+100+0)/4=75$; under the block mask only key `3` itself remains, so the output is $0$. Now change document 0's values from 100 to 1000: in block mode document 1's output is completely unchanged, still 0; in continuous-stream mode the same output goes from 75 to 750, because it never severed the coupling with document 0. Checking only the mask's shape, or whether each row normalizes to 1, cannot catch bugs where the boundary logic is wrong but the shape still looks normal. [D03] in §6 is the executable version of this test.

### 3.5　Normalizing the loss by valid targets

$$\mathcal{L} = \frac{\sum_i \mathrm{NLL}_i}{\#\{\text{valid target tokens}\}}$$

Sum the NLL over all valid target tokens in the **global batch**, then divide by the total valid-token count. In implementation, accumulate two quantities separately — the NLL sum and the valid token count — across the whole gradient-accumulation window and across DP-rank communication, dividing once at the end.

Another way to write it is to first compute the mean loss within each micro-batch, then average those means. When micro-batches have different valid-token counts — different padding ratios, different packing efficiency, or an in-progress short-to-long context-length switch — this implicitly overweights micro-batches with few tokens and underweights those with many, deviating from the truly token-weighted loss.

An MoE model's total loss usually also adds the router's auxiliary loss term per configuration (router aux loss), typically far smaller in magnitude than the main loss; its mechanics are in `moe_tutorial.md`.

### 3.6　Exact PPL/BPB formulas

$$\mathrm{PPL} = \exp\left(\frac{\sum_i \mathrm{NLL}_i}{\#\text{valid target tokens}}\right),\qquad \mathrm{BPB} = \frac{\sum_i \mathrm{NLL}_i}{\#\text{bytes}\cdot\ln 2}$$

PPL is directly comparable only when tokenizer, normalization, and evaluation text are all identical. The denominator's token count itself depends on the tokenizer, so even at identical total NLL, a tokenizer that segments more finely drives loss/token down. BPB replaces the denominator with the UTF-8 byte count of the same text, making it more robust to tokenizer differences, though still affected by whether encoding and preprocessing match. The full derivation and the numerical example of "loss/token differs by 2x yet BPB is identical" are in `tokenization_tutorial.md` §1.3 and §7 [D07].

## §4 Update accounting: what the global batch actually is

The conceptual execution order of the stages within one optimizer step: model/config contract → forward → loss (NLL sum divided by the valid target token count) → backward → gradient sync (DP all-reduce; mind sum/mean reduction semantics). Then comes unscale (AMP), one gradient clipping on the final gradient after the accumulation window ends, the optimizer step, the scheduler and consumed-token counter advance, and finally the atomic checkpoint write.

### 4.1　The global batch and three token counts

$B_{micro,seq}$ is the number of sequences per micro-batch per DP replica, $L_{seq}$ the sequence length, $G_{acc}$ the gradient-accumulation steps, $W_{DP}$ the data-parallel world size. When TP/PP > 1, one DP replica itself spans multiple GPUs, so "per GPU" and "per DP replica" are not the same thing.

**Tokens fed to the model** (processed token slots, regardless of valid loss participation):

$$B_{input} = B_{micro,seq}\times L_{seq}\times G_{acc}\times W_{DP}$$

**Only $W_{DP}$ directly multiplies the number of independent samples.** Tensor parallelism (TP) and pipeline parallelism (PP) partition the computation of the **same** samples, and multiplying their world sizes into $B_{input}$ is the most common source of global-batch miscalculation. Communication primitives and partitioning mechanics are in `distributed_training_tutorial.md`; the data consumed by DP ranks must not overlap, otherwise $D$ is double-counted.

**Valid targets actually participating in the loss**:

$$B_{target} = \sum_i m_i$$

where $m_i$ is the valid-target mask count of sequence $i$. $B_{input}$ and $B_{target}$ are generally unequal, even with **no padding at all**. Under shifted labels, each sequence's last position has no next token and is marked `IGNORE`, so valid targets per sequence number $L_{seq}-1$, not $L_{seq}$. Stacking on padding or packing's document-boundary loss mask shrinks $B_{target}$ further below $B_{input}$.

The three accounting conventions map to different quantities:

| Convention | Quantity | Where it is used |
| --- | --- | --- |
| processed token slots | $B_{input}$ ("how many token positions passed through") | advancing the consumed-token counter of §5.1, token-aligned scheduling |
| non-padding input tokens | $B_{input}$ minus padding positions | estimating actual effective GPU throughput, evaluating packing/padding efficiency |
| valid target tokens | $B_{target}$ | the loss-normalization denominator of §3.5; the $D$ in §1's scaling laws should track this convention when padding/boundary masking is present |

### 4.2　Conditions for gradient accumulation to be strictly equivalent to "one big batch"

For gradient accumulation to be **strictly equivalent** to running one big batch directly, six conditions must hold at once:

1. **Same samples** — the data consumed within the accumulation window must exactly match what the big batch would consume at once, with no samples missed or repeated due to different sharding;
2. **Normalize over all valid tokens** — not a mean over per-micro-batch means;
3. **Exactly one optimizer step** — gradients are only summed or averaged within the accumulation window, and the weight update happens once, at the window's end;
4. **Consistent final-gradient post-processing** — gradient clipping must act on the **final gradient** after accumulation completes, not clip each micro-batch's local gradient separately;
5. **Consistent randomness and batch-dependent state** — dropout and other randomness, plus any cross-sample running statistics, must behave as if computed once on the full big batch;
6. **Match the communication library's reduction semantics** — the global gradient should equal $\nabla\left(\sum_r \mathrm{NLL}_r \big/ \sum_r M_r\right)$, with $r$ ranging over DP ranks and $M_r$ that rank's valid-token count.

Condition 6 is the easiest to get wrong in implementation. Frameworks like DDP/FSDP **average** multi-rank gradients by default rather than summing: if the local loss was already normalized as "NLL sum / this rank's valid tokens", the default averaging implicitly divides by the world size once more. Pick one fix: explicitly switch to sum-reduce and normalize by the global valid-token count, or pre-multiply by $W_{DP}$ when normalizing locally. The local loss and the communication layer must not both divide by $W_{DP}$.

The points above guarantee **mathematical equivalence**: under exact real arithmetic, the accumulated gradient and the one-shot big-batch gradient are the same quantity. They do not guarantee bitwise-identical floating-point values: accumulation order and the all-reduce reduction-tree structure still cause differences in the last few significant digits.

### 4.3　BF16, FP16, and loss scaling

BF16 has the same exponent width as FP32 and a similar dynamic range, so it typically overflows less readily than FP16 in pretraining. FP16's narrower dynamic range makes it more dependent on loss scaling to keep small gradients from underflowing to zero.

## §5 Lifecycle: from warmup to deterministic resume

### 5.1　Warmup/stable/decay/cooldown + data-phase changes

Warmup suppresses early-training instability, stable/plateau occupies the bulk of training, and decay/cooldown steps the learning rate down to a very small value per the schedule. AdamW's update formula and the warmup/cosine/WSD scheduling math are in `optimizer_lr_schedule_tutorial.md`. What the pipeline side cares about is three things: optimizer state must be saved with the checkpoint, whether the schedule advances by step or by token count, and how the two couple when the batch size changes.

Two things often happen simultaneously in the final stretch of training, but they are controlled by different configurations. Decay/cooldown is an LR-side concept; data annealing / late-stage reweighting is an independent data-side concept that raises the weight of high-quality, curated data in the mixture near the end of training (described in public training reports such as Llama 3). A training recipe can perfectly well do LR decay without adjusting data weights, and vice versa.

Another lifecycle axis is the **context-length curriculum**: most of the training budget uses shorter sequences, switching to long context late. This changes $L_{seq}$, meaning the global-batch accounting of §4.1 and the consumed-token counter must both be recomputed consistently at the switch point.

### 5.2　Resuming training, sample order, and numerical reproduction

"After resume, the training trajectory is exactly as if uninterrupted" corresponds to three results of different strength, each requiring different state.

1. **Resumable training**: the model can restore weights and optimizer state from the checkpoint and continue lowering loss, with no requirement that data order or numerics align exactly. Only model weights, optimizer state, the LR schedule, and the consumed-token count are needed.
2. **Identical sample sequence**: after restart, the data order the model sees (which sample at which step) exactly matches the uninterrupted run, without requiring bitwise-identical numerics. Beyond tier 1, this additionally requires RNG states (the generators in Python, the framework, CUDA, and elsewhere), the data-iterator position, the shuffle epoch, the sampler state, and the dataloader worker/prefetch queue state — otherwise multi-worker prefetch order is itself a nondeterminism source; multi-rank training also needs per-rank checkpoint metadata.
3. **Numerically/bitwise identical trajectory** (usually not fully attainable): on top of tier 2, this needs the AMP loss-scaler state, otherwise the dynamic loss scale restarts from a different point and the gradient-scaling path forks within the first few steps; it needs any in-flight gradients inside an unfinished accumulation window, which is avoidable if checkpoints are only taken at accumulation boundaries; it needs deterministic kernels, with nondeterministic cuDNN/cuBLAS operators explicitly disabled; and it needs exactly identical software versions and communication topology, since different GPU counts and different collective implementations reduce floats in different orders, accumulating ULP-level differences. **When the world size changes**, even the most careful resharding semantics usually cannot guarantee a trajectory bitwise-identical to the unchanged-world-size run, because both the data partitioning and the gradient-reduction tree structure change.

Missing the state required by any tier means what you actually achieve is a weaker tier. Checkpoint metadata should explicitly declare which tier it guarantees.

### 5.3　Monitoring: per-domain validation loss

Validation loss should be reported **per domain** — code, math, low-resource languages, long context each counted separately — not just as one aggregate loss. The aggregate is easily dominated by the domain with the highest token weight: with web text at 80% of the mixture, the aggregate loss curve basically reflects only web-text quality. When a lower-weight but strategically important capability (code correctness, mathematical reasoning, some low-resource language, long-context retrieval) is degrading, the aggregate loss curve does not show it, and by the time it is noticed it is often very hard to recover.

### 5.4　Loss spike / NaN triage

The recommended order, arranged "cheapest checks first", is a heuristic rather than a mandatory procedure, and can be adjusted when you have stronger priors:

1. **Bad batch / data shard** — see whether the spike reproduces in isolation on the same batch or shard; this depends on deterministic sharding, the sampler's cursor state, and lineage (§2.8), without which you cannot even locate which batch it was;
2. **LR and resume state** — a misconfigured LR schedule, or a schedule counter accidentally reset after a resume, can both produce anomalous updates whose actual step count differs from expectations;
3. **Gradients and activations** — anomalous gradient norms, exploding activation norms, runaway logit magnitude in some layer, residual instability; the mechanisms behind these diagnostic quantities are derived in `normalization_init_tutorial.md`, and concrete monitoring thresholds still require your own engineering;
4. **Mixed precision** — BF16/FP16 overflow and underflow;
5. **Cross-rank anomalies** — one DP/TP/PP rank quietly diverging (e.g. a GPU with hardware trouble) while the others look normal on the surface; only per-rank loss/gradient statistics reveal this;
6. **Hardware faults** — ECC errors, single-GPU failures, collective-communication jitter.

Keeping reproducible batch IDs, step numbers, and shard attribution is what enables after-the-fact replay and localization. [D07] in §6 is the automated first line of defense: a rolling median/MAD detector that flags when to start the triage, not what the root cause is.

### 5.5　Checkpoint selection

The checkpoint used for downstream deployment is chosen from the per-domain validation sets plus a dedicated eval-set combination. Under a fixed data mixture, training loss mainly reflects fitting progress and does not directly correspond to generalization or to the specific capability profile you actually care about.

### 5.6　Common wrong answers

| Wrong answer | Correct judgment |
| --- | --- |
| $6ND$ is an exact formula, so dividing $C$ by peak FLOPs gives the training time | It is an algorithm-level approximation that ignores the long-context attention term, the vocabulary projection, activation recomputation, and FLOP-counting conventions; estimating wall-clock time additionally needs the GPU count, peak FLOPs, and MFU |
| Chinchilla gives one exact set of exponents, and its divergence from Kaplan is just "more data" | Different estimation methods give about 0.50/0.50, 0.49/0.51, 0.46/0.54; substituting the rounded $\alpha,\beta$ gives an analytic optimum of about 0.452/0.548, and the divergence also involves the fitting method, the training horizon, and whether the LR schedule matches the budget |
| "20 tokens per parameter" is a universal constant | It is an empirical approximation; $D_*/N_*$ also depends on the fitted constants $A,B,k$, and drifts slowly with $C$ when $\alpha\ne\beta$ |
| Training tokens beyond the Chinchilla ratio means overfitting, so the recipe is wrong | Feeding new, non-repeated, related data usually still lowers loss; over-trained is an economic judgment relative to the compute-optimal frontier |
| MinHash measures semantic similarity, and semantic dedup is its unconditional upgrade | MinHash estimates the surface-level Jaccard duplication of shingle sets; semantic dedup addresses another level of duplication and does wrongly delete legitimate text with different wording |
| Splitting first and deduplicating afterwards also blocks near-duplicates from crossing splits | Dedup clusters must form first and whole clusters be assigned to splits, merging by transitive closure documents that were never directly compared |
| The domain weights in the config are the token shares the model actually sees, and the nominal $D$ is the effective data volume | Configured by document but used as token weights, the two are usually unequal (equal in expectation when defined as token-level target shares); repeatedly presented tokens inflate the nominal $D$ without bringing equivalent new information |
| Cross-document attention after packing is a bug, and adding a block-diagonal mask achieves document independence | The continuous stream is a legitimate sequence-semantics choice; document independence additionally requires masking the cross-document label "previous document's EOS → next document's first token" |
| Multiply the GPU count or the TP/PP world size into the global batch; $B_{input}$ and $B_{target}$ are interchangeable | Only the DP world size multiplies independent samples, and under shifted labels each sequence has at most $L_{seq}-1$ valid targets |
| Mathematical equivalence means bitwise-identical numerics, and a run that resumes proves the trajectory was reproduced | Floating-point accumulation order and the reduction-tree structure change the last few digits; a bitwise-identical trajectory additionally needs the scaler state, in-flight gradients, deterministic kernels, and an identical topology, and is usually unattainable when the world size changes |

## §6 From-scratch implementation: corpus → packing → key-invariant sanity checks

The complete runnable script is [`code/pretraining_pipeline.py`](code/pretraining_pipeline.py) — pure Python standard library, zero third-party dependencies, all [D01]–[D07] sanity checks finishing in seconds on CPU. Four demo groups chain into a mini pipeline: text → dedup → pack + loss mask → synthetic budget planner → spike monitor. The script covers four invariants with executable assertions — dedup, packing, Chinchilla fitting, spike monitoring — and has no real forward/backward/optimizer step, and no checkpoint writing or resume. Below are the three core code segments; the full document-packing functional test, the EOS boundary (§3.2), and padding (§3.3) are folded into the same script as short test functions.

**[D01] MinHash / LSH / cluster-split transitive closure** (executable version of §2.3, §2.5):

```python
doc_a = set(range(1, 11))                 # {1,...,10}
doc_b = set(range(1, 9)) | {11, 12}        # {1,...,8,11,12}
j_exact = exact_jaccard(doc_a, doc_b)
assert abs(j_exact - 2 / 3) < 1e-12        # |A∩B|=8, |A∪B|=12 -> 8/12

sig_a = minhash_signature(doc_a, num_perm=1024, seed=12345)
sig_b = minhash_signature(doc_b, num_perm=1024, seed=12345)
j_est = minhash_estimate(sig_a, sig_b)     # MinHash ESTIMATE, not exact
tol = 6 * math.sqrt((2 / 3) * (1 / 3) / 1024)   # principled sampling tolerance
assert abs(j_est - 2 / 3) < tol             # close, not asserted exactly equal

p_high = lsh_candidate_probability(2 / 3, b=128, r=8)   # P = 1-(1-s^r)^b
p_low = lsh_candidate_probability(0.05, b=128, r=8)
assert p_high > 0.9 and p_low < 0.1         # high-similarity candidates recalled far more often

uf = UnionFind(4)               # docs: 0=A, 1=B, 2=C, 3=D(unrelated)
uf.union(0, 1); uf.union(1, 2)  # A~B found, B~C found (A,C never directly compared)
assert uf.find(0) == uf.find(2) # transitive closure still merges them into one cluster
# -> every doc in this cluster MUST receive the same train/val/test split label
```

**[D02]/[D03] document packing: shifted labels + block-diagonal mask + functional test** (executable version of §3.4):

```python
tokens = [11, 12, 13, 21, 22]
doc_ids = [0, 0, 0, 1, 1]

labels_indep = shifted_labels_independent_docs(tokens, doc_ids)
assert labels_indep == [12, 13, IGNORE, 22, IGNORE]      # doc boundary at index 2 masked
labels_cont = shifted_labels_continuous(tokens)
assert labels_cont == [12, 13, 21, 22, IGNORE]           # continuous stream: no boundary masking

m_block = block_diagonal_causal_mask(doc_ids)
m_cont = continuous_causal_mask(len(tokens))
assert m_block[3][2] is False and m_block[4][0] is False and m_block[4][3] is True
assert m_cont[3][2] is True          # SAME (q,k); continuous mode allows it -- a config choice

values = [100, 100, 100, 0, 0]       # doc0=100s, doc1=0s
assert masked_uniform_attention_output(m_cont[3], values) == 75.0    # (100+100+100+0)/4
assert masked_uniform_attention_output(m_block[3], values) == 0.0    # only key=3 allowed

values2 = [1000, 1000, 1000, 0, 0]   # bump doc0's values 10x
assert masked_uniform_attention_output(m_block[3], values2) == 0.0   # doc1's output: UNCHANGED
assert masked_uniform_attention_output(m_cont[3], values2) == 750.0  # continuous: DOES change
```

**[D06] synthetic Chinchilla curve fitting** (executable version of §1.3; **this is a fitting unit test, not a validation of the empirical law**):

```python
alpha, beta, k_flops = 0.34, 0.28, 6            # exponents straight from §1.3's CORRECTIONS
e_true, a_true, b_true = 1.5, 400.0, 400.0      # SYNTHETIC ground truth, not the paper's fitted values
points = [(n, d, loss_surface(n, d, e_true, a_true, b_true, alpha, beta))
          for n in logspace(1e7, 1e12, 6) for d in logspace(1e8, 1e13, 6)]

e_hat, a_hat, b_hat = fit_e_a_b(points, alpha, beta)    # linear least squares (alpha,beta fixed)
assert abs(e_hat - e_true) < 1e-6 and abs(a_hat - a_true) < 1e-3 and abs(b_hat - b_true) < 1e-3
assert e_hat < min(loss for _, _, loss in points)       # irreducible loss floor
assert a_hat > 0 and b_hat > 0

c_budget = 6e21
n_star, d_star = analytic_optimum(c_budget, k_flops, alpha, beta, a_true, b_true)
n_star_100x, d_star_100x = analytic_optimum(100 * c_budget, k_flops, alpha, beta, a_true, b_true)
assert abs(n_star_100x / n_star - 100 ** (beta / (alpha + beta))) < 1e-6   # ~8.0x
assert abs(d_star_100x / d_star - 100 ** (alpha / (alpha + beta))) < 1e-6  # ~12.5x
```

[D04] (EOS boundary, reusing the same doc-id rule as [D02]), [D05] (padding's dual masks + no-empty-row check), and [D07] (rolling median/MAD spike detection + unconditional NaN/Inf alerts + spike/sustained-shift event classification) are folded into `main()` as independent assertion groups.

## §7 26 High-Frequency Interview Questions

Three difficulty tiers; each question gives answer key points and common pitfalls.

### L1 Must-Know

<details>

<summary>Q1. What are $N$, $D$, $C$? Where does the "6" in $C\approx 6ND$ come from, and why is it only an approximation?</summary>

- $N$ = compute-bearing parameter count participating in the main matrix multiplies per token (active for MoE; the embedding lookup is usually excluded, see §1.1); $D$ = tokens presented to the model during training; $C$ = approximate training FLOPs
- One multiply-add counts as 2 FLOPs (one multiply, one add): the forward pass runs each token through the main matrix multiplies, about $2N$/token, so $D$ tokens give $2ND$; the backward pass computes both activation and weight gradients, roughly twice the forward, i.e. $4ND$; forward + backward $\approx 6ND$ — itself a counting convention (some tools count a multiply-add as 1 FLOP), not a physical constant
- It ignores the long-context attention $L_{seq}^2$ term (total cost approximately scales as $D\times L_{seq}$ at fixed $D$), vocabulary projection, norm/softmax, activation recomputation, and FLOP-counting convention differences; it is not an exact formula and cannot be converted directly to wall-clock time — a rough GPU-days estimate additionally needs GPU count, peak FLOPs, and MFU: $t\approx C/(n_{GPU}\times\text{peak FLOPs}\times MFU)$; communication overhead, recomputation, and data stalls make this an estimate, not a precise prediction (§1.1)

Treating $6ND$ as exact, counting even embedding parameters into $N$ indiscriminately, or dividing $C$ by peak FLOPs to get training time (ignoring MFU) are the three most common ways to lose points on this question.

</details>

<details>

<summary>Q2. How are a causal LM's inputs/labels offset? Why is a randomly initialized model's loss usually close to $\ln V$?</summary>

- Under teacher forcing, the label at position $i$ is $x_{i+1}$ (the sequence shifted right by one); the last position has no next token and is marked `IGNORE` ($-100$ here, §3.1)
- In packing/document-boundary settings, the rule "a label is valid iff the current position and the next position belong to the same document" uniformly handles both IGNORE sources — sequence end and document boundary (§3.2)
- A randomly initialized model assigns approximately uniform probability to the next token, so the NLL at each valid position is approximately $\ln V$ ($V$ the vocabulary size) — the loss observed at the very start of training should be near this value; a clear deviation (far higher, or an anomalously low value) usually signals a problem with initialization, label alignment, or the data itself

Believing "IGNORE only appears at the end of the batch", missing that document boundaries also produce IGNORE, or being unable to explain where the $\ln V$ initial-loss scale comes from are the common incomplete answers here.

</details>

<details>

<summary>Q3. How do BF16 and FP16 differ in dynamic range? When is loss scaling needed?</summary>

- BF16 has the same exponent width as FP32 and a similar dynamic range, so it typically overflows less readily than FP16 in pretraining; the cost is fewer mantissa bits, so per-value relative precision is lower than FP16
- FP16's exponent width is narrower than BF16/FP32, with a smaller dynamic range — small gradients easily underflow to zero in FP16, and large values overflow to Inf more easily
- **Loss scaling** (multiply the loss by a large factor before backprop, then divide it back out of the gradients) is the standard FP16 technique for "lifting" small gradients into FP16's representable range; BF16, with a dynamic range already close to FP32, usually does not need it (§4.3)
- This is also part of what the "mixed precision" layer checks during the loss spike/NaN triage of §5.4 — and the scaler state itself is part of the state that the deterministic resume of §5.2 must consider

Flatly equating "mixed-precision training" with "both BF16 and FP16 need loss scaling", without articulating the different engineering needs their dynamic-range difference creates, is the common wrong answer here.

</details>

<details>

<summary>Q4. Training-token count exceeds the Chinchilla-recommended ratio — is that overfitting?</summary>

- Not necessarily. "Exceeding the Chinchilla ratio" alone cannot establish overfitting — if the additional data is **new, non-repeated, same- or related-distribution**, a fixed-size model usually still lowers loss, failing the "training loss falls, validation loss rises" definition of overfitting
- What actually happens: as $D$ keeps growing, the final compute $C=kND$ itself grows; at that larger $C$, the configuration has a smaller $N$ and larger $D$ relative to its own compute-optimal frontier point, and is called "over-trained" — an economic judgment relative to the compute-optimal frontier, not a training error (§1.4)
- Deliberately over-training small models cuts deployment-side memory/latency/inference cost; training-compute-optimal $\ne$ lifecycle-cost-optimal

Conflating "over-trained" (a budget-relative economic concept) with "overfitting" (a generalization failure), or asserting "definitely not overfitting" while ignoring the "new/non-repeated data" premise, is the number-one way to fail this question.

</details>

<details>

<summary>Q5. What is document packing? Why doesn't a plain causal mask automatically isolate documents?</summary>

- Packing: concatenate multiple documents into one training sequence to raise utilization
- The causal mask only guarantees "cannot see the future" and does not check whether tokens belong to the same document — a later document can read an earlier document's history tokens by default; this is the continuous EOS-separated-stream semantics **many systems adopt deliberately**, not a bug
- For "document independence", you need the block-diagonal mask $M[q,k]=(k\le q)\wedge(\text{docid}[q]=\text{docid}[k])$ (§3.4)

Declaring "can attend across documents" an implementation error outright, without being able to say it is a legitimate semantic choice.

</details>

<details>

<summary>Q6. What is the difference between an attention mask and a loss mask? When are both mandatory, and when might one be redundant?</summary>

- The attention mask (key padding / block-diagonal) controls "who can see whom" and acts in the attention computation; the loss mask controls "which positions' predictions receive gradient" and acts in the loss computation — the two are independent, and blocking attention does not automatically mask the corresponding labels
- **The block-diagonal setting requires both**: even if document B cannot read document A, document A's last position may still be asked to predict document B's first token, unless excluded separately by the boundary loss mask (§3.2, §3.4)
- **The padding setting depends on layout**: the loss mask is mandatory; but under strict causal + right padding, the causal mask already blocks future padding keys, so the key mask may be redundant for real queries — under left padding or other layouts it cannot be omitted (§3.3)

Believing "both masks are equally mandatory at all times", or conversely "the key mask can be omitted unconditionally", both lose points; the correct answer analyzes cases by concrete layout.

</details>

<details>

<summary>Q7. How is global batch size / tokens-per-update actually computed? Which accounting conventions must be kept apart?</summary>

- Tokens fed to the model: $B_{input}=B_{micro,seq}\times L_{seq}\times G_{acc}\times W_{DP}$ ($B_{micro,seq}$ is sequences per DP replica, not "per GPU"); only the DP world size directly multiplies independent samples — TP/PP partition the computation of the same samples and must not be double-counted
- Valid targets actually in the loss: $B_{target}=\sum_i m_i$ ($m_i$ each sequence's valid-target mask count) — even with zero padding, since each sequence's last label is `IGNORE`, $B_{target}$ is $L_{seq}-1$ per sequence, not $L_{seq}$; padding/packing boundary masking shrinks it further below $B_{input}$
- The three accounting conventions must not be mixed: advancing the consumed-token counter/aligning the scheduler usually uses $B_{input}$; the loss-normalization denominator and the $D$ in scaling laws should track $B_{target}$ (§4.1)

Multiplying TP/PP world sizes into the batch-size formula, or swapping $B_{input}$ and $B_{target}$ as if they were the same number, are the two most common calculation errors here.

</details>

<details>

<summary>Q8. Why can't PPL be compared directly across tokenizers?</summary>

- PPL's denominator is the token count, which itself depends on the tokenizer — at the same total NLL, a tokenizer that segments more finely yields a lower loss/token
- BPB swaps the denominator for the byte count of the same text, making it more robust to tokenizers, but still requires both sides to evaluate the same normalized byte stream
- Exact formulas: $\mathrm{PPL}=\exp(\sum \mathrm{NLL}_i/\#\text{valid tokens})$, $\mathrm{BPB}=\sum \mathrm{NLL}_i/(\#\text{bytes}\cdot\ln 2)$ (§3.6; full derivation in tokenization_tutorial.md §1.3)

Knowing only "lower PPL is better" without being able to state its tokenizer-dependence precondition.

</details>

### L2 Advanced

<details>

<summary>Q9. How do the Kaplan (2020) and Chinchilla (2022) scaling-law conclusions differ?</summary>

- Kaplan: $N_{opt}\propto C^{0.73}, D_{opt}\propto C^{0.27}$ — growing compute budgets go mostly to parameters
- Chinchilla: no single "exact exponents" — different methods give about 0.50/0.50, 0.49/0.51, 0.46/0.54; the parametric model $L=E+A/N^\alpha+B/D^\beta$ fits $\alpha\approx0.34,\beta\approx0.28$, with analytic optimum $N_*\propto C^{0.452}, D_*\propto C^{0.548}$
- The divergence is not just "more data": training-horizon design, whether the LR schedule matches the budget, and fitting methods all differ (§1.3)

Answering only "Chinchilla says more data", without the exponents each of the three estimation methods gives, and without the key technical divergence of training horizon/LR schedule.

</details>

<details>

<summary>Q10. Derive the analytic $N_*,D_*$ solution of Chinchilla's constrained optimization.</summary>

- Objective: $\min L(N,D)=E+A/N^\alpha+B/D^\beta$ s.t. $C=kND$
- Substitute the constraint to eliminate one variable, then use Lagrange multipliers or differentiate directly in $\log N$ to find the stationary point, yielding $N_*=(\alpha A/\beta B)^{1/(\alpha+\beta)}(C/k)^{\beta/(\alpha+\beta)}$, $D_*=(\beta B/\alpha A)^{1/(\alpha+\beta)}(C/k)^{\alpha/(\alpha+\beta)}$
- Substituting $\alpha=0.34,\beta=0.28$ gives exponents 0.452/0.548; at 100x compute, $N_*$ grows about $100^{0.452}\approx 8.0$x and $D_*$ about $100^{0.548}\approx 12.5$x — this derivation is mathematically sound with $A,B,\alpha,\beta,k$ fixed, but must not be extrapolated into an empirical claim (§1.3, [D06])

The exponents are crossed: $N_*$ takes $\beta/(\alpha+\beta)$ and $D_*$ takes $\alpha/(\alpha+\beta)$ — the most common way to lose points is memorizing them backwards.

</details>

<details>

<summary>Q11. Given a compute budget, how do you decide the parameter count and training-token count?</summary>

- First separate the objectives: training-compute-optimal (apply $N_*,D_*$ directly) or inference-aware optimal (first cap the small $N$ the deployment budget allows; at fixed $C$, $D$ grows accordingly — deliberate over-training)
- MoE uses active parameters for the compute budget and total parameters for memory/checkpoints (§1.2)
- Long context introduces the $L_{seq}^2$ term $6ND$ ignores; budget for it separately (§1.5)

Only reciting the $N_*,D_*$ formulas without articulating the "training-optimal ≠ inference-aware-optimal" fork.

</details>

<details>

<summary>Q12. How should the three knobs — attention mask, loss mask, position id — be designed independently in packing?</summary>

- This is not "just two pre-packaged recipes, continuous stream vs independent documents" — **attention boundary, target boundary, and position id are three independently combinable knobs**, not bound into one choice:
  - **Attention boundary**: plain causal ($k\le q$, allowing cross-document history reads) or block-diagonal ($k\le q \wedge \text{docid}[q]=\text{docid}[k]$, documents mutually invisible);
  - **Target boundary**: even with plain causal attention (documents can "see" each other), the document-boundary rule can still separately mask the cross-document label "previous document's EOS → next document's first token" (§3.2) — attention allowing cross-document history reads does not mean that target must be kept;
  - **Position id**: whether to reset is a choice determined by the position-encoding scheme, not a mathematical necessity — standard RoPE is insensitive, in ideal math, to a whole-document position shift; absolute position encodings should usually reset; trade-offs in §3.4
- The two most common combinations are "continuous stream = causal attention + no position-id reset + no extra target masking" and "independent documents = block-diagonal attention + position-id reset + cross-document target masking", but intermediate configurations like "causal attention yet still masking cross-document targets" exist in practice — the two extremes must not be treated as the only two options
- Both keep the supervision "document's last content token → EOS"; they differ only in how the "cross-document" part is handled (§3.2, §3.4)

Answering this question by "reciting two fixed recipes", without saying the three knobs combine independently, or that resetting position ids under RoPE is not mathematically required.

</details>

<details>

<summary>Q13. What conditions must gradient accumulation satisfy to be equivalent to one big batch? Be clear about which tier of "equivalence"</summary>

- Same samples (data consumed in the accumulation window must exactly match what the big batch would consume at once)
- Normalize over all valid tokens, not a mean over per-micro-batch means
- Exactly one optimizer step; gradient clipping acts on the final gradient after accumulation completes, not per micro-batch
- This only guarantees **mathematical equivalence** (under exact real arithmetic, the accumulated gradient and the one-shot big-batch gradient are the same formula). Going further requires: DDP **averages** gradients by default rather than summing, so you must explicitly compensate with the global valid-token denominator, otherwise you effectively divide by the world size once more (Q16); dropout/RNG state and any batch-dependent running statistics must traverse exactly the same random-number sequence in both settings before comparison is meaningful; the AMP loss-scaler state and the scheduler update count (each "big batch" should trigger exactly one scheduler.step, not one per micro-batch) must also be aligned
- Even with all of the above aligned, differing **floating-point reduction orders** (the order of summation across micro-batches, the tree structure of cross-DP-rank all-reduce) still change the last few significant digits — this step reaches only "mathematical equivalence", never unconditional **bitwise numerical equivalence** (§4.2; the three-tier framework of §5.2 applies here too)

Missing "clipping must happen once, after accumulation", substituting a mean of means for total-over-count, or mistaking "mathematical equivalence" for "bitwise-identical numerics" are the three most common pitfalls here.

</details>

<details>

<summary>Q14. How do you design a near-dedup (MinHash+LSH) pipeline? What is LSH's candidate-probability formula?</summary>

- Document → shingle set → MinHash signature (minimum over multiple hash functions) → estimate Jaccard similarity
- LSH splits the signature into $b$ bands of $r$ rows each; under the idealized assumption that MinHash rows/hash functions are approximately independent, the candidate probability is $P=1-(1-s^r)^b$ — candidate recall only
- Candidate pairs usually still need exact-Jaccard verification, and MinHash estimates set overlap, not semantic similarity (§2.3, [D01])

Treating LSH candidates directly as "confirmed near-duplicates", skipping the exact-Jaccard verification step.

</details>

<details>

<summary>Q15. Why must dedup come before splitting?</summary>

- If you split first and dedup later, near-duplicate content (mirror sites, rewritten reposts) can land one copy in train and one in val
- The model has "memorized" a near-copy of validation samples during training; validation loss is spuriously low, misleading early stopping and model selection
- Correct approach: run near dedup first to get connected components (dedup clusters), then assign whole clusters atomically to a split; even if A and C were never directly compared, transitive connection through B still forces them into the same split (§2.5, [D01])

Knowing only "dedup is needed", without saying the ordering of dedup and split, and without the hidden transitive-closure requirement.

</details>

<details>

<summary>Q16. What is the most common trap in loss normalization?</summary>

- Correct formula: $\sum \mathrm{NLL}_i / \#\text{valid targets}$, normalized once over the global batch
- Common mistake: compute a mean per micro-batch, then average those means — introducing weighting bias when micro-batches have different valid-token counts (padding/packing/variable length)
- The multi-rank setting has a variant mistake: DDP **averages** gradients by default, so if the local loss was already normalized as "NLL sum / this rank's valid tokens", letting all-reduce average again over ranks effectively scales the denominator by the world size once more; the fix is either to compute only "sums" locally and divide by the **global** valid-token total after all-reduce, or to explicitly switch to sum-reduce and control the normalization timing yourself (§3.5, §4.2)
- A correct implementation accumulates "NLL sum" and "valid token count" separately across micro-batches/ranks, dividing once at the end

Writing a "mean of means" implementation, or letting all-reduce's default averaging semantics fight the local normalization logic, silently distorts the loss in variable-length batch/multi-rank settings without raising any error — the most easily overlooked hazard of this question.

</details>

<details>

<summary>Q17. What are the EOS/BOS supervision rules at packing boundaries?</summary>

- Keep: the loss for "document's last content token → EOS" (teaching the model when to stop)
- Mask (per configuration): the loss for "previous document's EOS → next document's first token/BOS" (that label is merely a concatenation artifact)
- Both rules are produced automatically by the single rule "a label is valid iff the current and next positions share a document" — no separate special case for EOS is needed (§3.2, [D04])

Assuming EOS needs extra special-case logic, not realizing the same doc-id boundary rule handles it uniformly.

</details>

<details>

<summary>Q18. When PAD and EOS share the same token ID, what must mask construction watch for? Separate the two causal chains</summary>

- PAD=EOS is not wrong per se; many tokenizers have no dedicated PAD token
- **Building the loss mask via `target==pad_id`**: this also masks the real supervision "content-end token → EOS" as "target is padding", directly deleting the signal from which the model learns "where to stop" — the most severe accident chain
- **Building the key mask via `key==pad_id`**: a different consequence — it wrongly hides real EOS positions as padding keys (real queries cannot read that EOS's contextual representation), but deletes no loss automatically, since the loss mask is computed separately by target position; this is a "hide context" error, not a "delete loss" error
- The uniform correct approach: build masks from each sequence's true valid length `valid_len` (or an explicit padding flag), never from token-ID equality (§3.3, [D05])

Only reciting the absolutist conclusion "PAD=EOS is wrong", without being able to say which distinct kind of error ID-based loss masks and ID-based key masks each cause.

</details>

### L3 Expert

<details>

<summary>Q19. Design a complete data pipeline from web crawling to training shards.</summary>

- **The eleven steps of §2.1 are a reference flow; the main hard constraint is a single one: dedup clusters must form first, then the cluster-level split** (which depends on the already-computed connected components, §2.5); privacy/safety processing and quality filtering are often moved earlier in practice, or run once at several stages, and need not be locked into their fixed table positions
- After the raw-document stage comes a frequently omitted **tokenized stage**: tokenizer materialization (actually running the tokenizer over raw-document shards to produce the token stream), token-level mixture sampling (correcting the gap between document-level quotas and actual token shares), long-document chunk/truncate, and the data-stream/sampler layer that decides how those tokenized shards are read at training time (shard shuffle, domain-sampling RNG, mutually exclusive DP-rank sharding, worker/prefetch state, cursor position) — see §2.7–§2.10
- Every step must keep lineage: source/snapshot version/filter version/dedup cluster ID/split assignment/chunk attribution/final shard position (§2.1, §2.9)
- Benchmark decontamination targets eval sets and is a different matter from the train-val split (§2.4, §2.6)

Memorizing only the eleven steps' names and order as the one true answer, without stating that "dedup before split" is the only hard constraint, and omitting the tokenizer/chunking/sampler stages of the tokenized phase.

</details>

<details>

<summary>Q20. What state must a checkpoint save for "deterministic resume"? Which determinism, exactly?</summary>

- First decide which tier you want (§5.2): **resumable training** needs only model weights + optimizer state (Adam's $m,v$) + LR schedule/consumed-token count; **identical sample sequence** additionally needs RNG states, data-iterator position, shuffle epoch, sampler state, and dataloader worker/prefetch state, plus per-rank checkpoint metadata for multi-rank training
- The strongest tier, **numerically/bitwise identical trajectory**, additionally needs the AMP loss-scaler state, in-flight pending gradients (from an unfinished accumulation window), deterministic kernels, and exactly identical software versions/communication topology — and even then, changing the world size usually still cannot guarantee a bitwise-identical trajectory; how a checkpoint reshards when the world size changes, and whether the resharded run can match the original trajectory, has no general answer, and `distributed_training_tutorial.md` does not cover it either
- Missing the data-iterator position or RNG state drops you straight out of the "identical sample sequence" tier; satisfying the first two tiers but missing the AMP scaler/pending gradients/deterministic kernels gives you "restartable" and "order-consistent" but not full numerical reproduction

Treating "deterministic resume" as a single black-or-white promise, without mapping the three strength tiers to their necessary states.

</details>

<details>

<summary>Q21. Pretraining suddenly hits a loss spike or NaN — how do you triage in layers?</summary>

- This is a **recommended heuristic triage order** (arranged "cheapest checks first"), not a mandatory procedure — adjust when you have stronger priors: bad batch/data shard (can it reproduce on the same batch? depends on deterministic sharding + lineage) → LR and resume state → gradients and activations (diagnostic quantities deferred to normalization_init_tutorial.md) → mixed-precision overflow/underflow → cross-rank anomalies (needs per-rank statistics) → hardware faults
- Keep reproducible batch IDs/step numbers for after-the-fact replay and localization
- Automated first line of defense: a rolling median/MAD detector responsible for flagging "when to start the triage", not for judging root cause (§5.4, [D07])

Jumping straight to "it must be the learning rate being too high", skipping "can it reproduce on the same batch" — the cheapest and most deserving first step.

</details>

<details>

<summary>Q22. Why report validation loss per domain?</summary>

- The aggregate loss is easily dominated by the domain with the highest token weight (e.g. web text with a large share), masking degradation of low-weight but important capabilities like code/math/low-resource languages/long context
- Per-domain monitoring is what catches problems while there is still time to adjust the mixture/checkpoint selection
- Checkpoint selection should also rely on per-domain validation sets/eval-set combinations, not just training loss (§5.3, §5.5)

Concluding "the model is training fine" from a single aggregate loss curve, blind to the diluted-away local degradation.

</details>

<details>

<summary>Q23. When do mixture-sampling "domain weights" equal the token weights the model actually sees, and when not?</summary>

- **Two cases, no unconditional claim**: when domain weights are defined as token-level target shares (the config literally says "this domain should contribute this fraction of tokens"), the two can be equal in expectation; they diverge only when configured as document counts/document weights yet used directly as token weights — because domains differ in average document length (§2.7)
- **Keep the three sampling units apart**: sample by document (no domain distinction; draw whole documents by pool probabilities), sample domain first then document (a two-stage process — only here does the first-level "domain weight" have a clear meaning), sample by token (cut blocks directly from the concatenated token stream, decoupled from document boundaries)
- **Response**: measure and report the "actual token-level share" separately after sampling, not just the configured values; this also connects to the accounting for $D$ in scaling laws — resampling/residual near-duplicates push the "effective data volume" below the nominal $D$

Unconditionally asserting "domain weights definitely do not equal token weights", or blurring "sample by document" and "sample domain first then document" into one concept, are the two easiest pitfalls here.

</details>

<details>

<summary>Q24. What are the "three leakages" (train-val leakage / benchmark contamination / packed cross-document context), and how do you respond?</summary>

| Type | Core judgment | Method boundary |
| --- | --- | --- |
| train-val leakage | split-before-dedup lets near-duplicates cross splits → cluster-level split **mitigates** (§2.5) | MinHash/LSH+n-gram misses heavy rewrites, cross-language translation, and "answer variants" semantically identical but superficially different — it only sharply reduces surface-duplication leakage probability, it does not "solve" |
| benchmark contamination | training corpus contains eval-set content → n-gram overlap decontamination **mitigates** (§2.4) | missed detection is worse (benchmarks are often manually rewritten to evade detection), and templated/boilerplate text (common code snippets, legal templates) can trigger false positives |
| packed cross-document history context | **usually not data leakage but the "cross-sample context coupling" sequence-semantics choice** (§2.6, §3.4) | whether to sever it with a block-diagonal mask depends on whether the "document independence" modeling assumption is required |

Claiming the first two detections "solve it completely", or treating the third as leakage that must always be severed, are the point-losers here.

</details>

<details>

<summary>Q25. How are an MoE model's compute budget and memory budget each calculated?</summary>

- **Compute budget uses active parameters**: the $N$ in $6ND$ takes the expert parameters that actually participate in that token's forward/backward computation
- **$6N_{active}D$ is itself still an approximation**: it excludes MoE-specific all-to-all communication overhead, inter-expert load imbalance, and the extra compute of capacity padding (tokens exceeding routing capacity must be dropped or handled specially); details deferred to `moe_tutorial.md`
- **Aggregate checkpoint/model-state size grows with total parameters**: all expert weights must be storable in full
- **Per-GPU memory cannot simply be counted by total**: it also depends on the sharding scheme — EP (expert parallelism)/FSDP etc. — parameter dtype, optimizer state (Adam's $m,v$ counted at post-sharding scale), gradients, and activation memory; at the same total parameter count, different sharding strategies give very different per-GPU memory; if the total loss includes a router aux loss, its magnitude is usually far below the main loss (§1.2, §3.5)

Mixing up active and total parameters — estimating $6ND$ training compute with total parameters significantly overestimates; conversely, estimating per-GPU memory by simply dividing total parameters by GPU count, ignoring sharding and optimizer/activation footprints, is the other common oversimplification here.

</details>

<details>

<summary>Q26. Design a data-lineage scheme supporting "audit the origin of a training sample six months later".</summary>

- **Minimum traceability (the passing floor, insufficient for strict audits)**: source dataset/URL → snapshot/crawl version → extraction and filtering pipeline version (with thresholds) → dedup cluster ID → split assignment → final shard file + offset
- **Strong audit (recorded on top of minimum traceability)**: sample/content hash (confirming "the content now" matches "as written then"), raw snapshot hash (not just a version number), tokenizer/normalizer hashes (the concrete artifact, not a version string), code/container versions, mixture-sampling random seeds, shard checksums, exact document→segment/token mapping (especially with chunking, §2.9), and license plus takedown-request status
- Strong-audit records serve four scenarios: benchmark-contamination investigation, loss-spike reproduction and localization, tracing suspicious model outputs to sources, and compliance takedown-request response (§2.1, §2.8, §2.9, §5.4)

Storing just a "source URL + version number" and calling it enough — unable to name the content hash, tokenizer hash, mixture seed, and document→token mapping that equally determine "what this sample looks like now", and unable to name the license/takedown compliance dimension.

</details>

## §A Appendix: Sanity Checks

The from-scratch implementation should satisfy the following key invariants; script at [`code/pretraining_pipeline.py`](code/pretraining_pipeline.py):

1. **[D01] dedup**: exact Jaccard $J(A,B)=8/12=2/3$ ($A=\{1..10\}$, $B=\{1..8,11,12\}$); the fixed-seed, 1024-permutation MinHash estimate falls within a tolerance set by sampling theory (exact equality is not asserted); LSH ($b=128,r=8$) candidate probability is far higher for the high-similarity pair ($s=2/3$) than for the low-similarity pair ($s=0.05$); union-find verifies that the transitive closure of A~B, B~C puts A and C in the same dedup cluster, which must receive the same split.
2. **[D02] packing shifted labels**: with `tokens=[11,12,13,21,22]`, `doc_ids=[0,0,0,1,1]`, independent-document semantics give labels `[12,13,IGNORE,22,IGNORE]`; continuous-stream semantics give `[12,13,21,22,IGNORE]` (the same token stream, two legitimate configurations, different labels).
3. **[D03] block-diagonal mask + functional test**: `M[3,2]=False`, `M[4,0]=False`, `M[4,3]=True` (block mode); the same coordinate in continuous-stream mode has `M[3,2]=True`. With $QK^\top=0$, when doc0's values change from `[1,2,3]` to `[1000,2000,3000]` while doc1's values stay `[10,20]`, block mode's document-1 outputs at q3 and q4 stay unchanged at `10.0` and `15.0`, while continuous-stream mode's q3 output goes from `4.0` to `1502.5`.
4. **[D04] EOS boundary**: on `[11,12,13,EOS,21,22,EOS]` / `doc_ids=[0,0,0,0,1,1,1]`, the label content-end token (13)→EOS is kept (=EOS) and the label EOS→next document's first token (21) is masked (IGNORE) — both produced automatically by the same doc-id boundary rule.
5. **[D05] padding**: with `valid_len=3` the loss mask is `[1,1,0,0,0]`; real queries cannot attend to padding keys; every attention row is non-empty (no all-`-inf` row causing softmax NaN).
6. **[D06] Chinchilla fit**: from a log-spaced grid ($N\in[10^7,10^{12}]$, $D\in[10^8,10^{13}]$, 6 points each) the noiseless recovery of $E,A,B$ is within $<10^{-3}$ of the synthetic ground truth; the recovered $E$ is below all finite observed losses; the analytic $N_*,D_*$ differ from the fine-grid-search minimum by at most one grid step; at 100x compute, $N_*$ grows about $100^{0.452}\approx 8.0$x and $D_*$ about $100^{0.548}\approx 12.5$x.
7. **[D07] spike monitoring**: in the sequence `[2.00,2.02,1.98,2.01,1.99,2.00,2.03,1.97,3.00]` only the last point is flagged; a smoothly declining sequence triggers no false alarms throughout; same-magnitude single-point jumps are labeled as two different event types — `spike` (subsequent fall-back) vs `sustained_shift` (subsequent stay-high); NaN/Inf is unconditionally and immediately flagged `non_finite`.

**Actual output** of running `python3 code/pretraining_pipeline.py`:

```text
[D01] dedup: exact J(A,B)=0.6667, MinHash est=0.6250 (tol 0.0884), LSH P_high=0.9939 vs P_low=0.0000; real banding candidates=[(0, 1), (1, 2)], confirmed edges=[(0, 1), (1, 2)], cluster split={0: 'test', 1: 'test', 2: 'test', 3: 'train'}  PASS
[D02] packing labels: independent-doc=[12, 13, -100, 22, -100], continuous=[12, 13, 21, 22, -100]  PASS
[D03] block mask: doc1 out unaffected by doc0 values changing (q3: 10.0->10.0, q4: 15.0->15.0); doc0 out unaffected by doc1 values changing (q0/q1/q2: 1.0/1.5/2.0); continuous mode DOES change (q3: 4.0->1502.5)  PASS
[D04] EOS boundary: labels=[12, 13, 999, -100, 22, 999, -100] (content-end->EOS kept, EOS->next-doc masked)  PASS
[D05] padding: loss_mask=[1, 1, 0, 0, 0], no attention row is empty, PAD==EOS adversarial loss_mask=[1, 1, 0, 0, 0] (real EOS target NOT masked)  PASS
[D06] Chinchilla fit: E=1.5000 A=400.00 B=400.00 (true 1.5/400.0/400.0); N*(from fitted A_hat/B_hat)=4.167e+09 matches grid (4.112e+09) and true-coefficient optimum (4.167e+09); 100x compute -> N* x8.003, D* x12.496  PASS
[D07] spike monitor: main seq flags=[8], decline seq flags=0, spike->spike vs shift->sustained_shift, NaN/Inf-> non_finite  PASS

all pretraining pipeline sanity checks passed
```

## 📚 References

- **Scaling Laws (Kaplan)** — Kaplan et al. (OpenAI), *Scaling Laws for Neural Language Models*, arXiv 2001.08361 (2020).
- **Chinchilla (Hoffmann)** — Hoffmann et al. (DeepMind), *Training Compute-Optimal Large Language Models*, arXiv 2203.15556 (2022).
- **GPT-3** — Brown et al. (OpenAI), *Language Models are Few-Shot Learners*, arXiv 2005.14165 (2020), NeurIPS 2020.
- **Gopher** — Rae et al. (DeepMind), *Scaling Language Models: Methods, Analysis & Insights from Training Gopher*, arXiv 2112.11446 (2021).
- **OPT (training logs and instability records)** — Zhang et al. (Meta), *OPT: Open Pre-trained Transformer Language Models*, arXiv 2205.01068 (2022).
- **PaLM (incl. loss-spike / checkpoint-restart practice)** — Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311 (2022).
- **Mixed-precision training** — Micikevicius et al. (NVIDIA/Baidu), *Mixed Precision Training*, arXiv 1710.03740 (2017), ICLR 2018.
- **The Pile (data-pipeline design reference)** — Gao et al. (EleutherAI), *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, arXiv 2101.00027 (2020).
- **C4 corpus audit (extraction-quality issues)** — Dodge et al., *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*, arXiv 2104.08758 (2021).
- **RefinedWeb (web-corpus extraction/filtering pipeline)** — Penedo et al., *The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*, arXiv 2306.01116 (2023).
- **FineWeb (large-scale web corpus + ablations)** — Penedo et al. (Hugging Face), *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, arXiv 2406.17557 (2024).
- **Training-data dedup (near-duplicates' downstream effects)** — Lee et al. (Google), *Deduplicating Training Data Makes Language Models Better*, arXiv 2107.06499 (2021), ACL 2022.
- **MinHash / Jaccard estimation original source** — Broder, *On the Resemblance and Containment of Documents*, Proceedings of Compression and Complexity of Sequences (1997) (no arXiv id).
- **Data mixtures / domain reweighting** — Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*, arXiv 2305.10429 (2023), NeurIPS 2023.
- **Data-constrained scaling (repeated data's effect on effective D)** — Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv 2305.16264 (2023), NeurIPS 2023.
- **Llama 2 (incl. pretraining data/recipe description)** — Touvron et al. (Meta), *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288 (2023).
- **Llama 3 (incl. data-pipeline/training-recipe description)** — Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783 (2024).
- **DeepSeek-V3 (active/total parameters and MoE training practice)** — DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437 (2024).
- **C4 / T5 (classic reference for tokenizer and corpus preprocessing)** — Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, arXiv 1910.10683 (2019), JMLR 2020.
