## §0 "Input-State-Output" Overview + TL;DR Cheat Sheet

> **Scope statement**: this tutorial only covers the **decoder-only, causal language modeling (next-token prediction)** pretraining objective — it does not cover encoder-decoder, masked language modeling (BERT-style token/span-masking reconstruction), or any of the span-corruption/denoising objectives (e.g., T5's span corruption). The shifted labels, mask construction, and loss normalization for those objectives differ from the §3 derivation here; wherever you see "teacher-forcing shifted labels" or "causal mask," assume the decoder-only causal LM context by default.

**Pretraining is not "feed in data, tune hyperparameters against loss" — it is a complete state machine running from raw documents to a checkpoint.** A large share of interview points lost ("$6ND$ is an exact formula," "going past Chinchilla means overfitting," "the causal mask naturally isolates documents") comes from compressing this pipeline down into one or two isolated formulas. Let's first build the overall mental model:

```text
Raw web/book/code snapshots (with license records)
  → ① corpus factory (raw-document stage): extraction/encoding normalization → language ID → basic quality filtering → exact dedup → near-dedup
                      → privacy/safety processing → benchmark decontamination → cluster-level data split (dedup FIRST, split SECOND!)
                      → document-level mixture quotas → deterministic sharding, writing out raw-document shards + full-chain lineage (§2.1-§2.6)
  → ② tokenizer (a version-frozen contract; training details not re-derived here — see tokenization_tutorial.md)
token stream
  → ③ tokenized stage: token-level mixture sampling (corrects the gap between document-level quotas and actual token share, §2.7)
                      → long-document chunk/truncate (§2.9) → write out tokenized training shards (§2.8)
  → ④ data-stream/sampler layer: shard shuffle → domain-sampling RNG → with/without replacement
                      → DP-rank-disjoint sharding → worker/prefetch queue → cursor position (§2.10)
train token stream                                    val/test token stream (frozen version, independent scoring protocol, §2.11)
  → ⑤ objective-to-tensor: teacher-forcing shifted labels → causal / block-diagonal mask
                      → EOS/BOS boundary rules → dual padding mask → document packing (§3)
packed batch (micro-batch × seq_len)
  → ⑥ conceptual state machine for a single update: model/config contract → forward → loss (Σ NLL / Σ valid target tokens)
                      → backward → gradient sync (DP all-reduce, mind the sum/mean reduction semantics, §4.2)
                      → unscale (AMP) / gradient clipping (applied once to the final gradient after the accumulation window closes)
                      → optimizer step (AdamW etc.; formulas pointed to optimizer_lr_schedule_tutorial.md)
                      → scheduler / consumed-token counter advance → atomic checkpoint write (checkpoint boundary)
  → ⑦ lifecycle: warmup/stable/decay (LR) vs. late-stage data reweighting (data side — two separate things, §5.1)
                      → per-domain monitoring / spike-NaN triage (§5.4)
                      → checkpoint (weights + optimizer + RNG + data cursor + …, three strength tiers in §5.2) → resume
```

> Step ⑥ above is the **conceptual execution order** of the stages within a single optimizer step, meant to make explicit a chain that is often glossed over — "model/config contract → forward → backward → gradient sync → unscale/clip → optimizer step → scheduler/token counter → checkpoint." The minimal implementation accompanying this tutorial (§6) covers **sanity checks for a handful of key invariants** (dedup / packing / Chinchilla fitting / spike monitoring); it does not include a runnable training loop or real checkpoint/resume code — the state machine here is a conceptual diagram for building a mental model, not a claim that the §6 code reproduces the whole chain.

**Notation conventions**: $N$ = the compute-bearing parameter count participating in the main matmuls per token (dense models: total; MoE models: **active** — see §1.1 for whether embedding parameters count); $D$ = the number of tokens actually presented to the model and participating in supervision during training (not guaranteed to equal the corpus's "unique token" total — see §2.7; the precise accounting is $B_{target}$ in §4.1); $C$ = training compute (an approximate FLOPs count, not actual GPU wall-clock time); $B_{input}$ = the number of tokens fed into the model in one optimizer step (processed token slots, regardless of whether they participate in the loss); $B_{target}$ = the number of valid target tokens that actually participate in the loss for that same update — the exact formulas and the distinction between $B_{input}$ and $B_{target}$ are in §4.1; $L$ denotes the loss $L(N,D)$ in scaling-law contexts and context length in attention/sequence contexts — this tutorial disambiguates by context, writing $L_{\text{seq}}$ where needed.

> 💡 **The LLM pretraining pipeline in one sitting** — everything you need on one page for interviews (full derivations in §1–§7 below).

1. **$C\approx 6ND$ is only an approximation**: the coefficient $6$ comes from "forward ≈ $2ND$ + backward ≈ $4ND$" (one multiply-add = 2 FLOPs), and $N$ should be understood as compute-bearing/non-embedding parameter count (the counting convention varies across papers); it ignores the $L_{\text{seq}}^2$ term from long-context attention (with total token count $D$ fixed, total attention cost scales approximately with $D\times L_{\text{seq}}$, not $D\times L_{\text{seq}}^2$), vocabulary projection, norm/softmax, activation recomputation, and differing FLOP-counting conventions; it is an algorithm-level approximate training cost, not actual GPU FLOPs, and cannot be converted directly into wall-clock time.
2. **MoE bookkeeping splits into two separate lines**: per-token compute is estimated with **active** parameters; memory/checkpoint size is computed with **total** parameters (details pointed to moe_tutorial.md).
3. **Kaplan (2020) and Chinchilla (2022) exponents are not the same thing**: Kaplan gives $N_{opt}\propto C^{0.73}, D_{opt}\propto C^{0.27}$; Chinchilla has no single "exact exponent" — different estimation methods give roughly 0.50/0.50, 0.49/0.51, 0.46/0.54; the parametric loss model $L(N,D)=E+A/N^\alpha+B/D^\beta$ fits to about $\alpha\approx0.34,\beta\approx0.28$, and the analytic optimum gives $N_*\propto C^{0.452}, D_*\propto C^{0.548}$ (papers reporting full precision often quote roughly 0.46/0.54). The disagreement is not just "more data" — training-horizon design, whether the LR schedule matches the budget, and the fitting method all differ.
4. **"~20 tokens per parameter" is an empirical approximation, not a universal constant**; merely training past this ratio **cannot by itself be judged overfitting** — feeding a fixed model new, non-duplicated, related-distribution data typically still lowers loss; what actually happens is that the configuration becomes "over-trained" relative to the compute-optimal frontier point for its resulting total compute $C=kND$; deliberately over-training a small model is done to cut deployment cost — training-compute-optimal $\ne$ lifecycle-cost-optimal.
5. **The corpus factory is the largest, most self-contained block in this tutorial** (§2): upstream extraction errors easily produce systematic noise, so extraction quality is often a high-leverage step; MinHash estimates the **Jaccard similarity** of shingle sets, not semantic similarity; LSH only recalls candidates — candidates still need exact-Jaccard verification; **dedup must happen before splitting** — near-duplicate samples must be assigned to train/val/test as whole dedup clusters, or a rewritten copy of the same content crossing splits will make validation loss artificially low.
6. **The causal mask in document packing does not automatically isolate documents — this is not a bug**: many systems treat the packed token stream as one continuous EOS-delimited stream and use a plain causal mask as a legitimate choice; if the semantics require documents to be independent, you need a block-diagonal mask **plus** a separate boundary loss mask — blocking attention does not automatically mask cross-document prediction labels.
7. **Loss normalization must be "sum divided by count"**: first averaging within each micro-batch and then averaging those averages introduces bias whenever the number of valid tokens differs across batches.
8. **Global batch: distinguish "fed into the model" from "participates in loss"**: with no padding and equal lengths, the number of tokens fed into the model is $B_{input}=B_{micro,seq}\times L_{seq}\times G_{acc}\times W_{DP}$ — only DP world size directly scales the number of independent samples; TP/PP must not be double-counted. The number of valid targets that actually participate in the loss, $B_{target}=\sum_i m_i$, is usually smaller (even with no padding, shifted labels leave only $L_{seq}-1$ valid targets per sequence — see §4.1).
9. **"Deterministic resume" must be examined in layers**: weights + optimizer state + LR/consumed-token count is only enough for "basic resumability"; adding RNG state + data iterator position + shuffle epoch + sampler/worker state gets you "identical sample sequence"; adding a matching AMP scaler, in-flight gradients, deterministic kernels, and identical software version/communication topology (with unchanged world size) can approach "numerically bitwise-identical trajectory" — the three strength tiers must not be conflated (§5.2).
10. **Loss spike/NaN triage should follow a layered order** (not the sole mandatory procedure): bad batch/data shard → LR and resume state → gradients and activations → mixed precision → cross-rank anomalies → hardware failure, keeping reproducible batch IDs/step numbers throughout.

## §1 Training Budget and Scaling Laws

### 1.1　$C\approx 6ND$: Where the Coefficient Comes From, and What It Ignores

The training compute of a decoder-only dense Transformer is commonly approximated as

$$C \approx 6ND$$

Intuition: count one multiply-add as 2 FLOPs; for each token, the forward pass runs through all the main matrix multiplications (QKV projections, attention output projection, FFN up/down projections, etc.), costing approximately $2N$ FLOPs — here $N$ should be understood as **compute-bearing / non-embedding parameter count**: the embedding lookup itself is an indexing operation, not a matrix multiplication, and mechanically counting embedding parameters into $N$ systematically overestimates this piece of compute; whether the output vocabulary projection is counted separately, and how, is not fully consistent across papers/implementations, so before reading any specific FLOPs estimate you should first confirm which parameter-counting convention it uses — the forward pass over $D$ tokens totals $\approx 2ND$. The backward pass computes both activation gradients and weight gradients, roughly twice the forward cost, i.e., $\approx 4ND$. Forward + backward $\approx 6ND$ (the $6$ here is likewise an approximate coefficient; more precisely it should be written $k\approx 6$).

> ⚠️ **$6ND$ is an algorithm-level approximation, not an exact formula**
> At minimum it ignores/simplifies: the $O(L_{\text{seq}}^2)$ term from long-context attention — this is the attention-score compute **within a single sequence** growing as $L_{\text{seq}}^2$ ($N$ itself does not change with sequence length); but with the total training token count $D$ held fixed, the number of sequences is approximately $D/L_{\text{seq}}$, and multiplying each sequence's $O(L_{\text{seq}}^2)$ by the sequence count means the **total** attention cost grows approximately as $D\times L_{\text{seq}}$, not $D\times L_{\text{seq}}^2$ — don't conflate "$L_{\text{seq}}^2$ within a single sequence" with "total cost growing linearly in $L_{\text{seq}}$ at fixed $D$"; these are different quantities, and the longer the sequence, the less this term can be neglected relative to the other linear terms in $6ND$; vocabulary projection (the output-layer matmul itself is non-trivial when $V$ is large); non-matmul operators such as norm/softmax; **activation recomputation** (activation checkpointing re-runs part of the forward pass to save memory, pushing the effective backward multiplier above 4); MoE's active-vs-total parameter accounting (§1.2); and differing **counting conventions** across communities on whether "1 FLOP" counts a multiply and an add separately or together. $6ND$ gives an algorithm-level approximate training cost — it is **not equal to the FLOPs the GPU actually executes**, and still less can it be used directly to predict wall-clock time (that also depends on MFU, communication overhead, memory swap-in/out, and other engineering factors) — a rough GPU-day estimate additionally needs the GPU count, peak FLOPs, and MFU: $t\approx C/(n_{GPU}\times\text{peak FLOPs}\times MFU)$, and communication overhead, recomputation, and data stalls mean this remains an estimate rather than an exact prediction.

(This approximation was first systematically laid out by Kaplan et al. (2020); full citation in the references at the end.)

### 1.2　MoE: Active Parameters for Compute, Total Parameters for Memory

The per-token compute estimate for an MoE model should plug in **active parameters** (the expert parameters that actually participate in that token's forward/backward computation), not total parameters; but model capacity, memory footprint, and checkpoint file size are still tied to **total parameters** — for a 671B/37B-style model (a DeepSeek-V3-like public configuration, see references), the $N$ in $6ND$ should use the ~37B scale to estimate compute budget, but the checkpoint still needs to be sized at 671B. The specific derivation of routing, load balancing, and aux loss is out of scope here; see `moe_tutorial.md`.

### 1.3　Kaplan (2020) vs. Chinchilla (2022): the Exponents Are Not Interchangeable

The two foundational papers give **different** optimal-configuration exponents; the common interview answer "not enough data, so the exponent is wrong" is incomplete.

**Kaplan et al. (2020)**:

$$N_{opt}\propto C^{0.73},\qquad D_{opt}\propto C^{0.27}$$

As compute budget grows, Kaplan's conclusion is to allocate **more to parameters** rather than data.

**Hoffmann et al. (2022, "Chinchilla")**: the paper cross-validates with multiple estimation methods and **does not give a single "exact exponent"** — different estimation methods yield roughly **0.50/0.50**, **0.49/0.51**, **0.46/0.54**, a family of close but not identical results. The most commonly cited among these is the **parametric loss model**, written as

$$L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

which fits to about $\alpha\approx 0.34,\ \beta\approx 0.28$. $E$ is the **asymptotic irreducible loss floor** fit under this parametric form and this data distribution — the fit itself cannot uniquely decompose it into "data entropy lower bound" and "model family's expressiveness ceiling"; at most it roughly bundles together the data's own entropy and the model family's approximation error, and how much each contributes cannot be determined from this single fit alone. $A/N^\alpha$ is the cost paid for "not enough parameters," and $B/D^\beta$ is the cost paid for "not enough data."

Doing constrained optimization of $L(N,D)$ under the compute-budget constraint $C=kND$ ($k=6$, see §1.1) and eliminating via Lagrange multipliers gives the analytic solution:

$$N_* = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}\left(\frac{C}{k}\right)^{\frac{\beta}{\alpha+\beta}},\qquad D_* = \left(\frac{\beta B}{\alpha A}\right)^{\frac{1}{\alpha+\beta}}\left(\frac{C}{k}\right)^{\frac{\alpha}{\alpha+\beta}}$$

Plugging in the rounded $\alpha=0.34,\beta=0.28$: $\dfrac{\beta}{\alpha+\beta}=\dfrac{0.28}{0.62}\approx 0.452$, $\dfrac{\alpha}{\alpha+\beta}=\dfrac{0.34}{0.62}\approx 0.548$, i.e.,

$$N_*\propto C^{0.452},\qquad D_*\propto C^{0.548}$$

(computing with the paper's full-precision $\alpha,\beta$ is usually reported as roughly 0.46/0.54; [D06] in §6 turns this pair of exponents into an executable fitting unit test, including the numerical assertion that at 100x compute, $N_*$ grows by about $100^{0.452}\approx 8.0$x and $D_*$ by about $100^{0.548}\approx 12.5$x).

> ⚠️ **Three common confusions, clarified one by one**
> 1. **"~0.50/0.50" and "~0.46/0.54" are not the same exact result** — the former is the coarse-grained empirical takeaway that "parameters and tokens grow roughly proportionally," while the latter is the specific exponent fit by the parametric loss model; they are close but should not stand in for each other.
> 2. **"~20 tokens per parameter" is an empirical approximation**, not a universal constant derivable solely from $\alpha,\beta$ — from the analytic forms of $N_*,D_*$ you can see that $D_*/N_*$ also depends on the constants $A,B$ (and $k$), and when $\alpha\ne\beta$, $D_*/N_*\propto (C/k)^{(\alpha-\beta)/(\alpha+\beta)}$ drifts slowly with $C$ rather than staying a strictly fixed ratio (**this derivation is mathematically valid on its own terms**: holding $A,B,\alpha,\beta,k$ fixed and continuing to use this fitted model, the ratio does grow by about 1.56x at 100x compute, per the exponent above). But this drift conclusion **cannot be directly extrapolated into the empirical claim that "the true optimal ratio reliably grows 1.56x for every 100x increase in compute"** — for three reasons: the Chinchilla paper's other two independent estimation methods (IsoFLOP curve fitting, parameter/data ratio fitting) give ratios close to 0.5/0.5, i.e., approximately constant, which doesn't fully match the direction and magnitude of drift seen here; when $\alpha,\beta$ are close to each other, fitting error itself can significantly affect the exponent $(\alpha-\beta)/(\alpha+\beta)$, amplifying the uncertainty in the drift estimate; and $A,B,k$, along with data distribution, model architecture, and training recipe, need not stay constant across many orders of magnitude of compute — yet the premise of this derivation is that they all remain fixed, and 100x compute is itself already a substantial extrapolation.
> 3. **The technical reason for the Kaplan/Chinchilla disagreement** is not just "more data": the two differ in datasets used, experimental coverage, loss-fitting method, and training-horizon design — Kaplan heavily uses loss at midpoints of single training curves (the LR schedule endpoints corresponding to those midpoints were not specifically configured for that budget), while Chinchilla emphasizes configuring a separately matched training schedule for each predetermined token horizon (e.g., aligning cosine-decay length to the target token count) before comparing — this is an important technical factor in the disagreement, not simply "whose dataset is bigger."

### 1.4　"Exceeding the Chinchilla Ratio = Overfitting" Is Wrong

**This is a common misconception that must be corrected**: fixing a model size $N$ and continuing to feed it new (non-duplicated, same- or related-distribution) data typically **still lowers the loss** — this is not the definition of overfitting (overfitting should manifest as training loss dropping while validation loss rises). What actually happens is: as $D$ keeps growing with $N$ fixed, the resulting total compute $C=kND$ (§1.1) is itself growing in step — this is not "training a few extra steps under a fixed budget," it is arriving at a **larger $C$**; it's just that under this new, larger $C$, the configuration $(N,D)$ has a **smaller $N$ and larger $D$** relative to the compute-optimal frontier point $(N_*,D_*)$ (§1.3) for that $C$ — this configuration is called "**over-trained**" relative to the compute it actually consumes, which is an economic judgment relative to the compute-optimal frontier, not an error in training, and certainly not overfitting.

Industry deliberately trains many small models far past the Chinchilla-optimal ratio (e.g., choosing a smaller model at equivalent capability and feeding it more tokens), for reasons of **deployment-side inference cost** (memory, latency, per-token serving cost) — trading a one-time extra training compute cost for cheaper inference over the model's whole lifecycle. This is a tradeoff between "training-compute-optimal" and "lifecycle-cost-optimal," not "the recipe is wrong."

### 1.5　Choosing $N,D$ for a Given Budget: an Engineering Checklist

1. First pin down the optimization target: **training-compute-optimal** (use the $N_*,D_*$ analytic solution directly) or **inference-aware-optimal** (first pin down a smaller $N$ dictated by the deployment budget, letting $D$ grow correspondingly at fixed $C$ — i.e., deliberately "over-training" a small model);
2. Long-context training introduces the $L_{\text{seq}}^2$ attention term that $6ND$ ignores in §1.1; this needs to be budgeted separately rather than folding everything into $6ND$;
3. For MoE models, use active parameters to compute the budget (§1.2), and total parameters when reporting storage/memory budget;
4. Pin down what $D$ actually means — is it the token count "presented during training," or the "effective" data volume after deduplication (§2.7)? High corpus duplication makes the "effective data volume" lower than the nominal $D$, which is exactly the concern behind the line of "data-constrained scaling" work that followed Chinchilla.

## §2 The Corpus Factory: from Web Snapshots to Training Shards

**This is the largest, most self-contained section of the tutorial** — whether pretraining runs stably and correctly often depends more on the corpus-pipeline design than on model architecture or hyperparameter choices.

### 2.1　The Eleven-Step Pipeline Overview + Lineage

An auditable, recoverable data pipeline roughly comprises the following stages (order matters, especially "dedup before split"):

| Step | What it does | Key pitfall |
| --- | --- | --- |
| 1. Raw snapshot + license record | Save crawl timestamp, source, license metadata | Copyright/license traceability |
| 2. Extraction/encoding normalization | Web body-text extraction, strip nav/ads, Unicode normalization | Often matters more for quality than downstream classifiers (§2.2) |
| 3. Language ID | Route by language | Code-mixed text, low confidence on short text |
| 4. Basic quality filtering | Length/symbol-ratio/repeated n-gram/perplexity filters | Uniform thresholds wrongly drop long-tail but legitimate text |
| 5. Exact dedup | Fully duplicate document/substring dedup | Only catches byte-level duplicates, misses near-duplicates |
| 6. Near-dedup | MinHash/LSH to find near-duplicates | Similarity ≠ semantic similarity (§2.3) |
| 7. Privacy/safety processing | PII redaction, illegal content filtering | Needs a dedicated pipeline, independent of quality filtering |
| 8. Benchmark decontamination | n-gram overlap detection against eval sets | A different problem from train-val leakage (§2.6) |
| 9. **Cluster-level data split** | **Assign splits by the dedup clusters from step 6, as whole groups** | **Dedup must be complete before this step** (§2.5) |
| 10. Mixture sampling (document-level quota) | Domain weights decide how many documents each domain contributes | Document-level weight ≠ token-level weight (§2.7) |
| 11. Deterministic sharding | Write out reproducible raw-document shard files | Must be versioned together with lineage (§2.8) |

> **Two stages, not one line**: the 11 steps above happen in the **raw-document stage** — "mixture sampling" in step 10 here refers to the **document-level** quota decision of "what proportion of documents each domain contributes"; "deterministic sharding" in step 11 produces the raw-document shards. **The tokenizer runs after this** (turning raw-document shards into a token stream — see the interface contract in §2.8), and there is then a separate flow occurring in the **tokenized stage**: the actual token-level share resulting from document-level quotas need not match the configured value (this is exactly the token-level sampling-unit issue discussed in §2.7); long documents still need chunking/truncation (§2.9); finally the tokenized training shards are written, feeding into the data-stream/sampler layer of §2.10, and after the split, train vs. val/test diverge into separate handling (§2.11). Reading "mixture sampling" as simply a single step in the table, missing that it happens once at the document level and once at the token level with different meanings, is the easiest place to misread the order of this pipeline.

**Data lineage**: every training sample should ideally be traceable back to — its source dataset/URL, snapshot/crawl version, extraction and filtering pipeline version, which quality filters and thresholds it hit, its dedup cluster ID, its split assignment, and finally which shard file and byte offset it ended up at. This record is the only basis for investigating, months later, whether a given output "memorized" training data or whether a benchmark "got contaminated."

### 2.2　Extraction Quality: Often High-Leverage, Not "Rigorously Proven More Important"

The **most upstream** steps — web body-text extraction (stripping nav bars, ads, footer boilerplate, comment-section noise) and encoding/Unicode normalization — often have quality issues that produce more systematic impact than fine-tuning downstream quality classifiers. This is indirectly supported by empirical observations from work such as the C4 corpus audit (Dodge et al.), RefinedWeb (Penedo et al.), and FineWeb (Penedo et al.), but none of these works rigorously prove a universal ordering that "extraction quality priority is always higher than the classifier's." A more accurate statement is: upstream extraction errors can systematically push large amounts of structural noise (nav text, repeated boilerplate, garbled characters) into every document, and this kind of noise is very "conspicuous" in the token distribution — so extraction quality is often a **high-leverage step** worth prioritizing, not an optional preprocessing detail. The normalization rules used on the training corpus must stay consistent with the rules used when training the tokenizer, or you get a "train/serve split mismatch" drift — the tokenizer's own training details, normalization options, and byte fallback are covered in `tokenization_tutorial.md` §3–§4; this section does not re-derive them.

Language ID typically uses a lightweight classifier (e.g., a fastText-style n-gram model) to route by document; code-switched text and low-confidence language judgments on short text are common engineering edge cases. Basic quality filtering generally layers several heuristic rules (line-length distribution, symbol/letter ratio, repeated-n-gram share) and/or perplexity scoring from a small auxiliary language model — essentially "use cheap signals to filter out the worst tail," not fine-grained semantic judgment.

### 2.3　Exact Dedup vs. Near-Dedup

**Exact dedup**: byte-level/hash-level exact matching (whole-document hash dedup, or suffix-array-based cross-document repeated-substring dedup) — this only catches content that is completely identical, and misses near-duplicates like "the title had one word changed."

**Near-dedup** (MinHash + LSH): represent documents as shingle (n-gram) sets, estimate Jaccard similarity between sets with MinHash (Broder, 1997, see references), then use LSH banding for candidate recall. Three boundaries that must be clarified:

> ⚠️ **MinHash estimates the Jaccard similarity of shingle sets, not semantic similarity**
> Two documents discussing the same concept in completely different wording can have low Jaccard similarity; two documents full of boilerplate (nav bars, license notices, sample code) but unrelated in topic can conversely have high Jaccard similarity. MinHash/LSH catches **surface-level duplication** (mirror sites, near-verbatim copies, templated documents), not "means the same thing."

- **LSH banding only does candidate recall**: candidate pairs typically still need exact-Jaccard verification, because LSH banding by itself only guarantees that candidate probability rises monotonically with similarity (**under the idealized assumption that MinHash rows/hash functions are approximately independent**, $P=1-(1-s^r)^b$, with $b$ bands of $r$ rows each) — it does not guarantee a candidate is actually a true near-duplicate; treating candidates as the conclusion directly introduces false positives.
- **Semantic dedup is not an unconditional upgrade over near-duplicate dedup**: embedding-threshold-based semantic dedup can wrongly delete legitimate text that is "same topic, different phrasing," and this mis-deletion affects low-resource languages, dialects, and templated technical documents unevenly (these texts' own embedding distributions are more concentrated) — you cannot treat it as "a fancier MinHash" and swap it in directly; the two address different levels of duplication.

### 2.4　Privacy/Safety Processing + Benchmark Decontamination

Privacy/safety processing (PII redaction — emails, phone numbers, keys; illegal and harmful content filtering) is a pipeline independent of "quality," usually requiring its own dedicated detection rules and thresholds, and should not be tuned mixed in with quality filters.

**Benchmark decontamination**: detect n-gram overlap between the training corpus and target evaluation sets (e.g., common QA/reasoning benchmarks), and remove or flag overlapping documents. **Exact matching at the 13-gram scale** is the specific approach explicitly used by the GPT-3 paper (Brown et al., 2020, see references) to detect overlap between training corpus and evaluation sets — this is a widely referenced concrete precedent, but the n-gram length actually used differs across teams in practice and should not be treated as a unified industry standard. This guards against **benchmark contamination** — "having seen the eval answers during training" — which is a completely different problem from the train-val leakage discussed in §2.6 below; don't conflate the two.

### 2.5　Dedup Before Split: Cluster-Level Split (Critical)

If you **split first and dedup second**: near-duplicate content (mirror sites, reprinted/rewritten articles, different instances of templated documents) can very plausibly end up with one copy in train and another in val — the model "has seen" a near-verbatim copy of a validation sample during training, so validation loss is **artificially low** — not because the model generalizes well, but because it has effectively memorized a rewritten version of the validation sample. This misleads early stopping, model selection, and any downstream quality reporting based on that validation loss (a systematic discussion of dedup's effect on downstream model quality, closely related to this phenomenon, is in Lee et al., 2021, see references).

**Correct order**: first run near-dedup to partition all documents into connected components (dedup clusters); then atomically assign **whole clusters** to train/val/test — documents in the same cluster must never be split across different sets. The key requirement is **transitive closure**: even if document A was only ever directly compared with B, and B only ever directly compared with C (A and C were never directly compared), as long as A~B and B~C, A and C must end up in the same split — [D01] in §6 turns this rule into an executable assertion using union-find.

### 2.6　Three Kinds of "Leakage" — Must Be Discussed Separately

| Type | Definition | Root cause | Mitigation |
| --- | --- | --- | --- |
| Train-val leakage | Near-duplicate content crosses splits | Split before dedup | Cluster-level split (§2.5) |
| Benchmark contamination | Training corpus contains eval-set content | No decontamination check done | n-gram overlap detection (§2.4) |
| Packed-sequence cross-document historical context | After packing, document B can read document A's tokens | Causal mask permits reading cross-document history | **Usually not data leakage — it's a sequence-semantics choice** (§3.4) |

The third is the easiest to misjudge as a "leakage": a plain causal mask lets a later document read an earlier document's historical tokens, but this is merely "reading prior tokens," not "future-label leakage" in a causal sense — a more accurate term is **"cross-sample context coupling"**: the model, during training, concatenates two independent documents into the same sequence, and their representations end up influencing each other — this is a modeling choice at the level of training dynamics/statistical independence, not information leakage in the evaluation sense. Whether you need a block-diagonal mask to cut this off depends on whether you require the modeling assumption that "every document must be a statistically independent sample" — see §3.4 for details.

### 2.7　Mixture Sampling: Domain Weight Doesn't Necessarily Equal Token Weight

Mixture sampling has at least three different "sampling units," and the same "domain weight" configuration produces **different** final token distributions under each:

- **Document-level sampling**: first select a document pool by domain weight, then draw whole documents uniformly/weighted from within the pool;
- **Token-level sampling**: cut fixed-length chunks directly from the concatenated token stream, decoupled from document boundaries;
- **Sample domain, then sample document**: first select a domain by domain weight, then select a document within that domain.

The difference among the three: different domains have different **average document lengths** — a domain with longer average documents contributes more tokens even if its "document count" weight is configured the same as another domain's. So **when "domain weight" is configured by document count/document weight and then directly used as token weight, the two are usually not equal** — but if the configuration is itself defined as a token-level target share, they can be equal in expectation, so you cannot unconditionally assert "domain weight is always unequal to token weight." A systematic method for domain reweighting can be found in DoReMi (Xie et al., 2023, see references). Whichever sampling unit is used, the "actual token share" must be measured and reported separately after sampling, rather than only reporting the configured value.

This also ties back to what $D$ means in §1: the $D$ in scaling laws usually refers to the number of tokens **presented** to the model during training (the number of times total tokens flow through the optimizer), which is **not guaranteed to equal the corpus's unique-token total** — if some domain is repeatedly oversampled (multiple epochs), or the corpus still retains a large amount of near-duplicate content, the nominal $D$ will be inflated relative to the "effective" data volume, and the extra portion does not carry the same statistical information as genuinely new data; the systematic effect of duplicate data on effective $D$ in such "data-constrained" settings is discussed in Muennighoff et al. (2023, see references) — this section only states the qualitative conclusion, without plugging in its specific numbers.

### 2.8　Deterministic Sharding + the Tokenizer Interface Contract

**Deterministic sharding**: given the same corpus + the same mixture configuration + the same random seed, "which document goes into which shard file, at what position" must be reproducible — this directly underpins deterministic resume in §5.2 (after a restart you must be able to pinpoint exactly where consumption stopped) and also underpins the spike triage in §5.4 (reproducing "the specific batch that caused this spike" depends on being able to precisely reconstruct the data consumed at that time).

**Tokenizer interface contract** (pointer only, not re-derived): the output interface from the corpus factory toward the tokenizer should be a small, frozen contract — a version-locked tokenizer artifact (with its hash), fixed BOS/EOS/PAD IDs, fixed normalization rules, byte-fallback behavior, and the bytes/token ratio needed for budget estimation. How the tokenizer itself is trained (BPE/WordPiece/Unigram) and SentencePiece framework details are covered in `tokenization_tutorial.md` §2–§5; this tutorial only consumes that contract without re-deriving it.

### 2.9　Long-Document Chunking / Truncation

A single document's raw length frequently exceeds the training context length (sometimes by a large multiple — a book or a large code repository can easily exceed a window of a few thousand to tens of thousands of tokens). Such documents need to be **chunked** before entering packing, with several decision points that must be made explicit:

- **Whether to overlap**: non-overlapping chunking is simplest but hard-cuts the context at each chunk boundary; overlapping chunking (e.g., keeping tens to hundreds of tokens of overlap between adjacent chunks) mitigates the loss of context at boundaries, at the cost of the same original text being shared across multiple chunks;
- **Whether the overlap region double-counts loss**: if two adjacent chunks share an overlapping span of tokens, you need to decide explicitly whether that content contributes target loss in only one of the chunks, or in both — counting it in both effectively "oversamples" that content, distorting its true weight within total $D$ (§1, and $B_{target}$ in §4.1); usually the loss for that span should be kept only in the chunk that sees the longer preceding context, and marked as pure context (no loss) in the other chunk;
- **Whether to insert EOS at manual chunk boundaries**: this is a semantic choice, not a standard answer — if you want each chunk treated as an independent sample (echoing the packing semantics of §3.4), the chunk boundary should insert an EOS/BOS boundary and assign a new doc-id, so that the downstream block-diagonal mask and boundary loss-mask rules can correctly recognize "this is actually two artificially split pieces of the same document"; if no boundary marker is inserted, multiple chunks get treated as "a continuous continuation of the same document";
- **How position id / doc-id continue**: if you choose "treat each chunk as an independent sample," each chunk should have its own doc-id (and whether to reset the position id follows the discussion in §3.4); if you choose "chunking is just a split, logically still a continuation of the same document," the doc-id should stay consistent across the chunks, and the position id should also be numbered continuously rather than restarting at 0 for each chunk — this choice must stay consistent with how "document" is defined at training vs. inference time, or you get inconsistencies like "treated as an independent sample during training but assumed to have continuous position at inference."

These decisions have no one-size-fits-all default, but they must be recorded explicitly in the lineage (§2.1): if a piece of training data has been chunked, its lineage should be traceable to "which chunk number of which original document it came from, whether it overlaps with an adjacent chunk, and whether the overlap region counts toward loss" — otherwise, after the fact, you cannot determine whether some observed issue (e.g., an anomalous loss weight for some span) was caused by the chunking strategy.

### 2.10　The Data-Stream / Sampler Layer

Once tokenized training shards are written to disk, the layer that actually reads these shards during training is a **data-stream / sampler layer** that is often skipped over in discussion, but whose state definitions §4 (gradient accumulation) and §5.2 (deterministic resume) repeatedly depend on:

- **Shard shuffle**: whether the read order of shard files is shuffled before each epoch (or each pass), and the random seed/state used for shuffling;
- **Domain-sampling RNG**: if the mixture is sampled dynamically during training (rather than mixed offline ahead of time at fixed ratios), you need a domain-selection random number generator independent of other sources of randomness (dropout, data shuffling), and its state must also be captured by checkpointing;
- **With replacement / without replacement**: whether documents/shards within a domain may be resampled — multi-epoch training, or a data volume insufficient to support compute-optimal $D$ (§1.5), usually requires sampling with replacement, and in that case the "repetition" itself must be recorded, since it directly affects the "nominal $D$ vs. effective data volume" accounting discussed in §2.7;
- **DP-rank-disjoint sharding**: the set of shards/samples read by each DP rank must be pairwise disjoint from every other rank's, or the same piece of data gets counted by multiple ranks simultaneously, equivalent to implicitly double-counting $D$ (§4.4 also mentions this constraint);
- **Worker/prefetch state**: multi-process/multi-thread dataloaders typically prefetch several batches ahead into a queue; the state of this queue itself (what has been prefetched, what has not yet been consumed) is also part of the training state — if resume only restores the cursor for "which sample number we've read up to" but doesn't correctly handle data already pulled out of the prefetch queue but not yet consumed, that data may get skipped or consumed twice;
- **Cursor position**: all of the randomness and ordering decisions above must ultimately converge onto a "cursor" that can be precisely saved and restored — usually "which byte/sample offset of which shard file" — and the precision of this cursor itself determines how precisely the "identical sample sequence" tier in §5.2 can be achieved.

This layer's state is often glossed over as a single phrase, "data iterator position," when discussing deterministic resume (§5.2), but in practice it comprises at least the five independent categories of state above — failing to save any one of them discounts the "identical sample sequence" guarantee of resume.

### 2.11　After the Cluster Split: Train and Val/Test Diverge

The cluster-level split in §2.5 only settles one thing — which dedup clusters belong to which split; once the split is done, train and val/test follow two completely different subsequent flows and cannot be vaguely handled by "reusing the same training mixture":

- **Train**: continues through the full flow of §2.7-§2.10 — mixture sampling (possibly dynamically adjusting domain weights, echoing the late-stage data reweighting of §5.1), shuffling, (if needed) resampling with replacement, and all the state of the data-stream/sampler layer;
- **Val/test**: must be **version-frozen** — once fixed, the same val/test set should keep its sample content, count, and scoring protocol unchanged for the entire training lifecycle (and even across multiple training runs when making fair comparisons); it should not be affected by a dynamically adjusted mixture the way train is, nor should it be "diluted" or "reweighted" by resampling — the per-domain reporting required in §5.3 needs a validation set that is independently comparable per domain, not a subset that follows the train mixture;
- If val/test also needs to cover multiple domains, the sampling within each domain should be determined once and fixed at split time, rather than dynamically resampled every epoch/every run the way train is — otherwise validation loss across different checkpoints is no longer measured on the same yardstick.

Treating val/test as "a scaled-down version of the train mixture" and handling it in passing is the step in this pipeline most easily overlooked — and it directly contaminates the conclusions of the per-domain monitoring in §5.3 and the checkpoint selection in §5.5.

## §3 Objective-to-Tensor: How a Token Stream Becomes a Supervision Signal

### 3.1　Teacher Forcing + Shifted Labels

Standard next-token training uses teacher forcing: given a token sequence $x_1,\dots,x_T$, the **label** at position $i$ is $x_{i+1}$ (shifted right by one), the model outputs the predicted distribution $p(x_{i+1}\mid x_{\le i})$ at position $i$, and the loss is the NLL over all **valid** target positions (§3.5 gives the precise normalization formula). There is no next token at the end of the sequence, so the corresponding position's label is marked `IGNORE` (this tutorial follows PyTorch convention and uses the sentinel value $-100$).

### 3.2　Causal Mask + EOS/BOS Boundary Rules

The plain causal mask $M[q,k]=\mathbb{1}[k\le q]$ only guarantees "can't see the future." Around document boundaries there is one more supervision rule that is frequently implemented incorrectly:

- **Keep** the supervision "last content token of a document → EOS" — this is the direct signal that teaches the model when to stop;
- **Mask (per configuration)** the supervision "EOS of the previous document → first token (or BOS) of the next document" — the "label" at this position is merely an artifact of concatenation, not any meaningful generative continuation.

Both rules can be produced automatically by **the same** document-boundary-based rule: treat EOS as the last token belonging to its own document (sharing the same id as the rest of that document in the doc-id array), and construct labels using the single rule "a label is valid iff the current position and the next position belong to the same document" — the last content token to EOS naturally belongs to the same document (the rule marks it valid), and EOS to the next document's first token naturally crosses documents (the rule marks it invalid). No special-cased logic for EOS is needed; [D04] in §6 demonstrates this.

One more pointer: when assembling a chat template, if you use both the template's own built-in BOS and the tokenizer's `add_special_tokens`, you can easily insert BOS twice — this is a tokenizer/inference-side correctness issue; details are in `tokenization_tutorial.md` §5.5. This section only flags that it pollutes the supervision signal, without re-deriving it.

### 3.3　Padding: the Loss Mask Is Mandatory; Whether the Key Mask Is Needed Depends on the Layout

Padding needs to address two different things:

1. **Key padding mask**: prevents query positions from attending to padding keys;
2. **Loss mask**: excludes positions whose target is padding from the loss (otherwise gradients get computed against meaningless filler targets).

**The loss mask is non-negotiable** — without it, the loss will inevitably be polluted by padding targets. **Whether the key mask is required depends on the sequence layout**: in the common special case of **strict causal + right-side padding**, real queries are naturally positioned before where padding starts, and the causal mask ($k\le q$) already blocks these future padding keys by itself — the key mask is redundant for real queries here, and omitting it does not change real tokens' outputs. But under **left padding, holes within the sequence, or any layout other than "pure causal + right padding,"** the causal condition alone is insufficient to exclude padding keys, and real queries can very well read meaningless content — in that case the key mask cannot be omitted. **A general-purpose implementation should construct both** — omitting the key mask can only be argued for once you've confirmed you are strictly in the right-padding + causal special case and are willing to maintain that implicit assumption long-term; this special case must not be treated as a license to skip the general handling unconditionally.

> ⚠️ **The classic accident when PAD shares the same token ID as EOS**
> Many tokenizers have no dedicated PAD token and simply set `pad_id` to `eos_id` itself — this is **not necessarily wrong**. The real accident happens when downstream code indiscriminately constructs a mask via `input_ids == pad_id`: this masks out **every real EOS position** in the corpus along with actual padding, systematically removing the training signal for "where a document should end," and significantly weakening the model's ability to stop (especially fatal when training from scratch). The correct approach is to construct the mask from **actual padding positions** (each sequence's valid length `valid_len`, or an explicit padding flag carried alongside the batch), never by testing "equal to some token ID." [D05] in §6 turns this mask construction into an executable assertion.

### 3.4　Document Packing: Cross-Document Attention Is Not a Bug

Concatenating multiple documents into the same training sequence (packing) is a common way to improve effective batch utilization. Core facts:

- A plain causal mask only prevents "reading the future"; it **does not automatically prevent a later document from reading an earlier document's historical tokens** — this is not an implementation flaw; many systems deliberately treat the packed token stream as one continuous **EOS-delimited stream** and use a plain causal mask (as noted in §2.6, this is a sequence-semantics choice, not data leakage);
- If the semantics require "every document is an independent sample," you need a **block-diagonal causal mask**:

$$M[q,k] = (k\le q)\ \wedge\ (\text{docid}[q]=\text{docid}[k])$$

Whether to reset the position ID within each document depends on the positional encoding scheme and the training semantics — it is not an unconditional hard requirement: under **absolute positional embeddings**, not resetting causes later documents to start from a huge position number that doesn't belong to them, so resetting is usually appropriate; **standard RoPE** depends only on the relative position difference between query and key, and once the block-diagonal mask has already confined each document's attention to within that document, adding the same overall offset to all positions within one document does not, under idealized math, change the relative phase within the document — it will not pollute the document's internal attention computation merely for "not starting numbering from 0." Still, some RoPE length-extrapolation/scaling variants, numerical behavior at finite precision, and the engineering desire for every document to strictly start training at position 0 (e.g., to align with the position distribution at inference) lead most implementations to reset anyway;

- **Blocking attention does not automatically mask cross-document prediction labels** — this is the most common bug in packing implementations: a team adds a block-diagonal mask and assumes "documents are now fully independent," but if the labels aren't handled separately, document A's last position still defaults to being required to predict document B's first token (an unambiguous cross-document target) unless it is separately excluded with the boundary loss mask from §3.2.

**Numerical example** (tokens = `[11,12,13,21,22]`, doc IDs = `[0,0,0,1,1]`): under independent-document semantics, the shifted labels are `[12,13,IGNORE,22,IGNORE]`; the block mask asserts `M[3,2]=False` (position 3 belongs to document 1, position 2 to document 0, so it's blocked for crossing documents even though $k\le q$), `M[4,0]=False`, `M[4,3]=True`; under continuous-stream semantics the same position `M[3,2]=True` — this is a legitimate configuration choice, not an implementation failure.

> ✅ **A stronger functional test than "just checking mask shape"**
> Construct $QK^\top=0$ (softmax degenerates to uniformly distributing attention over the allowed keys), set document 0's three positions' values to `[100,100,100]`, and document 1's two positions' values to `[0,0]`. Under a plain causal mask, position 3's output (document 1's first token) is the mean over allowed keys `[0,1,2,3]` $= (100+100+100+0)/4=75$; under the block mask, only key `3` itself is available, so the output $=0$. Now change document 0's values from 100 to 1000: **under block mode, document 1's output stays completely unchanged (still 0)** — this is exactly the property "document independence" should have; under continuous-stream mode the same output changes from 75 to 750, because it was never decoupled from document 0. Merely checking the mask's shape or whether each row normalizes to 1 will not catch this kind of bug, where the boundary logic is wrong but the shape still "looks right"; this "change a value, recompute, and check whether the other document doesn't budge" test is the assertion that actually locks down independence. [D03] in §6 is the executable version of this test.

### 3.5　Loss Normalization: Sum Divided by Count, Not the Mean of Means

The correct training loss is

$$\mathcal{L} = \frac{\sum_i \mathrm{NLL}_i}{\#\{\text{valid target tokens}\}}$$

Sum the NLL over all valid target tokens in the **global batch**, and divide by the total count of valid tokens. A common but subtle bug: first computing "the mean loss within this micro-batch" separately for each micro-batch, then averaging those means (e.g., across the multiple micro-batches of gradient accumulation, or across multiple DP ranks). When micro-batches have different numbers of valid tokens (different padding ratios, different packing efficiency, or an ongoing short-to-long context-length transition), this "mean of means" implicitly overweights micro-batches with fewer tokens and underweights those with more, deviating from a loss that is genuinely weighted per token. A correct implementation must separately accumulate the two quantities "sum of NLL" and "total count of valid tokens" across the entire gradient accumulation window (and even across DP-rank communication), and only divide once at the very end.

If the model is MoE, the total loss usually also needs a router auxiliary loss term added per configuration (router aux loss, typically much smaller in magnitude than the main loss) — the specific mechanism here is out of scope for this tutorial; see `moe_tutorial.md`.

### 3.6　Exact PPL/BPB Formulas

$$\mathrm{PPL} = \exp\left(\frac{\sum_i \mathrm{NLL}_i}{\#\text{valid target tokens}}\right),\qquad \mathrm{BPB} = \frac{\sum_i \mathrm{NLL}_i}{\#\text{bytes}\cdot\ln 2}$$

PPL is only suitable for direct comparison when tokenizer, normalization, and eval text are all fully consistent — the denominator (token count) itself depends on the tokenizer, so even with the same total NLL, a tokenizer that splits more finely will pull loss/token lower. BPB replaces the denominator with the UTF-8 byte count of the same text, making it more robust to tokenizer differences, but it is still affected by whether encoding/preprocessing is consistent. The full derivation and a numerical example of "loss/token differs by 2x while BPB is the same" are in `tokenization_tutorial.md` §1.3 and §7 [D07]; this section does not repeat them, and only emphasizes the exact form of these two formulas.

## §4 Update Accounting: What Exactly Is the Global Batch

### 4.1　Micro-batch / DP / Gradient Accumulation: the Global Batch Formula, and Three Different Token-Counting Conventions

Let's pin down notation first: $B_{micro,seq}$ is the number of sequences per micro-batch **per DP replica** (not "per GPU" — when TP/PP > 1, a single DP replica itself spans multiple GPUs, so "per GPU" and "per DP replica" are not the same thing), $L_{seq}$ is the sequence length, $G_{acc}$ is the number of gradient accumulation steps, and $W_{DP}$ is the data-parallel world size.

**Number of tokens fed into the model** (processed token slots, regardless of whether they participate in the loss):

$$B_{input} = B_{micro,seq}\times L_{seq}\times G_{acc}\times W_{DP}$$

**Only $W_{DP}$ directly scales the number of independent samples** — tensor parallelism (TP) and pipeline parallelism (PP) split the computation for **the same** sample (splitting a matmul's dimensions, or splitting layers across a pipeline), and must not be double-counted into the independent-sample count; multiplying TP/PP world size into $B_{input}$ is the most common source of global-batch miscalculation. (The communication primitives and splitting mechanisms of TP/PP/CP/ZeRO/FSDP are out of scope here; see `distributed_training_tutorial.md`.)

**Number of valid targets that actually participate in the loss**:

$$B_{target} = \sum_i m_i$$

where $m_i$ is the valid-target mask count for the $i$-th sequence (the sum of the loss mask from §3.5). $B_{input}$ and $B_{target}$ are generally not equal, even with **absolutely no padding**: under teacher-forcing shifted labels, the last position of every length-$L_{seq}$ sequence has no next token and is marked `IGNORE` (§3.1), so with no padding and no document-boundary masking, the number of valid targets per sequence is $L_{seq}-1$, not $L_{seq}$; layering on padding (§3.3) or packing's document-boundary loss mask (§3.2, §3.4) further shrinks $B_{target}$ below $B_{input}$.

**Which quantity each of the three accounting conventions corresponds to — do not mix them up**:

- **Processed token slots** (i.e., $B_{input}$, "how many token positions were passed through") — this is usually the convention used to advance the consumed-token counter in §5.1 and when "aligning the scheduler by token";
- **Non-padding input tokens** ($B_{input}$ minus padding positions) — this convention is used to estimate actual effective GPU throughput and evaluate packing/padding efficiency;
- **Valid target tokens** (i.e., $B_{target}$) — this is the denominator for §3.5's loss normalization, and the $D$ in the §1 scaling law (the number of tokens presented to the model and participating in supervision during training) should track this convention more closely than the blanket $B_{input}$ whenever there is padding/document-boundary masking.

Once these three get mixed up, the question "how many tokens did this training run feed in total" ends up with several mutually inconsistent numbers appearing in reports.

### 4.2　Conditions for Gradient Accumulation to Be Strictly Equivalent to "One Big Batch"

For gradient accumulation to be **strictly equivalent** to directly running one big batch, all of the following must hold simultaneously:

1. **Identical samples** — the data consumed within the accumulation window must exactly match what the big batch would consume in one shot; sharding differences must not cause missed or duplicated samples;
2. **Normalization over all valid tokens** (§3.5) — not the mean of each micro-batch's mean;
3. **Only one optimizer step executed** — gradients are only summed/averaged within the accumulation window, and the weight update happens exactly once, at the end of the window;
4. **Correct handling of randomness and state** — gradient clipping must act on the **final gradient** after accumulation completes (not applied separately to each micro-batch's local gradient); dropout and other sources of randomness, as well as any running statistics computed across samples, must behave as though computed in one shot on the full big batch;
5. **Matching the communication library's reduction semantics** — frameworks like DDP/FSDP by default **average** gradients across ranks rather than summing them; if local code has already computed a normalized loss as "sum of NLL / this rank's valid-token count," the default "averaging" during all-reduce implicitly divides by world size one more time — this must be explicitly changed to sum-reduce, or conversely the local normalization must premultiply by $W_{DP}$ to compensate; the two must be matched, or equivalence breaks at this very step.

The points above guarantee **mathematical equivalence** — under exact real-number arithmetic, the gradient obtained via accumulation is the same quantity as the gradient computed from a one-shot big batch; it does **not** guarantee **bitwise-identical floating-point values** — the order of summation, the reduction-tree structure of all-reduce, and so on will still introduce differences in the last few significant digits (this distinction is consistent with the three-tier strength framework for deterministic resume in §5.2).

### 4.3　Numerical Precision: Operational Constraints, Not a Numeric-Format Primer

BF16 has the same exponent width as FP32 (similar dynamic range), and in pretraining is generally less prone to overflow than FP16; FP16 has a narrower dynamic range and relies more on loss scaling to prevent small gradients from underflowing to zero. This section only gives the operational constraints relevant to triaging spikes/NaN in §5.4, without expanding on the derivation of floating-point formats themselves.

### 4.4　Sibling Boundaries (Strictly Enforced)

- **optimizer_lr_schedule_tutorial.md** owns the derivation of the AdamW update formula and the warmup/cosine/WSD scheduling math. This tutorial only cares about: why optimizer state (Adam's $m,v$) must be saved together with the checkpoint for correct resume (§5.2), whether the schedule advances by step or by token count, how the schedule/LR configuration must be co-adjusted when batch size changes, and what role each of the warmup/stable/cooldown phases plays across the whole lifecycle (§5.1) — the concrete formulas are always pointed to that tutorial.
- **distributed_training_tutorial.md** owns the communication primitives and splitting mechanisms of ZeRO/FSDP/TP/PP/CP. This tutorial only cares about: the global-batch accounting in §4.1, and that the data consumed by each DP rank must not overlap with any other's (or $D$ gets double-counted). **Exactly how a checkpoint gets re-sharded when world size changes, and whether the resulting training trajectory can be guaranteed to match the trajectory at unchanged world size, is a question neither this tutorial nor that sibling gives a systematic answer to** — this document only states, in §5.2, the weaker qualitative conclusion that "a world-size change usually cannot guarantee a bitwise-identical trajectory"; the derivation of the specific communication/splitting mechanism is always pointed to that tutorial, but readers should not assume it has already systematically covered this specific question of "determinism guarantees after re-sharding."

## §5 Lifecycle: from Warmup to Deterministic Resume

### 5.1　Warmup/Stable/Decay/Cooldown + Data-Stage Changes

The lifecycle divides roughly into several phases along the LR schedule (this only discusses each phase's **role** — the concrete math is in `optimizer_lr_schedule_tutorial.md`): warmup suppresses early-training instability; stable/plateau occupies the bulk of training; **decay/cooldown is an LR-side concept** — the learning rate decreases per schedule down to a very small value, with the formula and schedule shape pointed to `optimizer_lr_schedule_tutorial.md`. **Data annealing/late-stage reweighting is an independent, data-side concept** — many modern training recipes increase the weight of high-quality/curated data within the mixture toward the end of training (this practice is described in public training reports such as Llama 3's — see references). These two things **often happen simultaneously during the same late stage of training**, but they are controlled by two independent mechanisms — the LR schedule and the data-mixture configuration — respectively, and "LR is decaying" and "data is being reweighted" should not be treated as two ways of saying the same thing: a training recipe can perfectly well do LR decay without adjusting data weights, and vice versa.

Another lifecycle axis is the **context-length curriculum**: most of the training budget uses shorter sequences, switching to long context later. This changes $L_{seq}$, meaning both the global-batch accounting in §4.1 and the consumed-token counter must be recomputed consistently at the switch point, rather than silently carrying over the pre-switch assumptions.

### 5.2　"Deterministic Resume" Must Be Split into Three Different Strength Tiers

The phrase "the training trajectory after resume is exactly the same as if it had never been interrupted" is itself ambiguous — different engineering goals need very different necessary conditions, and it's best split into three increasingly strong tiers discussed separately:

1. **Trainable to continue** (weakest): the model can restore weights and optimizer state from a checkpoint and continue lowering the loss, without requiring the data order or numerical values to match exactly. Only model weights, optimizer state, and the LR schedule/consumed-token count are needed.
2. **Identical sample sequence**: after restart, the order of data the model sees (which sample at which step) exactly matches the uninterrupted run, without requiring bitwise-identical numerics. Beyond tier 1's content, you must also have: RNG state (generators across Python/framework/CUDA, etc.), data iterator position, shuffle epoch, sampler state, and the state of the data-loading worker/prefetch queue (§2.10 — otherwise the multi-worker prefetch order is itself a source of nondeterminism); multi-rank training also needs each rank's checkpoint metadata.
3. **Numerically/bitwise identical trajectory** (strongest, usually not fully achievable): on top of tier 2, you additionally need the AMP loss-scaler's state (otherwise resume starts from a different dynamic loss-scale value, and the gradient-scaling path diverges in the first few steps), any gradient still "in flight" within an incomplete gradient-accumulation window (pending gradients — avoidable if checkpoints are only taken at accumulation boundaries), deterministic kernels (non-deterministic operators like cuDNN/cuBLAS must be explicitly disabled), and an identical software version and communication topology (different GPU counts or different collective-communication implementations reduce floating-point sums in different orders, accumulating ULP-level differences). **When world size changes**, no matter how carefully the re-sharding semantics are written, it usually still cannot guarantee a bitwise-identical trajectory relative to unchanged world size — the data-splitting scheme and the gradient-reduction tree structure both change.

Missing the state required by any tier downgrades the guarantee to a weaker tier: missing the data iterator position immediately invalidates tier 2 (after restart, either already-trained data gets consumed again, or a segment gets skipped); missing RNG state means dropout/data shuffling diverges from the original trajectory immediately; missing the AMP scaler/pending gradients/deterministic-kernel settings means tier 3 is unreachable even if the first two tiers are satisfied. Details of how re-sharding works when world size changes are pointed to `distributed_training_tutorial.md` (that sibling currently does not systematically cover the stronger guarantee of "still bitwise-reproducible after re-sharding" — it only covers the mechanism of sharding itself, §4.4).

### 5.3　Monitoring: Per-Domain Validation Loss

Validation loss should be reported **per domain** (code/math/low-resource languages/long context, etc.) separately, not just as one aggregate loss — the aggregate loss is easily dominated by the domain with the highest token weight (e.g., if web text is 80% of the mixture, the aggregate loss curve basically only reflects web-text quality), which can mask a lower-weight but strategically important capability (code correctness, math reasoning, some low-resource language, long-context retrieval) degrading — by the time it's noticed, it's often already deeply entrenched.

### 5.4　Loss Spike / NaN: Layered Triage

The recommended triage order (not a one-shot guess, but working down a hierarchy of "cheapest to check first"):

1. **Bad batch / data shard** — first check whether this spike can be reproduced in isolation on the same batch/shard (this depends on the deterministic sharding of §2.8, the sampler/cursor state of §2.10, and lineage — without them you can't even pin down "which batch it was");
2. **LR and resume state** — a misconfigured LR schedule, or an LR-schedule counter accidentally reset after a resume, can both produce an anomalous update whose actual step count doesn't match expectations;
3. **Gradients and activations** — anomalous gradient norms, exploding activation norms, a layer's logit magnitude going out of control — why these quantities go out of control, and how norm/residual/initialization mechanistically suppress this kind of blowup, is pointed to `normalization_init_tutorial.md` (that sibling covers **the stability mechanisms themselves**, not a ready-made monitoring-threshold/diagnostic-metric operations manual; what thresholds to use to trigger alerts in your own training still needs to be calibrated by you). This section only lists these quantities as things to check, without re-deriving the formulas;
4. **Mixed precision** — BF16/FP16 overflow/underflow (§4.3);
5. **Cross-rank anomalies** — some DP/TP/PP rank silently diverging (e.g., a hardware issue on one card) while the rest of the ranks appear normal on the surface — this is only visible if per-rank loss/gradient statistics are recorded separately;
6. **Hardware failure** — ECC errors, single-card failure, collective-communication jitter.

Keeping reproducible batch IDs / step numbers / shard assignments (§2.8, §2.10) is what allows after-the-fact replay-based triage, rather than guessing from memory. [D07] in §6 is the automated first line of defense — a detector based on rolling median/MAD, whose job is to flag "when this triage should begin," not to determine the root cause.

### 5.5　Checkpoint Selection

Checkpoint selection for downstream deployment should be based on the per-domain validation sets / dedicated evaluation-set combination from §5.3, not just training loss — under a fixed data mixture, training loss mainly reflects fitting progress, and does not directly correspond to generalization ability or the specific capability profile you actually care about.

### 5.6　Sibling Boundaries

- **normalization_init_tutorial.md** owns the complete formula derivations for norm/residual/initialization and the **stability mechanisms** themselves; this tutorial only references the **names of the diagnostic quantities** — activation norm / gradient norm / logit magnitude / residual instability — in NaN diagnosis (§5.4), without re-deriving them. That sibling explains "why these mechanisms make training more stable," not a monitoring-metrics manual you can copy directly; specific monitoring thresholds still need to be engineered on your own.
- **moe_tutorial.md** owns the complete mechanism derivation for routing and load-balancing; this tutorial only cares about the router aux loss term that may appear in the total loss (§3.5), and the two bookkeeping rules of using active parameters for compute budget and total parameters for storage/memory (§1.2) — the routing mechanism is always pointed to that tutorial.

## §6 From-Scratch Implementation: Corpus → Packing → Key-Invariant Sanity Checks (Not a Complete Training Loop)

See [`code/pretraining_pipeline.py`](code/pretraining_pipeline.py) for the complete runnable script (pure Python standard library, zero third-party dependencies, runs all of [D01]–[D07] sanity checks in seconds on CPU). Four demo groups are chained into a mini pipeline: text → dedup → pack + loss mask → synthetic budget planner → spike monitor — this covers a handful of the **key invariants** most prone to being implemented wrong and most worth backstopping with executable assertions; it is not an end-to-end runnable training loop: the script contains no real forward/backward/optimizer step, and no real checkpoint writing and resume (that conceptual state machine is in the §0 overview and the three-tier strength framework of §5.2). The main text only shows the three most essential code blocks; the full functional test for document packing, the EOS boundary (§3.2), and padding (§3.3) are folded into the same script as short test functions and are not pasted line by line here.

**[D01] MinHash / LSH / cluster-split transitive closure** (the executable version of §2.3 and §2.5):

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

**[D02]/[D03] document packing: shifted labels + block-diagonal mask + functional test** (the executable version of §3.4 — the most essential and most error-prone segment in the whole codebase):

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

**[D06] synthetic Chinchilla curve fitting** (the executable version of §1.3; **this is a fitting unit test, not a validation of the empirical law**):

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

[D04] (EOS boundary, reusing [D02]'s same doc-id rule), [D05] (padding's dual mask + no-empty-row check), and [D07] (rolling median/MAD spike detection + unconditional NaN/Inf alerting + spike/sustained-shift event classification) are folded into `main()` as independent groups of assertions; their logic is already covered clearly in §3.2, §3.3, and §5.4 respectively, so the code is not repeated here.

## §7 26 High-Frequency Interview Questions

Split into three difficulty tiers; expand to see the answer's key points + common pitfalls. When answering, **don't re-derive the main text** — organize by "framework → key formula → common mistakes."

### L1 Must-Know Questions

<details>

<summary>Q1. What are $N$, $D$, $C$? Where does the "6" in $C\approx 6ND$ come from, and why is it only an approximation?</summary>

- $N$ = the compute-bearing parameter count per token in the main matmuls (MoE uses active; embedding lookups are usually excluded — see §1.1); $D$ = the number of tokens presented to the model during training; $C$ = approximate training FLOPs
- One multiply-add counts as 2 FLOPs (one multiply, one add): the forward pass runs each token through the main matmuls, totaling about $2N$/token, or $2ND$ for $D$ tokens; the backward pass computes both activation gradients and weight gradients, roughly twice the forward cost, i.e., $4ND$; forward + backward $\approx 6ND$ — this is itself a counting convention (some tools count a multiply-add as 1 FLOP), not a physical constant
- It ignores the $L_{seq}^2$ term from long-context attention (total cost scales approximately with $D\times L_{seq}$ at fixed $D$), vocabulary projection, norm/softmax, activation recomputation, and differing FLOP-counting conventions; it is not an exact formula and cannot be directly converted into wall-clock time — a rough GPU-day estimate additionally needs GPU count, peak FLOPs, and MFU: $t\approx C/(n_{GPU}\times\text{peak FLOPs}\times MFU)$, and communication overhead, recomputation, and data stalls mean this remains an estimate, not an exact prediction (§1.1)

Treating $6ND$ as an exact formula, indiscriminately folding embedding parameters into $N$, or dividing $C$ by peak FLOPs directly as training time (ignoring MFU) are the three most common ways to lose points on this question.

</details>

<details>

<summary>Q2. How are inputs and labels shifted relative to each other in a causal LM? Why is the loss of a randomly initialized model usually close to $\ln V$?</summary>

- Under teacher forcing, the label at position $i$ is $x_{i+1}$ (shifted right by one); the last position of the sequence has no next token and is marked `IGNORE` ($-100$ in this tutorial, §3.1)
- In packing/document-boundary settings, the rule "a label is valid iff the current position and the next position belong to the same document" uniformly handles both sources of IGNORE — sequence-end and document boundary (§3.2)
- A randomly initialized model assigns approximately uniform probability to the next token, so the NLL at each valid position is approximately $\ln V$ ($V$ is vocabulary size) — the loss observed at the very start of training should be close to this value; a noticeable deviation (much higher, or an anomalously low value) usually points to a problem with initialization, label alignment, or the data itself

Assuming "IGNORE only occurs at the end of a batch," missing that document boundaries also produce IGNORE, or being unable to explain where the $\ln V$ initial-loss magnitude comes from, are the common incomplete answers to this question.

</details>

<details>

<summary>Q3. What's the difference in dynamic range between BF16 and FP16? When is loss scaling needed?</summary>

- BF16 has the same exponent width as FP32, so its dynamic range is similar and it's generally less prone to overflow than FP16 in pretraining; the cost is fewer mantissa bits, so the relative precision of a single value is lower than FP16's
- FP16's exponent width is narrower than BF16/FP32's, so its dynamic range is smaller — small gradients easily underflow to zero in FP16, and large values also overflow to Inf more easily
- **Loss scaling** (multiplying the loss by a large factor before backprop, then unscaling the gradients by the same factor before clipping or the optimizer step) is the standard practice in FP16 training used to "lift" small gradients into FP16's representable range; since BF16's dynamic range is already close to FP32's, this kind of loss scaling is usually unnecessary (§4.3)
- This is also one of the things checked under the "mixed precision" layer when triaging loss spikes/NaN in §5.4 — the scaler's state itself is also part of the state that deterministic resume in §5.2 needs to consider

Equating "mixed-precision training" with "both BF16 and FP16 need loss scaling," without being able to explain the different engineering needs that follow from their different dynamic ranges, is the common wrong answer to this question.

</details>

<details>

<summary>Q4. If the number of training tokens exceeds the Chinchilla-recommended ratio, is that overfitting?</summary>

- Not necessarily. "Exceeding the Chinchilla-recommended ratio" alone cannot by itself be judged as overfitting — if the additional data being fed is **new, non-duplicated, same- or related-distribution** data, a fixed model size can typically still lower the loss, which does not satisfy the overfitting definition of "training loss drops while validation loss rises"
- What actually happens: as $D$ keeps growing, the resulting compute $C=kND$ itself grows too; under this larger $C$, the configuration has a smaller $N$ and larger $D$ relative to the compute-optimal frontier point for its own resulting compute — this is called "over-trained," an economic judgment relative to the compute-optimal frontier, not an error in training (§1.4)
- Deliberately over-training a small model is done to reduce deployment-side memory/latency/inference cost — training-compute-optimal $\ne$ lifecycle-cost-optimal

Conflating "over-trained" (an economic concept relative to budget) with "overfitting" (a generalization failure), or asserting "definitely not overfitting" while ignoring the "new data/non-duplicated" precondition, is the number-one way to lose points on this question.

</details>

<details>

<summary>Q5. What is document packing? Why doesn't a plain causal mask automatically isolate documents?</summary>

- Packing: concatenating multiple documents into one training sequence to improve utilization
- The causal mask only guarantees "can't see the future" — it doesn't check whether tokens belong to the same document; a later document can by default read an earlier document's historical tokens. This is the continuous EOS-delimited stream semantics that **many systems deliberately adopt**, not a bug
- If "document independence" is required, you need a block-diagonal mask $M[q,k]=(k\le q)\wedge(\text{docid}[q]=\text{docid}[k])$ (§3.4)

Directly judging "being able to attend cross-document" as an implementation error, without being able to explain that this is a legitimate semantic choice, is the pitfall here.

</details>

<details>

<summary>Q6. What's the difference between the attention mask and the loss mask? When are both required, and when might one be redundant?</summary>

- The attention mask (key padding / block-diagonal) controls "who can see whom" and operates within the attention computation; the loss mask controls "which positions' predictions participate in the gradient" and operates within the loss computation — the two are independent, and blocking attention does not automatically mask the corresponding label
- **In block-diagonal settings, both are mandatory**: even if document B can't read document A, document A's last position can still be required to predict document B's first token, unless separately excluded with the boundary loss mask (§3.2, §3.4)
- **In padding settings, it depends on the layout**: the loss mask is mandatory; but with strict causal + right-side padding, the causal mask itself already blocks future padding keys, so the key mask may be redundant for real queries — under left padding or other layouts it cannot be omitted (§3.3)

Assuming "both masks are always equally required," or the opposite, "the key mask can always be omitted unconditionally," are both pitfalls here; the correct answer must be discussed case by case according to the specific layout.

</details>

<details>

<summary>Q7. How exactly is global batch size / tokens-per-update computed? Distinguish the different conventions</summary>

- Tokens fed into the model: $B_{input}=B_{micro,seq}\times L_{seq}\times G_{acc}\times W_{DP}$ ($B_{micro,seq}$ is the number of sequences per DP replica, not "per GPU"); only DP world size directly scales the number of independent samples — TP/PP split the computation of the same sample and must not be double-counted
- Valid targets actually participating in the loss: $B_{target}=\sum_i m_i$ ($m_i$ is each sequence's valid-target mask count) — even with absolutely no padding, because the label at each sequence's last position is `IGNORE`, $B_{target}$ is $L_{seq}-1$ per sequence rather than $L_{seq}$; with padding/packing boundary masking it shrinks further below $B_{input}$
- The three accounting conventions must not be mixed: advancing the consumed-token counter/aligning the scheduler usually uses $B_{input}$; the loss-normalization denominator and the $D$ in the scaling law should track $B_{target}$ more closely (§4.1)

Multiplying TP/PP world size into the batch-size formula as well, or treating $B_{input}$ and $B_{target}$ as the same number and swapping them freely, are the two most common types of computational error here.

</details>

<details>

<summary>Q8. Why can't PPL from different tokenizers be compared directly?</summary>

- PPL's denominator is token count, which itself depends on the tokenizer — for the same total NLL, a tokenizer that splits more finely yields a lower loss/token
- BPB replaces the denominator with the byte count of the same text, making it more robust to tokenizer differences, but it still requires both sides to be evaluated on the same normalized byte stream
- Exact formulas: $\mathrm{PPL}=\exp(\sum \mathrm{NLL}_i/\#\text{valid tokens})$, $\mathrm{BPB}=\sum \mathrm{NLL}_i/(\#\text{bytes}\cdot\ln 2)$ (§3.6; full derivation in tokenization_tutorial.md §1.3)

Only knowing "lower PPL is better," without being able to state the precondition that it depends on the tokenizer, is the incomplete answer here.

</details>

### L2 Advanced Questions

<details>

<summary>Q9. What's different between the Kaplan (2020) and Chinchilla (2022) scaling-law conclusions?</summary>

- Kaplan: $N_{opt}\propto C^{0.73}, D_{opt}\propto C^{0.27}$ — as compute budget grows, more goes to parameters
- Chinchilla: no single "exact exponent" — different methods give roughly 0.50/0.50, 0.49/0.51, 0.46/0.54; the parametric model $L=E+A/N^\alpha+B/D^\beta$ fits to $\alpha\approx0.34,\beta\approx0.28$, with analytic optimum $N_*\propto C^{0.452}, D_*\propto C^{0.548}$
- The disagreement is not just "more data": training-horizon design, whether the LR schedule matches the budget, and the fitting method all differ (§1.3)

Answering only "Chinchilla says you need more data," without being able to state the three specific sets of exponents or the key technical disagreement around training horizon/LR schedule — this question requires remembering all three estimation methods and their applicable scenarios simultaneously, which goes beyond an L1-level shallow answer and is the crux of this question.

</details>

<details>

<summary>Q10. Derive the analytic solution $N_*,D_*$ under Chinchilla's constrained optimization.</summary>

- Objective: $\min L(N,D)=E+A/N^\alpha+B/D^\beta$ s.t. $C=kND$
- Substitute the constraint to eliminate one variable, then use Lagrange multipliers or directly differentiate with respect to $\log N$ to find the stationary point, giving $N_*=(\alpha A/\beta B)^{1/(\alpha+\beta)}(C/k)^{\beta/(\alpha+\beta)}$, $D_*=(\beta B/\alpha A)^{1/(\alpha+\beta)}(C/k)^{\alpha/(\alpha+\beta)}$
- Plugging in $\alpha=0.34,\beta=0.28$ gives exponents 0.452/0.548; at 100x compute budget, $N_*$ grows by about $100^{0.452}\approx 8.0$x and $D_*$ by about $100^{0.548}\approx 12.5$x — this derivation is mathematically valid when $A,B,\alpha,\beta,k$ are held fixed, but it cannot be directly extrapolated into the empirical claim that "the true optimal ratio must grow 1.56x for every 100x increase in compute" (§1.3, [D06])

Failing to derive that the exponents take the cross form $\beta/(\alpha+\beta)$ and $\alpha/(\alpha+\beta)$ (easy to misremember as $\alpha/(\alpha+\beta)$ corresponding to $N_*$), or mistaking this mathematical derivation for an empirical rule that can be extrapolated unconditionally, are the pitfalls here.

</details>

<details>

<summary>Q11. Given a compute budget, how do you decide model parameter count and training token count?</summary>

- First separate the goals: training-compute-optimal (directly plug into $N_*,D_*$) vs. inference-aware-optimal (first pin down a small $N$ dictated by the deployment budget, letting $D$ grow correspondingly at fixed $C$ — i.e., deliberately over-training)
- For MoE, use active parameters for compute budget, total parameters for memory/checkpoint (§1.2)
- Long context introduces the $L_{seq}^2$ term that $6ND$ ignores, requiring a separately reserved budget (§1.5)

Only being able to recite the $N_*,D_*$ formulas, without being able to state the "training-optimal ≠ inference-aware-optimal" fork (this is also a frequently cited hotspot in design reviews).

</details>

<details>

<summary>Q12. How should the three knobs — attention mask, loss mask, position id — be designed independently within packing?</summary>

- This is not a choice between two pre-packaged recipes, "continuous stream" vs. "independent documents" — **the attention boundary, the target boundary, and the position id are three knobs that can be combined independently**, not necessarily bundled together:
  - **Attention boundary**: plain causal ($k\le q$, allows reading history across documents) or block-diagonal ($k\le q \wedge \text{docid}[q]=\text{docid}[k]$, documents mutually invisible);
  - **Target boundary**: even when using plain causal attention (documents can "see" each other), you can still separately use the document-boundary rule to mask the cross-document label "previous document's EOS → next document's first token" (§3.2) — attention allowing cross-document history reads does not mean this target must be kept;
  - **Position id**: whether to reset it is a choice determined by the positional-encoding scheme, not a mathematical necessity — standard RoPE is insensitive under idealized math to an overall position shift within one document; by contrast, absolute positional embeddings should usually be reset, and the specific tradeoff is discussed in §3.4
- The two most common combinations are "continuous stream = causal attention + no position-id reset + no extra target masking" and "independent documents = block-diagonal attention + position-id reset + cross-document target masking," but in engineering there are also intermediate configurations like "causal attention but still masking cross-document targets" — the two extremes should not be treated as the only two options
- Both must retain the supervision "last content token of a document → EOS"; only the handling of the "cross-document" portion differs (§3.2, §3.4)

Treating this question as "memorize two fixed recipes," without being able to state that these three knobs can be combined independently, and without being able to state that resetting the position id for RoPE is not mathematically necessary, is the crux of this question.

</details>

<details>

<summary>Q13. What conditions must gradient accumulation satisfy to be equivalent to one big batch? Be clear about which tier of "equivalent" you mean</summary>

- Identical samples (the data consumed within the accumulation window must exactly match what the big batch would consume in one shot)
- Normalized over all valid tokens, not the mean of each micro-batch's mean averaged again
- Only one optimizer step executed; gradient clipping acts on the final gradient after accumulation completes, not applied separately per micro-batch
- This only guarantees **mathematical equivalence** (under exact real-number arithmetic, the gradient obtained via accumulation is the same formula as the gradient from a one-shot big batch). To go further, you also need: DDP by default **averages** gradients rather than summing them, so you must explicitly compensate with the global valid-token denominator, or it's equivalent to dividing by world size one extra time (Q16); dropout/RNG state and any batch-related running statistics must follow exactly the same random-number sequence under both settings for the comparison to even be meaningful; AMP's loss-scaler state and the scheduler's update count (each "big batch" should trigger exactly one scheduler.step, not one per micro-batch) must also be aligned
- Even with all of the above aligned, differences in the **order of floating-point reduction** (the order of summation across micro-batches, the reduction-tree structure of all-reduce across DP ranks) will still cause the last few significant digits to differ — this can only reach "mathematical equivalence," not unconditional **numerical bitwise equivalence** (the three-tier strength framework of §4.2 and §5.2 applies here as well)

Missing "clipping must be done once after accumulation," substituting the mean of means for sum-divided-by-count, or mistaking "mathematical equivalence" for "numerically bitwise identical," are the three most common pitfalls here.

</details>

<details>

<summary>Q14. How do you design a near-dedup (MinHash+LSH) pipeline? What's the LSH candidate-probability formula?</summary>

- Document → shingle set → MinHash signature (for each of multiple hash functions, take the minimum hash value over the document's shingles) → estimate Jaccard similarity
- LSH splits the signature into $b$ bands of $r$ rows each; under the idealized assumption that MinHash rows/hash functions are approximately independent, the candidate probability is $P=1-(1-s^r)^b$, doing only candidate recall
- Candidate pairs usually still need exact-Jaccard verification, and MinHash estimates set overlap, not semantic similarity (§2.3, [D01])

Treating LSH candidates directly as "confirmed near-duplicates," skipping the exact-Jaccard verification step, or ignoring that the candidate-probability formula itself relies on the idealized assumption of "approximately independent hash rows," are the pitfalls here.

</details>

<details>

<summary>Q15. Why must you "dedup before splitting"?</summary>

- If you split first and dedup second, near-duplicate content (mirror sites, rewritten reposts) may end up with one copy in train and one in val
- The model "memorizes" a near-verbatim copy of a validation sample during training, making validation loss artificially low, misleading early stopping and model selection
- Correct approach: first run near-dedup to get connected components (dedup clusters), then atomically assign whole clusters to a split; even if A and C were never directly compared, as long as they're transitively connected through B, they must still be in the same split (§2.5, [D01])

Only knowing "you need to dedup," without being able to state that "the order" is the real point of this question, and without being able to state the hidden requirement of transitive closure, are the pitfalls here.

</details>

<details>

<summary>Q16. What's the easiest pitfall in loss normalization?</summary>

- Correct formula: $\sum \mathrm{NLL}_i / \#\text{valid targets}$, normalized once over the global batch
- Common mistake: averaging within each micro-batch first, then averaging those averages — this introduces weighting bias whenever micro-batches have different numbers of valid tokens (padding/packing/variable length)
- There's also a variant mistake in multi-rank settings: DDP by default **averages** gradients; if the local code has already computed a normalized loss as "sum of NLL / this rank's valid-token count," and then all-reduce averages over rank count, that's equivalent to erroneously scaling the denominator by world size one extra time; the correct approach is either to compute only the "sum" locally and divide by the **global** valid-token total after all-reduce, or to explicitly switch to sum-reduce and control the normalization timing yourself (§3.5, §4.2)
- A correct implementation must separately accumulate "sum of NLL" and "count of valid tokens" across micro-batches/ranks, and divide only once at the end

Writing a "mean of means" implementation, or letting all-reduce's default averaging semantics clash with the local normalization logic, will silently distort the loss in variable-length-batch/multi-rank settings without raising an error — this is the most easily overlooked hazard in this question.

</details>

<details>

<summary>Q17. What are the supervision rules for EOS/BOS at packing boundaries?</summary>

- Keep: the loss for "last content token of a document → EOS" (teaches the model when to stop)
- Mask (per configuration): the loss for "previous document's EOS → next document's first token/BOS" (this label is just an artifact of concatenation)
- Both rules can be automatically produced by the same rule, "a label is valid iff the current position and the next position are in the same document" — no need to write special-cased logic for EOS (§3.2, [D04])

Assuming extra special-case logic is needed for EOS, without realizing it can be uniformly handled by the same doc-id boundary rule, is the pitfall here.

</details>

<details>

<summary>Q18. When PAD and EOS share the same token ID, what should you be careful of when constructing masks? Separate the two causal chains</summary>

- PAD=EOS is not necessarily wrong by itself; many tokenizers have no dedicated PAD token
- **If the loss mask is constructed via `target==pad_id`**: this would mistake the genuine content-token → EOS target for padding and mask it out, directly deleting the model's signal for learning "where to stop" — this is the most severe accident chain
- **If the key mask is constructed via `key==pad_id`**: the consequence is different — it wrongly hides real EOS positions as if they were padding keys (real queries can't read that EOS's context representation), but it doesn't automatically delete any loss, since the loss mask is computed separately by target position — this is an error of "hiding context," not "deleting loss"
- The correct, unified approach: construct the mask from each sequence's actual valid length `valid_len` (or an explicit padding flag), never by testing equality with a token ID (§3.3, [D05])

Only being able to recite the absolute conclusion "PAD=EOS is just wrong," without being able to state the different kinds of errors caused by constructing the loss mask vs. the key mask by ID, is the pitfall here.

</details>

### L3 High-Level Questions

<details>

<summary>Q19. Design a complete data pipeline from web crawling to training shards.</summary>

- The eleven steps in §2.1 present a **common, order-sensitive reference flow**, not the one fixed order — the real hard constraint is essentially just one: **dedup clusters must be formed before splitting** (cluster-level split depends on already-computed connected components, §2.5); privacy/safety processing and quality filtering are often moved earlier in engineering practice, or run in multiple passes across the pipeline — they don't have to be pinned to the fixed positions in the table
- After the raw-document stage there is a commonly missed **tokenized stage**: tokenizer materialization (actually running raw-document shards through the tokenizer to produce a token stream, §2.8), token-level mixture sampling (correcting the gap between document-level quotas and actual token share, §2.7), long-document chunking/truncation (§2.9), and the data-stream/sampler layer that decides how these tokenized shards are read during training (shard shuffle, domain-sampling RNG, DP-rank-disjoint sharding, worker/prefetch state, cursor position, §2.10)
- Every step must retain lineage: source/snapshot version/filter version/dedup cluster ID/split assignment/chunk assignment/final shard position (§2.1, §2.9)
- Benchmark decontamination targets evaluation sets and is a different matter from the train-val split (§2.4, §2.6)

Only memorizing the eleven steps' names and order, treating it as the one correct answer, without being able to state that "dedup before split" is the sole hard constraint, and missing the tokenizer/chunking/sampler pieces of the tokenized stage, is the deep water of this system-design question.

</details>

<details>

<summary>Q20. To achieve "deterministic resume," what state does the checkpoint need to save? What specific kind of "determinism" is meant?</summary>

- First distinguish which tier you need (§5.2): **basic resumability** only needs model weights + optimizer state (Adam's $m,v$) + LR schedule/consumed-token count; **identical sample sequence** additionally needs RNG state, data iterator position, shuffle epoch, sampler state, dataloader worker/prefetch state, and, for multi-rank training, each rank's checkpoint metadata
- The strongest tier, **numerically/bitwise identical trajectory**, additionally needs the AMP loss-scaler state, in-flight (incomplete-accumulation-window) pending gradients, deterministic kernels, and an identical software version/communication topology — even then, a world-size change usually still cannot guarantee a bitwise-identical trajectory (re-sharding semantics pointed to `distributed_training_tutorial.md`, which currently does not systematically cover this stronger guarantee of "still bitwise-reproducible after re-sharding")
- Missing the data iterator position or RNG state immediately falls out of the "identical sample sequence" tier; satisfying only the first two tiers while missing the AMP scaler/pending gradients/deterministic kernels lets you "resume training" with "identical order," but falls short of full numerical reproduction

Treating "deterministic resume" as a single black-or-white guarantee, without being able to state what necessary state each of the three strength tiers corresponds to, is the real crux of this question.

</details>

<details>

<summary>Q21. Pretraining suddenly has a loss spike or NaN — how do you triage in layers?</summary>

- This is a **recommended heuristic triage order** (arranged by "cheapest to check first"), not a universally mandatory procedure that must be strictly followed — if a specific project already has stronger priors (e.g., last time this kind of spike was a hardware issue), the order can absolutely be adjusted: bad batch/data shard (whether it's reproducible on the same batch, depending on deterministic sharding + lineage) → LR and resume state → gradients and activations (diagnostic quantities pointed to normalization_init_tutorial.md) → mixed-precision overflow/underflow → cross-rank anomalies (requires per-rank recorded statistics) → hardware failure
- Keep reproducible batch IDs/step numbers for after-the-fact replay-based localization
- The automated first line of defense: a rolling median/MAD detector, whose job is to flag "when triage should begin," not to determine the root cause (§5.4, [D07])

Jumping straight to guessing "it must be the learning rate is too high," skipping the cheapest and most appropriate first step of "can it be reproduced," or treating this triage order as a mandatory procedure that must be followed layer by layer without adjustment, are the pitfalls here.

</details>

<details>

<summary>Q22. Why report validation loss separately per domain?</summary>

- The aggregate loss is easily dominated by the domain with the highest token weight (e.g., web text being a large share), masking degradation in low-weight but important capabilities like code/math/low-resource languages/long context
- Per-domain monitoring is what allows problems to be caught while there's still time to adjust the mixture/checkpoint selection
- Checkpoint selection should also be based on the per-domain validation set/evaluation-set combination, not just training loss (§5.3, §5.5)

Looking at only one aggregate loss curve and concluding "the model trained fine," missing the local degradation that got diluted away.

</details>

<details>

<summary>Q23. For mixture sampling's "domain weight" vs. the token weight actually seen by the model — when are they equal, and when aren't they?</summary>

- Not always unequal — **if the configured domain weight is itself defined as a token-level target share** (i.e., the config literally specifies "what proportion of tokens this domain should contribute"), then the two can be equal in expectation; what actually decouples is the case where weights are configured **by document count/document weight** but then used directly as token weight — that's when a discrepancy arises from different domains having different average document lengths (§2.7)
- **Distinguish two different sampling units** (the main text can easily conflate these as the same thing):
  - **Document-level sampling**: doesn't distinguish domains — directly draws whole documents from a document pool with equal/weighted probability;
  - **Sample domain, then sample document**: first selects a domain by domain weight, then selects a document within that domain — this is a distinct two-stage process, and it's here that the first-stage "domain weight" has a clear meaning;
  together with "token-level sampling" (cutting chunks directly from the concatenated token stream, decoupled from document boundaries), these form three distinct sampling units
- Mitigation: separately measure and report the "actual token-level share" after sampling, rather than only reporting the configured value; this also connects to what $D$ means in the scaling law — resampling/residual near-duplicates make the "effective data volume" lower than the nominal $D$

Unconditionally asserting "domain weight is definitely not equal to token weight," or conflating "document-level sampling" with "sample domain, then sample document" as the same concept, are the two pitfalls most easily hit here.

</details>

<details>

<summary>Q24. What are the "three kinds of leakage" (train-val leakage / benchmark contamination / packed cross-document context) and how do you address each?</summary>

- **Train-val leakage**: splitting before deduping causes near-duplicate content to cross splits → **mitigated** with cluster-level split (§2.5) — but MinHash/LSH+n-gram detection itself can miss heavily rewritten paraphrases, cross-lingual translated versions, or semantically identical but surface-completely-different "answer variants"; cluster split should not be considered to have "solved" leakage, only to have significantly reduced the probability of leakage caused by surface-level duplication
- **Benchmark contamination**: training corpus contains eval-set content → **mitigated** with n-gram overlap decontamination detection (§2.4) — the same miss-detection problem is even more severe here (benchmarks are often manually rewritten to evade detection), and it can produce false positives on templated/boilerplate text (e.g., common code snippets, legal templates), wrongly flagging unrelated content as "contamination"
- **Packed cross-document historical context**: a plain causal mask lets a later document read an earlier document's history → **this is usually not data leakage — it's the sequence-semantics choice of "cross-sample context coupling"**; whether it needs to be cut off with a block-diagonal mask depends on whether the modeling assumption of "document independence" is required (§2.6, §3.4)

Claiming the first two kinds of leakage detection "completely solve" the problem, not knowing that n-gram methods themselves have clear miss-detection and false-positive boundaries, or treating the third kind as "leakage" too and unconditionally requiring it to be cut off, are the common incomplete answers to this question.

</details>

<details>

<summary>Q25. How are the compute budget and memory budget computed separately for an MoE model?</summary>

- Compute budget (the $N$ in $6ND$) is estimated with **active** parameters — the expert parameters that actually participate in that token's forward/backward computation; but this $6N_{active}D$ is itself still an approximation, excluding MoE-specific all-to-all communication overhead, load imbalance across experts, and the extra compute from capacity padding (tokens exceeding routing capacity need to be dropped or handled specially) — these details are pointed to `moe_tutorial.md`
- **Memory budget cannot simply be said to be "computed with total parameters"**: the model's **aggregate** checkpoint/model-state size does grow with total parameters (all expert weights need to be fully storable), but the memory actually occupied on a single card also depends on the sharding scheme such as EP (expert parallelism)/FSDP, parameter dtype, optimizer state (Adam's $m,v$ usually also needs to be computed at the sharded scale), gradients, and activation memory — for the same total parameter count, different sharding strategies can lead to very different per-card memory usage
- If there's a router aux loss in the total loss, its magnitude is usually much smaller than the main loss; the routing/load-balancing mechanism itself is pointed to moe_tutorial.md (§1.2, §3.5, §5.6)

Mixing up active and total parameters — using total parameters to estimate $6ND$ training compute significantly overestimates it; conversely, estimating per-card memory by simply dividing total parameters by card count, ignoring the sharding scheme and optimizer/activation memory, is another common oversimplification here.

</details>

<details>

<summary>Q26. Design a data lineage scheme that supports "auditing the provenance of a piece of training data six months later."</summary>

- Basic traceability chain: source dataset/URL → snapshot/crawl version → extraction and filtering pipeline version (including thresholds) → dedup cluster ID → split assignment → final shard file + offset — this is a **qualifying minimal lineage**, but not yet enough to support a genuinely rigorous audit
- Supporting a strong audit additionally requires recording: **sample/content hash** (a hash of the sample content itself, confirming that "the content now" exactly matches "the content when it was originally written"), **raw snapshot hash** (rather than just a version number), **tokenizer/normalizer hash** (confirming the specific artifact used at tokenization time, rather than just a version string), **code/container version** (the executable environment of the processing pipeline itself), **the random seed used for mixture sampling**, **shard checksum** (integrity verification of the whole shard file), **the precise document→segment/token mapping** (which token range a document corresponds to, especially with chunking involved, §2.9), and **license and takedown-request status**
- This more complete record simultaneously serves four scenarios: benchmark contamination investigation, loss-spike reproduction and localization, tracing the origin when a model's output looks suspicious, and responding to takedown requests in a compliance sense (§2.1, §2.8, §2.9, §5.4)

Storing just a "source URL + version number" and thinking that's enough, without being able to state that content hash, tokenizer hash, mixture seed, and document→token mapping are equally important pieces of intermediate state for "what this sample looks like now," and without being able to state the compliance dimension of license/takedown requests, is the crux of this question.

</details>

## §A Appendix: Sanity Checks

The from-scratch implementation in this tutorial should satisfy the following key invariants (pure Python standard library, no third-party dependencies, seconds on CPU; script at [`code/pretraining_pipeline.py`](code/pretraining_pipeline.py)):

1. **[D01] dedup**: exact Jaccard $J(A,B)=8/12=2/3$ ($A=\{1..10\}$, $B=\{1..8,11,12\}$); the MinHash estimate with a fixed seed + 1024 permutations falls within the tolerance set by sampling theory (exact equality is not asserted); LSH ($b=128,r=8$) candidate probability is far higher for a high-similarity pair ($s=2/3$) than a low-similarity pair ($s=0.05$); union-find verifies that the transitive closure of A~B, B~C places A and C into the same dedup cluster, which must be assigned the same split.
2. **[D02] packing shifted labels**: with `tokens=[11,12,13,21,22]`, `doc_ids=[0,0,0,1,1]`, the independent-document semantics label is `[12,13,IGNORE,22,IGNORE]`; the continuous-stream semantics label is `[12,13,21,22,IGNORE]` (the same token stream, two legitimate configurations giving different labels).
3. **[D03] block-diagonal mask + functional test**: `M[3,2]=False`, `M[4,0]=False`, `M[4,3]=True` (block mode); the same coordinates under continuous-stream mode give `M[3,2]=True`. Under $QK^\top=0$, when doc0's values change from 100 to 1000, block mode's document-1 output stays unchanged at $0$, while continuous-stream mode's output changes from $75$ to $750$.
4. **[D04] EOS boundary**: on `[11,12,13,EOS,21,22,EOS]` / `doc_ids=[0,0,0,0,1,1,1]`, the label for the last content token (13) → EOS is kept (=EOS), and the label for EOS → the next document's first token (21) is masked (IGNORE) — both are produced automatically by the same doc-id boundary rule.
5. **[D05] padding**: with `valid_len=3`, the loss mask is `[1,1,0,0,0]`; real queries cannot attend to padding keys; no attention row is ever empty (no all-`-inf` row causing softmax NaN).
6. **[D06] Chinchilla fitting**: from a log-spaced grid ($N\in[10^7,10^{12}]$, $D\in[10^8,10^{13}]$, 6 points each) with no noise, $E,A,B$ are recovered with error $<10^{-3}$ from the synthetic ground truth; the recovered $E$ is less than all finite observed losses; the analytic $N_*,D_*$ differ from the fine-grid search minimum by no more than one grid step; at 100x compute budget, $N_*$ grows by about $100^{0.452}\approx 8.0$x and $D_*$ by about $100^{0.548}\approx 12.5$x.
7. **[D07] spike monitoring**: in the sequence `[2.00,2.02,1.98,2.01,1.99,2.00,2.03,1.97,3.00]` only the last point is flagged; a smoothly declining sequence produces no false positives throughout; single-point jumps of the same magnitude are flagged as two distinct event types, `spike` (subsequent drop-back) vs. `sustained_shift` (subsequently stays elevated); NaN/Inf are unconditionally and immediately flagged as `non_finite`.

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
- **PaLM (includes loss spike / checkpoint restart practices)** — Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311 (2022).
- **Mixed-precision training** — Micikevicius et al. (NVIDIA/Baidu), *Mixed Precision Training*, arXiv 1710.03740 (2017), ICLR 2018.
- **The Pile (data pipeline design reference)** — Gao et al. (EleutherAI), *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, arXiv 2101.00027 (2020).
- **C4 corpus audit (extraction quality issues)** — Dodge et al., *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*, arXiv 2104.08758 (2021).
- **RefinedWeb (web-corpus extraction/filtering pipeline)** — Penedo et al., *The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*, arXiv 2306.01116 (2023).
- **FineWeb (large-scale web corpus + ablations)** — Penedo et al. (Hugging Face), *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, arXiv 2406.17557 (2024).
- **Training data dedup (near-duplicate effects on downstream models)** — Lee et al. (Google), *Deduplicating Training Data Makes Language Models Better*, arXiv 2107.06499 (2021), ACL 2022.
- **Original source of MinHash / Jaccard estimation** — Broder, *On the Resemblance and Containment of Documents*, Proceedings of Compression and Complexity of Sequences (1997) (no arXiv id).
- **Data mixture / domain reweighting** — Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*, arXiv 2305.10429 (2023), NeurIPS 2023.
- **Data-constrained scaling (effect of duplicate data on effective D)** — Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv 2305.16264 (2023), NeurIPS 2023.
- **Llama 2 (includes pretraining data/training recipe description)** — Touvron et al. (Meta), *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288 (2023).
- **Llama 3 (includes data pipeline/training recipe description)** — Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783 (2024).
- **DeepSeek-V3 (active/total parameters and MoE training practice)** — DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437 (2024).
- **C4 / T5 (classic reference for tokenizer and corpus preprocessing)** — Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, arXiv 1910.10683 (2019), JMLR 2020.
