## §0 TL;DR Cheat Sheet

> 💡 **9 sentences to nail RAG + Embedding** — one page covering the interview essentials (see §2–§11 for derivations).

1. **What RAG is**: Retrieval-Augmented Generation = first **retrieve** relevant passages from an external knowledge base with the query, then stitch them into the prompt and let the LLM **generate** the answer. It addresses three things: stale knowledge, hallucination, private data (Lewis et al. 2020).

2. **Two complementary halves**: the left half "**train embeddings**" is contrastive learning (bi-encoder + InfoNCE + hard negatives), math-dense; the right half "**use embeddings**" is the RAG pipeline (chunk → retrieve → rerank → assemble prompt), systems-dense.

3. **The embedding main loss, InfoNCE**: $\mathcal{L} = -\log \frac{\exp(\text{sim}(q,d^+)/\tau)}{\sum_i \exp(\text{sim}(q,d_i)/\tau)}$ — essentially the softmax cross-entropy of "the positive vs a pile of negatives," with temperature $\tau$ tuning sharpness.

4. **Bi-encoder vs cross-encoder**: the bi-encoder encodes query/doc separately, is **precomputable**, fast, used for recall; the cross-encoder encodes them together, is **not precomputable**, slow but accurate, used to **rerank top-k**.

5. **Hard negatives**: pick "similar but irrelevant" docs from the retrieval top-k as negatives — more informative than random negatives; but beware the **false negative trap** — a picked "hard negative" may actually be relevant and become noise.

6. **Sparse vs dense vs hybrid**: BM25 (exact term matching, strong on rare words / entities) + dense vectors (semantic matching) complement each other, fused by rank with **RRF** (Reciprocal Rank Fusion): $\text{RRF}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$, $k\approx60$.

7. **Matryoshka representations**: one training run makes the **first $m$ dims** of the embedding also a good embedding, so deployment truncates dimensions on demand (save storage / speed up recall) via the multi-granularity loss $\sum_m \mathcal{L}(\text{emb}[:m])$.

8. **RAG vs long-context vs fine-tuning**: RAG injects **updatable external facts** and is traceable; long-context stuffs the full text but is expensive and suffers "lost in the middle"; fine-tuning changes **capability/style**, not great at injecting massive new facts. The three are often combined.

9. **Evaluation**: don't only look at generation — evaluate in layers: **retrieval quality** (recall@k / nDCG) + **generation faithfulness** (faithfulness / context relevance / answer relevance, e.g. RAGAS).

## §1 Intuition: why we need RAG

LLMs have three structural weaknesses: **knowledge cutoff** (they don't know post-training events), **hallucination** (they confidently make things up when they don't know), and **private data** (they haven't seen your company's docs). Fine-tuning to inject knowledge is expensive, slow, and goes stale.

**RAG (Retrieval-Augmented Generation) moves "knowledge" out of the parameters into an external knowledge base**: treat the weights as the "reasoning engine" and the vector store as "external memory you can update any time." When answering, first retrieve relevant passages, then let the LLM answer based on them — knowledge updates only need a store update (no retraining), answers can be traced (cite sources), and private data never enters the training set.

This chain splits naturally into **two complementary halves**:

- **Train embeddings (left half, math-dense)**: how to turn a sentence/passage into a vector such that "semantically close → vectors close." The core is **contrastive learning** (bi-encoder + InfoNCE + hard negatives).
- **Use embeddings (right half, systems-dense)**: how to build a usable pipeline from these vectors — chunking, indexing (ANN), recall, reranking, prompt assembly, evaluation.

> 💡 **One-sentence mental model** — An LLM is a "closed-book exam relying only on memory"; RAG is an "open-book exam: first use the query to find the relevant pages in the book (retrieve), then answer following those pages (generate)." How accurately you find = embedding + retrieval quality; how well you answer = rerank + prompt + LLM.

## §2 Text embeddings: bi-encoder + contrastive learning

### 2.1　Bi-encoder structure

Pass the query and the document each through an encoder (usually a shared or siamese BERT-like model), take a sentence vector (`[CLS]` or mean pooling), and L2-normalize:

$$\mathbf{q} = \text{normalize}(\text{Enc}(\text{query})), \qquad \mathbf{d} = \text{normalize}(\text{Enc}(\text{doc}))$$

Similarity is the dot product (which equals cosine after normalization): $\text{sim}(q,d) = \mathbf{q}^\top \mathbf{d}$.

Key advantage: **doc vectors can be precomputed offline and indexed**, so online you only encode the query once and do nearest-neighbor search. This is the root reason the bi-encoder scales to huge bases (contrast the cross-encoder in §6).

### 2.2　InfoNCE / contrastive loss (must-know, know the derivation)

Goal: make the query close to its positive doc $d^+$ and far from negatives $d^-$. Write it as "pick the positive out of a pile of candidates" — a softmax cross-entropy over the candidate set $\{d^+, d_1^-, \dots, d_{N}^-\}$:

$$\boxed{\;\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_{i} \exp(\text{sim}(q, d_i^-)/\tau)}\;}$$

Derivation view: this is just **cross-entropy** with "similarity $/\tau$" as logits and the positive index as the label. Minimizing it = maximizing $q$'s similarity to $d^+$ and suppressing similarity to all negatives (InfoNCE, van den Oord 2018; applied to retrieval it's DPR, Karpukhin 2020).

**Role of temperature $\tau$**: small $\tau$ → logits amplified → softmax sharper → the model "cares" more about the hardest negative (gradient leans toward the hardest negative; but when the positive already dominates, small $\tau$ can also saturate and shrink the gradient); large $\tau$ → smoother, more uniform negative weights. Too-small $\tau$ is sensitive to noise/false-negatives, too-large lacks discrimination — a key hyperparameter (commonly $0.01\sim0.07$).

### 2.3　In-batch negatives and hard negatives

**In-batch negatives**: a batch has $B$ pairs $(q_i, d_i^+)$; for $q_i$, **the other $B-1$ positives $d_{j\ne i}^+$ serve as negatives**. One forward pass yields $B-1$ negatives at almost zero extra cost (DPR's key trick). Bigger batch → more negatives → stronger contrast (large batch matters for contrastive learning).

**Hard negatives**: random negatives are often too easy (obviously irrelevant), with small gradients. Picking "lexically/semantically similar but actually irrelevant" docs from a BM25 / previous-model retrieval top-k is far more informative — a key to leaderboard gains.

> ⚠️ **False negative trap** — a "hard negative" mined from the top-k may **actually be relevant** (just unlabeled), and treating it as a negative gives a wrong gradient. Mitigation: drop candidates too similar to the positive (threshold filtering), use a cross-encoder to score and remove high-scoring ones, or distill soft labels. Hard negatives are a double-edged sword — harder is not always better.

## §3 The dual of hard negatives + Matryoshka representations

### 3.1　Cosine / dot product / normalization

After L2 normalization $\lVert \mathbf{q} \rVert = \lVert \mathbf{d} \rVert = 1$, so $\mathbf{q}^\top\mathbf{d} = \cos\theta \in [-1,1]$. The benefit of normalization: similarity looks only at **direction**, not magnitude — training is more stable (an absolute threshold still needs calibration per model/corpus). Most retrieval embeddings are normalized and use the dot product (= cosine).

### 3.2　Matryoshka Representation Learning (MRL)

A plain embedding is fixed-length (e.g. 768-dim); to save storage / speed up you can only reduce dimensions afterward (PCA), at a big loss. **Matryoshka (Kusupati 2022) gets "nested" embeddings from one training run**: the first $m$ dims ($m \in \{64,128,256,\dots,768\}$) are each a valid embedding. This is done by computing the loss at multiple granularities simultaneously:

$$\mathcal{L}_{\text{MRL}} = \sum_{m \in \mathcal{M}} w_m \, \mathcal{L}_{\text{InfoNCE}}\big(\mathbf{q}[:m],\ \mathbf{d}[:m]\big)$$

At deployment you truncate by budget: coarse ranking uses the first 64/128 dims (fast, cheap), fine ranking uses the full dims. OpenAI `text-embedding-3`, Nomic, BGE etc. all support adjustable MRL dimensions.

> 💡 **Why MRL works** — the multi-granularity loss forces the model to pack "the most important semantics" into the first few dims (like principal components ordered by importance), so a truncated prefix is still usable. Note: when used for cosine/dot retrieval, each prefix $\mathbf{q}[:m]$ should typically be **re-L2-normalized** (otherwise differing prefix norms change the logit scale). Cost: a more complex training objective and a slight possible drop in full-dim quality, traded for deployment flexibility.

## §4 From scratch: bi-encoder + InfoNCE training

Below is a runnable bi-encoder + in-batch InfoNCE training step (with a toy encoder; the focus is the contrastive loss and the hard-negative logic).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    """Toy bi-encoder: query/doc share one encoder; swap in a BERT-like model in practice."""

    def __init__(self, vocab, dim=128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")  # bag-of-words mean as sentence vector
        self.proj = nn.Linear(dim, dim)

    def encode(self, ids, offsets):
        h = self.proj(self.emb(ids, offsets))
        return F.normalize(h, dim=-1)                        # L2 normalize -> dot product = cosine


def info_nce_in_batch(q, d_pos, tau=0.05):
    """q, d_pos: [B, dim] normalized. Other positives in the batch serve as negatives."""
    logits = (q @ d_pos.t()) / tau          # [B, B], entry (i, i) is the positive
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)  # diagonal is positive -> multiclass cross-entropy


def info_nce_with_hard(q, d_pos, d_hard, tau=0.05):
    """Append each query's own hard negative d_hard: [B, dim]."""
    pos = (q * d_pos).sum(-1, keepdim=True)         # [B,1] positive similarity
    in_batch = q @ d_pos.t()                        # [B,B] in-batch negatives
    in_batch.fill_diagonal_(float("-inf"))          # drop self-positive to avoid double counting
    hard = (q * d_hard).sum(-1, keepdim=True)       # [B,1] hard-negative similarity
    logits = torch.cat([pos, in_batch, hard], dim=1) / tau   # positive at column 0
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)
```

Key points:

- The essence of `info_nce_in_batch`: the **diagonal of the similarity matrix** $q d^\top$ **is the positive**, so the label is just `arange(B)`, and one cross_entropy line does it.
- When adding hard negatives, put the positive logit at column 0 and set the label to all 0; `fill_diagonal_(-inf)` avoids the in-batch matrix counting the self-positive again.
- The smaller `tau` is, the more the logits are amplified, the more it leans toward the hardest negative.

## §5 Retrieval: sparse / dense / hybrid

### 5.1　Sparse retrieval: BM25

BM25 is the classic exact-term-matching baseline, still strong for **rare words, entity names, code/IDs** — queries that "must match literally":

$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\big(1 - b + b\,\frac{\lvert d\rvert}{\text{avgdl}}\big)}$$

where $f(t,d)$ is term frequency, $\lvert d\rvert$ is document length, $\text{avgdl}$ is average length, $k_1$ (≈1.2–2.0) controls term-frequency saturation, and $b$ (≈0.75) controls length normalization. **Term-frequency saturation**: appearing 10 times isn't 10× more important than once.

### 5.2　Dense retrieval + ANN (HNSW)

Dense = encode the query into a vector and find nearest neighbors in the vector store. Once the store is large you can't brute-force all of it — use **approximate nearest neighbor (ANN)**. The mainstream is **HNSW** (Malkov & Yashunin 2016): build a multi-layer "navigable small world" graph, with sparse long hops up top and dense fine search at the bottom, giving query cost **empirically near-logarithmic / sublinear** (not a strict guarantee; depends on efSearch, graph degree, recall target).

> ⚠️ **Dense is not a panacea** — dense vectors excel at **semantics/synonyms/paraphrase**, but for "exact entities, rare proper nouns, serial numbers" they may lose to BM25 (these words have no discriminative power in the semantic space). So industry rarely goes pure dense — mostly hybrid.

### 5.3　Hybrid retrieval + RRF

Fuse BM25's and dense's respective rankings with **Reciprocal Rank Fusion** (using only ranks, not scores, sidestepping scale alignment):

$$\boxed{\;\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)}\;}, \qquad k \approx 60$$

$\text{rank}_r(d)$ is document $d$'s rank in retriever $r$ (1-based). The constant $k$ damps the head ranks' over-dominance. RRF is simple, stable, training-free — the default fusion for hybrid retrieval (Cormack 2009).

> 💡 **learned sparse: SPLADE** — besides BM25 (terms) and dense (semantics) there's a middle path: **SPLADE** (Formal 2021) uses an MLM head to expand query/doc into **sparse term vectors with learned weights** (including synonyms/expansion terms) — it fits an inverted index (fast, interpretable) yet has semantic expansion, often a strong baseline or one leg of the hybrid.

## §6 Reranking: cross-encoder

Recall (bi-encoder + ANN) chases **high recall, fast**, but the precision of the top-k isn't enough. **A reranker re-scores the recalled top-k (say 100) precisely and takes the top few**.

| | Bi-encoder | Cross-encoder | ColBERT (late interaction) |
| --- | --- | --- | --- |
| Encoding | query, doc encoded **separately** | `[query; doc]` encoded **together** | doc stores **per-token** vectors |
| Interaction | only a final dot product (no token-level interaction) | full attention throughout (early, strong) | token-level late interaction (MaxSim) |
| Precomputable doc | ✅ (build an index) | ❌ (every pair recomputed) | ✅ (store token vectors, larger) |
| Speed / scale | fast / millions–billions | slow / rerank top-k only | mid / larger storage |
| Use | **recall** | **rerank** | recall+rerank compromise |

Cross-encoder: $\text{score}(q,d) = \text{MLP}(\text{BERT}([q;d])_{\texttt{[CLS]}})$ — the query's and doc's tokens attend to each other throughout, hence accurate; the cost is **it can't be precomputed** (the doc representation depends on the query), so it only runs over a small number of recalled candidates.

**ColBERT's late interaction**: the doc stores each token's vector, and scoring uses MaxSim — each query token finds the most similar token in the doc and sums:

$$\text{score}(q,d) = \sum_{i \in q} \max_{j \in d} \mathbf{E}_{q_i}^\top \mathbf{E}_{d_j}$$

Finer-grained than the bi-encoder's single vector, yet the doc side is still precomputable — a compromise between recall and reranking (Khattab & Zaharia 2020). The cost is **storage/latency**: each doc stores per-token vectors, far larger than a single vector; **ColBERTv2** (quantization + residual compression) and **PLAID** (centroid pruning for speed) are the key engineering that make multi-vector late interaction land at acceptable storage/latency (Santhanam 2021/2022). Single-vector vs multi-vector is the classic "save storage vs higher precision" tradeoff.

## §7 The RAG pipeline: from query to answer

```
query
  │  ① query rewrite / expand (HyDE / multi-query)
  ▼
[sparse BM25]  +  [dense ANN]
  │                │
  └──── RRF fuse ───┘   ② recall top-100
  ▼
cross-encoder rerank  ③ take top-5
  ▼
assemble prompt (context + question)  ④ with source labels
  ▼
LLM generate + cite   ⑤ optional self-check: Self-RAG(generation reflection) · CRAG(retrieval-quality → correction loop)
```

- **⓪ Chunking (offline)**: split documents into passages then embed (this is offline preprocessing, not shown in the online flow above). Too large → coarse retrieval, diluted relevant info; too small → broken context. Common is 200–500 token + overlap, or split by semantics/heading structure. **Chunking strategy hugely affects results** — one of the most worth-tuning knobs.
- **① Query rewrite**: **HyDE** (Gao 2022) first lets the LLM generate a **hypothetical document/passage** then retrieves with its embedding (aligning the query-vector distribution with real documents; in QA this pseudo-document often looks answer-like, but the core is using a pseudo-document to query the real corpus); multi-query generates several rewrites and merges recall.
- **④ Prompt assembly**: context passages + the original question + an instruction to "answer only from the context and give sources"; mind **lost-in-the-middle** (Liu 2023) — key passages placed in the middle are easily ignored, so put the important ones at the head and tail.
- **⑤ Self-checking RAG**: **Self-RAG** (Asai 2023) trains the model to emit reflection tokens ("should I retrieve / is the retrieved useful / is the answer grounded"); **CRAG** (Yan 2024) uses a lightweight evaluator to score the relevance/confidence of the **retrieved documents**, and on low confidence/ambiguity triggers corrective retrieval (web search / rewrite).

## §8 Advanced RAG: GraphRAG · comparison · evaluation

### 8.1　GraphRAG

Naive RAG recalls passages independently and is weak at "global questions spanning many documents" (e.g. "the thematic thread of the whole report"). **GraphRAG** (Microsoft, Edge 2024) first uses an LLM to extract the corpus into an **entity-relation knowledge graph + community summaries**, and global questions go through "community summaries" rather than scattered passages. Strong at query-focused summarization; the cost is expensive graph building.

### 8.2　RAG vs long-context vs fine-tuning

| | RAG | Long-context | Fine-tuning |
| --- | --- | --- | --- |
| What it injects | updatable **external facts** | the full text stuffed in this time | **capability / style / format** |
| Knowledge update | update the store, traceable | re-stuff each time, no persistence | retrain, easily stale |
| Cost | retrieval + short prompt | long prompt, expensive, slow | training expensive, inference cheap |
| Weakness | wrong retrieval → wrong answer | lost-in-the-middle | bad at injecting massive new facts |

In practice the three are often combined: fine-tune for style + RAG for facts + moderate long-context to hold the recalled passages.

### 8.3　Evaluation (don't only look at generation)

Evaluate in two layers:

- **Retrieval layer**: recall@k, nDCG, MRR — whether the recalled passages are right; pick an embedding/retriever using standard benchmarks like **MTEB** (general embedding leaderboard, Muennighoff 2022) and **BEIR** (zero-shot retrieval generalization, Thakur 2021), not just the end-to-end score.
- **Generation layer**: **faithfulness** (is the answer faithful to the context, no fabrication), **context relevance** (is the recall relevant), **answer relevance** (does the answer hit the point). RAGAS (Es 2023) computes these automatically with an LLM.

> ❌ **Classic faceplant** — only looking at "does the answer read correctly," without layering. The retrieval recalls a wrong passage, the LLM "faithfully" gives a wrong answer based on it, and end-to-end it looks like hallucination while the root cause is in retrieval. You must attribute retrieval and generation separately.

## §9 From scratch: a mini-RAG pipeline

```python
import re, math
import torch, torch.nn.functional as F

def chunk(text, size=40, overlap=10):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += size - overlap                      # sliding window with overlap, avoid broken context
    return out

def embed(texts, encoder):                       # encoder: str -> [dim] normalized vector
    return F.normalize(torch.stack([encoder(t) for t in texts]), dim=-1)

def dense_topk(qv, dv, k):                        # cosine top-k
    sims = dv @ qv                                # [N]
    return sims.topk(min(k, len(dv))).indices.tolist()

def bm25_scores(query, docs, k1=1.5, b=0.75):
    toks = [d.lower().split() for d in docs]
    avgdl = sum(len(t) for t in toks) / len(toks)
    N = len(docs)
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    scores = []
    for d in toks:
        s, dl = 0.0, len(d)
        for w in set(query.lower().split()):
            if w not in df:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            f = d.count(w)
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores

def rrf(rank_lists, k=60):                        # rank_lists: [[doc_idx by rank]...]
    score = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=score.get, reverse=True)

def mini_rag(query, corpus, encoder, top_recall=10, top_final=3):
    chunks = [c for doc in corpus for c in chunk(doc)]
    dv = embed(chunks, encoder)
    qv = F.normalize(encoder(query), dim=-1)
    dense_rank = dense_topk(qv, dv, top_recall)                      # dense ranks
    bm25 = bm25_scores(query, chunks)                               # compute once, avoid O(N^2)
    bm25_rank = sorted(range(len(chunks)), key=lambda i: bm25[i], reverse=True)[:top_recall]
    fused = rrf([dense_rank, bm25_rank])[:top_final]                # fuse + truncate
    context = "\n".join(f"[{r}] {chunks[i]}" for r, i in enumerate(fused, 1))
    return f"Answer based on the following passages (cite the source number):\n{context}\n\nQuestion: {query}"
```

This ties §5–§7 together: chunk → dense + BM25 dual recall → RRF fuse → assemble a source-labeled prompt (in practice add a cross-encoder rerank after fusion).

## §10 Engineering practice and common bugs

- **Chunk size is the number-one knob**: too large → coarse retrieval, too small → broken context; start with 256/512 token + 10–20% overlap, tune by evaluation.
- **Higher-value than tuning chunk size**: **Contextual Retrieval** (Anthropic 2024, prepend an LLM-generated whole-document context to each chunk before embedding) and **late chunking** (Jina 2024, run the full doc through a long-context encoder first, then chunk-pool) both target the root cause "plain chunking loses document-level context," often paying off more than just tuning chunk size.
- **Match the embedding model to the domain**: a generic embedding may be weak in specialized domains (medical/legal/code); consider domain fine-tuning or a matching model (BGE-M3 / E5 / domain models).
- **Don't go pure dense**: for entity/rare-word/exact-match scenarios always add BM25 hybrid, otherwise "you can't retrieve the one that's obviously there."
- **Hard-negative false negatives** (§2.3): dedup and filter when mining negatives, or you train it crooked.
- **Lost in the middle** (§7): put key passages at the head/tail of the context, not piled in the middle.
- **Don't skip reranking**: when recall is enough but precision isn't, a cross-encoder rerank of the top-k is often a high-ROI step (whether it is the highest depends on recall quality / latency budget / candidate scale).
- **The index goes stale**: after the store updates, incrementally re-embed / rebuild the index; a stale index = stale answers.
- **Evaluate in layers** (§8.3): attribute retrieval and generation separately, or you can't locate where the problem is.
- **Temperature $\tau$ and batch**: in contrastive learning a large batch (more in-batch negatives) + a suitable $\tau$ matter a lot; small batches perform poorly.

> ⚠️ **"Still hallucinating with RAG"** — usually not the LLM's fault: the retrieval recalled an irrelevant/wrong passage and the LLM "faithfully" answered from it. Check retrieval first (recall@k, are the passages right), then suspect generation.

## §11 Complexity and resources

| Stage | Compute | Note |
| --- | --- | --- |
| Embed docs (offline) | $O(N \cdot C_{\text{enc}})$ | one-off, batchable; MRL can save storage |
| Build ANN index (HNSW) | $O(N \log N)$ build | memory for query speed |
| Dense recall (online) | empirically near-log / sublinear per query | ANN approximate, not exact, not a strict guarantee |
| BM25 recall | inverted index $O(\lvert q\rvert \cdot \text{postings})$ | term-level |
| RRF fusion | $O(R \cdot k_{\text{recall}})$ | ranks only, dirt cheap |
| Cross-encoder rerank | $O(k_{\text{recall}} \cdot C_{\text{enc}})$ | one forward per candidate, **top-k only** |
| LLM generation | $O(L_{\text{ctx}}^2)$ prefill | longer context costs more (see the attention sheet) |

The bi-encoder's core dividend: doc encoding is $O(N)$ but **done offline once**, and online each query is just one encode + an ANN query; the cross-encoder can't be precomputed, so it's **only usable for top-k reranking** and must never scan the whole base.

## §12 25 high-frequency interview questions

Sorted into three tiers. Click to expand for answer points + pitfalls.

### L1 must-know (any role that has used RAG)

<details>

<summary>Q1. What is RAG? What problem does it solve?</summary>

- Retrieval-augmented generation: first retrieve relevant passages, then let the LLM generate based on them
- Mitigates: knowledge cutoff / hallucination / private data (mitigates factual hallucination + adds traceability; does not guarantee elimination)
- Knowledge updates only need a store update, traceable

Saying only "let the LLM look things up," without the "retrieve→assemble prompt→generate" three steps and the specific problems it addresses.

</details>

<details>

<summary>Q2. What are the basic steps of the RAG pipeline?</summary>

- Offline: chunk documents → embed → build index
- Online: embed query → recall top-k → (rerank) → assemble prompt → LLM generate
- Optional: query rewrite, source labeling, self-check

Missing chunking/reranking, or conflating recall and reranking.

</details>

<details>

<summary>Q3. Bi-encoder vs cross-encoder?</summary>

- Bi-encoder: query/doc encoded separately, doc vectors precomputable, fast, for recall
- Cross-encoder: encoded together, full attention, accurate but not precomputable, for reranking
- Bi-encoder for recall + cross-encoder for reranking is the standard pairing

Saying a cross-encoder can also index and scan the whole base (wrong, it's not precomputable).

</details>

<details>

<summary>Q4. Why do we need reranking?</summary>

- Recall chases high recall + fast, precision isn't enough
- The cross-encoder scores the top-k precisely, boosting precision
- Runs only over the few recalled candidates, cost controllable

Thinking recall is enough, not knowing reranking is often a high-ROI step.

</details>

<details>

<summary>Q5. What are sparse (BM25) and dense retrieval each strong at?</summary>

- BM25: exact term matching, strong on rare words / entities / serial numbers
- Dense: semantic / synonym / paraphrase matching
- Complementary, industry mostly uses hybrid

Saying dense fully dominates BM25 (wrong, for exact-match scenarios BM25 is often better).

</details>

<details>

<summary>Q6. What is chunking? How does chunk size affect results?</summary>

- Split documents into passages then embed
- Too large: coarse retrieval, diluted relevant info
- Too small: broken context
- Commonly 256/512 token + overlap

Treating chunking as an unimportant preprocessing step, not knowing it's the number-one tuning knob.

</details>

<details>

<summary>Q7. Why is cosine / normalized dot product commonly used?</summary>

- After L2 normalization, dot product = cosine
- Looks only at direction, not magnitude — more stable (absolute threshold still needs calibration)
- Most embeddings are normalized and use the dot product

Not knowing the normalized dot product is the cosine, or unable to state the benefit of normalization.

</details>

<details>

<summary>Q8. How does RAG differ from stuffing documents into a long context?</summary>

- RAG: retrieve relevant passages first, short prompt, traceable, store updatable
- Long-context: stuff the full text, expensive, slow, with lost-in-the-middle
- For large stores RAG is more economical/controllable (the full text also doesn't fit)

Thinking a long context removes the need for RAG (a large store won't fit and is expensive).

</details>

<details>

<summary>Q9. Why use approximate nearest neighbor (ANN) for the vector store?</summary>

- Brute-forcing all of a large store $O(N)$ is too slow
- HNSW and other ANN bring queries down to empirically near-logarithmic / sublinear (not a strict guarantee)
- Trade a little recall for huge speed

Saying vector retrieval is exact nearest neighbor (at large scale it's basically approximate).

</details>

<details>

<summary>Q10. Possible reasons for "still hallucinating with RAG"?</summary>

- Usually the retrieval recalled an irrelevant/wrong passage and the LLM "faithfully" answered from it
- Could also be chunks too fragmented, or the prompt didn't strongly constrain "answer only from context"
- Check retrieval quality first, then suspect generation

Blaming the LLM directly, without checking the retrieval layer.

</details>

### L2 advanced (research / deep-engineering)

<details>

<summary>Q11. Write the InfoNCE loss and explain temperature $\tau$.</summary>

- $\mathcal{L} = -\log \frac{\exp(\text{sim}(q,d^+)/\tau)}{\sum_i \exp(\text{sim}(q,d_i)/\tau)}$
- Essentially cross-entropy with similarity as logits and the positive as label
- Small $\tau$ → sharper softmax → cares more about the hardest negative; too small is noise-sensitive

Only writing the formula, unable to explain "it is softmax cross-entropy" and the sharpness role of $\tau$.

</details>

<details>

<summary>Q12. What are in-batch negatives? Why efficient?</summary>

- Each query in the batch uses the other samples' positives as negatives
- One forward yields $B-1$ negatives at almost zero extra cost
- The similarity matrix diagonal is the positive, label = arange(B)

Saying you need a separate forward for negatives (wrong, in-batch reuses the same batch).

</details>

<details>

<summary>Q13. Why are hard negatives important? What is the false-negative trap?</summary>

- Random negatives are too easy with small gradients; hard negatives (similar but irrelevant) are far more informative
- False negative: a mined "hard negative" may actually be relevant → wrong gradient
- Mitigation: threshold filtering, cross-encoder removal, soft-label distillation

Only saying "hard negatives are good," not knowing they can introduce false-negative noise.

</details>

<details>

<summary>Q14. How does RRF fuse multiple retrievers? Why use ranks not scores?</summary>

- $\text{RRF}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$, $k\approx60$
- Uses only ranks, avoiding aligning different retrievers' score scales
- Simple, stable, training-free

Trying to add BM25 scores and cosine scores directly (incomparable scales).

</details>

<details>

<summary>Q15. What does Matryoshka representation solve? How is it trained?</summary>

- One training run yields nested embeddings, the first $m$ dims are also a good embedding
- Multi-granularity loss $\sum_m \mathcal{L}(\text{emb}[:m])$
- Deployment truncates dimensions by budget (re-L2-normalize after truncation; save storage / speed up coarse ranking)

Thinking it's post-hoc PCA reduction (MRL packs the important semantics into the prefix during training).

</details>

<details>

<summary>Q16. What do $k_1$ and $b$ control in BM25?</summary>

- $k_1$: term-frequency saturation (diminishing marginal returns of repeated occurrences)
- $b$: document-length normalization strength ($b=1$ full, $b=0$ none)
- Typical $k_1\approx1.2$–$2$, $b\approx0.75$

Unable to articulate the two mechanisms of term-frequency saturation and length normalization.

</details>

<details>

<summary>Q17. How does ColBERT's late interaction relate to the bi-encoder and cross-encoder?</summary>

- Bi-encoder: single vector, interaction only at the final dot product (no token-level interaction), precomputable
- Cross-encoder: early, full attention, strongest, not precomputable
- ColBERT: token-level late interaction, stores token vectors, MaxSim $\sum_i\max_j E_{q_i}^\top E_{d_j}$, between the two, doc precomputable

Treating ColBERT as a plain bi-encoder, not knowing it's token-level MaxSim.

</details>

<details>

<summary>Q18. What is HyDE's idea? Why is it useful?</summary>

- First let the LLM generate a **hypothetical document/passage**, retrieve with its embedding
- Align the query-vector distribution with real documents (the pseudo-document looks more like a corpus document than the raw question)
- Zero-shot dense retrieval often improves as a result

Thinking HyDE rewrites query keywords (it actually generates a hypothetical document then embeds it).

</details>

<details>

<summary>Q19. How do you evaluate a RAG system?</summary>

- Retrieval layer: recall@k, nDCG, MRR
- Generation layer: faithfulness, context relevance, answer relevance (e.g. RAGAS)
- Layered attribution: first whether retrieval is right, then generation

Only looking at end-to-end answer correctness, not separating retrieval / generation.

</details>

<details>

<summary>Q20. What is lost-in-the-middle? How to mitigate?</summary>

- In a long context, key info placed in the middle is easily ignored (Liu 2023)
- Mitigation: after reranking, put the most relevant passages at head/tail, compress the context, reduce irrelevant passages
- Also a reason "more recall is always better" doesn't hold

Thinking a longer/fuller context is always better, ignoring that middle info is weakened.

</details>

### L3 top-lab questions (deep end)

<details>

<summary>Q21. Why can't a cross-encoder scan the whole base? Where does complexity differ?</summary>

- The cross-encoder's doc representation depends on the query, so it **can't be precomputed offline**
- Every (query, doc) pair needs a full forward, the whole base = $O(N \cdot C_{\text{enc}})$/query
- The bi-encoder encodes docs offline, online only an ANN query (empirically near-log / sublinear); so the cross-encoder is only for reranking top-k

Saying a cross-encoder can also index (its representation is coupled to the query, you can't build a static index).

</details>

<details>

<summary>Q22. Why does batch size matter so much in contrastive learning? Relation to negatives?</summary>

- in-batch negatives = batch − 1, bigger batch → more negatives → stronger contrast
- a large batch makes InfoNCE's denominator cover more negatives, a more accurate estimate
- small batch has few negatives and overfits easily; can compensate with cross-batch memory / hard negatives

Not knowing batch size directly determines the number of in-batch negatives.

</details>

<details>

<summary>Q23. How is GraphRAG stronger than naive RAG? At what cost?</summary>

- Naive RAG recalls passages independently, weak at "cross-document global questions"
- GraphRAG uses an LLM to extract an entity-relation graph + community summaries, global questions go through the summaries
- Cost: high offline LLM cost of graph/summary building, complex to maintain

Treating GraphRAG as "swapping the vector store," not knowing it builds a knowledge graph + community summaries.

</details>

<details>

<summary>Q24. What do "self-checking RAG" methods like Self-RAG / CRAG do?</summary>

- Self-RAG: trains the model to emit reflection tokens (should I retrieve / is the passage useful / is the answer grounded)
- CRAG: at the **retrieval layer**, an evaluator scores the relevance/confidence of the retrieved docs (you can't measure true recall at inference without a labeled set), and on low confidence triggers corrective retrieval (web search, rewrite)
- Goal: keep RAG from blindly answering when retrieval is useless/wrong

Thinking they're better retrievers; actually they add a self-evaluation/correction loop — Self-RAG focuses on generation-time reflection, CRAG focuses on evaluating retrieval quality and triggering corrective retrieval.

</details>

<details>

<summary>Q25. When to choose RAG, when fine-tuning, when long-context for the same knowledge?</summary>

- Inject large amounts of **updatable external facts** + need traceability → RAG
- Change **capability/style/format/output structure** → fine-tuning
- A few documents at once, needing global reasoning and they fit → long-context
- In practice often combined: fine-tune for style + RAG for facts + long-context to hold the recall

Absolutely saying "RAG fully beats fine-tuning" or vice versa, ignoring the essential split of "inject facts vs change capability."

</details>

## §A Appendix: sanity check

Key invariants of `info_nce_in_batch` and mini-RAG (verifiable with a script):

- **Diagonal is the positive**: the diagonal of the similarity matrix $qd^\top$ should be each row's max (after training); `cross_entropy(logits, arange(B))` converging means alignment.
- **Normalization**: encoder output $\lVert\mathbf{q}\rVert\approx1$, dot product in $[-1,1]$.
- **Temperature direction**: with fixed logits, lowering $\tau$ makes the softmax probabilities more concentrated (lower entropy).
- **RRF monotone**: if a document moves earlier in any retriever's rank, its RRF score should not drop.
- **Hybrid recall ⊇ exact hits**: a precise-entity passage that BM25 can hit should be kept in the hybrid result (pure dense may miss it).

Below is the **real run** output of [`code/rag_embedding.py`](code/rag_embedding.py) on **PyTorch 2.10 / CPU** (each line has an `assert`; the summary prints only if all pass):

```
[a] in-batch InfoNCE: argmax==idx, loss aligned 0.0000 < shuffled 22.9564  OK
[b] DualEncoder ||v|| = 1.0000 (~1), max|dot| = 1.0000 (<=1)  OK
[c] entropy(tau=0.05) = 0.0000 < entropy(tau=0.5) = 1.2945  OK
[d] with-hard loss = 0.0000 finite, logits (6, 8) == (B, B+2)  OK
[e] RRF(doc7): base = 0.03226 -> better = 0.03252 (non-decreasing); rrf top = 7  OK
[f] BM25 top = 1 (entity chunk 1); fused top2 = [1, 2]  OK

all RAG / embedding sanity checks passed ✓
```

Here [a] shows the InfoNCE loss is ≈0 when aligned and jumps to ~23 when the positives are shuffled, proving the loss really responds to "alignment" (not just that a diagonal matrix was constructed); [c] under the same logits the softmax entropy at $\tau=0.05$ (≈0) is far below that at $\tau=0.5$ (≈1.29), verifying a smaller temperature is sharper; [f] the passage with the rare entity `Zephyrnaut9000` is **ranked #1 by BM25** and kept in the hybrid (RRF) top-2, demonstrating hybrid recall's robustness to exact entities (§5.3; note: in this toy case dense happens to recall it too — in real scenarios dense more easily misses rare entities / serial numbers).

---

## 📜 Runnable Code

The embedding / retrieval core of this tutorial has a minimal runnable version at [`docs/tutorials/code/rag_embedding.py`](code/rag_embedding.py):

- [`rag_embedding.py`](code/rag_embedding.py) — `DualEncoder` (L2-normalized bi-encoder) + `info_nce_in_batch` / `info_nce_with_hard` (in-batch + hard-negative contrastive loss) + `bm25_scores` / `rrf` / hybrid recall, with 6 `assert` sanity checks (diagonal-is-positive / normalization / temperature sharpness / with-hard shape / RRF monotone / hybrid keeps the entity chunk).

Pure PyTorch, runs on CPU in seconds with no GPU: `python docs/tutorials/code/rag_embedding.py`. The §A output above is this script's real run result.

---

## 📚 References

- **RAG** — Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv 2005.11401 (2020), NeurIPS 2020.
- **DPR** — Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, arXiv 2004.04906 (2020), EMNLP 2020.
- **InfoNCE / CPC** — van den Oord et al., *Representation Learning with Contrastive Predictive Coding*, arXiv 1807.03748 (2018).
- **SimCSE** — Gao et al., *SimCSE: Simple Contrastive Learning of Sentence Embeddings*, arXiv 2104.08821 (2021), EMNLP 2021.
- **ColBERT** — Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, arXiv 2004.12832 (2020), SIGIR 2020.
- **Matryoshka Representation Learning** — Kusupati et al., arXiv 2205.13147 (2022), NeurIPS 2022.
- **BGE-M3** — Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*, arXiv 2402.03216 (2024).
- **HyDE** — Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, arXiv 2212.10496 (2022), ACL 2023.
- **Self-RAG** — Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, arXiv 2310.11511 (2023), ICLR 2024.
- **CRAG** — Yan et al., *Corrective Retrieval Augmented Generation*, arXiv 2401.15884 (2024).
- **GraphRAG** — Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv 2404.16130 (2024).
- **RAGAS** — Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv 2309.15217 (2023).
- **RRF** — Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009.
- **HNSW** — Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, arXiv 1603.09320 (2016), IEEE TPAMI 2020.
- **Lost in the Middle** — Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv 2307.03172 (2023), TACL 2024.
- **BM25** — Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in IR (2009).
- **SPLADE** — Formal et al., *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*, arXiv 2107.05720 (2021), SIGIR 2021.
- **ColBERTv2** — Santhanam et al., *ColBERTv2: Efficient and Effective Retrieval via Lightweight Late Interaction*, arXiv 2112.01488 (2021), NAACL 2022.
- **PLAID** — Santhanam et al., *PLAID: An Efficient Engine for Late Interaction Retrieval*, arXiv 2205.09707 (2022), CIKM 2022.
- **Contextual Retrieval** — Anthropic, *Introducing Contextual Retrieval* (engineering blog, 2024).
- **Late Chunking** — Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*, arXiv 2409.04701 (2024).
- **BEIR** — Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, arXiv 2104.08663 (2021), NeurIPS 2021 D&B.
- **MTEB** — Muennighoff et al., *MTEB: Massive Text Embedding Benchmark*, arXiv 2210.07316 (2022), EACL 2023.
