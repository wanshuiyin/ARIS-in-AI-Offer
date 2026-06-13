"""
RAG + Embedding / Retrieval - minimal runnable implementation
=============================================================

Educational PyTorch reference for the contrastive-embedding + retrieval pieces
of a RAG pipeline. Standalone script: runs six sanity checks on CPU in seconds.

Pairs with: docs/tutorials/rag_embedding_retrieval_tutorial.md (concept reference).

What it demonstrates:
    DualEncoder        - toy bi-encoder, L2-normalized -> dot product = cosine
    info_nce_in_batch  - InfoNCE with in-batch negatives (diagonal = positive)
    info_nce_with_hard - InfoNCE with explicit hard negatives
    bm25_scores / rrf  - sparse retrieval + reciprocal-rank fusion
    mini-RAG retrieval - hybrid (dense + BM25) fused recall

    Sanity checks:
      [a] in-batch InfoNCE: with d_pos == q, each row's argmax is its own index
      [b] encoder output is L2-normalized (||v|| ~ 1), dot in [-1, 1]
      [c] temperature: smaller tau -> lower softmax entropy (sharper)
      [d] info_nce_with_hard: finite loss, no NaN, logits shape [B, B+2]
      [e] RRF is monotone: improving a doc's rank never lowers its RRF score
      [f] hybrid recall keeps an exact-entity chunk that pure dense may miss

Run:
    python rag_embedding.py
"""
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------- embedding side (contrastive) ----------

class DualEncoder(nn.Module):
    """Toy bi-encoder; real systems use a BERT-like tower. L2-normalized output."""

    def __init__(self, vocab, dim=128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")
        self.proj = nn.Linear(dim, dim)

    def encode(self, ids, offsets):
        return F.normalize(self.proj(self.emb(ids, offsets)), dim=-1)


def info_nce_in_batch(q, d_pos, tau=0.05):
    """q, d_pos: [B, dim] normalized. In-batch negatives; diagonal is the positive."""
    logits = (q @ d_pos.t()) / tau                    # [B, B]
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def info_nce_with_hard(q, d_pos, d_hard, tau=0.05):
    """Append each query's own hard negative; positive sits at column 0."""
    pos = (q * d_pos).sum(-1, keepdim=True)           # [B, 1]
    in_batch = q @ d_pos.t()                          # [B, B]
    in_batch.fill_diagonal_(float("-inf"))            # drop self-positive
    hard = (q * d_hard).sum(-1, keepdim=True)         # [B, 1]
    logits = torch.cat([pos, in_batch, hard], dim=1) / tau   # [B, B+2]
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels), logits


# ---------- retrieval side ----------

def toy_encode(text, dim=64):
    """Deterministic bag-of-words hashing encoder (no training needed)."""
    v = torch.zeros(dim)
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return F.normalize(v, dim=-1)


def bm25_scores(query, docs, k1=1.5, b=0.75):
    toks = [d.lower().split() for d in docs]
    avgdl = sum(len(t) for t in toks) / len(toks)
    N = len(docs)
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    out = []
    for d in toks:
        s, dl = 0.0, len(d)
        for w in set(query.lower().split()):
            if w not in df:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            f = d.count(w)
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        out.append(s)
    return out


def rrf(rank_lists, k=60):
    score = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=score.get, reverse=True)


def softmax_entropy(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * (p + 1e-12).log()).sum(-1).mean().item()


def main():
    B, dim = 6, 32

    # [a] in-batch InfoNCE: d_pos == q -> argmax on diagonal AND aligned loss is
    #     strictly lower than a misaligned (shuffled-positive) batch -> the CE/label
    #     wiring actually responds to alignment (not just the raw sim matrix).
    q = F.normalize(torch.randn(B, dim), dim=-1)
    d_pos = q.clone()
    logits = q @ d_pos.t()
    perm = torch.tensor([1, 2, 3, 4, 5, 0])            # deterministic shuffle
    loss_aligned = info_nce_in_batch(q, d_pos).item()
    loss_shuffled = info_nce_in_batch(q, d_pos[perm]).item()
    a_ok = bool((logits.argmax(1) == torch.arange(B)).all()) and loss_aligned < loss_shuffled
    print(f"[a] in-batch InfoNCE: argmax==idx, loss aligned {loss_aligned:.4f} < shuffled "
          f"{loss_shuffled:.4f}  {'OK' if a_ok else 'FAIL'}")
    assert a_ok

    # [b] the REAL DualEncoder output is L2-normalized; dot in [-1, 1]
    enc = DualEncoder(vocab=50, dim=dim)
    ids = torch.tensor([3, 7, 9, 2, 5, 1, 4, 8])       # flattened tokens
    offsets = torch.tensor([0, 3, 5])                  # 3 "documents"
    ev = enc.encode(ids, offsets)                      # [3, dim]
    b_ok = torch.allclose(ev.norm(dim=-1), torch.ones(ev.size(0)), atol=1e-5) \
        and logits.abs().max() <= 1 + 1e-5
    print(f"[b] DualEncoder ||v|| = {ev.norm(dim=-1).mean():.4f} (~1), max|dot| = "
          f"{logits.abs().max():.4f} (<=1)  {'OK' if b_ok else 'FAIL'}")
    assert b_ok

    # [c] temperature: smaller tau -> sharper softmax -> lower entropy
    row = (q @ d_pos.t())[0]
    e_sharp, e_soft = softmax_entropy(row / 0.05), softmax_entropy(row / 0.5)
    c_ok = e_sharp < e_soft
    print(f"[c] entropy(tau=0.05) = {e_sharp:.4f} < entropy(tau=0.5) = {e_soft:.4f}  "
          f"{'OK' if c_ok else 'FAIL'}")
    assert c_ok

    # [d] info_nce_with_hard: finite loss, no NaN, logits shape [B, B+2]
    d_hard = F.normalize(torch.randn(B, dim), dim=-1)
    loss_d, logits_d = info_nce_with_hard(q, d_pos, d_hard)
    d_ok = torch.isfinite(loss_d) and logits_d.shape == (B, B + 2)
    print(f"[d] with-hard loss = {loss_d.item():.4f} finite, logits {tuple(logits_d.shape)} == (B, B+2)  "
          f"{'OK' if d_ok else 'FAIL'}")
    assert d_ok

    # [e] RRF monotone: move doc 7 earlier in one list -> its RRF score must not drop
    base = [[3, 7, 1, 9], [5, 7, 2, 8]]
    better = [[7, 3, 1, 9], [5, 7, 2, 8]]            # doc 7 rank 2 -> 1 in list-1
    def rrf_score(lists, doc):
        return sum(1.0 / (60 + (lst.index(doc) + 1)) for lst in lists if doc in lst)
    # doc 7 is the only doc in BOTH lists -> rrf() must rank it first
    e_ok = rrf_score(better, 7) >= rrf_score(base, 7) and rrf(base)[0] == 7
    print(f"[e] RRF(doc7): base = {rrf_score(base,7):.5f} -> better = {rrf_score(better,7):.5f} "
          f"(non-decreasing); rrf top = {rrf(base)[0]}  {'OK' if e_ok else 'FAIL'}")
    assert e_ok

    # [f] hybrid recall keeps an exact-entity chunk pure dense may miss
    corpus = [
        "the cat sat on the warm mat in the sun",
        "device model Zephyrnaut9000 supports 128 channels at 9 ghz",
        "a quick brown fox jumps over the lazy dog near the river",
    ]
    query = "Zephyrnaut9000 channel spec"
    dv = torch.stack([toy_encode(t) for t in corpus])
    qv = toy_encode(query)
    dense_rank = (dv @ qv).topk(len(corpus)).indices.tolist()
    bm25_rank = sorted(range(len(corpus)), key=lambda i: bm25_scores(query, corpus)[i],
                       reverse=True)
    fused = rrf([dense_rank, bm25_rank])
    entity_idx = 1                                    # the Zephyrnaut9000 chunk
    f_ok = entity_idx == bm25_rank[0] and entity_idx in fused[:2]
    print(f"[f] BM25 top = {bm25_rank[0]} (entity chunk {entity_idx}); fused top2 = {fused[:2]}  "
          f"{'OK' if f_ok else 'FAIL'}")
    assert f_ok

    print("\nall RAG / embedding sanity checks passed ✓")


if __name__ == "__main__":
    main()
