"""
Linear / Sparse Attention - minimal runnable implementation
===========================================================

Educational PyTorch reference for linear attention, the chunkwise-parallel
bridge, the delta rule (DeltaNet), and trainable block-sparse attention.
Standalone script: runs six sanity checks on CPU in a few seconds.

Pairs with: docs/tutorials/linear_sparse_attention_tutorial.md (concept reference).

What it demonstrates:
    linear_attn_recurrent  - token-by-token causal linear attention, matrix state S in R^{d_k x d_v}
    linear_attn_chunkwise  - intra-chunk quadratic + inter-chunk O(1) state recurrence
    delta_rule_recurrent   - DeltaNet update S <- S(I - beta k k^T) + beta v k^T (overwrite)
    block_topk_attention   - keep softmax, attend only top-k key blocks (NSA / MoBA "select" idea)
    Sanity checks:
      [a] chunkwise == recurrent (the core equivalence) for chunk sizes C in {1, 4, 7, L}
      [b] chunk-size invariance: output is identical for every C (only the compute path changes)
      [c] linear-attn state is a [d_k, d_v] matrix independent of L (constant-memory, O(1) decode)
      [d] delta rule degenerates to additive linear attention when the erase term is dropped & beta=1
      [e] delta != additive for non-orthogonal keys (overwrite term is active);
          delta == additive for orthonormal keys & beta=1 (the precise sufficient condition)
      [f] block-sparse keeps exactly top-k blocks & softmax rows sum to 1,
          for BOTH divisible (L=12) and non-divisible (L=13) lengths

Run:
    python linear_sparse_attention.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def linear_attn_recurrent(Q, K, V):
    """Token-by-token causal linear attention. Q,K,V: [L, d] -> O: [L, d].

    State S: [d_k, d_v] matrix (here d_k = d_v = d), accumulated outer products
    sum_{j<=t} k_j v_j^T. Self-inclusive: S is updated before reading o_t.
    """
    L, d = Q.shape
    S = torch.zeros(d, d, dtype=Q.dtype)              # [d_k, d_v] matrix state
    O = torch.zeros(L, d, dtype=Q.dtype)
    for t in range(L):
        S = S + torch.outer(K[t], V[t])               # S += k_t v_t^T
        O[t] = Q[t] @ S                               # o_t = q_t^T S
    return O


def linear_attn_chunkwise(Q, K, V, C=4):
    """Chunkwise parallel form. Mathematically identical to the recurrent form.

    Per chunk: inter = Q_c @ S (history state, tokens strictly before this chunk),
    intra = ((Q_c K_c^T) * tril) @ V_c (in-chunk causal, diagonal inclusive),
    then update S += K_c^T V_c. Handles ragged last chunk (L not divisible by C).
    """
    L, d = Q.shape
    S = torch.zeros(d, d, dtype=Q.dtype)              # cross-chunk state [d_k, d_v]
    O = torch.zeros(L, d, dtype=Q.dtype)
    tri = torch.tril(torch.ones(C, C, dtype=Q.dtype))  # in-chunk causal mask, diagonal incl.
    for s in range(0, L, C):
        e = min(s + C, L)
        Qc, Kc, Vc = Q[s:e], K[s:e], V[s:e]           # [c, d]
        c = e - s
        m = tri[:c, :c]                               # ragged-safe slice
        inter = Qc @ S                                # history-state contribution [c, d]
        intra = ((Qc @ Kc.t()) * m) @ Vc             # in-chunk causal attention [c, d]
        O[s:e] = inter + intra
        S = S + Kc.t() @ Vc                           # update state [d_k, d_v]
    return O


def delta_rule_recurrent(K, V, beta, use_delta=True):
    """DeltaNet recurrence. K,V: [L, d] (k assumed unit-norm), beta: [L] -> states [L, d, d].

    S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T : the (I - beta k k^T) term
    first ERASES the old association in the k direction, then writes the new v k^T.
    use_delta=False with beta=1 falls back to additive linear attention S += v k^T.
    """
    L, d = K.shape
    S = torch.zeros(d, d, dtype=K.dtype)              # [d_k, d_v] matrix state
    I = torch.eye(d, dtype=K.dtype)
    states = []
    for t in range(L):
        k = K[t]                                      # [d]
        if use_delta:
            S = S @ (I - beta[t] * torch.outer(k, k))  # erase old association in k direction
        S = S + beta[t] * torch.outer(V[t], k)        # write new v k^T  (order: v k^T)
        states.append(S.clone())
    return torch.stack(states)                        # [L, d, d]


def block_topk_attention(Q, K, V, block=4, topk=2):
    """Top-k block-sparse attention: each query attends only its top-k key blocks
    (in-block softmax). Q,K,V: [L, d] -> out: [L, d], keep_blk: [L, nblk] bool.

    Block representative score = MEAN over the block's REAL key columns only (padding
    ignored), so L not divisible by `block` is handled correctly. The final attention
    still uses the original dense scores, restricted to the selected blocks.
    """
    L, d = Q.shape
    nblk = (L + block - 1) // block
    scores = (Q @ K.t()) / (d ** 0.5)                 # [Lq, Lk] full scores (select + final attn)
    pad = nblk * block - L
    # masked block mean: pad scores with 0, count real columns, divide by the count
    sc  = F.pad(scores, (0, pad), value=0.0).view(L, nblk, block)
    cnt = F.pad(torch.ones_like(scores), (0, pad)).view(L, nblk, block)
    blk_score = sc.sum(-1) / cnt.sum(-1).clamp(min=1)  # [Lq, nblk] mean over real cols only
    topk = min(topk, nblk)
    sel = blk_score.topk(topk, dim=-1).indices         # [Lq, topk]
    keep_blk = torch.zeros(L, nblk, dtype=torch.bool)
    keep_blk.scatter_(1, sel, True)                    # [Lq, nblk] True = keep this block
    keep = keep_blk.repeat_interleave(block, dim=1)[:, :L]   # token-level mask [Lq, Lk]
    masked = scores.masked_fill(~keep, float("-inf"))
    w = F.softmax(masked, dim=-1)                      # in-selected-block normalized, rows sum to 1
    return w @ V, keep_blk


def orthonormal_rows(n, d):
    """Return n orthonormal row vectors in R^d (n <= d), via QR of a random matrix."""
    q, _ = torch.linalg.qr(torch.randn(d, d))          # q: [d, d] orthonormal columns
    return q.t()[:n]                                   # [n, d] orthonormal rows


def main():
    L, d = 13, 6                                       # ragged: L not divisible by C below
    Q, K, V = torch.randn(L, d), torch.randn(L, d), torch.randn(L, d)
    ref = linear_attn_recurrent(Q, K, V)

    # [a] chunkwise == recurrent (core equivalence) across chunk sizes
    sizes = [1, 4, 7, L]
    max_diff = max((linear_attn_chunkwise(Q, K, V, C=c) - ref).abs().max().item() for c in sizes)
    a_ok = max_diff < 1e-5
    print(f"[a] chunkwise == recurrent, C in {sizes}: max |Δ| = {max_diff:.2e}  "
          f"{'OK' if a_ok else 'FAIL'}")
    assert a_ok

    # [b] chunk-size invariance: every C gives the SAME output (path changes, result doesn't)
    outs = [linear_attn_chunkwise(Q, K, V, C=c) for c in sizes]
    b_ok = all(torch.allclose(o, outs[0], atol=1e-5) for o in outs)
    print(f"[b] chunk-size invariance over C in {sizes}: all equal = {b_ok}  "
          f"{'OK' if b_ok else 'FAIL'}")
    assert b_ok

    # [c] linear-attn state is [d_k, d_v], independent of L (constant memory, O(1) decode)
    shapes = set()
    S = torch.zeros(d, d)
    for t in range(L):
        S = S + torch.outer(K[t], V[t])
        shapes.add(tuple(S.shape))
    c_ok = shapes == {(d, d)}
    print(f"[c] state shape over all {L} steps = {shapes} (expect {{({d}, {d})}}), L-independent  "
          f"{'OK' if c_ok else 'FAIL'}")
    assert c_ok

    # [d] delta rule -> additive linear attention when erase term dropped & beta=1
    Kn = F.normalize(torch.randn(L, d), dim=-1)
    beta1 = torch.ones(L)
    S_no_delta = delta_rule_recurrent(Kn, V, beta1, use_delta=False)[-1]   # additive
    S_additive = torch.zeros(d, d)
    for t in range(L):
        S_additive = S_additive + torch.outer(V[t], Kn[t])                 # plain S += v k^T
    d_ok = torch.allclose(S_no_delta, S_additive, atol=1e-6)
    print(f"[d] delta(use_delta=False, beta=1) == additive S+=vk^T: |Δ| = "
          f"{(S_no_delta - S_additive).abs().max():.2e}  {'OK' if d_ok else 'FAIL'}")
    assert d_ok

    # [e] overwrite term is real: delta != additive for non-orthogonal keys,
    #     but delta == additive for orthonormal keys & beta=1 (the precise condition)
    S_delta_rnd = delta_rule_recurrent(Kn, V, beta1, use_delta=True)[-1]
    diff_rnd = (S_delta_rnd - S_no_delta).norm().item()                    # expect > 0
    Ko = orthonormal_rows(d, d)                                            # L=d orthonormal keys
    Vo = torch.randn(d, d)
    be = torch.ones(d)
    S_delta_o = delta_rule_recurrent(Ko, Vo, be, use_delta=True)[-1]
    S_add_o = delta_rule_recurrent(Ko, Vo, be, use_delta=False)[-1]
    diff_orth = (S_delta_o - S_add_o).abs().max().item()                  # expect ~0
    e_ok = diff_rnd > 1e-3 and diff_orth < 1e-5
    print(f"[e] overwrite term: non-orthogonal ||Δ|| = {diff_rnd:.3e} (>0), "
          f"orthonormal |Δ| = {diff_orth:.2e} (~0)  {'OK' if e_ok else 'FAIL'}")
    assert e_ok

    # [f] block-sparse keeps exactly top-k blocks & softmax rows sum to 1,
    #     for both divisible (L=12) and non-divisible (L=13) lengths
    f_ok = True
    for Lf in (12, 13):
        Qf, Kf, Vf = torch.randn(Lf, d), torch.randn(Lf, d), torch.randn(Lf, d)
        out, keep_blk = block_topk_attention(Qf, Kf, Vf, block=4, topk=2)
        nblk = (Lf + 3) // 4
        kept = keep_blk.sum(dim=-1)                                        # blocks kept per query
        exact = int((kept == min(2, nblk)).all())
        rows_sum1 = torch.allclose((F.softmax(
            (Qf @ Kf.t() / d ** 0.5).masked_fill(
                ~keep_blk.repeat_interleave(4, 1)[:, :Lf], float("-inf")), dim=-1)).sum(-1),
            torch.ones(Lf), atol=1e-5)
        ok = bool(exact) and rows_sum1 and out.shape == (Lf, d)
        f_ok = f_ok and ok
        print(f"[f] L={Lf}: blocks kept/query = {kept.unique().tolist()} (expect [2]), "
              f"rows sum to 1 = {rows_sum1}  {'OK' if ok else 'FAIL'}")
    assert f_ok

    print("\nall linear / sparse attention sanity checks passed ✓")


if __name__ == "__main__":
    main()
