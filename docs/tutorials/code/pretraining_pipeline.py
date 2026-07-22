"""
LLM Pretraining Pipeline - minimal runnable implementation
===========================================================

Pure-stdlib (Python 3.9+, no torch/numpy) sanity checks, runs in <1s on CPU.
Pairs with: docs/tutorials/llm_pretraining_pipeline_tutorial.md (concept reference).

    [D01] exact Jaccard + MinHash estimate + REAL b x r LSH banding (candidate
          generation) + exact-Jaccard-thresholded union-find dedup-cluster
          split (near-duplicate documents must share a split via TRANSITIVE
          closure, even for pairs the banding step never directly compares)
    [D02] document packing: shifted labels under independent-doc vs
          continuous-stream semantics (same causal-mask family, different
          label/attention scoping)
    [D03] block-diagonal causal mask vs continuous causal mask + a
          FUNCTIONAL test over ALL of one document's query positions (swap
          another document's values, check this document's output is
          unaffected under block masking -- and the reverse direction too)
    [D04] EOS boundary supervision: content-end -> EOS kept, EOS -> next
          document's first token masked -- both fall out of the SAME
          doc-id-boundary rule used in [D02], no special-casing needed
    [D05] padding: combined key-padding + causal mask, loss mask for padding
          targets (target validity decided by POSITION, i+1 < valid_len --
          NEVER by comparing a token id to pad_id, since PAD may legitimately
          equal EOS), and a check that no attention row is completely empty
          (which would produce NaN after softmax)
    [D06] synthetic Chinchilla-style curve fit: recover E, A, B by linear
          least squares (alpha, beta held fixed) from a noiseless log-spaced
          grid, then feed the FITTED A_hat/B_hat (not the ground truth) into
          the analytic compute-optimal N*, D* solver and check it against a
          grid search and against the 100x compute-scaling ratios quoted in
          the tutorial. This is a FITTING-TO-DECISION UNIT TEST for the
          optimization code, not empirical validation of the Chinchilla law.
    [D07] loss-spike / NaN monitor: causal rolling median/MAD robust
          z-score, NaN/Inf immediate flag, spike-vs-sustained-shift event
          classification, and a check that a smooth declining loss curve
          does not false-positive.

Boolean attention-mask polarity convention used throughout this file:
    True = ALLOWED (this key position MAY be attended). This is the
    OPPOSITE of PyTorch's common key_padding_mask convention (True usually
    means MASKED OUT / excluded) -- invert before feeding a mask from this
    file into an API that expects that convention.

Run:
    python3 pretraining_pipeline.py
"""
import math
import statistics
import sys

IGNORE = -100  # PyTorch's default cross-entropy ignore_index convention


# =====================================================================
# [D01] Corpus factory: exact Jaccard, MinHash, LSH banding, cluster split
# =====================================================================
# Teaching simplification: documents are represented directly as sets of
# shingle IDs (small ints). A real pipeline hashes n-gram shingles of the
# normalized text into this same integer space before running MinHash.

def exact_jaccard(a, b):
    """J(A,B) = |A n B| / |A u B|. This is a SET-OVERLAP statistic on
    shingles, not a semantic-similarity measure (tutorial CORRECTIONS)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def minhash_signature(s, num_perm, seed):
    """MinHash signature via num_perm random affine hashes h(x) = (a*x+b) mod p.
    Elements are ints, so Python's hash(int) == int (no PYTHONHASHSEED noise);
    a fixed `seed` makes the whole signature deterministic and reproducible.

    Raises ValueError on a degenerate call instead of returning a signature
    that would silently misbehave downstream: `s` must be non-empty (min()
    over an empty hash set is undefined), and `num_perm` must be positive
    (num_perm == 0 would produce a length-0 signature that later divides by
    zero in minhash_estimate)."""
    if num_perm <= 0:
        raise ValueError(f"num_perm must be a positive integer, got {num_perm}")
    if not s:
        raise ValueError("cannot compute a MinHash signature of an empty document/set")
    p = (1 << 61) - 1  # Mersenne prime, standard MinHash modulus
    rng_a, rng_b = [], []
    rnd = _SplitMix(seed)
    for _ in range(num_perm):
        rng_a.append(1 + rnd.next() % (p - 1))
        rng_b.append(rnd.next() % p)
    sig = [min((a_i * x + b_i) % p for x in s) for a_i, b_i in zip(rng_a, rng_b)]
    return sig


class _SplitMix:
    """Tiny deterministic PRNG (splitmix64-style) so this file has zero
    dependency on Python's `random` module internals across versions."""
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)


def minhash_estimate(sig1, sig2):
    """Estimated Jaccard = fraction of matching signature positions. Both
    signatures MUST come from the same hash-function family (same seed,
    same num_perm) and therefore must be the same nonzero length -- this is
    checked explicitly rather than silently truncated by zip(), which would
    otherwise produce a wrong and ASYMMETRIC estimate (the shorter signature
    would dominate the sample) whenever the two lengths differ."""
    if len(sig1) != len(sig2):
        raise ValueError(f"signature length mismatch: {len(sig1)} vs {len(sig2)}")
    if len(sig1) == 0:
        raise ValueError("cannot estimate Jaccard from empty signatures (num_perm must be >= 1)")
    matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
    return matches / len(sig1)


def lsh_candidate_probability(s, b, r):
    """P(candidate pair) under b-band/r-row LSH: 1 - (1 - s^r)^b. This is
    the THEORETICAL closed-form probability, assuming each MinHash row is
    an independent hash draw -- a separate check from lsh_candidate_pairs
    below, which does the actual banding on real signatures."""
    return 1.0 - (1.0 - s ** r) ** b


def lsh_bands(sig, b, r):
    """Slice a MinHash signature into b bands of r rows each, for REAL LSH
    banding (as opposed to just the theoretical probability formula above).
    Requires len(sig) == b * r -- a signature that doesn't divide evenly
    into (band, row) tuples cannot be banded and must fail loudly rather
    than silently drop or misalign rows."""
    if len(sig) != b * r:
        raise ValueError(f"signature length {len(sig)} != b*r ({b}*{r}={b * r})")
    return [tuple(sig[i * r:(i + 1) * r]) for i in range(b)]


def lsh_candidate_pairs(signatures, b, r):
    """Real LSH banding over a list of signatures: hash each document's
    bands into buckets keyed by (band_index, band_tuple); any two documents
    that land in the SAME bucket for at least one band are candidate pairs.
    This is the actual banding step -- candidates from here still need an
    exact-Jaccard confirmation before being trusted as near-duplicates."""
    buckets = {}
    for doc_idx, sig in enumerate(signatures):
        for band_idx, band in enumerate(lsh_bands(sig, b, r)):
            buckets.setdefault((band_idx, band), []).append(doc_idx)
    candidates = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add((min(members[i], members[j]), max(members[i], members[j])))
    return candidates


class UnionFind:
    """Minimal union-find for the dedup-cluster-split test: near-duplicate
    edges must propagate to a shared split ID via TRANSITIVE closure, even
    for pairs that were never directly compared."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


# =====================================================================
# [D02]/[D04] Document packing: doc-boundary label masking
# =====================================================================

def validate_doc_ids(doc_ids):
    """Precondition shared by shifted_labels_independent_docs and
    block_diagonal_causal_mask: each logical document's id must occupy ONE
    CONTIGUOUS run of positions. Reusing an id after a different id has
    already appeared (e.g. [0, 1, 0]) is NOT a second, later document --
    it silently merges the reused id's positions into a single doc-mask
    block, letting the final run wrongly attend the FIRST run too. Reject
    that here instead of producing a silently-wrong mask or label set."""
    seen = set()
    prev = object()  # sentinel, never equal to a real doc id
    for d in doc_ids:
        if d != prev:
            if d in seen:
                raise ValueError(
                    f"doc_ids reuse id {d!r} non-contiguously (each logical document "
                    f"must occupy one contiguous run of positions): {doc_ids}"
                )
            seen.add(d)
            prev = d


def shifted_labels_independent_docs(tokens, doc_ids):
    """Teacher-forcing labels under INDEPENDENT-DOCUMENT semantics: label[i]
    is valid (tokens[i+1]) only if i+1 is inside the SAME document; otherwise
    IGNORE. Feeding an EOS-terminated doc list through this one rule
    automatically yields BOTH: (a) content-end -> EOS kept valid (EOS shares
    its own doc's id), and (b) EOS -> next-doc-first-token masked (the id
    changes at that boundary) -- no separate EOS special-case is needed.

    Preconditions (validated, not assumed): `tokens` and `doc_ids` must be
    the same length, and `doc_ids` must satisfy validate_doc_ids above."""
    if len(tokens) != len(doc_ids):
        raise ValueError(
            f"tokens and doc_ids must be the same length: {len(tokens)} != {len(doc_ids)}"
        )
    validate_doc_ids(doc_ids)
    n = len(tokens)
    labels = []
    for i in range(n):
        if i + 1 < n and doc_ids[i] == doc_ids[i + 1]:
            labels.append(tokens[i + 1])
        else:
            labels.append(IGNORE)
    return labels


def shifted_labels_continuous(tokens):
    """Plain next-token labels, no doc-boundary masking at all: this is the
    CONTINUOUS EOS-delimited stream semantics -- a legitimate config choice,
    not a bug (tutorial CORRECTIONS)."""
    n = len(tokens)
    return [tokens[i + 1] if i + 1 < n else IGNORE for i in range(n)]


def block_diagonal_causal_mask(doc_ids):
    """allowed[q][k] = (k <= q) AND (doc_ids[q] == doc_ids[k]).

    Polarity: True = key k is ALLOWED for query q (see module docstring;
    opposite of the common PyTorch key_padding_mask convention).

    Precondition: doc_ids must assign each logical document one contiguous
    run of positions -- validated below rather than silently trusted."""
    validate_doc_ids(doc_ids)
    n = len(doc_ids)
    return [[(k <= q) and (doc_ids[q] == doc_ids[k]) for k in range(n)] for q in range(n)]


def continuous_causal_mask(n):
    """M[q,k] = (k <= q) only -- ordinary causal mask, no doc restriction.
    Polarity: True = ALLOWED (may attend), the same convention as
    block_diagonal_causal_mask above."""
    return [[k <= q for k in range(n)] for q in range(n)]


def masked_uniform_attention_output(mask_row, values):
    """QK^T = 0 -> softmax over the ALLOWED keys is uniform. This makes the
    output a plain average of `values` at allowed positions -- a stronger,
    numeric functional test than just checking mask shapes. `mask_row`
    follows this file's ALLOWED=True polarity convention."""
    allowed = [k for k, m in enumerate(mask_row) if m]
    return sum(values[k] for k in allowed) / len(allowed)


def key_padding_and_loss_mask(tokens, valid_len):
    """Combined key-padding + causal attention mask, and a separate loss
    mask that excludes (a) padding query positions and (b) real positions
    whose TARGET falls in the padding region. Two independent maskings --
    doing only one is incomplete (tutorial CORRECTIONS).

    Target validity is decided PURELY BY POSITION (i + 1 < valid_len),
    never by comparing tokens[i + 1] to a pad id. Checking by token VALUE
    is exactly the bug this function must avoid: if PAD and EOS ever share
    the same id (a realistic tokenizer choice), a value-based check would
    silently mask out a real end-of-document EOS target that sits INSIDE
    the valid region -- see the PAD==EOS adversarial check in [D05] below.

    Polarity: the returned attention mask uses True = ALLOWED (may attend),
    the opposite of PyTorch's common key_padding_mask convention.

    Precondition: 0 <= valid_len <= len(tokens), validated below."""
    if not (0 <= valid_len <= len(tokens)):
        raise ValueError(f"valid_len={valid_len} out of range for {len(tokens)} tokens")
    n = len(tokens)
    allowed_attention_mask = [[(k <= q) and (k < valid_len) for k in range(n)] for q in range(n)]
    loss_mask = []
    for i in range(n):
        if i >= valid_len:
            loss_mask.append(0)          # padding query itself
        elif i + 1 >= valid_len:
            loss_mask.append(0)          # target position falls in the padding region
        else:
            loss_mask.append(1)
    return allowed_attention_mask, loss_mask


# =====================================================================
# [D06] Synthetic Chinchilla-style curve fit (fitting unit test)
# =====================================================================

def loss_surface(n, d, e, a, b, alpha, beta):
    return e + a * n ** (-alpha) + b * d ** (-beta)


def solve_linear_3x3(mat, vec):
    """Gaussian elimination with partial pivoting for a 3x3 linear system.

    Raises ValueError if the design matrix is singular or numerically
    ill-conditioned (near-zero pivot magnitude after row-swap partial
    pivoting), instead of silently dividing by ~0 and returning garbage
    coefficients for degenerate/collinear design points."""
    m = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        if abs(pv) < 1e-12:
            raise ValueError(
                f"solve_linear_3x3: singular or ill-conditioned design matrix "
                f"(pivot magnitude {abs(pv):.3e} at column {col}) -- check for "
                f"duplicated or collinear (n, d) fit points"
            )
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r != col:
                factor = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_e_a_b(points, alpha, beta):
    """Ordinary least squares for L = E*1 + A*N^-alpha + B*D^-beta, with
    alpha/beta held FIXED (this makes the model linear in E, A, B)."""
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for n, d, loss in points:
        feats = [1.0, n ** (-alpha), d ** (-beta)]
        for i in range(3):
            xty[i] += feats[i] * loss
            for j in range(3):
                xtx[i][j] += feats[i] * feats[j]
    return solve_linear_3x3(xtx, xty)  # [E_hat, A_hat, B_hat]


def analytic_optimum(c_budget, k, alpha, beta, a, b):
    """N*, D* minimizing L(N,D) subject to C = k*N*D."""
    a_exp = beta / (alpha + beta)
    b_exp = alpha / (alpha + beta)
    g = (alpha * a / (beta * b)) ** (1.0 / (alpha + beta))
    n_star = g * (c_budget / k) ** a_exp
    d_star = (c_budget / k) / n_star
    return n_star, d_star


def logspace(start, stop, num):
    if num == 1:
        return [start]
    step = (math.log10(stop) - math.log10(start)) / (num - 1)
    return [10 ** (math.log10(start) + i * step) for i in range(num)]


# =====================================================================
# [D07] Loss-spike / NaN monitor
# =====================================================================

def rolling_median_mad(window):
    s = sorted(window)
    n = len(s)
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    mad = statistics.median([abs(x - median) for x in window])
    return median, mad


def robust_zscore(x, window, eps=1e-8):
    median, mad = rolling_median_mad(window)
    return (x - median) / (1.4826 * mad + eps)


def detect_anomalies(seq, window_size=5, threshold=5.0, lookahead=3, shift_tol=0.05):
    """Causal detector: the window for point i is seq[i-window_size:i] --
    strictly BEFORE i, so an anomaly never dilutes its own reference stats.
    NaN/Inf are flagged unconditionally, ahead of any windowing logic."""
    events = []
    for i, x in enumerate(seq):
        if not math.isfinite(x):
            events.append({"index": i, "type": "non_finite", "z": None})
            continue
        if i < window_size:
            continue  # not enough causal history yet
        window = seq[i - window_size:i]
        if any(not math.isfinite(w) for w in window):
            continue  # window contaminated by an already-flagged point
        z = robust_zscore(x, window)
        if abs(z) < threshold:
            continue
        median, _ = rolling_median_mad(window)
        after = [a for a in seq[i + 1:i + 1 + lookahead] if math.isfinite(a)]
        if not after or abs(sum(after) / len(after) - median) < shift_tol:
            etype = "spike"            # transient, or no data to confirm persistence
        else:
            etype = "sustained_shift"  # level shift persists past the anomalous point
        events.append({"index": i, "type": etype, "z": z})
    return events


def event_at(events, idx):
    for e in events:
        if e["index"] == idx:
            return e
    return None


def main():
    if not __debug__:
        raise SystemExit("run without -O: the checks below are assert-based")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    # ---------------- [D01] dedup: exact Jaccard / MinHash / LSH / cluster split ----------------
    doc_a = set(range(1, 11))               # {1,...,10}
    doc_b = set(range(1, 9)) | {11, 12}      # {1,...,8,11,12}
    j_exact = exact_jaccard(doc_a, doc_b)
    assert abs(j_exact - 2 / 3) < 1e-12       # |A n B|=8, |A u B|=12 -> 8/12
    assert exact_jaccard(doc_a, doc_a) == 1.0
    assert exact_jaccard(doc_a, set(range(1000, 1010))) == 0.0

    num_perm, b_bands, r_rows = 1024, 128, 8   # b*r == num_perm
    sig_a = minhash_signature(doc_a, num_perm, seed=12345)
    sig_b = minhash_signature(doc_b, num_perm, seed=12345)
    j_est = minhash_estimate(sig_a, sig_b)
    # sampling std-error ~ sqrt(J(1-J)/num_perm) ~= 0.0147 at J=2/3; use a
    # generous (but principled) multiple of that as tolerance, not exact eq.
    tol = 6 * math.sqrt((2 / 3) * (1 / 3) / num_perm)
    assert abs(j_est - 2 / 3) < tol, (j_est, tol)

    p_high = lsh_candidate_probability(2 / 3, b_bands, r_rows)
    p_low = lsh_candidate_probability(0.05, b_bands, r_rows)
    assert p_high > 0.9 and p_low < 0.1 and (p_high - p_low) > 0.3

    # --- real banding -> candidate generation -> exact-Jaccard threshold -> union-find ---
    # A bridges B and C via two DIFFERENT halves (core_ab, core_bc): B shares
    # core_ab with A and core_bc with C, but A and C share NOTHING directly.
    # A and C must only end up in the same split via TRANSITIVE closure
    # through B, and B<->C's candidacy must come from REAL LSH banding on
    # real MinHash signatures, not a hand-written uf.union() call.
    core_ab, core_bc = set(range(1, 9)), set(range(101, 109))
    pdoc_a, pdoc_b, pdoc_c, pdoc_d = core_ab, core_ab | core_bc, core_bc, set(range(5000, 5010))
    corpus = [pdoc_a, pdoc_b, pdoc_c, pdoc_d]
    assert exact_jaccard(pdoc_a, pdoc_b) == 0.5 and exact_jaccard(pdoc_b, pdoc_c) == 0.5
    assert exact_jaccard(pdoc_a, pdoc_c) == 0.0        # truly nothing shared, directly

    lsh_num_perm, lsh_b, lsh_r = 128, 64, 2            # tuned for reliable recall at s=0.5
    sigs = [minhash_signature(doc, lsh_num_perm, seed=12345) for doc in corpus]
    candidates = lsh_candidate_pairs(sigs, lsh_b, lsh_r)
    assert candidates == {(0, 1), (1, 2)}              # LSH itself never proposes (A,C) or (*,D)

    jaccard_threshold = 0.4
    confirmed_edges = [(i, j) for (i, j) in candidates
                        if exact_jaccard(corpus[i], corpus[j]) >= jaccard_threshold]
    assert set(confirmed_edges) == {(0, 1), (1, 2)}    # both candidates pass the EXACT-Jaccard gate

    uf = UnionFind(len(corpus))
    for i, j in confirmed_edges:
        uf.union(i, j)
    assert uf.find(0) == uf.find(1) == uf.find(2)      # A,B,C merge -- A,C only via TRANSITIVE closure
    assert uf.find(3) != uf.find(0)                    # D (never a candidate with anyone) stays apart
    split_of = {doc: ["train", "val", "test"][uf.find(doc) % 3] for doc in range(len(corpus))}
    assert split_of[0] == split_of[1] == split_of[2]   # whole cluster -> one split

    # negative tests: degenerate inputs must fail loudly, not silently miscompute
    try:
        minhash_signature(set(), lsh_num_perm, seed=1)
        raise AssertionError("expected ValueError for an empty document set")
    except ValueError:
        pass
    try:
        minhash_signature(pdoc_a, 0, seed=1)
        raise AssertionError("expected ValueError for num_perm <= 0")
    except ValueError:
        pass
    try:
        minhash_estimate([1, 2, 3], [1, 2])
        raise AssertionError("expected ValueError for mismatched signature lengths")
    except ValueError:
        pass
    try:
        minhash_estimate([], [])
        raise AssertionError("expected ValueError for empty signatures")
    except ValueError:
        pass
    try:
        lsh_bands([1, 2, 3], b=2, r=2)                 # 3 != 2*2
        raise AssertionError("expected ValueError for len(sig) != b*r")
    except ValueError:
        pass

    print(f"[D01] dedup: exact J(A,B)={j_exact:.4f}, MinHash est={j_est:.4f} (tol {tol:.4f}), "
          f"LSH P_high={p_high:.4f} vs P_low={p_low:.4f}; real banding candidates={sorted(candidates)}, "
          f"confirmed edges={sorted(confirmed_edges)}, cluster split={split_of}  PASS")

    # ---------------- [D02] document packing: independent-doc vs continuous labels ----------------
    tokens = [11, 12, 13, 21, 22]
    doc_ids = [0, 0, 0, 1, 1]
    labels_indep = shifted_labels_independent_docs(tokens, doc_ids)
    assert labels_indep == [12, 13, IGNORE, 22, IGNORE]
    labels_cont = shifted_labels_continuous(tokens)
    assert labels_cont == [12, 13, 21, 22, IGNORE]     # doc1's first token IS a valid label here

    # negative tests: shifted_labels_independent_docs must validate its own preconditions
    try:
        shifted_labels_independent_docs([11, 12, 13], [0, 0])       # length mismatch
        raise AssertionError("expected ValueError for tokens/doc_ids length mismatch")
    except ValueError:
        pass
    try:
        shifted_labels_independent_docs([11, 12, 13], [0, 1, 0])    # non-contiguous doc id reuse
        raise AssertionError("expected ValueError for non-contiguous doc_ids reuse")
    except ValueError:
        pass

    print(f"[D02] packing labels: independent-doc={labels_indep}, continuous={labels_cont}  PASS")

    # ---------------- [D03] block-diagonal vs continuous causal mask + a FUNCTIONAL test ----------------
    allowed_block_mask = block_diagonal_causal_mask(doc_ids)
    allowed_cont_mask = continuous_causal_mask(len(tokens))
    expected_block = [
        [True,  False, False, False, False],
        [True,  True,  False, False, False],
        [True,  True,  True,  False, False],
        [False, False, False, True,  False],
        [False, False, False, True,  True],
    ]
    assert allowed_block_mask == expected_block         # EVERY element, not just a handful of samples
    assert allowed_cont_mask[3][2] is True               # same (q,k), continuous mode allows it -- a config choice

    values = [1, 2, 3, 10, 20]       # every position DISTINCT so any wrong mask element changes the output
    out_block_q3 = masked_uniform_attention_output(allowed_block_mask[3], values)
    out_block_q4 = masked_uniform_attention_output(allowed_block_mask[4], values)
    out_cont_q3 = masked_uniform_attention_output(allowed_cont_mask[3], values)
    out_cont_q4 = masked_uniform_attention_output(allowed_cont_mask[4], values)
    assert out_block_q3 == 10.0      # only k=3 (own doc) allowed -> values[3]
    assert out_block_q4 == 15.0      # k=3,4 allowed -> avg(10,20)
    assert out_cont_q3 == 4.0        # avg(1,2,3,10) over k<=3
    assert out_cont_q4 == 7.2        # avg(1,2,3,10,20) over k<=4

    # forward test: perturb ONLY doc0's values at every doc0 position; doc1's output at
    # BOTH of its query positions (q=3 AND q=4, not just q=3) must be UNCHANGED under
    # block masking, and DIFFERENT under continuous masking.
    values_doc0_changed = [1000, 2000, 3000, 10, 20]
    out_block_q3_v2 = masked_uniform_attention_output(allowed_block_mask[3], values_doc0_changed)
    out_block_q4_v2 = masked_uniform_attention_output(allowed_block_mask[4], values_doc0_changed)
    out_cont_q3_v2 = masked_uniform_attention_output(allowed_cont_mask[3], values_doc0_changed)
    out_cont_q4_v2 = masked_uniform_attention_output(allowed_cont_mask[4], values_doc0_changed)
    assert out_block_q3_v2 == out_block_q3 and out_block_q4_v2 == out_block_q4
    assert out_cont_q3_v2 != out_cont_q3 and out_cont_q4_v2 != out_cont_q4

    # reverse test: perturb ONLY doc1's values; doc0's output at EVERY one of ITS query
    # positions (q=0,1,2) must be unchanged (plain causality: a later document's values
    # cannot leak into an earlier query, independent of the doc-boundary rule).
    values_doc1_changed = [1, 2, 3, -999, -888]
    out_block_q0_v2 = masked_uniform_attention_output(allowed_block_mask[0], values_doc1_changed)
    out_block_q1_v2 = masked_uniform_attention_output(allowed_block_mask[1], values_doc1_changed)
    out_block_q2_v2 = masked_uniform_attention_output(allowed_block_mask[2], values_doc1_changed)
    assert out_block_q0_v2 == 1.0 and out_block_q1_v2 == 1.5 and out_block_q2_v2 == 2.0

    # negative test: block_diagonal_causal_mask must validate doc_ids itself
    try:
        block_diagonal_causal_mask([0, 1, 0])            # non-contiguous doc id reuse
        raise AssertionError("expected ValueError for non-contiguous doc_ids reuse")
    except ValueError:
        pass

    print(f"[D03] block mask: doc1 out unaffected by doc0 values changing "
          f"(q3: {out_block_q3}->{out_block_q3_v2}, q4: {out_block_q4}->{out_block_q4_v2}); "
          f"doc0 out unaffected by doc1 values changing (q0/q1/q2: "
          f"{out_block_q0_v2}/{out_block_q1_v2}/{out_block_q2_v2}); "
          f"continuous mode DOES change (q3: {out_cont_q3}->{out_cont_q3_v2})  PASS")

    # ---------------- [D04] EOS boundary supervision (same doc-id rule, no special-casing) ----------------
    EOS = 999
    tokens_eos = [11, 12, 13, EOS, 21, 22, EOS]
    doc_ids_eos = [0, 0, 0, 0, 1, 1, 1]         # EOS belongs to the document it closes
    labels_eos = shifted_labels_independent_docs(tokens_eos, doc_ids_eos)
    assert labels_eos == [12, 13, EOS, IGNORE, 22, EOS, IGNORE]
    assert labels_eos[2] == EOS                 # content-end(13) -> EOS: loss KEPT
    assert labels_eos[3] == IGNORE              # doc0's EOS -> doc1's first token: loss MASKED
    print(f"[D04] EOS boundary: labels={labels_eos} "
          f"(content-end->EOS kept, EOS->next-doc masked)  PASS")

    # ---------------- [D05] padding: key-padding mask + loss mask + no empty rows ----------------
    PAD = 0
    tokens_pad = [11, 12, 13, PAD, PAD]
    allowed_mask_pad, loss_mask_pad = key_padding_and_loss_mask(tokens_pad, valid_len=3)
    assert loss_mask_pad == [1, 1, 0, 0, 0]
    for q in range(3):                                    # real queries...
        assert allowed_mask_pad[q][3] is False and allowed_mask_pad[q][4] is False   # ...can't attend padding keys
    assert allowed_mask_pad[0][2] is False   # causal: query 0 can't attend a FUTURE real key either --
                                              # guards against silently dropping the k<=q term while every
                                              # OTHER assertion in this block would still pass
    assert all(any(row) for row in allowed_mask_pad)      # no all-False row -> no NaN after softmax

    # --- P1 regression check: target validity must be POSITION-based (i+1 < valid_len),
    # NEVER token-VALUE-based (tokens[i+1] == pad_id). If PAD and EOS share the same id,
    # a value-based check would wrongly zero out the loss on a REAL end-of-document EOS
    # sitting INSIDE the valid region -- exactly the bug this file's own docstring (and
    # the tutorial's CORRECTIONS section) warns against.
    EOS_AS_PAD = 999
    tokens_pad_eos = [11, 12, EOS_AS_PAD, EOS_AS_PAD, EOS_AS_PAD]   # PAD and EOS share one id
    _, loss_mask_pad_eos = key_padding_and_loss_mask(tokens_pad_eos, valid_len=3)
    assert loss_mask_pad_eos == [1, 1, 0, 0, 0]
    assert loss_mask_pad_eos[1] == 1        # query 1's target is tokens[2]==EOS_AS_PAD, INSIDE the
                                             # valid region -> loss MUST be kept (a token-value check
                                             # would have wrongly zeroed this out)

    # negative tests: illegal valid_len must be rejected explicitly, not silently mis-masked
    try:
        key_padding_and_loss_mask(tokens_pad, valid_len=6)     # > len(tokens)
        raise AssertionError("expected ValueError for valid_len > len(tokens)")
    except ValueError:
        pass
    try:
        key_padding_and_loss_mask(tokens_pad, valid_len=-1)    # negative
        raise AssertionError("expected ValueError for negative valid_len")
    except ValueError:
        pass

    print(f"[D05] padding: loss_mask={loss_mask_pad}, no attention row is empty, "
          f"PAD==EOS adversarial loss_mask={loss_mask_pad_eos} (real EOS target NOT masked)  PASS")

    # ---------------- [D06] synthetic Chinchilla-style curve fit (unit test, not law validation) ----------------
    alpha, beta, k_flops = 0.34, 0.28, 6        # exponents from the tutorial's CORRECTIONS section
    e_true, a_true, b_true = 1.5, 400.0, 400.0  # SYNTHETIC ground truth for this unit test only
    n_grid = logspace(1e7, 1e12, 6)
    d_grid = logspace(1e8, 1e13, 6)
    points = [(n, d, loss_surface(n, d, e_true, a_true, b_true, alpha, beta))
              for n in n_grid for d in d_grid]
    e_hat, a_hat, b_hat = fit_e_a_b(points, alpha, beta)
    assert abs(e_hat - e_true) < 1e-6
    assert abs(a_hat - a_true) < 1e-3
    assert abs(b_hat - b_true) < 1e-3
    assert e_hat < min(loss for _, _, loss in points)     # irreducible loss floor
    assert a_hat > 0 and b_hat > 0 and alpha > 0 and beta > 0

    # fit -> optimize actually WIRED TOGETHER: the compute-optimal point below uses the
    # FITTED a_hat/b_hat, not a_true/b_true -- that is the whole point of fitting a curve,
    # namely letting the recovered coefficients drive the downstream allocation decision.
    c_budget = 6e21
    n_star, d_star = analytic_optimum(c_budget, k_flops, alpha, beta, a_hat, b_hat)
    assert abs(k_flops * n_star * d_star - c_budget) / c_budget < 1e-9   # respects C=kND exactly

    fine_n_grid = logspace(1e6, 1e13, 400)
    grid_losses = [(n, loss_surface(n, c_budget / (k_flops * n), e_hat, a_hat, b_hat, alpha, beta))
                   for n in fine_n_grid]
    n_grid_argmin = min(grid_losses, key=lambda t: t[1])[0]
    log_step = (math.log10(1e13) - math.log10(1e6)) / (len(fine_n_grid) - 1)
    assert abs(math.log10(n_grid_argmin) - math.log10(n_star)) <= log_step * 1.5

    # sanity check only (not the primary decision path): since the fit is essentially
    # exact on this noiseless grid, the FITTED optimum should also land close to the
    # ground-truth-coefficient optimum.
    n_star_true, d_star_true = analytic_optimum(c_budget, k_flops, alpha, beta, a_true, b_true)
    assert abs(n_star - n_star_true) / n_star_true < 1e-3
    assert abs(d_star - d_star_true) / d_star_true < 1e-3

    n_star_100x, d_star_100x = analytic_optimum(100 * c_budget, k_flops, alpha, beta, a_hat, b_hat)
    ratio_n = n_star_100x / n_star
    ratio_d = d_star_100x / d_star
    assert abs(ratio_n - 100 ** (beta / (alpha + beta))) < 1e-6
    assert abs(ratio_d - 100 ** (alpha / (alpha + beta))) < 1e-6
    assert abs(ratio_n - 8.0) < 0.05     # matches the tutorial's quoted ~8.0x
    assert abs(ratio_d - 12.5) < 0.1     # matches the tutorial's quoted ~12.5x

    # negative test: solve_linear_3x3 must reject a singular design matrix explicitly
    # instead of dividing by a ~0 pivot and returning silently-wrong coefficients.
    try:
        solve_linear_3x3([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]], [1.0, 2.0, 3.0])
        raise AssertionError("expected ValueError for a singular design matrix")
    except ValueError:
        pass

    print(f"[D06] Chinchilla fit: E={e_hat:.4f} A={a_hat:.2f} B={b_hat:.2f} (true {e_true}/{a_true}/{b_true}); "
          f"N*(from fitted A_hat/B_hat)={n_star:.3e} matches grid ({n_grid_argmin:.3e}) "
          f"and true-coefficient optimum ({n_star_true:.3e}); "
          f"100x compute -> N* x{ratio_n:.3f}, D* x{ratio_d:.3f}  PASS")

    # ---------------- [D07] loss-spike / NaN monitor ----------------
    seq_main = [2.00, 2.02, 1.98, 2.01, 1.99, 2.00, 2.03, 1.97, 3.00]
    events_main = detect_anomalies(seq_main)
    flagged_idx = {e["index"] for e in events_main}
    assert flagged_idx == {8}                          # only the final point is flagged
    assert event_at(events_main, 8)["type"] in ("spike", "sustained_shift")

    seq_decline = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2]
    assert detect_anomalies(seq_decline) == []          # smooth decreasing loss: no false positive

    seq_spike = [2.0] * 5 + [3.0] + [2.0] * 3
    seq_shift = [2.0] * 5 + [3.0] * 4
    ev_spike = event_at(detect_anomalies(seq_spike), 5)
    ev_shift = event_at(detect_anomalies(seq_shift), 5)
    assert ev_spike["type"] == "spike"
    assert ev_shift["type"] == "sustained_shift"        # SAME magnitude jump, different aftermath

    seq_nan = [2.0, 2.0, 2.0, 2.0, 2.0, float("nan"), 2.0]
    seq_inf = [2.0, 2.0, 2.0, 2.0, 2.0, float("inf"), 2.0]
    assert event_at(detect_anomalies(seq_nan), 5)["type"] == "non_finite"
    assert event_at(detect_anomalies(seq_inf), 5)["type"] == "non_finite"
    print(f"[D07] spike monitor: main seq flags={sorted(flagged_idx)}, decline seq flags=0, "
          f"spike->{ev_spike['type']} vs shift->{ev_shift['type']}, NaN/Inf-> non_finite  PASS")

    print("\nall pretraining pipeline sanity checks passed")


if __name__ == "__main__":
    main()
