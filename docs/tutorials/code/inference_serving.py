"""
LLM Inference & Serving Stack - minimal runnable implementation
================================================================

Pure-stdlib (Python 3.9+, no torch/numpy) sanity checks, runs in <1s on CPU.
Pairs with: docs/tutorials/llm_inference_serving_tutorial.md (concept reference).

    [D01] sampling operators: exact support-set math for greedy/temperature/
          top-k/top-p/min-p/typical + penalties. Split into two kinds of
          functions: "compute filtered/renormalized distribution" (pure,
          deterministic -- almost every assert below is on THIS half) vs.
          "draw a token" (the only function that touches an RNG).
    [D02] filter-order non-commutativity: top_k->top_p vs top_p->top_k
          produce different support sets from the SAME distribution.
    [D03] streaming stop-sequence detector: a stop string can span multiple
          token chunks; the buffer must never leak an unsafe partial prefix.
    [D04] static vs. continuous batching simulator: exact A/B/C completion
          times + scheduler-capacity utilization, plus an equal-length
          counterexample showing continuous batching does NOT strictly
          dominate every workload, plus generic scheduling invariants.
    [D05] chunked-prefill toy model: reduces max decode-service gap (ITL)
          at the cost of the long prompt's own TTFT -- not a free lunch.
    [D06] KV-cache per-token bytes (reuses the KV-cache sibling tutorial's
          own formula verbatim; no re-derivation of MQA/GQA/MLA here).
    [D07] block allocator: logical vs. allocated slots/bytes (internal
          fragmentation from block-size rounding).
    [D08] prefix-sharing physical block count + the "don't divide KV bytes
          by TP degree unless heads divide evenly" caveat.
    [D09] (optional) roofline ridge point as an upper-bound sanity number,
          explicitly NOT a latency predictor.

Run:
    python3 inference_serving.py
"""
import math
import random
from collections import namedtuple


# ============================================================
# [D01]/[D02] Sampling: logits -> filtered distribution -> draw
# ============================================================
# Convention pinned to the tutorial's CORRECTIONS section:
#   - greedy() is a SEPARATE branch from temperature sampling; T=0 must
#     never be passed into z_i / T.
#   - every filter below takes an already-computed probability vector
#     `probs` (i.e. temperature has already been applied) and returns a
#     new, renormalized probability vector with zero mass outside its
#     support set. None of them touch randomness.
#   - sample() is the ONLY function below that calls the RNG.
#   - every support-set threshold below is an EXACT `>=` comparison against
#     the caller-supplied parameter -- never a fudged/lowered threshold.
#     Cumulative sums use math.fsum() for precision, but the comparison
#     itself is never loosened by an epsilon; that would silently redefine
#     which support set the filter computes.

def _validate_logits(logits):
    if len(logits) == 0:
        raise ValueError("logits must be non-empty")
    for z in logits:
        if isinstance(z, float) and math.isnan(z):
            raise ValueError("logits must not contain NaN")
    if all(z == -math.inf for z in logits):
        raise ValueError("logits must not be all -inf (no finite/legal token)")


def _validate_probs(probs):
    if len(probs) == 0:
        raise ValueError("probs must be non-empty")
    for p in probs:
        if isinstance(p, float) and math.isnan(p):
            raise ValueError("probs must not contain NaN")
        if p < 0:
            raise ValueError("probs must be non-negative")


def softmax(logits):
    _validate_logits(logits)
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def greedy(logits):
    """argmax with a DETERMINISTIC tie-break rule: lowest index wins ties.
    (T=0 must go through this branch, never through temperature_probs.)"""
    _validate_logits(logits)
    best_i, best_v = 0, logits[0]
    for i, v in enumerate(logits):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def temperature_probs(logits, T):
    """p_i(T) = softmax(z_i / T). Defined ONLY for finite T > 0.

    Computes (z_i - z_max) / T directly -- NOT z_i / T followed by a SECOND
    max-subtraction inside softmax(). Dividing by T first means an extreme
    logit combined with a very small T can overflow to +-inf (or produce
    inf - inf = nan) before the max-shift ever gets a chance to cancel it;
    subtracting z_max in logit space FIRST keeps every exponent <= 0.
    """
    _validate_logits(logits)
    if not math.isfinite(T) or T <= 0:
        raise ValueError(
            f"temperature must be finite and > 0; got {T} (call greedy() for T=0)")
    zmax = max(logits)
    exps = [math.exp((z - zmax) / T) for z in logits]
    total = math.fsum(exps)
    return [e / total for e in exps]


def top_k_filter(probs, k):
    """Support S_k = the k highest-probability tokens (ties broken by lower
    index, for a deterministic ranking). Positive-temperature scaling does
    not change the ranking, so S_k by itself does not depend on T>0.

    k must satisfy 1 <= k <= len(probs): k=0 would legally return the
    all-zero vector, k<0 would silently trigger Python's negative-slice
    semantics on `order[:k]`, and k>len(probs) can never produce a support
    set of size exactly k -- all three are rejected outright."""
    _validate_probs(probs)
    n = len(probs)
    if k != int(k) or not (1 <= k <= n):
        raise ValueError(f"k must be an integer with 1 <= k <= {n} (len(probs)), got {k!r}")
    k = int(k)
    order = sorted(range(n), key=lambda i: (-probs[i], i))
    support = set(order[:k])
    mass = sum(probs[i] for i in support)
    return [probs[i] / mass if i in support else 0.0 for i in range(n)]


def top_p_filter(probs, p0):
    """Sort descending; take the SMALLEST prefix whose cumulative mass first
    reaches p0 (the boundary token is KEPT). NOT "all p_i >= p0" and NOT
    "cumulative strictly < p0" -- both of those can wrongly drop the
    boundary token.

    p0 must satisfy 0 < p0 <= 1. The cumulative sum used for the `>= p0`
    test is recomputed via math.fsum() over the tokens included so far on
    every step (rather than a running `cum += p`) to keep the ONE
    comparison that matters -- reaching the exact threshold -- as precise
    as floating point allows; the threshold itself is never lowered by an
    epsilon, which would silently stop one token too early."""
    _validate_probs(probs)
    if not (0.0 < p0 <= 1.0):
        raise ValueError(f"p0 must satisfy 0 < p0 <= 1, got {p0}")
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    support, included = set(), []
    for i in order:
        support.add(i)
        included.append(probs[i])
        if math.fsum(included) >= p0:
            break
    mass = sum(probs[i] for i in support)
    return [probs[i] / mass if i in support else 0.0 for i in range(len(probs))]


def min_p_filter(probs, alpha):
    """S = {i : p_i >= alpha * max_j p_j}, renormalized within S.

    alpha must satisfy 0 < alpha <= 1: alpha<=0 would keep every token
    (min-p degenerating to a no-op, and disagreeing with the logit-space
    formula below where log(alpha) is undefined at alpha<=0), and alpha>1
    can produce an EMPTY support set (no p_i can exceed alpha*p_max when
    alpha>1) -- silently returning an all-zero, non-normalized "distribution"
    instead of raising, which would only surface downstream as a confusing
    failure the next time something tries to sample from it."""
    _validate_probs(probs)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}")
    pmax = max(probs)
    support = {i for i, p in enumerate(probs) if p >= alpha * pmax}
    mass = sum(probs[i] for i in support)
    return [probs[i] / mass if i in support else 0.0 for i in range(len(probs))]


def min_p_support_from_logits(logits, T, alpha):
    """Equivalent logit-space condition when p_i = softmax(z_i/T):
    z_i >= z_max + T*log(alpha). This support set MOVES with T -- it is
    not a fixed logit-gap threshold.

    Requires finite T > 0 (the derivation assumes softmax(z/T), which is
    undefined for T<=0) and 0 < alpha <= 1 (same domain as min_p_filter,
    so the two independent derivations of the SAME support set stay
    consistent with each other)."""
    _validate_logits(logits)
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {T}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}")
    zmax = max(logits)
    thresh = zmax + T * math.log(alpha)
    return {i for i, z in enumerate(logits) if z >= thresh}


def typical_filter(probs, typical_p):
    """H(p) = -sum p_i log p_i; d_i = |-log p_i - H(p)|; sort by d_i
    ascending (closest to the entropy first); take the smallest prefix in
    THAT order whose cumulative mass reaches typical_p. The resulting
    support need not be a contiguous prefix of the probability-sorted
    order -- see the D01 assertions below.

    typical_p must satisfy 0 < typical_p <= 1, mirroring top_p_filter's
    p0 domain; the cumulative-mass comparison is likewise an exact `>=`
    over an math.fsum()-recomputed running total, never epsilon-loosened."""
    _validate_probs(probs)
    if not (0.0 < typical_p <= 1.0):
        raise ValueError(f"typical_p must satisfy 0 < typical_p <= 1, got {typical_p}")
    H = -sum(p * math.log(p) for p in probs if p > 0)
    d = [abs(-math.log(p) - H) if p > 0 else math.inf for p in probs]
    order = sorted(range(len(probs)), key=lambda i: (d[i], i))
    support, included = set(), []
    for i in order:
        support.add(i)
        included.append(probs[i])
        if math.fsum(included) >= typical_p:
            break
    mass = sum(probs[i] for i in support)
    return [probs[i] / mass if i in support else 0.0 for i in range(len(probs))]


def freq_presence_penalty(logits, counts, alpha_f, alpha_p):
    """z_i' = z_i - alpha_f*c_i - alpha_p*1[c_i>0]. `counts` semantics
    (whether prompt tokens are counted) is an API-level choice, not a
    universal constant -- the caller decides what goes into `counts`."""
    return [z - alpha_f * counts[i] - (alpha_p if counts[i] > 0 else 0.0)
            for i, z in enumerate(logits)]


def repetition_penalty(logits, seen, r):
    """Sign-aware penalty, r > 1, applied ONLY to already-seen tokens:
    z_i/r if z_i>0 else z_i*r. This is NOT "subtract a constant from every
    seen token" -- the direction depends on the sign of z_i."""
    assert r > 1
    out = []
    for i, z in enumerate(logits):
        if i in seen:
            out.append(z / r if z > 0 else z * r)
        else:
            out.append(z)
    return out


def apply_hard_mask(logits, allowed):
    """Constrained decoding primitive: illegal tokens -> -inf logit, then a
    normal softmax renormalizes over the legal ones."""
    return [z if allowed[i] else -math.inf for i, z in enumerate(logits)]


class StubRNG:
    """Deterministic stand-in for random.Random, for tests that need an
    EXACT `u` value (0.0, an exact cumulative boundary, a value close to 1)
    rather than a statistical frequency check over many draws."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def random(self):
        v = self._values[self._i]
        self._i += 1
        return v


def sample(probs, rng):
    """The ONLY function in this module that touches randomness.

    u ~ Uniform[0, 1). Uses `u < cum` (NOT `u <= cum`): with `<=`, u==0.0
    would select index 0 even when probs[0] == 0.0 (a token filtered out by
    top-k/top-p/min-p/typical has probability EXACTLY 0, and must never be
    drawn). The fallback -- needed only when floating-point summation
    leaves the true final cumulative sum fractionally below 1.0 -- returns
    the LAST token with POSITIVE probability, never an unconditional
    len(probs)-1: that trailing index may itself have been filtered to
    zero mass."""
    positive = [i for i, p in enumerate(probs) if p > 0]
    if not positive:
        raise ValueError("sample() requires at least one token with positive probability")
    u = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if u < cum:
            return i
    return positive[-1]


def run_d01():
    # --- greedy: no softmax needed, deterministic tie-break ---
    assert greedy([5, 5, 1]) == 0                     # lowest-index tie-break
    assert greedy([1, 5, 5]) == 1
    for bad_T in (0.0, -1.0, float("inf"), float("nan")):
        try:
            temperature_probs([1.0, 2.0], bad_T)
            raise AssertionError(f"temperature_probs must reject T={bad_T}")
        except ValueError:
            pass

    # --- degenerate logits/probs inputs: empty / NaN / all -inf must raise
    #     a clear ValueError, never silently produce nan or garbage output ---
    for bad_fn, args in (
        (softmax, ([],)),
        (greedy, ([],)),
        (temperature_probs, ([], 1.0)),
        (top_k_filter, ([], 1)),
        (top_p_filter, ([], 0.5)),
        (min_p_filter, ([], 0.5)),
        (typical_filter, ([], 0.5)),
        (min_p_support_from_logits, ([], 1.0, 0.5)),
    ):
        try:
            bad_fn(*args)
            raise AssertionError(f"{bad_fn.__name__} must reject empty input")
        except ValueError:
            pass
    try:
        softmax([1.0, float("nan"), 2.0])
        raise AssertionError("softmax must reject NaN logits")
    except ValueError:
        pass
    try:
        softmax([-math.inf, -math.inf])
        raise AssertionError("softmax must reject all -inf logits (no legal token)")
    except ValueError:
        pass
    try:
        temperature_probs([-math.inf, -math.inf], 1.0)
        raise AssertionError("temperature_probs must reject all -inf logits")
    except ValueError:
        pass
    try:
        top_p_filter([0.5, float("nan"), 0.5], 0.5)
        raise AssertionError("top_p_filter must reject NaN probabilities")
    except ValueError:
        pass

    # --- temperature: T->0+ limit concentrates on argmax (unique max) ---
    unique_logits = [5.0, 3.0, 1.0]
    p_tiny = temperature_probs(unique_logits, 1e-4)
    assert p_tiny[0] > 0.999999                       # concentrates on argmax

    # --- tied max logits: math limit SPLITS mass among ties, not one-hot ---
    tied_logits = [5.0, 5.0, 1.0]
    p_tied = temperature_probs(tied_logits, 1e-4)
    assert abs(p_tied[0] - 0.5) < 1e-4 and abs(p_tied[1] - 0.5) < 1e-4
    assert p_tied[2] < 1e-6

    # --- lower T sharpens (raises max prob) for a fixed logits vector ---
    p_hot = temperature_probs(unique_logits, 2.0)
    p_cold = temperature_probs(unique_logits, 0.5)
    assert max(p_cold) > max(p_hot)

    # --- top-k: exact support size, excludes anything ranked below k,
    #     stable tie-break for equal logits ---
    probs5 = softmax([4.0, 3.0, 2.0, 1.0, 0.0])
    for k in (1, 2, 3):
        filt = top_k_filter(probs5, k)
        support = {i for i, p in enumerate(filt) if p > 0}
        assert len(support) == k
        assert max(support) == k - 1                  # never keeps a lower-ranked token
        assert abs(sum(filt) - 1.0) < 1e-12
        assert all(p >= 0 for p in filt)
        if k >= 2:
            # relative proportions WITHIN the support must match the ORIGINAL
            # distribution -- catches a broken renormalization that flattens
            # the support to a uniform distribution instead of rescaling it.
            assert abs(filt[0] / filt[1] - probs5[0] / probs5[1]) < 1e-9
    tie_probs = softmax([3.0, 3.0, 1.0])               # index0==index1 tie
    assert {i for i, p in enumerate(top_k_filter(tie_probs, 1)) if p > 0} == {0}

    # --- top-k domain/boundary: k must satisfy 1 <= k <= n (n=5 here) ---
    n5 = len(probs5)
    for bad_k in (0, -1, n5 + 1):
        try:
            top_k_filter(probs5, bad_k)
            raise AssertionError(f"top_k_filter must reject k={bad_k}")
        except ValueError:
            pass
    filt_kn = top_k_filter(probs5, n5)                 # k == n keeps everything
    assert all(p > 0 for p in filt_kn)
    assert abs(sum(filt_kn) - 1.0) < 1e-12

    # --- top-p: design-review numeric example -----------------------------
    # p = [0.4, 0.3, 0.2, 0.1], threshold 0.65 -> boundary token MUST be kept
    p_ex = [0.4, 0.3, 0.2, 0.1]
    filt = top_p_filter(p_ex, 0.65)
    support = {i for i, p in enumerate(filt) if p > 0}
    assert support == {0, 1}                           # exactly the first two
    assert abs(sum(p_ex[i] for i in support) - 0.7) < 1e-12   # cumulative mass 0.7
    # naive "cumulative strictly < p0" would stop BEFORE the boundary token
    # (mass 0.4 < 0.65) and wrongly drop it:
    assert sum([p_ex[0]]) < 0.65
    # naive "keep all p_i >= p0" keeps NOTHING here (no single token >= 0.65):
    assert all(p < 0.65 for p in p_ex)
    assert abs(sum(filt) - 1.0) < 1e-12 and all(p >= 0 for p in filt)
    # relative proportions within the support must match the original ratio
    # (a bug that renormalizes to a uniform 0.5/0.5 would also pass a bare
    # "sums to 1" check, so check the actual values):
    assert abs(filt[0] - p_ex[0] / 0.7) < 1e-12
    assert abs(filt[1] - p_ex[1] / 0.7) < 1e-12
    # top_p = 1 keeps every token with nonzero probability
    filt_all = top_p_filter(p_ex, 1.0)
    assert all(f > 0 for f in filt_all)

    # --- exact boundary: p0 == 0.7 == the true cumulative mass of {0,1} ---
    # this is the case that distinguishes a correct `>=` from a wrong
    # strict `>` (which would keep scanning past the boundary token):
    filt_boundary = top_p_filter(p_ex, 0.7)
    support_boundary = {i for i, p in enumerate(filt_boundary) if p > 0}
    assert support_boundary == {0, 1}

    # --- epsilon regression: p0 a hair ABOVE the true cumulative mass of
    #     {0,1} (0.7 exactly) must NOT stop at {0,1} -- a threshold that has
    #     been fudged downward by e.g. `- 1e-12` would wrongly treat 0.7 as
    #     "close enough" and drop the token that should push the support to
    #     {0,1,2} ---
    p0_just_above = 0.7 + 5e-13
    filt_eps = top_p_filter(p_ex, p0_just_above)
    support_eps = {i for i, p in enumerate(filt_eps) if p > 0}
    assert support_eps == {0, 1, 2}

    # --- p0 domain: must satisfy 0 < p0 <= 1 ---
    for bad_p0 in (0.0, -0.1, 1.1):
        try:
            top_p_filter(p_ex, bad_p0)
            raise AssertionError(f"top_p_filter must reject p0={bad_p0}")
        except ValueError:
            pass

    # --- min-p ---
    z = [3.1, 1.0, 0.4, -2.0]
    probs_mp = softmax(z)
    argmax_i = probs_mp.index(max(probs_mp))
    prev_support_size = None
    prev_support_set = None
    for alpha in (0.9, 0.5, 0.2, 0.05):
        filt = min_p_filter(probs_mp, alpha)
        support = {i for i, p in enumerate(filt) if p > 0}
        assert argmax_i in support                    # max-prob token always kept
        size = len(support)
        if prev_support_size is not None:
            assert size >= prev_support_size           # smaller alpha -> support grows (monotonic)
            assert prev_support_set <= support          # NESTED: shrinking alpha only ADDS
                                                         # tokens, it never drops one that a
                                                         # larger alpha had already kept
        if size >= 2:
            # relative proportions within the support must match the
            # original distribution here too (not just "sums to 1")
            j, k = sorted(support)[0], sorted(support)[1]
            assert abs(filt[j] / filt[k] - probs_mp[j] / probs_mp[k]) < 1e-9
        prev_support_size = size
        prev_support_set = support
    only_argmax = min_p_filter(probs_mp, 1.0)
    assert {i for i, p in enumerate(only_argmax) if p > 0} == {argmax_i}

    # --- alpha=1 with a TIE for the max: must keep ALL tied-max tokens,
    #     not just one of them (the earlier test only covered a unique max) ---
    tied_max_probs = softmax([3.0, 3.0, 1.0])
    only_tied = min_p_filter(tied_max_probs, 1.0)
    tied_support = {i for i, p in enumerate(only_tied) if p > 0}
    assert tied_support == {0, 1}
    assert abs(only_tied[0] - 0.5) < 1e-12 and abs(only_tied[1] - 0.5) < 1e-12

    # --- alpha domain: must satisfy 0 < alpha <= 1 ---
    for bad_alpha in (0.0, -0.1, 1.1):
        try:
            min_p_filter(probs_mp, bad_alpha)
            raise AssertionError(f"min_p_filter must reject alpha={bad_alpha}")
        except ValueError:
            pass

    # logit shift invariance: adding a constant to all logits changes nothing
    shifted = [x + 7.0 for x in z]
    assert min_p_filter(softmax(shifted), 0.3) == min_p_filter(softmax(z), 0.3)

    # --- logit shift invariance for top-k and top-p too (the tutorial
    #     appendix claims this holds for min-p/top-k/top-p; only min-p was
    #     actually checked above -- close that gap here) ---
    z_shift_base = [2.0, 1.0, 0.0, -1.0, -3.0]
    probs_shift_base = softmax(z_shift_base)
    probs_shift_plus5 = softmax([zz + 5.0 for zz in z_shift_base])
    assert top_k_filter(probs_shift_base, 2) == top_k_filter(probs_shift_plus5, 2)
    assert top_p_filter(probs_shift_base, 0.7) == top_p_filter(probs_shift_plus5, 0.7)

    # --- min-p support set MOVES with temperature (not a fixed logit gap) ---
    logits_t = [4.0, 2.0, 1.0, -1.0]
    alpha = 0.3
    sizes = []
    for T in (0.2, 0.5, 1.0, 2.0, 5.0):
        s = min_p_support_from_logits(logits_t, T, alpha)
        # cross-check: same support via probs-space filter at this T
        probs_T = temperature_probs(logits_t, T)
        s_via_probs = {i for i, p in enumerate(min_p_filter(probs_T, alpha)) if p > 0}
        assert s == s_via_probs
        sizes.append(len(s))
    assert sizes == sorted(sizes)                      # non-decreasing as T grows (alpha<1)
    assert sizes[0] < sizes[-1]                        # and it strictly changes somewhere

    # --- min_p_support_from_logits domain: T must be finite and > 0
    #     (negative/zero temperature has no softmax(z/T) interpretation) ---
    for bad_T in (0.0, -1.0, float("inf"), float("nan")):
        try:
            min_p_support_from_logits(logits_t, bad_T, alpha)
            raise AssertionError(f"min_p_support_from_logits must reject T={bad_T}")
        except ValueError:
            pass

    # --- typical sampling: NOT equivalent to top-p; support need not be a
    #     contiguous prefix of the probability-sorted order (it can skip the
    #     single highest-probability token while keeping two lower ones) ---
    p_typ = [0.02, 0.35, 0.33, 0.30]                    # sums to 1.0
    assert abs(sum(p_typ) - 1.0) < 1e-12
    argmax_typ = p_typ.index(max(p_typ))                # index 1 (prob 0.35)
    filt_typ = typical_filter(p_typ, 0.63)
    support_typ = {i for i, p in enumerate(filt_typ) if p > 0}
    assert support_typ == {2, 3}                        # skips index1 (the argmax!)
    assert argmax_typ not in support_typ
    # this set is not equal to ANY top-m-by-probability prefix (m=1,2,3):
    order_by_prob = sorted(range(len(p_typ)), key=lambda i: (-p_typ[i], i))
    for m in (1, 2, 3):
        assert support_typ != set(order_by_prob[:m])
    assert abs(sum(filt_typ) - 1.0) < 1e-12
    # relative proportions within the support must match the original ratio
    assert abs(filt_typ[2] / filt_typ[3] - p_typ[2] / p_typ[3]) < 1e-9

    # --- typical_p domain: must satisfy 0 < typical_p <= 1 ---
    for bad_tp in (0.0, -0.1, 1.1):
        try:
            typical_filter(p_typ, bad_tp)
            raise AssertionError(f"typical_filter must reject typical_p={bad_tp}")
        except ValueError:
            pass

    # --- penalties ---
    logits_p = [2.0, -1.0, 0.5]
    counts = [3, 0, 1]
    pen = freq_presence_penalty(logits_p, counts, alpha_f=0.2, alpha_p=0.5)
    assert pen[1] == -1.0                               # never appeared -> untouched
    assert pen[0] == 2.0 - 0.2 * 3 - 0.5
    assert pen[2] == 0.5 - 0.2 * 1 - 0.5
    rp = repetition_penalty([2.0, -2.0, 1.0], seen={0, 1}, r=2.0)
    assert rp == [1.0, -4.0, 1.0]                       # positive/r, negative*r, untouched

    # --- constrained decoding: illegal tokens get exactly zero mass ---
    masked = apply_hard_mask([1.0, 2.0, 0.5, 9.0], allowed=[True, True, False, False])
    p_masked = softmax(masked)
    assert p_masked[2] == 0.0 and p_masked[3] == 0.0
    assert abs(p_masked[0] + p_masked[1] - 1.0) < 1e-12

    # --- sample(): deterministic checks via a stub RNG (the PRIMARY
    #     verification -- exact `u` values pinned to exact bucket boundaries,
    #     rather than relying only on statistical tolerance) ---
    probs_det = [0.5, 0.2, 0.15, 0.1, 0.05]                # cumulative: .5 .7 .85 .95 1.0
    assert sample(probs_det, StubRNG([0.0])) == 0          # u=0.0 -> first bucket
    assert sample(probs_det, StubRNG([0.5])) == 1          # u==cum(bucket0) -> NOT bucket0
                                                            # (u < cum is false at the boundary,
                                                            # so it must fall into the NEXT bucket)
    assert sample(probs_det, StubRNG([0.999999999])) == 4  # near 1 -> last bucket
    assert sample(probs_det, StubRNG([1.0])) == 4           # forces the fallback path (true final
                                                             # cumulative sum can undershoot 1.0 due
                                                             # to floating point); must still land on
                                                             # the LAST bucket, not silently misbehave

    # zero-probability-token regression: u=0.0 must never select a token
    # whose probability was filtered to exactly zero, whether it's at the
    # front of the vector or the very last (fallback) index
    probs_zero_head = [0.0, 0.6, 0.4]
    assert sample(probs_zero_head, StubRNG([0.0])) == 1     # NOT index 0 (zero mass)

    # this specific construction (ten 0.1's, naive running sum lands EXACTLY
    # on the largest double < 1.0, plus a filtered zero-prob tail) forces the
    # fallback branch for u=1.0: an old buggy `return len(probs) - 1` would
    # return the ZERO-probability tail index (10); the correct fallback must
    # return the last POSITIVE-probability index (9) instead.
    probs_zero_tail = [0.1] * 10 + [0.0]
    assert sample(probs_zero_tail, StubRNG([1.0])) == 9
    try:
        sample([0.0, 0.0], StubRNG([0.5]))
        raise AssertionError("sample() must reject an all-zero probability vector")
    except ValueError:
        pass

    # --- draw(): empirical frequency check kept as a SUPPLEMENTARY
    #     statistical sanity test (fixed seed, wide tolerance) -- the
    #     deterministic StubRNG checks above are the primary verification ---
    rng = random.Random(42)
    two_way = [0.7, 0.3]
    n = 4000
    count0 = sum(1 for _ in range(n) if sample(two_way, rng) == 0)
    assert 0.65 <= count0 / n <= 0.75

    print("[D01] sampling operators: greedy tie-break, T=0 rejected, T->0 limits "
          "(unique->one-hot, tied->split), top-k/top-p/min-p/typical support sets "
          "(exact boundaries, nested/shift-invariant, domain-checked), penalties, "
          "constrained masking, deterministic + empirical draw checks PASS")


def run_d02():
    """Filter-order non-commutativity: top_k->top_p vs top_p->top_k on the
    SAME distribution give DIFFERENT final support sets."""
    probs = [0.5, 0.2, 0.15, 0.1, 0.05]
    order_a = top_p_filter(top_k_filter(probs, 3), 0.8)     # top_k THEN top_p
    order_b = top_k_filter(top_p_filter(probs, 0.8), 3)     # top_p THEN top_k
    support_a = {i for i, p in enumerate(order_a) if p > 0}
    support_b = {i for i, p in enumerate(order_b) if p > 0}
    assert support_a == {0, 1}
    assert support_b == {0, 1, 2}
    assert support_a != support_b                            # order matters
    print(f"[D02] filter order is NOT commutative: top_k->top_p support={sorted(support_a)}, "
          f"top_p->top_k support={sorted(support_b)}  PASS")


# ============================================================
# [D03] Streaming stop-sequence detection across token chunks
# ============================================================

def _stop_overlap_len(buffer, stop):
    """Longest suffix of `buffer` that is also a (proper) prefix of `stop` --
    that suffix must be held back because it MIGHT still grow into a full
    match with the next chunk."""
    max_k = min(len(buffer), len(stop) - 1)
    for k in range(max_k, 0, -1):
        if buffer.endswith(stop[:k]):
            return k
    return 0


def stream_with_stop(chunks, stop):
    """Feed token chunks one at a time. Returns (emitted_text, stopped,
    leftover_buffer). Never emits text that could still be a partial match
    of `stop` -- that text stays in the internal buffer until either (a)
    the match completes (truncate + stop) or (b) more chunks arrive and
    prove it can't complete after all (then it becomes safe to emit)."""
    emitted, buffer = "", ""
    for chunk in chunks:
        buffer += chunk
        if stop in buffer:
            idx = buffer.index(stop)
            emitted += buffer[:idx]
            return emitted, True, ""
        k = _stop_overlap_len(buffer, stop)
        emitted += buffer[:len(buffer) - k]
        buffer = buffer[len(buffer) - k:]
    return emitted, False, buffer


def run_d03():
    # stop string "STOP" spans three separate token chunks: "cS" | "TO" | "Pxyz"
    chunks = ["ab", "cS", "TO", "Pxyz"]
    emitted, stopped, _ = stream_with_stop(chunks, "STOP")
    assert emitted == "abc" and stopped is True
    # never leaked "S", "ST", or "STO" as committed output at any point:
    partial_emits = []
    buf, acc = "", ""
    for c in chunks:
        buf += c
        if "STOP" in buf:
            acc += buf[:buf.index("STOP")]
            partial_emits.append(acc)
            break
        k = _stop_overlap_len(buf, "STOP")
        acc += buf[:len(buf) - k]
        buf = buf[len(buf) - k:]
        partial_emits.append(acc)
    assert all(not p.endswith(("S", "ST", "STO")) for p in partial_emits[:-1])

    # no stop match at all: nothing is lost, everything reconstructible via flush
    chunks_nostop = ["ab", "cd", "ef"]
    emitted2, stopped2, remainder = stream_with_stop(chunks_nostop, "STOP")
    assert stopped2 is False
    assert emitted2 + remainder == "".join(chunks_nostop)   # full text recoverable
    print(f"[D03] stop-string streaming: 'STOP' split across 3 chunks still detected "
          f"(emitted={emitted!r}), no-match case fully recoverable via flush  PASS")


# ============================================================
# [D04] Static vs. continuous batching simulator
# ============================================================

Request = namedtuple("Request", ["name", "arrival", "ticks_needed"])


def _validate_requests(requests, capacity):
    """Reject anything that could make the tick loop below fail to
    terminate: duplicate names (dict/active/completed all key on `name`, so
    duplicates silently merge and the loop's `len(completed) < len(requests)`
    target can never be reached), non-positive/non-integer capacity (nothing
    can ever be admitted), and non-positive/non-integer ticks_needed (a
    request with ticks_needed=0 gets decremented straight to -1 and never
    equals 0 again; a non-integer can step over 0 the same way) or a
    negative/non-integer arrival (outside the tick domain)."""
    if not requests:
        raise ValueError("requests must be non-empty")
    names = [r.name for r in requests]
    if len(set(names)) != len(names):
        raise ValueError(f"request names must be unique, got {names}")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
    for r in requests:
        if not isinstance(r.arrival, int) or isinstance(r.arrival, bool) or r.arrival < 0:
            raise ValueError(
                f"arrival must be a non-negative integer, got {r.arrival!r} for request {r.name!r}")
        if not isinstance(r.ticks_needed, int) or isinstance(r.ticks_needed, bool) or r.ticks_needed <= 0:
            raise ValueError(
                f"ticks_needed must be a positive integer, got {r.ticks_needed!r} for request {r.name!r}")


def simulate_batching(requests, capacity, policy):
    """Tick-based scheduler-capacity simulator. NOT a model of real GPU
    utilization -- it is a unit-cost toy: every ACTIVE request advances by
    exactly one decode token per tick.

    policy == "static": a batch is formed (up to `capacity` arrived, not-yet-
        started requests, FCFS) only when the active set is completely
        empty. A finished member's slot sits idle -- it is NOT backfilled --
        until every member of that batch has finished.
    policy == "continuous": every tick, after removing finished requests,
        the scheduler tops the active set back up to `capacity` with
        whichever arrived requests are waiting (FCFS).
    """
    _validate_requests(requests, capacity)
    by_name = {r.name: r for r in requests}
    waiting = sorted(requests, key=lambda r: (r.arrival, r.name))
    active = {}       # name -> remaining ticks
    started = set()
    completed = {}
    serviced_log = []      # (tick, name) once per unit of service
    t, guard = 0, 0
    n_requests = len(requests)
    while len(completed) < n_requests:
        guard += 1
        if guard >= 100_000:
            # an explicit exception (not `assert`) so this guard survives
            # even when the interpreter is run with `python -O`, which
            # strips bare `assert` statements -- illegal input must never
            # be able to turn into a silent infinite loop.
            raise RuntimeError("simulation did not terminate within the tick guard")
        arrived = [r for r in waiting if r.arrival <= t and r.name not in started]

        if policy == "continuous":
            while len(active) < capacity and arrived:
                r = arrived.pop(0)
                active[r.name] = r.ticks_needed
                started.add(r.name)
        elif policy == "static":
            if len(active) == 0 and arrived:
                for r in arrived[:capacity]:
                    active[r.name] = r.ticks_needed
                    started.add(r.name)
        else:
            raise ValueError(policy)

        if not active:
            # Idle tick: nothing is running. Given validated input this can
            # only mean some requests simply haven't arrived yet (every
            # already-arrived, not-yet-started request would have been
            # admitted above, by either policy, since active was empty) --
            # so jump straight to the next arrival instead of burning one
            # tick at a time waiting for it.
            not_yet_started = [r for r in waiting if r.name not in started]
            if not_yet_started:
                t = max(t, min(r.arrival for r in not_yet_started))
                continue
            raise RuntimeError(
                "no active or pending requests but the simulation is not complete "
                "(internal inconsistency)")

        # -- invariants, checked every tick --
        assert len(active) <= capacity                                # capacity respected
        for name in active:
            assert by_name[name].arrival <= t                         # never runs before arrival
            assert name not in completed                              # completed -> no longer occupies a slot (no KV held)

        for name in list(active.keys()):
            serviced_log.append((t, name))                            # exactly one unit of service this tick
            active[name] -= 1
        t += 1
        for name in list(active.keys()):
            if active[name] == 0:
                completed[name] = t
                del active[name]                                      # slot freed, KV reclaimed

    # generation-token-count conservation
    per_request_service = {}
    for tick, name in serviced_log:
        per_request_service[name] = per_request_service.get(name, 0) + 1
    for r in requests:
        assert per_request_service[r.name] == r.ticks_needed
    return completed, serviced_log


def utilization(completed, requests, capacity):
    total_work = sum(r.ticks_needed for r in requests)
    makespan = max(completed.values())
    return total_work / (capacity * makespan), makespan


def run_d04():
    reqs = [Request("A", 0, 1), Request("B", 0, 4), Request("C", 0, 1)]
    cap = 2
    comp_static, _ = simulate_batching(reqs, cap, "static")
    comp_cont, _ = simulate_batching(reqs, cap, "continuous")

    assert comp_static == {"A": 1, "B": 4, "C": 5}          # C waits until t=4
    assert comp_cont == {"A": 1, "C": 2, "B": 4}            # C is refilled into A's slot at t=1

    u_static, ms_static = utilization(comp_static, reqs, cap)
    u_cont, ms_cont = utilization(comp_cont, reqs, cap)
    assert ms_static == 5 and ms_cont == 4
    assert abs(u_static - 0.60) < 1e-9
    assert abs(u_cont - 0.75) < 1e-9

    # --- equal-length counterexample: continuous does NOT strictly dominate ---
    reqs_eq = [Request("X", 0, 3), Request("Y", 0, 3)]
    comp_static_eq, _ = simulate_batching(reqs_eq, 2, "static")
    comp_cont_eq, _ = simulate_batching(reqs_eq, 2, "continuous")
    assert comp_static_eq == comp_cont_eq == {"X": 3, "Y": 3}   # identical makespan
    u_s_eq, ms_s_eq = utilization(comp_static_eq, reqs_eq, 2)
    u_c_eq, ms_c_eq = utilization(comp_cont_eq, reqs_eq, 2)
    assert ms_s_eq == ms_c_eq and abs(u_s_eq - u_c_eq) < 1e-12

    # --- delayed-arrival workload: every previous test used arrival=0 for
    #     everything, which can never independently verify "a request must
    #     never be serviced before it arrives" -- P arrives at 0, Q at 3,
    #     R at 5; check every logged service record against arrival/completion ---
    reqs_delayed = [Request("P", 0, 2), Request("Q", 3, 2), Request("R", 5, 1)]
    comp_delayed, log_delayed = simulate_batching(reqs_delayed, capacity=2, policy="continuous")
    by_name_delayed = {r.name: r for r in reqs_delayed}
    assert comp_delayed == {"P": 2, "Q": 5, "R": 6}
    for tick, name in log_delayed:
        assert by_name_delayed[name].arrival <= tick < comp_delayed[name]

    # --- illegal input must raise immediately, never hang (previously:
    #     duplicate names broke the loop's termination condition; ticks_needed
    #     <=0 or non-integer could step over the "== 0 -> completed" check;
    #     capacity <=0 could never admit anything; the old termination guard
    #     was a bare `assert`, stripped by `python -O`) ---
    illegal_cases = [
        ([Request("A", 0, 1), Request("A", 0, 2)], 2, "continuous"),   # duplicate name
        ([Request("A", 0, 0)], 2, "continuous"),                       # ticks_needed == 0
        ([Request("A", 0, 1.5)], 2, "continuous"),                     # non-integer ticks_needed
        ([Request("A", 0, 1)], 0, "continuous"),                       # capacity == 0
        ([Request("A", 0, 1)], -1, "continuous"),                      # capacity < 0
        ([Request("A", -1, 1)], 2, "continuous"),                      # negative arrival
        ([Request("A", 0.5, 1)], 2, "continuous"),                     # non-integer arrival
    ]
    for bad_reqs, bad_cap, bad_policy in illegal_cases:
        try:
            simulate_batching(bad_reqs, bad_cap, bad_policy)
            raise AssertionError(f"simulate_batching must reject {bad_reqs, bad_cap}")
        except ValueError:
            pass

    print(f"[D04] static: A=1,B=4,C=5 (util={u_static:.2f}); "
          f"continuous: A=1,C=2,B=4 (util={u_cont:.2f}); "
          f"equal-length counterexample: both makespan={ms_s_eq} (no strict dominance); "
          f"delayed-arrival workload respects arrival<=tick<completion; "
          f"illegal inputs ({len(illegal_cases)} cases) rejected immediately  PASS")


# ============================================================
# [D05] Chunked prefill: decode-gap vs. TTFT tradeoff (toy token-budget model)
# ============================================================

def chunked_vs_unchunked(total_prefill_work, budget_per_iter, decode_share_per_iter):
    """Stylized per-iteration token-budget model (NOT a real scheduler):
    - unchunked: a long prefill takes the ENTIRE iteration budget exclusively
      until it's done; any concurrently active decode request gets 0
      service during those iterations (max gap = #iterations).
    - chunked: every iteration reserves `decode_share_per_iter` units for
      the decode request (gap = 1, serviced every iteration) and gives the
      REST of the budget to a prefill chunk -- so the prefill itself takes
      MORE iterations to finish (its own TTFT gets worse) in exchange for
      the decode request never stalling.
    """
    iters_unchunked = math.ceil(total_prefill_work / budget_per_iter)
    max_gap_unchunked = iters_unchunked

    prefill_rate_chunked = budget_per_iter - decode_share_per_iter
    assert prefill_rate_chunked > 0, "decode share must leave room for prefill progress"
    iters_chunked = math.ceil(total_prefill_work / prefill_rate_chunked)
    max_gap_chunked = 1

    return {
        "ttft_unchunked": iters_unchunked, "max_gap_unchunked": max_gap_unchunked,
        "ttft_chunked": iters_chunked, "max_gap_chunked": max_gap_chunked,
    }


def run_d05():
    r = chunked_vs_unchunked(total_prefill_work=12, budget_per_iter=4, decode_share_per_iter=1)
    assert r["max_gap_chunked"] < r["max_gap_unchunked"]        # decode ITL improves
    assert r["ttft_chunked"] > r["ttft_unchunked"]              # but this prompt's own TTFT worsens
    assert r == {"ttft_unchunked": 3, "max_gap_unchunked": 3, "ttft_chunked": 4, "max_gap_chunked": 1}
    print(f"[D05] chunked prefill: decode max-gap {r['max_gap_unchunked']}->{r['max_gap_chunked']} "
          f"(ITL improves) but long-prompt TTFT {r['ttft_unchunked']}->{r['ttft_chunked']} "
          f"(worsens) -- not a free lunch  PASS")


# ============================================================
# [D06]-[D08] KV-cache calculator (reuses the KV-cache sibling tutorial's
# own per-token formula; no re-derivation of MQA/GQA/MLA cache-ratio math)
# ============================================================

def kv_bytes_per_token(n_layer, n_kv_head, d_head, bytes_per_elem):
    """m_token = 2 * N_layer * N_kv_head * d_head * bytes_per_elem
    (the leading 2 is for K and V). Same formula as the sibling KV-cache
    tutorial's §2.1 (per-token slice of its L_ctx-scaled cache formula)."""
    return 2 * n_layer * n_kv_head * d_head * bytes_per_elem


def block_alloc_slots(lengths, block_size):
    """logical slots = sum of true lengths; allocated slots round EACH
    request up to a whole number of blocks (internal fragmentation in the
    last, partially-filled block)."""
    logical = sum(lengths)
    allocated = sum(block_size * math.ceil(L / block_size) for L in lengths)
    return logical, allocated


def physical_blocks_no_sharing(lengths, block_size):
    return sum(math.ceil(L / block_size) for L in lengths)


def shared_prefix_blocks(token_seqs, block_size):
    """Number of COMPLETE blocks shared as an identical prefix across EVERY
    sequence in `token_seqs` -- an exact, block-by-block token comparison,
    not an assumed/hand-picked prefix length. Sharing can only happen at
    whole-block granularity: a common prefix shorter than one block shares
    nothing, and comparison stops at the first block where any sequence's
    content diverges (or where the shortest sequence runs out)."""
    if len(token_seqs) < 2:
        return 0
    min_len = min(len(s) for s in token_seqs)
    n_blocks = min_len // block_size
    shared = 0
    for b in range(n_blocks):
        lo, hi = b * block_size, (b + 1) * block_size
        block0 = token_seqs[0][lo:hi]
        if all(s[lo:hi] == block0 for s in token_seqs[1:]):
            shared += 1
        else:
            break
    return shared


def physical_blocks_with_sharing(token_seqs, block_size):
    """B_physical = B_no_share - (n-1) * B_shared: every request AFTER the
    first reuses the B_shared blocks of common prefix instead of allocating
    its own copy, so the saving scales with (n-1), not with a fixed constant.
    Returns (physical_blocks, shared_blocks, no_share_blocks)."""
    no_share = physical_blocks_no_sharing([len(s) for s in token_seqs], block_size)
    b_shared = shared_prefix_blocks(token_seqs, block_size)
    n = len(token_seqs)
    physical = no_share - (n - 1) * b_shared
    return physical, b_shared, no_share


def kv_bytes_under_tp(total_bytes, n_kv_head, tp_degree):
    """Only divide by TP degree when KV heads are evenly sharded across
    ranks. Otherwise this is an outright modeling error (replicated heads,
    uneven layouts, etc. need their own accounting)."""
    if n_kv_head % tp_degree != 0:
        raise ValueError(
            f"{n_kv_head} KV heads not evenly divisible by TP degree {tp_degree}: "
            "cannot assume a uniform per-rank split")
    return total_bytes / tp_degree


def run_d06_d07_d08():
    # [D06] per-token bytes, synthetic config
    m_token = kv_bytes_per_token(n_layer=2, n_kv_head=2, d_head=4, bytes_per_elem=2)
    # 2 (K,V) * 2 (layers) * 2 (kv heads) * 4 (head dim) * 2 (bytes/elem) = 64 bytes/token
    assert m_token == 64

    # [D07] block allocator: logical vs allocated slots (and bytes)
    lengths = [1, 5]
    logical_slots, allocated_slots = block_alloc_slots(lengths, block_size=4)
    assert (logical_slots, allocated_slots) == (6, 12)
    logical_bytes = logical_slots * m_token
    allocated_bytes = allocated_slots * m_token
    assert (logical_bytes, allocated_bytes) == (384, 768)
    assert allocated_bytes > logical_bytes                 # internal fragmentation is real

    # [D08] prefix-sharing block count -- computed from REAL token sequences
    # (not a hand-picked constant): two requests share tokens [1,2,3,4] as a
    # complete first block (block_size=4), then diverge.
    shared_prefix = [1, 2, 3, 4]
    seq_a = shared_prefix + [5, 6]          # length 6
    seq_b = shared_prefix + [7, 8, 9]       # length 7
    no_share = physical_blocks_no_sharing([len(seq_a), len(seq_b)], block_size=4)
    assert no_share == 4                                   # ceil(6/4)+ceil(7/4) = 2+2
    with_share, shared_blocks, no_share_check = physical_blocks_with_sharing([seq_a, seq_b], block_size=4)
    assert no_share_check == no_share
    assert shared_blocks == 1               # verified token-by-token, not asserted
    assert with_share == 3                  # 4 - (2-1)*1 = 3

    # a common prefix SHORTER than one block shares nothing (block-granularity only)
    seq_c = [1, 2, 3] + [10, 11, 12, 13]     # only 3 tokens overlap with seq_a's start
    seq_d = [1, 2, 3] + [20, 21, 22, 23, 24]
    physical_partial, shared_partial, no_share_partial = physical_blocks_with_sharing(
        [seq_c, seq_d], block_size=4)
    assert shared_partial == 0
    assert physical_partial == no_share_partial

    # three requests sharing the SAME full first block: the saving scales
    # as (n-1)*B_shared, not a fixed constant
    seq_e = shared_prefix + [30]
    seq_f = shared_prefix + [31, 32]
    physical_3, shared_3, no_share_3 = physical_blocks_with_sharing(
        [seq_a, seq_e, seq_f], block_size=4)
    assert shared_3 == 1
    assert physical_3 == no_share_3 - 2 * shared_3          # (n-1) = 2 requests reuse the block

    # TP-division caveat
    assert kv_bytes_under_tp(1024.0, n_kv_head=8, tp_degree=4) == 256.0   # 8 % 4 == 0, fine
    try:
        kv_bytes_under_tp(1024.0, n_kv_head=2, tp_degree=4)
        raise AssertionError("must reject uneven KV-head / TP-degree split")
    except ValueError:
        pass

    print(f"[D06] KV bytes/token (L=2,kv_head=2,d_head=4,fp16)={m_token}B; "
          f"[D07] block alloc lengths=[1,5],P=4: logical={logical_slots} slots/{logical_bytes}B, "
          f"allocated={allocated_slots} slots/{allocated_bytes}B; "
          f"[D08] prefix-sharing (real token-sequence match) blocks {no_share}->{with_share}, "
          f"sub-block overlap shares 0, 3-way sharing scales as (n-1)*B_shared, "
          f"TP-divisibility guard enforced  PASS")


# ============================================================
# [D09] (optional) roofline ridge point -- an upper-bound number, NOT a
# latency predictor
# ============================================================

def ridge_point(peak_flops_per_s, hbm_bytes_per_s):
    """I* = peak FLOP/s / HBM byte/s. Purely a units ratio used to compare
    against a kernel's own arithmetic intensity; it bounds a BEST CASE, it
    does not predict actual latency (real kernels sit below the roofline)."""
    assert peak_flops_per_s > 0 and hbm_bytes_per_s > 0
    return peak_flops_per_s / hbm_bytes_per_s


def run_d09():
    i_star = ridge_point(peak_flops_per_s=312e12, hbm_bytes_per_s=1.55e12)  # illustrative A100-class numbers
    assert i_star > 0
    # a decode step's arithmetic intensity in a small-batch regime is typically
    # far below any realistic ridge point; this demo only checks the ratio is
    # well-formed, not that it "predicts" any specific latency
    small_batch_intensity = 8.0     # illustrative FLOPs/byte, well under I*
    assert small_batch_intensity < i_star
    print(f"[D09] roofline ridge point I* = {i_star:.1f} FLOPs/byte (upper-bound reference only)  PASS")


def main():
    run_d01()
    run_d02()
    run_d03()
    run_d04()
    run_d05()
    run_d06_d07_d08()
    run_d09()
    print("\nall inference serving sanity checks passed")


if __name__ == "__main__":
    main()
