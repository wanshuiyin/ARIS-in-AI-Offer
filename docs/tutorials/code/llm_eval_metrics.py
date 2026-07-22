"""
LLM Evaluation & Benchmarking - minimal runnable implementation
================================================================

Pure-stdlib + numpy/scipy (Python 3.9+), runs in a few seconds on CPU.
Pairs with: docs/tutorials/llm_evaluation_benchmarking_tutorial.md (concept reference).

    [D01] pass@k: high-variance direct estimate vs unbiased combinatorial
          estimate (Chen et al.) vs biased plug-in estimate
    [D02] bootstrap CI: paired per-item difference, degenerate constant array,
          brute-force enumeration check, Wilson interval vs naive percentile
          bootstrap at a boundary proportion, and the multi-judge-vote /
          prompt-cluster independence trap
    [D03] Bradley-Terry MLE: closed-form 2-player case, standard-error scaling
          under count multiplication, cyclic (intransitive) data, complete
          separation detection, disconnected-graph detection
    [D04] position bias: equal-quality responses under a positional logit
          shift, order-balancing recovers the marginal win rate, and
          swap-inconsistency exposes what a deterministic "always pick first"
          judge hides behind that balanced marginal

Run:
    python3 llm_eval_metrics.py
"""
import itertools
import math
import sys
from math import comb, log

import numpy as np
from scipy import optimize


# ==========================================================================
# [D01] pass@k: three estimators, not two
# ==========================================================================
# Teaching note: "run exactly k samples, check if >=1 passes" is UNBIASED
# (its expectation is exactly the true pass@k) but high-variance (a single
# Bernoulli draw per problem). The common BIASED estimator is the plug-in
# form 1-(1-c/n)^k, which substitutes empirical accuracy into the population
# formula. The Chen et al. combinatorial estimator is unbiased AND lower
# variance because it averages over all C(n,k) size-k subsets of n>=k
# independently drawn samples instead of using only one subset.

def _validate_pass_at_k_inputs(n, c, k):
    """Shared validation for the n/c/k-based pass@k estimators below. n is
    the number of Monte Carlo samples actually generated, c is how many of
    them passed, and k is the estimand's allowed-attempts budget -- silently
    accepting out-of-range values would otherwise produce nonsense statistics
    (a negative pass@k for c<0, a silently-wrong 1.0 for c>n, or a
    ZeroDivisionError deep inside pass_at_k_plugin for n=0)."""
    for name, val in (("n", n), ("c", c), ("k", k)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {val!r}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got n={n}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")
    if not (0 <= c <= n):
        raise ValueError(f"c must satisfy 0 <= c <= n, got c={c}, n={n}")


def pass_at_k_true(p, k):
    """True pass@k for a FIXED per-problem success probability p under a
    fixed sampling policy (temperature/top-p/max tokens all fixed)."""
    if not (isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k >= 1):
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must satisfy 0 <= p <= 1, got p={p}")
    return 1 - (1 - p) ** k


def pass_at_k_unbiased(n, c, k):
    """Chen et al. unbiased, lower-variance estimator from n>=k samples with
    c correct. Convention: if fewer than k samples are wrong (n-c < k), the
    estimate is 1 (every size-k subset necessarily contains a correct one)."""
    _validate_pass_at_k_inputs(n, c, k)
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)


def pass_at_k_plugin(n, c, k):
    """Common BIASED plug-in estimator: substitute p_hat=c/n into the
    population formula. Biased DOWNWARD for k>=2 because 1-(1-p)^k is
    concave in p (Jensen's inequality applied to the random variable c/n)."""
    _validate_pass_at_k_inputs(n, c, k)
    return 1 - (1 - c / n) ** k


def binomial_pmf(n, p):
    return {c: comb(n, c) * p ** c * (1 - p) ** (n - c) for c in range(n + 1)}


# ==========================================================================
# [D02] bootstrap CI: paired design, degeneracy, and cluster dependence
# ==========================================================================

def percentile_ci(values, alpha=0.05):
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def wilson_ci(x, n, z=1.96):
    """Wilson score interval for a binomial proportion x/n -- does not
    degenerate to a zero-width interval at x=n like a naive percentile
    bootstrap of an all-successes sample does."""
    phat = x / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return center - margin, center + margin


# ==========================================================================
# [D03] Bradley-Terry MLE: identifiability, separation, non-transitivity
# ==========================================================================
# comparisons: dict {(i, j): (wins_i_over_j, wins_j_over_i)} for i < j.
# Player 0's theta is fixed at 0 during optimization (removes the additive
# non-identifiability); the caller may re-center to sum-zero afterward.

def _validate_comparisons(comparisons, n_players):
    """Validate a comparisons dict before any BT computation. Keys must be
    (i, j) integer pairs with i < j and both indices in range(n_players)
    (rejects self-comparisons and out-of-range player ids); counts must be
    finite, non-negative, and sum to a positive total (a (0, 0) entry
    carries no information and is almost certainly a bug, so it is
    rejected rather than silently treated as "not an edge")."""
    if not (isinstance(n_players, (int, np.integer)) and n_players >= 2):
        raise ValueError(f"n_players must be an integer >= 2, got {n_players!r}")
    for (i, j), (wij, wji) in comparisons.items():
        if isinstance(i, bool) or isinstance(j, bool) or not (
                isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer))):
            raise ValueError(f"player indices must be integers, got key {(i, j)!r}")
        if not (0 <= i < n_players and 0 <= j < n_players):
            raise ValueError(
                f"player indices {(i, j)} out of range for n_players={n_players}")
        if i == j:
            raise ValueError(f"self-comparison is not allowed: key {(i, j)}")
        if i >= j:
            raise ValueError(f"comparisons keys must satisfy i < j, got {(i, j)}")
        if not (np.isfinite(wij) and np.isfinite(wji)):
            raise ValueError(f"non-finite comparison count at {(i, j)}: {(wij, wji)}")
        if wij < 0 or wji < 0:
            raise ValueError(f"negative comparison count at {(i, j)}: {(wij, wji)}")
        if wij + wji <= 0:
            raise ValueError(
                f"comparison {(i, j)} has zero total games (0, 0), which "
                "carries no information; remove it from the input")


def _is_connected(comparisons, n_players):
    """Undirected weak-connectivity check. Assumes comparisons has already
    been validated (see _validate_comparisons), so every entry has
    wij + wji > 0 and is a genuine edge."""
    adj = {i: set() for i in range(n_players)}
    for (i, j), (wij, wji) in comparisons.items():
        if wij + wji > 0:
            adj[i].add(j)
            adj[j].add(i)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n_players


def _reachable(start, adj, n_players):
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def detect_complete_separation(comparisons, n_players):
    """Full strong-connectivity check on the directed winner-to-loser graph
    (edge i -> j iff player i has at least one recorded win over player j).
    Under the standard unregularized Bradley-Terry model, a finite MLE
    exists (unique up to the usual additive shift) if and only if this
    directed graph is STRONGLY CONNECTED: every non-trivial partition of
    the players must have at least one win crossing in each direction. When
    it is not strongly connected, the negative log-likelihood's infimum is
    not attained at any finite parameter vector -- the optimal parameters
    diverge to +/-infinity along the direction separating the strongly
    connected components (the supremum of the likelihood is finite, but no
    finite parameter vector attains it).

    For n_players == 2 this collapses to the familiar "one player never
    lost a single comparison" rule, but that pairwise rule does NOT
    generalize to n >= 3: e.g. A beats B 10-0 while A:C and B:C are both
    5-5 is NOT complete separation (C's split record against both A and B
    makes the whole graph strongly connected, hence a finite MLE exists),
    even though the isolated A-vs-B pair looks exactly like the classic
    separation pattern.

    Returns a list of the strongly connected components (each a sorted
    list of player indices), only when there is more than one component
    (i.e. separation is present); returns an empty list when the graph is
    strongly connected (no separation). Assumes comparisons has already
    been validated (see _validate_comparisons).
    """
    out_adj = {i: set() for i in range(n_players)}
    for (i, j), (wij, wji) in comparisons.items():
        if wij > 0:
            out_adj[i].add(j)
        if wji > 0:
            out_adj[j].add(i)
    in_adj = {i: set() for i in range(n_players)}
    for u, vs in out_adj.items():
        for v in vs:
            in_adj[v].add(u)

    components, assigned = [], set()
    for u in range(n_players):
        if u in assigned:
            continue
        scc = _reachable(u, out_adj, n_players) & _reachable(u, in_adj, n_players)
        components.append(sorted(scc))
        assigned |= scc

    return components if len(components) > 1 else []


def bt_negloglik(theta_free, comparisons, n_players, ridge=0.0):
    theta = np.concatenate(([0.0], theta_free))
    nll = 0.0
    for (i, j), (wij, wji) in comparisons.items():
        d = theta[i] - theta[j]
        # log sigma(d)  = -log(1+exp(-d)) = -logaddexp(0, -d)
        # log sigma(-d) = -log(1+exp(d))  = -logaddexp(0, d)
        nll -= wij * (-np.logaddexp(0, -d))
        nll -= wji * (-np.logaddexp(0, d))
    if ridge > 0:
        # Penalize the SUM-ZERO-CENTERED theta, not theta_free directly.
        # theta_free implicitly pins player 0's parameter at 0 purely as an
        # optimization device to remove the additive non-identifiability;
        # penalizing theta_free alone (as if player 0 were a fixed
        # reference point) makes the ridge ESTIMATE depend on which player
        # happens to be numbered 0. Centering first makes the penalty (and
        # the resulting estimate) invariant to that arbitrary choice.
        theta_centered = theta - theta.mean()
        nll += ridge * np.sum(theta_centered ** 2)
    return nll


def bt_mle(comparisons, n_players, ridge=0.0, allow_separation=False, x0=None):
    """Batch Bradley-Terry MLE with theta_0 fixed at 0 during optimization,
    then re-centered to sum-zero. Raises on invalid input (see
    _validate_comparisons), on a disconnected comparison graph (relative
    strength across components is not identifiable), and, unless ridge>0
    or the caller explicitly opts in, on detected complete separation (no
    finite unregularized MLE exists). Also raises if the optimizer fails to
    converge to a finite solution, rather than silently returning whatever
    scipy last tried.

    x0: optional custom starting point for theta_free (shape
    (n_players-1,)); defaults to all-zeros. Mainly useful for tests where
    starting exactly at the true solution would make the optimizer's
    correctness untestable.
    """
    _validate_comparisons(comparisons, n_players)
    if not (np.isfinite(ridge) and ridge >= 0):
        raise ValueError(f"ridge must be a finite non-negative real number, got {ridge!r}")

    if not _is_connected(comparisons, n_players):
        raise ValueError(
            "comparison graph is disconnected: relative strength across "
            "components is not identifiable from this data alone")
    flagged = detect_complete_separation(comparisons, n_players)
    if flagged and not (ridge > 0) and not allow_separation:
        raise ValueError(
            f"complete separation detected: the winner-to-loser graph "
            f"splits into strongly connected components {flagged} with no "
            "win flowing back into an earlier component from a later one; "
            "no finite unregularized MLE exists; pass ridge>0 or "
            "allow_separation=True")

    if x0 is None:
        x0 = np.zeros(n_players - 1)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (n_players - 1,):
            raise ValueError(
                f"x0 must have shape ({n_players - 1},), got {x0.shape}")

    res = optimize.minimize(
        bt_negloglik, x0, args=(comparisons, n_players, ridge), method="BFGS")
    theta = np.concatenate(([0.0], res.x))
    theta = theta - theta.mean()
    if not res.success or not np.isfinite(res.fun) or not np.all(np.isfinite(theta)):
        raise RuntimeError(
            "BT MLE optimization did not converge to a finite solution: "
            f"success={res.success}, fun={res.fun}, message={res.message!r}")
    return theta, res


def bt_pairwise_se(n_ij, p_ij):
    """Standard error of theta_i - theta_j for a pair observed in ISOLATION
    (no other comparisons touching i or j): the log-likelihood restricted to
    d=theta_i-theta_j is exactly a binomial-logistic model with n_ij trials,
    Fisher information n_ij * p * (1-p)."""
    info = n_ij * p_ij * (1 - p_ij)
    return 1.0 / math.sqrt(info)


# ==========================================================================
# [D04] position bias: positional logit shift + swap consistency
# ==========================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


POSITION_BIAS_B = math.log(7 / 3)  # calibrated so P(A wins | shown first) = 0.7


def p_a_wins(delta, position_a):
    """position_a: +1 if A is shown first, -1 if A is shown second.
    delta is the TRUE quality gap (0 for equal-quality responses)."""
    return sigmoid(delta + POSITION_BIAS_B * position_a)


def order_balanced_marginal_win_rate(judge_fn, x, y):
    """Directly compute, for the ACTUAL judge_fn (not a separate smooth
    logistic model of it), the marginal win rate for x when the pair (x, y)
    is order-balanced: judged once with x shown first and once with x
    shown second, each counted with weight 0.5. This is what "order
    balancing" means operationally, and it is what hides a purely
    positional judge like always_pick_first behind an innocuous-looking
    marginal of 0.5, even though the judge never once looked at content."""
    win_when_first = judge_fn(x, y) == x     # x shown first
    win_when_second = judge_fn(y, x) == x    # x shown second
    return 0.5 * win_when_first + 0.5 * win_when_second


def swap_inconsistency_rate(judge_fn, pairs):
    """For each (X, Y) pair, ask the judge in original order (X first, Y
    second) and swapped order (Y first, X second); report the fraction of
    pairs where the WINNER IDENTITY flips between the two orders. A judge
    that only tracks true quality gives rate 0; a position-driven judge can
    give rate 1 even though its balanced marginal win rate looks like 0.5."""
    flips = 0
    for x, y in pairs:
        winner_original = judge_fn(x, y)          # x shown first
        winner_swapped = judge_fn(y, x)           # y shown first
        if winner_original != winner_swapped:
            flips += 1
    return flips / len(pairs)


def main():
    if not __debug__:
        raise SystemExit("run without -O: the checks below are assert-based")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    # ---------------------------------------------------------------- D01
    p, k, n = 0.2, 2, 5
    true_pk = pass_at_k_true(p, k)
    assert abs(true_pk - 0.36) < 1e-12                       # 1 - 0.8^2

    pmf = binomial_pmf(n, p)
    e_unbiased = sum(pass_at_k_unbiased(n, c, k) * w for c, w in pmf.items())
    e_plugin = sum(pass_at_k_plugin(n, c, k) * w for c, w in pmf.items())
    assert abs(e_unbiased - 0.36) < 1e-9                     # exactly unbiased
    assert abs(e_plugin - 0.328) < 1e-9                      # biased downward
    bias_pred = -p * (1 - p) / n                             # closed form for k=2
    assert abs((e_plugin - true_pk) - bias_pred) < 1e-9

    # "run exactly k times, record >=1 success": unbiased single Bernoulli obs
    var_naive = true_pk * (1 - true_pk)
    assert abs(var_naive - 0.2304) < 1e-9

    # Chen et al.'s claim is not just "unbiased" but "unbiased AND lower
    # variance than the naive k-sample estimator, for the SAME n". Actually
    # compute Var[unbiased estimator] over repeated draws of n samples
    # (c ~ Binomial(n, p)) via the same pmf used for e_unbiased above, and
    # check it against var_naive rather than only asserting the mean matches.
    e_unbiased_sq = sum(pass_at_k_unbiased(n, c, k) ** 2 * w for c, w in pmf.items())
    var_unbiased = e_unbiased_sq - e_unbiased ** 2
    assert var_unbiased < var_naive                          # Rao-Blackwell: strictly lower here
    assert abs(var_unbiased - 0.08448) < 1e-4

    # hand example: n=10, c=3, k=5 -> 1 - C(7,5)/C(10,5) = 11/12
    assert abs(pass_at_k_unbiased(10, 3, 5) - 11 / 12) < 1e-12

    # edge cases
    assert abs(pass_at_k_unbiased(6, 4, 1) - 4 / 6) < 1e-12   # k=1 -> c/n
    assert pass_at_k_unbiased(6, 0, 3) == 0.0                 # c=0 -> 0
    assert pass_at_k_unbiased(6, 5, 3) == 1.0                 # n-c=1<k -> 1
    try:
        pass_at_k_unbiased(3, 1, 5)
        raise AssertionError("expected ValueError for k>n")
    except ValueError:
        pass

    # illegal n/c/k must raise, not silently produce a negative pass@k, a
    # falsely-perfect 1.0 for c>n, or (pre-fix) a ZeroDivisionError for n=0
    for bad_call in (
        lambda: pass_at_k_unbiased(5, -1, 2),      # negative c
        lambda: pass_at_k_unbiased(5, 6, 2),       # c > n
        lambda: pass_at_k_unbiased(0, 0, 1),       # n < 1
        lambda: pass_at_k_unbiased(5, 2, 0),       # k < 1
        lambda: pass_at_k_unbiased(5.5, 2, 2),     # non-integer n
        lambda: pass_at_k_plugin(0, 0, 2),         # n=0 -> used to ZeroDivisionError
        lambda: pass_at_k_plugin(5, -1, 2),        # negative c
        lambda: pass_at_k_plugin(5, 6, 2),         # c > n
        lambda: pass_at_k_true(0.2, 0),            # k < 1
        lambda: pass_at_k_true(1.5, 2),            # p out of [0, 1]
    ):
        try:
            bad_call()
            raise AssertionError(f"expected ValueError for invalid pass@k input {bad_call}")
        except ValueError:
            pass

    # multi-problem averaging: per-problem first, THEN average -- not pooled
    per_problem_avg = (pass_at_k_unbiased(5, 1, 2) + pass_at_k_unbiased(5, 4, 2)) / 2
    pooled_wrong = pass_at_k_unbiased(10, 5, 2)               # c and n summed first
    assert abs(per_problem_avg - 0.7) < 1e-12
    assert abs(pooled_wrong - 0.7778) < 1e-3
    assert abs(per_problem_avg - pooled_wrong) > 0.05         # nonlinear transform bites

    print(f"[D01] pass@k(p=0.2,k=2)=0.360 true; E[unbiased]={e_unbiased:.3f} "
          f"(exact) vs E[plugin]={e_plugin:.3f} (biased {e_plugin - true_pk:+.3f}); "
          f"Var[unbiased]={var_unbiased:.5f} < Var[naive k-sample]={var_naive:.4f}; "
          f"n=10,c=3,k=5 -> {pass_at_k_unbiased(10, 3, 5):.4f}=11/12; "
          f"per-problem avg={per_problem_avg:.3f} != pooled={pooled_wrong:.4f}  PASS")

    # ---------------------------------------------------------------- D02
    rng = np.random.default_rng(0)
    n_items = 30
    A = rng.normal(0.70, 0.10, n_items)
    B = rng.normal(0.65, 0.10, n_items)
    d = A - B
    n_boot = 4000
    idx_matrix = rng.integers(0, n_items, size=(n_boot, n_items))
    boot_paired = np.array([A[idx].mean() - B[idx].mean() for idx in idx_matrix])
    boot_direct = np.array([d[idx].mean() for idx in idx_matrix])
    # algebraic identity (same bootstrap indices -> identical means): check
    # against an absolute tolerance ONLY (rtol=0), since the default rtol
    # would otherwise dominate atol here and mask a real discrepancy
    assert np.allclose(boot_paired, boot_direct, rtol=0, atol=1e-9)

    # degenerate constant array
    const_data = np.full(20, 0.07)
    idx_matrix2 = rng.integers(0, 20, size=(2000, 20))
    const_means = np.array([const_data[idx].mean() for idx in idx_matrix2])
    lo_c, hi_c = percentile_ci(const_means)
    assert lo_c == hi_c
    assert abs(lo_c - 0.07) < 1e-9

    # brute-force enumeration: mean of ALL 3^3=27 bootstrap-sample means
    # equals the original sample mean exactly (bootstrap-mean identity)
    d_small = np.array([0.2, -0.1, 0.4])
    all_means = [d_small[list(combo)].mean()
                 for combo in itertools.product(range(3), repeat=3)]
    assert len(all_means) == 27
    assert abs(sum(all_means) / len(all_means) - d_small.mean()) < 1e-12
    assert abs(d_small.mean() - 1 / 6) < 1e-12

    # all-correct edge case: naive percentile bootstrap collapses to a point,
    # Wilson score interval does not
    all_correct = np.ones(20)
    idx_matrix3 = rng.integers(0, 20, size=(2000, 20))
    ac_means = np.array([all_correct[idx].mean() for idx in idx_matrix3])
    lo_ac, hi_ac = percentile_ci(ac_means)
    assert lo_ac == hi_ac == 1.0
    lo_w, hi_w = wilson_ci(20, 20)
    assert abs(lo_w - 0.839) < 0.005

    # multi-judge-vote / prompt-cluster independence trap: 4 prompts, each
    # duplicated by 5 (perfectly agreeing) judge votes. Treating all 20 votes
    # as independent draws understates uncertainty; resampling by PROMPT
    # cluster (then keeping each prompt's 5 duplicate votes together) is honest.
    prompt_values = np.array([0.9, 0.5, 0.5, 0.1])
    votes_flat = np.repeat(prompt_values, 5)                  # 20 "votes"
    idx_naive = rng.integers(0, 20, size=(5000, 20))
    naive_means = np.array([votes_flat[idx].mean() for idx in idx_naive])
    lo_n, hi_n = percentile_ci(naive_means)
    idx_cluster = rng.integers(0, 4, size=(5000, 4))          # resample PROMPTS
    cluster_means = np.array([prompt_values[idx].mean() for idx in idx_cluster])
    lo_cl, hi_cl = percentile_ci(cluster_means)
    width_naive, width_cluster = hi_n - lo_n, hi_cl - lo_cl
    assert width_cluster > 1.5 * width_naive                  # cluster CI much wider

    print(f"[D02] paired==direct bootstrap: "
          f"{np.allclose(boot_paired, boot_direct, rtol=0, atol=1e-9)}; "
          f"constant-array CI=[{lo_c:.4f},{hi_c:.4f}]; 27-sample enum mean="
          f"{sum(all_means)/len(all_means):.4f}=1/6; all-correct naive=[1.0,1.0] vs "
          f"Wilson=[{lo_w:.3f},{hi_w:.3f}]; cluster CI width {width_cluster:.3f} > "
          f"naive width {width_naive:.3f}  PASS")

    # ---------------------------------------------------------------- D03
    # closed-form 2-player case: A beats B 75/25 -> theta_A-theta_B = log(3)
    comparisons_2p = {(0, 1): (75, 25)}
    theta_2p, res_2p = bt_mle(comparisons_2p, n_players=2)
    assert res_2p.success
    assert abs((theta_2p[0] - theta_2p[1]) - log(3)) < 1e-6
    assert abs(theta_2p[0] - log(3) / 2) < 1e-6
    assert abs(theta_2p[1] + log(3) / 2) < 1e-6
    pred_win = sigmoid(theta_2p[0] - theta_2p[1])
    assert abs(pred_win - 0.75) < 1e-6

    # scaling: multiply win/loss counts by 10 -> same point estimate, smaller SE
    se_orig = bt_pairwise_se(100, 0.75)
    se_scaled = bt_pairwise_se(1000, 0.75)
    comparisons_2p_scaled = {(0, 1): (750, 250)}
    theta_2p_scaled, res_2p_scaled = bt_mle(comparisons_2p_scaled, n_players=2)
    assert res_2p_scaled.success
    assert abs((theta_2p_scaled[0] - theta_2p_scaled[1]) - log(3)) < 1e-6
    assert se_scaled < se_orig
    assert abs(se_scaled / se_orig - 1 / math.sqrt(10)) < 1e-6

    # cyclic (intransitive) data: A>B>C>A each 2:1 (key (0,2)=(1,2) encodes
    # "C beats A 2:1" under the i<j key convention) -> scalar model forces
    # equal strengths, predicting ~0.5 for every pair despite the 2:1 cycles.
    # Start the optimizer AWAY from the all-zeros initial guess: zeros
    # happen to be exactly the symmetric solution here, so starting there
    # would let a broken objective/gradient pass by sheer luck of never
    # having to move. res_cyc.success is also checked explicitly rather
    # than only checking theta_cyc for finiteness.
    comparisons_cyc = {(0, 1): (2, 1), (1, 2): (2, 1), (0, 2): (1, 2)}
    theta_cyc, res_cyc = bt_mle(comparisons_cyc, n_players=3, x0=np.array([1.0, -1.0]))
    assert res_cyc.success
    assert max(theta_cyc) - min(theta_cyc) < 1e-4
    for (i, j) in comparisons_cyc:
        assert abs(sigmoid(theta_cyc[i] - theta_cyc[j]) - 0.5) < 1e-3

    # complete separation (2-player): A beats B 10-0, no losses -> flagged,
    # no finite unregularized MLE; with ridge>0 the estimate is finite AND
    # the optimizer must actually report success (not just finite numbers).
    comparisons_sep = {(0, 1): (10, 0)}
    assert detect_complete_separation(comparisons_sep, n_players=2) == [[0], [1]]
    try:
        bt_mle(comparisons_sep, n_players=2)
        raise AssertionError("expected ValueError for complete separation")
    except ValueError:
        pass
    theta_reg, res_reg = bt_mle(comparisons_sep, n_players=2, ridge=1.0)
    assert res_reg.success
    assert np.all(np.isfinite(theta_reg))

    # generalization check (the actual P1 bug): A:B=10:0 in isolation looks
    # exactly like the separation case above, but with A:C=5:5 and B:C=5:5
    # added, the winner-to-loser graph IS strongly connected (C's split
    # record against both A and B provides a path back), so a finite
    # unregularized MLE exists and must NOT be flagged or rejected.
    comparisons_not_sep = {(0, 1): (10, 0), (0, 2): (5, 5), (1, 2): (5, 5)}
    assert detect_complete_separation(comparisons_not_sep, n_players=3) == []
    theta_not_sep, res_not_sep = bt_mle(comparisons_not_sep, n_players=3)
    assert res_not_sep.success
    assert np.all(np.isfinite(theta_not_sep))

    # a genuine multi-player separation that the naive "one pair only-wins"
    # rule and the graph-based rule both catch: a strict win chain 0->1->2
    # with no reverse wins anywhere splits into 3 singleton components.
    comparisons_chain_sep = {(0, 1): (10, 0), (1, 2): (10, 0)}
    assert detect_complete_separation(comparisons_chain_sep, n_players=3) == [[0], [1], [2]]
    try:
        bt_mle(comparisons_chain_sep, n_players=3)
        raise AssertionError("expected ValueError for chain complete separation")
    except ValueError:
        pass

    # disconnected graph: {0,1} and {2,3} never compared -> must raise
    comparisons_disc = {(0, 1): (5, 5), (2, 3): (5, 5)}
    try:
        bt_mle(comparisons_disc, n_players=4)
        raise AssertionError("expected ValueError for disconnected graph")
    except ValueError:
        pass

    # invalid comparisons input must raise, not silently corrupt the graph
    # (out-of-range id, self-comparison, negative count, zero-total count,
    # i>=j key ordering, non-integer id) or bypass separation via bad ridge
    for bad_call in (
        lambda: bt_mle({(0, 5): (3, 2)}, n_players=3),          # id out of range
        lambda: bt_mle({(1, 1): (3, 2)}, n_players=3),          # self-comparison
        lambda: bt_mle({(0, 1): (-1, 2)}, n_players=2),         # negative count
        lambda: bt_mle({(0, 1): (0, 0)}, n_players=2),          # zero-total count
        lambda: bt_mle({(1, 0): (3, 2)}, n_players=2),          # i >= j
        lambda: bt_mle({(0.5, 1): (3, 2)}, n_players=2),        # non-integer id
        lambda: bt_mle(comparisons_2p, n_players=2, ridge=-1.0),        # negative ridge
        lambda: bt_mle(comparisons_2p, n_players=2, ridge=float("nan")),  # NaN ridge
    ):
        try:
            bad_call()
            raise AssertionError(f"expected ValueError for invalid BT input {bad_call}")
        except ValueError:
            pass

    print(f"[D03] 2-player MLE: theta_A-theta_B={theta_2p[0]-theta_2p[1]:.6f}=log3, "
          f"pred_win={pred_win:.3f}; SE shrinks {se_orig:.4f}->{se_scaled:.4f} "
          f"(x10 counts, ratio {se_scaled/se_orig:.4f}=1/sqrt(10)); cyclic strengths "
          f"span={max(theta_cyc)-min(theta_cyc):.2e} (all ~0.5 pairwise); "
          f"2-player separation detected+regularized finite; 3-player A:B=10:0/"
          f"A:C=5:5/B:C=5:5 correctly NOT flagged (strongly connected); chain "
          f"separation 0->1->2 flagged; disconnected graph -> raises  PASS")

    # ---------------------------------------------------------------- D04
    p_first = p_a_wins(delta=0.0, position_a=1)
    p_second = p_a_wins(delta=0.0, position_a=-1)
    assert abs(p_first - 0.7) < 1e-9
    assert abs(p_second - 0.3) < 1e-9
    marginal = 0.5 * p_first + 0.5 * p_second
    assert abs(marginal - 0.5) < 1e-12

    def always_pick_first(x, y):
        return x                                              # x is shown first

    pairs = [("item1", "item2"), ("item3", "item4"), ("item5", "item6"),
             ("item7", "item8"), ("item9", "item10")]
    rate = swap_inconsistency_rate(always_pick_first, pairs)
    assert rate == 1.0

    # the claim "always-pick-first's order-balanced marginal is 0.5" is
    # about the deterministic judge itself, not about the separate smooth
    # logistic model (p_a_wins) used above -- compute it directly for the
    # actual judge_fn on a concrete pair.
    marginal_first_direct = order_balanced_marginal_win_rate(
        always_pick_first, "item1", "item2")
    assert marginal_first_direct == 0.5

    def quality_only_judge(x, y, quality):
        return x if quality[x] >= quality[y] else y

    # distinct qualities per item (never tied) so the judge below is a clean
    # position-INVARIANT baseline: it always favors the higher-quality item
    # regardless of which slot it is shown in
    quality = {name: q for q, name in enumerate(
        item for pair in pairs for item in pair)}
    consistent_rate = swap_inconsistency_rate(
        lambda x, y: quality_only_judge(x, y, quality), pairs)
    assert consistent_rate == 0.0

    print(f"[D04] position bias b=log(7/3)={POSITION_BIAS_B:.6f}: "
          f"P(A wins|first)={p_first:.1f}, P(A wins|second)={p_second:.1f}, "
          f"balanced marginal={marginal:.1f}; always-pick-first swap "
          f"inconsistency={rate:.1f} vs quality-only judge={consistent_rate:.1f}  PASS")

    print("\nall llm_eval sanity checks passed ✓")


if __name__ == "__main__":
    main()
