"""
World models: three toy experiments
===================================

Pairs with: docs/tutorials/world_models_tutorial.md
            (RSSM / JEPA / latent action models).

Three claims from the tutorial, each checked against an ANALYTIC answer:

  A. One KL term with a large weight collapses the RSSM posterior (§3.4):
     w -> 0, d -> 0, and the reconstruction loses the observation entirely.
     a Dreamer-style BALANCED KL (two stop-gradient branches; toy weights
     9 : 0.1, not the paper's) has a non-degenerate fixed point that is solvable
     in closed form.
  B. JEPA without a stop-gradient takes the shrink-together shortcut (§4.4):
     the encoder decays geometrically to zero. A stop-gradient / EMA teacher
     removes that shortcut — but it is NOT an anti-collapse theorem: the
     collapsed point is still a fixed point of the same dynamics.
  C. A latent action model on a gridworld recovers the four true actions up to
     a permutation of the codes (§5.1): one VQ assignment + centroid update
     gives exact reconstruction, purity 1 and I(K;A) = 2 bits on held-out
     start cells. This is a DETERMINISTIC COORDINATE LAM used to exhibit the
     identifiability question (actions are recovered only up to relabelling),
     not a pixel-level inverse-dynamics learner.

Everything is enumerated exactly — no sampling, no free bits, no frameworks.

Run:
    python world_models_toy.py      # CPU, single thread, a couple of seconds
"""
import math
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# A. RSSM: single KL vs balanced KL
# ---------------------------------------------------------------------------
# World: x_{t+1} = 0.5 x_t + a_t + eps,  eps in {-1,+1} balanced (enumerated).
# The deterministic path is FIXED at the truth, h = 0.5 x_t + a_t, so the
# residual the stochastic path must carry is exactly eps = o_{t+1} - h.  Three
# learnable scalars: q(z|h,o) = N(w eps, 1), p(z|h) = N(v, 1), decoder
# o_hat = h + d z.  Both expectations are integrated analytically:
#   L_rec = E_eps E_{z~q} (o_hat - o)^2 / 2 = [ (1 - d w)^2 + d^2 ] / 2
#           -- the d^2 is the posterior SAMPLING variance; dropping it removes
#              the only pressure that keeps d finite.
#   KL(N(mq,1) || N(mp,1)) = log(sp/sq) + (sq^2 + (mq-mp)^2)/(2 sp^2) - 1/2
#                          = (mq - mp)^2 / 2   for sq = sp = 1, so
#   E_eps KL(q||p) = E_{eps=±1} (w eps - v)^2 / 2 = (w^2 + v^2) / 2.
#
# ANALYTIC BALANCED FIXED POINT (stationarity of the two-path gradient).
# For L = L_rec + b_p KL(sg(q)||p) + b_q KL(q||sg(p)), b_p = 9, b_q = 0.1,
# autograd sees the prior only through the first KL and the posterior only
# through the second, so with MSE = 2 L_rec:
#   dL/dv = b_p (v - E[w eps]) = b_p v            => v* = 0
#   dL/dd = -w (1 - d w) + d                      => d = w (1 - d w)
#   dL/dw = -d (1 - d w) + b_q w                  => d (1 - d w) = b_q w
# Write A = 1 - d w.  The d-equation gives d = w A, hence d w = w^2 A and
# A = 1 - w^2 A, i.e.  A = 1/(1 + w^2)  and  d* = w / (1 + w^2).
# Substituting into the w-equation: w/(1+w^2)^2 = b_q w, so either w = 0 or
#   (1 + w^2)^2 = 1/b_q   =>   w*^2 = 1/sqrt(b_q) - 1 = sqrt(10) - 1,
# and MSE* = A^2 (1 + w^2) = 1/(1 + w^2) = sqrt(b_q) = sqrt(0.1).
# A single KL of weight b (b_p = b_q = b) needs (1 + w^2)^2 = 1/b instead,
# which has no real solution at b = 9: only w = d = 0 survives, MSE = 1.
BETA_PRIOR, BETA_POST, LR_A, STEPS_A = 9.0, 0.1, 0.05, 1000
EPS_SUPPORT = torch.tensor([-1.0, 1.0])


def kl_unit_gaussians(mu_q, mu_p):
    """KL(N(mu_q,1) || N(mu_p,1)), reduced from the general Gaussian formula."""
    return 0.5 * (mu_q - mu_p) ** 2


def rssm_terms(w, v, d):
    """Returns (L_rec, per-eps posterior mean, per-eps prior mean)."""
    mu_q = w * EPS_SUPPORT                                   # posterior mean N(w eps, 1)
    mu_p = v * torch.ones_like(EPS_SUPPORT)                  # prior mean, eps-independent
    rec = 0.5 * ((d * mu_q - EPS_SUPPORT) ** 2 + d ** 2).mean()
    return rec, mu_q, mu_p


def train_rssm(balanced):
    w, v, d = (torch.tensor(x, requires_grad=True) for x in (0.5, 1.0, 0.5))
    opt = torch.optim.SGD([w, v, d], lr=LR_A)                # synchronous updates
    for _ in range(STEPS_A):
        opt.zero_grad()
        rec, mu_q, mu_p = rssm_terms(w, v, d)
        if balanced:
            # two REAL gradient paths: prior-side branch and posterior-side branch
            loss = (rec
                    + BETA_PRIOR * kl_unit_gaussians(mu_q.detach(), mu_p).mean()
                    + BETA_POST * kl_unit_gaussians(mu_q, mu_p.detach()).mean())
        else:
            loss = rec + BETA_PRIOR * kl_unit_gaussians(mu_q, mu_p).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec, _, _ = rssm_terms(w, v, d)
    return w.item(), v.item(), d.item(), 2.0 * rec.item()    # MSE = 2 L_rec


def experiment_A():
    print("== A. RSSM: one KL collapses the posterior, the balanced KL does not ==")
    w, v, d, mse = train_rssm(balanced=False)
    print(f"  single KL (beta=9)   w={w:+.3e}  v={v:+.3e}  d={d:+.3e}  MSE={mse:.6f}")
    assert abs(w) < 1e-6 and abs(d) < 1e-6, (w, d)
    assert abs(v) < 1e-6, v
    assert abs(mse - 1.0) < 1e-6, mse                        # decoder ignores o: MSE = Var(eps) = 1

    w, v, d, mse = train_rssm(balanced=True)
    w2_star = 1.0 / math.sqrt(BETA_POST) - 1.0               # (1 + w^2)^2 = 1/b_q
    w_star, mse_star = math.sqrt(w2_star), math.sqrt(BETA_POST)
    d_star = w_star / (1.0 + w2_star)
    print(f"  balanced (9 / 0.1)   w={w:+.6f}  v={v:+.3e}  d={d:+.6f}  MSE={mse:.6f}")
    print(f"  analytic fixed point w={w_star:+.6f}  v=+0.000000  d={d_star:+.6f}  MSE={mse_star:.6f}")
    assert abs(v) < 1e-5, v
    assert abs(w - w_star) < 1e-5, (w, w_star)
    assert abs(d - d_star) < 1e-5, (d, d_star)
    assert abs(mse - mse_star) < 1e-5, (mse, mse_star)
    print("  PASS: beta=9 alone kills the stochastic path; balancing keeps MSE at sqrt(0.1)\n")


# ---------------------------------------------------------------------------
# B. JEPA: the shrink-together shortcut vs a stop-gradient teacher
# ---------------------------------------------------------------------------
# Enumerate independent s, eps in {-1,+1}: context x = s, target y = s + eps.
# Encoder E_w(u) = w u, predictor = identity.  E[x^2] = E[x y] = 1, E[y^2] = 2.
S_EPS = torch.tensor([[s, e] for s in (-1.0, 1.0) for e in (-1.0, 1.0)])
X_B, Y_B = S_EPS[:, 0], S_EPS[:, 0] + S_EPS[:, 1]


def experiment_B():
    print("== B. JEPA: shrink-together shortcut vs a stop-gradient/EMA teacher ==")
    lr, steps = 0.1, 100

    # --- shared weights on both branches: L = E(w x - w y)^2 / 2 = w^2 / 2
    w = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * (w * X_B - w * Y_B) ** 2).mean().backward()
        opt.step()
    w_shared = w.item()
    want = 0.9 ** steps                                      # exact: w_{n+1} = (1 - lr) w_n
    assert abs(w_shared - want) < 1e-12 * want, (w_shared, want)
    assert abs(w_shared ** 2 - 0.9 ** (2 * steps)) < 1e-12 * 0.9 ** (2 * steps)
    print(f"  no stop-grad: w_100={w_shared:.6e} (=0.9^100), context var={w_shared ** 2:.6e} (=0.9^200)")
    assert w_shared < 1e-4, "the shortcut should have shrunk the representation to nothing"

    # --- stop-gradient target branch with an EMA teacher b
    def run_ema(w0, b0):
        w = torch.tensor(w0, requires_grad=True)
        b = torch.tensor(b0)
        opt = torch.optim.SGD([w], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = (0.5 * (w * X_B - b.detach() * Y_B) ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                b = 0.9 * b + 0.1 * w                        # EMA teacher
        with torch.no_grad():
            loss = (0.5 * (w * X_B - b * Y_B) ** 2).mean().item()
        return w.item(), b.item(), loss

    # grad_w = w E[x^2] - b E[x y] = w - b, so w = b = 1 is EXACTLY stationary
    w_on, b_on, loss_on = run_ema(1.0, 1.0)
    assert abs(w_on - 1.0) < 1e-12 and abs(b_on - 1.0) < 1e-12, (w_on, b_on)
    assert abs(loss_on - 0.5) < 1e-12, loss_on                # = E[eps^2]/2, the unpredictable part
    assert abs(w_on ** 2 - 1.0) < 1e-12
    print(f"  stop-grad + EMA from w=b=1: w={w_on:.12f} b={b_on:.12f} loss={loss_on:.12f} (=0.5), context var=1")

    # ... but the collapsed point is still a fixed point of the same dynamics
    w_z, b_z, loss_z = run_ema(0.0, 0.0)
    assert abs(w_z) < 1e-12 and abs(b_z) < 1e-12 and abs(loss_z) < 1e-12, (w_z, b_z, loss_z)
    print(f"  stop-grad + EMA from w=b=0: w={w_z:.1e} b={b_z:.1e} loss={loss_z:.1e} — stays collapsed")
    print("  PASS: stop-grad removes the shrink-together shortcut, it does not forbid collapse\n")


# ---------------------------------------------------------------------------
# C. Coordinate LAM on a gridworld — exact recovery up to a code permutation
# ---------------------------------------------------------------------------
GRID_N, N_CODES = 5, 4
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))                   # up, down, left, right


def gridworld_transitions():
    """Interior start cells only, so every start has all four moves available.
    The split is BY START CELL: no val cell is ever seen during fitting."""
    starts = [(r, c) for r in range(1, GRID_N - 1) for c in range(1, GRID_N - 1)]
    tr, va, va_a = [], [], []
    for i, (r, c) in enumerate(starts):
        for a, (dr, dc) in enumerate(MOVES):
            (va if i % 3 == 2 else tr).append(([r, c], [r + dr, c + dc]))
            if i % 3 == 2:
                va_a.append(a)
    def pack(rows):
        return (torch.tensor([s for s, _ in rows], dtype=torch.get_default_dtype()),
                torch.tensor([p for _, p in rows], dtype=torch.get_default_dtype()))
    return pack(tr), pack(va) + (torch.tensor(va_a),)


def fit_lam(states, next_states):
    """Receives ONLY (s, s') pairs. Returns the codebook. No action labels here."""
    delta = next_states - states                             # encoder
    uniq = torch.unique(delta, dim=0)
    # farthest-first init: pick the four most spread-out distinct deltas, halved
    chosen = [int(torch.argmax(uniq.norm(dim=1)))]
    while len(chosen) < N_CODES:
        dist = torch.cdist(uniq, uniq[chosen]).min(dim=1).values
        dist[torch.tensor(chosen)] = -1.0
        chosen.append(int(torch.argmax(dist)))
    codes = 0.5 * uniq[chosen]
    print(f"  codebook init (farthest-first, x1/2): {[[round(x, 2) for x in c] for c in codes.tolist()]}")
    k = torch.cdist(delta, codes).argmin(dim=1)              # one VQ assignment
    for j in range(N_CODES):                                 # one centroid update
        codes[j] = delta[k == j].mean(dim=0)
    return codes


def mutual_information_bits(k, a, n_k, n_a):
    joint = torch.zeros(n_k, n_a)
    for ki, ai in zip(k.tolist(), a.tolist()):
        joint[ki, ai] += 1.0
    joint /= joint.sum()
    pk, pa = joint.sum(dim=1, keepdim=True), joint.sum(dim=0, keepdim=True)
    nz = joint > 0
    return float((joint[nz] * torch.log2(joint[nz] / (pk * pa).expand_as(joint)[nz])).sum())


def experiment_C():
    print("== C. Coordinate LAM: four codes recover four actions up to permutation ==")
    (tr_s, tr_sp), (va_s, va_sp, va_a) = gridworld_transitions()
    print(f"  {GRID_N}x{GRID_N} grid, interior starts: {len(tr_s)} train / {len(va_s)} val transitions")
    codes = fit_lam(tr_s, tr_sp)                             # action labels never passed in
    print(f"  codebook after 1 assignment + centroid: {[[round(x, 6) for x in c] for c in codes.tolist()]}")

    k = torch.cdist(va_sp - va_s, codes).argmin(dim=1)
    err = ((va_s + codes[k]) - va_sp).abs().max().item()     # additive decoder s + e_k
    assert err < 1e-12, err
    assert len(torch.unique(k)) == N_CODES, torch.unique(k)

    # purity: each code maps to exactly one action (a bijection, not the identity)
    perm = {j: int(torch.bincount(va_a[k == j], minlength=len(MOVES)).argmax()) for j in range(N_CODES)}
    purity = sum(int((va_a[k == j] == perm[j]).sum()) for j in range(N_CODES)) / len(va_a)
    mi = mutual_information_bits(k, va_a, N_CODES, len(MOVES))
    print(f"  val recon error={err:.1e}  purity={purity:.6f}  I(K;A)={mi:.6f} bits  codes used={len(torch.unique(k))}")
    print(f"  code -> action permutation: {perm}  (identity is not required, a bijection is)")
    assert abs(purity - 1.0) < 1e-12, purity
    assert abs(mi - 2.0) < 1e-12, mi
    assert sorted(perm.values()) == list(range(len(MOVES))), perm
    print("  PASS: exact recovery of the four transition vectors, up to relabelling the codes\n")


if __name__ == "__main__":
    experiment_A()
    experiment_B()
    experiment_C()
    print("All checks passed.")
