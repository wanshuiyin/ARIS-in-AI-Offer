"""
Modern diffusion post-training: three toy experiments
=====================================================

Pairs with: docs/tutorials/modern_diffusion_post_training_tutorial.md
            (Flow-GRPO / DGPO / DiffusionNFT).

Three claims from the tutorial, each checked against an ANALYTIC answer:

  A. Flow-GRPO's ODE->SDE conversion preserves marginals (§3.1) —
     and the "wrong" SDE (noise added, score correction forgotten) does not.
  B. DiffusionNFT's reflected two-branch loss (§5.6) pushes v_theta toward the
     target for positives, away for negatives, and is neutral at r = 1/2;
     the per-sample optimum is v_old + (2r-1)(v - v_old)/beta, NOT the target itself.
  C. DGPO's |A| weights balance exactly (§4.3), so the loss is invariant to a
     constant shift of the implicit scores (the log Z cancellation) — and
     unit weights on a 1:3 split are not.

Convention (same as the tutorial): x_t = (1-t) x_0 + t eps, t=0 clean,
t=1 noise, generation runs t: 1 -> 0, target v = eps - x_0.

Run:
    python diffusion_online_rl.py      # CPU, single thread, a few seconds
"""
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# A. marginal preservation — 1-D Gaussian data, everything analytic
# ---------------------------------------------------------------------------
# x_0 ~ N(1, 0.25)  =>  x_t ~ N(m_t, q_t),  m_t = 1-t,  q_t = 0.25(1-t)^2 + t^2
# Gaussian conditioning gives the TRUE velocity and score fields:
#   v*(x,t) = -1 + k_t (x - m_t),   k_t = (t - 0.25(1-t)) / q_t
#   s*(x,t) = -(x - m_t) / q_t
def m(t):
    return 1.0 - t

def q(t):
    return 0.25 * (1 - t) ** 2 + t ** 2

def k(t):
    return (t - 0.25 * (1 - t)) / q(t)

def g2(t, a=0.7):                                # Flow-GRPO's g(t)^2 = a^2 t/(1-t)
    return a * a * t / (1 - t)


def experiment_A(n=16384, steps=400, t0=0.9, t1=0.1):
    print("== A. ODE -> SDE marginal preservation ==")
    h = (t0 - t1) / steps
    checkpoints = {0.7, 0.5, 0.3, 0.1}
    x = m(t0) + math.sqrt(q(t0)) * torch.randn(n)      # start from the analytic p_{t0}
    xs = {"ode": x.clone(), "sde_ok": x.clone(), "sde_wrong": x.clone()}
    # exact variance recursion of the discretised (linear-Gaussian) processes.
    # writing y = x - m_t, one Euler step gives y' = (1 - h c) y + g sqrt(h) xi with
    #   c = k_t + g^2/(2 q_t)   (correct SDE: drift v - g^2/2 s)
    #   c = k_t                 (ODE, and the wrong SDE which forgets the score term)
    var = {"ode": q(t0), "sde_ok": q(t0), "sde_wrong": q(t0)}
    t = t0
    for _ in range(steps):
        v = {kk: -1.0 + k(t) * (xs[kk] - m(t)) for kk in xs}
        s_ok = -(xs["sde_ok"] - m(t)) / q(t)
        xi = torch.randn(n)
        xs["ode"] = xs["ode"] - h * v["ode"]
        xs["sde_ok"] = xs["sde_ok"] - h * (v["sde_ok"] - 0.5 * g2(t) * s_ok) + math.sqrt(g2(t) * h) * xi
        xs["sde_wrong"] = xs["sde_wrong"] - h * v["sde_wrong"] + math.sqrt(g2(t) * h) * xi
        var["ode"] = (1 - h * k(t)) ** 2 * var["ode"]
        var["sde_ok"] = (1 - h * (k(t) + g2(t) / (2 * q(t)))) ** 2 * var["sde_ok"] + g2(t) * h
        var["sde_wrong"] = (1 - h * k(t)) ** 2 * var["sde_wrong"] + g2(t) * h
        t -= h
        if any(abs(t - c) < 1e-9 for c in checkpoints):
            se_mean = math.sqrt(q(t) / n)
            se_var = q(t) * math.sqrt(2.0 / n)
            for kk in ("ode", "sde_ok"):
                mu_mc, var_mc = xs[kk].mean().item(), xs[kk].var(unbiased=False).item()
                assert abs(mu_mc - m(t)) < 6 * se_mean + 0.005, (kk, t, mu_mc, m(t))
                assert abs(var_mc - var[kk]) < 6 * se_var + 0.005, (kk, t, var_mc, var[kk])
                assert abs(var[kk] - q(t)) < 0.002, (kk, t, var[kk], q(t))       # discretisation error
            print(f"  t={t:.1f}  truth var={q(t):.4f}  ode={xs['ode'].var(unbiased=False):.4f}"
                  f"  sde_ok={xs['sde_ok'].var(unbiased=False):.4f}"
                  f"  sde_wrong={xs['sde_wrong'].var(unbiased=False):.4f}")
    assert abs(var["sde_wrong"] - q(t1)) > 0.15, "the wrong SDE should visibly break the marginal"
    print(f"  terminal analytic: truth {q(t1):.4f} | correct-SDE recursion {var['sde_ok']:.5f}"
          f" | wrong-SDE recursion {var['sde_wrong']:.5f}")
    print("  PASS: ODE and score-corrected SDE match the analytic marginal; noise-only SDE does not\n")


# ---------------------------------------------------------------------------
# B. DiffusionNFT reflected two-branch loss — scalar, closed-form gradient
# ---------------------------------------------------------------------------
def nft_loss(z, v_old, v, r, beta):
    d = z - v_old
    v_pos = v_old + beta * d                    # (1-beta) v_old + beta v_theta
    v_neg = v_old - beta * d                    # (1+beta) v_old - beta v_theta
    return r * (v_pos - v) ** 2 + (1 - r) * (v_neg - v) ** 2


def experiment_B():
    print("== B. DiffusionNFT: sign of the update and the per-sample optimum ==")
    t, x0, eps = 0.5, 0.0, 1.0
    v = eps - x0                                # target = 1
    v_old, beta, lr = 0.0, 0.5, 0.1
    expect_grad = {1.0: -1.0, 0.0: +1.0, 0.5: 0.0}
    for r, want in expect_grad.items():
        z = torch.zeros((), requires_grad=True)                   # current = old
        loss = nft_loss(z, v_old, v, r, beta)
        (grad,) = torch.autograd.grad(loss, z)
        assert abs(grad.item() - want) < 1e-12, (r, grad.item(), want)
        z_after = (z - lr * grad).item()
        assert abs(z_after - (-lr * want)) < 1e-12
        print(f"  r={r:<4} grad at current=old: {grad.item():+.1f}  ->  after one SGD step z={z_after:+.2f}")
    # per-sample optimum: (2r-1)(v - v_old)/beta = ±2, not the target v = 1
    for r, want in ((1.0, +2.0), (0.0, -2.0)):
        z = torch.zeros((), requires_grad=True)
        opt = torch.optim.SGD([z], lr=0.5)
        for _ in range(200):
            opt.zero_grad(); nft_loss(z, v_old, v, r, beta).backward(); opt.step()
        assert abs(z.item() - want) < 1e-6, (r, z.item(), want)
        branch = v_old + (beta if r == 1.0 else -beta) * (z.item() - v_old)
        assert abs(branch - v) < 1e-6, "it is the implicit branch that reaches the target"
        print(f"  r={r:<4} optimum z*={z.item():+.4f} (= (2r-1)(v-v_old)/beta); its implicit branch = {branch:.4f} = target")
    print("  PASS: positives pull toward the target, negatives push away, r=1/2 is neutral\n")


# ---------------------------------------------------------------------------
# C. DGPO balanced weights — log Z cancellation, with a counterexample
# ---------------------------------------------------------------------------
def advantages(rewards):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False)


def dgpo_logit(theta, ref, x0, adv, t, eps, weights=None, score_shift=0.0, beta=1.0, T=1.0):
    """theta/ref: (w, b) of a linear velocity model v(x_t) = w x_t + b.  Returns the
    argument of log-sigmoid: -beta T * sum_i w_i * sign_i * d_i, with d_i the
    per-sample DSM difference (theta minus ref) plus an optional constant shift."""
    x_t = (1 - t) * x0 + t * eps
    v = eps - x0
    d = (theta[0] * x_t + theta[1] - v) ** 2 - (ref[0] * x_t + ref[1] - v) ** 2 + score_shift
    if weights is None:                          # DGPO: w = |A|, sign carried by A
        return -beta * T * (adv * d).sum()
    sign = torch.where(adv > 0, 1.0, -1.0)      # counterexample: unit weights, same split
    return -beta * T * (weights * sign * d).sum()


def experiment_C():
    print("== C. DGPO: |A| weights balance => invariant to the partition-function term ==")
    rewards = torch.tensor([0.0, 1.0, 2.0, 7.0])                  # 1 positive : 3 negative
    adv = advantages(rewards)
    pos, neg = adv[adv > 0].abs().sum(), adv[adv <= 0].abs().sum()
    assert abs(pos - neg) < 1e-12, (pos.item(), neg.item())
    print(f"  advantages {adv.tolist()}  ->  sum|A| positive = {pos:.4f}, negative = {neg:.4f}")

    x0 = torch.tensor([0.3, -0.8, 1.1, 0.4]); t = 0.6; eps = torch.tensor([0.5, -0.2, 0.9, -1.3])
    w = torch.tensor(0.7, requires_grad=True); b = torch.tensor(-0.2, requires_grad=True)
    ref = (torch.tensor(0.4), torch.tensor(0.1))

    def loss_and_grad(adv_, shift=0.0, weights=None):
        L = -F.logsigmoid(dgpo_logit((w, b), ref, x0, adv_, t, eps, weights, shift))
        gw, gb = torch.autograd.grad(L, (w, b))
        return L.item(), gw.item(), gb.item()

    base = loss_and_grad(adv)
    shifted_r = loss_and_grad(advantages(rewards + 11.0))           # reward shift -> same A
    assert all(abs(a - b_) < 1e-10 for a, b_ in zip(base, shifted_r)), "centering broke"
    shifted_s = loss_and_grad(adv, shift=7.0)                       # implicit-score shift
    assert all(abs(a - b_) < 1e-10 for a, b_ in zip(base, shifted_s)), "log Z did not cancel"
    print(f"  loss {base[0]:.6f}: unchanged by reward+11 ({shifted_r[0]:.6f}) and by score+7 ({shifted_s[0]:.6f})")

    ones = torch.ones(4)
    unit = loss_and_grad(adv, weights=ones)
    unit_shift = loss_and_grad(adv, shift=7.0, weights=ones)
    assert abs(unit[0] - unit_shift[0]) > 1e-3, "unit weights on a 1:3 split should NOT cancel"
    print(f"  counterexample, unit weights: loss {unit[0]:.6f} -> {unit_shift[0]:.6f} after score+7 (1-3 = -2 != 0)")
    print("  PASS: balanced |A| weights make the group logit invariant to log Z; unit weights do not\n")


if __name__ == "__main__":
    experiment_A()
    experiment_B()
    experiment_C()
    print("All checks passed.")
