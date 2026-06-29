"""
Optimizers / LR Schedules - minimal runnable implementation
===========================================================

Educational PyTorch reference for SGD-momentum / Adam / AdamW from scratch, the
AdamW-vs-Adam-L2 decoupling, Adam's bias correction, and a cosine-with-warmup schedule.
Standalone script: runs six sanity checks on CPU in a few seconds.

Pairs with: docs/tutorials/optimizer_lr_schedule_tutorial.md (concept reference).

What it demonstrates:
    sgd_momentum_step / adam_step / adamw_step  - match torch.optim.{SGD,Adam,AdamW}
    Sanity checks:
      [a] SGD+momentum from scratch == torch.optim.SGD(momentum=...) after K steps
      [b] Adam from scratch (with bias correction) == torch.optim.Adam after K steps
      [c] AdamW != Adam+L2: same weight_decay gives DIFFERENT params (decoupled vs coupled),
          and from-scratch decoupled decay == torch.optim.AdamW
      [d] bias correction matters: at step 1 it shrinks the update to sqrt(1-b2)/(1-b1)~0.32x the uncorrected one (else the under-estimated v_hat makes the first step ~3x too large)
      [e] cosine-with-warmup schedule: lr~0 at start, peaks at the warmup boundary, decays to lr_min at the end
      [f] momentum accelerates on an ill-conditioned quadratic: lower loss than plain GD after K steps

Run:
    python optimizer_lr_schedule.py
"""
import math

import torch

torch.manual_seed(0)


# ---- from-scratch optimizer steps (match PyTorch's update rules) ----

def sgd_momentum_step(p, g, buf, lr, mu):
    """PyTorch SGD with momentum (dampening=0): buf = mu*buf + g; p -= lr*buf."""
    buf = mu * buf + g
    return p - lr * buf, buf


def adam_step(p, g, m, v, t, lr, b1, b2, eps):
    """PyTorch Adam: EMA m,v + bias correction; p -= lr * m_hat/(sqrt(v_hat)+eps)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)                         # bias correction (m,v start at 0)
    v_hat = v / (1 - b2 ** t)
    return p - lr * m_hat / (v_hat.sqrt() + eps), m, v


def adamw_step(p, g, m, v, t, lr, b1, b2, eps, wd):
    """PyTorch AdamW: DECOUPLED decay (p *= 1-lr*wd) then the Adam step on the raw grad."""
    p = p * (1 - lr * wd)                             # decoupled weight decay (acts on the weight, not the grad)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return p - lr * m_hat / (v_hat.sqrt() + eps), m, v


def cosine_warmup_lr(t, warmup, total, lr_peak, lr_min):
    """Linear warmup to lr_peak over `warmup` steps, then cosine decay to lr_min by `total`."""
    if t < warmup:
        return lr_peak * t / warmup
    progress = (t - warmup) / (total - warmup)        # 0 -> 1
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * progress))


def run_torch(opt_cls, p0, target, steps, **kw):
    """Drive a torch optimizer on loss = 0.5*||p - target||^2 (grad = p - target)."""
    p = torch.nn.Parameter(p0.clone())
    opt = opt_cls([p], **kw)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * ((p - target) ** 2).sum()
        loss.backward()
        opt.step()
    return p.detach()


def main():
    torch.manual_seed(0)
    d = 16
    p0 = torch.randn(d)
    target = torch.randn(d)
    lr, b1, b2, eps, mu, wd, K = 0.1, 0.9, 0.999, 1e-8, 0.9, 0.1, 20

    # [a] SGD+momentum from scratch == torch.optim.SGD(momentum)
    p_t = run_torch(torch.optim.SGD, p0, target, K, lr=lr, momentum=mu)
    p, buf = p0.clone(), torch.zeros(d)
    for _ in range(K):
        g = p - target                                # grad of 0.5*||p-target||^2
        p, buf = sgd_momentum_step(p, g, buf, lr, mu)
    a_ok = torch.allclose(p, p_t, atol=1e-5)
    print(f"[a] SGD+momentum from scratch vs torch.optim.SGD: max|Δ| = {(p - p_t).abs().max():.2e}  "
          f"{'OK' if a_ok else 'FAIL'}")
    assert a_ok

    # [b] Adam from scratch (bias-corrected) == torch.optim.Adam
    p_t = run_torch(torch.optim.Adam, p0, target, K, lr=lr, betas=(b1, b2), eps=eps)
    p, m, v = p0.clone(), torch.zeros(d), torch.zeros(d)
    for t in range(1, K + 1):
        g = p - target
        p, m, v = adam_step(p, g, m, v, t, lr, b1, b2, eps)
    b_ok = torch.allclose(p, p_t, atol=1e-5)
    print(f"[b] Adam from scratch vs torch.optim.Adam: max|Δ| = {(p - p_t).abs().max():.2e}  "
          f"{'OK' if b_ok else 'FAIL'}")
    assert b_ok

    # [c] AdamW != Adam+L2 (same wd), and from-scratch decoupled == AdamW
    p_adamw = run_torch(torch.optim.AdamW, p0, target, K, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    p_adaml2 = run_torch(torch.optim.Adam, p0, target, K, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    p, m, v = p0.clone(), torch.zeros(d), torch.zeros(d)
    for t in range(1, K + 1):
        g = p - target                                # raw grad, NO L2 added (decoupling)
        p, m, v = adamw_step(p, g, m, v, t, lr, b1, b2, eps, wd)
    differ = (p_adamw - p_adaml2).abs().max().item()  # AdamW vs coupled-L2: should DIFFER
    match = (p - p_adamw).abs().max().item()           # from-scratch decoupled vs AdamW: should MATCH
    c_ok = differ > 1e-3 and match < 1e-5
    print(f"[c] AdamW vs Adam+L2 (same wd={wd}): max|Δ| = {differ:.2e} (DIFFER, decoupled≠coupled); "
          f"from-scratch decoupled vs AdamW = {match:.2e} (MATCH)  {'OK' if c_ok else 'FAIL'}")
    assert c_ok

    # [d] bias correction matters: at step 1 it SHRINKS the update to sqrt(1-b2)/(1-b1) of the
    #     uncorrected one -- v's 1/(1-b2)=1000 under-estimate dominates m's 1/(1-b1)=10, so WITHOUT
    #     correction v_hat is too small and the first step is ~3x too large (instability source).
    g1 = torch.ones(d)                                 # a unit gradient at step 1
    m1 = (1 - b1) * g1                                 # m after one EMA step (m0=0), biased toward 0
    v1 = (1 - b2) * g1 * g1                            # v after one EMA step, biased toward 0
    step_uncorrected = m1 / (v1.sqrt() + eps)                            # forgetting bias correction
    step_corrected = (m1 / (1 - b1)) / ((v1 / (1 - b2)).sqrt() + eps)    # with bias correction
    ratio = (step_corrected / step_uncorrected).mean().item()
    expected = math.sqrt(1 - b2) / (1 - b1)            # = sqrt(1-b2)/(1-b1) ~ 0.316
    d_ok = abs(ratio - expected) < 0.05
    print(f"[d] bias correction at step 1: corrected/uncorrected = {ratio:.3f} "
          f"(= √(1-β2)/(1-β1) ≈ {expected:.3f}); without it the under-estimated v̂ makes the first step "
          f"~{1/ratio:.1f}× too large  {'OK' if d_ok else 'FAIL'}")
    assert d_ok

    # [e] cosine-with-warmup schedule shape
    warmup, total, lr_peak, lr_min = 100, 1000, 1.0, 0.0
    lr0 = cosine_warmup_lr(0, warmup, total, lr_peak, lr_min)
    lrW = cosine_warmup_lr(warmup, warmup, total, lr_peak, lr_min)
    lrT = cosine_warmup_lr(total, warmup, total, lr_peak, lr_min)
    lr_mid_warmup = cosine_warmup_lr(warmup // 2, warmup, total, lr_peak, lr_min)
    e_ok = lr0 < 1e-9 and abs(lrW - lr_peak) < 1e-9 and lrT < 1e-6 and abs(lr_mid_warmup - 0.5) < 1e-6
    print(f"[e] cosine+warmup lr: t=0 -> {lr0:.3f}, t=warmup -> {lrW:.3f} (peak), "
          f"t=end -> {lrT:.3f} (min); linear warmup midpoint -> {lr_mid_warmup:.3f}  {'OK' if e_ok else 'FAIL'}")
    assert e_ok

    # [f] momentum accelerates on an ill-conditioned quadratic f(x)=0.5*(kappa*x1^2 + x2^2)
    kappa = 20.0
    A = torch.tensor([kappa, 1.0])
    x_gd = torch.tensor([1.0, 1.0]); x_mom = x_gd.clone(); buf = torch.zeros(2)
    glr, N = 0.25 / kappa, 100                          # small lr: GD crawls on the low-curvature dir, momentum accelerates it
    for _ in range(N):
        x_gd = x_gd - glr * (A * x_gd)                  # plain gradient descent
    for _ in range(N):
        g = A * x_mom
        buf = mu * buf + g
        x_mom = x_mom - glr * buf                       # GD + momentum
    loss_gd = 0.5 * (A * x_gd * x_gd).sum().item()
    loss_mom = 0.5 * (A * x_mom * x_mom).sum().item()
    f_ok = loss_mom < loss_gd
    print(f"[f] ill-conditioned quadratic (κ={kappa:.0f}) after {N} steps: "
          f"GD loss = {loss_gd:.2e}  vs  GD+momentum loss = {loss_mom:.2e}  "
          f"(momentum lower)  {'OK' if f_ok else 'FAIL'}")
    assert f_ok

    print("\nall optimizer / LR-schedule sanity checks passed ✓")


if __name__ == "__main__":
    main()
