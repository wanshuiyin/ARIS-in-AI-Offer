"""
Normalization / Residual / Initialization - minimal runnable implementation
===========================================================================

Educational PyTorch reference for normalization layers, the Pre-LN vs Post-LN
gradient behaviour, and variance-preserving initialization.
Standalone script: runs six sanity checks on CPU in a few seconds.

Pairs with: docs/tutorials/normalization_init_tutorial.md (concept reference).

What it demonstrates:
    layernorm_from_scratch  - LayerNorm matches torch.nn.LayerNorm
    rmsnorm_from_scratch    - RMSNorm matches torch.nn.RMSNorm, and (unlike LayerNorm)
                              is NOT mean-shift invariant (it only re-scales, no re-centering)
    Sanity checks:
      [a] LayerNorm from scratch == nn.LayerNorm (population var, eps inside the sqrt)
      [b] RMSNorm from scratch == nn.RMSNorm; LayerNorm(x+c)==LayerNorm(x) but RMSNorm(x+c)!=RMSNorm(x)
      [c] BatchNorm train != eval: train uses batch stats (+updates running stats), eval uses running stats
      [d] Post-LN piles parameter gradients near the OUTPUT (top-heavy, last/first>1, needs warmup); Pre-LN spreads them evenly across depth
      [e] Kaiming preserves the second moment E[y^2] through Linear+ReLU (factor 2/fan_in); Xavier-for-ReLU halves it to ~0.5
      [f] GPT-2 residual scaling 1/sqrt(2N) keeps the residual-stream variance bounded vs linear growth

Run:
    python normalization.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def layernorm_from_scratch(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim. x: [..., d]. Population variance (unbiased=False),
    eps added INSIDE the sqrt, then affine: y = (x-mean)/sqrt(var+eps) * weight + bias."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)        # population var, matches torch
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def rmsnorm_from_scratch(x, weight, eps=1e-6):
    """RMSNorm over the last dim: only re-scale by the RMS, NO mean-centering, NO bias.
    y = x / sqrt(mean(x^2) + eps) * weight."""
    ms = x.pow(2).mean(dim=-1, keepdim=True)                 # mean of squares
    return x / torch.sqrt(ms + eps) * weight


class ResidualStack(nn.Module):
    """A deep stack of identical residual blocks, either Pre-LN or Post-LN, plus a final
    linear head, used to show how parameter-gradient norms distribute across depth at init.
        Pre-LN  block:  h = h + Linear(LayerNorm(h))
        Post-LN block:  h = LayerNorm(h + Linear(h))
    The head matters: without it, a Post-LN stack's output is LayerNorm-normalized, so a
    loss like mean(out^2) is ~scale-invariant and artificially suppresses ALL gradients
    (a confound). The head makes the loss depend on the actual output, so the only signal
    left is the genuine cross-depth gradient distribution.
    """

    def __init__(self, depth, d, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.lns = nn.ModuleList(nn.LayerNorm(d) for _ in range(depth))
        self.lins = nn.ModuleList(nn.Linear(d, d) for _ in range(depth))
        self.head = nn.Linear(d, d)                         # final projection (un-normalizes the output)

    def forward(self, h):
        for ln, lin in zip(self.lns, self.lins):
            if self.pre_ln:
                h = h + lin(ln(h))
            else:
                h = ln(h + lin(h))
        return self.head(h)


def block_grad_topheavy(depth, d, pre_ln):
    """Build a fresh stack, one forward+backward, return the LAST-block / FIRST-block
    weight-grad-norm ratio. >1 means the gradient is concentrated near the OUTPUT
    (top-heavy). This ratio is robust to the loss choice (Xiong et al. 2020's actual claim)."""
    torch.manual_seed(0)                                    # same init for both variants
    stack = ResidualStack(depth, d, pre_ln=pre_ln)
    stack(torch.randn(16, d)).pow(2).mean().backward()
    gn = [lin.weight.grad.norm().item() for lin in stack.lins]
    return gn[-1] / gn[0]


def main():
    B, d, eps = 8, 64, 1e-5

    # [a] LayerNorm from scratch == nn.LayerNorm
    x = torch.randn(B, d)
    ln = nn.LayerNorm(d, eps=eps)                            # affine: weight=1, bias=0 at init
    mine = layernorm_from_scratch(x, ln.weight, ln.bias, eps=eps)
    a_ok = torch.allclose(mine, ln(x), atol=1e-5)
    print(f"[a] LayerNorm from scratch vs nn.LayerNorm: max|Δ| = {(mine - ln(x)).abs().max():.2e}  "
          f"{'OK' if a_ok else 'FAIL'}")
    assert a_ok

    # [b] RMSNorm matches nn.RMSNorm; and RMSNorm is NOT mean-shift invariant (LayerNorm is)
    rms = nn.RMSNorm(d, eps=1e-6)
    mine_r = rmsnorm_from_scratch(x, rms.weight, eps=1e-6)
    b1 = torch.allclose(mine_r, rms(x), atol=1e-5)
    c = 5.0                                                  # constant shift on every feature
    ln_shift = (ln(x + c) - ln(x)).abs().max().item()       # LayerNorm removes the mean -> ~0
    rms_shift = (rmsnorm_from_scratch(x + c, rms.weight) - mine_r).abs().max().item()  # RMS keeps it -> >0
    b_ok = b1 and ln_shift < 1e-4 and rms_shift > 1e-2
    print(f"[b] RMSNorm vs nn.RMSNorm: max|Δ| = {(mine_r - rms(x)).abs().max():.2e}; "
          f"mean-shift LN |Δ|={ln_shift:.2e} (~0, re-centers) vs RMS |Δ|={rms_shift:.2e} (>0, only re-scales)  "
          f"{'OK' if b_ok else 'FAIL'}")
    assert b_ok

    # [c] BatchNorm: train uses batch stats (+ updates running stats), eval uses running stats
    bn = nn.BatchNorm1d(d)                                   # running_mean=0, running_var=1 at init
    xb = torch.randn(B, d) * 3 + 7                           # shifted/scaled so batch stats != running
    bn.train()
    y_train = bn(xb)
    running_moved = bn.running_mean.abs().mean().item()      # moved away from 0 toward batch mean
    bn.eval()
    y_eval = bn(xb)
    differ = (y_train - y_eval).abs().max().item()
    # train output is normalized by the batch -> ~0 mean / ~1 std per feature; eval is not
    train_mean = y_train.mean(dim=0).abs().mean().item()
    c_ok = running_moved > 0 and differ > 1e-2 and train_mean < 1e-4
    print(f"[c] BatchNorm train!=eval: max|Δ| = {differ:.2e}; running_mean moved {running_moved:.3f} from 0; "
          f"train per-feature mean = {train_mean:.2e} (~0)  {'OK' if c_ok else 'FAIL'}")
    assert c_ok

    # [d] Post-LN concentrates parameter gradients near the OUTPUT (top-heavy -> needs warmup);
    #     Pre-LN's clean identity path spreads them evenly across depth (Xiong et al. 2020)
    depth = 48
    r_pre = block_grad_topheavy(depth, d, pre_ln=True)     # ~0.4: balanced (slightly bottom-heavy)
    r_post = block_grad_topheavy(depth, d, pre_ln=False)   # ~2.3: top-heavy (gradient piled at the top)
    d_ok = r_post > 1.3 and r_post > 2 * r_pre              # Post-LN top-heavy AND more imbalanced than Pre-LN
    print(f"[d] per-block weight-grad top/bottom ratio (last/first over {depth} blocks): "
          f"Pre-LN={r_pre:.2f} (balanced)  Post-LN={r_post:.2f} (top-heavy, >1)  "
          f"-> Post-LN piles gradient near the output, needs warmup  {'OK' if d_ok else 'FAIL'}")
    assert d_ok

    # [e] Kaiming preserves the SECOND MOMENT E[y^2] through Linear+ReLU (the quantity He et al.
    #     actually propagate): ReLU halves E[pre^2], so the factor 2/fan_in restores E[y^2]~E[x^2]=1.
    #     Xavier (1/fan_in) lacks the factor 2 -> E[y^2]~0.5, decaying by half every layer.
    fan_in, fan_out, N = 512, 512, 4096
    xin = torch.randn(N, fan_in)                            # E[x^2] = 1 (unit second moment)
    W_kaiming = torch.randn(fan_out, fan_in) * (2.0 / fan_in) ** 0.5   # He: std = sqrt(2/fan_in)
    W_xavier = torch.randn(fan_out, fan_in) * (1.0 / fan_in) ** 0.5    # Xavier: std = sqrt(1/fan_in)
    ms_kaiming = F.relu(xin @ W_kaiming.t()).pow(2).mean().item()   # E[y^2] ~ 1 (preserved)
    ms_xavier = F.relu(xin @ W_xavier.t()).pow(2).mean().item()     # E[y^2] ~ 0.5 (halved -> decays)
    e_ok = abs(ms_kaiming - 1.0) < 0.1 and abs(ms_xavier - 0.5) < 0.1
    print(f"[e] post-ReLU second moment E[y^2] (input E[x^2]=1): Kaiming = {ms_kaiming:.3f} (~1, preserved)  "
          f"Xavier = {ms_xavier:.3f} (~0.5, halves per layer)  {'OK' if e_ok else 'FAIL'}")
    assert e_ok

    # [f] GPT-2 residual scaling 1/sqrt(2N): residual-stream variance stays bounded vs linear growth
    Nlayers, width = 50, 256
    h_plain = torch.randn(N, width)
    h_scaled = h_plain.clone()
    v0 = h_plain.var().item()
    for _ in range(Nlayers):
        delta = torch.randn(N, width)                        # each block adds an O(1)-variance update
        h_plain = h_plain + delta                            # no scaling -> Var grows ~linearly with depth
        h_scaled = h_scaled + delta * (1.0 / (2 * Nlayers) ** 0.5)   # GPT-2 1/sqrt(2N) residual scaling
    growth_plain = h_plain.var().item() / v0                 # ~ (1 + Nlayers)
    growth_scaled = h_scaled.var().item() / v0               # ~ O(1), bounded
    f_ok = growth_plain > 10 * growth_scaled
    print(f"[f] residual-stream var growth over {Nlayers} blocks: "
          f"unscaled ×{growth_plain:.1f}  vs  1/sqrt(2N)-scaled ×{growth_scaled:.2f}  "
          f"{'OK' if f_ok else 'FAIL'}")
    assert f_ok

    print("\nall normalization / residual / init sanity checks passed ✓")


if __name__ == "__main__":
    main()
