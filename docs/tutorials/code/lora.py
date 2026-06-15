"""
LoRA / PEFT - minimal runnable implementation
=============================================

Educational PyTorch reference for LoRA and DoRA parameter-efficient fine-tuning.
Standalone script: runs six sanity checks on CPU in a few seconds.

Pairs with: docs/tutorials/lora_peft_tutorial.md (concept reference).

What it demonstrates:
    LoRALinear  - frozen base W0 + low-rank bypass (alpha/r) B A; merge/unmerge
    DoRALinear  - magnitude-direction decomposition (DoRA)
    Sanity checks:
      [a] B=0 init  -> dW=0 -> output identical to frozen base
      [b] after 1 step -> dW != 0 -> output changes
      [c] merge()/unmerge() are numerically consistent and reversible
      [d] trainable params = r*(in+out) per layer; base fully frozen
      [e] rsLoRA scaling = alpha/sqrt(r) (vs standard alpha/r)
      [f] DoRA: magnitude m trainable, base frozen, B=0 identity start

Run:
    python lora.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class LoRALinear(nn.Module):
    """Frozen base nn.Linear + low-rank bypass. forward: W0 x + (alpha/r) B A x."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, rslora: bool = False):
        super().__init__()
        assert r > 0
        self.base = base
        for p in self.base.parameters():           # freeze the base
            p.requires_grad_(False)

        in_f, out_f = base.in_features, base.out_features
        self.r, self.alpha = r, alpha
        # standard alpha/r; rsLoRA uses alpha/sqrt(r)
        self.scaling = alpha / (math.sqrt(r) if rslora else r)

        dev, dt = base.weight.device, base.weight.dtype          # follow the base
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=dt))   # [r, in]
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))  # [out,r] zero -> dW=0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            out = out + self.scaling * lora
        return out

    @torch.no_grad()
    def merge(self):
        # NOTE: merge/unmerge are exact inverses ONLY while lora_A/lora_B are unchanged.
        # Do not run optimizer steps while merged, or unmerge() will not restore W0.
        if self.merged:
            return
        dW = self.scaling * (self.lora_B @ self.lora_A)        # [out, in]
        self.base.weight.add_(dW.to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        dW = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.sub_(dW.to(self.base.weight.dtype))
        self.merged = False


class DoRALinear(nn.Module):
    """DoRA: W' = m * (V + dV)/||V + dV||, dV = (alpha/r) B A, m = ||W0|| per OUTPUT row.

    Magnitude is per output neuron (weight-normalization style): for a PyTorch
    weight [out, in], the norm is over the input dim (dim=1) -> m shape [out, 1].
    This matches the reference DoRA / HF PEFT (torch.linalg.norm(weight, dim=1)).
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # per-output magnitude: norm over the in dim (dim=1) -> [out, 1]
        self.m = nn.Parameter(self.base.weight.norm(dim=1, keepdim=True))  # [out, 1]

    def forward(self, x):
        dW = self.scaling * (self.lora_B @ self.lora_A)        # [out, in]
        V = self.base.weight + dW
        Vnorm = V.norm(dim=1, keepdim=True) + 1e-8            # [out, 1] per-output norm
        W_eff = self.m * V / Vnorm                            # [out,1]*[out,in]/[out,1]
        return F.linear(x, W_eff, self.base.bias)


def trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    B, IN, OUT, r, alpha = 4, 32, 48, 8, 16
    x = torch.randn(B, IN)

    # ---- LoRA ----
    base = nn.Linear(IN, OUT)
    lora = LoRALinear(base, r=r, alpha=alpha)

    # [a] B=0 init -> dW=0 -> output identical to frozen base
    with torch.no_grad():
        base_only = F.linear(x, base.weight, base.bias)
    out0 = lora(x)
    a_ok = torch.allclose(out0, base_only, atol=1e-6)
    print(f"[a] B=0 init: |out_lora - out_base| = {(out0 - base_only).abs().max():.2e}  "
          f"{'OK' if a_ok else 'FAIL'}")
    assert a_ok

    # [b] one optimization step -> dW != 0 -> output changes
    opt = torch.optim.SGD([p for p in lora.parameters() if p.requires_grad], lr=0.1)
    out0_detached = lora(x).detach()
    loss = lora(x).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    dW_norm = (lora.scaling * (lora.lora_B @ lora.lora_A)).norm().item()
    out1 = lora(x)
    b_ok = dW_norm > 0 and not torch.allclose(out1, out0_detached, atol=1e-6)
    print(f"[b] after 1 step: ||dW|| = {dW_norm:.3e}, output changed  "
          f"{'OK' if b_ok else 'FAIL'}")
    assert b_ok

    # [c] merge()/unmerge() numerically consistent + reversible (non-vacuous:
    #     assert merged flag flips AND base.weight actually moved by dW)
    w_before = lora.base.weight.clone()
    dW_expect = lora.scaling * (lora.lora_B @ lora.lora_A)
    out_unmerged = lora(x).detach()
    lora.merge()
    out_merged = lora(x).detach()
    c1 = torch.allclose(out_merged, out_unmerged, atol=1e-5)
    c_flag = lora.merged
    c_moved = torch.allclose(lora.base.weight, w_before + dW_expect, atol=1e-6)
    lora.unmerge()
    c2 = torch.allclose(lora.base.weight, w_before, atol=1e-6) and not lora.merged
    print(f"[c] merge: |out_merged - out_unmerged| = {(out_merged - out_unmerged).abs().max():.2e}; "
          f"weight += dW = {c_moved}; unmerge restores base = {c2}  "
          f"{'OK' if (c1 and c_flag and c_moved and c2) else 'FAIL'}")
    assert c1 and c_flag and c_moved and c2

    # [d] trainable params = r*(in+out); base frozen
    expect = r * (IN + OUT)
    got = trainable_params(lora)
    base_frozen = all(not p.requires_grad for p in lora.base.parameters())
    d_ok = (got == expect) and base_frozen
    print(f"[d] trainable params = {got} (expect r*(in+out) = {expect}); base frozen = {base_frozen}  "
          f"{'OK' if d_ok else 'FAIL'}")
    assert d_ok

    # [e] rsLoRA scaling = alpha/sqrt(r) vs standard alpha/r
    std = LoRALinear(nn.Linear(IN, OUT), r=r, alpha=alpha, rslora=False)
    rs = LoRALinear(nn.Linear(IN, OUT), r=r, alpha=alpha, rslora=True)
    e_ok = (abs(std.scaling - alpha / r) < 1e-9
            and abs(rs.scaling - alpha / math.sqrt(r)) < 1e-9)
    print(f"[e] scaling: standard alpha/r = {std.scaling:.4f}, rsLoRA alpha/sqrt(r) = {rs.scaling:.4f}  "
          f"{'OK' if e_ok else 'FAIL'}")
    assert e_ok

    # [f] DoRA: m trainable, base frozen, B=0 identity start
    dora = DoRALinear(nn.Linear(IN, OUT), r=r, alpha=alpha)
    with torch.no_grad():
        dora_base = F.linear(x, dora.base.weight, dora.base.bias)
    dora_out0 = dora(x)
    f_ok = (dora.m.requires_grad
            and all(not p.requires_grad for p in dora.base.parameters())
            and torch.allclose(dora_out0, dora_base, atol=1e-5))
    print(f"[f] DoRA: m.requires_grad = {dora.m.requires_grad}, base frozen, "
          f"identity start |Δ| = {(dora_out0 - dora_base).abs().max():.2e}  "
          f"{'OK' if f_ok else 'FAIL'}")
    assert f_ok

    print("\nall LoRA / DoRA sanity checks passed ✓")


if __name__ == "__main__":
    main()
