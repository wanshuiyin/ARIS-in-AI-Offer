"""
DiT - minimal runnable implementation
=====================================

Educational PyTorch reference for the class-conditional Diffusion Transformer
(Peebles & Xie 2023): patchify -> N x [adaLN-Zero DiT block] -> final layer
-> unpatchify. Sizes are tiny so the whole script runs on CPU in seconds.

Pairs with: docs/tutorials/image_generation_systems_tutorial.md §4
(DiT architecture and adaLN-Zero).

What the sanity checks verify (the claims in §4.2):
    1. At init every block is the identity: gate alpha = 0, and the modulated
       norm uses (1 + gamma) so gamma = 0 leaves LN(x) intact.
    2. The identity is NOT a dead end: for a block with a loss on its output,
       dL/d(alpha) != 0 at step 0, so the zero-initialised modulation MLP
       receives gradient and the block starts learning.
    3. gamma and beta get zero gradient at step 0 — their only path to the
       loss runs through alpha = 0.
    4. patchify / unpatchify round-trip exactly.
    5. In the whole model the final layer is zero-initialised as well, so step
       0 updates only its output linear; the blocks are reached from step 1.

Run:
    python dit.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of (possibly continuous) timesteps. [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sincos_pos_embed_2d(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding for a grid x grid patch grid. -> [grid*grid, dim]."""
    assert dim % 4 == 0
    quarter = dim // 4
    omega = 1.0 / 10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter)
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                            torch.arange(grid, dtype=torch.float32), indexing="ij")
    out = []
    for coord in (ys.reshape(-1), xs.reshape(-1)):  # [N]
        a = coord[:, None] * omega[None]            # [N, quarter]
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, dim=1)                    # [N, dim]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    """Class embedding with an extra "null" class for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden: int):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, hidden)
        self.null_id = num_classes

    def forward(self, y: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        if drop_mask is not None:
            y = torch.where(drop_mask, torch.full_like(y, self.null_id), y)
        return self.table(y)


# ---------------------------------------------------------------------------
# adaLN-Zero block
# ---------------------------------------------------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # (1 + scale), not scale: with a zero-initialised modulation MLP the
    # normalised activations pass through unchanged instead of being zeroed.
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)          # [B, H, N, d]
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    """x -> x + alpha1 * Attn(mod(LN(x))) -> x + alpha2 * MLP(mod(LN(x)))."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden * mlp_ratio)), nn.GELU(approximate="tanh"),
                                 nn.Linear(int(hidden * mlp_ratio), hidden))
        # One MLP produces all six modulation vectors; its last layer is
        # zero-initialised -> shift = scale = gate = 0 at step 0 (adaLN-Zero).
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mod: torch.Tensor | None = None) -> torch.Tensor:
        # `mod` may be passed in so a caller can inspect its gradient (see [D03]).
        if mod is None:
            mod = self.ada(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=1)
        x = x + gate1[:, None, :] * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2[:, None, :] * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class FinalLayer(nn.Module):
    """adaLN (shift, scale only) + linear to patch pixels; both zero-initialised."""

    def __init__(self, hidden: int, patch: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch * patch * out_channels)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        for lin in (self.ada[1], self.linear):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class DiT(nn.Module):
    def __init__(self, input_size: int = 8, patch: int = 2, in_channels: int = 4, hidden: int = 64,
                 depth: int = 4, heads: int = 4, num_classes: int = 10, learn_sigma: bool = True):
        super().__init__()
        assert input_size % patch == 0
        self.patch, self.in_channels = patch, in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.grid = input_size // patch
        self.x_embed = nn.Linear(patch * patch * in_channels, hidden)   # patchify == linear on flattened patches
        self.register_buffer("pos_embed", sincos_pos_embed_2d(hidden, self.grid), persistent=False)
        self.t_embed = TimestepEmbedder(hidden)
        self.y_embed = LabelEmbedder(num_classes, hidden)
        self.blocks = nn.ModuleList(DiTBlock(hidden, heads) for _ in range(depth))
        self.final = FinalLayer(hidden, patch, self.out_channels)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N, p*p*C] with N = (H/p)*(W/p), row-major over patches."""
        B, C, H, W = x.shape
        p = self.patch
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)   # [B, h, w, p, p, C]
        return x.reshape(B, (H // p) * (W // p), p * p * C)

    def unpatchify(self, tokens: torch.Tensor, channels: int) -> torch.Tensor:
        """[B, N, p*p*C] -> [B, C, H, W]; exact inverse of patchify."""
        B, N, _ = tokens.shape
        p, g = self.patch, self.grid
        x = tokens.reshape(B, g, g, p, p, channels).permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        return x.reshape(B, channels, g * p, g * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.x_embed(self.patchify(x)) + self.pos_embed[None]   # [B, N, hidden]
        c = self.t_embed(t) + self.y_embed(y, drop_mask)            # [B, hidden]
        for blk in self.blocks:
            h = blk(h, c)
        return self.unpatchify(self.final(h, c), self.out_channels)  # [B, out_C, H, W]


# ---------------------------------------------------------------------------
# Sanity checks — the §4.2 claims, executed
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(0)
    model = DiT()
    B = 3
    x = torch.randn(B, 4, 8, 8)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 10, (B,))

    # 4. patchify / unpatchify round-trip is exact
    tokens = model.patchify(x)
    assert tokens.shape == (B, 16, 16)
    assert torch.equal(model.unpatchify(tokens, 4), x)
    print(f"[D01] patchify: {tuple(x.shape)} -> {tuple(tokens.shape)} -> back, exact  PASS")

    # 1. every block is the identity at init
    h = model.x_embed(tokens) + model.pos_embed[None]
    c = model.t_embed(t) + model.y_embed(y)
    for i, blk in enumerate(model.blocks):
        assert torch.equal(blk(h, c), h), f"block {i} not identity at init"
    out = model(x, t, y)
    assert out.shape == (B, 8, 8, 8) and torch.equal(out, torch.zeros_like(out))
    print(f"[D02] adaLN-Zero: {len(model.blocks)} blocks are the identity at init; final layer outputs 0  PASS")

    # 2./3. gradient at step 0 for ONE block with a loss on its output (the
    # per-block derivation in §4.2): dL/d(alpha) != 0, dL/d(gamma) = dL/d(beta) = 0.
    blk = model.blocks[0]
    mod = blk.ada(c.detach())                          # [B, 6D] = (shift1, scale1, gate1, shift2, scale2, gate2), all 0
    mod.retain_grad()
    loss = (blk(h.detach(), c, mod=mod) - torch.randn_like(h)).pow(2).mean()
    loss.backward()
    shift1, scale1, gate1, shift2, scale2, gate2 = mod.grad.chunk(6, dim=1)
    gates, others = torch.cat([gate1, gate2]), torch.cat([shift1, scale1, shift2, scale2])
    assert gates.abs().max() > 0, "gate alpha got no gradient at init"
    assert torch.equal(others, torch.zeros_like(others)), "shift/scale got gradient through alpha=0"
    print(f"[D03] one block, step 0: |dL/d(alpha)|max={gates.abs().max():.3e} (nonzero), "
          f"dL/d(gamma)=dL/d(beta)=0 exactly  PASS")

    # Whole model from init: the final layer is zero-initialised too, so at
    # step 0 only its output linear gets a nonzero gradient — nothing reaches
    # the blocks. After that update the linear is nonzero, step 1's backward
    # reaches the blocks' gates, and after step 1 they leave the identity.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.randn(B, 8, 8, 8)
    for step in range(2):
        opt.zero_grad()
        F.mse_loss(model(x, t, y), target).backward()
        blocks_have_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in model.blocks.parameters())
        assert blocks_have_grad == (step == 1), f"step {step}: blocks_have_grad={blocks_have_grad}"
        opt.step()
    assert not torch.equal(model.blocks[0](h, c), h)
    print("[D04] full model: step 0 updates only the final linear; step 1's gradient reaches the blocks; "
          "after it block 0 is no longer the identity  PASS")

    # CFG plumbing: dropping the label routes to the null class
    drop = torch.tensor([True, False, True])
    emb = model.y_embed(y, drop)
    assert torch.equal(emb[0], model.y_embed.table.weight[model.y_embed.null_id])
    assert torch.equal(emb[1], model.y_embed.table.weight[y[1]])
    print("[D05] label dropout -> null-class embedding (CFG training)  PASS")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nall DiT sanity checks passed  ({n_params:,} params, hidden=64, depth=4, 8x8x4 latent, patch=2)")


if __name__ == "__main__":
    main()
