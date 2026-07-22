"""
Transformer Block: from 2017 seq2seq to modern LLM blocks - minimal runnable implementation
=============================================================================================

Educational PyTorch reference for how a Transformer *block* is assembled: residual
topology (Post-LN / serial Pre-LN / parallel Pre-LN), the encoder-decoder -> decoder-only
topological fork, RoPE + KV-cache assembly, MHA/GQA/MQA as one configurable axis, and
dense-vs-MoE FFN as an orthogonal, swappable interface.
Standalone script: runs sanity checks [D01]-[D15] on CPU in well under a minute
([D01]-[D09] are the core assembly/contract checks; [D10]-[D15] are targeted
regression tests, one per code-review bug fix, see §5 below).

Pairs with: docs/tutorials/transformer_block_tutorial.md (concept reference).

Scope note (assembly-level, not derivation-level): this file does NOT re-derive
softmax attention, FlashAttention, RoPE/YaRN/MLA algebra, MoE routing/capacity math,
or Xiong/DeepNorm gradient proofs -- those live in the sibling tutorials'
code/*.py files (normalization.py, linear_sparse_attention.py) and are only cited
here. This file only assembles known pieces into four canonical block classes and
verifies the *contracts* between them.

Entry classes (four clearly distinct block "eras"):
    VanillaEncoderLayer2017        - Post-LN, self-attn -> FFN (Vaswani et al. 2017)
    VanillaSeq2SeqDecoderLayer2017 - Post-LN, masked self-attn -> cross-attn -> FFN
    GPT2StyleDecoderLayer          - Pre-LN, LayerNorm, causal self-attn -> FFN, GELU, bias
    ModernDecoderBlock             - Pre-RMSNorm, RoPE, GQA-capable, SwiGLU/MoE-swappable

Sanity checks:
    [D01] Shape contract: all four blocks preserve residual width [B,T,d];
          cross-attn correctly decouples query length T from memory length S,
          AND changing memory actually changes the output (memory is read,
          not merely shape-compatible).
    [D02] Residual topology test: zeroing every sublayer's output projection makes
          a Pre-LN block return x EXACTLY; a real two-sublayer Post-LN block
          returns exactly ln2(ln1(x)) (not just "something != x").
    [D03] Causal non-leakage: perturbing tokens after position t must not change
          the output at positions <= t; perturbing position 0 instead must
          change positions > 0 (proves attention mixes forward, ruling out a
          dead/zeroed attention branch trivially "passing" the first check).
    [D04] Full forward vs cached decoding: eval(), no dropout, full-sequence
          forward vs token-by-token incremental KV-cache decoding must match
          numerically within a small tolerance (NOT bit-identical); the
          cache's actual K/V CONTENT is separately checked against an
          independently projected+rotated reference, not just cache shape.
    [D05] MHA is the n_kv_heads==n_q_heads degenerate case of GQA (compared
          against an independent plain-MHA reference function); a REAL grouped
          case (n_kv_heads=2, group=4) and MQA (n_kv_heads=1) are both checked
          against an independent repeat_interleave() reference, and MQA's
          per-query-head outputs are confirmed distinct (group dim isn't
          accidentally collapsed) while the cache never grows past 1 head.
    [D06] FFN parameter parity (d=96, dense_hidden=384, gated_hidden=256 both
          give exactly 73,728 matrix parameters) + attention parameter formula
          2*d^2*(1+n_kv/n_q), verified against real .numel() counts.
    [D07] Packed projection equivalence: a single packed QKV (or gate+up) Linear
          built from the same weights as three (or two) separate Linears must
          give numerically equivalent output (not a bit-identical guarantee
          across arbitrary GEMM shapes/backends).
    [D08] Parallel vs sequential dependency: forward hooks on BOTH the attn
          module (input+output) and the FFN module, captured during the SAME
          real forward call, prove that a serial Pre-LN block's FFN reads the
          attn-UPDATED residual stream while a GPT-J/PaLM-style parallel
          block's attn and FFN read the literal SAME normalized tensor.
    [D09] Backward smoke test: all block parameters (including the MoE
          router/gate and the rest of the block, not just expert weights) get
          finite AND non-zero gradients; for an FFN swapped to a minimal
          top-1 MoE, only experts that actually received a token this batch
          may have a non-None/non-zero gradient.

Additional regression tests [D10]-[D15] (§5): one per B1-class bug found
during code review -- config validation (indivisible d_model/heads, MoE
top_k), KV-cache/start_pos consistency, cross-attention + RoPE/cache
restrictions, masked-softmax NaN safety + mask-shape collision guards,
default-causal safety net, and RoPE cache device/dtype handling.

Run:
    python transformer_block.py
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# =============================================================================
# §1 Shared building blocks (used by more than one of the four block classes)
# =============================================================================

def build_causal_mask(q_positions: torch.Tensor, k_positions: torch.Tensor) -> torch.Tensor:
    """Boolean keep-mask [Tq, Tk]: True where key position <= query position.

    Works for both a full-sequence forward (q_positions == k_positions == arange(T))
    and a single incremental decode step (q_positions = [t], k_positions =
    arange(0, cache_len)) -- same formula, same function, see [D04].
    """
    return k_positions[None, :] <= q_positions[:, None]


def build_rope_cache(max_len: int, head_dim: int, base: float = 10000.0,
                      device=None, dtype=None):
    """cos/sin cache for RoPE (rotate-half convention), shape [max_len, head_dim].

    This is an assembly-level utility only -- for the *why* (complex-plane
    derivation, frequency choice, YaRN/NTK extensions), see
    long_context_rope_yarn_mla_tutorial.md §2/§5/§6.

    `device`/`dtype` are accepted explicitly (bug fix: a cache silently fixed
    to CPU/float32 breaks the moment the model is moved to GPU or run under
    mixed precision -- see [D15]). The trig itself is always computed in
    float32 for numerical stability, then cast to `dtype` at the end.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got head_dim={head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))  # [dh/2]
    t = torch.arange(max_len, device=device).float()                              # [L]
    freqs = torch.outer(t, inv_freq)                                             # [L, dh/2]
    emb = torch.cat([freqs, freqs], dim=-1)                                      # [L, dh]
    cos, sin = emb.cos(), emb.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, L, dh]; cos/sin: [L, dh]. Applied to Q/K ONLY -- V is never
    rotated (assembly rule, see markdown §3 / §4 [C007]).

    Defensively casts cos/sin to x's device/dtype (bug fix: a RoPE cache built
    once on CPU/float32 must not silently device-mismatch-crash on GPU, nor
    silently upcast a low-precision matmul -- see [D15])."""
    cos = cos[None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin = sin[None, None, :, :].to(device=x.device, dtype=x.dtype)
    return x * cos + _rotate_half(x) * sin


class KVCache:
    """Minimal single-layer incremental KV cache.

    Stores K, V at shape [B, n_kv_heads, T, d_head] and grows along the T axis
    via .append(). The head dimension is ALWAYS n_kv_heads -- it must never be
    physically broadcast/copied up to n_q_heads before being written back
    (that broadcast happens only transiently inside attention, see
    MultiHeadAttention.forward and [D05]).

    Bug fix (see [D11]): a cache used to silently accept anything .append()-ed
    to it, with no record of "what RoPE position comes next" and no check that
    a new chunk actually belongs to the same (batch, n_kv_heads, d_head,
    device, dtype) sequence as what's already stored. That made two silent
    failure modes possible: (1) calling MultiHeadAttention.forward with a
    stale/mismatched `start_pos` against a non-empty cache re-uses the wrong
    RoPE angles for the new tokens; (2) reusing one KVCache instance across two
    different sequences/configs silently concatenates unrelated K/V. This
    class now tracks `next_pos` (the RoPE position the next appended chunk
    must start at) and validates shape/device/dtype identity on every append;
    call `.reset()` to reuse an instance for a new sequence.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.k = None
        self.v = None
        self.next_pos = 0
        self._sig = None    # (batch, n_kv_heads, d_head, device, dtype) of the stored sequence

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        B, H, T, dh = k_new.shape
        sig = (B, H, dh, k_new.device, k_new.dtype)
        if self.k is None:
            self._sig = sig
            self.k, self.v = k_new, v_new
        else:
            if sig != self._sig:
                raise ValueError(
                    f"KVCache.append: shape/device/dtype {sig} does not match the cache's "
                    f"existing sequence {self._sig}. Reusing one KVCache across two different "
                    f"sequences or configs silently splices unrelated K/V together -- call "
                    f".reset() before starting a new sequence.")
            # Performance note (teaching limitation, not a correctness bug):
            # torch.cat here re-copies the ENTIRE history on every single
            # append, an O(T^2) total-copy cost over a T-token decode. A real
            # serving cache pre-allocates a fixed-capacity buffer and writes
            # new tokens in place (or manages paged/chunked blocks). Fine for
            # this file's 6-token demo; not representative of production KV
            # cache implementations -- see run-experiment/inference_serving
            # sibling material for the real thing.
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        self.next_pos += T
        return self.k, self.v


def _infer_start_pos(kv_cache: "KVCache", start_pos):
    """Single source of truth for "what RoPE position does this call start at".

    Bug fix (see [D11]): the old default `start_pos: int = 0` meant that
    calling with a non-empty cache and no explicit start_pos silently reused
    RoPE angle 0 for new tokens instead of continuing from the cache. Passing
    `start_pos=None` (the new default) now means "ask the cache"; passing an
    explicit start_pos that disagrees with the cache's recorded position is a
    ValueError instead of a silent wrong-position bug.
    """
    if kv_cache is not None:
        inferred = kv_cache.next_pos
        if start_pos is None:
            return inferred
        if start_pos != inferred:
            raise ValueError(
                f"start_pos={start_pos} is inconsistent with kv_cache.next_pos={inferred}. "
                "Pass start_pos=None to let the cache supply the position, or call "
                "kv_cache.reset() if this is meant to start a new sequence.")
        return start_pos
    return 0 if start_pos is None else start_pos


class MultiHeadAttention(nn.Module):
    """Generic multi-head attention: self- or cross-attention, optional causal
    masking, optional GQA (n_kv_heads <= n_q_heads, MHA when equal, MQA when
    n_kv_heads==1), optional RoPE, optional incremental KV cache.

    Interface-level only: for *why* softmax(QK^T/sqrt(d)) and *why* GQA saves KV
    cache memory, see attention_tutorial.md §2 / §6. This class only assembles
    the pieces into the shape contract a Transformer block needs.

    Mask convention (bug fix, see [D13]): two SEPARATE, unambiguous mask
    arguments are accepted instead of one shape-guessed `mask` -- `causal_mask`
    is the [Tq, Tk] connectivity pattern, `key_padding_mask` is the [B, Tk]
    per-batch-element key validity. Both use "True = ATTEND / keep" (the
    OPPOSITE of the common PyTorch nn.MultiheadAttention `key_padding_mask`
    True-means-mask-OUT convention -- double check when porting real code).
    A single generic `mask` argument that could be either [Tq, Tk] or [B, Tk]
    is ambiguous and, worse, silently WRONG whenever B happens to equal Hkv or
    Tq (the tensor still broadcasts, just onto the wrong dimension).

    Combination limits that are NOT supported by this minimal reference (bug
    fix, see [D12]): cross-attention (`x_kv is not x_q`) together with RoPE or
    with a KV cache. Cross-attention's query and memory generally occupy
    different position spaces (a single `start_pos` cannot represent both),
    and repeatedly caching the same encoder memory would re-append it every
    call. Both combinations raise ValueError rather than silently computing
    something wrong.
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int = None,
                 bias: bool = False, rope: bool = False):
        super().__init__()
        n_kv_heads = n_q_heads if n_kv_heads is None else n_kv_heads
        if n_q_heads <= 0 or n_kv_heads <= 0:
            raise ValueError(f"n_q_heads ({n_q_heads}) and n_kv_heads ({n_kv_heads}) must be positive")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        if d_model % n_q_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_q_heads ({n_q_heads}); "
                f"floor-dividing here would silently narrow attention to "
                f"{n_q_heads * (d_model // n_q_heads)} of {d_model} dims while still "
                f"projecting back out to d_model, passing shape checks with a wrong answer.")
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads
        self.d_head = d_model // n_q_heads
        if rope and self.d_head % 2 != 0:
            raise ValueError(
                f"RoPE requires an even per-head dim, got d_head={self.d_head} "
                f"(d_model={d_model}, n_q_heads={n_q_heads})")
        self.group = n_q_heads // n_kv_heads          # Q heads sharing one KV head
        self.rope = rope
        self.q_proj = nn.Linear(d_model, n_q_heads * self.d_head, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.o_proj = nn.Linear(n_q_heads * self.d_head, d_model, bias=bias)

    def _shape(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, n_heads, self.d_head).transpose(1, 2)   # [B, H, L, dh]

    def forward(self, x_q, x_kv=None, causal_mask=None, key_padding_mask=None,
                cos=None, sin=None, kv_cache: KVCache = None, start_pos: int = None):
        is_cross_attn = x_kv is not None and x_kv is not x_q
        x_kv = x_q if x_kv is None else x_kv              # self-attn vs cross-attn source
        B, Tq, _ = x_q.shape

        if is_cross_attn:
            # Bug fix (see [D12]): cross-attention's Q and K/V generally live in
            # DIFFERENT position spaces. A single start_pos/cache cannot represent
            # both without silently mis-positioning RoPE or re-appending the same
            # memory into the cache on every call -- refuse instead of guessing.
            if self.rope:
                raise ValueError(
                    "RoPE is not supported for cross-attention in this minimal reference: "
                    "query and memory occupy different position spaces and a single "
                    "start_pos cannot represent both. Construct this module with rope=False "
                    "for cross-attention, or apply positions explicitly outside this class.")
            if kv_cache is not None:
                raise ValueError(
                    "KV caching is not supported for cross-attention in this minimal reference: "
                    "repeated decode-step calls would re-append the SAME encoder memory into the "
                    "cache every time. Encode memory once outside any KVCache, or extend KVCache "
                    "with a write-once static-memory mode.")

        resolved_start = _infer_start_pos(kv_cache, start_pos)

        q = self._shape(self.q_proj(x_q), self.n_q_heads)      # [B, Hq,  Tq, dh]
        k_new = self._shape(self.k_proj(x_kv), self.n_kv_heads)  # [B, Hkv, Tk_new, dh]
        v_new = self._shape(self.v_proj(x_kv), self.n_kv_heads)

        if self.rope:
            Tk_new = k_new.shape[2]
            q = apply_rope(q, cos[resolved_start:resolved_start + Tq], sin[resolved_start:resolved_start + Tq])
            # K carries RoPE BEFORE entering the cache; V is never rotated.
            k_new = apply_rope(k_new, cos[resolved_start:resolved_start + Tk_new],
                                sin[resolved_start:resolved_start + Tk_new])

        if kv_cache is not None:
            k, v = kv_cache.append(k_new, v_new)          # cache head dim stays n_kv_heads
        else:
            k, v = k_new, v_new

        Tk = k.shape[2]
        # GQA broadcast: view Q as (Hkv groups x group size), broadcast K/V across
        # the group dim via unsqueeze (a view, NOT a physical copy written to cache).
        q_g = q.view(B, self.n_kv_heads, self.group, Tq, self.d_head)
        k_g = k.unsqueeze(2)                               # [B, Hkv, 1, Tk, dh]
        v_g = v.unsqueeze(2)

        scores = (q_g @ k_g.transpose(-2, -1)) / math.sqrt(self.d_head)   # [B, Hkv, G, Tq, Tk]

        # Bug fix (see [D13]): causal_mask [Tq, Tk] and key_padding_mask [B, Tk]
        # are reshaped EXPLICITLY by their known rank/role, never shape-guessed,
        # so there is no B==Hkv / B==Tq collision with the 5-D `scores` tensor.
        keep = None
        if causal_mask is not None:
            keep = causal_mask[None, None, None, :, :]                 # -> [1, 1, 1, Tq, Tk]
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, None, :]             # -> [B, 1, 1, 1, Tk]
            keep = kpm if keep is None else (keep & kpm)
        if keep is not None:
            scores = scores.masked_fill(~keep, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # Bug fix (see [D13]): a query row with NO valid keys (all -inf) produces
        # softmax NaN, not a numerical accident but a real hazard for padded
        # batches/empty cross-memory. Safe masked softmax: treat a fully-masked
        # row as contributing zero attention weight rather than propagating NaN.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = attn @ v_g                                    # [B, Hkv, G, Tq, dh]

        out = out.reshape(B, self.n_q_heads, Tq, self.d_head).transpose(1, 2)
        out = out.reshape(B, Tq, self.n_q_heads * self.d_head)
        return self.o_proj(out), (k, v)


class DenseFFN(nn.Module):
    """2-matrix FFN: down(act(up(x))). GELU by default (GPT-2-style)."""

    def __init__(self, d_model: int, d_ff: int, act=F.gelu, bias: bool = False):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.act = act

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class GatedFFN(nn.Module):
    """3-matrix gated FFN: down(act(gate(x)) * up(x)). SiLU gate -> SwiGLU
    (Llama-style); GELU gate -> GeGLU. Full "why gated FFN helps" discussion:
    markdown §3 (only the parameter-matching arithmetic is re-derived here, [D06])."""

    def __init__(self, d_model: int, d_ff: int, act=F.silu, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.act = act

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Minimal top-1 token-choice MoE FFN (Switch-style): same [B,T,d] -> [B,T,d]
    interface as DenseFFN/GatedFFN, so it drops into ModernDecoderBlock unchanged
    (markdown §5/§7 "FFN interface stays fixed" point). Routing / capacity /
    load-balancing-loss math is intentionally NOT re-derived here -- see
    moe_tutorial.md §2-§4 for the full treatment. `tokens_per_expert` is recorded
    each forward for the [D09] "only touched experts get gradients" check."""

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, top_k: int = 1):
        super().__init__()
        if top_k != 1:
            # Bug fix (see [D10]): this used to be `assert top_k == 1`, which (a)
            # disappears entirely under `python -O`, and (b) even when it DOES
            # fire, `forward()` below never reads `top_k` at all -- it always does
            # a `.max(dim=-1)` top-1 route regardless. Any top_k != 1 must be
            # rejected explicitly rather than silently downgraded to top-1.
            raise NotImplementedError(
                f"MoEFFN in this minimal assembly demo only implements top-1 routing; "
                f"got top_k={top_k}. Extending to top_k>1 needs per-token weighted "
                f"multi-expert combination, intentionally out of scope here -- see "
                f"moe_tutorial.md §2-§4.")
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [GatedFFN(d_model, d_ff, bias=False) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.tokens_per_expert = [0] * n_experts

    def forward(self, x):
        # Performance note (teaching limitation, not a correctness bug): this
        # is a plain Python for-loop over experts, and `sel.sum().item()` /
        # `sel.any()` each force a GPU->CPU sync every iteration. A real MoE
        # kernel dispatches tokens via a single vectorized gather/scatter (or
        # a fused grouped-GEMM) instead of per-expert boolean masking in a
        # Python loop -- see moe_tutorial.md for the real routing/dispatch
        # implementation. Fine for this file's few-token demo.
        B, T, d = x.shape
        flat = x.reshape(-1, d)                                # [B*T, d]
        gate_w, idx = self.gate(flat).softmax(-1).max(dim=-1)   # top-1 weight + expert id
        out = torch.zeros_like(flat)
        self.tokens_per_expert = [0] * len(self.experts)
        for e, expert in enumerate(self.experts):
            sel = idx == e
            self.tokens_per_expert[e] = int(sel.sum().item())
            if sel.any():
                out[sel] = expert(flat[sel]) * gate_w[sel].unsqueeze(-1)
        return out.view(B, T, d)


# =============================================================================
# §2 Four canonical block classes ("eras")
# =============================================================================

class VanillaEncoderLayer2017(nn.Module):
    """Post-LN encoder layer (Vaswani et al. 2017): self-attn -> FFN, TWO
    sublayers, each y = N(x + Sublayer(x)). Self-attn is bidirectional (mask, if
    any, is padding-only, never causal). Sinusoidal PE lives OUTSIDE this block
    (added to embeddings before the stack) -- see markdown §1."""

    def __init__(self, d_model: int = 96, n_heads: int = 8, d_ff: int = 384):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, n_heads, bias=True)
        self.ffn = DenseFFN(d_model, d_ff, act=F.relu, bias=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        a, _ = self.attn(x, key_padding_mask=pad_mask)
        x = self.ln1(x + a)          # Post-LN: y = N(x + F(x))
        f = self.ffn(x)
        x = self.ln2(x + f)
        return x


class VanillaSeq2SeqDecoderLayer2017(nn.Module):
    """Post-LN decoder layer (Vaswani et al. 2017): masked self-attn -> cross-attn
    -> FFN, THREE sublayers -- the sublayer that decoder-only LLM blocks later
    drop entirely (not merely reconfigure, see markdown §2). Cross-attn: Q comes
    from THIS stream (length T), K/V come from encoder memory (length S); T and
    S need not match."""

    def __init__(self, d_model: int = 96, n_heads: int = 8, d_ff: int = 384):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, n_heads, bias=True)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, n_heads, bias=True)
        self.ffn = DenseFFN(d_model, d_ff, act=F.relu, bias=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, causal_mask=None, cross_mask=None):
        # Bug fix (see [D14]): self_attn here is documented as MASKED
        # (causal) self-attention. The old code passed `mask=None` straight
        # through whenever the caller omitted it, which silently ran
        # BIDIRECTIONAL self-attention instead of causal -- no error, just a
        # quietly wrong contract. Auto-build the causal mask when omitted.
        if causal_mask is None:
            T = x.shape[1]
            causal_mask = build_causal_mask(torch.arange(T), torch.arange(T))
        a, _ = self.self_attn(x, causal_mask=causal_mask)
        x = self.ln1(x + a)
        c, _ = self.cross_attn(x, x_kv=memory, causal_mask=cross_mask)   # Q len T, K/V len S
        x = self.ln2(x + c)
        f = self.ffn(x)
        x = self.ln3(x + f)
        return x


class GPT2StyleDecoderLayer(nn.Module):
    """Pre-LN decoder-only block (GPT-2, Radford et al. 2019): causal self-attn
    -> FFN, TWO sublayers -- cross-attn is structurally ABSENT, not merely
    unused (markdown §2 explains why this is a topological fork). GPT-2 is
    ALREADY Pre-LN with LayerNorm, learned absolute PE added outside the stack,
    a GELU dense FFN, and bias everywhere. A final LayerNorm after the LAST
    block in the stack is required (not modeled per-block here) -- see
    normalization_init_tutorial.md §5.3 and markdown §2.2."""

    def __init__(self, d_model: int = 96, n_heads: int = 8, d_ff: int = 384):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, n_heads, bias=True)
        self.ffn = DenseFFN(d_model, d_ff, act=F.gelu, bias=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask=None):
        # Bug fix (see [D14]): same silent-bidirectional hazard as above --
        # this block's docstring/class name promise CAUSAL self-attention, so
        # omitting the mask must not silently fall back to full attention.
        if causal_mask is None:
            T = x.shape[1]
            causal_mask = build_causal_mask(torch.arange(T), torch.arange(T))
        a, _ = self.attn(self.ln1(x), causal_mask=causal_mask)   # Pre-LN: y = x + F(N(x))
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class ModernDecoderBlock(nn.Module):
    """Pre-RMSNorm + RoPE + (GQA-capable) causal self-attn + gated FFN
    ("Llama-style core recipe", markdown §3/§7). No bias. RoPE applied inside
    attention to Q/K only. Supports incremental KV-cache decoding. `ffn` is
    swappable (DenseFFN / GatedFFN / MoEFFN) -- the block's own interface never
    changes, only what sits behind it (markdown §5)."""

    def __init__(self, d_model: int = 96, n_q_heads: int = 8, n_kv_heads: int = 2,
                 d_ff: int = 256, ffn: nn.Module = None):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_q_heads, n_kv_heads, bias=False, rope=True)
        self.ffn = ffn if ffn is not None else GatedFFN(d_model, d_ff, act=F.silu, bias=False)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x, cos, sin, causal_mask=None, kv_cache: KVCache = None, start_pos: int = None):
        # Bug fix (see [D14]): resolve the position BEFORE deciding the mask, so
        # an omitted causal_mask still gets built over the correct absolute
        # positions when a non-empty KV cache is present (query positions
        # continue from where the cache left off, not from 0).
        resolved_start = _infer_start_pos(kv_cache, start_pos)
        Tq = x.shape[1]
        if causal_mask is None:
            q_pos = torch.arange(resolved_start, resolved_start + Tq)
            k_pos = torch.arange(0, resolved_start + Tq)
            causal_mask = build_causal_mask(q_pos, k_pos)
        a, _ = self.attn(self.norm1(x), causal_mask=causal_mask, cos=cos, sin=sin,
                          kv_cache=kv_cache, start_pos=resolved_start)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# §3 Two extra minimal blocks, ONLY for the [D08] parallel-vs-sequential test
# =============================================================================

class SequentialPreLNBlock(nn.Module):
    """Serial Pre-LN, independent norms: attn updates the residual stream BEFORE
    FFN reads it. x1 = x + Attn(N1(x)); y = x1 + FFN(N2(x1))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, n_heads, bias=False)
        self.ffn = GatedFFN(d_model, d_ff, bias=False)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x, causal_mask=None):
        a, _ = self.attn(self.norm1(x), causal_mask=causal_mask)
        x1 = x + a
        return x1 + self.ffn(self.norm2(x1))


class ParallelPreLNBlock(nn.Module):
    """GPT-J/PaLM-style parallel residual: y = x + Attn(N(x)) + FFN(N(x)) -- FFN
    reads the SAME normalized ORIGINAL x that attention reads, never the
    attn-updated stream. A single shared norm is used here to match the
    GPT-J/PaLM core formula; sharing norm PARAMETERS is not required for the
    "no dependency" property this test checks (markdown §6)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, n_heads, bias=False)
        self.ffn = GatedFFN(d_model, d_ff, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x, causal_mask=None):
        n = self.norm(x)
        a, _ = self.attn(n, causal_mask=causal_mask)
        f = self.ffn(n)
        return x + a + f


def reference_mha(x, q_w, k_w, v_w, o_w, n_heads, causal_mask, cos, sin):
    """Plain, non-grouped reference MHA (every head has its own independent K/V,
    no group broadcasting at all) -- written independently of MultiHeadAttention,
    used only to validate that MultiHeadAttention(n_kv_heads=n_q_heads) is
    mathematically the same computation ([D05])."""
    B, T, d = x.shape
    dh = d // n_heads

    def shape(w):
        return (x @ w.t()).view(B, T, n_heads, dh).transpose(1, 2)

    q, k, v = shape(q_w), shape(k_w), shape(v_w)
    q = apply_rope(q, cos[:T], sin[:T])
    k = apply_rope(k, cos[:T], sin[:T])
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, T, d)
    return out @ o_w.t()


def reference_gqa(x, q_w, k_w, v_w, o_w, n_q_heads, n_kv_heads, causal_mask, cos, sin,
                   return_per_head=False):
    """Independent GQA/MQA reference that PHYSICALLY repeat_interleave()s K/V
    heads up to n_q_heads (group = n_q_heads // n_kv_heads) and runs ordinary
    per-head attention -- no view+unsqueeze broadcasting trick at all. Used to
    validate that MultiHeadAttention's view-based GQA broadcast is
    mathematically equivalent to literally copying K/V ([D05]). Unlike
    reference_mha, this covers group sizes > 1 (n_kv_heads < n_q_heads), which
    reference_mha (group == 1 only) cannot exercise."""
    B, T, d = x.shape
    dh = d // n_q_heads
    group = n_q_heads // n_kv_heads

    def shape(w, n_heads):
        return (x @ w.t()).view(B, T, n_heads, dh).transpose(1, 2)

    q = shape(q_w, n_q_heads)
    k = shape(k_w, n_kv_heads)
    v = shape(v_w, n_kv_heads)
    q = apply_rope(q, cos[:T], sin[:T])
    k = apply_rope(k, cos[:T], sin[:T])
    k_rep = k.repeat_interleave(group, dim=1)     # [B, Hq, T, dh] -- physical copy, not a view
    v_rep = v.repeat_interleave(group, dim=1)
    scores = (q @ k_rep.transpose(-2, -1)) / math.sqrt(dh)
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    per_head_out = attn @ v_rep                    # [B, Hq, T, dh], BEFORE o_proj
    out = per_head_out.transpose(1, 2).reshape(B, T, d)
    result = out @ o_w.t()
    if return_per_head:
        return result, per_head_out
    return result


# =============================================================================
# §4 Sanity checks [D01]-[D09]
# =============================================================================

def check_d01_shape_contract():
    B, T, S, d, n_q, n_kv, d_ff = 2, 7, 5, 96, 8, 2, 384
    x = torch.randn(B, T, d)
    memory = torch.randn(B, S, d)          # deliberately S != T
    causal = build_causal_mask(torch.arange(T), torch.arange(T))

    enc = VanillaEncoderLayer2017(d, n_q, d_ff)
    dec2017 = VanillaSeq2SeqDecoderLayer2017(d, n_q, d_ff)
    gpt2 = GPT2StyleDecoderLayer(d, n_q, d_ff)
    cos, sin = build_rope_cache(T, d // n_q)
    modern = ModernDecoderBlock(d, n_q, n_kv, d_ff=256)

    out_enc = enc(x)
    out_dec = dec2017(x, memory, causal_mask=causal)                  # Q len T, K/V len S
    out_gpt2 = gpt2(x, causal_mask=causal)
    out_modern = modern(x, cos, sin, causal_mask=causal)

    shapes = [out_enc.shape, out_dec.shape, out_gpt2.shape, out_modern.shape]
    shapes_ok = all(s == (B, T, d) for s in shapes)
    # cross-attn ran with S=5 != T=7 memory and still returned length-T output:
    # this only works if cross-attn correctly decouples query length from K/V length.
    cross_len_ok = out_dec.shape[1] == T and memory.shape[1] == S and S != T

    # Bug-strength fix (see [D01] in the review ledger): matching shapes alone
    # passes even if cross-attn completely IGNORES memory (e.g. a bug that
    # accidentally reads x_q instead of x_kv). Re-run with different memory and
    # require the output to actually change -- this proves memory is read, not
    # just shape-compatible.
    memory2 = torch.randn(B, S, d)
    out_dec2 = dec2017(x, memory2, causal_mask=causal)
    memory_matters = not torch.allclose(out_dec, out_dec2, atol=1e-6)

    ok = shapes_ok and cross_len_ok and memory_matters
    print(f"[D01] shapes (enc/dec2017/gpt2/modern) = {shapes} (expect 4x ({B},{T},{d})); "
          f"cross-attn ran with Tq={T} != S={S}; changing memory changes cross-attn output "
          f"(memory is actually read, not ignored) = {memory_matters}  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d02_residual_topology():
    d = 96
    x = torch.randn(4, 5, d)
    ln = nn.LayerNorm(d)

    def zero_sublayer(_x):
        return torch.zeros_like(_x)

    # Formula-level proof (§0's three canonical equations, F(.) == 0):
    post_ln_out = ln(x + zero_sublayer(x))       # y = N(x + F(x)) -> y = N(x)
    pre_ln_out = x + zero_sublayer(ln(x))        # y = x + F(N(x)) -> y = x

    # rtol=0 below: these are declared-EXACT claims (F=0 should give EXACTLY
    # N(x) / EXACTLY x), so the default rtol=1e-5 fudge-factor must not be
    # allowed to paper over a real discrepancy.
    post_matches_Nx = torch.allclose(post_ln_out, ln(x), atol=1e-6, rtol=0)
    post_matches_x = torch.allclose(post_ln_out, x, atol=1e-4)
    pre_matches_x = torch.allclose(pre_ln_out, x, atol=1e-6, rtol=0)

    # Class-level corroboration: zero out every sublayer's OUTPUT projection
    # (weight AND bias) in a real Pre-LN block and a real Post-LN block.
    modern = ModernDecoderBlock(d, 8, 2, d_ff=256)
    with torch.no_grad():
        modern.attn.o_proj.weight.zero_()
        modern.ffn.down.weight.zero_()
    cos, sin = build_rope_cache(5, d // 8)
    causal = build_causal_mask(torch.arange(5), torch.arange(5))
    modern_out = modern(x, cos, sin, causal_mask=causal)
    modern_is_identity = torch.allclose(modern_out, x, atol=1e-6, rtol=0)

    enc = VanillaEncoderLayer2017(d, 8, 384)
    with torch.no_grad():
        enc.attn.o_proj.weight.zero_()
        enc.attn.o_proj.bias.zero_()
        enc.ffn.down.weight.zero_()
        enc.ffn.down.bias.zero_()
    enc_out = enc(x)
    enc_not_identity = not torch.allclose(enc_out, x, atol=1e-3)
    # Bug-strength fix (see [D02] in the review ledger): the old check only
    # verified enc_out != x, which any arbitrary wrong transform would also
    # satisfy. A REAL two-sublayer Post-LN block with both sublayer outputs
    # zeroed must equal exactly ln2(ln1(x) + 0) = ln2(ln1(x)) -- compare
    # against that closed form directly, not a generic "!= x" sanity check.
    enc_expected = enc.ln2(enc.ln1(x))
    enc_matches_expected = torch.allclose(enc_out, enc_expected, atol=1e-6, rtol=0)

    ok = (pre_matches_x and post_matches_Nx and (not post_matches_x)
          and modern_is_identity and enc_not_identity and enc_matches_expected)
    print(f"[D02] formula-level: Pre-LN(F=0)==x: {pre_matches_x}; Post-LN(F=0)==N(x): {post_matches_Nx} "
          f"(and != x: {not post_matches_x}); class-level: zeroed ModernDecoderBlock==x: {modern_is_identity}, "
          f"zeroed Post-LN encoder layer != x: {enc_not_identity}, == ln2(ln1(x)) exactly: "
          f"{enc_matches_expected}  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d03_causal_non_leakage():
    d, n_heads, d_ff, T, t = 96, 8, 384, 6, 3
    x = torch.randn(2, T, d)

    # Perturb ONLY a future token (after t) -- tests that changes after
    # position t do not leak BACKWARD into positions <= t.
    x_future_perturbed = x.clone()
    x_future_perturbed[:, t + 1:, :] = torch.randn_like(x_future_perturbed[:, t + 1:, :])

    # Bug-strength fix (see [D03] in the review ledger): "future positions
    # changed" is true even with a completely dead/zeroed attention branch,
    # because each future token's OWN value changed and still feeds its own
    # FFN through the residual stream -- that alone says nothing about causal
    # MIXING. Perturb an EARLIER token instead and require LATER positions to
    # change: that can only happen if attention is actually carrying
    # information FORWARD from position 0 to positions > 0.
    x_past_perturbed = x.clone()
    x_past_perturbed[:, 0, :] = torch.randn_like(x_past_perturbed[:, 0, :])

    causal = build_causal_mask(torch.arange(T), torch.arange(T))
    blk = GPT2StyleDecoderLayer(d, n_heads, d_ff)
    blk.eval()
    with torch.no_grad():
        o1 = blk(x, causal_mask=causal)
        o2 = blk(x_future_perturbed, causal_mask=causal)
        o3 = blk(x_past_perturbed, causal_mask=causal)

    # rtol=0: "unchanged" is a declared-EXACT claim (no backward leakage at all).
    past_unchanged = torch.allclose(o1[:, :t + 1], o2[:, :t + 1], atol=1e-6, rtol=0)
    future_did_change = not torch.allclose(o1[:, t + 1:], o2[:, t + 1:], atol=1e-6)
    forward_mixing_present = not torch.allclose(o1[:, 1:], o3[:, 1:], atol=1e-6)

    ok = past_unchanged and future_did_change and forward_mixing_present
    print(f"[D03] perturbing tokens after t={t}: positions 0..{t} unchanged = {past_unchanged}, "
          f"positions {t + 1}..{T - 1} changed = {future_did_change}; perturbing position 0 changes "
          f"positions 1..{T - 1} (attention actually mixes forward, not a dead branch) = "
          f"{forward_mixing_present}  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d04_full_vs_cached():
    d, n_q, n_kv, T = 96, 8, 2, 6
    x = torch.randn(2, T, d)
    cos, sin = build_rope_cache(T, d // n_q)
    blk = ModernDecoderBlock(d, n_q, n_kv, d_ff=256)
    blk.eval()

    causal = build_causal_mask(torch.arange(T), torch.arange(T))
    with torch.no_grad():
        out_full = blk(x, cos, sin, causal_mask=causal)

    cache = KVCache()
    outs = []
    with torch.no_grad():
        for t in range(T):
            xt = x[:, t:t + 1, :]
            q_pos = torch.tensor([t])
            k_pos = torch.arange(t + 1)               # cache so far, INCLUDING this new token
            mask_t = build_causal_mask(q_pos, k_pos)   # a single new query attends to all of 0..t
            ot = blk(xt, cos, sin, causal_mask=mask_t, kv_cache=cache, start_pos=t)
            outs.append(ot)
    out_cached = torch.cat(outs, dim=1)

    TOL = 1e-4   # bug-fix wording (see [D04] ledger): this is a numerical-equivalence
                 # tolerance, NOT bit-exactness -- do not call it "exact" in the report.
    max_diff = (out_full - out_cached).abs().max().item()
    cache_len_ok = cache.k.shape[2] == T and cache.k.shape[1] == n_kv

    # Bug-strength fix (see [D04] in the review ledger): full-vs-cached agreement
    # alone can't catch a bug shared by BOTH code paths (they route through the
    # same apply_rope/k_proj/v_proj), and only checks cache SHAPE, never cache
    # CONTENT. Independently project+rotate K/V for the WHOLE sequence in one
    # shot (bypassing the incremental per-step append path entirely) and
    # compare against what actually ended up sitting in the cache.
    with torch.no_grad():
        x_norm = blk.norm1(x)
        k_expected = blk.attn._shape(blk.attn.k_proj(x_norm), n_kv)
        k_expected = apply_rope(k_expected, cos[:T], sin[:T])
        v_expected = blk.attn._shape(blk.attn.v_proj(x_norm), n_kv)
    cache_k_diff = (cache.k - k_expected).abs().max().item()
    cache_v_diff = (cache.v - v_expected).abs().max().item()
    cache_content_ok = cache_k_diff < TOL and cache_v_diff < TOL

    ok = max_diff < TOL and cache_len_ok and cache_content_ok
    print(f"[D04] full-forward vs token-by-token cached decode: max|Δ| = {max_diff:.2e} "
          f"(tol {TOL:.0e}, numerically equivalent, NOT bit-identical); cache CONTENT vs "
          f"independently projected+rotated K/V: max|ΔK|={cache_k_diff:.2e}, "
          f"max|ΔV|={cache_v_diff:.2e}; final cache shape head_dim={cache.k.shape[1]} "
          f"(expect {n_kv}) len={cache.k.shape[2]} (expect {T})  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d05_mha_gqa_mqa():
    d, n_heads, T = 96, 8, 5
    x = torch.randn(2, T, d)
    cos, sin = build_rope_cache(T, d // n_heads)
    causal = build_causal_mask(torch.arange(T), torch.arange(T))

    # n_kv_heads == n_q_heads must match an INDEPENDENT plain-MHA reference.
    mha = MultiHeadAttention(d, n_heads, n_heads, bias=False, rope=True)
    out_mha, _ = mha(x, causal_mask=causal, cos=cos, sin=sin)
    out_ref = reference_mha(x, mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight,
                             mha.o_proj.weight, n_heads, causal, cos, sin)
    degenerate_to_mha_ok = torch.allclose(out_mha, out_ref, atol=1e-5)

    # Bug-strength fix (see [D05] in the review ledger): n_kv==n_q above only
    # exercises group==1 and can't catch a wrong repeat ORDER (repeat vs
    # repeat_interleave) or a wrong group-to-head assignment. Test a REAL
    # grouping (n_kv_heads=2, group size 4) against an independent
    # repeat_interleave reference.
    n_kv2 = 2
    gqa2 = MultiHeadAttention(d, n_heads, n_kv2, bias=False, rope=True)
    out_gqa2, _ = gqa2(x, causal_mask=causal, cos=cos, sin=sin)
    out_gqa2_ref = reference_gqa(x, gqa2.q_proj.weight, gqa2.k_proj.weight, gqa2.v_proj.weight,
                                  gqa2.o_proj.weight, n_heads, n_kv2, causal, cos, sin)
    gqa2_ok = torch.allclose(out_gqa2, out_gqa2_ref, atol=1e-5)

    # n_kv_heads == 1 (MQA): cache head dim must stay 1 (never physically copied
    # up to n_q_heads); output must match the SAME repeat_interleave reference
    # specialized to n_kv_heads=1 (not just "some shape"); and different query
    # heads must produce genuinely DIFFERENT per-head outputs (proving the
    # group dimension is broadcast independently per head, not collapsed to a
    # single shared result before o_proj mixes everything back together).
    mqa = MultiHeadAttention(d, n_heads, 1, bias=False, rope=True)
    cache = KVCache()
    out_mqa, (k_cached, _) = mqa(x, causal_mask=causal, cos=cos, sin=sin, kv_cache=cache)
    out_mqa_ref, per_head_out = reference_gqa(x, mqa.q_proj.weight, mqa.k_proj.weight, mqa.v_proj.weight,
                                               mqa.o_proj.weight, n_heads, 1, causal, cos, sin,
                                               return_per_head=True)
    mqa_matches_ref = torch.allclose(out_mqa, out_mqa_ref, atol=1e-5)
    mqa_cache_ok = k_cached.shape[1] == 1
    mqa_shape_ok = out_mqa.shape == (2, T, d)
    heads_are_distinct = not torch.allclose(per_head_out[:, 0], per_head_out[:, 1], atol=1e-6)

    ok = (degenerate_to_mha_ok and gqa2_ok and mqa_matches_ref and mqa_cache_ok
          and mqa_shape_ok and heads_are_distinct)
    print(f"[D05] GQA(n_kv=n_q) == independent reference MHA: max|Δ|="
          f"{(out_mha - out_ref).abs().max():.2e}; GQA(n_kv=2, group=4) == independent "
          f"repeat_interleave reference: max|Δ|={(out_gqa2 - out_gqa2_ref).abs().max():.2e}; "
          f"MQA == same reference (n_kv=1): max|Δ|={(out_mqa - out_mqa_ref).abs().max():.2e}, "
          f"cache head_dim={k_cached.shape[1]} (expect 1), output shape={tuple(out_mqa.shape)}, "
          f"per-query-head outputs distinct (not collapsed) = {heads_are_distinct}  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok


def check_d06_param_formulas():
    # Note: this formula only holds for a config where d_model divides evenly
    # by n_q_heads (as validated at construction time -- see [D10]); an
    # indivisible config is now rejected with ValueError instead of silently
    # producing a DIFFERENT, wrong parameter count that would fail this check
    # for the wrong reason.
    d = 96
    dense_hidden, gated_hidden = 384, 256                 # 4d and 8d/3 respectively
    dense = DenseFFN(d, dense_hidden, act=F.relu, bias=False)
    gated = GatedFFN(d, gated_hidden, act=F.silu, bias=False)
    dense_params = sum(p.numel() for p in dense.parameters())
    gated_params = sum(p.numel() for p in gated.parameters())
    ffn_ok = dense_params == 73728 and gated_params == 73728

    n_q, n_kv = 8, 2
    gqa = MultiHeadAttention(d, n_q, n_kv, bias=False)
    gqa_params = sum(p.numel() for p in gqa.parameters())
    gqa_formula = int(2 * d ** 2 * (1 + n_kv / n_q))
    gqa_ok = gqa_params == gqa_formula == 23040

    mha = MultiHeadAttention(d, n_q, n_q, bias=False)
    mha_params = sum(p.numel() for p in mha.parameters())
    mha_ok = mha_params == 4 * d ** 2 == 36864

    ok = ffn_ok and gqa_ok and mha_ok
    print(f"[D06] FFN params: dense(4d)={dense_params}, gated(8d/3)={gated_params} (both expect 73728); "
          f"attn params: GQA={gqa_params} (formula 2d^2(1+n_kv/n_q)={gqa_formula}), MHA={mha_params} "
          f"(formula 4d^2={4 * d ** 2})  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d07_packed_projection():
    d, T = 96, 5
    x = torch.randn(2, T, d)

    n_q, n_kv, dh = 8, 2, 12
    q_lin = nn.Linear(d, n_q * dh, bias=False)
    k_lin = nn.Linear(d, n_kv * dh, bias=False)
    v_lin = nn.Linear(d, n_kv * dh, bias=False)
    packed_qkv = nn.Linear(d, n_q * dh + 2 * n_kv * dh, bias=False)
    with torch.no_grad():
        packed_qkv.weight.copy_(torch.cat([q_lin.weight, k_lin.weight, v_lin.weight], dim=0))
    sep_qkv = torch.cat([q_lin(x), k_lin(x), v_lin(x)], dim=-1)
    max_diff_qkv = (sep_qkv - packed_qkv(x)).abs().max().item()

    d_ff = 256
    gate_lin = nn.Linear(d, d_ff, bias=False)
    up_lin = nn.Linear(d, d_ff, bias=False)
    packed_gu = nn.Linear(d, 2 * d_ff, bias=False)
    with torch.no_grad():
        packed_gu.weight.copy_(torch.cat([gate_lin.weight, up_lin.weight], dim=0))
    sep_gu = torch.cat([gate_lin(x), up_lin(x)], dim=-1)
    max_diff_gu = (sep_gu - packed_gu(x)).abs().max().item()

    ok = max_diff_qkv < 1e-6 and max_diff_gu < 1e-6
    print(f"[D07] packed QKV vs separate: max|Δ|={max_diff_qkv:.2e}; packed gate+up vs separate: "
          f"max|Δ|={max_diff_gu:.2e}  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d08_parallel_vs_sequential():
    d, n_heads, d_ff, T = 96, 8, 256, 5
    x = torch.randn(2, T, d)
    causal = build_causal_mask(torch.arange(T), torch.arange(T))

    seq_blk = SequentialPreLNBlock(d, n_heads, d_ff)
    par_blk = ParallelPreLNBlock(d, n_heads, d_ff)

    # Bug-strength fix (see [D08] in the review ledger): the old test hooked
    # only the FFN's input and derived "what attn saw" by a SEPARATE, redundant
    # call to seq_blk.attn(...) outside the real forward pass -- for the
    # parallel block it never captured attn's input at all, so "both attn and
    # FFN read the SAME normalized stream" was never actually verified end to
    # end. Hook BOTH attn (input+output) and FFN (input) DURING the one real
    # forward call, for both blocks.
    captured_seq, captured_par = {}, {}

    def make_attn_io_hook(store):
        def hook(_module, inputs, output):
            store["attn_in"] = inputs[0].detach().clone()
            store["attn_out"] = output[0].detach().clone()   # (out, (k, v))
        return hook

    def make_ffn_in_hook(store):
        def hook(_module, inputs, _output):
            store["ffn_in"] = inputs[0].detach().clone()
        return hook

    h1 = seq_blk.attn.register_forward_hook(make_attn_io_hook(captured_seq))
    h2 = seq_blk.ffn.register_forward_hook(make_ffn_in_hook(captured_seq))
    seq_blk(x, causal)
    h1.remove(); h2.remove()

    # rtol=0 throughout: these are declared "exact reconstruction" claims.
    seq_attn_sees_norm1 = torch.allclose(captured_seq["attn_in"], seq_blk.norm1(x), atol=1e-6, rtol=0)
    expected_seq_ffn_in = seq_blk.norm2(x + captured_seq["attn_out"])   # FFN should see the UPDATED stream
    seq_sees_updated_stream = torch.allclose(captured_seq["ffn_in"], expected_seq_ffn_in, atol=1e-6, rtol=0)
    seq_not_original_norm = not torch.allclose(captured_seq["ffn_in"], seq_blk.norm2(x), atol=1e-3)

    h1 = par_blk.attn.register_forward_hook(make_attn_io_hook(captured_par))
    h2 = par_blk.ffn.register_forward_hook(make_ffn_in_hook(captured_par))
    par_blk(x, causal)
    h1.remove(); h2.remove()

    par_attn_sees_norm = torch.allclose(captured_par["attn_in"], par_blk.norm(x), atol=1e-6, rtol=0)
    par_sees_original_stream = torch.allclose(captured_par["ffn_in"], par_blk.norm(x), atol=1e-6, rtol=0)
    # The strongest form of "SAME stream": compare the two captured tensors to
    # EACH OTHER directly, not just each individually to norm(x).
    par_both_read_same_tensor = torch.allclose(captured_par["attn_in"], captured_par["ffn_in"],
                                                atol=1e-6, rtol=0)

    ok = (seq_attn_sees_norm1 and seq_sees_updated_stream and seq_not_original_norm
          and par_attn_sees_norm and par_sees_original_stream and par_both_read_same_tensor)
    print(f"[D08] sequential: attn input == norm1(x): {seq_attn_sees_norm1}, FFN input == "
          f"norm2(x+attn_out) (updated stream): {seq_sees_updated_stream} (and != norm2(x) alone: "
          f"{seq_not_original_norm}); parallel: attn input == norm(x): {par_attn_sees_norm}, FFN "
          f"input == norm(x): {par_sees_original_stream}, attn and FFN inputs are the literal SAME "
          f"tensor: {par_both_read_same_tensor}  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d09_backward_smoke():
    d, n_q, n_kv, T = 96, 8, 2, 6
    x = torch.randn(2, T, d)
    cos, sin = build_rope_cache(T, d // n_q)
    causal = build_causal_mask(torch.arange(T), torch.arange(T))

    dense_blk = ModernDecoderBlock(d, n_q, n_kv, d_ff=256)
    out = dense_blk(x, cos, sin, causal_mask=causal)
    out.pow(2).mean().backward()
    dense_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                        for p in dense_blk.parameters())
    # Bug-strength fix (see [D09] in the review ledger): "finite" alone passes
    # for an accidentally-detached branch whose grad is a finite ZERO tensor.
    # Require non-zero too.
    dense_nonzero = all(p.grad is not None and p.grad.abs().sum().item() > 0
                         for p in dense_blk.parameters())

    # MoE case: force n_tokens < n_experts so at least one expert is GUARANTEED
    # to receive zero tokens this batch, regardless of routing outcome (pigeonhole).
    n_experts, Tm = 4, 3
    xm = torch.randn(1, Tm, d)
    cos_m, sin_m = build_rope_cache(Tm, d // n_q)
    causal_m = build_causal_mask(torch.arange(Tm), torch.arange(Tm))
    moe_ffn = MoEFFN(d, d_ff=256, n_experts=n_experts, top_k=1)
    moe_blk = ModernDecoderBlock(d, n_q, n_kv, ffn=moe_ffn)
    out_m = moe_blk(xm, cos_m, sin_m, causal_mask=causal_m)
    out_m.pow(2).mean().backward()

    tokens_per_expert = moe_ffn.tokens_per_expert
    unused_exists = any(c == 0 for c in tokens_per_expert)
    moe_experts_ok = True
    for e_idx, expert in enumerate(moe_ffn.experts):
        hit = tokens_per_expert[e_idx] > 0
        for p in expert.parameters():
            if hit:
                # touched experts must get a NON-ZERO gradient, not merely finite
                grad_ok = (p.grad is not None and torch.isfinite(p.grad).all().item()
                           and p.grad.abs().sum().item() > 0)
                moe_experts_ok = moe_experts_ok and grad_ok
            else:
                moe_experts_ok = moe_experts_ok and (
                    p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)))

    # Bug-strength fix (see [D09] in the review ledger): the old check only
    # inspected expert parameters. A detached/STE-broken router, or a bug that
    # accidentally cuts the FFN branch out of the residual stream, would leave
    # the router and/or the rest of the block gradient-free while this
    # experts-only check still passes. Check the router AND the rest of the
    # block (attention + both norms) too.
    router_ok = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                     and p.grad.abs().sum().item() > 0
                     for p in moe_ffn.gate.parameters())
    other_params = (list(moe_blk.attn.parameters()) + list(moe_blk.norm1.parameters())
                     + list(moe_blk.norm2.parameters()))
    rest_of_block_ok = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            and p.grad.abs().sum().item() > 0
                            for p in other_params)

    moe_ok = moe_experts_ok and router_ok and rest_of_block_ok
    ok = dense_finite and dense_nonzero and unused_exists and moe_ok
    print(f"[D09] dense block: all grads finite = {dense_finite}, all non-zero = {dense_nonzero}; "
          f"MoE tokens_per_expert={tokens_per_expert} (<{n_experts} tokens, so >=1 expert unused by "
          f"pigeonhole); touched experts non-zero+finite grad, untouched experts no/zero grad: "
          f"{moe_experts_ok}; router gate non-zero+finite grad: {router_ok}; rest-of-block "
          f"(attn+norms) non-zero+finite grad: {rest_of_block_ok}  {'PASS' if ok else 'FAIL'}")
    assert ok


# =============================================================================
# §5 Additional regression tests [D10]-[D15], one per B1-class bug found in
# code review (each targets a real silent-failure mode, not just formula
# correctness -- see the review ledger).
# =============================================================================

def check_d10_config_validation():
    """[D10] targets bug #2 (indivisible d_model/n_q_heads silently narrows
    attention) and the MoEFFN top_k silent-downgrade bug."""
    raised_divisibility = False
    try:
        MultiHeadAttention(100, 8)     # 100 % 8 != 0
    except ValueError:
        raised_divisibility = True

    raised_group = False
    try:
        MultiHeadAttention(96, 8, n_kv_heads=3)   # 8 % 3 != 0
    except ValueError:
        raised_group = True

    raised_odd_head = False
    try:
        MultiHeadAttention(48, 16, rope=True)     # d_head = 48//16 = 3 (odd)
    except ValueError:
        raised_odd_head = True

    raised_topk = False
    try:
        MoEFFN(96, d_ff=256, n_experts=4, top_k=2)
    except NotImplementedError:
        raised_topk = True

    ok = raised_divisibility and raised_group and raised_odd_head and raised_topk
    print(f"[D10] d_model % n_q_heads != 0 raises ValueError: {raised_divisibility}; "
          f"n_q_heads % n_kv_heads != 0 raises ValueError: {raised_group}; "
          f"RoPE + odd head_dim raises ValueError: {raised_odd_head}; "
          f"MoEFFN(top_k=2) raises NotImplementedError (no silent top-1 downgrade): {raised_topk}  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok


def check_d11_kv_cache_consistency():
    """[D11] targets bug #4 (KV cache + start_pos has no consistency
    contract, and the cache never validates shape/device/dtype identity)."""
    d, n_heads, T = 32, 4, 3
    x = torch.randn(1, T, d)
    cos, sin = build_rope_cache(10, d // n_heads)
    mha = MultiHeadAttention(d, n_heads, n_heads, bias=False, rope=True)

    cache = KVCache()
    causal1 = build_causal_mask(torch.arange(T), torch.arange(T))
    mha(x, causal_mask=causal1, cos=cos, sin=sin, kv_cache=cache)   # cache.next_pos becomes T

    # A stale/wrong explicit start_pos against a non-empty cache must raise --
    # NOT silently reuse the wrong RoPE angles for the new chunk.
    x2 = torch.randn(1, 1, d)
    k_pos = torch.arange(T + 1)
    mask_wrong = build_causal_mask(torch.tensor([0]), k_pos)   # deliberately wrong query position
    raised_inconsistent = False
    try:
        mha(x2, causal_mask=mask_wrong, cos=cos, sin=sin, kv_cache=cache, start_pos=0)
    except ValueError:
        raised_inconsistent = True

    # The CORRECT explicit start_pos (== cache.next_pos) must still work fine.
    mask_ok = build_causal_mask(torch.tensor([T]), k_pos)
    no_error_when_consistent = True
    try:
        mha(x2, causal_mask=mask_ok, cos=cos, sin=sin, kv_cache=cache, start_pos=T)
    except ValueError:
        no_error_when_consistent = False

    cache.reset()
    reset_ok = cache.k is None and cache.next_pos == 0

    cache2 = KVCache()
    dh = d // n_heads
    cache2.append(torch.randn(1, n_heads, 2, dh), torch.randn(1, n_heads, 2, dh))
    raised_shape_mismatch = False
    try:
        k_wrong_batch = torch.randn(2, n_heads, 1, dh)   # different batch size than what's cached
        cache2.append(k_wrong_batch, k_wrong_batch)
    except ValueError:
        raised_shape_mismatch = True

    ok = raised_inconsistent and no_error_when_consistent and reset_ok and raised_shape_mismatch
    print(f"[D11] inconsistent start_pos vs non-empty cache raises ValueError: {raised_inconsistent}; "
          f"consistent start_pos does not raise: {no_error_when_consistent}; cache.reset() clears "
          f"state: {reset_ok}; append() with mismatched batch/shape raises ValueError: "
          f"{raised_shape_mismatch}  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d12_cross_attention_guards():
    """[D12] targets bug #5 (self/cross + RoPE + cache "any combination" isn't
    actually a coherent interface -- cross-attention + RoPE/cache must be
    refused, plain cross-attention must keep working)."""
    d, n_heads, T, S = 32, 4, 4, 3
    x_q = torch.randn(1, T, d)
    memory = torch.randn(1, S, d)
    cos, sin = build_rope_cache(10, d // n_heads)

    cross_rope = MultiHeadAttention(d, n_heads, n_heads, bias=False, rope=True)
    raised_rope_cross = False
    try:
        cross_rope(x_q, x_kv=memory, cos=cos, sin=sin)
    except ValueError:
        raised_rope_cross = True

    cross_plain = MultiHeadAttention(d, n_heads, n_heads, bias=False, rope=False)
    cache = KVCache()
    raised_cache_cross = False
    try:
        cross_plain(x_q, x_kv=memory, kv_cache=cache)
    except ValueError:
        raised_cache_cross = True

    # Plain cross-attention (no RoPE, no cache) must still work -- this is
    # exactly what VanillaSeq2SeqDecoderLayer2017 relies on.
    out, _ = cross_plain(x_q, x_kv=memory)
    plain_cross_ok = out.shape == (1, T, d)

    ok = raised_rope_cross and raised_cache_cross and plain_cross_ok
    print(f"[D12] cross-attention + RoPE raises ValueError: {raised_rope_cross}; cross-attention + "
          f"KV cache raises ValueError: {raised_cache_cross}; plain cross-attention (no RoPE/cache) "
          f"still works, shape={tuple(out.shape)}: {plain_cross_ok}  {'PASS' if ok else 'FAIL'}")
    assert ok


def check_d13_masking_safety_and_broadcast():
    """[D13] targets bug #6 (fully-masked row -> NaN) and bug #3 (mask
    broadcasting collides when B == Hkv or B == Tq)."""
    d, n_heads, T = 32, 4, 5

    x1 = torch.randn(1, T, d)
    mha_nan = MultiHeadAttention(d, n_heads, n_heads, bias=False)
    bad_causal = build_causal_mask(torch.arange(T), torch.arange(T)).clone()
    bad_causal[0, :] = False   # query 0 has NO valid key at all (not even itself)
    out_nan, _ = mha_nan(x1, causal_mask=bad_causal)
    no_nan = torch.isfinite(out_nan).all().item()

    # B == n_kv_heads: a shape-guessed [B, Tk] mask could be silently
    # misread as a [1, B, 1, Tq, Tk] KV-head-dim mask instead of the intended
    # per-BATCH mask. Verify per-sample isolation against an independent
    # per-sample reference computed one batch element at a time.
    B = n_kv_heads_eq_B = 2
    mha_b = MultiHeadAttention(d, n_heads, n_kv_heads_eq_B, bias=False)
    x2 = torch.randn(B, T, d)
    causal = build_causal_mask(torch.arange(T), torch.arange(T))
    key_padding_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])
    out_batched, _ = mha_b(x2, causal_mask=causal, key_padding_mask=key_padding_mask)
    per_sample_outs = [mha_b(x2[b:b + 1], causal_mask=causal,
                              key_padding_mask=key_padding_mask[b:b + 1])[0] for b in range(B)]
    out_per_sample = torch.cat(per_sample_outs, dim=0)
    batch_hkv_isolation_ok = torch.allclose(out_batched, out_per_sample, atol=1e-5)

    # B == Tq: another special shape where a shape-guessed mask could collide.
    Tq_eq_B = 3
    mha_t = MultiHeadAttention(d, n_heads, n_heads, bias=False)
    x3 = torch.randn(Tq_eq_B, Tq_eq_B, d)
    causal3 = build_causal_mask(torch.arange(Tq_eq_B), torch.arange(Tq_eq_B))
    kpm3 = torch.tensor([[True, True, False], [True, True, True], [False, True, True]])
    out_batched3, _ = mha_t(x3, causal_mask=causal3, key_padding_mask=kpm3)
    per_sample_outs3 = [mha_t(x3[b:b + 1], causal_mask=causal3,
                               key_padding_mask=kpm3[b:b + 1])[0] for b in range(Tq_eq_B)]
    out_per_sample3 = torch.cat(per_sample_outs3, dim=0)
    batch_tq_isolation_ok = torch.allclose(out_batched3, out_per_sample3, atol=1e-5)

    ok = no_nan and batch_hkv_isolation_ok and batch_tq_isolation_ok
    print(f"[D13] fully-masked query row produces finite output (no NaN): {no_nan}; "
          f"key_padding_mask correctly isolated per-batch-sample when B==n_kv_heads ({B}): "
          f"max|Δ| vs per-sample reference={(out_batched - out_per_sample).abs().max():.2e} -> "
          f"{batch_hkv_isolation_ok}; same check when B==Tq ({Tq_eq_B}): "
          f"max|Δ|={(out_batched3 - out_per_sample3).abs().max():.2e} -> {batch_tq_isolation_ok}  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok


def check_d14_default_causal_behavior():
    """[D14] targets bug #1 (decoder blocks defaulted to mask=None, silently
    running bidirectional attention instead of causal)."""
    d, n_heads, d_ff, T = 32, 4, 128, 5
    x = torch.randn(2, T, d)

    gpt2 = GPT2StyleDecoderLayer(d, n_heads, d_ff)
    gpt2.eval()
    causal = build_causal_mask(torch.arange(T), torch.arange(T))
    with torch.no_grad():
        out_explicit = gpt2(x, causal_mask=causal)
        out_omitted = gpt2(x)                              # no mask argument at all
    gpt2_matches = torch.allclose(out_explicit, out_omitted, atol=1e-6, rtol=0)

    t = 2
    x_future_perturbed = x.clone()
    x_future_perturbed[:, t + 1:, :] = torch.randn_like(x_future_perturbed[:, t + 1:, :])
    with torch.no_grad():
        o1 = gpt2(x)
        o2 = gpt2(x_future_perturbed)
    gpt2_default_is_causal = torch.allclose(o1[:, :t + 1], o2[:, :t + 1], atol=1e-6, rtol=0)

    n_q, n_kv, d_modern = 8, 2, 32
    modern = ModernDecoderBlock(d_modern, n_q, n_kv, d_ff=64)
    modern.eval()
    cos, sin = build_rope_cache(T, d_modern // n_q)
    x_m = torch.randn(2, T, d_modern)
    with torch.no_grad():
        out_m_explicit = modern(x_m, cos, sin, causal_mask=causal)
        out_m_omitted = modern(x_m, cos, sin)               # no mask argument at all
    modern_matches = torch.allclose(out_m_explicit, out_m_omitted, atol=1e-6, rtol=0)

    ok = gpt2_matches and gpt2_default_is_causal and modern_matches
    print(f"[D14] GPT2StyleDecoderLayer: omitted causal_mask == explicit causal_mask: {gpt2_matches}, "
          f"default is genuinely causal (no future->past leakage): {gpt2_default_is_causal}; "
          f"ModernDecoderBlock: omitted causal_mask == explicit causal_mask: {modern_matches}  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok


def check_d15_rope_device_dtype():
    """[D15] targets bug #7 (RoPE cache hard-fixed to CPU/float32 with no
    device/dtype parameters)."""
    d, n_heads, T = 32, 4, 4
    head_dim = d // n_heads

    cos64, sin64 = build_rope_cache(T, head_dim, dtype=torch.float64)
    dtype_honored = cos64.dtype == torch.float64 and sin64.dtype == torch.float64

    cos32, sin32 = build_rope_cache(T, head_dim)              # default float32 cache
    x64 = torch.randn(1, n_heads, T, head_dim, dtype=torch.float64)
    out64 = apply_rope(x64, cos32, sin32)                     # deliberately mismatched dtype
    cast_ok = out64.dtype == torch.float64

    out64_native = apply_rope(x64, cos64, sin64)
    matches_native_cache = torch.allclose(out64, out64_native, atol=1e-6)

    ok = dtype_honored and cast_ok and matches_native_cache
    print(f"[D15] build_rope_cache(dtype=float64) produces float64 cos/sin: {dtype_honored}; "
          f"apply_rope auto-casts a mismatched (float32) cache to a float64 input's dtype "
          f"(no crash, correct output dtype): {cast_ok}; matches a natively-float64-built cache: "
          f"{matches_native_cache} (max|Δ|={(out64 - out64_native).abs().max():.2e})  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok


def main():
    if not __debug__:
        print("WARNING: running under `python -O` -- Python's `assert` statement is a no-op in "
              "this mode, which silently disables EVERY correctness check in this script "
              "(it would print 'all ... passed' unconditionally regardless of actual results). "
              "Re-run without -O for the checks below to mean anything.", file=sys.stderr)

    check_d01_shape_contract()
    check_d02_residual_topology()
    check_d03_causal_non_leakage()
    check_d04_full_vs_cached()
    check_d05_mha_gqa_mqa()
    check_d06_param_formulas()
    check_d07_packed_projection()
    check_d08_parallel_vs_sequential()
    check_d09_backward_smoke()
    check_d10_config_validation()
    check_d11_kv_cache_consistency()
    check_d12_cross_attention_guards()
    check_d13_masking_safety_and_broadcast()
    check_d14_default_causal_behavior()
    check_d15_rope_device_dtype()
    print("\nall transformer block sanity checks passed ✓")


if __name__ == "__main__":
    main()
