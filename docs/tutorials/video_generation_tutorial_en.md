## §0 TL;DR Cheat Sheet

> 💡 **Video Generation in 7 sentences** — the 2024-2025 video generation explosion. One page covering the most frequently-asked interview points (see §1–§11 below for derivations).

1. **Paradigm**: mainstream video generation = **3D Causal VAE compression + Latent DiT diffusion / Flow Matching**. Sora (2024-02) was the first to push the Transformer to 60s + high resolution; Hunyuan-Video (Tencent 2024-12) / Wan 2.x (Alibaba 2024-2025) / Mochi-1 / CogVideoX / Movie Gen / Kling / Veo 2 are all variants of this architectural family.

2. **3D Causal VAE**: compresses an $H \times W \times T$ video into an $h \times w \times t$ latent (typical spatial downsampling $8\times$, temporal downsampling $4\times$). The key is **causal**: the current frame latent **cannot see future frames** — this lets the trained VAE simultaneously support images ($t{=}1$) and videos ($t{>}1$), and enables streaming / autoregressive generation of subsequent frames.

3. **Spacetime Patches (Sora)**: slice the video latent into $p_t \times p_h \times p_w$ 3D patches as tokens; like ViT but with the added time dimension. Supports **variable resolution / variable duration / variable aspect ratio**: directly pack patch sequences of different shapes into a single batch (using masks to distinguish them), with no need to resize to a fixed shape.

4. **Three spatiotemporal attention variants**: (a) **Factorized 2+1D** (Latte / OpenSora / AnimateDiff): spatial-only first, then temporal-only, complexity $O(T \cdot S^2 + S \cdot T^2)$; (b) **Full 3D** (Sora / Hunyuan-Video / Mochi): all tokens attend to each other, complexity $O((ST)^2)$, most expensive but best quality; (c) **Window / sparse ST** (Wan 2.x / portions of CogVideoX): sliding-window 3D, a middle ground.

5. **MM-DiT for Video (Hunyuan-Video / Mochi)**: text tokens and video tokens share the **same sequence** in self-attention (no longer cross-attn), with each stream having its own QKV projection + AdaLN modulation; conditioning information interacts directly at the token level — Hunyuan / Mochi / Wan extend the SD3 image MM-DiT idea to video.

6. **Image-to-Video (I2V)**: three mainstream techniques — (i) **First-frame concat**: encode the ref image and concat along the channel dim of the latent; (ii) **Cross-attention injection**: ref image tokens serve as K/V; (iii) **AnimateDiff** style: freeze the T2I backbone and insert only a temporal module. SVD / DynamiCrafter / I2VGen-XL / Wan-I2V are representative.

7. **Long video** = keyframe + interpolation / hierarchical / autoregressive chunks. **Evaluation**: VBench (Huang CVPR 2024) provides 16-dimensional fine-grained scoring and is the de facto standard; FVD (Unterthiner 2018) remains as an auxiliary metric; CLIPSim-V evaluates text-video alignment.

> ⚠️ **Caveat** — the model-specific numbers in this article (parameter counts, compression ratios, attention types) are based on each model's public paper / tech report; refer to the originals for exact training hyperparameters and final architectures.

## §1 Intuition and Landscape

### 1.1 Why video is harder than images

For an $H{\times}W{\times}T$ video, the pixel count grows linearly with $T$, and the token count (after patchification) also grows linearly. **Attention complexity is quadratic in token count** — meaning full 3D attention in raw pixel space already explodes for $T \ge 16$. So the core engineering problem of video generation is:

- **Compression**: 3D VAE compresses token count in both spatial and temporal dimensions (typically $8 \times 8 \times 4$)
- **Architecture**: run attention on the latent (spatiotemporal patterns are the design point)
- **Generative paradigm**: DDPM / Rectified Flow / FM (v-prediction is mainstream; SD3-style)

### 1.2 2024-2025 timeline (by release order)

| Time | Model | Source | Key contribution |
| --- | --- | --- | --- |
| 2023-07 → ICLR'24 | **AnimateDiff** | Guo et al. | T2I backbone + plug-in motion module; the open-source ancestor of I2V/T2V |
| 2023-11 | **SVD** | Stability | I2V open-source baseline; SD2.1 + temporal layers |
| 2024-01 | **Latte** | Ma et al. | Early DiT-Video; factorized spatial+temporal |
| 2024-02 | **Sora** | OpenAI | DiT + spacetime patches + large-scale captioning; closed source |
| 2024-05 | **Veo** | Google DeepMind | Closed-source 1080p / 1 minute |
| 2024-06 | **Kling** | Kuaishou | Closed-source from China, up to 2 minutes |
| 2024-08 | **CogVideoX** (arXiv) | Zhipu/THU | Open-source 2B/5B; Expert Transformer + 3D VAE |
| 2024-10 | **Movie Gen** | Meta | 30B; joint video + audio; DiT + FM |
| 2024-10 | **Mochi-1** | Genmo | 10B open source; AsymmDiT asymmetric MM-DiT |
| 2024-11 | **LTX-Video** | Lightricks | Real-time (2B); strong-compression VAE + DiT |
| 2024-12 | **Hunyuan-Video** | Tencent | 13B open-source SoTA; 3D Causal VAE + MM-DiT + prompt rewriter |
| 2024-12 | **Veo 2** | Google | 4K / 2 minutes |
| 2025-02 | **Wan 2.1** | Alibaba (Team Wan) | 14B open source; dual T2V + I2V |
| 2025-07 | **Wan 2.2** | Alibaba | Upgraded; MoE experts + longer temporal range |
| 2024-2025 | **OpenSora / OpenSora-Plan** | HPC-AI / PKU | Fully open-source training stack replicating the Sora pipeline |

> ⚠️ **Closed vs open source** — Sora / Veo / Kling / Movie Gen do not release weights; all architectural details are based on official technical reports. When interviewing, do not take internal ablations as hard numbers; explicitly say "according to their report".

### 1.3 Overall pipeline (the common skeleton)

```
Text  --T5/CLIP/MLLM-->  text tokens [B, L_t, D]
Video --3D Causal VAE-->  latent z₁ [B, C, t, h, w] --patchify (p_t×p_h×p_w)--> tokens [B, N_v, D]
                                       │
                              Latent DiT / MM-DiT
                (full 3D / factorized 2+1D / window ST attention)
                              │  Train: regress v_θ(τ, z_τ, text) -> z₁ - z₀ (RF)
                              ↓
                       sampled z₁ --3D Causal VAE Decoder--> Generated video [B, 3, T, H, W]
```

## §2 3D Causal VAE — the foundation of video compression

### 2.1 Formalization

VAE encoder $E$ and decoder $D$:

$$E: \mathbb{R}^{3 \times T \times H \times W} \to \mathbb{R}^{C \times t \times h \times w}, \quad D: \mathbb{R}^{C \times t \times h \times w} \to \mathbb{R}^{3 \times T \times H \times W}$$

Downsampling ratios $T/t \in \{4, 8\}$ (temporal), $H/h, W/w \in \{8, 16\}$ (spatial). Hunyuan-Video / CogVideoX / Wan use $4{\times}8{\times}8$; LTX-Video pushes to an extreme $8{\times}32{\times}32 = 8192\times$ token compression.

Latent dim $C$ is typically $16$ (Hunyuan) or $4$ (OpenSora); larger retains more information, but the latent prior becomes farther from $\mathcal{N}(0,I)$, slowing diffusion convergence.

### 2.2 Why must it be "Causal" — three key benefits

> ✅ **Three advantages of 3D Causal VAE**

- **Image / video unified**: when $T=1$ (single frame), causal 3D conv degenerates to 2D conv (the kernel only looks at history in the time dim, but history is empty → equivalent to no time dim). The encoder can compress both video and images, **and images and video can share the same latent space** — this is the prerequisite for Hunyuan-Video / Wan to simultaneously support I2V/T2V.

- **Streaming / autoregressive inference**: causality guarantees the current frame latent only depends on past frames, so **long videos can be processed in chunks**, without having to decode all $T$ frames at once; same idea as KV cache in LLMs.

- **Training data utilization**: when mixing image + video training, images can be treated as $T{=}1$ videos. A standard 3D VAE has its kernel center looking into the future, so it cannot do this.

### 2.3 Causal Conv3d implementation

A standard 3D conv pads $\lfloor k_t/2 \rfloor$ zeros on each side of the time dimension (symmetric), so output[t] depends on input[t-1], input[t], input[t+1] — **leaking the future**.

**Causal 3D conv** = stack all the padding on the left (past direction) of the time dim, with no padding on the right (future):

$$\text{output}[t] = \sum_{\tau=0}^{k_t - 1} W[\tau] \cdot \text{input}[t - (k_t - 1) + \tau]$$

So the output at time $t$ only sees the historical window $[t - k_t + 1, t]$. Spatial dims still use symmetric padding (image problem, no directionality).

> ⚠️ **Making downsampling causal** — for downsampling layers with time stride > 1 (e.g. $T \to T/2$), window alignment must be preserved. The common practice: have stride see only the historical window `[t-1, t]` in time, outputting `t' = t//2`. Hunyuan-Video / Mochi both draw this layer in their papers.

### 2.4 Loss function (same backbone as image VAE)

$$\mathcal{L}_\text{VAE} = \mathcal{L}_\text{recon} + \lambda_\text{KL} \cdot \mathcal{L}_\text{KL} + \lambda_\text{LPIPS} \cdot \mathcal{L}_\text{LPIPS} + \lambda_\text{GAN} \cdot \mathcal{L}_\text{GAN}$$

Video VAE additionally needs a **temporal consistency loss** (e.g. adjacent-frame reconstruction difference + optical flow constraint) to prevent flickering. Hunyuan-Video uses GAN (PatchGAN over spatiotemporal patches) + 3D perceptual loss.

### 2.5 Code: Causal Conv3d and encoder skeleton

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    """
    Causal conv3d along the temporal dimension.
    Input [B, C_in, T, H, W] -> Output [B, C_out, T, H, W] (shape preserved when stride=1).
    Temporal kernel size k_t -> left pad (k_t-1), no right padding; spatial dims use symmetric padding.
    """
    def __init__(self, in_ch, out_ch, kernel=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1)):
        super().__init__()
        k_t, k_h, k_w = kernel
        d_t, d_h, d_w = dilation
        # Causal padding in time dim (all stacked on the left)
        self.t_pad_left = (k_t - 1) * d_t
        # Symmetric padding in spatial dims
        self.h_pad = ((k_h - 1) * d_h) // 2
        self.w_pad = ((k_w - 1) * d_w) // 2
        # nn.Conv3d's built-in padding is symmetric; pad manually + disable conv's own padding
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, dilation=dilation, padding=0)

    def forward(self, x):
        # x: [B, C, T, H, W]
        # F.pad order: (W_left, W_right, H_left, H_right, T_left, T_right)
        x = F.pad(x, (self.w_pad, self.w_pad,
                      self.h_pad, self.h_pad,
                      self.t_pad_left, 0))   # No right padding in time => no future
        return self.conv(x)


class CausalResBlock3D(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = CausalConv3d(ch, ch, kernel=(3, 3, 3))
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = CausalConv3d(ch, ch, kernel=(3, 3, 3))

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class CausalVAEEncoder(nn.Module):
    """
    Pedagogical 3D Causal VAE encoder. Real Hunyuan / CogVideoX use deeper stacks + GAN discriminators.
    Downsampling uses CausalConv3d with stride=(2,2,2) (still causal in time).
    """
    def __init__(self, in_ch=3, base_ch=64, z_ch=16, num_levels=3, blocks_per_level=2):
        super().__init__()
        self.in_proj = CausalConv3d(in_ch, base_ch, kernel=(3, 3, 3))
        levels, ch = [], base_ch
        for _ in range(num_levels):
            blocks = [CausalResBlock3D(ch) for _ in range(blocks_per_level)]
            blocks.append(CausalConv3d(ch, ch * 2, kernel=(3, 3, 3), stride=(2, 2, 2)))
            levels.append(nn.Sequential(*blocks))
            ch *= 2
        self.levels = nn.ModuleList(levels)
        self.norm_out = nn.GroupNorm(8, ch)
        self.head = CausalConv3d(ch, 2 * z_ch, kernel=(1, 1, 1))     # mu + logvar

    def forward(self, x):
        # Expects T, H, W to be divisible by 2^num_levels
        h = self.in_proj(x)
        for lvl in self.levels:
            h = lvl(h)
        h = F.silu(self.norm_out(h))
        return self.head(h).chunk(2, dim=1)                          # (mu, logvar)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
```

> 💡 **First-frame / chunk boundary handling (interview plus)** — causal padding makes the first frame "see 0" along the time dim. During VAE training, either randomly drop the loss contribution from the first $k_t{-}1$ frames, or use anchor padding (replicate the first frame as padding). Hunyuan-Video discusses boundary frame consistency for chunk-by-chunk encoding in their paper; in engineering this is often handled by chunked overlapping.

## §3 Spacetime Patches (the Sora perspective)

### 3.1 One-liner

Just as ViT slices images into $p \times p$ patches as tokens, Sora slices the **video latent** into $p_t \times p_h \times p_w$ **3D patches** as tokens. $N = (t/p_t) \cdot (h/p_h) \cdot (w/p_w)$ tokens are fed to the Transformer.

Formally, rearrange a latent $z \in \mathbb{R}^{C \times t \times h \times w}$:

$$z \mapsto Z \in \mathbb{R}^{N \times (C \cdot p_t \cdot p_h \cdot p_w)} \xrightarrow{W_\text{proj}} Z' \in \mathbb{R}^{N \times D}$$

### 3.2 Why patchify (instead of one token per voxel)

- **Token count reduction**: $N$ is reduced by a factor of $p_t \cdot p_h \cdot p_w$, hugely cutting the quadratic attention cost.
- **Local inductive bias**: within each patch, a fixed MLP projection acts like a conv with local fusion.
- **Variable shape support**: different resolutions / durations directly produce different $N$, and the Transformer is insensitive to $N$.

### 3.3 Variable resolution / duration / aspect ratio (Sora's key contribution)

Sora does not resize every video clip to a fixed shape (e.g. $256 \times 256 \times 16$). It slices every $(T, H, W)$ video into a patch sequence and uses **packing** (different samples mixed into the same batch, separated by segment masks) for training.

Key implementation points:

- **Positional encoding uses relative / decoupled** schemes: RoPE-2D for space, RoPE-1D for time; not bound to a maximum length.
- **Attention mask**: tokens from different samples in the same batch are masked out from each other.
- **Target aspect ratio token**: an additional conditioning token telling the model whether to generate 16:9 or 9:16.

This is what lets Sora train / generate videos at arbitrary aspect ratios; Wan-2.x and Hunyuan-Video later borrowed this.

### 3.4 Code: Spacetime patchify / unpatchify

```python
class SpacetimePatchify(nn.Module):
    """
    Slice a video latent [B, C, t, h, w] into 3D patch tokens [B, N, D].
    p_t / p_h / p_w must divide t / h / w (ensure this on the input side).
    """
    def __init__(self, in_ch, patch=(2, 2, 2), embed_dim=1024):
        super().__init__()
        self.p_t, self.p_h, self.p_w = patch
        self.proj = nn.Conv3d(
            in_ch, embed_dim,
            kernel_size=(self.p_t, self.p_h, self.p_w),
            stride=(self.p_t, self.p_h, self.p_w),
        )

    def forward(self, z):
        # z: [B, C, t, h, w]
        x = self.proj(z)                        # [B, D, t/p_t, h/p_h, w/p_w]
        B, D, tt, hh, ww = x.shape
        N = tt * hh * ww
        x = x.flatten(2).transpose(1, 2)        # [B, N, D]
        return x, (tt, hh, ww)


class SpacetimeUnpatchify(nn.Module):
    """ Inverse of Patchify: [B, N, D] + shape -> [B, C_out, t, h, w] """
    def __init__(self, embed_dim, out_ch, patch=(2, 2, 2)):
        super().__init__()
        self.p_t, self.p_h, self.p_w = patch
        self.out_ch = out_ch
        # Each token decodes back into an (out_ch * p_t * p_h * p_w) voxel block
        self.proj = nn.Linear(embed_dim, out_ch * self.p_t * self.p_h * self.p_w)

    def forward(self, x, grid):
        # x: [B, N, D], grid = (tt, hh, ww)
        B, N, D = x.shape
        tt, hh, ww = grid
        assert N == tt * hh * ww
        y = self.proj(x)                                 # [B, N, out_ch * p_t * p_h * p_w]
        y = y.view(B, tt, hh, ww, self.out_ch, self.p_t, self.p_h, self.p_w)
        # Splice intra-patch coordinates back into original space: (tt, p_t) -> T, (hh, p_h) -> H, ...
        y = y.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return y.view(B, self.out_ch, tt * self.p_t, hh * self.p_h, ww * self.p_w)
```

> ⚠️ **How to choose patch size** — $p_t$ is typically 1-2 (time was already compressed 4x by the VAE, no room to compress further), $p_h = p_w$ typically 2. Open-source models often use $p_t{=}1, p_h{=}p_w{=}2$ (Mochi-1 / Hunyuan-Video, paired with a 4×8×8 latent VAE). Sora has not disclosed implementation details, only describing the "spacetime patches" concept.

## §4 Three variants of Spatiotemporal Attention

### 4.1 Complexity comparison (a core interview question)

Define $S = h \cdot w / (p_h p_w)$ (tokens per frame), $T_t = t / p_t$ (temporal tokens), $N = S \cdot T_t$.

| Type | Formula | Time | Score memory (vanilla) | Where used |
| --- | --- | --- | --- | --- |
| **Factorized 2+1D** | $\text{TempAttn}(\text{SpatialAttn}(x))$ | $O(T_t S^2 + S T_t^2) \cdot d$ | $O(T_t S^2 + S T_t^2)$ | Latte, OpenSora, AnimateDiff |
| **Full 3D ST** | $\text{Attn}_{\text{ST}}(x \in \mathbb{R}^{N \times D})$ | $O(N^2 d) = O(S^2 T_t^2 d)$ | $O(N^2)$ | Hunyuan-Video, Mochi, (Sora speculated) |
| **2+1D + occasional full** | Mostly 2+1D, a few full blocks | Between the two | Between the two | OpenSora-Plan, CogVideoX |
| **Window 3D / sparse** | Local 3D window | $O(N \cdot w^3 d)$ ($w$ = window) | $O(N \cdot w^3)$ | Some engineering practice |

> 💡 **Core comparison** — Full 3D has all spacetime tokens fully interacting and is **most expressive**; Factorized does first spatial then temporal attn twice, **strongly assuming space ⊗ time is separable**, but at $\min(S, T_t)$ times lower cost. Sora has not officially disclosed its attention type; community speculation based on its "spacetime patches" description generally guesses full 3D / a single-sequence unified attention (Hunyuan-Video / Mochi already adopt this design).

### 4.2 Factorized 2+1D Attention (Latte / OpenSora)

```

Input [B, T_t, S, D]
   │
   │  rearrange -> [B*T_t, S, D]
   ↓
SpatialAttn (intra-frame self-attention)  →    [B*T_t, S, D]
   │
   │  rearrange -> [B*S, T_t, D]
   ↓
TempAttn (cross-frame self-attention)      →    [B*S, T_t, D]
   │
   │  rearrange -> [B, T_t, S, D]
   ↓
FFN
```

Strong assumption: per-frame spatial relations and cross-frame temporal relations are **separable**. This is a simplification but very engineering-friendly — most of the time, adjacent frames in a video are largely similar (temporal "optical flow" information can be captured by 1D attn), and intra-frame patch relations resemble those in images.

### 4.3 Full 3D Spatiotemporal Attention (Sora / Hunyuan / Mochi)

Treat $N = S \cdot T_t$ tokens as a single 1D sequence and apply standard self-attention:

$$\text{Attn}(Z) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V, \quad Z \in \mathbb{R}^{N \times D}$$

Complexity $O(N^2 d)$. $N$ is large for long videos: e.g. 720p × 5s × 24fps after a $4{\times}8{\times}8$ VAE yields a latent shape of about $30 \times 90 \times 160$, then (1,2,2) patchify gives $N = 30 \cdot 45 \cdot 80 = 108000$ tokens, $N^2 \approx 1.17 \times 10^{10}$. **So in production, FlashAttention v2/v3 + Tensor Parallelism + Sequence Parallelism are mandatory**.

### 4.4 RoPE-3D (positional encoding)

Video tokens need to encode (time, height, width) all at once. **Approach 1 (decoupled)**: split the $d$-dim vector into three chunks, applying RoPE to (t, h, w) separately. **Approach 2 (shared)**: each chunk stacks all three frequency rotations (Hunyuan-Video).

CogVideoX uses **3D RoPE**: split the $d$-dim query/key into three chunks $q = [q^{(t)} \,|\, q^{(h)} \,|\, q^{(w)}]$ (disjoint subspaces), and apply 1D RoPE to each chunk with the corresponding (t, h, w) frequency, then concat:

$$\text{RoPE}_{3D}(q,\, t, h, w) = \big[\, R_t(\theta_t) q^{(t)} \,\big|\, R_h(\theta_h) q^{(h)} \,\big|\, R_w(\theta_w) q^{(w)} \,\big]$$

In practice this is three independent 1D RoPE rotations concatenated (block-diagonal rotation), naturally yielding relative spatiotemporal position encoding.

### 4.5 Code: Factorized 2+1D implementation

```python
class AxisAttention(nn.Module):
    """
    Self-attention along a specified axis (spatial 'S' or temporal 'T').
    Input [B, T, S, D] -> Output [B, T, S, D].
    """
    def __init__(self, dim, heads, head_dim, axis="S", causal=False):
        super().__init__()
        assert axis in ("S", "T")
        self.heads, self.head_dim, self.axis, self.causal = heads, head_dim, axis, causal
        self.qkv = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, dim, bias=False)

    def forward(self, x):
        B, T, S, D = x.shape
        if self.axis == "S":
            x_ = x.reshape(B * T, S, D); L_seq = S; merge_batch = T
        else:   # "T": collapse spatial dim into batch, time dim becomes seq
            x_ = x.permute(0, 2, 1, 3).reshape(B * S, T, D); L_seq = T; merge_batch = S
        qkv = self.qkv(x_).reshape(-1, L_seq, 3, self.heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # [B', H, L, d_k]
        # PyTorch SDPA automatically picks FlashAttention v2/v3
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(-1, L_seq, self.heads * self.head_dim)
        out = self.out(out)
        if self.axis == "S":
            return out.reshape(B, T, S, D)
        else:
            return out.reshape(B, S, T, D).permute(0, 2, 1, 3)


class Factorized2plus1DBlock(nn.Module):
    """ Latte-style: SpatialAttn -> TempAttn -> FFN. """
    def __init__(self, dim, heads, head_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.spatial = AxisAttention(dim, heads, head_dim, "S")
        self.norm2 = nn.LayerNorm(dim); self.temporal = AxisAttention(dim, heads, head_dim, "T")
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.spatial(self.norm1(x))
        x = x + self.temporal(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x
```

### 4.6 Code: Full 3D Spatiotemporal Attention (with FlashAttention via SDPA)

```python
class Full3DSpatiotemporalAttention(nn.Module):
    """
    Treat [B, T, S, D] as [B, N, D] (N = T*S) and run self-attention.
    Uses PyTorch SDPA; on H100/A100 it dispatches to FlashAttention v2/v3 automatically.
    When combined with MM-DiT, text/video tokens are directly concat into this sequence (see §5.4).
    """
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, dim, bias=False)

    def forward(self, x, attn_bias=None):
        # x: [B, N, D] (caller flattens T*S first)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                          # [B, N, H, d_k]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))      # [B, H, N, d_k]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)
```

## §5 MM-DiT for Video (Hunyuan-Video / Mochi-1 idea)

### 5.1 From SD3 MM-DiT to video

SD3 (Esser et al. 2024) introduced **MM-DiT**: concat text tokens and image tokens into the same sequence for self-attention, where **each stream has its own QKV / FFN / AdaLN parameters**, but the attention matrix is shared:

$$Z = [Z_\text{text} \,|\, Z_\text{img}], \quad \text{Attn}(Z) \text{ shared, FFN per-stream}$$

Going video = replace the image stream with a video stream (spacetime tokens); the text stream is unchanged.

### 5.2 Hunyuan-Video's dual-stream → single-stream design

The Hunyuan-Video tech report splits DiT into two phases:

- **Dual-stream blocks** (first N layers): text tokens and video tokens share attention but have independent FFN / AdaLN — letting each modality first refine its own representation.
- **Single-stream blocks** (last M layers): text and video tokens fully share weights, behaving like an ordinary Transformer over a unified sequence.

Similar to Mochi-1's **AsymmDiT** (Mochi makes the video stream 4x wider than the text stream, since video carries far more information than text).

### 5.3 Text encoders (overview)

| Model | Text encoder | Notes |
| --- | --- | --- |
| Hunyuan-Video | CLIP-L + MLLM (LLaVA-style) | MLLM provides long prompts + object relationships |
| Mochi-1 | T5-XXL | Same as SD3 |
| Wan 2.1 / 2.2 | UMT5 (multilingual) | Supports Chinese + English |
| CogVideoX | T5-XXL | — |
| Sora | Not disclosed | OpenAI's report only mentions a re-captioning pipeline |

> 💡 **Prompt rewriter (Hunyuan-Video / Sora reports)** — training-data captions are usually long and detailed, while user short prompts have a gap with the training distribution. The Sora report mentions dense re-captioning of training videos; Hunyuan-Video uses an LLM at inference time to rewrite short prompts into dense captions before feeding the model. A common interview question is "why do we need a prompt rewriter" — answer: align train / inference caption distributions + user prompts are too short on their own.

### 5.4 Code: MM-DiT video block

```python
class MMDiTVideoBlock(nn.Module):
    """
    Dual-stream MM-DiT block (Hunyuan-Video / Mochi-1 / SD3 style).
    Text and video tokens share the self-attention matrix, but QKV / FFN / AdaLN have separate parameters.
    """
    def __init__(self, dim, heads, head_dim, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Separate QKV for each stream
        self.qkv_text = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.qkv_vid  = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.out_text = nn.Linear(heads * head_dim, dim, bias=False)
        self.out_vid  = nn.Linear(heads * head_dim, dim, bias=False)

        # Separate FFN per stream
        hidden = int(dim * mlp_ratio)
        self.mlp_text = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mlp_vid  = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        # Separate AdaLN per stream (modulated by timestep + pooled text condition)
        self.norm_text = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_vid  = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ff_text = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ff_vid  = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_text = nn.Linear(dim, 6 * dim)
        self.cond_vid  = nn.Linear(dim, 6 * dim)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _proj_qkv(self, x, qkv_mod):
        B, L, _ = x.shape
        qkv = qkv_mod(x).reshape(B, L, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                         # [B, L, H, d_k]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [B, H, L, d_k]

    def forward(self, text, vid, cond):
        """
        text: [B, L_t, D]  vid: [B, N_v, D]
        cond: [B, D]  (timestep emb + pooled text)
        """
        # Modulation params
        st_a, sc_t_a, gt_a, st_m, sc_t_m, gt_m = self.cond_text(F.silu(cond)).chunk(6, dim=-1)
        sv_a, sc_v_a, gv_a, sv_m, sc_v_m, gv_m = self.cond_vid(F.silu(cond)).chunk(6, dim=-1)

        # Pre-norm + modulate (per stream, independent)
        t = self._modulate(self.norm_text(text), st_a, sc_t_a)
        v_ = self._modulate(self.norm_vid(vid),  sv_a, sc_v_a)

        # Per-stream QKV (independent parameters)
        qt, kt, vt = self._proj_qkv(t, self.qkv_text)
        qv, kv, vv = self._proj_qkv(v_, self.qkv_vid)

        # Concat both streams along seq dim, then joint self-attention
        q = torch.cat([qt, qv], dim=2)                       # [B, H, L_t + N_v, d_k]
        k = torch.cat([kt, kv], dim=2)
        v = torch.cat([vt, vv], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)        # [B, H, L_t + N_v, d_k]

        L_t = text.shape[1]
        out_t, out_v = out[:, :, :L_t], out[:, :, L_t:]
        out_t = out_t.transpose(1, 2).reshape(*text.shape[:2], -1)
        out_v = out_v.transpose(1, 2).reshape(*vid.shape[:2], -1)

        # Out projection (per stream)
        text = text + gt_a.unsqueeze(1) * self.out_text(out_t)
        vid  = vid  + gv_a.unsqueeze(1) * self.out_vid(out_v)

        # FFN (per stream)
        text = text + gt_m.unsqueeze(1) * self.mlp_text(
            self._modulate(self.norm_ff_text(text), st_m, sc_t_m))
        vid = vid + gv_m.unsqueeze(1) * self.mlp_vid(
            self._modulate(self.norm_ff_vid(vid), sv_m, sc_v_m))
        return text, vid
```

### 5.5 Training objective

Mainstream = **v-prediction or rectified flow vector** (consistent with SD3). Given video latent $x_1$, noise $x_0$, and random $\tau \in [0, 1]$ (logit-normal distribution is more common):

$$x_\tau = (1 - \tau) x_0 + \tau\, x_1, \quad u_\tau = x_1 - x_0$$

$$\mathcal{L} = \mathbb{E}_{\tau, x_0, x_1, c} \, \| v_\theta(\tau, x_\tau, c) - u_\tau \|^2$$

where $c$ is the text condition (+ optional image / video conditioning). Hunyuan-Video / Mochi / Wan all use RF (same paradigm as SD3 on the image side); Sora is not disclosed but the community speculates it is also in the RF / v-pred family.

## §6 Image-to-Video (I2V) — three main conditioning injection methods

### 6.1 First-Frame Channel Concat (simplest, most common)

Encode the reference image $I_\text{ref}$ into a latent $z_\text{ref}^{(0)} \in \mathbb{R}^{C \times 1 \times h \times w}$ (the VAE degenerates to 2D encoding when $T{=}1$), broadcast along the latent's temporal dim $t$ (not the original frame dim $T$) with zero padding, and concat with the noisy latent along the channel dim:

$$\tilde{z}_\tau = \text{concat}_C\!\left(z_\tau, \, \tilde{z}_\text{ref}, \, m\right) \in \mathbb{R}^{(2C+1) \times t \times h \times w}$$

where $\tilde{z}_\text{ref}[:, t', :, :] = z_\text{ref}^{(0)}$ if $t'$ is a reference position (typically $t' = 0$), and 0 elsewhere; $m \in \{0, 1\}^{1 \times t \times h \times w}$ is a mask channel of the same shape (1 at reference positions, 0 at positions to be generated). The first conv layer of the model has its in_channels changed from $C$ to $2C + 1$.

**Pros**: simple to implement, fully compatible with the T2V backbone; only one conv layer needs to change. **Cons**: reference information is mixed into the channels and the model has to extract it via conv; long-range frame guidance is weak.

SVD (Stable Video Diffusion, Blattmann 2023) / DynamiCrafter / I2VGen-XL all use this approach.

### 6.2 Cross-Attention injection (fine-grained control)

Patchify $z_\text{ref}$ into tokens, then inject them as additional K/V into the video stream's attention:

$$\text{CrossAttn}: \quad Q = z_\text{video tokens}, \, K, V = [z_\text{text}, z_\text{ref tokens}]$$

Or more complex still — add a dedicated cross-attention layer for the ref image. AnimateDiff-style LoRA tuning basically follows this route too.

### 6.3 AnimateDiff-style plug-in motion module

> ✅ **AnimateDiff (Guo et al. arXiv 2023-07, ICLR 2024)** — freeze a T2I (Stable Diffusion) and **only insert and train a temporal motion module** (temporal self-attention), turning a T2I into a T2V. Pros: reuse the entire T2I ecosystem (LoRA / ControlNet); cons: quality is bounded by the base T2I. The motion module is usually inserted after each spatial block: `[B,T,C,H,W] -> rearrange [B*H*W, T, C] -> temporal self-attn -> back`.

### 6.4 Code: first-frame concat I2V

```python
class FirstFrameI2VAdapter(nn.Module):
    """
    Broadcast the ref image latent along the time dim + concat it along the channel dim with the noisy latent.
    Usage: replace the DiT's patchify input projection; change in_ch from C to (2C + 1).
    """
    def __init__(self, latent_ch=16, embed_dim=1024, patch=(1, 2, 2)):
        super().__init__()
        self.patch = patch
        # +1 is the mask channel
        self.proj = nn.Conv3d(
            in_channels=2 * latent_ch + 1,
            out_channels=embed_dim,
            kernel_size=patch, stride=patch,
        )

    def forward(self, noisy_latent, ref_latent, ref_mask):
        """
        noisy_latent: [B, C, t, h, w]   - noised latent
        ref_latent:   [B, C, t, h, w]   - reference-frame latent (0 at positions to be generated)
        ref_mask:     [B, 1, t, h, w]   - 1 = this position (along t) is a reference frame, 0 = to be generated
        """
        x = torch.cat([noisy_latent, ref_latent, ref_mask], dim=1)  # [B, 2C+1, t, h, w]
        x = self.proj(x)                                            # [B, D, t', h', w']
        B, D, tt, hh, ww = x.shape
        return x.flatten(2).transpose(1, 2), (tt, hh, ww)           # [B, N, D]
```

> 💡 **Pseudocode for building ref_latent + mask** — given a video latent `z_full` and reference frame indices `ref_indices` (e.g. `[0]` for first-frame I2V), copy `z_full[:, :, idx]` to the same positions in `ref_latent`, set the others to 0, and set the corresponding positions in `ref_mask` to 1. See §A.2 `video_gen_forward` for the concrete implementation.

### 6.5 The "noise-only on future frames" training trick

During I2V training, reference-frame latents get no noise (or extremely small noise), while frames-to-generate get the normal $\tau$ noise. Loss is computed only at to-be-generated positions. This trick is mentioned in SVD / Hunyuan-Video I2V variants.

## §7 Long video generation (open problems)

### 7.1 Three mainstream routes

| Route | Method | Representatives | Drawbacks |
| --- | --- | --- | --- |
| **Keyframe + interpolation** | Generate $K$ keyframes (far apart) first, then interpolate intermediate frames in each segment | NUWA-XL, parts of Movie Gen's pipeline | Keyframe transitions may be unnatural |
| **Hierarchical (coarse → fine)** | First generate a low-fps low-res version, then cascade super-resolution + frame interpolation | Phenaki, Imagen Video, parts of Sora | Multi-stage error accumulation |
| **Autoregressive chunks** | Generate $T$ frames per chunk, conditioning the next chunk on the end of the previous chunk | Mochi-1 partial mode, StreamingT2V | Error accumulation → drift, "looping" |

### 7.2 Sora's "duration as condition"

Sora treats video duration as conditioning (instead of a fixed $T$); training sees videos of different durations, and inference, given a target duration, generates a corresponding patch sequence length. Only briefly mentioned in the public report.

### 7.3 Autoregressive video generation compatibility (Causal VAE earns its keep)

Since the 3D **Causal** VAE encoder does not look into the future, it naturally supports chunk-wise autoregression: the encoder can process segments one by one. But for the DiT part to be autoregressive, causality at the token level is also required — this is an active research direction ("VAR for Video" and friends). Current versions of Hunyuan-Video / Wan are still non-AR diffusion.

### 7.4 Simple chunk autoregressive pseudocode

```python
@torch.no_grad()
def chunk_autoregressive(rf_sample_fn, vae, text, num_chunks, chunk_t_lat=4, overlap=1,
                         latent_shape=(16, 60, 90)):
    """
    Pseudocode skeleton for chunk-wise autoregressive long-video generation.
    rf_sample_fn(z_init, text, ref_full, ref_mask) -> z_out
        — uses the §A.3 video_gen_sample style, but accepts external z_init and ref clamp.
    latent_shape = (C, h_lat, w_lat); each chunk's latent temporal length = chunk_t_lat.
    """
    chunks = []
    ref_latent = None
    C, hl, wl = latent_shape
    device = next(vae.parameters()).device
    for c in range(num_chunks):
        # Initial noise
        z_init = torch.randn(1, C, chunk_t_lat, hl, wl, device=device)
        # Use the last `overlap` frames of the previous chunk as the reference
        if ref_latent is not None:
            z_init[:, :, :overlap] = ref_latent[:, :, -overlap:]
            ref_full = torch.zeros_like(z_init)
            ref_full[:, :, :overlap] = ref_latent[:, :, -overlap:]
            ref_mask = torch.zeros(1, 1, chunk_t_lat, hl, wl, device=device)
            ref_mask[:, :, :overlap] = 1.0
        else:
            ref_full, ref_mask = None, None
        z = rf_sample_fn(z_init, text, ref_full, ref_mask)   # [1, C, chunk_t_lat, h, w]
        chunks.append(z)
        ref_latent = z

    # Concatenate chunks (deduplicating overlap), then VAE decode
    z_full = torch.cat([chunks[0]] + [c[:, :, overlap:] for c in chunks[1:]], dim=2)
    return vae.decoder(z_full)
```

> ⚠️ **Drift problem** — even with overlap, later chunks under AR mode tend to drift in content (characters disappear, color shifts). Movie Gen / Hunyuan generate short clips (5–10s) in one shot with global attention + extend duration via post-processing super-resolution, rather than pure AR.

## §8 Control / editing / character consistency (overview table)

| Task | Representative method | Injection mechanism |
| --- | --- | --- |
| Camera control | **MotionCtrl** (Wang et al. 2023) | CMCM: camera trajectory $R, T$ sequence → temporal embedding |
| Object trajectory control | **MotionCtrl** OMCM module | 2D trajectory points → spatial heatmap as conditioning |
| Character animation | **AnimateAnyone** (Hu et al. CVPR 2024) | ReferenceNet (sibling SD backbone) + Pose Guider (OpenPose/DWPose) + temporal layers |
| Character consistency | IP-Adapter / DreamBooth-style | ID embedding injected via cross-attn |
| Local editing | Mask-guided generation | Character mask + ref image injected together |

## §9 Evaluation: VBench / FVD / CLIPSim-V

> ✅ **VBench (Huang et al. CVPR 2024) — the current SoTA standard** — 16 fine-grained dimensions across two categories.

- **Video Quality (7 dims)**: Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality
- **Video-Condition Consistency (9 dims)**: Object Class, Multiple Objects, Human Action, Color, Spatial Relationship, Scene, Appearance Style, Temporal Style, Overall Consistency

Each dimension has a custom prompt set + automatic evaluator (GroundingDINO / DOVER / RAFT / CLIP, etc.), with a final weighted composite score. Hunyuan-Video / Mochi / Movie Gen / Sora etc. all self-report or are independently measured on VBench.

**FVD (Unterthiner et al. 2018)** — generalization of FID to video. Generated / real videos are each passed through a pretrained video classifier (I3D / VideoMAE / S3D) to get intermediate features, then the Fréchet distance between the two distributions is computed:

$$\text{FVD} = \|\mu_g - \mu_r\|^2 + \text{Tr}\!\left(\Sigma_g + \Sigma_r - 2\sqrt{\Sigma_g \Sigma_r}\right)$$

Lower FVD = generated distribution closer to real; drawback: depends on classifier domain and doesn't reflect text-video alignment.

**CLIPSim-V** — for each frame compute the cosine between the CLIP image embedding and the prompt CLIP text embedding: $\frac{1}{T} \sum_t \cos(\text{CLIP-I}(f_t), \text{CLIP-T}(\text{prompt}))$. Measures text-video alignment but ignores temporal coherence.

**In practice**: the Hunyuan-Video tech report reports both VBench + human eval (1-5 Likert user study); Sora / Veo demos give only qualitative visualizations. When asked "how do you evaluate" in an interview — answer VBench + human eval + FVD on UCF-101 / MSR-VTT.

## §10 Complexity and memory

### 10.1 Token count and attention cost

| Video spec | VAE downsample | Post-VAE latent shape | $N$ after (1,2,2) patchify | Full-3D Attn $N^2$ |
| --- | --- | --- | --- | --- |
| $256{\times}256{\times}16$ (2s, 8fps) | $4{\times}8{\times}8$ | $4 \times 32 \times 32$ | $4 \cdot 16 \cdot 16 = 1024$ | $\approx 10^6$ |
| $720p{\times}48$ (2s, 24fps, latent $12{\times}90{\times}160$) | $4{\times}8{\times}8$ | $12 \times 90 \times 160$ | $12 \cdot 45 \cdot 80 = 43200$ | $\approx 1.9 \times 10^9$ |
| $1080p{\times}120$ (5s, 24fps, latent $30{\times}136{\times}240$) | $4{\times}8{\times}8$ | $30 \times 136 \times 240$ | $30 \cdot 68 \cdot 120 = 244800$ | $\approx 6.0 \times 10^{10}$ |

Blows up fast. A common interview question is "the attention bottleneck for long video / high resolution" — answer: $O(N^2)$ quadratic cost, and even FlashAttention does not reduce the token count itself; the fix is factorized / window / cascaded multi-stage.

### 10.2 Memory and FLOPs key points

- **Score matrix memory**: vanilla is $O(L^2)$; FlashAttention drops it to $O(L)$ activation
- **KV cache**: pure diffusion has no AR steps, so there is no KV cache during training; activation checkpointing + ZeRO-3 are essential
- **3D Causal VAE inference**: chunkable, memory $O(\text{chunk}_T \cdot h \cdot w \cdot C)$
- **Training FLOPs**: a 13B model's per-batch attention FLOPs ≈ $4 B N^2 d$ × layers ($B$ = batch size); Hunyuan-Video full model ≈ several hundred to a thousand PFLOP / sample, requiring thousands of H100s for months of training

## §11 Relation to other generative models

### 11.1 Image diffusion vs Video diffusion

| Dimension | Image (SD3 / SDXL / FLUX) | Video (Hunyuan / Mochi / Wan) |
| --- | --- | --- |
| VAE | 2D VAE, $\downarrow 8\times$ | 3D Causal VAE, $\downarrow 4\times8\times8$ |
| Token count | $\sim 10^3$ (1024² @ 16²) | $\sim 10^4 - 10^5$ |
| Attention | Self + Cross | ST (factorized / full 3D) + Cross / MM-DiT |
| Training token volume | Billions of image-text pairs | Tens of millions of video-text pairs (video data is scarce) |
| Evaluation | FID, CLIPScore | VBench, FVD, CLIPSim-V |
| Main challenges | Aesthetics / prompt following | Temporal consistency / physical plausibility / long duration |

### 11.2 Video LLM vs Video Generation vs AR Video

- **Video LLM** (VideoChat / Video-LLaVA / NaVit): input video, output text (understanding direction); CLIP/ViViT/VideoMAE features + LLM
- **Diffusion Video Gen** (the focus of this article): high quality, with long video via chunk / hierarchical
- **AR Video** (Cosmos-Predict, VAR-Video family): video tokenized (VQ-VAE / FSQ) + LM-style generation. Pros: natural streaming + long video; cons: quality currently below diffusion
- **Unified** (Show-o / Emu3, etc. — emerging 2025): simultaneous understanding + generation; the video version has not yet taken off
- 2024-2025 SoTA is still dominated by diffusion

## §12 25 frequently-asked interview questions

Grouped into 3 levels: L1 must-know (any MLE position), L2 advanced (research-oriented), L3 top labs (diffusion / video specialist). Each item opens to answer key points + common pitfalls.

### L1 must-know (basics — any video-generation-related role)

<details>

<summary>Q1. What is the overall pipeline of today's mainstream video generation models?</summary>

- **3D Causal VAE** compresses video from $H{\times}W{\times}T$ to an $h{\times}w{\times}t$ latent
- **Latent DiT / MM-DiT** performs diffusion / flow matching in the latent space
- **Text encoder** (T5 / CLIP / MLLM) provides conditioning
- **VAE Decoder** decodes the latent back to video
- Inference uses an ODE/SDE sampler (Euler / Heun / DDIM) integrating from noise to data

Pitfall: saying only "diffusion + UNet" while forgetting the latent / 3D / VAE trio; or claiming attention runs in pixel space (long since infeasible).

</details>

<details>

<summary>Q2. What does "Causal" in 3D Causal VAE mean? Why must it be causal?</summary>

- In the temporal dim, conv padding is all stacked on the **left (past)**, with no right padding, so the **current frame cannot see the future**
- Three key benefits: (1) **images and video can share the latent space** (when $T{=}1$ it degenerates to 2D, so images also pass through the encoder), enabling joint image+video training; (2) **streaming / autoregressive inference**, long videos can be processed chunk-wise; (3) during training images count as $T{=}1$ videos for high data utilization
- Hunyuan-Video / Wan / Mochi / CogVideoX all use causal 3D VAE

Pitfall: saying "causal just means masking the future" — that is the surface; the key is why those 3 engineering benefits matter.

</details>

<details>

<summary>Q3. What problem do Sora's "spacetime patches" solve?</summary>

- Slice the video latent into 3D patches $p_t \times p_h \times p_w$, each patch becoming a token
- **Supports variable resolution / duration / aspect ratio** — different-shape videos can be packed into the same batch (separated by attention masks)
- Like ViT but with the added time dim; decouples network architecture from concrete input shape

Pitfall: saying only "slice patches" without explaining the key benefit (variable-shape training / inference).

</details>

<details>

<summary>Q4. How big is the complexity gap between Factorized 2+1D attention and Full 3D attention?</summary>

- Let per-frame tokens be $S$, temporal tokens $T_t$, total tokens $N = S T_t$
- **Factorized 2+1D**: spatial $O(T_t S^2)$ + temporal $O(S T_t^2)$
- **Full 3D**: $O(N^2) = O(S^2 T_t^2)$
- Full 3D is $\min(S, T_t)$ times more expensive than Factorized
- But Full 3D **has more expressive power** (allowing joint spatial + temporal patterns)

Pitfall: comparing only Full 3D without counting the temporal term; or ignoring that the essential difference is "whether you assume space ⊗ time is separable".

</details>

<details>

<summary>Q5. How does MM-DiT for video inject text?</summary>

- Text tokens and video tokens are **concatenated into a single sequence** for self-attention
- Each stream has independent QKV / FFN / AdaLN parameters (dual-stream), but the attention matrix is shared
- Compared to traditional cross-attention (Q from video, KV from text), MM-DiT lets the two modalities interact **at the token level on equal footing**
- Hunyuan-Video / Mochi adopt this architecture (video side); SD3 (Esser et al. 2024) is the original MM-DiT on the image side

Pitfall: saying "video backbone + cross-attn from text" — that's SDXL style; MM-DiT has replaced it.

</details>

<details>

<summary>Q6. What is the simplest implementation of I2V (Image-to-Video)?</summary>

- Encode the reference image $I_\text{ref}$ into a latent $z_\text{ref}$
- Broadcast along the time dim, concat along the channel dim with the noisy latent: change input channels from $C$ to $2C{+}1$ (plus a mask channel marking which positions are ref)
- Other parts of the model unchanged; only the first patchify conv layer is modified
- During training, ref frames receive no noise / extremely small noise, and loss is computed only on to-be-generated frames

Pitfall: saying "just feed in another image" without explaining where the concat happens or how the mask is handled.

</details>

<details>

<summary>Q7. What is the core idea of AnimateDiff?</summary>

- **Freeze** a T2I model (Stable Diffusion)
- **Insert + only train a temporal motion module** (temporal self-attention), placed after each spatial block
- Pros: directly reuses the entire T2I ecosystem (LoRA, ControlNet, custom checkpoints)
- Cons: quality is bounded by the base T2I model; long video is hard

Pitfall: saying "replace UNet with Transformer" — wrong; AnimateDiff is a plug-in that does not replace the backbone.

</details>

<details>

<summary>Q8. What does VBench evaluate? How does it differ from FVD?</summary>

- **VBench** (Huang CVPR 2024): 16 **fine-grained dimensions** (Subject Consistency, Motion Smoothness, Object Class, Color, ...), each scored by a dedicated detector / classifier; the current SoTA video generation standard
- **FVD** (Unterthiner 2018): FID generalized to video; uses I3D / VideoMAE features to compute Fréchet distance between two distributions; a single number
- VBench is more interpretable and pinpoints which dimension is weak; FVD is single-number but black box
- In practice: both are reported + human eval

Pitfall: only knowing FVD without VBench — a big gap after 2024.

</details>

<details>

<summary>Q9. Which of Sora / Hunyuan-Video / Mochi-1 / Kling are open source?</summary>

- **Open source**: Hunyuan-Video (13B, 2024-12), Mochi-1 (10B, 2024-10), CogVideoX (2B/5B, arXiv 2024-08), OpenSora / OpenSora-Plan, LTX-Video (2B, 2024-11), Wan 2.1/2.2 (14B, 2025), SVD (2023-11)
- **Closed source**: Sora (2024-02), Veo / Veo 2 (Google), Kling (Kuaishou 2024-06), Movie Gen (Meta, 30B)
- Domestic Chinese open-source mainstays: Hunyuan / Wan / CogVideoX

Pitfall: only remembering Sora and SVD — you must know both open and closed timelines.

</details>

<details>

<summary>Q10. Text encoder choice: T5 vs CLIP vs MLLM?</summary>

- **CLIP-L / G**: aligned well with image latents (image-text jointly trained), works well for short text but poor at long-prompt following
- **T5-XXL**: sequence-to-sequence model, **better at long-prompt following**; SD3 / Mochi / CogVideoX all use it
- **MLLM** (LLaVA-like): strong at understanding object relationships within prompts; Hunyuan-Video uses a dual CLIP-L + MLLM encoder
- In production, long prompts are also handled at inference time by a **prompt rewriter** (LLM rewrites) to bridge the train/test caption distribution gap

Pitfall: saying "just CLIP" — after 2024 it has largely been replaced by T5 / MLLM or used alongside them.

</details>

### L2 advanced (research-oriented / mid-level role)

<details>

<summary>Q11. Derive why a 3D Causal VAE at $T{=}1$ is "behaviorally" equivalent to 2D conv.</summary>

- Causal 3D conv with temporal kernel size $k_t$, left pad $k_t - 1$, right pad 0, stride=1.
- Input shape $[B, C, 1, H, W]$ ($T{=}1$ single frame); after left-padding $k_t - 1$, the time-dim length becomes $1 + (k_t - 1) = k_t$, where the first $k_t - 1$ positions are padding zeros and the last 1 is the true frame.
- Output time length $= (k_t + 0 - k_t)/1 + 1 = 1$ (using the conv output length formula $(L_\text{in} + p - k) / s + 1$, with $L_\text{in}=1$, $p=k_t-1$, $k=k_t$, $s=1$). So **only a single time slot is produced**.
- Let the time-dim weight sequence be $W_0, W_1, \dots, W_{k_t-1}$; the computation at the sole output position is $\sum_{\tau=0}^{k_t-1} W_\tau \star x_\tau$. But only $\tau = k_t - 1$ corresponds to the real frame; the rest of the $x_\tau$ are padding zeros — **only the time slice $W_{k_t - 1}$ contributes to the output**.
- **Conclusion**: at $T{=}1$ the entire causal 3D conv degenerates to an **effective** 2D conv (kernel = $W_{k_t - 1}[:, :, :, :]$); the remaining time-slice weights are not exercised by any real input on the image path.

> Engineering: at training time, image $T{=}1$ batches traverse the "2D conv subset" path, with no need for an additional 2D head; this is the prerequisite that lets Hunyuan-Video / Mochi / CogVideoX jointly train image + video.

Pitfall: treating "equivalent to 2D conv" as a physical weight merge — actually, padding-0 simply causes the other time slices to have no effect on image inputs.

</details>

<details>

<summary>Q12. Expressive-power difference between Full 3D attention and factorized 2+1D? (between L2/L3)</summary>

- Factorized: first $\text{Attn}_S$ lets spatial tokens attend to each other (within each frame), then $\text{Attn}_T$ attends along the time dim
- **Assumption**: spatiotemporal interactions are separable — the relation between any two spacetime positions $(t_1, s_1)$ and $(t_2, s_2)$ decomposes into "first $s_1 \leftrightarrow s_2$ at $t_1$" then "$t_1 \leftrightarrow t_2$ at $s_2$"
- Full 3D makes no separability assumption: any two tokens attend directly; can learn **diagonal spatiotemporal patterns** (e.g. diagonal motion)
- **Mathematically**: factorized is a **strict subset** of full 3D (a constrained parameterization)
- Empirically: full 3D is better at fast motion / complex spatiotemporal patterns

Pitfall: saying only "full 3D is more accurate" without identifying what patterns it can learn (diagonal / non-separable motion).

</details>

<details>

<summary>Q13. Why are video VAEs mainstream-bounded in the $\downarrow 4{\sim}8 \times 8 \times 8$ range?</summary>

- Voxel compression ratio $= 4 \cdot 8 \cdot 8 = 256$ ($1{:}256$); with $C=16$ latent channels, the net ratio is about $1{:}48$
- **More aggressive (e.g. LTX-Video reports $1{:}8192$ overall) → fewer tokens but degraded reconstruction quality** (detail / sharpness loss)
- DiT backbone attention is $O(N^2)$, so aggressive compression is engineering-friendly; but lost detail is hard to recover
- $4{\times}$ temporal downsampling is empirically the upper bound where motion smoothness stays acceptable — higher temporal compression introduces jitter in fast-motion scenes
- Each model is configured differently: CogVideoX uses $4{\times}8{\times}8$; Hunyuan-Video uses $4{\times}8{\times}8$; Mochi-1 is more aggressive on the temporal dim (about $6{\times}8{\times}8$, per the Mochi blog); LTX-Video goes for extreme high compression + real-time
- Even with the same backbone, VAE choice is an active design decision

Pitfall: lumping all models into the same $4{\times}8{\times}8$ bucket — Mochi-1 / LTX-Video use more aggressive temporal compression.

</details>

<details>

<summary>Q14. Why is logit-normal $t$ sampling useful in video FM training?</summary>

- SD3 (Esser 2024) found that $t \sim \mathcal{U}[0,1]$ is suboptimal: the middle region $t \approx 0.5$ has the hardest noise/signal ratio
- Switch to $t = \sigma(\tau), \tau \sim \mathcal{N}(m, s^2)$, concentrating $t$ near 0.5
- The same phenomenon is observed in video (Hunyuan / Mochi both default to logit-normal)
- Intuition: away from $t=0$ and $t=1$, the model must learn on mixed noise/signal, which is the hardest region

Pitfall: saying "loss reweighting" — logit-normal is not loss reweighting but a change of the $t$ sampling distribution.

</details>

<details>

<summary>Q15. How does RoPE-3D encode (time, height, width)?</summary>

- Split the $d$-dim query/key into 3 chunks $q = [q^{(t)} \,|\, q^{(h)} \,|\, q^{(w)}]$ (disjoint subspaces), one per (t, h, w)
- Each chunk uses 1D RoPE (frequencies $\theta_d = 1 / 10000^{2k/d}$ like vanilla Transformer)
- Result is the concat of block-diagonal rotations: $\text{RoPE}_{3D}(q) = [R_t(\theta_t) q^{(t)} \,|\, R_h(\theta_h) q^{(h)} \,|\, R_w(\theta_w) q^{(w)}]$
- Done per head independently; after attention, **relative spatiotemporal position** information emerges naturally
- CogVideoX / Hunyuan-Video both use similar schemes

Pitfall: treating it as absolute embedding — RoPE is fundamentally a relative position encoding.

</details>

<details>

<summary>Q16. Difference between Stable Video Diffusion (SVD) and AnimateDiff?</summary>

- **SVD** (Blattmann 2023): finetune the entire SD2.1 UNet, add temporal layers; I2V-focused, with 14-frame and 25-frame variants; entire model trained
- **AnimateDiff** (Guo 2024 ICLR): **freeze SD-T2I** and only train a plug-in temporal motion module; T2V-focused
- SVD has better quality but loses the SD T2I ecosystem; AnimateDiff sacrifices quality for plug-and-play
- Both are early 2023 methods, surpassed in 2024 by DiT-based large models

Pitfall: saying "both are video diffusion" — the difference is finetune vs plug-in.

</details>

<details>

<summary>Q17. Comparison between DiT with AdaLN vs cross-attn? Why does DiT-Video also prefer AdaLN?</summary>

- **AdaLN** (DiT default): pool the condition into a single vector $c$ and predict the LayerNorm scale + shift + gate
- **Cross-attn**: condition tokens are K/V, video tokens are Q
- AdaLN is simple / cheap / stable; but provides **global conditioning** — every token sees the same modulation
- Cross-attn lets each token selectively attend to the condition; expensive when tokens are many
- MM-DiT (SD3 / Hunyuan / Mochi) actually **combines** the two: the MM part shares attention while AdaLN modulates the norm
- Pure video DiT with AdaLN started with Latte / OpenSora; later supplanted by MM-DiT

Pitfall: treating AdaLN as outdated — MM-DiT still embeds AdaLN modulation.

</details>

<details>

<summary>Q18. What is special about CFG in video generation?</summary>

- Same formula as image: $v_\text{CFG} = v_\theta(\emptyset) + s \cdot (v_\theta(c) - v_\theta(\emptyset))$
- **CFG done per frame**, so 2x forward cost $\times$ T frames
- **Guidance scale typically 5-7.5** (close to SDXL / SD3)
- In video, an overly large $s$ easily causes **frame flickering / oversaturation** — more sensitive than images
- Some implementations use **temporal-aware CFG** (different $s$ at different timesteps / frames)

Pitfall: saying "CFG is the same in video as in image" — usually true, but the flickering sensitivity is video-specific.

</details>

<details>

<summary>Q19. What is Hunyuan-Video's dual-stream → single-stream design?</summary>

- **Dual-stream blocks** (early layers): text and video tokens share the attention matrix but have independent QKV / FFN / AdaLN per stream — letting each modality first refine its own representation
- **Single-stream blocks** (later layers): fully shared parameters, equivalent to a unified Transformer
- Intuition: early stages need to preserve modality specificity (text is sequential, video is spatiotemporal); later stages can unify
- Mochi's AsymmDiT is similar in spirit but more aggressive — the video stream is 4x wider than the text stream

Pitfall: saying "they are all self-attn on a unified sequence" — missing stream-specific parameters / the early-vs-late distinction.

</details>

<details>

<summary>Q20. What is Mochi-1's AsymmDiT (asymmetric MM-DiT)?</summary>

- The two streams (text / video) have different hidden dims: video is 4x wider than text (e.g. video $D{=}3072$ / text $D{=}768$)
- Intuition: video carries much more information than text, so allocate more parameters to the video stream
- During attention, text and video project to a common head dim and run self-attention
- FFN is independent per stream, sized per its own width
- Uses fewer text-side parameters than standard MM-DiT, saving memory without losing quality

Pitfall: treating "asymmetric" as if the two streams don't interact — wrong; they still share attention.

</details>

### L3 top-lab questions (deep — diffusion / video specialist)

<details>

<summary>Q21. 3D Causal VAE vs standard 3D VAE: besides streaming, what other essential differences are there? (must-ask)</summary>

- **Standard 3D VAE**: the time kernel is center-aligned, needs symmetric padding, so **output[t] depends on input[t-\lfloor k_t/2 \rfloor \ldots t+\lfloor k_t/2 \rfloor]** — looks at the future
- **3D Causal VAE**: the time kernel is causally aligned, with left pad $k_t - 1$ and no right pad, so **output[t] depends only on input[t-k_t+1 \ldots t]**

**Essential differences**:

- **Streaming / autoregressive inference**: causal allows chunk-wise encoding, encoding each new segment of video on arrival; a standard VAE must wait for the full clip (the kernel center needs future frames)
- **Image-video same space**: both, at $T{=}1$, can in principle "degenerate to 2D" (zero padding renders other time slices ineffective), but the causal design is "past all visible + future all zero", consistent with its behavior at large $T$ during training; with symmetric padding at $T{=}1$, the kernel sees $\lfloor k_t/2 \rfloor$ padding positions on each side, behaviorally inconsistent with the $T \gg 1$ boundary case — **worse consistency**
- **Train + inference consistency**: a causal VAE uses one padding rule for training (full clip) and inference (AR chunk); a standard 3D VAE lacks the future frames at AR chunk inference time and needs special padding
- **Compatibility with AR video**: causal is a necessary prerequisite for AR video generation (VAR-Video / Cosmos)
- **Training data utilization**: image-only batches behave the same in the causal VAE encoder as during training; a standard 3D VAE needs dummy symmetric padding, with a gap to the normal training distribution

> 💡 **Bonus**: explain how a causal VAE's downsampling layer works (preserving causality when time stride is 2); explain the implementation details for mixing image + video in training batches.

Pitfall: saying only "causal means masking the future" — necessary but not sufficient; expand on "why this closes the ecosystem loop (image + video same stack, AR compatibility, streaming)".

</details>

<details>

<summary>Q22. Engineering implementation for handling arbitrary resolution / duration / aspect ratio with spacetime patches?</summary>

- **Patchify token count $N = (t/p_t)(h/p_h)(w/p_w)$ varies with input shape**; the Transformer is insensitive to $N$ (self-attn is set-like)
- **RoPE-3D**: relative position encoding, not bound to a maximum length; any $(t, h, w)$ triple can be rotated
- **Packing**: within a batch, concatenate samples of mixed shapes into one seq + a segment attention mask (visible within a sample, masked across samples) + pad to max length
- **Aspect ratio / duration as conditioning**: extra tokens / scalars injected via AdaLN, so the model can learn reasonable composition for different ratios
- Sora first productionized; community reproductions include OpenSora-Plan / OpenSora 1.2+

Pitfall: saying only "Transformers don't care about length" — expand on the full engineering stack: packing + segment mask + RoPE-3D + aspect ratio conditioning.

</details>

<details>

<summary>Q23. Precise complexity formulas for Factorized 2+1D vs Full 3D attention + when to choose which?</summary>

- Let per-frame spatial tokens $S = (h/p_h)(w/p_w)$, temporal tokens $T_t = t/p_t$, total tokens $N = S T_t$
- **Factorized 2+1D**:
  - Spatial attn (independent per frame): $T_t \cdot S^2 \cdot d$
  - Temporal attn (independent per spatial position): $S \cdot T_t^2 \cdot d$
  - Total $O\!\left(d \cdot (T_t S^2 + S T_t^2)\right) = O\!\left(d \cdot S T_t (S + T_t)\right)$
- **Full 3D**: $O(d \cdot N^2) = O\!\left(d \cdot S^2 T_t^2\right)$
- Ratio = full / factorized = $\frac{S^2 T_t^2}{S T_t(S + T_t)} = \frac{S T_t}{S + T_t}$, of order $\min(S, T_t)$
- For $S \approx 10^3$ (720p latent), $T_t \approx 30$ (5s): full is $\approx 30\times$ more expensive than factorized (dominated by $\min$)

**When to choose which**:

- **Sufficient compute + high quality target**: Full 3D (Sora / Hunyuan / Mochi)
- **Budget constrained / fast iteration**: Factorized 2+1D (Latte / OpenSora / AnimateDiff)
- **Quality–speed compromise**: most blocks factorized + a few full 3D (the OpenSora-Plan approach)
- For memory-sensitive cases, factorized + recompute is the cheaper option

Pitfall: giving only a qualitative answer "full is more expensive" — for L3, give precise formulas + engineering trade-off.

</details>

<details>

<summary>Q24. What is the most promising route for long video (>30s) right now? Why does chunk AR drift easily?</summary>

- **Drift, mathematically**: each chunk samples from $p(x | \hat{z}_\text{prev})$, where $\hat{z}_\text{prev}$ is itself model-generated (not real data); multi-step conditioning accumulates error (like RNN exposure bias)
- **Mitigation**: overlap + clamp the known portion; multi-frame reference (not just the last 1 frame); global learned motif tokens
- **More promising routes**:
  - **Hierarchical**: first generate low-fps keyframes, then temporal super-resolution to interpolate middle frames
  - **Long-context full attention**: long video directly + sparse / sliding / hierarchical to control cost
  - **Diffusion Forcing and other hybrid approaches**
- Sora's public 60s demo doesn't disclose internal details; Movie Gen uses multi-stage keyframe + interp + super-res

Pitfall: saying only "AR drifts" — give a concrete mathematical explanation of drift + the current SoTA response.

</details>

<details>

<summary>Q25. Hunyuan-Video / Mochi-1 / Wan all use Rectified Flow; what's the advantage over DDPM ε-pred?</summary>

- **Target is stationary**: RF's $u_t = x_1 - x_0$ is a constant given $(x_0, x_1)$; magnitude variation across $t$ is much smaller than ε-pred
- **Few-step sampling**: the path is a straight line (ideal), so few-step ODE has small error; after Reflow, 1-4 steps suffice
- **Natural loss conditioning**: no need for explicit reweighting like SNR / VLB / EDM-preconditioning of DDPM
- **Shared stack with SD3**: image / video use one recipe (v-pred + logit-normal $t$)
- DDPM ε-pred needs careful noise schedule design + the target scale at different $t$ varies hugely, complicating training reweighting; RF "works once for all".

Pitfall: saying only "RF has a straighter path" — expand on training objective / numerical stability / sampler compatibility from multiple angles.

</details>

## §A Appendix: Full from-scratch video generation model skeleton

### A.1 Overall class diagram

```

VideoGenModel
├── encoder: CausalVAEEncoder        (3D Causal VAE)
├── decoder: CausalVAEDecoder
├── text_encoder: T5Encoder / MLLM   (frozen)
├── patchify: SpacetimePatchify      (3D patch -> token)
├── unpatchify: SpacetimeUnpatchify
├── time_embed: SinusoidalTimeEmbed  (timestep τ)
├── transformer:
│     ├── MMDiTVideoBlock × N_dual   (dual-stream)
│     └── SingleStreamBlock × N_sing (single-stream)
└── i2v_adapter: FirstFrameI2VAdapter (optional, for I2V mode)
```

### A.2 Training forward + A.3 Euler RF sampler

```python
def run_dit(model, z, text, tau, ref_full=None, ref_mask=None):
    """ DiT forward: noisy latent z -> predicted vector field. """
    if ref_full is None:
        tokens, grid = model.patchify(z)
    else:
        tokens, grid = model.i2v_adapter(z, ref_full, ref_mask)
    text_tok = model.text_encoder(text)
    cond = model.time_embed(tau) + text_tok.mean(dim=1)        # [B, D]
    for blk in model.transformer.dual_stream:
        text_tok, tokens = blk(text_tok, tokens, cond)
    for blk in model.transformer.single_stream:
        cat = torch.cat([text_tok, tokens], dim=1)
        cat = blk(cat, cond)
        text_tok, tokens = cat[:, :text_tok.size(1)], cat[:, text_tok.size(1):]
    return model.unpatchify(tokens, grid)                       # [B, C, t, h, w]


def video_gen_forward(model, video_clean, text, ref_image=None):
    """ RF v-prediction MSE loss. video_clean: [B,3,T,H,W]; ref_image: optional [B,3,H,W]. """
    B = video_clean.size(0); device = video_clean.device
    mu, logvar = model.encoder(video_clean)
    z1 = model.encoder.reparameterize(mu, logvar)
    z0 = torch.randn_like(z1)
    tau = torch.sigmoid(torch.randn(B, device=device))          # SD3 logit-normal
    tau_b = tau.view(-1, 1, 1, 1, 1)
    z_tau = (1 - tau_b) * z0 + tau_b * z1
    u_target = z1 - z0
    ref_full, ref_mask = None, None
    if ref_image is not None:
        mu_r, _ = model.encoder(ref_image.unsqueeze(2))         # T=1 path
        ref_full = torch.zeros_like(z1); ref_full[:, :, :1] = mu_r
        ref_mask = torch.zeros(B, 1, *z1.shape[2:], device=device); ref_mask[:, :, :1] = 1.0
        # I2V trick: ref frames kept clean (no noise), aligning train and inference distributions
        ref_mask_C = ref_mask.expand_as(z_tau).bool()
        z_tau = torch.where(ref_mask_C, z1, z_tau)
    v_pred = run_dit(model, z_tau, text, tau, ref_full, ref_mask)
    # I2V: compute loss only on to-be-generated frames
    if ref_mask is not None:
        gen = (1 - ref_mask).expand_as(v_pred)
        return ((v_pred - u_target).pow(2) * gen).sum() / gen.sum().clamp(min=1.0)
    return F.mse_loss(v_pred, u_target)


@torch.no_grad()
def video_gen_sample(model, text, t_lat=12, h_lat=90, w_lat=160, steps=50,
                     ref_image=None, guidance_scale=6.0, C=16):
    """ End-to-end inference: t_lat/h_lat/w_lat directly specify the latent shape (avoiding the T/4 divisibility trap). """
    device = next(model.parameters()).device
    z = torch.randn(1, C, t_lat, h_lat, w_lat, device=device)
    ref_full, ref_mask = None, None
    if ref_image is not None:
        mu_r, _ = model.encoder(ref_image.unsqueeze(2))
        ref_full = torch.zeros_like(z); ref_full[:, :, :1] = mu_r
        ref_mask = torch.zeros(1, 1, t_lat, h_lat, w_lat, device=device); ref_mask[:, :, :1] = 1.0
        # At inference time, clamp the ref frame positions to clean latent right from start
        # (consistent with training), otherwise the first DiT step sees noise there
        z = torch.where(ref_mask.bool().expand_as(z), ref_full, z)
    taus = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        tau_i = taus[i].expand(1)
        v_cond   = run_dit(model, z, text, tau_i, ref_full, ref_mask)
        v_uncond = run_dit(model, z, [""],  tau_i, ref_full, ref_mask)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        z = z + (taus[i + 1] - taus[i]) * v
        if ref_mask is not None:                                # I2V: clamp ref frames at every step
            z = torch.where(ref_mask.bool().expand_as(z), ref_full, z)
    return model.decoder(z)
```

> ✅ **Sanity checks commonly done on video generation models** —

- **Reconstruction-only**: VAE encode then decode, check PSNR / SSIM / LPIPS to verify the model can reconstruct the original
- **Random latent decode**: sample latent from $\mathcal{N}(0, I)$ and decode, to see whether the VAE has degraded (should yield "natural-looking" videos whose semantics may not be coherent)
- **Class / prompt overfit**: train on a single prompt for 1000 steps to see if the model can memorize
- **VBench partial dimensions**: mid-training, first evaluate Subject Consistency / Motion Smoothness to verify temporal stability
- **CFG sweep**: scale = 0/1/3/6/10 to see the effect (too high → saturation / flicker)

### A.4 Current SoTA overview

| Dimension | Current SoTA | Notes |
| --- | --- | --- |
| Best open-source T2V | Hunyuan-Video 13B | 2024-12, quality approaches closed-source |
| Open-source I2V | Wan 2.2 I2V / Hunyuan-Video I2V | Friendly to Chinese prompts |
| Best closed-source | Veo 2 / Sora / Kling | 4K / long duration |
| Real-time | LTX-Video 2B | Real-time on RTX 4090 |
| Video + audio | Movie Gen (closed) | 30B joint generation |
| Control | AnimateAnyone / MotionCtrl | Character / camera control |

---

**Video Generation Quick Reference** · Main references: Sora technical report (OpenAI 2024), Hunyuan-Video tech report (Tencent 2024), Mochi-1 blog (Genmo 2024), CogVideoX (Yang et al. arXiv 2024-08 → ICLR 2025), Wan tech report (Team Wan / Ang Wang et al. arXiv 2025), Movie Gen tech report (Meta 2024), VBench (Huang et al. CVPR 2024), AnimateDiff (Guo et al. arXiv 2023-07 → ICLR 2024), SVD (Blattmann et al. arXiv 2023-11), Latte (Ma et al. arXiv 2024-01 → TMLR 2025), MotionCtrl (Wang et al. 2023 → SIGGRAPH 2024), AnimateAnyone (Hu et al. CVPR 2024), SD3 (Esser et al. ICML 2024)
