## §0 TL;DR Cheat Sheet

> 💡 **7 句话搞定 Video Generation** — 2024-2025 视频生成大爆发。一页吃下面试高频要点（详见后文 §1–§11 推导）。

1. **范式**：主流视频生成 = **3D Causal VAE 压缩 + Latent DiT 扩散 / Flow Matching**。Sora（2024-02）首次把 Transformer 推到 60s + 高分辨率；Hunyuan-Video (Tencent 2024-12) / Wan 2.x (Alibaba 2024-2025) / Mochi-1 / CogVideoX / Movie Gen / Kling / Veo 2 都是这一架构家族的变体。

2. **3D Causal VAE**：把 $H \times W \times T$ 视频压成 $h \times w \times t$ latent（典型空间下采样 $8\times$、时间下采样 $4\times$）。**因果 (causal)** 关键：当前帧 latent **不能看未来帧**——使训练好的 VAE 同时支持图像（$t{=}1$）与视频（$t{>}1$）、并允许后续帧流式 / 自回归生成。

3. **Spacetime Patches (Sora)**：把视频 latent 切成 $p_t \times p_h \times p_w$ 的 3D patch 当 token；像 ViT 但增加了时间维。支持**变分辨率 / 变时长 / 变长宽比**：直接把不同 shape 的 patch 序列打包到一个 batch（用 mask 区分），无需 resize 到固定尺寸。

4. **时空 Attention 三大变体**：(a) **Factorized 2+1D**（Latte / OpenSora / AnimateDiff）：先 spatial-only，再 temporal-only，复杂度 $O(T \cdot S^2 + S \cdot T^2)$；(b) **Full 3D**（Sora / Hunyuan-Video / Mochi）：所有 token 互相 attend，复杂度 $O((ST)^2)$，最贵但效果最好；(c) **Window / 稀疏 ST**（Wan 2.x / CogVideoX 部分块）：滑窗 3D，折中。

5. **MM-DiT for Video (Hunyuan-Video / Mochi)**：text token 与 video token **同序列**做 self-attention（不再用 cross-attn），两个 stream 各自的 QKV 投影 + AdaLN 调制；条件信息在 token-level 直接交互——Hunyuan/Mochi/Wan 把 SD3 image MM-DiT 思路扩展到视频。

6. **Image-to-Video (I2V)**：主流三招——(i) **First-frame concat**：把 ref image 编码后沿 channel 拼到 latent；(ii) **Cross-attention 注入**：ref image token 作 K/V；(iii) **AnimateDiff** 风格：冻结 T2I 主干，只插入 temporal module。SVD / DynamiCrafter / I2VGen-XL / Wan-I2V 是代表。

7. **长视频** = keyframe + interpolation / hierarchical / autoregressive chunks。**评测**：VBench (Huang CVPR 2024) 16 维细粒度评分，是当下事实标准；FVD（Unterthiner 2018）仍作辅助；CLIPSim-V 评估 text-video 对齐。

> ⚠️ **Caveat** — 本文中 model-specific 数字（参数量、压缩比、attention 类型）均依据各模型公开 paper / tech report；具体训练 hyperparam 与最终架构以原文为准。

## §1 直觉与全景

### 1.1 为什么视频比图像难

记 $H{\times}W{\times}T$ 视频，pixel 数随 $T$ 线性增长，token 数（patchify 后）也线性增长。**Attention 复杂度二次于 token 数**——这意味着在原 pixel 空间做 full 3D attention 对 $T \ge 16$ 已经爆炸。所以视频生成的核心工程问题是：

- **压缩**：3D VAE 在空间 + 时间双维度压缩 token 数（典型 $8 \times 8 \times 4$）
- **架构**：在 latent 上跑 attention（spatiotemporal patterns 是设计点）
- **生成范式**：DDPM / Rectified Flow / FM（v-prediction 主流；SD3-style）

### 1.2 2024-2025 时间线（按发布顺序）

| 时间 | 模型 | 出品 | 关键贡献 |
| --- | --- | --- | --- |
| 2023-07 → ICLR'24 | **AnimateDiff** | Guo et al. | T2I 主干 + plug-in motion module，开源 I2V/T2V 鼻祖 |
| 2023-11 | **SVD** | Stability | I2V 开源 baseline；SD2.1 + temporal layers |
| 2024-01 | **Latte** | Ma et al. | 早期 DiT-Video；factorized spatial+temporal |
| 2024-02 | **Sora** | OpenAI | DiT + spacetime patches + 大规模 caption；闭源 |
| 2024-05 | **Veo** | Google DeepMind | 闭源 1080p / 1 分钟 |
| 2024-06 | **Kling** | Kuaishou | 国产闭源，时长 2 分钟 |
| 2024-08 | **CogVideoX** (arXiv) | Zhipu/THU | 开源 2B/5B；Expert Transformer + 3D VAE |
| 2024-10 | **Movie Gen** | Meta | 30B；视频 + 音频联合；DiT + FM |
| 2024-10 | **Mochi-1** | Genmo | 10B 开源；AsymmDiT 非对称 MM-DiT |
| 2024-11 | **LTX-Video** | Lightricks | 实时（2B）；强压缩 VAE + DiT |
| 2024-12 | **Hunyuan-Video** | Tencent | 13B 开源 SoTA；3D Causal VAE + MM-DiT + prompt rewriter |
| 2024-12 | **Veo 2** | Google | 4K / 2 分钟 |
| 2025-02 | **Wan 2.1** | Alibaba (Team Wan) | 14B 开源；T2V + I2V 双模 |
| 2025-07 | **Wan 2.2** | Alibaba | 升级版；MoE expert + 更长时序 |
| 2024-2025 | **OpenSora / OpenSora-Plan** | HPC-AI / PKU | 全开源训练栈复刻 Sora pipeline |

> ⚠️ **闭源 vs 开源** — Sora / Veo / Kling / Movie Gen 没放 weights，所有架构细节基于官方 technical report；面试时不要把内部 ablation 当真硬数字，要主动说 "according to their report"。

### 1.3 整体 pipeline（共性骨架）

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

## §2 3D Causal VAE — 视频压缩的基石

### 2.1 形式化

VAE encoder $E$ 与 decoder $D$：

$$E: \mathbb{R}^{3 \times T \times H \times W} \to \mathbb{R}^{C \times t \times h \times w}, \quad D: \mathbb{R}^{C \times t \times h \times w} \to \mathbb{R}^{3 \times T \times H \times W}$$

下采样率 $T/t \in \{4, 8\}$（时间）、$H/h, W/w \in \{8, 16\}$（空间）。Hunyuan-Video / CogVideoX / Wan 用 $4{\times}8{\times}8$，LTX-Video 极限 $8{\times}32{\times}32 = 8192\times$ token 压缩。

潜空间维度 $C$ 一般 $16$（Hunyuan）或 $4$（OpenSora）；越大保留信息越多，但 latent prior 离 $\mathcal{N}(0,I)$ 越远，diffusion 收敛越慢。

### 2.2 为什么必须 "Causal" — 三个关键收益

> ✅ **3D Causal VAE 三大优势**

- **Image / video 同构**：当 $T=1$（单帧）时，causal 3D conv 退化为 2D conv（kernel 在时间维只看历史，但历史为空 → 等价于无时间维）。Encoder 既能压视频也能压图像，**图像与视频可以共享同一 latent 空间**——这是 Hunyuan-Video / Wan 同时支持 I2V/T2V 的前提。

- **流式 / 自回归推理**：因果保证当前帧 latent 只依赖历史帧，**长视频可分 chunk 处理**，无需一次性 decode 全部 $T$ 帧；与 LLM 的 KV cache 同源思路。

- **训练数据 utilization**：图像 + 视频混合训练时，图像可视作 $T{=}1$ 视频；标准 3D VAE 在时间维 kernel 中心要看未来，不能这么做。

### 2.3 Causal Conv3d 的实现

标准 3D conv 在时间维做 zero padding 时左右各 pad $\lfloor k_t/2 \rfloor$（symmetric），这会让 output[t] 依赖 input[t-1], input[t], input[t+1]——**泄露未来**。

**Causal 3D conv** = 把 padding 全堆到时间左侧（过去方向），右侧（未来）不 pad：

$$\text{output}[t] = \sum_{\tau=0}^{k_t - 1} W[\tau] \cdot \text{input}[t - (k_t - 1) + \tau]$$

即 output 在时间 $t$ 只看 $[t - k_t + 1, t]$ 这一段历史。空间维仍 symmetric padding（图像问题，无方向）。

> ⚠️ **下采样的 causal 化** — 时间 stride > 1 的下采样层（如 $T \to T/2$）要保证窗口对齐。常见做法：让 stride 在时间维只看历史窗 `[t-1, t]`，输出 `t' = t//2`。Hunyuan-Video / Mochi 在 paper 里都画了这一层。

### 2.4 损失函数（与图像 VAE 同骨架）

$$\mathcal{L}_\text{VAE} = \mathcal{L}_\text{recon} + \lambda_\text{KL} \cdot \mathcal{L}_\text{KL} + \lambda_\text{LPIPS} \cdot \mathcal{L}_\text{LPIPS} + \lambda_\text{GAN} \cdot \mathcal{L}_\text{GAN}$$

视频 VAE 还需要 **temporal consistency loss**（如相邻帧重建差异 + 光流约束）防闪烁。Hunyuan-Video 用 GAN（PatchGAN 在 spatiotemporal patch 上）+ 3D 感知损失。

### 2.5 代码：Causal Conv3d 与编码器骨架

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    """
    时间维度因果 conv3d.
    输入 [B, C_in, T, H, W] -> 输出 [B, C_out, T, H, W]（stride=1 时形状不变）.
    时间 kernel 大小 k_t -> 左侧 pad (k_t-1)，右侧不 pad；空间维 symmetric pad.
    """
    def __init__(self, in_ch, out_ch, kernel=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1)):
        super().__init__()
        k_t, k_h, k_w = kernel
        d_t, d_h, d_w = dilation
        # 时间维 causal padding（全部堆到左侧）
        self.t_pad_left = (k_t - 1) * d_t
        # 空间维 symmetric padding
        self.h_pad = ((k_h - 1) * d_h) // 2
        self.w_pad = ((k_w - 1) * d_w) // 2
        # 注意 nn.Conv3d 自带 padding 是 symmetric, 这里手动 pad + 关闭 conv 自带 pad
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, dilation=dilation, padding=0)

    def forward(self, x):
        # x: [B, C, T, H, W]
        # F.pad order: (W_left, W_right, H_left, H_right, T_left, T_right)
        x = F.pad(x, (self.w_pad, self.w_pad,
                      self.h_pad, self.h_pad,
                      self.t_pad_left, 0))   # 时间右侧不 pad => 不看未来
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
    教学版 3D Causal VAE encoder. 实际 Hunyuan / CogVideoX 用更深的 stack + GAN 判别器.
    Downsample 用 stride=(2,2,2) 的 CausalConv3d (时间维仍 causal).
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
        # 期望 T, H, W 能整除 2^num_levels
        h = self.in_proj(x)
        for lvl in self.levels:
            h = lvl(h)
        h = F.silu(self.norm_out(h))
        return self.head(h).chunk(2, dim=1)                          # (mu, logvar)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
```

> 💡 **首帧 / chunk boundary 处理（面试加分）** — 因果 pad 让首帧时间维"看 0"——VAE 训练时随机扔掉时间起点的 $k_t{-}1$ 帧损失贡献，或者用 anchor padding（首帧复制填充）。Hunyuan-Video 在 paper 里讨论了 chunk-by-chunk 编码时 boundary frame 的一致性问题，工程上常通过 chunked overlapping 解决。

## §3 Spacetime Patches (Sora 视角)

### 3.1 一句话

像 ViT 把图像切 $p \times p$ patch 当 token，Sora 把 **视频 latent** 切 $p_t \times p_h \times p_w$ 的 **3D patch** 当 token。$N = (t/p_t) \cdot (h/p_h) \cdot (w/p_w)$ 个 token 喂给 Transformer。

公式上把一段 latent $z \in \mathbb{R}^{C \times t \times h \times w}$ 重排：

$$z \mapsto Z \in \mathbb{R}^{N \times (C \cdot p_t \cdot p_h \cdot p_w)} \xrightarrow{W_\text{proj}} Z' \in \mathbb{R}^{N \times D}$$

### 3.2 为什么 patch 化（而非每 voxel 一个 token）

- **token 数缩减**：$p_t \cdot p_h \cdot p_w$ 倍降低 $N$，attention 二次成本大幅下降。
- **局部归纳偏置**：每个 patch 内是固定 mlp 投影，类似 conv 的局部融合。
- **支持变形**：不同分辨率 / 时长直接产生不同 $N$，模型在 Transformer 内对 $N$ 不敏感。

### 3.3 Variable resolution / duration / aspect ratio（Sora 关键贡献）

Sora 不把每段视频 resize 到固定尺寸（如 $256 \times 256 \times 16$）。它把不同 $(T, H, W)$ 的视频都切成 patch 序列，**用 packing**（不同样本混到同一 batch 用 segment mask 分离）训练。

实现要点：

- **位置编码用相对 / 解耦** ：space 用 RoPE-2D，time 用 RoPE-1D；不绑死最大长度。
- **Attention mask**：同一 batch 内不同 sample 的 token 之间 mask 掉。
- **target aspect ratio token**：作为额外条件让模型知道当前生成的是 16:9 还是 9:16。

这一招让 Sora 可以训 / 生成任意比例的视频；后续 Wan-2.x、Hunyuan-Video 都借鉴。

### 3.4 代码：Spacetime patchify / unpatchify

```python
class SpacetimePatchify(nn.Module):
    """
    把 video latent [B, C, t, h, w] 切成 3D patch token [B, N, D].
    p_t/p_h/p_w 必须能整除 t/h/w（输入时保证）.
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
    """ Patchify 的逆操作: [B, N, D] + shape -> [B, C_out, t, h, w] """
    def __init__(self, embed_dim, out_ch, patch=(2, 2, 2)):
        super().__init__()
        self.p_t, self.p_h, self.p_w = patch
        self.out_ch = out_ch
        # 每个 token 解码回 (out_ch * p_t * p_h * p_w) 维 voxel block
        self.proj = nn.Linear(embed_dim, out_ch * self.p_t * self.p_h * self.p_w)

    def forward(self, x, grid):
        # x: [B, N, D], grid = (tt, hh, ww)
        B, N, D = x.shape
        tt, hh, ww = grid
        assert N == tt * hh * ww
        y = self.proj(x)                                 # [B, N, out_ch * p_t * p_h * p_w]
        y = y.view(B, tt, hh, ww, self.out_ch, self.p_t, self.p_h, self.p_w)
        # 把 patch 内坐标拼回原始空间：(tt, p_t) -> T,  (hh, p_h) -> H, ...
        y = y.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return y.view(B, self.out_ch, tt * self.p_t, hh * self.p_h, ww * self.p_w)
```

> ⚠️ **patch 大小如何选** — $p_t$ 通常 1-2（时间维已被 VAE 压过 4x，不能再压太多），$p_h = p_w$ 通常 2。开源模型常用 $p_t{=}1, p_h{=}p_w{=}2$（Mochi-1 / Hunyuan-Video，配合 4×8×8 latent VAE）。Sora 未公开实现细节，只描述了 "spacetime patches" 概念。

## §4 Spatiotemporal Attention 三大变体

### 4.1 复杂度对比（核心面试题）

定义 $S = h \cdot w / (p_h p_w)$（每帧 token 数），$T_t = t / p_t$（时间 token 数），$N = S \cdot T_t$。

| 类型 | 公式形式 | Time | Score memory (vanilla) | 用在哪 |
| --- | --- | --- | --- | --- |
| **Factorized 2+1D** | $\text{TempAttn}(\text{SpatialAttn}(x))$ | $O(T_t S^2 + S T_t^2) \cdot d$ | $O(T_t S^2 + S T_t^2)$ | Latte, OpenSora, AnimateDiff |
| **Full 3D ST** | $\text{Attn}_{\text{ST}}(x \in \mathbb{R}^{N \times D})$ | $O(N^2 d) = O(S^2 T_t^2 d)$ | $O(N^2)$ | Hunyuan-Video, Mochi, (Sora 推测) |
| **2+1D + 周期 full** | 大部分块 2+1D, 少量块 full | 介于两者 | 介于两者 | OpenSora-Plan, CogVideoX |
| **Window 3D / 稀疏** | 局部 3D window | $O(N \cdot w^3 d)$ ($w$ = window) | $O(N \cdot w^3)$ | 部分工程实践 |

> 💡 **核心对比** — Full 3D 是 spacetime token 全交互，**表达力最强**；Factorized 是先空间后时间两次 attn，**强假设 space ⊗ time 可分**，但成本低 $\min(S, T_t)$ 量级倍。Sora 官方未披露具体 attention 类型，社区基于其 "spacetime patches" 描述普遍推测是 full 3D / 大序列统一 attention（Hunyuan-Video / Mochi 已采用此设计）。

### 4.2 Factorized 2+1D Attention（Latte / OpenSora）

```

Input [B, T_t, S, D]
   │
   │  rearrange -> [B*T_t, S, D]
   ↓
SpatialAttn (每帧内 self-attention)    →    [B*T_t, S, D]
   │
   │  rearrange -> [B*S, T_t, D]
   ↓
TempAttn (跨帧 self-attention)         →    [B*S, T_t, D]
   │
   │  rearrange -> [B, T_t, S, D]
   ↓
FFN
```

强假设：每帧的 spatial relations 与跨帧的 temporal relations **可分离**。这是简化但工程上非常好用——多数情况下视频里相邻帧大体相似（temporal "光流" 信息可用 1D attn 抓到），帧内 patch 关系类似图像。

### 4.3 Full 3D Spatiotemporal Attention（Sora / Hunyuan / Mochi）

直接把 $N = S \cdot T_t$ 个 token 串成一维序列，标准 self-attention：

$$\text{Attn}(Z) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V, \quad Z \in \mathbb{R}^{N \times D}$$

复杂度 $O(N^2 d)$。$N$ 在长视频下大：例如 720p × 5s × 24fps 经 $4{\times}8{\times}8$ VAE 后 latent shape 约为 $30 \times 90 \times 160$，再做 (1,2,2) patchify 得 $N = 30 \cdot 45 \cdot 80 = 108000$ token，$N^2 \approx 1.17 \times 10^{10}$。**所以工程上必须用 FlashAttention v2/v3 + Tensor Parallelism + Sequence Parallelism**。

### 4.4 RoPE-3D（位置编码）

视频 token 需要同时编码 (time, height, width) 三维位置。**做法 1（解耦）**：把 $d$ 维向量切三段，分别对 (t, h, w) 做 RoPE。**做法 2（共享）**：每段都 stack 三种频率的旋转（Hunyuan-Video）。

CogVideoX 用 **3D RoPE**：把 $d$ 维 query/key 切成三段 $q = [q^{(t)} \,|\, q^{(h)} \,|\, q^{(w)}]$（disjoint subspace），三段分别用 (t, h, w) 对应的频率做 1D RoPE 后 concat：

$$\text{RoPE}_{3D}(q,\, t, h, w) = \big[\, R_t(\theta_t) q^{(t)} \,\big|\, R_h(\theta_h) q^{(h)} \,\big|\, R_w(\theta_w) q^{(w)} \,\big]$$

实操即三段独立 1D RoPE 拼接（block-diagonal 旋转），自然得到相对时空位置编码。

### 4.5 代码：Factorized 2+1D 实现

```python
class AxisAttention(nn.Module):
    """
    沿一个指定轴 (spatial 'S' 或 temporal 'T') 做 self-attention.
    输入 [B, T, S, D] -> 输出 [B, T, S, D].
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
        else:   # "T": 把 spatial 维 collapse 进 batch, 时间维当 seq
            x_ = x.permute(0, 2, 1, 3).reshape(B * S, T, D); L_seq = T; merge_batch = S
        qkv = self.qkv(x_).reshape(-1, L_seq, 3, self.heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # [B', H, L, d_k]
        # PyTorch SDPA 自动选 FlashAttention v2/v3
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

### 4.6 代码：Full 3D Spatiotemporal Attention（with FlashAttention via SDPA）

```python
class Full3DSpatiotemporalAttention(nn.Module):
    """
    把 [B, T, S, D] 当作 [B, N, D] 做 self-attention（N = T*S）.
    使用 PyTorch SDPA, 底层在 H100/A100 上会自动调度 FlashAttention v2/v3.
    与 MM-DiT 结合时, text/video token 直接 concat 到这一序列做 attention(见 §5.4).
    """
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, dim, bias=False)

    def forward(self, x, attn_bias=None):
        # x: [B, N, D] (调用前先把 T*S flatten)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                          # [B, N, H, d_k]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))      # [B, H, N, d_k]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)
```

## §5 MM-DiT for Video（Hunyuan-Video / Mochi-1 思路）

### 5.1 从 SD3 MM-DiT 推广到视频

SD3 (Esser et al. 2024) 引入 **MM-DiT**：把 text token 和 image token concat 到同一序列做 self-attention，**两个 stream 各自有独立的 QKV / FFN / AdaLN 参数**，但 attention 是同一矩阵：

$$Z = [Z_\text{text} \,|\, Z_\text{img}], \quad \text{Attn}(Z) \text{ shared, FFN per-stream}$$

视频化 = 把 image stream 换成 video stream（spacetime token），text stream 不变。

### 5.2 Hunyuan-Video 的 dual-stream → single-stream 设计

Hunyuan-Video tech report 把 DiT 分成两段：

- **Dual-stream blocks**（前 N 层）：text token 与 video token 在 attention 共享，但 FFN / AdaLN 独立——让两个模态先各自精炼表示。
- **Single-stream blocks**（后 M 层）：text token 和 video token 完全 share weight，相当于普通 Transformer 处理统一序列。

这与 Mochi-1 的 **AsymmDiT** 类似（Mochi 让 video stream 比 text stream 宽 4 倍参数，因为视频信息远多于文本）。

### 5.3 文本编码器（一览）

| 模型 | Text encoder | 备注 |
| --- | --- | --- |
| Hunyuan-Video | CLIP-L + MLLM (LLaVA 风) | MLLM 提供长 prompt + 物体关系 |
| Mochi-1 | T5-XXL | 与 SD3 同 |
| Wan 2.1 / 2.2 | UMT5 (multilingual) | 支持中英双语 |
| CogVideoX | T5-XXL | — |
| Sora | 未公开 | OpenAI 报告只提到使用 re-captioning 流程 |

> 💡 **prompt rewriter（Hunyuan-Video / Sora 报告）** — 训练数据 caption 通常长且细致，用户 short prompt 与训练分布有 gap。Sora 官方报告提到训练前对视频做 dense re-captioning；Hunyuan-Video 在推理时用 LLM 改写短 prompt 成 dense caption 再喂模型。面试常问"为什么需要 prompt rewriter"——答：训练/推理 caption 分布对齐 + 用户原 prompt 太短。

### 5.4 代码：MM-DiT video block

```python
class MMDiTVideoBlock(nn.Module):
    """
    Dual-stream MM-DiT block (Hunyuan-Video / Mochi-1 / SD3 style).
    Text token 与 video token 共享 self-attention 矩阵, 但 QKV / FFN / AdaLN 各自参数.
    """
    def __init__(self, dim, heads, head_dim, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # 各自的 QKV
        self.qkv_text = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.qkv_vid  = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.out_text = nn.Linear(heads * head_dim, dim, bias=False)
        self.out_vid  = nn.Linear(heads * head_dim, dim, bias=False)

        # 各自的 FFN
        hidden = int(dim * mlp_ratio)
        self.mlp_text = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mlp_vid  = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        # 各自的 AdaLN(由 timestep + pooled text condition 调制)
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

        # Pre-norm + modulate (各 stream 独立)
        t = self._modulate(self.norm_text(text), st_a, sc_t_a)
        v_ = self._modulate(self.norm_vid(vid),  sv_a, sc_v_a)

        # 各 stream 的 QKV (独立参数)
        qt, kt, vt = self._proj_qkv(t, self.qkv_text)
        qv, kv, vv = self._proj_qkv(v_, self.qkv_vid)

        # Concat 两 stream 沿 seq 维一起 self-attention
        q = torch.cat([qt, qv], dim=2)                       # [B, H, L_t + N_v, d_k]
        k = torch.cat([kt, kv], dim=2)
        v = torch.cat([vt, vv], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)        # [B, H, L_t + N_v, d_k]

        L_t = text.shape[1]
        out_t, out_v = out[:, :, :L_t], out[:, :, L_t:]
        out_t = out_t.transpose(1, 2).reshape(*text.shape[:2], -1)
        out_v = out_v.transpose(1, 2).reshape(*vid.shape[:2], -1)

        # Out projection (各 stream)
        text = text + gt_a.unsqueeze(1) * self.out_text(out_t)
        vid  = vid  + gv_a.unsqueeze(1) * self.out_vid(out_v)

        # FFN (各 stream)
        text = text + gt_m.unsqueeze(1) * self.mlp_text(
            self._modulate(self.norm_ff_text(text), st_m, sc_t_m))
        vid = vid + gv_m.unsqueeze(1) * self.mlp_vid(
            self._modulate(self.norm_ff_vid(vid), sv_m, sc_v_m))
        return text, vid
```

### 5.5 训练目标

主流 = **v-prediction 或 rectified flow vector**（与 SD3 一致）。给定 video latent $x_1$、噪声 $x_0$、随机 $\tau \in [0, 1]$（logit-normal 分布更常用）：

$$x_\tau = (1 - \tau) x_0 + \tau\, x_1, \quad u_\tau = x_1 - x_0$$

$$\mathcal{L} = \mathbb{E}_{\tau, x_0, x_1, c} \, \| v_\theta(\tau, x_\tau, c) - u_\tau \|^2$$

其中 $c$ 是 text condition（+ optional 图像 / 视频条件）。Hunyuan-Video / Mochi / Wan 都用 RF（与 image 端 SD3 同 paradigm）；Sora 未公开但社区推测也是 RF / v-pred 家族。

## §6 Image-to-Video (I2V) — 主流三种条件注入

### 6.1 First-Frame Channel Concat（最简、最常用）

把参考图 $I_\text{ref}$ 编码成 latent $z_\text{ref}^{(0)} \in \mathbb{R}^{C \times 1 \times h \times w}$（VAE 在 $T{=}1$ 下退化为 2D 编码），沿 latent 时间维 $t$（不是原始帧 $T$）做 zero-padded broadcast，并和 noisy latent 沿 channel 维拼接：

$$\tilde{z}_\tau = \text{concat}_C\!\left(z_\tau, \, \tilde{z}_\text{ref}, \, m\right) \in \mathbb{R}^{(2C+1) \times t \times h \times w}$$

其中 $\tilde{z}_\text{ref}[:, t', :, :] = z_\text{ref}^{(0)}$ 若 $t'$ 是参考位（一般为 $t' = 0$），其余位为 0；$m \in \{0, 1\}^{1 \times t \times h \times w}$ 是同 shape 的 mask channel（参考位为 1，待生成位为 0）。模型第一层 conv 的 in_channels 从 $C$ 改成 $2C + 1$。

**优点**：实现简单，与 T2V 主干完全兼容；改一层 conv 就行。**缺点**：参考信息混在 channel 里，模型要靠 conv 自己提取；远距离帧的引导能力弱。

SVD (Stable Video Diffusion, Blattmann 2023) / DynamiCrafter / I2VGen-XL 都用这套。

### 6.2 Cross-Attention 注入（细粒度控制）

把 $z_\text{ref}$ patchify 成 token，作为额外 K/V 注入到 video stream 的 attention：

$$\text{CrossAttn}: \quad Q = z_\text{video tokens}, \, K, V = [z_\text{text}, z_\text{ref tokens}]$$

或更复杂——为 ref image 增加独立 cross-attention 层。AnimateDiff 风格的 LoRA tuning 也基本用这条路。

### 6.3 AnimateDiff 风格 plug-in motion module

> ✅ **AnimateDiff (Guo et al. arXiv 2023-07, ICLR 2024)** — 冻结一个 T2I (Stable Diffusion)，**只插入并训练 temporal motion module**（temporal self-attention），就把 T2I 变 T2V。优势：复用 T2I 生态（LoRA / ControlNet）；劣势：质量受 base T2I 限制。Motion module 通常插在每个 spatial block 后：`[B,T,C,H,W] -> rearrange [B*H*W, T, C] -> temporal self-attn -> back`。

### 6.4 代码：first-frame concat I2V

```python
class FirstFrameI2VAdapter(nn.Module):
    """
    把 ref image latent 沿时间 broadcast + 沿 channel concat 到 noisy latent.
    用法: 替换 DiT 的 patchify 输入投影, in_ch 从 C 改为 (2C + 1).
    """
    def __init__(self, latent_ch=16, embed_dim=1024, patch=(1, 2, 2)):
        super().__init__()
        self.patch = patch
        # +1 是 mask channel
        self.proj = nn.Conv3d(
            in_channels=2 * latent_ch + 1,
            out_channels=embed_dim,
            kernel_size=patch, stride=patch,
        )

    def forward(self, noisy_latent, ref_latent, ref_mask):
        """
        noisy_latent: [B, C, t, h, w]   - 加噪 latent
        ref_latent:   [B, C, t, h, w]   - 参考帧 latent (待生成位为 0)
        ref_mask:     [B, 1, t, h, w]   - 1 = 该位置 (在 t 维) 是参考帧, 0 = 待生成
        """
        x = torch.cat([noisy_latent, ref_latent, ref_mask], dim=1)  # [B, 2C+1, t, h, w]
        x = self.proj(x)                                            # [B, D, t', h', w']
        B, D, tt, hh, ww = x.shape
        return x.flatten(2).transpose(1, 2), (tt, hh, ww)           # [B, N, D]
```

> 💡 **build ref_latent + mask 的伪代码** — 给定 video latent `z_full` 和参考帧索引 `ref_indices`（如 `[0]` 表示 first-frame I2V），把 `z_full[:, :, idx]` 拷贝到 `ref_latent` 同位置、其余位 0，并在 `ref_mask` 对应位置赋 1。具体实现见 §A.2 `video_gen_forward` 内部。

### 6.5 训练时的"noise-only on未来帧" trick

I2V 训练时给参考帧 latent 不加噪（或加极小噪声），待生成帧加正常 $\tau$ 噪声。loss 只在待生成帧位置计算。这点在 SVD / Hunyuan-Video I2V variant 中都有提到。

## §7 长视频生成（开放问题）

### 7.1 三大主流路线

| 路线 | 方法 | 代表 | 缺点 |
| --- | --- | --- | --- |
| **Keyframe + interpolation** | 先生成 $K$ 个关键帧（远距），再每段插值出中间帧 | NUWA-XL, Movie Gen 部分 pipeline | 关键帧 transition 可能不自然 |
| **Hierarchical (coarse → fine)** | 先生成 low-fps 低分辨率版本，再 cascade 超分 + 帧插 | Phenaki, Imagen Video, Sora 部分 | 多阶段累积误差 |
| **Autoregressive chunks** | 每次生成 $T$ 帧 chunk，下一 chunk 以前 chunk 末尾为条件 | Mochi-1 部分模式, StreamingT2V | 误差累积 → drift, "loop" |

### 7.2 Sora 的 "duration as condition"

Sora 把 video duration 当成 conditioning（不是固定 $T$），训练时把不同时长视频都见过；推理时给定目标时长，生成对应长度 patch 序列。这在公开 report 中只是简短提到。

### 7.3 Autoregressive Video Generation 兼容性（Causal VAE 派上用场）

由于 3D **Causal** VAE encoder 不看未来，自然支持 chunk-wise 自回归：encoder 可以一段一段处理。但 DiT 部分要做到自回归还需在 token 维度也加 causal——这是研究方向（"VAR for Video" 等）。Hunyuan-Video / Wan 现版本仍是 non-AR diffusion。

### 7.4 简单的 chunk 自回归伪代码

```python
@torch.no_grad()
def chunk_autoregressive(rf_sample_fn, vae, text, num_chunks, chunk_t_lat=4, overlap=1,
                         latent_shape=(16, 60, 90)):
    """
    Chunk-wise 自回归生成长视频伪代码框架.
    rf_sample_fn(z_init, text, ref_full, ref_mask) -> z_out
        ——使用 §A.3 的 video_gen_sample 风格, 但接受外部 z_init 与 ref clamp.
    latent_shape = (C, h_lat, w_lat); 每段 latent 时间长度 = chunk_t_lat.
    """
    chunks = []
    ref_latent = None
    C, hl, wl = latent_shape
    device = next(vae.parameters()).device
    for c in range(num_chunks):
        # 初始 noise
        z_init = torch.randn(1, C, chunk_t_lat, hl, wl, device=device)
        # 上一段末尾 overlap 帧当 ref
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

    # 拼接 chunks(去重 overlap), VAE decode
    z_full = torch.cat([chunks[0]] + [c[:, :, overlap:] for c in chunks[1:]], dim=2)
    return vae.decoder(z_full)
```

> ⚠️ **drift 问题** — 即使有 overlap，AR 模式下后期 chunk 容易出现内容 drift（人物消失、色调变化）。Movie Gen / Hunyuan 采用全局 attention 一次生成短视频（5-10s）+ 后处理超分时长 trick 而非纯 AR。

## §8 控制 / 编辑 / 角色一致性（简表）

| 任务 | 代表方法 | 注入手段 |
| --- | --- | --- |
| Camera 控制 | **MotionCtrl** (Wang et al. 2023) | CMCM: 相机轨迹 $R, T$ 序列 → temporal embedding |
| Object trajectory 控制 | **MotionCtrl** OMCM 模块 | 2D 轨迹点 → spatial heatmap 当 condition |
| 人物动画 | **AnimateAnyone** (Hu et al. CVPR 2024) | ReferenceNet (sibling SD 主干) + Pose Guider (OpenPose/DWPose) + temporal layers |
| 角色一致 | IP-Adapter / DreamBooth-style | ID embedding 经 cross-attn 注入 |
| 局部编辑 | Mask-guided generation | 角色 mask + ref image 一同注入 |

## §9 评测：VBench / FVD / CLIPSim-V

> ✅ **VBench (Huang et al. CVPR 2024) — 现 SoTA 标准** — 16 个细粒度维度，分两大类。

- **Video Quality (7 维)**：Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality
- **Video-Condition Consistency (9 维)**：Object Class, Multiple Objects, Human Action, Color, Spatial Relationship, Scene, Appearance Style, Temporal Style, Overall Consistency

每维有定制 prompt 集 + 自动评测器（GroundingDINO / DOVER / RAFT / CLIP 等），最后给加权综合分。Hunyuan-Video / Mochi / Movie Gen / Sora 等都在 VBench 上自报或被独立测。

**FVD (Unterthiner et al. 2018)** — FID 推广到视频。生成 / 真实视频分别过一个预训练 video classifier（I3D / VideoMAE / S3D）拿中间层 feature，计算两组分布的 Fréchet 距离：

$$\text{FVD} = \|\mu_g - \mu_r\|^2 + \text{Tr}\!\left(\Sigma_g + \Sigma_r - 2\sqrt{\Sigma_g \Sigma_r}\right)$$

低 FVD = 生成分布更接近真实；缺点：依赖 classifier domain，不反映 text-video alignment。

**CLIPSim-V** — 每帧取 CLIP image embedding 与 prompt CLIP text embedding 计算余弦：$\frac{1}{T} \sum_t \cos(\text{CLIP-I}(f_t), \text{CLIP-T}(\text{prompt}))$。评 text-video 对齐，但忽略时序连贯。

**实务**：Hunyuan-Video tech report 同时报 VBench + 人评（user study 1-5 likert）；Sora / Veo demos 只给定性可视化。面试碰到"你怎么评测"——说 VBench + 人评 + 在 UCF-101 / MSR-VTT 上算 FVD。

## §10 复杂度与显存

### 10.1 Token 数与 attention cost

| 视频规格 | VAE 下采样 | VAE 后 latent shape | Patchify (1,2,2) 后 N | Full-3D Attn $N^2$ |
| --- | --- | --- | --- | --- |
| $256{\times}256{\times}16$ (2s, 8fps) | $4{\times}8{\times}8$ | $4 \times 32 \times 32$ | $4 \cdot 16 \cdot 16 = 1024$ | $\approx 10^6$ |
| $720p{\times}48$ (2s, 24fps, latent $12{\times}90{\times}160$) | $4{\times}8{\times}8$ | $12 \times 90 \times 160$ | $12 \cdot 45 \cdot 80 = 43200$ | $\approx 1.9 \times 10^9$ |
| $1080p{\times}120$ (5s, 24fps, latent $30{\times}136{\times}240$) | $4{\times}8{\times}8$ | $30 \times 136 \times 240$ | $30 \cdot 68 \cdot 120 = 244800$ | $\approx 6.0 \times 10^{10}$ |

很快爆。面试常问 "长视频/高分辨率 attention 瓶颈"——答：$O(N^2)$ 二次成本 + FlashAttention 也减不了 token 数本质；解法是 factorized / window / cascaded 多阶段。

### 10.2 显存与 FLOPs 要点

- **Score 矩阵显存**：vanilla 是 $O(L^2)$；FlashAttention 降到 $O(L)$ activation
- **KV cache**：纯 diffusion 无 AR step，训练时无 KV cache 概念；activation checkpointing + ZeRO-3 是必备
- **3D Causal VAE 推理**：可分 chunk，显存 $O(\text{chunk}_T \cdot h \cdot w \cdot C)$
- **训练 FLOPs**：13B 模型 per-batch attention FLOPs ≈ $4 B N^2 d$ × layers（B = batch size）；Hunyuan-Video 全模型 ≈ 几百到上千 PFLOP / sample 量级，用数千 H100 数月级别训出

## §11 与其他生成模型的关系

### 11.1 Image diffusion vs Video diffusion

| 维度 | Image (SD3 / SDXL / FLUX) | Video (Hunyuan / Mochi / Wan) |
| --- | --- | --- |
| VAE | 2D VAE, $\downarrow 8\times$ | 3D Causal VAE, $\downarrow 4\times8\times8$ |
| Token 数 | $\sim 10^3$ (1024² @ 16²) | $\sim 10^4 - 10^5$ |
| Attention | Self + Cross | ST (factorized / full 3D) + Cross/MM-DiT |
| 训练 token 量 | 数十亿 image-text pair | 数千万 video-text pair（视频数据稀缺） |
| 评测 | FID, CLIPScore | VBench, FVD, CLIPSim-V |
| 主要难点 | 美学 / prompt 跟随 | 时序一致性 / 物理合理性 / 长时长 |

### 11.2 Video LLM vs Video Generation vs AR Video

- **Video LLM** (VideoChat / Video-LLaVA / NaVit)：输入视频，输出文本（理解方向）；CLIP/ViViT/VideoMAE feature + LLM
- **Diffusion Video Gen**（本文主线）：质量高，长视频靠 chunk / hierarchical
- **AR Video** (Cosmos-Predict, VAR-Video 系列)：视频 token 化 (VQ-VAE / FSQ) + LM 风格生成。优势：天然 streaming + 长视频；劣势：质量目前不如 diffusion
- **Unified** (Show-o / Emu3 等 2025 苗头)：同时理解 + 生成；视频版本尚未爆发
- 2024-2025 SoTA 仍以 diffusion 占主流

## §12 25 高频面试题

按难度分 3 档：L1 必会（任何 MLE 岗）、L2 进阶（research-oriented）、L3 顶级 lab（diffusion / video specialist）。每题点开看答案要点 + 易踩坑。

### L1 必会题（基础 — 任何视频生成相关岗）

<details>

<summary>Q1. 当下主流视频生成模型的整体 pipeline 是什么？</summary>

- **3D Causal VAE** 把视频从 $H{\times}W{\times}T$ 压到 $h{\times}w{\times}t$ latent
- **Latent DiT / MM-DiT** 在 latent 空间做扩散 / flow matching
- **Text encoder**（T5 / CLIP / MLLM）提供 condition
- **VAE Decoder** 把 latent decode 回视频
- 推理用 ODE/SDE sampler（Euler / Heun / DDIM）从噪声积分到数据

只说 "diffusion + UNet"，忘了 latent / 3D / VAE 三件套；或说 pixel-space 跑 attention（早就不可行）。

</details>

<details>

<summary>Q2. 3D Causal VAE 的 "Causal" 是什么意思？为什么必须 causal？</summary>

- 时间维度 conv padding 全堆到**左侧（过去）**，右侧不 pad，**不让 current frame 看未来**
- 三大收益：(1) **图像和视频可共享 latent 空间**（$T{=}1$ 时退化成 2D，图像也能进 encoder），便于 image+video 联合训练；(2) **流式 / 自回归推理**，长视频可 chunk-wise 处理；(3) 训练时 image 视为 $T{=}1$ 视频，数据利用率高
- Hunyuan-Video / Wan / Mochi / CogVideoX 都用 causal 3D VAE

说 "causal 就是 mask 掉未来"——这是表面，关键是为什么有那 3 个工程收益。

</details>

<details>

<summary>Q3. Sora 的 "spacetime patches" 解决了什么问题？</summary>

- 把视频 latent 切 3D patch $p_t \times p_h \times p_w$，每个 patch 是一个 token
- **支持变 resolution / 变 duration / 变 aspect ratio**——不同 shape 视频直接打包到同 batch（用 attention mask 分离）
- 类似 ViT 但加了时间维；解耦了网络结构与具体输入形状

只说"切 patch"，没说核心收益（变形状训练 / 推理）。

</details>

<details>

<summary>Q4. Factorized 2+1D attention 和 Full 3D attention 的复杂度差多少？</summary>

- 记每帧 token 数 $S$、时间 token 数 $T_t$、总 token 数 $N = S T_t$
- **Factorized 2+1D**：spatial $O(T_t S^2)$ + temporal $O(S T_t^2)$
- **Full 3D**：$O(N^2) = O(S^2 T_t^2)$
- Full 3D 比 Factorized 多 $\min(S, T_t)$ 倍
- 但 Full 3D **表达力更强**（允许 spatial + temporal 联合 pattern）

只比 Full 3D 不算时间项；或忽略两者本质差是"是否假设 space ⊗ time 可分"。

</details>

<details>

<summary>Q5. MM-DiT for video 是怎么注入文本的？</summary>

- Text token 和 video token **拼接到同一序列**做 self-attention
- 每个 stream 各自的 QKV / FFN / AdaLN 参数（dual-stream），但 attention 矩阵共享
- 对比传统 cross-attention（Q from video, KV from text）：MM-DiT 让两个模态在 token 级别**对等**交互
- Hunyuan-Video / Mochi 用此架构（video 端）；SD3 (Esser et al. 2024) 是 image 端的原始 MM-DiT

说 video 主干 + cross-attn from text——这是 SDXL 风格，MM-DiT 已替代它。

</details>

<details>

<summary>Q6. I2V (Image-to-Video) 最简单的实现是怎样的？</summary>

- 参考图 $I_\text{ref}$ 编码为 latent $z_\text{ref}$
- 沿时间维 broadcast，沿 channel 维与 noisy latent 拼接：input channel 从 $C$ 改成 $2C{+}1$（加一个 mask channel 标注哪些帧是 ref）
- 模型其他部分不变；改 patchify 第一层 conv
- 训练时参考帧不加噪 / 加极小噪，loss 只在待生成帧上算

只说"再喂一张图"，没说 concat 在哪 / mask 怎么处理。

</details>

<details>

<summary>Q7. AnimateDiff 的核心思路？</summary>

- **冻结**一个 T2I 模型（Stable Diffusion）
- **插入 + 仅训 temporal motion module**（temporal self-attention），插在每个 spatial block 后
- 优势：直接复用所有 T2I 生态（LoRA, ControlNet, custom checkpoint）
- 劣势：质量受限于 base T2I 模型；难做长视频

说"用 transformer 替换 UNet"——错，AnimateDiff 是 plug-in，不替换主干。

</details>

<details>

<summary>Q8. VBench 评测什么？与 FVD 有什么区别？</summary>

- **VBench** (Huang CVPR 2024)：16 个**细粒度维度**（如 Subject Consistency, Motion Smoothness, Object Class, Color, ...），每维用专门 detector / classifier 评分；现 SoTA 视频生成标准
- **FVD** (Unterthiner 2018)：FID 推广到视频；用 I3D / VideoMAE feature 算两组分布的 Fréchet 距离；单一数字
- VBench 更解释性强，能定位哪个维度差；FVD 单一但黑盒
- 实务：两者一起报 + 人评

只听过 FVD 不知道 VBench——2024 后这是大缺口。

</details>

<details>

<summary>Q9. Sora / Hunyuan-Video / Mochi-1 / Kling 哪些开源？</summary>

- **开源**：Hunyuan-Video (13B, 2024-12), Mochi-1 (10B, 2024-10), CogVideoX (2B/5B, arXiv 2024-08), OpenSora / OpenSora-Plan, LTX-Video (2B, 2024-11), Wan 2.1/2.2 (14B, 2025), SVD (2023-11)
- **闭源**：Sora (2024-02), Veo / Veo 2 (Google), Kling (Kuaishou 2024-06), Movie Gen (Meta, 30B)
- 国内开源主力：Hunyuan / Wan / CogVideoX

只记得 Sora 和 SVD——闭源开源都得知道时间线。

</details>

<details>

<summary>Q10. Text encoder 选择：T5 vs CLIP vs MLLM？</summary>

- **CLIP-L / G**：与图像 latent 对齐好（image-text 联合训练），短文本工作得好，但长 prompt 跟随差
- **T5-XXL**：sequence-to-sequence 模型，**长 prompt 跟随更好**；SD3 / Mochi / CogVideoX 都用
- **MLLM**（LLaVA-like）：理解 prompt 内物体关系强；Hunyuan-Video 用 CLIP-L + MLLM 双编码器
- 工程上长 prompt 推理时还会用 **prompt rewriter**（LLM 改写）补齐 train/test caption 分布差

只说 CLIP——2024 后基本被 T5 / MLLM 替代或并用。

</details>

### L2 进阶题（research-oriented / 中级岗）

<details>

<summary>Q11. 推导一下 3D Causal VAE 在 $T{=}1$ 时为什么能"行为上"等价于 2D conv？</summary>

- Causal 3D conv 在时间维 kernel size $k_t$，左 pad $k_t - 1$，右 pad 0；stride=1。
- 输入 shape $[B, C, 1, H, W]$（$T{=}1$ 单帧）；经左 pad $k_t - 1$ 后，时间维长度变为 $1 + (k_t - 1) = k_t$，前 $k_t - 1$ 个时间位是 padding 0，最后 1 个是真实 frame。
- Output 时间长度 = $(k_t + 0 - k_t)/1 + 1 = 1$（用 conv 输出长度公式 $(L_\text{in} + p - k) / s + 1$，其中 $L_\text{in}=1$、$p=k_t-1$、$k=k_t$、$s=1$）。所以**只产生一个时间位输出**。
- 设时间维权重序列 $W_0, W_1, \dots, W_{k_t-1}$，对该唯一输出位的计算是 $\sum_{\tau=0}^{k_t-1} W_\tau \star x_\tau$。但只有 $\tau = k_t - 1$ 对应真实 frame，其余 $x_\tau$ 是 padding 0——**只有 $W_{k_t - 1}$ 这一片时间权重对输出有贡献**。
- **结论**：$T{=}1$ 时整个 causal 3D conv 退化为一个**有效的** 2D conv（kernel = $W_{k_t - 1}[:, :, :, :]$）；其余时间片权重对 image batch "看不到"任何真实输入，相当于在 image path 上不被激活。

> 工程上：训练时 image 用 $T{=}1$ batch 走 "2D conv 子集" 路径，无需额外 2D head；这是 Hunyuan-Video / Mochi / CogVideoX 等同时训 image + video 的前提。

把"等价于 2D conv"当成权重物理合并——其实是 padding-0 让其他 time slice 权重对 image 输入不起作用。

</details>

<details>

<summary>Q12. Full 3D attention 与 factorized 2+1D 的表达力差异？(L2/L3 之间)</summary>

- Factorized：先 $\text{Attn}_S$ 把空间 token 互相 attend (在每帧内)，再 $\text{Attn}_T$ 在时间维做 attention
- **假设**：时空交互是 separable——任意两个时空位置 $(t_1, s_1)$ 与 $(t_2, s_2)$ 的关系，可以分解为"先 $s_1 \leftrightarrow s_2$ 在 $t_1$"再"$t_1 \leftrightarrow t_2$ 在 $s_2$"两步
- Full 3D 不假设可分：任意两 token 直接 attention；可学**斜向时空 pattern**（如 diagonal motion）
- **数学上**：factorized 是 full 3D 的一个**严格子集**（参数化受限）
- 实测：full 3D 在 fast motion / 复杂时空 pattern 上更好

只说"full 3D 更准确"，不知具体能学到的 pattern 多在哪里（diagonal / non-separable motion）。

</details>

<details>

<summary>Q13. 为什么 video VAE 主流停在 $\downarrow 4{\sim}8 \times 8 \times 8$ 范围？</summary>

- 体素级压缩比 $= 4 \cdot 8 \cdot 8 = 256$（$1{:}256$），$C=16$ latent channel 后净比约 $1{:}48$
- **更激进（如 LTX-Video 报告的 $1{:}8192$ 总压缩）→ token 数变少但重建质量下降**（细节 / 锐度损失）
- DiT 主干 attention $O(N^2)$，所以激进压缩工程友好；但 lost detail 难以恢复
- 时间下采样 $4{\times}$ 是经验上 motion 平滑性可接受的上限——更高时间压缩在快动作场景会出抖动
- 各家具体配置不同：CogVideoX 用 $4{\times}8{\times}8$；Hunyuan-Video 用 $4{\times}8{\times}8$；Mochi-1 在时间维更激进（约 $6{\times}8{\times}8$，见 Mochi blog）；LTX-Video 走极端高压缩 + 实时
- 即使主架构相同，VAE 选择仍是 active design choice

把所有模型都归到 $4{\times}8{\times}8$ 同一档——Mochi-1 / LTX-Video 走的是更激进时间压缩。

</details>

<details>

<summary>Q14. logit-normal $t$ 采样在 video FM 训练中为什么有用？</summary>

- SD3 (Esser 2024) 发现 $t \sim \mathcal{U}[0,1]$ 不最优：中间区域 $t \approx 0.5$ 的 noise/signal ratio 最难学
- 改成 $t = \sigma(\tau), \tau \sim \mathcal{N}(m, s^2)$，让 $t$ 集中在 0.5 附近
- 视频中也观察到同样现象（Hunyuan / Mochi 都默认 logit-normal）
- 直觉：远离 $t=0$ 和 $t=1$ 时模型需要在 mixed noise/signal 上学，正是 hardest 区域

说"loss 加权"——logit-normal 不是 loss reweighting 而是改 $t$ 采样分布。

</details>

<details>

<summary>Q15. RoPE-3D 怎么编码 (time, height, width)？</summary>

- 把 $d$ 维 query/key 切成 3 段 $q = [q^{(t)} \,|\, q^{(h)} \,|\, q^{(w)}]$（disjoint subspace），每段对应 (t, h, w)
- 每段用 1D RoPE（频率 $\theta_d = 1 / 10000^{2k/d}$ 类似 Transformer）
- 结果是 block-diagonal 旋转 concat：$\text{RoPE}_{3D}(q) = [R_t(\theta_t) q^{(t)} \,|\, R_h(\theta_h) q^{(h)} \,|\, R_w(\theta_w) q^{(w)}]$
- 对每个 head 独立做；attention 后自然得到**相对时空位置**信息
- CogVideoX / Hunyuan-Video 都用类似方案

把它当成 absolute embedding——RoPE 本质是相对位置编码。

</details>

<details>

<summary>Q16. Stable Video Diffusion (SVD) 与 AnimateDiff 的区别？</summary>

- **SVD** (Blattmann 2023)：finetune 整个 SD2.1 UNet，加 temporal layers；I2V 主，14 帧 / 25 帧两版；新模型整体训练
- **AnimateDiff** (Guo 2024 ICLR)：**冻结 SD-T2I**，只训 plug-in temporal motion module；T2V 主
- SVD 质量更好但失去 SD T2I 生态；AnimateDiff 牺牲质量换 plug-and-play
- 都是 2023 的早期方法，2024 被 DiT-based 大模型超越

只说"两者都是 video diffusion"——区别是 finetune vs plug-in。

</details>

<details>

<summary>Q17. DiT 用 AdaLN 与 cross-attn 的对比？为什么 DiT-Video 也偏好 AdaLN？</summary>

- **AdaLN**（DiT 默认）：把 condition pool 成单向量 $c$，预测 LayerNorm 的 scale + shift + gate
- **Cross-attn**：condition token 作 K/V, video token 作 Q
- AdaLN 简单 / 计算少 / 训练稳；但**全局 condition**——每 token 看到同一个 modulation
- Cross-attn 允许 token-level 选择性看 condition；token 多时贵
- MM-DiT（SD3 / Hunyuan / Mochi）实际把两者**结合**：MM 部分 share attention，AdaLN 调制 norm
- 单纯 video DiT 用 AdaLN 起家是 Latte / OpenSora 思路；后续被 MM-DiT 取代

把 AdaLN 当过时——MM-DiT 仍内嵌 AdaLN modulation。

</details>

<details>

<summary>Q18. CFG 在视频生成里的特殊点？</summary>

- 与 image 完全同公式：$v_\text{CFG} = v_\theta(\emptyset) + s \cdot (v_\theta(c) - v_\theta(\emptyset))$
- **每帧都做 CFG**，所以两倍 forward 成本 $\times$ T 帧
- **guidance scale 通常 5-7.5**（与 SDXL / SD3 接近）
- 视频中过大 $s$ 容易导致**帧间闪烁 / 颜色饱和** —— 比图像更敏感
- 部分实现用 **temporal-aware CFG**（不同时间步 / 不同帧用不同 $s$）

说 "CFG 在视频里和图像一样"——通常对，但闪烁更敏感是 video specific。

</details>

<details>

<summary>Q19. Hunyuan-Video 的 dual-stream → single-stream 设计是什么？</summary>

- **Dual-stream blocks**（前段 layers）：text token + video token 共享 attention 矩阵，但 QKV / FFN / AdaLN 各 stream 独立——让两个模态先精炼自己的表征
- **Single-stream blocks**（后段 layers）：完全 share 参数，相当于 unified Transformer
- 直觉：早期需要保留模态特异性（text 是 sequential，video 是 spatiotemporal），后期可以统一处理
- Mochi 的 AsymmDiT 类似思路但更激进——video stream 比 text stream 宽 4 倍

只说"都是 self-attn 同序列"——遗漏了 stream-specific 参数 / 早晚不同的 design。

</details>

<details>

<summary>Q20. Mochi-1 的 AsymmDiT (asymmetric MM-DiT) 是什么？</summary>

- 两个 stream（text / video）的 hidden dim 不一样：video 比 text 宽 4 倍（如 video $D{=}3072$ / text $D{=}768$）
- 直觉：视频信息量远超文本，应分配更多参数给 video stream
- Attention 时 text 和 video 投影到共同 head dim，做 self-attention
- FFN 各 stream 独立，按各自宽度
- 比标准 MM-DiT 用更少 text-side 参数，省显存又不损质量

把 "asymmetric" 当成两个 stream 不交互——错，它们仍同 attention。

</details>

### L3 顶级 lab 题（深度 — diffusion / video specialist）

<details>

<summary>Q21. 3D Causal VAE vs 标准 3D VAE：除了 streaming，还有什么本质差别？(必考)</summary>

- **标准 3D VAE**：时间 kernel center 对齐，需 symmetric padding，**output[t] 依赖 input[t-\lfloor k_t/2 \rfloor \ldots t+\lfloor k_t/2 \rfloor]**——看了未来
- **3D Causal VAE**：时间 kernel 因果对齐，左侧 pad $k_t - 1$，右侧不 pad，**output[t] 只依赖 input[t-k_t+1 \ldots t]**

**本质差别**：

- **Streaming / 自回归推理**：causal 允许 chunk-wise encoding，每来一段视频立即编码；标准 VAE 要等完整 clip 才能算（kernel center 需未来帧）
- **图像视频同空间**：两者 $T{=}1$ 时其实都能"退化到 2D"（zero pad 让其他时间片不起作用），但 causal 设计是"过去全见 + 未来全 0"，与训练时大 $T$ 的子集行为一致；symmetric pad 在 $T{=}1$ 时让 kernel 左右各看 $\lfloor k_t/2 \rfloor$ 个 padding，行为与训练时 $T \gg 1$ 的 boundary 行为不同——**一致性更差**
- **训练 + inference 一致性**：causal VAE 训练（完整 clip）与推理（AR chunk）共用一套 padding 规则；标准 3D VAE 在 AR chunk 推理时缺右侧未来 frame，需要特殊 pad
- **与 AR video 兼容**：causal 是 AR video 生成 (VAR-Video / Cosmos) 的必要前提
- **训练数据 utilization**：image-only batch 在 causal VAE encoder 里行为与训练时一致；标准 3D VAE 需要 dummy symmetric padding，与正常训练分布有 gap

> 💡 **加分**：解释 causal VAE 的 downsample 层怎么做（time stride 2 时仍保 causal）；解释训练 batch 里 image + video 混合的 implementation 细节。

只说"causal 是 mask 未来"——这是必要不充分；要展开"为何这件事让生态闭环（image + video 同 stack, AR 兼容, streaming）"。

</details>

<details>

<summary>Q22. 用 spacetime patches 处理任意 resolution / duration / aspect ratio 的工程实现？</summary>

- **Patchify token 数 $N = (t/p_t)(h/p_h)(w/p_w)$ 随输入 shape 变**；Transformer 对 $N$ 不敏感（self-attn 是 set-like）
- **RoPE-3D**：相对位置编码，不与最大长度绑定；任意 $(t, h, w)$ 三元组都可旋转
- **Packing**：一 batch 内 mixed shape sample concat 到同 seq + segment attention mask（同 sample 内可见、跨 sample 屏蔽）+ pad 到 max length
- **Aspect ratio / duration 作为 condition**：额外 token / scalar 经 AdaLN 注入，让模型学不同比例的合理构图
- Sora 首次 production 化；社区复刻见 OpenSora-Plan / OpenSora 1.2+

只说"transformer 不挑长度"——要展开 packing + segment mask + RoPE-3D + aspect ratio condition 这一整套工程组件。

</details>

<details>

<summary>Q23. Factorized 2+1D vs Full 3D attention 复杂度对比的精确公式 + 何时选哪个？</summary>

- 记每帧 spatial token $S = (h/p_h)(w/p_w)$，时间 token $T_t = t/p_t$，总 token $N = S T_t$
- **Factorized 2+1D**：
  - Spatial attn (每帧内独立): $T_t \cdot S^2 \cdot d$
  - Temporal attn (每空间位置内独立): $S \cdot T_t^2 \cdot d$
  - 合计 $O\!\left(d \cdot (T_t S^2 + S T_t^2)\right) = O\!\left(d \cdot S T_t (S + T_t)\right)$
- **Full 3D**: $O(d \cdot N^2) = O\!\left(d \cdot S^2 T_t^2\right)$
- 比值 = full / factorized = $\frac{S^2 T_t^2}{S T_t(S + T_t)} = \frac{S T_t}{S + T_t}$，等于 $\min(S, T_t)$ 量级
- $S \approx 10^3$（720p latent）、$T_t \approx 30$（5s）时：full 比 factorized 贵 $\approx 30\times$（被 $\min$ 主导）

**何时选哪个**：

- **算力充足 + 高质量目标**：Full 3D（Sora / Hunyuan / Mochi）
- **预算受限 / 快速迭代**：Factorized 2+1D（Latte / OpenSora / AnimateDiff）
- **质量与速度折中**：大部分块 factorized + 几块 full 3D（OpenSora-Plan 思路）
- 显存敏感时 factorized + recompute 是更经济选择

只给定性回答"full 更贵"——L3 题要给精确公式 + 工程选择 trade-off。

</details>

<details>

<summary>Q24. 长视频 (>30s) 现阶段最 promising 的路线是什么？为什么 chunk AR 容易 drift？</summary>

- **Drift 数学解释**：每 chunk 从 $p(x | \hat{z}_\text{prev})$ 采样，$\hat{z}_\text{prev}$ 本身是模型生成（非真实数据），多步条件后误差累积（类似 RNN exposure bias）
- **缓解**：overlap + clamp 已知部分；多帧 ref（不只末尾 1 帧）；global learned motif token
- **更 promising 路线**：
  - **Hierarchical**：先生 low-fps keyframe，再 temporal super-resolution 插中间帧
  - **Long-context full attention**：直接长视频 + sparse / sliding / hierarchical 控制成本
  - **Diffusion Forcing 等 hybrid**
- Sora 公开 demo 60s 内部细节未披露；Movie Gen 多阶段 keyframe + interp + super-res

只说"AR drift"——要给具体 drift 数学解释 + 当前 SoTA 应对。

</details>

<details>

<summary>Q25. Hunyuan-Video / Mochi-1 / Wan 都用 Rectified Flow，比 DDPM ε-pred 强在哪？</summary>

- **Target 平稳**：RF 的 $u_t = x_1 - x_0$ 给定 $(x_0, x_1)$ 后是常数，跨 $t$ 量级变化比 $\epsilon$-pred 小
- **少步采样**：path 是直线（理想），少步 ODE 误差小；Reflow 后 1-4 步可生成
- **Loss conditioning 自然**：不像 DDPM 需 SNR / VLB / EDM-preconditioning 等显式 reweighting
- **与 SD3 共享 stack**：image / video 同一套 recipe（v-pred + logit-normal $t$）
- DDPM ε-pred 需 noise schedule 设计 + 不同 $t$ 处 target 尺度差大，训练 reweighting 复杂；RF 更"一招吃遍"

只说"RF 路径更直"——要展开训练目标 / 数值稳定性 / sampler 兼容多角度。

</details>

## §A 附录：完整 from-scratch 视频生成模型骨架

### A.1 总体类图

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

### A.2 训练 forward + A.3 Euler RF sampler

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
        mu_r, _ = model.encoder(ref_image.unsqueeze(2))         # 走 T=1 path
        ref_full = torch.zeros_like(z1); ref_full[:, :, :1] = mu_r
        ref_mask = torch.zeros(B, 1, *z1.shape[2:], device=device); ref_mask[:, :, :1] = 1.0
        # I2V trick: 参考帧保持 clean (不加噪), 训练分布与推理一致
        ref_mask_C = ref_mask.expand_as(z_tau).bool()
        z_tau = torch.where(ref_mask_C, z1, z_tau)
    v_pred = run_dit(model, z_tau, text, tau, ref_full, ref_mask)
    # I2V: 只对待生成帧算 loss
    if ref_mask is not None:
        gen = (1 - ref_mask).expand_as(v_pred)
        return ((v_pred - u_target).pow(2) * gen).sum() / gen.sum().clamp(min=1.0)
    return F.mse_loss(v_pred, u_target)


@torch.no_grad()
def video_gen_sample(model, text, t_lat=12, h_lat=90, w_lat=160, steps=50,
                     ref_image=None, guidance_scale=6.0, C=16):
    """ 端到端推理: t_lat/h_lat/w_lat 直接指定 latent 形状(避开 T/4 整除陷阱). """
    device = next(model.parameters()).device
    z = torch.randn(1, C, t_lat, h_lat, w_lat, device=device)
    ref_full, ref_mask = None, None
    if ref_image is not None:
        mu_r, _ = model.encoder(ref_image.unsqueeze(2))
        ref_full = torch.zeros_like(z); ref_full[:, :, :1] = mu_r
        ref_mask = torch.zeros(1, 1, t_lat, h_lat, w_lat, device=device); ref_mask[:, :, :1] = 1.0
        # 推理初始即把参考帧位 clamp 到 clean latent (与训练一致), 不然首步 DiT 看到的是 noise
        z = torch.where(ref_mask.bool().expand_as(z), ref_full, z)
    taus = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        tau_i = taus[i].expand(1)
        v_cond   = run_dit(model, z, text, tau_i, ref_full, ref_mask)
        v_uncond = run_dit(model, z, [""],  tau_i, ref_full, ref_mask)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        z = z + (taus[i + 1] - taus[i]) * v
        if ref_mask is not None:                                # I2V: 每步后 clamp 参考帧
            z = torch.where(ref_mask.bool().expand_as(z), ref_full, z)
    return model.decoder(z)
```

> ✅ **Sanity checks 视频生成模型常做的** —

- **Reconstruction-only**：先 VAE encode 后 decode，PSNR / SSIM / LPIPS 看是否能复原原视频
- **Random latent decode**：从 $\mathcal{N}(0, I)$ 采 latent decode，看 VAE 是否退化（应该出"自然 looking" 但语义不一定连贯的视频）
- **Class / prompt overfit**：单一 prompt 训练 1000 步看模型能不能 memorize
- **VBench 局部维度**：训练中段先评 Subject Consistency / Motion Smoothness 看时序是否稳定
- **CFG sweep**：scale = 0/1/3/6/10 看影响（过高会饱和 / 闪烁）

### A.4 当前 SoTA 一览

| 维度 | 当前 SoTA | 备注 |
| --- | --- | --- |
| 开源最强 T2V | Hunyuan-Video 13B | 2024-12，质量逼近闭源 |
| 开源 I2V | Wan 2.2 I2V / Hunyuan-Video I2V | 中文 prompt 友好 |
| 闭源最强 | Veo 2 / Sora / Kling | 4K / 长时长 |
| 实时 | LTX-Video 2B | RTX 4090 实时 |
| 视频 + 音频 | Movie Gen (闭源) | 30B 联合生成 |
| 控制类 | AnimateAnyone / MotionCtrl | 人物 / 相机控制 |

---

**Video Generation Quick Reference** · 主要参考：Sora technical report (OpenAI 2024), Hunyuan-Video tech report (Tencent 2024), Mochi-1 blog (Genmo 2024), CogVideoX (Yang et al. arXiv 2024-08 → ICLR 2025), Wan tech report (Team Wan / Ang Wang et al. arXiv 2025), Movie Gen tech report (Meta 2024), VBench (Huang et al. CVPR 2024), AnimateDiff (Guo et al. arXiv 2023-07 → ICLR 2024), SVD (Blattmann et al. arXiv 2023-11), Latte (Ma et al. arXiv 2024-01 → TMLR 2025), MotionCtrl (Wang et al. 2023 → SIGGRAPH 2024), AnimateAnyone (Hu et al. CVPR 2024), SD3 (Esser et al. ICML 2024)
