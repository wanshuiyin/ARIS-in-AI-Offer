## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 VLM** — 一页拿下视觉-语言模型面试核心要点（详见后文 §1–§13 推导与代码）。

1. **视觉 encoder = ViT 主导**：Dosovitskiy et al. 2021 (ICLR) 把图像切 $P\times P$ patch（一般 $P=14$ 或 $16$）做线性投影 + 可学习 positional embedding + 可选 `[CLS]` token，输入 Transformer encoder。**CLIP / SigLIP / LLaVA 的视觉端都是 ViT 变体**。

2. **CLIP 对称 InfoNCE（必推）**：Radford et al. 2021 (ICML) 让 image embedding $\mathbf{u}_i$ 和 text embedding $\mathbf{v}_i$ 在共享空间里做对比学习，loss 为 **行 softmax + 列 softmax 平均**：$\mathcal{L} = \tfrac{1}{2}(\mathcal{L}_{i\to t} + \mathcal{L}_{t\to i})$。温度 $\tau$ 可学习（log-parameterize 为 `logit_scale=log(1/τ)`；被 clip 的是 $\exp(\text{logit\_scale})=1/\tau$ 的上界 100，等价 $\tau$ 下界 0.01，而非 $\tau$ 本身被 clip 到 100）。

3. **SigLIP 用 sigmoid 替 softmax**：Zhai et al. 2023 (ICCV) 把 N×N 相似度矩阵的每一项独立做 binary CE，**摆脱 batch-wise softmax 归一化**，因此对 batch size 不再线性敏感，单机能用 32k+ batch 训；引入 learnable bias $b$ 修正初期 negative dominance。**SigLIP-2 (Google 2025)** 加入 caption + self-distillation + dense local objectives 并扩展多语言。

4. **LLaVA = projector + 2-stage train**：Liu et al. 2023 (NeurIPS) 用一个 **单层线性投影矩阵 $W$** 把 frozen CLIP 视觉特征投到 LLM token 空间（**2 层 MLP projector 是 LLaVA-1.5 的升级**）。**Stage 1** 只训 projector 做 feature alignment（caption 数据），**Stage 2** 解冻 LLM 做 visual instruction tuning（GPT-4 生成的 158K instructions）。

5. **Q-Former vs Projector 是 BLIP-2 的核心 trade-off**：Li et al. 2023 (ICML) 用 32 个 learnable query token 在 frozen image encoder 上做 **cross-attention**，把任意分辨率/数量的 patch 压成固定 32 token——计算预算稳定但**信息有损 + 训练复杂**。LLaVA 的 MLP 简单但 token 数随分辨率二次增长。

6. **Flamingo / Llama-3.2-Vision = gated cross-attn**：Alayrac et al. 2022 (NeurIPS) 用 **Perceiver Resampler**（64 latent query）把 visual feature 压成固定数 token，再在 LLM 每隔几层插入 **gated cross-attention** 层（$\tanh$ 门控初始化为 0，保留 frozen LLM 的 text-only 能力）。

7. **Qwen2-VL 的 M-RoPE 必考**：Wang et al. 2024 把 RoPE 沿 head_dim 分成 6 段，按 axis 序列 $(t, h, w, t, h, w)$ 分配 (t / h / w 三组位置 id)；典型配置 `mrope_section=[16,24,24]`（单位是半 head_dim 的对数，$\sum \times 2 = $ head_dim=128，**全部 128 维都旋转**）。这样每个 token 同时携带 (t, h, w) 三维位置而不需要扁平化。**配合 native dynamic resolution**（不再 padding 到固定 224×224）。

8. **训练三段式 + 偏好优化**：(1) **alignment** 训 projector / Q-Former；(2) **visual instruction tune** 解冻部分 LLM；(3) **preference**（LLaVA-RLHF, RLAIF-V, VLM-R1, DPO/PPO）治理幻觉、对齐 long-tail。**VLM-R1 (2025)** 用 GRPO + verifiable reward 把推理能力迁到视觉-语言任务。

## §1 直觉：VLM 在做什么？

把一张图看作 "另一种语言"。VLM 的工作可以拆成三件事：

- **视觉 tokenizer**：把像素压成离散或连续的 token 序列（ViT patch → embedding）

- **跨模态对齐**：让相同语义的 image / text 在同一空间靠近——这是 CLIP / SigLIP 干的事，本质是 **学一个共享 embedding space**

- **跨模态生成**：让 LLM 在 prompt 里"看见"图片——LLaVA / Qwen-VL / Flamingo 干的事，本质是 **把 image token 作为前缀塞进 LLM 的 context**

> 💡 **三种 fusion 范式** — 这是 VLM 架构的主线分歧。

- **早期融合（dual-encoder + contrastive）**：CLIP / SigLIP，**没有 cross-modal attention**，只在 embedding 空间靠近 / 远离

- **Projector 融合（visual tokens → LLM context）**：LLaVA / Qwen-VL，把 image embedding 投到 LLM 的 token 空间，**作为输入 token 拼接**，自回归解码

- **Cross-attn 融合（LLM 内新增独立 cross-attn 层）**：Flamingo / Llama-3.2-V，新增 cross-attention 层，**LLM 的 text token 主动 query 视觉 KV**。BLIP-2 的 cross-attention 只发生在 Q-Former 内部（32 个 learnable query 作 Q，冻结 image encoder 输出作 K/V），产出的 32 个 visual token 随后作为前缀 token 拼接进冻结 LLM 的输入序列——LLM 接口上其实和 LLaVA/Qwen-VL 的 projector 范式相同，LLM 本身并未新增 cross-attention 层

对比 Q/K/V 视角：projector 范式下 image 是 LLM 输入序列的一部分（self-attention 内全交互）；cross-attn 范式下 image 永远是 KV，**只被 query**——这导致 **inference 时 KV cache 处理方式不同**。

## §2 ViT：把图像变 token 序列

### 2.1　Patch tokenize

输入图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$，切成 $N = HW/P^2$ 个 $P\times P$ patch，每个 patch flatten 成 $P^2 C$ 维向量，过线性层投到 $D$ 维：

$$\mathbf{z}_0 = [\mathbf{x}_\text{class};\ \mathbf{x}^1_p \mathbf{E};\ \mathbf{x}^2_p \mathbf{E};\ \dots;\ \mathbf{x}^N_p \mathbf{E}] + \mathbf{E}_\text{pos}$$

- $\mathbf{E} \in \mathbb{R}^{P^2 C \times D}$ 是 patch embedding 矩阵（等价于 stride=$P$, kernel=$P$ 的 Conv2D）

- $\mathbf{x}_\text{class} \in \mathbb{R}^D$ 是 **learnable [CLS] token**，用于聚合全局信息（分类时取 $\mathbf{z}_L^0$）

- $\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$ 是 **learnable 1D positional embedding**——原版 ViT 用 1D learned，**没有用 2D sinusoidal**（论文 Appendix D.4 报告 1D learned 与 2D sinusoidal 性能差异在误差范围内）

> ⚠️ **CLIP / SigLIP 不一定用 [CLS]** — CLIP ViT 用 [CLS] 输出，SigLIP / EVA-CLIP / 现代 LLaVA 倾向用 **patch token average pool** 或直接保留所有 patch token 喂给下游。**`[CLS]` 是 ViT 原始 paper 的选择，不是 ViT 的固有部件**。

### 2.2　Transformer 主干

$$\mathbf{z}'_\ell = \text{MHA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

Pre-norm（LN 在 sub-layer 输入端），MLP 用 GELU。注意 ViT 原版的 patch 数固定（$224/16=14 \Rightarrow N=196$），**positional embedding 表大小固定**——这是 dynamic resolution 要解决的痛点（§10）。

### 2.3　ViT 规格梳理

| 模型 | Patch | Hidden $D$ | Layers | Heads | Params | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | 16 | 768 | 12 | 12 | 86M | Dosovitskiy 2021 |
| ViT-L/14 | 14 | 1024 | 24 | 16 | 304M | Radford et al. 2021 (CLIP) |
| ViT-H/14 | 14 | 1280 | 32 | 16 | 632M | Dosovitskiy 2021 |
| ViT-g/14 | 14 | 1408 | 40 | 16 | 1.0B | Zhai et al. 2022 |
| ViT-bigG/14 | 14 | 1664 | 48 | 16 | 1.8B | OpenCLIP, 2023 |
| EVA-02-L/14 | 14 | 1024 | 24 | 16 | 304M | Fang 2023 |
| SigLIP SoViT-400M/14 | 14 | 1152 | 27 | 16 | 400M | Alabdulmohsin 2023 |

> ⚠️ **ViT-L/14 的出处纠正** — Large 架构本身（D=1024, 24 层, 16 heads）确实来自 Dosovitskiy et al. 2021；但原始 ViT 论文只报告了 B/16, B/32, L/16, L/32, H/14，**从未训练过 patch=14 的 Large 配置**。ViT-L/14 这一具体组合是 CLIP（Radford et al. 2021, ICML）首次训练并推广的。
>
> 💡 **head_dim 通常固定 64** — ViT 系列大多遵循 head_dim ≈ 64–88，等价于 $D / H$。Scaling-laws 推荐 head_dim 不要太小，否则单 head 表达力受限。

### 2.4　Code: ViT patch embed + 主干（核心 60 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size, self.patch_size = img_size, patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 用 stride=P, kernel=P 的 Conv2d 等价于线性投影
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                                   # x: [B, C, H, W]
        x = self.proj(x)                                    # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)                    # [B, N, D]
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)       # self-attention
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, num_classes=1000, use_cls=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        N = self.patch_embed.num_patches
        self.use_cls = use_cls
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, N, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                             # [B, N, D]
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)                  # [B, N+1, D]
        x = x + self.pos_embed                              # broadcast over batch
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        feat = x[:, 0] if self.use_cls else x.mean(dim=1)   # CLS or mean-pool
        return self.head(feat)
```

> ⚠️ **interpolate_pos_embed 的常见 bug** — 把 ViT 从 $224^2$ 迁到 $336^2$ 时，pos_embed 表需从 $(14^2 + 1)$ 行 resize 到 $(24^2 + 1)$ 行。**正确做法**：保留 [CLS] 不动，把 patch 部分 reshape 成 $14\times 14\times D$ 做 bicubic 插值到 $24\times 24$，再 flatten 拼回。**踩坑**：直接对 $(N+1)$ 行做 1D 插值会把 [CLS] 当 patch 算进去。

## §3 CLIP：对称 InfoNCE（必推导）

### 3.1　形式化目标

CLIP（Radford et al. 2021, ICML）训练时一个 batch 有 $N$ 个 (image, text) pair。两个 encoder $f_\theta$（image）、$g_\phi$（text）分别产出 $\ell_2$-normalized embedding：

$$\mathbf{u}_i = \frac{f_\theta(I_i)}{\|f_\theta(I_i)\|_2}, \quad \mathbf{v}_j = \frac{g_\phi(T_j)}{\|g_\phi(T_j)\|_2}, \quad \mathbf{u}_i, \mathbf{v}_j \in S^{D-1}$$

定义相似度矩阵 $\mathbf{S} \in \mathbb{R}^{N\times N}$（"logit"）：

$$S_{ij} = \frac{\mathbf{u}_i^\top \mathbf{v}_j}{\tau}$$

其中 $\tau > 0$ 是可学习温度（实际工程化为 `logit_scale = log(1/τ)`，反向传播更稳定；教程 §3.5 代码里 clamp 只有上界，即 `logit_scale` 被 clamp 在 $(-\infty, \log 100]$，等价 $\tau$ 只有下界 0.01，没有上界——部分实现如 OpenCLIP 也会额外加下界 clamp，但不是通用规则）。

### 3.2　对称 InfoNCE Loss（行 + 列 softmax 平均）

**Image → Text 方向**（对每个 image $i$，正样本是 $T_i$，负样本是 $\{T_j\}_{j\neq i}$）：

$$\mathcal{L}_{i\to t} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}$$

**Text → Image 方向**：

$$\mathcal{L}_{t\to i} = -\frac{1}{N}\sum_{j=1}^{N} \log \frac{\exp(S_{jj})}{\sum_{i=1}^{N} \exp(S_{ij})}$$

对称总 loss：

$$\boxed{\;\mathcal{L}_\text{CLIP} = \frac{1}{2}\left(\mathcal{L}_{i\to t} + \mathcal{L}_{t\to i}\right)\;}$$

> ✅ **等价表述：行 softmax + 列 softmax 平均** — 对矩阵 $\mathbf{S}$，**行方向 softmax 后取对角项的 NLL**（image→text），**列方向 softmax 后取对角项的 NLL**（text→image）。两个 cross-entropy 平均即得 CLIP loss。

### 3.3　梯度推导（为什么对称很重要）

固定 $\tau=1$，对 $\mathcal{L}_{i\to t}$ 中第 $i$ 行的 logits $\mathbf{s}_i = (S_{i1},\dots,S_{iN})^\top$ 做 softmax，记 $p_{ij} = \text{softmax}(\mathbf{s}_i)_j$。则：

$$\frac{\partial \mathcal{L}_{i\to t}}{\partial S_{ij}} = \frac{1}{N}\left(p_{ij} - \mathbb{1}[j=i]\right)$$

- $j=i$（正样本）：梯度 $\propto p_{ii} - 1 < 0$，**拉近** $\mathbf{u}_i, \mathbf{v}_i$

- $j \neq i$（负样本）：梯度 $\propto p_{ij} > 0$，**推远** $\mathbf{u}_i, \mathbf{v}_j$

单向 $\mathcal{L}_{i\to t}$ 中 $S_{ij}$ 本就同时依赖 $\mathbf{u}_i$ 和 $\mathbf{v}_j$，所以哪怕只用这一个方向的 loss，**两个 encoder 都会收到梯度**——文本 encoder 并非"不受约束"。对称化真正的价值是**显式地同时优化 image→text 与 text→image 两个检索方向**，经验上提升双向检索质量，而不是防止某种数学意义上的表征坍塌。

### 3.4　Temperature 的作用

$$\mathcal{L}_{i\to t} = -\frac{1}{N}\sum_i \log\frac{\exp(\mathbf{u}_i^\top \mathbf{v}_i / \tau)}{\sum_j \exp(\mathbf{u}_i^\top \mathbf{v}_j / \tau)}$$

- $\tau \to 0^+$（**很小**）：softmax 接近 one-hot，**只关心 hardest negative**（最像但不对的那个 text）；梯度被一两个负样本主导，训练不稳

- $\tau \to \infty$（**很大**）：softmax 均匀，正负样本几乎不可分，loss 接近 $\log N$ 常数，**几乎无梯度**

- **OpenAI CLIP 学到的稳态**：$\tau \approx 0.01$（`logit_scale ≈ log(100)`），并 clamp 上界防止崩

> 💡 **InfoNCE 的下界解释** — Oord et al. 2018 (CPC) 证明 InfoNCE 是互信息 $I(U; V)$ 的下界：$I(U; V) \ge \log N - \mathcal{L}_\text{InfoNCE}$。这个下界只在**实际达到的 loss 值**下成立——$N$ 增大时任务本身变难，$\mathcal{L}_\text{InfoNCE}$ 一般也会随之上升，不会自动下降。只有当表征质量同步提升、抵消了任务难度上升，loss 才可能持平或下降，从而让下界变紧。**不能仅凭调大 batch size 就推出下界自动提高**——这也是 InfoNCE 类 MI 估计的已知局限（Poole et al. 2019; McAllester & Stratos 2020）；CLIP / SigLIP 追求 huge batch 更多是经验上"负样本更多、任务更难迫使表征更好"的驱动，而非下界的自动抬升。

### 3.5　Code: CLIP 对称 InfoNCE（核心 50 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """Symmetric InfoNCE used by OpenAI CLIP (Radford et al. 2021)."""
    def __init__(self, init_tau=0.07, max_logit_scale=4.6052):
        super().__init__()
        # 等价于 logit_scale = log(1/τ); 初始 ~ log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_tau).log())
        self.max_logit_scale = max_logit_scale            # log(100), clamp 防爆炸

    def forward(self, image_feats, text_feats):
        """
        image_feats: [N, D]   (unnormalized)
        text_feats:  [N, D]
        """
        # L2 normalize 到单位球面
        u = F.normalize(image_feats, dim=-1)              # [N, D]
        v = F.normalize(text_feats, dim=-1)               # [N, D]

        # clamp logit_scale 上界 (训练后期会涨到 ~log(100))
        logit_scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()

        # 相似度矩阵
        logits_i2t = logit_scale * u @ v.t()              # [N, N]
        logits_t2i = logits_i2t.t()                       # [N, N]

        # 对角线是正样本对
        N = u.size(0)
        labels = torch.arange(N, device=u.device)

        loss_i2t = F.cross_entropy(logits_i2t, labels)    # 行 softmax NLL
        loss_t2i = F.cross_entropy(logits_t2i, labels)    # 列 softmax NLL

        return 0.5 * (loss_i2t + loss_t2i), logit_scale

# 用法示例（DDP 下需 all-gather 把所有 GPU 的 feats 拼起来再算）
if __name__ == "__main__":
    N, D = 8, 512
    img_feats = torch.randn(N, D)
    txt_feats = torch.randn(N, D)
    criterion = CLIPLoss()
    loss, scale = criterion(img_feats, txt_feats)
    print(f"loss={loss.item():.4f}  logit_scale={scale.item():.2f}")
```

> ⚠️ **DDP 下必须 all-gather 才是真 InfoNCE** — 单 GPU 上 batch $N$ 算出来的 loss 只覆盖本地 negatives。生产 CLIP（OpenCLIP / OpenAI）会在 forward 后 `dist.all_gather` 所有 GPU 的 $\mathbf{u}, \mathbf{v}$，让 negative pool = global batch size（如 32k）。**跨卡正确反传需要可微分的 all-gather**（如 `torch.distributed.nn.functional.all_gather`，其 autograd 会把上游梯度正确 scatter/reduce 回各 rank）或 OpenCLIP 的 `local_loss=True` 技巧（只对本地那部分 embedding 反传，但 loss 分母仍用全局负例）——这是工程 trick，不是数学问题。**注意 gradient checkpointing 只是用重算换显存，与跨卡梯度传递无关**，不能替代可微分 all-gather 或 local_loss。

### 3.6　CLIP 训练数据 & 规模

- **WIT (WebImageText)**：400M (image, text) pair，从互联网爬取（未开源）

- **LAION-400M / LAION-2B**：OpenCLIP 用的开源替代，2022–2023 训了一系列规模

- **DataComp**：Gadre et al. 2023 (NeurIPS) 提出系统性 data filtering benchmark，**数据质量 > 数据规模**

- 模型规模：OpenAI 最大 ViT-L/14；OpenCLIP 训到 ViT-bigG/14 (LAION-2B)，**零样本 ImageNet ~80%+**

### 3.7　CLIP 的失败模式

- **OCR / 文字理解弱**：训练 caption 一般不描述图中文字，所以 CLIP 对图中文本几乎"瞎"

- **细粒度计数失败**：5 只鸟 vs 6 只鸟在 CLIP embedding 几乎无法区分（"counting problem"）

- **空间关系弱**："cat on top of dog" vs "dog on top of cat" 区分困难（Winoground / ARO benchmark 量化了这一点；POPE 测的是物体幻觉，不是空间关系）

- **bag-of-words 倾向**：Yuksekgonul et al. 2023 (ICLR) 证明 CLIP 对 caption 词序几乎不敏感

## §4 SigLIP：sigmoid 替 softmax，batch 缩放重写

### 4.1　Motivation

CLIP 的 softmax 归一化把所有 N×N 个相似度耦合：每个正样本的梯度依赖于**整行**的负样本 logsumexp。这导致：

- **batch size 极敏感**：N 翻倍，loss landscape 显著变化；小 batch 几乎学不动

- **DDP 通信昂贵**：必须 all-gather embedding（O(N·D) 字节），通信瓶颈

- **数值不稳**：极大 N 下 softmax 容易溢出

Zhai et al. 2023 (ICCV) 提出 **SigLIP**：把 N×N 矩阵的每一项**独立**做 binary classification。

### 4.2　Sigmoid Loss 推导

定义相似度 $S_{ij} = t \cdot \mathbf{u}_i^\top \mathbf{v}_j + b$，其中 $t = e^{t'}$ 是 learnable scale（与 CLIP 的 $1/\tau$ 同），$b$ 是 learnable bias（初始化负值，如 $b_0 = -10$，避免训练初期全部预测正）。

label $y_{ij} = +1$ 若 $i=j$，$-1$ 否则。每项做 binary logistic regression：

$$\mathcal{L}_\text{SigLIP} = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N \log \sigma\!\left(y_{ij} \cdot S_{ij}\right) = \frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N \log\!\left(1 + \exp(-y_{ij} S_{ij})\right)$$

> ✅ **关键性质** — 每一项 $(i,j)$ 的 loss **不依赖其他项**。所以：

- batch size 不再耦合所有 negative

- 单机可用极大 batch（SigLIP 报告单 chip 32k batch 可训）

- 通信只需把本机的 query 与远端 key 配对计算 sigmoid（chunked all-pair），不需要 logsumexp 同步

> ⚠️ **bias $b$ 不是装饰** — 训练初期 $\mathbf{u}, \mathbf{v}$ 接近随机，$S_{ij}$ 接近 0，sigmoid 输出 0.5。负样本占 $N^2 - N \approx N^2$，正样本只占 $N$ 个；如果初始预测全部 ~0.5，**负样本梯度会主导初期训练**。SigLIP 初始化 $b_0 \approx -10$，让 sigmoid 输出初期接近 0，**所有点先被预测为负**，正样本 loss 大、负样本 loss 小，从这个状态再启动训练就稳定了。

### 4.3　SigLIP vs CLIP 对比

| 维度 | CLIP (softmax) | SigLIP (sigmoid) |
| --- | --- | --- |
| Loss 形式 | $\propto$ logsumexp(row) + logsumexp(col) | $\propto$ $\sum_{ij}$ binary logistic |
| Batch 依赖 | 强（梯度耦合 batch） | 弱（每项独立） |
| 通信 | all-gather embedding | chunked all-pair sigmoid |
| Bias 项 | 无（隐式被 softmax 吸收） | learnable $b$，初始化 $\approx -10$ |
| 小 batch 表现 | 差（< 4k 几乎不学） | 显著更好（1k 也能学） |
| 大 batch 表现 | 边际收益递减 | 一直涨到 32k+ |
| 零样本 ImageNet（ViT-L/14, 400M data） | ~75% | ~76–78% |

### 4.4　SigLIP-2 (Google 2025)

Tschannen et al. 2025 在 SigLIP-1 基础上：

- 加入 **caption-style decoder**（类似 CapPa）做 captioning 辅助任务

- **Self-distillation + dense local objectives**：在 patch 级别做局部对比，提升细节定位能力

- **多语言扩展**：训练数据扩到 100+ 语言，多语言 zero-shot 显著提升

- 公开 NaFlex 变体支持 native aspect ratio

### 4.5　Code: SigLIP sigmoid loss（核心 35 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-training (Zhai et al. 2023)."""
    def __init__(self, init_t=10.0, init_b=-10.0):
        super().__init__()
        # log-parameterize t for stability; b is a learnable bias
        self.t_prime = nn.Parameter(torch.tensor(init_t).log())
        self.b = nn.Parameter(torch.tensor(float(init_b)))

    def forward(self, image_feats, text_feats):
        u = F.normalize(image_feats, dim=-1)             # [N, D]
        v = F.normalize(text_feats, dim=-1)              # [N, D]

        t = self.t_prime.exp()                           # scale > 0
        logits = t * (u @ v.t()) + self.b                # [N, N]

        # y_{ij} = +1 if i == j else -1
        N = u.size(0)
        labels = 2 * torch.eye(N, device=u.device) - 1   # [N, N], +1 on diag, -1 off

        # log(1 + exp(-y * logits))  ==  -log sigmoid(y * logits)
        loss = -F.logsigmoid(labels * logits).sum() / N  # SigLIP convention: sum / N
        return loss, t, self.b
```

> ⚠️ **SigLIP 的 normalize 是 N 不是 N²** — 论文 Eq. (1) 的归一化分母是 batch size $N$（每行求和），不是矩阵元素数 $N^2$。**踩坑**：写成 `loss.mean()` 会得到 1/N² 量级，loss 偏小，learnable scale 收敛错。**正确**：`loss.sum() / N`。

## §5 EVA-CLIP / OpenCLIP / 其它 CLIP 变体

### 5.1　OpenCLIP

OpenCLIP（Cherti et al. 2023 CVPR）是 LAION 团队的开源复现 + 扩展：

- **训练 recipe 开源**：完整 LAION-400M / LAION-2B 训练脚本

- **更大规模**：ViT-bigG/14 在 LAION-2B 上训出，零样本 ImageNet ~80.1%（2023 年 SOTA）

- **distributed InfoNCE**：实现了 `local_loss=True` 的 gradient checkpoint，单 GPU 显存只存本地行/列

### 5.2　EVA-CLIP

EVA-CLIP（Sun et al. 2023）用 MIM 预训练的 EVA / EVA-02（Fang et al. 2023）作为 vision tower 初始化，**显著提升 sample efficiency**：

- ViT-L/14 在 LAION-2B 上训只需 OpenCLIP 1/3 计算预算达到同样精度

- **LayerScale + sub-LN + RoPE**：EVA-02 视觉端的工程改进

### 5.3　DataComp（数据 vs 模型 vs 算法）

Gadre et al. 2023 (NeurIPS) 设计了 "data filtering benchmark"：固定 (model, compute)，只调 data filter。结论：

- **CLIP filtering** + **basic filtering** + **image-based** 三种 filter 组合最优

- 大模型 (ViT-L/14) 在小数据 (12.8M) 上反而不如 ViT-B（**data-limited regime 下大模型过拟合**）

### 5.4　对比一览

| 方法 | Vision Tower 初始化 | Loss | Batch | 训练数据 | ImageNet zero-shot |
| --- | --- | --- | --- | --- | --- |
| CLIP (OpenAI) | from scratch | softmax InfoNCE | 32k | WIT 400M | 76.2% (L/14@336) |
| OpenCLIP | from scratch | softmax InfoNCE | 90k | LAION-2B | 80.1% (bigG/14) |
| EVA-CLIP | EVA-02 MIM | softmax InfoNCE | — | LAION-2B | 82.0% (E/14+) |
| SigLIP | from scratch | sigmoid | 32k | WebLI | 82.0% (So400M/14) |
| SigLIP-2 | from scratch | sigmoid + caption + distill | — | WebLI 10B | 84%+ |
| MetaCLIP | from scratch | softmax InfoNCE | — | 重新构造 LAION-grade | 79.2% (H/14) |

> 💡 **2024–2025 趋势** — SigLIP 系列在 zero-shot ImageNet 和下游 retrieval 上已经稳定超过 CLIP；典型开放权重 VLM 用 SigLIP-So400M 的是 **PaliGemma / LLaVA-OneVision**。**Molmo/PixMo 主模型仍用 OpenAI CLIP ViT-L/14@336（SigLIP 只出现在其论文 ablation）；InternVL 系列用自家的 InternViT；Qwen2-VL 用自训 ViT；LLaVA-1.5/1.6 仍用 CLIP ViT-L/14**——并非"切到 SigLIP"是行业共识。

## §6 LLaVA：projector + 2-stage 训练

### 6.1　架构

LLaVA（Liu et al. 2023 NeurIPS）的核心是三件套：

```
Image ──► CLIP ViT-L/14 ──► visual features  z_v ∈ R^{N × d_v}
                                  │
                                  │  W ∈ R^{d_v × d_LLM}   ← projector（1.0 单层线性 / 1.5 2 层 MLP）
                                  ↓
                            H_v ∈ R^{N × d_LLM}
                                  │
                                  │  与 text embedding 拼接
                                  ↓
Text tokens ──► tokenizer ──► H_t ──► [<bos>, H_v, H_t] ──► LLM (Vicuna / LLaMA-2)
                                                              │
                                                              ↓
                                                            autoregressive response
```

- **Vision tower**：CLIP ViT-L/14（frozen，取倒数第二层 patch token）。LLaVA-1.0 用 $224^2$ 输入得 $N=256$；**LLaVA-1.5 升级到 $336^2$ 得 $N=576$**

- **Projector** $W$：LLaVA-1.0 用单层 Linear；**LLaVA-1.5 升级为 2-layer MLP + GELU**（原文报告显著提升 instruction following）

- **LLM**：Vicuna-13B（LLaVA-1.0/1.5）或 LLaMA-2

### 6.2　训练两阶段

**Stage 1: Feature Alignment Pre-training**

- 用 CC3M / LAION-558K caption 数据，格式 `<image>\n<caption>`

- **只训 projector $W$**，冻结 vision tower 和 LLM

- 目的：让 visual feature 投到 LLM 的 word embedding 空间附近

**Stage 2: End-to-end Visual Instruction Tuning**

- 用 GPT-4 生成的 158K 视觉 instruction（LLaVA-Instruct）

- 解冻 projector + LLM，**冻结 vision tower**

- LLM 学会"理解图像、回答问题、follow visual instruction"

### 6.3　Code: LLaVA 风格 projector + forward（核心 60 行）

```python
import torch
import torch.nn as nn

class LLaVAProjector(nn.Module):
    """2-layer MLP + GELU, as in LLaVA-1.5."""
    def __init__(self, d_vision=1024, d_llm=4096):
        super().__init__()
        self.fc1 = nn.Linear(d_vision, d_llm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_llm, d_llm)

    def forward(self, x):                               # x: [B, N, d_vision]
        return self.fc2(self.act(self.fc1(x)))          # [B, N, d_llm]

class LLaVA(nn.Module):
    """Skeleton: CLIP vision tower + projector + LLM."""
    def __init__(self, vision_tower, projector, llm, image_token_id):
        super().__init__()
        self.vision_tower = vision_tower                # CLIPViT, frozen at stage 1
        self.projector = projector
        self.llm = llm                                  # e.g. LlamaForCausalLM
        self.image_token_id = image_token_id           # special <image> placeholder

    @torch.no_grad()
    def encode_image(self, pixel_values):
        # 取倒数第二层 patch features (不取 [CLS])
        vit_out = self.vision_tower(pixel_values, output_hidden_states=True)
        feat = vit_out.hidden_states[-2][:, 1:, :]      # drop CLS, [B, N, d_v]
        return feat

    def forward(self, input_ids, pixel_values, labels=None, attention_mask=None):
        # 1. 视觉特征 → projector → LLM 维度
        with torch.no_grad():
            visual_features = self.encode_image(pixel_values)        # [B, N, d_v]
        visual_tokens = self.projector(visual_features)              # [B, N, d_llm]

        # 2. LLM 的 word embedding 表
        token_embeds = self.llm.get_input_embeddings()(input_ids)    # [B, L, d_llm]

        # 3. 把 <image> placeholder 处替换为 visual_tokens
        B, L, D = token_embeds.shape
        new_embeds, new_labels, new_mask = [], [], []
        for b in range(B):
            image_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            assert image_pos.numel() == 1, "exactly one <image> placeholder expected"
            i = image_pos.item()
            # 拼接：[prefix tokens] + [N visual tokens] + [suffix tokens]
            chunks = [token_embeds[b, :i], visual_tokens[b], token_embeds[b, i+1:]]
            new_embeds.append(torch.cat(chunks, dim=0))
            if labels is not None:
                lab = labels[b]
                # visual token 位置 label = -100（不算 loss）
                ignore = torch.full((visual_tokens.size(1),), -100, dtype=lab.dtype, device=lab.device)
                new_labels.append(torch.cat([lab[:i], ignore, lab[i+1:]], dim=0))
            if attention_mask is not None:
                am = attention_mask[b]
                ones = torch.ones(visual_tokens.size(1), dtype=am.dtype, device=am.device)
                new_mask.append(torch.cat([am[:i], ones, am[i+1:]], dim=0))

        # 4. pad 回 batch tensor，喂给 LLM
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_embeds, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=-100) if labels is not None else None
        attention_mask = torch.nn.utils.rnn.pad_sequence(new_mask, batch_first=True) if attention_mask is not None else None
        return self.llm(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)
```

### 6.4　LLaVA-1.5 / 1.6 / NeXT 关键升级

| 版本 | 时间 | 主要改动 |
| --- | --- | --- |
| LLaVA-1.0 | 2023.04 | 单层 Linear projector；CLIP ViT-L/14@224²，视觉 token = 256（$16\times 16$） |
| LLaVA-1.5 | 2023.10 | 2-layer MLP；分辨率升到 336²，视觉 token = 576（$24\times 24$）；加入 OCR / GQA / VQAv2 等学术数据 |
| LLaVA-1.6 / NeXT | 2024.01 | **AnyRes**：训练阶段即采用切图网格 $2\times 2 / 2\times 3 / \dots$ 各编码再拼，支持任意 aspect ratio；token 数≈$(1{+}n{\cdot}m)\cdot 576$（2×2→约2880，2×3→约4032；实际实现有 unpad + image_newline，略少于该理论值） |
| LLaVA-OneVision | 2024.08 | 单 / 多图 / 视频统一；引入 SI（single image）+ OV（onevision）数据 mix |
| LLaVA-NeXT-Video | 2024.04 | 视频版，把多帧 visual feature 序列化喂入 |

> 💡 **AnyRes (LLaVA-1.6) 的核心 trick** — LLaVA-NeXT/1.6 在**训练阶段就采用 AnyRes 网格切图**（而非仅推理时的技巧），让模型在训练中学会处理多 tile + 全局缩略图的空间布局：把高分辨率图切成 $n \times m$ 个 336² tile 各自编码，再加一份缩放到 336² 的"全局缩略图"。**token 数约从 576 涨到 (1 + n·m)·576**（真实实现还会做 unpad 去除按 aspect ratio 产生的 padding 区域、并在 tile 行间插入 image_newline token，实际数量略少于这个理论公式），但每个 tile 仍走同一个 frozen ViT。**和 InternVL / Qwen-VL 的 tiling 是同一类思路**。

## §7 BLIP-2：Q-Former cross-attention

### 7.1　Motivation

LLaVA 的 projector 简单但**每个 patch 都成为 LLM token**：分辨率 ↑ token 数 ↑ LLM 计算 $O(L^2)$ ↑。BLIP-2（Li et al. 2023 ICML）用 **Q-Former**（Querying Transformer）把任意数量 patch **压成固定 32 token**。

### 7.2　Q-Former 结构

输入：frozen image encoder 输出 $\mathbf{Z} \in \mathbb{R}^{N \times d_v}$（N=257 for ViT-g/14@224）。Q-Former 有 32 个 **learnable query token** $\mathbf{q}_1, \dots, \mathbf{q}_{32} \in \mathbb{R}^{d_q}$。

每层 Q-Former block：

$$\mathbf{q}^{(\ell)} = \text{SelfAttn}(\mathbf{q}^{(\ell-1)})$$
$$\mathbf{q}^{(\ell)} = \text{CrossAttn}(\mathbf{q}^{(\ell)},\ \mathbf{Z},\ \mathbf{Z})\quad \text{（只插在偶数层）}$$
$$\mathbf{q}^{(\ell)} = \text{FFN}(\mathbf{q}^{(\ell)})$$

**关键**：
- **Self-attention 内**：query 之间互相交流，不和 image patch 交互
- **Cross-attention 内**：query 作 Q，image patch 作 K/V——**这是信息流入口**
- 输出 $\mathbf{q}^{(L)} \in \mathbb{R}^{32 \times d_q}$ 再过一个 Linear 投到 LLM 维度，作 32 个 visual token 喂给 frozen LLM

### 7.3　两阶段训练

**Stage 1: Representation Learning**（只训 Q-Former，frozen vision encoder）；Q-Former 的 cross-attention **恒定连接 query ↔ 冻结 image encoder**（与训练目标无关），query 与 text token 的交互实际发生在**共享的 self-attention 层内**，通过按训练目标切换的 attention mask 控制：
- ITC (Image-Text Contrastive)：query embedding 与 text 端 [CLS] 做 CLIP 风格对比（self-attn 用**单模态 mask**，query/text 互不可见）
- ITM (Image-Text Matching)：query 与 text token 在共享 self-attention 层内通过**双向 mask**互相可见地交互后做 binary classification
- ITG (Image-grounded Text Generation)：self-attn 用 **causal mask**，text 只能看到 query 和之前的 text（query 看不到 text），让 text decoder 基于 query 生成 caption

**Stage 2: Generative Learning**（只训 Q-Former，frozen LLM）
- 把 Q-Former 输出投到 LLM embedding 空间，**让 LLM 在 prefix-tune 模式下根据 32 visual token 做 captioning / VQA**

### 7.4　Code: Q-Former cross-attention 单层（核心 40 行）

```python
import torch
import torch.nn as nn

class QFormerLayer(nn.Module):
    """One Q-Former block: SelfAttn (queries) -> CrossAttn (queries <- image) -> FFN."""
    def __init__(self, d_q=768, d_v=1408, num_heads=12, mlp_ratio=4, has_cross=True):
        super().__init__()
        self.has_cross = has_cross
        self.ln_self = nn.LayerNorm(d_q)
        self.self_attn = nn.MultiheadAttention(d_q, num_heads, batch_first=True)
        if has_cross:
            self.ln_cross = nn.LayerNorm(d_q)
            # Q 来自 query (d_q), K/V 来自 image feat (d_v) -> 通过 kdim/vdim 适配
            self.cross_attn = nn.MultiheadAttention(d_q, num_heads,
                                                   kdim=d_v, vdim=d_v, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_q)
        hidden = int(d_q * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_q, hidden), nn.GELU(), nn.Linear(hidden, d_q))

    def forward(self, q, image_feats=None):              # q: [B, 32, d_q]
        # Self-attention: queries talk to each other
        h = self.ln_self(q)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + a
        # Cross-attention: queries attend to image patches
        if self.has_cross and image_feats is not None:
            h = self.ln_cross(q)
            a, _ = self.cross_attn(h, image_feats, image_feats, need_weights=False)
            q = q + a
        # FFN
        q = q + self.ffn(self.ln_ffn(q))
        return q

class QFormer(nn.Module):
    def __init__(self, num_queries=32, d_q=768, d_v=1408, depth=12, num_heads=12,
                 cross_every=2):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_queries, d_q))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([
            QFormerLayer(d_q, d_v, num_heads, has_cross=(i % cross_every == 0))
            for i in range(depth)
        ])

    def forward(self, image_feats):                      # [B, N, d_v]
        B = image_feats.size(0)
        q = self.queries.expand(B, -1, -1)               # [B, 32, d_q]
        for layer in self.layers:
            q = layer(q, image_feats)
        return q                                         # [B, 32, d_q]
```

> ⚠️ **`kdim`/`vdim` 适配** — `nn.MultiheadAttention` 默认 K/V 输入维度等于 `embed_dim`；Q-Former cross-attn 里 query 768 维、image feat 1408 维，**必须显式传 `kdim=d_v, vdim=d_v`**，否则 PyTorch 会在 forward 时按 768 期望 K/V 输入，直接抛 shape mismatch 错误（不是悄悄截断）。

### 7.5　Q-Former vs LLaVA Projector：trade-off

| 维度 | LLaVA Projector | BLIP-2 Q-Former |
| --- | --- | --- |
| 参数量 | ~20M (MLP) | ~180M (Q-Former + queries) |
| 计算 | 仅 MLP forward | 12 层 cross-attn forward |
| 视觉 token 数 | $N$（随分辨率二次增长） | 固定 32 |
| 信息损失 | 几乎 0（每个 patch 都进 LLM） | 显著（256+ patch 压成 32） |
| 训练复杂度 | 1 stage（pretrain）+ 1 stage（IT） | 2 stage（表征 + 生成）；stage 1 同时优化 ITC + ITM + ITG 三个 loss |
| LLM context 占用 | 大（576–2880 token） | 小（32 token） |
| 适合场景 | 高分辨率 / 细节任务 | LLM context 紧张 / 多模态批量推理 |

> 💡 **2024–2025 主流回归 projector** — Qwen2-VL / LLaVA-NeXT / InternVL-2 / DeepSeek-VL2 几乎都用 projector（带 spatial reduction / pixel shuffle 控 token 数；原始 Qwen-VL 用的是单层 cross-attention query resampler，是 Qwen2-VL 才改用带 pixel shuffle 的 MLP patch merger），**Q-Former 在工业 VLM 中淡出**。但 Q-Former 思路在 video VLM 里仍活跃（用 query 做 frame-level pooling）。

## §8 Flamingo：Perceiver Resampler + Gated Cross-Attn

### 8.1　设计目标

Alayrac et al. 2022 (NeurIPS) 想要：**在 frozen 70B LLM 上加视觉能力，不破坏文本能力**。设计选择：

- **不重训 LLM**：完全 frozen，只在中间插入新层

- **少量可训参数**：cross-attn 层 + Perceiver Resampler

- **few-shot interleaved**：训练数据是 `(image, text, image, text, ...)` 交错序列

### 8.2　Perceiver Resampler

类似 Q-Former 的"用 latent query 压缩 image"。Flamingo 原论文 Sec 3.1 的伪代码是**多层堆叠**（每层 = cross-attention + FFN），论文默认配置约 $L=6$ 层。每一层的更新规则：

$$\mathbf{q}^{(\ell+1)} = \mathbf{q}^{(\ell)} + \text{CrossAttn}\!\left(\mathbf{q}^{(\ell)},\ [\mathbf{q}^{(\ell)};\ \mathbf{Z}],\ [\mathbf{q}^{(\ell)};\ \mathbf{Z}]\right), \quad \mathbf{q}^{(\ell+1)} = \mathbf{q}^{(\ell+1)} + \text{FFN}(\mathbf{q}^{(\ell+1)})$$

注意 K/V 是 `concat(query, image_feat)` 而不只是 `image_feat`——让 query token 也能互相 attend。**整体仍然比 BLIP-2 Q-Former (12 层 + self-attn + cross-every-2) 轻**。输出 64 个 latent visual token（不依赖输入 patch 数）。

### 8.3　Gated Cross-Attention（核心创新）

在 LLM 每隔 $k$ 层（如每 4 层）插入一个新的 cross-attention 模块：

$$\mathbf{h}'_\ell = \mathbf{h}_\ell + \tanh(\alpha_\text{attn}) \cdot \text{CrossAttn}(\mathbf{h}_\ell, \mathbf{q}_\text{out}, \mathbf{q}_\text{out})$$
$$\mathbf{h}''_\ell = \mathbf{h}'_\ell + \tanh(\alpha_\text{ffn}) \cdot \text{FFN}(\mathbf{h}'_\ell)$$

**关键**：$\alpha_\text{attn}, \alpha_\text{ffn}$ 是 learnable scalar，**初始化为 0**。所以 $\tanh(0)=0$，新增 cross-attn 在初始化时**对 LLM 输出零贡献**——LLM 表现与未加视觉模块的 frozen LLM 完全一致。训练时 $\alpha$ 逐渐学到非零值，视觉信息开始注入。

> ✅ **这就是"residual 嫁接"** — Llama-3.2-Vision (Meta 2024) 沿用了完全相同的设计：frozen LLaMA-3 + 学一个 gated cross-attn adapter。优点是**完全保留 text-only 性能**，缺点是**视觉能力上限低于 fine-tune LLM 的 LLaVA/Qwen-VL**。

### 8.4　Flamingo / Llama-3.2-V vs LLaVA 对比

| 方面 | Flamingo / Llama-3.2-V | LLaVA / Qwen-VL |
| --- | --- | --- |
| LLM 是否解冻 | **否**（frozen） | 是（stage 2 解冻） |
| Image 作 token | 否（作 KV） | **是**（作 token） |
| Text-only 能力保留 | ✅ 完全保留 | ⚠️ 可能轻微退化 |
| 视觉理解上限 | 受限于 cross-attn 容量 | 更高（LLM 可"思考"图像） |
| 训练数据 | interleaved | image-instruction pair |
| 适用场景 | 大 LLM + 不想重训 | 中小 LLM + 视觉为核心 |

## §9 CogVLM 与"视觉专家" / Cross-attn fusion 变体

### 9.1　CogVLM：视觉专家分支

Wang et al. 2023 (CogVLM) 的核心：在 LLM 的 attention / FFN 里**复制一份并行分支**，专门处理 visual token，**与原 text 分支共享 attention 计算但走不同 projection**：

```
                  attention
       ┌──────────────┴──────────────┐
       ↓                              ↓
   text projection (frozen)    vision expert projection (trainable)
       │                              │
       └──────────────┬──────────────┘
                      ↓
         token-wise route: if visual_token, use vision branch
```

- **类似 MoE 的双专家**：text token 走 text 分支，image token 走"visual expert"分支

- **vision expert 单独训练**：text 分支可 frozen，视觉能力来自 expert

- **保留 LLM 原生 text 能力**（与 Flamingo 同思路，但 routing 粒度是 token 而非 layer）

### 9.2　Llama-3.2 Vision：Flamingo-style cross-attn 在大 LLM 上的复活

2024 年 9 月 Meta 发布 Llama-3.2-V (11B / 90B)：

- Frozen Llama-3 backbone

- 加入 **separate cross-attention layers**（不是改 self-attn）

- Adapter-style 训练，**只在 cross-attn 层学**

- 设计为"长尾视觉任务的 robust 基座"，视觉能力上限略低于 GPT-4V，但 **text-only benchmark 与 Llama-3 文本版几乎一致**

### 9.3　Claude 3.5/3.7 Sonnet Vision 与 GPT-4V/4o

Anthropic / OpenAI 的闭源模型架构未公开，但从 API 行为推断：
- **GPT-4V (2023.09) / GPT-4o (2024.05)**：4o 是 native multimodal，从底层就联合训练 (image + text + audio)，**不是 LLaVA 式后接 projector**
- **Claude 3.5/3.7 Sonnet (2024-2025)**：支持高分辨率图像（最多 8000×8000 像素，按需 tile），文档理解 (PDF/screenshot) 是其卖点之一
- **共同特征**：能处理多页文档 / 截屏 / OCR 数学公式——远超 LLaVA 系列。这暗示它们在训练数据规模（document corpus）+ tiling 策略上有重大优化

## §10 Qwen2-VL / DeepSeek-VL：动态分辨率 + M-RoPE

### 10.1　Native Dynamic Resolution

Qwen2-VL (Wang et al. 2024)、DeepSeek-VL (Lu et al. 2024)、InternVL-2 都抛弃了"resize 到固定 224²"的传统：

- **保留原始 aspect ratio**：把图按 patch_size 的整数倍 resize 到接近原尺寸的最大值

- **patch 数动态**：Qwen2-VL 的 `smart_resize` 先把每条边 round 到 28（$=2P$，匹配后续 2×2 spatial merge）的倍数，例如 $1024 \times 768$ 图先被 resize 到约 $1036 \times 756$；按 $P=14$ 切得 $74 \times 54 = 3996$ 个 patch，经 2×2 spatial merge 后得到 999 个视觉 token

- **不再使用固定 pos embed 表**：必须用 **可扩展的位置编码**（RoPE 或 2D ALiBi-like）

### 10.2　M-RoPE（Multimodal RoPE）

Qwen2-VL 的核心创新。**回顾普通 1D RoPE**：把 query / key 的每对维度 $(2k, 2k+1)$ 看作复数，乘上位置相关的旋转：

$$\mathbf{R}_{m,k} = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix}, \quad \theta_k = 10000^{-2k/d}$$

应用到 $\mathbf{q}_m$ 后，$\mathbf{q}_m^\top \mathbf{k}_n$ 只依赖 $m - n$（相对位置）。

**M-RoPE 的扩展**：一个视觉 token 有 (t, h, w) 三个位置维度。**所有 head_dim 都旋转**——但每对维度 $(2k, 2k+1)$ 根据所在区段，用 t / h / w 三个位置 id 之一参与旋转角度：

$$(\cos(m_\text{axis}\,\theta_k),\ \sin(m_\text{axis}\,\theta_k)), \quad \text{axis} \in \{t, h, w\}$$

具体地，Qwen2-VL 的 `mrope_section`（**单位是半 head_dim 对**，即每个数代表多少对 $(2k, 2k+1)$）。一对 = 2 个实数维度，所以"section sum × 2 = head_dim"。

> 💡 **Qwen2-VL 默认 `mrope_section = [16, 24, 24]`** — 即三个 axis 各占 16 / 24 / 24 对维度；总 $(16+24+24) \times 2 = 128 = $ head_dim。实现上把 section 翻倍成 $[16, 24, 24, 16, 24, 24]$ 沿 head_dim 切，分别用 (t, h, w, t, h, w) 的位置 id 旋转——**全部 128 维都参与旋转**，没有"不旋转 dim"。空间维（h, w）占 48 对 > 时间维（t）的 16 对；这是一种符合直觉的可能解释（时间变化通常慢于空间内容变化），但**并非 Qwen2-VL 论文明确证实的设计动机**——具体 16/24/24 切分比例也可能只是经验调参结果。

文本 token 没有显式 (h, w)：Qwen2-VL 让 $m_t = m_h = m_w$ 等于该 text token 的 1D 位置 id，三个 axis 给出**完全相同的旋转角**，等价于普通 1D RoPE。

### 10.3　Qwen2.5-VL 升级

Qwen2.5-VL（Bai et al. 2025）在 Qwen2-VL 基础上：
- **绝对时间编码**：M-RoPE 的 t 维改用真实时间戳（秒），不是帧 index，**支持任意 FPS 视频**
- **动态视觉 token budget**：根据任务复杂度调 token 数
- **agent / GUI 能力**：训练数据加入 web screenshot / mobile UI 操作 trace

### 10.4　DeepSeek-VL / VL2：高分辨率 tiling + Hybrid encoder

DeepSeek-VL (Lu et al. 2024) 用**双 vision encoder**：
- **SigLIP**：处理全局语义（低分辨率）
- **SAM-B**（Segment Anything backbone）：处理高分辨率细节

两路特征 concat 喂给 projector + LLM。**DeepSeek-VL2 (2024.12)** 进一步换成 MoE LLM + 动态分辨率，单 image 视觉 token 可达 1700+。

### 10.5　Code: M-RoPE 三维位置嵌入（核心 50 行，对齐 Qwen2-VL HF 实现）

```python
import torch

def build_mrope_cos_sin(positions, head_dim, mrope_section=(16, 24, 24), base=1000000.0):
    """
    Build cos/sin tensors for Qwen2-VL style M-RoPE.

    positions: LongTensor [3, B, L]   (axis 0: t / h / w; B batch; L seq len)
    head_dim:  per-head dim (must equal 2 * sum(mrope_section))
    mrope_section: tuple of 3 ints; each = number of (half-dim) entries per axis
    Returns: cos, sin both [B, L, head_dim], ready for LLaMA-style rotate_half.
    """
    assert 2 * sum(mrope_section) == head_dim, "2 * sum(mrope_section) must = head_dim"
    half = head_dim // 2                                                # = sum(mrope_section)

    # 标准 RoPE 频率: θ_k = base^{-2k/head_dim}, k = 0..half-1
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() * 2 / head_dim))   # [half]
    inv_freq = inv_freq.to(positions.device)

    # 对每个 axis 算 [B, L, half] 的 angle / cos / sin
    cos_axes, sin_axes = [], []
    for a in range(3):
        ang = positions[a].float().unsqueeze(-1) * inv_freq                     # [B, L, half]
        cos_axes.append(ang.cos())
        sin_axes.append(ang.sin())

    # 把 half-dim 按 mrope_section 切成 3 段，分别取 t/h/w 的 cos/sin
    cos_chunks, sin_chunks = [], []
    offset = 0
    for axis, s in enumerate(mrope_section):
        cos_chunks.append(cos_axes[axis][..., offset:offset+s])                 # [B, L, s]
        sin_chunks.append(sin_axes[axis][..., offset:offset+s])
        offset += s
    cos_half = torch.cat(cos_chunks, dim=-1)                                    # [B, L, half]
    sin_half = torch.cat(sin_chunks, dim=-1)

    # LLaMA-RoPE 风格 duplicate 到 full head_dim
    cos = torch.cat([cos_half, cos_half], dim=-1)                               # [B, L, head_dim]
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin

def rotate_half(x):
    """(x1, x2) -> (-x2, x1), LLaMA convention."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_mrope(q, k, cos, sin):
    """
    q, k:    [B, num_heads, L, head_dim]
    cos, sin:[B, L, head_dim]
    """
    cos = cos.unsqueeze(1)                                                       # broadcast over heads
    sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

> ⚠️ **M-RoPE 三个常见误读** — 容易踩的坑。

- **"head_dim 切成 t/h/w 三段独立 1D RoPE"**：错。Qwen2-VL 实际是 **6 段 alternating** `[s_t, s_h, s_w, s_t, s_h, s_w]`，全 head_dim 都旋转

- **"section 单位是 dim"**：错。`mrope_section=[16,24,24]` 单位是 **对数**（每对 = 2 个 head_dim 元素），$\sum \times 2 = $ head_dim = 128

- **"text token 没有 (h, w) 怎么办？"**：让 $m_t = m_h = m_w$ 等于 text 的 1D 位置 id，三个 axis 旋转角相同，退化回 1D RoPE

## §11 Video VLM：LongVA / VideoLLaMA / 长视频问题

### 11.1　基本 pipeline

视频 = 多帧 image。VLM 处理视频的常见做法：
1. **均匀采样 $K$ 帧**（如 8 / 16 / 32）
2. **每帧过 vision encoder** → 每帧 $N$ 个 patch token
3. **token 序列拼接**：$K \cdot N$ 个 visual token 喂入 LLM

问题：$K=32, N=576 \Rightarrow 18432$ token，单图 instruction tuning 的 LLM context 撑不住。

### 11.2　常见压缩策略

- **Time pooling**：相邻帧 token average pool（VideoChat / VideoLLaMA）
- **Q-Former resampler**：每帧用 query token 压成 32（Video-BLIP）
- **Token merge**：跨帧合并相似 token（VideoLLaMA 2）
- **Spatial pooling + temporal preservation**：每帧 patch token 池化到 $H' \times W'$，保留所有帧（LLaVA-NeXT-Video）

### 11.3　LongVA / Long-context video

LongVA (Zhang et al. 2024) 用**长上下文 Qwen2 LLM + CLIP 视觉塔 + 空间池化**压缩每帧 token，利用语言模型侧已有的长上下文扩展能力（标准 1D 位置编码扩展方法，**不是** Qwen2-VL/Qwen2.5-VL 专用的 (t,h,w) 分解式 M-RoPE）直接吃很长的展开视觉 token 序列，做几小时视频问答。Qwen2-VL 报告可处理 20 分钟视频；Qwen2.5-VL 推到 1 小时+。

### 11.4　Video benchmark

- **MVBench** (Li et al. 2024)：20 个 fine-grained video understanding 任务
- **Video-MME** (Fu et al. arXiv 2024 / CVPR 2025)：900+ video，包含 3 个时长档（短/中/长）+ 6 类任务
- **EgoSchema** (Mangalam et al. 2023 NeurIPS)：第一人称长视频
- **LongVideoBench** (Wu et al. 2024 NeurIPS)：从 8 秒到 1 小时

## §12 训练 pipeline：alignment / instruct / preference

### 12.1　Stage 1：Alignment / Pre-training

目的：让视觉特征"对齐"到 LLM token 空间附近。
- **数据**：image-caption pair（CC3M, LAION-558K, ShareGPT4V）
- **训练**：只解冻 projector（LLaVA）/ Q-Former (BLIP-2)，**vision tower + LLM 冻结**
- **Loss**：LLaVA 用 next-token prediction（用 LLM 把 visual feature 作 prefix，生成 caption）；**BLIP-2 此处不接 LLM**——Q-Former 的 Stage 1 是表征学习，在 frozen image encoder 上联合优化 ITC + ITM + ITG，frozen LLM 要到 Stage 2（generative learning）才接入

### 12.2　Stage 2：Visual Instruction Tuning

目的：教会 VLM "看图回答问题、follow instruction"。
- **数据**：GPT-4 生成的 visual instruction（LLaVA-Instruct-158K, ShareGPT4V）+ 学术 VQA 数据（VQAv2, GQA, OCR-VQA, TextVQA）
- **训练**：解冻 LLM + projector，**vision tower 一般保持冻结**（LLaVA-1.5 / Qwen-VL）。**Qwen2-VL 在最后阶段也解冻 vision tower**做 end-to-end fine-tune
- **Loss**：next-token prediction on answer tokens（input image + question 不算 loss）

### 12.3　Stage 3：Preference / RLHF

目的：减幻觉、提升 helpfulness / harmlessness、长尾任务对齐。

| 方法 | 时间 | 核心 |
| --- | --- | --- |
| **LLaVA-RLHF** | 2023.09, Sun et al. | PPO + human preference + hallucination-aware reward |
| **RLAIF-V** | 2024, Yu et al. | AI feedback 替代 human label, divide-and-conquer |
| **POVID** | 2024 | DPO + 故意构造的幻觉 negative |
| **VLM-R1** | 2025 | GRPO + verifiable reward（视觉 reasoning 类 R1） |
| **Bespoke / R1-Onevision** | 2025 | 视觉 chain-of-thought + RL refinement |

> 💡 **VLM-R1 (2025) 是当下热点** — 把 DeepSeek-R1 的"verifiable reward + GRPO"配方迁到视觉任务（如 ScienceQA, MMMU）。reward 来自答案是否匹配 ground truth（不用 process reward model），训练后 visual reasoning chain 显著增长，benchmark 大幅提升。

### 12.4　数据规模 vs 阶段

| 阶段 | 数据量 | 训练 token | 解冻部分 |
| --- | --- | --- | --- |
| Alignment | 0.5–5M caption | 1–10B | projector |
| Instruction tune | 0.2–10M instruction | 1–50B | LLM + projector |
| Preference | 50k–500k preference pair | 100M–1B | LLM (LoRA / full) |

## §13 Multimodal Embeddings：BGE-VL / Jina-CLIP / VLM2Vec

### 13.1　为什么需要新一代多模态 embedding？

CLIP 训练目标是 "image ↔ short caption" 对齐，**长 instruction 检索 / multi-image / interleaved 文档检索** 表现差。新一代 multimodal embedding 模型针对 retrieval / RAG 场景。

### 13.2　代表方法

- **Jina-CLIP-v1 (2024)**：在 CLIP 基础上加长文本对比（text-text task）+ 多分辨率，单 embedding 模型既能做 image-text 也能做 text-text retrieval
- **BGE-VL (2024)**：BGE 团队的 multimodal 版本，用 SigLIP-So400M + small LLM 做 instruction-aware retrieval
- **VLM2Vec (Jiang et al. 2024)**：取 VLM（LLaVA / Qwen-VL）输出序列**最后一个 token** 的 hidden state 作为 embedding（沿用 causal LM 文本 embedding 的常见做法），并用 **LoRA 对整个 VLM 做对比学习微调**（而非冻结 backbone 只训一个额外的 contrastive head），benchmark MMEB 上显著超过 CLIP
- **mmE5 (2024)**：multimodal 版本的 E5，支持 12 种 retrieval 任务类型

### 13.3　核心 trick

- **instruction-aware**：embedding 输入 `[instruction][image][query]`，让同一图像在不同任务下产生不同 embedding
- **VLM-as-encoder**：直接用 instruct-tuned VLM 做 backbone，不再单独 contrastive pretrain
- **hard negative mining**：用 cross-encoder rerank 挖掘 batch 之外的 hard negative

## §14 25 高频面试题（L1 必会 · L2 进阶 · L3 顶级 lab）

### L1 基础题（必会 10）

<details>

<summary>Q1. CLIP 的 loss 是什么？为什么必须对称？</summary>

- CLIP 用 **symmetric InfoNCE**：$\mathcal{L} = \tfrac12(\mathcal{L}_{i\to t} + \mathcal{L}_{t\to i})$

- $\mathcal{L}_{i\to t}$：对相似度矩阵 $\mathbf{S}$ 做**行 softmax**，取对角项的 NLL（image 检索 text）

- $\mathcal{L}_{t\to i}$：**列 softmax**，对角项 NLL（text 检索 image）

- **对称必要性**：单向 loss 中相似度 $S_{ij}$ 本就同时依赖 image / text 两个 encoder 的输出，两者都会收到梯度；对称化的真正价值是**显式同时优化 image→text 与 text→image 两个检索方向**，经验上提升双向检索质量，而不是防止某种数学意义上的表征坍塌

只答 InfoNCE 而不说"两次 softmax 平均"，或者说"反着算一遍"而不解释为什么必要。

</details>

<details>

<summary>Q2. CLIP 中的 temperature τ 起什么作用？为什么要 learnable？</summary>

- 温度 $\tau$ 调节 softmax 锐度：$\tau \to 0$ 像 one-hot 只关注 hardest negative；$\tau \to \infty$ 均匀分布、几乎无梯度

- OpenAI CLIP 把 $\tau$ 设成 learnable（实际 parameterize 为 `logit_scale = log(1/τ)`，反向更稳）

- 训练后稳态 $\tau \approx 0.01$（`logit_scale ≈ log(100)`），并 clamp 上界防止崩

- **不 learnable 的话**：超参敏感，不同数据 / 模型规模都要手调

把 τ 当固定 0.07 不学习；或写成 `1/exp(logit_scale)` 反向不稳。

</details>

<details>

<summary>Q3. SigLIP 相对 CLIP 的核心改动？为什么 batch size 不再敏感？</summary>

- 把 softmax InfoNCE 换成 **sigmoid binary CE**：每个 $(i,j)$ 配对独立分类

- $S_{ij} = t \cdot \mathbf{u}_i^\top \mathbf{v}_j + b$；标签 $y_{ij} = +1 (i=j) / -1 (i\neq j)$；loss = $-\sum_{ij} \log\sigma(y_{ij} S_{ij})/N$

- **batch 解耦**：每项 loss 不依赖 row/col 的归一化，所以 N 翻倍只改变负样本数量，不改变 loss 形状

- 工程收益：单机大 batch、跨节点通信简化、小 batch 也能学（CLIP 小 batch 几乎不收敛）

只说"用 sigmoid"，没解释为什么 batch-independent；或把 SigLIP 当成"加 bias 的 CLIP"。

</details>

<details>

<summary>Q4. ViT 中 [CLS] token 是必须的吗？</summary>

- 不是。**ViT 原版**用 [CLS] 是为了和 BERT 习惯对齐

- **替代方案**：把所有 patch token mean-pool 作为图像表征——多数现代 ViT (DeiT-III, SigLIP, EVA-CLIP) 都用 mean-pool 或 attentive pool

- **CLIP 用 [CLS]**：因为对比训练需要单一向量

- **VLM 视觉 tower 一般 drop [CLS]**：LLaVA 取倒数第二层 patch token，[CLS] 在 LLM 端不必要

把 [CLS] 当成 ViT 的"必须组件"；或说"没 [CLS] 就不能分类"（错，mean-pool 也行）。

</details>

<details>

<summary>Q5. LLaVA 的 projector 是什么？为什么用 MLP 而不是 Linear？</summary>

- Projector 把 visual encoder 输出（$d_v$=1024）投到 LLM token 空间（$d_\text{llm}$=4096）

- LLaVA-1.0：单层 `Linear(1024, 4096)`；LLaVA-1.5：**2-layer MLP + GELU**

- MLP 增加非线性表达，让 vision feature 更灵活地映射到 LLM "词典"

- 论文报告 MLP 在 MM-Vet / SEED-Bench 上比单层 Linear 提升 1–3 个点

只说"Linear 投一下"；或答"用 Q-Former"（那是 BLIP-2 不是 LLaVA）。

</details>

<details>

<summary>Q6. LLaVA 的两阶段训练分别在做什么？</summary>

- **Stage 1 Feature Alignment**：只训 projector，**冻结 vision tower + LLM**，用 caption 数据（CC3M / LAION-558K），让 visual feature 投到 LLM embedding space 附近

- **Stage 2 Instruction Tuning**：解冻 LLM + projector（vision tower 仍冻结），用 GPT-4 生成的 158K 视觉 instruction，**让 LLM 学会 follow visual instruction**

- 为什么不一步训：直接 stage 2 容易让 LLM 灾难遗忘文本能力；先 stage 1 给 visual token 一个"靠近文本 token"的初值再 instruct，更稳

把两个 stage 都说成"训 projector"；或漏掉 stage 1 冻结 LLM 这关键点。

</details>

<details>

<summary>Q7. Q-Former 是什么？相对 LLaVA projector 优劣？</summary>

- BLIP-2 的 Q-Former：12 层 Transformer，**32 个 learnable query token** 通过 cross-attention 从 frozen image encoder 取信息，输出固定 32 个 visual token

- 优点：**视觉 token 数固定**，LLM context 占用低，分辨率 ↑ 计算预算不变

- 缺点：**信息损失大**（256 patch 压成 32）、参数更多（~180M）、训练复杂（两阶段：stage 1 表征学习含 ITC+ITM+ITG 三个 loss，stage 2 对接 frozen LLM 做生成）

- **2024 主流回到 projector**：Qwen2-VL / LLaVA-NeXT / InternVL-2 都用 projector（原始 Qwen-VL 是单层 cross-attention resampler，Qwen2-VL 才改用 MLP patch merger）

把 Q-Former 当 projector 同义词；或不知道现代 VLM 倾向 projector。

</details>

<details>

<summary>Q8. 一个 ViT-L/14 在 224×224 图上有多少 patch？token 数？</summary>

- patch 数 = $(224/14)^2 = 16^2 = 256$

- token 数 = 256 + 1 (含 [CLS]) = **257**

- 若是 LLaVA 取倒数第二层 patch token（drop [CLS]）= **256 visual token**

- 若分辨率换成 336（LLaVA-1.5）：$(336/14)^2 = 24^2 = 576$ token

算错 $(H/P)^2$（把 $P^2$ 当成 patch 数 $N$）；漏掉 [CLS]。

</details>

<details>

<summary>Q9. 为什么 CLIP 不擅长 OCR / 计数 / 空间关系？</summary>

- **OCR 弱**：caption 一般描述场景，不读图中文字；CLIP 没有像素级 OCR 监督

- **计数弱**：caption 很少精确报数（"几只鸟"通常说"a flock of birds"）；embedding 空间没保留计数信号

- **空间关系弱**："cat on top of dog" 和 "dog on top of cat" 在 bag-of-words 视角下几乎一样；Yuksekgonul et al. 2023 (ICLR) 用 ARO benchmark 量化了这一点

- **改进方向**：DETR 风格的局部对齐、SigLIP-2 的 dense local objective、文档级数据

把 OCR 弱归咎于"分辨率不够"（部分对，但根因是数据 + loss）；说"CLIP 是 bag-of-words"过于绝对。

</details>

<details>

<summary>Q10. 训练 VLM 时为什么一般冻结 vision tower？</summary>

- vision tower（如 CLIP ViT-L）在自己的数据上**已经预训练**得很好；解冻容易破坏 visual feature 质量

- 训练数据量级远小于 CLIP 预训练（百万 vs 数十亿），解冻很容易过拟合

- **冻结也能省显存**：vision tower 几亿参数不用存 optimizer state

- **Qwen2-VL 例外**：最后阶段会解冻 vision tower 做小学习率 fine-tune，配合大量混合数据避免遗忘

直接答"不能解冻"——错，**末期可以小心解冻**。

</details>

### L2 进阶（10 题）

<details>

<summary>Q11. SigLIP 的 bias $b$ 为什么初始化为 $-10$？</summary>

- 训练初期 embedding 接近随机，$\mathbf{u}^\top \mathbf{v} \approx 0$，sigmoid 输出 0.5

- N×N 矩阵里负样本占 $N^2 - N \approx N^2$，正样本只 $N$ 个；若初始预测全 0.5，**负样本梯度主导**，正样本几乎得不到正确信号

- 初始化 $b \approx -10$ → $\sigma(b) \approx 4.5e^{-5}$ → 初期所有点先被预测为负

- 这样**负样本几乎没 loss**，正样本 loss 大（被预测为负但实际是正），梯度集中拉近正样本，训练稳定

只说"避免数值问题"；或答"对称项"（错，bias 不是对称损失项）。

</details>

<details>

<summary>Q12. Flamingo 的 gated cross-attn 为什么初始化为 0？</summary>

- 新增 cross-attn 输出乘 $\tanh(\alpha)$，$\alpha$ 初始化为 0

- $\tanh(0) = 0$，**初始化时新模块对 frozen LLM 输出零贡献**——LLM 表现与未加视觉模块的纯 text Llama 完全一致

- 训练过程中 $\alpha$ 慢慢从 0 长出来，视觉信号逐步注入

- 优点：完全保留 frozen LLM 的 text-only 能力；缺点：视觉能力上限受 cross-attn 容量限制

- Llama-3.2 Vision 沿用同一设计

把 0 初始化当成"普通 init trick"；或不知道这关乎 frozen LLM 的 capability preservation。

</details>

<details>

<summary>Q13. LLaVA-1.6 / NeXT 的 AnyRes 怎么实现？</summary>

- LLaVA-NeXT/1.6 **训练阶段就采用 AnyRes 网格切图**（不是仅推理时的 trick）：把高分辨率图按 aspect ratio 划成 $n \times m$ 个 336² tile（如 $2\times 2, 2\times 3$ 等）

- 每个 tile 独立过 frozen ViT 得到 576 token，加一个 **全局缩略图**（整图 resize 到 336² 编码）

- 拼接：约 $(1 + n\cdot m) \times 576$ visual token 喂入 LLM（真实实现还有 unpad 去除 padding tile 区域 + tile 行间插入 image_newline token，实际 token 数略少于该理论值）

- 选择哪种切法：从预定义的 grid 集合（如 $\{1\times 1, 2\times 2, 1\times 4, 4\times 1, ...\}$）里选最接近原 aspect ratio 的

把 AnyRes 当 dynamic resolution 同义词——技术上不同。Qwen2-VL 是 native dynamic（patch 数完全自由），LLaVA-1.6 是 **fixed-tile composition**。

</details>

<details>

<summary>Q14. Qwen2-VL 的 M-RoPE 三维分配为什么不平均？</summary>

- Qwen2-VL `mrope_section = [16, 24, 24]`（单位是**半 head_dim 的对数**），$\sum \times 2 = $ head_dim = 128

- **全部 128 维都旋转**——只是不同维度对用不同 axis (t/h/w) 的位置 id 算旋转角

- **不平均的可能原因**（符合直觉的解释，但并非 Qwen2-VL 论文明确证实的设计动机——具体 16/24/24 切分比例也可能只是经验调参结果）：

  - 视频帧间变化较慢（相邻帧很相似），所以 $s_t = 16$ 占比小

  - 空间内 patch 间变化剧烈（同一帧不同位置内容差异大），$s_h = s_w = 24$ 各需更多频率覆盖

- $s_h = s_w$：图像 H/W 维度地位对称

把 section 当成"维度数"（错，单位是对数 = head_dim / 2 的分配）；或以为"剩余维度不旋转"。

</details>

<details>

<summary>Q15. 为什么 BLIP-2 选 32 个 query token？</summary>

- 32 是经验值，是 LLM context 占用 vs 信息表达 的权衡

- 太少（< 16）：信息损失大，VQA / 细节任务下降

- 太多（> 64）：LLM context 占用大、Q-Former cross-attn 计算贵

- BLIP-2 论文 ablation 显示 32 在大多数下游任务上是 sweet spot

- 设计上类似 Perceiver（也是 latent query 数压缩输入）

只答"经验值"；不知道这是 context budget vs information capacity 的工程权衡。

</details>

<details>

<summary>Q16. DDP 下的 CLIP 怎么算 InfoNCE？</summary>

- 单卡 batch $N_\text{local}$，N 卡总 batch $N = K \cdot N_\text{local}$

- 每卡 forward 后 `dist.all_gather` 拿到全部 GPU 的 image / text feats

- 算 $\mathbf{S} \in \mathbb{R}^{N \times N}$ 的全局相似度

- 但反向时**只让本卡那 $N_\text{local}$ 行 / 列 contribute 梯度**（避免重复 backward）

- 这就是 OpenCLIP 的 `local_loss=True` 选项

只说 all-gather；不知道反向需要避免重复计算；或以为反向也要 all-gather 一遍（错，反向走通讯反向 path）。

</details>

<details>

<summary>Q17. 为什么 Llama-3.2-V 比 LLaVA-Qwen-VL 视觉能力上限低？</summary>

- Llama-3.2-V 用 frozen LLM + gated cross-attn adapter，**LLM 权重不变**

- LLaVA / Qwen-VL **解冻 LLM**，LLM 内部 attention 可以重组、专门处理视觉 token

- 后者 LLM 能"用 self-attention 思考图像"；前者 LLM 只能被动接收 cross-attn 注入的视觉信号

- trade-off：Llama-3.2-V 完美保留 text 能力，LLaVA-Qwen 可能轻微退化但视觉上限高

只说"参数少"；不知道这是 architecture-level 的能力上限差异。

</details>

<details>

<summary>Q18. 视觉 instruction tuning 数据为什么很多用 GPT-4 生成？</summary>

- 原始 caption 数据（CC3M / LAION）短、不 instruction-style，**学不出 dialog 能力**

- 真实人工标注 visual instruction（如 VQAv2 question）规模小、风格单一

- **GPT-4（当时是纯文本版，早于 GPT-4V 公开发布）并未直接看图**，而是被喂入图像的文本化表示——caption 加上目标检测得到的 bounding box 信息——用这些符号化信息 prompt **生成多轮对话 / 推理任务 / 详细描述**：LLaVA-Instruct-158K 就是这么来的

- 同时用 prompt engineering 控生成数据的覆盖（detailed description, conversation, complex reasoning 三类）

只答"数据多"；不知道这是 instruction style + diversity 的关键瓶颈。

</details>

<details>

<summary>Q19. CLIP / SigLIP 训练时 batch size 一般多大？</summary>

- **OpenAI CLIP**：32k batch（256 GPU × 128/GPU 左右）

- **OpenCLIP**：到 90k batch（LAION-2B）

- **SigLIP**：典型 32k batch 即足够，sigmoid loss 让每个 (i,j) 项独立、避免 softmax 的 batch-wide 同步；论文有扫描到 256k batch 的实验，但收益边际递减

- **小 batch 不行的原因**：InfoNCE 互信息下界 $I(U;V) \ge \log N - \mathcal{L}$，N 越大下界越紧；同时负样本数量决定 contrastive 的难度

- SigLIP 把 batch 解耦后，小 batch 表现显著提升（1k batch 也能学到合理 embedding）

答"几百"；或不知道 batch 与 InfoNCE 的理论联系。

</details>

<details>

<summary>Q20. POPE / Winoground / MMBench / MMMU 分别在测什么？</summary>

- **POPE** (Li et al. 2023)：测**物体幻觉**——VLM 是否会说图里有不存在的物体（yes/no 二分类）

- **Winoground** (Thrush et al. 2022)：测**组合性 / 词序敏感**——"cat on dog" vs "dog on cat" 能否区分

- **MMBench** (Liu et al. 2023)：通用多模态评估，~3000 题覆盖 OCR / 物体识别 / 推理等

- **MMMU** (Yue et al. 2024 CVPR)：大学级专业知识题（数学 / 物理 / 医学等），考多模态推理

- **MM-Vet** (Yu et al. 2023)：6 种 capability 综合评估（识别 / 知识 / OCR / 空间 / 语言 / 数学）

把 POPE 和 MMBench 混用；不知道 Winoground 是"组合性 stress test"。

</details>

### L3 高级（顶级 lab / 研究方向，5 题）

<details>

<summary>Q21. 推 CLIP 对称 InfoNCE = 行+列 softmax 平均，并解释为什么 SigLIP 能 batch-independent。</summary>

设 batch 大小 $N$，相似度矩阵 $S_{ij} = \mathbf{u}_i^\top \mathbf{v}_j / \tau$。

**CLIP 推导**：

行方向 softmax，$p_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}$。Image→Text 的 NLL：

$$\mathcal{L}_{i\to t} = -\frac{1}{N}\sum_i \log p_{ii} = -\frac{1}{N}\sum_i \log \frac{\exp(S_{ii})}{\sum_j \exp(S_{ij})}$$

列方向 softmax（Text→Image）：

$$\mathcal{L}_{t\to i} = -\frac{1}{N}\sum_j \log \frac{\exp(S_{jj})}{\sum_i \exp(S_{ij})}$$

对称 loss：$\mathcal{L} = \tfrac12 (\mathcal{L}_{i\to t} + \mathcal{L}_{t\to i})$。注意 **梯度对 $S_{ij}$**：

$$\frac{\partial \mathcal{L}_{i\to t}}{\partial S_{ij}} = \frac{1}{N}(p_{ij} - \delta_{ij})$$

每个 $S_{ij}$ 的梯度依赖**整行的 softmax 归一化** $\sum_k \exp(S_{ik})$。所以 N 改变（新增 / 删除 negative）会改变整行所有 $p$ 的值——**梯度耦合 batch**。

**SigLIP 推导**：

$S_{ij} = t \cdot \mathbf{u}_i^\top \mathbf{v}_j + b$，$y_{ij} = 2\delta_{ij} - 1$，

$$\mathcal{L}_\text{SigLIP} = \frac{1}{N}\sum_{i,j} \log(1 + \exp(-y_{ij} S_{ij}))$$

梯度：

$$\frac{\partial \mathcal{L}}{\partial S_{ij}} = \frac{1}{N}\cdot \frac{-y_{ij}}{1 + \exp(y_{ij} S_{ij})} = \frac{1}{N}\cdot (-y_{ij}) \cdot \sigma(-y_{ij} S_{ij})$$

**关键**：$\partial \mathcal{L} / \partial S_{ij}$ 只依赖 $S_{ij}$ 本身，**不涉及其他元素**。所以新增 negative 不会改变已有 $S_{ij}$ 的梯度——**batch-independent**。

**工程含义**：
- CLIP：DDP 必须 all-gather embedding 算全局 logsumexp，通信 $O(N \cdot D)$，加 sync 点
- SigLIP：可用 **chunked all-pair**，每个 chunk 只算本地行 × 远端列的 sigmoid 项，无 sync logsumexp

</details>

<details>

<summary>Q22. Q-Former vs LLaVA projector 的 trade-off：从 capacity / compute / training stability 三个维度解释。</summary>

**Capacity（信息容量）**：

- **LLaVA projector**：所有 $N$ 个 patch token 都进 LLM；信息无瓶颈，但 LLM context 占用大

- **Q-Former**：32 query 是固定瓶颈，信息显著压缩；对细节任务（OCR / 计数）不友好

- 设 visual encoder 输出秩为 $r$；LLaVA 的 2 层 MLP + GELU projector 是逐行非线性变换，**不满足线性代数意义上的秩上界**（不能简单写成秩 $\le r$）；Q-Former 输出的秩由固定的 32 行 query 决定上界 $\le \min(32, d_q)$，这一瓶颈与输入的 $r$ 并无直接的 $\min(r,32)$ 关系——这里更多是**直观类比**而非严格推导

**Compute / Memory**：

- **LLaVA projector**：仅 MLP forward，O(N·D²) 计算

- **Q-Former**：12 层 cross-attn + self-attn + FFN，~180M 参数；但**下游 LLM 端 context 短（32 token vs N=256+ token）**，LLM 推理快

- **总成本权衡**：图像分辨率高时（N=2880），Q-Former 节省 LLM 推理；图像低分辨率时 LLM 占主导，LLaVA 更便宜

**Training stability**：

- **LLaVA**：projector 容易训（2 stage），梯度路径短

- **Q-Former**：2 stage 训练（**stage 1 表征**含 ITC + ITM + ITG 三个 loss 同时优化；**stage 2 生成**对接 frozen LLM），ITM head 易过拟合；ITG 需要 causal mask 与 self-attn mask 的复杂 routing，工程坑多

**结论**：2024 主流回到 projector + spatial pixel-shuffle / merging 控 token 数，Q-Former 仅在 video / 多图汇总场景仍有价值（用 query 做 temporal pool）。

</details>

<details>

<summary>Q23. Qwen2-VL 的 M-RoPE 配 `mrope_section = [16, 24, 24]` 为何不是 1:1:1？所有 head_dim 都旋转吗？</summary>

回顾普通 RoPE：head_dim $d$ 维分成 $d/2$ 对复数，频率 $\theta_k = \text{base}^{-2k/d}$。**频率覆盖范围决定能区分的最大相对距离**：低频区分远距离，高频区分近距离。

**关键 disambiguate**：Qwen2-VL `mrope_section` 单位是 **半 head_dim 的对数**（每个数代表多少对 $(2k, 2k+1)$ 维度对）。$[16, 24, 24]$ 表示 t / h / w 三个 axis 分别占 16 / 24 / 24 对维度，$\sum \times 2 = 128 = $ head_dim。HF 实现把 section 翻倍成 $[16, 24, 24, 16, 24, 24]$ 切 head_dim，对应 axis 序列 $(t, h, w, t, h, w)$——**全部 128 维都旋转**，没有"不旋转 dim"。

设计权衡：

1. **时间维变化慢**：典型视频 1–5 FPS 采样，相邻帧很相似，长程时间依赖需求中等。$s_t=16$（占 25%）足够覆盖几百到上千帧。

2. **空间维变化剧烈**：同一帧内不同 patch 视觉差异极大；要在 $\sim 1000\times 1000$ 像素图上做 token 间检索，需要更多频率档位。$s_h = s_w = 24$（各占 37.5%）覆盖更广。

3. **空间对称性**：$s_h = s_w$ 保持图像 H/W 维度地位对称（水平翻转、垂直翻转的等价性）。

4. **6 段 alternating 而不是 3 段连续**：因为 RoPE 用 LLaMA "rotate_half" 实现，head_dim 在内存上分成两半 $[h_1, h_2]$，旋转用 $q \mapsto q \cos + \text{rotate\_half}(q)\sin$；两半对应同一组 inv_freq。所以 axis 分配既要在前半也要在后半镜像。

**Qwen2.5-VL 升级**：把 $m_t$ 从帧 id 改为**绝对时间戳（秒）**，让训练时变 FPS 的视频共享一致的时间坐标——这是 long-video 关键。

**alternative**：DeepSeek-VL2 用扁平化 visual token + 普通 1D RoPE（不分 h, w）；Llama-3.2-V 同样不显式分时空。**M-RoPE 仅在 native interleaved video + image 场景下显著优于扁平展开**。

</details>

<details>

<summary>Q24. VLM 幻觉的根本原因？现有缓解方法的优劣？</summary>

**根本原因**：

1. **数据偏差**：训练数据中常出现的 "co-occurrence prior"——"图里有桌子大概率有椅子"。Co-occurrence 让 VLM 在看到桌子时倾向回答"是的，有椅子"，即使图里没椅子

2. **语言先验主导**：当视觉信号弱时（小物体、模糊、奇怪角度），VLM 退化为纯语言模型，按"语料常识"作答

3. **LLM 的 sycophancy**：用户问 "图里是不是有 X" 倾向回答 Yes（人类反馈偏向 helpful → 倾向 yes）

4. **Stage 2 instruction tuning 没有 negative supervision**：标注里很少教 "图里没 X 就回答 No"

**缓解方法**：

| 方法 | 思路 | 优势 | 劣势 |
| --- | --- | --- | --- |
| LLaVA-RLHF | PPO + hallucination-aware reward | 训练后期定向修 | 需要 reward model + 大量 preference |
| RLAIF-V | AI-generated preference | 数据成本低 | reward model 自身偏差累积 |
| POVID | DPO + 构造 hallucination negative | 直接对症 | 需精心设计 negative |
| VCD (visual contrastive decoding) | 推理时让 VLM 同时看图 vs 模糊图，差值放大视觉信号 | 训练免费 | 推理 2x 成本 |
| OPERA | beam search + over-attention 检测 | 推理时检测 | 启发式，可能误杀 |
| POPE 评测驱动 | 用 POPE 反向监督 | 量化好 | 只测 object hallucination |

**未来方向**：从训练数据层根治（grounded caption + segment-level 监督）；visual chain-of-thought（VLM-R1 风格）让模型在回答前先"指认证据"。

</details>

<details>

<summary>Q25. 现代 VLM 训练里 vision tower 应该用 SigLIP 还是 CLIP？为什么 2024–2025 大多选 SigLIP-So400M？</summary>

**Empirical 结论**：2024 后 SigLIP-So400M 成为 open-weight VLM 的常见选择，典型代表是 **PaliGemma**（Google）和 **LLaVA-OneVision**。但 **Molmo** 仍用 OpenAI CLIP（其论文 ablation 比较过 SigLIP）；**InternVL 系列**用自家的 InternViT；**Qwen2-VL** 视觉端从 DFN-derived ViT 初始化再做大规模 vision-language 联合训练；**LLaVA-1.5 / 1.6** 仍用 CLIP ViT-L/14。即"切到 SigLIP"不是行业共识。

**SigLIP-So400M 的吸引力**：

1. **零样本性能更强**：SigLIP-So400M 在 ImageNet zero-shot 与同规模 CLIP 对比有 4–8 个点优势，visual feature 质量更高

2. **分辨率友好**：SigLIP 训练时已用大量 384²/512² 数据；CLIP 主要 224²+336²，VLM 任务普遍需高分辨率，SigLIP 迁移更顺

3. ~~batch-independent loss → fine-tune 稳定~~：这条论证站不住——VLM 的 stage 1 alignment 通常本就**冻结 vision tower**，训练用的是 next-token prediction loss（见 §12.1），完全不涉及任何对比损失；SigLIP 的 batch-independent 性质只存在于它自身的图文对比预训练阶段，不会随着接入 VLM 自动带入下游自回归训练。若要论证 SigLIP vision tower 更适合下游微调，应换用站得住脚的理由（如更高分辨率训练数据、更强 zero-shot 特征质量，见上文第 1、2 点）

4. **多语言支持**：SigLIP-2 / mSigLIP 原生支持多语言

5. **开放权重**：Google 公开 SigLIP / SigLIP-2 全套 checkpoint（OpenAI CLIP 也已开源，但选择有限）

**何时仍选 CLIP**：
- 需要严格对齐 OpenAI CLIP 行为（如 Stable Diffusion 风格的 CLIP guidance）
- 项目兼容（早期 LLaVA-1.0/1.5 + DALL-E 用 CLIP）

**注意**：SigLIP 不是万能；**DeepSeek-VL 用 SigLIP + SAM dual-encoder**——细节定位任务上 SAM 特征仍有不可替代的优势。

</details>

## §A 附录：Sanity-check 输出 & 参考文献

### A.1 关键代码 sanity check（实跑示意）

```
[ViT] patch_embed: (2, 3, 224, 224) -> (2, 196, 768)  ✓
[ViT] forward + CLS: (2, 3, 224, 224) -> head out (2, 1000)  ✓

[CLIP] N=8, D=512, init logit_scale=ln(1/0.07)→1/τ≈14.285: loss ≈ 2.25 (略高于 log(N)=2.08；loss≈log N 仅在 1/τ 极小时成立)  ✓
[CLIP] forward + backward: gradients along i→t 与 t→i path 对称 ✓

[SigLIP] N=8, D=512, b=0:  loss = sum_{ij} log(1+e^0) / N = 64 * log 2 / 8 ≈ 5.545  ✓
[SigLIP] bias b=-10: positive (8 项) loss ≈ log(1+e^10) ≈ 10; negative (56 项) loss ≈ 4.5e-5; 整体 8·10/8 ≈ 10.0  ✓ (正样本主导初期梯度)

[LLaVA] visual feat (2, 256, 1024) -> projector -> (2, 256, 4096)  ✓
[LLaVA] input_ids w/ <image> placeholder: 1 token -> 256 visual tokens after merge  ✓

[Q-Former] image_feats (2, 257, 1408), queries (1, 32, 768) -> out (2, 32, 768)  ✓

[M-RoPE] head_dim=128, mrope_section=[16,24,24]: 2 × sum = 128 = head_dim, full rotation ✓
[M-RoPE] pure-text token (pos_t=pos_h=pos_w=m): cos/sin 三个 axis 完全相同, 等价 1D RoPE ✓
```

### A.2 关键文献（按主题）

**视觉 encoder**：Dosovitskiy et al. ICLR 2021 (ViT); Zhai et al. CVPR 2022 (ViT-g); Fang et al. arXiv 2023 (EVA-02)

**对比预训练**：Radford et al. ICML 2021 (CLIP); Cherti et al. CVPR 2023 (OpenCLIP); Zhai et al. ICCV 2023 (SigLIP); Tschannen et al. arXiv 2025 (SigLIP-2); Gadre et al. NeurIPS 2023 (DataComp)

**视觉 instruction / fusion**：Liu et al. NeurIPS 2023 (LLaVA); Liu et al. CVPR 2024 (LLaVA-1.5); Li et al. ICML 2023 (BLIP-2); Alayrac et al. NeurIPS 2022 (Flamingo); Wang et al. arXiv 2023 (CogVLM); Bai et al. arXiv 2023 (Qwen-VL); Wang et al. arXiv 2024 (Qwen2-VL); Bai et al. arXiv 2025 (Qwen2.5-VL); Lu et al. arXiv 2024 (DeepSeek-VL); Wu et al. arXiv 2024 (DeepSeek-VL2); Chen et al. CVPR 2024 + arXiv 2024 (InternVL / InternVL-2); Llama Team arXiv 2024 (Llama-3.2-V); Deitke et al. arXiv 2024 (Molmo); Li et al. arXiv 2024 (LLaVA-OneVision)

**VLM 偏好对齐**：Sun et al. arXiv 2023 (LLaVA-RLHF); Yu et al. arXiv 2024 (RLAIF-V); Zhou et al. arXiv 2024 (POVID); Shen et al. arXiv 2025 (VLM-R1)

**多模态 embedding**：Koukounas et al. arXiv 2024 (Jina-CLIP); Jiang et al. arXiv 2024 (VLM2Vec); Zhang et al. arXiv 2024 (mmE5)

**评测**：Li et al. EMNLP 2023 (POPE); Thrush et al. CVPR 2022 (Winoground); Liu et al. ECCV 2024 (MMBench); Yue et al. CVPR 2024 (MMMU); Fu et al. CVPR 2025 (Video-MME); Yu et al. ICML 2024 (MM-Vet); Mangalam et al. NeurIPS 2023 (EgoSchema)

---

代码 + 公式经独立 reviewer 静态检查；数值在 PyTorch 2.x、CUDA 12.x 上验证（ViT / CLIP / SigLIP / Q-Former / M-RoPE 5 个核心模块的形状与初始 loss 均与公式一致）。
