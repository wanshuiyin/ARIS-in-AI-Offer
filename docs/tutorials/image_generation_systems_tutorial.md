## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 Image Generation 系统** — 一页拿下 production text-to-image 栈核心（详见 §1–§10 推导）。

1. **LDM 关键**：VAE encode 把 $H\times W\times 3$ 压成 $h\times w\times c$（SD 1.x: $8\times$ 下采样、$c=4$），扩散在 latent 上做，token 数降到 $1/8^2=1/64$（**数量级估计**，假设每 token 计算量不变；实际 FLOPs 节省还受 latent 通道更宽、self-attention FLOPs 按 token 数平方 scale 等因素影响，非精确的 $64\times$ 等价），最后再 VAE decode 出像素（Rombach et al. 2022 CVPR）。

2. **SD 1.x → SDXL → SD3 → FLUX 主线**：1.x 用 CLIP-L 文本编码 + U-Net；SDXL 双编码器（OpenCLIP-G + CLIP-L）+ 2.6B U-Net + size/crop conditioning + Refiner（Podell et al. 2024 ICLR）；SD3 换 **MM-DiT** + Rectified Flow（Esser et al. 2024 ICML）；FLUX.1 12B MM-DiT 加 parallel attention（Black Forest Labs 2024）。

3. **CFG（必考）**：训练时按概率 $p_\text{drop}\approx 0.1$ 把条件 $c$ 替换成 $\emptyset$；推理时输出 $\hat\epsilon_\text{cfg} = \hat\epsilon_\emptyset + s\,(\hat\epsilon_c - \hat\epsilon_\emptyset)$，$s \in [1.5, 12]$（Ho & Salimans 2022）。

4. **ControlNet 零卷积**（Zhang et al. 2023 ICCV）：把 U-Net encoder 整段 **trainable copy**，每个连接到主干的卷积 $W=0,b=0$ 初始化——前向恒等、**梯度非零**（$\partial L/\partial W = \delta \cdot x \neq 0$），训练时从干净恒等映射出发逐步注入条件信号，从而保护预训练能力。

5. **IP-Adapter**（Ye et al. 2023）：**Decoupled Cross-Attention**——为图像条件**新增一组** $W_K', W_V'$，与原文本 cross-attn 并行，输出相加：$\text{out} = \text{Attn}(Q, K_\text{txt}, V_\text{txt}) + \lambda\,\text{Attn}(Q, K_\text{img}, V_\text{img})$。仅训新增 K/V + projector，~22M 参数。

6. **LoRA**（Hu et al. 2022 ICLR）：$\Delta W = B A,\ B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d,k)$；只训 $A, B$，原 $W$ 冻结。SD 上典型 $r \in \{4,8,16,32\}$，比全量小 50–200×；推理可 merge $W' = W + \alpha B A$。

7. **DreamBooth**（Ruiz et al. 2023 CVPR）：rare-token（如 `sks dog`）+ **prior preservation loss** $L = \|\epsilon - \hat\epsilon(x_t, t, \text{"a sks dog"})\|^2 + \lambda \|\epsilon' - \hat\epsilon(x_t', t, \text{"a dog"})\|^2$，第二项防 language drift / 过拟合。

8. **DiT vs MM-DiT**：DiT 用 **AdaLN-Zero**（条件 $\to$ MLP $\to$ scale/shift/gate，最后一层 $W_\text{gate}=0$ 实现恒等启动，Peebles & Xie 2023 ICCV）；MM-DiT 把 text token 和 image token **拼成一个序列做联合 self-attention**（每个模态独立 QKV 投影，但 attention 是全局的），信息流双向（Esser et al. 2024）。

## §1 直觉与全景

为什么需要 LDM？**Pixel-space 扩散太贵**：SD 时代 1024² 图像 = 3M 像素，U-Net 每个 timestep 都要全分辨率前向，单卡只能训 64×64 这种 toy 尺寸。LDM 的思路是先用预训练 **VQ-VAE / KL-VAE** 把图压到 latent（SD 1.x: $512\times 512\times 3 \to 64\times 64\times 4$，$64\times$ token 数减少），扩散只在 latent 上做，最后再 VAE decode 回像素，把"语义 / 结构 / 纹理"三件事解耦：

```
    pixels x        latent z (perceptually similar)        latent z_T (noise)
   [H,W,3]    →     [h,w,c]     →     diffuse     →     [h,w,c]
                       ▲                                    │
                       │  VAE decoder                       │  reverse SDE / ODE
                       │                                    ▼
                    pixels x̂          ←      latent z_0      ←    [h,w,c]
```

整个 production 栈分成 5 个**正交可替换**的模块：

| 模块 | 作用 | SD 1.x | SDXL | SD3 / FLUX |
|---|---|---|---|---|
| **VAE** | pixel ↔ latent | KL-VAE ($f=8$, $c=4$) | KL-VAE ($f=8$, $c=4$) | $f=8$, $c=16$（更宽 latent） |
| **Text encoder** | text → token embedding | CLIP-L | OpenCLIP-G + CLIP-L | CLIP-L + OpenCLIP-G + **T5-XXL**（SD3）/ T5-XXL + CLIP-L（FLUX） |
| **Denoiser** | $\epsilon$ / $v$ / $u$ 预测 | U-Net 860M | U-Net 2.6B | MM-DiT 2B / 8B / 12B |
| **Objective** | 训练目标 | $\epsilon$-pred (DDPM) | $\epsilon$-pred | **Rectified Flow** (v-pred 一族) |
| **Sampler** | reverse 过程 | DDIM / PLMS / DPM++ | DDIM / DPM++ | Euler / Heun (RF ODE) |

> 💡 **Conditioning 三态** — 面试时主动 disambiguate 你说的"条件"是哪一种。

- **Semantic conditioning（文本）**：text encoder embedding → cross-attention K/V，主要驱动"画什么"

- **Structural conditioning（边缘 / 深度 / 姿态）**：ControlNet / T2I-Adapter，驱动"按什么结构画"

- **Identity / style conditioning（人脸 / 风格）**：IP-Adapter / InstantID / PuLID / DreamBooth / LoRA，驱动"画成谁的样子 / 谁的画风"

## §2 LDM 核心：VAE Compression + Latent Diffusion

### 2.1　LDM 损失（Rombach et al. 2022 CVPR）

预训练 KL-VAE 给一个 encoder $\mathcal{E}: \mathbb{R}^{H\times W\times 3} \to \mathbb{R}^{h\times w\times c}$ 和 decoder $\mathcal{D}$，满足 $\mathcal{D}(\mathcal{E}(x)) \approx x$（感知重建），$h = H/f$，下采样倍率 $f \in \{4, 8, 16\}$，SD 一族用 $f=8$。

在 latent 空间训 diffusion，目标与像素 DDPM 一致：

$$\boxed{\;\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0, \epsilon, t, c}\left[\,\big\|\epsilon - \epsilon_\theta\!\left(z_t,\, t,\, \tau_\theta(c)\right)\big\|^2\,\right]\;}$$

其中 $z_0 = \mathcal{E}(x)$，$z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，$\tau_\theta(c)$ 是 text encoder 输出。

### 2.2　为什么 $f=8$ 是 sweet spot

Rombach 2022 Table 8 ablation：

| 下采样 $f$ | Token 数降低（$f^2$） | 重建质量（FID↓） | 生成质量（FID↓） |
|---|---|---|---|
| $f=4$ | $16\times$ | 最好（latent 接近 pixel） | 一般（扩散仍然太贵） |
| $f=8$ | $64\times$ | 略损（PSNR 微降） | **最佳** |
| $f=16$ | $256\times$ | 显著降（VAE 重建变差） | 重建瓶颈拖垮生成 |
| $f=32$ | $1024\times$ | VAE 几乎无法重建细节 | 生成质量大幅下降 |

> ⚠️ **token 数下降 ≠ 精确 FLOPs 等比下降** — 上表 $f^2$ 是空间 token 数量的精确折算，但实际计算量节省还取决于：(i) latent 空间网络通道宽度通常远大于 pixel-space 的 3 通道，会部分抵消 token 数下降带来的节省；(ii) self-attention FLOPs 按 token 数**平方**scale，attention 层的节省可能远超 $f^2$（如 $f=8$ 时理论上可达 $\sim f^4=4096\times$）。因此表中数字应理解为数量级上的显著节省，而非精确的 FLOPs 等价。

**关键洞察**：VAE 重建质量是上限——latent diffusion 无论多强都画不出 VAE 解不出来的图。所以 $f$ 不是越大越好，要在"压缩率"和"重建上限"之间找平衡。

### 2.3　VAE 的"小 KL"细节

SD VAE 是 **KL-VAE，不是 VQ-VAE**——latent 是**连续高斯**，KL 项极小（$\sim 10^{-6}$ 量级），近似一个 AE 加微弱正则。Rombach 2022 Appendix 解释：太强 KL 会让 latent 退化成纯高斯，丢掉结构信息。

> ⚠️ **SD VAE scaling factor** — 直接用 $\mathcal{E}(x)$ 训扩散，raw latent 标准差远离 1（SD 1.x raw std 约 5.5）。SD 的做法是 latent 乘标量 `0.18215`（"scaling factor"，约等于 $1/5.5$）让 std 接近 1。SDXL/SD3 重新校准了这个常数（SDXL `0.13025`；SD3 `scaling_factor=1.5305` + `shift_factor=0.0609`，diffusers 的 SD3 pipeline 是 `z = (z_raw - shift) * scaling`）。**不一致就会 over/under noising**——pipeline 代码里这是经典 bug 源。

## §3 SD 1.x → SDXL → SD3 → FLUX 主线

### 3.1　SD 1.x / 2.x（Stability AI 2022）

- **U-Net 860M 参数**：cross-attention 装在 latent 分辨率 64 / 32 / 16 三档下采样 + 对应上采样块 + middle block（SD v1 config `attention_resolutions = [4, 2, 1]`，DS=1/2/4 即 64/32/16；最深的 DS=8 也即 8×8 下采样和上采样块只有 ResBlock 不带 transformer，但 middle block 8×8 有 transformer），文本通过 CLIP-L (`openai/clip-vit-large-patch14`，768-d) 提供 77 token embedding

- **Objective**：$\epsilon$-prediction（DDPM Ho et al. 2020 NeurIPS）

- **训练分辨率**：512²，1B+ LAION 图像

- 2.x 换 OpenCLIP-H/14（更强但 license cleaner），并在 768² 上 fine-tune

### 3.2　SDXL（Podell et al. 2024 ICLR）

三个关键升级：

| 改动 | 细节 |
|---|---|
| **2.6B U-Net** | 更宽更深；3× 1.x 参数 |
| **双文本编码器** | OpenCLIP-G (1280-d) + CLIP-L (768-d)；concat 后做 cross-attn |
| **Size & crop conditioning** | 训练时 $(h_\text{orig}, w_\text{orig})$、$(h_\text{crop}, w_\text{crop})$ 与 **target size** $(h_\text{tgt}, w_\text{tgt})$（multi-aspect-ratio 训练用）共 3 组 6 个标量，各自 Fourier-embed 后**加到 timestep embedding 上**，让模型显式知道"这张图原本多大、被裁了哪里、目标分辨率是多少"，避免低分辨率 / 裁剪伪影泄漏到推理 |
| **Refiner** | 一个独立 latent diffusion 模型，专做最后 ~20% noise level（$t < 0.2$），仅细节精修；可选 |
| **训练分辨率** | 1024²（最终），bucketing 不同 aspect ratio |

> ✅ **Size conditioning 训练效果** — SDXL Table 1：没有 size conditioning 时，模型见到 512² LAION 图，会把"低分辨率感"当成数据先验，1024² 推理时模糊；加入 size conditioning 后，推理时填 `(1024, 1024)` 就能告诉模型"我要 1024 质量"，**显著降低模糊 / 块状伪影**。Crop conditioning 同理修复 LAION center-crop 偏置。

### 3.3　SD3（Esser et al. 2024 ICML）

核心改动两个：

**1）训练目标换成 Rectified Flow（RF）**

$$x_t = (1-t)\, x_0 + t\, x_1,\quad u_t = x_1 - x_0$$

其中 $x_0$ 数据端（$t=0$），$x_1 \sim \mathcal{N}(0, I)$ 噪声端（$t=1$）。模型学 $v_\theta(x_t, t, c) \approx x_1 - x_0$。Loss 用 logit-normal $t$ sampling（中间 $t$ 概率更高）+ RF 加权。**timestep convention 与 DDPM 方向相同**（$t=0$ 数据、$t$ 增大 → 噪声增多，并非相反）；但不同代码库（如 diffusers）里具体变量命名 / schedule 细节可能不同，**面试要主动 disambiguate**。

**2）Denoiser 换成 MM-DiT（Multimodal Diffusion Transformer）**

文本和图像 token **拼成一个序列**做联合 self-attention，每个模态有**独立**的 QKV 投影 + AdaLN-Zero MLP 参数，但 attention 矩阵是全局的（双向）。这意味着：

```
[txt tokens, img tokens]  ──╮
       │                     ├─ joint self-attention  ──→  双向信息流
       │                     │                            (txt sees img, img sees txt)
       │                     │
   独立 QKV (txt, img)       │
   独立 LN/MLP gate (txt, img)
```

对比 SD 1.x/SDXL 的 **cross-attention**：image queries 单向读 text K/V，**text 不会被更新**。MM-DiT 让 text 也被 image 更新（"图像 → 文本"流向打开），实验上文本对齐显著提升。

### 3.4　FLUX.1（Black Forest Labs 2024）

12B 参数 MM-DiT v2，主要差异：

| 维度 | FLUX.1 |
|---|---|
| 参数量 | dev: 12B，schnell: 同 12B 但蒸馏过 |
| 架构 | MM-DiT + **parallel attention block**（attn 和 MLP 并行而非串行，类似 PaLM / GPT-J） |
| 文本编码 | T5-XXL (4096-d) + CLIP-L |
| 训练目标 | Rectified Flow（与 SD3 同族） |
| 采样 | dev：~28-50 步；schnell：1-4 步（adversarial diffusion distillation） |
| Position encoding | 多轴 RoPE（对图像 token 按 2D 坐标编码相对位置）；文本 token 的 RoPE 位置 id 通常直接置零（`txt_ids` 全 0，无显式位置编码），顺序信息由 T5 encoder 自身的相对位置机制隐式携带 |

> 💡 **Parallel attention** — Standard transformer block: `y = x + Attn(LN(x)); y = y + MLP(LN(y))`。Parallel block: `y = x + Attn(LN(x)) + MLP(LN(x))`，两支并行算、加在一起。**收益**是 GPU 上 attn 和 MLP 可以重叠 launch / overlap，且权重融合更简洁；轻微的表达力损失通常被规模补回。

### 3.5　并列开源线

- **PixArt-α / Σ**（Chen et al. 2024）：DiT-XL/2 + T5 文本，强调"训练成本只有 SDXL 的 12%"，small but capable。

- **Hunyuan-DiT**（Tencent 2024 arXiv 2405.08748）：中文友好双语 DiT，1.5B 参数，CLIP + mT5 双编码。

- **DiT**（Peebles & Xie 2023 ICCV）：把扩散去噪从 U-Net 换成 ViT-style transformer，类 token + AdaLN-Zero conditioning，scale 律比 U-Net 平滑——这是 SD3/FLUX 的祖先。

- **U-ViT**（Bao et al. 2023）：U-Net 风骨架但纯 transformer block + long skip connection，是早期 transformer-based diffusion 探索。

- **Imagen**（Saharia et al. 2022 NeurIPS）：Google **pixel-space**（不是 latent）级联扩散——$64\times 64$ base + $256\times 256$ super-res + $1024\times 1024$ super-res；文本用大 T5-XXL，结果显示**文本编码器规模 > U-Net 规模**对文本对齐影响更大。

## §4 DiT 架构与 AdaLN-Zero（必考）

### 4.1　DiT block（Peebles & Xie 2023 ICCV）

DiT 把 ViT block 改成扩散友好：每个 block 由 condition $c = \text{embed}(t) + \text{embed}(\text{class})$ 控制。

```
Input tokens x_l (shape [B, N, D]),  condition c (shape [B, D])
                │
        ┌───────┴────────┐
        │                │
   MLP(c) → (α₁, β₁, γ₁) │  scale / shift / gate parameters
        │                │
        ▼                │
   LayerNorm(x_l)        │
        │                │
   scale·γ₁ + shift·β₁   │  ← AdaLN: normalize then conditioned affine
        │                │
   Multi-Head Attention  │
        │                │
   × α₁ (gate, 0-init)   │  ← gate × residual; α₁ starts at 0
        │                │
   +  x_l                │  residual
        │                │
        ▼                │
   ┌────────────┐        │
   │ second half│        │
   │ (LN + MLP) │       同样 (α₂, β₂, γ₂) ← MLP(c)
   └────────────┘
        │
        ▼
   Output x_{l+1}
```

### 4.2　AdaLN-Zero 推导（"为何 gate 初始化为 0"）

DiT 实际形式（Peebles & Xie 2023, Eqn. (5)–(6)）：condition $c$ 经一个 MLP 一次性产出 $(\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2)$，归一化层用 **(1 + gamma)** 而不是 `gamma` 直乘：

$$\text{AdaLN}(x, c) = \big(1 + \gamma(c)\big) \odot \text{LN}(x) + \beta(c)$$

**AdaLN-Zero** 让产出 $(\beta, \gamma, \alpha)$ 的 MLP 最后一层 weight + bias **初始化为 0**：

$$\text{Block}(x, c) = x + \alpha(c) \cdot f\!\left(\big(1 + \gamma(c)\big) \odot \text{LN}(x) + \beta(c)\right)$$

训练 step 0：MLP 全 0 → $\gamma = 0, \beta = 0, \alpha = 0$ → AdaLN 退化为 $\text{LN}(x)$、gate $\alpha = 0$ → 整个 block 输出 $= x$（恒等）。**注意是 `1 + gamma`，所以 gamma=0 时归一化路径不会被乘 0 抹掉，只是恒等于 LN(x)**。

> ✅ **关键性质**：$\alpha = 0$ 时 block 是恒等，但 **梯度非零**。链式法则：

$$\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial \text{out}} \cdot f\!\left((1+\gamma)\odot\text{LN}(x) + \beta\right)$$

step 0 时 $\gamma = \beta = 0$，$f((1+0)\cdot\text{LN}(x) + 0) = f(\text{LN}(x))$ **非零**（LN(x) 一般非零，attention/MLP 也不会把任意输入映成 0）；因此 $\partial L/\partial \alpha \neq 0$，再链式回 $W^{\text{last}}_\text{MLP}$ 即得到非零梯度——$\alpha$ 从 0 开始长大，block 逐步从恒等映射 fork 出非平凡变换，训练稳定不发散。

注意 $\gamma, \beta$ 自身在 step 0 梯度为 0（因为它们的下游被 $\alpha = 0$ 截断：$\partial L/\partial \gamma = (\partial L/\partial \alpha\cdot$ ...) 这条路径要经过 $\alpha$，而 $\alpha = 0$ 时 $\partial \text{Block}/\partial \gamma$ 包含 $\alpha\cdot f'(\cdot)$ 因子等于 0）。但 $\alpha$ 一旦长出来，下一步 $\gamma, \beta$ 立刻拿到非零梯度——所以"first $\alpha$ 长大，再 $\gamma, \beta$ 跟进"是 AdaLN-Zero 的两阶段动力学。

对比朴素初始化（标准随机 $\alpha \neq 0$）：早期 block 输出已经有大方差，叠 24-32 层后激活炸掉、训练发散。AdaLN-Zero 是 DiT scale 上去的关键设计。

### 4.3　时间嵌入

$$\text{TimeEmbed}(t) = \text{MLP}\!\left(\text{SinusoidalEmb}(t)\right),\quad \text{SinusoidalEmb}(t)_{2i} = \sin\!\left(t / 10000^{2i/D}\right)$$

奇偶位置分别用 sin / cos，类似 Transformer 位置编码。$t$ 在 SD 中是离散 timestep $\in \{0, 1, ..., T-1\}$，在 RF / FM 中是连续 $\in [0, 1]$。

## §5 SD 推理循环 + CFG（核心代码）

### 5.1　CFG 公式

训练时以概率 $p_\text{drop} \approx 0.1$ 把 $c$ 替换成 null（空 text embedding 或 zero embedding），让同一个网络既学条件也学非条件。推理：

$$\boxed{\;\hat\epsilon_\text{cfg}(z_t, t, c) = \hat\epsilon_\theta(z_t, t, \emptyset) + s\cdot\left[\hat\epsilon_\theta(z_t, t, c) - \hat\epsilon_\theta(z_t, t, \emptyset)\right]\;}$$

$s$ 是 guidance scale；SD 1.x 一般 $s \in [5, 12]$；SDXL $s \approx 5$-$7$；FLUX dev $\approx 3.5$（更小，因为 RF 模型 CFG 敏感）。

**v-prediction / RF 下**形式相同，把 $\hat\epsilon$ 换成 $\hat v$ 即可。

### 5.2　SD inference loop（核心 40 行）

```python
import torch

@torch.no_grad()
def sd_sample(unet, vae, text_encoder, tokenizer, scheduler,
              prompt, neg_prompt="", num_steps=30, cfg_scale=7.0,
              height=512, width=512, device="cuda", dtype=torch.float16):
    # 1) text 编码：把 prompt 和 negative prompt 都 forward 一次
    ids_pos = tokenizer(prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    ids_neg = tokenizer(neg_prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    emb_pos = text_encoder(ids_pos)[0]                  # [1, 77, D_text]
    emb_neg = text_encoder(ids_neg)[0]
    emb = torch.cat([emb_neg, emb_pos], dim=0)          # [2, 77, D_text]  (uncond, cond)

    # 2) latent 初始化：纯噪声 [1, 4, h/8, w/8]
    lat_shape = (1, 4, height // 8, width // 8)
    z = torch.randn(lat_shape, device=device, dtype=dtype) * scheduler.init_noise_sigma

    # 3) scheduler 设定 timestep
    scheduler.set_timesteps(num_steps, device=device)

    # 4) 主循环
    for t in scheduler.timesteps:
        # 拼 batch：两份 latent 分别配 (uncond, cond) 一次性 forward,
        # 比顺序两次 forward 省一次 kernel launch + 利用 cuDNN batch 友好
        z_in = torch.cat([z, z], dim=0)                  # [2, 4, h, w]
        z_in = scheduler.scale_model_input(z_in, t)      # 某些 sampler 要 sigma scaling

        eps = unet(z_in, t, encoder_hidden_states=emb).sample   # [2, 4, h, w]
        eps_neg, eps_pos = eps.chunk(2, dim=0)

        # 5) CFG combine
        eps_cfg = eps_neg + cfg_scale * (eps_pos - eps_neg)

        # 6) scheduler step：根据 eps 推回 z_{t-1}
        z = scheduler.step(eps_cfg, t, z).prev_sample

    # 7) VAE decode + denormalize 回像素 [0, 1]
    z = z / 0.18215                                       # SD 1.x scaling factor
    x = vae.decode(z).sample                              # [1, 3, H, W] in [-1, 1]
    x = ((x.clamp(-1, 1) + 1) / 2)                        # → [0, 1]
    return x
```

> ⚠️ **CFG 双 forward 不要忘合并** — 新手常写两次 unet forward，吞 2× 时间；正确做法是 `torch.cat([z, z])` + `torch.cat([emb_neg, emb_pos])` 一次性 forward。**进一步**：CFG-distilled / Guidance-distilled 模型（如 SDXL-Turbo、FLUX schnell）甚至不需要双 forward。

> ⚠️ **scaling factor 务必对齐** — SD 1.x: `0.18215`，SDXL: `0.13025`，SD3: scalar `scaling_factor=1.5305` + `shift_factor=0.0609`（diffusers 的 SD3 pipeline 里 `z = (z_raw - shift) * scaling`）。错了图就出错色 / 高频伪影。

## §6 ControlNet 与 IP-Adapter：条件扩展

### 6.1　ControlNet 架构（Zhang et al. 2023 ICCV）

**问题**：要把 edge / depth / pose 等结构条件灌进预训练 SD，从头训成本太高，全量 fine-tune 又会破坏 text-to-image 能力。

**方案**：

```
    Input latent z_t  ──┬───────────────────►  原 SD U-Net Encoder (frozen)
                        │                              │
                        │    Condition image c_img     │
                        │           │                  │
                        │           ▼                  │
                        │    Hint Encoder (conv stack) │
                        │           │                  │
                        │           ▼                  │
                        └────► Trainable Copy ←────────┤  encoder 部分 deep copy,
                                    │                  │  开始训练
                                    │                  │
                              Zero Conv (W=0, b=0)     │
                                    │                  │
                                    ▼                  │
                              add to skip connection ──┘  ─────►  Decoder (frozen)
```

- **Trainable copy**：完整复制 SD U-Net **encoder + middle block**，作为副本，初始权重 = 原 SD encoder weights

- **Zero-conv**：每个连接到主干 skip 路径的 $1\times 1$ 卷积，weight 与 bias **全初始化为 0**

- **Hint encoder**：4 层 conv 把 condition image (1 或 3 channel) 投到 latent shape

- **训练时**：原 SD encoder/decoder 冻结，只训 trainable copy + zero-conv + hint encoder

### 6.2　零卷积梯度推导（L3 必问）

零卷积 layer：$y = W \star x + b$，$W = 0, b = 0$，则 $y = 0$；加到主干 skip 上等于"什么都不加"，前向恒等。

**反向传播时分两条路径**：

**(a) zero-conv 自身权重**：

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} \cdot x_j$$

$x_j$（zero-conv 的输入，来自 trainable copy 输出）非零、$\partial L / \partial y_i$ 非零，**梯度非零**——所以 $W$ 会从 0 开始更新。

**(b) trainable copy 的参数 $\theta_c$**：要经过 $x \to y$ 这条路径，链式法则的关键因子是 $\partial y / \partial x = W$。step 0 时 $W = 0$，**所以第一步 trainable copy 自身的梯度也是 0**。

**收敛过程因此是两阶段**：

1. Step 0：$W = 0$ → ControlNet 完全不影响主干 → 输出 = 原 SD 输出 → **不可能比 baseline 差**

2. Step 1：zero-conv 自己 break 零（路径 (a)）→ $W \neq 0$ → 路径 (b) 解锁

3. Step 2+：trainable copy 开始接收梯度并学习

4. 训练结束：$W$ 与 trainable copy 共同达到合适幅度，结构条件被纳入

> ✅ **为什么这么巧妙** — 普通随机初始化 trainable copy 会让"未训练好的副本"提前污染主干信号，导致训练初期 SD 能力被破坏（catastrophic forgetting）。零卷积保证了一个 **clean warm start**——梯度信号在 zero-conv 自身那条路径上立即可传（破零只需 1 步），随后 trainable copy 跟进，**先解耦再纳入**。

### 6.3　Hint encoding + zero-conv（核心 50 行）

```python
import torch
import torch.nn as nn

def zero_module(m: nn.Module) -> nn.Module:
    """ 把模块所有参数清零（用于 ControlNet zero-conv 与 IP-Adapter projector） """
    for p in m.parameters():
        nn.init.zeros_(p)
    return m

class HintEncoder(nn.Module):
    """ 把 condition image (e.g. canny edge, depth) 编码到 latent 分辨率 """
    def __init__(self, in_ch=3, out_ch=320):  # 320 = SD U-Net first hidden dim
        super().__init__()
        # 渐进下采样到 1/8（与 VAE 同倍率）；最后一层是 zero-conv
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),       nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),          nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.SiLU(),   # /2
            nn.Conv2d(32, 32, 3, padding=1),          nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2), nn.SiLU(),   # /4
            nn.Conv2d(96, 96, 3, padding=1),          nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2), nn.SiLU(),  # /8
            zero_module(nn.Conv2d(256, out_ch, 3, padding=1)),
        )

    def forward(self, hint):     # hint: [B, 3, H, W]
        return self.net(hint)    # [B, out_ch, H/8, W/8]

class ControlNetBlock(nn.Module):
    """ Trainable copy 输出 → zero-conv → 加到主干 skip 上 """
    def __init__(self, ch):
        super().__init__()
        # 这是连接到主干 skip 的"输出 zero-conv"
        self.zero_conv = zero_module(nn.Conv2d(ch, ch, 1))   # 1×1, init 0

    def forward(self, x_copy, x_main_skip):
        # x_copy: trainable copy 当前层输出
        # x_main_skip: 原 SD U-Net 对应层的 skip activation
        return x_main_skip + self.zero_conv(x_copy)
```

> ⚠️ **常见误解** — 零卷积不是 dropout，不是 LoRA，不是 BatchNorm。它是**初始化策略**：weight=0 让"前向恒等 + 梯度非零"二者兼得。

### 6.4　T2I-Adapter（Mou et al. 2024）对比

ControlNet 的 trainable copy 太重（参数量 ≈ 半个 SD），T2I-Adapter 的思路是**纯 adapter**：

| 维度 | ControlNet | T2I-Adapter |
|---|---|---|
| 主干干预 | 复制整个 encoder | 4 个轻量 conv block 直接喂 skip |
| 参数量 | ~360M（SD 1.5） | ~77M |
| 质量 | 较高（结构跟随强） | 略弱（但够用） |
| 推理速度 | 慢（双 encoder） | 几乎免费 |

### 6.5　IP-Adapter（Ye et al. 2023）

**IP-Adapter 用 reference image 做 conditioning，跨身份 / 风格保持**。核心是 **Decoupled Cross-Attention**：

```
    image  ──► CLIP image encoder ──► [N_img, D_clip]
                                            │
                                            ▼
                                    Projector (Linear, ~22M params)
                                            │
                                            ▼
                                    image embeddings [N_img, D_text]
                                            │
                                            │  与 text embedding 并行使用
                                            │
    text   ──► text encoder ──► [N_txt, D_text]
                │                                                          ▲
                │                                                          │
                ▼                                                          │
   ┌───────────────────────────────────────────────────────────────┐
   │   每个 U-Net cross-attention 层:                                │
   │                                                                │
   │   Q = z W_Q                                                    │
   │                                                                │
   │   原文本路径:    K_txt = c_txt W_K^txt,   V_txt = c_txt W_V^txt   │
   │   新图像路径:    K_img = c_img W_K^img,   V_img = c_img W_V^img   │
   │                                                                │
   │   out = Attn(Q, K_txt, V_txt) + λ · Attn(Q, K_img, V_img)      │
   │                                                                │
   │   只训练 W_K^img, W_V^img, projector                            │
   └───────────────────────────────────────────────────────────────┘
```

参数量约 22M（projector ~10M + 每层新增 K/V 投影合计 ~12M），是 SD 全量的 ~1%。

### 6.6　Decoupled Cross-Attention（核心 45 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupledCrossAttention(nn.Module):
    """ 文本 + 图像并行 cross-attention，输出相加 """
    def __init__(self, d_model, num_heads, d_text, d_image, lam=1.0):
        super().__init__()
        self.h = num_heads
        self.d = d_model
        self.d_head = d_model // num_heads
        self.lam = lam

        # 文本路径：与原 SD 一致，加载预训练权重后**冻结**
        self.W_Q     = nn.Linear(d_model, d_model, bias=False)   # query 来自 latent
        self.W_K_txt = nn.Linear(d_text,  d_model, bias=False)
        self.W_V_txt = nn.Linear(d_text,  d_model, bias=False)
        self.W_O     = nn.Linear(d_model, d_model, bias=False)
        for p in (*self.W_Q.parameters(),
                  *self.W_K_txt.parameters(),
                  *self.W_V_txt.parameters(),
                  *self.W_O.parameters()):
            p.requires_grad_(False)        # ← IP-Adapter 只训新增 K/V

        # 图像路径：新增 K/V，仅这两个 + projector 是 trainable 的
        self.W_K_img = nn.Linear(d_image, d_model, bias=False)
        self.W_V_img = nn.Linear(d_image, d_model, bias=False)

    def _split_heads(self, x):     # [B, L, D] → [B, H, L, d_head]
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_head).transpose(1, 2)

    def _attn(self, Q, K, V):       # 标准 scaled dot-product
        return F.scaled_dot_product_attention(Q, K, V)   # [B, H, L_q, d_head]

    def forward(self, z, c_text, c_image):
        # z: [B, L_z, D]   c_text: [B, L_t, d_text]   c_image: [B, L_i, d_image]
        Q     = self._split_heads(self.W_Q(z))
        K_txt = self._split_heads(self.W_K_txt(c_text))
        V_txt = self._split_heads(self.W_V_txt(c_text))
        K_img = self._split_heads(self.W_K_img(c_image))
        V_img = self._split_heads(self.W_V_img(c_image))

        out_txt = self._attn(Q, K_txt, V_txt)
        out_img = self._attn(Q, K_img, V_img)
        out = out_txt + self.lam * out_img            # ← 解耦相加

        # [B, H, L_q, d_head] → [B, L_q, D]
        B, _, L_q, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d)
        return self.W_O(out)
```

> 💡 **为何 decoupled 比 concat 好** — 一种 naive 思路是把 image embedding **concat** 到 text embedding（变长序列做单 cross-attn）。但 IP-Adapter 论文 Table 4 显示，concat 会让文本对齐显著下降（CLIP-Score 掉点）——因为 Q 对 K_txt 和 K_img 用了**同一组 softmax**，长度上 image token 多就抢走 text 注意力。Decoupled 用**两个独立 softmax 然后线性相加**，两路信号互不挤压，是更优的工程方案。

### 6.7　InstantID / PuLID / PhotoMaker

**目标**都是单参考图 → 保 ID 文生图，主流路线：

| 方法 | 核心机制 |
|---|---|
| **InstantID** (Wang, Bai et al. 2024) | IP-Adapter 风格 decoupled cross-attention，直接注入 face embedding（ArcFace/insightface 提取，该 embedding 本身即 identity embedding）+ **IdentityNet**（ControlNet 类分支，条件为 5 点 facial landmark，负责空间姿态控制） |
| **PuLID** (Guo, Wu et al. 2024 NeurIPS) | 双分支 + contrastive alignment，避免 ID 信号污染 prompt 跟随性 |
| **PhotoMaker** (Li, Cao et al. 2024 CVPR) | **Stacked ID Embedding**：每张参考图各自把 CLIP image embedding 与 class word text embedding 融合成该图的 ID embedding，再把多张图的 ID embedding 沿 token 维**堆叠（stack）**（而非先对多图 CLIP embedding 取平均、再统一融合），注入到 cross-attn |

共同思路：**ID-relevant 信号** 用专门 adapter，**ID-irrelevant 信号**（pose / 表情 / 光照）让 prompt 控制，避免身份直接 paste。

## §7 个性化微调：DreamBooth / Textual Inversion / LoRA / Custom Diffusion

### 7.1　Textual Inversion（Gal et al. 2023 ICLR）

**只训一个 token embedding，不动模型**：

1. 引入新 token `S*`（如 `<my-cat>`），其 embedding $e_{S^*} \in \mathbb{R}^{d_\text{text}}$ 是**唯一可训参数**

2. 训练目标：

$$e_{S^*}^* = \arg\min_{e} \mathbb{E}_{z, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c(\text{"a photo of } S^*\text{"}; e))\|^2\right]$$

3. ~3-5K steps 收敛，**embedding 总参数 768-1024 维，文件 < 10KB**

**优点**：极轻，无 catastrophic forgetting；**缺点**：表达力有限（一个 embedding 装不下复杂概念）。

### 7.2　DreamBooth（Ruiz et al. 2023 CVPR）

核心配方两条：

**1）Rare token + class word**：用罕见 token（如 `sks`、`zwx`）+ 类别词（`dog`、`person`），prompt 形如 `"a photo of sks dog"`。罕见 token 在预训练时 embedding 是"语义死角"，不会被原有概念干扰。

**2）Prior Preservation Loss**：

$$\boxed{\;\mathcal{L} = \mathbb{E}\!\left[\|\epsilon - \hat\epsilon_\theta(z_t, t, c_\text{sks})\|^2\right] + \lambda\,\mathbb{E}\!\left[\|\epsilon' - \hat\epsilon_\theta(z_t', t, c_\text{class})\|^2\right]\;}$$

第二项是"class-prior preservation"——用模型**自身生成的** class images（如 200 张 `"a photo of dog"`）做 anchor，告诉模型"sks dog 是特殊的，但普通 dog 还得画对"。$\lambda$ 一般 1.0。

> ⚠️ **没有 prior preservation 的失败模式** — (i) **Language drift**：模型把 "dog" 的概念整体偏移到 sks 的特定形态；(ii) **Concept bleed**：所有 `dog` prompt 都画成 sks dog；(iii) **Overfitting**：~5 张训练图被记死，不同 prompt 生成结果几乎一致。**生产实践**：DreamBooth 一定要配合 prior preservation 或 LoRA-DreamBooth（更稳）。

### 7.3　LoRA（Hu et al. 2022 ICLR）

**核心数学**：对 weight $W \in \mathbb{R}^{d \times k}$ 不直接 fine-tune，而是学**低秩增量**：

$$\boxed{\;W' = W + \Delta W,\quad \Delta W = B A,\quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)\;}$$

通常 $A$ 用 $\mathcal{N}(0, \sigma^2)$ 初始化，$B$ 初始化为 0 → $\Delta W = 0$（保留预训练）→ 训练时只更新 $A, B$。推理时 $W' = W + \alpha B A$（$\alpha$ scaling factor）。

**参数节省**：原 $W$ 有 $d \cdot k$ 参数；LoRA 有 $r(d + k)$ 参数。SD U-Net cross-attn 一个 $W_K$ 是 $D \times d_\text{text}$（如 $1280 \times 2048 = 2.6M$）；$r = 8$ 时 LoRA 是 $8(1280 + 2048) = 26K$，**100× 少**。

### 7.4　LoRA inject 到 nn.Linear（核心 40 行）

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """ Wrap nn.Linear with low-rank delta. Original weight is frozen. """
    def __init__(self, base: nn.Linear, rank=8, alpha=8.0, dropout=0.0):
        super().__init__()
        self.base = base                          # 冻结原 Linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        d_in, d_out = base.in_features, base.out_features
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank                 # 推理时统一 scaling

        # ΔW = B A,  A: [r, d_in],  B: [d_out, r]
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))   # B = 0  → ΔW = 0
        nn.init.kaiming_uniform_(self.A, a=5**0.5)        # 类似 nn.Linear 默认

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 原路径：用 frozen base
        out = self.base(x)
        # LoRA 增量：x @ A^T @ B^T  (顺序很重要——避免 [d_out, d_in] 大矩阵)
        out = out + self.drop(x) @ self.A.t() @ self.B.t() * self.scale
        return out

def inject_lora(model, target_module_names=("to_q", "to_k", "to_v"),
                rank=8, alpha=8.0):
    """ 把 SD U-Net 内 cross-attn / self-attn 的 to_q/to_k/to_v Linear 替换为 LoRALinear；
        to_out 在 Diffusers 里是 nn.Sequential([Linear, Dropout])，要用 to_out.0 单独处理 """
    replaced = 0
    for name, mod in model.named_modules():
        for child_name, child in list(mod.named_children()):
            # 1) to_q / to_k / to_v：单层 Linear，直接替换
            if (child_name in target_module_names
                    and isinstance(child, nn.Linear)):
                setattr(mod, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1
            # 2) to_out：Diffusers Attention 的 to_out 是 nn.Sequential(Linear, Dropout)
            #    SDXL LoRA 习惯只 wrap 第 0 个（Linear）
            if child_name == "to_out" and isinstance(child, nn.Sequential):
                lin = child[0]
                if isinstance(lin, nn.Linear):
                    child[0] = LoRALinear(lin, rank=rank, alpha=alpha)
                    replaced += 1
    return replaced
```

> 💡 **LoRA 放在 attention QKV vs MLP** — 经验上 SD 类模型 LoRA **放 attention 的 Q/K/V/O 比 MLP 收益高**（attention 是文本-图像跨模态交互瓶颈，调权重直接改 conditioning 行为）。LLM 则相反，MLP 含更多知识 → MoE LoRA 也常放 FFN。SDXL LoRA 默认覆盖：`to_q`, `to_k`, `to_v`, `to_out.0` + 部分 conv（如 ResBlock 内的 `conv1`, `conv2`）。

### 7.5　DreamBooth + LoRA = LoRA-DreamBooth

实战中**很少用纯 DreamBooth**（全量训太重），主流是 LoRA-DreamBooth：只对 attention 的 Q/K/V/O 注入 LoRA，配 prior preservation。文件 ~50-200MB，单卡 30 分钟，复现性远好于纯 DreamBooth。

### 7.6　DreamBooth 训练 step（核心 35 行）

```python
import torch
import torch.nn.functional as F

def dreambooth_train_step(unet, vae, text_encoder, scheduler,
                          x_instance, c_instance,    # 训练图像 + "a sks dog" embedding
                          x_class, c_class,          # 自生 class 图像 + "a dog" embedding
                          lam_prior=1.0, dtype=torch.float16, device="cuda"):
    """ 一个 DreamBooth + prior preservation 训练 step """
    bs = x_instance.shape[0]

    # 1) 把 instance & class 拼成 2× batch 一次 forward
    x = torch.cat([x_instance, x_class], dim=0)
    c = torch.cat([c_instance, c_class], dim=0)        # text embeddings

    # 2) encode 到 latent + scale
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample() * 0.18215    # [2bs, 4, h, w]

    # 3) 随机采 timestep + 噪声
    t = torch.randint(0, scheduler.num_train_timesteps, (z.shape[0],), device=device)
    eps = torch.randn_like(z)
    z_t = scheduler.add_noise(z, eps, t)

    # 4) 预测 ε
    eps_pred = unet(z_t, t, encoder_hidden_states=c).sample

    # 5) 分两组算 loss：instance / class 各一份
    eps_pred_inst, eps_pred_cls = eps_pred.chunk(2, dim=0)
    eps_inst, eps_cls = eps.chunk(2, dim=0)

    loss_inst = F.mse_loss(eps_pred_inst.float(), eps_inst.float(),
                           reduction="mean")
    loss_cls  = F.mse_loss(eps_pred_cls.float(),  eps_cls.float(),
                           reduction="mean")
    loss = loss_inst + lam_prior * loss_cls

    return loss
```

### 7.7　HyperDreamBooth / Custom Diffusion 对比

- **HyperDreamBooth** (Ruiz et al. 2024)：用 hypernetwork 直接预测每张参考图的 LoRA 权重，**inference-time 个性化**（~5 秒，对比 DreamBooth ~10 分钟训练）。

- **Custom Diffusion** (Kumari et al. 2023 CVPR)：直接（非低秩分解）微调 cross-attention 的完整 $W_K, W_V$ 矩阵（不动 $W_Q$），配合真实同类图像做 regularization 防概念遗忘/过拟合。它是独立于 LoRA 的**选择性全参数微调**方法，并非 LoRA-DreamBooth 的更窄变体。

| 方法 | 训参数 | 推理代价 | 表达力 | 文件大小 |
|---|---|---|---|---|
| Textual Inversion | embedding（1 个 token） | 0 | 弱 | < 10 KB |
| DreamBooth (full) | 整个 U-Net | 0 | 强 | ~5 GB |
| LoRA-DreamBooth | LoRA on Q/K/V/O | merge 后 0 | 较强 | 50-200 MB |
| Custom Diffusion | 只 W_K, W_V | 0 | 中 | ~70 MB |
| HyperDreamBooth | 一个 hypernet 输出 LoRA | 多一次 hypernet 前向 | 中 | ~120 MB 主网 |

## §8 图像编辑：SDEdit / InstructPix2Pix / Prompt-to-Prompt

### 8.1　SDEdit（Meng et al. 2022 ICLR）

**思路**：图像编辑 = "把输入图加部分噪声 → 用 prompt 引导 reverse 回去"。

```
   输入图 x (e.g. 草图)
         │
         │   noise_strength = 0.6 (例)
         ▼
   z_0 = VAE_enc(x)
         │
   z_τ = √(ᾱ_τ) z_0 + √(1 - ᾱ_τ) ε,   τ = noise_strength × T
         │
         ▼
   reverse SDE / ODE from t=τ to t=0    (受 prompt c 引导)
         │
         ▼
   z_0' →  VAE_dec → 编辑后图 x'
```

**关键参数 strength $\in [0, 1]$**：

- $\text{strength} \to 0$：noise 极少，输出 $\approx$ 输入（不编辑）

- $\text{strength} \to 1$：完全噪声化，输出 = 纯文生图（输入信号丢失）

- 常用 $0.3$–$0.8$ 范围找 balance

### 8.2　SDEdit 核心代码

```python
import torch

@torch.no_grad()
def sdedit_sample(unet, vae, text_encoder, tokenizer, scheduler,
                  init_image, prompt, neg_prompt="",
                  strength=0.7, num_steps=30, cfg_scale=7.0,
                  device="cuda", dtype=torch.float16):
    assert 0.0 < strength <= 1.0, "strength=0 == 不编辑（直接返回输入）；strength>1 不合法"

    # 1) text 编码（与 §5 一致）
    ids_pos = tokenizer(prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    ids_neg = tokenizer(neg_prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
    emb = torch.cat([text_encoder(ids_neg)[0],
                     text_encoder(ids_pos)[0]], dim=0)

    # 2) 把输入图 encode 到 latent
    z0 = vae.encode(init_image).latent_dist.sample() * 0.18215   # [1, 4, h, w]

    # 3) 设 timestep 子集，只走后 strength 段
    scheduler.set_timesteps(num_steps, device=device)
    import math
    # 用 ceil + max(1, .) 保证 strength>0 时至少有 1 步要跑
    n_edit = max(1, math.ceil(num_steps * strength))
    t_start = num_steps - n_edit               # 0 ≤ t_start < num_steps
    timesteps = scheduler.timesteps[t_start:]  # 至少有 1 个 timestep

    # 4) 给 z_0 加 timesteps[0] 那一档噪声
    eps = torch.randn_like(z0)
    z = scheduler.add_noise(z0, eps, timesteps[:1])

    # 5) 主循环（与 §5 完全一致）
    for t in timesteps:
        z_in = torch.cat([z, z], dim=0)
        z_in = scheduler.scale_model_input(z_in, t)
        eps_pred = unet(z_in, t, encoder_hidden_states=emb).sample
        eps_neg, eps_pos = eps_pred.chunk(2, dim=0)
        eps_cfg = eps_neg + cfg_scale * (eps_pos - eps_neg)
        z = scheduler.step(eps_cfg, t, z).prev_sample

    # 6) decode
    z = z / 0.18215
    return ((vae.decode(z).sample.clamp(-1, 1) + 1) / 2)
```

### 8.3　InstructPix2Pix（Brooks et al. 2023 CVPR）

**做指令式编辑**："add a hat to the dog"。训练数据用 GPT-3 + Prompt-to-Prompt 合成（pair-wise (源图, 指令, 目标图)），fine-tune SD 1.x。架构：

- U-Net 输入 channel 数 4 → 8（原 latent 4 + 源图 latent 4）

- 两种 CFG scale：text guidance $s_T$ 和 image guidance $s_I$，独立调

$$\hat\epsilon = \hat\epsilon(\emptyset, \emptyset) + s_I [\hat\epsilon(c_I, \emptyset) - \hat\epsilon(\emptyset, \emptyset)] + s_T [\hat\epsilon(c_I, c_T) - \hat\epsilon(c_I, \emptyset)]$$

### 8.4　Prompt-to-Prompt（Hertz et al. 2023 ICLR）

**Training-free 编辑**：通过**操纵 cross-attention map** 实现"换词不换结构"。

- 跑原 prompt P 得到每层每步的 cross-attn map $M_t$（shape $[H_q, L_t]$，对每个图像 token 给出每个文本 token 的权重）

- 跑新 prompt P\*（只改一个词，如 `cat` → `dog`）但**强制把对应文本 token 的 attention map 替换为原 P 的对应位置**

- 因此结构（哪些图像 patch 关注哪些文本位置）保留，只有内容（softmax 之后被指向的 value）更新

适合 "swap one word"、"add adjective"、"reweight emphasis" 三类编辑。

## §9 蒸馏：少步采样

### 9.1　LCM / LCM-LoRA（Luo et al. 2023）

**Latent Consistency Model**：把 latent diffusion 蒸馏成 **Consistency Model**（Song et al. 2023），让单步预测 $f_\theta(z_t, t)$ 直接给出 $z_0$，从 50 步压到 4-8 步。

**LCM-LoRA**：把上述蒸馏写成 **LoRA**，base SDXL/SDXL-1.0 + 一个 ~200MB LCM-LoRA = 4 步出图。零样本切换，**可与个性化 LoRA 叠加**。

### 9.2　SDXL-Turbo：ADD（Sauer et al. 2024 ECCV）/ SD3-Turbo：LADD（Sauer et al. 2024）

**SDXL-Turbo = ADD (Adversarial Diffusion Distillation, arXiv 2311.17042 / ECCV 2024)**；**SD3-Turbo = LADD (Latent Adversarial Diffusion Distillation, arXiv 2403.12015)** —— 两套独立方法，ADD 在 pixel 空间用 vision encoder 做判别器，LADD 改到 latent 空间训判别器，规模化到 SD3 大模型。

**Adversarial Diffusion Distillation (ADD)**：

- **Teacher**：原 SDXL（多步扩散模型）

- **Student**：与 teacher 同架构，目标 1-4 步出图

- **两个 loss**（不是三个——蒸馏项本身就是 score-distillation 形式，不再额外拆出第三项）：

  1. Adversarial loss：DINOv2-based discriminator 判别 student 单步输出真假

  2. Distillation loss（score-distillation 形式）：把 student 单步去噪输出重新加噪到随机采样的更高噪声水平，再用**冻结的 teacher** 在该噪声水平上做去噪重建，取 student 输出与该 teacher 重建之间的距离作为回归目标——**不存在**独立于此之外、单独匹配 teacher 多步完整采样轨迹的第三个 MSE/LPIPS 项

ADD 是少步蒸馏 SOTA 路线之一；FLUX schnell 也基于类似思想（"timestep distillation" + adversarial）。

### 9.3　DMD / DMD2 / Hyper-SD 简表

| 方法 | 核心 |
|---|---|
| DMD (Yin et al. 2024) | 用 KL divergence 间接对齐 student 与 teacher 分布，1 步生成 |
| DMD2 (Yin et al. 2024) | DMD + 一些训练稳定性技巧 |
| Hyper-SD (Ren et al. 2024) | trajectory-segmented consistency distillation |
| Lightning SDXL | 4-8 步 SDXL 蒸馏，开源版本流行 |

## §10 评测：FID / CLIP-Score / ImageReward / HPSv2 / PickScore

| 指标 | 计算 | 评什么 |
|---|---|---|
| **FID** (Heusel et al. 2017 NeurIPS) | Inception-V3 pool3 特征的 Fréchet distance（real vs gen） | 整体分布相似度（多样性 + 真实度） |
| **CLIP-Score** | CLIP 算 (text, image) cosine similarity，平均 | 文本对齐 |
| **ImageReward** (Xu et al. 2023 NeurIPS) | 拿人类偏好数据训的 reward model（ViT + CLIP backbone） | 人类整体偏好（含审美 / 文本对齐 / 真实度） |
| **HPSv2** (Wu et al. 2023) | Human Preference Score V2，类似 ImageReward 但更多数据 | 人类偏好（更细分类别） |
| **PickScore** (Kirstain et al. 2023 NeurIPS) | Pick-a-Pic 数据集训练，CLIP-based | 用户偏好 |

> ⚠️ **FID 局限** — (i) 对 mode collapse 不敏感（生成方差小 FID 反而可能涨）；(ii) Inception-V3 是 ImageNet 训的，对人脸 / 艺术 / 非自然图偏置严重；(iii) **生成数应至少 10K**，<5K 时方差极大，paper 间不可比；(iv) FID-30K vs FID-10K 不能直接比。**面试要主动指出 FID 不是 final word**，必须配合 human preference 指标（IR / HPSv2 / PickScore）。

## §11 复杂度 / 资源

> ⚠️ **本节数字为粗估** — 训练 A100 小时、推理秒数、显存峰值都来自社区拼图与个别公开报告。准确度依赖 batch size / sequence 实现 / 优化器 / 显存策略；面试时主动声明"这是数量级估计、非官方数字"。

### 11.1　训练侧（数量级估计）

| 模型 | 参数 | 训练数据 | 训练算力（数量级估计） |
|---|---|---|---|
| SD 1.5 | 860M U-Net + 84M VAE + 123M CLIP-L | LAION-5B → LAION-aesthetics 2B | ~150K A100 hr 级 |
| SDXL | 2.6B U-Net + Refiner | 内部 + LAION | ~250-300K A100 hr 级 |
| SD3 | 2B / 8B MM-DiT（最大 8B） | 内部 ~1B | 未公开（≫ SDXL） |
| FLUX.1 | 12B MM-DiT | 内部 | 未公开（极大） |

### 11.2　推理侧（粗估，依赖实现）

**SD 1.5 512×512（原生分辨率，latent 64×64×4）**：

- 单步 U-Net FLOPs：~0.5T

- 单步 cross-attn QKV：(L_z=4096) × (L_t=77) → 0.3M score 矩阵 × 多层

- 单步显存：~2-3 GB（FP16，无 cross-attn KV cache）

- 30 步 ≈ 1.5-2 秒 / 张（A100，FP16）

（SD 1.5 在 1024² 上运行会比 512² 大约慢 4×；社区不建议非原生分辨率直接推理，常配合 SDXL 或 super-res。）

**SDXL 1024×1024**：

- 单步 U-Net FLOPs：~1.2T

- 30 步 ≈ 4-5 秒 / 张（A100，FP16）

**FLUX.1-dev 1024×1024**：

- MM-DiT FLOPs ≈ 2.5T / step

- 28 步 ≈ 12-15 秒（A100）；H100 ~5-7 秒

### 11.3　Memory footprint cheat sheet

| 组件 | 估算 |
|---|---|
| Latent ($1024^2 / f=8$) | $128 \times 128 \times 4 \times 2$ bytes (FP16) ≈ 130 KB |
| U-Net activations (SDXL, batch 1) | ~7 GB（FP16，需 gradient checkpointing 才能训） |
| Cross-attn scores (peak) | $16384 \times 77 \times 2 \times \text{heads} \approx$ 几 MB / 层 |
| KV cache for text | 文本固定 77 token，CFG 双 batch → 154 token 等价；可全 batch 缓存 |

## §12 与相关方法对比

### 12.1　扩散族 vs 其他生成模型

| 类 | 速度 | 质量 | 训练稳定性 |
|---|---|---|---|
| **Diffusion / Flow** | 慢（多步） | 高 | 高（MSE 回归） |
| GAN (StyleGAN, BigGAN) | 快（1 步） | 高（特定 domain） | 低（mode collapse） |
| VAE | 快 | 低（模糊） | 高 |
| Autoregressive (DALL-E, Parti) | 中（token by token） | 中（DALL-E 1）/ 高（Parti） | 高 |
| Hybrid (Muse) | 中（少步并行 token） | 中-高 | 中 |

### 12.2　扩散 internal route

| 路线 | 表征 | 训练目标 | 代表 |
|---|---|---|---|
| Pixel-space DDPM | pixel | $\epsilon$-pred | Imagen, GLIDE |
| Latent diffusion | VAE latent | $\epsilon$-pred / $v$-pred | SD 1.x/2.x, SDXL |
| Latent rectified flow | VAE latent | $u_t = x_1 - x_0$ | SD3, FLUX |
| Score-based EDM | pixel / latent | preconditioned $\epsilon$ | EDM, EDM2 |

## §13 25 高频面试题（codex 5.5 xhigh 顶级 lab 面试官视角）

每题点开看答案要点 + 易踩坑。

### L1必会题（任何视觉 / multimodal 工程岗都会问）

<details>

<summary>Q1.LDM 相比 pixel-space diffusion 为何更省算？</summary>

- VAE 把 $H\times W$ 下采样 $f=8$ 倍：$(H/8)\times(W/8)$ token

- token 数下降 $f^2 = 64$ 倍，FLOPs 数量级上也相应下降（非精确等比——latent 通道更宽、self-attention FLOPs 按 token 数平方 scale，会让实际节省偏离 $64\times$）

- 同等算力可上更高分辨率（512² → 1024²）

只说"压缩"，不给倍率推导；把 token 数降低当成精确的 FLOPs 节省；忘了 VAE 重建质量是上限。

</details>

<details>

<summary>Q2.SD 1.x 的 latent shape 是多少？</summary>

- 输入 $512\times 512\times 3$，VAE encode 后 $64\times 64\times 4$

- $f = 8$ 下采样，$c = 4$ channel

- **VAE encode 之后** 把 latent **乘** scaling factor `0.18215`（SD 1.x），让方差靠近 1 再喂扩散；**VAE decode 之前**再 **除回**。训练 / 推理两端方向一致——encode 后乘、decode 前除

scaling factor 给错（SDXL 是 0.13025，SD3 用 scalar `scaling_factor=1.5305` + `shift_factor=0.0609`）；或者把方向搞反（如果只在推理时做，训练时不做，latent 方差就完全错位）。

</details>

<details>

<summary>Q3.CFG 公式是什么？训练时怎么造 uncond 分支？</summary>

- 推理：$\hat\epsilon_\text{cfg} = \hat\epsilon_\emptyset + s(\hat\epsilon_c - \hat\epsilon_\emptyset)$，$s \in [1.5, 12]$

- 训练：以概率 $p_\text{drop} \approx 0.1$ 把 condition 替换为 null embedding

- 同一模型既学条件分支也学无条件分支

把 $s$ 当作 temperature 乱调；忘了训练时的 dropout 步骤。

</details>

<details>

<summary>Q4.cross-attention 在 SD U-Net 中怎么用？</summary>

- Image latent token 做 Q

- Text token embedding 做 K, V

- 出现在 U-Net **装了 transformer block 的那几级**（SD 1.5 config `attention_resolutions=[4,2,1]`：DS=1/2/4，即 latent 64/32/16 三档下采样 + 对应上采样 + middle block 8×8（512² 输入下 f=8 VAE → latent 64²，DS=8 即为 8×8 middle）；SD/SDXL 的 middle block 是标准 SpatialTransformer / `UNetMidBlock2DCrossAttn`：ResBlock → (self-attention → cross-attention → FFN 的 transformer block) → ResBlock，**同样有 cross-attn**（与 §3.1 的描述一致），并非"纯 ResBlock + self-attention、不做 cross-attn"）

- 每个 transformer block 内顺序：self-attention → cross-attention → FFN

说反 Q 和 K/V 的来源；以为 8×8 下采样 / 上采样块也有 cross-attn（其实只有 middle 才有）；或反过来以为只在 bottleneck 一处。

</details>

<details>

<summary>Q5.SDXL 相比 SD 1.5 的主要改动？</summary>

- U-Net 860M → 2.6B

- CLIP-L → OpenCLIP-G + CLIP-L 双编码器

- Size + crop conditioning：训练分辨率信号显式注入

- Refiner（可选）做最后 ~20% 噪声段精修

- 训练分辨率 512² → 1024²，bucketing 多 aspect ratio

只说"参数变大"，漏 size conditioning / 双编码器。

</details>

<details>

<summary>Q6.ControlNet 零卷积的作用是什么？</summary>

- 在 trainable copy 连接到主干的 1×1 conv 上，weight 与 bias 全初始化为 0

- 前向：$y = 0 \cdot x + 0 = 0$，加到主干 = 不影响（恒等启动）

- 反向：$\partial L / \partial W = \partial L / \partial y \cdot x \neq 0$，仍可更新

- 既保护预训练 SD 能力，又能学到条件信号

误以为零卷积"训不出来"；或者以为它和 BN / dropout 类似。

</details>

<details>

<summary>Q7.LoRA 公式是什么？参数节省比例多少？</summary>

- $W' = W + B A$，$A \in \mathbb{R}^{r \times k}$，$B \in \mathbb{R}^{d \times r}$，$r \ll \min(d, k)$

- 原 $W$ 冻结，只训 $A, B$

- 参数从 $dk$ 降到 $r(d+k)$，精确比例是 $r(d+k)/dk = r(1/d+1/k)$；当 $d=k=n$（方阵，如常见 attention Q/K/V/O 投影）时为 $2r/n$，**不是** $r/n=r/\min(d,k)$（相差 2 倍）；SD 上 $W_K$（$d=1280, k=2048$）取 $r=8$ 时约 100× 少（具体倍数依 $d,k$ 形状而定）

把参数比例记成 $r/\min(d,k)$（漏了 2 倍因子）；把 $A, B$ 的形状写反；忘了 $W$ 是冻结的。

</details>

<details>

<summary>Q8.DreamBooth 的 prior preservation loss 解决什么？</summary>

- 防止 language drift：模型把 class concept 整体替换为 sks 个性化形态

- 防 concept bleed：所有 `dog` prompt 都画成 sks

- 用模型自生 class 图像做 anchor，$L = L_\text{instance} + \lambda L_\text{class}$

只记得 sks 罕见 token，漏掉 prior loss；以为 prior loss 是 regularization on parameters。

</details>

<details>

<summary>Q9.IP-Adapter 与直接把图像 token concat 到 text 上有何区别？</summary>

- IP-Adapter 用 **decoupled cross-attn**：新增一组 $W_K', W_V'$ 与原文本并行，输出**线性相加**

- Concat 会让 Q 对 (text, image) 共享一个 softmax，长度上 image 多就挤压文本对齐

- Decoupled 两路独立 softmax，互不干扰

只说"加了 image embedding"，不解释 softmax 互斥问题。

</details>

<details>

<summary>Q10.FID 评测有哪些局限？</summary>

- Inception-V3 是 ImageNet 训练，特征偏置（人脸 / 艺术风格不准）

- 对 mode collapse 不敏感（小方差可能 FID 反低）

- 生成数 <10K 时方差大，跨 paper 不可比

- 不评单图质量，只评分布；需配合 CLIP-Score / ImageReward / HPSv2

把 FID 当万能金标；FID-10K vs FID-30K 直接对比。

</details>

### L2进阶题（research / production 岗位）

<details>

<summary>Q11.SDXL 的 size conditioning 是怎么训练 / 推理的？</summary>

- 训练时记录每张图原始 $(h_\text{orig}, w_\text{orig})$、crop 起点 $(h_\text{crop}, w_\text{crop})$，以及 **target size** $(h_\text{tgt}, w_\text{tgt})$（multi-aspect-ratio 训练用，SDXL 论文 §2.3）

- 这 3 组共 6 个标量各自 Fourier-embed 后过 MLP，**加到 timestep embedding 上**送入 U-Net

- 推理时填 $(1024, 1024)$ 与 $(0, 0)$ 告诉模型"我要 1024 全图质量"，避免低分辨率 / 裁剪伪影泄露到生成

- 也可故意填小 size / 非零 crop 控制风格

只说"加了分辨率信号"，漏掉 target size 这组；不讲注入位置（时间 embedding）与 Fourier-embed 细节。

</details>

<details>

<summary>Q12.MM-DiT 与 SDXL cross-attn 的信息流差别？</summary>

- SDXL: cross-attn 中 image latent 做 Q，text 做 K/V，**单向**（text 不更新）

- MM-DiT: text token 与 image token **拼成一个序列做联合 self-attn**，独立 QKV 投影但全局 attention 矩阵，**双向**（text 也被 image 更新）

- 实证：MM-DiT 显著提升复杂 prompt 的文本对齐（SD3 论文 Table 1）

说反方向；以为 MM-DiT 只是"更大的 cross-attn"。

</details>

<details>

<summary>Q13.AdaLN-Zero 中 gate 为何初始化为 0？训练初期梯度怎么传？</summary>

- $\text{Block}(x, c) = x + \alpha(c) \cdot f\big((1+\gamma(c)) \odot \text{LN}(x) + \beta(c)\big)$（DiT 用 **1+γ**，不是 γ 直乘）

- $\alpha = 0$ 时 block 输出 = $x$（恒等），保证深层稳定 warm start

- 梯度通过 $\partial L / \partial \alpha = \partial L / \partial \text{out} \cdot f((1+0)\text{LN}(x)+0) \neq 0$，$\alpha$ 从 0 慢慢长大

- 注意 $\gamma, \beta$ 自身在 step 0 梯度为 0（被 $\alpha = 0$ 截断），$\alpha$ 长大后才跟进，是两阶段动力学

- 等价于"learnable identity shortcut"，DiT scale 上去的关键

写错 AdaLN 形式为 $\gamma\odot\text{LN}(x)+\beta$（漏掉 1+γ 的偏置）；以为 $\alpha = 0$ 训不动；把 AdaLN-Zero 当作普通 dropout。

</details>

<details>

<summary>Q14.LoRA 放 attention QKV 与放 MLP 谁更好？为什么？</summary>

- SD 类视觉生成：**attention QKV** 收益更大（跨模态条件信号瓶颈在 cross-attn）

- LLM：MLP 含更多 task knowledge → 放 MLP / FFN 收益更高

- SDXL LoRA 默认覆盖 `to_q, to_k, to_v, to_out.0` + 部分 conv

- 这是 SD vs LLM 的 fine-tune 经验性差异，与"哪部分参数承担条件交互"相关

把 SD 与 LLM 的 LoRA 配置混用；以为"哪都行"。

</details>

<details>

<summary>Q15.Rectified Flow 与 DDPM 在训练目标 / 采样上的差别？</summary>

- **DDPM** ε-pred：$\mathcal{L} = \|\epsilon - \hat\epsilon_\theta(x_t, t)\|^2$，$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$

- **RF**：$x_t = (1-t)x_0 + tx_1$（$x_0$ 数据，$x_1$ 噪声），$\mathcal{L} = \|u_t - v_\theta(x_t, t)\|^2$，$u_t = x_1 - x_0$ 是常数

- **采样**：DDPM 多用 DDIM / DPM++（基于 SDE / ODE 推导的高阶 solver），RF 直接 Euler / Heun

- RF 路径直，少步采样质量好（SD3 / FLUX 选 RF 的关键原因）

把 RF 当作"另一个噪声 schedule"；忘了 RF target 是 $x_1 - x_0$ 不是 $\epsilon$。

</details>

<details>

<summary>Q16.SD VAE 的 scaling factor 是什么？为什么需要？</summary>

- VAE encode 输出的 raw latent 标准差远离 1（SD 1.x raw std 约 5.5）

- 直接用会让扩散噪声 schedule 失配（$x_t$ 信号 / 噪声比错）

- SD 1.x 乘标量 `0.18215`，让 latent 方差 ≈ 1；SDXL 用 `0.13025`；SD3 用 scalar `scaling_factor=1.5305` + 标量 `shift_factor=0.0609`（不是 per-channel mean/std）

- VAE encode 后 × scaling（SD3 还要先减 shift），VAE decode 前 ÷ scaling（SD3 再加回 shift），必须严格对齐

值给错；以为 SD3 用 per-channel 数组（其实是 scalar+shift）；忘了 encode 与 decode 两端都要做。

</details>

<details>

<summary>Q17.SDEdit 的 strength 参数怎么影响输出？</summary>

- strength $\in [0, 1]$ 决定加噪到哪个 timestep $\tau = \text{strength} \cdot T$

- strength → 0：几乎无噪声，输出 ≈ 输入（不编辑）

- strength → 1：完全噪声化，输出 = 纯文生图（输入完全丢失）

- 常用 0.3-0.8，找"保留结构 + 跟 prompt"的 sweet spot

把 strength 当 CFG scale 乱解释；忘了 0 是"不编辑"。

</details>

<details>

<summary>Q18.Prompt-to-Prompt 为什么能保结构？</summary>

- 跑原 prompt 时存每层每步的 cross-attn map $M_t$（image patch → text token 权重）

- 跑新 prompt 时，对**未变化的 token**强制使用原 $M_t$

- 结构（哪些 patch 关注哪些位置）保留，内容（softmax 后聚合的 value）跟着新 prompt

- 适合 single-word swap / 词权重 reweight，不适合大幅 prompt 重写

以为 P2P 改的是 latent；不知道它操纵 attention map。

</details>

<details>

<summary>Q19.LCM / Consistency Model 与 ADD 蒸馏的区别？</summary>

- **LCM** (Luo 2023)：训练 student 直接预测 $z_0$，目标是 ODE trajectory 的"自一致"（consistency loss）

- **ADD** (Sauer et al. 2024 ECCV，preprint arXiv 2311.17042 / 2023)：adversarial loss（DINOv2 discriminator）+ distillation loss（score-distillation 形式）两项组成，并非三件套

- LCM-LoRA 是 LoRA 形式，可零成本接入任意 base；ADD 是全量蒸馏，效果略强但需重训

- 都把 NFE 从 25-50 压到 1-4

把 LCM 当 GAN；以为 ADD 只有 adversarial 一项 loss。

</details>

<details>

<summary>Q20.SDXL Refiner 是必须的吗？什么时候用？</summary>

- 不必须；Refiner 是独立 latent diffusion 模型专做最后 ~20% noise level

- 主要修细节（皮肤 / 头发 / 纹理）

- 在 base U-Net 跑到 $t < 0.2$ 时切到 Refiner 继续

- 实际中 SDXL Refiner 涨幅小、增加时间成本，社区常**不用**

以为 Refiner 是必选；说它"训练时一起用"（实际是两阶段独立训）。

</details>

### L3顶级 lab / 深入题（research lead 视角）

<details>

<summary>Q21.详细推导 ControlNet 零卷积"前向 0、梯度非零"为何成立？</summary>

零卷积层 $y = W \star x + b$，$W \in \mathbb{R}^{c_\text{out} \times c_\text{in} \times 1 \times 1}$，初始化 $W = 0, b = 0$，输入 $x$ 来自 trainable copy 输出（非零）：

- **前向**：$y = 0 \star x + 0 = 0$。加到主干 skip 上后等价于"没加"，所以 step 0 输出 = baseline SD 输出。

- **反向（zero-conv 自身权重）**：$\partial L / \partial W_{ij} = \partial L / \partial y_i \cdot x_j$。$x_j$ 来自 trainable copy（一开始就是预训练 SD encoder weights，对任何非零输入都给非零激活），$\partial L / \partial y_i$ 来自下游 loss，**乘积非零**，$W$ 即可更新——这是 zero-conv 自己能"破零"的关键。

- **反向（trainable copy 的参数）—— 微妙之处**：trainable copy 的参数 $\theta_c$ 必须通过 $x \to y$ 这条路径接受梯度。链式法则给出 $\partial L / \partial \theta_c = (\partial L / \partial y) \cdot (\partial y / \partial x) \cdot (\partial x / \partial \theta_c)$。**注意 $\partial y / \partial x = W$**——step 0 时 $W = 0$，所以**第一步 trainable copy 的参数梯度确实为 0**！但只要 $W$ 在第 1 步更新出非零值（如上一条所述），第 2 步 $\partial y / \partial x = W \neq 0$，trainable copy 也开始学。

  这就是 ControlNet "warm start 但仍可学"的两阶段机制：**先 zero-conv 自己 break 零**（第 1 步），**再带动 trainable copy 学习**（第 2 步开始）。

这是 ControlNet 比朴素 "freeze + adapter" 强的关键工程巧思——既零干扰启动，又能整个分支逐步学起来。**类似思想出现在**：AdaLN-Zero（DiT，$\alpha = 0$ 也只是第一步阻断、不阻断 $\alpha$ 自身的梯度）、LoRA 的 $B = 0$ 初始化（$B = 0$ 时 $\partial L / \partial A$ 会含 $B$ 因子也为 0，但 $\partial L / \partial B$ 含 $A \neq 0$ 因子，所以 $B$ 先动起来）。

不会写出 chain rule，或者把"前向 0"误以为"梯度全部为 0"（忽略 $\partial y / \partial W = x$ 这条路径），或者反过来错说"trainable copy 第一步就有梯度"（实际要等 zero-conv 破零之后）。

</details>

<details>

<summary>Q22.SDXL 的 size / crop conditioning，训练数据有什么真实分布特性，conditioning 又怎么补救？</summary>

LAION 数据集训练时遇到 3 个分布问题：

1. **分辨率多样**：从 256² 到 4K，多数 < 1024²。Naive 训练会被 "low-res 占多数" 拖向模糊偏置。

2. **center crop 偏置**：很多 pipeline 把图 center-crop 到正方形，丢掉边缘信息。模型学到"主体在中心"先验，生成时常裁掉头脚 / 边缘。

3. **aspect ratio 偏单一**：直接 resize 到 $1024^2$ 浪费横竖图信息，bucket 不同 ratio 才解决（SDXL 用 ~30 个 bucket）。

SDXL conditioning 的修复：

- 把 $(h_\text{orig}, w_\text{orig})$ 做 sinusoidal Fourier embed → MLP → 加到 timestep embedding。模型可以"知道"原始分辨率，**推理时填 $(1024, 1024)$ 就触发"按 1024 质量生成"模式**。

- 把 $(h_\text{crop}, w_\text{crop})$ 同样 embed 后注入。推理填 $(0, 0)$ 等于"我从原图左上角开始裁"。**故意填非零 crop 可控制生成视角**（如让主体偏右下角）。

- 训练时 size/crop 信号准确反映该样本，推理时把它们当 control knob。

Ablation 结论（Podell 2024 ICLR Table 1）：去掉 size conditioning 后 FID 显著升高，特别在大分辨率 prompt 上模糊感强。

只说"加了 size 标签"，不解释 LAION 分布问题；忘了 Fourier embed + timestep injection 的具体位置。

</details>

<details>

<summary>Q23.MM-DiT vs SDXL cross-attn 的信息流差异及对文本对齐的影响？</summary>

**SDXL cross-attention 一层的信息流**：

```
image latent Q  ─►───┐
                     │   softmax(QK^T/√d) V
                     │
text K/V (frozen)  ──┘──► attended_out → 加到 image latent residual
```

- text token 的表示**不被更新**，每层 cross-attn 后 text 仍是输入 embedding（只是被多次"读"）

- 信息流单向：text → image，image **看不进** text 的 "context" 更新

- 复杂 prompt（如"a red cube to the LEFT of a blue cube, with a small green ball between them"）下，模型对**位置 / 关系**的理解只能从静态 text embedding 里抽

**MM-DiT 一层信息流**：

```
                 [txt | img] 拼成一个序列
                       │
        ┌──────────────┼──────────────┐
        │              │              │
  独立 QKV (txt)   独立 QKV (img)
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              joint self-attention
                  (全局 attn matrix)
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   text 输出更新    image 输出更新
```

- **每个 token (text or image) 既是 Q 也是 K/V**，且 attention matrix 是全局的（$L_\text{txt} + L_\text{img}$ token 互相看）

- **text 也被 image 更新**：当前层 text 表示融合了 image 的信号，下一层 text 已经是"上下文感知"的 conditioning

- 复杂 prompt 的空间关系 / 多对象绑定显著改善（SD3 论文 Figure 5 显示 SD3 在 GenEval 各项 ~10-20% 提升）

代价：参数和算力增加（每 token 都做完整 MHA）。

把 MM-DiT 简单当"更大的 cross-attn"；不区分单向 vs 双向信息流；不能说出对 multi-object prompt 的具体提升。

</details>

<details>

<summary>Q24.LoRA 在 SD attention 上放 Q/K/V 哪个更关键？理论上怎么解释？</summary>

经验上 SD attention LoRA 配置主要覆盖 $W_Q, W_K, W_V, W_O$ 四件套（**全套**通常最稳）。如果只能选一个，社区经验上：

- **$W_V$ 影响内容**：cross-attn 中 $V$ 直接进入残差，调 $W_V$ 等于调"text token 传给 image latent 的具体内容"。**风格 LoRA** 倾向于动 $W_V$。

- **$W_K$ 影响选择**：调 $W_K$ 改变"哪些 text token 被 Q 选中"。**身份 / 概念 LoRA** 常重 $W_K$（让特定 token 被精准选中）。

- **$W_Q$ 影响 image 端**：image latent 的检索方式，对 image latent 自身分布敏感。

- **$W_O$ 影响混合**：concat heads 后的线性变换；改 $W_O$ 等于调"多头如何被融合"。

理论上：$\Delta(QK^T) = \Delta Q \cdot K + Q \cdot \Delta K + \Delta Q \cdot \Delta K$。前两项一阶项分别由 LoRA-Q 和 LoRA-K 贡献——所以 K 和 Q 影响 **attention map**；而 V 和 O 影响 **map 之后的 value pathway**。修 attention map 改"语义对齐"，修 value pathway 改"风格 / 内容"。这是社区"风格 LoRA 多动 V/O，身份 LoRA 多动 Q/K"经验的一种解释。

**对比 LLM**：LLM 推理时 Q 来自当前 token，K/V 来自 cache；调 Q 直接改"当前 query 的检索方式"，但 K/V 受 cache 影响——这是 LoRA 在 LLM 上更倾向 MLP / FFN 的部分原因（cross-attn 不是 LLM 主交互通道）。

只说"全套加 LoRA"；不区分 V / K / Q 的语义影响；不能解释为什么 SD 偏 attn、LLM 偏 MLP。

</details>

<details>

<summary>Q25.从生产部署视角，把 SDXL + LCM-LoRA + ControlNet + IP-Adapter 同时启用，怎么管理 sampler / CFG / 内存？</summary>

**Sampler 选择**：

- 加了 LCM-LoRA 必须用 **LCM scheduler**（4-8 步），不再适合 DPM++ / DDIM 的多步配置

- 4 步典型配置：scheduler `LCMScheduler`, num_steps=4, CFG **较低**（LCM 已做 distilled guidance，再叠普通 CFG 会过度饱和——LCM-Distill 模型常用 $s = 1.0$，即关 CFG；或显式用 W-CFG）

**CFG 与 ControlNet / IP-Adapter 的叠加**：

- CFG 是 text 的"差分放大"

- ControlNet 的 condition embedding 加在 U-Net side branch，**不参与** CFG 双 forward 内部数学（两份 batch 各自带相同 ControlNet input）

- IP-Adapter 的图像条件**应当**和 text 一起做 CFG drop（训练时图像条件也有 drop_rate），推理时 unconditional batch 的 IP 条件也置空

**内存 estimation**（FP16，1024²，batch 1）：

- SDXL U-Net forward activations: ~6-7 GB（gradient checkpointing 训练才能省）

- ControlNet (full): + ~3-4 GB

- IP-Adapter: +~0.2 GB（小）

- LoRA merged: 0（merge 后无额外开销）

- VAE decode peak: ~1.5 GB

- **总计 1024² 推理峰值 ~12-14 GB**，A10 (24GB) 单卡可，T4 (16GB) 紧但可 with attention slicing

**优化措施**：

1. **LoRA merge**：把 LCM-LoRA 与 style LoRA 都 merge 进 base，避免每 forward 多算 $\Delta W$

2. **xformers / FlashAttention**：把 cross-attn 与 self-attn 都 fused，节省 ~30% 时间和 ~20% 显存

3. **ControlNet quantize / pruning**：production 时常 quantize ControlNet 到 INT8，~1.5GB

4. **Schedule 上 sequential ControlNet 调用**：multiple ControlNets 不要并行算（OOM），sequential 调用聚合

5. **Cache text embedding**：同 prompt 多张图时只算一次 text encoder

**坑**：

- LCM-LoRA + ControlNet 经常出现"结构跟随减弱"——LCM 蒸馏路径里没见过 ControlNet 信号，需要 fine-tune ControlNet 在 LCM 路径上重训（或用 `ControlNet-LCM` 社区版本）

- IP-Adapter "Plus" 版本（ViT-G + image patches）会更吃显存，普通 ID 场景 ViT-L 版本即可

只会列工具名；不能算显存；不知道 LCM + ControlNet 的兼容性坑；不能区分 CFG 与 ControlNet / IP-Adapter 注入位置。

</details>

## §A 附录：参考与 reference list

### 主要论文（按时间）

- DDPM — Ho, Jain, Abbeel 2020 NeurIPS

- LDM / Stable Diffusion — Rombach, Blattmann et al. 2022 CVPR

- Classifier-Free Guidance — Ho & Salimans 2022 (workshop / arXiv)

- DiT — Peebles & Xie 2023 ICCV

- U-ViT — Bao, Nie et al. 2023 CVPR

- LoRA — Hu, Shen et al. 2022 ICLR

- DreamBooth — Ruiz, Li et al. 2023 CVPR

- Textual Inversion — Gal, Alaluf et al. 2023 ICLR

- Custom Diffusion — Kumari, Zhang et al. 2023 CVPR

- HyperDreamBooth — Ruiz, Li et al. 2024 CVPR

- ControlNet — Zhang, Rao, Agrawala 2023 ICCV

- T2I-Adapter — Mou, Wang et al. 2024 AAAI

- IP-Adapter — Ye, Zhang et al. 2023 (arXiv 2308.06721)

- InstantID — Wang, Bai et al. 2024 (arXiv 2401.07519)

- PuLID — Guo, Wu et al. 2024 NeurIPS

- PhotoMaker — Li, Cao et al. 2024 CVPR

- SDEdit — Meng, He et al. 2022 ICLR

- InstructPix2Pix — Brooks, Holynski, Efros 2023 CVPR

- Prompt-to-Prompt — Hertz, Mokady et al. 2023 ICLR

- SDXL — Podell, English et al. 2024 ICLR

- SD3 — Esser, Kulal et al. 2024 ICML

- FLUX.1 — Black Forest Labs 2024 (technical report)

- PixArt-α / Σ — Chen, Yu et al. 2024 ICLR / ECCV

- Hunyuan-DiT — Zhimin Li, Jianwei Zhang et al. 2024 (arXiv 2405.08748)

- Imagen — Saharia, Chan et al. 2022 NeurIPS

- ADD / SDXL-Turbo — Sauer, Lorenz et al. 2024 ECCV (arXiv 2311.17042)

- LCM — Luo, Tan et al. 2023 (arXiv 2310.04378)

- LCM-LoRA — Luo, Tan et al. 2023 (arXiv 2311.05556)

- DMD — Yin, Gharbi et al. 2024 CVPR

- ImageReward — Xu, Liu et al. 2023 NeurIPS

- HPSv2 — Wu, Hao et al. 2023 (arXiv 2306.09341)

- PickScore / Pick-a-Pic — Kirstain, Polyak et al. 2023 NeurIPS

- FID — Heusel, Ramsauer et al. 2017 NeurIPS

### 一句话总结

本 cheat sheet 覆盖从 latent diffusion 数学（VAE + DDPM/RF + CFG）到主流架构演进（SD 1.x → SDXL → SD3 → FLUX）、conditioning 体系（ControlNet / T2I-Adapter / IP-Adapter / InstantID）、个性化微调（DreamBooth / Textual Inversion / LoRA / Custom Diffusion）、编辑（SDEdit / InstructPix2Pix / Prompt-to-Prompt）、蒸馏（LCM / ADD / DMD）与评测（FID / CLIP-Score / ImageReward / HPSv2）。25 题按 L1/L2/L3 分布，L3 题强调 production lab 视角（零卷积 chain rule 推导、size conditioning 训练效果、MM-DiT 信息流、LoRA Q/K/V 选择、SDXL + LCM-LoRA + ControlNet + IP-Adapter 同栈部署 trade-off）。

---

## 📜 Runnable Code

本 tutorial 涉及的 MM-DiT / latent diffusion 组件在 [`docs/tutorials/code/`](code/) 里有最小可跑的 PyTorch 实现：

- [`mmdit_block.py`](code/mmdit_block.py) — 双流 MMDiT block：text + image 独立 Q/K/V、沿 seq 维 concat 跑一次 joint attention、per-stream FFN、AdaLN-Zero gated modulation
- [`toy_mmdit_t2i_pipeline.py`](code/toy_mmdit_t2i_pipeline.py) — 端到端 toy T2I pipeline：toy text encoder + 8× 16-channel VAE + 4-layer MMDiT + FlowMatchEuler scheduler + norm-preserving true CFG

每个脚本默认 CPU 几秒跑完，自带 `assert` sanity check（shape 对齐、AdaLN-Zero identity 验证、CFG scale=1 等价测试）。完整说明见 [`code/README.md`](code/README.md)。
