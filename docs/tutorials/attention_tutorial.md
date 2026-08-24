## §0 TL;DR Cheat Sheet

> 💡 **7 句话搞定 attention** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **公式**：$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V$。

2. **为什么除 √d_k**：若 $q_i, k_i \sim \mathcal{N}(0,1)$ 独立，$q\cdot k$ 方差 $= d_k$；除 $\sqrt{d_k}$ 把方差拉回 1，避免 softmax 饱和。

3. **Multi-Head**：把 $D$ 拆成 $H$ 个 head，每个 head 在不同 subspace 独立做 attention，concat 后 $W_o$ 投影。**固定 $D$ 且 $d_k=D/H$ 时，标准 MHA 参数量 $\approx 4D^2$（不随 $H$ 变）；MQA/GQA 下 K/V 投影变小**。

4. **Self vs Cross**：Self 的 Q/K/V 同源；Cross 的 Q 来自 query stream，K/V 来自 context stream（encoder output / image tokens / text embedding）。

5. **Causal mask vs Padding mask**：前者用下三角阻断未来；后者用 `[B,1,1,L_k]` 屏蔽 padding 列。

6. **复杂度**：$O(B H L^2 d_k)$ 时间，$O(B H L^2)$ score 显存——长序列瓶颈在二次项。

7. **易踩坑**：全 masked row → softmax NaN；FP16 下 $QK^\top$ 可能 overflow；attention weight ≠ 因果解释。

## §1 Attention 直觉

Attention 的本质是 **可学习的检索（learned retrieval）**：

- 每个 **query**（"我现在需要什么信息？"）

- 对所有 **key**（"每个位置宣称自己能提供什么"）计算相似度

- 用 softmax 归一化得到 **权重分布**

- 对所有 **value**（"每个位置实际提供的内容"）做加权求和

对比 RNN：RNN 把过去信息**压缩进一个固定大小的 hidden state**，长序列必丢信息；attention 在每一步都**直接、全局、动态地**检索过去所有位置，因此适合长程依赖。

"在 self-attention 中，Q/K/V 是同一个向量经过三个不同投影"——这点要主动说明，因为面试常考新手会以为 Q/K/V 是三份不同输入（注意 cross-attention 中 Q 与 K/V 并不同源，见 §4）。

## §2 Scaled Dot-Product Attention

### 2.1　公式

$$\boxed{\;\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V\;}$$

形状：

- $Q \in \mathbb{R}^{L_q \times d_k}$, $K \in \mathbb{R}^{L_k \times d_k}$, $V \in \mathbb{R}^{L_k \times d_v}$

- Scores $QK^\top \in \mathbb{R}^{L_q \times L_k}$（每个 query 对所有 key 的相似度）

- Softmax over **key 维度**：每个 query 行的权重和为 1

- Output $\in \mathbb{R}^{L_q \times d_v}$

### 2.2　为什么除以 √d_k（必考题，要会推方差）

假设 $q, k \in \mathbb{R}^{d_k}$ 的每个分量独立同分布，$q_i, k_i \sim \mathcal{N}(0,1)$。考虑点积：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

由独立性，每项 $q_i k_i$ 均值 $= \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$，方差 $= \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$。所以：

$$\mathbb{E}[q\cdot k] = 0, \quad \text{Var}[q\cdot k] = d_k$$

当 $d_k$ 大（如 64、128），$q\cdot k$ 的典型量级是 $\sqrt{d_k}$，进入 softmax 后**最大 logit 容易抢走绝大部分概率**，softmax 进入饱和区，梯度量级显著缩小，训练收敛变慢甚至停滞。除以 $\sqrt{d_k}$ 把方差拉回 1，**减轻饱和、改善梯度尺度**。

> ⚠️ **面试加分：FP16 下还有 overflow** — 即使除了 √d_k，FP16 下 `QK^T` 自己累加时也可能 overflow（fp16 max ≈ 65504）。生产实现用 fused SDPA / FlashAttention 或 **fp32 accumulation** 解决。`torch.softmax` 内部有 log-sum-exp 稳定化（减最大 logit 再 exp），但那是在 softmax 一步内做的，挡不住 matmul 累加的 overflow。

### 2.3　Mask 与 NaN 陷阱（💣 经典 bug，面试必问）

标准做法：把要屏蔽的位置 score 填成 $-\infty$，softmax 后那些位置概率 = 0。

但有个陷阱：**如果某一行所有 key 都被 mask**（如 query 0 在 cross-attn 中 context 全 padding；causal + 左 padding；某 query 后无任何合法 token），那一行 score 全是 $-\infty$，softmax 输出：

$$\text{softmax}([-\infty, -\infty, ..., -\infty]) = \text{NaN}$$

因为分子分母都是 $e^{-\infty} = 0$，$0/0 = $ NaN，污染整个 batch 的梯度。

> ✅ **修复：检测全 mask 行 → softmax 后清 0** —

```python
# 检测全 mask 行
all_masked = (~mask).all(dim=-1, keepdim=True)   # [..., L_q, 1]
# 临时给该行放开 (避免 NaN)
safe_mask = mask | all_masked
scores = scores.masked_fill(~safe_mask, float("-inf"))

# Softmax 正常计算
weights = F.softmax(scores, dim=-1)

# 把全 mask 行的输出强制设为 0 (否则会得到均匀分布)
weights = weights.masked_fill(all_masked, 0.0)
```

> ⚠️ **Mask 语义不对齐 (面试要主动 disambiguate)** — 本实现 / `F.scaled_dot_product_attention`：**True = keep**

`nn.MultiheadAttention` 的 `attn_mask` / `key_padding_mask`：**True = mask out**（相反！）

面试写代码前先问面试官约定，或主动声明你的约定，否则容易被搞反。

### 2.4　代码（核心 20 行）

```python
def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, training=True):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)                 # [..., L_q, L_k]
    scores = scores / math.sqrt(d_k)                 # ← 关键 scale

    if mask is not None:
        all_masked = (~mask).all(dim=-1, keepdim=True)
        safe_mask = mask | all_masked
        scores = scores.masked_fill(~safe_mask, float("-inf"))
    else:
        all_masked = None

    weights = F.softmax(scores, dim=-1)

    if all_masked is not None:
        weights = weights.masked_fill(all_masked, 0.0)   # NaN 防护

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    return weights @ V, weights                       # output, weights
```

## §3 Multi-Head Attention

### 3.1　公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_o$$

$$\text{head}_h = \text{Attention}(Q W_q^{(h)},\; K W_k^{(h)},\; V W_v^{(h)})$$

每个 head 的 $W_q^{(h)}, W_k^{(h)}, W_v^{(h)} \in \mathbb{R}^{D \times d_k}$，$d_k = D/H$。**工程上把 H 个 head 的投影矩阵 concat 成一个 $D \times D$ 大矩阵**，一次 matmul 跑完所有 head 的投影（GPU 友好）：

```

Input X [B, L, D]
   │
   │  W_q, W_k, W_v ∈ R^{D×D}   (每个 = concat of H 个 W^{(h)} ∈ R^{D×d_k})
   ↓
Q, K, V [B, L, D]
   │
   │  reshape [B, L, D] → [B, L, H, d_k] → transpose → [B, H, L, d_k]
   ↓
对每个 head 独立做 Scaled-Dot-Product Attention (batched matmul, 并行)
   ↓
heads [B, H, L_q, d_k]
   │
   │  transpose + reshape → [B, L_q, D]   (concat heads)
   ↓
W_o ∈ R^{D×D}    →    Output [B, L_q, D]
```

### 3.2　为什么要 multi-head（不是单 head 也行吗？）

- **不同 subspace**：每个 head 在自己的 $d_k$ 维子空间里学一种关系模式（语法、共指、远距依赖、局部 n-gram...）

- **表达力**：单 head 只能学一种 attention 模式；H 个 head 在 inference 时**并行**给出 H 种不同的 weighted sum 结果

- **参数效率**：$d_k = D/H$ 而不是 $D$，所以参数量不会随 H 线性增加

- 面试常问：head 越多越好吗？**不**。$d_k = D/H$ 太小（如 $d_k < 16$）会让每个 head 表达力受限；Mistral / LLaMA 用 head_dim ≈ 64-128 是 sweet spot

### 3.3　参数量与 FLOPs

| 组件 | 形状 | 参数量 |
| --- | --- | --- |
| $W_q$ | $D \times D$ | $D^2$ |
| $W_k$ | $D \times D$ | $D^2$ |
| $W_v$ | $D \times D$ | $D^2$ |
| $W_o$ | $D \times D$ | $D^2$ |
| **总计** |  | **$4D^2$**（不随 $H$ 变） |

FLOPs（单次 self-attention forward，$L_q = L_k = L$）：

- QKV projection: $3 \cdot 2 B L D^2 = 6 B L D^2$

- $QK^\top$: $2 B H L^2 d_k = 2 B L^2 D$

- Softmax weight × V: $2 B L^2 D$

- Output projection $W_o$: $2 B L D^2$

- **总计 $\approx 8 B L D^2 + 4 B L^2 D$**——前者随 $L$ 线性，后者随 $L$ 平方（长序列瓶颈）

### 3.4　代码（核心 30 行）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads

        # 合并 H 个 W^(h) 成一个 [D, D] 矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = dropout

    def _split(self, x):  # [B, L, D] → [B, H, L, d_k]
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):  # [B, H, L, d_k] → [B, L, D]
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split(self.W_q(query))
        K = self._split(self.W_k(key))
        V = self._split(self.W_v(value))

        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)   # [1,1,L_q,L_k]
            elif mask.dim() == 3: mask = mask.unsqueeze(1)              # [B,1,L_q,L_k]
            # dim=4: 已对齐

        out, w = scaled_dot_product_attention(Q, K, V, mask=mask, dropout_p=self.dropout_p, training=self.training)
        return self.W_o(self._merge(out)), w
```

### 3.5　fused QKV + chunk 写法（工程实现都这么写）

§3.4 用三个独立的 `W_q / W_k / W_v`，读起来最贴公式。**self-attention** 实现里更常见的是把三个投影**合成一个 Linear**，再把输出切开——nanoGPT、HF GPT-2、timm 的 ViT 都是这个写法（也有反例：HF LLaMA 就保持三个独立的 `q_proj / k_proj / v_proj`；cross-attention 则根本没法 fuse，因为 Q 和 K/V 来自不同输入）。下面三种切法**逐元素完全相同**（`torch.equal` 为 True），彼此只差可读性；省 GEMM 是 fused 相对 §3.4 三个 Linear 而言的。

```python
D, H, d_k = 64, 4, 16                       # D = H * d_k
qkv_proj = nn.Linear(D, 3 * D, bias=False)  # 一个 Linear 产出 Q/K/V
qkv = qkv_proj(x)                           # [B, N, 3D]

# 写法 1：chunk —— 最好读，白板上不容易写错
q, k, v = qkv.chunk(3, dim=-1)                          # 每个 [B, N, D]
q, k, v = [t.view(B, N, H, d_k).transpose(1, 2)         # → [B, H, N, d_k]
           for t in (q, k, v)]

# 写法 2：nanoGPT 的 split —— 与 chunk 等价（3D/3 = D），后面同样要接 view+transpose
q, k, v = qkv.split(D, dim=2)
q, k, v = [t.view(B, N, H, d_k).transpose(1, 2) for t in (q, k, v)]

# 写法 3：5-D reshape + permute —— timm ViT / 本仓库 code/mha.py 用的
q, k, v = qkv.reshape(B, N, 3, H, d_k).permute(2, 0, 3, 1, 4)
```

> 💡 **为什么要 fuse** —— 相比 §3.4 的三个 Linear，Q/K/V 只需**一次 GEMM** 而不是三次。FLOPs 完全一样，省的是 kernel launch 和激活读写，大矩阵乘的硬件利用率通常也更好（幅度取决于形状和后端）。另一个现实理由是权重排布——HF GPT-2 的 `c_attn`（`Conv1D`，权重 `[D, 3D]`）就是 Q/K/V 沿输出维按 `[Q | K | V]` 顺序拼的，想加载预训练权重就必须对齐这个顺序（搬到 `nn.Linear` 还得转置成 `[3D, D]`）。

| 写法 | 切分 | 可读性 | 用在哪 |
| --- | --- | --- | --- |
| 三个独立 Linear | 不用切 | 最贴公式 | 教学代码、§3.4 |
| fused + `chunk(3, dim=-1)` | 均分三份 | **最好** | 大部分手写实现 |
| fused + `split(D, dim=2)` | 每份宽 D | 好 | nanoGPT |
| fused + `reshape` + 5-D `permute` | 一步到位 | 差，容易写错轴顺序 | timm ViT、`code/mha.py` |

> ⚠️ **GQA 一来 `chunk(3)` 就是错的（高频追问）** —— MQA/GQA 下 fused 投影不再是三等份，而是 `D + 2·G·d_k`（见 §6.2）。取 `D=64, H=4, d_k=16, G=2`，fused 宽度是 `64 + 32 + 32 = 128`，`chunk(3)` 切出来是 `[43, 43, 42]`——**它不报错**，Q 拿到 43 而不是 64，要等到后面 `view` 才炸出一句和真正病因毫无关系的 `shape '[2, 7, 4, 16]' is invalid for input of size 602`。

`torch.chunk` 除不尽时不会抛异常，只是给出不等长的几份（宽 5 分 3 份 → `[2, 2, 1]`）。所以规则很简单：**Q 和 K/V 头数不一致的结构，一律用 `split` 显式给出三段宽度，别用 `chunk`。**

```python
q, k, v = qkv.split([D, G * d_k, G * d_k], dim=-1)      # 显式宽度，MQA/GQA 下的标准写法
```

## §4 Self / Cross / Causal / Padding

### 4.1　Self vs Cross Attention（必考）

|  | Self-Attention | Cross-Attention |
| --- | --- | --- |
| **Q 来源** | $X$ | $X_\text{decoder}$ / latent / learnable queries |
| **K, V 来源** | $X$（同源） | $X_\text{encoder}$ / context / memory |
| **$L_q$ vs $L_k$** | 相等 | 可以不同 |
| **典型 mask** | causal (decoder) 或 padding (encoder) | K/V 端 padding mask（不用 causal） |
| **用途** | 内部位置相关性 | 从外部 memory 检索相关信息 |
| **例子** | BERT 各层；GPT 各层；ViT | Transformer Decoder 第二子层；DETR；Perceiver；Stable Diffusion (image Q × text K/V) |

### 4.2　Causal Mask（Decoder / GPT）

下三角矩阵（含对角线）：第 $i$ 行可以看 $j \le i$ 的 key。

```python
def causal_mask(L, device=None):
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
# L=4 →
# [[T F F F]
#  [T T F F]
#  [T T T F]
#  [T T T T]]
```

### 4.3　Padding Mask（变长序列）

每个 sample 有效长度不同，padding token 不应被 attend：

```python
def padding_mask(lengths, max_len=None):
    if max_len is None: max_len = int(lengths.max())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(len(lengths), -1)
    return idx < lengths.unsqueeze(1)    # [B, L]    True=valid, False=padding

# 用法：必须 unsqueeze 成 [B, 1, 1, L_k] 才能 broadcast 到 MHA 内部 [B, H, L_q, L_k]
pmask = padding_mask(lengths).unsqueeze(1).unsqueeze(1)   # [B, 1, 1, L_k]
out, _ = mha(x, x, x, mask=pmask)
```

> 💡 **Causal + Padding 同时用** — 两个 mask 取 AND：`combined = causal_mask & padding_mask_4d`。注意 broadcast 维度对齐：causal 是 `[L,L]`，padding 是 `[B,1,1,L_k]`，AND 出来 `[B,1,L,L]`。

## §5 复杂度分析

|  | Time | Memory | 瓶颈 |
| --- | --- | --- | --- |
| RNN | $O(L \cdot D^2)$ | $O(D)$ | 顺序计算不可并行 |
| Self-Attention | $O(L^2 \cdot D)$ | $O(L^2 + L \cdot D)$ | $L^2$ score 矩阵（长序列） |
| Conv (kernel $k$) | $O(L \cdot k \cdot D^2)$ | $O(D)$ | 感受野有限 |

关键点：

- Self-attention 的 $L^2$ 项**计算**可以接受（GPU 并行），但 **$L^2$ 显存**（score 矩阵）是真正瓶颈——这是 Flash Attention 攻击的痛点

- LLM inference 时 prefill stage 是 $O(L^2)$；decode stage 用 KV cache 后每步是 $O(L)$（见 §6）

- 当 $L \approx D$ 时，attention 与 FFN 时间相当；当 $L \gg D$ 时，attention 占绝大部分时间

## §6 KV Cache + MQA / GQA

### 6.1　KV Cache（autoregressive inference 关键优化）

问题：GPT 自回归生成时，每生成一个 token，把整个 prefix 重新过 forward——单步代价是 $O(t^2)$，对 $T$ 步求和，累计冗余计算是 $O(T^3)$（而非 $O(T^2)$）。

解：把每层 $K^{(\ell)}, V^{(\ell)}$ 缓存。生成第 $t+1$ 个 token 时：

- 只对新 token 算 $q_{t+1}, k_{t+1}, v_{t+1}$（1 × D 大小）

- 将 $k_{t+1}, v_{t+1}$ append 到 cache

- 新 $q$ 对整个 cache 做 attention（$O(t)$ 不是 $O(t^2)$）

> ⚠️ **易踩坑** — KV cache 是 **推理优化**；标准 teacher-forcing 的并行训练（SFT / 预训练的前向-反向）用不到它，因为所有位置同时并行计算，没有"逐个生成"的概念。但 RLHF / GRPO 等含 on-policy rollout 采样的训练中，rollout 生成阶段本质上是自回归推理，同样会用 KV cache 加速。

**KV cache 显存（per sample）**：

$$\text{KV cache} = L_\text{ctx} \cdot n_\text{layers} \cdot \underbrace{2}_{K, V} \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes\_per\_elem}$$

注意：MQA/GQA 下 $H_\text{kv} \ll H$，cache 显著缩减。对 LLaMA-2-70B（GQA, $H_\text{kv}=8$）、$L_\text{ctx}=4096$、80 层、fp16：约 **1.25 GiB**（≈1.34 GB）/ sample——这就是 LLaMA-2 用 GQA 不用 MHA 的原因（vanilla MHA 会到约 **10 GiB ≈ 10.74 GB** / sample）。

### 6.2　MQA / GQA（attack KV cache 显存）

| 变体 | Q heads | K/V heads | KV cache 缩减 | 用在哪 |
| --- | --- | --- | --- | --- |
| **MHA** (Vanilla) | $H$ | $H$ | 1× | 原始 Transformer |
| **MQA** (Multi-Query) | $H$ | **1** | $H \times$ | PaLM, Falcon |
| **GQA** (Grouped-Query) | $H$ | $G$（$1 < G < H$） | $H/G \times$ | LLaMA-2-70B（7B/13B 仍是 MHA）、LLaMA-3 全系列、Mistral |

核心：**多个 Q head 共享一组 K/V**。MQA 极端但质量略降；GQA 是折中（如 H=32, G=8），显存 / 带宽降 4 倍，质量基本不掉。

> ❌ **易踩坑** — MQA/GQA 减少的是 **KV cache 显存 + 显存带宽**，**不是 Q projection 计算**（Q 头数不变）。面试常被反问"减了什么"。

## §7 FlashAttention 核心 Trick

问题：标准 attention 需要物化 $L \times L$ 的 score 矩阵，HBM 读写 IO 是瓶颈（不是 FLOPs）。

FlashAttention 思路（**IO-aware exact attention**，不是近似）：

1. **Block Tiling**：把 $Q, K, V$ 切成 block，每次只把一个 $Q$ block 和一个 $K, V$ block 加载到 SRAM

2. **Online Softmax**：边算边维护 running max $m$ 和 running sum $\ell$，避免一次性物化全部 scores

3. **Recompute on backward**：反向时重算 attention 而不存 $L^2$ scores

效果：

- **避免 materialize** 完整 $L \times L$ 的 scores / probs 矩阵到 HBM

- 论文给出的 HBM IO 复杂度约 $O(L^2 d^2 / M + Ld)$，对比标准 attention 的 $O(L^2 + Ld)$ HBM traffic——当 $L$ 大、$M$（SRAM）合适时 IO 减少显著

- 显存峰值从 $O(L^2)$ 降到 $O(L)$（不存中间 scores）

- 典型速度提升 2-4 倍（取决于 sequence length & GPU 架构）

- **数学上完全等价**（exact attention，不是 sparse / linear approximation）

> 💡 **FlashAttention v1/v2/v3 关键区别** — v1 (2022)：online softmax + block tiling + recompute。v2 (2023)：换内外循环 (Q-outer, KV-inner) + 更好 warp-level parallelism + 减少非 matmul FLOPs。v3 (2024)：针对 H100 Hopper，使用 WGMMA / TMA / FP8 + asynchronous pipeline。面试一般问 v1/v2，会问到 online softmax 细节。

## §8 Position Encoding (RoPE / ALiBi / Absolute)

| 方法 | 原理 | 外推性 | 用在哪 |
| --- | --- | --- | --- |
| **Sinusoidal absolute** | 固定 sin/cos 位置向量加到 input embedding | 位置编码本身可定义任意长度，但模型未必学到长度外的泛化 | 原始 Transformer (Vaswani 2017) |
| **Learned absolute** | 把位置当 token，学一个 embedding 表 | 差（位置 embedding 表是定长，硬限制） | BERT, GPT-2 |
| **RoPE** (Rotary) | 对 $Q, K$ 做位置相关的旋转：$q_m \to q_m e^{im\theta}$（复数视角）——**位置相关项通过相对位移 $m-n$ 进入内积**（内容向量仍影响分数） | 中等（自然包含相对位置；长度外推需 NTK-aware / YaRN） | LLaMA-1/2/3, Mistral, Qwen |
| **ALiBi** | 在 score 上加位置距离 bias：$\text{score}_{ij} - m \cdot \lvert i-j \rvert$ | 好（线性 bias 自然外推） | BLOOM, MPT |

### 8.1　Attention Sink（高级题）

训练好的 LLM 在 decode 时，注意力会异常集中在前 1-4 个 token（特别是 [BOS] / 第一个 token），即使内容无关。这种现象叫 **attention sink**。**常见直觉解释**：softmax 强制权重和为 1，当一个 query 实际不想 attend 任何 key 时，需要一个"垃圾位"来吸收概率质量；又因为 early tokens 对所有后续 token 都可见，训练中容易自然形成全局 sink。StreamingLLM (Xiao et al., ICLR 2024) 利用这个现象做长序列推理（保留 attention sink + 滑动窗口）。

## §9 Attention in Diffusion（generative 方向必问）

对 diffusion 背景的候选人，几乎必问 attention 在生成模型里的用法。

### 9.1　Latent Diffusion (Stable Diffusion) 里的 Cross-Attention

```

Image latent (z_t)  [B, C, H, W]
   │
   │  flatten to tokens [B, HW, D]
   ↓
Self-Attention (Q=K=V from image)
   ↓
Cross-Attention:
    Q = image tokens [B, HW, D]
    K, V = text embedding [B, L_text, D]    ← 文本条件
   ↓
FFN → next layer
```

关键点：

- Text-to-image conditioning 通过 cross-attention 实现：image tokens 是 query，text embedding 是 key/value

- Classifier-Free Guidance (CFG)：两次 forward（with text / without text）做差值。$\epsilon$-pred 时：$\epsilon_\text{CFG} = \epsilon_\text{uncond} + s (\epsilon_\text{cond} - \epsilon_\text{uncond})$；v-pred / x0-pred 时换成对应预测量，线性 guidance 形式类似

- SD / SDXL 的 U-Net 在多个 spatial resolution 的 transformer block 里都有 self-attn 与 cross-attn 交替

- DiT (Diffusion Transformer) 把 U-Net 换成 pure Transformer，conditioning 通过 AdaLN / cross-attn / token-concat 等方式

### 9.2　Attention 在 video diffusion

- **Spatial attention**：每帧内部（image patches 之间）

- **Temporal attention**：跨帧（同位置在不同时间步之间）

- **Spatiotemporal / full attention**：所有 frame × all positions——最昂贵，长视频不可行

- Long video 的 attention 是开放问题（$L \sim 10^4$-$10^5$ token），常见路线：因式化（空间 + 时间交替）、sparse window、hierarchical pooling、chunked attention

## §10 25 高频面试题

codex (gpt-5.5 xhigh) 作为顶级 lab 面试官视角列的，按难度分 3 档。每题点开看答案要点 + 易踩坑。

### L1必会题（任何 ML 工程岗都会问）

<details>

<summary>Q1.Attention 公式是什么？</summary>

- $\text{softmax}(QK^\top / \sqrt{d_k}) V$

- Softmax over keys 维度

- 输出是 value 的加权和

把 softmax 维度写到 query 维。

</details>

<details>

<summary>Q2.为什么除以 √d_k？</summary>

- 若 $q_i, k_i$ 独立零均值单位方差

- Dot product 方差约 $d_k$

- 缩放后方差回到 1，避免 softmax 饱和

只说"防止数值太大"，不给方差推导。

</details>

<details>

<summary>Q3.Q/K/V 分别代表什么？</summary>

- Q 是检索请求

- K 是匹配索引

- V 是被聚合内容

说 Q/K/V 是三份不同输入；self-attn 中它们同源但投影不同。

</details>

<details>

<summary>Q4.Multi-head 为什么有用？</summary>

- 不同子空间建模不同关系

- 多种位置/语义模式并行

- Concat 后再融合

说 head 越多一定越好。实际 $d_k$ 太小会限制表达力。

</details>

<details>

<summary>Q5.MHA 参数量随 head 数怎么变？</summary>

- 固定 $D$ 且 $d_k = D/H$（标准 MHA）

- $W_q + W_k + W_v + W_o$ 共 $4D^2$，**不随 $H$ 变**

- 但若用 MQA/GQA，K/V 投影矩阵会变小（$H_\text{kv} < H$ 个 head）

- 这就是为什么"head 数是免费的"在标准 MHA 下成立，但在 MQA/GQA 下有显存收益

误以为 head 多参数也线性多 H 倍；或忘了 MQA/GQA 改变了 K/V 投影维度。

</details>

<details>

<summary>Q6.Self-attention 和 cross-attention 区别？</summary>

- Self: Q/K/V 同源

- Cross: Q 来自 target，K/V 来自 context

- Cross 常用于 encoder-decoder、diffusion text conditioning

只说"cross 有两个输入"，不说明 Q 与 KV 来源。

</details>

<details>

<summary>Q7.写 causal mask 怎么做？</summary>

- `torch.tril(torch.ones(L, L, dtype=torch.bool))`

- 明确说 True=keep 还是 True=mask（API 间不一致）

- Broadcast 到 `[B, H, L, L]` 或让框架隐式 broadcast

上下三角写反；忘记 broadcast 维度对齐。

</details>

<details>

<summary>Q8.Padding mask mask 的是哪一维？</summary>

- 通常 mask key/value 列（让 padding 位置概率为 0）

- Shape 可为 `[B, 1, 1, L_k]` 对齐 head + query 维

- 注意：mask key 列**不足以**让 padded query 输出为 0；padded query 行通常用 loss ignore / output zeroing / packed sequence 等手段单独处理

以为 padding mask 一手包办——它只防止"看到 padding"，但 padded query 自己的输出还需要外部处理。

</details>

<details>

<summary>Q9.Attention 复杂度？</summary>

- 时间 $O(B H L_q L_k d_k) = O(B L^2 D)$

- Score memory $O(B H L_q L_k)$

- 长序列瓶颈是二次项

只说 $O(n^2)$，漏 head 和 hidden 维。

</details>

<details>

<summary>Q10.Attention dropout 放在哪里？</summary>

- 放在 softmax weights 之后、与 V matmul 之前

- Training 才启用，eval 时关闭

- Dropout 后权重行和不一定是 1（期望意义上为 1）

Sanity check 时还要求 dropout 后 row-sum = 1（错的）。

</details>

### L2进阶题（research-oriented 岗位）

<details>

<summary>Q11.手推 softmax 的 Jacobian。</summary>

- $y_i = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$

- $\dfrac{\partial y_i}{\partial x_j} = y_i (\delta_{ij} - y_j)$

- 矩阵形式：$J = \text{diag}(y) - yy^\top$

只写对角项，漏交叉项 $-y_i y_j$。

</details>

<details>

<summary>Q12.用 -∞ 做 mask 有什么坑？</summary>

- 正常情况 masked 位置 softmax 概率为 0 ✓

- **全 masked row → softmax 输出 NaN**（$0/0$）

- 修复路径：先避免 all-`-inf` 行（临时放开），softmax 后把该行 weights 与 output 强制清 0，并确保该 query 不进入 loss / 残差累积

- Fused kernel / API 对 sentinel 数值有约束；fp16 下用一个 dtype-safe 大负数（如 `finfo(dtype).min`）更稳

以为 -inf 永远安全；或只在 softmax 后清 0 而不防 NaN。

</details>

<details>

<summary>Q13.Log-sum-exp trick 是什么？</summary>

- softmax 前先减 max(logits)，等价不改变概率

- 防止 $e^{x_i}$ overflow（fp32 max ≈ 3.4e38，但 $e^{100}$ 已经溢出）

- $\log \sum_j e^{x_j} = m + \log \sum_j e^{x_j - m}$ 其中 $m = \max_j x_j$

忘了 $QK^\top$ overflow 可能发生在 softmax 之前（matmul 累加阶段）。

</details>

<details>

<summary>Q14.PyTorch nn.MultiheadAttention 的 in_proj_weight 顺序？</summary>

- Shape `[3D, D]`

- 顺序：**Q, K, V**（cat dim=0）

- Linear weight 是 `[out, in]`，所以 `cat([W_q.weight, W_k.weight, W_v.weight], dim=0)`

拼成 K/Q/V 或转置 weight。

</details>

<details>

<summary>Q15.attn_mask 和 key_padding_mask 区别？</summary>

- `attn_mask` 控制 query-key pair 级别（一般是 causal）

- `key_padding_mask` 控制 key token 整体可见性（一般是 padding）

- 两者 bool 语义：`nn.MultiheadAttention` 是 **True = mask out**；`F.scaled_dot_product_attention` 的 bool mask 是 **True = keep**（相反！）

- 同时用时，在 mask-out 语义下合并是 **OR**（任一为 True 就屏蔽）；在 keep 语义下是 AND（两者都 True 才保留）

不查 API 文档直接套用 True/False；或把 AND/OR 搞反。

</details>

<details>

<summary>Q16.Cross-attention 中 L_q 和 L_k 能否不同？</summary>

- 可以——这正是 cross-attention 的常态

- Scores shape 是 $[L_q, L_k]$

- Mask 必须对齐 key 维度

默认 cross-attn 必须等长。

</details>

<details>

<summary>Q17.为什么需要 output projection W_o？</summary>

- 融合不同 head 的输出

- 映射回 $d_\text{model}$ 与残差相加

- 给模型学习 head 间组合（不是简单 concat）

以为 concat 后已经结束。

</details>

<details>

<summary>Q18.Pre-norm vs post-norm 对 attention block 的影响？</summary>

- Pre-norm：`x + Attn(LN(x))`，深层训练更稳定，gradient 沿残差路径相对保持

- Post-norm：`LN(x + Attn(x))`，Vaswani 原论文用，超深时需 warmup / careful init

- **多数 decoder-only LLM 用 pre-norm（常配 RMSNorm 变体）**，但具体架构有例外

把 norm 位置当纯工程细节，或说"现代 LLM 都用 pre-norm" 太绝对。

</details>

<details>

<summary>Q19.Attention weight 等于"模型解释"吗？</summary>

- 可视化有参考价值（注意聚焦位置）

- 但 **不等于因果解释**

- Value 路径和后续层都会改变实际贡献

- Jain & Wallace "Attention is not Explanation" (2019)

直接把高 attention 权重当成"模型理由"。

</details>

<details>

<summary>Q20.Mixed-precision attention 注意什么？</summary>

- **fp32 accumulation**：matmul 累加 / softmax 关键步骤在 fp32 完成，再 cast 回低精度

- **Softmax max-subtraction**（log-sum-exp）防 exp overflow，PyTorch `F.softmax` 内部已做

- **Mask sentinel**：fp16 下用 `torch.finfo(dtype).min` 而非字面 -inf

- **BF16 vs FP16**：BF16 动态范围与 fp32 相近，更适合 attention；fp16 表数范围窄，QK^T 易 overflow

- **Fused kernels**（FlashAttention, `F.scaled_dot_product_attention`）内置 kernel-level 稳定化，比手写 naive 安全

FP16 下直接手写 naive attention 不做 fp32 accumulation。

</details>

### L3高级变体（顶级 lab / diffusion 方向）

<details>

<summary>Q21.KV cache 如何优化自回归解码？</summary>

- 解码第 $t+1$ 步时，只为新 token 算 $Q$（1×D）

- 复用历史 $K, V$（已经在 cache 里），append 新 $k_{t+1}, v_{t+1}$

- 每步 attention 从 $O(t^2)$ 变 $O(t)$，整段生成从 $O(L^3)$ 变 $O(L^2)$

- Per-sample 显存：$L_\text{ctx} \cdot n_\text{layers} \cdot 2 \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$（MQA/GQA 下 $H_\text{kv} \ll H$）

说 KV cache 减少训练成本——错。它只用于 autoregressive inference。另：cache 是 KV heads 数量，不是 Q heads。

</details>

<details>

<summary>Q22.MQA 和 GQA 解决什么？</summary>

- MQA：多个 Q head 共享一组 K/V（K/V 只有 1 个 head）

- GQA：折中，K/V 有 $G$ 组（$1 < G < H$）

- 主要收益：**decode 时 KV cache 显存 + 显存带宽**（大幅降低）

- 同时也减少 K/V projection 的参数和计算（K/V 投影矩阵变小），**但不减少 Q / O projection**

- 质量影响：通常 **GQA 质量损失小于 MQA**，具体取决于模型规模和训练方式（LLaMA-2 70B / LLaMA-3 / Mistral / Qwen-2 都用 GQA）

以为减少了 Q projection；或说"GQA 质量基本不掉"过于绝对。

</details>

<details>

<summary>Q23.FlashAttention 核心 trick？</summary>

- **Block tiling**：把 $Q, K, V$ 切成 SRAM-sized block，分批 load

- **Online softmax**：增量维护 running max $m$ 与 running sum $\ell$，**避免 materialize** 完整 $L \times L$ scores / probs 矩阵到 HBM

- **Recompute on backward**：反向时根据 saved $m, \ell$ 重算 scores，不存中间结果

- 关键：**IO-aware exact attention**（数学等价，不是近似）

- HBM IO 复杂度约 $O(L^2 d^2 / M + Ld)$，对比标准 attention 的 $O(L^2 + Ld)$ HBM traffic——长序列下显著减少 IO（不是 FLOPs）

说它是近似 attention（如 Performer / Linformer）——错，FlashAttn 是 exact；或把 IO 复杂度和 FLOPs 复杂度混淆。

</details>

<details>

<summary>Q24.RoPE, ALiBi, absolute position 的区别？什么是 attention sink？</summary>

- **Absolute**：位置向量加到 input embedding 上（Vaswani sinusoidal / GPT-2 learned）

- **RoPE**：对 $Q, K$ 做位置相关的旋转，保留**相对位置**信息（位置项只经由 $m-n$ 进入 $q_m^\top k_n$，但内容向量仍决定具体数值——不是只依赖位置）

- **ALiBi**：在 score 上加距离 bias $-m |i-j|$，自然外推

- **Attention sink**：训练好的 LLM 会让前 1-4 个 token（特别是 [BOS]）获得异常高的 attention，即使内容无关——softmax 强制和为 1，模型需要"垃圾位"。StreamingLLM 利用此现象做长序列推理。

把 attention sink 当成 padding / CLS token 的正常 attention 行为。

</details>

<details>

<summary>Q25.Attention 在 diffusion / latent diffusion 里怎么用？</summary>

- **U-Net latent tokens 作 Q**，text embedding 作 K/V，做 **cross-attention** 注入文本条件

- Self-attention 在每个 spatial resolution 内做（image patches × image patches）

- **CFG (Classifier-Free Guidance)**：两次 forward，差值放大 conditional 信号

- DiT (Diffusion Transformer)：把 U-Net 换成 pure Transformer，conditioning 通过 AdaLN / cross-attn / token-concat

- Video diffusion：空间 attn + 时间 attn + 时空 attn 的组合（长 video 是开放问题，$L \sim 10^5$）

说 diffusion 只靠卷积；或者只在 DiT 里才有 attention（错，U-Net 里也有大量）。

</details>

## §A 附录：完整 from-scratch 代码骨架

参考 from-scratch 实现（`code/mha.py`）包含：

- `MultiHeadAttention`——标准多头 self-attention（单个 fused `qkv` Linear 一次产出 Q/K/V，支持 `[N,N]` additive mask；无 NaN 全 mask 防护，无 cross-attention wrapper）；`__init__` 里 `assert embed_dim % num_heads == 0` 校验维度，不满足时抛 `AssertionError`（注意 `assert` 在 `python -O` / `PYTHONOPTIMIZE` 下会被整体移除，生产代码应改用显式 `raise ValueError(...)`）

- `make_causal_mask()`—— 生成上三角填 `-inf` 的因果 mask

- 3 个 sanity check：`sanity_check()`（对齐 `nn.MultiheadAttention`，diff < 1e-5）、`shape_check()`、`causal_check()`

实跑 sanity check 输出（PyTorch 2.x，CPU 几秒跑完）：

```
== Multi-Head Attention sanity ==
[sanity] max diff vs nn.MultiheadAttention = 5.96e-08
[sanity] PASS
[shape] input  x: (2, 1024, 768)
[shape] output y: (2, 1024, 768)
[causal] applied causal mask, output shape (1, 8, 32)

All checks passed.
```

代码经独立 reviewer 静态检查 + PyTorch 实跑 sanity check，与 `nn.MultiheadAttention` diff < 1e-5。

---

## 📜 Runnable Code

本 tutorial 的核心概念在 [`docs/tutorials/code/`](code/) 里有最小可跑的 PyTorch 实现：

- [`mha.py`](code/mha.py) — 标准 Multi-Head Self-Attention + causal mask + 跟 `nn.MultiheadAttention` 数值对齐验证
- [`axial_attention.py`](code/axial_attention.py) — H/W 轴向 attention + 复杂度对比表 + 感受野隔离测试

每个脚本默认 CPU 几秒跑完，自带 `assert` sanity check。完整说明见 [`code/README.md`](code/README.md)。
