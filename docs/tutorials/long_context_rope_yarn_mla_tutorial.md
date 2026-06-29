## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 Long Context** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **RoPE**：对每对维度 $(2i, 2i+1)$ 做位置 $m$ 相关的 2D 旋转，$\theta_i = 10000^{-2i/d}$。$q_m^\top k_n$ 仅依赖**相对位置** $m-n$（不依赖绝对 $m, n$ 各自），且无需训练参数。

2. **PI (Position Interpolation, Chen 2023)**：把所有 $\theta_i$ 同除以 $s = L_\text{new}/L_\text{train}$（等价于把绝对位置 $m$ 缩到 $m/s$）。**伤害高频**（早期维度的相位分辨率被压缩），但实现简单。

3. **NTK-aware (bloc97 2023)**：换底，新底 $b' = b \cdot s^{d/(d-2)}$。**低频维度被强压缩、高频维度几乎不变**，零样本外推优于 PI。

4. **YaRN (Peng 2023)**：NTK-by-parts（分段处理频率）+ temperature scaling（拟合公式 $\sqrt{1/t} \approx 0.1\ln s + 1$，即 $t \approx 1/(0.1\ln s + 1)^2$）+ attention scale。三组件分别解决：高频/低频分别处理、稀释 softmax、补偿外推后注意力熵增。

5. **LongRoPE (Ding 2024 ICML)**：演化搜索每维独立的缩放因子 $\lambda_i$，加上 short-context "rescue"，把上下文推到 2M tokens。

6. **MLA (DeepSeek-V2)**：$\mathbf{c}_t^{KV} = W_\text{DKV}\mathbf{h}_t$ 把 K/V 压成 $d_c \ll N_h d_h$ 的 latent；RoPE 必须**解耦**——单独留一份 $d_h^R$ 维 RoPE key（共享 across heads），否则旋转矩阵不能"吸进" $W_\text{UK}$ 里。

7. **Streaming + Sink (Xiao 2024 ICLR)**：保留前 4 个 token（attention sink，softmax 的"垃圾桶"）+ 滑动窗口；window 外的 token 直接丢，但 sink 不能丢，否则 PPL 爆炸。

8. **System**：Ring Attention / Context Parallel 跨设备 chunk K/V；FlashAttention 2/3 块化 + online softmax；Mistral SWA 把每层感受野从 $L$ 降到 $W$（多层叠加仍可看远）。

## §1 Long Context 为什么难 — 一段直觉

把 Transformer 推到 100K-2M token 上下文，难点其实是 **三件事并存**：

- **位置编码外推 (extrapolation)**：训练时只见过 $m \in [0, 4096)$，推理时给 $m = 100{,}000$，模型必须知道这是"很远"而不是数值崩坏。**RoPE 默认不外推**：未见过的旋转相位让 $q_m^\top k_n$ 的相对位置信号失效。

- **KV cache 显存**：自回归 decode 时 $\text{cache} \propto L \cdot n_\text{layers} \cdot 2 \cdot N_h \cdot d_h \cdot \text{bytes}$。对 LLaMA-2-7B（32 层, $N_h=32, d_h=128$, fp16，MHA）一条 4K 上下文 $\approx 2.1$ GB，一条 100K $\approx 52$ GB，单卡装不下。MQA/GQA 把 $N_h \to G$，MLA 把 $N_h d_h \to d_c$。

- **注意力本身的 $O(L^2)$**：$L=100\text{K}$ 时 $L^2 = 10^{10}$，分数矩阵装不下。两条路：**算法变稀疏/线性**（sliding window, sparse attention, linear attention）或 **系统切分**（Ring Attention, Context Parallelism, FlashAttention 块化）。

> ⚠️ **一句话区分扩展方法** — RoPE 系（PI / NTK / YaRN / LongRoPE）解决"位置编码外推"；MLA / MQA / GQA 解决"KV cache 显存"；FlashAttention / Ring / SWA / Sink 解决"算注意力的时间和显存"。**三者正交**，工业级长上下文模型（如 DeepSeek-V2、Qwen2.5-1M、Llama-3.1-405B）通常同时用三类。

## §2 RoPE — 旋转位置编码

### 2.1 复数视角推导

**目标**：找一个对 query/key 的位置相关变换 $f(\mathbf{x}, m)$，使得内积 $\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle$ 只依赖于**相对位置** $m - n$（以及 $\mathbf{q}, \mathbf{k}$ 内容本身），不再依赖绝对 $m, n$。

把 $\mathbf{x} \in \mathbb{R}^d$ 按相邻两维分组打包成 $d/2$ 个复数：$\mathbf{x} \leftrightarrow [x_0 + ix_1, x_2 + ix_3, \dots] \in \mathbb{C}^{d/2}$。定义

$$f(\mathbf{x}, m) = \mathbf{x} \odot e^{im\boldsymbol\theta}, \quad e^{im\boldsymbol\theta}_i = e^{im\theta_i}, \quad \theta_i = b^{-2i/d}\ (b = 10000)$$

其中 $\odot$ 是逐元素复数乘法。由复数乘法性质：

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle_\mathbb{R} = \mathrm{Re}\!\left[(\mathbf{q} \odot e^{im\boldsymbol\theta})^* (\mathbf{k} \odot e^{in\boldsymbol\theta})\right] = \mathrm{Re}\!\left[\sum_{i=0}^{d/2-1} \bar{q_i} k_i \cdot e^{i(n-m)\theta_i}\right]$$

只依赖 $n - m$（与 $\bar{q_i}k_i$，即内容项），**绝对位置项消掉**——这就是 RoPE 给出相对位置的根本原因。

> ✅ **几何直觉** — 把每对维度 $(x_{2i}, x_{2i+1})$ 想成 2D 平面上的向量，RoPE 就是在每个 2D 子空间里转一个角度 $m\theta_i$（不同 $i$ 转速不同）。query 和 key 都被旋转后做内积，**相对角度**保留，绝对方向被抵消。

### 2.2 实数矩阵形式

每对维度上是 2D 旋转矩阵：

$$R_{\theta_i, m} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$$

把 $\mathbf{x}$ 视作 $d/2$ 个 2D 向量的拼接，整体 $R_m = \mathrm{blkdiag}(R_{\theta_0, m}, \dots, R_{\theta_{d/2-1}, m})$。则：

$$\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{n-m} \mathbf{k}$$

最后一步用 $R_m^\top R_n = R_{n-m}$（2D 旋转的可加性）。**相对位置 $n - m$ 被显式编码到内积里**。

### 2.3 为什么 $\theta_i = 10000^{-2i/d}$（频率分布）

把 $\theta_i$ 看作角速度。维度 $i$ 越大，$\theta_i$ 越小，旋转**越慢**（低频）；维度 $i$ 越小（接近 0），$\theta_i$ 越接近 1，旋转**越快**（高频）。

- **高频维度**：周期短（$2\pi/\theta_i$ 短），位置变化对相位敏感——编码精细的局部相对位置
- **低频维度**：周期长（最大 $2\pi \cdot 10000$），相位随位置缓变——编码粗粒度的远程位置

这种 **几何级数频率分布** 与 Vaswani 2017 sinusoidal PE 相同（不是巧合：sinusoidal PE 也是 $10000^{-2i/d}$），让模型在多个时间尺度上同时分辨位置。

> 💡 **波长 vs 训练上下文** — 维度 $i$ 的波长 $\lambda_i = 2\pi b^{2i/d}$。当 $\lambda_i$ **超过训练长度** $L$ 时，这个维度在训练中没见过完整周期——这就是 NTK-by-parts 的关键观察：低频维度的相位插值很危险（外推时进入未见区域），高频维度安全。

### 2.4 from-scratch RoPE 代码

> 📝 **说明** — 后续所有 Python 代码块共享本块的 import 前导（`import torch` 等）；单独复制运行某个后续块时需自行补上 `import torch`。

```python
import torch

def precompute_rope_cache(seq_len: int, dim: int, base: float = 10000.0, device=None):
    """
    返回 cos / sin tensor，shape [seq_len, dim/2]，逐对维度共享旋转角。
    dim 必须是偶数 (RoPE 按相邻两维成对旋转)。
    """
    assert dim % 2 == 0, "RoPE dim must be even"
    half = dim // 2
    # θ_i = base^{-2i/dim},  i = 0, 1, ..., dim/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()              # [L]
    freqs = torch.outer(pos, inv_freq)                              # [L, dim/2]
    return freqs.cos(), freqs.sin()                                 # [L, dim/2] each

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   [..., L, dim]              (Q 或 K)
    cos: [L, dim/2]  sin: [L, dim/2]
    实现实数形式: 把 x 拆成两半 x1, x2，对应复数实/虚部，做 2D 旋转。
    约定 (HuggingFace LLaMA style): pair = (x[..., :half], x[..., half:])
        而不是 (x[..., 0::2], x[..., 1::2])。
    数学等价 (取决于实现约定，二者只是排列不同)。
    """
    x1, x2 = x.chunk(2, dim=-1)                                     # 每个 [..., L, dim/2]
    # 旋转: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)

# 完整流程示例 ——————————————————————————————————————
def rope_attention(Q, K, V, cos, sin, mask=None):
    """
    Q, K, V: [B, H, L, d_head]
    cos, sin: [L, d_head/2]  (可被 broadcast)
    """
    Q = apply_rope(Q, cos, sin)
    K = apply_rope(K, cos, sin)
    scores = (Q @ K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ V
```

> ⚠️ **复数 vs 实数实现差异** — Meta 官方 LLaMA repo 用 **复数 view** (`torch.view_as_complex`)，HuggingFace transformers 用 **实数 chunk 形式**（上面代码）。HF 的"前半 / 后半"约定与原始论文"奇偶交错"约定**仅排列不同**，对最终 attention 输出**数学等价**。**但 RoPE cache 的预计算与你选择的 pairing 必须一致**——混用会导致旋转作用在错维度上，效果近似随机。这个 bug 在 HuggingFace `LlamaRotaryEmbedding` 版本变迁中真出现过。

## §3 朴素位置编码回顾（对比基线）

| 方法 | 形式 | 相对位置？ | 外推性 | 用在哪 |
| --- | --- | --- | --- | --- |
| **Sinusoidal absolute** (Vaswani 2017) | $\mathrm{PE}_{m, 2i} = \sin(m / 10000^{2i/d})$，加到 input | 否（绝对） | 差（外推区域模型未见过） | 原始 Transformer |
| **Learned absolute** | 位置当 token，查 embedding 表 | 否 | **极差**（embedding 表定长） | BERT, GPT-2 |
| **Relative bias** (T5) | 加到 logits 的 learned bias（按相对距离 bucket） | 是 | 中等（bucket 外饱和） | T5 |
| **ALiBi** (Press 2022) | $\text{score}_{ij} - m_h \cdot \lvert i - j \rvert$，head 相关斜率 $m_h$ | 是 | **好**（线性 bias 自然外推） | BLOOM, MPT |
| **RoPE** (Su 2021/2024) | $q_m \to q_m e^{im\boldsymbol\theta}$ 旋转 | 是 | 中等（默认）；配合 NTK/YaRN 可推至 100K-2M | LLaMA-1/2/3, Mistral, Qwen, DeepSeek |
| **NoPE** (Kazemnejad 2023) | 完全不加位置编码 | 经由 causal mask 间接 | 意外地不错（decoder-only 小模型场景） | 研究性 |

> 💡 **为什么社区收敛到 RoPE** — 三点合一：(1) 无训练参数（vs learned absolute），(2) 显式相对位置（vs sinusoidal），(3) 实现简单且与 multi-head 兼容（每 head 独立旋转）。ALiBi 外推更好但表达力略弱（只是单调距离衰减），RoPE 让模型自己学复杂的位置-内容耦合。

## §4 PI — Position Interpolation（最简单的 RoPE 外推）

### 4.1 动机

训练时 $m \in [0, L_\text{train})$；推理给 $m \in [0, L_\text{new})$，$L_\text{new} > L_\text{train}$。**RoPE 直接外推会崩溃**：当 $m \theta_i$ 超过训练见过的相位范围（特别是低频维度上 $m\theta_i$ 已经接近 $2\pi$ 时），相位进入未见区域，attention 行为不可预测。

PI 的想法（Chen et al., Meta, 2023, "Extending Context Window of LLMs via Position Interpolation"）：**不外推，去插值**。把 $m \in [0, L_\text{new})$ 线性压缩到 $[0, L_\text{train})$。

### 4.2 形式

设缩放因子 $s = L_\text{new} / L_\text{train}$。把绝对位置 $m$ 替换为 $m / s$：

$$\text{PI:}\quad f(\mathbf{x}, m) = \mathbf{x} \odot e^{i (m/s) \boldsymbol\theta}$$

等价地（更常见的实现）：保持 $m$ 不变，把所有 $\theta_i$ 替换为 $\theta_i / s$。两种说法**完全等价**。

### 4.3 副作用：高频被破坏

低频维度上 $m\theta_i$ 本来就 $\ll 2\pi$（在训练长度内未完成一周期），缩到 $m\theta_i / s$ 仍在合理范围。**问题在高频**：高频维度 $\theta_i \approx 1$，训练时 $m\theta_i$ 已经在 $[0, L_\text{train}]$ 范围内自由旋转，缩到 $1/s$ 后**相对位置分辨率下降 $s$ 倍**——本来相邻 token 之间相位差为 $\theta_i$（接近 1 rad），现在只剩 $\theta_i/s$，模型分辨"相距 1 token vs 相距 2 token"的能力下降。

> ⚠️ **必须 fine-tune 才能恢复** — 原论文用 PI 做 zero-shot 评估时 PPL 会变差；用 1000 步 fine-tune 即可基本恢复，并能稳定扩到 32K 上下文。

### 4.4 PI 代码

```python
def precompute_rope_cache_pi(seq_len: int, dim: int,
                              base: float = 10000.0,
                              scale: float = 1.0,        # s = L_new / L_train
                              device=None):
    """PI: 把 θ_i 同除以 s（等价于把 m 缩到 m/s）"""
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    inv_freq = inv_freq / scale                         # ← PI 的关键一行
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()
```

## §5 NTK-aware RoPE — 保留高频的换底方案

### 5.1 起源与直觉

PI 把高频维度也压扁了，社区觉得太粗暴。**bloc97 / jquesnelle** 在 LocalLLaMA 2023 年 7 月的 reddit 帖里提出 NTK-aware scaling（"NTK-Aware Scaled RoPE"），名字来源于 Neural Tangent Kernel 中"高频 vs 低频"的视角：神经网络对高频信号学习慢，破坏高频比破坏低频更伤模型。

**核心想法**：**换底**而不是统一缩放——让高频维度几乎不变（保护精细位置），低频维度被强压缩（这部分本来训练就没见过完整周期，影响小）。

### 5.2 推导：怎样换底才能让低频被压到 $1/s$？

回顾 RoPE 频率 $\theta_i = b^{-2i/d}$。

- **最高频** ($i = 0$)：$\theta_0 = 1$
- **最低频** ($i = d/2 - 1$)：$\theta_{d/2-1} = b^{-(d-2)/d} \approx b^{-1}$（当 $d$ 大时）

PI 把所有 $\theta_i$ 都除以 $s$，等价于把所有维度的位置分辨率打 $s$ 折。

NTK-aware：换底从 $b$ 到 $b'$，使得**最低频**的 $\theta$ 被缩到 $1/s$，**最高频**的 $\theta$ 几乎不变。

设 $b' = b \cdot \alpha$。则新最低频是

$$\theta'_{d/2-1} = (b')^{-(d-2)/d} = b^{-(d-2)/d} \cdot \alpha^{-(d-2)/d}$$

要让 $\theta'_{d/2-1} = \theta_{d/2-1} / s$，需要

$$\alpha^{-(d-2)/d} = 1/s \quad\Longrightarrow\quad \alpha = s^{d/(d-2)}$$

所以

$$\boxed{\;b' = b \cdot s^{\,d/(d-2)}\;}$$

**验证最高频**：$\theta'_0 = (b')^0 = 1 = \theta_0$，完全不变。$\checkmark$

**渐进**：维度 $i$ 上 $\theta'_i / \theta_i = \alpha^{-2i/d} = s^{-2i/(d-2)}$。$i = 0$ 时比值 1（不变），$i = d/2-1$ 时比值 $1/s$（强压缩）。从高频到低频压缩比**指数过渡**——这就是"NTK-aware"的几何含义。

### 5.3 与 PI 的对比

| 维度 | PI | NTK-aware |
| --- | --- | --- |
| **最高频 ($i=0$)** 缩放 | $1/s$（破坏） | **$1$**（保留） |
| **最低频** 缩放 | $1/s$ | $1/s$ |
| **中间维度** 缩放 | 一律 $1/s$（线性） | $s^{-2i/(d-2)}$（指数过渡） |
| **零样本 PPL** ($s=4$ on LLaMA-7B) | 大幅恶化 | 接近原 PPL |
| **需要 fine-tune** | 是 | 否（零样本可用） |

### 5.4 NTK-aware 代码

```python
def precompute_rope_cache_ntk(seq_len: int, dim: int,
                               base: float = 10000.0,
                               scale: float = 1.0,        # s = L_new / L_train
                               device=None):
    """
    NTK-aware: 换底 b' = b * s^{d/(d-2)}
    - 最高频维度 (i=0) θ 不变;
    - 最低频维度 (i=d/2-1) θ 被压到 1/s;
    - 中间维度按 i 指数过渡。
    """
    new_base = base * (scale ** (dim / (dim - 2)))
    half = dim // 2
    inv_freq = 1.0 / (new_base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()
```

> ⚠️ **NTK-aware 的局限** — 在更大的扩展比 ($s \ge 8$) 下，最低频维度被压得太狠，仍会出现性能下降。这促成了 NTK-by-parts 的进一步改进，把不同频段**分段处理**——这正是 YaRN 的起点。

## §6 YaRN — Yet another RoPE extensioN

### 6.1 总览

Peng et al. 2023 ("YaRN: Efficient Context Window Extension of Large Language Models") 把 NTK-aware 思路系统化，拆成三个相对独立的组件：

1. **NTK-by-parts**：按波长把维度分三段，分别处理
2. **Temperature scaling**：在 softmax 之前对 logits 整体加温度
3. **Attention scale**（与温度等价的另一种实现）：把 Q/K 的范数同步缩放

下面分别推导。

### 6.2 NTK-by-parts — 分频段处理

设 $L_\text{train}$ 是训练上下文长度。维度 $i$ 的波长 $\lambda_i = 2\pi / \theta_i = 2\pi b^{2i/d}$。定义比值

$$r_i = \frac{L_\text{train}}{\lambda_i} = \frac{L_\text{train} \cdot \theta_i}{2\pi}$$

$r_i$ 表示训练长度内维度 $i$ 转了多少圈。把维度分三段：

| 区间 | 条件 | 含义 | 处理 |
| --- | --- | --- | --- |
| **高频** | $r_i \ge \beta$ ($\beta=32$) | 训练内 ≥ 32 圈，相对位置完全采样 | **不缩放** ($\theta'_i = \theta_i$) |
| **中频** | $\alpha < r_i < \beta$ ($\alpha=1$) | 部分采样 | **线性插值**（PI 的局部应用） |
| **低频** | $r_i \le \alpha$ | 训练内 < 1 圈，位置编码没见过整周期 | **完全缩放到 $1/s$**（PI 行为） |

形式化：对维度 $i$，定义 ramp 函数

$$\gamma(r_i) = \mathrm{clip}\!\left(\frac{r_i - \alpha}{\beta - \alpha},\; 0,\; 1\right) \in [0, 1]$$

新频率取 NTK-aware 与 PI 的插值：

$$\theta'_i = (1 - \gamma(r_i)) \cdot \frac{\theta_i}{s} + \gamma(r_i) \cdot \theta_i$$

- $r_i \ge \beta$（高频）：$\gamma = 1$，$\theta'_i = \theta_i$（不变）
- $r_i \le \alpha$（低频）：$\gamma = 0$，$\theta'_i = \theta_i / s$（PI 全缩放）
- 中间：平滑过渡

> 💡 **三段化的理由** — 高频维度在训练时已完整旋转过多周期，外推时**只要相位别跳跃**就能继续工作（旋转的周期性）；低频维度在训练时连一圈都没转完，"外推区域"对模型完全是未见数据，必须**插值**进训练见过的相位范围。中频维度居中处理。

### 6.3 Temperature Scaling — 注意力熵补偿

**问题**：扩展上下文后 softmax 输入的有效统计变了——同样的 query 现在面对 $L_\text{new} \gg L_\text{train}$ 个 key，attention 分布更**扁平**（熵更高），有效信号被稀释。

**解决**：在 softmax 前给 logits 除以温度 $t$（$t < 1$ 让分布更尖锐，补偿稀释）：

$$\mathrm{Attention} = \mathrm{softmax}\!\left(\frac{QK^\top}{t \sqrt{d_k}}\right) V$$

YaRN 论文的拟合公式（来自经验性 ablation）：

$$\boxed{\;\sqrt{1/t} \approx 0.1 \ln s + 1 \quad\Longleftrightarrow\quad t \approx \frac{1}{(0.1 \ln s + 1)^2}\;}$$

例如 $s = 8$ ($L_\text{new} = 32\text{K}$ from $L_\text{train} = 4\text{K}$)：$\sqrt{1/t} \approx 0.1 \cdot 2.08 + 1 \approx 1.21$，$t \approx 0.68$。

### 6.4 Attention Scale — 与 Temperature 等价的另一实现

直接改 softmax 温度需要改 attention kernel。等价做法：把 query 和 key 的范数同时**乘以** $\sqrt{1/t} = (0.1\ln s + 1) > 1$（$t < 1$ 时是放大），这样 $QK^\top$ 自然被放大 $1/t$ 倍，softmax 看到的就是除以 $t$ 后的 logits。

YaRN 通过把 RoPE cache 直接乘进缩放因子来实现：

$$\text{cos}'_m = \cos(m \theta'_i) \cdot \sqrt{1/t}, \quad \text{sin}'_m = \sin(m \theta'_i) \cdot \sqrt{1/t}$$

注意这只对 RoPE 部分起作用，**整体效果等价于把 query/key 范数放大 $\sqrt{1/t}$ 倍**（$t < 1$ 时该因子 $> 1$）——前提是 Q/K 的范数主要由 RoPE 后的部分主导。实践中 YaRN 的 attention scale 实现就是把 cos/sin 缓存里乘上 $\sqrt{1/t}$。**这等价于改温度且不动 attention kernel**。

### 6.5 YaRN 三组件各解决什么（L3 必问）

| 组件 | 解决的问题 | 不要它会怎样 |
| --- | --- | --- |
| **NTK-by-parts** | 高频应保留，低频应插值，中频要平滑过渡 | 用 NTK-aware 全局换底，扩展比大时低频压崩 |
| **Temperature scaling** | 上下文变长后 softmax 分布被稀释 | 注意力熵过高，长程信号被淹没 |
| **Attention scale (实现层面)** | 不动 softmax kernel 实现温度 | 需要重写 FlashAttention kernel |

YaRN 论文展示：仅 400 步 fine-tune 把 LLaMA-2-7B 从 4K 推到 128K（$s = 32$），优于 PI 和 NTK-aware。

### 6.6 YaRN 代码（NTK-by-parts + temperature）

```python
import math

def precompute_rope_cache_yarn(
    seq_len: int, dim: int,
    base: float = 10000.0,
    scale: float = 1.0,            # s = L_new / L_train
    original_max_pos: int = 4096,  # L_train
    alpha: float = 1.0,            # ramp 下界 (圈数)
    beta: float = 32.0,            # ramp 上界 (圈数)
    device=None,
):
    """
    YaRN: NTK-by-parts + temperature scaling (实现为 attention scale).
    - 高频维 (r_i ≥ β):  不缩放
    - 低频维 (r_i ≤ α):  PI 风格全缩放
    - 中间维 (α < r_i < β): 平滑过渡
    """
    half = dim // 2
    i = torch.arange(0, half, device=device).float()                 # [half]
    inv_freq = 1.0 / (base ** (i / half))                            # θ_i
    wavelen = 2.0 * math.pi / inv_freq                               # λ_i
    r = original_max_pos / wavelen                                   # r_i = L_train / λ_i

    gamma = torch.clamp((r - alpha) / (beta - alpha), 0.0, 1.0)      # ramp ∈ [0,1]
    inv_freq_pi   = inv_freq / scale                                  # PI 全缩放
    inv_freq_ntk  = inv_freq                                          # NTK 不缩 (高频)
    inv_freq_yarn = (1.0 - gamma) * inv_freq_pi + gamma * inv_freq_ntk

    # Temperature scaling (作为 attention scale 实现到 cos/sin 上)
    # YaRN 经验公式:  sqrt(1/t) ≈ 0.1 ln(s) + 1
    # 目标: 让 effective QK^T 被放大 1/t 倍 (等价于 softmax 温度变 t<1 → 更尖锐).
    # 实现: Q 和 K 的范数各乘 sqrt(1/t), 则 QK^T 自然乘以 1/t.
    # 因为 RoPE 用 cos/sin 旋转, 把 sqrt(1/t) 乘到 cos/sin 上即可.
    sqrt_inv_t = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0
    attn_scale = sqrt_inv_t                                           # ← 乘到 cos/sin 上, 放大 Q/K 范数

    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq_yarn)
    return freqs.cos() * attn_scale, freqs.sin() * attn_scale
```

> ⚠️ **YaRN 的 attention scale 副作用** — Q/K 范数被**放大** $\sqrt{1/t} > 1$ 倍（不是缩小），但 V 没有同步放大。在多层 transformer 里这等价于改了每层 attention 的有效温度，反向传播时梯度尺度也会不同。实践中 fine-tune 才能稳定（YaRN 论文用 ≈ 400 步）。

## §7 LongRoPE — 演化搜索 + 短上下文 rescue

Ding et al., ICML 2024 (Microsoft) 进一步问：**每个维度的最优缩放因子能否独立搜索**，而不是用同一个 ramp 函数？

### 7.1 关键观察

1. **不同维度对扩展长度的敏感性差异极大**（同一 ramp 函数不一定最优）。
2. **超长上下文模型在短上下文 (≤ $L_\text{train}$) 上反而退化**——因为 RoPE 缓存被改了，原训练分布被打乱。

### 7.2 三阶段方案

| 阶段 | 做什么 |
| --- | --- |
| **Stage 1: Evolution search (256K)** | 每个 RoPE 维度独立缩放 $\lambda_i$，演化算法搜索使长上下文 PPL 最低的 $\{\lambda_i\}$ |
| **Stage 2: Fine-tune at 256K** | 用搜出的 $\{\lambda_i\}$ 短期微调（≈ 400 步） |
| **Stage 3: Re-search at 2M + short-context rescue** | 进一步搜到 2M；并维护两套缩放因子，short context 用 $\{\lambda_i^\text{short}\}$（接近 1），long context 用 $\{\lambda_i^\text{long}\}$ |

### 7.3 搜索空间

每个维度 $i$ 的 $\lambda_i \in [1, s_\text{max}]$（$s_\text{max} = L_\text{new}/L_\text{train}$），新频率 $\theta'_i = \theta_i / \lambda_i$。

搜索目标：

$$\min_{\{\lambda_i\}} \mathrm{PPL}\!\left(M; \theta'_i = \theta_i / \lambda_i\right) \quad \text{on a long-context validation set}$$

演化算法（CMA-ES 或类似）维护 population，迭代选优。论文实验上演化几百代即可收敛。

### 7.4 与 YaRN 对比

| 方法 | 缩放粒度 | Fine-tune 需求 | 最大上下文 |
| --- | --- | --- | --- |
| PI | 全维度同 $1/s$ | 是 (≥ 1000 步) | 32K |
| NTK-aware | 渐变 (单参 $\alpha$) | 否 (zero-shot) | 16K |
| YaRN | 三段 ramp (固定 $\alpha, \beta$) | 是 (≈ 400 步) | 128K |
| LongRoPE | **每维独立** | 是 (≈ 400 步) | **2M** |

> 💡 **Short-context rescue 的意义** — 直接套用 long-context 缩放会让模型在短上下文（如 1K-4K，覆盖大多数实际用例）上变差。LongRoPE 在推理时根据当前 batch 的实际长度切换缩放表，是工业级长上下文模型的常见技巧（DeepSeek-V2 / Qwen2.5 / Llama-3.1 也有类似 dual-table 设计）。

## §8 ABF 与 NoPE — 两种"非主流"扩展

### 8.1 ABF — Adjusted Base Frequency (Xiong et al., "Effective Long-Context Scaling of Foundation Models", arXiv:2309.16039, NAACL 2024, Meta)

最朴素的"换底"——直接把 RoPE 底从 10000 改大（如 500000）。等价于全维度同步 NTK-aware 缩放，但不用考虑 ramp。

- **优点**：最简单，1 行配置改动。
- **缺点**：$\theta_0 = (b')^0 = 1$ 同样不变（与 NTK-aware 一致，最高频被精确保留），但 ABF 的 $b'$ 是凭经验拍出来的（如 $10^6$），**没有按训练长度比例校准**——不像 NTK-aware 那样让最低频精确压到 $1/s$。结果：低频维度的相位压缩力度全凭手感。
- **用在哪**：CodeLlama 用 $b = 10^6$ 扩到 16K；Llama-3 / Llama-3.1 沿用大 base 加上更精细的 RoPE scaling。

### 8.2 NoPE — No Position Encoding

Kazemnejad et al. 2023 ("The Impact of Positional Encoding on Length Generalization in Transformers")：**decoder-only 模型不加位置编码，仅靠 causal mask 也能学到位置信息**。

观察：causal attention 的 mask 已经破坏了交换对称性（位置 $i$ 看不到位置 $j > i$ 的 token），这本身就编码了**先后顺序**。在小模型 / 短上下文上 NoPE 甚至外推性更好。

> ⚠️ **NoPE 的局限** — 仅适用于 decoder-only + causal mask。Encoder-only (BERT-like) 没有 causal mask，去掉位置编码后会退化为 bag-of-words。大模型上 NoPE 也未广泛验证。**记住它是个有意思的研究结论**，不是工业默认选择。

## §9 MLA — Multi-Head Latent Attention（DeepSeek-V2 May 2024）

### 9.1 动机

GQA 把 KV cache 从 $2 N_h d_h$ 压到 $2 G d_h$（per-token, per-layer），但 $G$ 至少要 4-8 才能保住质量。**能不能压得更狠？** MLA 把 KV cache 压到一个低秩 latent，理论上可以做到 $d_c \ll N_h d_h$ 而几乎不掉点。

### 9.2 朴素 low-rank K/V — 第一步推导

定义压缩矩阵 $W_\text{DKV} \in \mathbb{R}^{d_c \times d_\text{model}}$，把每个 token 的 hidden state $\mathbf{h}_t \in \mathbb{R}^{d_\text{model}}$ 投影到一个 KV latent：

$$\boxed{\;\mathbf{c}_t^{KV} = W_\text{DKV}\, \mathbf{h}_t \in \mathbb{R}^{d_c}\;}$$

每 head 的 K, V 通过**上投影**矩阵从这个 latent 恢复：

$$\mathbf{k}_t^{(h)} = W_\text{UK}^{(h)}\, \mathbf{c}_t^{KV}, \qquad \mathbf{v}_t^{(h)} = W_\text{UV}^{(h)}\, \mathbf{c}_t^{KV}$$

其中 $W_\text{UK}^{(h)}, W_\text{UV}^{(h)} \in \mathbb{R}^{d_h \times d_c}$。

**关键：cache 只存 $\mathbf{c}_t^{KV}$**（$d_c$ 维，外加一小段解耦-RoPE 分量 $d_r$，见 §9.6），不存 $\mathbf{k}, \mathbf{v}$ 本身。Per-token-per-layer cache 从 $2 N_h d_h$ 降到 $d_c+d_r$。DeepSeek-V2 选 $d_c = 4 d_h$（vs $2 N_h d_h$ for $N_h = 128$），**完整 cache 压缩约 57×**（精确算账见 §9.6 / 表 §12.3；只数裸 latent $d_c$ 则约 64×）。

### 9.3 Absorbing trick — 避免显式上投影

朴素做法：每次 attention 都从 $\mathbf{c}_t^{KV}$ 算出 $\mathbf{k}_t^{(h)} = W_\text{UK}^{(h)} \mathbf{c}_t^{KV}$，再计算 $\mathbf{q}_t^{(h)\top} \mathbf{k}_t^{(h)}$。这等价于：

$$\mathbf{q}_t^{(h)\top} \mathbf{k}_t^{(h)} = \mathbf{q}_t^{(h)\top} (W_\text{UK}^{(h)} \mathbf{c}_t^{KV}) = (W_\text{UK}^{(h)\top} \mathbf{q}_t^{(h)})^\top \mathbf{c}_t^{KV}$$

设 $\tilde{\mathbf{q}}_t^{(h)} := W_\text{UK}^{(h)\top} \mathbf{q}_t^{(h)}$，则 attention 分数变成 $\tilde{\mathbf{q}}_t^{(h)\top} \mathbf{c}_s^{KV}$——**直接和 latent cache 做内积**，不需要再算 $\mathbf{k}_s^{(h)}$。同理 $W_\text{UV}^{(h)}$ 可以被吸收进 output projection $W_O$ 的左乘。这就是**MLA 的 absorbing trick**：训练时显式分两步，推理时把上投影矩阵 absorb 进 query/output 投影，**实际 cache 读出 → matmul 一次完成**。

### 9.4 为什么 RoPE 必须解耦（最关键的 L3 题）

**问题**：把 RoPE 加进来会怎样？传统 RoPE 直接乘到 $\mathbf{q}, \mathbf{k}$ 上：

$$\mathbf{q}_t^{(h)} \leftarrow R_t\, \mathbf{q}_t^{(h)}, \qquad \mathbf{k}_t^{(h)} \leftarrow R_t\, \mathbf{k}_t^{(h)} = R_t\, W_\text{UK}^{(h)}\, \mathbf{c}_t^{KV}$$

但 $R_t$ 是**位置依赖**的——对 cache 中不同 token $t$ 是不同的旋转矩阵。如果还想用 absorbing trick，把 $R_t W_\text{UK}^{(h)}$ 吸进 query 侧，会变成

$$\mathbf{q}_t^{(h)\top} \mathbf{k}_s^{(h)} = \mathbf{q}_t^{(h)\top} (R_s\, W_\text{UK}^{(h)}\, \mathbf{c}_s^{KV})$$

这里 $R_s$ 对每个 cache 位置 $s$ 都不同——**没法 absorb 一个固定矩阵进 query 投影**。换句话说：

$$(W_\text{UK}^{(h)\top} R_s^\top)\, \mathbf{q}_t^{(h)} \quad \text{中 } R_s \text{ 随 } s \text{ 变}$$

如果非要保留 RoPE 同时做 absorbing，等价于 **per-position 的 query 投影**，破坏了 absorbing trick 的全部 cache 友好性，cache 必须重新存 RoPE 后的 K（回到 $N_h d_h$ 大小）。

### 9.5 MLA 的解耦方案 — 共享 RoPE key + 非 RoPE 主体

DeepSeek-V2 的解法：**把 K 拆成两段**：

1. **Non-RoPE 主体**：从 latent 上投影得到，大小 $d_h$，参与 absorbing。
2. **RoPE 部分**：单独一份 key，大小 $d_h^R$（一般 64），**所有 head 共享**这一份，独立施加 RoPE，不参与 absorbing。

形式化（DeepSeek-V2 论文 Eq. 5-11）：

$$
\begin{aligned}
\mathbf{c}_t^{KV} &= W_\text{DKV}\, \mathbf{h}_t \in \mathbb{R}^{d_c} \\
\mathbf{k}_t^{C,(h)} &= W_\text{UK}^{(h)}\, \mathbf{c}_t^{KV} \in \mathbb{R}^{d_h} \quad\text{(non-RoPE 主体, per head)}\\
\mathbf{k}_t^{R} &= \mathrm{RoPE}(W_\text{KR}\, \mathbf{h}_t) \in \mathbb{R}^{d_h^R} \quad\text{(共享 RoPE key, all heads share)}\\
\mathbf{k}_t^{(h)} &= [\mathbf{k}_t^{C,(h)}\; ;\; \mathbf{k}_t^{R}] \in \mathbb{R}^{d_h + d_h^R}
\end{aligned}
$$

Query 端也类似分成两半：

$$
\begin{aligned}
\mathbf{c}_t^{Q} &= W_\text{DQ}\, \mathbf{h}_t \in \mathbb{R}^{d_c'} \\
\mathbf{q}_t^{C,(h)} &= W_\text{UQ}^{(h)}\, \mathbf{c}_t^{Q} \in \mathbb{R}^{d_h} \quad\text{(非 RoPE, 与 } \mathbf{k}_t^{C,(h)} \text{ 配对)}\\
\mathbf{q}_t^{R,(h)} &= \mathrm{RoPE}(W_\text{QR}^{(h)}\, \mathbf{c}_t^{Q}) \in \mathbb{R}^{d_h^R} \quad\text{(per head 的 RoPE query, 与 } \mathbf{k}_t^{R} \text{ 配对)}\\
\mathbf{q}_t^{(h)} &= [\mathbf{q}_t^{C,(h)}\; ;\; \mathbf{q}_t^{R,(h)}]
\end{aligned}
$$

Attention 分数（同一 head）：

$$\mathbf{q}_t^{(h)\top} \mathbf{k}_s^{(h)} = \underbrace{\mathbf{q}_t^{C,(h)\top}\, \mathbf{k}_s^{C,(h)}}_{\text{absorb 进 q 投影}} + \underbrace{\mathbf{q}_t^{R,(h)\top}\, \mathbf{k}_s^{R}}_{\text{RoPE 部分, 直接算}}$$

第一项中 $\mathbf{k}_s^{C,(h)} = W_\text{UK}^{(h)} \mathbf{c}_s^{KV}$，按 §9.3 absorbing trick 把 $W_\text{UK}^{(h)}$ 吸进 query 侧。第二项 RoPE key 所有 head 共享，cache 只存一份 $\mathbf{k}_s^R$。

### 9.6 MLA KV cache 总大小

Per token per layer：

$$\boxed{\;\text{MLA cache} = \underbrace{d_c}_{\mathbf{c}^{KV}} + \underbrace{d_h^R}_{\mathbf{k}^R} \quad \text{vs} \quad \text{MHA cache} = 2 N_h d_h\;}$$

DeepSeek-V2 数值（$N_h = 128, d_h = 128, d_c = 512, d_h^R = 64$）：

- MHA: $2 \cdot 128 \cdot 128 = 32{,}768$ 元素 / token / layer
- MLA: $512 + 64 = 576$ 元素 / token / layer
- 压缩比 **57×**（vs MHA），同时 KV cache 总比 GQA-8 还小约 4×。

### 9.7 MLA 简化代码

```python
import torch
import torch.nn as nn
# 复用 §2.4 的 apply_rope 函数 (略).

class MultiHeadLatentAttention(nn.Module):
    """
    简化 MLA: 训练版 (推理可加 absorbing trick).
    Per token per layer cache: c_kv [d_c] + k_R [d_h_R]
    """
    def __init__(self, d_model: int, n_heads: int,
                 d_c: int = 512, d_h: int = 128, d_h_R: int = 64,
                 d_c_q: int = 1536):
        super().__init__()
        self.n_heads, self.d_h, self.d_h_R = n_heads, d_h, d_h_R

        # Down-projection 到 latent
        self.W_DKV = nn.Linear(d_model, d_c,  bias=False)
        self.W_DQ  = nn.Linear(d_model, d_c_q, bias=False)

        # Up-projection (non-RoPE 主体)
        self.W_UK = nn.Linear(d_c,   n_heads * d_h,   bias=False)
        self.W_UV = nn.Linear(d_c,   n_heads * d_h,   bias=False)
        self.W_UQ = nn.Linear(d_c_q, n_heads * d_h,   bias=False)

        # RoPE 解耦部分
        self.W_KR = nn.Linear(d_model, d_h_R,            bias=False)  # 共享 across heads
        self.W_QR = nn.Linear(d_c_q,   n_heads * d_h_R,  bias=False)  # per head

        self.W_O  = nn.Linear(n_heads * d_h, d_model, bias=False)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor = None):
        # h: [B, L, d_model]; cos/sin for RoPE: [L, d_h_R/2]
        B, L, _ = h.shape
        H, dH, dHR = self.n_heads, self.d_h, self.d_h_R

        # ----- KV path -----
        c_kv = self.W_DKV(h)                                              # [B, L, d_c]
        k_C  = self.W_UK(c_kv).view(B, L, H, dH).transpose(1, 2)          # [B, H, L, d_h]
        v    = self.W_UV(c_kv).view(B, L, H, dH).transpose(1, 2)          # [B, H, L, d_h]

        k_R  = self.W_KR(h)                                                # [B, L, d_h_R]  (shared)
        k_R  = apply_rope(k_R, cos, sin)                                   # 共享 RoPE
        # broadcast 到 H 个 head 上做 concat
        k_R_per_head = k_R.unsqueeze(1).expand(B, H, L, dHR)                # [B, H, L, d_h_R]
        k = torch.cat([k_C, k_R_per_head], dim=-1)                          # [B, H, L, d_h+d_h_R]

        # ----- Q path -----
        c_q  = self.W_DQ(h)                                                 # [B, L, d_c_q]
        q_C  = self.W_UQ(c_q).view(B, L, H, dH).transpose(1, 2)             # [B, H, L, d_h]
        q_R  = self.W_QR(c_q).view(B, L, H, dHR).transpose(1, 2)            # [B, H, L, d_h_R]
        q_R  = apply_rope(q_R, cos, sin)                                    # per-head RoPE
        q = torch.cat([q_C, q_R], dim=-1)                                   # [B, H, L, d_h+d_h_R]

        # ----- Attention -----
        scores = (q @ k.transpose(-2, -1)) / ((dH + dHR) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, L, H * dH)   # [B, L, H*d_h]
        return self.W_O(out)                                                # [B, L, d_model]
```

> ⚠️ **常见误解** — "MLA 只是 GQA 的进一步压缩" — 不准确。GQA 是 head 维压缩（多 Q head 共享 K/V head），MLA 是 hidden 维低秩压缩（$N_h d_h \to d_c$）+ 共享 RoPE。GQA 仍然每 KV head 独立做 RoPE；MLA 必须解耦 RoPE 才能保住 absorbing trick。

### 9.8 训练成本

MLA 引入额外的下投影 / 上投影，**训练 FLOPs 略增**（DeepSeek-V2 报告 ≈ 2% 增加），换来推理时 KV cache 砍数十倍——属于"训练贵一点，推理便宜很多"的取舍。

## §10 滑动窗口与流式注意力

### 10.1 Sliding Window Attention (Mistral 2023)

每个 query 只看前 $W$ 个 key（$W$ = window size，Mistral-7B 用 $W = 4096$）。

- **复杂度**：从 $O(L^2 d)$ 降到 $O(L W d)$，长序列线性。
- **多层叠加感受野扩展**：第 1 层看 $W$，第 2 层每个位置看前 $W$（其中每个 token 又看了它的前 $W$），有效感受野 $2W$；$\ell$ 层后感受野 $\ell W$。**所以 32 层 × 4096 window ≈ 131K 有效感受野**。

```python
def sliding_window_mask(L: int, W: int, device=None) -> torch.Tensor:
    """
    L: sequence length; W: window size (含 self)
    返回 [L, L] bool mask, True=可见.
    位置 i 可看 j ∈ [max(0, i-W+1), i] (causal + sliding window)
    """
    idx_q = torch.arange(L, device=device).unsqueeze(1)   # [L, 1]
    idx_k = torch.arange(L, device=device).unsqueeze(0)   # [1, L]
    causal     = idx_k <= idx_q
    in_window  = idx_k > (idx_q - W)
    return causal & in_window

# 示例 L=8, W=4:
# row 0: [T F F F F F F F]
# row 1: [T T F F F F F F]
# row 2: [T T T F F F F F]
# row 3: [T T T T F F F F]
# row 4: [F T T T T F F F]    ← 0 号被推出 window
# row 5: [F F T T T T F F]
# row 6: [F F F T T T T F]
# row 7: [F F F F T T T T]
```

> 💡 **SWA 的实战意义** — Mistral-7B 训练长度 8K，推理时配合 SWA 可处理 32K+ 上下文（每层只看本地 4K，多层叠出全局），同时显存/计算线性。但纯 SWA 对**远距精确检索**（如 needle-in-haystack 远端针）能力较弱——这正是 StreamingLLM 加 attention sink 的动机。

### 10.2 StreamingLLM — Attention Sink + Sliding Window

Xiao et al. ICLR 2024 ("Efficient Streaming Language Models with Attention Sinks") 提出推理时一个关键发现：

**LLM decode 时，softmax 强制 attention 权重和为 1，但 query 实际可能"什么也不想 attend"。模型于是把权重大量倒给前 1-4 个 token（特别是 `<bos>`），形成 attention sink。** 这些 token 内容上没什么信息，但它们的 KV cache **不能丢弃**——丢了之后 softmax 没有"垃圾桶"，剩下 token 的注意力分布被强行重整，PPL 爆炸。

StreamingLLM 推理策略：

1. **永远保留** 前 $S$ 个 token 的 KV cache（$S = 4$ 经验值）作为 sink。
2. **滑动窗口** 保留最近 $W$ 个 token 的 KV cache。
3. window 之外、sink 之外的 token，**直接丢弃 KV**。

总 KV cache 大小为 $S + W$，与序列长度 $L$ 解耦，达到**真正的流式**生成。

### 10.3 StreamingLLM 推理循环代码

下面是 **教学示意版**，重点展示控制流。生产实现（HuggingFace `streaming_llm` / 原作者 `streaming-llm` repo）有两个关键细节，下面注释里说明。

```python
@torch.no_grad()
def streaming_decode(model, input_ids, max_new_tokens,
                     sink_size=4, window_size=2044):
    """
    教学版 streaming inference: sink 区 + sliding window.
    总 cache = sink_size + window_size, 与生成长度无关.

    关键细节 (生产代码必须做):
    (a) Cache 存的是 *RoPE 之前* 的 K (即 W_K @ h, 未旋转), 同时记录该 token 的"逻辑位置".
        每次 forward 时, 根据当前 cache 中各 token 的*新* 逻辑位置, 对 sink / recent K 重新施加 RoPE.
        否则裁剪 + 位置漂移会让 cache 中的旋转角对不上新的逻辑位置.
    (b) Sink 区位置永远固定在 [0, S), recent window 位置永远固定在 [S, S+W),
        新 token 用 S+W (即 cache 容量上限) 作为它的逻辑位置.
        这样模型见到的"最大相对位置"始终 ≤ S+W, 永远不会触碰 RoPE 训练上限.
    """
    device = input_ids.device
    B = input_ids.size(0)
    total = sink_size + window_size

    # ----- 1) Prefill -----
    # past_kv_pre[i] = (k_pre, v) 其中 k_pre = W_K @ h, 未做 RoPE
    past_kv_pre = _prefill_unrotated(model, input_ids)            # 实现细节略

    # 若 prompt 已超 sink+window, 裁剪 (sink 段 + 最近 window 段)
    def trim_unrotated(past_kv_pre):
        new_past = []
        for (k_pre, v) in past_kv_pre:
            if k_pre.size(-2) <= total:
                new_past.append((k_pre, v));  continue
            sink = (k_pre[..., :sink_size, :], v[..., :sink_size, :])
            recent = (k_pre[..., -window_size:, :], v[..., -window_size:, :])
            new_past.append(
                (torch.cat([sink[0], recent[0]], dim=-2),
                 torch.cat([sink[1], recent[1]], dim=-2))
            )
        return new_past

    past_kv_pre = trim_unrotated(past_kv_pre)
    # logits 来自 prefill 最后一步
    next_token = _last_logits(model, past_kv_pre).argmax(-1, keepdim=True)
    generated = [next_token]

    # ----- 2) Autoregressive decode -----
    for step in range(max_new_tokens - 1):
        cur_len = past_kv_pre[0][0].size(-2)                       # 当前 cache 中 token 数
        # 给 cache 中每个 token 分配"逻辑位置"; 注意 prompt 极短时 cur_len < sink_size,
        # 此时所有 token 都视为 sink (没有 recent window).
        if cur_len <= sink_size:
            cache_pos = torch.arange(cur_len, device=device)        # [cur_len]
        else:
            cache_pos = torch.cat([
                torch.arange(sink_size, device=device),             # sink 段: [0..S)
                torch.arange(sink_size, cur_len, device=device),    # window 段: [S..cur_len)
            ])                                                       # 长度 = cur_len
        new_pos = torch.tensor([cur_len], device=device)             # 新 token 逻辑位置

        # 对 cache 中的 K_pre 重新施加 RoPE (按 cache_pos), 对新 token 按 new_pos.
        out = model(next_token, past_kv_pre=past_kv_pre,
                    cache_pos=cache_pos, new_pos=new_pos, use_cache=True)
        past_kv_pre = trim_unrotated(out.past_kv_pre)
        next_token = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_token)

    return torch.cat(generated, dim=-1)
```

> ⚠️ **直接裁剪 RoPE 之后的 K cache 是错的** — 一个常见 bug：把 HF 默认的 K cache（已 RoPE）直接按上面的方式裁剪 + 用逻辑位置 id 喂新 token，会得到自相矛盾的相对位置（cache 中的 K 用原始绝对位置旋转过，但新 query 用逻辑位置旋转）。**正确做法**：保留未旋转的 K（`W_K @ h`，未乘 cos/sin），每步根据当前逻辑位置重新做 RoPE；或者用作者 repo 提供的 `enable_streaming_llm()` patch，它修改了 attention layer 以接受 "position-shift" 形式的旋转。

> ⚠️ **StreamingLLM 不增加模型有效上下文** — 它让模型可以**永久流式生成**而不爆显存，但实际能看到的还是 sink + window 范围内的 token。中间被丢弃的内容**真的看不到了**。要长上下文检索能力还是得依赖 YaRN / LongRoPE / SSM 等真正的上下文扩展。

### 10.4 Lost-in-the-Middle (Liu 2023)

Liu et al. 2023 ("Lost in the Middle: How Language Models Use Long Contexts") 经验观察：**长上下文模型对 prompt 头部和尾部的关注度远高于中间**，造成"中间内容更难被检索到"。

- **U-shaped curve**：把 key info 放在 prompt 不同位置，检索准确率随位置呈 U 形（首尾高，中间低）。
- **原因**：causal LM 训练分布中，第一个 token 影响最广（attention sink 同源问题）；最后一个 token 是 next-token 预测的直接前驱。中间内容被两端"挤压"。
- **缓解**：(a) 把重要信息放在 prompt 开头或结尾；(b) Recurrent retrieval (chain prompts)；(c) 训练时增加中段权重 (位置感知 loss weighting)。

> 💡 **面试要点** — 这不是"位置编码外推失败"——是模型**有效**学到了长上下文，但 attention 分布存在偏好。和 RoPE/YaRN 解决的问题不同。

## §11 System 级长上下文 — Ring / CP / FlashAttn

### 11.1 Ring Attention (Liu et al. 2023)

把序列在 $P$ 张 GPU 上切成 $P$ 段，每段独立持有自己的 Q/K/V chunk。Attention 通过 **K/V chunk 在 GPU 间环形传递** 实现：

```
GPU 0: 持 Q0, K0, V0  ←→  GPU 1: 持 Q1, K1, V1  ←→  ...  ←→  GPU P-1
            │                       │
            └─ pass K1, V1 to GPU 0, 同时 GPU 0 把 K0, V0 传给 GPU P-1
            (环形 P-1 次后, 每个 GPU 都看过所有 K, V)
```

- **每个 GPU 在每一轮**用自己当前持有的 K/V 段，对本地 Q 段做局部 attention，累加 partial output。
- **通信与计算 overlap**（下一段 K/V 在传时，本段 attention 在算）。
- 总通信量 $O(L \cdot d)$ per GPU（每个 GPU 收发 $P-1$ 次 chunk）；总计算 $O(L^2 d / P)$ per GPU。

**关键效果**：单 GPU 显存只需放 $L/P$ 的 K/V，**有效上下文随 GPU 数线性扩展**——理论上 8 GPU × 128K 单卡 = 1M 上下文。

### 11.2 Context Parallelism (Megatron 2024)

Megatron-Core 的 Context Parallel (CP) 是 Ring Attention 的工程化版本，集成进现有 tensor/pipeline parallelism。主要工程点：

- 配合 FlashAttention 块化做 fused all-to-all 通信
- 解决 causal mask 下 chunk 间负载不均（前半 chunk attention 计算少、后半多，需要负载均衡）
- 与 ZeRO-3 兼容

### 11.3 FlashAttention 2/3 与长上下文

FlashAttention v1 (Dao 2022) 的核心是 IO-aware exact attention，但 v1 的循环结构对长序列负载不均。

- **v2 (Dao 2023)**：换内外循环 (Q-outer, KV-inner)，更好 warp-level parallelism，长序列吞吐提升 2×。
- **v3 (Dao 2024)**：针对 H100，使用 WGMMA / TMA / FP8 asynchronous pipeline。

长上下文场景下，FlashAttention 是**几乎所有训练 / 推理 stack 的默认**（避免物化 $L \times L$ 分数矩阵）。

### 11.4 Differential Attention（可选，Microsoft 2024）

Ye et al. 2024 ("Differential Transformer") 提出每个 attention head 用**两个独立 Q/K 投影做差**：

$$\mathrm{Diff} = \mathrm{softmax}(Q_1 K_1^\top / \sqrt{d}) - \lambda \cdot \mathrm{softmax}(Q_2 K_2^\top / \sqrt{d})$$

- **直觉**：第一项学"信号"，第二项学"噪声"，差值更尖锐。
- **效果**：长上下文 needle-in-haystack 任务上比 vanilla attention 提升明显。
- **代价**：实现上每 head 多算一组 Q/K 投影，但论文通过**减半 head 数**使总参数 / FLOPs 与标准 attention **基本持平**（≈0 额外开销，不是 +50%）。

> 💡 **是否选用** — Differential Attention 是 2024 末的新方向，业界采用率还不高（DeepSeek-V3 没用，Llama-3 也没用），但研究上有意思。面试问"长上下文新方向"可以提一句。

## §12 复杂度与显存总表

### 12.1 KV cache per-token-per-layer 大小（attention 变体）

| 方法 | KV cache 大小（元素） | 与 MHA 对比（$N_h=128, d_h=128, G=8, d_c=512, d_h^R=64$） |
| --- | --- | --- |
| **MHA** | $2 N_h d_h$ | 32,768（基准 1×） |
| **MQA** | $2 d_h$ | 256（128×） |
| **GQA-8** | $2 G d_h$ | 2,048（16×） |
| **MLA** | $d_c + d_h^R$ | 576（57×） |

### 12.2 KV cache 总占用（per-sample-per-layer，受到 SWA / Streaming 等"窗口"机制影响）

| 方法 | 总 cache 大小（元素） | 与 vanilla cache 对比（同 attention 变体下） |
| --- | --- | --- |
| **Vanilla (整段序列)** | $L \cdot 2 N_h d_h$ | 基准 1× |
| **SWA (window=W)** | $W \cdot 2 N_h d_h$（每层只看 W 个最近 token） | $W/L \times$ |
| **Streaming (sink+win)** | $(S + W) \cdot 2 N_h d_h$（常数, 与 L 解耦） | $(S{+}W)/L \times$ |

注：SWA / Streaming 与 GQA / MLA 是**正交**的——两者乘起来即得到工业级 stack 的实际 cache 大小。

### 12.3 Attention 时间与显存

| 方法 | Time per token (decode) | Memory peak (prefill) |
| --- | --- | --- |
| Vanilla MHA | $O(L \cdot N_h d_h)$ | $O(L^2)$ scores |
| FlashAttention | $O(L \cdot N_h d_h)$ | $O(L)$（无中间 scores） |
| Sliding Window | $O(W \cdot N_h d_h)$ | $O(L \cdot W)$ |
| Streaming (S+W) | $O((S+W) \cdot N_h d_h)$ | $O((S+W)^2)$ |
| Ring (P GPU) | $O(L \cdot N_h d_h / P)$ per GPU | $O(L^2 / P)$ per GPU |
| MLA | $O(L \cdot N_h (d_c + d_h^R))$ | + 投影开销 |

## §13 综合对比与选型决策树

```
Q: 我要把上下文从 4K 推到 N tokens, N=?
│
├── N ≤ 16K, zero-shot, 不能 fine-tune
│    └── NTK-aware (1 行配置, base 改大)
│
├── N ≤ 32K, 可少量 fine-tune (~1000 步)
│    └── PI (简单稳定) or YaRN (更好)
│
├── 32K < N ≤ 128K, fine-tune 预算 < 500 步
│    └── YaRN (NTK-by-parts + temperature)
│
├── N > 128K (256K-2M)
│    └── LongRoPE (每维独立搜索 + short-context rescue)
│
└── 流式生成 (无限长度, 不需要远端检索)
     └── StreamingLLM (sink + sliding window)

Q: KV cache 显存压不住?
│
├── 想保住质量, 适度压
│    └── GQA (LLaMA-2/3, Mistral)
│
├── 想极致压, 接受重训练
│    └── MLA (DeepSeek-V2/V3): cache 砍 50×, RoPE 必须解耦
│
└── 推理 server 端
     └── 配合 PagedAttention (vLLM) 做 cache 分页管理

Q: Attention 算不动 (L^2 太大)?
│
├── 单卡推理
│    └── FlashAttention 2/3 (exact, 必装)
│
├── 多卡训练 / 推理
│    └── Ring Attention / Context Parallelism (chunk K/V 环传)
│
└── 不要远距精确检索, 只要本地依赖
     └── Sliding Window Attention (Mistral 风格)
```

## §14 25 高频面试题

按难度分 L1（必会）/ L2（进阶）/ L3（顶级 lab）三档。每题展开看答案要点和易踩坑。

### L1 必会题（任何长上下文相关岗位都问）

<details>

<summary>Q1. RoPE 的核心公式是什么？为什么给出"相对位置"？</summary>

- 对每对相邻维度做 2D 旋转：$f(\mathbf{x}, m) = \mathbf{x} \odot e^{im\boldsymbol\theta}$（复数视角），$\theta_i = 10000^{-2i/d}$
- $\langle f(\mathbf{q}, m), f(\mathbf{k}, n)\rangle = \mathrm{Re}\!\sum_i \bar{q}_i k_i\, e^{i(n-m)\theta_i}$，**只依赖 $n-m$**
- 关键：旋转矩阵的可加性 $R_m^\top R_n = R_{n-m}$

只说"RoPE 编码相对位置"而不会推导。

</details>

<details>

<summary>Q2. RoPE 频率为什么是 $10000^{-2i/d}$？</summary>

- 沿用 Vaswani 2017 sinusoidal 的几何级数频率分布
- 高频维度（小 $i$）周期短、编码精细局部位置；低频维度（大 $i$）周期长、编码粗远程位置
- 多时间尺度同时分辨位置

只回答"为了让不同维度看不同位置"，不指出几何级数与高/低频意义。

</details>

<details>

<summary>Q3. 朴素 RoPE 为什么不能直接外推？</summary>

- 训练时 $m \in [0, L_\text{train})$，低频维度 $m\theta_i$ 还远小于 $2\pi$
- 推理给 $m > L_\text{train}$，低频维度进入未见相位区
- 模型对这些区域的 attention 行为没学过 → PPL 爆炸 / 上下文崩溃

说"RoPE 周期外推自然 OK" — 错。周期性只在维度内成立，跨上下文长度外推的是"位置 → 相位"映射，模型从未见过 $m\theta_i$ 超出训练范围的相位组合。

</details>

<details>

<summary>Q4. PI (Position Interpolation) 怎么做？有什么副作用？</summary>

- 把 $\theta_i$ 同除以 $s = L_\text{new}/L_\text{train}$（或等价地把 $m$ 缩到 $m/s$）
- 副作用：**高频维度被破坏**——高频本来已经在训练中分辨细粒度位置，现在分辨率被压 $s$ 倍
- 必须 fine-tune（≥ 1000 步）才能恢复

以为"插值就一定无损"。

</details>

<details>

<summary>Q5. NTK-aware 与 PI 的核心区别是什么？</summary>

- **PI**：所有维度同除 $s$（高频破坏）
- **NTK-aware**：换底 $b' = b \cdot s^{d/(d-2)}$，让最高频几乎不变、最低频被压到 $1/s$
- **NTK-aware 零样本可用**（不需 fine-tune），PI 必须 fine-tune

说"NTK 和 PI 没区别"。

</details>

<details>

<summary>Q6. ALiBi 与 RoPE 的区别？哪种外推更好？</summary>

- ALiBi：在 logit 上加 $-m_h |i-j|$ 距离 bias，head 相关斜率，**无 Q/K 旋转**
- RoPE：通过 Q/K 旋转编码位置，**没有显式 bias**
- 外推：ALiBi 更好（线性 bias 自然外推），但表达力弱（只有单调距离衰减）
- 工业选择：RoPE 配合 YaRN/LongRoPE 用得多（表达力 + 可扩展）

把 RoPE 和 ALiBi 当作同类方法（一个是 score-shift，一个是 Q/K 变换）。

</details>

<details>

<summary>Q7. KV cache 显存怎么算？</summary>

- 公式：$L_\text{ctx} \cdot n_\text{layers} \cdot 2 \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$
- $2$ 是因为存 K + V；MQA 时 $H_\text{kv} = 1$；GQA 时 $H_\text{kv} = G$；MLA 时换成 $d_c + d_h^R$（不再有 2× 分别）
- 对 LLaMA-2-7B (32 层, $N_h=32, d_h=128$, fp16, MHA), 4K 上下文 $\approx 2.1$ GB / sample；100K $\approx 52$ GB / sample
- LLaMA-2-70B 用 GQA-8 ($H_\text{kv}=8$, 80 层, $d_h=128$), 4K $\approx 1.25$ GB / sample——GQA 大幅压缩

漏 $n_\text{layers}$；或把 $2$ 当作 head 因子。

</details>

<details>

<summary>Q8. MQA / GQA 减的是什么？</summary>

- **KV cache 显存** + **显存带宽**（decode 阶段每步要从 HBM 读 K/V cache）
- 也减少 K/V projection 的参数和计算
- **不减少 Q projection**，Q head 数不变

误说"GQA 减少 Q head" — 错。

</details>

<details>

<summary>Q9. Sliding Window Attention 怎么让模型看远？</summary>

- 每层只看 $W$ 个，但多层叠加：第 $\ell$ 层每个位置的感受野是 $\ell \cdot W$
- Mistral-7B: 32 层 × 4K window ≈ 131K 有效感受野
- 但**远距精确检索**能力弱（信息要经过多层 "tunnel" 传递）

以为"window 内才能看，所以只能看 W 个 token" — 错，这是只对一层而言。

</details>

<details>

<summary>Q10. 什么是 Attention Sink？</summary>

- LLM decode 时，前 1-4 个 token (特别是 `<bos>`) 获得异常高的 attention，即使内容无关
- 直觉：softmax 强制权重和为 1，模型需要"垃圾桶"吸收概率
- 工程利用：StreamingLLM 永远保留前 $S$ 个 token 的 KV cache + 滑动窗口

以为 attention sink 是 BOS / CLS token 的"语义正常"的注意力 — 错，sink 通常出现在所有 query 上，**与内容无关**。

</details>

### L2 进阶题（research-oriented 岗位）

<details>

<summary>Q11. NTK-aware 中 $b' = b \cdot s^{d/(d-2)}$ 的推导？</summary>

- 设 $b' = b \cdot \alpha$
- 最高频 $\theta_0 = (b')^0 = 1$，不随 $\alpha$ 变 ✓
- 最低频 $\theta'_{d/2-1} = b^{-(d-2)/d} \cdot \alpha^{-(d-2)/d}$
- 要求 $\theta'_{d/2-1} = \theta_{d/2-1}/s$ → $\alpha^{-(d-2)/d} = 1/s$ → $\alpha = s^{d/(d-2)}$

不会推导只背公式。

</details>

<details>

<summary>Q12. YaRN 的三个组件各解决什么？</summary>

- **NTK-by-parts**：高/中/低频分段处理，比 NTK-aware 单参数 ramp 更精细
- **Temperature scaling**：上下文变长后 softmax 分布扁平化，加温度 $t < 1$ 让分布更尖锐
- **Attention scale (实现层面)**：把温度 $1/t$ 实现为 Q/K 范数缩放（等价于乘到 cos/sin cache），不动 attention kernel

只说"YaRN 是 NTK-aware 改进版"，不分解。

</details>

<details>

<summary>Q13. YaRN 温度公式 $\sqrt{1/t} \approx 0.1 \ln s + 1$ 怎么来？</summary>

- 这是**经验拟合公式**，不是闭式推导
- 基于不同扩展比 $s$ 下 attention 熵的实验测量
- 想法：扩展比越大，需要越低的温度（更尖锐的分布）来补偿稀释

把它当成"严格推导的最优温度" — 错，YaRN 论文明确是 empirical fit。

</details>

<details>

<summary>Q14. RoPE 实数实现有哪两种 pairing？</summary>

- **奇偶交错**：$(x_0, x_1), (x_2, x_3), \dots$（原始 RoFormer 论文）
- **前半 / 后半**：$(x_0, x_{d/2}), (x_1, x_{d/2+1}), \dots$（HuggingFace LLaMA 实现）
- 数学上仅是维度排列，对最终内积**等价**
- 但 RoPE cache 的预计算与 pairing 必须**一致**，混用会导致旋转作用在错维度

不知道 HF 和 Meta 官方实现有这种差异。

</details>

<details>

<summary>Q15. LongRoPE 与 YaRN 的核心区别？</summary>

- **YaRN**：基于波长的固定 ramp 函数，所有维度按统一规则分段
- **LongRoPE**：每个维度独立缩放因子 $\lambda_i$，**演化算法**搜索
- LongRoPE 还引入 short-context rescue（短上下文用另一套缩放表）
- 最大上下文：YaRN 128K vs LongRoPE 2M

说 LongRoPE "和 YaRN 没区别"。

</details>

<details>

<summary>Q16. Mistral-7B 用 SWA + 多层堆叠怎么算有效感受野？</summary>

- 单层感受野 $W = 4096$
- $\ell$ 层后理论感受野 $\ell W$；32 层 × 4096 = 131K
- 但实际"信息传递"是稀疏的——远距 token 必须经过多层传递，相当于一个 deep pipeline
- 实测 Mistral 在 32K 以内表现不错，更远开始衰减

以为 SWA 直接看 4K 就是硬上限。

</details>

<details>

<summary>Q17. Streaming LLM 中为什么 position id 要用"逻辑位置"而非绝对位置？</summary>

- 如果用绝对位置：cache 里 sink 是 [0,4)，最近 window 是 [L-W, L)，新 token 是 L
- 但 $L$ 可以无限增长，RoPE 在 $m > L_\text{train}$ 时未见过，PPL 爆炸
- **逻辑位置**：sink 用 [0, S)，window 内用 [S, S+W)，新 token 用 S+W
- 这样 RoPE 永远在训练见过的范围内 → 流式可无限生成

说"用绝对位置才正确" — 错，绝对位置会撞 RoPE 外推上限。

</details>

<details>

<summary>Q18. Ring Attention 的通信量与计算量？</summary>

- $P$ 卡, 每卡持序列长 $L/P$ 的 Q/K/V
- 环形传 K/V chunk，$P-1$ 轮后每卡看过所有 K/V
- 每卡通信量 $O(L \cdot d)$（收/发各一份 K/V）
- 每卡计算量 $O(L^2 d / P)$
- **通信与计算 overlap**：下一轮 K/V 在传时本轮 attention 在算

说"Ring Attention 只是 chunk attention" — 漏掉环形通信关键点。

</details>

<details>

<summary>Q19. Lost in the Middle 是什么现象？与位置编码外推是同一问题吗？</summary>

- 现象：长上下文中模型对**首尾 token 关注高、中段关注低**（U 形曲线）
- 原因：causal LM 训练分布偏好首尾（attention sink 同源 + next-token 直接前驱）
- **不是位置编码外推问题**——是 attention 分布偏好问题
- 即使位置编码完美外推，也存在此偏好

把它和 RoPE 外推混淆。

</details>

<details>

<summary>Q20. ABF 和 NTK-aware 的关系？</summary>

- ABF (Adjusted Base Frequency)：直接把 RoPE base 改大（如 10000 → 500000），全维度同步换底
- NTK-aware：换底 $b' = b \cdot s^{d/(d-2)}$，**和 ABF 形式上一样**（都是改大 base）
- 区别在**为什么这么改**：NTK-aware 有数学推导（保最高频不变 + 最低频压 $1/s$），ABF 是经验选择
- CodeLlama 用 ABF (base=$10^6$)；LLaMA-3 也大幅增大 base + 配合 RoPE scaling

说"ABF 和 NTK-aware 完全没关系" — 错，公式同形，只是动机不同。

</details>

### L3 顶级 lab 难题（DeepSeek / Anthropic / OpenAI / Google 系）

<details>

<summary>Q21. NTK-aware base scaling 为什么能精确保留高频？</summary>

- 高频对应 $i = 0$，$\theta_0 = b^{-0} = 1$，**与 $b$ 无关**
- 换底 $b \to b' = b \cdot \alpha$ 后，$\theta'_0 = (b')^0 = 1$，仍然 1
- 中间维度 $\theta'_i / \theta_i = \alpha^{-2i/d}$，从 1（$i=0$）指数过渡到 $1/s$（$i=d/2-1$）
- **几何含义**：换底是在对数频率空间做"剪切"（高频锚定不动，低频被压缩 $\log s$ 量）

只说"NTK 不改高频" — 不解释为什么换底自动有这效果。

</details>

<details>

<summary>Q22. MLA 中 RoPE 解耦后，绝对位置信息如何注入到 K/V 的 latent 上投影部分？</summary>

- **关键回答：不注入**。MLA 的 non-RoPE 主体 $\mathbf{k}_t^{C,(h)} = W_\text{UK}^{(h)} \mathbf{c}_t^{KV}$ 完全没有位置编码
- 位置信号**仅由共享 RoPE key** $\mathbf{k}_t^R = \mathrm{RoPE}(W_\text{KR} \mathbf{h}_t)$ 提供
- Attention 分数被加性分解：$\mathbf{q}_t^{C\top} \mathbf{k}_s^C$ (内容) + $\mathbf{q}_t^{R\top} \mathbf{k}_s^R$ (位置)
- 这就是"解耦"的含义：内容路径和位置路径**独立**，互不污染 absorbing trick

以为 MLA 把 RoPE 也吸进 latent 里 — 错。

</details>

<details>

<summary>Q23. 为什么 MLA 不能简单地"在上投影后再加 RoPE"？哪一步算不出来？</summary>

- 假设 cache 存 $\mathbf{c}_s^{KV}$，attention 时算 $\mathbf{k}_s^{(h)} = R_s\, W_\text{UK}^{(h)}\, \mathbf{c}_s^{KV}$
- 想 absorb：query 内积变 $\mathbf{q}_t^{(h)\top} (R_s W_\text{UK}^{(h)} \mathbf{c}_s^{KV}) = (W_\text{UK}^{(h)\top} R_s^\top \mathbf{q}_t^{(h)})^\top \mathbf{c}_s^{KV}$
- 这里 $R_s$ 是**位置 $s$ 依赖**的旋转——每个 cache 位置 $s$ 对应不同 $R_s$
- 不能 absorb 一个固定矩阵进 query 投影，**absorb 必须 per-position**
- 等价于每个 query 对每个 cache 位置算一次 $W_\text{UK}^{(h)\top} R_s^\top$ 矩阵——**O(L) 个 matmul**，比直接物化 K 还贵
- 所以"加完 RoPE 再吸"在算力上比不解耦还差，**absorbing trick 完全失效**

只说"RoPE 是位置相关的" — 不够，要说出**absorb 需要的常数性被破坏**这一关键点。

</details>

<details>

<summary>Q24. YaRN 的 attention scale 和直接改 softmax 温度有什么实现层面的区别？</summary>

- **直接改温度**：在 attention kernel 里把 logits 除以 $t$，需要修改 FlashAttention 等 fused kernel
- **Attention scale**：把 $\sqrt{1/t}$ 乘进 RoPE cos/sin cache，等价于 Q/K 范数**放大** $\sqrt{1/t}$ 倍（$t < 1$ 时 $\sqrt{1/t} > 1$），$QK^\top$ 自然放大 $1/t$ 倍
- 两者**数学等价**（前提：Q/K 范数主要来自 RoPE 后的部分）
- 工程优势：**完全不动 attention kernel**，只改 RoPE 预计算
- 这是 YaRN "infrastructure-friendly" 的一大卖点

说"两个就是同一件事" — 数学等价但工程意义不同。

</details>

<details>

<summary>Q25. 设计一个 1M context、可流式生成、单卡推理的方案。</summary>

参考 Qwen2.5-1M / DeepSeek-V3 思路：

- **位置编码**：YaRN / LongRoPE 把 RoPE 推到 1M（per-dim 缩放搜索）
- **KV cache 压缩**：MLA (cache 砍 50×) 让单卡能装下 1M cache 的"latent"
- **Attention 算法**：FlashAttention 3 + Ring Attention（如果跨多卡）或 Sliding Window 配合 sink（如果要流式）
- **Inference 优化**：vLLM PagedAttention 做 cache 分页；Speculative decoding 加速 decode；Chunked prefill 分批喂 prompt（避免一次性 OOM）
- **训练**：必须真的在长上下文数据上 fine-tune（≥ 1000 步），仅靠 zero-shot RoPE 改造不够

完整 stack：MLA + YaRN/LongRoPE + FlashAttn3 + (Ring/CP if 多卡) + StreamingLLM(if 流式) + vLLM 推理。

- 只说一种方法（如只说 YaRN）— 不够完整
- 没区分"扩上下文"和"压 cache"两个独立维度
- 漏掉"必须 fine-tune"

</details>

## §A 附录：实现要点 checklist

### A.1 RoPE 工程实现要点

- **pairing 一致性**：奇偶交错 vs 前半/后半，必须和 cache 预计算保持一致
- **half-dim 注意**：cos/sin cache shape 是 $[L, d/2]$，应用时要 broadcast 到 $[L, d]$ 或在两半上分别乘
- **dtype 处理**：cos/sin 用 fp32 计算后转 dtype，避免 fp16/bf16 下旋转角累积误差
- **YaRN 时**：cos/sin cache 已经乘了 $1/\sqrt{t}$，attention 内部不要再缩放
- **MLA 时**：query / key 的 RoPE 部分和 non-RoPE 部分要 concat（一般 RoPE 在后），attention scale 用 $\sqrt{d_h + d_h^R}$ 而不是 $\sqrt{d_h}$

### A.2 Long Context fine-tune 关键超参（YaRN 经验）

- 训练 token 数：≈ 1 B tokens（≈ 400-1000 步）即可显著改善 PPL
- 数据：必须含**真实长上下文**（books / arxiv / code repos），不能只用拼接的短文档
- 学习率：通常 $1\times 10^{-5}$ 到 $5\times 10^{-5}$，比 pretrain 小一个量级
- 不冻：所有层都参与 fine-tune；冻 attention 层效果显著更差
- Eval：PPL 在长上下文上、Needle-in-Haystack 检索准确率

### A.3 StreamingLLM 部署 checklist

- Sink size $S = 4$ 经验最优（前 4 个 token）
- Window size $W$ 选择：吞吐 vs 质量权衡，常用 $W \in [1024, 4096]$
- Position ID 必须用**逻辑位置**而非绝对位置
- 与 RoPE / YaRN 兼容；按 §10.3 的做法 cache 应存 *RoPE 之前* 的 K，每次 forward 根据 sink / window 内 token 的**当前逻辑位置**重新施加 RoPE（部分 vendor 实现会把 sink 段当作"固定旋转 + 平移 attention key 索引"等价处理，效果近似）

### A.4 速查表

| 上下文 | 推荐方案 | KV cache 优化 |
| --- | --- | --- |
| 4K-16K | RoPE + ABF / NTK-aware (zero-shot) | GQA |
| 16K-32K | PI / YaRN + fine-tune | GQA |
| 32K-128K | YaRN + fine-tune | GQA / MLA |
| 128K-2M | LongRoPE + fine-tune | MLA + Ring/CP |
| 流式生成 | StreamingLLM (sink + window) | 任何，cache 常数大小 |

**Long Context Quick Reference** · 主要参考：Su et al. 2021/2024 (RoPE/RoFormer, Neurocomputing), Chen et al. 2023 (PI, arXiv:2306.15595, Meta), bloc97 / jquesnelle 2023 (NTK-aware, LocalLLaMA community), Peng et al. 2023 (YaRN, arXiv:2309.00071), Ding et al. 2024 (LongRoPE, ICML 2024, Microsoft), DeepSeek-AI 2024 (DeepSeek-V2, arXiv:2405.04434), Jiang et al. 2023 (Mistral 7B, arXiv:2310.06825), Xiao et al. 2024 (StreamingLLM, ICLR 2024), Nelson F. Liu et al. (Lost in the Middle, arXiv:2307.03172, arXiv 2023 / TACL 2024), Hao Liu et al. 2023 (Ring Attention, arXiv:2310.01889), Dao et al. 2022-2024 (FlashAttention 1/2/3)
