## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 LLM Quantization** — 一页拿下面试核心要点（详见后文 §2–§11 推导）。

1. **Affine quantization 公式**：$q = \mathrm{round}(x / s) + z$，反量化 $\hat{x} = s\,(q - z)$。对称量化 $z = 0$；非对称量化 $z$ 把 zero-point 对齐到一个整数。

2. **粒度三档**（scale 共享范围越小，精度越高、开销越大）：per-tensor → per-channel → per-group（一行/一列内每 $g$ 个元素共享一个 scale，$g = 32 / 64 / 128$ 主流）。

3. **LLM 量化痛点**：activation 存在 **per-channel 系统性 outlier**（少数 channel 量级是平均的 $50\text{-}100\times$），均匀量化必崩。Weight 分布相对平坦，weight-only 量化（GPTQ / AWQ）天生比 weight+act 容易。

4. **GPTQ (Frantar 2023 ICLR)**：基于 OBS 推导的最优 weight update 公式 $\delta_{\mathbf{w}} = -\dfrac{w_q - \mathrm{quant}(w_q)}{[H^{-1}]_{qq}}\,[H^{-1}]_{q,:}$，逐列量化 + Cholesky 加速 + 128 列 block，4-bit weight 几乎无损。

5. **AWQ (Lin 2024 MLSys)**：观察 "1% salient weights drive most loss"，按 activation 幅度选 salient channel，**per-channel scale $s_c$ 在 $w \to w\cdot s_c$ / $x \to x/s_c$ 下数学等价但量化误差降低**，grid search $s_c = \mathrm{mean}(|x_c|)^\alpha$。

6. **SmoothQuant (Xiao 2023 ICML)**：把 activation outlier **迁移到 weight**——$Y = (X \mathrm{diag}(s)^{-1})(\mathrm{diag}(s)\,W)$，数学完全等价但 $X / s$ 平滑得多，得以做 W8A8。

7. **低精度浮点格式族**：FP8 (E4M3/E5M2, Hopper)、MX (OCP MXFP8/MXFP6/MXFP4, 32-elem block + E8M0 shared exp)、NVFP4 (Blackwell B100/B200, FP4 E2M1 + per-16-elem FP8 E4M3 scale + per-tensor FP32 scale)。Blackwell tensor core 原生支持 FP4 matmul。

8. **KV cache quant**：K 用 **per-channel**（K 的 outlier 沿 channel 维稳定），V 用 **per-token**（V outlier 沿 token 维变化）——KIVI / KVQuant 的基本设计。QServe 进一步把 W4A8KV4 整套量化做 SM89/SM90 kernel-level co-design。

## §1 直觉：为什么 LLM 需要量化、为什么这么难

### 1.1 量化的本质

把高精度浮点数（FP16 / BF16）映射到低位宽整数（INT8 / INT4）或低精度浮点（FP8 / FP4），换取：

- **显存**：FP16 → INT4 直接 $4\times$ 压缩（70B 模型从 140 GB → 35 GB，单卡可推）。

- **显存带宽**：decode 阶段是 memory-bound（每 token 读一遍全部 weights），位宽降一半延迟降一半。

- **算力**：INT8 / FP8 / FP4 tensor core throughput 比 FP16 高 $2\text{-}8\times$（Hopper / Blackwell 上更显著）。

代价：**精度损失**。对 LLM 而言，损失主要来自两个源——activation outlier 和 weight 边界效应。

### 1.2 LLM 量化为什么比 CNN 难

CNN 时代的 INT8 PTQ（NVIDIA TensorRT 2017 那一套）几乎免费：CNN activation 分布近似 Gaussian，per-tensor calibration 即可。LLM 上同样的方法用到 OPT-66B / LLaMA-70B 直接掉 5-10 个 PPL 点。原因：

- **Outlier scaling 现象**（Dettmers 2022, LLM.int8()）：模型 ≥ 6.7B 之后，少数 channel（约 0.1%-1%）的 activation 量级显著高于其他（$50\text{-}100\times$），且**这些 channel 在不同 token / sample 上稳定地是同一批**。

- **Outlier 抢走 scale**：per-tensor scale 由 max 决定。少数 outlier 把 scale 顶得很大，大部分 "normal" channel 量化后落入很窄的整数区间（INT8 下只用 ±10 / ±127），有效 bit width 大幅缩减。

- **Per-channel weight 易、per-channel activation 难**：activation 的 channel 维（hidden dim, $D$）是矩阵乘的内积维（K 维），per-channel scale 沿 K 维变化意味着不能直接在 GEMM 上 fuse；weight 的 output-channel 维（N 维）天然 per-channel 可行（output dim 之间无相互内积）。

### 1.3 三类主流方案

| 方案 | 量化谁 | 代表方法 | 核心 trick |
|---|---|---|---|
| **Weight-only PTQ** | W4/W8, A 保持 FP16 | GPTQ, AWQ, QuIP, GGUF Q4_K | weight 易量化；用 calibration data 找最优量化误差补偿 |
| **Weight + Activation PTQ** | W8A8 | SmoothQuant, ZeroQuant, FP8 | 必须处理 activation outlier（迁移 / 旋转） |
| **Weight + Act + KV (低 bit)** | W4A8KV4 / W4A4 | QuaRot, QServe, SpinQuant | Hadamard / 学习旋转把所有方向上的 outlier 打平 |

> ⚠️ **decode 与 prefill 区别** — Decode 是 memory-bound（KV cache + weight 读取量主导），低 bit weight + KV 直接降延迟。Prefill 是 compute-bound（$L^2$ 的 attention + 大 batch GEMM），W4A4 / FP8 才有意义；纯 weight-only 量化在 prefill 上几乎不省时间（甚至因 dequant overhead 反而变慢）。

## §2 量化数学基础

### 2.1 Affine quantization（uniform / linear 量化）

把实数 $x \in [\alpha, \beta]$ 映射到整数 $q \in [Q_\min, Q_\max]$（如 INT8: $[-128, 127]$，INT4: $[-8, 7]$）。

$$\boxed{\;q = \mathrm{clamp}\!\left(\mathrm{round}(x / s) + z,\; Q_\min,\; Q_\max\right),\quad \hat{x} = s\,(q - z)\;}$$

- **Scale** $s = (\beta - \alpha) / (Q_\max - Q_\min)$，**zero-point** $z = \mathrm{round}(Q_\min - \alpha/s)$。

- **对称量化**（symmetric）：$\alpha = -\beta$，强制 $z = 0$，反量化退化为 $\hat{x} = s\cdot q$。优点：no zero-point addition in GEMM；缺点：分布不对称时浪费 1 bit 表达范围。

- **非对称量化**（asymmetric）：保留 $z$，可精确覆盖任意 $[\alpha, \beta]$。代价：GEMM 内 $W\cdot X$ 展开时多 4 项（cross-zero-point 项），实际实现常用 "subtract zero-point first then GEMM" 把 cross 项归零。

### 2.2 量化误差与 SNR

舍入误差 $\epsilon = x - \hat{x}$ 在 $[-s/2, s/2]$ 上近似均匀分布，方差 $\mathbb{E}[\epsilon^2] = s^2/12$。信号 $\mathrm{Var}(x) = \sigma^2$，则量化 SNR：

$$\mathrm{SNR} = 10 \log_{10} \frac{\sigma^2}{s^2/12} = 10 \log_{10}(12) + 20 \log_{10}\frac{\sigma}{s}$$

INT8 对 $\mathcal{N}(0, 1)$ 用 $\beta = 3\sigma$ 截断时，每减 1 bit SNR 降约 6 dB（每 bit 增加 1 倍量化 step → SNR 下降 $20\log_{10} 2 \approx 6$ dB）。但 LLM 实际衡量是 PPL / 任务指标，远比 SNR 复杂。

### 2.3 粒度（granularity）

| 粒度 | Scale 共享范围 | 显存开销 | 精度 |
|---|---|---|---|
| **per-tensor** | 整个张量一个 $s$ | $O(1)$ | 最差（outlier 一坏全坏） |
| **per-channel** | weight 沿 output channel（行） / activation 沿 hidden（列） | $O(N)$ | 中等 |
| **per-group** | 每 $g$ 个元素共享一个 $s$（一般沿 input/K 维分组） | $O(NK/g)$ | 高（$g = 32 / 64 / 128$） |
| **per-token (act only)** | activation 每个 token (一行) 一个 $s$ | $O(L)$ | 高，但每 step 算一次 |

> 💡 **Per-group 是 W4 量化的工业标准** — GPTQ / AWQ / GGUF Q4_K 默认 group_size = 128：在 input dim 上每 128 个 weight 共享一个 scale。Storage 开销（对称量化，无 zero-point）：W4 + group128（INT4 quant + FP16 scale per 128 weights）≈ $4 + 16/128 = 4.125$ bits/weight；group32 ≈ $4 + 16/32 = 4.5$ bits/weight，精度更高。若是非对称量化（每组还需存一个 INT4 zero-point），公式应为 $4 + (16+4)/g$：group128 ≈ 4.156 bits/weight，group32 ≈ 4.625 bits/weight。

### 2.4 Rounding 模式

- **Round-to-nearest, ties to even** (RNE)：IEEE 754 默认，无偏。LLM 量化一般默认用。

- **Stochastic rounding**：$\mathrm{round}(x/s + u)$，$u \sim \mathcal{U}[-0.5, 0.5)$。期望无偏，适合 QAT 反向传播。

- **AdaRound** (Nagel 2020 ICML)：把舍入决策当作 $\{0, 1\}$ 二值优化变量，逐 layer 用代理 loss 学习（"应该 round up 还是 down"），比 RTN 显著提升 PTQ。

### 2.5 简洁可运行代码：非对称 INT8 per-channel q/dq

```python
import torch

def quantize_per_channel_asym(x: torch.Tensor, n_bits: int = 8, channel_dim: int = 0):
    """
    Asymmetric per-channel quantization.

    Args:
        x: float tensor, e.g. weight [out_features, in_features]
        n_bits: target bit width (e.g. 8 -> INT8)
        channel_dim: which dim to share scale on (typically 0 for weight rows)

    Returns:
        q_int: int quantized tensor (still stored as int32 here for simplicity)
        scale: [N] float scale per channel
        zero_point: [N] int zero-point per channel
    """
    q_min = -(1 << (n_bits - 1))           # -128 for INT8
    q_max = (1 << (n_bits - 1)) - 1        # 127  for INT8

    # Reduce over all dims except channel_dim
    reduce_dims = [d for d in range(x.dim()) if d != channel_dim]
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)

    # Avoid degenerate (zero range) channels
    eps = 1e-8
    scale = (x_max - x_min).clamp(min=eps) / (q_max - q_min)
    zero_point = (q_min - x_min / scale).round().clamp(q_min, q_max)

    q = (x / scale + zero_point).round().clamp(q_min, q_max).to(torch.int32)
    return q, scale, zero_point.to(torch.int32)


def dequantize_per_channel(q_int: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """ x_hat = scale * (q - zero_point)   (per channel) """
    return scale * (q_int.to(scale.dtype) - zero_point.to(scale.dtype))


# Sanity check
if __name__ == "__main__":
    W = torch.randn(64, 1024) * 0.1
    W[3, :] *= 10.0   # one outlier channel
    q, s, z = quantize_per_channel_asym(W, n_bits=8, channel_dim=0)
    W_hat = dequantize_per_channel(q, s, z)
    err = (W - W_hat).abs().max().item()
    print(f"max abs err = {err:.4e} (should be ≤ max_scale/2)")
```

> ⚠️ **PyTorch 早期版本的 `round` 是 banker's rounding (RNE)，与 CUDA / TensorRT 不一致** — 跨 backend 部署时务必用同一 rounding；在 tie 点（$x/s$ 恰为整数 + 0.5）不同 rounding 惯例给出的整数 $q$ 相差整整 1 个 quantization level，反量化后差一个完整的 quantization step $s$（1 LSB），而不是 0.5 ULP。

## §3 LLM 的 Outlier 问题

### 3.1 经验观察（Dettmers 2022, LLM.int8()）

模型规模超过 6.7B 后，每一层 attention input / FFN input activation 中，**少数 hidden channel 的幅值是其他 channel 的 $50\text{-}100\times$**，且：

- **稳定性**：同一 channel 在所有 token、所有 sample 上都是 outlier（不是随机出现）。

- **集中性**：约 0.1%-1% 的 channel 贡献了几乎所有的最大值。

- **后果**：per-tensor INT8 几乎所有 PPL 损失来自这些 channel；per-channel weight + per-tensor activation INT8 在 < 6.7B 模型上 OK，到 OPT-13B / 66B 上崩盘。

### 3.2 LLM.int8() — 第一个能落地的 8-bit LLM

Dettmers 2022 (NeurIPS) 的核心思路：**Mixed-precision decomposition**。在每一层把 activation 拆成两路：

- **Outlier path**：选出 outlier channel（threshold $\alpha = 6.0$ 即 channel-max $> 6$），把对应的 weight 列和 activation 列**保留 FP16** 做矩阵乘。

- **Normal path**：其余 99% 的 channel 做 INT8 vector-wise（per-row activation × per-column weight）矩阵乘。

- **合并**：两路结果相加。

数学等价于把矩阵分块：

$$Y = X W = \underbrace{X_O W_O}_{\text{FP16, outlier 列}} + \underbrace{X_N W_N}_{\text{INT8, 其余}}$$

效果：OPT-175B INT8 推理几乎零 PPL 损失，但 outlier path 的 FP16 GEMM 是 throughput 瓶颈（~10-15% 延迟），且 outlier mask 在每个 forward step 都要 detect。这促使后续工作（SmoothQuant / AWQ）转向"把 outlier 干掉而不是绕开"的思路。

### 3.3 不同 outlier 形态（GPTQ / AWQ / SmoothQuant 攻击的对象）

| Outlier 类型 | 沿哪一维稳定 | 谁能解决 |
|---|---|---|
| **Activation channel outlier**（hidden dim 维） | input channel（K 维） | SmoothQuant（迁移到 weight）, AWQ（per-channel scale 保护） |
| **Token outlier**（少数 token 整行偏大） | sequence 维（L 维） | per-token activation quant（zeroquant / SmoothQuant 默认）|
| **Weight outlier**（少数 weight 偏大） | output dim（N 维） | per-channel weight quant 即可吸收 |
| **KV cache K-channel outlier** | K 的 head_dim 维 | per-channel K quant（KIVI / KVQuant） |

## §4 GPTQ：基于 OBS 的最优 weight 量化（必考推导）

GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023) 是 weight-only PTQ 的工业标准。其数学基础是 **Optimal Brain Surgeon (OBS)**（Hassibi & Stork, NeurIPS 1992），把"删除/修改一个 weight 后如何最小化 loss 增量"推广到"量化一个 weight 后如何更新剩余 weight 以补偿"。

### 4.1 问题设定

对单个 linear layer 输出 $Y = X W$，$X \in \mathbb{R}^{B\times K}$ 来自 calibration set，$W \in \mathbb{R}^{K \times N}$。量化目标：找 $\hat{W}$（每个元素是 INT4 / INT3）最小化：

$$\min_{\hat{W}} \|X W - X \hat{W}\|_F^2$$

这是关于 $\hat{W}$ 的 layer-wise reconstruction objective。注意：

- 这是 **layer-wise**，不是 end-to-end loss（PTQ 假设：layer-wise 重建好 → end-to-end loss 增量小，对线性 / 弱非线性架构基本成立）。

- 由于 GEMM 在 $N$ 维独立可分（每列 $w_j$ 自己一个最小化问题），把 $W$ 按列拆开是常见做法。下面对**一列** $w \in \mathbb{R}^K$ 推导。

### 4.2 二阶 Taylor 展开

设 $L(w) = \frac{1}{2}\|Xw - Xw^*\|^2$（$w^*$ 是 FP16 原始 weight，$w$ 待优化）。在 $w = w^*$ 处展开：

$$L(w^* + \delta) = \underbrace{L(w^*)}_{= 0} + \nabla L(w^*)^\top \delta + \frac{1}{2}\delta^\top H \delta + O(\|\delta\|^3)$$

由于 $L$ 在 $w^*$ 是全局极小，$\nabla L(w^*) = 0$。Hessian：

$$H = X^\top X \in \mathbb{R}^{K\times K}$$

注意 $H$ 与 $w^*$ 无关（calibration data 算一次即可全列复用），且与 column index $j$ 无关（每列共享同一 $H$）。所以：

$$L(w^* + \delta) \approx \frac{1}{2}\delta^\top H \delta$$

### 4.3 OBS：固定一个坐标到目标值后的最优 $\delta$

OBS 的关键问题：**强制把 $w$ 的第 $q$ 个分量 $w_q$ 改成目标值 $w_q^{\mathrm{target}}$**（在我们这里 $w_q^{\mathrm{target}} = \mathrm{Quant}(w_q)$），其他坐标如何调整才能最小化 $\delta^\top H \delta / 2$？

约束 $e_q^\top \delta = w_q^{\mathrm{target}} - w_q^* := c_q$（其中 $e_q$ 是第 $q$ 个标准基）。用 Lagrangian：

$$\mathcal{L}(\delta, \lambda) = \frac{1}{2}\delta^\top H \delta - \lambda (e_q^\top \delta - c_q)$$

求导：$\nabla_\delta \mathcal{L} = H\delta - \lambda e_q = 0 \Rightarrow \delta = \lambda H^{-1} e_q$。

代入约束 $e_q^\top \delta = c_q$：

$$\lambda \cdot e_q^\top H^{-1} e_q = c_q \;\Rightarrow\; \lambda = \frac{c_q}{[H^{-1}]_{qq}}$$

所以：

$$\boxed{\;\delta^* = \frac{w_q^{\mathrm{target}} - w_q^*}{[H^{-1}]_{qq}}\; H^{-1} e_q\;}$$

即 $\delta^*$ 的第 $j$ 个分量 = $\dfrac{c_q}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{jq}$。**所有其他坐标的最优补偿全部来自 $H^{-1}$ 的第 $q$ 列**。

最小损失增量：

$$\Delta L^* = \frac{1}{2}\delta^{*\top} H \delta^* = \frac{1}{2}\cdot \frac{c_q^2}{[H^{-1}]_{qq}}$$

> ✅ **GPTQ 的最优 weight update（必考）** — 量化第 $q$ 列后，剩余未量化的所有列 $j > q$ 按下式更新一次：
$$w_j \leftarrow w_j - \frac{w_q - \mathrm{Quant}(w_q)}{[H^{-1}]_{qq}}\,[H^{-1}]_{jq}$$
之后再量化 $q+1$ 列。这就是 GPTQ "iterative columnwise quantization" 的数学公式。

### 4.4 工程加速：Cholesky 解 $H^{-1}$ 子矩阵

直接对每个 $q$ 求 $H^{-1}$ 的列代价 $O(K^2)$；GPTQ 用 Cholesky 分解 $H^{-1} = U^\top U$（$U$ 上三角，等价地 $H^{-1} = L L^\top$ 若取下三角 $L = U^\top$），扫到第 $q$ 列时只需 $U$ 的子矩阵，且更新可向量化。

GPTQ 的实际实现是把 $K$ 列按 **block size = 128** 分块，每个 block 内部按列 quantize 并 update，block 间用懒惰的批量（lazy batched）方式一次性 sync update（Cholesky decomposition based），从而减少内存访问次数。整体复杂度：$O(MK^2 + K^3 + NK^2)$ per layer（$M$ = calibration 样本数，$K$ = input dim，$N$ = output dim；$MK^2$ 来自算 Hessian $H=X^\top X$，$K^3$ 来自 Cholesky 分解，$NK^2$ 来自跨 $N$ 列共享 $H^{-1}$ 的更新传播），对 7B 模型单 A100 上 ~30 分钟即可量化完。

> 💡 **block size ≠ group_size** — 这里的 "block size = 128"（更新批处理粒度，纯工程加速手段，决定多少列的更新被延迟合并再一次性写回）与 §2.3 提到的 **group_size**（量化 scale 的共享粒度）是两个独立概念，只是常见实现里两者都取 128，容易被混为一谈。

### 4.5 GPTQ-style 伪代码（必考写法）

```python
import torch

@torch.no_grad()
def gptq_quantize_layer(
    W: torch.Tensor,             # [N, K]  weight (one linear layer)
    X: torch.Tensor,             # [B, K]  calibration input to this layer
    n_bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
):
    """
    Direct (non-batched) GPTQ for ONE linear weight matrix.

    Hessian H = X^T X is shared across rows of W.
    We quantize columns of W one-by-one (so we walk along K).
    After quantizing column q, propagate the residual error
    along H^-1 to all yet-unquantized columns j > q.

    NOTE: for clarity this directly updates ALL remaining columns
    W[:, q+1:] after every single column (mathematically identical to
    §4.3's OBS update). The real GPTQ implementation instead delays
    the update within a `block_size`-wide window (lazy batched update):
    it only updates columns inside the current block after each step,
    and applies the full cross-block update once per block — same
    result, far fewer memory passes. `block_size` here is an update-
    batching granularity, distinct from `group_size` (the quantization
    scale granularity above), even though both commonly default to 128.
    """
    N, K = W.shape
    device = W.device
    H = X.t() @ X                                              # [K, K]
    # Damping: avoid singular H. Replace zero diag with mean diag * damp_percent.
    mean_diag = torch.mean(torch.diag(H))
    diag = torch.arange(K, device=device)
    H[diag, diag] += damp_percent * mean_diag

    # Cholesky on H^-1: GPTQ uses upper-triangular inverse Cholesky factor.
    Hinv = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)

    Q = torch.zeros_like(W)                                    # quantized W
    W = W.clone()                                              # we will mutate

    q_min = -(1 << (n_bits - 1))
    q_max = (1 << (n_bits - 1)) - 1

    for q in range(K):
        # Per-group scale: when entering a new group, recompute scale on W[:, q:q+group]
        if q % group_size == 0:
            end = min(q + group_size, K)
            w_grp = W[:, q:end]
            s = w_grp.abs().amax(dim=1, keepdim=True) / q_max
            s = s.clamp(min=1e-8)                              # [N, 1]

        w = W[:, q:q+1]                                        # [N, 1]   current column
        # Quantize this column (symmetric, per-row scale `s`)
        q_int = (w / s).round().clamp(q_min, q_max)
        w_q = q_int * s                                        # [N, 1]   dequantized
        Q[:, q:q+1] = w_q

        # OBS-derived weight update on all columns to the right
        err = (w - w_q) / Hinv[q, q]                           # [N, 1]
        W[:, q+1:] -= err @ Hinv[q:q+1, q+1:]                  # [N, K-q-1]

    return Q
```

> ⚠️ **GPTQ 工程坑** — (1) Calibration data 量：典型 128 samples × 2048 tokens，太少 Hessian 病态；(2) Damp 不能省，$H$ 经常有 zero diag（某些 input channel 在 calib set 上恒为 0）；(3) Activation reorder（`act_order=True`，按 $\mathrm{diag}(H)$ 降序量化）显著提升 W3 / W2 精度但增加 inference dequant 索引开销，AutoGPTQ 默认关。

### 4.6 GPTQ vs 早期 round-to-nearest (RTN)

|  | RTN | GPTQ |
|---|---|---|
| 误差补偿 | 无（每个 weight 独立 round） | 有（OBS 公式向后传播误差） |
| Calibration | 不需要 | 需要 128-512 samples |
| 4-bit on LLaMA-7B | PPL +1.5 | PPL +0.1 |
| 3-bit on LLaMA-7B | PPL +14 (崩) | PPL +0.7 |
| 时间 | 秒级 | 单卡 30-60 分钟（7B） |

## §5 AWQ：Activation-Aware Weight Quantization

AWQ (Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han, MLSys 2024) 的核心洞察：**1% 的 "salient" weights 决定了几乎全部的量化损失**——这些 salient weight 的输入 activation 幅值大。所以**不应该独立量化所有 weight**；要先把 salient channel 放大（量化前），等价地把对应 input scale 缩小，最终量化误差降低。

### 5.1 Salient channel 的识别

不是看 weight 自己的大小，而是看**对应 activation channel 的幅值**：

$$\text{salience}(c) = \mathrm{mean}_{x \sim \mathrm{calib}}\,|x_c|$$

把 channel 按 salience 排序，前 1% 是 "salient channel"，对应 weight 列 $w_{\cdot, c}$ 是 "salient weight"。

### 5.2 Per-channel scale 等价变换（数学等价 ≠ 量化误差等价）

考虑 $Y = X W$，$X \in \mathbb{R}^{B\times K}$，$W \in \mathbb{R}^{K\times N}$。对每个 input channel $c$ 引入正 scale $s_c > 0$：

$$Y = (X / S) (S \cdot W) = \tilde{X} \tilde{W}$$

其中 $S = \mathrm{diag}(s_1, \ldots, s_K)$，$\tilde{X}_{:, c} = X_{:, c} / s_c$，$\tilde{W}_{c, :} = s_c \cdot W_{c, :}$。**在 FP16 下两者完全相等**。但量化后：

- $\tilde{W} = S W$：salient row $w_{c, :}$ 被乘以 $s_c$（变大）。**注意**：保护不是来自"per-channel scale 变小、量化更精细"——group 内所有 channel 共享同一个 $\Delta$，salient channel 被放大反而可能把 $\Delta$（组内 absmax）顶得更大。真正的保护来自反变换：量化误差经 $w \to w\cdot s_c$（对应 activation 除以 $s_c$）后，salient channel 换算回原始尺度的有效误差被压低约 $1/s_c$ 倍，只要 $s_c$ 不过大导致 $\Delta$ 涨幅超过 $s_c$ 本身。

- $\tilde{X} = X / S$：activation 没量化（weight-only PTQ 不动 activation），但下游若有 act quant 则误差也降。

**问题**：$s_c$ 怎么选？过大会让 salient row 太突出抢爆 scale；过小起不到保护效果。AWQ 给出 grid search：

$$s_c = \mathrm{mean}(|x_c|)^\alpha,\quad \alpha \in \{0.0, 0.1, \ldots, 1.0\}$$

对每层独立 grid search $\alpha$，最小化 layer-wise MSE $\|Y_{\mathrm{fp}} - Y_{\mathrm{quant}}\|^2$。$\alpha = 0$ 退化为 RTN（无 scale）；$\alpha = 1$ 直接用 mean act 做 scale；典型最优 $\alpha \in [0.4, 0.7]$。

### 5.3 与 GPTQ 的区别

| 项 | GPTQ | AWQ |
|---|---|---|
| 数据依赖 | Hessian $X^\top X$（需 calibration） | $\text{mean}(\lvert x \rvert)$ per channel（也需 calibration） |
| 优化对象 | 所有 weight 的最优误差补偿 | salient channel 的 scale |
| 量化流程 | iterative, columnwise + OBS update | one-shot scaling + RTN |
| 时间 | 30-60 min (7B) | 5-15 min (7B) |
| 推理 dequant | 可能需要 act reordering | 无额外开销（scale 可吸收进 LayerNorm / W） |
| 与 W 重排兼容 | 较弱 | 强（scale 是 elementwise，不破坏 GEMM 结构） |

> 💡 **AWQ 的工程亲和性** — Per-channel $s_c$ 可以**预 merge 进上游 LayerNorm / RMSNorm 的 weight**：$\mathrm{LN}(x) \cdot \gamma$ 中 $\gamma \leftarrow \gamma / s$，下游 $w \leftarrow s \cdot w$，运行时**完全没有额外 elementwise 操作**。这是 AWQ 比 SmoothQuant 更易部署的原因之一（SmoothQuant 也能 merge，但若 LN 后接 cat / residual 则不能）。**注意**：这个等价变换要求 LN 无 bias（或 bias 同步除以 $s$）——带 bias 的标准 LayerNorm（$\mathrm{LN}(x)=\gamma\odot\mathrm{norm}(x)+\beta$，OPT/BLOOM 用）只除 $\gamma$ 不够，须同时 $\beta \leftarrow \beta / s$ 才能保持等价；RMSNorm 因无 $\beta$ 不受影响。

### 5.4 AWQ-style per-channel scale 搜索代码

```python
import torch

@torch.no_grad()
def awq_search_scale(
    W: torch.Tensor,             # [N, K]   FP16 weight
    X: torch.Tensor,             # [B, K]   calibration activations (post-LayerNorm)
    n_bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
):
    """
    Search per-channel scale s in [0, 1] that minimizes layer-wise MSE
    of dequantized W·X relative to FP16 W·X.

    s_c = mean(|x_c|) ** alpha,   alpha in {0, 1/n_grid, ..., 1}
    """
    device = W.device
    x_mean = X.abs().mean(dim=0).clamp(min=1e-5)              # [K]

    # Y_fp is the FP16 reference output (compute once)
    Y_fp = X @ W.t()                                          # [B, N]

    q_min = -(1 << (n_bits - 1))
    q_max = (1 << (n_bits - 1)) - 1

    best_alpha, best_err = None, float("inf")
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = x_mean.pow(alpha)
        # Normalize s so that its arithmetic mean is 1: keeps scale of W stable.
        s = s / s.mean()
        s = s.clamp(min=1e-4)

        # Apply: W' = W · diag(s),  X' = X / s
        Wp = W * s.unsqueeze(0)                               # [N, K]
        # Group-wise symmetric quantize Wp along K dim
        Wq = _group_quant_dequant(Wp, n_bits, group_size, dim=-1, q_min=q_min, q_max=q_max)
        # Equivalent activation rescaling: divide each input channel by s
        Xp = X / s.unsqueeze(0)
        Y_q = Xp @ Wq.t()

        err = (Y_fp - Y_q).pow(2).mean().item()
        if err < best_err:
            best_err, best_alpha, best_s = err, alpha, s

    return best_alpha, best_s, best_err


def _group_quant_dequant(W, n_bits, g, dim, q_min, q_max):
    """ Symmetric per-group quant-dequant along last dim (assumed K). Assumes K % g == 0. """
    N, K = W.shape
    assert K % g == 0, f"K={K} must be divisible by group_size={g}; pad W or pick g | K."
    Wg = W.view(N, K // g, g)                                # [N, K/g, g]
    s = Wg.abs().amax(dim=-1, keepdim=True) / q_max
    s = s.clamp(min=1e-8)
    Wq = (Wg / s).round().clamp(q_min, q_max) * s
    return Wq.view(N, K)
```

## §6 SmoothQuant：把 activation outlier 迁移到 weight (W8A8)

SmoothQuant (Xiao, Lin, Seznec, Wu, Demouth, Han, ICML 2023) 解决 weight + activation 同时量化（W8A8）的核心痛点：activation outlier 让 per-tensor INT8 崩盘。

### 6.1 核心数学

对 $Y = X W$，引入**对角 smoothing matrix** $S = \mathrm{diag}(s_1, \ldots, s_K)$，$s_c > 0$：

$$Y = (X S^{-1})(S W) = \hat{X} \hat{W}$$

- $\hat{X}_{:,c} = X_{:,c} / s_c$：activation 的 outlier channel 被压平。

- $\hat{W}_{c,:} = s_c \cdot W_{c,:}$：weight 的对应行被放大（吸收 outlier 量级）。

数学上**完全等价**（FP16 下逐元素相等）；但量化后：

- $\hat{X}$ 的最大值显著下降，per-tensor INT8 的有效 bit width 提升。

- $\hat{W}$ 因为 weight 本身**分布平坦**且**可以 per-channel quant**（per output dim 维度，与 $S$ 操作的 K 维正交），少数 row 被放大无伤大雅。

### 6.2 Migration strength $\alpha$（关键超参）

最优 $s_c$ 应平衡"activation 平滑了多少"和"weight 被放大多少"：

$$\boxed{\;s_c = \dfrac{\max(|X_{:, c}|)^\alpha}{\max(|W_{c, :}|)^{1 - \alpha}}\;}$$

- $\alpha = 0$：$s_c = 1/\max|W_c|$。$\hat X = X \cdot \max|W_c|$（activation 被放大，量化反而更难），$\hat W = W / \max|W_c|$（weight 被归一化，量化变易）。**burden 全压在 activation 上**。

- $\alpha = 1$：$s_c = \max|X_c|$。$\hat X = X / \max|X_c|$（activation 被压平，量化变易），$\hat W = W \cdot \max|X_c|$（weight 被放大，量化变难）。**burden 全压在 weight 上**。

- $\alpha = 0.5$（默认）：$s_c = \sqrt{\max|X_c| / \max|W_c|}$，两边各让一半。

SmoothQuant 论文在 OPT / BLOOM 上扫 $\alpha \in [0.3, 0.7]$，typically 0.5 即可。

### 6.3 等价变换不破坏 GEMM（必考）

>  ✅ **为什么 SmoothQuant migration 不破坏 GEMM** — 等价变换 $Y = X W = (XS^{-1})(SW)$，**$S$ 是对角矩阵 (per-channel scale)，对 $X$ 的列 / $W$ 的行做 elementwise rescale**：

- 数学等价：对角矩阵作用是 channelwise multiplication，与 GEMM 的内积顺序无关，最终输出 $Y$ 在 FP16 下逐元素相等。

- 工程实现：$S^{-1}$ 可以 fuse 进**上一层的 LayerNorm weight**（$\gamma \leftarrow \gamma / s$），$S$ 可以 fuse 进**本层 weight**（$W \leftarrow SW$，离线一次性），inference 时**完全没有 elementwise overhead**。若上一层 LN 带 bias（$\beta$），需同步 $\beta \leftarrow \beta / s$ 才能保持等价；RMSNorm 无 $\beta$ 不受此限制；若 LN 输出还喂给其他未同步缩放的下游消费者，也需一并处理。

- 不破坏的关键：$S$ 是 diagonal，rescale 在 K 维上每个 channel 独立。如果 $S$ 是 dense / rotation matrix，则需 explicit matmul，下面的 QuaRot / SpinQuant 选择了那个方向，但代价更大。

### 6.4 SmoothQuant 伪代码

```python
import torch

@torch.no_grad()
def compute_smooth_scale(
    X: torch.Tensor,             # [B*L, K]  per-token flattened activations (calibration)
    W: torch.Tensor,             # [N, K]    weight
    alpha: float = 0.5,
):
    """
    SmoothQuant migration scale (per input channel c):
        s_c = max(|x_c|)^alpha / max(|w_c|)^(1-alpha)
    """
    x_max = X.abs().amax(dim=0)                             # [K]   per channel
    w_max = W.abs().amax(dim=0)                             # [K]   per input ch of W
    s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
    return s                                                # [K]


@torch.no_grad()
def apply_smoothing(
    W: torch.Tensor,
    s: torch.Tensor,
    prev_ln_weight: torch.Tensor,
    prev_ln_bias: torch.Tensor = None,   # pass this if the upstream LN has a bias (e.g. OPT/BLOOM)
):
    """
    Fuse smoothing into upstream LayerNorm weight (+ bias) and current layer W:
        gamma_new = gamma / s   (so output of LN becomes x / s)
        beta_new  = beta / s    (REQUIRED if LN has bias: LN(x) = gamma*norm(x) + beta;
                                  dividing gamma alone gives (gamma/s)*norm(x)+beta,
                                  which is NOT (gamma*norm(x)+beta)/s unless beta is
                                  also divided by s)
        W_new     = W * s       (broadcasts over output dim of W)
    After this, FP16 forward is identical, but X and W are reshaped
    such that simple per-tensor / per-channel quant works well.
    RMSNorm has no bias (beta), so prev_ln_bias is simply omitted there.
    """
    prev_ln_weight.div_(s)                                  # in-place modify γ
    if prev_ln_bias is not None:
        prev_ln_bias.div_(s)                                # in-place modify β (bias-bearing LN only)
    W.mul_(s.unsqueeze(0))                                  # [N, K] broadcasts s along K dim
    return W
```

### 6.5 SmoothQuant 适用范围

- ✅ Decoder-only LLM (OPT, BLOOM, LLaMA family)：W8A8 几乎无损（PPL +0.1）。

- ✅ FFN 和 attention input projection（K, V, Q）：smoothing 可与上游 LN merge。

- ⚠️ Out projection 和 down projection：上游不是 LN（是 residual / attention output），smoothing 需要 explicit elementwise op，工程上 SmoothQuant 默认跳过这两层的 smoothing 预处理——但这不等于保留 FP16 input：这两层仍会做 INT8 量化（通常是简单的 per-tensor / per-token 动态量化，不做离线 outlier-migration），只是不享受 smoothing 带来的额外精度保护。

- ⚠️ INT4 activation：W4A4 用 SmoothQuant 单独不够，需要配合 QuaRot / SpinQuant 旋转。

## §7 旋转方法：QuIP / QuaRot / SpinQuant

SmoothQuant 用 **diagonal**（per-channel）scale 抑制 outlier；但 outlier 仍存在于某些 channel 子空间。**Rotation methods** 用一个 random / learned **正交矩阵** $R$ 把 outlier 在 hidden dim 上"打散"，让分布更接近 Gaussian。

### 7.1 QuIP (Chee et al. 2023, NeurIPS)

核心：用一个随机的 **incoherence-inducing 矩阵** $U$（例如随机 Hadamard 或 Householder）旋转 weight，使量化更友好。

- **Hadamard 变换**：$H \in \mathbb{R}^{d \times d}$，$H_{ij} \in \{+1, -1\} / \sqrt{d}$，正交矩阵。

- 关键性质：随机 sign flip 后的 Hadamard 变换可证明把 weight 的 incoherence（最大 column $\ell_2$ norm 与 Frobenius norm 比）压到 $O(\sqrt{\log d / d})$ 级别。

- $W' = U W V^\top$（$U, V$ orthogonal）；量化 $W'$ 比量化 $W$ 损失小（因为 incoherent）。**完整等价关系**是 $Wx = U^\top W'(Vx)$：需先用 $V$ 旋转输入，用 $W'$ 计算后再用 $U^\top$ 旋转输出才能复原 $Wx$，而不是简单地"$W'$ 与 $W$ FP16 等价"。工程上 $V$ 需吸收进上一层的输出权重，$U^\top$ 需吸收进下一层的输入权重，才能做到 runtime 零额外开销。

代价：inference 时需保留 $U, V$ 的 matmul（一次 dense rotation）。Hadamard 变换有 fast algorithm（$O(d \log d)$），但仍比 SmoothQuant 的对角 fuse 慢。

### 7.2 QuaRot (Ashkboos et al. 2024 NeurIPS)

把 Hadamard 推广到 LLM 全栈：**weight + activation + KV cache 全 INT4**。核心：

- 在每个 residual stream 进入 transformer block 前，**乘上一个 Hadamard $H$**。

- Hadamard 是 orthogonal，纯归一化部分（无 $\gamma$）与 $H$ 可交换（$\mathrm{rms}(Hx) = \mathrm{rms}(x)$），但 $\mathrm{diag}(\gamma)$ 与稠密的 $H$ 一般不交换，直接对带 $\gamma$ 的 RMSNorm 输出穿过 $H$ 不严格成立。QuaRot 的做法是先把 $\gamma$ **离线吸收进下一层权重**（$W \leftarrow \mathrm{diag}(\gamma) W$），使 RMSNorm 变成不含 $\gamma$ 的纯归一化后再与 $H$ 精确交换；"online Hadamard" 只用于少数无法离线吸收旋转的位置（如非线性之后、KV cache），不是解决 $\gamma$ 不交换问题本身的机制。

- Activation 旋转后呈现更 Gaussian 的分布，**outlier 被打散到所有维度**，INT4 activation 量化误差大幅下降。

效果：LLaMA-2 70B 在 W4A4KV4 下 PPL 增量约 +0.5（vs SmoothQuant 的 +5）。代价：每个 block 需要 1-2 次 Hadamard matmul（实际上 fast Hadamard transform 在 H100 上很便宜）。

### 7.3 SpinQuant (Liu et al. 2024 → ICLR 2025, Meta)

把 QuaRot 的随机 Hadamard 换成**学习的旋转矩阵** $R_1, R_2, R_3, R_4$，分别作用于 residual stream / attention input / FFN input / KV cache。优化目标：layer-wise output MSE。

- $R_i \in SO(d)$（特殊正交群），用 Cayley parameterization 或 stochastic gradient on Stiefel manifold 优化。

- 比 QuaRot 多 ~0.5 PPL 改进，但训练时间增加（每个模型需 ~1 GPU 小时学 R）。

> 💡 **旋转方法 vs SmoothQuant** — Smoothing 解决"channel 维 outlier"；rotation 解决"channel-subspace outlier"。Rotation 更通用，但工程成本更高（dense matmul 不能 fuse 进 LN，需要 online compute 或 explicit kernel）。LLaMA-3 / Qwen-2 部署上 W4A8KV4 主流仍是 SmoothQuant + GPTQ；W4A4 才需要 QuaRot / SpinQuant 级别旋转。

## §8 低精度浮点：FP8 / MX / NVFP4

低精度浮点不是新事物（FP16 / BF16 已普遍），新的是 **FP8 (E4M3 / E5M2)** 在 Hopper 上原生 tensor core 支持，以及 **MX / NVFP4** 在 Blackwell 上的 block-scaled 浮点。

### 8.1 IEEE 754-style 浮点编码

一个浮点数 $x = (-1)^s \cdot (1 + m) \cdot 2^{e - \text{bias}}$（normal）或 $x = (-1)^s \cdot m \cdot 2^{1 - \text{bias}}$（subnormal）。

| 格式 | Sign | Exp bits | Mantissa bits | Bias | Max | Min normal |
|---|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 127 | $\sim 3.4\times 10^{38}$ | $\sim 1.2\times 10^{-38}$ |
| FP16 | 1 | 5 | 10 | 15 | 65504 | $\sim 6.1\times 10^{-5}$ |
| BF16 | 1 | 8 | 7 | 127 | $\sim 3.4\times 10^{38}$ | $\sim 1.2\times 10^{-38}$ |
| **FP8 E4M3** | 1 | 4 | 3 | 7 | 448 (no Inf) | $\sim 1.5\times 10^{-2}$ |
| **FP8 E5M2** | 1 | 5 | 2 | 15 | 57344 | $\sim 6.1\times 10^{-5}$ |
| **FP4 E2M1** | 1 | 2 | 1 | 1 | 6 | 1 |

> ✅ **E4M3 vs E5M2 forward/backward 选择** — NVIDIA Transformer Engine 默认：

- **Forward (activation, weight)**：用 **E4M3**——dynamic range 小但分辨率高 (mantissa 3 bit)。Activation 经 layer-wise scale 后落在 $[-448, 448]$ 内，mantissa 精度更重要。

- **Backward (gradient)**：用 **E5M2**——dynamic range 大但分辨率低 (mantissa 2 bit, 形态接近 FP16)。Gradient 量级跨越多个数量级，需要大动态范围。

注意 E4M3 没有 Inf，只有 NaN（最大 normal 即 448）；E5M2 有 Inf 和 NaN（与 FP16 类似）。这点 hardware 设计上有意区分。

### 8.2 FP8 E4M3 bit-level encoding 代码

```python
def fp8_e4m3_encode(x: float) -> int:
    """
    Encode a float into FP8 E4M3 8-bit pattern (returned as int 0..255).
    Format: 1 sign | 4 exponent | 3 mantissa, bias = 7, no Inf, NaN = S.1111.111
    Subnormals: exp = 0, value = (-1)^s * (mantissa/8) * 2^(1-7) = (-1)^s * (m/8) * 2^-6
    Max normal: S.1111.110 -> 448.0
    """
    import math
    if math.isnan(x):
        return 0b0_1111_111
    sign = 0 if x >= 0 else 1
    ax = abs(x)
    if ax == 0:
        return sign << 7
    if ax >= 448.0:
        return (sign << 7) | 0b1111_110                    # saturate to max normal

    # Decompose ax = m * 2^e with m in [1, 2)
    e = int(math.floor(math.log2(ax)))
    m = ax / (2 ** e)                                       # m in [1, 2)

    # Adjust to FP8 E4M3 representation
    biased_e = e + 7                                        # bias = 7

    if biased_e <= 0:
        # Subnormal: shift mantissa right by (1 - biased_e), set exp = 0
        shift = 1 - biased_e
        m_int = int(round((m * 2 ** -shift) * 8))           # 3-bit mantissa
        exp_bits = 0
    else:
        m_int = int(round((m - 1.0) * 8))                   # 3-bit mantissa
        if m_int == 8:                                      # mantissa overflow
            m_int = 0
            biased_e += 1
        if biased_e > 15:                                   # exceeds max exp (biased_e==15 with
                                                             # mantissa 0-6 is still legal: 256-448)
            return (sign << 7) | 0b1111_110                 # saturate
        exp_bits = biased_e

    return (sign << 7) | (exp_bits << 3) | m_int


def fp8_e4m3_decode(b: int) -> float:
    """ Decode an 8-bit FP8 E4M3 pattern (int 0..255) back to a Python float. """
    sign = -1.0 if (b >> 7) & 1 else 1.0
    exp_bits = (b >> 3) & 0b1111
    m_bits = b & 0b111

    if exp_bits == 0b1111 and m_bits == 0b111:
        return float("nan")
    if exp_bits == 0:                                        # subnormal
        return sign * (m_bits / 8.0) * (2 ** -6)
    return sign * (1.0 + m_bits / 8.0) * (2 ** (exp_bits - 7))


# Sanity check: round-trip 448 (max normal)
assert abs(fp8_e4m3_decode(fp8_e4m3_encode(448.0)) - 448.0) < 1e-6
```

### 8.3 MX 格式（OCP / Microsoft 2023）

OCP (Open Compute Project) MX (Microscaling) 规范：把 32 个元素组成一个 block，共享一个 **8-bit shared scale**（E8M0 格式，即 power-of-two scale）；block 内每个元素用 FP4/FP6/FP8 编码。

| MX 格式 | Element type | Block size | Shared scale | 总 bits/element |
|---|---|---|---|---|
| MXFP8 | FP8 (E5M2 or E4M3) | 32 | E8M0 | $8 + 8/32 = 8.25$ |
| MXFP6 | FP6 (E3M2 or E2M3) | 32 | E8M0 | $6 + 8/32 = 6.25$ |
| MXFP4 | FP4 (E2M1) | 32 | E8M0 | $4 + 8/32 = 4.25$ |
| MXINT8 | INT8 | 32 | E8M0 | $8.25$ |

E8M0 是 1 字节、纯指数（无 mantissa、无 sign）的 power-of-two scale：$s = 2^{e - 127}$，$e \in [0, 254]$（255 = 0xFF 保留给 NaN，故 E8M0 无法表示 NaN 以外的 255 号码点，也没有精确的 zero scale）。这种 scale 在 dequant 时是 bit shift（最便宜的硬件操作）。

### 8.4 NVFP4（Blackwell 2025 NVIDIA）

NVFP4 是 NVIDIA 在 Blackwell（B100 / B200 / GB200）上推的 FP4 格式，与 OCP MXFP4 区别：

- **Element**：FP4 E2M1（与 MX 相同）。

- **Block size**：**16**（不是 32），更细粒度。

- **Per-block scale**：**FP8 E4M3**（而非 E8M0），保留 mantissa，scale 自身精度更高。

- **Per-tensor scale**：额外一个 FP32 全局 scale（block scale 是 FP8 受 $\pm 448$ 限制，per-tensor scale 拉大动态范围）。

总 bits/element：$4 + 8/16 + \text{negligible per-tensor} \approx 4.5$ bits。Blackwell tensor core 原生支持 NVFP4 × NVFP4 matmul，throughput 在 B200 上号称 FP16 的 $\sim 8\times$。

> ⚠️ **NVFP4 ≠ MXFP4** — 工业界经常混用 "FP4" 字眼。NVFP4（block=16, FP8 E4M3 scale, +FP32 tensor scale）是 NVIDIA Blackwell 专有；OCP MXFP4（block=32, E8M0 scale）是开放规范，AMD MI350 / Intel Gaudi 3 部分支持。两者在数值精度和硬件路径上不通用。

### 8.5 FP8 在大模型训练中的使用（Transformer Engine）

NVIDIA Transformer Engine (TE) 在 Hopper / Blackwell 上用 FP8 训 LLM 的标准做法：

- **每个 GEMM 维护两个 amax history**（forward / backward 各一），每 8-16 step 更新一次 scale。

- **Delayed scaling**：用上一窗口的 amax 算 scale，避免阻塞当前 step。

- **Hybrid precision**：linear weight 和 grad accumulator 仍 FP32 / BF16，只有 GEMM 的输入 cast 到 FP8。Optimizer state（Adam $m, v$）保持 FP32。

- **Loss scaling**：与 FP16 类似，但因 E5M2 dynamic range 已经接近 FP16，loss scale 可设较小或省略。

LLaMA-3 / DeepSeek-V3 系列 FP8 训练在 Hopper 上吞吐比 BF16 提升 ~1.5-2$\times$。

## §9 KV Cache 量化

LLM inference decode 阶段，per-sample KV cache 显存 = $L_\text{ctx} \cdot 2 \cdot n_\text{layers} \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$。LLaMA-2-70B (80 层, GQA $H_\text{kv}=8$, $d_\text{head}=128$, FP16, $L_\text{ctx}=4096$)：

$$4096 \times 2 \times 80 \times 8 \times 128 \times 2\text{B} = 1.34\text{ GB / sample}$$

batch 64 → 86 GB（A100 80GB 一卡塞不下，必须 KV 量化）。

### 9.1 KIVI / KVQuant 的关键观察

经验：

- **K cache** 的 outlier 沿 **head_dim** 维度 (channel 维) **稳定出现**（与 RoPE 编码的相位有关，某些 freq band 量级大）。

- **V cache** 的 outlier 沿 **token** 维度（sequence 维）出现，且每个 token 独立。

所以最优粒度：

- **K**：**per-channel** quant（每个 head_dim 一个 scale）。

- **V**：**per-token** quant（每个 sequence position 一个 scale）。

### 9.2 KIVI (Liu et al. 2024 ICML)

KIVI = "K per-channel + V per-token" + INT2 quant，搭配 sliding window outlier residual。流程：

1. K cache update 时按 head_dim 维度计算 scale（**per-channel**），quant 到 INT2/INT4。

2. V cache update 时按 token 维度计算 scale（**per-token**），quant 到 INT2/INT4。

3. 最近 $W$ 个 token 保留 FP16（sliding window），避免 quant 噪声主导最新 attention。

LLaMA-2-7B INT2 KV：PPL 增量 $\sim 0.5$；KV cache 本身 FP16→INT2 理论 $8\times$ 压缩，去除 scale / outlier residual 开销后实测 KIVI 论文报告 peak memory（含 weight + activation）约 $2.35\text{-}2.6\times$ 降低、batch size 可放大 $\sim 4\times$。

### 9.3 KVQuant (Hooper et al. 2024, NeurIPS)

进一步分析：

- Per-channel K 不够，**Pre-RoPE quant** 比 post-RoPE 更稳（RoPE 引入相位混合，破坏 channel 维结构）。

- V 用 per-token + non-uniform quant（density-aware）。

效果：LLaMA-2-70B INT4 KV PPL 增量 ~0.04。

### 9.4 QServe / QoQ (Lin et al. 2024 MLSys 2025)

QServe 推出 **W4A8KV4** 全栈量化 + 自定义 GPU kernel。关键工程点：

- **W4A8 GEMM**：weight INT4, activation INT8。dequant 路径：每个 weight 在 register 内用 lookup table 展开成 INT8 后做 INT8×INT8 matmul，无 FP16 dequant 开销。

- **KV4 attention**：K 和 V 都 INT4，与 INT8 query 做 mixed-precision dot product。

- **QoQ (quattuor-octo-quattuor)**：4+8+4 命名，4-bit weight, 8-bit activation, 4-bit KV。

QServe 论文主要实测硬件是 A100 / L40S（部分对比含 A6000），并非 H100；相比 vanilla TensorRT-LLM FP16，throughput 提升约 1.2-3.5$\times$（例如让更便宜的 L40S 在吞吐上追平/超过更贵的硬件跑 TensorRT-LLM FP16，是论文的核心卖点之一）。

### 9.5 KV cache quant 代码示意（per-channel K, per-token V）

```python
import torch

@torch.no_grad()
def quantize_kv_cache(
    K: torch.Tensor,             # [B, H_kv, L, d_head]
    V: torch.Tensor,             # [B, H_kv, L, d_head]
    n_bits: int = 4,
):
    """
    K: per-channel (head_dim) symmetric quant
    V: per-token   (sequence) symmetric quant

    Returns int8-stored tensors plus scales.
    For real deployment, pack two INT4 values into one INT8 byte.
    """
    q_min = -(1 << (n_bits - 1))
    q_max = (1 << (n_bits - 1)) - 1

    # ---- K: per-channel along the last dim (d_head). Same scale across L, H_kv per-batch. ----
    # Common choice: per (batch, head, channel) scale, reduce over L only.
    s_K = K.abs().amax(dim=2, keepdim=True) / q_max         # [B, H_kv, 1, d_head]
    s_K = s_K.clamp(min=1e-8)
    K_q = (K / s_K).round().clamp(q_min, q_max).to(torch.int8)

    # ---- V: per-token, scale per (batch, head, token) reducing over d_head ----
    s_V = V.abs().amax(dim=-1, keepdim=True) / q_max        # [B, H_kv, L, 1]
    s_V = s_V.clamp(min=1e-8)
    V_q = (V / s_V).round().clamp(q_min, q_max).to(torch.int8)

    return K_q, s_K, V_q, s_V


def dequantize_kv(K_q, s_K, V_q, s_V, dtype=torch.float16):
    K = K_q.to(dtype) * s_K.to(dtype)
    V = V_q.to(dtype) * s_V.to(dtype)
    return K, V
```

> ⚠️ **RoPE 前还是 RoPE 后量化？** — 学术界共识：**RoPE 前量化 K**（KVQuant 主张）。原因：RoPE 是 frequency-band 上的 rotation，把 channel 维度上的 outlier"打散"到其他 dim，破坏 per-channel scale 的稳定性。RoPE 前每个 head_dim 的 outlier 是固定 channel，post-RoPE 则在每个 token 上不同。但 RoPE 前 quant 需要在 attention kernel 内 dequant 再做 RoPE，工程上 fuse 比较麻烦；折中方案：post-RoPE 但用更细 group_size（如 32）。

## §10 QAT 与训练时量化

PTQ (Post-Training Quantization) 不动 weight；QAT (Quantization-Aware Training) 在训练或 finetune 时模拟量化，让模型适应。

### 10.1 STE (Straight-Through Estimator)

Round / clamp 在数学上是不可导（round 的导数几乎处处为 0），反向传播无信号。**STE** 把量化-反量化函数 $\mathrm{QDQ}(x) = s\,(\mathrm{clamp}(\mathrm{round}(x/s), Q_\min, Q_\max))$（对称量化为例）的梯度近似为：

$$\frac{\partial \mathrm{QDQ}(x)}{\partial x} \;\overset{\text{STE}}{:=}\; \mathbf{1}\!\left[\,s\,Q_\min \le x \le s\,Q_\max\,\right]$$

即"前向用量化值，反向在 clipping 范围内 pass-through 梯度（饱和区梯度置 0）"。这是 LSQ / DoReFa / PACT 等 QAT 方法的基础。

### 10.2 LLM-QAT (Liu et al. 2023)

- 用**自蒸馏 data**（teacher 是 FP16 模型本身生成的 sequence）做 QAT，避免下载额外训练数据。

- 在每个 forward step 内 simulate INT4 weight quant，反向用 STE。

- 适合 W4A8 / W4A4 finetune 几千 step 后 PPL 接近 FP16。

代价：QAT 比 PTQ 慢 100-1000$\times$。Production 上 PTQ (GPTQ + AWQ) 已经够好，QAT 主要用于 < 4-bit（W2A4 / W1.58 ternary 等）。

### 10.3 FP8 Training（Transformer Engine）

见 §8.5。FP8 training 是 QAT 的特例：训练全程用 FP8 GEMM，scale 用 amax history 周期更新，loss / opt state 仍 FP32。

### 10.4 BitNet b1.58 / b2

最近 (Ma et al. 2024) 微软推 **BitNet b1.58**：weight 是 ternary $\{-1, 0, +1\}$（$\log_2 3 \approx 1.58$ bits），activation INT8。需要 from-scratch QAT 训练（不能 PTQ 转换），3B 规模与 FP16 LLaMA 持平。这是当前最低 bit 的 production-ready LLM 量化方案。

## §11 框架与生态对照

| 框架 | 量化方法支持 | 推理后端 | 典型用例 |
|---|---|---|---|
| **bitsandbytes** | LLM.int8(), NF4, FP4 | PyTorch + Triton | HuggingFace transformers 集成；QLoRA finetune 必备 |
| **AutoGPTQ** | GPTQ (W4, W3, W2) | ExLlama / Marlin kernels | 4-bit 推理 ~2× FP16 throughput |
| **AutoAWQ** | AWQ (W4) | GEMM kernel | 比 GPTQ 略快、精度相当 |
| **llama.cpp / GGUF** | Q4_K, Q5_K, Q6_K, Q8_0, Q3_K, IQ2_XXS... | CPU + GPU + Metal + ROCm | 端侧推理首选 |
| **TensorRT-LLM** | INT8 SmoothQuant, FP8, W4A8, NVFP4 | NVIDIA fused kernel | 生产服务首选（Hopper / Blackwell） |
| **vLLM** | GPTQ, AWQ, FP8, INT8 SmoothQuant | PagedAttention + 自定义 kernel | 多用户 serving 首选 |
| **SGLang** | GPTQ, AWQ, FP8, W4A8 KV | radix-tree + 自定义 kernel | latency-sensitive serving |
| **Transformer Engine** | FP8 training + inference | H100 / B100 cuBLAS | FP8 训练首选 |

> 💡 **Marlin kernel** — Frantar 2024 的 W4A16 GEMM kernel，针对 Ampere / Ada / Hopper 设计，4-bit weight + FP16 activation 在 batch 1-32 上比 FP16 cuBLAS 快 1.5-2$\times$。vLLM / SGLang 默认 W4 路径。

## §12 25 高频面试题

codex (gpt-5.5 xhigh) 顶级 lab 面试官视角列的，按难度分 3 档。每题点开看答案要点 + 易踩坑。

### L1必会题（任何 ML 工程岗都会问）

<details>

<summary>Q1. Affine 量化的 quant / dequant 公式？</summary>

- Quant：$q = \mathrm{clamp}(\mathrm{round}(x/s) + z,\; Q_\min,\; Q_\max)$

- Dequant：$\hat{x} = s\,(q - z)$

- 对称量化 $z = 0$，dequant 退化为 $\hat{x} = s\cdot q$

把 round 和 clamp 顺序写反，或忘掉 zero-point 的减法。

</details>

<details>

<summary>Q2. 对称 vs 非对称量化的取舍？</summary>

- 对称：$z = 0$，GEMM 实现简单（无 cross-zero 项），但分布偏态时浪费 1 bit

- 非对称：精确覆盖任意 $[\alpha, \beta]$，GEMM 需 fuse 掉 zero-point 项

- LLM weight 一般近似零均值 → 对称即可；activation 可能偏（如 ReLU 输出非负）→ 非对称更好

说"非对称一定更精确所以一定更好"，忘了 GEMM cross 项工程开销。

</details>

<details>

<summary>Q3. Per-tensor / per-channel / per-group 区别？</summary>

- per-tensor：整张矩阵一个 scale，开销 $O(1)$，最差精度

- per-channel：weight 沿 output dim 每行一个 scale（or activation 沿 hidden dim）

- per-group：每 $g$ 个 weight 一个 scale，$g = 32 / 64 / 128$，精度最高

- Storage 影响：W4 + group128 ≈ 4.125 bits/weight；group32 ≈ 4.5 bits/weight

混淆"per-channel along which dim"——weight 是 output channel 安全（GEMM K 维独立），activation 沿 hidden（K 维）则不能直接 fuse 进 GEMM。

</details>

<details>

<summary>Q4. LLM 量化为什么比 CNN 难？</summary>

- LLM ≥ 6.7B 后出现 systematic activation outlier（0.1%-1% channel 量级 $50\text{-}100\times$）

- Outlier 在不同 token / sample 上稳定（不是噪声，是结构）

- per-tensor INT8 在小模型 OK，大模型崩 5-10 PPL 点

- 这是 LLM.int8() / SmoothQuant / AWQ 都在攻击的痛点

说"LLM 量化和 CNN 一样"或者"只是规模大"。

</details>

<details>

<summary>Q5. INT8 / INT4 / FP8 区别？</summary>

- INT8：8-bit 整数 $[-128, 127]$，配 scale 表实数

- INT4：4-bit 整数 $[-8, 7]$，必须配 group quant + 比较精细的 calibration

- FP8 E4M3：1S/4E/3M，dynamic range $\pm 448$，forward 用

- FP8 E5M2：1S/5E/2M，dynamic range 与 FP16 相当，backward 用

把 FP8 当 INT8 用；忘了 E4M3 没有 Inf 只有 NaN。

</details>

<details>

<summary>Q6. GPTQ 是什么？它和 RTN 比好在哪？</summary>

- GPTQ (Frantar 2023) = OBQ 在 LLM 上的高效化版本，基于 OBS 的 layer-wise PTQ

- 用 Hessian $H = X^\top X$ 信息，量化第 $q$ 列后**更新剩余列**补偿误差

- W4 LLaMA-7B：RTN PPL +1.5；GPTQ +0.1

- 时间代价：单卡 30-60 分钟/7B

只说"GPTQ 是 4-bit 量化"，不提 OBS 误差传播。

</details>

<details>

<summary>Q7. AWQ 的核心思路？</summary>

- 1% salient weight (input activation 大的 channel) 决定大部分量化损失

- 引入 per-input-channel scale $s_c$，做等价变换 $W \to s W$, $X \to X/s$

- grid search $\alpha \in [0, 1]$，$s_c = \mathrm{mean}(|x_c|)^\alpha$

- 比 GPTQ 快、与 Marlin / W4A16 kernel 配合好

只说"AWQ 比 GPTQ 快"，不提 activation-aware scale 等价变换的数学。

</details>

<details>

<summary>Q8. SmoothQuant 解决什么？为什么 W8A8 不能直接量化？</summary>

- W8A8 直接量化崩 in part because activation per-tensor scale 被 outlier 顶死

- SmoothQuant：$Y = (X/S)(SW)$，$S$ 对角矩阵，**等价变换**

- $X/S$ 平滑、$SW$ 仍然 per-channel 可量

- $s_c = \max|X_c|^\alpha / \max|W_c|^{1-\alpha}$，$\alpha = 0.5$ 默认

只说"SmoothQuant 是 W8A8"，不解释 outlier migration 数学。

</details>

<details>

<summary>Q9. KV cache 占多少显存？怎么算？</summary>

- 公式：$L_\text{ctx} \cdot 2 \cdot n_\text{layers} \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$

- LLaMA-2-70B FP16 4K ctx：$4096 \times 2 \times 80 \times 8 \times 128 \times 2 \approx 1.34$ GB / sample

- batch 64 → 86 GB，需要 KV4 / KV8 才能装下

- MQA / GQA 让 $H_\text{kv} \ll H$（70B 是 GQA G=8，否则 vanilla MHA 是 10 GB / sample）

只说"KV cache 大"，不会算具体数字。

</details>

<details>

<summary>Q10. Bitsandbytes 的 NF4 和 INT4 区别？</summary>

- INT4：均匀量化，16 个等间距 level

- NF4 (Normal Float 4)：非均匀，根据 standard normal 的分位数选 16 个 level

- NF4 假设 weight $\sim \mathcal{N}(0, \sigma^2)$ 后归一化到 $[-1, 1]$，比 INT4 期望意义上更优

- QLoRA (Dettmers 2023) 用 NF4 + double quantization

说 NF4 是非整数所以更慢——错。它是 lookup table-based dequant，速度与 INT4 相当。

</details>

### L2进阶题（research-oriented 岗位）

<details>

<summary>Q11. GPTQ 最优 weight update 公式从 OBS 怎么推？</summary>

- 二阶 Taylor：$L(w^* + \delta) \approx \frac{1}{2}\delta^\top H \delta$（$H = X^\top X$）

- 约束 $e_q^\top \delta = c_q := \mathrm{Quant}(w_q^*) - w_q^*$（注意符号：$c_q$ 是 quant 后减原值）

- Lagrangian + KKT 得 $\delta^* = \lambda H^{-1} e_q$，$\lambda = c_q / [H^{-1}]_{qq}$

- 所以剩余列更新 $w_j \mathrel{+}= (c_q / [H^{-1}]_{qq}) \cdot [H^{-1}]_{jq}$（等价于 §4.3 中用 $-(w_q-\mathrm{Quant}(w_q))/[H^{-1}]_{qq}\cdot[H^{-1}]_{jq}$ 的写法）

只背公式不会推；或混淆 $H = X^\top X$ 的双重身份——它既是对 $w$ 的真实 Hessian，也恰好是输入 Gram 矩阵，二者是同一件事，不是需要区分对错的两个不同概念；或忘了量化第 $q$ 列后还要 propagate 误差到剩余列。

</details>

<details>

<summary>Q12. SmoothQuant migration 为什么不破坏 GEMM？数学上证明。</summary>

- $S = \mathrm{diag}(s_1, \ldots, s_K)$ 对角矩阵

- $Y = X W = X S^{-1} \cdot S W = \hat{X} \hat{W}$ — 矩阵乘法关联律 + 对角矩阵可吸收

- 等价：$\hat{X}_{:, c} = X_{:, c} / s_c$，$\hat{W}_{c, :} = s_c W_{c, :}$（channelwise rescale，不改变 K 维内积结构）

- 工程：$S^{-1}$ fuse 进上游 LN weight，$SW$ 离线 merge 一次，runtime 零开销

只说"对角矩阵可以 fuse"，不写出 $X S^{-1} \cdot S W$ 的代数等价。

</details>

<details>

<summary>Q13. FP8 E4M3 vs E5M2 forward / backward 怎么选？为什么？</summary>

- **Forward (W, A)**：E4M3 — 4E/3M, dynamic range $\pm 448$ 够覆盖经 layer scale 后的 weight/activation，**mantissa 多 1 bit 精度更高**

- **Backward (gradient)**：E5M2 — 5E/2M, 与 FP16 相同 dynamic range，**gradient 量级跨越 $10^{-8}$ 到 $10^4$ 必须大动态范围**

- E4M3 无 Inf 只有 NaN（最大 normal = 448），E5M2 有 Inf + NaN

- NVIDIA Transformer Engine 默认采用此分工

倒过来用（FP backward 用 E4M3）会 overflow（gradient 经常 > 448）。

</details>

<details>

<summary>Q14. AWQ 与 GPTQ 哪个更快？精度差多少？</summary>

- 量化耗时：AWQ 更快（一次 grid search $\alpha$，5-15 min/7B）；GPTQ 慢（Hessian + Cholesky 迭代，30-60 min/7B）

- 精度：W4 上几乎打平（LLaMA-7B 两者都 < +0.2 PPL）

- 推理：AWQ 的 scale 可 merge 进 LN weight，runtime 零开销；GPTQ act_order=True 时有 reorder 索引开销

- 工程：AWQ 与 Marlin W4A16 kernel 配合好，vLLM 默认 W4 路径用 AWQ

说"GPTQ 一定更准"——错，W4 上等价；说"AWQ 不需要 calibration"——错，需要 mean(|x|) per channel。

</details>

<details>

<summary>Q15. INT8 量化后 GEMM 会有 cross zero-point 项，怎么消？</summary>

- $\hat{x}_a = s_a (q_a - z_a)$，$\hat{x}_b = s_b (q_b - z_b)$

- $\hat{x}_a \hat{x}_b = s_a s_b (q_a q_b - z_a q_b - z_b q_a + z_a z_b)$

- 展开后有 4 项；通常预先**让 weight 对称量化 $z_W = 0$** 消两项

- 剩下 $- z_a \cdot q_b$ 项可以**用一行 reduce sum 预算**（per-batch only），inference 时一次性减掉

只说"对称量化"，不解释如何处理 activation 非对称的 cross 项。

</details>

<details>

<summary>Q16. PTQ vs QAT 区别？什么时候用 QAT？</summary>

- PTQ：训练后 calibration + closed-form 量化（GPTQ, AWQ, SmoothQuant）

- QAT：训练或 finetune 时模拟量化（STE 反向）

- LLM 工业现状：W8 / W4 PTQ 已足够（< 0.2 PPL 损失），不需要 QAT

- W2 / 1.58-bit BitNet 必须 from-scratch QAT；finetune 后 W4A4 也常做 QAT

说"QAT 一定更准所以总是用它"——成本 100-1000$\times$ PTQ，对 W8/W4 没必要。

</details>

<details>

<summary>Q17. KV cache 量化 K 和 V 为什么粒度不同？</summary>

- K 的 outlier 沿 **head_dim** 维稳定（特定 channel 大），所以 K 用 **per-channel quant**

- V 的 outlier 沿 **token** 维变化（每个 token 自己 magnitude），所以 V 用 **per-token quant**

- KIVI / KVQuant 都是这个设计

- RoPE 前 quant K 更稳（post-RoPE 把 channel outlier 打散到不同 freq band）

说 "K 和 V 一视同仁 per-tensor"——这正是早期 KV cache 量化崩盘的原因。

</details>

<details>

<summary>Q18. NVFP4 和 MXFP4 区别？</summary>

- 都是 FP4 E2M1 element type（1S/2E/1M）

- **MXFP4** (OCP)：block size **32**，shared scale **E8M0** (8-bit pure exponent, 即 $2^{e-127}$)

- **NVFP4** (NVIDIA Blackwell)：block size **16**，shared scale **FP8 E4M3**（带 mantissa），**额外一个 per-tensor FP32 scale**

- NVFP4 更细粒度、scale 精度更高，但 storage overhead 也大（$4 + 8/16 \approx 4.5$ bits）

- Blackwell tensor core 原生支持 NVFP4，MXFP4 需 AMD MI350 / Intel Gaudi 3

混为一谈或说 "FP4 就是 INT4 加 sign"——错。

</details>

<details>

<summary>Q19. LLM.int8() 的 mixed-precision decomposition 怎么做？</summary>

- 每层把 activation 拆两路：outlier path (channel-max > 6, 保留 FP16) + normal path (其余, INT8 vector-wise quant)

- 数学：$Y = X_O W_O + X_N W_N$，两路独立 GEMM 后相加

- Outlier mask 在每个 forward step detect（不能预先 baked）

- 第一个能落地的 OPT-175B INT8 推理方案，PPL 几乎无损

- 缺点：FP16 outlier path 是 throughput 瓶颈（~10-15% 延迟），后续 SmoothQuant / AWQ 走"消灭 outlier"路线

说 "LLM.int8() 是 pure INT8"——错，是 mixed-precision。

</details>

<details>

<summary>Q20. STE (Straight-Through Estimator) 怎么用？为什么 work？</summary>

- Round 函数处处导数 0 或不存在，反向无信号

- STE: $\partial \mathrm{Round}(x) / \partial x := 1$（在 clamp 范围内），范围外置 0

- 直觉：前向用量化值（discrete），反向当作恒等（pass gradient through）

- 有偏估计但实践上 work；LSQ (Esser 2020) 进一步学习 scale，PACT 学习 clamp threshold

- 不 work 的情况：量化太激进（W2 from scratch），梯度方向显著偏离真实梯度，需要 BinaryConnect 等专门方法

说 STE 是 unbiased estimator——错，是 biased but useful。

</details>

### L3高级变体（顶级 lab / 系统方向）

<details>

<summary>Q21. QuaRot / SpinQuant 的 Hadamard 旋转为什么能消 outlier？</summary>

- 任意正交矩阵 $R$ 把 vector $x$ 变 $Rx$，$\|Rx\|_2 = \|x\|_2$ 不变，但 $\max|x|$ 可以显著减小

- Hadamard $H \in \{+1, -1\}^{d\times d} / \sqrt{d}$：把每个 channel 变成所有 channel 的 $\pm$ 等权平均，**集中 outlier 被打散到所有维度**

- 数学：若 $x$ 有 $k \ll d$ 个 outlier，$Hx$ 的 $\ell_\infty$ norm 约 $\sqrt{k/d} \cdot \max|x|$（incoherence 性质）。**注意**：这需要 $H$ 是带随机符号翻转的 Hadamard（$HD$），且是高概率（非确定性）成立——固定、不加随机化的 Hadamard 矩阵可能与 $x$ 对齐（如 $x$ 正比于某一行）而完全打不散 outlier；精确的集中不等式还需再乘一个 $\sqrt{\log d}$ 因子：$\|Hx\|_\infty \lesssim \sqrt{k/d}\cdot\max|x|\cdot\sqrt{\log d}$。

- QuaRot 在 RMSNorm 处穿过（用 online Hadamard 绕开 $\gamma$ 不通过 $H$ 的问题），SpinQuant 学习 $R$ 而非随机

- W4A4 LLaMA-2-70B：QuaRot PPL +0.5，比 SmoothQuant 的 +5 显著好

说"旋转就是降维 PCA"——错，正交变换保 norm 不降维。

</details>

<details>

<summary>Q22. QServe / QoQ (W4A8KV4) 为什么需要自定义 kernel？</summary>

- W4 weight + A8 activation 的 GEMM 在 stock cuBLAS / cuDNN 上没有直接路径

- QServe 的 kernel：每个 W4 weight 在 register 里 dequant 到 INT8（lookup table），然后做 INT8×INT8 matmul

- 关键优化：dequant 与 Tensor Core MMA（`mma.sync`，一条固定的硬件指令，本身不含通用 dequant 逻辑）在同一个 kernel 内通过**软件流水线（pipelining）**时间重叠、寄存器内完成——下一个 tile 的 dequant 与当前 tile 的 MMA 并行执行，而不是融合成同一条 warp instruction，**避免 FP16 中间 buffer 的写回/读取**

- KV4 attention：K, V 都 INT4，与 INT8 query 做 mixed-precision dot，需要 attention kernel 内的 dequant 路径

- 端到端：QServe 论文主要实测硬件是 A100 / L40S（不是 H100），比 FP16 TensorRT-LLM 快 1.2-3.5$\times$

只说"W4A8 比 FP16 快"，不解释为什么需要 kernel-level co-design（stock GEMM 不支持 W4 input）。

</details>

<details>

<summary>Q23. FP8 training 的 amax history / delayed scaling 是什么？</summary>

- 每个 GEMM 的输入 / 输出维护一个 amax history（最近 N 个 step 的 max abs，N = 16 典型）

- Scale 用 history max 算（$s = \max\text{history} / 448$ for E4M3），保证下一 window 不 overflow

- "Delayed"：用**上一**窗口的 amax 算**当前**窗口的 scale，避免阻塞 forward 等 amax 算出来

- Cast 在 GEMM 入口：FP32 → FP8 用 scale，GEMM 输出累积 FP32 再 cast 出去

- LLaMA-3 / DeepSeek-V3 FP8 train 比 BF16 提升 1.5-2$\times$ throughput

把 delayed scaling 当成 loss scaling 同一个东西——loss scaling 是 backward 路径上抗 underflow，delayed scaling 是 per-GEMM forward/backward 的 amax 用法。

</details>

<details>

<summary>Q24. NVFP4 的 per-tensor + per-block FP8 scale 双层结构为什么必要？</summary>

- FP4 E2M1 max normal = 6，动态范围极窄

- 单层 per-block FP8 scale (E4M3, max 448)：单 block 内可表 $\pm 6 \times 448 \approx \pm 2700$；但跨 block 仍受 FP8 scale 自己范围限制

- LLM activation amax 经常 > 2700 (outlier channel 出现 $10^4$ 级)，per-block FP8 scale 不够

- **额外 per-tensor FP32 scale**：把整个 tensor 拉到 FP8 scale 的合理动态范围内，相当于先全局粗调再 block-wise 细调

- 类比：FP32 用 mantissa+exp 表大 range；NVFP4 用 FP4 mantissa + FP8 block exp + FP32 tensor exp，三层 hierarchy

只说 "NVFP4 = FP4 + scale"——错，是 FP4 + per-block FP8 + per-tensor FP32 三层。

</details>

<details>

<summary>Q25. 设计一个 W4 量化方案给一个未知 LLM，你会怎么做？</summary>

- **Step 1**：跑 layer-wise activation profiling，看是否 ≥ 6.7B 有 systematic outlier；如果有，weight-only 必须配 AWQ scale 或 GPTQ Hessian

- **Step 2**：选 PTQ 方法：

  - 简单部署：AWQ (W4) + Marlin kernel，最佳精度-工程 trade-off

  - 极致精度：GPTQ (W4) + act_order，多 0.5-1% throughput 但 < 0.1 PPL

  - W4A8 / W4A4：必须额外 SmoothQuant / QuaRot

- **Step 3**：calibration data 选择：

  - 通用 LLM：128 samples × 2048 tokens from C4 / WikiText

  - 任务专用：用 in-domain data，PPL 差异显著（数学 task on math data 比 web data 好 2-3 PPL）

- **Step 4**：选 group_size：

  - W4 group128 默认（4.125 bits/weight, 良好精度）

  - W3 / W2 必须 group32 或更小

- **Step 5**：验证：跑 PPL + 下游 task (MMLU / GSM8K)，PPL diff < 0.2 + task drop < 1% 算 PASS

只说"用 GPTQ 4-bit"——没解释如何选 calibration / group_size / kernel backend。

</details>

## §A 附录：完整的工程参考

### A.1 主要论文 reference list

| 方法 | 论文 | 关键贡献 |
|---|---|---|
| LLM.int8() | Dettmers et al., NeurIPS 2022 | Mixed-precision INT8 decomposition for outlier |
| GPTQ | Frantar et al., ICLR 2023 | OBS-based W4 PTQ for LLM |
| SmoothQuant | Xiao et al., ICML 2023 | Migrate activation outlier to weight (W8A8) |
| AWQ | Lin et al., MLSys 2024 | Activation-aware per-channel scale (W4) |
| QuIP | Chee et al., NeurIPS 2023 | Random Hadamard for weight incoherence |
| QuaRot | Ashkboos et al., NeurIPS 2024 | Full Hadamard rotation, W4A4KV4 |
| SpinQuant | Liu et al., ICLR 2025 (Meta) | Learned rotations $R_1$-$R_4$ |
| OmniQuant | Shao et al., ICLR 2024 | Learnable equivalent transforms |
| KIVI | Liu et al., ICML 2024 | per-channel K + per-token V, INT2 KV |
| KVQuant | Hooper et al., NeurIPS 2024 | Pre-RoPE quant K, non-uniform V |
| QServe / QoQ | Lin et al., MLSys 2025 | W4A8KV4 GPU kernel co-design |
| LLM-QAT | Liu et al., 2023 | Self-distillation QAT for W4 |
| BitNet b1.58 | Ma et al., 2024 | Ternary weight LLM from scratch |
| FP8 Training | Micikevicius et al., 2022 | E4M3 forward / E5M2 backward |
| MX formats | OCP / Microsoft, 2023 | Block-scaled FP4/6/8 with E8M0 |
| NVFP4 | NVIDIA Blackwell, 2025 | FP4 E2M1 + FP8 E4M3 block + FP32 tensor scale |

### A.2 一图速查：选什么量化方案

```

  ┌─────────────────────────┐
  │ 是否能接受 PPL +0.5 ?  │
  └────────┬────────────────┘
           │
     ┌─────┴─────┐
     │ 能         │  不能
     ↓           ↓
 [W4 weight-only]   [INT8 / FP8 W+A 量化]
 GPTQ / AWQ          SmoothQuant W8A8
 + Marlin kernel     + per-channel W
 群中典型 PPL: +0.1   PPL: +0.05

  ┌─────────────────────────┐
  │ 是否做 INT4 activation?│
  └────────┬────────────────┘
           │
     ┌─────┴─────┐
     │ 是         │  否
     ↓           ↓
 [W4A4]            [W4A8KV4 / QoQ]
 QuaRot / SpinQuant  AWQ + KV4
 + Hadamard rotation 用 QServe kernel
 PPL +0.5 (70B)      PPL +0.2
```

### A.3 量化 quick reference 卡片

| 任务 | 推荐方案 | 框架 |
|---|---|---|
| 单卡推理 LLaMA-2-70B | AWQ W4 + Marlin (vLLM / SGLang) | AutoAWQ |
| 多卡推理 LLaMA-3-405B | SmoothQuant + GPTQ W4A8 + KV8 | TensorRT-LLM |
| 端侧 (Apple Silicon / CPU) | GGUF Q4_K_M / Q5_K_S | llama.cpp |
| 训练 fp8 LLM | TE FP8 (E4M3/E5M2) + amax history | Transformer Engine |
| QLoRA finetune | NF4 + double quant + LoRA | bitsandbytes + peft |
| 极致 throughput H100/B200 serving | QoQ W4A8KV4 / NVFP4 | QServe / TensorRT-LLM |
| 边缘 < 100 MB model | BitNet b1.58 (1.58-bit) from scratch | 自定义 / bitnet.cpp |

### A.4 Sanity check checklist

实战部署任何量化模型前，必跑：

- [ ] **PPL on calibration domain**：量化前后差 < 0.2 算 OK

- [ ] **PPL on out-of-domain**（重要！）：差 < 0.5 算 OK，> 1 重新选 calibration data

- [ ] **MMLU / GSM8K 等下游 task**：drop < 1% 算 OK

- [ ] **Long context PPL**（最考验 KV cache quant）：4K / 8K / 32K 三档对比

- [ ] **生成质量主观评估**：让人盲评 50 个 prompt 的输出，比 FP16 不显著差

- [ ] **Throughput / TTFT (Time To First Token)**：实测 vs 理论数字差 > 30% 说明 kernel 未充分优化

- [ ] **峰值显存**：实测 vs nominal $\text{bits} \cdot \text{params} / 8 + \text{KV cache} + \text{activations}$

**Quantization Quick Reference** · 主要参考：Dettmers 2022 (LLM.int8()), Frantar 2023 (GPTQ), Xiao 2023 (SmoothQuant), Lin 2024 (AWQ), Ashkboos 2024 (QuaRot), Lin 2025 (QServe). 最后更新：2026-05。