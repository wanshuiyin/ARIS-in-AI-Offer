## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 LoRA / PEFT** — 一页拿下面试核心要点（详见后文 §2–§10 推导）。

1. **核心公式**：冻结预训练权重 $W_0$，只学一个低秩增量 $\Delta W = BA$，前向 $h = W_0 x + \frac{\alpha}{r} BA\,x$。其中 $B \in \mathbb{R}^{d\times r}$、$A \in \mathbb{R}^{r\times k}$、$r \ll \min(d,k)$。

2. **为什么低秩有效**：预训练模型有很低的"内在维度"（intrinsic dimension，Aghajanyan 2020）——低维子空间里优化即可逼近满参；Hu 2021 据此**假设**微调的权重更新 $\Delta W$ 近似低秩，用 $r=4\sim64$ 的低秩矩阵就能逼近（经验假设 + 实证，非严格定理）。

3. **初始化**：$A$ 随机（Kaiming）、$B = 0$，使得**训练起点 $\Delta W = 0$**（不扰动预训练），但梯度不为零、能学起来。两个都置零会永远学不动。

4. **缩放 $\alpha/r$**：解耦 $r$ 与学习率，换 $r$ 不用重调 lr。rsLoRA 指出高秩时应改用 $\alpha/\sqrt{r}$ 防梯度坍缩。

5. **零推理延迟**：训练后可把 $\frac{\alpha}{r}BA$ **合并**进 $W_0$ 得 $W' = W_0 + \frac{\alpha}{r}BA$，推理与原模型完全同构、无额外延迟（这是 LoRA 相对 Adapter / Prefix 的关键优势）。

6. **省的是什么显存**：主要省**优化器状态 + 梯度**（Adam 下满参微调要 16 bytes/param，LoRA 只对 ~0.1% 的可训练参数付这笔钱）；**激活显存不自动省**（仍要反传过冻结基座），需配合 gradient checkpointing。

7. **QLoRA**：基座用 **NF4**（4-bit NormalFloat，对正态权重信息论最优）量化 + **双重量化** + **paged optimizer**，LoRA adapter 走 bf16，单卡 48GB 即可微调 65B。

8. **家族**：DoRA（幅度-方向分解）、rsLoRA（$\sqrt{r}$ 缩放）、PiSSA（主成分初始化）、AdaLoRA（自适应秩预算）、(IA)³（缩放向量）、LoRA+（A/B 不同 lr）—— 都在补 LoRA 的某一块短板。

## §1 直觉：为什么需要 PEFT

**满参微调（full fine-tuning）的痛点是显存，不是算力。** 用 Adam 微调一个 $\Psi$ 参数的模型，混合精度下每个参数要存：bf16 权重 2 B + bf16 梯度 2 B + fp32 master 权重 4 B + Adam 一阶动量 $m$ 4 B + 二阶动量 $v$ 4 B = **16 B/param**。7B 模型光这部分就 112 GB，单张 80GB 卡放不下——还没算激活。

PEFT（Parameter-Efficient Fine-Tuning）的核心思想：**冻结绝大多数预训练参数，只训练极小一部分新增或挑选出的参数**，把可训练量从 100% 压到 0.01%~1%。这样：

- 优化器状态 / 梯度只对那一小撮可训练参数付费 → 显存暴降；
- 一个基座 + 多个轻量 adapter（每个几十 MB）即可服务多任务，热插拔；
- 小数据上更不容易过拟合、灾难性遗忘更轻。

**低秩假设（low-rank hypothesis）是 LoRA 的理论支点。** Aghajanyan et al. (2020, arXiv 2012.13255) 实证发现：预训练语言模型有很低的"内在维度"（intrinsic dimension）——在一个远小于全参数量的随机子空间里优化就能达到满参微调 90% 的效果。Hu et al. (2021) 顺着这个直觉提出：既然微调"走得不远"，那权重的**更新量** $\Delta W$ 本身应当是低秩的，于是用两个瘦矩阵的乘积 $BA$ 去参数化它。

> 💡 **一句话心智模型** — 满参微调是"重写整本书"，LoRA 是"在书页边上贴低秩的便利贴 $\Delta W = BA$"；推理时把便利贴的内容誊抄回正文（merge），读者（推理引擎）完全感觉不到便利贴存在过。

## §2 LoRA 核心公式与推导

### 2.1　主公式与形状

对一个被适配的线性层，原权重 $W_0 \in \mathbb{R}^{d \times k}$（冻结），LoRA 学一个低秩增量：

$$\boxed{\;h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A\, x\;}$$

形状：

- $A \in \mathbb{R}^{r \times k}$（down-projection，把输入压到 $r$ 维）
- $B \in \mathbb{R}^{d \times r}$（up-projection，从 $r$ 维升回 $d$ 维）
- $\Delta W = BA \in \mathbb{R}^{d \times k}$，秩 $\le r$
- $r \ll \min(d, k)$，典型 $r \in \{4, 8, 16, 32, 64\}$
- $\alpha$ 是缩放超参，$\frac{\alpha}{r}$ 是缩放因子（见 §2.3）

可训练参数量从 $d \times k$ 降到 $r(d + k)$。例如 $d=k=4096$、$r=8$：从 16.8M 降到 65.5K，**压缩 256 倍**。

### 2.2　为什么 $B=0$、$A$ 随机初始化（必考）

LoRA 官方初始化：$A$ 用 Kaiming uniform 随机初始化，$B$ 初始化为**全零**。这样：

$$\Delta W_{t=0} = B_0 A_0 = 0 \cdot A_0 = 0 \;\Rightarrow\; h_{t=0} = W_0 x$$

即**训练起点完全等于预训练模型**，不引入任何扰动。这点很关键：如果起点就偏离预训练，等于丢掉了预训练知识、还可能炸 loss。

那为什么不**两个都置零**？因为会永远学不动。看第一步梯度（设损失 $L$，输出方向梯度 $g = \partial L / \partial h$）：

$$\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, g\, (A x)^\top, \qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top g\, x^\top$$

- $A$ 非零 → 一般情况下 $Ax \neq 0$ → $\partial L/\partial B \neq 0$：**第一步 $B$ 能动**（除非 $g=0$ 或 $x$ 恰落在 $A$ 的零空间等退化情形）；
- $B = 0$ → $\partial L/\partial A = 0$：第一步 $A$ 不动，但只要 $B$ 动起来变非零，下一步 $A$ 就有梯度了。

如果 $A=B=0$，则两个梯度恒为零，$\Delta W$ 永远是 0。**所以必须"一零一非零"：$A$ 非零保证可学性，$B=0$ 保证干净起点。**（对称地，$A=0$、$B$ 随机也可以，PEFT 库里两种约定都见过；关键是恰好一个为零。）

> ⚠️ **常见误解** — "$B=0$ 所以 LoRA 第一步什么都没学到" 是错的。第一步 $B$ 确实在更新（梯度非零），只是 $\Delta W$ 在 $t=0$ 那一刻数值为 0。是"输出增量为零"而非"梯度为零"。

### 2.3　缩放因子 $\alpha/r$ 与 rsLoRA

LoRA 把增量缩放为 $\frac{\alpha}{r} BA$。Hu et al. 的说法是："把 $\alpha$ 当作类似学习率的东西来调，固定为最先试的那个 $r$ 值，之后换 $r$ 时不用重调 lr。" 直觉：$\frac{\alpha}{r}$ 让不同 $r$ 下 $\Delta W$ 的量级大致可比，从而**解耦秩与学习率**。

但 Kalajdzievski (2023, rsLoRA, arXiv 2312.03732) 指出 $\frac{\alpha}{r}$ 在**大 $r$ 时会过度缩小**梯度，导致高秩 LoRA 学习坍缩、效果不升反平。其分析：要让前向/反向激活的量级在 $r$ 增大时保持稳定（rank-stabilized），缩放因子应当是 $\frac{\alpha}{\sqrt{r}}$：

$$\gamma_r^{\text{LoRA}} = \frac{\alpha}{r} \quad\text{vs}\quad \gamma_r^{\text{rsLoRA}} = \frac{\alpha}{\sqrt{r}}$$

直觉来自方差：$BA$ 的每个输出元素是 $r$ 项之和，若各项独立同方差，则其标准差 $\propto \sqrt{r}$。除以 $\sqrt{r}$ 才能把量级拉回常数；除以 $r$ 则过度压制，$r$ 越大压得越狠。所以**低秩（$r\le 32$）用 $\alpha/r$ 通常无碍，想吃高秩红利时换 rsLoRA 的 $\alpha/\sqrt{r}$**。

### 2.4　合并与零推理延迟（LoRA 的杀手锏）

训练完成后，$\Delta W$ 可以**一次性合并**进基座权重：

$$W' = W_0 + \frac{\alpha}{r} BA$$

之后推理只用 $W'$，前向 $h = W' x$ 和原始线性层**完全同构**——没有额外矩阵乘、没有额外延迟、没有额外显存。这是 LoRA 相对 Adapter（插入串行子模块）和 Prefix-Tuning（占用序列长度 / KV）的根本优势：**它们都会增加推理开销，LoRA 合并后为零。**

需要切回基座或换 adapter 时，**反合并（unmerge）**即可：$W_0 = W' - \frac{\alpha}{r}BA$。多任务服务时一种常见做法是**不合并**、在线加 $\frac{\alpha}{r}B(Ax)$，这样一个基座可同时挂多个 adapter 动态路由（代价是恢复了一点点推理开销）。

> ⚠️ **QLoRA 不能直接合并** — 基座是 NF4 量化的，把 bf16 的 $\Delta W$ 加进 4-bit 权重要先反量化 → 合并 → （可选）再量化，过程有精度损失。实践里通常**反量化基座到 fp16 再合并**保存，而不是合并回 4-bit。详见 §5.5。

### 2.5　适配哪些矩阵 / 怎么选 $r$

原始论文在 Transformer 的注意力投影上做 LoRA，消融发现：**同样参数预算下，适配 $W_q, W_v$ 比只配 $W_q$ 好；把预算摊到更多矩阵（$q,k,v,o$）通常优于堆在少数矩阵上的高秩**。后续实践（如 QLoRA）进一步建议**把 LoRA 加到所有线性层**（含 FFN 的 gate/up/down），往往比只配注意力更稳。

- $r$ 选择：简单任务 / 小数据 $r=4\sim8$；难任务 / 大数据 $r=16\sim64$。$r$ 过大不仅省不了多少、还可能过拟合或踩到 $\alpha/r$ 的坍缩（见 rsLoRA）。
- $\alpha$ 经验：常设 $\alpha = 2r$（即缩放 $\approx 2$）或 $\alpha = r$（缩放 $=1$），按验证集调。
- target_modules：注意力 `q_proj,k_proj,v_proj,o_proj` + FFN `gate_proj,up_proj,down_proj` 是 LLaMA 系常用全配方。

## §3 从零实现 LoRALinear

下面是一个可跑的 `LoRALinear`：包一个冻结的 `nn.Linear`，加上 $A,B$ 与缩放，支持 merge / unmerge。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """冻结的 base linear + 低秩旁路 BA。forward: W0 x + (alpha/r) B A x。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, rslora: bool = False):
        super().__init__()
        assert r > 0
        self.base = base
        for p in self.base.parameters():       # 冻结基座
            p.requires_grad_(False)

        in_f, out_f = base.in_features, base.out_features
        self.r, self.alpha = r, alpha
        # 缩放：标准 alpha/r；rsLoRA 用 alpha/sqrt(r)
        self.scaling = alpha / (math.sqrt(r) if rslora else r)

        # A: [r, in]  Kaiming 随机；B: [out, r] 置零 -> 起点 ΔW = 0
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        out = self.base(x)                      # 冻结主路 W0 x (+bias)
        if not self.merged:
            lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            out = out + self.scaling * lora      # 加旁路
        return out

    @torch.no_grad()
    def merge(self):
        """把 scaling·BA（标准 alpha/r 或 rsLoRA alpha/sqrt(r)）加进 base.weight，推理零延迟。"""
        if self.merged:
            return
        dW = self.scaling * (self.lora_B @ self.lora_A)   # [out, in]
        self.base.weight.add_(dW.to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        dW = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.sub_(dW.to(self.base.weight.dtype))
        self.merged = False


def inject_lora(model: nn.Module, target_names=("q_proj", "v_proj"),
                r=8, alpha=16, dropout=0.0):
    """把 model 中名字命中 target_names 的 nn.Linear 替换成 LoRALinear。"""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name,
                        LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return model
```

要点：

- 注意 `x @ A.t() @ B.t()`：$x$ 形状 $[\dots, k]$，$A^\top$ 是 $[k, r]$，$B^\top$ 是 $[r, d]$，输出 $[\dots, d]$，与 base 输出对齐。
- `merge()` 用 `@torch.no_grad()` + `add_` 原地改 `base.weight`，合并后 `forward` 跳过旁路。
- dropout 只作用在**进入旁路的输入**上，不动主路（这是 LoRA 的标准做法）。

> 💡 **dropout 放在旁路输入** — LoRA dropout 作用于 $x$ 进入 $A$ 之前，相当于对低秩适配做正则；主路 $W_0 x$ 不加 dropout（基座是冻结的、不需要正则）。

## §4 显存账：LoRA 到底省在哪

把一个 $\Psi$ 参数模型用 Adam 微调，按字节拆解（混合精度训练，per param）：

| 组件 | 满参微调 | LoRA（基座 bf16） |
| --- | --- | --- |
| 基座权重（bf16） | $2\Psi$ | $2\Psi$（冻结，仍需常驻） |
| 基座梯度 | $2\Psi$ | $0$ |
| 基座 fp32 master | $4\Psi$ | $0$ |
| 基座 Adam $m,v$（fp32） | $8\Psi$ | $0$ |
| LoRA 参数 + 梯度 + 优化器 | — | $16 \cdot \Psi_{\text{lora}}$ |
| **可训练相关合计** | $\approx 16\Psi$ | $\approx 2\Psi + 16\Psi_{\text{lora}}$ |

其中 $\Psi_{\text{lora}} \ll \Psi$（典型 0.1%~1%）。所以 LoRA 把"满参微调 16 bytes/param 的大头"几乎全砍掉，只剩基座那份 $2\Psi$ 常驻 + 一丢丢 LoRA 开销。

**但有一个必须主动澄清的点：激活显存（activation memory）LoRA 不自动省。** 因为旁路 $BA$ 挂在冻结层上，反向传播仍要穿过整个网络计算 $\partial L / \partial x$，中间激活照存。LoRA 省的是**优化器状态 + 梯度 + master 权重**，不是激活。要省激活得另上 **gradient checkpointing**（用算力换激活显存）。

可训练参数量（适配 $M$ 个线性层，第 $i$ 层 $d_i \times k_i$）：

$$\Psi_{\text{lora}} = \sum_{i=1}^{M} r_i (d_i + k_i)$$

**具体例子**：LLaMA-7B，适配 $q,k,v,o$（各 $4096\times4096$），$r=8$，32 层：

$$\Psi_{\text{lora}} = 32 \times 4 \times 8 \times (4096+4096) = 8.39\text{M} \approx 0.12\% \text{ of } 7\text{B}$$

显存对比（7B，单卡，忽略激活）：

- 满参 Adam：$7\text{e}9 \times 16 \approx 112$ GB（单卡放不下）
- LoRA（bf16 基座）：$7\text{e}9 \times 2 + 8.4\text{e}6 \times 16 \approx 14$ GB + 0.13 GB ≈ **14 GB**
- QLoRA（NF4 基座）：$7\text{e}9 \times 0.5\,\text{B/param} + \text{overhead} \approx 3.5$ GB + LoRA ≈ **<5 GB**（见 §5；NF4 = 4 bit/param = 0.5 **byte**/param）

## §5 QLoRA：4-bit 基座 + LoRA

QLoRA（Dettmers et al., 2023, arXiv 2305.14314, NeurIPS 2023）让单张 48GB 卡微调 65B 模型，核心是三件套：**NF4 量化** + **双重量化** + **paged optimizer**，再叠加"基座 4-bit 冻结、LoRA 走 bf16"的配方。

### 5.1　NF4：4-bit NormalFloat（信息论最优）

观察：神经网络权重近似服从零均值正态分布 $\mathcal{N}(0,\sigma^2)$。普通 4-bit 整数量化（INT4，等间距 bin）在正态分布上是浪费的——尾部 bin 几乎没有值落入。**NF4 的思路是让每个量化 bin 落入相等数量的权重（等概率质量），在"权重服从零均值正态 + quantile 量化"的假设下，这是对该固定分布信息论最优的（quantile quantization）。**

构造（简化）：取标准正态 $\mathcal{N}(0,1)$ 的 $2^4=16$ 个分位点作为量化级别，使相邻级别之间的概率质量相等；并做**非对称**处理使 0 能被精确表示（zero-preserving，对剪枝 / padding 友好）。量化时按 block（QLoRA 用 block size 64）算 absmax 把权重归一化到 $[-1,1]$，再查最近的 NF4 级别：

$$w \;\xrightarrow{\text{normalize by absmax}}\; \hat{w} \in [-1,1] \;\xrightarrow{\text{nearest NF4 level}}\; q \in \{n_0,\dots,n_{15}\}$$

反量化：$w \approx c \cdot n_q$，其中 $c$ 是该 block 的 absmax 缩放常数。

> 💡 **NF4 vs INT4 vs FP4** — INT4 等间距，对正态分布尾部浪费 bin；FP4 用浮点指数分配非均匀级别但未必匹配权重分布；NF4 直接按正态分位点放级别，对"权重本就近正态"这件事做了最优匹配。论文实测 NF4 显著优于 INT4/FP4。

### 5.2　双重量化（Double Quantization）

NF4 每个 block（64 个权重）要存一个 fp32 的 absmax 缩放常数 $c$，摊到每个参数是 $32/64 = 0.5$ bit/param 的额外开销。**双重量化把这些缩放常数自己再量化一次**：把 $c$（fp32）按 block size 256 用 8-bit 量化，第二层缩放常数用 fp32：

$$\text{开销}: \underbrace{0.5\,\text{bit/param}}_{\text{单次}} \;\to\; \underbrace{\frac{8}{64} + \frac{32}{64\times256}}_{\text{双重}} \approx 0.127\,\text{bit/param}$$

平均每参数省约 **0.37 bit**——对 65B 模型就是约 3 GB，足以决定能不能塞进一张卡。

### 5.3　Paged Optimizer

长序列 / 大 batch 时，优化器状态可能出现瞬时显存尖峰导致 OOM。QLoRA 用 NVIDIA **unified memory** 给优化器状态做**分页**：显存吃紧时把优化器状态分页换出到 CPU 内存，需要时再换回，像操作系统的内存分页一样，避免 OOM 崩溃。

### 5.4　前向 / 反向怎么走

关键：**基座以 NF4 存储，但计算时按 block 反量化到 bf16 再做矩阵乘**；梯度只流向 bf16 的 LoRA 参数，NF4 基座始终冻结、不存梯度。所以 QLoRA 训练显存 ≈ NF4 基座（4 bit/param = 0.5 byte/param，外加双重量化后约 0.127 bit/param 的量化常数）+ bf16 LoRA 的优化器 + 激活。

```python
# QLoRA 前向的概念骨架（非完整 kernel，仅示意数据流）
def qlora_linear_forward(x, nf4_weight, absmax, lora_A, lora_B, scaling):
    # 1) 基座：NF4 -> bf16 反量化，再做主路 matmul（基座无梯度）
    W = dequantize_nf4(nf4_weight, absmax).to(x.dtype)   # [out, in] bf16
    base_out = x @ W.t()
    # 2) 旁路：LoRA 在 bf16 上，可训练
    lora_out = (x @ lora_A.t() @ lora_B.t()) * scaling
    return base_out + lora_out
```

### 5.5　QLoRA 的合并坑

如 §2.4 所述，NF4 基座不能无损合并 bf16 的 $\Delta W$。标准做法：**反量化基座 → 合并 LoRA → 以 fp16/bf16 保存**（得到一个全精度合并模型），或者干脆推理时保持"NF4 基座 + 不合并的 LoRA 旁路"。直接把 $\Delta W$ 量化回 NF4 再加会引入明显误差。

> ⚠️ **QLoRA ≠ 推理省显存** — QLoRA 解决的是**微调时**把大模型塞进单卡。推理阶段如果反量化合并成 fp16，显存又回到全精度；要推理也省显存得用专门的 4-bit 推理路径（如 bitsandbytes / GPTQ / AWQ，见 quantization 篇）。

## §6 DoRA：幅度-方向分解

DoRA（Weight-Decomposed Low-Rank Adaptation，Liu et al., 2024, arXiv 2402.09353, ICML 2024）观察到：满参微调和 LoRA 在"权重的**幅度（magnitude）**与**方向（direction）**如何协同变化"上有系统性差异，LoRA 的表达模式受限。DoRA 把预训练权重显式分解为幅度和方向两部分，分别学习。

把权重按**输出神经元（行）**分解（沿用 weight normalization：每个输出的权重向量 = 幅度 × 单位方向；$\lVert\cdot\rVert_r$ 表示逐行 / 逐输出 2-范数，对 PyTorch 权重 $[\text{out},\text{in}]$ 即沿 in 维 `dim=1`）：

$$W_0 = m \odot \frac{V}{\lVert V \rVert_r}, \qquad m = \lVert W_0 \rVert_r \in \mathbb{R}^{d\times 1}, \quad V = W_0$$

其中 $m$ 是**每个输出神经元一个标量幅度**（$d$ 维），$V/\lVert V\rVert_r$ 是每行单位方向。DoRA 让 **$m$ 可训练**，并用 **LoRA 更新方向**：

$$\boxed{\;W' = m \odot \frac{V + \Delta V}{\lVert V + \Delta V \rVert_r}, \qquad \Delta V = BA\;}$$

（轴向易错：DoRA 沿用 weight normalization，幅度是**逐输出**的——HF PEFT 实现即 `torch.linalg.norm(weight, dim=1)`；DoRA 原文把 $W$ 记作 $\mathbb{R}^{\text{in}\times\text{out}}$ 的逐列范数，等价于此处 PyTorch $[\text{out},\text{in}]$ 的逐行 / per-output。）

直觉：LoRA 把幅度和方向耦合在一个 $\Delta W$ 里一起调；DoRA 把幅度抽出来单独给一个标量序列 $m$，让低秩旁路专注学方向。实证上 DoRA 的"幅度-方向更新模式"更接近满参微调，常在同等参数量下小幅超过 LoRA，尤其在低秩（$r$ 小）时增益明显。代价：逐行归一化带来一点额外计算和实现复杂度；合并时需把 $m$ 和归一化一起折算回 $W'$（合并后仍是单一矩阵、推理零延迟）。

```python
class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # 逐输出幅度：W 形状 [out, in]，沿 in 维(dim=1)求范数 -> [out, 1]（同 HF PEFT）
        self.m = nn.Parameter(self.base.weight.norm(dim=1, keepdim=True))  # [out, 1]

    def forward(self, x):
        dW = self.scaling * (self.lora_B @ self.lora_A)       # [out, in]
        V = self.base.weight + dW
        Vnorm = V.norm(dim=1, keepdim=True) + 1e-8           # [out, 1] 逐输出范数
        W_eff = self.m * V / Vnorm                            # 方向 × 幅度（逐输出）
        return F.linear(x, W_eff, self.base.bias)
```

> 💡 **面试点：DoRA 改了什么** — LoRA 只学 $\Delta W=BA$；DoRA = 可训练幅度 $m$ + 用 LoRA 学方向。它解释了"LoRA 为何在某些任务弱于满参"——因为 LoRA 的幅度/方向更新被绑死，DoRA 解绑了。DoRA 可叠加 QLoRA（**QDoRA**）在量化基座上做分解，兼顾显存与效果。

## §7 LoRA 家族变体一览

| 变体 | 一句话 | 改了 LoRA 的什么 | 论文 |
| --- | --- | --- | --- |
| **LoRA** | 低秩旁路 $BA$ | 基线 | Hu 2021, 2106.09685 |
| **rsLoRA** | 缩放改 $\alpha/\sqrt{r}$ | 高秩梯度稳定 | Kalajdzievski 2023, 2312.03732 |
| **DoRA** | 幅度-方向分解 | 解绑幅度与方向 | Liu 2024, 2402.09353 |
| **PiSSA** | 用 $W_0$ 主成分初始化 $A,B$ | 初始化（更快收敛） | Meng 2024, 2404.02948 |
| **LoRA-GA** | 让首步梯度逼近满参 | 初始化（梯度对齐） | Wang 2024, 2407.05000 |
| **AdaLoRA** | SVD 形式 + 按重要性剪秩 | 自适应分配秩预算 | Zhang 2023, 2303.10512 |
| **LoRA+** | $A,B$ 用不同学习率 | 优化（$B$ 更大 lr） | Hayou 2024, 2402.12354 |
| **(IA)³** | 学缩放向量乘激活 | 非低秩、参数更省 | Liu 2022, 2205.05638 |

几个值得展开的：

- **PiSSA**：不再随机初始化，而是对 $W_0$ 做 SVD，用**主奇异值/向量**初始化 $A,B$（适配"最重要"的子空间），残差 $W_0 - BA$ 冻结。因为一开始就对准主成分，收敛更快、效果常更好。
- **AdaLoRA**：把 $\Delta W$ 参数化成 SVD 形式 $P\Lambda Q$，训练中按奇异值重要性**动态剪枝**，把有限的秩预算分给更需要的层（不同层 $r$ 不同），而非均匀分配。
- **(IA)³**：不走低秩矩阵，而是学三个**缩放向量** $l_k, l_v, l_{ff}$，逐元素乘到 key、value、FFN 隐藏激活上：$k \to l_k \odot k$。参数量比 LoRA 还小一个量级，T-Few 用它做少样本超过 in-context learning。

## §8 与其他 PEFT 范式对比

PEFT 大致分四类：**加性低秩（LoRA 系）**、**加性模块（Adapter）**、**软提示（Prompt/Prefix）**、**选择性（BitFit）**。

| 方法 | 训练什么 | 插在哪 | 推理额外开销 | 典型可训练比例 | 论文 |
| --- | --- | --- | --- | --- | --- |
| **LoRA** | 低秩 $BA$ | 并联到线性层 | **零**（可合并） | 0.1%~1% | Hu 2021, 2106.09685 |
| **Adapter** | 瓶颈 MLP | 串联在子层后 | 有（串行子模块） | 0.5%~5% | Houlsby 2019, 1902.00751 |
| **Prefix-Tuning** | 每层前缀 K/V | 注意力 K/V 前缀 | 有（占 KV / 序列） | <0.1%~1% | Li-Liang 2021, 2101.00190 |
| **Prompt Tuning** | 输入软提示 | 仅 embedding 层 | 有（占序列长度） | 极小 | Lester 2021, 2104.08691 |
| **P-Tuning v2** | 每层软提示 | 各层前缀 | 有 | 0.1%~3% | Liu 2021, 2110.07602 |
| **BitFit** | 仅 bias | 各层 bias 项 | 零（原地） | ~0.08% | Ben-Zaken 2021, 2106.10199 |
| **(IA)³** | 缩放向量 | K/V/FFN 激活 | 有（可部分合并） | <0.05% | Liu 2022, 2205.05638 |

核心对比维度（面试爱问）：

- **推理延迟**：LoRA（合并后）和 BitFit 是真正零开销；(IA)³ 的缩放向量在**固定单 adapter** 时也常可折叠进相邻线性层权重（动态多 adapter 才有在线缩放开销）。Adapter 串行非线性子模块、Prompt/Prefix 占用序列或 KV，都有不可消除的额外推理成本。
- **占用上下文**：Prompt/Prefix 系会吃掉一部分可用序列长度（soft tokens 占位）；LoRA / Adapter / BitFit 不占。
- **表达力 vs 参数**：Prompt Tuning 参数最省但小模型上效果弱（Lester 指出"规模够大才追平满参"）；LoRA 在参数/效果上折中最好，成为事实标准。

> ⚠️ **Adapter 为什么有推理延迟而 LoRA 没有** — Adapter 是**串联**新模块（down→非线性→up），推理时必须额外跑这几层，且打断了原算子融合；LoRA 是**并联**线性旁路，数学上能合并进原权重（线性 + 线性 = 线性），合并后消失。非线性的 Adapter 无法这样吸收。

## §9 工程实践与常见 bug

- **LoRA 学习率要比满参高**：LoRA 参数少、且只调一个低秩子空间，经验上 lr 取 $1\text{e-}4 \sim 3\text{e-}4$（比满参的 $1\text{e-}5\sim2\text{e-}5$ 高一个量级）。LoRA+ 进一步指出 $B$ 应比 $A$ 用更大 lr。
- **target_modules 选择**：只配 `q,v` 省但弱；配全部线性层（含 FFN）通常最稳。配错模块名（不同模型命名不同，如 `c_attn` vs `q_proj`）会导致"LoRA 没挂上去、loss 不降"。
- **$\alpha/r$ 设置**：常见 $\alpha=2r$。换 $r$ 后若忘了 $\alpha$ 随动，等于改了有效学习率。高秩想避免坍缩用 rsLoRA。
- **QLoRA 合并坑**（§5.5）：别把 LoRA 合并回 4-bit 基座，反量化到 fp16 再合并。
- **多 adapter 服务**：不合并、在线加旁路可热插拔多个 adapter；要极致推理速度就为每个任务合并出独立权重。
- **dtype 一致性**：旁路 $A,B$ 与反量化后的基座要在同一精度做加法，混 dtype 会报错或精度异常。
- **保存的是 adapter 不是整模型**：`save` 只存 $A,B$（几十 MB），分发轻量；加载时挂回同一基座。版本要对齐基座，否则 $W_0$ 不一致、$\Delta W$ 无意义。
- **梯度检查点配合**：LoRA 不省激活，长序列训练务必开 gradient checkpointing，否则激活显存照样 OOM。

> ❌ **经典翻车** — "LoRA 把显存降下来了，所以 batch 可以开很大" —— 错。激活显存没省，开大 batch 仍可能 OOM。LoRA 省的是优化器/梯度那部分，不是激活那部分。

## §10 复杂度与资源

| 维度 | 满参微调 | LoRA | QLoRA |
| --- | --- | --- | --- |
| 可训练参数 | $\Psi$ | $\sum_i r_i(d_i+k_i)$（~0.1%~1%） | 同 LoRA |
| 基座存储 | bf16 $2\Psi$ B | bf16 $2\Psi$ B | NF4 4 bit/param $= 0.5\Psi$ B |
| 优化器+梯度显存 | $\approx 14\Psi$ B | $\approx 14\Psi_{\text{lora}}$ B | 同 LoRA |
| 激活显存 | 高 | **同满参**（需 checkpointing） | 同满参 |
| 训练前向计算 | $W_0 x$ | $W_0 x + \frac{\alpha}{r}B(Ax)$（多两次小 matmul） | + 反量化开销 |
| 推理延迟（合并后） | 基线 | **零额外** | 反量化或合并到 fp16 |
| 单卡可训最大模型（80GB·vanilla Adam 量级） | ~3B（7B 满参需 ZeRO / offload / 8-bit optimizer） | ~13B~33B | ~65B |

旁路前向开销：每个适配层多 $2 \cdot 2 L r d$ FLOPs（$Ax$ 与 $B(\cdot)$ 两次小 matmul），因 $r \ll d$，相对主路 $2Ld^2$ 可忽略。合并后旁路彻底消失。

> ⚠️ **"单卡最大模型"只是经验量级** — 真实上限强依赖序列长度、batch、gradient checkpointing、优化器选择（8-bit / offload）、target_modules 与是否 ZeRO 切分，上表仅为 vanilla Adam 假设下的粗略数量级，不是硬上限。

## §11 25 高频面试题

下列题目按难度分三档，点开看答案要点 + 易踩坑。

### L1必会题（任何用过微调的岗位都会问）

<details>

<summary>Q1. LoRA 的核心公式是什么？</summary>

- 冻结 $W_0$，学低秩增量 $\Delta W = BA$
- 前向 $h = W_0 x + \frac{\alpha}{r} BA\,x$
- $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$、$r\ll\min(d,k)$

只说"加了个小矩阵"，说不清 $\Delta W$ 是两个瘦矩阵的积、秩 $\le r$。

</details>

<details>

<summary>Q2. 为什么 $B$ 初始化为 0、$A$ 随机？</summary>

- 让起点 $\Delta W = B A = 0$，不扰动预训练
- 但 $A$ 非零保证梯度不全为零，能学起来
- 两个都置零 → 梯度恒为零 → 永远学不动

以为"$B=0$ 第一步没梯度所以学不了"——错，第一步 $B$ 有梯度（$\partial L/\partial B \propto (Ax)^\top \ne 0$），只是 $\Delta W$ 在 $t=0$ 数值为 0。

</details>

<details>

<summary>Q3. LoRA 相比满参微调省了什么显存？没省什么？</summary>

- 省：基座的优化器状态（Adam 的 $m,v$，8 B）+ 梯度（2 B）+ fp32 master（4 B）≈ **14 B/param**（满参总足迹 ~16 B/param，含 2 B 可训练权重）；基座 bf16 权重 2 B/param 仍常驻
- 没省：**激活显存**（仍要反传过冻结基座）、基座权重本身（仍常驻）
- 省激活要另开 gradient checkpointing

说"LoRA 全面省显存所以能开大 batch"——激活没省，大 batch 仍会 OOM。

</details>

<details>

<summary>Q4. 缩放因子 $\alpha/r$ 是干嘛的？</summary>

- 解耦秩 $r$ 与学习率，换 $r$ 不用重调 lr
- 让不同 $r$ 下 $\Delta W$ 量级可比
- 常设 $\alpha=2r$ 或 $\alpha=r$

不知道 $\alpha$ 是超参，或以为 $\alpha$ 必须等于 $r$。

</details>

<details>

<summary>Q5. LoRA 为什么推理没有额外延迟？</summary>

- 训练后可合并：$W' = W_0 + \frac{\alpha}{r}BA$
- 推理只用 $W'$，与原线性层完全同构
- 线性旁路能被吸收进原权重（线性+线性=线性）

说"LoRA 推理时也要多算 $BA$"——那是没合并的情况；合并后为零。

</details>

<details>

<summary>Q6. 可训练参数量怎么算？给个例子。</summary>

- 单层 $d\times k$：$r(d+k)$
- 全模型：$\sum_i r_i(d_i+k_i)$
- 例：7B 配 $q,k,v,o$（$4096^2$）、$r=8$、32 层 ≈ 8.4M ≈ 0.12%

把可训练量算成 $r\cdot d\cdot k$（那是 $\Delta W$ 的元素数，不是 LoRA 参数数）。

</details>

<details>

<summary>Q7. LoRA 一般加在 Transformer 的哪些层？</summary>

- 原论文：注意力 $W_q, W_v$（同预算下优于只配 $W_q$）
- 实践（QLoRA 起）：加到所有线性层（含 FFN gate/up/down）更稳
- target_modules 命名随模型而异，配错名会"挂不上"

只会说"加在 attention 上"，不知道现代实践常配全部线性层。

</details>

<details>

<summary>Q8. LoRA 的 $r$ 怎么选？越大越好吗？</summary>

- 简单任务 $r=4\sim8$，难任务 $r=16\sim64$
- 不是越大越好：过大易过拟合、省得少、还可能踩 $\alpha/r$ 高秩坍缩
- 高秩想要红利用 rsLoRA（$\alpha/\sqrt r$）

说"$r$ 越大效果越好"，忽略过拟合与缩放坍缩。

</details>

<details>

<summary>Q9. LoRA 的学习率相比满参该大还是小？</summary>

- 通常**大一个量级**（$1\text{e-}4\sim3\text{e-}4$ vs 满参 $1\text{e-}5\sim2\text{e-}5$）
- 因为只调低秩子空间、参数少
- LoRA+ 进一步主张 $B$ 比 $A$ 用更大 lr

直接套满参的小 lr，导致 LoRA 学得太慢。

</details>

<details>

<summary>Q10. 保存 LoRA 微调结果存的是什么？</summary>

- 只存 adapter（$A,B$），几十 MB
- 加载时挂回同一基座
- 基座版本必须对齐，否则 $W_0$ 不符、$\Delta W$ 失效

以为存的是整个微调后大模型。

</details>

### L2进阶题（research / 工程深入岗）

<details>

<summary>Q11. 为什么相信"权重更新是低秩的"？理论依据？</summary>

- Aghajanyan 2020（arXiv 2012.13255）：预训练模型有很低的内在维度，低维子空间优化即可逼近满参
- Hu 2021 顺此假设：$\Delta W$ 低秩，用 $BA$ 参数化
- 是**经验假设 + 实证**，不是严格定理

把"低秩有效"当成已证明的定理，或答不出 intrinsic dimension 的来源。

</details>

<details>

<summary>Q12. 推导 LoRA 第一步对 $A$、$B$ 的梯度，解释初始化。</summary>

- 设 $g=\partial L/\partial h$。$\partial L/\partial B = \frac{\alpha}{r} g (Ax)^\top$，$\partial L/\partial A = \frac{\alpha}{r} B^\top g\, x^\top$
- $t=0$：$B=0$ → $\partial L/\partial A=0$（$A$ 不动）；$A\ne0$ → $\partial L/\partial B\ne0$（$B$ 动）
- 一旦 $B$ 变非零，下一步 $A$ 也有梯度

把两个梯度都算成非零，或说"$B=0$ 所以两个都为零"。

</details>

<details>

<summary>Q13. rsLoRA 解决什么问题？缩放为什么是 $\alpha/\sqrt r$？</summary>

- 问题：$\alpha/r$ 在高秩时过度缩小梯度，学习坍缩
- $BA$ 输出是 $r$ 项之和，标准差 $\propto\sqrt r$；除 $\sqrt r$ 才把量级拉回常数
- 除以 $r$ 过度压制，$r$ 越大压得越狠

只记住"rsLoRA 用 $\sqrt r$"，讲不出方差/量级稳定的原因。

</details>

<details>

<summary>Q14. NF4 是什么？为什么比 INT4 适合量化权重？</summary>

- NF4 = 4-bit NormalFloat，按标准正态分位点放 16 个量化级别，使各 bin 等概率质量
- 权重近似 $\mathcal{N}(0,\sigma^2)$，**正态假设下**等概率 bin 对该固定分布是信息论最优（quantile 量化）
- INT4 等间距，对正态尾部浪费 bin；NF4 还做 zero-preserving

说 NF4 是普通 4-bit 整数，或答不出"等概率质量 / quantile"这个关键。

</details>

<details>

<summary>Q15. QLoRA 的双重量化省了多少？怎么做的？</summary>

- 每 block(64) 的 fp32 absmax 常数本身再量化
- 单次开销 $32/64=0.5$ bit/param → 双重 $\approx0.127$ bit/param
- 平均每参省约 0.37 bit（65B 约省 3GB）

不知道"被二次量化的是缩放常数 absmax"，或把它和 NF4 本身搞混。

</details>

<details>

<summary>Q16. QLoRA 训练时基座是 4-bit，梯度怎么流？</summary>

- 基座 NF4 存储，计算时按 block 反量化到 bf16 再 matmul
- 梯度只流向 bf16 的 LoRA 参数，NF4 基座冻结无梯度
- 所以省的是基座的优化器/梯度，激活仍在

以为在 4-bit 上直接算梯度，或以为基座也在更新。

</details>

<details>

<summary>Q17. DoRA 相比 LoRA 改了什么？为什么可能更好？</summary>

- 把 $W_0$ 分解为幅度 $m=\lVert W_0\rVert_r$（**逐输出行**，沿 in 维 `dim=1`，同 weight norm / HF PEFT）与方向 $V/\lVert V\rVert_r$
- $m$ 可训练，LoRA 只负责更新方向：$W'=m\odot\frac{V+\Delta V}{\lVert V+\Delta V\rVert_r}$
- 解绑幅度与方向，更新模式更接近满参，低秩时增益明显

只说"DoRA 加了个 magnitude"，讲不清是**逐输出**归一 + 方向用 LoRA 学；或把幅度轴搞成逐输入（错，DoRA 是逐输出）。

</details>

<details>

<summary>Q18. Adapter 为什么有推理延迟，LoRA 合并后没有？</summary>

- Adapter 串联非线性瓶颈模块，推理必须额外跑且打断算子融合
- LoRA 并联线性旁路，线性+线性=线性，可吸收进原权重
- 非线性模块无法这样合并

把两者都说成"加了小模块所以都有延迟"，没抓住线性可合并这点。

</details>

<details>

<summary>Q19. Prompt/Prefix Tuning 和 LoRA 的本质区别？</summary>

- Prompt/Prefix：在输入或各层注入可训练 soft tokens / 前缀 K/V，**占用序列长度或 KV**，有推理开销
- LoRA：改权重的低秩旁路，**不占序列**，合并后零开销
- Prompt Tuning 在小模型上偏弱（Lester：规模够大才追平满参）

把 LoRA 也归成"加 token"，或不知道 prompt 系占序列长度。

</details>

<details>

<summary>Q20. PiSSA / LoRA-GA 这类"换初始化"的方法在改什么？</summary>

- PiSSA：用 $W_0$ 的主奇异向量初始化 $A,B$（对准主成分），残差冻结，收敛更快
- LoRA-GA：让 LoRA 首步梯度方向逼近满参梯度（梯度对齐初始化）
- 共同点：不改结构，只改**初始化**以加速收敛 / 提质

把它们当成新结构，或说不出"改初始化"这个共性。

</details>

### L3高级题（顶级 lab / 深水区）

<details>

<summary>Q21. 为什么 LoRA 不自动省激活显存？要省怎么办？</summary>

- 旁路挂在冻结层上，反向仍要穿过整个网络求 $\partial L/\partial x$，中间激活照存
- LoRA 省的是优化器状态 + 梯度 + master 权重，与激活无关
- 省激活要 gradient checkpointing（重算换显存），或序列并行等

笼统说"LoRA 省显存"，被追问"省哪部分"答不上来。

</details>

<details>

<summary>Q22. AdaLoRA 的"自适应秩"怎么实现的？</summary>

- 把 $\Delta W$ 参数化为 SVD 形式 $P\Lambda Q$
- 训练中按奇异值重要性对 $\Lambda$ 剪枝，给不同层分配不同有效秩
- 把有限秩预算动态投到更需要的层，而非均匀分配

以为 AdaLoRA 是"自动选一个全局 $r$"，没说到 SVD 参数化 + 按层剪秩。

</details>

<details>

<summary>Q23. 多任务部署时，一个基座挂多个 LoRA 有哪两种策略？各自代价？</summary>

- 合并法：为每个任务合并出独立全权重，推理零延迟，但每任务一份完整权重、不能动态切换
- 旁路法：基座不动、在线加各自 $\frac{\alpha}{r}B(Ax)$，可热插拔 / batch 内混合不同 adapter，代价是恢复了一点推理开销
- 工业界多用旁路法 + 批处理优化；专门的**多 LoRA serving 系统**（S-LoRA、Punica）能在同一基座上高吞吐并发服务成百上千个 adapter（统一显存分页 + 自定义 batched kernel）

只知道合并，不知道"不合并以支持动态多 adapter"。

</details>

<details>

<summary>Q24. (IA)³ 和 LoRA 的本质区别？为什么 (IA)³ 参数更省？</summary>

- LoRA：加低秩矩阵 $BA$，参数 $r(d+k)$
- (IA)³：学三个**缩放向量** $l_k,l_v,l_{ff}$ 逐元素乘到 K/V/FFN 激活，参数仅 $O(d)$
- (IA)³ 是"逐元素重标定激活"而非"加一个矩阵增量"，所以省一个量级；T-Few 用它少样本超过 ICL

把 (IA)³ 也当成低秩矩阵，没抓住"缩放向量 × 激活"。

</details>

<details>

<summary>Q25. LoRA 微调能学到"新知识"吗？低秩是不是表达瓶颈？</summary>

- 低秩旁路擅长**风格 / 格式 / 任务适配 / 对齐**这类"调整已有能力"的场景
- 注入大量**全新事实知识**时低秩可能受限，需更大 $r$、配更多层，或继续预训练 / 全参
- Biderman et al. (2024, *LoRA Learns Less and Forgets Less*, arXiv 2405.09673) 实证：难任务（如代码 / 数学）上 LoRA 常学得不如满参，但**遗忘也更轻**——低秩约束是一把双刃剑
- 这也是 DoRA / 高秩 / PiSSA 等想突破表达力的动机；秩是容量旋钮，不是免费午餐

绝对化地说"LoRA 和满参等价"或"LoRA 学不了任何新东西"——都过头，取决于 $r$、适配层数与任务类型。

</details>

## §A 附录：sanity check

`LoRALinear` 的关键不变量（可写一段脚本验证）：

- **起点等价**：注入 LoRA 后、训练前，模型输出应与基座逐元素相等（因 $B=0 \Rightarrow \Delta W=0$）。
- **合并一致性**：`merge()` 前后对同一输入的输出应数值一致（fp32 / eval / dropout=0 下误差 ~1e-6 级，纯浮点累加导致；bf16 或 train-mode dropout 下误差更大，不应承诺该数值）。
- **可训练量**：只有 `lora_A`、`lora_B`（以及 DoRA 的 `m`）`requires_grad=True`，基座全 `False`。
- **参数计数**：适配 $q,v$、$r=8$、$d=4096$、$L$ 层时，可训练参数 $= L \times 2 \times 8 \times (4096+4096)$。

下面是 [`code/lora.py`](code/lora.py)（$\text{IN}=32, \text{OUT}=48, r=8, \alpha=16$）在 **PyTorch 2.10 / CPU** 上的**真实运行**输出（每行带 `assert`，全过才打印汇总）：

```
[a] B=0 init: |out_lora - out_base| = 0.00e+00  OK
[b] after 1 step: ||dW|| = 4.768e-02, output changed  OK
[c] merge: |out_merged - out_unmerged| = 3.58e-07; weight += dW = True; unmerge restores base = True  OK
[d] trainable params = 640 (expect r*(in+out) = 640); base frozen = True  OK
[e] scaling: standard alpha/r = 2.0000, rsLoRA alpha/sqrt(r) = 5.6569  OK
[f] DoRA: m.requires_grad = True, base frozen, identity start |Δ| = 1.19e-07  OK

all LoRA / DoRA sanity checks passed ✓
```

其中 $640 = r(\text{IN}+\text{OUT}) = 8\times(32+48)$ 验证了 §4 的可训练参数公式；merge 前后 $3.58\text{e-}7$ 的纯浮点误差验证了零延迟合并的数值等价；DoRA 恒等起点 $\lvert\Delta\rvert=1.19\text{e-}7$ 验证了 $B=0$ 时 $W'=W_0$。

---

## 📜 Runnable Code

本 tutorial 的 LoRA / DoRA 核心实现在 [`docs/tutorials/code/lora.py`](code/lora.py) 有最小可跑版本：

- [`lora.py`](code/lora.py) — `LoRALinear`（$B=0$ 恒等起点 · $\alpha/r$ 与 rsLoRA $\alpha/\sqrt{r}$ 缩放 · merge/unmerge）+ `DoRALinear`（幅度-方向分解），含 6 个 `assert` sanity check（起点等价 / 一步后 $\Delta W \ne 0$ / merge 一致且可逆 / 参数量 $= r(\text{in}+\text{out})$ / rsLoRA 缩放 / DoRA 冻结基座）。

纯 PyTorch，CPU 几秒跑完、无需 GPU：`python docs/tutorials/code/lora.py`。上方 §A 的输出即该脚本的真实运行结果。

---

## 📚 参考文献

- **LoRA** — Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv 2106.09685 (2021), ICLR 2022.
- **Intrinsic Dimension** — Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, arXiv 2012.13255 (2020).
- **QLoRA** — Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv 2305.14314 (2023), NeurIPS 2023.
- **DoRA** — Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, arXiv 2402.09353 (2024), ICML 2024.
- **rsLoRA** — Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*, arXiv 2312.03732 (2023).
- **PiSSA** — Meng et al., *PiSSA: Principal Singular Values and Singular Vectors Adaptation*, arXiv 2404.02948 (2024), NeurIPS 2024.
- **LoRA-GA** — Wang et al., *LoRA-GA: Low-Rank Adaptation with Gradient Approximation*, arXiv 2407.05000 (2024).
- **AdaLoRA** — Zhang et al., *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, arXiv 2303.10512 (2023), ICLR 2023.
- **LoRA+** — Hayou et al., *LoRA+: Efficient Low Rank Adaptation of Large Models*, arXiv 2402.12354 (2024).
- **(IA)³ / T-Few** — Liu et al., *Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper than In-Context Learning*, arXiv 2205.05638 (2022).
- **Adapter** — Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, arXiv 1902.00751 (2019), ICML 2019.
- **Prefix-Tuning** — Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, arXiv 2101.00190 (2021), ACL 2021.
- **Prompt Tuning** — Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning*, arXiv 2104.08691 (2021), EMNLP 2021.
- **P-Tuning v2** — Liu et al., *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks*, arXiv 2110.07602 (2021).
- **BitFit** — Ben-Zaken et al., *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models*, arXiv 2106.10199 (2021), ACL 2022.
- **S-LoRA** — Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, arXiv 2311.03285 (2023).
- **LoRA Learns Less and Forgets Less** — Biderman et al., arXiv 2405.09673 (2024), TMLR 2024.
