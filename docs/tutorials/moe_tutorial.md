## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 MoE** — 一页拿下 2026 秋招核心要点（详见后文 §1–§9 推导）。

1. **核心思想**：把单个 FFN 换成 $N$ 个 expert + 一个 router，每个 token 只走 $k \ll N$ 个 expert，**总参数 ↑、激活参数不变**（sparse activation）。计算量约等于 $k/N$ × dense，但内存 / 显存按总参数计。

2. **路由公式**（Token-Choice top-k）：$g_i(x) = \text{softmax}(W_g x)_i$，选 $\mathcal{T}_k(x) = \text{TopK}_i\, g_i(x)$，输出 $y = \sum_{i \in \mathcal{T}_k(x)} g_i(x) \cdot E_i(x)$。Gate 概率作为 **soft 权重**乘到 expert 输出上，反传时让 router 可微。

3. **历史脉络（必背 5 篇）**：Shazeer 2017（首个深度学习里可工作的 MoE 层）→ GShard 2020（top-2 + capacity factor + all-to-all）→ Switch Transformer 2021（top-1 极简，aux loss + load balance）→ Mixtral 8x7B/8x22B 2024（首个开源主流 MoE，top-2）→ DeepSeek-V3 2024（671B/37B，**aux-loss-free**，fine-grained + 1 shared）。

4. **DeepSeek 路线（2026 面试热点）**：DeepSeekMoE 2024 提出 **fine-grained experts**（拆细，$mN$ 个小 expert 选 $mK$ 个）+ **shared experts**（少数 expert 给所有 token，吸收通用知识）。V2 用 MLA + DeepSeekMoE，V3 把路由专家做到 256 + 1 shared，**取消 aux loss**，改用 expert bias 在线更新。

5. **Aux-loss-free balance**：在 router score 上加每 expert 的偏置 $b_i$，**只用于 top-k 选择**，不进梯度（也不进最终 gate 权重）；每个 step 按"实际 load - 期望 load"反向更新 $b_i$（多负载 expert 减偏置）。**不破坏 sparse gradient，不引入干扰梯度**。

6. **容量与丢 token**：每个 expert 的容量 $C = \lceil \alpha \cdot T \cdot k / N \rceil$（$\alpha$ 是 capacity factor，常用 1.0-1.25）。超过容量的 token 走 **residual bypass**（直接跳过 expert，原始残差通过），或被 drop。Switch Transformer 论文里这是核心工程细节。

7. **并行**：MoE 几乎必须用 **Expert Parallelism (EP)**——把不同 expert 放到不同 GPU，token 路由后做 **all-to-all**（dispatch）→ expert 算 → all-to-all（combine）。DeepEP 与 DualPipe 是 DeepSeek-V3 在 H800 上把 EP 通信和计算 overlap 的两大工程武器。

8. **常见 bug**：①routing collapse（某些 expert 被全部 token 选、其他 expert 饿死）；②fp16 下 router logit overflow；③推理时 **all experts 必须全部加载到显存**——只是激活参数少，**显存依然按总参数算**；④小 batch 时 EP 通信占满，吞吐反而比 dense 差。

## §1 直觉：稀疏激活 = 用"参数容量"换"计算量"

Dense FFN 长这样：每个 token 都过同一个 $D \times 4D \to 4D \times D$ 大矩阵，参数全部参与计算。

MoE 把 FFN 换成 $N$ 个独立 expert（每个本质仍是 FFN），但每个 token 由 router 决定**只走 $k$ 个**：

```

                    Token x  [B, L, D]
                          │
                          ↓
                  Router W_g ∈ R^{D × N}        (gate)
                          │
                          ↓
                  scores  g(x) ∈ R^N
                          │
                  TopK + softmax-renormalize
                          │
              ┌─────┬─────┼─────┬─────┐
              ↓     ↓     ↓     ↓     ↓
            E_1   E_2   ...   E_{N-1}  E_N    (每个 expert = FFN)
              │     │           │     │
              └─ × g_1 ─ ... ─ × g_N ─┘        (只有 k 个 g_i ≠ 0)
                          │
                          ↓
                  Sum → output y [B, L, D]
```

为什么这是好主意？三个层次的回答：

- **容量论**：模型容量主要来自参数量，泛化主要来自数据 × 容量。Sparse activation 允许把总参数推到上千 B，而每 token 计算仍像 30B dense——**参数容量 ≫ 计算量**。
- **专精论**：不同 expert 在训练中天然分工——code / math / 多语言 / 通用语义。Switch Transformer 给过 t-SNE 可视化，DeepSeekMoE 用 fine-grained 进一步加剧这种专精。
- **推理效率**：固定 active 参数 $\approx$ 固定 FLOPs，可与 dense 同台比，但参数容量大很多——这是 Mixtral 8x7B 用 13B active 干过 LLaMA-2 70B 的根本原因。

> ⚠️ **MoE 不是免费午餐** — sparse activation 减的是 **FLOPs**，**不减显存**（推理时所有 expert 必须常驻 GPU）；EP 训练时 all-to-all 通信能轻易吃掉一半时间；负载不均会让多数 GPU 空转。下面 §3-§7 是这些工程现实的全部展开。

## §2 路由：从 Top-k Token-Choice 到 Expert Choice 到 Loss-Free

### 2.1　Token-Choice Top-k（GShard / Switch / Mixtral / DeepSeek）

最经典也最常考。Gate 计算每个 (token, expert) 对的 affinity，token 选 top-k expert：

$$\boxed{\;s(x) = W_g x \in \mathbb{R}^N, \quad g(x) = \text{softmax}(s(x)), \quad \mathcal{T}_k(x) = \text{TopK}_i\, g_i(x)\;}$$

输出（routing weight 重新归一化版本，Mixtral / DeepSeek 默认这种）：

$$y = \sum_{i \in \mathcal{T}_k(x)} \tilde{g}_i(x) \cdot E_i(x), \quad \tilde{g}_i(x) = \frac{g_i(x)}{\sum_{j \in \mathcal{T}_k(x)} g_j(x)}$$

> 💡 **重归一 vs 直用 softmax 概率** — 两种工程实现都见过：Switch 直接用 $g_i$（top-1 时 $g_i$ 直接乘）；Mixtral / DeepSeek 在 top-k 上再 renormalize 让权重和 = 1。renormalize 让输出尺度稳定，不依赖 softmax 在 top-k 之外的"漏出"概率质量。

**几个变体的 $k$**：

| 模型 | $k$ | 备注 |
| --- | :-: | --- |
| Switch Transformer | 1 | 极简，被证明也能 work |
| GShard / Mixtral / Qwen3-MoE | 2 | 主流；2 个 expert 足够"集成" |
| DeepSeek-V2 / V3 | 6 / 8（routed）+ 1 shared | 配合 fine-grained 拆细 |
| Llama 4 Scout | 1（仅 1 个 routed）+ 1 shared | 16 experts，1 shared |
| Llama 4 Maverick | 1 routed + 1 shared | 128 experts |

### 2.2　Auxiliary Load Balancing Loss（Switch 公式，必背）

问题：纯 top-k 没有任何机制让 router **均衡使用**所有 expert——很容易 collapse 成"只用 1-2 个 expert"。

Switch Transformer 引入 differentiable load balancing loss：

$$\boxed{\;\mathcal{L}_\text{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i\;}$$

其中

- $f_i = \dfrac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{\arg\max_j s_j(x_t) = i\}$ ——分配给 expert $i$ 的 token 比例（不可微，sample-level 估计）
- $P_i = \dfrac{1}{T} \sum_{t=1}^{T} g_i(x_t)$ ——expert $i$ 的平均 gate 概率（可微，softmax 输出）
- $\alpha$ 是 loss 权重（常用 $10^{-2}$）
- $N$ 是 expert 数量；$T$ 是 batch 中 token 数

**关键点 / 易混淆**：

1. $f_i$ 给"实际频率"，$P_i$ 给"梯度通路"。乘起来 $\sum_i f_i P_i$ 当两者都集中在同一组 expert 时最大；均匀分布时最小（$= 1/N$，乘以 $N$ 后 = 1）。
2. 这是 **encourage 均匀分布**，不是硬约束。极端 collapse 会被惩罚，但日常小幅不均不会被强力惩罚。
3. Switch top-1 公式直接如此；top-k 时把 $f_i$ 改成"top-k 命中频率"（更准确写法见 GShard 论文）。
4. **$\alpha$ 太大会干扰主任务梯度**——这是 DeepSeek 改 aux-loss-free 的根本动机。

### 2.3　Expert Choice Routing（Zhou et al. 2022，NeurIPS）

反向思维：不让 token 挑 expert，让 **expert 挑 token**。每个 expert 根据 capacity 选 top-$M$ 个 token，$M = (T \cdot k) / N$（每个 expert 平均分到 token 数）。

$$s(x) \in \mathbb{R}^{T \times N}, \quad \mathcal{T}_M^{(i)} = \text{TopM}_{t}\, s_{t,i}, \quad y_t = \sum_{i: t \in \mathcal{T}_M^{(i)}} g_{t,i} \cdot E_i(x_t)$$

**优点**：

- **天然平衡**：每个 expert 严格选 $M$ 个 token，不需要 aux loss
- **不丢 token**：理论上不会有"溢出"（capacity 是 hard 给定）

**缺点 / 限制**：

- **不是 causal**：第 $t$ 个 token 的归属依赖 $T$ 个 token 全部的 score——decoder 推理时**无法 token-by-token 生成**，需要 batch-level 全局视角（autoregressive 不友好）
- 每个 token 实际激活的 expert 数 **不固定**（可能没被选 / 被多个选）
- 主要在 encoder（如 T5、BERT-style）或 vision 用

> ⚠️ **面试陷阱** — 被问"Expert Choice 解决了 routing collapse，那 Mixtral / DeepSeek 为什么还用 token-choice？" 答案：autoregressive decoder + 每 token 严格走固定 $k$ expert 的需求让 expert-choice 用不了；DeepSeek-V3 改用 aux-loss-free 在 **token-choice 框架内**做平衡。

### 2.4　Auxiliary-Loss-Free Balance（DeepSeek-V3 招牌，必考）

DeepSeek（Wang et al., arXiv 2408.15664, 2024）提出的方案，被 V3 全面采纳。核心一句话：**给每个 expert 加一个偏置项 $b_i$，只用于 top-k 选择，不进梯度**。

具体地，原 score 是 $s_i(x)$，做 top-k 选择时用 **biased score**：

$$\tilde{s}_i(x) = s_i(x) + b_i$$

$$\mathcal{T}_k(x) = \text{TopK}_i\, \tilde{s}_i(x)$$

但**最终 gating weight** 仍用原始 $s_i$ 的 softmax（或 sigmoid 后 renormalize，DeepSeek-V3 用 sigmoid）：

$$g_i(x) = \frac{\sigma(s_i(x))}{\sum_{j \in \mathcal{T}_k(x)} \sigma(s_j(x))}, \quad i \in \mathcal{T}_k(x)$$

$b_i$ 的更新规则（每 step 一次，**out-of-graph**，非梯度更新）：

$$b_i \leftarrow b_i + u \cdot \text{sign}(\bar{c}_i - c_i)$$

- $c_i$ = 本 step 中 expert $i$ 实际接到的 token 数
- $\bar{c}_i = T k / N$ = 均匀分布下每个 expert 应接到的期望 token 数
- 若 $c_i > \bar{c}_i$（过载）→ $b_i$ 减小 → 下次更难被选
- 若 $c_i < \bar{c}_i$（欠载）→ $b_i$ 增大 → 下次更易被选
- $u$ 是固定步长（V3 报告里很小，量级 $10^{-3}$）

> ✅ **为什么这是非平凡的胜利** — Aux loss 的问题是它把"平衡"作为 **梯度信号** 注入 router → 干扰主任务梯度，模型为了平衡可能牺牲质量。Loss-free 把平衡作为 **离线控制信号**（PID 风味的 sign update），完全不进入计算图——router 仍然只为"语言建模 loss"求梯度，但选 expert 时被 $b_i$ 推着走。**不破坏 sparse routing 的语义、不污染主梯度、不引入需要 tune 的 $\alpha$**。

### 2.5　Noisy Top-k（Shazeer 2017 原始方案，已较少用）

Shazeer 等 2017 论文里在 score 上加可学习的 Gaussian noise：

$$s_i(x) = (W_g x)_i + \text{StandardNormal}() \cdot \text{Softplus}((W_\text{noise} x)_i)$$

噪声为了让 top-k 选择"软化"、避免 collapse。后被 GShard / Switch 用更明确的 load balance loss 取代——但概念（"top-k 不可微 → 需要某种 stochastic 软化"）仍有教学价值。

## §3 实现细节：核心 60 行 PyTorch

最小可跑实现：token-choice top-k MoE 层 + Switch aux loss。不考虑 EP，单 GPU 教学用。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """每个 expert 就是一个标准 FFN（SwiGLU 风格也可，这里用 GELU 简化）"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class MoELayer(nn.Module):
    """
    Token-choice top-k MoE layer.
    输入 x: [B, L, D]
    输出 y: [B, L, D], aux_loss: scalar (Switch load-balance loss)
    """
    def __init__(self, d_model, d_ff, num_experts, top_k=2, capacity_factor=1.25, aux_loss_coef=0.01):
        super().__init__()
        assert top_k <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef

        # Router: D -> N, no bias (Switch / Mixtral 都不用 bias)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # N 个 expert
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        B, L, D = x.shape
        T = B * L
        x_flat = x.view(T, D)                              # [T, D]

        # ----- 1. Router 计算 score + top-k -----
        logits = self.router(x_flat)                       # [T, N]
        probs = F.softmax(logits, dim=-1)                  # [T, N]
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # [T, k], [T, k]
        # Mixtral / DeepSeek 风格：在 top-k 上重归一
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)  # [T, k]

        # ----- 2. Aux load-balance loss (Switch 公式) -----
        # f_i = 分给 expert i 的 token 比例 (每 token 算 top_k 次命中, 与 top_k 一致)
        with torch.no_grad():
            one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()  # [T, k, N]
            f = one_hot.sum(dim=(0, 1)) / (T * self.top_k)                       # [N]
        P = probs.mean(dim=0)                                                    # [N], 可微
        aux_loss = self.aux_loss_coef * self.num_experts * (f * P).sum()

        # ----- 3. Capacity 计算 + token dispatch -----
        # 与正文公式一致：ceil(capacity_factor * T * top_k / N_E)，避免小 batch 截断为 0
        import math as _math
        capacity = max(1, _math.ceil(self.capacity_factor * T * self.top_k / self.num_experts))
        # 把 [T, k] 展开成 [T*k] 的 (token_idx, expert_idx) pair
        flat_expert = top_idx.view(-1)                     # [T*k]
        flat_weight = top_probs.view(-1)                   # [T*k]
        flat_token = torch.arange(T, device=x.device).repeat_interleave(self.top_k)  # [T*k]

        # ----- 4. Expert forward (capacity-aware) -----
        y_flat = torch.zeros_like(x_flat)                  # [T, D]
        for e in range(self.num_experts):
            mask_e = (flat_expert == e)                    # [T*k]
            if mask_e.sum() == 0:
                continue
            tok_e = flat_token[mask_e]                     # 要给 expert e 处理的 token 全局 idx
            w_e   = flat_weight[mask_e]                    # 对应 gate weight
            # 容量截断：超过 capacity 的 token drop (Switch residual bypass)
            if tok_e.numel() > capacity:
                tok_e = tok_e[:capacity]
                w_e   = w_e[:capacity]
            inp_e = x_flat[tok_e]                          # [≤cap, D]
            out_e = self.experts[e](inp_e)                 # [≤cap, D]
            # 按 token idx scatter-add (一个 token 可能被多个 expert 命中,所以是 add)
            y_flat.index_add_(0, tok_e, out_e * w_e.unsqueeze(-1))

        return y_flat.view(B, L, D), aux_loss
```

> ⚠️ **生产实现关键差异** — 教学版用 Python `for e in range(num_experts)` 串行调每个 expert，**生产 (Megatron / vLLM) 都是 grouped GEMM 或 fused kernel**，把所有 expert 的计算 batch 到一起，否则 N 大时 GPU 大部分时间在 launch kernel。MoE 真正难的是这一层。

### 3.1　Aux-Loss-Free Bias Update（DeepSeek-V3 风格）

把上面 §3 中的 aux loss 删掉，改用 expert bias 更新：

```python
class MoEAuxFree(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2, bias_step=1e-3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_step = bias_step
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # b_i: 非参数, 不进梯度图 (buffer)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        B, L, D = x.shape
        T = B * L
        x_flat = x.view(T, D)

        # Router score (V3 用 sigmoid 不是 softmax, 但 top-k 逻辑一样)
        logits = self.router(x_flat)                       # [T, N]
        # ★ Top-k 用 biased score, 但 gating weight 用原始 score
        biased_logits = logits + self.expert_bias.unsqueeze(0)
        _, top_idx = biased_logits.topk(self.top_k, dim=-1)         # [T, k]
        # 取出原始 score 对应位置, sigmoid 后 renormalize
        gate_raw = torch.sigmoid(logits).gather(-1, top_idx)        # [T, k]
        top_weights = gate_raw / (gate_raw.sum(dim=-1, keepdim=True) + 1e-9)

        # ----- 在线更新 expert_bias (无梯度, 论文 sign 更新) -----
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()  # [T, k, N]
                c = one_hot.sum(dim=(0, 1))                          # [N], 实际 load
                c_bar = T * self.top_k / self.num_experts            # 期望 load (scalar)
                # 过载 -> 减偏置; 欠载 -> 加偏置
                self.expert_bias -= self.bias_step * torch.sign(c - c_bar)

        # Dispatch + expert forward (与 §3 同, 这里省略)
        y_flat = torch.zeros_like(x_flat)
        flat_expert = top_idx.view(-1)
        flat_weight = top_weights.view(-1)
        flat_token  = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        for e in range(self.num_experts):
            mask = (flat_expert == e)
            if mask.sum() == 0: continue
            tok_e = flat_token[mask]
            w_e   = flat_weight[mask]
            out_e = self.experts[e](x_flat[tok_e]) * w_e.unsqueeze(-1)
            y_flat.index_add_(0, tok_e, out_e)
        return y_flat.view(B, L, D)
```

注意：

1. `expert_bias` 是 buffer 不是 parameter，**不进梯度图、不被 optimizer step**——只被 §forward 末尾的"sign 更新"修改。
2. **biased_logits 只用于选 top-k，gating weight 来自原始 logits**（这是 V3 论文反复强调的点）。如果连 gating weight 都用 biased，相当于把控制信号污染到主梯度路径，aux-loss-free 的好处就丢了。
3. V3 实际是 sigmoid + 256 experts + 9 选（8 routed + 1 shared 在 §6 讲），这里简化成 softmax + top-k 演示原理。

### 3.2　Fine-Grained + Shared Expert Layer（DeepSeekMoE 风格）

```python
class DeepSeekMoELayer(nn.Module):
    """
    Fine-grained + shared expert: 
      - 总共 N 个 routed expert (拆细, 每个比标准 expert 小 m 倍)
      - 加 N_shared 个 shared expert (所有 token 都走)
      - 每个 token 路由到 K 个 routed expert (K = m × baseline_top_k)
    论文 (Dai et al. 2024, arXiv 2401.06066): 把 N 个原始 expert 拆成 mN 个,
    每个 expert 的 d_ff 缩到 d_ff/m, 路由数从 K 提到 mK, 总计算 / 参数量不变,
    但组合数 C(mN, mK) >> C(N, K) -> 专业化粒度提升.
    """
    def __init__(self, d_model, d_ff_per_expert, num_routed_experts, num_shared_experts, top_k):
        super().__init__()
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)
        # 拆细后的 routed experts (每个比 baseline 小 m 倍)
        self.routed_experts = nn.ModuleList(
            [Expert(d_model, d_ff_per_expert) for _ in range(num_routed_experts)]
        )
        # Shared experts: 永远激活, 不进 router
        self.shared_experts = nn.ModuleList(
            [Expert(d_model, d_ff_per_expert) for _ in range(num_shared_experts)]
        )

    def forward(self, x):
        # 1. Shared expert: 直接累加, 不路由
        shared_out = sum(e(x) for e in self.shared_experts)

        # 2. Routed top-k (这里简化省略 aux loss / capacity)
        B, L, D = x.shape
        T = B * L
        x_flat = x.view(T, D)
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + 1e-9)

        routed_out = torch.zeros_like(x_flat)
        flat_e = top_i.view(-1)
        flat_w = top_p.view(-1)
        flat_t = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        for e in range(self.num_routed):
            mask = (flat_e == e)
            if mask.sum() == 0: continue
            tok = flat_t[mask]
            w   = flat_w[mask]
            routed_out.index_add_(0, tok, self.routed_experts[e](x_flat[tok]) * w.unsqueeze(-1))

        return shared_out + routed_out.view(B, L, D)
```

> 💡 **Fine-grained 的"组合爆炸"直觉** — 标准 8 选 2 → 28 种组合。拆细成 64 选 16（每个 expert 缩到 1/8，路由数 ×8），组合 $\binom{64}{16} \approx 4.9 \times 10^{14}$。同样的参数/算力预算下，**每个 token 可表达的"专家组合身份"指数级丰富**，是 DeepSeekMoE 的核心论点之一。

## §4 容量、Token Dropping、Routing Collapse

### 4.1　Capacity Factor（必背工程细节）

每个 expert 不可能接受任意多 token——GPU memory + EP 通信都需要静态 buffer。所以预先给每个 expert 设容量 $C$：

$$\boxed{\;C = \left\lceil \alpha \cdot \frac{T \cdot k}{N} \right\rceil\;}$$

- $T$：batch 中 token 总数
- $k$：每 token 走的 expert 数
- $N$：expert 总数
- $\alpha$：capacity factor（常用 1.0、1.25、2.0）

**$\alpha = 1$**：均匀分布下刚好够用，但稍微不均就 drop；**$\alpha = 1.25$**（Switch / GShard 默认）：留 25% 余量缓冲不均匀；**$\alpha > 2$**：基本不 drop 但浪费显存。

### 4.2　Token Dropping vs Residual Bypass

当某 expert 已收满 $C$ 个 token，新来的 token 怎么办？**两条路线**：

| 方案 | 行为 | 谁用 |
| --- | --- | --- |
| **Token Drop** | 直接丢，输出 0 | GShard 早期；通常 ↓ 训练稳定性 |
| **Residual Bypass** | Expert 输出置 0，但 **residual $x$ 自然通过 layer norm + skip connection 透传** | Switch Transformer / Mixtral / DeepSeek 默认 |

Residual bypass 的好处：被 drop 的 token **不是变 0**——它在残差通路上仍然带着自己的表示，只是这层"没被任何 expert 加工"，相当于退化成 identity layer。整个网络仍然能学到东西。

> ⚠️ **必考易踩坑** — 面试常问"MoE 为什么不 catastrophic 地坏掉"。答案：residual bypass + 多层堆叠下，**单层 drop 不致命**——下一层 router 仍然能为该 token 选到合适 expert。

### 4.3　Routing Collapse（💣 经典 bug）

如果 router 没有 balance 机制，训练初期某些 expert 偶然被多选 → 它们参数被更新更多 → 下次更可能被选 → 强者愈强 → **少数 expert 通吃，多数饿死**。这就是 routing collapse。

诊断信号：

- 监控 expert load（每 step 每 expert 接到的 token 数）→ 极不均
- 监控 aux loss → 大或在涨
- Validation loss 平台期早出现

修复手段：

1. **Aux loss**（Switch 公式，§2.2）
2. **Expert dropout** / **Z-loss**（对 router logits 加 entropy regularizer）
3. **Expert Choice**（§2.3，硬约束每 expert 选固定数 token）
4. **Aux-loss-free bias**（§2.4，DeepSeek-V3）
5. **Capacity factor 适度调大**（让 collapse 触发 drop，间接惩罚集中）

### 4.4　Top-k 的"硬"选择如何反传梯度？

经典问题：$\text{TopK}$ 是不可微的离散操作。

答案：**top-k 只决定走哪条路径（discrete），不参与梯度**；softmax 权重 $g_i(x)$ 进入最终输出 $y = \sum_i g_i E_i(x)$，对 $g_i$ 求导是连续的（standard softmax 链式法则）。所以 router 的 $W_g$ 是可微的（注意 softmax / renorm 的耦合带来 **完整 Jacobian**，不只是对角项）：

记 $s = W_g x$（logits），$p = \text{softmax}(s)$，top-k 选 $\mathcal{T}_k(x)$ 后做 renormalize $g_i = p_i / Z$（$Z = \sum_{j \in \mathcal{T}_k} p_j$）：

$$\frac{\partial g_i}{\partial s_l} = \mathbf{1}[i \in \mathcal{T}_k]\cdot\left(\frac{\partial p_i / \partial s_l}{Z} - \frac{p_i}{Z^2}\cdot \mathbf{1}[l \in \mathcal{T}_k] \cdot \frac{\partial Z}{\partial s_l}\right)$$

其中 $\partial p_i/\partial s_l = p_i(\delta_{il} - p_l)$ 是 softmax Jacobian（含交叉项 $-p_i p_l$，不只是对角 $p_i(1-p_i)$）。汇总：

$$\frac{\partial \mathcal{L}}{\partial W_g} = \sum_{i \in \mathcal{T}_k(x)}\sum_{l} \frac{\partial \mathcal{L}}{\partial g_i} \cdot \frac{\partial g_i}{\partial s_l} \cdot \frac{\partial s_l}{\partial W_g}$$

对 $l \notin \mathcal{T}_k$（被 top-k 排除），$\partial g_i/\partial s_l$ 通过 $p_l$ 仍非零（softmax 的全局耦合），但这一项在 forward 不影响输出、在 backward 的贡献仅经 $\partial p_i / \partial s_l = -p_i p_l$ 间接进入。**实操上**，只有 top-k 内的 expert 才能拿到主导梯度信号；top-k 外的 expert 长期得不到 "我应该被选" 的训练 push——这是 chicken-and-egg：不被选 → 不更新 → 永远不被选。Aux loss / bias 就是为了打破这个循环。

## §5 经典模型谱系：从 Shazeer 到 DeepSeek-V3

按时间和"工程门槛"排成 7 行（必背年表 + 关键贡献）：

| 年份 | 论文 / 模型 | 核心贡献 | arXiv |
| --- | --- | --- | --- |
| 2017 | **Shazeer et al., Outrageously Large NN** | 首个深度 MoE 层（LSTM 旁挂），noisy top-k, load-balance loss 雏形 | 1701.06538 |
| 2020 | **GShard (Lepikhin et al.)** | Top-2 routing, capacity factor, **automatic sharding / all-to-all**，600B 多语言 NMT | 2006.16668 |
| 2021 | **Switch Transformer (Fedus, Zoph, Shazeer)** | Top-1 极简，aux loss 标准公式，1.6T 参数 T5-MoE | 2101.03961 |
| 2022 | **Expert Choice (Zhou et al., NeurIPS)** | Expert-choice routing，硬平衡，无需 aux loss（但 autoregressive 不友好） | 2202.09368 |
| 2024.1 | **DeepSeekMoE (Dai et al.)** | **Fine-grained experts + shared experts**，16B / 145B 验证 | 2401.06066 |
| 2024.1 | **Mixtral 8x7B (Jiang et al.)** | 首个开源主流 MoE，8 experts top-2，13B active 干过 LLaMA-2 70B | 2401.04088 |
| 2024.4 | **Mixtral 8x22B** | 141B 总 / 39B 激活，64k context | (Mistral blog) |
| 2024.5 | **DeepSeek-V2 (DeepSeek-AI)** | 236B / 21B，**MLA + DeepSeekMoE**，KV cache 减 93.3% | 2405.04434 |
| 2024.8 | **Loss-Free Balance (Wang et al.)** | Aux-loss-free 偏置更新方案 | 2408.15664 |
| 2024.12 | **DeepSeek-V3** | **671B / 37B**，256 routed + 1 shared expert，aux-loss-free, MTP, DualPipe | 2412.19437 |
| 2025.4 | **Llama 4 (Meta)** | Scout (17B/16E)、Maverick (17B/128E)、Behemoth (288B/16E) | (Meta blog) |
| 2025.5 | **Qwen3 MoE** | Qwen3-235B-A22B（22B 激活） | 2505.09388 |

### 5.1　主流开源 MoE 横向对比（2026 面试常考）

| 模型 | 总参数 | 激活 / token | Routed experts | Top-k | Shared expert | Aux 方案 | Released |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 | 0 | aux loss | 2023.12 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 | 0 | aux loss | 2024.4 |
| Qwen2-57B-A14B | 57B | 14B | 64 | 8 | 1 | aux loss | 2024.6 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | 2 | aux loss + device-level | 2024.5 |
| **DeepSeek-V3** | **671B** | **37B** | **256** | **8** | **1** | **Aux-loss-free + sequence-aux** | 2024.12 |
| Llama 4 Scout | 109B | 17B | 16 | 1 | 1 | undisclosed | 2025.4 |
| Llama 4 Maverick | 400B | 17B | 128 | 1 | 1 | undisclosed | 2025.4 |
| Qwen3-235B-A22B | 235B | 22B | 128 | 8 | 0 | (loss-free 变体) | 2025.5 |

> 💡 **读这张表的姿势** — 注意 "总参 / 激活" 比例（Mixtral $\approx 4\times$，V3 $\approx 18\times$，Maverick $\approx 24\times$）。这个比例越高，"用稀疏换容量"的杠杆越大，但也越依赖 routing 质量。V3 把这个比值推到极致并 still work，靠的就是 fine-grained + shared + loss-free。

## §6 DeepSeek-V3 全景（必背重点）

DeepSeek-V3（DeepSeek-AI, arXiv 2412.19437, 2024.12）是 2026 面试季最高频的 MoE 系统。把它的设计点拆开：

### 6.1　架构层面

```
                  DeepSeek-V3 MoE Layer
                 ────────────────────────
       x  [B, L, D=7168]
         │
         ├──────► Shared Expert (1 个, d_ff ≈ 2048)   →  shared_out
         │         (永远激活, 不进 router)
         │
         └──────► Router (D → 256)
                   │
                   ├ score s_i(x) + b_i  (bias 仅用于 top-k 选择)
                   │
                   ↓
                Top-8 (256 选 8)
                   │
                   │  实际 gating weight: sigmoid(s_i) / Σ sigmoid(s_j)
                   ↓
        ┌────────┬────────┬────────┬────────┐
        ↓        ↓        ↓        ↓        ↓
      E_a      E_b      E_c     ...      E_h         (8 个 routed expert)
        │        │        │        │        │
        └─ × g ─┴─ × g ─┴─ × g ─┴─ × g ─┘
                          │
                          ↓
                routed_out  +  shared_out  →  y
```

关键数字：

- **Total** 671B parameters
- **Active per token** 37B parameters
- **MoE layer**：256 routed experts + 1 shared，每 token 选 **9 个 expert**（8 routed + 1 shared）
- **Attention** 用 **MLA**（Multi-head Latent Attention），把 KV 压缩到低秩 latent → KV cache 降 ~93%
- **MTP**（Multi-Token Prediction）作为辅助 objective，inference 时可做 speculative decode

### 6.2　Auxiliary-Loss-Free Balance（详见 §2.4）

V3 论文里写了两件事的组合：

1. **Per-expert bias** $b_i$（核心）—— §2.4 描述
2. **Sequence-wise auxiliary loss**（兜底）—— 极小权重的辅助 loss，用于防止单条序列内部极端不均（不是跨 batch 的传统 aux loss）

第二点经常被忽略：V3 不是"完全无 aux loss"，而是**主要靠 bias 做平衡，留一个 sequence-level 的极小 aux 兜底**。面试时如果说"V3 完全没有 aux loss"是不准确的。

### 6.3　Node-Limited Routing

V3 训练时跑 64-way EP 跨 8 个 node。Naïve top-8 routing 可能让一个 token 的 8 个 expert 散布到所有 8 个 node → 严重 all-to-all 通信。V3 加了硬约束：**每个 token 最多路由到 4 个 node**，且每 node 最多选 3 个 expert。这是 algorithm × system 联合优化。

### 6.4　System：DualPipe + DeepEP

- **DualPipe**（github: deepseek-ai/DualPipe）：双向 pipeline parallelism，把 forward 和 backward 的 compute / comm 重叠到几乎 0 bubble。
- **DeepEP**（github: deepseek-ai/DeepEP）：专门为 MoE all-to-all 优化的通信库，支持 FP8 dispatch + asymmetric inter/intra-node bandwidth。

这两个开源库是 V3 在 2048 张 H800 上把 MoE 训练 cost 压到 $5.6M USD 的关键工程基础。

## §7 复杂度、显存、推理算账

### 7.1　训练时 FLOPs / 参数对比

设 dense baseline 用 $D$ hidden, $d_\text{ff} = 4D$, 层数 $L$, token 数 $T$。FFN 部分 FLOPs（dense）：

$$\text{FLOPs}_\text{FFN}^\text{dense} \approx 2 \cdot T \cdot L \cdot D \cdot d_\text{ff} \cdot 2 = 16 T L D^2$$

MoE：把 FFN 换成 $N$ expert, 每 expert $d_\text{ff}$, 每 token 走 $k$。**计算量只算被激活的 expert**：

$$\text{FLOPs}_\text{FFN}^\text{MoE} \approx 16 T L D^2 \cdot \frac{k}{N} \cdot \frac{N \cdot d_\text{ff}^\text{expert}}{4D}$$

如果 $d_\text{ff}^\text{expert} = 4D$（"每个 expert 等于一个完整 FFN"，Mixtral 风格），$\text{FLOPs} \propto k$，**与 $N$ 无关**（典型 Mixtral 8x7B $k=2$ → 计算 $\approx$ 2 个 FFN）。

参数量则与 $N$ 线性增长：

$$\text{Params}_\text{FFN}^\text{MoE} = N \cdot 8D^2 + D \cdot N \;(\text{router}) \approx 8 N D^2$$

**结论**：MoE 的"参数容量 $N$ 倍, 计算量 $k$ 倍"——这正是为什么 Mixtral 8x7B 总参 47B / active 13B 比同 active 13B 的 dense 表现强。

### 7.2　推理显存（必考！）

**面试常见陷阱问题：MoE 推理时显存是按"active 参数"还是"总参数"算？**

答案：**几乎按总参数**——所有 expert 必须**常驻 GPU**，因为下一个 token 可能路由到任何 expert。不能"按需 load 一个 expert"，因为 PCIe / NVLink 加载延迟远高于 expert 计算延迟。

具体地，单 GPU 推理 671B model（FP8 / FP16）：

| 模型 | 存储 (FP16) | 存储 (FP8) | 单卡能塞下吗 |
| --- | :-: | :-: | :-: |
| LLaMA-2 70B (dense) | 140 GB | 70 GB | H100 80G 单卡 FP8 可 |
| Mixtral 8x22B (141B) | 282 GB | 141 GB | 单卡不行，2× H100 / 8×A100 |
| DeepSeek-V3 (671B) | 1342 GB | 671 GB | **8× H100 80G = 640 GB 还差一点**；实际部署需 **2 节点 16× H100 80G** 或 **8× H200 141G**（≈1128 GB）才能塞下 FP8 权重 + KV cache |

**带宽的好处**：每 token 只取 `active_param / total_param` 比例的权重做 matmul → **memory bandwidth bottleneck 缓解**（推理多数是 memory-bound）。671B / 37B → 每 token 只读 ~5% 权重——这是 V3 在 H800 上 inference throughput 与 dense 30B 同台的根源。

### 7.3　KV Cache

MoE 不直接影响 KV cache 大小——KV 与 attention 相关，与 FFN sparsity 无关。但 V2/V3 同时引入 **MLA** 把 KV 压缩到低秩 latent，所以 V3 的 KV cache 是 LLaMA-3 70B 的 ~5%。**面试要区分"MoE 减的是 FFN 显存（FP8 OK），MLA 减的是 KV 显存"**——是两条互相独立的优化线。

## §8 EP / TP / PP / DP：MoE 并行的 4 维交织

Dense LLM 训练常用 DP + TP + PP（数据 / 张量 / 流水）三维并行。MoE 必须加第 4 维：**Expert Parallelism (EP)**。

### 8.1　Expert Parallelism

把 $N$ 个 expert 切到不同 GPU 上（如 64 EP, 每 GPU 4 个 expert）。Forward 流程：

```

   Step 1.  Token x [local batch] -> Router -> top-k expert IDs
   Step 2.  ★ all-to-all dispatch:
            按 expert ID 把 token 发到对应 GPU
            (每 GPU 收到的 token 数 ≤ capacity)
   Step 3.  本地 expert forward
   Step 4.  ★ all-to-all combine:
            把 expert 输出按原始 token ID 发回起点 GPU
   Step 5.  Gate 加权求和 -> 下一层
```

两次 **all-to-all** 是 MoE 的灵魂也是痛点：通信量 $\propto T \cdot k \cdot D$，可以轻易吃掉一半时间。

### 8.2　EP × DP × TP × PP

| 维度 | 拆什么 | 通信 |
| --- | --- | --- |
| **DP** | Batch | all-reduce（gradient） |
| **TP** | Hidden / heads（每层内部） | all-reduce（in-layer） |
| **PP** | 层（不同 stage） | point-to-point（cross-stage） |
| **EP** | Expert | all-to-all（每 MoE layer 2 次） |

实践组合（DeepSeek-V3 训练）：

- 16-way PP × 64-way EP × ZeRO-1 DP
- 单 H800 cluster: 8 GPU per node × 256 nodes = 2048 GPU
- DualPipe 让 PP bubble 接近 0；DeepEP 把 EP all-to-all 与 expert compute overlap

### 8.3　EP 通信量算账

设每 token 平均 $k$ expert, $D$ hidden, $G$ EP group size, 每 step token 数 $T$：

$$\text{all-to-all volume per layer} \approx 2 \cdot T \cdot k \cdot D \cdot \frac{(G-1)}{G}$$

（$\times 2$ 是 dispatch + combine 各一次。$(G-1)/G$ 是因为 1/G 的 token 本地命中无需通信。）

对 V3：$T \sim$ few-M tokens/step, $k=8$, $D=7168$, $G=64$ → 单 layer 通信量 ~ TB 级。这就是为什么 DeepEP / NVLink-aware kernel 这么关键。

## §9 与相关方法对比

### 9.1　MoE vs Dense（同 FLOPs / 同参数）

| 角度 | Dense | MoE |
| --- | --- | --- |
| 同 active 参数 | 表现更稳 | 容量上限低 |
| 同 total 参数 | 计算成本高 | **MoE 算力高效，容量大** |
| 训练稳定性 | 高 | 中（routing collapse / EP comm 不稳） |
| 推理显存 | 按 total ≈ 按 active | **按 total（必须全 expert 常驻）** |
| 推理 latency | memory-bound | **memory bandwidth 优势**（active 部分小） |
| 部署难度 | 低 | 高（需 EP / 多卡） |
| 微调 | 简单 | **复杂**（见 §9.3 ESFT） |

### 9.2　MoE vs Mixture-of-Depths (MoD)

MoD（Raposo et al. 2024）思路不同：**每层选 top-k 个 token 走完整层，其他 token skip 整层**（沿 depth 方向稀疏化）。对比 MoE 沿 width 方向稀疏化（每个 token 选部分 expert）。

| 维度 | MoE | MoD |
| --- | --- | --- |
| 稀疏方向 | width (FFN expert) | depth (whole layer) |
| 每 token 计算 | 固定（top-k expert） | 动态（每层是否走） |
| Router 决策 | per-layer per-token | per-layer 选 top-k token |
| 训练稳定性 | 中 | 较新，工程化弱 |

不互斥——理论上可以同时用，2025 后的几个工作（如 Llama 4 的细节、Qwen3 的某些实验）开始混用。

### 9.3　MoE 微调：ESFT 和它的同类

标准 LoRA 在 MoE 上有几个坑：

- LoRA 加到 $W_q/W_k/W_v$ 上是 attention 层级，**不动 expert**
- 加到每个 expert 上 → $N$ 倍 LoRA 参数，部分 expert 训不到（因为不被路由）

**ESFT**（Expert-Specialized Fine-Tuning, Wang et al. arXiv 2407.01906）：

- 先用任务数据做 forward，统计 router score
- 找出**任务最相关的 top-k expert**（per-layer），其余冻结
- 只训这些"任务专家"，**显存 ↓ 90%、时间 ↓ 30%、性能 ≈ 全参 fine-tune**

> 💡 **面试加分** — 比 LoRA 更适合 MoE 的本质：MoE 已经 specialize 了，微调只该 specialize 进一步，不该把通用 expert 也搅动。这是"sparse pretrain → sparse fine-tune"的自然延伸。

### 9.4　MoE vs MQA/GQA

完全正交：

- **MQA/GQA** 减 KV cache（attention 端）
- **MoE** 减 FFN 算力（FFN 端）

可以同时用（V2/V3 = MLA + MoE；LLaMA-3 = GQA + dense；理论上 GQA + MoE 也合法）。

## §10 25 高频面试题

按难度分 L1（10 必会）/ L2（10 进阶）/ L3（5 顶级 lab）。每题点开看答案要点 + 易踩坑。

### L1必会题（任何 ML 工程岗都会问）

<details>

<summary>Q1.MoE 在做什么？为什么用 sparse activation？</summary>

- 把 dense FFN 换成 $N$ 个独立 expert + router，每 token 只走 $k \ll N$ 个
- **总参数 ↑（容量）但激活参数不变（FLOPs）**——用参数换计算效率
- 训练存得下，推理算力可控
- 关键：MoE 减少 FLOPs，但**不减少推理显存**（all experts 必须常驻）

把 MoE 想成 ensemble。MoE 是**单一前向通路上选 expert**，与"训多个模型 vote"完全不同。

</details>

<details>

<summary>Q2.Top-k token-choice MoE 的 routing 公式是什么？</summary>

- Router: $s(x) = W_g x \in \mathbb{R}^N$，$g(x) = \text{softmax}(s(x))$
- 选 $\mathcal{T}_k(x) = \text{TopK}_i\, g_i$，输出 $y = \sum_{i \in \mathcal{T}_k} \tilde{g}_i E_i(x)$
- Mixtral/DeepSeek：top-k 上 **renormalize**（$\tilde{g}_i = g_i / \sum_{j \in \mathcal{T}_k} g_j$）让权重和 = 1
- Router 没 bias（标准做法）

只说"选 top 几个 expert"不写公式；忘了 renormalize；说 Q/K/V 那套（attention 公式，搞混了）。

</details>

<details>

<summary>Q3.Switch Transformer 的 aux load-balance loss 是什么？</summary>

- $\mathcal{L}_\text{aux} = \alpha \cdot N \sum_i f_i P_i$
- $f_i$ = expert $i$ 接到的 token 比例（不可微，但有数值）
- $P_i$ = expert $i$ 的平均 gate 概率（可微）
- 乘积越小说明分布越均匀（均匀时 $\sum_i f_i P_i = 1/N$）

只说"鼓励均匀"但写不出公式；只写 $\sum P_i^2$（这是 entropy regularizer，不是 Switch 公式）。

</details>

<details>

<summary>Q4.Capacity factor 是什么？为什么需要？</summary>

- $C = \lceil \alpha \cdot Tk/N \rceil$，每个 expert 的容量上限
- $\alpha = 1.25$ 是 Switch / GShard 默认（留 25% 缓冲）
- 超 capacity 的 token → **residual bypass**（expert 输出 0，残差通过）
- 没 capacity 限制无法静态 buffer，EP 通信无法 schedule

只答"防止 OOM"；不知道 residual bypass，以为 drop 就是直接归零。

</details>

<details>

<summary>Q5.MoE 推理时显存按什么算？</summary>

- **按总参数算，不是激活参数**——所有 expert 必须常驻 GPU
- 因为下一个 token 可能路由到任意 expert，按需 load 延迟无法接受
- 671B MoE FP8 ≈ 671 GB，**8× H100 80G = 640 GB 还差一点**；实际部署常用 16× H100 80G（2 节点）或 8× H200 141G
- 真正节省的是 memory bandwidth（每 token 只读 active 部分权重）

以为 MoE 显存按 active 算，所以"Mixtral 13B 单卡能跑"——错，Mixtral 47B 总参，单 24G 卡跑不下。

</details>

<details>

<summary>Q6.DeepSeek-V3 总参数 / 激活参数 / expert 数？</summary>

- **总参数 671B，激活 37B / token**
- **256 routed experts + 1 shared expert**
- Top-8 routed + 1 shared = 9 个 expert per token
- Attention 用 **MLA**，FFN 用 DeepSeekMoE
- arXiv: 2412.19437, 2024.12

把 V3 跟 V2 (236B/21B) 数字搞混；说 V3 有"多少 expert"但记错 256。

</details>

<details>

<summary>Q7.Mixtral 8x7B 真的有 7B × 8 = 56B 参数吗？</summary>

- **不是**。8x7B 的命名只表示"8 expert, 每 expert 7B 量级"
- **实际总参 46.7B**——因为 attention / norm / embedding 在所有 expert 间共享，不是每个 expert 复制一份
- 激活参数 12.9B（不是 14B，因为 router gate 是 sparse 加权）

被名字误导，直接 $8 \times 7 = 56$。

</details>

<details>

<summary>Q8.MoE 训练为什么常 routing collapse？怎么治？</summary>

- 没 balance 机制时，强者愈强：被多选 → 训得多 → 更可能被选
- 治疗：**aux loss (Switch)** / Expert Choice / Aux-loss-free bias (V3) / Z-loss / expert dropout / 适当增大 capacity factor
- 监控：每 expert 的 load + aux loss 数值 + validation loss

只说"加 aux loss"但不给具体公式；不知道还有 expert-choice 这类硬约束方法。

</details>

<details>

<summary>Q9.MoE 训练需要哪些并行？</summary>

- **DP（数据）+ TP（张量）+ PP（流水）+ EP（expert）**
- EP 是新增维度：把 $N$ 个 expert 分到不同 GPU
- 每个 MoE layer 需要 **2 次 all-to-all**（dispatch + combine）
- DeepSeek-V3：16-way PP × 64-way EP × ZeRO-1 DP，跨 2048 卡 H800

忘了 EP；以为 dense 的 DP+TP+PP 就够用。

</details>

<details>

<summary>Q10.MoE 在 inference 端的好处是什么？</summary>

- 同 active 参数下，**memory bandwidth 优势**：每 token 只读 ~active/total 比例权重
- 推理多数 token-by-token 是 memory-bound，bandwidth 减小 → throughput ↑
- 671B / 37B → 每 token 读 ~5% 权重 → 与 ~30B dense 同 latency

只说"算力少"——但显存上不去（全 expert 加载），所以单卡 deployment 还是难。

</details>

### L2进阶题（research-oriented 岗位）

<details>

<summary>Q11.Expert Choice routing 是什么？为什么 autoregressive decoder 不能用？</summary>

- 反向：每个 expert 选 top-$M$ 个 token（$M = Tk/N$）
- **天然 balance**：每 expert 严格选 $M$ 个 token，不需 aux loss
- 不 drop token（capacity 是 hard 给定）
- **不能 autoregressive**：第 $t$ 个 token 的路由依赖整个 batch（含未来 token）的 score → 不能 token-by-token 生成
- 用在 encoder / vision / BERT-style；decoder 用 token-choice
- Zhou et al. 2022 NeurIPS, arXiv 2202.09368

以为它能直接替代 Mixtral / DeepSeek 的 token-choice；忘了 causal 这一限制。

</details>

<details>

<summary>Q12.Top-k 的离散操作怎么反传梯度？</summary>

- Top-k 本身**不可微，不参与梯度**
- 但被选中的 $k$ 个 expert 的 gate 权重 $g_i$（softmax 后）**进入 output 加权求和**，对 $g_i$ 可微
- 未被选中的 expert 的 router weight 这次 step 拿不到梯度（chicken-and-egg → 需 aux loss / bias 打破）
- 类似 hard attention，但因为加权求和，梯度信号还是能到 router

以为"top-k 必须用 Gumbel-softmax 才能反传"——其实标准做法直接用 softmax + top-k 选择，gate weight 就是梯度通路。

</details>

<details>

<summary>Q13.DeepSeekMoE 的 fine-grained expert 是什么意思？</summary>

- 把 $N$ 个 baseline expert 拆成 $mN$ 个更小的 expert（每 expert $d_\text{ff}$ 缩到 $d_\text{ff}/m$）
- 路由数从 $K$ 提到 $mK$
- **总计算 / 总参数不变**，但组合数 $\binom{mN}{mK} \gg \binom{N}{K}$
- 每 token 的"专家组合身份"指数级丰富 → 专精度提升
- Dai et al. 2024, arXiv 2401.06066

以为 fine-grained = 增加 expert 数量（部分对）但漏掉"每个 expert 缩小、路由数同比例放大"这个保持算力的关键。

</details>

<details>

<summary>Q14.Shared expert 是什么？为什么需要？</summary>

- 一种**永远激活、不经 router**的 expert（每 token 都走它）
- 吸收"通用知识"（语言 / 常识），让 routed expert 专心做 specialization
- DeepSeekMoE / Llama 4 都有；Mixtral / Switch 没有
- DeepSeek-V3: 1 shared expert + 256 routed (8 selected)

误以为 shared expert 是 ensemble；或不知道它不通过 router（router 不为 shared expert 投票）。

</details>

<details>

<summary>Q15.DeepSeek-V3 的 aux-loss-free balance 怎么做的？</summary>

- 每 expert 一个偏置 $b_i$，top-k 选择用 **biased score** $s_i + b_i$
- 但 **gating weight 用原始 $s_i$**（保持主梯度路径干净）
- 每 step 按 sign 更新：$b_i \leftarrow b_i - u \cdot \text{sign}(c_i - \bar{c}_i)$
- $b_i$ 是 buffer 不进梯度图，不被 optimizer 更新
- 兜底还有极小 sequence-level aux loss（防单序列内部极端不均）
- Wang et al. arXiv 2408.15664；V3 在 2412.19437 中应用

说"V3 完全没有 aux loss"——不准确（还有 sequence-wise 兜底）；或把 biased score 用到 gating weight 上（会污染主梯度，违背设计意图）。

</details>

<details>

<summary>Q16.MoE 训练的 all-to-all 是什么？通信量怎么算？</summary>

- 每 MoE layer 做 2 次 all-to-all：dispatch（按 expert 发 token）+ combine（按 token 收回）
- 通信量 per layer ≈ $2 T k D (G-1)/G$（$G$ 是 EP group size）
- 通常占训练总时间 30-50%，DeepEP / DualPipe 是用来 overlap 通信与计算的工程武器
- $G$ 大 → 更稀疏 → 通信比例反而 ↑

只答"用 NCCL 通信"；不清楚通信量随 $T \cdot k \cdot D$ 线性增长；不知 DualPipe / DeepEP。

</details>

<details>

<summary>Q17.MoE 上做 LoRA 微调有什么问题？ESFT 怎么解？</summary>

- LoRA 加到 attention 层 → 不动 expert；加到所有 expert → $N$ 倍 LoRA 参数，且部分 expert 训不到
- **ESFT** (arXiv 2407.01906)：先 forward 统计 router score → 选任务最相关的 top-$k$ expert → 只训这些
- 性能 ≈ 全参 fine-tune，显存 ↓ 90%、时间 ↓ 30%

不知道 ESFT；以为 "MoE = 大模型，LoRA 一招通吃"。

</details>

<details>

<summary>Q18.MoE 推理时显存全 load，那 sparse activation 的好处去哪了？</summary>

- 好处转移到 **memory bandwidth**——每 token 只读 active 比例权重
- LLM decode 是 memory-bound（不是 compute-bound），所以 bandwidth 减小直接转化为 throughput
- 671B/37B 模型：每 token 读 ~5% 权重 → throughput ≈ 30B dense
- 但显存仍按总参数预算：单 H100 80G **跑不动 V3**，需 8 卡 minimum

以为 sparse 推理可以"按需 load 1 个 expert"——硬件延迟禁止；或以为 throughput 收益来自 FLOPs 节省（实际更多来自 bandwidth）。

</details>

<details>

<summary>Q19.MoE 和 MoD (Mixture-of-Depths) 区别？</summary>

- MoE 沿 **width** 稀疏：每 token 选部分 expert
- MoD 沿 **depth** 稀疏：每层选 top-$k$ 个 token 走整层，其他 token skip
- 每 token 计算量：MoE 固定（top-k expert），MoD 动态（取决于多少层接受它）
- 不互斥，可以叠加

以为 MoD = MoE 的别名；或不知道沿 depth 也能稀疏化。

</details>

<details>

<summary>Q20.Mixtral 8x7B 实际总参数为什么不是 56B 而是 46.7B？</summary>

- 8 个 expert 只在 **FFN 层** —— attention / norm / embedding / unembedding **全部共享，只算一份**
- 设 Mistral-7B 的 dense 总参 $\approx 7.24\text{B}$（FFN 约 $4.8\text{B} \approx 2/3$，shared = attention+embed+norm $\approx 2.4\text{B} \approx 1/3$）
- Mixtral 8x7B 总参 = **共享部分 + 8 × FFN** = $2.4 + 8 \times 4.8 \approx 2.4 + 38.4 \approx 40.8\text{B}$；再加上 Mistral-7B 比 LLaMA-2-7B 略大、Mixtral 还增加了 router weight，官方 **46.7B**
- 激活参数（per token，top-2）= **共享部分 + 2 × FFN** = $2.4 + 2 \times 4.8 \approx 12\text{B}$（论文给出 12.9B / 13B active）

直接 $8 \times 7 = 56$ 当总参（把 attention/embed 也乘 8）；或激活算成 $2 \times 7 = 14$ 忘掉 shared 已经只算一份。

</details>

### L3顶级 lab / Research 方向

<details>

<summary>Q21.DeepSeek-V3 aux-loss-free 为何不破坏 sparse routing semantics？</summary>

- **关键设计**：bias $b_i$ **只用于 top-k 选择**，不进入 gating weight，更不进入梯度图
- 这意味着：router 的梯度信号**只来自语言建模 loss**，与传统 aux loss "把平衡目标作为额外梯度" 完全不同
- 类比控制系统：bias 是 outer-loop controller（按 load 反馈调节），主梯度是 inner-loop optimizer（按 task loss 优化表达力）——两个 loop **解耦**
- 对比 aux loss：aux loss 的梯度会拉 router 权重往"均匀分布"方向，可能与"任务最优 routing"方向冲突
- Loss-free 让 router **在主任务下仍然 sparse + specialize**，bias 只是后处理纠偏
- 一个细节：bias **不进 gating weight** 这一点至关重要——否则相当于通过反向通路偷偷影响梯度，loss-free 名义就名不副实

把 bias 当成 learnable parameter 进 optimizer；以为 loss-free 就是"啥都不加"（实际有 sign 更新和 sequence-aux 兜底）。

</details>

<details>

<summary>Q22.Fine-grained + shared 比同总参 dense 优势在哪？理论 / 实证？</summary>

- **理论容量论**：$mN$ 个小 expert 的 routed combination 数 $\binom{mN}{mK}$ 远超 $\binom{N}{K}$，每 token 可表达的"专家组合身份"指数级丰富，更接近"每 token 独立小子网"的极限
- **专精论**：shared expert 吸收通用知识，让 routed expert 摆脱"既要又要"的负担，更可能 specialize
- **算力论**：同 active FLOPs 下，dense 必须把容量塞进激活参数；MoE 把容量塞进 dormant expert，活时只取一小撮——可以在不增加 FLOPs 的前提下把总参数推到 10x+
- **实证**：DeepSeekMoE 16B vs LLaMA-2 7B：FLOPs 相近，benchmark 上接近或超过（论文 Table 3）；DeepSeekMoE 145B vs DeepSeek 67B dense：性能持平，但 145B MoE 训练 FLOPs 只用 ~28%
- **失败模式**：若 routing 不够好（collapse），fine-grained 反而退化（少数小 expert 通吃，等效 dense 但效率更低）
- **2025-2026 实证趋势**：V3 把这条路线推到极致（256 routed），证明在 1T 级 token 训练下 fine-grained 优势显著

只说"参数多就行"——但没拆开"组合容量 ≠ 单参数容量"；忽略 routing 质量是这条路线 work 的前提。

</details>

<details>

<summary>Q23.MoE inference latency 真的能与 dense active 同台吗？瓶颈在哪？</summary>

- **理论**：每 token 只读 active 比例权重，bandwidth-bound 下 throughput 应正比于 active 参数
- **实际几个 caveat**：
  1. **Routing overhead**：router 计算 + top-k 选择 + scatter/gather 本身有 latency，长 context 下也分摊不少
  2. **Expert load 不均**：单 batch 内 routing 不均 → 部分 expert 排队，wave 浪费
  3. **EP 通信**（多卡推理）：dispatch / combine 的 all-to-all 占很大比例
  4. **Memory bandwidth ≠ memory size**：每 token 5% 权重，但 5% 落到哪些 expert 是 token-dependent，cache 命中差 → 实际带宽利用率不如理论上限
- **Mixtral 8x7B 在单卡 H100 80G**：bandwidth 利用率 ~70-80%（vs dense 70B 利用率类似），latency 与 13B dense 接近
- **V3 671B**：8 卡部署下 inference 延迟略高于 30B dense（多卡通信不可忽略）
- **结论**：理论可以，工程上还有 5-30% gap 取决于 batch size / 上下文长度 / 多卡拓扑

只说"理论上一样"，不知道实际有 routing / EP 通信 / cache 命中等几个隐藏 cost；或反过来说"MoE 推理一定慢"，也不全对（small batch / single GPU 下 Mixtral 与 dense 13B 接近）。

</details>

<details>

<summary>Q24.MoE 训练的 stability 问题有哪些？怎么 mitigate？</summary>

- **Router logit 爆炸**：训练初期 router 权重大幅波动，fp16 下 logit 可 overflow → **Z-loss** $\mathcal{L}_z = \beta \sum_t (\log \sum_i e^{s_i(x_t)})^2$ 惩罚 logit 量级（ST-MoE 2022 提出）
- **Routing collapse**：解 §4.3 提到的方案（aux loss / bias / capacity factor / expert dropout）
- **Expert representation drift**：训练中 expert 间表示渐变（被 router 协同 shape），可能让 fine-tune 时小数据 overfit 某些 expert
- **Aux loss 干扰**：$\alpha$ 大 → loss 拉 router 往均匀，损害性能；$\alpha$ 小 → balance 不够。**这是 loss-free 的根本动机**
- **bf16 vs fp16**：bf16 动态范围与 fp32 接近，更适合 router；fp16 下 router logit 容易 overflow，必须 z-loss + softmax 稳定化
- **Small batch 下 router 噪声大**：每 step 的 $f_i$ 估计噪声大，aux loss 不稳。生产里通常用 micro-batch grad accumulation 增大 effective batch

只答"加 aux loss"——但忽略 z-loss、router 数值稳定、bf16 重要性、small-batch 的统计估计问题。

</details>

<details>

<summary>Q25.如果让你设计下一代 MoE，会改什么？</summary>

这是开放题，没标准答案。给一些 2026 前沿 lab 关心的方向 + 你的判断角度：

- **更细的 expert**：DeepSeek 把 expert 推到 256，进一步推到 1024 / 4096？瓶颈在 routing collapse + EP 通信，需要新的 balance 方法
- **动态 top-k**：每 token 选不同数量 expert（easy token 走 1，hard token 走 4）。和 MoD 思想结合，"既稀疏 width 又稀疏 depth"
- **Differentiable routing**：Sinkhorn / hash-based / learned permutation，让 top-k 部分可微，router 训练更快收敛
- **Expert sharing across layers**：跨层共享一部分 expert pool（类似 ALBERT 的 cross-layer sharing），进一步压总参数
- **更好的 MoE post-training**：ESFT 之后是什么？对 SFT/RLHF 阶段 router 不一定保持 pretrain 时的 specialization，需要专门的 alignment 方案
- **MoE × long context**：1M context 下每层 routing 决策乘以 1M，router 本身可能变成瓶颈 → routing reuse / hierarchical routing
- **MoE 服务化**：单卡跑不动 671B，怎么让中小公司也能部署？expert offload / streaming weights / disaggregated expert servers
- **理论分析**：fine-grained vs dense 容量的严格量化？routing 的 implicit regularization？aux-loss-free 的收敛保证？目前都还很少

回答时**先说一个具体方向**，给清楚 motivation + technical sketch + 可能的 failure mode，比泛泛而谈"我会让 expert 更多"加分多。

</details>

## §A 附录：参考文献清单（按时间）

按出现先后整理。注：**所有 arXiv ID 已交叉验证**。

1. **Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.** (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR. arXiv:1701.06538.
2. **Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., Chen, Z.** (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. arXiv:2006.16668.
3. **Fedus, W., Zoph, B., Shazeer, N.** (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR. arXiv:2101.03961.
4. **Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A., Chen, Z., Le, Q., Laudon, J.** (2022). Mixture-of-Experts with Expert Choice Routing. NeurIPS. arXiv:2202.09368.
5. **Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., Fedus, W.** (2022). ST-MoE: Designing Stable and Transferable Sparse Expert Models. arXiv:2202.08906.（Z-loss 来源）
6. **Dai, D., Deng, C., Zhao, C., Xu, R., et al.** (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. ACL. arXiv:2401.06066.
7. **Jiang, A. Q., Sablayrolles, A., et al.** (2024). Mixtral of Experts. arXiv:2401.04088.
8. **DeepSeek-AI** (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.
9. **Wang, Z., Chen, D., Dai, D., Xu, R., Li, Z., et al.** (2024). Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning (ESFT). arXiv:2407.01906.
10. **Wang, L., Gao, H., Zhao, C., Sun, X., Dai, D.** (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. arXiv:2408.15664.
11. **DeepSeek-AI** (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
12. **Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P., Santoro, A.** (2024). Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models. arXiv:2404.02258.
13. **Meta AI** (2025). Llama 4: Scout, Maverick, Behemoth (blog announcement, April 5, 2025).
14. **Qwen Team** (2025). Qwen3 Technical Report. arXiv:2505.09388.
15. **DeepSeek-AI / DualPipe / DeepEP** (2025). Open-source releases at github.com/deepseek-ai/{DualPipe,DeepEP}.

> 💡 **2026 秋招高频提问 paper top-5**（按面试出现频率）：DeepSeek-V3 (2412.19437) > Mixtral (2401.04088) > Switch Transformer (2101.03961) > DeepSeekMoE (2401.06066) > Loss-Free Balance (2408.15664)。准备这 5 篇基本覆盖 95% MoE 问题。
