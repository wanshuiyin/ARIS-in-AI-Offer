## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 Reasoning Model** — 2024-2026 LLM 最大范式转移，一页拿下面试核心。

1. **范式转移**：以前 scale **训练算力**（参数 + 数据），现在 scale **推理算力**（reasoning tokens / search / verification）。Snell et al. 2024 (arXiv 2408.03314) 给出 **compute-optimal test-time scaling** 配方：相同推理 FLOPs 下，best-of-N + PRM beam search + sequential revision 的混合策略比单一 best-of-N 高 **>4×** 效率；FLOPs-matched 设置下，小模型 + 优化 test-time compute 在某些任务上能匹配/超过 **14×** 更大的模型。

2. **o1 (OpenAI Sep 2024)**：用 RL 训练 hidden chain-of-thought，API 只返回 `reasoning_tokens` 计数而非内容。**o3 (Dec 2024)** 在 ARC-AGI 上 75.7% (低算力) / 87.5% (高算力 172× 预算)——首次在抽象推理 benchmark 上接近人类。

3. **DeepSeek-R1-Zero (arXiv 2501.12948, Jan 2025)**：**纯 RL 从 base model 直接训**，无 SFT cold-start，rule-based reward（答案对/错 + 格式），用 **GRPO**（无 critic），涌现出 "aha moment"——模型自己学会反思、回溯、验证。

4. **DeepSeek-R1**：四阶段 pipeline = SFT 冷启动（数千高质 CoT）→ reasoning-oriented RL → rejection sampling + 通用 SFT → 全场景 RL。在 MATH-500、AIME 等数学/代码 benchmark 上对齐 o1。

5. **GRPO（DeepSeekMath, arXiv 2402.03300）**：去掉 critic value network；对每个 prompt sample $G$ 个回答 $\{o_i\}$，用 **group-relative advantage** $A_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$ 替代 GAE。显存降一半 + 训练更稳。

6. **PRM vs ORM**：ORM（outcome reward）只评 final answer；PRM（process reward，Lightman 2023 "Let's Verify Step by Step"）对每个推理 step 打分，best-of-N 选 trace 上更优。Math-Shepherd (Wang et al. 2023, arXiv 2312.08935) 从中间 step 采样 **Monte Carlo completion rollouts**（不是 MCTS tree search），用最终答案的 soft/hard estimation 自动标 step label，省去人标。

7. **s1 (Muennighoff Feb 2025, arXiv 2501.19393)**："Wait" budget forcing——在 `</think>` 处强行追加 "Wait" 让模型继续思考，1K 样本 SFT + 推理控制超 o1-preview 27% (AIME24)。

8. **易踩坑**：CoT ≠ 推理（模型可能 post-hoc 编故事）；self-consistency 在 distribution-shifted 题上崩；PRM 训练易过拟合 step pattern；GRPO 在 long-CoT 上 critic-free 反而是优势（critic 难学）。

## §1 为什么这是 2024-2026 最大范式转移

### 1.1　从 train-time scaling 到 test-time scaling

2020-2023 的 scaling law (Kaplan, Hoffmann/Chinchilla)：**性能 ∝ log(参数 × 数据 × FLOPs)**——但这是 **训练时算力**。模型一旦训完，推理时算力是固定的（一次 forward）。

2024 出现两个 anomaly：
- **OpenAI o1**：花更多 inference tokens 思考 → 性能持续提升，**呈现 log-linear scaling**（OpenAI 公开的 figure：accuracy vs reasoning compute 是直线）
- **Snell et al. 2024**：固定 inference/test-time compute 预算下用 compute-optimal scaling（best-of-N + PRM beam search + sequential revision 混合），可让小模型在 FLOPs-matched 设置下 **匹配甚至超过 14× 更大模型**——前提是 base model 在该任务上已有 non-trivial success rate

> 💡 **范式 mental model** — 把推理看成 **search over reasoning paths**：

- **System 1**（快）：一次 greedy decode = 一条路径
- **System 2**（慢）：multiple paths + verifier + backtrack = 树搜索（Tree-of-Thought）或长 CoT（o1）

把 LLM 当成 **policy + value** 的组合（像 AlphaZero），推理是 MCTS-like 的搜索过程。

### 1.2　Reasoning Model 的核心组件

```

       问题 prompt
           │
           ↓
   ┌───────────────────────────┐
   │   Policy LLM (sampler)    │ ← 生成 candidate reasoning traces
   │   - greedy / temperature  │
   │   - long CoT (o1, R1)     │
   │   - tree expansion (ToT)  │
   └───────────────────────────┘
           │  N traces
           ↓
   ┌───────────────────────────┐
   │  Verifier / Reward Model  │
   │  - ORM (outcome only)     │
   │  - PRM (per-step)         │
   │  - rule-based (math/code) │
   └───────────────────────────┘
           │
           ↓
   ┌───────────────────────────┐
   │  Aggregator               │
   │  - majority vote (SC)     │
   │  - best-of-N (verifier)   │
   │  - beam search (PRM)      │
   │  - MCTS (rStar)           │
   └───────────────────────────┘
           │
           ↓
       最终答案
```

不同 reasoning model 路线本质上是在这三个组件上做选择：
- **o1 / R1**：把 search 内化到 policy 里（long CoT 单次生成，policy 自己反思）
- **ToT / rStar**：外部显式 search（MCTS / BFS / beam）
- **Best-of-N + PRM**：sampling + verifier（最朴素的 test-time scaling）

## §2 CoT 演进：从 Wei 2022 到 o1

### 2.1　Chain-of-Thought (Wei et al., NeurIPS 2022, arXiv 2201.11903)

核心发现：**few-shot prompt 给模型展示 "step-by-step reasoning"**，性能在大模型上（>62B PaLM）出现 emergent jump，GSM8K 从 18% → 57%。

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11.
The answer is 11.
```

关键点：
- **不是 fine-tune**——纯 prompting，base model 已有能力
- **emergence**：小模型用 CoT 反而**变差**（噪声大于信号）
- 后续 Kojima et al. 2022 "Let's think step by step" 发现 zero-shot CoT 也能触发

### 2.2　Self-Consistency (Wang et al. 2022, arXiv 2203.11171)

观察：CoT decoding 是 stochastic（temperature > 0），同一题 sample 多次会得到不同推理路径，但**正确答案往往更多次出现**（如果模型能力足够）。

算法：
1. 对同一 prompt sample $N$ 条 CoT
2. 提取每条的 final answer
3. **多数投票** 选 majority answer

$$\hat{y} = \arg\max_{y \in \mathcal{Y}} \sum_{i=1}^{N} \mathbb{1}[\text{extract}(\text{trace}_i) = y]$$

效果：GSM8K +17.9%（Wang et al. 报告的 PaLM-540B 数字）。

> ⚠️ **常见 bug** — 投票前必须做 **answer extraction normalization**（去单位、化简分数、整数化）。否则 "1/2" 和 "0.5" 会被算成不同 answer，多数票被打散。

### 2.3　Tree of Thoughts (Yao et al., NeurIPS 2023, arXiv 2305.10601)

把 CoT 从 **chain** 升级成 **tree**：每个节点是一个 "thought"（一步推理），从根节点 BFS/DFS 展开多个候选 child，**LLM 自己评分**（"is this step promising?"），保留 top-k。

```

          root (problem)
           │
   ┌───────┼───────┐
  step1a  step1b  step1c        ← sample k thoughts
   │       ✗       ✗            ← LLM evaluator score
   │       prune  prune
   ├──────┼──────┐
 step2a step2b step2c           ← expand surviving node
```

Game of 24 (用 4 个数字 + 四则运算得到 24)：
- **GPT-4 CoT**: 4% 成功率
- **GPT-4 ToT** (b=5, depth=3): **74%**——巨大跳变

> 💡 **ToT vs CoT 的本质区别** — CoT 是 **autoregressive decoding**（一条路径），ToT 是 **deliberate search**（多条路径 + 显式回溯 + evaluator）。前者快但容易陷入早期错误，后者慢但能跳出局部最优。

### 2.4　从 ToT 到 long-CoT (o1 / R1 路线)

ToT 需要外部搜索框架（递归 prompt + state management）。**o1 / R1 走另一条路**：把 search **训练进 policy 里**——单次长 CoT，但模型自己学会：
- "wait, let me reconsider..."
- "actually that's wrong, the correct way is..."
- "let me verify by trying a different approach"

这些回溯 / 反思 token 在 base model 里很罕见，靠 RL **反复采样 + reward 信号** 才能放大成稳定行为。R1-Zero 报告的 "aha moment" 就是这种行为在 RL 训练中突然涌现的时刻。

## §3 PRM vs ORM：reasoning verifier 的两条路线

### 3.1　定义

| 维度 | ORM（Outcome Reward Model） | PRM（Process Reward Model） |
| --- | --- | --- |
| 监督粒度 | 整条 trace 一个 reward | 每个 step 一个 reward |
| Label 来源 | answer 对 → +1，错 → 0 | 人工标 (PRM800K) 或 MC completion rollout 估计 (Math-Shepherd) |
| 训练目标 | $\max \mathbb{E}[r(\text{trace})]$ | $\max \sum_t \mathbb{E}[r_t(\text{step}_t)]$ |
| 优势 | 标注便宜（只要 ground-truth answer） | 信号密集；能定位错误步 |
| 劣势 | 稀疏 reward，credit assignment 难 | 标注昂贵；step boundary 难定义 |

### 3.2　PRM800K (Lightman et al. 2023, arXiv 2305.20050)

OpenAI 在 MATH 数据集上人工标了 80 万个 step-level 标签：每个 step 是 `positive` / `neutral` / `negative`。

训练目标（per-step classification）：

$$\mathcal{L}_\text{PRM} = -\sum_{t=1}^{T} \log p_\phi(\ell_t \mid s_{\leq t})$$

其中 $\ell_t \in \{+, 0, -\}$，$s_{\leq t}$ 是前 $t$ 步推理。

推理时把 PRM 当 verifier：

$$\text{score}(\text{trace}) = \prod_{t} p_\phi(\ell_t = +\mid s_{\leq t}) \quad \text{或} \quad \min_t p_\phi(\ell_t = +\mid s_{\leq t})$$

（前者乘积形式更标准；min 形式更悲观但抓最弱步）

> ✅ **关键发现** — Lightman 2023 报告：用 PRM800K 做 best-of-1024 verifier，在 MATH test 上 78%（vs ORM 72%，majority vote 70%）。**PRM > ORM > self-consistency**，但代价是 80 万人工标注。

### 3.3　Math-Shepherd（Wang et al. 2023, arXiv 2312.08935）—— 自动标 step label

人标贵，怎么 scale？Math-Shepherd 思路：**用 Monte Carlo completion rollout 估每个 step 的"潜在正确率"**。

```
对每个 step s_t（部分推理）：
  从 s_t 出发 rollout K 条 completion
  count 多少条最终 answer 正确
  reward_t = (正确数) / K
```

直觉：如果一个 step 是好的，从它出发往下走应该容易得到正确答案；如果是坏的，怎么走都错。

训练 PRM 用 Math-Shepherd 标签：MSE 回归或 BCE 分类。Mistral-7B 在 GSM8K 上 77.9% → 84.1%，无需人标。

> 💡 **MCTS-label 的隐含假设** — base model 在该任务上有 non-trivial success rate（否则 rollout 全错，标签全 0）。所以 PRM 训练需要 base model 至少能解出部分题——这是个 bootstrap 问题。

### 3.4　Generative PRM 与 Critic LM

2024 后流行 "generative verifier"：把 PRM 当 **next-token-prediction**（"is this step correct? yes/no"），直接复用 LLM 架构，不用单独 reward head。代表作：Generative Verifiers (Zhang et al. 2024, arXiv 2408.15240)。优势：能利用 in-context reasoning 评分，比 scalar head PRM 更准。

## §4 Best-of-N + Verifier：最朴素的 test-time scaling

### 4.1　公式

给定 prompt $x$、policy $\pi$、verifier $V$、采样数 $N$：

$$i^{*} = \arg\max_{i \in [N]} V(\text{trace}_i), \quad \text{trace}_i \sim \pi(\cdot \mid x); \quad \hat{y} = \text{extract}(\text{trace}_{i^{*}})$$

理论极限（"oracle BoN"，假设有完美 verifier）：

$$\text{pass}@N = 1 - (1 - \text{pass}@1)^N$$

实际：verifier 不完美，BoN 的 saturation curve 远低于 oracle。Snell 2024 发现：
- 在 **简单题** 上 BoN 很快饱和（N=4 之后边际收益小）
- 在 **难题** 上 BoN 持续涨直到 N=64 / 128
- **compute-optimal**：题难 → 加 N；题易 → greedy

### 4.2　Best-of-N + PRM 代码

```python
import torch
from typing import Callable

def best_of_n_prm(
    policy_sample: Callable,          # prompt -> (trace, step_list)
    prm_score: Callable,              # step_list -> scalar in [0, 1]
    prompt: str,
    n: int = 16,
    aggregation: str = "min",         # "min" | "prod" | "mean"
) -> tuple[str, float]:
    """
    Best-of-N with process reward model verifier.
    Returns (best_trace, best_score). Note: in `prod` mode, `best_score`
    is the cumulative log-product (negative); compare scores within the same
    aggregation mode only.
    """
    best_trace, best_score = None, -float("inf")
    for _ in range(n):
        trace, steps = policy_sample(prompt)
        step_probs = [prm_score(steps[: t + 1]) for t in range(len(steps))]
        if aggregation == "min":
            score = min(step_probs)
        elif aggregation == "prod":
            # log-sum 避免数值下溢
            score = sum(torch.log(torch.tensor(p) + 1e-9) for p in step_probs).item()
        elif aggregation == "mean":
            score = sum(step_probs) / len(step_probs)
        else:
            raise ValueError(aggregation)
        if score > best_score:
            best_score, best_trace = score, trace
    return best_trace, best_score
```

> ⚠️ **PRM aggregation 选择** — Lightman 2023 实验：**min** 和 **prod** 在 BoN 上接近，**mean** 显著差（被高分 step 掩盖低分错步）。生产里常用 **min**——直觉是"trace 强度取决于最弱一环"。

### 4.3　Self-Consistency 代码

```python
from collections import Counter
from typing import Callable
import re

def _extract_braced(s: str, open_idx: int) -> str | None:
    """从 `\boxed{` 后第一个 `{` 起，按平衡括号提取内层（支持嵌套 LaTeX）。"""
    if open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i]
    return None  # 未闭合

def extract_answer(trace: str) -> str:
    """从 trace 提取 final answer。注意：生产里要做更全的规范化（单位、分数→小数、LaTeX 化简等）。"""
    # 用平衡括号匹配 `\boxed{...}`，避免 `\boxed{\frac{1}{2}}` 被截成 `\frac{1`
    pos = trace.find(r"\boxed")
    if pos != -1:
        inner = _extract_braced(trace, pos + len(r"\boxed"))
        if inner is not None:
            return inner.replace(",", "").replace("$", "").strip()
    m = re.search(r"answer is[:\s]+([^.\n]+)", trace, re.IGNORECASE)
    if m:
        return m.group(1).strip().replace(",", "").replace("$", "").strip()
    return ""

def self_consistency(
    policy_sample: Callable,
    prompt: str,
    n: int = 40,
    temperature: float = 0.7,
) -> tuple[str, dict]:
    """Self-Consistency: sample N traces, majority vote on extracted answers."""
    answers = []
    for _ in range(n):
        trace = policy_sample(prompt, temperature=temperature)
        ans = extract_answer(trace)
        if ans:                            # 跳过解析失败
            answers.append(ans)
    if not answers:
        return "", {}
    counts = Counter(answers)
    return counts.most_common(1)[0][0], dict(counts)
```

> ✅ **生产细节** — Wang 2022 报告 GSM8K 上 N=40 收益开始饱和；N=64 接近极限。Temperature 太高（>1.0）会引入 garbage 推理，太低（<0.3）多次 sample 几乎重复。**sweet spot 通常 0.5-0.7**。

## §5 RL 路线：PPO → GRPO

### 5.1　Vanilla PPO 回顾（Schulman et al. 2017）

对每个 token $t$，policy gradient with clipping：

$$\mathcal{L}_\text{PPO} = -\mathbb{E}_t \left[ \min\!\left( \rho_t A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中
- $\rho_t = \pi_\theta(o_t \mid s_t) / \pi_{\theta_\text{old}}(o_t \mid s_t)$ 是 importance ratio
- $A_t$ 是 advantage，常用 GAE-$\lambda$：$A_t = \sum_{l\geq 0} (\gamma\lambda)^l \delta_{t+l}$，$\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$

**痛点**：
1. 需要训一个 **value network** $V_\psi$（critic），通常和 policy 同等大小——显存翻倍
2. Critic 训练在 long-CoT 上很难收敛（reward 极度稀疏，只有最终答案对错）
3. GAE 需要 token-level value 估计；long-CoT (4K+ tokens) 上每个 token 的 value 噪声很大

### 5.2　GRPO（Shao et al. 2024, DeepSeekMath, arXiv 2402.03300）

核心想法：**用 group statistics 替代 critic**。

算法：
1. 对每个 prompt $x$，从 $\pi_{\theta_\text{old}}$ 采样 $G$ 个 completion $\{o_1, \dots, o_G\}$（通常 $G = 16$ 或 $64$）
2. 用 reward model 给每个 completion 打分 $\{r_1, \dots, r_G\}$
3. **Group-relative advantage**（trace-level，不是 token-level）：

$$\boxed{\;A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}\;}$$

4. 把 trace-level $A_i$ 广播给该 trace 的所有 token：$A_t = A_i \;\forall t \in o_i$
5. 用 PPO clipping objective 更新 policy（同上公式，但 $A_t$ 来自步骤 3-4）；DeepSeekMath/R1 同时加 KL 正则 $\beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_\text{ref})$。实践中用 Schulman 的 **unbiased k3 估计** $\widehat{\mathrm{KL}}_t = e^{\log\pi_\text{ref}(o_t|s_t) - \log\pi_\theta(o_t|s_t)} - (\log\pi_\text{ref}(o_t|s_t) - \log\pi_\theta(o_t|s_t)) - 1$ 对每个 token 取值（恒 $\geq 0$），再 mask + 平均

> ✅ **GRPO 关键洞察** — 为什么 group-relative 比 critic 好用？

- **同 prompt 同源**：$G$ 个 completion 共享 prompt 难度，差异完全来自 policy 输出；mean 自动减掉 prompt-specific baseline，等价做了 control variate
- **无需 value network**：直接省一半显存 + 一倍计算；critic 在 long-CoT 上本来就难学（reward 稀疏 + episode 长）
- **稳定性来自 group size $G$**：$G$ 越大，advantage 估计方差越小；DeepSeek-R1 用 $G \approx 16$

### 5.3　GRPO advantage 计算代码

```python
import torch

def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Group-relative advantage estimation.

    Args:
        rewards: [G] tensor of trace-level rewards (one prompt's group).
        eps: numerical stability.

    Returns:
        advantages: [G] tensor, mean ≈ 0, std ≈ 1.
    """
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)        # 用 biased std（除 G 而非 G-1）
    return (rewards - mu) / (sigma + eps)


def grpo_loss(
    log_probs_new: torch.Tensor,               # [G, T] new policy log p
    log_probs_old: torch.Tensor,               # [G, T] old policy log p (detached)
    log_probs_ref: torch.Tensor,               # [G, T] reference policy log p
    rewards: torch.Tensor,                     # [G] trace-level
    mask: torch.Tensor,                        # [G, T] valid token mask
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> torch.Tensor:
    """Single-prompt GRPO loss (group size = G traces).
    要求 `log_probs_old` 和 `log_probs_ref` 都是 detached（无梯度）；
    若调用方传入带梯度 tensor，会把梯度错误地传进 old/reference policy。
    """
    adv = grpo_advantage(rewards)              # [G]
    adv = adv.unsqueeze(-1)                    # [G, 1] broadcast to tokens

    # 防御性 detach：即使调用方忘了，old / ref policy 也不会被反传
    log_probs_old = log_probs_old.detach()
    log_probs_ref = log_probs_ref.detach()

    # PPO clipping
    ratio = torch.exp(log_probs_new - log_probs_old)        # [G, T]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    pg_loss = -torch.min(surr1, surr2)                       # [G, T]

    # KL 正则：约束 policy 不漂离 reference 太远
    # DeepSeekMath/R1 用 Schulman 的 unbiased k3 估计：
    #   KL ≈ exp(log_ref - log_new) - (log_ref - log_new) - 1   ≥ 0
    # 这是 KL(π_θ || π_ref) 的样本无偏估计，且总是非负，比 raw log-ratio 稳得多
    log_diff = log_probs_ref - log_probs_new               # [G, T]
    kl = torch.exp(log_diff) - log_diff - 1.0              # [G, T], ≥ 0
    loss = pg_loss + kl_beta * kl                          # [G, T]

    # Masked mean over valid tokens
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
```

> ⚠️ **常见 bug** — GRPO 实现里几个坑：

- **`log_probs_old` 必须 detach**（不参与梯度），否则和 `ratio` 形成奇怪的梯度路径
- **`std` 用 biased 还是 unbiased** 影响小但要一致（DeepSeek 公开版用 biased）
- **当 $G$ 个 rewards 全相同（如全错或全对）**，`std = 0`，且 $r_i - \mu \equiv 0$，加 `eps` 后 advantage = 0 → policy-gradient 项归零；但 KL 正则项仍存在，总 loss 退化成 `kl_beta * KL`（仍会把 policy 拉回 reference）。生产里通常 skip 这种 prompt（data filtering）以避免无信号更新，或者 clamp std 下限（如 0.1）
- **Mask 必须 cover 所有 padding token**，否则 padded log-prob 会污染 loss

### 5.4　GRPO 为什么 sample efficient（L3 高频题）

观察：R1 论文公开的 ~80 万样本（≈ 600k reasoning + 200k non-reasoning）是 **Stage 3 rejection sampling + SFT 数据**，不是 RL prompt 数；R1 Stage 2 reasoning RL 的 prompt 总量论文未严格公开，但整体训练计算量（~147K H800-hours）仍比同期 RLHF (InstructGPT 几百万人类偏好 pair) 显著小，GRPO 是核心算法贡献。

**根因**（不止是"省 critic"）：
1. **Trace-level reward 与 trace-level credit 自然对齐**：GRPO 直接把 trace reward 当 advantage 给所有 token；不需要 critic 做 per-token credit assignment。在 reward 完全只看 final answer 的设定下，这就是 sequence-level Monte-Carlo return 的标准 policy-gradient 估计（不是"最优"，但 trace-level 比 critic 在 long-CoT 上的 noisy per-token V 更稳）
2. **同 prompt 内的 contrast 自动消除 prompt-level noise**：advantage = $(r - \mu) / \sigma$ 让 advantage 分布稳定，policy update 更新方向更准；等价于在每个 prompt 内做了 paired comparison
3. **PPO clipping 限制单步更新幅度**：和 $G$ 一起保证不会因某个 prompt outlier reward 把 policy 推飞
4. **Rule-based reward 难以被 RM-hacking**：r1 用规则 reward（答案正则匹配 + 格式 `<think></think>`），没有 learned RM 可被 over-optimization；advantage 主要反映"是否答对"。注意规则 reward 并非完全不可 hack——policy 仍可能利用正则漏洞、格式 trick 或测试集泄露——但避开了 learned-RM 漂出训练分布的主要 failure mode

R1 训练 GPU 小时 ≈ 147K H800-hours（DeepSeek 公开数）——比 GPT-4 训练量小两个数量级，是 GRPO + 数据精炼 + 规则 reward 联合作用的结果。

## §6 DeepSeek-R1-Zero / R1 全流程

### 6.1　R1-Zero：纯 RL 不要 SFT 的极端实验

**唯一假设**：base model（DeepSeek-V3-Base, 671B MoE）已具备基础语言 + 数学能力。

**训练**：
- Reward = `accuracy_reward + format_reward`（rule-based，无 RM）
  - `accuracy_reward`：答案能否被自动 extract + match ground truth
  - `format_reward`：是否用 `<think>...</think>` 包围推理过程
- 算法：GRPO，$G = 16$，KL $\beta = 0.001$
- 数据：数学（MATH, AIME, 等）+ 代码（LeetCode-like 可执行评测）

**涌现行为**（"aha moment"，DeepSeek-R1 paper Fig 3）：
- 训练几 K steps 后，CoT 长度自发增长（200 token → 数千 token）
- 出现 "Wait, let me reconsider..."、"Actually that's wrong"、"Let me verify by..." 等 self-reflection 句式
- 在 AIME 2024 上 pass@1 从 base 的 15.6% → 71.0%（远超 GPT-4o-0513 的 9.3%）

> 💡 **R1-Zero 历史意义** — 在 R1-Zero 之前，社区普遍认为 reasoning 能力必须靠 **SFT cold start**（先给模型看大量 demonstrated CoT）才能 RL bootstrap。R1-Zero 证明 base model + 纯规则 reward 也能做到——这是 LLM 训练历史上第一次大规模复现 "RL from scratch elicit reasoning"（对应 AlphaGo Zero 在 LLM 上的等价物，虽然 base 已 pretrained）。

**问题**：R1-Zero 的输出有 **可读性问题** —— 推理过程语言混杂（中英文 + 数学符号乱跳）、有时候不分段、人类读不懂。所以 R1 加了后续阶段。

### 6.2　R1 完整 4 阶段 pipeline

| 阶段 | 输入 | 方法 | 目标 |
| --- | --- | --- | --- |
| **Stage 1: Cold-start SFT** | DeepSeek-V3-Base | 数千条精选 long-CoT（部分来自 R1-Zero 输出 + 人工修正可读性） | 让模型学到 "human-readable reasoning format" |
| **Stage 2: Reasoning-oriented RL** | Stage 1 模型 | GRPO + rule-based reward + 语言一致性 reward（penalize 中英混杂） | 提升推理能力 |
| **Stage 3: Rejection sampling + SFT** | Stage 2 模型 | 用 Stage 2 模型大量 sample → 用 PRM/规则筛 → 60 万 reasoning + 20 万通用 数据再 SFT | 扩展到非数学/代码领域；保留通用能力 |
| **Stage 4: All-scenario RL** | Stage 3 模型 | GRPO + (rule reward for math/code) + (RM for helpfulness/harmlessness) | 全场景对齐 |

**核心 insight**：
- **Stage 1 = readability injection**，不是为了 reasoning（reasoning 来自 RL）
- **Stage 3 = generalization injection**，把数学/代码 RL 学到的推理能力 transfer 到非可验证领域（写作、对话、QA）
- **Stage 4 = safety + helpfulness alignment**，等价于 RLHF 收尾

### 6.3　R1-Distill：把 R1 蒸到小模型

DeepSeek 用 Stage 3 数据（60 万 reasoning 样本）对 Qwen2.5-{1.5B, 7B, 14B, 32B} 和 Llama3-{8B, 70B} 做 **纯 SFT**（无 RL），得到 R1-Distill 系列。

**关键发现**（论文 Table 5）：
- DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 上 72.6 vs o1-mini 63.6——**SFT 蒸馏的小模型超过 o1-mini**
- 1.5B 模型在 MATH-500 上 83.9，远超原 Qwen2.5-Math-1.5B 的 51.0
- 论文公开版 R1-Distill **只做了 SFT，没有再叠 RL**；作者明确指出"incorporating RL could substantially boost performance"——所以"distill 后能否 RL 出更多"是未结论的开放问题，而不是"做不到"

> ⚠️ **L3 高频题** — "为什么 R1-Distill 远超原 model，但比直接对小模型做 RL 反而更好？"

- DeepSeek-R1-Distill-Qwen-32B (SFT only) > DeepSeek-Qwen-32B-RL (直接 RL from scratch on Qwen)
- 原因：小模型 base 太弱，纯 RL 难涌现 reasoning；用大模型 R1 生成的 reasoning trace 做 SFT 等价于 **distillation through demonstrations**，把"reasoning 行为模式"直接复制过来。
- Implication：**reasoning 能力的涌现需要大 base + 强 RL；小模型上靠蒸馏复制是最经济路线**。

## §7 Test-Time Scaling：Snell 2024 与 s1 budget forcing

### 7.1　Snell et al. 2024（arXiv 2408.03314）核心结论

题目：固定 inference FLOPs，怎么分配最优？

实验设置：
- 同一 base model（PaLM-2 系列）
- 不同 test-time 策略：BoN、majority vote、ToT-like beam search、PRM beam search、sequential revision（让模型看自己的回答再改）
- **compute-optimal**：每个 prompt 根据难度动态选策略（简单题 greedy，难题 beam search + PRM）

核心 finding：
- **在固定预算下，optimal test-time scaling > 14× 模型 scaling**（在某些 MATH 子集上）
- **简单题**：majority vote / BoN 4-8 即可
- **难题**：PRM-guided sequential revision + beam search 收益最大

> ✅ **Compute-optimal scaling 公式** —

$$\text{Compute}_\text{optimal}(x) = \arg\min_{(\theta, N)} \mathbb{E}[L(\pi_\theta(x; N))] \;\text{s.t.}\; \text{FLOPs}(\theta, N) \leq B$$
- $\theta$ = 模型规模，$N$ = test-time samples / beam width
- 论文给的实际经验：在 base model 能 pass@1 > 30% 的题上，加 N 比加参数更划算

### 7.2　s1: Simple Test-Time Scaling（Muennighoff et al. Feb 2025, arXiv 2501.19393）

最简单的 reasoning model：**1000 条样本 SFT + budget forcing 推理控制**。

**数据**（s1K，1000 条）：
- 来自三个 criteria 筛选：difficulty、diversity、quality
- 每条样本：question + reasoning trace（用 Gemini Thinking 生成）+ answer

**训练**：Qwen2.5-32B-Instruct 上做 26 分钟 SFT（16×H100，1 epoch）——堪称史上最便宜 reasoning model。

**推理 trick：budget forcing**

```
模型生成：
<think>
[推理 token...]
[模型试图输出 </think>，但当前 token 数 < target_budget]
[强行替换 </think> 为 "Wait"]
[模型继续推理...]
...
[当 token 数 >= target_budget 或自然结束]
</think>
答案：...
```

效果：在 AIME 2024 上 s1-32B（用 budget forcing）= 56.7%，**超过 o1-preview 的 44.6%**。

> 💡 **为什么 "Wait" 这么简单的 trick 有用？**

- SFT 让 base model 学会了 `<think>...</think>` 格式，但 reasoning 长度有 distribution（短题短推理）
- 强行注入 "Wait" 让模型停在 `</think>` 的 boundary，激活其 in-context 的"反思"能力
- 等价于 forcing 模型留在 "thinking mode" 多 sample 几条推理路径
- 失败案例：base model 完全没见过反思 pattern 时，"Wait" 后面接的内容是 garbage——s1 的 1K SFT 数据已经隐含了反思 pattern

### 7.3　Sequential vs Parallel test-time compute

| 维度 | Parallel (BoN, Self-Consistency) | Sequential (long CoT, s1 budget forcing, o1) |
| --- | --- | --- |
| 实现 | Sample N 次，verifier/vote 选 | 单条 trace 拉长，模型自己反思 |
| 延迟 | $N$ 倍延迟（并行可降低） | 单条但很长（不可并行降低） |
| 显存 | KV cache × N 或顺序复用 | 单 KV cache 但 sequence 长 |
| Plateau | 早期饱和（N=8-16） | 持续扩展（10K-100K token） |
| 适合 | 浅推理（GSM8K、commonsense） | 深推理（AIME、Codeforces） |

Snell 2024 报告：**对简单题 parallel 更划算，对难题 sequential 显著更好**。

## §8 MCTS for Reasoning：rStar 系列

### 8.1　PUCT 公式（AlphaGo / AlphaZero 起源）

在节点 $s$，对每个 action $a$，PUCT score：

$$\boxed{\;U(s, a) = Q(s, a) + c_\text{puct} \cdot \pi(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}\;}$$

- $Q(s, a)$ = 当前 $(s, a)$ 的平均 value（exploitation）
- $\pi(a \mid s)$ = policy prior（来自 policy network）
- $N(s)$ = node $s$ 的总访问次数
- $N(s, a)$ = action $a$ 被选过几次
- $c_\text{puct}$ = exploration constant（典型 1.0-2.0）

直觉：访问次数少的 + policy 看好的 + 当前 value 高的 action 被偏好。

### 8.2　rStar (Microsoft Research, arXiv 2408.06195) 关键 idea

把 LLM 当 policy + value 套进 MCTS：
- **State**：当前部分推理 $s_{<t}$
- **Action**：下一步要做什么（rStar 定义了 5 种 reasoning action：propose one-step, propose sub-question, generate full CoT, decompose, rephrase）
- **Reward**：终止节点用 mutual consistency check（另一个 LLM verify）

**rStar-Math (arXiv 2501.04519)** 进一步：用 **MCTS rollout 自动标 process label**（类似 Math-Shepherd 思路），训出 process preference model (PPM)，policy + PPM 反复 self-evolve 四轮。Qwen2.5-Math-7B 在 MATH 上从 58.8 → 90.0，逼近 o1-preview。

### 8.3　简化 MCTS for reasoning 伪代码

```python
import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MCTSNode:
    state: str                          # 部分推理 trace
    parent: Optional["MCTSNode"] = None
    prior: float = 0.0                  # policy network prob
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

def puct_score(node: MCTSNode, parent_visits: int, c_puct: float = 1.5) -> float:
    """PUCT: Q + c * P * sqrt(N_parent) / (1 + N)"""
    return node.q_value + c_puct * node.prior * math.sqrt(parent_visits) / (1 + node.visit_count)

def select(root: MCTSNode) -> MCTSNode:
    """Traverse tree by PUCT until reaching a leaf."""
    node = root
    while node.children and not node.is_terminal:
        node = max(node.children.values(),
                   key=lambda c: puct_score(c, node.visit_count))
    return node

def expand(node: MCTSNode, policy_lm, k: int = 4):
    """Sample k next-step candidates from policy LM."""
    if node.is_terminal:
        return
    candidates = policy_lm.sample_next_steps(node.state, k=k)   # list of (step_text, prior)
    for step_text, prior in candidates:
        new_state = node.state + "\n" + step_text
        is_term = step_text.startswith("Final answer:")
        node.children[step_text] = MCTSNode(
            state=new_state, parent=node, prior=prior, is_terminal=is_term,
        )

def rollout(node: MCTSNode, policy_lm, verifier) -> float:
    """Simulate from node to terminal; return reward."""
    if node.is_terminal:
        return verifier(node.state)
    state = node.state
    while len(state) < 4096:
        step = policy_lm.sample_next_step(state, temperature=1.0)
        state += "\n" + step
        if step.startswith("Final answer:"):
            break
    return verifier(state)   # ORM or PRM final score

def backup(node: MCTSNode, reward: float):
    """Propagate reward up the path."""
    while node is not None:
        node.visit_count += 1
        node.value_sum += reward
        node = node.parent

def mcts_search(prompt: str, policy_lm, verifier, n_simulations: int = 100, k: int = 4) -> str:
    """Standard MCTS for reasoning."""
    root = MCTSNode(state=prompt)
    # 首轮先 expand 根节点，保证 root.children 非空
    expand(root, policy_lm, k=k)
    for _ in range(n_simulations):
        leaf = select(root)
        if leaf.visit_count > 0 and not leaf.is_terminal:
            expand(leaf, policy_lm, k=k)
            if leaf.children:
                leaf = max(leaf.children.values(),
                           key=lambda c: c.prior)        # 先选 prior 最高
        reward = rollout(leaf, policy_lm, verifier)
        backup(leaf, reward)
    if not root.children:
        return root.state                                 # safety guard
    # 沿 visit_count 最高路径下行，直到 terminal 或 leaf（叶子未 expand）
    # 这样返回的是完整 trace 而非仅第一层 child；如果中间层最强路径未到 terminal，
    # 也至少返回该路径上当前已 expand 的最深 state
    node = root
    while node.children and not node.is_terminal:
        node = max(node.children.values(), key=lambda c: c.visit_count)
    return node.state
```

> ⚠️ **MCTS for LLM 的真实开销** — 上面伪代码假设 `policy_lm.sample_next_steps` 是廉价 op，实际上每个 expansion 是 **一次完整 LLM forward pass**（生成几十 token + 评分）。100 simulations × 4 children = 400 次 LLM call，比单次 greedy decode 慢 100-400×。这是为什么实际 deployment 用 long-CoT (o1/R1) 比 MCTS 更主流——把 search 内化到 policy。

### 8.4　PRM-Guided Beam Search（更轻量替代）

不做完整 MCTS，只保留 top-$b$ 个 partial trace，每步用 PRM 评分：

```python
import math

def prm_beam_search(
    policy_lm,
    prm,
    prompt: str,
    beam_width: int = 4,
    expansion: int = 4,
    max_steps: int = 16,
) -> str:
    """PRM-guided step-level beam search."""
    # (cumulative_log_score, partial_trace, is_done)
    beams = [(0.0, prompt, False)]
    for _ in range(max_steps):
        new_beams = []
        for score, trace, done in beams:
            if done:
                new_beams.append((score, trace, True))
                continue
            # 从当前 trace expand expansion 个候选 step
            candidates = policy_lm.sample_next_steps(trace, k=expansion, temperature=0.8)
            for step, _ in candidates:
                new_trace = trace + "\n" + step
                step_prob = prm.score_step(new_trace)        # in [0, 1]
                # 累加 log 概率（数值稳定）
                new_score = score + math.log(step_prob + 1e-6)
                is_terminal = step.startswith("Final answer:")
                new_beams.append((new_score, new_trace, is_terminal))
        # 防御性 edge case：若所有 beam 都 expand 出空候选，避免 max(empty)
        if not new_beams:
            break
        # 按累积 log-score 选 top-b
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(b[2] for b in beams):
            break
    # 返回最高分的 trace（空 beam 时回退到初始 prompt）
    return max(beams, key=lambda x: x[0])[1] if beams else prompt
```

> 💡 **beam search vs MCTS** — beam search 是 **deterministic + breadth-bounded**（每层固定 b），不回溯；MCTS 是 **stochastic + adaptive**（visit count 决定深度），可回溯。前者快 5-10×，后者在搜索深的树时质量更高。Snell 2024 推荐：题难度中等 → PRM beam search；超难 → MCTS。

## §9 Reasoning Model 全景对比

### 9.1　主流 reasoning model 对比表（截至 2026）

| 模型 | 厂商 | 发布 | 训练范式 | 推理控制 | 公开性 |
| --- | --- | --- | --- | --- | --- |
| **o1-preview / o1** | OpenAI | 2024-09 | RL on hidden CoT (细节闭源) | reasoning_effort: low/med/high | 闭源 |
| **o3** | OpenAI | 2024-12 / 2025 | o1 后继；ARC-AGI 75.7%/87.5% | 推理 budget 可调；high-compute 172× | 闭源 |
| **DeepSeek-R1-Zero** | DeepSeek | 2025-01 | 纯 RL (GRPO + 规则 reward) | 自然停止 | 完全开源（权重 + 论文） |
| **DeepSeek-R1** | DeepSeek | 2025-01 | 4 阶段：SFT cold-start + RL + rejection SFT + RL | 自然停止 | 完全开源 |
| **R1-Distill (1.5B-70B)** | DeepSeek | 2025-01 | SFT only on R1 reasoning data | 自然停止 | 完全开源 |
| **Claude 3.7 Sonnet** | Anthropic | 2025-02 | hybrid: standard + extended thinking | budget tokens 用户可设（up to 128K） | 闭源，但 thinking 内容可见 |
| **Gemini 2.0 Flash Thinking** | Google DeepMind | 2024-12 | 推理优化 (具体方法未公布) | 显式展示 thinking | 闭源 |
| **s1-32B** | Stanford/Allen AI | 2025-02 | 1K-sample SFT + "Wait" budget forcing | budget forcing 控制 thinking 长度 | 完全开源（含 s1K 数据） |
| **rStar-Math** | Microsoft | 2025-01 | MCTS + PPM self-evolution | 推理时显式 MCTS | 部分开源 |
| **DeepSeek-Prover-V2** | DeepSeek | 2025-04 | subgoal decomposition + Lean 4 + RL | Lean 形式化验证 | 完全开源 |

### 9.2　选型决策树（面试常问）

```

任务领域？
├── 数学/竞赛（AIME / IMO） ─→ R1 / R1-Distill-32B / o1 / s1-32B
├── 形式化定理证明（Lean / Coq） ─→ DeepSeek-Prover-V2
├── 代码（Codeforces / SWE-bench） ─→ o1-pro / R1 / Claude 3.7 + thinking
├── 通用 reasoning（agent / chain task） ─→ Claude 3.7 / Gemini 2 Thinking / R1
├── 小模型部署（边缘 / 移动） ─→ R1-Distill-1.5B-7B
└── ARC-AGI / 抽象推理 ─→ o3（高算力档）

部署预算？
├── 高（CoT 几千 token / 题）─→ o1, o3, R1, Claude 3.7-extended
├── 中（CoT 1-2K token）─→ s1-32B, R1-Distill-32B
└── 低（greedy 或 BoN=8）─→ R1-Distill-1.5B + best-of-N with verifier
```

### 9.3　CoT 是否真的反映推理？（深题）

经典争议：**Attention is not Explanation** (Jain & Wallace 2019) 在 CoT 上的 echo——模型展示的 reasoning 是否真实反映其内部计算？

证据：
- **支持真实**：Anthropic Sleeper Agents 实验 (Hubinger et al. 2024) 显示 CoT 内容会影响下游 action（不是纯 post-hoc rationalization）
- **支持非真实**：Turpin et al. 2023 "Language Models Don't Always Say What They Think" 发现给 biased exemplar 后，模型 CoT 给出 plausible 但错误的 reasoning（biased 但 CoT 没提 bias）
- **R1-Zero 的 "aha moment"** 是有限证据：行为变化（更长 + 更多反思）确实和性能涨同步，但仍可能是 surface pattern

**面试时务必给 balanced view**：CoT 是有用 + 部分可信的；但不能当 100% explanation。

## §10 25 高频面试题（L1 必会 / L2 进阶 / L3 顶级 lab）

按 gpt-5.5 xhigh 模拟的顶级 lab interviewer 视角排序。

### L1 必会题（任何 LLM 岗都会问）

<details>

<summary>Q1. Chain-of-Thought 是什么？什么时候用？</summary>

- few-shot prompt 给模型展示 "step-by-step" 推理 demonstration

- 在大模型（>62B）上 emergent 出推理能力（GSM8K +30+%）

- 小模型上 CoT 反而变差

- 后续 Kojima 2022 "Let's think step by step" 发现 zero-shot CoT 也行

把 CoT 当成"魔法 prompt"——它只是触发，能力来自 base model 自身

</details>

<details>

<summary>Q2. Self-Consistency 怎么工作？为什么比 greedy 好？</summary>

- temperature > 0 sample N 条 CoT

- 提取每条的 final answer，做 normalize（去单位、化分数）

- 多数票选 majority answer

- 直觉：正确答案是"吸引子"，多条 sampling 路径会收敛于此

- Wang 2022 报告 GSM8K +17.9%

不做 answer normalization；以为温度越高越好（实际 0.5-0.7 最佳）

</details>

<details>

<summary>Q3. Tree-of-Thought (ToT) 与 CoT 区别？</summary>

- CoT 是 autoregressive 单链路径

- ToT 是显式树搜索：每个节点是 thought，sample k 个 child，LLM 自评分

- ToT 可以回溯（backtrack）和剪枝

- 但 ToT 需要外部搜索框架 + 多次 LLM call，比 CoT 慢 5-50×

- Game of 24: CoT 4% → ToT 74%（GPT-4）

只说 ToT 是"多次 CoT"——它的核心是显式 evaluator + backtrack

</details>

<details>

<summary>Q4. ORM 和 PRM 区别？</summary>

- ORM (Outcome RM)：整条 trace 一个 reward（基于 final answer 对错）

- PRM (Process RM)：每个 reasoning step 一个 reward

- PRM 信号密集但标注贵（人标 PRM800K 80 万 step）

- Math-Shepherd (Wang et al. 2023, arXiv 2312.08935) 从中间 step 出发采样 **Monte Carlo completion rollouts**（不是 MCTS tree search），用「rollout 多少条最终对」的 soft/hard estimation 自动标 PRM step label

- Lightman 2023: best-of-1024 上 PRM 78% > ORM 72% > majority vote 70%

把 PRM 当成"训练 reward"——它主要用于推理时 verify，不一定参与 RL

</details>

<details>

<summary>Q5. Best-of-N 怎么工作？什么时候饱和？</summary>

- 同 prompt sample N 条 trace

- 用 verifier (ORM 或 PRM) 选最高分那条

- 简单题：N=4-8 饱和

- 难题：可以涨到 N=64-128 才饱和

- 理论上限 oracle pass@N = $1-(1-p_1)^N$，但 verifier 不完美时远低于此

以为 N 越大越好——verifier 误差会让 BoN 在某点之后开始下降（"verifier overfit to surface features"）

</details>

<details>

<summary>Q6. o1 / R1 的 reasoning_tokens 是什么？</summary>

- 模型生成的 hidden chain-of-thought tokens

- API 只返回 token 计数（如 OpenAI o1 返回 `reasoning_tokens` 字段），内容不公开

- 用户付费按 reasoning_tokens 计费（这就是 o1 贵的原因）

- R1 把 reasoning 包在 `<think>...</think>` 里，全文可见

- 推理算力 ≈ reasoning_tokens × model FLOPs/token

把 reasoning_tokens 当 "no-cost 优化"——它显著影响延迟和成本

</details>

<details>

<summary>Q7. GRPO 和 PPO 主要区别？</summary>

- PPO 需要 critic (value network)，估计每个 token 的 baseline

- GRPO 用 **group statistics 替代 critic**：同 prompt sample G 个 trace，advantage = $(r_i - \mu)/\sigma$

- GRPO 把 trace-level advantage 广播给该 trace 所有 token

- 显存降一半（无 critic），long-CoT 上 critic 本来就难学，所以 GRPO 反而稳

- 共享：PPO clipping、KL 正则 to reference policy

以为 GRPO 是"小改动"——它在 long-CoT 上的稳定性优势是质变

</details>

<details>

<summary>Q8. R1-Zero 的 "aha moment" 是什么？</summary>

- DeepSeek-R1 paper Fig 3 报告：纯 RL 训练几 K steps 后

- CoT 长度自发增长（数百 → 数千 token）

- 自发出现 "Wait, let me reconsider..." 等反思 pattern

- 性能跳变（AIME pass@1 15% → 70%）

- 直觉：rule-based reward + GRPO 让"思考更久 + 自验证"成为高 reward strategy

把 "aha moment" 当玄学——它是 reward shaping + 长 episode RL 的可预期涌现

</details>

<details>

<summary>Q9. R1-Distill 为什么用 SFT 不用 RL？</summary>

- R1 的 reasoning 能力靠大 base + 强 RL 涌现

- 直接对小模型 RL 难涌现（base 太弱，rollout 几乎全错，reward 信号过稀疏）

- 用 R1 生成的 60 万 reasoning trace 做 SFT，等价 demonstration learning

- 论文报告：32B SFT-distill 在 AIME 上 72.6 vs 直接对 Qwen-32B RL 的 47.0

- Implication：小模型上**蒸馏 reasoning > 直接 RL**（在当前算法下）

以为 RL 总是比 SFT 好——前提是 base 够强

</details>

<details>

<summary>Q10. Snell 2024 的 "test-time compute > parameter scaling" 是什么意思？</summary>

- 固定 inference 算力预算，让 1B 模型多 sample + verify

- 在某些 MATH 子集上，可超过 14× 大的模型的 greedy 性能

- 前提：base model 在该任务有 non-trivial pass@1（>30%）

- 不是普适：完全不会的任务，再多 test-time compute 也救不回来

- 实际工业部署常用 R1-Distill + BoN=8 + PRM 替代直接调 R1

把它当 "scaling law 终结" ——它只是"另一个维度的 scaling law"，不取代训练 scaling

</details>

### L2 进阶题（reasoning 方向 / research 岗）

<details>

<summary>Q11. 手推 GRPO advantage 公式，以及 std=0 的情况如何处理？</summary>

- 对每个 prompt sample G traces，得 rewards $\{r_1, \dots, r_G\}$

- $\mu = \frac{1}{G}\sum r_i$, $\sigma = \sqrt{\frac{1}{G}\sum(r_i - \mu)^2}$（biased std）

- $A_i = (r_i - \mu)/(\sigma + \epsilon)$

- 当 $G$ 个 rewards 全相同（全对或全错）→ $r_i - \mu = 0$ 且 $\sigma = 0$，所以
  - 加 $\epsilon$ 时 advantage $= 0/\epsilon = 0$ → **policy-gradient 项归零**；但 KL 正则项仍存在，**总 loss 仍可能有 KL 更新**（把 policy 拉回 reference）
  - 不加 $\epsilon$ 时是 $0/0 = $ NaN
  - 注意不是 $\pm\infty$（分子也是 0）

- 实践：要么 skip 该 prompt（GRPO_loss=0），要么 clamp $\sigma$ 到一个 floor（如 0.1）

- 这种情况说明该 prompt **过易或过难**，data filtering 应剔除

只写公式不说 std=0 的边界

</details>

<details>

<summary>Q12. R1 的 4 阶段 pipeline 每阶段目标是什么？</summary>

- **Stage 1 Cold-start SFT**：让 base 学会 human-readable reasoning format（不为 reasoning 能力本身）

- **Stage 2 Reasoning RL**：GRPO + 规则 reward 提升数学/代码推理

- **Stage 3 Rejection sampling + SFT**：扩展到非可验证领域 + 保留通用能力

- **Stage 4 All-scenario RL**：safety + helpfulness 收尾（类似 RLHF）

- 关键：reasoning 来自 Stage 2 RL；Stage 1 + 3 是 readability/generalization 注入；Stage 4 是对齐

把 4 阶段当"做菜步骤"——其实每阶段功能正交

</details>

<details>

<summary>Q13. 为什么 R1 的 reward 是 rule-based 而非学的 RM？</summary>

- 数学/代码可以**程序化验证**（答案正则匹配、单元测试）

- 学的 RM 容易被 hack（reward model overoptimization → policy 找到 trick 而非真正解题）

- 规则 reward 提供接近 ground-truth 的 signal，**避开了 learned-RM 漂出训练分布的主要 failure mode**（仍可能被正则漏洞、格式 trick、test-set 污染 hack，但远比 RM hacking 容易堵）

- 代价：只能用于可验证任务（math/code/format），不适合开放任务

- R1 Stage 4 又加了 RM 处理 helpfulness/harmlessness——可验证任务用 rule，开放任务用 RM

把 rule-based 当"简单"——它的关键是难以被 RM-hacking，而非简单

</details>

<details>

<summary>Q14. Budget forcing ("Wait" trick) 为什么有效？</summary>

- s1 在 1K reasoning trace 上 SFT，模型学会 `<think>...</think>` 格式 + 反思 pattern

- 推理时如果模型试图输出 `</think>` 但 token 数还没到 target budget

- 强行替换为 "Wait"，模型自然续接反思 token

- 等价 forcing 模型留在 thinking mode，多 sample 几条 internal reasoning path

- 失败模式：base 完全没见过反思 pattern → "Wait" 后接 garbage（s1 的 1K SFT 是必要前提）

以为 "Wait" 是 prompting trick——它依赖 SFT 注入的反思 pattern

</details>

<details>

<summary>Q15. PRM 怎么用在 best-of-N？aggregation 怎么选？</summary>

- 对 N 个 trace，每条用 PRM 给每个 step 打分 $p_1, \dots, p_T$

- 三种 aggregation：**min**, **prod (log-sum)**, **mean**

- Lightman 2023: **min ≈ prod >> mean**（mean 被高分 step 掩盖弱 step）

- min 直觉：trace 强度取决于最弱一环

- 代码细节：prod 用 log-sum 避免数值下溢

只用 mean——常见错误

</details>

<details>

<summary>Q16. PUCT 公式是什么？c_puct 怎么调？</summary>

- $U(s, a) = Q(s, a) + c_\text{puct} \cdot \pi(a \mid s) \cdot \sqrt{N(s)} / (1 + N(s, a))$

- $Q$ = exploit；$c_\text{puct} \cdot \pi \cdot \sqrt{N}/(1+N(s,a))$ = explore

- $c_\text{puct}$ 大：偏向 exploration（policy prior 和未访问 action 影响大）

- $c_\text{puct}$ 小：偏向 exploitation（已发现的高 Q action 主导）

- AlphaZero 用 $c_\text{puct} \approx 1.0$；MCTS-for-LLM 常用 1.5-2.0（因为 LLM policy prior 比围棋更准）

- 注意：分子是 $\sqrt{N(s)}$（parent visits），分母是 $1 + N(s, a)$（child visits）

把 $\sqrt{N}$ 错记为 child visits（错的）

</details>

<details>

<summary>Q17. Math-Shepherd 怎么自动标 step label？关键假设是什么？</summary>

- 对 trace 中每个 step $s_t$，从 $s_t$ 出发 rollout K 条 completion

- 数多少条最终 answer 正确，得 estimated step quality $\hat{q}_t = (正确数) / K$

- 用 $\hat{q}_t$ 当 BCE/MSE 标签训 PRM

- **关键假设**：base model 在该任务有 non-trivial success rate（否则 rollout 全错，$\hat{q}_t$ 全 0）

- Bootstrap 问题：弱 base → 没法用 MCTS-label；强 base → 不需要 PRM

- 实际：用中等强度 base（Mistral-7B post-SFT）作 rollout source

以为 MCTS-label 是免费——它需要 base 已有部分能力

</details>

<details>

<summary>Q18. CoT 是真的反映推理吗？怎么验证？</summary>

- 部分真实，部分 post-hoc rationalization（共识）

- Turpin 2023: 给 biased exemplar，模型 CoT 给出 plausible 但错误的解释（没 mention bias）

- Anthropic Sleeper Agents (Hubinger 2024): CoT 内容影响下游 action，不纯 post-hoc

- 验证方法：causal intervention（改 CoT 看 output 是否变）、faithfulness benchmark

- 面试 talking point：保持 balanced view，不要走极端

只说"CoT 是 explanation"或"CoT 全是 post-hoc"——两个极端都错

</details>

<details>

<summary>Q19. 怎么判断哪个 reasoning model 适合你的任务？</summary>

- 任务可验证 (math/code)：rule-based RL 路线（R1 / R1-Distill）

- 任务开放 (writing/dialogue)：hybrid RM 路线（Claude 3.7-extended / o1）

- 抽象推理 (ARC-AGI)：o3 高算力档（其他模型在 ARC-AGI 上仍很弱）

- 形式化证明：DeepSeek-Prover-V2（Lean 4 集成）

- 部署预算严：R1-Distill-7B/14B + best-of-N + PRM verifier（比直接 R1 便宜 10×+）

不看任务就推荐 o1——错配会贵且效果差

</details>

<details>

<summary>Q20. KL 正则在 GRPO 里的作用是什么？β 怎么调？</summary>

- $\mathcal{L}_\text{total} = \mathcal{L}_\text{PG} + \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_\text{ref})$

- $\pi_\text{ref}$ = 训练开始前的 policy（一般是 SFT 后或 base）

- $\beta$ 大：policy 不漂离 reference，但学不动新能力

- $\beta$ 小：policy 自由探索，但可能 collapse（生成 garbage）

- DeepSeek-R1 用 $\beta = 0.001$（很小，鼓励探索）；标准 RLHF 用 0.01-0.1

- 对 long-CoT，KL 在 token-level 累加，总量很大，所以 $\beta$ 要远小于短 CoT

照搬 RLHF 的 $\beta$ 到 long-CoT——会过度抑制探索

</details>

### L3 高级题（顶级 lab / 研究方向）

<details>

<summary>Q21. GRPO 比 PPO sample efficient 的 root cause 是什么？（不止"省 critic"）</summary>

- **Trace-level reward 与 trace-level credit 完美对齐**：当 reward 只来自 final answer，PPO 用 critic 做 per-token credit 反而引入噪声；GRPO 直接 broadcast trace-level advantage——advantage 估计本身仍带有 group baseline 引入的有限偏差（mean/std 都是有偏的样本统计），但比 critic 在长 episode 上的 high-variance 估计更稳，且对 trace-level reward 而言其方差更低

- **同 prompt 内 contrast 消除 prompt-level baseline noise**：advantage = $(r-\mu)/\sigma$ 等价于 paired comparison，比 critic 估的全局 baseline 准

- **Long-CoT 上 critic 难学**：reward 极度稀疏（episode 4K-32K token），$V_\psi$ 在中间 token 上几乎随机；GRPO 跳过这个学习问题

- **Rule-based reward 难以被 RM-hacking**：r1 用规则 reward，没有 learned RM 可被 over-optimization；policy 优化方向接近 ground-truth（仍可能被正则/格式漏洞 hack，但避开了 RM-distribution-shift 这条主路）

- **Group size G 控制 variance**：variance ∝ $1/G$；$G=16$ 给出足够低 variance 同时不爆显存

- 结论：GRPO 不是"小改动"，是在 long-CoT + rule reward 设定下的 algorithmically right answer

只说"省 critic"——表面原因

</details>

<details>

<summary>Q22. R1-Zero 的纯 RL 涌现 vs 历史 PPO RLHF (InstructGPT) 差别在哪？为什么前者突破后者不能？</summary>

- **Reward 来源**：R1-Zero 用 rule-based，InstructGPT 用学的 RM (preference model)

- **Reward density 与稀疏度**：rule reward 在 long-CoT 上是 response/trace-level sparse 但难以被 hacking 且 signal 接近 ground-truth；InstructGPT 的 learned RM 也输出 **response-level scalar preference reward**（不是逐 token 打分），token-level advantage 是 critic + GAE + KL penalty 共同构造的，RM 在 RLHF 全流程里仍易遭遇 reward overoptimization（policy 漂出 RM 训练分布 → 拿到不真实的高分）

- **Algorithm**：R1-Zero 用 GRPO；InstructGPT 用 PPO + critic + RM

- **Reward scope**：R1-Zero 训 reasoning（可验证）；InstructGPT 训 alignment（开放）——前者有 oracle reward，后者没有

- **Base model**：R1-Zero 用 V3-Base (671B MoE)，已有强 pretrained reasoning prior；InstructGPT 是 GPT-3 (175B dense)

- 历史原因：2022-2023 PPO+RM 范式被 RM overopt + critic 难学拖累；rule reward 在数学/代码上才走通

- 含义：**reasoning RL 突破 = rule reward + GRPO + 强 base + long-CoT 联合作用**，不是单点技术胜利

只说"DeepSeek 用了 GRPO"——错过整个 paradigm shift

</details>

<details>

<summary>Q23. 为什么 "Wait" 这么简单的 trick 能超 o1-preview？这告诉我们什么？</summary>

- s1 的核心：1K 精选 trace SFT (Qwen2.5-32B) + "Wait" budget forcing

- 第一层解释：1K SFT 已让模型学会 `<think>` 格式 + 反思 pattern 的"shape"

- 第二层：模型实际上**已经"知道怎么想"**（在 pretrain 中见过大量人类推理），SFT 只是激活 + 格式化

- 第三层："Wait" 强制模型在 thinking boundary 停下，重新采样——相当于强行做了一次 in-context self-revision

- 推论：**reasoning 能力的核心是激活而非注入**——base model 已有大量推理 prior

- Implication for research：
  - 不要假设 reasoning 必须靠大规模 RL 才能出
  - SFT data quality > quantity（s1K 1000 条 > 大量低质数据）
  - 推理控制 (budget forcing) 是 vastly underexplored 维度

- 反思：s1 不否定 R1 路线——R1-Distill 也是 distill 一种 SFT，s1 是这条思路的极端版本

只说 "s1 很简单很厉害"——错过 reasoning = activation 这个观察

</details>

<details>

<summary>Q24. 比较 sequential test-time compute (long CoT) 和 parallel test-time compute (BoN / MCTS) 的本质差异。什么时候选哪个？</summary>

- **Sequential（o1, R1, s1）**：单条 trace 拉长，模型自反思 + 自验证
  - 优势：单 KV cache（显存友好）；信息在 trace 内连续传递（后面 step 看得到前面所有 reasoning）
  - 劣势：早期错误传播到末端（无 backtrack）；难任务上需要极长 trace（10K-100K token）

- **Parallel（BoN, ToT, MCTS）**：多条独立 trace，外部 aggregator/verifier 选
  - 优势：可并行 → 延迟低；每条 trace 独立，错误不传播
  - 劣势：trace 之间无信息交换；verifier 必须精准否则 aggregation 失效

- **选型决策**：
  - 任务 sequential dependency 强（数学竞赛、定理证明）→ long CoT（错误信息后续可被反思修正）
  - 任务多解（codeforce、creative writing）→ BoN（多路径覆盖）
  - 任务有 well-defined intermediate verifier（math step）→ MCTS / PRM beam search
  - 延迟敏感 → parallel（可在 GPU 上并行）
  - 单 GPU 内存敏感 → sequential（单 KV cache）

- **未来方向**：sequential + parallel 混合——单条 long CoT 内嵌入 multi-path 探索（如 o1 内部可能就在做这个，但闭源不可知）

只说 "sequential 比 parallel 好"——任务依赖

</details>

<details>

<summary>Q25. 如果让你设计下一代 reasoning model，应该往哪几个方向走？（open-ended 顶级 lab interview）</summary>

可信回答框架（不需面面俱到，挑 2-3 个深入展开）：

- **方向 1 - 训练算法**：
  - GRPO 现在 trace-level；如何 token-level 又不引入 critic？（如学 PRM-as-critic）
  - Reward shaping：rule reward 太稀疏，能否 dense 化但保持 hard-to-hack（如形式化验证 intermediate steps）？
  - Continual RL：R1 训完就 freeze；能否 online RL during deployment？

- **方向 2 - Test-time compute scaling**：
  - Adaptive budget：根据题目难度动态分配 reasoning tokens（Snell 2024 起点）
  - Sequential + parallel 混合：long CoT 中嵌入 sub-tree exploration
  - Multi-agent debate：多个 LLM 互查、对抗

- **方向 3 - Verifier**：
  - Generative PRM 替代 scalar PRM（用 LLM 评 step quality 比 scalar head 更准）
  - Self-verifier：让模型自己 verify 自己（DeepSeek-Prover-V2 在 Lean 上是雏形）
  - Cross-domain transfer：math PRM 能否 transfer 到 code PRM？

- **方向 4 - 评测**：
  - 现有 reasoning benchmark (AIME, MATH) 接近饱和——下一代 evalution 标准？
  - Robustness：reasoning model 在 adversarial prompt 上是否 brittle？

- **方向 5 - 推理可解释性**：
  - CoT faithfulness（前 Q18）：让 CoT 真实反映 internal computation
  - Mechanistic 可解释性：能否定位到具体 attention head 负责"反思"？

- **方向 6 - Reasoning + agent**：
  - 现在 reasoning 主要在 single-turn；agentic setting 中 reasoning 怎么跨 turn 保持？
  - Tool use + reasoning 怎么 jointly optimize？

照搬现有方法 + 加一点——不展现 research taste

</details>

## §A 附录：核心 paper 时间线 + 一句话总结

按时间倒序：

| 日期 | Paper | arXiv | 一句话贡献 |
| --- | --- | --- | --- |
| 2025-04 | DeepSeek-Prover-V2 | 2504.21801 | subgoal decomposition + Lean 4 RL，MiniF2F 88.9% |
| 2025-02 | Claude 3.7 Sonnet | (no arXiv) | hybrid 模型，extended thinking budget 用户可控 |
| 2025-02 | s1: Simple Test-Time Scaling | 2501.19393 | 1K SFT + "Wait" budget forcing 超 o1-preview |
| 2025-01 | DeepSeek-R1 / R1-Zero | 2501.12948 | 纯 RL (GRPO + rule reward) 涌现推理；R1 = 4 阶段 pipeline |
| 2025-01 | rStar-Math | 2501.04519 | MCTS + PPM self-evolution, 7B 接近 o1-preview |
| 2024-12 | o3 (OpenAI) | (no arXiv) | ARC-AGI 75.7%-87.5%，首次抽象推理逼近人类 |
| 2024-12 | Gemini 2.0 Flash Thinking | (no arXiv) | Google 首个推理模型，thinking 显式可见 |
| 2024-09 | o1 (OpenAI) | (no arXiv) | 首个商用 reasoning model，hidden CoT + RL |
| 2024-08 | Snell et al. Test-Time Compute | 2408.03314 | 优化 test-time compute > 14× 模型 scaling |
| 2024-08 | rStar | 2408.06195 | MCTS + mutual reasoning，小 LM 大幅提升 |
| 2024-02 | DeepSeekMath / GRPO | 2402.03300 | GRPO 算法首次提出，去除 critic |
| 2023-12 | Math-Shepherd | 2312.08935 | MCTS rollout 自动标 PRM label |
| 2023-05 | Tree of Thoughts | 2305.10601 | 显式 tree search + LLM evaluator |
| 2023-05 | Let's Verify Step by Step | 2305.20050 | PRM > ORM > majority vote；PRM800K 数据集 |
| 2022-03 | Self-Consistency | 2203.11171 | sample N + 多数票，GSM8K +17.9% |
| 2022-05 | Zero-shot CoT (Kojima) | 2205.11916 | "Let's think step by step"，zero-shot 触发 CoT |
| 2022-01 | Chain-of-Thought (Wei) | 2201.11903 | few-shot step-by-step demonstration，CoT 在大模型上 emergent |

> 💡 **建议精读 4 篇** — 准备面试时间有限时，按优先级读：

1. DeepSeek-R1 (2501.12948) —— 涵盖 GRPO + 4 阶段 + R1-Zero "aha moment"
2. DeepSeekMath (2402.03300) —— GRPO 算法原始论文
3. Let's Verify Step by Step (2305.20050) —— PRM 基础
4. Snell et al. (2408.03314) —— test-time compute scaling 范式

读完这 4 篇 + 本 cheat sheet，reasoning model 面试题目应能 80%+ 覆盖。

> ⚠️ **常考开放题准备** — 顶级 lab interview 经常问 open-ended 题（如 Q25），关键是展现 **research taste**：能列出 3-5 个具体方向（不是"我会做 reasoning model" 这种空话），每个方向能给一个 concrete proposal + 一个 expected failure mode。准备时不要死记，多读最近 6 个月 arXiv 上的 reasoning paper，构建自己的 taxonomy。
