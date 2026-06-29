## §0 TL;DR Cheat Sheet

> 💡 **7 句话搞定 post-training alignment** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **RLHF pipeline (Ouyang 2022 InstructGPT)**：SFT → RM (Bradley-Terry pairwise) → PPO + per-token KL；价值模型 (value head) 单独训练，policy 与 reference policy 通过 KL 约束。

2. **PPO 核心 (Schulman 2017)**：clipped surrogate $L^{\text{CLIP}}(\theta) = \mathbb{E}[\min(r_t A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$，重要性比 $r_t = \pi_\theta / \pi_{\theta_\text{old}}$；advantage 用 **GAE** $A_t^{\text{GAE}} = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}$ 平衡偏差/方差。

3. **DPO 闭式 (Rafailov 2023 NeurIPS)**：KL-regularized RLHF 的最优策略 $\pi^*(y|x) \propto \pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$，反解得 $r = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$；代入 Bradley-Terry，**$\log Z$ 在 pairwise 差中消掉**，留下纯 SFT-style 损失 $-\log\sigma(\beta\log\frac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_\text{ref}(y_l)})$。

4. **GRPO (DeepSeekMath 2024, R1 2025)**：对每个 prompt 采样一组 $G$ 个回答，advantage 用**组内归一化** $\hat{A}_i = (r_i - \text{mean}(\mathbf{r}))/\text{std}(\mathbf{r})$；**省掉 value model**，省一半显存，特别适合 LLM 数学/代码 RL。

5. **DPO 变体生态**：KTO（仅需 thumbs up/down，无需 pair）、IPO（防 reward overfit 的 $\ell_2$ 形式）、SimPO（去 reference，length-normalized）、ORPO（SFT + odds-ratio 一阶段融合）、RLOO（多采样 leave-one-out baseline，无需 value）、ReMax（greedy baseline，进一步省）。

6. **PRM vs ORM (Lightman 2023 arXiv / 2024 ICLR)**：Process-reward 监督每步 reasoning，Math-Shepherd (Wang 2024 ACL) 自动标 PRM；Outcome-reward 只看最终答案。**PRM 在数学推理上稳压 ORM**，但标注成本高。

7. **Reward hacking 是核心痛点**：模型 over-optimize proxy reward 导致答案变长、谄媚、风格异常；缓解手段 = KL penalty、reward clipping、length penalty、ensemble RM、Constitutional AI / RLAIF（Bai 2022, Lee 2023）。

## §1 Post-Training Alignment 直觉

把 LLM 训练分成三段：

- **Pretraining**：next-token prediction on 万亿 token 语料 —— 学世界知识与语言模式
- **SFT (Supervised Fine-Tuning)**：在 instruction-response pair 上做 next-token —— 学指令格式与基础能力
- **Alignment / RL post-training**：让模型输出**与人类偏好对齐**（helpful / harmless / honest）—— 学"哪个回答更好"

为什么 SFT 不够？因为 SFT 只能模仿正例（"做得好的样子"），无法显式学**对比信号**（"A 比 B 好"）。RL post-training 提供了三种范式：

| 范式 | 信号 | 代表 | 一句话 |
| --- | --- | --- | --- |
| **RLHF + PPO** | 学一个 RM 模仿偏好，再 RL 优化 RM | InstructGPT, ChatGPT, Claude | RM-in-the-loop on-policy RL |
| **DPO 系** | 跳过 RM，直接在偏好数据上做 contrastive 损失 | DPO, IPO, SimPO, KTO, ORPO | offline，无需 sampling |
| **GRPO 系** | RM 在线打分但**省 value model**，组内归一化算 advantage | DeepSeek-R1, Kimi-K1.5 | 数学/代码任务首选 |

> 💡 **为什么不用纯 reward？** — 如果你直接 maximize reward，model 会找 RM 的捷径（reward hacking）。KL penalty $\beta \cdot \text{KL}(\pi || \pi_\text{ref})$ 是"防漂移"的核心机制：让 RL 后的策略不要离 SFT base 太远，相当于一个隐式正则。

### 1.1　语言任务 RL 的特殊性

经典游戏 RL（Atari、Go）和 LLM RL 差异很大，面试时常被反问：

| 维度 | 游戏 RL（PPO 经典场景） | LLM RL |
| --- | --- | --- |
| **状态空间** | 图像 / 棋盘 | token 序列，可 $\sim 10^4$ 长 |
| **动作空间** | 数十到数百离散动作 | vocab 大（$\sim 10^5$） |
| **轨迹长度** | 数千步 | 通常 1 个 response（整段 generation 后给一次 reward） |
| **Reward 稀疏度** | 中间也有 reward | 通常**只有 terminal reward**（response 末尾） |
| **环境** | 独立 simulator | RM（也是个神经网络，**会被 hack**） |
| **on/off-policy** | on-policy（PPO） | RLHF: on-policy；DPO: 完全 offline |

因为只有 terminal reward，**LLM RL 的 advantage 估计常常用粗粒度方案**：PPO+GAE 在每个 token 上分配 advantage（但 reward 大多数 token 是 0）；GRPO 直接用整段 response 的 reward，给所有 token 同一个组内归一化 advantage。

## §2 PPO 核心

### 2.1　Vanilla policy gradient 复盘

策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t)\, A^{\pi_\theta}(s_t, a_t)\right]$$

REINFORCE 是 $A \leftarrow R_t$（return），方差大；Actor-Critic 用 $A^{\pi}(s, a) = Q^\pi(s,a) - V^\pi(s)$ 减方差。

### 2.2　PPO clipped surrogate（必考公式）

定义重要性比：

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)}$$

PPO-Clip 的目标函数：

$$\boxed{\;L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\Big(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\Big)\right]\;}$$

直觉：

- 若 $A_t > 0$（这个 action 比 baseline 好），希望提升 $\pi_\theta(a_t|s_t)$，但**最多提升到 $r_t = 1+\epsilon$**（防止一次更新太激进）。
- 若 $A_t < 0$（这个 action 比 baseline 差），希望降低 $\pi_\theta(a_t|s_t)$，但**最多压到 $r_t = 1-\epsilon$**。
- `min` 选两者较小者 → **悲观估计**（pessimistic bound）：当我们想"加分"时，clip 上限；想"扣分"时，clip 下限。

典型 $\epsilon = 0.1 \sim 0.2$。LLM RLHF 实践中通常用 $0.1 \sim 0.2$，过大易爆 KL。

> ⚠️ **PPO-Clip vs PPO-Penalty** — 原论文还有一种 PPO-Penalty 形式：$L = \mathbb{E}[r_t A_t] - \beta\, \text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)$，并 adaptively 调 $\beta$。生产中**clipped 形式更常用**（rl4lms / TRL / OpenRLHF 默认）。

### 2.3　GAE：generalized advantage estimation

定义 TD residual：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE 是不同步长 advantage 的**指数加权平均**：

$$\boxed{\;A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}\;}$$

边界情况：

- $\lambda = 0$ → $A_t = \delta_t$，纯 TD(0)，偏差大方差小
- $\lambda = 1$ → $A_t = \sum_l \gamma^l r_{t+l} - V(s_t)$，纯 Monte Carlo（advantage 等于实际 return 减 baseline），偏差小方差大
- 典型 $\lambda = 0.95$，$\gamma = 0.99$

**LLM 中 GAE 的退化**：在 RLHF 中通常 $\gamma = 1$（不折扣），且只有 terminal reward，所以 $\delta_t = -V(s_t) + V(s_{t+1})$ 对中间 token、$\delta_T = R_T - V(s_T)$ 对终止 token。这种情况下 GAE 等价于做了"value baseline + reward 回传"。

### 2.4　PPO 在 RLHF 中的完整目标

LLM 中每个 timestep $t$ 对应生成第 $t$ 个 token，state = $(x, y_{\lt t})$，action = $y_t$。reward 一般加 KL penalty：

$$\tilde{r}_t = \mathbb{1}[t = T] \cdot R(x, y) - \beta \log \frac{\pi_\theta(y_t | x, y_{\lt t})}{\pi_\text{ref}(y_t | x, y_{\lt t})}$$

最终目标：

$$L^{\text{PPO}}(\theta) = L^{\text{CLIP}}(\theta) - c_v \cdot \underbrace{\mathbb{E}_t (V_\phi(s_t) - V_t^\text{target})^2}_{\text{value loss}} + c_e \cdot \underbrace{\mathbb{E}_t \mathcal{H}[\pi_\theta(\cdot | s_t)]}_{\text{entropy bonus}}$$

典型 $c_v = 0.5$，$c_e = 0.01$（LLM 中 entropy bonus 一般很小或为 0，因为 vocab 大本就高熵）。

> ⚠️ **代码块约定** — 后文 PPO / RM / DPO / GRPO 四块为**教学伪代码**，每块独立可读（已各自 `import torch / torch.nn.functional as F`）。生产实现需要额外补：(1) HF transformer forward 传 `attention_mask`；(2) decode-time `position_ids` 处理 padding；(3) `gather` 前对 `targets` clip 到 vocab；(4) RM 取 last-token 时按 `attention_mask` 找真实末尾。本文聚焦核心 loss 推导。

### 2.5　代码（核心 60 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_gae(rewards, values, dones, gamma=1.0, lam=0.95):
    """
    rewards: [B, T]    per-token reward (含 KL penalty)
    values:  [B, T+1]  value at s_0...s_T (s_T = terminal, V=0)
    dones:   [B, T]    1 if terminal, 0 otherwise
    returns advantages [B, T], returns [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        non_term = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * non_term - values[:, t]
        gae = delta + gamma * lam * non_term * gae
        advantages[:, t] = gae
    returns = advantages + values[:, :T]
    return advantages, returns


def ppo_step(policy, value, batch, eps_clip=0.2, c_v=0.5, c_e=0.01):
    """
    batch: dict with keys
      input_ids:    [B, L]     prompt + response tokens
      action_mask:  [B, L]     1 for response tokens, 0 for prompt/pad
      old_log_probs:[B, L]     log π_θold(y_t | s_t) at sample time
      advantages:   [B, L]     GAE advantages (already normalized)
      returns:      [B, L]     GAE returns (V target)
    """
    logits = policy(batch["input_ids"]).logits          # [B, L, V]
    # log-prob of taken action y_t at position t (shifted by 1: predict t+1 from t)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets   = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, targets).squeeze(-1)  # [B, L-1]
    new_log_probs = F.pad(new_log_probs, (1, 0))               # 对齐 [B, L]

    mask = batch["action_mask"].float()
    ratio = torch.exp(new_log_probs - batch["old_log_probs"])  # r_t

    # ── PPO-Clip 核心 ──
    A = batch["advantages"]
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * A
    policy_loss = -((torch.min(surr1, surr2) * mask).sum() / mask.sum())

    # ── Value loss ──
    V = value(batch["input_ids"]).squeeze(-1)                  # [B, L]
    value_loss = (((V - batch["returns"]) ** 2) * mask).sum() / mask.sum()

    # ── Entropy bonus ──
    probs = log_probs.exp()                                    # [B, L-1, V]
    entropy = -(probs * log_probs).sum(-1)                     # [B, L-1]
    entropy = F.pad(entropy, (1, 0))
    entropy_bonus = (entropy * mask).sum() / mask.sum()

    loss = policy_loss + c_v * value_loss - c_e * entropy_bonus

    # 监控（不参与反传）
    with torch.no_grad():
        approx_kl = ((ratio - 1) - torch.log(ratio.clamp_min(1e-8)))
        approx_kl = (approx_kl * mask).sum() / mask.sum()
        clip_frac = (((ratio < 1 - eps_clip) | (ratio > 1 + eps_clip)).float() * mask).sum() / mask.sum()

    return loss, {"policy": policy_loss.item(), "value": value_loss.item(),
                  "entropy": entropy_bonus.item(),
                  "approx_kl": approx_kl.item(), "clip_frac": clip_frac.item()}
```

> ⚠️ **PPO 工程踩坑 top 5** —

- `approx_kl` 用 $\mathbb{E}[r - 1 - \log r] \ge 0$（Schulman 2020 blog），比 $\mathbb{E}[\log r]$ 更稳，不会负。
- 每 epoch 多次 PPO update（典型 4 次），但要监控 `approx_kl`：超过 `target_kl`（如 0.02）就 early stop。
- Advantage 务必做 batch-level normalization（减均值除标准差），否则 scale 会让 learning rate 失效。
- 用 reference policy 算 KL penalty 时，KL 是 token-level（不是 sequence-level）；写错维度容易梯度爆炸。
- `c_v=0.5` 只是经验值；若 value loss 远大于 policy loss，会"吃掉"梯度。可以用 separate optimizer 或对 value 单独 lr。

## §3 RLHF Pipeline（InstructGPT 范式）

Ouyang et al. 2022 NeurIPS 论文 *Training language models to follow instructions with human feedback* 是 ChatGPT 的前身论文，定义了今天 RLHF 的标准三阶段：

### 3.1　Stage 1 — SFT（监督微调）

给定 instruction-response pair $\{(x, y)\}$（高质量人工撰写），做 next-token：

$$\mathcal{L}_\text{SFT}(\phi) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_\text{SFT}}\left[\sum_t \log \pi_\phi(y_t | x, y_{\lt t})\right]$$

输出 $\pi_\text{SFT}$（也叫 $\pi_\text{ref}$，因为下游 RL 拿它做 KL anchor）。

### 3.2　Stage 2 — RM（Reward Model）

对同一 prompt $x$ 用 $\pi_\text{SFT}$ 采样多个回答，让人**两两比较**得到偏好对 $(x, y_w, y_l)$，$y_w \succ y_l$。

**Bradley-Terry 偏好模型**：

$$P(y_w \succ y_l | x) = \sigma(r^*(x, y_w) - r^*(x, y_l))$$

其中 $r^*$ 是未知的"真实"奖励函数。我们用 $r_\psi$（一个 transformer，最后一层接 scalar head）拟合：

$$\boxed{\;\mathcal{L}_\text{RM}(\psi) = -\mathbb{E}_{(x, y_w, y_l)} \log \sigma\!\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big)\;}$$

实现细节：

- 一般用 SFT model 初始化 RM（共享 backbone），最后 token 的 hidden state 过线性层得到 scalar reward。
- 训练完后 RM 参数**冻结**。

```python
import torch
import torch.nn.functional as F

def rm_loss(reward_model, batch):
    """
    batch:
      chosen_ids:   [B, L]     prompt + y_w
      rejected_ids: [B, L]     prompt + y_l
    """
    r_w = reward_model(batch["chosen_ids"])      # [B]  scalar at last token
    r_l = reward_model(batch["rejected_ids"])    # [B]
    # Bradley-Terry NLL
    loss = -F.logsigmoid(r_w - r_l).mean()
    accuracy = (r_w > r_l).float().mean()
    return loss, accuracy
```

### 3.3　Stage 3 — PPO + KL（Policy Optimization）

目标：

$$\boxed{\;\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \big[r_\psi(x, y)\big] - \beta\, \mathbb{E}_x\, \text{KL}\!\big(\pi_\theta(\cdot|x) \,\big\|\, \pi_\text{ref}(\cdot|x)\big)\;}$$

实现时把 KL 拆到每个 token 上，与 RM reward 合并成 per-token reward $\tilde{r}_t$（见 §2.4），然后跑 PPO。

```

prompt x
   │
   │   π_θ generates response y
   ↓
(x, y)
   │
   │   r_ψ(x, y) → scalar reward       (Stage 2 RM, frozen)
   │   KL(π_θ || π_ref)               (per-token, on the fly)
   ↓
r̃_t = R · 1[t=T] − β log(π_θ/π_ref)
   │
   │   GAE on r̃_t → advantages
   ↓
PPO update on π_θ and V_φ
```

> ⚠️ **为什么需要 reference policy？** — 没有 KL anchor，policy 会**严重 over-optimize RM**（reward hacking）：输出越来越长、重复废话、谄媚（"As an AI..."）、风格漂移到 RM 的偏见样本上。$\beta = 0.01 \sim 0.1$ 是 InstructGPT 常用范围；太小漂移，太大学不到东西。

### 3.4　实际工程：4 个模型同时驻留

PPO RLHF 训练时**显存里同时有 4 个模型**：

1. **Policy** $\pi_\theta$（trainable）
2. **Reference policy** $\pi_\text{ref}$（frozen，算 KL）
3. **Reward model** $r_\psi$（frozen，算 reward）
4. **Value model** $V_\phi$（trainable，PPO 需要）

这是 RLHF 显存大的根本原因；4 倍 base model + optimizer state + gradient = 容易爆。**这也是 DPO / GRPO 在 LLM RL 中被广泛接受的根本原因——它们各自砍掉了某些模型。**

## §4 DPO：闭式直接偏好优化

Rafailov et al. 2023 NeurIPS *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* 是 RLHF 简化的里程碑。**核心观察**：KL-regularized RL 问题的最优解有闭式形式，可以反解得到隐式奖励，于是 PPO 步骤被替换成纯监督学习。

### 4.1　KL-regularized 最优策略（关键步骤）

考虑 RLHF 目标（公式 §3.3）：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}\big[r(x, y)\big] - \beta\, \text{KL}\!\big(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x)\big)$$

**对单个 $x$ 求最优 $\pi^*(\cdot | x)$**（Lagrangian）：

$$\mathcal{L}_x[\pi] = \sum_y \pi(y|x) r(x, y) - \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} + \mu\!\left(1 - \sum_y \pi(y|x)\right)$$

对 $\pi(y|x)$ 求偏导并令 = 0：

$$r(x, y) - \beta\!\left(\log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} + 1\right) - \mu = 0$$

整理：

$$\log \pi^*(y|x) = \log \pi_\text{ref}(y|x) + \frac{r(x, y)}{\beta} - \frac{\mu + \beta}{\beta}$$

记 $\log Z(x) = (\mu + \beta)/\beta$（partition function 的 log），得：

$$\boxed{\;\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right), \quad Z(x) = \sum_y \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right)\;}$$

### 4.2　反解：implicit reward

把上式取 log 反解 $r$：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

**关键洞察**：reward 是 $\pi^*$ 与 $\pi_\text{ref}$ 的 log-ratio（再加一个不依赖 $y$ 的项 $\beta \log Z(x)$）。所以一旦我们有 $\pi^*$，就有 reward；反之亦然。

### 4.3　代入 Bradley-Terry 得到 DPO 损失

Bradley-Terry 给出偏好概率：

$$P(y_w \succ y_l | x) = \sigma\!\big(r(x, y_w) - r(x, y_l)\big)$$

代入 §4.2 的反解（**注意 $\beta \log Z(x)$ 不依赖 $y$，在 $r(x, y_w) - r(x, y_l)$ 中消掉！**）：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}$$

把要学的 $\pi^*$ 改记为可学的 $\pi_\theta$，得到 **DPO 损失**：

$$\boxed{\;\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\;}$$

> ✅ **为什么 DPO 这么"反直觉地"有效？** —

- 它**没有显式 reward model**，但隐式 reward = $\beta \log(\pi/\pi_\text{ref})$。
- 它**不需要 sampling**（不像 PPO 要 on-policy rollouts），整个训练就是 offline contrastive learning。
- KL 约束是**隐式**写进 loss 的（通过 $\pi_\text{ref}$ 在 log-ratio 分母里）。
- 整个流水线变成 SFT-style：监督学习一遍就完事。

### 4.4　DPO 梯度的隐含意义

对 $\theta$ 求梯度（推导细节略，原论文 Appendix）：

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta \mathbb{E}\big[\sigma(\hat{r}_l - \hat{r}_w)\big(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\big)\big]$$

其中 $\hat{r} = \beta \log(\pi_\theta/\pi_\text{ref})$ 是 implicit reward。**直觉**：

- 系数 $\sigma(\hat{r}_l - \hat{r}_w)$ 是"当前模型对 $y_l \succ y_w$ 的错误置信度"——错得越自信，权重越大（hard-example mining 效应）。
- 接着提升 $\log \pi_\theta(y_w | x)$ 并降低 $\log \pi_\theta(y_l | x)$。

### 4.5　代码（DPO loss 核心 30 行）

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy, ref_policy, batch, beta=0.1):
    """
    batch:
      chosen_ids:    [B, L]     prompt + y_w
      rejected_ids:  [B, L]     prompt + y_l
      chosen_mask:   [B, L]     1 for y_w response tokens, 0 elsewhere (prompt + pad)
      rejected_mask: [B, L]
    """
    # log π(y | x) = sum_t log π(y_t | x, y_<t) over response tokens
    def log_prob_sum(model, ids, mask):
        logits = model(ids).logits[:, :-1]                  # [B, L-1, V]
        logp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:].unsqueeze(-1)                      # [B, L-1, 1]
        token_logp = logp.gather(-1, tgt).squeeze(-1)       # [B, L-1]
        token_mask = mask[:, 1:]                            # 对齐 next-token 预测
        return (token_logp * token_mask).sum(dim=-1)        # [B]

    pi_w  = log_prob_sum(policy,     batch["chosen_ids"],   batch["chosen_mask"])
    pi_l  = log_prob_sum(policy,     batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():
        ref_w = log_prob_sum(ref_policy, batch["chosen_ids"],   batch["chosen_mask"])
        ref_l = log_prob_sum(ref_policy, batch["rejected_ids"], batch["rejected_mask"])

    # log-ratios
    logits_w = pi_w - ref_w        # = log(π/π_ref)(y_w)
    logits_l = pi_l - ref_l
    diff = beta * (logits_w - logits_l)

    loss = -F.logsigmoid(diff).mean()
    # implicit reward margin (用来做监控)
    chosen_reward   = beta * logits_w.detach()
    rejected_reward = beta * logits_l.detach()
    margin = (chosen_reward - rejected_reward).mean()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, {"loss": loss.item(), "margin": margin.item(),
                  "accuracy": accuracy.item()}
```

> ⚠️ **DPO 已知失败模式** —

- **Likelihood decreases for both $y_w$ and $y_l$**（Pal et al. 2024, Saeidi et al.）：DPO 只要求 $\log\pi(y_w) - \log\pi(y_l)$ 涨，没强制 $\log\pi(y_w)$ 涨。实测中两者经常一起降，只是 $y_l$ 降得更多——可能导致模型"什么都不愿意说"。
- **对 $\beta$ 敏感**：$\beta$ 太小→失去 KL 约束→reward hacking；太大→学不动。常用 $0.05 \sim 0.5$。
- **数据质量决定一切**：偏好对噪声大时 DPO 比 PPO 更脆弱（PPO 通过 RM ensemble 缓冲，DPO 直接吃原数据）。
- **off-policy bias**：偏好数据是 $\pi_\text{SFT}$ 采的，但 $\pi_\theta$ 在训练中漂移，存在分布偏移。

## §5 GRPO：组相对优势

DeepSeekMath (Shao et al. 2024 arXiv 2402.03300) 提出 **Group Relative Policy Optimization**，DeepSeek-R1 (DeepSeek-AI 2025 arXiv 2501.12948) 把它推到 reasoning 旗舰位置。

### 5.1　动机

PPO 的 value model 在 LLM 上很难训：

- LLM token-level value 没有明确语义（中间 token 通常 reward = 0，只有 terminal 才有）
- Value model 与 policy 同等大小 → 显存双倍

GRPO 的**核心 idea**：对每个 prompt $x$ **采样一组 $G$ 个回答** $\{y_1, \dots, y_G\}$，用 RM 打 $G$ 个 reward $\{r_1, \dots, r_G\}$，**advantage 直接用组内归一化**：

$$\boxed{\;\hat{A}_{i, t} = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}\;}$$

整段 response 内**所有 token 共享同一个 $\hat{A}_i$**（因为没有 value model 给 per-token baseline，只用 sequence-level reward 算组相对优势）。

> ✅ **GRPO 的精髓** — 把 PPO 的 "$A_t = Q - V$"（Critic 出 $V$）换成 "$A_i = (r_i - \bar{r}) / \sigma$"（组内统计出 baseline）。**省掉 value model**，省下一半显存；同时组内 baseline 自动做了 variance reduction。

### 5.2　GRPO 目标

GRPO 保留 PPO-Clip 的形式，但加 KL penalty 进 loss（不是进 reward）：

记 importance ratio $\rho_{i, t}(\theta) = \pi_\theta(y_{i,t}|x_i, y_{i,\lt t}) / \pi_{\theta_\text{old}}(y_{i,t}|x_i, y_{i,\lt t})$（避免与 sequence-level reward $r_i$ 混淆）：

$$L^\text{GRPO}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \Big(\min(\rho_{i, t} \hat{A}_{i, t},\, \text{clip}(\rho_{i, t}, 1\!-\!\epsilon, 1\!+\!\epsilon) \hat{A}_{i, t}) - \beta\, \text{KL}_{i, t}(\pi_\theta \| \pi_\text{ref})\Big)\right]$$

其中 KL 用 K3 estimator (Schulman blog 2020)：

$$\text{KL}_{i, t} = \frac{\pi_\text{ref}(y_{i,t}|\cdot)}{\pi_\theta(y_{i,t}|\cdot)} - \log\frac{\pi_\text{ref}(y_{i,t}|\cdot)}{\pi_\theta(y_{i,t}|\cdot)} - 1$$

该 estimator 保证非负、低方差，相比直接 $\log(\pi_\theta/\pi_\text{ref})$ 更稳定。

### 5.3　DeepSeek-R1 的关键改造

DeepSeek-R1 (Jan 2025) 在 GRPO 基础上做了两件事：

1. **R1-Zero**：从 pretrain base 直接跑 GRPO，**无 SFT 阶段**，纯靠 rule-based reward（数学正确性 + 格式奖励）。emergent 长 CoT。
2. **R1**：加少量 SFT cold-start + 多阶段 RL（reasoning RL → SFT → general RL）。开源 32B/70B 在数学推理上对标 o1。

> 💡 **为什么 GRPO 在数学/代码 RL 上特别有效？** —

- **Rule-based reward**：数学有唯一答案、代码有单元测试，**绕过 neural RM，reward hacking 风险显著降低**（reward signal 接近地面真相，但 policy 仍可能找 grader 漏洞，例如猜答案、不写推理只写"42"、利用 test 覆盖盲区等，所以"完全没有 hacking"不严谨）。
- **组内归一化**：同一道题采 $G=16$ 个 solution，自动找出"哪些 reasoning path 更好"，不需要绝对 scale。
- **省 value model**：数学 prompt 多、reasoning 长，省一半显存能跑更大 batch。

### 5.4　代码（GRPO advantage + loss 核心 50 行）

```python
import torch
import torch.nn.functional as F

def grpo_loss(policy, ref_policy, batch, eps_clip=0.2, beta=0.04):
    """
    batch:
      input_ids:     [N, L]       N = sum_b G_b samples in the batch
      action_mask:   [N, L]
      old_log_probs: [N, L]
      rewards:       [N]          sequence-level reward (rule-based or RM)
      group_id:      [N]          which prompt each sample belongs to (long tensor)
    """
    rewards = batch["rewards"]
    gid = batch["group_id"].long()
    N = rewards.shape[0]

    # ── 组内归一化 (GRPO 核心) ──
    # 用 scatter 聚合每个 group 的 mean / std，不假设 group_id 已排序或等大小。
    num_groups = int(gid.max().item()) + 1
    counts = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, torch.ones_like(rewards))                              # [num_groups]
    sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, rewards)                                                # [num_groups]
    group_mean = sums / counts.clamp_min(1.0)                          # [num_groups]
    diff_sq = (rewards - group_mean[gid]) ** 2
    sq_sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, diff_sq)                                                # [num_groups]
    group_var = sq_sums / counts.clamp_min(1.0)
    group_std = group_var.sqrt()                                       # [num_groups]

    A = (rewards - group_mean[gid]) / (group_std[gid] + 1e-8)          # [N]
    A = A.unsqueeze(-1)                                                # [N, 1] 整段共享

    # ── log-prob ratio ──
    logits = policy(batch["input_ids"]).logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, tgt).squeeze(-1)
    new_log_probs = F.pad(new_log_probs, (1, 0))        # [B*G, L]
    mask = batch["action_mask"].float()

    ratio = torch.exp(new_log_probs - batch["old_log_probs"])

    # ── PPO-Clip surrogate（advantage 整段共享）──
    surr1 = ratio * A                                    # [N, L]
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * A

    # ── KL penalty (K3 estimator: KL ≈ exp(Δ) - Δ - 1) ──
    with torch.no_grad():
        ref_logits = ref_policy(batch["input_ids"]).logits[:, :-1]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_lp = ref_log_probs.gather(-1, tgt).squeeze(-1)
        ref_token_lp = F.pad(ref_token_lp, (1, 0))
    delta = ref_token_lp - new_log_probs                 # log(π_ref / π_θ)
    kl_per_token = torch.exp(delta) - delta - 1.0        # K3, non-negative [N, L]

    # 公式: (1/G) Σ_i (1/|y_i|) Σ_t [ min(...) - β·KL_{i,t} ]
    # 实现: 每个 sample 先按 response 长度 |y_i| 归一化, 再对 batch 取均值
    token_obj = torch.min(surr1, surr2) - beta * kl_per_token   # [N, L]
    seq_len = mask.sum(dim=-1).clamp_min(1.0)            # [N] = |y_i|
    per_seq = (token_obj * mask).sum(dim=-1) / seq_len   # [N]
    loss = -per_seq.mean()                               # 负号: 最大化 -> 最小化

    # 监控
    with torch.no_grad():
        policy_term = -(torch.min(surr1, surr2) * mask).sum(dim=-1) / seq_len
        kl_term = (kl_per_token * mask).sum(dim=-1) / seq_len
    return loss, {"policy": policy_term.mean().item(),
                  "kl": kl_term.mean().item(),
                  "reward_mean": rewards.mean().item(),
                  "advantage_std": A.squeeze(-1).std().item()}
```

> ⚠️ **GRPO 工程要点** —

- $G$ 一般取 8/16/32：太小组内统计噪声大，太大显存爆。
- `std` 为 0 的退化情况（组内 reward 全相同，all-pass / all-fail）：加 $\epsilon$ 或丢掉该组。
- **DAPO** (ByteDance Seed, 2025 arXiv 2503.14476) 对 GRPO 做了多项改进：clip higher (打破对称 clip)、dynamic sampling（丢全对全错 group）、token-level loss、overlong shaping，是 GRPO 的工业级进阶。
- **DeepSeek 在 R1 报告中也讨论了 GRPO 的限制**：组内 baseline 在 reward 方差小或 group size 小的情况下会失效；后续工作（如 VAPO、CISPO）尝试改进。

## §6 DPO 变体生态

DPO 之后 2024–2025 出现一大波改进，面试时常被问"X 和 Y 区别"。下表对**核心损失差异**做对比；记号统一为：$\pi$ = current policy, $\pi_\text{ref}$ = SFT model, $\beta$ = inverse temperature, $r$ = implicit reward。

| 方法 | 损失 | 关键变化 | 论文 |
| --- | --- | --- | --- |
| **DPO** | $-\log\sigma(\beta\log\tfrac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\tfrac{\pi(y_l)}{\pi_\text{ref}(y_l)})$ | 基础 | Rafailov 2023 NeurIPS |
| **IPO** | $(\log\tfrac{\pi(y_w)/\pi_\text{ref}(y_w)}{\pi(y_l)/\pi_\text{ref}(y_l)} - \tfrac{1}{2\beta})^2$ | sigmoid → squared loss，**防止 reward overfitting** | Azar 2024 AISTATS |
| **KTO** | per-sample sigmoid loss，only need binary good/bad，**不需 pair** | Kahneman-Tversky 启发，desirable/undesirable | Ethayarajh 2024 ICML |
| **SimPO** | $-\log\sigma(\beta\cdot\tfrac{1}{\lvert y_w\rvert}\log\pi(y_w) - \beta\cdot\tfrac{1}{\lvert y_l\rvert}\log\pi(y_l) - \gamma)$ | **去 reference**、length-normalized、加 margin $\gamma$ | Meng 2024 NeurIPS |
| **ORPO** | $\mathcal{L}_\text{SFT}(y_w) + \lambda\cdot\mathcal{L}_\text{OR}$，odds-ratio 偏好 | **SFT + preference 一阶段**，无需 reference | Hong 2024 EMNLP |
| **RLOO** | REINFORCE + leave-one-out baseline 从 $k$ 个 sample | online RL，无 value，强 baseline | Ahmadian 2024 ACL |
| **ReMax** | REINFORCE + greedy baseline（同 prompt greedy decode 作 baseline） | 进一步省 sample | Li 2024 ICML |

### 6.1　IPO：防止 reward overfit

Azar et al. 2024 *A General Theoretical Paradigm to Understand Learning from Human Preferences* 指出：Bradley-Terry + DPO 在偏好 deterministic（"$y_w$ 总是赢"）时会让 implicit reward 无限放大，过拟合。**IPO 把 sigmoid 换成 squared loss**：

$$\mathcal{L}_\text{IPO} = \mathbb{E}\!\left[\left(\log\frac{\pi(y_w)/\pi_\text{ref}(y_w)}{\pi(y_l)/\pi_\text{ref}(y_l)} - \frac{1}{2\beta}\right)^2\right]$$

直觉：希望 log-ratio 等于一个**固定 margin** $1/(2\beta)$，而不是越大越好。

### 6.2　KTO：no need for pairs

Ethayarajh et al. 2024 ICML *KTO: Model Alignment as Prospect Theoretic Optimization*。生产数据常常是 single label（"这条好"/"这条不好"），不是 pair。KTO 启发自前景理论（Kahneman-Tversky），对 desirable / undesirable 两类样本**对称地推开 reference point**：

记 implicit reward $\hat{r}_\theta(x, y) = \beta \log \tfrac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$，reference point $z_0 = \mathbb{E}_{x',\, y' \sim \pi_\theta(\cdot | x')}\!\big[\beta \cdot \text{KL}\!\big(\pi_\theta(\cdot | x') \,\|\, \pi_\text{ref}(\cdot | x')\big)\big]$（用 batch 内 mismatched pair 估计、且**不参与反传**——只当常数）。KTO 损失为：

$$\boxed{\;\mathcal{L}_\text{KTO}(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}} \big[\, w(y)\big(1 - v(x, y)\big)\,\big]\;}$$

其中 piecewise 定义（hat r 和 z0 已含 $\beta$，sigmoid 内**不再外乘** $\beta$）：

$$v(x, y) = \begin{cases}\sigma\!\big(\hat{r}_\theta(x, y) - z_0\big) & y \text{ desirable}\\[2pt] \sigma\!\big(z_0 - \hat{r}_\theta(x, y)\big) & y \text{ undesirable}\end{cases}$$

$$w(y) = \begin{cases}\lambda_D & y \text{ desirable}\\ \lambda_U & y \text{ undesirable}\end{cases}$$

直觉：desirable 样本希望 $\hat{r} > z_0$（推上去），undesirable 样本希望 $\hat{r} < z_0$（压下去）；$\sigma$ 给出"距离 reference point 多远"的概率打分。$\lambda_D, \lambda_U$ 是不平衡数据下的权重（如 desirable 多则 $\lambda_U > \lambda_D$）。当数据只有 single thumb up/down 时 KTO 比 DPO 实用。详细形式见 Ethayarajh et al. 2024 ICML、HuggingFace TRL `KTOTrainer` 实现。

### 6.3　SimPO：去 reference

Meng et al. 2024 NeurIPS *SimPO: Simple Preference Optimization with a Reference-Free Reward*。SimPO 观察：**DPO 推理时不用 $\pi_\text{ref}$，但训练时存（占显存）**。SimPO 直接把 implicit reward 改成 length-normalized log-prob：

$$r_\text{SimPO}(x, y) = \frac{\beta}{|y|}\log\pi(y|x)$$

损失：

$$\boxed{\;\mathcal{L}_\text{SimPO} = -\mathbb{E}\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi(y_w|x) - \frac{\beta}{|y_l|}\log\pi(y_l|x) - \gamma\right)\;}$$

其中 $\gamma$ 是 target reward margin（鼓励 $r_w - r_l \ge \gamma$，超出才不算 loss）。

> ✅ **SimPO 优势** —

- **训练时无需 $\pi_\text{ref}$**，省一份模型权重的显存。
- **Length-normalization** 缓解 DPO 的 length bias（DPO 倾向选长答案，因为它们 log-prob 更负，差更大）。
- 实验上 AlpacaEval-2 / Arena-Hard 上 SimPO 经常优于 DPO；但 reward-free 形式**没有 KL 锚定**，对 $\beta, \gamma$ 调参更敏感。

### 6.4　ORPO：SFT + 偏好一阶段

Hong et al. 2024 EMNLP *ORPO: Monolithic Preference Optimization without Reference Model*。ORPO 取消"先 SFT 再 DPO"两阶段，**一次训练同时做 SFT 和偏好**：

$$\mathcal{L}_\text{ORPO} = \mathcal{L}_\text{SFT}(y_w) - \lambda \cdot \log\sigma\!\left(\log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

其中 odds = $p / (1 - p)$。**无需 reference model**，但需要 chosen response 是高质量的（SFT 部分驱动）。

### 6.5　RLOO / ReMax

Ahmadian et al. 2024 ACL *Back to Basics: Revisiting REINFORCE Style Optimization*。在 LLM RLHF 上，**简单的 REINFORCE + 多采样 baseline 实测优于 PPO**：

- **RLOO**：对每 prompt 采 $k$ 个 response，每个的 baseline = 其余 $k-1$ 个的平均 reward（leave-one-out），$\hat{A}_i = r_i - \frac{1}{k-1}\sum_{j\ne i} r_j$。
- **ReMax** (Li et al. 2024 ICML)：baseline 直接用同 prompt 的 **greedy decode** reward，每次只需 1 sample + 1 greedy。

两者都**无 value model**，与 GRPO 同源——本质都是用**多 sample 或 greedy 的 baseline 替代 critic**。

> 💡 **GRPO vs RLOO vs ReMax 区分** —

- **GRPO**：组内 mean+std 归一化（$z$-score）+ PPO-clip。
- **RLOO**：组内 leave-one-out 均值做 baseline + REINFORCE。
- **ReMax**：greedy decode 做 baseline + REINFORCE。
- 三者哲学一致：**绕过 value model，用 sample baseline**；区别只在 baseline 形式。

## §7 Reward Modeling 进阶

### 7.1　PRM vs ORM

| 维度 | **ORM** (Outcome RM) | **PRM** (Process RM) |
| --- | --- | --- |
| **打分粒度** | 只看最终答案 | 每步 reasoning 都打分 |
| **标注成本** | 低（只看结果） | 高（每步都要标） |
| **奖励信号** | sparse（轨迹末尾） | dense（每步） |
| **数学推理表现** | 中 | **好** |
| **典型用法** | Best-of-N + ORM 重排 | PRM 用于 step-level search / RL |
| **代表论文** | InstructGPT (Ouyang 2022) | Let's Verify Step by Step (Lightman 2023 arXiv / 2024 ICLR) |

Lightman et al. *Let's Verify Step by Step* (OpenAI arXiv 2305.20050, 2023; ICLR 2024) 在 MATH 数据集上对比 ORM 和 PRM：

- PRM800K：80 万 step-level 标注（每步标 ✓ / ⚠️ / ✗）。
- 在 GPT-4 generator 上，PRM 重排 1860 个候选时 MATH 准确率 **78.2%**，明显高于 ORM 的 72.4%。

**Math-Shepherd** (Wang et al. 2024 ACL *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*) 用 **rollout-based 自动标注 PRM**：从每步出发跑 $K$ 次后续 generation，根据正确率自动给每步打 soft label，省去人工。

### 7.2　Constitutional AI / RLAIF

**Constitutional AI (Bai et al. 2022 Anthropic)** *Constitutional AI: Harmlessness from AI Feedback*：

1. **SL stage**：让 AI 根据"宪法"原则自己改写 harmful response。
2. **RL stage**：用 AI（而非人）做偏好标注 → 训 RM → PPO（即 RLAIF）。

整个 harmless 训练**无需人工偏好标注**，靠 LLM 自我评估。

**RLAIF (Lee et al. 2023 Google)** *RLAIF: Scaling RLHF with AI Feedback*：在 summarization、helpful dialogue 等任务上系统验证 AI feedback ≈ human feedback 效果。**条件**：LLM 评委足够强（GPT-4 / Claude-3 级），且 prompt 设计良好。

> ⚠️ **RLAIF 风险** — 如果评委 LLM 自己有偏见（如 sycophancy、长度偏好），偏见会被放大到 student model。生产中通常**混合使用**：rule-based（数学/代码）+ RM（一般质量）+ RLAIF（safety）。

### 7.3　Reward model ensemble & uncertainty

近几年（Coste 2024, Eisenstein 2024）发现：**single RM 容易被 hacking**，ensemble of RMs（5-7 个不同 seed 的 RM）配合 conservative aggregation（min / mean - $k\cdot$std）可以显著缓解 reward hacking 且不损失性能。代价是显存翻倍——所以工业部署多用 lightweight RM（如 6B 评 70B policy）。

## §8 Reward Hacking & 工程经验

Reward hacking 是 RL post-training 的核心痛点：model 找到 RM 的盲点拿高分，但人类觉得变差。常见症状：

| 现象 | 原因 | 缓解 |
| --- | --- | --- |
| **答案变长** | 长答案在 RM 训练数据里通常分高（信息量、礼貌套话） | length penalty / length normalize (SimPO) / 长度回归校准 |
| **谄媚 (sycophancy)** | 用户喜欢被认同 → RM 学到"agree better" | sycophancy probe + 对抗数据 |
| **重复 / 套话** | 句尾"Hope this helps!" 类型在 RM 数据里高分 | n-gram repetition penalty / 多样性 reward |
| **拒绝过度** | 安全 RM 学到"拒绝就安全" | helpfulness vs harmlessness 双 RM 平衡 |
| **格式 hack** | 用 markdown / bullet / emoji 拿分 | 格式 normalization |
| **数字操作** | 数学题硬编码常见答案分布 | rule-based reward 替代 RM |

### 8.1　核心缓解机制

1. **KL penalty** ($\beta \log\pi/\pi_\text{ref}$)：最基础的 anchor。
2. **Early stopping by KL budget**：跑到 $\text{KL}(\pi || \pi_\text{ref}) > K_\text{target}$ 时停。
3. **Reward model ensemble** (Coste 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*)：多个 RM 取 min 或 mean - std。
4. **Reward shaping**：把 reward 拆成多个项（helpfulness + length + diversity），单独 cap 每项。
5. **离线 + 在线混合**：先 DPO / KTO 拿 70 分，再 PPO + 强 RM 跑最后一公里。
6. **Composite reward**：rule-based + RM-based 加权，rule 部分不能被 hack。

### 8.2　Gao 2023 scaling law

Gao, Schulman, Hilton 2023 ICML *Scaling Laws for Reward Model Overoptimization*：

- Proxy reward (RM score) 随 KL 单调上升，但 gold reward 先升后降 (**inverted-U**)。
- Best-of-N 的 over-optimization 与 PPO 形式略不同：BoN 的 gold reward 在 $\sqrt{\text{KL}}$ 上接近线性 -减 + 线性，PPO 多了高阶项。
- 给出 RM 越大、过优化越慢的 scaling law，论文中也讨论了不同函数形式拟合。

**面试 takeaway**：reward hacking 不是 bug，是 RL 本质——proxy 与真目标分离的不可避免现象。

## §9 复杂度 / 资源 对比

### 9.1　四种范式资源对比

| 维度 | **PPO RLHF** | **DPO 系** | **GRPO** | **RLOO/ReMax** |
| --- | --- | --- | --- | --- |
| 需要 RM？ | ✅ frozen | ❌ implicit | ✅ frozen 或 rule | ✅ frozen 或 rule |
| 需要 value model？ | ✅ trainable | ❌ | ❌ | ❌ |
| 需要 reference policy？ | ✅ frozen | ✅ frozen (SimPO/ORPO 除外) | ✅ frozen | ✅ frozen 或不需 |
| Online sampling? | ✅ on-policy | ❌ offline | ✅ on-policy | ✅ on-policy |
| 模型副本数 | **4** (π, π_ref, RM, V) | **2** (π, π_ref) | **3** (π, π_ref, RM) | **3** |
| KL anchor | 显式 reward 中 | 隐式 in loss | 显式 loss 中 | 显式 reward 或 loss |
| 主要瓶颈 | 显存 + on-policy 慢 | 偏好数据质量 | sampling 多倍 | sampling 多倍 |
| 调参难度 | **高** ($\beta, \epsilon, \lambda, c_v, c_e$) | 中 ($\beta$) | 中 ($\beta, G$) | 低 ($\beta, k$) |
| 代表 | InstructGPT, Claude | Llama-3 instruct, Zephyr | DeepSeek-R1, Kimi-K1.5 | Cohere Command R |

### 9.2　LLM RLHF 训练显存粗算

以 7B base、fp16 forward + fp32 master、AdamW 为例（典型 RLHF）：

| 副本 | 用途 | bf16 权重 + 梯度 | fp32 (master + Adam m,v) | 合计 |
| --- | --- | --- | --- | --- |
| Policy $\pi_\theta$ | trainable | 14+14=28 GB | 28+28+28=84 GB | **112 GB** |
| Value $V_\phi$ | trainable | 28 GB | 84 GB | **112 GB** |
| Reference $\pi_\text{ref}$ | frozen | 14 GB | — | 14 GB |
| Reward $r_\psi$ | frozen | 14 GB | — | 14 GB |
| **合计** | | | | **~252 GB**（每 trainable 模型 16 B/param ×7B = 112 GB；与分布式训练篇一致） |

加上 activation、KV cache、generation buffer，单卡 80GB 极难放下；通常用 ZeRO-3 + offload 或多机分片。

**DPO 同 base**：只需 $\pi_\theta$ + $\pi_\text{ref}$ = 70 + 14 = **84 GB**（少一半）。

**GRPO 同 base**：$\pi_\theta$ + $\pi_\text{ref}$ + RM = 70 + 14 + 14 = **98 GB**（省 value 一份）。

### 9.3　训练吞吐对比

实测（开源 7B、8×H100，TRL/OpenRLHF 报告）：

- PPO：~50-100 token/s/GPU（瓶颈：on-policy generation + 4 副本）
- DPO：~500-1000 token/s/GPU（纯监督，无 generation）
- GRPO：~100-200 token/s/GPU（需要 generation，但少一个模型）
- SimPO：~600-1200 token/s/GPU（DPO 进一步去 reference）

DPO 训练快是因为**完全 offline**，几乎是 SFT 速度；PPO 慢主要在 rollout（generate 一段 response 才能算一次梯度）。

## §10 25 高频面试题

按难度分 3 档：L1 = 任何 LLM 工程岗都会问；L2 = research/alignment 团队会问；L3 = 顶级 lab / DeepSeek 量级团队的硬核题。每题点开看答案要点 + 易踩坑。

### L1 必会题（10 题）

<details>

<summary>Q1.RLHF 的三个阶段是什么？</summary>

- Stage 1: SFT（监督微调）
- Stage 2: RM（用 Bradley-Terry 训 reward model，frozen）
- Stage 3: PPO + KL penalty（policy 优化 RM 分，受 reference policy KL 约束）

跳掉 SFT 直接说"RM + PPO"；或漏掉 KL penalty。

</details>

<details>

<summary>Q2.PPO clipped surrogate 的公式是什么？</summary>

- $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ 是重要性比
- $L^\text{CLIP} = \mathbb{E}[\min(r_t A_t,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$
- 取 min 是悲观估计 (pessimistic bound)；advantage 为正时上限 $1+\epsilon$，为负时下限 $1-\epsilon$

说"clip 的是 reward"；或漏说 min 的作用。

</details>

<details>

<summary>Q3.GAE 是什么？$\lambda$ 怎么取？</summary>

- $A_t^\text{GAE} = \sum_l (\gamma\lambda)^l \delta_{t+l}$，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
- $\lambda = 0$ → TD(0)，偏差大方差小
- $\lambda = 1$ → Monte Carlo，偏差小方差大
- 典型 $\lambda = 0.95$，$\gamma = 0.99$

把 GAE 当成 advantage 本身（它是估计器）；或忘了 $\gamma\lambda$ 是联合衰减系数。

</details>

<details>

<summary>Q4.RM 的损失函数是什么？</summary>

- Bradley-Terry pairwise: $\mathcal{L} = -\mathbb{E}\log\sigma(r(x, y_w) - r(x, y_l))$
- 用 SFT model 初始化 backbone，最后接 scalar head
- 训完冻结，给 PPO 用

说 RM 用 BCE / MSE；或写成绝对 reward 监督。

</details>

<details>

<summary>Q5.DPO 损失公式？跟 RM 损失什么关系？</summary>

- $\mathcal{L}_\text{DPO} = -\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})$
- 形式跟 Bradley-Terry RM loss 一样，但**implicit reward** 是 $\beta\log(\pi/\pi_\text{ref})$，不是单独的 RM
- 起源：KL-regularized RLHF 最优解 $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$，反解得到 $r = \beta\log(\pi/\pi_\text{ref}) + \beta\log Z$，$\log Z$ 在 pairwise 差里消掉

只背公式不知道闭式推导；或说"DPO 不需要 reference"（错，需要 $\pi_\text{ref}$ 算 log-ratio）。

</details>

<details>

<summary>Q6.DPO 训练时需要哪些数据？</summary>

- 偏好对 $(x, y_w, y_l)$：同一 prompt 下，人或 AI 评判 $y_w \succ y_l$
- 需要 $\pi_\text{ref}$（一般是 SFT model）
- **不需要** RM、value model、online sampling

说要 reward scalar 标注；或忘了 $\pi_\text{ref}$。

</details>

<details>

<summary>Q7.GRPO 比 PPO 省了什么？</summary>

- **省掉 value model**：advantage 用组内归一化 $\hat{A}_i = (r_i - \bar{r}) / \sigma_r$
- 显存少一份；调参少一组 ($c_v$、value lr 不需要)
- 适合 LLM RL，因为 LLM 的 per-token value 很难学

说省了 RM（错，GRPO 还要 RM 或 rule-based reward）；或说 GRPO 是 offline（错，它是 on-policy）。

</details>

<details>

<summary>Q8.RLHF 中 KL penalty 起什么作用？</summary>

- Reward 上加 $-\beta\log\pi/\pi_\text{ref}$（per-token KL）
- 防止 policy 偏离 SFT 太远，缓解 reward hacking
- $\beta$ 太小 → 漂移，$\beta$ 太大 → 学不到东西
- 在 DPO 中通过 $\pi_\text{ref}$ 隐式实现

说 KL 是 RM 的一部分（错，KL 是 policy 与 ref 之间的，不涉及 RM）。

</details>

<details>

<summary>Q9.为什么 SFT 之后还要 RL/DPO？</summary>

- SFT 只能模仿正例，**学不到对比信号**（A 比 B 好）
- RM 把"哪个好"显式建模，policy 优化"获得高 RM 分"
- DPO 跳过 RM 但保留了同样的对比信号
- 实测 RLHF/DPO 后 helpfulness、harmlessness、honesty 都有显著提升（InstructGPT 报告）

只说"RL 比 SFT 强"；或不知道对比信号这一点。

</details>

<details>

<summary>Q10.Reward hacking 是什么？怎么缓解？</summary>

- Policy over-optimize RM，找到 RM 盲点拿高分，但人类觉得变差
- 典型症状：答案变长、谄媚、重复套话、过度拒绝
- 缓解：KL penalty、RM ensemble、length penalty、composite reward (rule + RM)、early stopping by KL budget

只说"加 KL"，不知道其他缓解；或不知道 length bias。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11.推导 DPO loss（从 KL-regularized RLHF 起）。</summary>

1. 目标：$\max_\pi \mathbb{E}[r] - \beta \text{KL}(\pi || \pi_\text{ref})$
2. Lagrangian + 对 $\pi$ 求导 $\Rightarrow \pi^*(y|x) = \frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp(r/\beta)$
3. 反解 $r(x, y) = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$
4. 代入 Bradley-Terry $P(y_w \succ y_l) = \sigma(r_w - r_l)$，**$\log Z$ 消掉**
5. 把 $\pi^*$ 替换为可学 $\pi_\theta$，对偏好数据做 NLL → DPO loss

直接背公式，问"$\log Z$ 为什么消掉"答不上来（因为它不依赖 $y$，差掉）。

</details>

<details>

<summary>Q12.PPO 的重要性比 $r_t = \pi_\theta/\pi_\text{old}$ 为什么需要？</summary>

- 标准 PG 是 on-policy 的，但 PPO 在同一 batch 上做多次 update（K 个 epoch）
- 第 2 次开始 $\pi_\theta \ne \pi_\text{old}$（即采样分布），需要 importance sampling 校正
- $\nabla \mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_{\pi_\text{old}}[(\pi_\theta/\pi_\text{old}) \nabla\log\pi_\theta A]$
- Clip 是为了防止 ratio 在多次 update 中飘太远

不知道 PPO 多次 update；或不清楚 IS 的角色。

</details>

<details>

<summary>Q13.DPO vs IPO 区别？什么时候用 IPO？</summary>

- DPO 用 sigmoid loss：偏好越极端，implicit reward 越大（无界）
- IPO 用 squared loss + 固定 margin $1/(2\beta)$：reward 有界
- **当偏好数据 deterministic**（每对都 $y_w$ 总赢）时 DPO 容易过拟合，IPO 更稳
- Azar et al. 2024 AISTATS 给出统一框架（$\Psi$PO）

说 IPO 是改进 DPO 的 hyperparameter；或不知道 sigmoid 在 deterministic preference 下的 issue。

</details>

<details>

<summary>Q14.SimPO 相比 DPO 改了什么？</summary>

- **去 reference**：$r = (\beta / |y|) \log\pi(y)$，不需要 $\pi_\text{ref}$
- **Length normalize**：除以 $|y|$，缓解 DPO 的长度偏好
- **Reward margin** $\gamma$：要求 $r_w - r_l \ge \gamma$，超出才停止 loss
- 显存省一倍；但失去 KL anchor，需要更小心调 $\beta, \gamma$

只说"去 reference"，不说 length-norm 和 margin；或不知道 SimPO 也是 contrastive。

</details>

<details>

<summary>Q15.GRPO 的 advantage 公式？为什么这样设计？</summary>

- $\hat{A}_i = (r_i - \text{mean}_g(r)) / (\text{std}_g(r) + \epsilon)$，$g$ 是同 prompt 的 group
- 整段 response 内所有 token 共享同一 $\hat{A}_i$
- **设计理由**：LLM token-level value 难学；用 sequence-level reward + 组内统计直接做 variance reduction
- 哲学和 RLOO（leave-one-out 均值）、ReMax（greedy baseline）一致：用 sample baseline 替代 critic

写成 per-token advantage（错，GRPO 整段共享）；或不知道 GRPO/RLOO/ReMax 的共性。

</details>

<details>

<summary>Q16.PRM vs ORM 差在哪？什么时候用 PRM？</summary>

- ORM 只在最终答案打分；PRM 每步 reasoning 打分
- PRM 在**多步推理任务**（数学、code）上明显更好（Lightman 2023, MATH 数据集 78% vs 72%）
- PRM 标注成本高 → Math-Shepherd 用 rollout-based 自动标
- PRM 既能做 search 重排，也能做 step-level RL（dense reward）

说 PRM 在所有任务都更好（不对，简单任务 ORM 足够）；或不知道 Math-Shepherd 的自动标注。

</details>

<details>

<summary>Q17.Constitutional AI / RLAIF 关键 idea？</summary>

- 用 AI（而非人）做偏好标注，省人工
- Constitutional AI (Bai 2022 Anthropic)：SL stage AI 自我改写 harmful → RL stage AI 偏好打分 → PPO
- RLAIF (Lee 2023 Google)：在 summarization 等任务系统验证 AI ≈ human
- **风险**：评委 LLM 有偏见会放大；通常和 rule-based / human RM 混合

说 RLAIF 就是 GPT-4 来标数据（不全对，CAI 强调"按宪法"自我评估）；或不知 CAI 早于 RLAIF 一年。

</details>

<details>

<summary>Q18.PPO 训练时为什么要在 reward 上加 KL，而不是在 loss 上？</summary>

- 加在 reward 上 → 通过 advantage 自然进入 PPO-clip surrogate，per-token 控制
- 加在 loss 上 → 整体 KL 约束，但失去 token-level resolution
- GRPO 反而把 KL 放在 loss 上（用 K3 estimator），因为 GRPO 不展开 per-token advantage
- 两种放法本质都是 KL anchor，但实现细节不同

混淆两种放法；或不知道 GRPO 的 K3 estimator。

</details>

<details>

<summary>Q19.RLHF 中 value model 怎么初始化？为什么？</summary>

- 通常用 **RM 或 SFT model 初始化** value backbone，加新 value head
- 用 RM 初始化的好处：RM 已经"理解 reward"，value 收敛更快
- 用 SFT 初始化的好处：value 与 policy 共享底层表征
- DeepSpeed-Chat 默认用 RM 初始化；TRL 默认用 SFT
- **共享 trunk vs 独立模型是 trade-off**：A3C/A2C 等经典 actor-critic 共享 trunk 省显存，但 policy 与 value loss scale 不同容易相互干扰；LLM RLHF 主流（DeepSpeed-Chat / TRL / OpenRLHF）选独立模型 + separate optimizer，稳定性优先

随机初始化 value（实践中不可行，太慢）；或说"value 必须共享 trunk"/"绝对不能共享"，两个极端都不对——是 trade-off 不是定理。

</details>

<details>

<summary>Q20.DPO 训练崩了（margin 不涨 / loss 不降），怎么诊断？</summary>

- 看 `chosen_logp` 和 `rejected_logp` 是否同时下降（典型 DPO 退化）
- 看 reference policy 是否正确加载（forgot to load → log-ratio 变成 raw log-prob）
- 看 $\beta$：太小（< 0.01）loss 几乎 = sigmoid 常数，太大 ($> 1$) 容易 collapse
- 看数据：$y_w$ 和 $y_l$ 是否真的差异显著
- 看 length：DPO 偏好长 response，若 $y_w$ 系统性比 $y_l$ 短就有 bug

只说"调 lr"；或不知道 likelihood-decrease-for-both 问题。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21.DeepSeek-R1 vs R1-Zero 的差异？为什么需要 cold-start？</summary>

- **R1-Zero**：从 pretrain base 直接跑 GRPO + rule-based reward（数学正确 + 格式），**无 SFT**
  - 优点：emergent 长 CoT，证明 RL 能自激发 reasoning
  - 缺点：可读性差、混语言、格式不稳定
- **R1**：cold-start SFT (几千条高质量 reasoning) → reasoning RL → SFT → general RL（多阶段）
  - 修复 R1-Zero 的可读性问题
  - SOTA 数学 / 代码推理
- **R1-Zero 重要性**：证明 RL 单独能激发 reasoning，不必先 SFT；后续 self-play / pure-RL 路线的基础

只说 R1 是 R1-Zero 的改进；或不知道 cold-start 修的是什么（可读性，不是性能）。

</details>

<details>

<summary>Q22.GAE 中 $\gamma$ 和 $\lambda$ 谁更重要？LLM RLHF 中通常怎么取？</summary>

- $\gamma$ 是 reward 折扣（任务级），$\lambda$ 是 TD 估计的 trace-decay（算法级）
- **LLM RLHF 中 $\gamma = 1$**（不折扣，因为 reward 只在 terminal，折扣会让 early token 收到信号过小）
- $\lambda = 0.95$ 仍然有用：在 token 维度做 bias-variance 折中
- $\gamma\lambda$ 联合衰减是 GAE 的有效 trace-decay；$\gamma=1, \lambda=0.95$ 时实际 trace 约 20 token
- 若 $\gamma = 1, \lambda = 1$ → GAE 退化为 MC，与"reward-to-go - V baseline" 等价

不假思索照搬 game RL 的 $\gamma = 0.99$，不知道 LLM 中 $\gamma = 1$ 是更主流选择（在 terminal-reward + 短上下文 generation 下更合理）；或反过来认定 $\gamma$ 必须为 1——具体仍取决于 reward 是否 dense / 是否长 horizon，是工程选择不是定理。

</details>

<details>

<summary>Q23.如何设计一个 RL 框架同时支持 DPO / PPO / GRPO？</summary>

抽象出三层：

1. **Data layer**：preference pair (DPO) / prompt + group (GRPO) / prompt only (PPO) → 统一 batch interface
2. **Trajectory layer**：on-policy rollout (PPO/GRPO) vs offline (DPO)
   - PPO/GRPO 需要 vLLM/sgl 推理加速 + 异步 trajectory queue
   - DPO 完全 dataloader
3. **Loss layer**：插件化
   - PPO: clipped surrogate + value + entropy + GAE
   - DPO: log-ratio sigmoid
   - GRPO: clipped surrogate + group-normalized advantage + K3 KL
4. **Reference 管理**：DPO/PPO/GRPO 都需 $\pi_\text{ref}$，统一封装为 frozen + lazy load
5. **RM/rule reward**：插件化 reward backend（neural RM、unit-test、math checker）

参考：TRL、OpenRLHF、verl (字节)、SimplePO。**verl 是 GRPO/RLOO/DAPO 的主流实现框架**。

只列 PPO；或没考虑 trajectory queue 和异步 rollout。

</details>

<details>

<summary>Q24.如果 RM 和 policy 一起训会怎样？为什么主流 RLHF 不这么做？</summary>

- 直觉：让 RM 也在线学，给 RM 更多新分布的数据 → adversarial training
- **问题 1**：RM 训练目标和 policy 训练目标耦合，loss landscape 不稳定（GAN-like）
- **问题 2**：RM 训练需要 fresh human label，否则会被 policy 拖着 drift
- **问题 3**：RM 信号变化太快 → policy 学到的"什么是好" 不一致
- 折中方案：**iterative preference optimization** (Xu et al. 2023 arXiv 2312.16682 *Some things are more CRINGE than others: Iterative Preference Optimization with the Pairwise Cringe Loss*)、**self-rewarding LM** (Yuan et al. 2024 *Self-Rewarding Language Models*) — 每轮用最新 policy 生成响应 + 重新打分（同模型 / 人 / GPT-4），让 RM 隐式更新
- **OpenAI / Anthropic 主流：固定 RM，多轮 PPO**；可视化解释更稳定

只说"会不稳定"，不给具体原因；或不知道 iterative DPO / self-rewarding 这条 emerging path。

</details>

<details>

<summary>Q25.如果让你设计 next-gen 后训练算法，你会怎么改进 GRPO？</summary>

可能的方向（任答 2-3 个，且要有 trade-off 讨论）：

- **Adaptive group size**：reward variance 大时小 $G$，小时大 $G$（DAPO dynamic sampling）
- **Token-level credit assignment**：GRPO 整段共享 advantage → 长 response 信号稀释。可以引入 lightweight critic（VAPO）或基于 step-level reward（PRM）的 partial credit
- **Off-policy correction**：GRPO 是 on-policy，rollout 慢；引入 V-trace / Retrace 让 stale samples 也能用
- **Multi-task reward**：rule + RM + style 合成 reward，每个维度独立归一化避免 reward scale 不平衡
- **Reward model uncertainty**：用 ensemble RM 的 min 或 mean - std 防止 over-optimization
- **Process reward integration**：PRM 给 step-level dense advantage，与 group baseline 联合（Math-Shepherd + GRPO 路线）
- **CISPO / 截断 IS** (Minimax 2025)：解决 GRPO 在 negative advantage 大 ratio 时的稳定性
- **DAPO** (ByteDance 2025)：clip higher + dynamic sampling + token-level loss + overlong shaping，已开源 verl 实现

只罗列"加 attention / 加更多模型"，没 trade-off；或不知道 DAPO / VAPO / CISPO 等 GRPO 后续工作。

</details>

## §A 附录：参考文献清单

按章节分组，全部经过 codex (gpt-5.5 xhigh) reviewer 验证作者-年份-会议正确：

**PPO / RL 基础**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms*
- Schulman et al. 2016 ICLR *High-Dimensional Continuous Control Using GAE*
- Schulman 2020 blog *Approximating KL Divergence*（K3 estimator 来源）

**RLHF**

- Christiano et al. 2017 NeurIPS *Deep Reinforcement Learning from Human Preferences*（pairwise preference + RM 第一篇）
- Stiennon et al. 2020 NeurIPS *Learning to Summarize from Human Feedback*（OpenAI summarization 报告）
- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback*（InstructGPT）
- Bai et al. 2022 Anthropic arXiv 2204.05862 *Training a Helpful and Harmless Assistant with RLHF*
- Bai et al. 2022 Anthropic arXiv 2212.08073 *Constitutional AI*
- Lee et al. 2023 Google arXiv 2309.00267 *RLAIF: Scaling RLHF with AI Feedback*

**DPO 系**

- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization*
- Azar et al. 2024 AISTATS *A General Theoretical Paradigm to Understand Learning from Human Preferences*（IPO）
- Ethayarajh et al. 2024 ICML *KTO: Model Alignment as Prospect Theoretic Optimization*
- Meng et al. 2024 NeurIPS *SimPO: Simple Preference Optimization with a Reference-Free Reward*
- Hong et al. 2024 EMNLP *ORPO: Monolithic Preference Optimization without Reference Model*
- Tang et al. 2024 ICML *Generalized Preference Optimization* (统一 DPO/IPO/SLiC 框架)

**Critic-free RL**

- Ahmadian et al. 2024 ACL *Back to Basics: Revisiting REINFORCE Style Optimization* (RLOO)
- Li et al. 2024 ICML *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method*
- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（GRPO 提出）
- DeepSeek-AI 2025 arXiv 2501.12948 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- Yu et al. 2025 ByteDance arXiv 2503.14476 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*

**Reward Modeling**

- Lightman et al. 2024 ICLR / OpenAI arXiv 2305.20050 (2023) *Let's Verify Step by Step*（PRM800K, PRM vs ORM）
- Wang et al. 2024 ACL *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*
- Eisenstein et al. 2024 COLM (arXiv 2312.09244, 2023) *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking*
- Gao, Schulman, Hilton 2023 ICML *Scaling Laws for Reward Model Overoptimization*

**Iterative / Self-rewarding**

- Xu et al. 2023 arXiv 2312.16682 *Some things are more CRINGE than others: Iterative Preference Optimization with the Pairwise Cringe Loss*
- Yuan et al. 2024 ICML *Self-Rewarding Language Models*
- Pal et al. 2024 arXiv 2402.13228 *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*

代码框架：TRL (HuggingFace)、OpenRLHF、verl (ByteDance Seed)、Axolotl、LLaMA-Factory、SimplePO。**verl 是当前 GRPO/RLOO/DAPO 主流实现**，DeepSeek 系工作多基于 verl 复现。
