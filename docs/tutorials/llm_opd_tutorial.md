## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 LLM OPD (On-Policy Distillation)** — 2025-2026 post-training 最热门的"便宜版 RL"范式，一页拿下面试核心要点（详见 §1-§9 推导 + §10 25 高频题）。

1. **OPD = On-Policy Distillation**：student 用自己当前 policy **采 trajectory**，teacher 在 student 自己访问到的状态上给 **per-token 监督信号**（KL / log-prob / soft label）。它**不是** DPO 的笔误，也**不是** Online Preference Distillation——这一术语在 LLM 上下文专指 "on-policy distillation"。代表 paper：MiniLLM (Gu 2024 ICLR, arXiv 2306.08543)、GKD (Agarwal 2024 ICLR, arXiv 2306.13649)、Thinking Machines blog (Lu 2025-10-27)、Qwen3 Technical Report (May 2025, arXiv 2505.09388)、Survey (Song & Zheng 2026, arXiv 2604.00626)。

2. **核心 loss（一行能写下来）**：sample $y \sim \pi_\theta(\cdot|x)$，对每个 token 算 reverse KL，定义为 $L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta}[\sum_t D_{\text{KL}}(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t}))]$。注意期望下标是 $y \sim \pi_\theta$（**student 自己采**），KL 方向是 $\pi_\theta \| \pi_T$（**reverse / mode-seeking**）——这两点是 OPD 与 vanilla KD 的本质区别。完整公式：
   $$L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]$$

3. **三种 distillation 范式速记**：
   - **SFT / Hard distillation**：teacher 生成 $\hat y$，student 在 $\hat y$ 上做 cross-entropy（off-policy + hard label）
   - **Vanilla KD / Soft distillation (Hinton 2015)**：teacher 生成 $\hat y$，student match teacher 的 soft logits（off-policy + soft label，forward KL，mode-covering）
   - **OPD**：**student 自己生成 $y$**，teacher 在 $y$ 上算 logits 给 KL 信号（on-policy + soft label，reverse KL，mode-seeking）

4. **为什么 on-policy 关键**：off-policy distillation 有 exposure bias——training 时 teacher 的 prefix 都"完美"，但推理时 student 遇到自己的错误前缀**没见过**，错误 compound（误差随序列长度 $L$ 大约按 $O(L^2)$ 放大，见 §1.3）。OPD 把训练分布 = 推理分布对齐，把 compound 误差从 $O(L^2)$ 压到 $O(L)$。

5. **OPD vs RL（核心卖点）**：Thinking Machines blog 给出经验法则——RL 每条 trajectory 教 $O(1)$ bits（一个 outcome reward），OPD 教 $O(N)$ bits（每个 token 都有 teacher 的 soft label 监督）。在 Qwen3-8B-Base + Qwen3-32B-teacher 数学推理实验上，OPD **匹配 RL 在 AIME'24 的 gain，但 compute 降到约 1/9-1/30**。Qwen3 Tech Report 也独立报告 OPD ≈ RL 性能但 GPU 时长**只需 1/10**。

6. **与 DPO / GRPO 的关系**：(a) **DPO** 用 offline preference pair 做 closed-form RLHF，**没有 teacher logits，没有 student rollout**——与 OPD 几乎正交；(b) **GRPO** 用 group-relative advantage 做 on-policy RL，**有 student rollout 但 reward 是 sparse outcome**；(c) **OPD = GRPO 的"dense teacher KL 替代 sparse outcome reward"** 版本。Survey (Song 2026) 给出统一视角：OPD ≈ "KL-constrained RL with $\beta \to \infty$ 且 token-level reward 来自 teacher log-prob"。

7. **2025-2026 工业采用**：Qwen3（off-policy + on-policy 两阶段蒸馏 small models）、DeepSeek-R1 distillation series（off-policy SFT 为主，但后续 follow-up 用 OPD）、MiMo-V2、Kimi distill 系列（Gemma 2/3 的官方蒸馏方法是固定数据集上的 off-policy soft-target KD，不含 student self-rollout，不应归为 OPD 采用案例，见 §5.3）。Thinking Machines (Murati 团队 2025) 把 OPD 包装成"便宜 RL 替代品"路线。

8. **三个最常考的 footgun**：(a) **Length inflation / truncation collapse**——student rollout 越训越长，触发 truncation 后 gradient 偏置，validation 暴跌（Demystifying OPD 2026, arXiv 2604.08527）；(b) **Reverse KL mode collapse**——student 收敛到 teacher 单一模，生成多样性塌缩；(c) **Teacher / student gap 太大**：student 完全采不到 teacher 概率高的 region——注意这不是"KL 信号消失"，采到的极少数 teacher 低概率 token 反而会产生数值极端的惩罚（"prefix teach, suffix fade" 现象，arXiv 2605.13643，详见 §7.5）。

## §1 直觉：为什么需要 On-Policy Distillation

### 1.1　Knowledge Distillation 简史（Hinton → DistilBERT → MiniLLM → OPD）

蒸馏的核心思想从 2015 年没变过：**用 teacher 的 soft label 训 student**。但在 LLM autoregressive 生成上，这个思想有一个隐藏假设需要重新审视——**训练时见到的 prefix 来自哪个分布**。

| 年份 | 方法 | 训练分布 | KL 方向 | 适用 |
|---|---|---|---|---|
| 2015 | Hinton soft target | teacher 在 dataset 上 forward | forward KL $D(\pi_T \,\Vert\, \pi_\theta)$ | 分类 / 检测，单步预测 |
| 2019 | DistilBERT (Sanh) | dataset 上 teacher logit | forward KL + MLM cross-entropy | encoder 类，单步预测 |
| 2016 | Seq-level KD (Kim & Rush 2016) | teacher beam search 的 $\hat y$ | hard token CE on $\hat y$ | NMT，autoregressive 但仍 off-policy |
| 2023 | MiniLLM (Gu, ICLR 2024) | **student rollout** $y \sim \pi_\theta$ | **reverse KL** $D(\pi_\theta \,\Vert\, \pi_T)$ | LLM instruction following |
| 2023 | GKD (Agarwal, ICLR 2024) | mix student rollout + dataset | generalized JSD（forward/reverse 插值） | LLM seq-to-seq |
| 2024-2026 | Qwen3 / Thinking Machines / Survey | student rollout + token-level teacher logit | reverse KL 为主 | 生产级 LLM post-training |
| 2025-01 | DeepSeek-R1-Distill 官方 pipeline | teacher 离线生成 800K trajectory | n/a（纯 off-policy CE，不含 KL 信号，见 §3.4） | 归入 Off-policy SFT KD 类别更准确，社区 follow-up 才会用 OPD 续训 |

关键演化：**off-policy → on-policy** 这一步是 LLM 蒸馏从"模仿数据集"升级成"模仿决策过程"的分水岭。

### 1.2　Forward KL vs Reverse KL：mode-covering vs mode-seeking

vanilla KD (Hinton) 用 forward KL：

$$D_{\text{KL}}(\pi_T \,\|\, \pi_\theta) = \sum_y \pi_T(y) \log \frac{\pi_T(y)}{\pi_\theta(y)}$$

期望在 $\pi_T$ 下取，**只有 teacher 给高概率的地方贡献 loss**。如果 $\pi_T(y) > 0$ 但 $\pi_\theta(y) = 0$，loss 爆炸 → student 被迫**覆盖 teacher 的每一个 mode**（mode-covering / mass-covering），即使有些 mode 对 student 来说不可达，结果是 student 把概率分散到很多 token 上，生成"安全但平庸"的回答。

MiniLLM 改用 reverse KL：

$$D_{\text{KL}}(\pi_\theta \,\|\, \pi_T) = \sum_y \pi_\theta(y) \log \frac{\pi_\theta(y)}{\pi_T(y)}$$

期望在 $\pi_\theta$ 下取，**只有 student 自己采到的地方贡献 loss**。如果 $\pi_T(y) = 0$ 但 $\pi_\theta(y) > 0$，loss 爆炸 → student 被迫**避开 teacher 认为不可能的 token**（mode-seeking / zero-forcing），结果 student 收敛到 teacher 的某一个高质量 mode 上，生成"sharp 且 coherent"的回答。

> 💡 **直观差异** — 想象 teacher 是一个 "有 3 个等价好答案" 的双峰分布：

- **forward KL** 让 student 学一个 trimodal 平均的分布，每个 mode 都有概率但每个都不够 sharp
- **reverse KL** 让 student 锁定其中一个 mode（哪个 mode 由 init + sampling 决定），生成最自信、最贴近 teacher 决策的那条路径

对 LLM 生成任务这一般是好事——我们要的是"流畅、自信、正确"的输出，不是"平均、模糊、平庸"的输出。这也是 MiniLLM 第一个把 reverse KL 用到 LLM 蒸馏的关键洞察。

### 1.3　Exposure bias：off-policy 蒸馏的结构缺陷

考虑一个 autoregressive 生成任务，序列长度 $L$。off-policy 蒸馏（包括 SFT / vanilla KD / seq-level KD）训练时 prefix 来自 teacher 或 dataset；推理时 prefix 来自 student 自己。

假设单步错误率（student 与 teacher 在某个 prefix 上偏离的概率）为 $\epsilon$。off-policy 训练只在 teacher prefix 上"教过"student，所以推理时 student 一旦走错一步，**后续步骤进入了训练中没见过的状态分布**——student 的错误率不再是 $\epsilon$，而是 $\epsilon' > \epsilon$（甚至跳到 0.5+ 的随机水平）。

Bagnell 2010 在 imitation learning 上证明：**off-policy 模仿的累积误差是 $O(L^2 \epsilon)$**（compound error），而 on-policy 监督（如 DAgger）能把累积误差压到 $O(L \epsilon)$。这正是 OPD 在 LLM 上的理论动机——把训练 state 分布对齐到推理 state 分布。DAgger 达到 $O(L\epsilon)$ 并非无条件成立，依赖两个前提：expert 可在 learner 访问到的任意 state 上被查询、且所用在线学习子程序满足 no-regret 保证。

Survey (Song & Zheng 2026 arXiv 2604.00626) 在 §3 复述了这个结果：
> "off-policy distillation 的 exposure bias 随序列长度大约按平方放大；on-policy distillation 把累积误差线性化。"

这条理论保证是 OPD 在 long-CoT、agent、code 这些**长序列 + 高复杂度任务**上特别有效的根本原因。

### 1.4　Bit-rate 视角：dense teacher signal vs sparse RL reward

Thinking Machines blog (Lu 2025-10) 给了一个简洁的 information-theoretic 直觉：

| 范式 | 每条 trajectory 提供的监督 bit 数 |
|---|---|
| **RL with outcome reward** | $O(1)$ —— 一个 scalar reward（典型 $\{0, 1\}$ 或 [-1, 1] 区间） |
| **OPD with per-token teacher KL** | $O(N \log V)$ —— $N$ 个 token × 每个 token 对 vocab $V$ 的完整分布 |

换句话说，OPD 在每个 token 都"教" student 一整个分布，而 RL 只在序列末尾"教"一个数。在数学推理、code、long-form QA 这些需要 dense supervision 的任务上，OPD 的 compute efficiency（达到同等准确率所需的 FLOPs）显著高于 RL——这是 FLOPs/compute 维度的效率，与"需要多少条 rollout 样本"的 sample efficiency 是不同的量，两者可能不同步变化。这也解释了为什么 Qwen3 / Thinking Machines 都报告 OPD 达到 RL 性能但 compute 降一个量级。

## §2 OPD 的精确定义与核心公式

### 2.1　形式化定义

给定：
- **prompt 分布** $D$（如 dataset 中 instruction 的分布）
- **teacher policy** $\pi_T$（固定 frozen，参数量 $\gg$ student）
- **student policy** $\pi_\theta$（可训练）

OPD 的目标函数（一般形式，from Survey arXiv 2604.00626 §2.1）：

$$\boxed{\;L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D}\,\mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\!\left[\sum_{t=1}^{|y|} D_f\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]\;}$$

其中 $D_f$ 是某个 $f$-divergence，常见选择：

| Divergence | 公式 | 来源 / 偏好 |
|---|---|---|
| **Reverse KL** | $D(\pi_\theta \,\Vert\, \pi_T) = \sum_v \pi_\theta(v) \log \frac{\pi_\theta(v)}{\pi_T(v)}$ | MiniLLM、Thinking Machines blog、Qwen3 主用 |
| **Forward KL** | $D(\pi_T \,\Vert\, \pi_\theta)$（但 sample 时仍从 $\pi_\theta$） | "Sampled-token OPD"，简单稳定 |
| **JSD** | $\frac{1}{2}D(\pi_\theta \,\Vert\, M) + \frac{1}{2}D(\pi_T \,\Vert\, M)$, $M = \frac{1}{2}(\pi_\theta + \pi_T)$ | GKD 默认（对称、bounded） |
| **Generalized JSD** | $D_\beta = \beta D(\pi_\theta \,\Vert\, M_\beta) + (1-\beta) D(\pi_T \,\Vert\, M_\beta)$, $M_\beta = \beta \pi_\theta + (1-\beta)\pi_T$ | GKD 的 $\beta$ 插值 |
| **Total Variation** | $\frac{1}{2}\sum_v \lvert\pi_\theta(v) - \pi_T(v)\rvert$ | 较少用，bounded but non-smooth |

> ⚠️ **Generalized JSD 的端点是极限，不是取值** — 直接代入 $\beta=0$：$M_0=\pi_T$，两项分别是 $0\cdot D(\pi_\theta\|\pi_T)$ 与 $1\cdot D(\pi_T\|\pi_T)=0$，即 $D_{\beta=0}=0$；同理 $D_{\beta=1}=0$。所以严格来说 $D_\beta$ 在两端点都恒为 0，"forward/reverse KL" 只能理解为**单侧极限**：$\lim_{\beta\to 0^+} D_\beta/\beta = D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)$（reverse KL），$\lim_{\beta\to 1^-} D_\beta/(1-\beta) = D_{\text{KL}}(\pi_T\,\Vert\,\pi_\theta)$（forward KL）——注意这个极限方向与直觉相反：$\beta\to 0$ 对应 reverse KL，$\beta\to 1$ 对应 forward KL（详见 §10 L2-2 的更正表述）。

**核心点**：不管 $D_f$ 是哪个，**期望下标必须包含 $y \sim \pi_\theta$**——student 自己采，这是 "on-policy" 的定义。

### 2.2　Reverse KL 展开 + token-level loss

把 reverse KL 在每个 prefix $s_t = (x, y_{<t})$ 展开：

$$D_{\text{KL}}\!\big(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t)\big) = \sum_{v=1}^{V} \pi_\theta(v|s_t) \left[\log \pi_\theta(v|s_t) - \log \pi_T(v|s_t)\right]$$

这是**完整 KL**（"full-distribution" 形式），需要 teacher 在完整 rollout 序列上做一次 teacher-forced forward（利用 causal self-attention 并行算出所有位置的 logits），而非对每个 prefix 长度单独跑一次前向。计算成本：$O(B \cdot L \cdot V)$ 的 teacher forward + softmax（$L$ 个位置共用同一次 forward 调用，不是 $L$ 次独立前向）。

实践中常用 **"sampled-token" 近似**——只在 student 实际采到的 token $v = y_t$ 上算：

$$\hat L_t = \log \pi_\theta(y_t|s_t) - \log \pi_T(y_t|s_t)$$

这是 reverse KL 的 single-sample Monte-Carlo 估计（$\mathbb{E}_{v \sim \pi_\theta}[\log \pi_\theta(v) - \log \pi_T(v)] \approx \log \pi_\theta(y_t) - \log \pi_T(y_t)$ when $y_t \sim \pi_\theta$）。这种形式称为 **per-token reverse KL**，是 Thinking Machines blog 与 Tinker cookbook 实现采用的版本。

> ⚠️ **Sampled-token KL ≠ 完整 KL** — sampled-token 是无偏 estimator，但**方差大**，因为只看一个 token 而不是整个 vocab 分布。两种形式各有偏好：

- **完整 KL**：信息密度高、方差小，但每步要 teacher 做一次 vocab-level forward（昂贵，尤其 teacher 大）
- **Sampled-token KL**：只需从结果里 `gather` 出 $\log \pi_T(y_t)$ 一个数，节省的是这个 $[B,L,V]$ logits 张量的**存储/传输**开销——但 teacher 仍要完整跑一遍 $V$ 维投影 + softmax 归一化才能拿到这一个精确值，**计算量（FLOPs）并没有减少**；换来的是方差大，可以加 control variate baseline（见 §4.4）降方差

### 2.3　两种实现路线：full-vocab supervised KL vs REINFORCE

OPD 的"理论目标"是 student 在自己访问到的 state 分布 $\rho_{\pi_\theta}$ 上最小化 per-state KL：

$$L_{\text{OPD}}(\theta) = \mathbb{E}_{s \sim \rho_{\pi_\theta}}\!\left[D_{\text{KL}}\!\big(\pi_\theta(\cdot|s)\,\|\,\pi_T(\cdot|s)\big)\right]$$

但 $\theta$ 同时出现在 (a) 内层 KL 和 (b) **state visitation** $\rho_{\pi_\theta}$ 里。**产业实现** 选择把这两个 $\theta$-依赖**解耦** —— 这才是 MiniLLM / GKD / Thinking Machines / Qwen3 真实做的事：

**路线 A：Full-vocab supervised KL + stop-grad rollouts（教科书清晰路径，teacher full logits 可用时的最简形式）**

把 rollout 当作"behavior policy"产生的数据（**对 $\theta$ stop-grad**），训练只反传 KL 内层：

$$\boxed{\;L_{\text{OPD}}^{\text{A}}(\theta) = \mathbb{E}_{s_t \sim \text{rollout}(\pi_{\theta^-})}\!\left[\sum_t \sum_{v \in V} \pi_\theta(v|s_t) \log\frac{\pi_\theta(v|s_t)}{\pi_T(v|s_t)}\right]\;}$$

这里 $\pi_{\theta^-}$ 表示 rollout 阶段对 $\theta$ stop-grad（与 PPO 的 old policy 同思路）；内层 full-vocab 求和**直接可微**，autograd 即可，**不需要 REINFORCE**。这是教程下方 §4.1 代码采用的形式。

> 💡 **直觉** — Full-vocab KL 在每个 student-visited prefix $s_t$ 上让 $\pi_\theta(\cdot|s_t)$ 对齐 $\pi_T(\cdot|s_t)$ 整条分布，不仅看 sampled token。信号密度 $O(\log V)$，但每步要 teacher forward 一次 logits（昂贵）。

> ⚠️ **实现层归属** — 不要把 Route A 等同于 "Tinker / Thinking Machines blog 默认"。Thinking Machines 的开源实现 (Tinker) 实际默认走 **Route B 的 importance-sampling 变体**（sampled-token logprob + negative-KL advantage，对应 `train_on_policy.py` 的 `loss_fn="importance_sampling"` + `incorporate_kl_penalty`）；**MiniLLM** (Gu 2024) 也是 sampled-token + REINFORCE 形式的 trajectory PG（含 single-step decomposition / length norm / teacher-mixed sampling 等稳定 trick）。Route A 的 full-vocab autograd 是**教学上最干净的形式**（一行 PyTorch 反传，方差为零），生产场景里当 teacher full logits 可用时也是简洁正确的实现，但它**不**是哪个具体大 lab 的 production 默认。两者**期望等价**，方差/工程成本不同——Route A 优势：零方差、不需 IS clip。Route B 优势：与 PPO/GRPO 共享 sampled-token 接口、节约 teacher vocab memory。（这里"零方差"特指：给定固定 prefix $s_t$ 后，内层 full-vocab KL 是闭式求和，不需要再对 Route B1 那样的采样估计做蒙特卡洛，因而消除了同一状态下的采样估计方差；训练流水线整体仍因 rollout 阶段 student 自身采样而带有跨 batch 的方差，并非"整条训练 pipeline 零方差"。）

**路线 B：Sampled-token REINFORCE / importance-sampling estimator（Tinker 默认 / MiniLLM trajectory PG 形式）**

Route B 需要先**讲清两种不同的 gradient 计算**，文献中经常混淆：

**(B1) Fixed-state inner KL** — 假设 prefix $s_t$ 给定，只对内层 $D_{\text{KL}}(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t))$ 做 REINFORCE 估计：
$$\nabla_\theta D_{\text{KL}}(s_t) = \mathbb{E}_{y_t \sim \pi_\theta(\cdot|s_t)}\!\big[\nabla_\theta \log \pi_\theta(y_t|s_t) \cdot G_t^{\text{detach}}\big],\quad G_t = \log\tfrac{\pi_\theta(y_t|s_t)}{\pi_T(y_t|s_t)}.$$
这是 §4.4 vOPD estimator 的形式，**与 trajectory 上的 state visitation 无关**。

**(B2) Trajectory objective with state visitation** — 若要严格保留 $\rho_{\pi_\theta}$ 对 $\theta$ 的依赖，policy gradient 要求 **return-to-go**（未来所有 KL cost 的累加）作为 score-function 权重：
$$\nabla_\theta\,\mathbb{E}_\tau\!\Big[\sum_t D_{\text{KL}}(s_t)\Big] = \mathbb{E}_\tau\!\Big[\sum_u \nabla_\theta \log\pi_\theta(a_u|s_u)\cdot \underbrace{\sum_{t \ge u} D_{\text{KL}}(s_t)^{\text{detach}}}_{\text{return-to-go}}\Big] + \mathbb{E}_\tau\!\Big[\sum_t \nabla_\theta D_{\text{KL}}(s_t)\Big].$$
注意 score-function 权重不是同 token $G_t$，**是后续所有 step 的 KL 之和**。

> ⚠️ **完整 B2 几乎没人原样做，但 MiniLLM 不是简单 semi-gradient**。MiniLLM (Gu et al. 2024) 实际的 policy gradient 用**带 future log-ratio return 的单步分解**（single-step decomposition）：$R_t = \sum_{t' \ge t} r_{t'}$，$r_{t'} = \log\frac{\pi_T(y_{t'}|y_{<t'})}{\pi_\theta(y_{t'}|y_{<t'})}$——这本质是 B2（return-to-go）的**方差削减近似**，不是仅对当前 state 做 fixed-state KL 的 B1，也不是对 state visitation 整体 stop-grad 的 Route A。完整、未削减的 B2 由于 return-to-go 跨 step 累加，方差极大、几乎无法稳定训练，才是真正"没人做"的部分；MiniLLM 走的是介于两者之间的实用折中。

**统一的实践规则**：
- **Route A**（教学清晰，§4.1）：full-vocab KL + stop-grad rollouts，直接 autograd，**不需要 REINFORCE**。零方差，但需要 teacher full logits。
- **Route B1**（§4.4 vOPD-style, Tinker / MiniLLM 默认）：sampled-token REINFORCE / IS estimator + control variate。期望等价于 A，方差更大但适配 sampled-token 接口、节约 teacher vocab memory、支持 logprob-only（灰盒）teacher——注意仍需 teacher 吐出 $\log \pi_T(y_t|s_t)$，真正 sample-only 的黑盒 teacher（无 logprob）需要 §6.5 的专门 Black-Box OPD 方法。
- **不能**对 sampled-token 的 $\log(\pi_\theta/\pi_T)$ 直接 `.backward()` —— 那是 pathwise gradient + 固定 sampled index，丢失 score-function 项，**不是 reverse-KL gradient**（既不是 MLE，也不是 KL 下降方向）

### 2.4　与 KL-constrained RL 的关系

考虑 RL with reverse-KL constraint（RLHF 标准目标）：

$$\max_\theta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[R(x, y)\big] - \beta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\big]$$

OPD 是这个公式在 $R \equiv 0$（无外部 reward）且 $\pi_{\text{ref}} = \pi_T$（reference 换成 teacher）的特例：

$$\min_\theta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta \,\|\, \pi_T)\big]$$

这意味着 **OPD 是"纯 KL 项、teacher 当 reference"的 RLHF**。从这个视角看 GRPO + KL 项等同于 OPD + GRPO 风格 group baseline；这就是为什么 Survey 把 OPD 归入 "KL-constrained RL 的特殊情况"，也是 §3 中我们能把 OPD 集成进 GRPO 的形式基础。

### 2.5　Forward KL 形式（"Sampled-Token OPD"）

某些工作（如 Qwen3 部分阶段、TRL GKD trainer）使用 **forward KL on student samples**：

$$L_{\text{OPD-fwd}}(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_t D_{\text{KL}}\!\big(\pi_T(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t)\big)\right]$$

注意 **sample 来自 $\pi_\theta$，但 KL 方向是 forward**。展开为 token-level：

$$L_{\text{OPD-fwd}} = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_t \sum_v \pi_T(v|s_t)\big(\log \pi_T(v|s_t) - \log \pi_\theta(v|s_t)\big)\right]$$

由于 $\theta$ 只出现在 $\log \pi_\theta$ 中，且不在期望下标里出现（虽然 $y \sim \pi_\theta$ 仍依赖 $\theta$，但 stop-gradient 处理后 forward 项的 $\theta$-依赖简化），训练就退化为**对每个 student-visited prefix，对 teacher 分布做 cross-entropy**——形式上是 **soft-label cross-entropy on student rollouts**。这是工程上最简单的 OPD 实现（不需要 REINFORCE / control variate），代价是丢掉了 reverse KL 的 mode-seeking 性质。

## §3 OPD 与 DPO / GRPO / RLHF 的关系（统一视角）

### 3.1　全景对比表

| 维度 | SFT | Vanilla KD (Hinton) | DPO | RLHF + PPO | GRPO | **OPD** |
|---|---|---|---|---|---|---|
| **训练数据来源** | dataset | dataset | offline preference pair | dataset prompt | dataset prompt | dataset prompt |
| **谁生成 $y$** | dataset | teacher | (offline pair) | student rollout | student rollout (group) | **student rollout** |
| **监督信号** | hard label | teacher soft logit | binary preference | scalar RM reward | scalar RM reward (group-normalized) | **teacher soft logit (per-token)** |
| **是否 on-policy** | no | no | no | yes | yes | **yes** |
| **是否需要 critic** | n/a | n/a | n/a | yes (value head) | no (group baseline) | no (teacher KL is closed-form value) |
| **每 traj 监督信息量** | $O(L \log V)$ | $O(L \log V)$ | $O(1)$ | $O(1)$ | $O(1)$ | **$O(L \log V)$** |
| **需要 teacher** | no | yes | no | no | no | **yes** |
| **典型 KL 方向** | n/a (CE) | forward | n/a (closed-form) | reverse (vs $\pi_{\text{ref}}$) | reverse (vs $\pi_{\text{ref}}$) | **reverse (vs $\pi_T$)** |

> 💡 **三句话定位 OPD** — 一行能记住：

- 与 **SFT**：OPD 在 student-rollout 上训（SFT 在 teacher/dataset 上训）
- 与 **vanilla KD**：OPD on-policy + reverse KL（vanilla KD off-policy + forward KL）
- 与 **RL (PPO/GRPO)**：OPD 用 teacher log-prob 当 dense reward（RL 用 RM 当 sparse reward）

### 3.2　OPD vs DPO

**DPO** (Rafailov 2023 NeurIPS, arXiv 2305.18290) 是 closed-form RLHF：把 KL-regularized RL 最优解代回 Bradley-Terry，消去 partition $\log Z$，得到 pairwise preference 损失：

$$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\big(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\right]$$

DPO 是 **completely offline**：无需 sampling、无 teacher、无 RM、无 critic，只要 preference pair。

| 维度 | DPO | OPD |
|---|---|---|
| 数据 | offline preference pair $(y_w, y_l)$ | online student rollout $y \sim \pi_\theta$ |
| Teacher | 无（只要 $\pi_{\text{ref}}$） | 需要更强 teacher $\pi_T$ |
| 监督 | binary preference | per-token soft logit |
| Compute | 最便宜（纯 forward + 反传） | 中等（student sampling + teacher forward） |
| 何时用 | 有偏好数据集、无 strong teacher | 有 strong teacher、想压缩到小 student |

**两者可以叠加**：先 DPO 对齐人类偏好，再 OPD 用更强的 teacher 压缩到小 student。Qwen3 的 production pipeline 用类似套路；R1-Distill 官方发布的流程本身是纯 off-policy SFT（不含 student rollout / KL 信号，见 §3.4），并未叠加 OPD——用 OPD 对 R1-Distill checkpoint 做进一步蒸馏属于社区 follow-up 工作，不是 DeepSeek 官方流程的一部分。

### 3.3　OPD vs GRPO

**GRPO** (DeepSeekMath 2024, arXiv 2402.03300) 把 PPO 的 critic 换成组内归一化 advantage：

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf r)}{\text{std}(\mathbf r)}, \quad r_i = R(x, y_i),\; i \in [1..G]$$

$$L_{\text{GRPO}}(\theta) = \mathbb{E}_x \frac{1}{G}\sum_i \frac{1}{|y_i|}\sum_t \min\!\big(\rho_t^i \hat A_i,\, \text{clip}(\rho_t^i, 1{-}\epsilon, 1{+}\epsilon) \hat A_i\big) - \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

其中 $\rho_t^i = \pi_\theta / \pi_{\theta_{\text{old}}}$ 是重要性比。

**OPD 与 GRPO 的关系**：把 GRPO 的 sparse outcome reward $R(x, y_i)$ 替换成对逐 token teacher KL log-ratio **求和得到的轨迹级 reward** $r^i = \sum_t \big(\log\pi_T(y_t^i|s_t^i) - \log\pi_\theta^{\text{old}}(y_t^i|s_t^i)\big) = -\sum_t\log\!\frac{\pi_\theta^{\text{old}}(y_t^i|s_t^i)}{\pi_T(y_t^i|s_t^i)}$（注意必须取 log-ratio，不是 LaTeX 容易误读成的 $-\log\pi_\theta/\pi_T$；且这是对轨迹内所有 token 求和后的**单个标量**，经组内归一化后会被广播到该轨迹每个 token 上作为 advantage，并不提供 token 级差异化 credit assignment，详见 §4.5），且 reference $\pi_{\text{ref}} \leftarrow \pi_T$，GRPO loss 就退化为 OPD 的 policy-gradient 形式（见 §2.3）。

> 💡 **集成实战** — 现代 production pipeline 常用 **OPD + GRPO 混合**：

- **outcome reward** 来自 verifier（math 正确 / test pass）
- **dense reward** 来自 teacher per-token KL
- 总 reward $r_t = \alpha \cdot r_{\text{outcome}} \cdot \mathbb{1}[t = T] + (1-\alpha) \cdot r_{\text{teacher-KL}}$

这样既得到 outcome 的 task-aligned signal，又得到 teacher 的 dense supervision——是 Qwen3 / DeepSeek 系列 small-model 后训练的标配。

### 3.4　OPD vs vanilla RL distillation（R1-Distill）

DeepSeek-R1-Distill 系列（Qwen / Llama 1.5B-70B）用 **off-policy distillation**：
1. 用 R1 teacher 生成 800K reasoning trajectory（math + code + general）
2. 在小 student 上做 token-level cross-entropy SFT
3. **不做 on-policy rollout、不做 KL 信号**

这是 hard-label seq-level KD（Kim & Rush 2016 的 LLM 版本），不是 OPD。R1-Distill 的成功主要靠 teacher trajectory 质量极高（R1 经过 RL 调教）+ 数据量大，**结构上仍属 off-policy**。

OPD 与 R1-Distill 是 **互补关系**：
- R1-Distill 的 trajectory 数据可以作为 OPD 的 init / SFT warm-start
- OPD 在 R1-Distill 之后跑一轮可以进一步消除 exposure bias

学术 / 工程上一般称这种组合为 "**Off-policy SFT + On-policy distillation**"，是 Qwen3 small-model recipe 的标准两阶段（见 §6.1）。

## §4 实现：核心代码块

> ⚠️ **教学版示意 — 生产实现请参考原论文** —— 本节代码展示 OPD 的核心 idea（per-token reverse KL on student rollout）。**注意**：
>
> - **Full-vocab reverse KL**（求和 over 整个 vocab）是可微的，autograd 直接工作；这是 OPD 的 **Route A 教学清晰形式**，要求 teacher full logits 可用。Route B（sampled-token + REINFORCE / IS，**Tinker、MiniLLM 等生产实现的默认**）期望等价（见 §2.3 callout），差异在方差 vs 工程接口 trade-off。
> - **Sampled-token estimator** 形式上的 $\log\pi_s(y) - \log\pi_t(y)$ 在 $y$ 是离散采样时，需要 **REINFORCE-style policy gradient**（带 baseline / control variate）才能正确反传，**不能直接** `.backward()`，否则梯度只通过 $\log\pi_s$ 项，丢失对样本分布选择的反传。
> - OPD + GRPO 集成（§4.5）的 ratio 应是 **new policy / old behavior policy**，teacher 用作 reward 或 reference 而非 PPO ratio 的分母。生产前请对照 verl / OpenRLHF / Tinker 实现校对。
>
> 下方代码标注了哪些是 full-vocab（可直接 autograd）vs sampled estimator（需 REINFORCE）；具体 production code 强烈建议参考各 framework 官方 release。

### 4.1　Per-token reverse KL（Route A，full-vocab 概念实现）

```python
import torch
import torch.nn.functional as F


def per_token_reverse_kl_loss(
    student,        # student model, requires_grad
    teacher,        # teacher model, frozen
    input_ids,      # [B, L]  prompt + student rollout (rollout 在外部完成，stop_grad)
    action_mask,    # [B, L]  1 for student-generated tokens, 0 for prompt / pad
):
    """
    Route A — full-vocab per-token reverse KL on student rollouts (concept-clean form).
    要求 teacher full logits 可用。Route B (Tinker / MiniLLM 等生产默认) 期望等价但
    采 sampled-token + IS / REINFORCE estimator，参见 §2.3 callout 和 §4.4。

        L(θ) = E_{s_t ~ rollout(π_{θ⁻})} [ Σ_t D_KL(π_θ(·|s_t) || π_T(·|s_t)) ]
             = E_{s_t} [ Σ_t Σ_v  π_θ(v|s_t) · ( log π_θ(v|s_t) − log π_T(v|s_t) ) ]

    Rollout 已在外部用 stop_grad(θ) 完成（vLLM / TGI 把 input_ids 拿回来）；
    本函数 **只** 计算可微的 full-vocab KL，autograd 直接反传，**不需要 REINFORCE**。
    """
    # Student full-vocab log-probs (requires_grad on θ)
    s_logits = student(input_ids).logits                   # [B, L, V]
    s_log_probs = F.log_softmax(s_logits[:, :-1], dim=-1)  # [B, L-1, V]
    s_probs = s_log_probs.exp()                            # [B, L-1, V]

    # Teacher full-vocab log-probs (frozen, no grad)
    with torch.no_grad():
        t_logits = teacher(input_ids).logits
        t_log_probs = F.log_softmax(t_logits[:, :-1], dim=-1)  # [B, L-1, V]

    # Full-vocab reverse KL per state s_t: Σ_v π_θ(v) (log π_θ(v) − log π_T(v))
    kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # [B, L-1]

    # Mask 到 student-generated 位置（prompt token 上算 KL 没意义）
    mask = action_mask[:, 1:].float()
    loss = (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0)

    # 诊断：在 sampled 位置上 student 是否比 teacher 更"自信"（监控 over-confidence）
    with torch.no_grad():
        targets = input_ids[:, 1:].unsqueeze(-1)
        s_logp_t = s_log_probs.gather(-1, targets).squeeze(-1)
        t_logp_t = t_log_probs.gather(-1, targets).squeeze(-1)
        overconf_frac = ((s_logp_t > t_logp_t).float() * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {
        "full_vocab_kl": loss.item(),
        "overconf_frac": overconf_frac.item(),
    }
```

注意几个生产实现的细节：

- **`action_mask`** 必须只 mask student rollout 的 token；prompt 上算 KL 没意义（teacher 也只是 condition）
- **teacher forward 用 `torch.no_grad()`** 否则 memory 翻倍
- **Full-vocab (Route A) vs sampled-token (Route B)**：上面的 loss 是 **Route A**（求和 over $V$ 个 token，autograd 直接反传）。**Route B**（sampled-token + REINFORCE/IS，§4.4）是 Tinker / MiniLLM 等生产实现的默认形式，与 PPO/GRPO 共享 sampled-token 接口，支持 logprob-only（灰盒）teacher（真正 sample-only 的黑盒 teacher 需要 §6.5 的 Black-Box OPD），A 与 B **期望等价**。无论选哪条，**不能**对 sampled-token 的 `log π_θ − log π_T` 直接 `.backward()` —— 那等价于对固定 sampled index 取 pathwise $\nabla\log\pi_\theta(y_t)$，**丢失了 score-function 项**，既不是 reverse-KL 梯度，也不是 MLE。Route B 必须配 REINFORCE 估计器（detached reward + score-function trick）
- **batched teacher inference** 在生产里通常**异步**：先用 vLLM 起 teacher server，把 student rollout batch 发过去拿 log-probs；本地只跑 student forward + backward。这是 Tinker / vLLM-based 框架的标配
- **mixed precision**：student 用 bf16，teacher logits 用 fp32 算 log-softmax（防数值不稳）
- **朴素实现的显存开销不小**：本函数同时物化 `s_logits / s_log_probs / s_probs / t_logits / t_log_probs` 五个 $[B,L,V]$ BF16 张量；按 §4.2/§4.6 给出的典型配置（`batch_size=64`、`max_new_tokens=512`、vocab 100K-152K），单个张量约 6.5-10 GB，五个合计约 **33-50 GB**，已超过 §8.2 显存表中 activations 的预算——生产实现通常做分块 / fused cross-entropy 来避免同时物化五个满 vocab 张量，见 §8.2 说明

#### 共用辅助函数（贯穿 §4.2 - §4.6）

```python
def per_token_logp(model, input_ids):
    """
    对每个位置返回 log π(y_{t+1}|s_t)。形状 [B, L-1]。grad 由 caller 决定
    （想要 grad: 直接调；想要 detached behavior log-prob: 包 with torch.no_grad()）。
    """
    logits = model(input_ids).logits[:, :-1]                     # [B, L-1, V]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)                     # [B, L-1, 1]
    return log_probs.gather(-1, targets).squeeze(-1)             # [B, L-1]


def teacher_kl_reward(student, teacher, input_ids, action_mask):
    """
    OPD-GRPO 中用作轨迹级 reward 的 teacher-vs-behavior KL（逐 token 计算后在轨迹内求和，
    不是逐 token 的 dense reward——见函数末尾 .sum(dim=-1)）：
        r_t = log π_T(y_t|s_t) - log π_θ_old(y_t|s_t)
    输入 action_mask 形状 [B, L]（同 input_ids），内部 shift 到 [B, L-1]。
    全程 no_grad，返回每轨迹的标量 reward（sum over generated tokens）。
    """
    with torch.no_grad():
        s_old = per_token_logp(student, input_ids)               # [B, L-1]
        t_lp  = per_token_logp(teacher, input_ids)               # [B, L-1]
        token_mask = action_mask[:, 1:].float()                  # 对齐到 [B, L-1]
        per_tok = (t_lp - s_old) * token_mask
    return per_tok.sum(dim=-1)                                   # [B]
```

> ⚠️ **生产前的 sanity check** — 上面是 **教学示意**：实际生产 (verl / OpenRLHF / Tinker) 会把 `per_token_logp` 拆成 logits-then-gather 以省显存、加 `attention_mask` 处理 padding、对 teacher 用 vLLM async server 而不是同进程 forward。Sampler 与 grad-aware forward 也通常分离（rollout phase 调 `student.generate`，update phase 用 grad-aware forward 算 `s_logp_new`）。下面代码省略这些工程细节。

### 4.2　完整 OPD 训练循环（带 student rollout）

```python
@torch.no_grad()
def student_rollout(student, prompts, max_new_tokens=512, temperature=1.0):
    """student 自己采 trajectory（on-policy 关键步骤）"""
    out = student.generate(
        prompts.input_ids,
        attention_mask=prompts.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        return_dict_in_generate=True,
    )
    full_ids = out.sequences           # [B, L_prompt + L_gen]
    prompt_len = prompts.input_ids.shape[1]
    action_mask = torch.zeros_like(full_ids)
    action_mask[:, prompt_len:] = 1
    # mask out pad
    action_mask = action_mask * (full_ids != student.config.pad_token_id).long()
    return full_ids, action_mask


def opd_train_step(student, teacher, optimizer, batch, kl_coef=1.0):
    """one step of OPD: sample rollout, compute teacher KL, update student"""
    # 1. student samples (on-policy)
    input_ids, action_mask = student_rollout(student, batch.prompts)

    # 2. compute per-token reverse KL with teacher
    loss, stats = per_token_reverse_kl_loss(
        student, teacher, input_ids, action_mask
    )

    # 3. backward
    optimizer.zero_grad()
    (kl_coef * loss).backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    return stats
```

这是 OPD 最朴素的训练循环。生产里会在第 2 步加 advantage / baseline / clipping（见 §4.4）；在第 1 步加 group sampling（GRPO 风格，见 §4.5）。

### 4.3　Off-policy KD vs OPD 的对比代码（差一行）

```python
# ── Off-policy KD (vanilla, Hinton-style) ──
def off_policy_kd_loss(student, teacher_outputs, dataset_y):
    """
    teacher_outputs 在 dataset_y 上 forward
    student 也在 dataset_y 上 forward
    forward KL: teacher → student
    """
    s_logits = student(dataset_y).logits
    t_logits = teacher_outputs                 # 已经预算
    return F.kl_div(
        F.log_softmax(s_logits, dim=-1),
        F.softmax(t_logits, dim=-1),
        reduction="batchmean",
    )


# ── On-Policy Distillation (OPD) ──
def opd_loss(student, teacher, prompts):
    student_y, mask = student_rollout(student, prompts)         # ← 关键差异：student 自己采
    loss, _ = per_token_reverse_kl_loss(student, teacher, student_y, mask)
    return loss
```

**唯一关键差异**：OPD 在自己 sample 的 trajectory 上算 loss；off-policy 在 dataset / teacher trajectory 上算。代码上只差一个 `student_rollout` 调用，但**训练分布与推理分布对齐**这件事的整体收益巨大。

### 4.4　Control Variate Baseline（vOPD 风格的 token-level KL）

> ⚠️ **使用场景**：本节是 Route B（sampled-token REINFORCE / importance sampling）的标准展示。**Route B 是 Tinker、MiniLLM 等生产实现的默认形式**，与 Route A (§4.1, full-vocab autograd) **期望等价**，trade-off 在方差 vs 工程接口：Route A 零方差但需 teacher full logits，Route B 适配 PPO/GRPO sampled-token 接口、支持 logprob-only（灰盒）teacher——仍需 teacher 吐出 $\log \pi_T(y_t|s_t)$，不是真正 sample-only 的黑盒 API（那需要 §6.5 的 Black-Box OPD）——并节约 teacher vocab memory。**closed-form baseline** 这个具体降方差技巧只在 teacher full logits 可用时算得出来；灰盒（logprob-only）teacher 下要换 learned / EMA / per-prompt 均值 baseline——见末尾 caveat。

> 注：vOPD（"KL for a KL"）的 control-variate 思想是 OPD 文献里反复出现的模式（Survey 2026 / Tinker blog 等都有等价讨论），下面给出**正确的 detach + sign + estimator** 形式。

**REINFORCE 形式的 sampled-token estimator**：对单个 sampled token $y_t \sim \pi_\theta(\cdot|s_t)$，无偏估计反向 KL 的 *gradient* 是

$$\nabla_\theta D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)(s_t) = \mathbb{E}_{y_t \sim \pi_\theta}\!\big[\,\nabla_\theta \log \pi_\theta(y_t|s_t) \cdot \underbrace{(\log \pi_\theta(y_t|s_t) - \log \pi_T(y_t|s_t))}_{\hat r_t,\;\textbf{detached}}\,\big].$$

注意 $\hat r_t$ 出现的位置只是 **scalar reward**（必须 detach），梯度仅来自外面那个 $\nabla_\theta \log\pi_\theta$。**Control variate** 减方差：

$$\hat r_t \leftarrow \hat r_t - B(s_t),\quad B(s_t) = \mathbb{E}_{y_t \sim \pi_\theta}[\hat r_t] = D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)(s_t).$$

$B(s_t)$ 是该 step **整 vocab** 的 KL（正号），可从 student forward 闭式算出。它的 detached value 不改变 estimator 的 unbiasedness（因为 $\mathbb{E}_{y_t}[\nabla \log\pi_\theta \cdot B(s_t)] = B(s_t)\cdot \mathbb{E}[\nabla\log\pi_\theta] = 0$），但显著降低方差。

```python
def vopd_token_kl_estimator(student, teacher, input_ids, action_mask):
    """
    Sampled-token REINFORCE estimator of ∇_θ D_KL(π_θ || π_T) with closed-form baseline.

    返回 surrogate loss L_surr，其 ∇L_surr 在期望意义下 = ∇ E[D_KL]。注意 L_surr 本身
    不是 D_KL 的数值估计 —— diagnostic 请用 full-vocab D_KL。
    """
    s_logits = student(input_ids).logits[:, :-1]                    # [B, L-1, V]
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    s_probs     = s_log_probs.exp()
    with torch.no_grad():
        t_logits = teacher(input_ids).logits[:, :-1]
        t_log_probs = F.log_softmax(t_logits, dim=-1)

    targets = input_ids[:, 1:].unsqueeze(-1)                        # [B, L-1, 1]
    s_logp_t = s_log_probs.gather(-1, targets).squeeze(-1)          # grad-aware: log π_θ(y_t|s_t)

    # ── reward signal (must detach: scalar weight, not part of grad path) ──
    with torch.no_grad():
        t_logp_t = t_log_probs.gather(-1, targets).squeeze(-1)
        r_hat    = s_logp_t.detach() - t_logp_t                     # scalar per token

        # closed-form baseline B(s_t) = D_KL(π_θ || π_T)(s_t), full vocab
        baseline = (s_probs.detach() * (s_log_probs.detach() - t_log_probs)).sum(-1)
        advantage = r_hat - baseline                                # both detached

    # ── REINFORCE-style surrogate: ∇ = E[ ∇log π_θ(y_t|s_t) · advantage_detached ] = +∇D_KL ──
    mask = action_mask[:, 1:].float()
    surrogate = (s_logp_t * advantage * mask).sum() / mask.sum().clamp_min(1.0)
    #            ↑ 注意符号：advantage = (logπ_θ − logπ_T) − B 是 KL cost。要 minimize D_KL，
    #            optimizer step 是 θ ← θ − η·∇L_surr，而 ∇L_surr = +E[∇logπ·advantage] = +∇D_KL，
    #            所以 surrogate **正号**，treat surrogate as the loss to .backward().

    # diagnostic（不进 grad）：监控真正的 full-vocab KL
    with torch.no_grad():
        full_kl = (baseline * mask).sum() / mask.sum().clamp_min(1.0)
    return surrogate, {"full_vocab_kl_diag": full_kl.item()}
```

> 💡 **代码里反复出现的"坑"** —
> 1. **`r_hat` 必须 detach**：它是 reward signal，不是 loss 的一部分；如果不 detach，会把 $\log\pi_T$ 和 student log-prob 同时拉进 backward，与论文 estimator 完全不同。
> 2. **`baseline` 必须 detach**：control variate 不能反传梯度，否则 unbiasedness 不再成立。
> 3. **surrogate 的符号**：我们想 **minimize** $D_{\text{KL}}$。REINFORCE 恒等式给出 $\nabla D_{\text{KL}} = \mathbb{E}[\nabla\log\pi_\theta\cdot \hat r]$，所以让 `loss = +E[\log\pi_\theta\cdot(\hat r - B).\text{detach}()]`，则 `loss.backward()` 得到 $+\nabla D_{\text{KL}}$，optimizer step `θ -= η·∇L` 就在做 KL 下降。**正号才对**——容易把符号写反。
> 4. **Route A vs Route B 的选择**：`full_kl_per_pos`（§4.1 Route A）零方差、一行 autograd，但要 teacher full logits；本节 Route B 适配 sampled-token 接口（与 §4.5 PPO clipping 自然衔接）、支持 logprob-only（灰盒）teacher、节约 vocab-size memory。两者**期望等价**，按工程约束选。
> 5. **Logprob-only（灰盒）teacher 的 baseline choices**：如果只能 query teacher log-prob 而拿不到 full vocab（注意这仍然是灰盒场景，不是真正 sample-only 的黑盒 API），closed-form `B(s_t)` 算不了；可改用 (a) `running mean of r_hat` 作 baseline；(b) 学习一个 lightweight value head；(c) per-prompt 经验均值。失去 closed-form 优势但仍保留 unbiased + 显著降方差。

### 4.5　OPD 集成进 GRPO（multi-sample group baseline）

GRPO 的 ratio 是 **new student policy vs old behavior student policy**（rollout 时记录 `s_logp_old`，update 时算 `s_logp_new`），teacher 进入两个独立位置：(i) **reward** 通过对 token-level KL reward $r_t^{\text{kl}} = \log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)$ 求和得到的**轨迹级标量 reward**（组内归一化后对该轨迹所有 token 广播相同 advantage，不提供 token 级差异化 credit assignment，见下方代码 `r_kl_dense` 的 `.sum().item()`），与 outcome reward 加权得到总 reward；(ii) **KL regularization** 通过 $D_{\text{KL}}(\pi_\theta\,\Vert\,\pi_T)$ 显式约束 student 不偏离 teacher——这一项才是真正的逐 token dense 监督（full-vocab 闭式，§4.1）。**Teacher 绝不进入 PPO ratio 的分母**。

```python
def opd_grpo_step(student, teacher, batch, G=8, alpha=0.5, kl_coef=0.1, clip=0.2):
    """
    OPD + GRPO hybrid（per-prompt group baseline）:
      - 对每个 prompt sample G 条 rollout，记录 behavior-policy log-prob
      - reward = α * outcome_reward + (1-α) * sum_t [log π_T(y_t|s_t) - log π_θ_old(y_t|s_t)]
      - GRPO group-relative advantage（组内归一化）
      - ratio = exp(s_logp_new - s_logp_old.detach())  ← 关键
      - KL penalty 项用 student vs teacher（不是 PPO ratio）

    Shape conventions:
      input_ids / action_mask:   [B, L]   (prompt + generated)
      per_token_logp output:     [B, L-1] (predict y_{t+1} from prefix up to t)
      token_mask = action_mask[:, 1:]：与 per-token logp 对齐
    """
    ids_all, action_mask_all, s_old_all, adv_all = [], [], [], []

    for prompt in batch.prompts:
        rollouts = []
        for _ in range(G):
            ids, action_mask = student_rollout(student, prompt)    # [B', L]; B'=1 for single prompt
            with torch.no_grad():
                s_old = per_token_logp(student, ids)               # [B', L-1] behavior log-prob
                t_lp  = per_token_logp(teacher, ids)               # [B', L-1]
            rollouts.append((ids, action_mask, s_old, t_lp))

        # 组内 G 条 rollout 的总 reward
        rewards = []
        for (ids, action_mask, s_old, t_lp) in rollouts:
            r_outcome = math_verifier(ids)                         # 标量 ∈ {0,1}
            token_mask = action_mask[:, 1:].float()                # [B', L-1]
            r_kl_dense = ((t_lp - s_old) * token_mask).sum().item()
            r_total = alpha * r_outcome + (1.0 - alpha) * r_kl_dense
            rewards.append(r_total)
        # 保证与 s_old / s_logp_new 同 device/dtype，避免 CPU/GPU mismatch
        rewards = torch.tensor(rewards, device=s_old.device, dtype=s_old.dtype)
        adv_per_traj = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # [G]

        for i, (ids, action_mask, s_old, _t_lp) in enumerate(rollouts):
            ids_all.append(ids); action_mask_all.append(action_mask)
            s_old_all.append(s_old)
            adv_all.append(adv_per_traj[i].view(1, 1).expand_as(s_old))     # broadcast 到 [B', L-1]

    ids         = torch.cat(ids_all, dim=0)                                  # [BG, L]
    action_mask = torch.cat(action_mask_all, dim=0)                          # [BG, L]
    token_mask  = action_mask[:, 1:].float()                                 # [BG, L-1]
    s_logp_old  = torch.cat(s_old_all, dim=0).detach()                       # [BG, L-1]
    A           = torch.cat(adv_all, dim=0).detach()                         # [BG, L-1]

    # ── new student log-prob（这步带 grad） ──
    s_logp_new = per_token_logp(student, ids)                                # [BG, L-1]

    # ── PPO clipped ratio: new student vs old student（不是 student vs teacher） ──
    ratio = torch.exp(s_logp_new - s_logp_old)
    pg = torch.min(ratio * A, torch.clamp(ratio, 1 - clip, 1 + clip) * A)
    pg_loss = -(pg * token_mask).sum() / token_mask.sum().clamp_min(1.0)

    # ── KL penalty: student vs teacher（OPD 的 reverse KL，full-vocab 闭式） ──
    # 传 [B, L] action_mask 进去；§4.1 内部自己 shift 到 [B, L-1]
    kl_loss, _ = per_token_reverse_kl_loss(student, teacher, ids, action_mask)

    return pg_loss + kl_coef * kl_loss
```

> 💡 **关键细节** —
> 1. **ratio 的分母是 `s_logp_old`（behavior policy）**，rollout 时一次性算出并 `.detach()`，与 teacher 无关。这与 vanilla GRPO/PPO 完全一致。
> 2. **Teacher 的两个角色互不重叠**：作为 reward source 提供 `r_kl_dense`（对逐 token KL 求和后的轨迹级标量，`.sum().item()` 后进 advantage，再经 `expand_as` 广播回每个 token，token 维度上是均匀的，并非 token 级差异化信号），作为 KL anchor 提供闭式 reverse-KL penalty（真正的逐 token dense 监督，直接进 loss）。
> 3. **`per_token_reverse_kl_loss` 是 full-vocab 闭式**（§4.1），不是 sampled-token estimator，所以 KL 项 unbiased 且低方差。
> 4. **mini-batch 多步更新**：实际 GRPO 一个 rollout batch 会做 `n_epochs` 次 inner update，ratio 不为 1（关键，否则 clipping 不起作用）。

实战 tip：**alpha = 0.5 是经验起点**；alpha → 1 退化为 pure GRPO（outcome only），alpha → 0 退化为 pure OPD（KL only）。Qwen3 / R1-Distill 类 recipe 通常 stage-wise 调度：早期 alpha 小（先把 student 拉近 teacher），后期 alpha 大（拼 outcome SOTA）。

### 4.6　Synthetic-data + distillation pipeline（顶层伪代码）

> 📝 **此处是 architecture-level 伪代码**（不是可运行 Python）：`sample / sft_train / math_verifier` 等都是 placeholder，对应到具体 framework 是 `torch.utils.data.DataLoader` / TRL `SFTTrainer` / 自定义 reward function。下面只展示 **三段式 recipe 的拓扑**。

```python
def synthetic_distillation_pipeline(
    teacher,             # 大 teacher（如 R1-671B / Qwen3-32B）
    student,             # 小 student（如 Qwen3-8B-Base）
    prompt_pool,         # 海量 prompt（可以用 unlabeled instruction）
    n_stage1=800_000,    # off-policy SFT 数据量
    n_stage2_steps=5_000,
    n_stage3_steps=1_000,
    verifiable_prompts=None,
):
    # ─── Stage 1: 用 teacher 生成 synthetic trajectory，做 off-policy SFT ───
    synthetic_data = []
    for prompt in random_sample(prompt_pool, n_stage1):           # placeholder sampler
        with torch.no_grad():
            y_teacher = teacher.generate(prompt, max_new_tokens=4096)
        synthetic_data.append((prompt, y_teacher))
    run_sft(student, synthetic_data, epochs=2)                    # cross-entropy on (prompt, y_teacher)

    # ─── Stage 2: 在 student 自己 rollout 上做 OPD ───
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    for step in range(n_stage2_steps):
        batch = make_batch(prompt_pool, batch_size=64)
        opd_train_step(student, teacher, optimizer, batch, kl_coef=1.0)

    # ─── Stage 3 (optional): OPD + outcome reward (math/code verifier) ───
    if verifiable_prompts is not None:
        for step in range(n_stage3_steps):
            batch = make_batch(verifiable_prompts, batch_size=64)
            loss  = opd_grpo_step(student, teacher, batch, G=8, alpha=0.5)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    return student
```

这是 Qwen3 / Thinking Machines 类似 recipe 的标准三段式（R1-Distill 官方发布的流程仅对应 Stage 1 的 off-policy SFT，不含 Stage 2/3 的 on-policy 部分，见 §3.4）。Stage 1 给 cold start（student 学会模仿 teacher 的 style 与 format），Stage 2 用 OPD 把 exposure bias 消掉，Stage 3 用 verifier 做 task-aligned 微调。

## §5 OPD 在 Knowledge Distillation 谱系中的位置

### 5.1　LLM 蒸馏方法谱系（从 Hinton 到 OPD）

```

   2015 ─────────── 2019 ─────── 2023 ────── 2024 ─────── 2025/2026
   Hinton          DistilBERT   MiniLLM     GKD          Thinking
   soft target     (Sanh)       (Gu)        (Agarwal)    Machines
                                                          blog
     │              │            │            │            │
     ↓              ↓            ↓            ↓            ↓
   分类             encoder-     LLM 首次     generalized   "便宜 RL"
   forward KL      level KD     reverse KL   JSD + mix     范式 + 大规模
                                + on-policy  on/off        实证（Qwen3/
                                              policy        MiMo/Kimi）
```

**核心演化方向**：
- **off-policy → on-policy**（处理 exposure bias）
- **forward KL → reverse KL / JSD**（处理 mode-covering）
- **hard label → soft logit**（保留 dark knowledge）
- **single-stage → multi-stage hybrid**（off-policy SFT cold start + on-policy OPD + outcome RL）

### 5.2　关键 paper 一句话索引

| 年份 | Paper | 关键贡献 |
|---|---|---|
| 2015 | Hinton, Vinyals, Dean — "Distilling the Knowledge in a Neural Network" (arXiv 1503.02531) | Soft target KD + temperature softmax 的开山之作 |
| 2016 | Kim & Rush — "Sequence-Level Knowledge Distillation" (EMNLP, arXiv 1606.07947) | 把 KD 推到 seq-to-seq NMT，提出 seq-level / token-level KD 区分 |
| 2019 | Sanh et al. — "DistilBERT" (NeurIPS workshop, arXiv 1910.01108) | 参数减少 40%（约保留 60% 原始参数量），保留约 97% GLUE 性能，推理速度提升约 60% |
| 2020 | Sun et al. — "MobileBERT" (ACL, arXiv 2004.02984) | 任务无关 KD + 渐进式蒸馏 |
| 2023 | Gu et al. — "MiniLLM" (ICLR 2024, arXiv 2306.08543) | **第一次正式把 reverse KL + on-policy 应用到 LLM**；policy gradient 优化 reverse KL |
| 2023 | Agarwal et al. — "GKD: On-Policy Distillation of LM" (ICLR 2024, arXiv 2306.13649) | 统一 forward/reverse KL + on/off-policy 数据，generalized JSD 插值；$\lambda$ 控制 student 数据比例 |
| 2025 | DeepSeek-R1-Distill (DeepSeek 2025-01, arXiv 2501.12948) | Off-policy SFT 蒸馏 800K reasoning trajectory，1.5B-70B family |
| 2025-05 | Qwen3 Tech Report (arXiv 2505.09388) | 工业级 OPD recipe：off-policy SFT 冷启动 + on-policy distillation；比 RL 省 10× GPU 时长 |
| 2025-10 | Thinking Machines — "On-Policy Distillation" blog (Lu et al.) | 把 OPD 包装成"便宜 RL"路线；Qwen3-8B + Qwen3-32B-teacher 复现 RL gain，9-30× FLOPs 节省（大方向可信，精确倍数区间 [needs-verify]） |
| 2025-11 | Black-Box OPD (arXiv 2511.10643) | 不需要 teacher logits，只用 teacher samples 做 on-policy 蒸馏 |
| 2026 | Song & Zheng — "A Survey of OPD for LLMs" (arXiv 2604.00626) | 把 OPD 形式化为 student-rollout 上的 $f$-divergence minimization；三轴 taxonomy |
| 2026 | "Rethinking OPD" (arXiv 2604.13016) | 现象学 + 机制 + recipe：truncation collapse / mode-seeking 失败 / 反事实回归 |
| 2026 | "vOPD: KL for a KL" (arXiv 2605.07865) | Closed-form control variate baseline 降方差 |

### 5.3　OPD 在 Reasoning Model 蒸馏中的位置（R1-Distill 时代）

reasoning model 蒸馏的两条主线：

| 路线 | 代表 | 范式 |
|---|---|---|
| **Off-policy SFT KD** | DeepSeek-R1-Distill, s1 (Muennighoff 2025), OpenThinker, Gemma 2/3（固定数据集上的 soft-target KD，不含 student self-rollout） | teacher 生成 trajectory / soft label → student SFT |
| **On-Policy Distillation (OPD)** | Qwen3, MiMo-V2, Thinking Machines | student 采 → teacher 给 per-token KL |

**何时用哪个**：
- 如果 teacher trajectory 质量极高、student 与 teacher capability gap 不大 → off-policy SFT 已经够
- 如果 student 比 teacher 小很多（>5×）、任务长 horizon、对推理时分布敏感 → OPD 是更稳的选择
- 实战通常**两者叠加**：先 off-policy SFT 给 cold start，再 OPD 消 exposure bias

## §6 工程案例：OPD 在生产里怎么用

### 6.1　Qwen3 Recipe（arXiv 2505.09388 §3.2）

Qwen3 small models（1.7B / 4B / 8B）的 post-training 采用两阶段蒸馏：

```
Qwen3-235B (teacher, /think + /no_think 双模)
    │
    │ Stage 1: Off-policy distillation
    │   - teacher 生成 trajectory（/think 与 /no_think 混合）
    │   - student SFT on teacher trajectory (cross-entropy)
    │   - 目的：basic reasoning + mode switching
    ↓
Qwen3-8B-mid
    │
    │ Stage 2: On-policy distillation (OPD)
    │   - student 自己 sample → teacher 给 per-token logit
    │   - per-token reverse KL loss
    │   - 目的：消 exposure bias + 拉 reasoning depth
    ↓
Qwen3-8B-final
```

**报告结果**：
- 比纯 RL 节省 ~10× GPU 时长
- 在 AIME'24 / AIME'25 上 pass@64 显著提升（说明 OPD 没塌缩 diversity）
- 比 off-policy SFT 在 long-CoT 任务上 +3-5pp

**[needs-verify]** pass@64 的具体数值与 "+3-5pp" 的精确度量口径待核对原始 Qwen3 Technical Report §3.2（arXiv 2505.09388），与 §A.4 的既有 needs-verify 惯例保持一致。

### 6.2　Thinking Machines Blog（Lu 2025-10-27）

实验 setting：
- **Student**: Qwen3-8B-Base
- **Teacher**: Qwen3-32B
- **Task**: 数学推理（AIME'24 主 benchmark）
- **Init**: 先用 OpenThoughts 数据 SFT 一轮
- **OPD**: 用 Tinker 实现，per-token reverse KL，无 outcome reward

报告结果（blog 内容）：
- 用 OPD 训 Qwen3-8B-Base 达到与 RL 几乎相同的 AIME'24 性能
- **总 FLOPs 节省约 9-30×**，取决于 batch / lr 设置
- 强调 "**OPD 是 RL 的便宜替代品**"，不是 RL 的补充

### 6.3　Tinker Cookbook 实现细节

GitHub: `thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation`

关键设计：
- **Environment 设计为 no-reward**：唯一 supervision 是 teacher KL
- **Reverse KL 实现**：`(student_logp - teacher_logp) * mask`
- **Advantage 计算**：把 KL 当 negative reward 加进 advantage（`advantage = -kl_penalty_coef * reverse_kl`）
- **LoRA**: rank 128 + lr 1e-4
- **Batch**: groups_per_batch = 64（GRPO 风格 grouping）

这种"把 KL 当 reward 注入 advantage"的设计正好对应 §3.3 的 OPD-GRPO 等价：把 sparse outcome reward 替换成 dense teacher KL reward。

### 6.4　DeepSeek-V4 报告（multi-teacher OPD 替代 RL）

**[needs-verify]** DeepSeek-V4 及其 multi-teacher OPD consolidation 细节来自二手/推断信息，未核实原始技术报告，与 §A.4 中对同一主张（L3-3）已有的 needs-verify 标注保持一致。

DeepSeek-V4 在 model consolidation 阶段**完全用 multi-teacher OPD 替代 mixed RL**——多个 teacher（specialized：math / code / reasoning / chat）的 logits 加权 ensemble 作为 student 的 OPD target。这是 OPD 范式扩展到 multi-teacher 与 specialist consolidation 的代表案例。

### 6.5　Black-Box OPD（不需要 teacher logits）

**arXiv 2511.10643** "Black-Box On-Policy Distillation"：当 teacher 是 closed-source API（如 GPT-4 / Claude）只能 sample 不能拿 logit 时怎么做 OPD？

核心思路：用 **teacher samples 做 trajectory-level reward**，组合 token-level student log-prob 做近似 reverse KL。这把 OPD 从"需要 teacher logit"扩展到"只需 teacher API"。trade-off 是丢掉了 per-token dense supervision，回退到 trajectory-level reward——本质上是 OPD 与 RL 之间的混合体。

## §7 失败模式与缓解策略

### 7.1　Truncation Collapse / Length Inflation

**现象**（Demystifying OPD 2026, arXiv 2604.08527）：训练中 student rollout 平均长度持续增长，最终撞 max_length truncation，被 truncate 的轨迹**贡献了大部分梯度**（因为它们 token 多），造成 gradient bias，validation 性能崩盘。

**根本原因**：reverse KL 的 mode-seeking 性质让 student 倾向于"持续输出 teacher 高概率 token"，而 teacher 在 long-CoT 上本来就喜欢长输出。这是一个**正反馈循环**。

**缓解**：
- **Length normalization**：loss 除以 trajectory length（per-token average，不是 sum）——这是 Tinker / 大多数生产实现的默认
- **Length penalty**：在 reward 里加 $-\lambda \cdot L$ 抑制超长
- **Max-length 比 sample max-length 大很多**：避免 truncation 进入训练
- **Early stop on val loss spike**：检测到 val 暴跌马上停

### 7.2　Mode Collapse / Diversity Loss

**现象**：reverse KL 是 mode-seeking 的，理论上 student 应该锁定到 teacher 的一个 mode；但在某些任务（如 open-ended generation）上 student 收敛到**单一回答 template**，丢掉多样性，pass@N（$N > 1$）下降。

**缓解**：
- **使用 forward KL / JSD 替代 reverse KL**（GKD 的 $\beta$ 插值）
- **加 entropy bonus**（类似 PPO）
- **Mixed loss**：reverse KL × $\alpha$ + forward KL × $(1-\alpha)$
- **MiniLLM 的 stabilization trick**：mixed policy $\pi_{\text{mix}} = (1-\alpha)\pi_\theta + \alpha\pi_T$ with $\alpha = 0.2$ 防止 mode collapse early

### 7.3　Reward Hacking（teacher-game）

**现象**：student 学到一种 hack——输出一些 **teacher 概率高但语义垃圾** 的 token（如 stopword 序列、重复 phrase），KL loss 极低但 task 性能差。

**缓解**：
- **混合 outcome reward**（§4.5 的 OPD-GRPO hybrid）
- **Trajectory-level filter**：过滤掉 teacher 都不高概率的 trajectory（"teacher-improbable filter"）
- **Token diversity regularizer**：penalize token entropy 过低

### 7.4　Prefix Teach, Suffix Fade（local teachability collapse）

**arXiv 2605.13643** 报告：训练后期，teacher 在 trajectory 后半段几乎"没东西可教"了——student 与 teacher 在 suffix 上的 log-prob 已经接近，loss 几乎全部来自 prefix。这导致 long-CoT 任务的后半推理段质量低。

**缓解**：
- **Token-level importance weighting**：对 suffix token 加权
- **Teacher 升级**：用更强的 teacher（多 teacher ensemble / 不同任务 specialist）
- **Curriculum**：先短 trajectory，后逐步增 max_length

### 7.5　Teacher / Student Gap 太大

**现象**：student 比 teacher 小很多（如 1B vs 70B）时，student 的 sample 完全采不到 teacher 概率高的 region（"student visits states teacher never thought of"）。这不是"信号消失"：sampled-token reverse KL 项 $\log\pi_\theta(y_t|s_t) - \log\pi_T(y_t|s_t)$ 在 $\pi_T(y_t|s_t)\to 0$ 时因 $\log\pi_T(y_t|s_t)\to -\infty$ 而**趋于 $+\infty$**（极端惩罚 / 数值不稳定）。真正的问题是：(a) student 自身采样分布可能从未覆盖 teacher 高概率 token（mode-covering 缺失，有效学习信号方向性差）；(b) 一旦采到 teacher 极不可能的 token，该 step 会产生数值极端（而非趋零）的 loss。

**缓解**：
- **off-policy SFT warm start**：先让 student 模仿 teacher 一段时间，状态分布拉近后再 OPD
- **Temperature scheduling**：训练初期 student temperature 高，扩大探索；后期降到 1.0
- **使用更接近的 teacher**：teacher 与 student gap 控制在 4-8×（Qwen3 选 32B teacher → 8B student 是这种考虑）

### 7.6　Catastrophic Forgetting（旧能力丢失）

**现象**：OPD 主要在数学 / 推理上训，但 student 通用能力（chat / safety / instruction-following）下降。

**缓解**：
- **Replay buffer**：保留一部分通用 SFT 数据，按比例混进 OPD 训练
- **Regularization to SFT init**：加 $\beta \cdot D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{SFT-init}})$ penalty
- **Multi-task OPD**：同时蒸馏多个领域的 teacher（math + chat + code）

## §8 复杂度与资源

### 8.1　每步训练成本对比

| 范式 | Student forward | Student backward | Teacher forward | Sampling cost | 总 wall-clock |
|---|---|---|---|---|---|
| **SFT** | 1× | 1× | 0 | 0（dataset 提供） | 1× |
| **Off-policy KD** | 1× | 1× | 1×（pre-computed cache） | 0 | 1×-1.2× |
| **OPD** | 1× | 1× | **1× per token** | **student rollout（$O(L)$ token）** | **2-3×**（看 teacher size） |
| **RLHF + PPO** | 1× | 1× | 0 | student rollout + RM forward | 2-4× |
| **GRPO** | $G$× (group sample) | 1× | 0 | $G$× student rollout | $G \cdot 1.5$× |

OPD 的额外开销主要来自：(1) student rollout（sampling 比 forward 慢，因 KV cache 累计）；(2) teacher forward（teacher 大）。但相比 RL，**OPD 不需要 reward model 训练 + RM forward**，所以总成本通常**小于纯 RL pipeline**。

### 8.2　显存

| 组件 | 显存（8B student + 32B teacher 例） |
|---|---|
| Student weight + optimizer | ~60 GB（Adam state + bf16 weight + bf16 grad） |
| Teacher weight (frozen, bf16) | ~64 GB |
| Student activations | ~10-20 GB（看 batch / seq） |
| Teacher activations (no grad) | ~5-10 GB |
| KV cache (student rollout) | ~5-10 GB |
| **Total** | **~150-170 GB** → 单 H100 (80GB) 跑不下，需 2-4 GPU |

> ⚠️ **本表假设了省显存的工程优化，§4.1 朴素代码原样跑会超出上表的 activations 预算** — 按本教程给出的典型配置（`batch_size=64`、`max_new_tokens=512`、vocab 100K-152K），§4.1 Route A 教学代码同时物化的 `s_logits / s_log_probs / s_probs / t_logits / t_log_probs` 五个 $[B,L,V]$ BF16 张量合计约 **33-50 GB**，已经超过上表 "Student activations (~10-20 GB) + Teacher activations (~5-10 GB)" 合计 15-30 GB 的预算。上表数字隐含假设了分块计算 / fused cross-entropy 等省显存优化；未做这些优化的朴素实现实际开销显著更高。

**降显存技巧**：
- Teacher 用 **separate inference server**（vLLM）跑，student 训练机不存 teacher weight
- Teacher 用 **fp8 / int8 quantization**（teacher 只 forward 不反传，精度损失小）
- Student 用 **LoRA**（rank 128）+ teacher 完整 weight
- **Async teacher inference**：student 训第 $N$ batch 时，teacher server 在算第 $N+1$ batch

### 8.3　Compute Efficiency

Thinking Machines blog 的核心数据点：

| Setting | AIME'24 准确率 | Total FLOPs |
|---|---|---|
| Qwen3-8B-Base + SFT only (OpenThoughts) | ~40% | 1× baseline |
| + RL (GRPO) | ~62% | 10-30× baseline |
| **+ OPD (Qwen3-32B teacher)** | **~62%** | **~1× baseline** |
| + RL 训到匹配 OPD | 62% | ~10× baseline |

**OPD 在同等准确率下 compute-efficient（以 Total FLOPs 计）约 9-30×**——这是 OPD 火起来的核心数字。注意这是 FLOPs/compute 维度的效率，与"需要多少条 rollout 样本"的 sample efficiency 是不同的量。

## §9 与相关方法的对比与定位

### 9.1　大表（cheat sheet 主考点）

| 方法 | 监督来源 | 数据收集 | KL 方向 | Critic | 适用场景 |
|---|---|---|---|---|---|
| **SFT** | dataset label | offline | n/a (CE) | no | warm start, instruction follow |
| **Hinton KD** | teacher logits | offline (dataset) | forward | no | 单步预测，分类 |
| **Kim-Rush Seq KD** | teacher beam search | offline | n/a (hard) | no | NMT, autoregressive |
| **DistilBERT** | teacher logits + MLM | offline | forward + CE | no | encoder model compression |
| **DPO** | offline preference pair | offline | closed-form | no | 无 strong teacher，有 pref data |
| **PPO RLHF** | learned RM | on-policy rollout | reverse (vs ref) | yes (value head) | 有 RM，want strict RM-aligned |
| **GRPO** | learned RM / verifier | on-policy rollout (group) | reverse (vs ref) | no (group baseline) | math / code，省 critic |
| **MiniLLM** | teacher logits | on-policy student rollout | reverse | no | LLM instruction tuning |
| **GKD** | teacher logits | mix offline + on-policy ($\lambda$) | JSD ($\beta$) | no | flexible，trade-off offline/online |
| **OPD** (general) | teacher logits per token | **on-policy student rollout** | **reverse (or JSD)** | no（baseline closed-form） | 小 student、long-CoT、强 teacher |
| **OPD + GRPO** | teacher logits + outcome verifier | on-policy group | reverse | no | math + dense supervision |
| **R1-Distill** | teacher trajectory | offline | n/a (CE) | no | massive synthetic SFT |

### 9.2　决策树：什么场景选什么方法

```
有 strong teacher 模型吗？
├── 否
│   ├── 有 preference dataset → DPO
│   ├── 有 verifier (math/code) → GRPO
│   └── 有 RM → PPO RLHF
└── 是
    ├── teacher 与 student gap 小 (< 4×) → off-policy SFT KD 就够
    ├── gap 中等 (4-10×) + long CoT 任务 → OPD（首选）
    ├── gap 大 (> 10×) + 复杂任务 → off-policy SFT warm start + OPD
    ├── 同时有 verifier → OPD + GRPO hybrid（生产标配）
    └── teacher 只能 API call → Black-Box OPD
```

## §10 25 高频面试题

### L1 必会（10 题，post-training 工程师 / LLM RL 岗）

<details>
<summary><strong>L1-1：OPD 是什么？为什么叫 "On-Policy"？</strong></summary>

**答**：OPD = On-Policy Distillation。"On-policy" 指**训练数据（trajectory）来自 student 当前 policy $\pi_\theta$ 自己采样**，而非来自 teacher / dataset。teacher 只在 student 自己访问到的状态上提供 per-token 监督信号（通常是 reverse KL）。这与 off-policy KD（teacher 自己生成数据 → student 模仿）形成对比。
</details>

<details>
<summary><strong>L1-2：OPD 与 vanilla KD（Hinton）的核心区别是什么？</strong></summary>

**答**：三点：(1) **数据来源**——vanilla KD 用 dataset / teacher 生成数据，OPD 用 student 自己 rollout；(2) **KL 方向**——vanilla KD 用 forward KL（mode-covering），OPD 主用 reverse KL（mode-seeking）；(3) **解决的问题**——vanilla KD 解决"压缩 teacher 知识"，OPD 解决"压缩 + exposure bias"（autoregressive 生成的 train/test 分布不一致）。
</details>

<details>
<summary><strong>L1-3：写出 OPD 的 reverse KL 损失公式。</strong></summary>

**答**：
$$L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]$$
关键：期望下标 $y \sim \pi_\theta$（on-policy），KL 方向是 $\pi_\theta$ 在前（reverse / mode-seeking）。Sampled-token 近似 = $\log \pi_\theta(y_t) - \log \pi_T(y_t)$。
</details>

<details>
<summary><strong>L1-4：什么是 exposure bias？OPD 怎么解决？</strong></summary>

**答**：exposure bias = autoregressive 模型训练时 prefix 来自 ground-truth / teacher（"完美 prefix"），推理时 prefix 来自模型自己（含错误）—— 训练 / 推理分布不一致。理论上累积误差按 $O(L^2)$ 放大（Bagnell 2010）。OPD 通过在 student 自己 rollout 的 trajectory 上算 loss，把训练分布 = 推理分布，把累积误差压到 $O(L)$。
</details>

<details>
<summary><strong>L1-5：reverse KL 与 forward KL 的区别？为什么 OPD 倾向用 reverse？</strong></summary>

**答**：forward KL $D(\pi_T \,\|\, \pi_\theta)$ 期望在 teacher 下取，**强制 student 覆盖 teacher 所有 mode**（mode-covering），易得到模糊平均的输出；reverse KL $D(\pi_\theta \,\|\, \pi_T)$ 期望在 student 下取，**强制 student 避开 teacher 不可能的 token**（mode-seeking / zero-forcing），易得到 sharp 自信的输出。LLM 生成任务通常要"流畅 + 自信"，所以 reverse KL 是首选。
</details>

<details>
<summary><strong>L1-6：OPD 需要 teacher 的什么？只有 teacher API（无 logit）能做 OPD 吗？</strong></summary>

**答**：标准 OPD 需要 teacher 在 student 访问的每个 prefix 上算 **logits** 或 **log-probabilities**。如果 teacher 是 closed-source API（如 GPT-4）只能拿 sample 不能拿 logit，可以用 "Black-Box OPD" (arXiv 2511.10643) ：用 teacher samples 做 trajectory-level reward，组合 student log-prob 做近似——但失去 per-token dense 监督，回退到 trajectory-level，性能介于 OPD 与 RL 之间。
</details>

<details>
<summary><strong>L1-7：OPD 与 RL 的核心差异是什么？为什么 OPD 更 sample efficient？</strong></summary>

**答**：RL 每条 trajectory 提供一个 $O(1)$ scalar reward（outcome），OPD 提供 $O(N \log V)$ bits（每 token 整个 teacher 分布）。所以同样数量的 rollout，OPD 提供的监督信息**多一到两个数量级**。Thinking Machines 报告 OPD 在数学推理上以 1/9-1/30 的 compute 达到 RL 同等性能。
</details>

<details>
<summary><strong>L1-8：OPD 需要 critic / value model 吗？</strong></summary>

**答**：**不需要**。在教程采用的 semi-gradient（B1，对 state visitation stop-grad）近似下，$D_{\text{KL}}(\pi_\theta(\cdot|s_t) \,\|\, \pi_T(\cdot|s_t))$ 是该 step **即时（single-step）reward** 的条件期望闭式值，可以从已经算好的 student 与 teacher logits 直接读出，用作该 step REINFORCE 估计量的零偏差控制变量 / baseline（"KL for a KL", arXiv 2605.07865），不需要单独训 critic——这是 OPD 相对 PPO 的一个工程优势。但严格来说它不是完整轨迹目标的真正状态价值函数 $V(s_t) = \mathbb{E}[\sum_{t'\ge t} D_{\text{KL}}(s_{t'})]$（后者需要对未来所有 step 的 KL 求 return-to-go，即 §2.3 中"几乎没人做"的完整 B2 形式）；把单步条件期望直接称为 $V(s_t)$ 是概念上的简化说法。
</details>

<details>
<summary><strong>L1-9：OPD 与 DPO 的关系？两者能叠加吗？</strong></summary>

**答**：DPO 是 offline、binary preference、closed-form RLHF；OPD 是 online、per-token teacher logit、policy-gradient KD。两者**几乎正交**：DPO 不需要 teacher，OPD 不需要 preference pair。可以叠加：先 DPO 对齐人类偏好（学"什么是好回答"），再 OPD 蒸馏到小 student（学"怎么生成"）。
</details>

<details>
<summary><strong>L1-10：OPD 在生产中最常用的 failure mode 是什么？怎么 mitigate？</strong></summary>

**答**：**Length inflation / truncation collapse**——student rollout 越训越长，撞 max_length 后被 truncate 的轨迹主导梯度，val 暴跌。Mitigate：(1) loss per-token average 而非 sum；(2) length penalty；(3) max_length 比 sample max 大 2-4×；(4) 监控 val，spike 立即停。
</details>

### L2 进阶（10 题，资深 post-training / 论文复现）

<details>
<summary><strong>L2-1：推导 OPD 的 policy gradient 形式（含两条路线）。</strong></summary>

**答**：reverse KL 在 trajectory 期望下：
$$L(\theta) = \mathbb{E}_{s_t \sim \rho_{\pi_\theta}}\!\big[D_{\text{KL}}(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t))\big].$$

$\theta$ 同时出现在 (a) **内层 KL**（$\pi_\theta(\cdot|s_t)$ 本身）和 (b) **state visitation** $\rho_{\pi_\theta}$。这导致 **两类不同的 gradient 计算**，文献常被混淆：

**(1) 内层 KL 的 REINFORCE 估计**（fixed-state $s_t$）：
$$\nabla_\theta D_{\text{KL}}(s_t) = \mathbb{E}_{y_t \sim \pi_\theta(\cdot|s_t)}\!\big[\nabla\log\pi_\theta(y_t|s_t)\cdot G_t^{\text{detach}}\big],\;\; G_t = \log\tfrac{\pi_\theta(y_t|s_t)}{\pi_T(y_t|s_t)}.$$
（推导：$\nabla\sum_v \pi_\theta(v)\log\tfrac{\pi_\theta(v)}{\pi_T(v)} = \sum_v\nabla\pi_\theta(v)\cdot(\log\tfrac{\pi_\theta}{\pi_T}+1) = \mathbb{E}_{y}[\nabla\log\pi_\theta(y)\cdot G + \nabla\log\pi_\theta(y)]$；第二项 $\mathbb{E}[\nabla\log\pi]=0$。）这是 §4.4 vOPD 用的 estimator。

**(2) Trajectory objective 的完整 policy gradient**（包含 state visitation）：
$$\nabla L = \mathbb{E}_\tau\!\Big[\underbrace{\sum_u \nabla\log\pi_\theta(a_u|s_u)\cdot {\textstyle\sum_{t\ge u}} D_{\text{KL}}(s_t)^{\text{detach}}}_{\text{return-to-go score term}} + \underbrace{\sum_t \nabla D_{\text{KL}}(s_t)}_{\text{inner grad}}\Big].$$
注意 score-function 权重是**未来所有 step 的 KL 之和**，不是同 token $G_t$。

**生产中的实践规则**：
- **Route A（教学清晰）**：rollout 时对 $\theta$ stop-grad（即 $\rho_{\pi_{\theta^-}}$ 固定），只反传内层 full-vocab KL（autograd 直接做），**不需要 REINFORCE**。这等价于"semi-gradient"近似。优势是零方差、一行 PyTorch；要求 teacher full logits 可用。（"零方差"特指给定固定 prefix 后内层 full-vocab KL 消除了 Route B1 的采样估计方差，训练流水线整体仍因 rollout 阶段 student 采样带有跨 batch 方差）
- **Route B（生产常见）**：sampled-token REINFORCE / importance-sampling estimator + control variate（§4.4）；与 PPO/GRPO 共享 sampled-token 接口、节约 teacher vocab memory、支持 logprob-only（灰盒）teacher（真正 sample-only 黑盒需 §6.5 Black-Box OPD）。MiniLLM (Gu 2024) 与 Thinking Machines Tinker 的开源实现都走这条，但两者近似程度不同：Tinker 走的是更贴近 (1) 的 fixed-state estimator；**MiniLLM 实际用带 future log-ratio return 的单步分解**（$R_t=\sum_{t'\ge t} r_{t'}$），是 (2) 的方差削减近似，不是纯粹的 semi-gradient。**完整、未削减的 (2) 才是几乎没人做的**——return-to-go 跨 step 累加，方差爆炸；生产中普遍在 (1) 附近或 MiniLLM 式的削减近似上折中。
- **不能**对 sampled-token $\log(\pi_\theta/\pi_T)$ 直接 `.backward()`：那是 pathwise + 固定 index，丢失 score-function 项，既不是 KL gradient 也不是 MLE。

**A 与 B 期望等价**，差异在方差 vs 工程接口 trade-off。

> "OPD 可以套 PPO/GRPO" 指的是 §4.5 中把 dense token KL **当作 reward** 喂进 GRPO 的 advantage（PPO ratio 仍是 student new vs old）；不是说"内层 KL gradient 形式与 PPO 等价"——这是常见的概念混淆。
</details>

<details>
<summary><strong>L2-2：GKD 与 MiniLLM 的关键差异？$\lambda$ 与 $\beta$ 分别控制什么？</strong></summary>

**答**：**MiniLLM** (Gu 2024)：纯 reverse KL on student rollout + REINFORCE 优化 + 一些稳定 trick（mixed policy $\pi_{\text{mix}} = (1-\alpha)\pi_\theta + \alpha\pi_T$，$\alpha = 0.2$ 防止 collapse；length penalty）。**GKD** (Agarwal 2024)：generalized 框架，两个 hyperparameter：(1) $\lambda \in [0, 1]$ 控制 **student-generated data fraction**（$\lambda=0$ 完全 off-policy on dataset $\hat y$，$\lambda=1$ 完全 on-policy on student $y$）；(2) $\beta \in [0, 1]$ 控制 **generalized JSD 的插值**——严格取极限时方向与直觉相反：$\beta \to 0$ 的单侧极限是 reverse KL（$D_\beta/\beta \to D_{\text{KL}}(\pi_\theta\|\pi_T)$），$\beta \to 1$ 的单侧极限是 forward KL（$D_\beta/(1-\beta) \to D_{\text{KL}}(\pi_T\|\pi_\theta)$），且 $\beta=0,1$ 处 $D_\beta$ 本身恒为 0（见 §2.1 表格端点说明）。GKD 是 MiniLLM 的 superset。
</details>

<details>
<summary><strong>L2-3：OPD 怎么集成进 GRPO？写出混合 reward 和正确的 ratio。</strong></summary>

**答**：核心两件事。
**(1) 混合 reward**（per-trajectory 标量，进 group-relative advantage）：
$$R(x,y) = \alpha\cdot R_{\text{outcome}}(x,y) + (1-\alpha)\cdot \sum_t \big(\log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)\big),$$
$\alpha=0$ 退化纯 OPD，$\alpha=1$ 退化纯 GRPO，生产 $\alpha\in[0.3,0.7]$。

**(2) ratio 的写法（最常错的地方）**：GRPO/PPO 的 ratio 是
$$\rho_t = \exp\big(\log\pi_\theta^{\text{new}}(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)\big),$$
即 **new student 与 old behavior student**——**teacher 绝不进入 ratio 分母**。Teacher 只在两个独立位置出现：(i) 作为 reward source（上面 $r_t^{\text{KL}}$）；(ii) 作为 reference distribution 出现在显式 KL penalty $D_{\text{KL}}(\pi_\theta\|\pi_T)$ 中（这一项闭式可微，§4.1 实现）。

最终 loss：
$$\mathcal{L} = -\mathbb{E}\!\left[\sum_t \min(\rho_t A_t, \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)\right] + \beta\, D_{\text{KL}}(\pi_\theta\,\|\,\pi_T).$$

advantage $A$ 用 GRPO 组内 z-score，不需要 critic。代码见 §4.5。
</details>

<details>
<summary><strong>L2-4：vOPD 的 control variate 是什么？为什么"免费"？</strong></summary>

**答**：vOPD ("KL for a KL"，2026 Survey 中常被引用的 control-variate 思路) 用在 Route B（sampled-token REINFORCE estimator）上降方差。对单个 sampled $y_t \sim \pi_\theta$，token-level reward $\hat r_t = \log\pi_\theta(y_t|s_t) - \log\pi_T(y_t|s_t)$（**必须 detach**），baseline 取 $B(s_t) = D_{\text{KL}}(\pi_\theta\,\|\,\pi_T)(s_t)$（也 detach）：

$$\nabla_\theta L \approx \mathbb{E}\!\big[\nabla\log\pi_\theta(y_t|s_t)\cdot (\hat r_t - B(s_t))\big].$$

$B(s_t)$ 是 $\hat r_t$ 在 $y_t \sim \pi_\theta(\cdot|s_t)$ 下的条件期望，加上后**保持 unbiased**（因为 $\mathbb{E}_y[\nabla\log\pi_\theta(y)\cdot B(s_t)] = B(s_t)\cdot\mathbb{E}[\nabla\log\pi_\theta] = 0$）且**通常显著降低方差**。严格意义上的 minimum-variance baseline 是按 score-norm 加权 $\mathbb{E}[\|g\|^2 r]/\mathbb{E}[\|g\|^2]$（一般不等于条件均值），但条件均值在实践中已经足够好用。**"免费"**指 $B(s_t)$ 就是 student forward 同步算出的 full-vocab 闭式 KL（同一次 logits → softmax → 求和），无需额外 critic / inference。

> ⚠️ 实现时反复出错的两点：(a) `r_hat` 和 `baseline` 都必须 `.detach()`，否则 unbiasedness 失效、gradient 变形；(b) surrogate 的**符号是正号**：因为 $\nabla D_{\text{KL}} = \mathbb{E}[\nabla\log\pi_\theta\cdot\hat r]$，所以 `loss = +E[\log\pi_\theta\cdot(\hat r - B).detach()]`，`∇loss = +∇D_{\text{KL}}`，optimizer step `θ -= η∇L` 就在做 KL 下降。**写成负号会让 KL 上升**——代码见 §4.4。
</details>

<details>
<summary><strong>L2-5：Qwen3 的 OPD recipe 与 Thinking Machines blog 实现的差异是什么？</strong></summary>

**答**：**Qwen3** 用 off-policy SFT 冷启动 + on-policy distillation 两阶段，目标是端到端 build small model；**Thinking Machines blog** 强调把 OPD 当 RL 的便宜替代品，core 实验是"用 OPD 复现 RL 的 AIME'24 gain，FLOPs 节省 9-30×"。技术细节上：Qwen3 同时蒸馏 /think 与 /no_think 两种模式（dual-mode logits）；Thinking Machines 用 Tinker，advantage 形式上把 KL 当 negative reward 注入（OPD-RL 视角）。两者本质都是 reverse KL on student rollout，差异在多任务 / 多模式蒸馏的处理。
</details>

<details>
<summary><strong>L2-6：OPD 的 "sampled-token KL" 与 "full-vocab KL" trade-off？</strong></summary>

**答**：(a) **sampled-token (REINFORCE / importance-sampling，Route B)**：reward 用 $G_t = \log \pi_\theta(y_t) - \log \pi_T(y_t)$.detach()（只 query teacher 一个 log-prob），grad = $\nabla\log\pi_\theta(y_t)\cdot G_t$；便宜但方差大。**MiniLLM (Gu 2024) 与 Thinking Machines Tinker 的开源实现都是这种 sampled-token PG/IS 形式**（Tinker `train_on_policy.py` 用 `loss_fn="importance_sampling"` + `incorporate_kl_penalty`；MiniLLM 用 single-step decomposition + length norm + teacher-mixed sampling 等 trick 稳定方差）。(b) **full-vocab (Route A)**：loss = $\sum_v \pi_\theta(v)(\log \pi_\theta(v) - \log \pi_T(v))$，直接 autograd；零方差但要 teacher full logits + 整 vocab 显存；GKD 等概念推导常用这种形式当教学起点，也是工程上 teacher full logits 可用时的最简实现。**两者期望等价**，trade-off：Route A 零方差、一行 PyTorch，但要 full logits + vocab memory；Route B 适配 PPO/GRPO sampled-token 接口、支持 logprob-only（灰盒）teacher（真正 sample-only 黑盒需 §6.5 Black-Box OPD）、节约 vocab memory，需要 control variate（§4.4）降方差。
</details>

<details>
<summary><strong>L2-7：OPD 为什么比 off-policy KD 在 long-CoT 任务上更好？</strong></summary>

**答**：long-CoT 上 student 自身 rollout 与 teacher rollout 的状态分布差距更大（错误累积 $O(L^2)$）。off-policy KD 训练时 student 见的全是 teacher prefix（光滑、正确），推理时遇到自己的错误 prefix 完全没见过——错误按 $O(L^2)$ 复合累积（而非指数级，与 §1.3/L1-4 的结论一致）。OPD 训练时就在 student 自己的错误 prefix 上学，teacher 给"我会怎么 recover"的监督——直接训"错误修复能力"。
</details>

<details>
<summary><strong>L2-8：如果 student 与 teacher gap 极大（如 1.5B vs 671B），OPD 会失败吗？怎么 mitigate？</strong></summary>

**答**：**会失败**——student 的 sample 大概率落在 teacher 概率极低的 region。这里的失败机制不是"KL 信号趋零"：一旦采到这种 token，$\log\pi_T(y_t|s_t)\to -\infty$ 会让该 token 的 reverse KL 项**趋于 $+\infty$**（数值极端、梯度不稳），而 gap 过大时真正拖累训练的是 student 自身采样分布可能从未覆盖 teacher 高概率 token，导致有效学习信号方向性差（"student 看起来在被 teacher 认可，但其实就是在乱说话"）。Mitigate：(1) **off-policy SFT warm start**：先让 student 在 teacher trajectory 上 SFT 一段，状态分布拉近后再 OPD；(2) **intermediate teacher**：用中等大小的 teacher（如 70B）当桥梁；(3) **curriculum**：先短 trajectory，再逐渐放长；(4) **temperature scheduling**：student temperature 初期高扩大探索。
</details>

<details>
<summary><strong>L2-9：OPD 与 R1-Distill 是同一类方法吗？</strong></summary>

**答**：**不是**。R1-Distill 是 **off-policy** SFT distillation——R1 teacher 离线生成 800K trajectory，student 在这些数据上做 token-level cross-entropy，**没有 student rollout、没有 KL 信号**。OPD 是 on-policy + teacher KL。两者互补：R1-Distill 可以作为 OPD 的 cold-start init（先模仿 teacher style），再用 OPD 消 exposure bias。
</details>

<details>
<summary><strong>L2-10：怎么诊断 OPD 训练是否 healthy？关键监控量是什么？</strong></summary>

**答**：监控五件套：(1) **per-token reverse KL**——应单调下降，但不能贴 0（贴 0 = student 与 teacher 完全一致，可能 overfit）；(2) **rollout length / truncation rate**——truncation rate < 5%，否则 length collapse；(3) **token entropy**——下降但不应接近 0（mode collapse）；(4) **pass@1 vs pass@N (N>1)**——pass@1 涨而 pass@64 跌 = diversity 塌缩；(5) **val accuracy on held-out**——这是终极信号，spike 立即停。
</details>

### L3 顶级 lab（5 题，研究 / 算法 lead）

<details>
<summary><strong>L3-1：从 KL-constrained RL 视角统一 OPD 与 RLHF / GRPO。</strong></summary>

**答**：KL-constrained RL 目标：
$$\max_\theta\, \mathbb{E}_{y \sim \pi_\theta}[R(x, y)] - \beta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\big]$$
- **RLHF / PPO**：$R$ = RM scalar reward, $\pi_{\text{ref}}$ = SFT init, $\beta$ 小
- **GRPO**：同 RLHF，但 advantage 用组内归一化代替 GAE
- **OPD**：$R \equiv 0$（无外部 reward）, $\pi_{\text{ref}} = \pi_T$（reference = teacher），$\beta = 1$
- **OPD + GRPO**（生产标配）：$R$ = outcome verifier + dense teacher KL reward, $\pi_{\text{ref}} = \pi_T$

Survey (arXiv 2604.00626) 把这统一称为 "$f$-divergence minimization on student rollout"，OPD 是这类问题的 $R \equiv 0$ 特例，DPO 是 closed-form $R$ + offline 特例。**注意 RLHF 不是 $\beta \to 0$ 特例**：一般 KL-constrained RL 目标在 $\beta$ 取常见小正值（如 0.01-0.1）时本身就是 RLHF/PPO 的标准形式，并非某个极限特例；相反，$\beta \to 0$ 对应去掉 KL 正则、纯粹 reward 最大化的无约束 RL（易 reward hacking），这与 RLHF 保留有限非零 KL 惩罚以防止偏离参考策略的核心特征相悖。
</details>

<details>
<summary><strong>L3-2：OPD 的理论收敛性分析有哪些已知结果？</strong></summary>

**答**：核心已知结果（截至 2026-05）：(1) **不动点**：$L_{\text{OPD}} = 0$ iff $\pi_\theta = \pi_T$ on the support of $\pi_\theta$（reverse KL 性质，且只在 student 访问的支撑上对齐）。所以 OPD 不能让 student "超越" teacher——但能让 student 在自己的 capacity 内最大化模仿 teacher 的某个 mode；(2) **收敛性**：在凸 policy parametrization 假设下，policy gradient 收敛到 reverse KL 的局部最小（Geist & Pietquin 2014 风格），但 LLM 的非凸 parametrization 没有 global 保证；(3) **Rethinking OPD** (arXiv 2604.13016) 指出 OPD 可以在 reasoning 任务上"超越 teacher"——这看似矛盾，但解释是 teacher logits 中包含 dark knowledge（如 self-correction signal），student 通过 on-policy 训练激活了 teacher 也未必能稳定发挥的能力。这是 OPD 与传统 imitation learning 的关键区别。

**[needs-verify]** "超越 teacher" 现象在不同 paper 报告不一致，需查 Rethinking OPD 原文细节。
</details>

<details>
<summary><strong>L3-3：multi-teacher OPD 怎么做？DeepSeek-V4 报告的"OPD 替代 mixed RL"是什么意思？</strong></summary>

**答**：multi-teacher OPD 把多个 specialist teacher 的 logits 在每个 token 上加权 ensemble：
$$\pi_T(v|s_t) = \sum_k w_k(s_t) \cdot \pi_{T_k}(v|s_t)$$
weight $w_k$ 可以是固定权重（如 math task 上 math teacher 权重高）、context-dependent（routing-style）或可学习。DeepSeek-V4 在 model consolidation 阶段用 math / code / chat / reasoning 多个 specialist teacher 做 multi-teacher OPD，完全替代了之前 mixed RL（多 RM 加权）。优势是 **每 token 都有 dense supervision**，比 multi-RM 的 mixed RL sample efficient 显著。

**[needs-verify]** DeepSeek-V4 的具体 multi-teacher 实现细节（weight 选择、是否 token-level routing）需查原 tech report；目前公开材料主要是 secondary source 引用。
</details>

<details>
<summary><strong>L3-4：OPD 的 "process reward" 视角与 PRM 的关系？两者能融合吗？</strong></summary>

**答**：把 OPD 的 per-token teacher KL 看成 "process reward"：每个 token（在 sampled-token / behavior-policy rollout 上）都有一个 dense reward $r_t = \log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t) = -\log\!\frac{\pi_\theta^{\text{old}}(y_t|s_t)}{\pi_T(y_t|s_t)}$（与 §3 / §4.5 的 OPD-GRPO 一致；上标 old 表示 rollout 时记录的 behavior policy log-prob，避免误读 $-\log\pi_\theta/\pi_T$）。这与 PRM (process reward model, Lightman 2023 "Let's Verify Step by Step") 在思想上一致——都是 dense 而非 sparse supervision。差异：PRM 是 step-level（每个推理 step 一个 0/1），OPD 是 token-level（每 token 一个 KL 值）。**融合方案**：(1) step-level reward = $\sum_{t \in \text{step}_k} r_t^{\text{OPD}} + \lambda \cdot r_k^{\text{PRM}}$；(2) 用 PRM 当 trajectory filter（PRM 高分的 trajectory 才进 OPD 训练）；(3) 用 OPD 蒸馏一个 PRM（dense teacher signal 训 process-level verifier）。学术上这块是 2026 active research。

**[needs-verify]** "OPD + PRM 融合"具体 paper 与实验结果尚不完整；上述方案是综合多源材料后的合理推断。
</details>

<details>
<summary><strong>L3-5：从 information-theoretic 视角分析 OPD 为什么能 9-30× sample efficient 于 RL。</strong></summary>

**答**：考虑 trajectory $y$ 长度 $N$，vocab $V$。每条 trajectory 上：

- **RL outcome reward**：1 个 scalar，最多 $\log_2 V_R$ bits（$V_R$ = reward 离散度，二元 reward $\log_2 2 = 1$ bit；real-valued 约 $\log_2 1000 \approx 10$ bits）
- **OPD per-token teacher KL（sampled）**：每 token 一个 $\log \pi_T(y_t)$ 值，约 $\log_2 V \approx 17$ bits（典型 vocab 100K-128K）；trajectory 累计 $N \cdot 17$ bits
- **OPD per-token full KL**：每 token 整个 vocab 分布，理论上限 $\log_2 V$ bits per token（但实际信息量取决于 teacher 分布 entropy）

bit-rate 比：OPD / RL ≈ $N \cdot 17 / 10 = 1.7N$。在 long-CoT 任务（$N = 2K$-$8K$）上该比值约为 **3400-13600×** 量级——这从信息论角度解释了 Thinking Machines 报告的 9-30× compute efficiency 为什么有数量级的空间存在，但实际效率远没达到这个信息论上限。

**caveat**：这是 upper bound 论证，有两层不严谨之处需要注意。(1) 把连续标量 $\log \pi_T(y_t)$ 直接等同于携带 $\log_2 V \approx 17$ bits 信息是一种启发式类比——$\log_2 V$ 只是"从 $V$ 个离散候选中选 1 个"这一分类结果的信息熵上界，并不严格等于对数概率这个连续值本身携带的信息量；此 bit-rate 论证应视为启发式上界，而非严格的信息论推导。(2) 实际 compute efficiency 还受梯度噪声、teacher / student gap、optimizer 等影响，远达不到 3400-13600× 的理论上限。OPD 在简单任务上的 advantage 通常不到 10×，在 long-horizon 复杂任务上才能逼近报告中的 9-30×。
</details>

## §A 附录

### A.1　Sanity-check：用 OPD 收敛性检验你的实现

实现 OPD 后，做这三个 micro-test 确认 loss 正确：

```python
# Test 1: student == teacher → loss should be ~0
student.load_state_dict(teacher.state_dict())
loss, _ = per_token_reverse_kl_loss(student, teacher, ids, mask)
assert abs(loss.item()) < 1e-4, f"identical model should give zero KL, got {loss}"

# Test 2: student random init → loss > 0
student = init_random_model(...)
loss, _ = per_token_reverse_kl_loss(student, teacher, ids, mask)
assert loss.item() > 0.5, f"random student should give positive KL, got {loss}"

# Test 3: loss should decrease over training
losses = []
for step in range(100):
    loss = opd_train_step(student, teacher, ...)
    losses.append(loss)
assert losses[-1] < losses[0], "OPD should reduce KL over training"
```

### A.2　常见错误与正确做法

| 错误做法 | 现象 | 正确做法 |
|---|---|---|
| 在 teacher trajectory 上算 OPD loss | 退化为 off-policy KD，丢失 on-policy 价值 | 必须 student 自己 rollout |
| KL 方向写反（forward KL 当 reverse KL） | mode-covering，输出平庸 | reverse KL 是 $\pi_\theta$ 在前 |
| Mask 把 prompt token 也算进 loss | prompt token 上 KL 信号无意义 | `action_mask` 只标 student 生成的 token |
| 用 sum reduction 不 normalize 长度 | length inflation 训练崩盘 | per-token average（除以 mask.sum()） |
| Teacher 用 train mode（dropout） | logits 不稳，loss 噪声大 | teacher 必须 `.eval()` + `torch.no_grad()` |
| Student rollout 用 greedy decode | trajectory diversity 太低，OPD 学不到 robustness | sampling with temperature ≥ 1.0 |
| 不监控 truncation rate | 撞 max_length 后 silent failure | monitor; 超 5% 就调 max_new_tokens |

### A.3　核心 paper 与资源列表

**核心 OPD paper**：
- MiniLLM (Gu et al. 2023, ICLR 2024) — arXiv 2306.08543
- GKD: On-Policy Distillation of Language Models (Agarwal et al. 2023, ICLR 2024) — arXiv 2306.13649
- A Survey of On-Policy Distillation for LLMs (Song & Zheng 2026) — arXiv 2604.00626
- Rethinking On-Policy Distillation (2026) — arXiv 2604.13016
- KL for a KL (vOPD, 2026) — arXiv 2605.07865
- Black-Box On-Policy Distillation (2026) — arXiv 2511.10643
- Decoupling KL and Trajectories (2026) — arXiv 2605.16826

**工业 tech report**：
- Qwen3 Technical Report (Qwen Team 2025-05) — arXiv 2505.09388 (§3.2 OPD recipe)
- DeepSeek-R1 (DeepSeek 2025-01) — arXiv 2501.12948 (off-policy distillation series)

**Blog / 代码**：
- Thinking Machines Lab — "On-Policy Distillation" blog (Lu et al. 2025-10-27) — `thinkingmachines.ai/blog/on-policy-distillation/`
- Tinker Cookbook — `github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation`
- TRL GKD Trainer — `huggingface.co/docs/trl/gkd_trainer`
- Awesome OPD list — `github.com/thinkwee/AwesomeOPD`, `github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation`

**相关基础**：
- Hinton, Vinyals, Dean — "Distilling the Knowledge in a Neural Network" (2015) — arXiv 1503.02531
- Sanh et al. — "DistilBERT" (2019) — arXiv 1910.01108
- Kim & Rush — "Sequence-Level Knowledge Distillation" (EMNLP 2016) — arXiv 1606.07947
- Bagnell — "Reinforcement Learning and Imitation Learning" (theoretical foundation for exposure bias)
- DeepSeekMath GRPO (2024) — arXiv 2402.03300
- DPO (Rafailov et al. NeurIPS 2023) — arXiv 2305.18290

### A.4　[needs-verify] 标记一览

本 cheat sheet 中下列内容标记为 **[needs-verify]**，建议在面试前后查原始 paper / tech report 确认：

1. **L3-2 "OPD 超越 teacher"**：Rethinking OPD (arXiv 2604.13016) 报告的具体 setting 与 magnitude
2. **L3-3 DeepSeek-V4 multi-teacher OPD 实现细节**：weight 选择策略、是否 token-level routing
3. **L3-4 "OPD + PRM 融合"**：截至 2026-05 active research，尚无单一权威 paper 整合两者
4. **§6.2 Thinking Machines 数据**："9-30× FLOPs 节省" 来自 blog secondary source，原 blog 数字与具体 setting 应核对
5. **§5.2 timeline**：2025-2026 多篇 OPD-related arXiv paper 编号（如 2604.* 系列）来自 2026 Q1-Q2 投稿/预印本，部分 ID 可能在投稿后更新版本号或重排
6. **OPD 在 Qwen3 / MiMo 上的具体采用细节**：多数信息来自 Thinking Machines blog 与 Qwen3 paper §3.2，但 MiMo 的 distillation 部分细节需查其 tech report；**Gemma 2/3 已重新归类为 off-policy soft-target KD（非 OPD，见 §5.3 与 §0 TL;DR 第 7 条）**，这一归类基于对 Gemma 2 技术报告方法描述的回忆，未做逐字核对，建议再查原始报告确认
7. **§6.1 Qwen3 具体数值**：pass@64（vs pass@1）与 "+3-5pp" 的精确度量口径待核对原始 Qwen3 Technical Report §3.2（arXiv 2505.09388）

### A.5　术语速查表

| 中文 | 英文 | 含义 |
|---|---|---|
| 在线策略蒸馏 | On-Policy Distillation (OPD) | student 在自己 rollout 上做 KL 蒸馏 |
| 离线蒸馏 | Off-Policy Distillation | student 在 teacher / dataset 数据上蒸馏 |
| 暴露偏置 | Exposure Bias | autoregressive 模型 train/test 分布不一致 |
| 反向 KL | Reverse KL | $D(\pi_\theta \,\mid \, \pi_T)$，mode-seeking |
| 正向 KL | Forward KL | $D(\pi_T \,\mid \, \pi_\theta)$，mode-covering |
| 模式寻找 | Mode-Seeking | 锁定一个 mode，sharp 输出 |
| 模式覆盖 | Mode-Covering | 覆盖所有 mode，平均输出 |
| 教师强制 | Teacher Forcing | 训练时用 ground-truth prefix |
| 控制变量 | Control Variate | 降方差用的 baseline 项 |
| 截断坍塌 | Truncation Collapse | rollout 越训越长撞 max_length 导致崩盘 |
| 局部可教性塌缩 | Local Teachability Collapse | trajectory 后半 teacher 没东西可教 |

> ⚠️ **caveat** — OPD 作为 LLM post-training 的独立技术名词主要在 **2025 下半年**（Qwen3 + Thinking Machines blog）才广泛流行。在此之前同样的方法在 MiniLLM (2023) 与 GKD (2023) 中已经提出。所以"OPD 是新方法"在严格意义上不正确——它是被重新命名 + 大规模工业化的旧方法，受益于 reasoning model 时代对 dense supervision 的渴求。这一历史脉络在面试 L3 上常被问及，请注意区分"方法首次提出年份"与"术语流行年份"。
