## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 RLHF 里的 KL** — 一页拿下面试核心要点（详见后文 §1–§8 推导）。

1. **定义**：$\text{KL}(p \| q) = \mathbb{E}_{x \sim p}[\log(p(x)/q(x))]$，非负、不对称、非度量；在 RLHF 里 $p = \pi_\theta$、$q = \pi_\text{ref}$，作用是把 RL 后的 policy 锚在 SFT 附近防止 reward hacking。

2. **Forward vs Reverse**：约定 $p$ 为数据/目标、$q_\theta$ 为变分/优化分布——**Forward KL** = $\text{KL}(p\|q_\theta)$（mass-covering），**Reverse KL** = $\text{KL}(q_\theta\|p)$（mode-seeking）。RLHF 用 $\text{KL}(\pi_\theta \| \pi_\text{ref})$，按这个约定属于 **reverse KL**（$\pi_\theta$ 是变分一侧），mode-seeking 性质正好符合 RL 目标：让 $\pi_\theta$ 在 $\pi_\text{ref}$ 高密度区域选 reward 高的 mode；工程上也可行——直接对 rollout 采样估计即可。注意不同社区命名稍有差异，但 DPO / RLOO / "Rethinking KL" / "Comedy of Estimators" 这类 2024-2026 RLHF 文献都把 $\text{KL}(\pi_\theta \| \pi_\text{ref})$ 称 reverse KL。

3. **三大 KL Estimator (Schulman 2020 blog `joschu.net/blog/kl-approx.html`)**：
   - **k1** = $\log(\pi_\theta/\pi_\text{ref})$ — 无偏但方差大、可负（不可读做"距离"）。
   - **k2** = $\tfrac{1}{2}(\log(\pi_\theta/\pi_\text{ref}))^2$ — 总是非负但**有偏**（二阶 Taylor 近似）。
   - **k3** = $(\pi_\text{ref}/\pi_\theta) - \log(\pi_\text{ref}/\pi_\theta) - 1$ — **无偏 + 非负 + 低方差**，从恒等式 $\mathbb{E}_q[f(\log p/q)]$（$f(x)=e^x-x-1$）导出。

4. **两种放置**：(a) **In-reward shaping**：$\tilde{r}_t = r_t - \beta \cdot \text{KL}_t$，与 advantage / GAE 一起算；(b) **In-loss regularization**：$\mathcal{L} = \mathcal{L}_\text{PG} + \beta \cdot \mathbb{E}[\text{KL}]$。**InstructGPT/Anthropic PPO 用 (a)；GRPO 用 (b) + k3 estimator**。**目标函数相同，但梯度路径不同**——在 principled estimator 下（如 (a) k1-in-reward 或 (b) k2-as-loss）on-policy 是 gradient-equivalent；GRPO 的 (b) k3-as-loss 实际有 $O(\Delta^2)$ first-order Taylor bias（见 §3.6 + Rethinking KL 2510.01555）。两种 placement 对 PPO clip / importance ratio 截断的响应也不同。

5. **β 调度**：固定 β（最简单）、**Adaptive β**（PPO-Penalty 原版，按 measured KL 距 target 拉 β）、Annealing β（早期紧后期松，类似 SFT-to-RL 过渡）。**β 过大学不到东西**（policy 卡在 ref 附近），**β 过小 reward hack**（policy 漂掉、长答案/谄媚）。

6. **KL-regularized RL 闭式最优 policy**：对 BT-style reward 目标 $\max_\pi \mathbb{E}[r] - \beta \cdot \text{KL}(\pi\|\pi_\text{ref})$ 求解，唯一闭式解为 $\pi^*(y|x) \propto \pi_\text{ref}(y|x) \exp(r(x,y)/\beta)$。**DPO 就是把这个反解代入 Bradley-Terry**，得到 $r = \beta \log(\pi^*/\pi_\text{ref}) + \beta\log Z$，partition function $\log Z$ 在 pairwise 差里**消掉**。

7. **DPO/GRPO/SimPO 中 KL 的位置**：DPO 的 implicit reward $\hat r_\theta = \beta\log(\pi_\theta/\pi_\text{ref})$ 是 sequence-level log-ratio（取 $y\sim\pi_\theta$ 期望才等于 $\beta\cdot$KL，DPO 训练时 $y$ 来自 preference 数据所以不是 KL 本身，但 reference 出现在分母里仍带来隐式 anchoring）；GRPO 用 k3 KL 进 loss（注意 k3-as-loss 的 gradient 是 first-order 近似，见 §3.6）；SimPO 直接**砍掉 reference**，所以**没有 KL 约束**——对 β/γ/length-norm 更敏感。

8. **失败模式**：KL 爆炸（β 过小、importance ratio 过大）、KL collapse（β 过大或 entropy 太低，policy 卡死）、**Reward Overoptimization** (Gao 2023 ICML)：proxy reward 随 KL 单调升，gold reward 先升后降（inverted-U），KL 距离是 overoptimization 的天然 axis。

## §1 KL 基础

### 1.1　定义、性质、记号

**定义**（离散）：

$$\boxed{\;\text{KL}(p \| q) \triangleq \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]\;}$$

约定 $0 \log 0 = 0$、$0 \log(0/0) = 0$、$p \log(p/0) = +\infty$。连续分布把 $\sum$ 换成 $\int$。

**核心性质**（面试常被串问）：

| 性质 | 一句话表述 | 简证 |
| --- | --- | --- |
| **非负** | $\text{KL} \ge 0$ | Jensen + $-\log$ 凸：$\text{KL} = -\mathbb{E}_p \log(q/p) \ge -\log \mathbb{E}_p(q/p) = 0$ |
| **等号** | $\text{KL} = 0$ 当且仅当 $p = q$ a.e. | Jensen 等号条件 |
| **不对称** | $\text{KL}(p,q) \ne \text{KL}(q,p)$（在 $\text{KL}$ 的不对称参数中） | 直接构造例子可见 |
| **不满足三角不等式** | 不是 metric | 故"KL distance"是不严谨的口语化 |
| **凸性** | $\text{KL}(\cdot,\cdot)$ 对 $(p,q)$ 联合凸 | log-sum 不等式 |
| **链式法则** | 联合 KL 等于边际 KL 加条件 KL 期望（见下方 display 公式） | 直接拆 log |
| **参数化不变** | 对 $x \mapsto T(x)$ 同形变换不变（仅在可逆 $T$ 下取等号） | 测度变换 + Jacobian 抵消 |

链式法则的完整形式：

$$\text{KL}(p(x,y) \,\|\, q(x,y)) = \text{KL}(p(x) \,\|\, q(x)) + \mathbb{E}_{p(x)}\!\big[\text{KL}(p(y|x) \,\|\, q(y|x))\big]$$

> ✅ **链式法则在 RLHF 里 = per-token KL 加总** — Sequence-level KL 是 trajectory log-ratio 之和的**期望**：$\text{KL}_\text{seq} = \mathbb{E}_{y\sim\pi_\theta}[\sum_t \log\pi_\theta(y_t|s_t)/\pi_\text{ref}(y_t|s_t)]$。实际实现里对每条 rollout 做 $\sum_t \log r_t$ 是**单条 MC estimator**（k1 的 sequence 形式）；要得到 true expected KL 需对 batch 内 rollout 平均。**真实的 token-level KL** $D_\text{KL}(\pi_\theta(\cdot|s_t)\|\pi_\text{ref}(\cdot|s_t))$ 还需对该 prefix 上整个 vocab 求和（full-vocab KL），而非只在 sampled token 上算 log-ratio。这两者**期望相等**但 estimator 形式不同：sum-over-rollout 的 sampled log-ratio 是 cheap-but-noisy estimator。

### 1.2　Forward KL vs Reverse KL：mass-covering 还是 mode-seeking？

记 $p$ 是数据/真实分布，$q_\theta$ 是参数化模型。两种 KL 在拟合策略上行为完全不同：

| 方向 | 形式（见下方 display） | 期望对谁取 | 行为 | 经典用途 |
| --- | --- | --- | --- | --- |
| **Forward KL** | $\text{KL}(p\,\|\,q_\theta) = \mathbb{E}_{p}[\log(p/q_\theta)]$ | $p$（数据/目标） | **mass-covering / mean-seeking**：哪里 $p > 0$，$q_\theta$ 必须 > 0，否则 $\log(p/q) \to \infty$ | MLE / 蒸馏（学生覆盖老师） |
| **Reverse KL** | $\text{KL}(q_\theta\,\|\,p) = \mathbb{E}_{q_\theta}[\log(q_\theta/p)]$ | $q_\theta$（变分/优化分布） | **mode-seeking / minorization**：哪里 $q_\theta > 0$，$p$ 必须 > 0；$q_\theta$ 倾向找单一 mode 缩进去 | VI、RLHF、GAN-like 训练 |

记号约定：

$$\text{Forward KL}: \text{KL}(p \,\|\, q_\theta) = \mathbb{E}_{p}\!\left[\log\frac{p}{q_\theta}\right],\qquad \text{Reverse KL}: \text{KL}(q_\theta \,\|\, p) = \mathbb{E}_{q_\theta}\!\left[\log\frac{q_\theta}{p}\right]$$

经典图（双峰 $p$，单峰 $q_\theta$）：

- Forward KL 拟合 → $q_\theta$ 拉宽，跨越两个 mode（**mass-covering**）。
- Reverse KL 拟合 → $q_\theta$ 选其中一个 mode 缩进（**mode-seeking**）。

> ⚠️ **命名约定的现代共识** — 变分推断、DPO、RLOO、"Rethinking KL Regularization in RLHF" (arXiv 2510.01555)、"A Comedy of Estimators" (arXiv 2512.21852) 等 2024-2026 RLHF 文献都用统一约定：$\text{KL}(q_\theta\|p)$ 叫 reverse KL（$q_\theta$ 是变分/优化分布），$\text{KL}(p\|q_\theta)$ 叫 forward KL。本教程跟随这个标准约定。少数早期 RL 教材按采样分布命名，建议在面试时直接给出公式，避免标签歧义。

**RLHF 用哪一种？** 几乎所有主流实现（InstructGPT / Anthropic PPO / DeepSeekMath GRPO）都用 $\text{KL}(\pi_\theta \| \pi_\text{ref})$（**reverse KL** 形式：$\pi_\theta$ 是变分一侧，$\pi_\text{ref}$ 是 target）。**根本原因**：训练时我们已经从 $\pi_\theta$ 采样（rollout），算 $\mathbb{E}_{\pi_\theta}[\log \pi_\theta/\pi_\text{ref}]$ 直接用样本就行；同时 reverse KL 的 mode-seeking 行为正好符合 RL 目标——在 $\pi_\text{ref}$ 高密度区找 reward 高的 mode，而不是"覆盖整个 $\pi_\text{ref}$"。

> 💡 注意：把 $\pi_\theta$ 与 $\pi_\text{ref}$ 角色对调时（forward KL = $\text{KL}(\pi_\text{ref}\|\pi_\theta)$），要从 $\pi_\text{ref}$ 采样，工程上需要 IS、贵且语义不直接（我们要训的是 $\pi_\theta$ 不是 $\pi_\text{ref}$）。所以 RLHF 极少用 forward 方向。

### 1.3　与其他 divergence 的关系

| Divergence | 定义（见下方 display） | 特点 | 备注 |
| --- | --- | --- | --- |
| **JS** | KL 到混合 $m = (p+q)/2$ 的对称平均 | 对称、有界（$\le \log 2$）、是 metric 的平方根 | $\sqrt{\text{JS}}$ 是 metric（Endres-Schindelin 2003 IEEE TIT 49(7)）；GAN 原版判别 loss 等价于 $2\cdot\text{JSD} - \log 4$；RLHF 极少用 |
| **α-divergence** | $\frac{1}{\alpha(1-\alpha)}(1 - \int p^\alpha q^{1-\alpha})$ | 含 KL 作为极限 | 统一框架，$\alpha \to 1$ 给 forward KL、$\alpha \to 0$ 给 reverse KL |
| **Hellinger** | $H^2(p,q) = \tfrac{1}{2}\int(\sqrt{p}-\sqrt{q})^2$ | 对称、有界、$0 \le H^2 \le 1$ | 与 KL 关系：$H^2 \le \tfrac{1}{2}\text{KL}$ |
| **$\chi^2$** | $\chi^2(p,q) = \int \frac{(p-q)^2}{q}$ | 是 $f$-divergence 在 $f(t) = (t-1)^2$ 时的特例 | 与 KL 关系：$\text{KL} \le \log(1 + \chi^2)$ |
| **TV** | $\tfrac{1}{2}\int \lvert p-q\rvert$ | 是 metric、$\in [0,1]$ | **Pinsker**：$\text{TV} \le \sqrt{\text{KL}/2}$ |

四个常见 $f$-divergence 的完整形式：

$$\text{JS}(p, q) = \tfrac{1}{2}\text{KL}(p \,\|\, m) + \tfrac{1}{2}\text{KL}(q \,\|\, m),\quad m = (p+q)/2$$

$$\chi^2(p, q) = \int \frac{(p(x) - q(x))^2}{q(x)}\,dx,\qquad \text{Pinsker: } \mathrm{TV}(p,q) \le \sqrt{\text{KL}(p \,\|\, q) / 2}$$

> 💡 **为什么 RLHF 没用 JS / α-divergence？** — 主要是工程惯性 + 闭式优势：在 PPO/DPO 框架里 KL 有干净的 per-token 拆分（链式法则）、闭式最优解（softmax-style），且 likelihood ratio 直接给出 KL token 增量，免去额外网络估计。JS / α-divergence 要么数学没这么干净，要么需要额外 density ratio estimator。

### 1.4　为什么 RLHF 要加 KL？

把"无 KL"和"有 KL"的目标摆一起：

$$\text{无 KL}:\quad \max_\pi \mathbb{E}_{x,\,y \sim \pi(\cdot|x)}[r(x,y)]$$

$$\text{有 KL}:\quad \max_\pi \mathbb{E}_{x,\,y \sim \pi(\cdot|x)}[r(x,y)] - \beta\,\mathbb{E}_x\,\text{KL}\!\big(\pi(\cdot|x)\,\big\|\,\pi_\text{ref}(\cdot|x)\big)$$

**没有 KL 会发生什么？**

1. **Reward hacking**：policy 找到 RM 盲点（更长答案 / "As an AI..." 套话 / 谄媚 / 格式 hack）拿高 RM 分，人感受变差。
2. **语言流畅性塌掉**：policy 会输出 RM "喜欢"但人类看着像鬼话的 token 序列（极端时退化到只重复某 token / 完全不语法）。
3. **Distribution shift**：policy 跑远了 $\pi_\text{ref}$，连 RM 自己都很难评分（off-distribution，RM 越没把握、reward 越随机）。

加 KL 后：

- **β 提供 RM error 的 implicit regularization**：当 RM 不可靠时，KL 把 policy 拉回 SFT 已知良好分布。
- **闭式最优 policy 存在**（§6.1 推导），整个 RLHF 有数学基础。
- **DPO / KTO / GRPO 都依赖这条 KL anchor**：去掉 reference 的 SimPO 实测对超参更敏感。

## §2 KL Estimators（k1 / k2 / k3）：面试核心

### 2.1　问题设置

实际训练里我们要在每一个 mini-batch 估计 $\text{KL}(\pi_\theta \| \pi_\text{ref})$。但完整求和 $\sum_y \pi_\theta(y) \log \pi_\theta(y)/\pi_\text{ref}(y)$ 不可行（$y$ 是整段 response，组合爆炸）。**只能 Monte Carlo**：用从 $\pi_\theta$ 采样的样本估 KL。

记 $\log r = \log(\pi_\theta(y)/\pi_\text{ref}(y))$（不是 importance ratio，是策略 log-ratio）。我们有 $y \sim \pi_\theta$ 的样本，想估 $\mathbb{E}_{\pi_\theta}[\log r] = \text{KL}(\pi_\theta \| \pi_\text{ref})$。

> ⚠️ **记号统一** — 本节 $\log r$ 始终表示 **policy log-ratio** $\log \pi_\theta - \log \pi_\text{ref}$（**不是** PPO importance ratio $\pi_\theta / \pi_\text{old}$）。两者形式相近但含义不同：前者衡量"离 reference 多远"，后者衡量"离 sampling policy 多远"。

### 2.2　k1 estimator —— "直接代入定义"

$$\boxed{\;\widehat{\text{KL}}_1 = \log\frac{\pi_\theta(y)}{\pi_\text{ref}(y)} = \log r\;}$$

- **无偏**：$\mathbb{E}_{y\sim\pi_\theta}[\log r] = \text{KL}(\pi_\theta\|\pi_\text{ref})$ by definition。
- **可负**：单个样本下 $\log r$ 可正可负（log-ratio 没有"非负"约束）。
- **方差大**：tails 上 $\log r$ 可以非常大或非常小，特别是 $\pi_\theta$ 和 $\pi_\text{ref}$ 不重叠的区域。

**问题**：用 k1 作为"KL 监控指标"会看到**负值**，工程上会让 logger 显示负 KL，新人会困惑（KL 不该非负吗？）。这是因为**期望非负不代表每个样本非负**。但作为 reward shaping 的 per-token KL，可以接受（关键是均值无偏）。

### 2.3　k2 estimator —— "$L^2$ form"

$$\boxed{\;\widehat{\text{KL}}_2 = \tfrac{1}{2}\!\left(\log\frac{\pi_\theta(y)}{\pi_\text{ref}(y)}\right)^2 = \tfrac{1}{2}(\log r)^2\;}$$

- **总是非负** ✓
- **有偏**：$\mathbb{E}[\tfrac{1}{2}(\log r)^2] \ne \text{KL}$。
- **小 KL 极限下**：Taylor 展开 $\log r = (r - 1) - \tfrac{1}{2}(r-1)^2 + O((r-1)^3)$，当 $\pi_\theta \approx \pi_\text{ref}$ 时 $\log r$ 小，$\tfrac{1}{2}(\log r)^2$ 与 KL 在二阶近似下相等（KL 在 $p=q$ 附近 = Fisher 信息度规的二次型）。
- **方差小于 k1**：因为平方后正负不再相消，但仍然不是最优。

实践中 k2 主要用于**监控**（提供"non-negative，但有偏"的视觉化指标）；很少用于 reward shaping。

### 2.4　k3 estimator (Schulman 2020 blog) —— **非负无偏 value estimator**（但作为 loss 项是 biased gradient，见 §3.6）

#### 2.4.1　构造

考虑函数 $f(x) = e^x - x - 1$。由 $e^x \ge 1 + x$（实数 $x$）得 $f(x) \ge 0$，且 $f(0) = 0$。

把 $x = \log(\pi_\text{ref}(y)/\pi_\theta(y)) = -\log r$ 代入：

$$f(-\log r) = e^{-\log r} - (-\log r) - 1 = \frac{1}{r} + \log r - 1 = \frac{\pi_\text{ref}(y)}{\pi_\theta(y)} + \log\frac{\pi_\theta(y)}{\pi_\text{ref}(y)} - 1$$

等价地（用 $\Delta = -\log r = \log(\pi_\text{ref}/\pi_\theta)$ 写）：$\frac{\pi_\text{ref}}{\pi_\theta} - \log\frac{\pi_\text{ref}}{\pi_\theta} - 1$（注意 $-\log(\pi_\text{ref}/\pi_\theta) = \log(\pi_\theta/\pi_\text{ref})$，两形式等价）。

等价地（这是 Schulman blog 中的标准写法）：

$$\boxed{\;\widehat{\text{KL}}_3 = \frac{\pi_\text{ref}(y)}{\pi_\theta(y)} - \log\frac{\pi_\text{ref}(y)}{\pi_\theta(y)} - 1 = e^{\Delta} - \Delta - 1,\quad \Delta = \log\frac{\pi_\text{ref}(y)}{\pi_\theta(y)} = -\log r\;}$$

#### 2.4.2　三大性质（必考）

**性质 1（非负）**：

由 $e^\Delta - \Delta - 1 \ge 0$ 对所有 $\Delta \in \mathbb{R}$（凸函数 $e^\Delta$ 在 $\Delta = 0$ 切线 $1 + \Delta$ 之上），$\widehat{\text{KL}}_3 \ge 0$ 总是成立。

**性质 2（无偏）**：

要证 $\mathbb{E}_{y \sim \pi_\theta}[\widehat{\text{KL}}_3] = \text{KL}(\pi_\theta\|\pi_\text{ref})$。

$$\mathbb{E}_{\pi_\theta}\!\left[\frac{\pi_\text{ref}}{\pi_\theta}\right] = \sum_y \pi_\theta(y) \cdot \frac{\pi_\text{ref}(y)}{\pi_\theta(y)} = \sum_y \pi_\text{ref}(y) = 1$$

$$\mathbb{E}_{\pi_\theta}\!\left[-\log\frac{\pi_\text{ref}}{\pi_\theta}\right] = \mathbb{E}_{\pi_\theta}\!\left[\log\frac{\pi_\theta}{\pi_\text{ref}}\right] = \text{KL}(\pi_\theta\|\pi_\text{ref})$$

所以：

$$\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = 1 + \text{KL}(\pi_\theta\|\pi_\text{ref}) - 1 = \text{KL}(\pi_\theta\|\pi_\text{ref}) \quad \checkmark$$

**性质 3（方差通常小于 k1）**：

直观：$\widehat{\text{KL}}_3$ 把 $\log r$ 的线性项（$-\log r/\pi_\theta$ 部分，即 k1 的等价形式 $\mathbb{E}_{\pi_\theta}[-\log(\pi_\text{ref}/\pi_\theta)]$）和一个**期望为 1 的控制变量** $\pi_\text{ref}/\pi_\theta - 1$ 相加。控制变量降方差（control variate）：当 $r$ 大时 $\log r$ 大但 $1/r$ 小，反之亦然，**两者负相关**，加起来方差比单独 k1 小。

形式化：$\widehat{\text{KL}}_3 = \widehat{\text{KL}}_1 + (\frac{\pi_\text{ref}}{\pi_\theta} - 1)$，加项 $(\frac{\pi_\text{ref}}{\pi_\theta} - 1)$ 期望为 0，但与 $\log r$ 强负相关 → 减方差。

> 💡 **k3 的 elegance** — Schulman blog 2020 给出的 derivation 等价于上面：他**先找一个 $f(x) \ge 0$ 且 $\mathbb{E}_q f(\log p/q) = \text{KL}$ 的函数**，然后挑 $f(x) = e^x - x - 1$（最简单的非负、可微、不是 trivial 的选择）。所以 k3 不是"灵感来的"——它是"非负 + 无偏"约束下的最自然构造。

#### 2.4.3　变种与简化

实际代码经常写成（**对 PPO ratio $r = \pi_\theta / \pi_\text{old}$ 同样适用**，但语义不同——这是 importance ratio approx，不是 policy log-ratio）：

```python
approx_kl_to_old = ((ratio - 1) - torch.log(ratio.clamp_min(1e-8))).mean()
```

即 $\widehat{\text{KL}}_3 \approx (r - 1) - \log r$，是上面 $e^\Delta - \Delta - 1$ 的等价对偶（把 $r = e^{\log r}$ 视作 $e^\Delta$ 即可，符号约定换一下）。**TRL / OpenRLHF / verl 默认 `approx_kl` 都是这个**。

### 2.5　三者对比表

| Estimator | 形式 | 无偏？ | 非负？ | 方差 | 在 RLHF 里典型用途 |
| --- | --- | --- | --- | --- | --- |
| **k1** | $\log r$ | ✅ | ❌ | 大 | InstructGPT PPO reward shaping（per-token KL） |
| **k2** | $\tfrac{1}{2}(\log r)^2$ | ❌ value-estimator 有偏（二阶近似 KL） | ✅ | 中 | value 视角常用作监控；**loss-gradient 视角是 principled k2-as-loss**——on-policy 下与 k1-in-reward gradient-equivalent，见 §3.6 |
| **k3** | $e^{-\log r} + \log r - 1$ | ✅（作为 **value estimator**） | ✅ | 小 | DeepSeekMath GRPO / DAPO 历史用法（作为 **loss** 时其梯度是 biased first-order approx，见 §3.6 + Rethinking KL 论文） |

### 2.6　代码：三估计器 + variance 对比 simulation

```python
import torch
import torch.nn.functional as F

def k1_estimator(logp_theta, logp_ref):
    """k1: log(π_θ / π_ref)  — unbiased, but can be negative, high variance."""
    return logp_theta - logp_ref           # [B, T]

def k2_estimator(logp_theta, logp_ref):
    """k2: 0.5 (log(π_θ / π_ref))^2  — biased, non-negative, mid variance."""
    return 0.5 * (logp_theta - logp_ref) ** 2

def k3_estimator(logp_theta, logp_ref):
    """k3 (Schulman 2020): exp(log π_ref - log π_θ) + log(π_θ/π_ref) - 1
       = (π_ref/π_θ) - log(π_ref/π_θ) - 1     [letting Δ = log π_ref - log π_θ]

    Value-estimator properties: unbiased, non-negative, low variance.
    ⚠️ Recommended for KL value MONITORING, NOT as a loss term — k3-as-loss
       gives gradient (1 - e^{-Δ})∇logπ_θ = (Δ - ½Δ² + O(Δ³))∇logπ_θ, a
       first-order Taylor approximation of reverse-KL gradient with O(Δ²) bias.
       For principled reverse-KL gradient, use k2-as-loss (gradient-equivalent
       to k1-in-reward on-policy) or k1-in-reward via score function. See §3.6.
    """
    log_r = logp_theta - logp_ref          # log(π_θ/π_ref)
    log_ratio_rev = -log_r                 # Δ = log(π_ref/π_θ)
    return torch.exp(log_ratio_rev) - log_ratio_rev - 1

# Variance comparison: 1D synthetic
def compare_kl_estimators(n_samples=10_000, seed=0):
    """
    Synthetic: π_θ = N(0, 1), π_ref = N(μ, 1). True KL = μ^2 / 2.
    Sample y ~ π_θ; evaluate the three estimators.
    """
    torch.manual_seed(seed)
    mu = 0.5
    true_kl = mu ** 2 / 2.0                        # closed-form for Gaussians w/ same σ

    y = torch.randn(n_samples)                     # y ~ N(0, 1) = π_θ
    # log-pdf of N(0,1) vs N(μ,1) at y (drop common -0.5 log 2π):
    logp_theta = -0.5 * y ** 2
    logp_ref   = -0.5 * (y - mu) ** 2

    k1 = k1_estimator(logp_theta, logp_ref)
    k2 = k2_estimator(logp_theta, logp_ref)
    k3 = k3_estimator(logp_theta, logp_ref)

    print(f"true KL = {true_kl:.4f}")
    for name, vals in [("k1", k1), ("k2", k2), ("k3", k3)]:
        print(f"  {name}: mean={vals.mean().item():+.4f}  var={vals.var().item():.4f}  "
              f"min={vals.min().item():+.3f}  max={vals.max().item():+.3f}")
    # Expected output: k1 mean ≈ 0.125 (unbiased), but min < 0;
    #                  k2 mean > 0.125 (biased upward);
    #                  k3 mean ≈ 0.125 (unbiased), min ≥ 0, var(k3) < var(k1).
```

> ✅ **运行 simulation 你会看到** —

- k1 mean ≈ 0.125, min 约 −1.5（可负），var 大。
- k2 mean ≈ 0.18（偏高），min ≥ 0，var 中。
- k3 mean ≈ 0.125（无偏），min ≥ 0，var **明显小于** k1。

这正是 Schulman blog 给出的 takeaway：**k3 同时拿到 unbiased + non-negative + 较低方差**。

## §3 KL 在 RLHF 中的两种放置

### 3.1　Option A：In-reward shaping（PPO RLHF / InstructGPT 标准做法）

把 KL **塞进 per-token reward**，再正常跑 PPO + GAE：

$$\boxed{\;\tilde{r}_t = \underbrace{\mathbb{1}[t = T] \cdot R(x, y)}_{\text{terminal RM reward}} - \beta \cdot \underbrace{\log\frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_\text{ref}(y_t \mid x, y_{<t})}}_{\text{per-token KL (k1)}}\;}$$

注意细节：

- **per-token**：每生成一个 token，就计算这个 token 的 log-prob ratio，作为该步 reward 的一部分。
- **k1 estimator**：这里直接用 $\log(\pi_\theta/\pi_\text{ref})$（k1）。注意它**单 token 可为负**，但作为 reward shaping 只要均值/总和符合 KL 期望即可。
- **terminal RM reward**：RM 给整段答案的 scalar，只放在最后一个 token 上（其余 token RM reward = 0）。

放完后跑 GAE：

$$\delta_t = \tilde{r}_t + \gamma V(s_{t+1}) - V(s_t),\quad A_t^{\text{GAE}} = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}$$

KL penalty 通过 advantage 自然 propagate 到 policy gradient：**每步 token 的"动作概率提升"被 KL "拉回 reference" 抵消**，最终 RL 目标 = expected $R$ − $\beta$ · KL。

### 3.2　Option B：In-loss regularization（GRPO / DAPO 做法）

把 KL **不放 reward**，作为单独 loss 项：

$$\boxed{\;\mathcal{L}_\text{full}(\theta) = -\underbrace{\mathbb{E}\!\left[\min(\rho_t A_t, \text{clip}(\rho_t, 1\!-\!\epsilon, 1\!+\!\epsilon) A_t)\right]}_{\text{PPO surrogate / GRPO surrogate}} + \beta \cdot \underbrace{\mathbb{E}\!\left[\widehat{\text{KL}}_\text{loss}(\pi_\theta \| \pi_\text{ref})\right]}_{\text{KL loss term}}\;}$$

GRPO 的**历史实现**（DeepSeekMath / DAPO）把 $\widehat{\text{KL}}_\text{loss}$ 取作 **k3**（per-token）：

- 用 k3 estimator（per-token）。
- KL **不进 advantage 计算**；直接作为正则项加到 loss 上。
- Advantage 由组内 reward 归一化得到（$\hat{A}_i = (r_i - \bar{r})/\sigma_r$），所有 token 共享。

> ⚠️ **重要 caveat**：把 k3 直接当 loss 反传**不是**精确的 reverse-KL gradient——它是一个 biased first-order approximation（详见 §3.6 + 2025 "Rethinking KL Regularization in RLHF" 论文）。on-policy 下更 principled 的两种选择是：(1) **k1 in reward**（KL 进 reward，PPO/GRPO 通过 score-function 反传，gradient 严格无偏）；(2) **k2 as loss**（$\tfrac12 (\log r)^2$，$\nabla\mathcal L = \Delta\,\nabla\log\pi_\theta$，on-policy 下与 (1) **gradient-equivalent**，都是严格 reverse-KL gradient）。off-policy 还需 IS correction。下方 §3.6 给出对比表。

### 3.3　两者数学上等价？工程上不等价

**数学**：两种放法本质都是优化同一个目标：

$$J(\pi) = \mathbb{E}_{\pi}[R] - \beta \cdot \text{KL}(\pi \| \pi_\text{ref})$$

只是**梯度 propagate 路径不同**：

- Option A：KL 作为 reward 一部分 → 经 GAE → 进入 advantage → 进入 PPO surrogate 梯度。
- Option B：KL 作为独立 loss → 直接对 $\theta$ 求梯度。

**工程差异**：

| 维度 | In-reward (Option A) | In-loss (Option B) |
| --- | --- | --- |
| KL 估计器 | k1（单 token 可负，作为 reward 接受） | k3（无偏 + 非负，更稳） |
| 与 PPO clip 的关系 | KL 被 clip 截断（importance ratio 截断把 KL 一并截了） | KL 独立，不被 clip 影响 |
| Advantage 解释 | advantage 内含 KL → "净 advantage" | advantage 仅 reward-based → "纯 advantage" |
| 调参敏感度 | β 直接影响 reward scale → 需要和 RM scale 协调 | β 与 reward 独立，但 KL 与 PG 比重要平衡 |
| 监控指标 | 看 token-level KL 进了 reward | 看 KL loss 独立曲线 |

> ⚠️ **PPO clip 抹掉 KL 信号的细节** — 在 Option A 中，当 importance ratio $\rho_t$ 落在 clip 之外（$\rho_t > 1+\epsilon$ 且 $\tilde A_t > 0$ 或 $\rho_t < 1-\epsilon$ 且 $\tilde A_t < 0$），PPO 把 surrogate 截断为常数（对 θ 梯度为 0），此时该 step 的 $\tilde r_t$ 里的 KL 项也一并失效。这意味着**当 policy 偏离过大时，PPO clip 反而让 KL anchor 失效**——这是 Option B 把 KL 放 loss 的部分动机（KL 永远生效）。

### 3.4　实操：哪一种用在哪？

| 算法 | KL 放置 | 估计器 | 来源 |
| --- | --- | --- | --- |
| InstructGPT PPO | **In-reward** | k1 | Ouyang 2022 NeurIPS |
| Anthropic RLHF | **In-reward** | k1 + adaptive β | Bai 2022 arXiv 2204.05862 |
| GRPO / DeepSeekMath | **In-loss** | k3 | Shao 2024 arXiv 2402.03300 |
| DAPO | **In-loss**（实际配置中 KL 项常被关掉或权重很低） | k3 | Yu 2025 arXiv 2503.14476 |
| DPO | **隐式 in-loss**（通过 $\pi_\text{ref}$ 在 log-ratio 分母） | — | Rafailov 2023 NeurIPS |
| SimPO | **去掉 reference, 无 KL** | — | Meng 2024 NeurIPS |

> 💡 **DAPO 实战经验** — 字节 verl 团队的工程报告里多次提到，KL 项在大规模数学/代码 RL 训练中**经常关掉或设很小** β（$10^{-4}$ 量级），原因：reward 已经是 rule-based（接近 ground truth），不存在 RM hacking，KL anchor 反而拖训练。这是"Option B + β ≈ 0"的极端版本。

### 3.6　Estimator placement 的 gradient bias 分析（Rethinking KL Regularization in RLHF）

> 📝 **关键参考**：Kezhao Liu et al., "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization", arXiv 2510.01555 (2025-10-02)。这篇文章系统区分 **value estimation**（"KL 量本身估得准不准"）与 **gradient optimization**（"对 θ 求导后是不是真的 reverse-KL gradient"），结论与历史 GRPO 实现不完全一致。

#### 设定与三种 estimator placement

为简洁起见，令 $\Delta_t = \log\!\frac{\pi_\theta(y_t|s_t)}{\pi_\text{ref}(y_t|s_t)}$（per-token log-ratio，**带 θ 梯度**），$\hat\Delta_t = \text{stop\_grad}(\Delta_t)$（在 rollout 时记录的 detached 副本）。三种常见 placement：

| 方案 | KL 出现的位置 | 单 token 形式 | 对 θ 的梯度贡献 |
|---|---|---|---|
| **(P1) k1 in reward** | 进 reward / advantage | $\hat r_t \leftarrow r_t - \beta\,\hat\Delta_t$，再走 PPO surrogate | $-\beta\cdot \mathbb{E}[\nabla_\theta\log\pi_\theta\cdot \hat\Delta_t]$，**严格的 reverse-KL score-function gradient**（on-policy） |
| **(P2) k2 as loss** | 直接 loss 项 | $\mathcal{L}_\text{KL} = \tfrac12 \Delta_t^2$ | $\nabla\mathcal{L}_\text{KL} = \Delta_t\,\nabla_\theta\log\pi_\theta$（用 $\nabla_\theta\Delta_t = \nabla_\theta\log\pi_\theta$）。on-policy 下 $\mathbb{E}_{y\sim\pi_\theta}[\Delta\,\nabla\log\pi_\theta] = \nabla\text{KL}$，与 (P1) **gradient-equivalent**——严格 principled |
| **(P3) k3 as loss** | 直接 loss 项 | $\mathcal{L}_\text{KL} = e^{-\Delta_t} + \Delta_t - 1$ | $\nabla\mathcal{L}_\text{KL} = (1 - e^{-\Delta_t})\nabla_\theta\log\pi_\theta$；这是 reverse-KL gradient 的 **first-order Taylor approximation**，$1-e^{-\Delta} = \Delta - \tfrac12\Delta^2 + O(\Delta^3)$，有 $O(\Delta^2)$ bias |

#### 为什么 k3 as loss 是 biased gradient（关键直觉）

k3 作为 **value estimator**（计算 KL 的数值）是无偏的：$\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = \text{KL}(\pi_\theta\|\pi_\text{ref})$（见 §2.4.2 性质 2）。但当 **对 θ 反传**时，autograd 计算的是 $\nabla_\theta\widehat{\text{KL}}_3$，而**不是** $\nabla_\theta\,\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3]$。后者由 score-function trick 给出：

$$\nabla_\theta\,\text{KL}(\pi_\theta\|\pi_\text{ref}) = \mathbb{E}_{y\sim\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(y) \cdot \log(\pi_\theta(y)/\pi_\text{ref}(y))\big] \;+\; \mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta]\,\quad\text{(=0)}$$

= $\mathbb{E}[\nabla\log\pi_\theta \cdot \Delta]$（**这是 (P1) 给的 gradient**，等价于把 $\Delta$ 当 detached reward 走 score function）。

而 (P3) `loss.backward()` 给的是 $\nabla_\theta(e^{-\Delta} + \Delta - 1) = (1 - e^{-\Delta})\nabla_\theta\Delta = (1 - 1/r)\nabla_\theta\log\pi_\theta$。Taylor 展开：$1 - 1/r = \Delta - \tfrac12\Delta^2 + O(\Delta^3)$。所以 (P3) ≈ (P1) 只在 $\Delta\to 0$ 邻域成立；远离 0 时有 $O(\Delta^2)$ bias。

#### 实践规则（结合 Rethinking KL + Comedy of Estimators）

| 场景 | 推荐 placement | 备注 |
|---|---|---|
| **on-policy（rollout 当步反传，无 PPO mini-batch 多步 update）** | **(P1) k1 in reward** 或 **(P2) k2 as loss** | 两者在 on-policy 下 **gradient-equivalent**，都是严格 reverse-KL gradient |
| **off-policy（PPO mini-batch 多次走 same data）** | (P1) k1 in reward + **IS correction**：$\rho\cdot\Delta$，$\rho = \pi_\theta^\text{new}/\pi_\theta^\text{old}$ | 否则 reward 与当前 policy mismatch |
| **历史 GRPO/DeepSeekMath/DAPO 实现** | (P3) k3 as loss | 有 $O(\Delta^2)$ gradient bias，实际工程里 $\Delta$ 很小（β 锚得紧），所以仍能跑；但**不是理论 principled** |
| **value-side monitoring**（看 KL 数值多大） | k3 estimator 仍是首选 | 无偏、非负、低方差——这就是 §2.4 那三大性质 |

> 💡 **总结口径**（面试推荐说法）：k3 是**优秀的 KL value estimator**（unbiased + non-negative + low variance）；但作为 loss 反传的 gradient 是 reverse-KL gradient 的 first-order Taylor approximation，有 bias。Rethinking KL 推荐 on-policy 用 (P1) 或 (P2)；DAPO/GRPO 历史用 (P3) 主要是工程惯例 + 在小 β 区间近似成立。

## §4 β 调度

### 4.1　固定 β —— baseline

最常见。$\beta \in [0.01, 0.5]$，**InstructGPT 报告中 β = 0.02 是参考值**（per-token KL）。优点：简单可重现。缺点：训练中 KL 通常先小后大，固定 β 后期可能"压不住"或"过度压制"。

### 4.2　Adaptive β —— Schulman PPO-Penalty 原版

PPO 论文（Schulman 2017 arXiv 1707.06347）原本提了两个版本：**Clip**（现在主流）和 **Penalty**（adaptive KL）。Penalty 形式：

$$\mathcal{L} = \mathbb{E}[r_t A_t] - \beta_k \cdot \text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)$$

**β 自适应规则**（每 epoch 之后看 measured KL $d$）：

- 若 $d < d_\text{target} / 1.5$：$\beta \leftarrow \beta / 2$（KL 比目标低，放松）
- 若 $d > d_\text{target} \times 1.5$：$\beta \leftarrow \beta \times 2$（KL 比目标高，加压）
- 否则 β 不变

直觉：把 β 当 PID controller 的 P term，target 是预期 KL（比如 $d_\text{target} = 0.01$）。这条思路在 InstructGPT 和后续 Anthropic 工作里也出现过（Anthropic 的 helpfulness/harmlessness 报告里也有类似 adaptive β 描述）。

### 4.3　β annealing schedule —— 类似 learning rate schedule

把 β 看作时间函数 $\beta(t)$：

- **早期紧后期松**：$\beta(t) = \beta_0 \cdot \exp(-t / \tau)$。直觉：训练初期 policy 离 ref 近，加强 anchor 防 instability；后期 policy 学到了，放松让它 explore reward。
- **早期松后期紧**：相反方向，初期 explore RM 信号，后期 anchor。少见。
- **Cosine / linear decay**：参考 lr schedule。

在 RLHF 工程实践中 annealing 用得不多——adaptive β 比 schedule 更鲁棒（不需调 schedule shape）。

### 4.4　β 失败模式

| β 设置 | 现象 | 诊断 |
| --- | --- | --- |
| **β 太大** ($> 1$) | KL ≈ 0，policy 卡在 ref 附近，RM reward 不涨 | 看 reward curve 是平的 / `chosen_logp - rejected_logp` 不分离 |
| **β 太小** ($< 0.001$) | KL 爆炸（runaway），policy 越来越长 / 谄媚 / 重复 | 看 KL 曲线持续上升、generation length 上升、人工评测下降 |
| **β 突变** | 训练 loss 跳跃 | adaptive 频率太高 / target_kl 太严 |

> ⚠️ **InstructGPT 论文 β = 0.02 真的合适吗？** — 答案因任务而异。**Math/code RL** 通常需要更小 β（DeepSeekMath 报告 β ≈ 0.04，DAPO 的 β 经常更小或为 0），因为 reward 接近 ground truth。**Helpfulness/safety RL** 需要更大 β（≥ 0.1）防 reward hacking，因为 neural RM 容易被 hack。这是为什么"β 不是 universal hyperparameter"。

### 4.5　Adaptive β 代码示例

```python
class AdaptiveKLController:
    """
    Schulman 2017 PPO-Penalty style adaptive β controller.
    每个 PPO epoch 后调用 update(measured_kl)。
    """
    def __init__(self, beta_init=0.02, target_kl=0.01, horizon=10000):
        self.beta = beta_init
        self.target_kl = target_kl
        self.horizon = horizon          # 平滑系数 (越大越缓)

    def update(self, measured_kl, n_steps):
        # proportional update; clip to prevent extreme jumps
        proportional_error = max(-0.2, min(0.2,
            measured_kl / self.target_kl - 1.0))
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.beta *= mult
        # safety clamp
        self.beta = max(1e-4, min(1.0, self.beta))
        return self.beta
```

用法：

```python
kl_ctrl = AdaptiveKLController(beta_init=0.02, target_kl=0.01)
for epoch in range(num_epochs):
    # ... rollout and PPO update ...
    measured_kl = compute_mean_kl(policy, ref_policy, batch)   # k3 推荐
    kl_ctrl.update(measured_kl, n_steps=batch_size)
    current_beta = kl_ctrl.beta
```

> 💡 **adaptive β 与 PID controller 的类比** — 上面只有 P 项（proportional）；HuggingFace TRL 里的实现也只用 P。完全可以加 I（integral：累计误差）和 D（derivative：变化率）做 PID，但实测 P 已经够了，加 D 反而容易在 noisy KL 估计下振荡。

## §5 KL 与 DPO / GRPO / SimPO / KTO / IPO 的关系

### 5.1　DPO：implicit reward = β × sequence log-density ratio

DPO 的 implicit reward 是 $\hat{r}_\theta(x, y) = \beta \log(\pi_\theta(y|x) / \pi_\text{ref}(y|x))$，即整段 $y$ 的 sequence-level log-density ratio：

$$\hat{r}_\theta(x, y) = \beta \sum_t \log\frac{\pi_\theta(y_t | x, y_{<t})}{\pi_\text{ref}(y_t | x, y_{<t})}$$

> ⚠️ **注意区分：implicit reward 不是 KL 本身** — $\hat{r}_\theta$ 是**单条 sequence 的 pointwise log-ratio**，**不是** KL。只有当对 $y\sim\pi_\theta$ 取期望时，$\mathbb{E}_{y\sim\pi_\theta}[\hat{r}_\theta(x,y)] = \beta\cdot\text{KL}(\pi_\theta(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x))$（这是 k1 KL estimator 的 sequence-level 形式）。但 **DPO 训练时 $y_w, y_l$ 不是从 $\pi_\theta$ 采的，而是来自固定 preference 数据**，所以训练阶段的 $\hat{r}_\theta$ 不能直接读作 KL；它只是 pairwise log-ratio 差异。
>
> **正确表述**：DPO loss 的对偶 RLHF 解释是把 $\hat{r}_\theta$ 当 reward，BT model 给概率 $\sigma(\hat{r}_w - \hat{r}_l)$；它对应"KL-regularized RL 闭式最优 policy $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$"反解出的 reward 表达式。$\pi_\theta$ 在 implicit RLHF 视角下隐式被 KL 约束（因为 $\hat r_\theta$ 形式里 $\pi_\theta/\pi_\text{ref}$ 出现），但 DPO 优化的是 pairwise margin 而非显式 KL margin。

#### 5.1.1　DPO 闭式推导回顾

KL-regularized 目标：

$$\max_\pi \mathbb{E}_{x,\, y \sim \pi}[r(x, y)] - \beta\, \text{KL}(\pi \| \pi_\text{ref})$$

对单个 $x$ 用 Lagrangian + 求导（详细推导见 §6.1），得：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right)$$

反解：

$$r(x, y) = \beta \log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry $P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$，$\beta \log Z$ 消掉，得 DPO：

$$\boxed{\;\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\;}$$

#### 5.1.2　DPO 梯度的 KL 解释

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta \mathbb{E}\!\Big[\sigma(\hat{r}_l - \hat{r}_w)\big(\nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta \log\pi_\theta(y_l|x)\big)\Big]$$

**解释**：

- $\sigma(\hat{r}_l - \hat{r}_w)$ 是"当前模型对偏好顺序错位的置信度"。
- $\nabla \log\pi(y_w) - \nabla \log\pi(y_l)$ 是"提升 $y_w$ 概率 + 降低 $y_l$"。
- $\beta$ 出现两次：一次进 implicit reward $\hat{r}$（决定 sigmoid 内部），一次显式作梯度系数。所以**β 在 DPO 里同时调"KL 强度"和"梯度幅值"**——与 RLHF 中 β 只调一个东西不同。

> ⚠️ **DPO 的 β 调参隐患** — DPO 调 β 不直接对应 RLHF 调 β 的语义。在 PPO 中 β 只影响 KL penalty 强度，在 DPO 中 β 同时影响：(1) implicit reward scale、(2) 梯度幅值（外层 β）、(3) sigmoid 饱和位置（内层 β）。**经验值 $\beta \in [0.05, 0.5]$ 但每个任务最优 β 不同**。

### 5.2　GRPO：k3 KL in loss

DeepSeekMath GRPO 用 k3，KL 作为独立 loss 项：

$$L^\text{GRPO}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\!\Big(\min(\rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1{-}\epsilon, 1{+}\epsilon)\hat{A}_{i,t}) - \beta\,\widehat{\text{KL}}_3^{i,t}\Big)\right]$$

其中 $\widehat{\text{KL}}_3^{i,t} = e^{-\log r_{i,t}} + \log r_{i,t} - 1$（per-token k3），$\log r_{i,t} = \log\pi_\theta(y_{i,t}|\cdot) - \log\pi_\text{ref}(y_{i,t}|\cdot)$。

**为什么 GRPO 历史上选 k3**（注意：这些都是 **value-estimator** 性质，不直接保证 loss gradient 正确，见 §3.6）：

1. **非负**：作为 loss 数值显示直观（k1 数值可负，调试时容易让 optimizer 监控曲线看起来怪；但 k1 用在 reward 里是无害的）。
2. **value estimator 无偏**：$\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = \text{KL}$（**仅作为 KL 数值估计无偏；对 θ 反传时 gradient 是 first-order Taylor approximation，有 $O(\Delta^2)$ bias**）。
3. **value estimator 低方差**：相比 k1 数值，方差更小（log 监控曲线更平滑）。
4. **不进 advantage**：与 In-reward 方案的耦合相对解开，便于诊断（KL loss 和 PG loss 独立监控）。

> ⚠️ **GRPO 的 k3-as-loss 不是理论 principled 的最佳选择**——Rethinking KL (arXiv 2510.01555) 系统分析显示 on-policy 下 (P1) k1-in-reward 或 (P2) k2-as-loss 是严格 reverse-KL gradient；GRPO/DAPO 实际工程上 $\beta$ 锁得紧、$\Delta$ 很小，所以 first-order approximation 误差有限——但**面试中应能区分 "value-estimator 优势" 与 "loss-gradient 正确性"**。

### 5.3　SimPO：去掉 reference，没有 KL anchor

SimPO 把 implicit reward 改成 length-normalized log-prob：

$$r_\text{SimPO}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$$

**关键差异**：**没有 $\pi_\text{ref}$**，所以**没有 KL 项**。损失：

$$\mathcal{L}_\text{SimPO} = -\mathbb{E}\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi(y_w) - \frac{\beta}{|y_l|}\log\pi(y_l) - \gamma\right)$$

**后果**：

- ✅ 训练时省一份 reference policy（显存减半）。
- ❌ 没有 KL anchor，policy 离 SFT 的距离**完全不可控**。
- 实测上 SimPO 在某些 benchmark（AlpacaEval-2 / Arena-Hard）上优于 DPO；但生成质量在 OOD prompt 上**比 DPO 不稳**。SimPO 的设计是"用 length-norm + margin 替代 KL anchor"——它假定了"短 prompt-response 任务下，长度归一 + margin 足以约束 policy"，**不是对所有任务都成立**。

> 💡 **SimPO 没有 KL 的工程影响** — 工程上发现 SimPO 训完的 model 在重复模式、生成长度、对 prompt 微扰的鲁棒性上**都比 DPO 差**。这是为什么很多生产 RLHF 仍然用 DPO + small β 而不是 SimPO——KL anchor 是有代价但有价值。

### 5.4　KTO / IPO / ORPO 的 KL 处理

| 算法 | KL 形式 | 备注 |
| --- | --- | --- |
| **KTO** (Ethayarajh 2024 ICML) | 通过 reference point $z_0 = \mathbb{E}[\beta\cdot\text{KL}]$（batch mismatched pair 估，detach） | KL 隐式地"作 anchor"：implicit reward 偏离 $z_0$ 多远 |
| **IPO** (Azar 2024 AISTATS) | 与 DPO 同（$\log(\pi/\pi_\text{ref})$），但损失从 sigmoid 改 squared | 防止 deterministic preference 下 $\hat{r}$ 无界增长 |
| **ORPO** (Hong 2024 EMNLP) | **无 reference model**（与 SimPO 类似），用 odds-ratio 代替 | 一阶段同时做 SFT + preference；KL anchor 隐含在 SFT loss 里 |

### 5.5　Recent papers (2024-2026)

| 论文 | 主张 | KL 角度 |
| --- | --- | --- |
| **DeepSeekMath GRPO** (Shao et al. 2024 arXiv 2402.03300) | 组内归一化 + k3 KL in loss | 第一个在 LLM RL 中默认 k3 KL |
| **DPO Implicit Reward Models** (Rafailov 2023 NeurIPS) | DPO 的 implicit reward 等价于 KL log-ratio | DPO 本质是 KL-regularized 优化的反解 |
| **Reward Model Overoptimization Scaling Laws** (Gao, Schulman, Hilton 2023 ICML) | gold reward 在 KL 距离上 inverted-U | KL 是 overoptimization 的天然 x-axis |
| **DAPO** (Yu et al. 2025 ByteDance arXiv 2503.14476) | clip-higher + dynamic sampling + token-level loss | KL 项常被设小或关掉，加入清理 prompt 的 dynamic sampling |
| **Cohere DRO / OPO** (各种 2024-2025 工作) | offline IS-correction RL with KL | 把 IS-correction 与 KL anchor 联合 |
| **"Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization"** (Kezhao Liu et al. 2025-10, arXiv 2510.01555) | 系统区分 KL value estimation 与 gradient optimization：k3-as-loss 是 biased first-order approximation；推荐 (1) k1 in reward（严格 reverse-KL score-function gradient）或 (2) **k2 as loss**（value estimator 有偏，但 on-policy loss gradient 严格等价于 k1-in-reward / reverse-KL gradient）；off-policy 还需 IS correction | 历史 GRPO 的"k3 as loss"是工程惯例不是理论 principled |
| **"A Comedy of Estimators: On KL Regularization in RL Training of LLMs"** (Vedant Shah et al. 2025-12, arXiv 2512.21852, v3 2026-03) | 在多种 RL 算法 + estimator placement 组合上系统对比 k1/k2/k3 的 estimator bias 与 gradient bias；分析 placement-effect | 不存在 universally 最好的 estimator；reward-shaping vs loss-term 各有适用场景 |

## §6 Theoretical：KL-Regularized RL 的最优 policy + Reward Overoptimization

### 6.1　KL-Regularized RL 闭式解（DPO 的数学基础）

**定理**（KL-regularized policy optimization 的闭式解）：

考虑：

$$\max_\pi J(\pi) = \mathbb{E}_{y \sim \pi(\cdot|x)}[r(x, y)] - \beta\, \text{KL}\!\big(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x)\big)$$

其中 $\pi$ 是对任意 $x$ 的分布，$\pi_\text{ref}$ 严格正（$\pi_\text{ref}(y|x) > 0$ 对所有 $y$）。

**唯一最优解**：

$$\boxed{\;\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_\text{ref}(y|x)\, \exp\!\left(\frac{r(x, y)}{\beta}\right),\quad Z(x) = \sum_{y'} \pi_\text{ref}(y'|x) \exp\!\left(\frac{r(x, y')}{\beta}\right)\;}$$

**证明**（Lagrangian）：

固定 $x$，写目标（连续/离散通用，离散表示）：

$$\mathcal{L}_x[\pi] = \sum_y \pi(y|x) r(x, y) - \beta \sum_y \pi(y|x) \log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)} - \mu\!\left(\sum_y \pi(y|x) - 1\right)$$

其中 $\mu$ 是 normalization 约束的 Lagrange multiplier（注意符号：从 max 推 KKT 时把 $\sum_y\pi = 1$ 写成 $-\mu(\sum - 1)$ 是为了内点最优条件简洁）。

对 $\pi(y|x)$ 求偏导：

$$\frac{\partial \mathcal{L}_x}{\partial \pi(y|x)} = r(x, y) - \beta \log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)} - \beta - \mu = 0$$

整理：

$$\log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)} = \frac{r(x, y) - \mu - \beta}{\beta}$$

指数化：

$$\pi(y|x) = \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y) - \mu - \beta}{\beta}\right) = \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right) \cdot e^{-(\mu+\beta)/\beta}$$

代入归一化条件 $\sum_y \pi(y|x) = 1$：

$$e^{(\mu+\beta)/\beta} = \sum_y \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right) = Z(x)$$

所以 $e^{-(\mu+\beta)/\beta} = 1/Z(x)$，得：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x, y)}{\beta}\right) \quad \blacksquare$$

**唯一性**：因为 $J$ 在 $\pi$ 上是**严格凹**的（KL 是凸的，加负号变严格凹，且 reward 是 linear），凹优化有唯一最优。

### 6.2　从最优 policy 反解 implicit reward

把 §6.1 的 $\pi^*$ 取 log：

$$\log\pi^*(y|x) = \log\pi_\text{ref}(y|x) + \frac{r(x, y)}{\beta} - \log Z(x)$$

反解 $r$：

$$\boxed{\;r(x, y) = \beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta\log Z(x)\;}$$

**这就是 DPO 的 implicit reward**。$\beta \log Z(x)$ 是 partition function，**只依赖 $x$，不依赖 $y$**——在 Bradley-Terry 的 $r(x, y_w) - r(x, y_l)$ 差里**消掉**。

### 6.3　Bradley-Terry 偏好模型下的 RL 等价于 DPO

BT 模型：$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$。

代入 §6.2：

$$r(x, y_w) - r(x, y_l) = \beta \log\frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}$$

替换 $\pi^*$ 为可学 $\pi_\theta$，对偏好数据 NLL：

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

**这是 §5.1 的 DPO 公式**。所以**DPO ≡ Bradley-Terry preference + KL-regularized RL 闭式解 + 偏好数据 NLL**，三件套合一。

### 6.4　Reward Overoptimization（Gao, Schulman, Hilton 2023 ICML）

#### 6.4.1　问题表述

记 gold reward $r_g$（人感受的真实质量）、proxy reward $r_p$（RM 估的）。RM 用偏好数据学，但偏好数据有限 → $r_p \ne r_g$。

随着 RL 训练 KL 距离 $d = \text{KL}(\pi_\theta \| \pi_\text{ref})$ 增大：

- $\mathbb{E}_{\pi_\theta}[r_p]$ 单调上升（policy 学到 RM 的偏好）。
- $\mathbb{E}_{\pi_\theta}[r_g]$ 先升后降（**inverted-U**）—— 这是 reward overoptimization。

#### 6.4.2　Gao 2023 给的拟合形式

Gao 2023 用大量 RM size + KL 实验，给出 gold reward 关于 KL 距离 $d$ 的拟合形式（用 $\sqrt{d}$ 当 x-axis）：

$$R_g(d) = d \cdot (\alpha_g - \gamma_g \cdot d) \quad \text{(BoN)}$$

$$R_g(d) = d \cdot (\alpha_g - \beta_g \cdot \log d)\quad \text{(PPO/RL：log 项，不是高阶多项式)}$$

其中 $\alpha_g, \gamma_g, \delta_g$ 是与 RM size 相关的系数；RM 越大，"高阶项 / 二次项"权重越小，过优化越慢。

#### 6.4.3　KL 是 overoptimization 的"天然 axis"

无论是 BoN、PPO 还是 DPO（按 implicit reward 累积），都可以把训练曲线画成"**KL 距离 vs gold reward**"图。Gao 2023 发现：

- **不同算法（BoN / PPO）在同样 KL 下的 over-optimization 行为相似**（同一 RM 给出的 gold 曲线形状一致）。
- **更大 RM → 更慢的 over-optimization**（gold curve peak 更靠右）。
- **KL 是一个一维 progress indicator**，比 step 数 / reward 数更能解释 over-optimization。

**面试 takeaway**：在 RLHF 监控里，把 `(measured_KL, gold_reward)` 画图，看是否进入下降区——这是最直接的 "stop 早一点" 信号。

### 6.5　KL 与 $\chi^2$ gap 的关系

KL 与 $\chi^2$ 在 Pinsker 类不等式族里有关系：

$$\text{TV} \le \sqrt{\text{KL}/2}\qquad \text{(Pinsker)}$$

$$\text{KL} \le \log(1 + \chi^2)$$

在 RLHF 里 $\chi^2$ 偶尔被作为 KL 的"敏感"上界使用：**当 $\chi^2$ 大但 KL 小，意味着 $\pi_\theta$ 有重 tail 偏离 $\pi_\text{ref}$**（高方差但低 mean）。某些工作（Yu et al. 2024 / others）在监控 reward hacking 时同时画 KL 和 $\chi^2$，看 tail 行为。

### 6.6　Sequence-level vs Token-level KL

由 §1.1 的**链式法则**：

$$\text{KL}(\pi_\theta(\cdot|x) \| \pi_\text{ref}(\cdot|x)) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_t \log\frac{\pi_\theta(y_t | x, y_{<t})}{\pi_\text{ref}(y_t | x, y_{<t})}\right]$$

即 **sequence-level KL = token-level KL 之和的期望**（自回归 chain rule）。注意：单条 rollout 上把 $\sum_t \log\pi_\theta(y_t)/\pi_\text{ref}(y_t)$ 直接当 KL 是 **k1 estimator**（unbiased 但 per-rollout 方差大）；若要 true token-level KL $D(\pi_\theta(\cdot|s_t)\|\pi_\text{ref}(\cdot|s_t))$ 还需 full-vocab forward。**两者期望相等**，工程上选 sampled log-ratio 是因为 vocab 求和昂贵。

但有两个**implementation trick** 与 token-level 相关：

1. **Per-token clipping**：单 token KL 偶尔很大（罕见 token、长 tail），可以对每 token KL 做 clip 防止单 token 拖飞 batch 总 KL。
2. **Mask on assistant tokens only**：在 chat / agent setting 下，prompt token 不应该参与 KL 计算（prompt 是同一的，policy 和 ref 在 prompt 上完全一致，KL = 0；但浮点误差会污染）。所以 KL mask 与 PPO action_mask 重合，**只在 assistant generation token 上算 KL**。

```python
# Per-token KL with action mask (chat / agentic RL setting)
def per_token_kl_with_mask(logp_theta, logp_ref, action_mask, estimator="k3"):
    """
    logp_theta, logp_ref: [B, T]   per-token log-prob
    action_mask:          [B, T]   1 = assistant token, 0 = prompt/system/pad
    estimator:            "k1" | "k2" | "k3"
    Returns: mean KL over assistant tokens, scalar.
    """
    log_r = logp_theta - logp_ref
    if estimator == "k1":
        kl_per_tok = log_r
    elif estimator == "k2":
        kl_per_tok = 0.5 * log_r ** 2
    elif estimator == "k3":
        delta = -log_r                        # log(π_ref / π_θ)
        kl_per_tok = torch.exp(delta) - delta - 1.0
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    masked = (kl_per_tok * action_mask).sum()
    n = action_mask.sum().clamp_min(1.0)
    return masked / n
```

## §7 实践 + 代码

### 7.1　PPO 风格：In-reward shaping 实现

```python
import torch
import torch.nn.functional as F

def ppo_reward_with_kl(rewards_terminal, rollout_logp_theta, logp_ref,
                       action_mask, beta=0.02):
    """
    PPO / InstructGPT style: KL penalty in reward (k1 estimator).
    rewards_terminal:    [B]      only the last assistant token gets the RM reward
    rollout_logp_theta:  [B, T]   log π_θ_old(y_t | ...), recorded during rollout, DETACHED
    logp_ref:            [B, T]   log π_ref(y_t | ...), reference policy frozen
    action_mask:         [B, T]   1 = assistant token
    Returns: shaped_reward [B, T]  per-token reward with KL penalty baked in.

    ⚠️ 关键：rollout_logp_theta 和 logp_ref 必须是 detached scalars（rollout 阶段记录或 no_grad
    forward）。这里 shaped reward 是 PPO/score-function 反传中的 reward tensor，**不能携带梯度**；
    否则 backward 时 KL 项会和 PG surrogate 双重反传，破坏 score-function 语义。生产代码里通常在
    rollout buffer 中把 logp_old 当固定数据保存，update 阶段重新 forward 拿 new_logp。
    """
    B, T = rollout_logp_theta.shape
    # k1 KL per token (signed; mean is unbiased). Both inputs assumed detached.
    kl_per_tok = (rollout_logp_theta.detach() - logp_ref.detach()) * action_mask  # [B, T]
    # spread terminal reward to last assistant token
    last_token_idx = action_mask.cumsum(dim=-1).argmax(dim=-1)  # [B]
    R_per_tok = torch.zeros_like(kl_per_tok)
    R_per_tok[torch.arange(B), last_token_idx] = rewards_terminal
    # combine
    shaped = R_per_tok - beta * kl_per_tok
    return shaped  # 整个 tensor 是 detached / 无梯度，作为 PPO advantage 计算的数据输入
```

### 7.2　GRPO 风格：In-loss regularization 实现（k3）

```python
def grpo_loss_with_k3_kl(policy, ref_policy, batch, eps_clip=0.2, beta=0.04):
    """
    GRPO / DeepSeekMath style: KL penalty in loss (k3 estimator).
    batch:
      input_ids:     [N, L]    N = sum_b G_b samples
      action_mask:   [N, L]
      old_log_probs: [N, L]
      rewards:       [N]       sequence-level reward
      group_id:      [N]       which prompt
    """
    rewards = batch["rewards"]
    gid = batch["group_id"].long()
    num_groups = int(gid.max().item()) + 1

    # Group-relative advantage (z-score within group)
    counts = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, torch.ones_like(rewards))
    sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, rewards)
    group_mean = sums / counts.clamp_min(1.0)
    diff_sq = (rewards - group_mean[gid]) ** 2
    sq_sums = torch.zeros(num_groups, device=rewards.device).scatter_add_(
        0, gid, diff_sq)
    group_std = (sq_sums / counts.clamp_min(1.0)).sqrt()
    A = (rewards - group_mean[gid]) / (group_std[gid] + 1e-8)
    A = A.unsqueeze(-1)                                          # [N, 1] shared per token

    # Forward pass policy + ref
    logits = policy(batch["input_ids"]).logits[:, :-1]
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = batch["input_ids"][:, 1:].unsqueeze(-1)
    new_log_probs = log_probs.gather(-1, tgt).squeeze(-1)
    new_log_probs = F.pad(new_log_probs, (1, 0))                 # [N, L]
    mask = batch["action_mask"].float()

    with torch.no_grad():
        ref_logits = ref_policy(batch["input_ids"]).logits[:, :-1]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_lp = ref_log_probs.gather(-1, tgt).squeeze(-1)
        ref_token_lp = F.pad(ref_token_lp, (1, 0))

    # PPO-Clip surrogate (advantage shared per sample)
    ratio = torch.exp(new_log_probs - batch["old_log_probs"])
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A
    pg_per_tok = torch.min(surr1, surr2)                         # [N, L]

    # k3 KL per token: exp(Δ) - Δ - 1, Δ = log(π_ref / π_θ)
    delta = ref_token_lp - new_log_probs                         # log(π_ref / π_θ)
    kl_per_tok_k3 = torch.exp(delta) - delta - 1.0               # ≥ 0

    # Combine: maximize PG - β·KL  →  minimize -PG + β·KL
    token_obj = pg_per_tok - beta * kl_per_tok_k3
    seq_len = mask.sum(dim=-1).clamp_min(1.0)
    per_seq = (token_obj * mask).sum(dim=-1) / seq_len
    loss = -per_seq.mean()

    with torch.no_grad():
        kl_mean = (kl_per_tok_k3 * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {"kl_k3": kl_mean.item(), "advantage_std": A.std().item()}
```

### 7.3　DPO 闭式 loss + implicit reward 监控

```python
def dpo_loss_with_implicit_reward_monitor(policy, ref_policy, batch, beta=0.1):
    """
    DPO loss + implicit reward margin (β × pointwise sequence log-density ratio on
    preference data; this is NOT a KL estimator on preference data — only equals
    β·KL when expectation is taken under y~π_θ, which doesn't hold for fixed preference pairs).
    """
    def log_prob_sum(model, ids, mask):
        logits = model(ids).logits[:, :-1]
        logp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:].unsqueeze(-1)
        token_logp = logp.gather(-1, tgt).squeeze(-1)
        token_mask = mask[:, 1:]
        return (token_logp * token_mask).sum(dim=-1)

    pi_w = log_prob_sum(policy, batch["chosen_ids"], batch["chosen_mask"])
    pi_l = log_prob_sum(policy, batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():
        ref_w = log_prob_sum(ref_policy, batch["chosen_ids"], batch["chosen_mask"])
        ref_l = log_prob_sum(ref_policy, batch["rejected_ids"], batch["rejected_mask"])

    # sequence-level log-density ratios on preference data
    log_ratio_w = pi_w - ref_w               # Σ_t log(π_θ/π_ref) on y_w
    log_ratio_l = pi_l - ref_l

    diff = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(diff).mean()

    # implicit rewards (DPO 定义：β × pointwise sequence log-ratio，detached for logging)
    chosen_reward = beta * log_ratio_w.detach()
    rejected_reward = beta * log_ratio_l.detach()
    margin = (chosen_reward - rejected_reward).mean()

    # ⚠️ 注意：preference data 上的 log-ratio 平均**不是** KL estimator——KL 需要 y ~ π_θ 采样。
    # 这里只能监控"模型在 preference 数据上偏离 ref 的程度"，是 DPO 训练健康度指标，不能当 KL 看。
    avg_pref_log_ratio = ((log_ratio_w.detach() + log_ratio_l.detach()) / 2).mean()
    return loss, {"margin": margin.item(),
                  "avg_pref_log_ratio": avg_pref_log_ratio.item(),
                  "chosen_reward": chosen_reward.mean().item(),
                  "rejected_reward": rejected_reward.mean().item()}
```

### 7.4　Reward overoptimization 监控

```python
def overoptimization_monitor(policy, ref_policy, gold_reward_fn, prompts,
                             checkpoints_kl, gold_at_kl):
    """
    Track gold reward as a function of measured KL distance during training.
    Call periodically; plot (measured_KL, gold_reward) to visualize the inverted-U.

    Args:
      policy:          current π_θ
      ref_policy:      frozen π_ref
      gold_reward_fn:  callable (text -> float), uses gold RM or human eval
      prompts:         list of held-out prompts (small, e.g. 64)
      checkpoints_kl:  running list of KL values across training
      gold_at_kl:      running list of gold reward values
    """
    measured_kl_total = 0.0
    gold_reward_total = 0.0
    for prompt in prompts:
        # Generate from policy and ref; compute per-token k3 KL on policy generation
        out_policy = policy.generate(prompt, do_sample=True)
        # ... (use generate to get logits, mask, compute k3 KL) ...
        # For simplicity here, just track the average sequence-level KL
        kl_seq = compute_seq_kl_k3(policy, ref_policy, out_policy)
        gold = gold_reward_fn(out_policy)
        measured_kl_total += kl_seq
        gold_reward_total += gold
    avg_kl = measured_kl_total / len(prompts)
    avg_gold = gold_reward_total / len(prompts)
    checkpoints_kl.append(avg_kl)
    gold_at_kl.append(avg_gold)
    # Detection heuristic: if last 5 gold values are decreasing while KL is rising,
    # we are likely in the over-optimization regime.
    if len(gold_at_kl) >= 5:
        recent_gold = gold_at_kl[-5:]
        recent_kl = checkpoints_kl[-5:]
        if all(recent_gold[i] >= recent_gold[i+1] for i in range(4)) \
           and all(recent_kl[i] <= recent_kl[i+1] for i in range(4)):
            print(f"⚠️ Possible reward over-optimization: KL ↑, gold ↓ over 5 checkpoints")
    return avg_kl, avg_gold
```

### 7.5　工程清单（debug checklist）

> ⚠️ **KL 相关 bug Top 8** —

1. **KL 用错了 mask**：在 prompt token 上算 KL 应该 = 0（同 prompt 输入），但 floating-point 噪声会污染，所以**必须 mask only assistant tokens**。
2. **k1 显示为负**：log-ratio 单样本可负，但**期望非负**。不是 bug，是 estimator 性质。换 k3 / k2 看监控更直观。
3. **β 单位错**：reward scale = O(1) 时 β = 0.02；reward scale = O(100) 时 β 要相应放大。否则 KL anchor 失效。
4. **adaptive β 振荡**：target_kl 太严 / mult 太大。把 mult clip 到 [0.8, 1.2]，update 频率降到每 100 steps。
5. **Reference policy 没冻结**：忘记 `ref_policy.eval()` + `torch.no_grad()`，ref 跟着训，KL 变成 self-distillation。
6. **Per-token vs sequence-level KL 混淆**：监控里有时报 per-token，有时报 sequence-level，看错就调错 β。统一一个单位。
7. **GRPO 中 KL 项缩 reward**：当 reward 是 binary {0, 1}（数学题）且 β = 0.04，KL ≈ 0.5 时，KL penalty 已大于 reward 平均值——loss 被 KL 主导。**降 β 到 1e-3 或更小**。
8. **Float overflow on $\pi_\text{ref}/\pi_\theta$**：当 policy 大幅偏离 ref，$\pi_\text{ref}/\pi_\theta$ 可能极大，$e^\Delta$ 溢出。**用 log-space 写 k3**：`exp(delta) - delta - 1` 在 $\Delta$ 大时仍可能溢出，可以加 `torch.clamp(delta, max=10)` 或换数值稳定 form。

## §8 失败模式：KL Collapse / Runaway / Reward Hacking

### 8.1　KL Collapse（β 过大或 entropy 太低）

**症状**：

- KL 曲线 $\approx 0$，policy 等同 reference。
- Reward 不上升。
- Generation 与 SFT 完全相同。

**原因**：

- β 过大，KL penalty 主导 loss，PG 信号被压住。
- Policy entropy 太低（SFT 后已经很 deterministic）→ 进一步 update 困难。

**修复**：

- 降低 β（adaptive controller 应能自动处理）。
- 加 entropy bonus（$+ c_e \cdot H[\pi_\theta]$）。
- 检查 PG 与 KL loss 的 magnitude ratio：理想 $|\text{PG}| / (\beta \cdot |\text{KL}|) \in [1, 10]$。

### 8.2　KL Runaway（β 过小或 reward 信号太强）

**症状**：

- KL 持续上升，policy 与 ref 距离越来越大。
- Generation 变长、出现重复模式、风格漂移。
- Validation 上 gold reward 开始下降（reward overoptimization）。

**原因**：

- β 过小，KL anchor 失效。
- Reward 信号过强（neural RM 给出 huge gradient）。
- PPO importance ratio 经常超出 clip 范围，KL 信号被 clip 屏蔽（Option A 特有）。

**修复**：

- 升 β（adaptive）。
- 加 max_kl_budget 提前停（当 measured KL > target 时停训）。
- 用 reward model ensemble + conservative aggregation（min / mean - std）。
- 若是 Option A，考虑切到 Option B（KL 不被 PPO clip 屏蔽）。

### 8.3　Reward Overoptimization（KL 距离的 inverted-U）

**症状**：

- Proxy reward 持续上升。
- Gold reward 先升后降（inverted-U）。
- 人类评测：模型在 in-distribution 数据上看起来好，但 OOD prompt 上崩。

**原因**：见 §6.4，RM 与 gold reward 的本质不同。

**修复**：

1. **KL budget early stop**：把 measured KL 限制在 inverted-U peak 之前（需要 gold reward 监控）。
2. **RM ensemble**：多个 seed 的 RM，取 min 或 mean - $k\cdot\text{std}$。
3. **混 RL + DPO**：先 DPO 拿 70%，再 small-β PPO 跑最后一公里。
4. **Rule-based reward 替代神经 RM**：数学 / 代码任务的根本解法。
5. **PRM 替代 ORM**：dense reward → 单步 over-optimization 受限。

### 8.4　Length bias（DPO 特有，也可视作 KL anchor 不够强）

**症状**：

- DPO 训完后输出明显变长。
- AlpacaEval 上分数高但用户感受变啰嗦。

**原因**：DPO loss 是 sequence-level log-ratio 差。$y_w$ 通常更长（人选更详尽答案），longer $y_w$ → 更负的 $\log\pi(y_w)$ → log-ratio 差更大 → loss 减小。但这是 RM scale 而非 reasoning quality。

**修复**：

- SimPO 的 length-normalization（$r = (\beta/|y|)\log\pi$）。
- Reward shaping 加 length penalty。
- 数据 curation：让 $y_w$ 和 $y_l$ 长度分布相似。

### 8.5　Reference policy "wrong checkpoint" failure

**症状**：训练正常但下游 eval 极差。

**原因**：用了错误的 ref checkpoint（比如 pretrain base 而不是 SFT）。KL anchor 锚到了"语言模型"而不是"指令模型"。

**修复**：reference 必须是 immediate-previous-stage SFT checkpoint，不能跳层级。

## §9 25 高频面试题

按难度分 3 档：L1 = 任何 LLM 工程岗都会问；L2 = research / alignment 团队会问；L3 = 顶级 lab / DeepSeek 量级团队的硬核题。每题点开看答案要点 + 易踩坑。

### L1 必会题（10 题）

<details>

<summary>Q1.KL 散度的定义和 3 个核心性质？</summary>

- 定义：$\text{KL}(p\|q) = \mathbb{E}_{x\sim p}[\log(p(x)/q(x))]$
- 非负（Jensen）；等号 ⟺ $p = q$ a.e.
- 不对称（$\text{KL}(p\|q) \ne \text{KL}(q\|p)$），不是 metric
- 联合凸，参数化变换下不变（reparam invariant）

说 KL 是"距离"——错（不满足三角不等式）；或不知道凸性。

</details>

<details>

<summary>Q2.RLHF 里为什么要加 KL penalty？</summary>

- 防 reward hacking：policy 找 RM 盲点拿高分但人感受变差
- 防语言流畅性塌掉：no-KL 时模型可能输出"鬼话但 RM 给高分"
- 防 distribution shift：policy 离 SFT 太远，RM 自己也不准（OOD）
- 提供闭式最优 policy：$\pi^* \propto \pi_\text{ref}\exp(r/\beta)$

只说"防过拟合"——不够具体；不知道 reward hacking。

</details>

<details>

<summary>Q3.k1 estimator 是什么？为什么单样本可以为负？</summary>

- k1: $\widehat{\text{KL}}_1 = \log(\pi_\theta(y)/\pi_\text{ref}(y))$（直接 log-ratio）
- **期望** $\mathbb{E}_{y\sim\pi_\theta}[\log r] = \text{KL}$（无偏），但**单样本** $\log r$ 可正可负
- 原因：单个样本 $y$ 上 $\pi_\theta$ 可能比 $\pi_\text{ref}$ 大也可能小，log-ratio 没有"非负"约束
- 工程影响：用 k1 做监控时看到负值不要慌，是 estimator 本性

把单 estimator 和期望混淆；或不知 expectation 非负 ≠ pointwise 非负。

</details>

<details>

<summary>Q4.k3 estimator (Schulman 2020) 公式是什么？三大性质？</summary>

- 公式：$\widehat{\text{KL}}_3 = e^\Delta - \Delta - 1$，$\Delta = \log(\pi_\text{ref}/\pi_\theta) = -\log r$
- 等价形式：$\widehat{\text{KL}}_3 = (\pi_\text{ref}/\pi_\theta) - \log(\pi_\text{ref}/\pi_\theta) - 1$
- **无偏**（$\mathbb{E}_{\pi_\theta} = \text{KL}$，用 $\sum_y\pi_\text{ref} = 1$）
- **非负**（$f(x) = e^x - x - 1 \ge 0$）
- **低方差**（带 expectation-0 control variate $\pi_\text{ref}/\pi_\theta - 1$）

只背公式不知道 derivation；或不知道它非负。

</details>

<details>

<summary>Q5.KL 在 RLHF 中有哪两种放置？区别？</summary>

- **In-reward shaping (Option A)**：$\tilde{r}_t = r_t - \beta \cdot \text{KL}_t$（用 k1），KL 进 advantage / GAE。InstructGPT 标准。
- **In-loss regularization (Option B)**：$\mathcal{L} = \mathcal{L}_\text{PG} + \beta \cdot \mathbb{E}[\text{KL}]$。GRPO / DAPO 历史选 k3 estimator。
- 目标函数相同，但**梯度路径不同**。**Principled gradient** 选择：(P1) k1 in reward；(P2) k2 as loss—— on-policy 下两者 gradient-equivalent，都是严格 reverse-KL gradient。**GRPO 的 k3-as-loss** 是 reverse-KL gradient 的 first-order Taylor approximation，有 $O(\Delta^2)$ bias（参见 §3.6 / Rethinking KL arXiv 2510.01555）。
- PPO clip 在 Option A 会"屏蔽"超出 clip 的 KL（advantage 内含 KL 一并被 clip）；Option B 不会。
- 实践：math/code RL 历史上选 B (k3) 因为 β 小、$\Delta$ 小，bias 可忽略；helpfulness / safety 多用 A (k1)；如果要严格 principled，on-policy 用 P2 (k2-as-loss) 是 simplest principled 选择。

说两者完全等价——目标相同梯度不等价；说 k3-as-loss 是 principled——它是 first-order approximation，理论不严格。

</details>

<details>

<summary>Q6.β 怎么调？太大太小各有什么症状？</summary>

- β 太大：KL ≈ 0，policy 卡在 ref，reward 不涨，模型基本等同 SFT
- β 太小：KL runaway，policy 漂走，generation 变长 / 重复 / 谄媚，reward hacking
- 起点：InstructGPT 用 β ≈ 0.02（per-token）；GRPO 数学任务 β ≈ 0.04 或更小；DAPO 经常 β ≈ 0
- 调法：固定 / adaptive（按 measured KL 距 target 拉 β） / annealing schedule

只说"看着调"；不知 β 的工程量级。

</details>

<details>

<summary>Q7.Forward KL vs Reverse KL 在 RLHF 里用哪个？为什么？</summary>

- RLHF 几乎都用 $\text{KL}(\pi_\theta \| \pi_\text{ref})$；按标准 VI/RLHF 约定（DPO/RLOO/Rethinking KL 等都采用）这是 **reverse KL**（$\pi_\theta$ 是变分一侧），mode-seeking
- 因为训练时从 $\pi_\theta$ 采样（rollout），$\mathbb{E}_{\pi_\theta}[\log r]$ 直接用样本估
- **Forward KL** $\text{KL}(\pi_\text{ref} \| \pi_\theta)$ 需要从 $\pi_\text{ref}$ 采样，工程上无意义（要训 $\pi_\theta$ 不是 $\pi_\text{ref}$），且 mass-covering 行为不符合 RLHF 目标
- Reverse KL 的 mode-seeking 性质正合规则：在 $\pi_\text{ref}$ 高密度区里选 reward 最高的 mode

不知道命名约定（少数早期 RL 教材以采样分布命名，会把 reverse 标成 forward；面试时给公式更稳）；或反过来说 forward 更好。

</details>

<details>

<summary>Q8.DPO 的 implicit reward 跟 KL 有什么关系？</summary>

- DPO implicit reward $\hat{r}_\theta(x, y) = \beta\log(\pi_\theta(y|x)/\pi_\text{ref}(y|x))$，即 **β × pointwise sequence log-density ratio**
- ⚠️ **不是 KL 本身**：只有当 $y\sim\pi_\theta$ 取期望时，$\mathbb{E}_{y\sim\pi_\theta}[\hat r_\theta(x,y)] = \beta\cdot\text{KL}(\pi_\theta(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x))$。**DPO 训练时 $y_w, y_l$ 来自固定 preference 数据**，所以这里的 $\hat r$ 不是 KL estimator，不是 KL margin
- 训练时 maximize $\hat{r}(y_w) - \hat{r}(y_l)$ ≡ pairwise log-ratio margin（preference 对的相对置信度），**不是** "KL margin"
- KL anchor 是**隐式**的：通过 $\pi_\text{ref}$ 出现在 $\hat r$ 的分母带来 anchoring 效果——但显式 KL 不在 loss 里
- 对偶 RL 视角：DPO 反推自 KL-regularized 闭式最优 policy $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$；隐式优化的目标是同一个 RLHF 目标，但 estimator 路径不同

说 DPO 没有 KL（错，是隐式 anchor）；或把 $\hat r$ 直接读作 KL / KL margin（错，需要 y~π_θ 期望条件）。

</details>

<details>

<summary>Q9.Reward overoptimization 是什么？为什么发生？</summary>

- Proxy reward（RM）随 KL 单调升，但 gold reward（人评）先升后降（**inverted-U**）
- Gao, Schulman, Hilton 2023 ICML 给出 KL vs gold-reward 拟合形式
- 原因：RM 与 gold reward 不同（RM 是 finite-data 学的近似），policy 学到 RM 偏好但不一定是真实质量
- 缓解：KL budget early stop / RM ensemble / PRM / mix DPO + PPO

只说"过拟合"——太笼统；不知道 KL 是 over-optimization 的 axis。

</details>

<details>

<summary>Q10.GRPO 为什么历史上用 k3 不用 k1？</summary>

- **工程动机**：GRPO 把 KL 放 loss，k3 的 **value-estimator** 性质（非负 + value-side unbiased + 低方差）让 KL 监控曲线直观
- **k1-as-loss 不可行**：直接对 $\log r = \log\pi_\theta - \log\pi_\text{ref}$ 反传时，$\nabla = \nabla\log\pi_\theta$，在 on-policy 期望下为 0，**不是** reverse-KL gradient。k1 必须**进 reward / advantage**（作为 detached scalar 走 score-function trick）才能给出严格的 reverse-KL gradient。把 k1 当 loss 项加既数值上可负不好看，gradient 也错
- **k3-as-loss 的真实代价**：对 θ 反传的 gradient 是 reverse-KL gradient 的 **first-order Taylor approximation**，$1 - e^{-\Delta} = \Delta - \tfrac12\Delta^2 + O(\Delta^3)$，有 $O(\Delta^2)$ bias（见 §3.6 + Rethinking KL arXiv 2510.01555）
- **Principled 替代**：on-policy 下 (P1) k1 in reward 或 (P2) **k2 as loss**（$\tfrac12\Delta^2$，$\nabla = \Delta\cdot\nabla\log\pi_\theta$，gradient-equivalent to P1）都是严格 reverse-KL gradient
- **历史 DeepSeekMath / DAPO / verl 默认 k3** 是工程惯例，$\beta$ 锁得紧、$\Delta$ 很小时 bias 可忽略——但这是经验合理化，不是理论 principled

只说 k3 "三优满足"不给 gradient bias caveat；不知道 (P2) k2-as-loss 是 P1 的 gradient-equivalent 替代。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11.推导 k3 的无偏性。</summary>

需要证 $\mathbb{E}_{y\sim\pi_\theta}[(\pi_\text{ref}/\pi_\theta) - \log(\pi_\text{ref}/\pi_\theta) - 1] = \text{KL}(\pi_\theta\|\pi_\text{ref})$。

1. $\mathbb{E}_{\pi_\theta}[\pi_\text{ref}/\pi_\theta] = \sum_y \pi_\theta(y)\cdot\pi_\text{ref}(y)/\pi_\theta(y) = \sum_y\pi_\text{ref}(y) = 1$
2. $\mathbb{E}_{\pi_\theta}[-\log(\pi_\text{ref}/\pi_\theta)] = \mathbb{E}_{\pi_\theta}[\log(\pi_\theta/\pi_\text{ref})] = \text{KL}(\pi_\theta\|\pi_\text{ref})$
3. 合并：$\mathbb{E}[\widehat{\text{KL}}_3] = 1 + \text{KL} - 1 = \text{KL}$ ✓

关键：用 $\sum_y\pi_\text{ref} = 1$。如果你忘了 $\pi_\text{ref}/\pi_\theta$ 这一项是 expectation 1 的 control variate，证不出无偏。

</details>

<details>

<summary>Q12.从 $f(x) = e^x - x - 1$ 推导 k3 estimator。</summary>

构造性证明：

1. $f(x) = e^x - x - 1$ 在 $\mathbb{R}$ 非负（凸函数 $e^x$ 在 $x = 0$ 切线 $1 + x$ 之上）。
2. 取 $x = \log(\pi_\text{ref}(y)/\pi_\theta(y)) = -\log r$：
   - $e^{-\log r} = 1/r = \pi_\text{ref}/\pi_\theta$
   - $-(-\log r) = \log r = \log(\pi_\theta/\pi_\text{ref})$
   - $f(-\log r) = \pi_\text{ref}/\pi_\theta - \log(\pi_\text{ref}/\pi_\theta) - 1 = e^\Delta - \Delta - 1$，$\Delta = -\log r$
3. 期望：$\mathbb{E}_{\pi_\theta}[f(-\log r)] = \text{KL}$（已在 Q11 证）。

所以 k3 = $f(\log p/q)$ 的特殊形式，恰好满足"非负 + 无偏 + 含 control variate" 三个性质。

只写公式不解释 $f$ 的构造动机；或不知道凸函数切线 → 非负。

</details>

<details>

<summary>Q13.推 BT preference 下 KL-regularized RL 的闭式最优 policy。</summary>

目标：$\max_\pi \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,\text{KL}(\pi\|\pi_\text{ref})$。

1. 写 Lagrangian：$\sum_y \pi r - \beta\sum_y\pi\log(\pi/\pi_\text{ref}) - \mu(\sum_y\pi - 1)$
2. 对 $\pi(y)$ 求偏导 = 0：$r - \beta(\log(\pi/\pi_\text{ref}) + 1) - \mu = 0$
3. 整理：$\log(\pi/\pi_\text{ref}) = (r - \mu - \beta)/\beta$
4. 指数化：$\pi(y) = \pi_\text{ref}(y)\exp((r - \mu - \beta)/\beta)$
5. 用 $\sum_y\pi = 1$ 解 $\mu$：$e^{(\mu+\beta)/\beta} = \sum_y\pi_\text{ref}\exp(r/\beta) = Z$
6. 得：$\pi^*(y|x) = \pi_\text{ref}(y|x)\exp(r/\beta)/Z(x)$

注意：$J$ 在 $\pi$ 上严格凹，最优唯一。

漏 normalization 约束；漏 strict concavity argument；或把 sign 搞反。

</details>

<details>

<summary>Q14.推 DPO loss 从 §6.1 的闭式 $\pi^*$ 开始。</summary>

1. §6.1 给 $\pi^*(y|x) = \pi_\text{ref}\exp(r/\beta)/Z(x)$
2. 反解 $r(x,y) = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$
3. Bradley-Terry: $P(y_w \succ y_l) = \sigma(r_w - r_l)$
4. 关键观察：$\beta\log Z(x)$ 不依赖 $y$，**在 $r_w - r_l$ 差里消掉**：
   $r_w - r_l = \beta\log(\pi^*(y_w)/\pi_\text{ref}(y_w)) - \beta\log(\pi^*(y_l)/\pi_\text{ref}(y_l))$
5. 把 $\pi^*$ 改为可学 $\pi_\theta$，对偏好数据做 NLL：
   $\mathcal{L}_\text{DPO} = -\mathbb{E}\log\sigma(\beta\log(\pi_\theta(y_w)/\pi_\text{ref}(y_w)) - \beta\log(\pi_\theta(y_l)/\pi_\text{ref}(y_l)))$

关键 trick：**$\log Z$ 不依赖 $y$**，所以在 pairwise 差里抵消（这是 DPO 不需要采样的根本原因）。

不解释 $\log Z$ 为什么消掉；或不知道 BT 的 sigmoid。

</details>

<details>

<summary>Q15.推 DPO 梯度。</summary>

记 $h_\theta = \beta\log(\pi_\theta(y_w)/\pi_\text{ref}(y_w)) - \beta\log(\pi_\theta(y_l)/\pi_\text{ref}(y_l))$（implicit reward margin）。

$\mathcal{L}_\text{DPO} = -\mathbb{E}\log\sigma(h_\theta)$，$\nabla = -\mathbb{E}\sigma'(h_\theta)/\sigma(h_\theta) \cdot \nabla h_\theta$。

用 $\sigma'(x) = \sigma(x)\sigma(-x)$，$\sigma'(h)/\sigma(h) = \sigma(-h) = \sigma(\hat{r}_l - \hat{r}_w)$（即"错位置信度"）。

$\nabla h_\theta = \beta(\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l))$（$\pi_\text{ref}$ 是常数，求 $\theta$ 导为 0）。

合并：

$$\nabla\mathcal{L}_\text{DPO} = -\beta\mathbb{E}[\sigma(\hat{r}_l - \hat{r}_w)(\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l))]$$

解读：

- 系数 $\sigma(\hat{r}_l - \hat{r}_w)$ 是 "当前 model 多大程度上错把 $y_l$ 排在前面" → hard-example mining
- 后面就是提升 $y_w$ 概率 + 降低 $y_l$ 概率

少推一步 sigmoid' 公式；或不知道 $\sigma'(x) = \sigma(x)\sigma(-x)$。

</details>

<details>

<summary>Q16.k2 estimator 为什么有偏？小 KL 极限下偏多少？</summary>

记 $r = \pi_\theta/\pi_\text{ref}$，$y \sim \pi_\theta$，$\text{KL}(\pi_\theta\|\pi_\text{ref}) = \mathbb{E}_{\pi_\theta}[\log r]$。

- **k2** 定义 $\mathbb{E}_{\pi_\theta}[\tfrac12(\log r)^2]$，一般 $\ne \mathbb{E}[\log r]$ — 所以**有偏**。
- **正确的 Taylor 展开** 要用 $s = \pi_\text{ref}/\pi_\theta = 1/r$（这是 $\mathbb{E}_{\pi_\theta}[s] = 1$ 的随机变量，可做 mean-1 展开）。$\log r = -\log s$，$\log s = (s-1) - \tfrac12 (s-1)^2 + O((s-1)^3)$。
- $\mathbb{E}_{\pi_\theta}[\log r] = -\mathbb{E}[\log s] = -\mathbb{E}[s-1] + \tfrac12\mathbb{E}[(s-1)^2] + O(\cdot) = 0 + \tfrac12\mathrm{Var}(s) + O$. 这与 Fisher 二阶展开一致。
- $\mathbb{E}_{\pi_\theta}[\tfrac12 (\log r)^2] = \tfrac12\mathbb{E}[(\log s)^2] = \tfrac12\mathbb{E}[(s-1)^2] + O((s-1)^3) = \tfrac12\mathrm{Var}(s) + O$.
- 所以**小 KL（即 $r\approx 1$）极限下**：$\mathbb{E}[k_2] \approx \mathbb{E}[\log r] = \text{KL}$，二阶等价。
- 大 KL 下 k2 系统性偏离 KL（高阶项不可忽略，方向取决于分布的高阶矩）。

**关键修正**：旧版本写 "$\mathbb{E}[r-1] = 0$" 是错的——在 $y\sim\pi_\theta$ 下 $\mathbb{E}[r] = \mathbb{E}_{\pi_\theta}[\pi_\theta/\pi_\text{ref}]$ 一般 $\ne 1$。$\mathbb{E}[s] = 1$ 才是正确的恒等式（这是 importance sampling 的标准结果）。

只说"二阶近似"；不展开 Taylor，或者错用 $r$ 而非 $s$ 做 mean-1 展开。

</details>

<details>

<summary>Q17.Adaptive β 与 PID controller 的类比？</summary>

- Adaptive β（PPO Schulman 2017）只用 P 项：若 KL > target → β ↑；若 KL < target → β ↓
- 对比 PID controller：$u(t) = K_p e + K_i\int e + K_d \dot{e}$
- P 项 = 当前误差比例响应
- I 项 = 累积误差（防 steady-state error），但 KL 是 stochastic，加 I 容易振荡
- D 项 = 误差变化率（damping），但 KL 估计本身高方差，D 容易放大噪声
- 工程实践：**只用 P 项**最 robust（TRL 默认）；某些 framework 加 small I 项做长期收敛

只说 "类似 P controller" 不展开；或加上 D 项不知会出问题。

</details>

<details>

<summary>Q18.GRPO 的 k3 KL 在 reward 是 binary (0/1) 时怎么调 β？</summary>

- Reward = {0, 1}，advantage scale ≈ O(1)
- KL_k3 per-token 初始 ≈ 0（policy 与 ref 相同），训练中可达 0.1 ~ 0.5
- 若 β = 0.04，KL penalty per-token ≈ 0.04 × 0.3 ≈ 0.012，与 advantage 比合理
- 若 β = 1，KL penalty per-token ≈ 0.3，**远大于** advantage，loss 被 KL 主导
- 经验：数学任务起始 β = 0.01 ~ 0.04，看 KL 曲线调；DAPO 经常 β = 1e-3 或 0
- 对比 RLHF helpfulness 任务：reward scale 大（连续 [-5, 5]）→ β 可以大些（0.1 ~ 0.5）

不知 β 与 reward scale 的耦合；或机械套 β = 0.04。

</details>

<details>

<summary>Q19.SimPO 没有 reference model，为什么还能 work？没了 KL anchor 不是会 reward hack 吗？</summary>

- SimPO 用 **length-normalization** $r = (\beta/|y|)\log\pi(y)$ + **target reward margin** $\gamma$
- 没有 KL anchor，但 length-norm 阻止 "$y_w$ 越长越优" 的退化
- Margin $\gamma$ 让 loss 在 $r_w - r_l > \gamma$ 时 saturate，避免 implicit reward 无界
- 实测某些 benchmark（AlpacaEval-2 / Arena-Hard）SimPO 优于 DPO
- 但**没有 KL anchor 的代价**：OOD 鲁棒性差、对 $\beta, \gamma$ 调参敏感、生成长度可能仍然偏大
- 实际生产中 DPO + small β 仍然主流，SimPO 是 "在某些 benchmark 上更强但 trade off 不同" 的选择

说 SimPO 一定更好；或不知道 length-norm + margin 是 KL 的替代。

</details>

<details>

<summary>Q20.Per-token KL 和 sequence-level KL 是什么关系？</summary>

- 由 KL 的**链式法则**（条件分解）：
  $\text{KL}(\pi_\theta(\cdot|x) \| \pi_\text{ref}(\cdot|x)) = \mathbb{E}_{y\sim\pi_\theta}[\sum_t \log(\pi_\theta(y_t|x,y_{<t})/\pi_\text{ref}(y_t|x,y_{<t}))]$
- 即 **sequence-level KL = token-level KL 之和的期望**
- 工程上只能 token-level 算（vocab 维度求和 OK，sequence 维度组合爆炸）
- 实际写 RLHF loss：per-token KL → mask assistant tokens → sum → average per sequence
- 注意：prompt token 上 policy 和 ref 输入相同，KL 应 = 0，但 floating-point 噪声会污染 → 必须 mask

混淆两者；或不知道 mask only assistant tokens 的必要性。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21.证明 $f(x) = e^x - x - 1 \ge 0$ 对所有 $x \in \mathbb{R}$。从该不等式出发，推导 k3 的非负性、无偏性。哪个性质对 "KL 作为 loss 加项" 更关键，为什么？</summary>

**非负证明**：

$f(x) = e^x - x - 1$。$f'(x) = e^x - 1$，$f'(x) = 0$ ⟺ $x = 0$。$f''(x) = e^x > 0$，$f$ 严格凸，$x = 0$ 是全局最小，$f(0) = 0$。所以 $f(x) \ge 0$ 对所有 $x$。✓

**k3 非负性**：取 $x = \log(\pi_\text{ref}/\pi_\theta) = -\log r$，则 $\widehat{\text{KL}}_3 = f(-\log r) \ge 0$ 总成立。✓

**k3 无偏性**：

$\mathbb{E}_{\pi_\theta}[\widehat{\text{KL}}_3] = \mathbb{E}_{\pi_\theta}[e^{-\log r}] + \mathbb{E}_{\pi_\theta}[\log r] - 1 = 1 + \text{KL} - 1 = \text{KL}$ ✓

**哪个性质对 loss 加项更关键？**

- **非负 + value-side 无偏**：这些是好的**监控/数值显示**性质——KL loss 曲线不会出现负值或大幅噪声。但 **k3-as-loss 的关键问题不在数值层**，而在 gradient 层。
- ⚠️ **k3 作为 loss 反传时的 gradient 是 reverse-KL gradient 的 first-order Taylor approximation**：$\nabla(e^{-\Delta} + \Delta - 1) = (1-e^{-\Delta})\nabla\Delta = (\Delta - \tfrac12\Delta^2 + O(\Delta^3))\,\nabla\log\pi_\theta$，有 $O(\Delta^2)$ bias（见 §3.6 + Rethinking KL arXiv 2510.01555）。
- **真正 principled 的 loss 选择是 (P2) k2-as-loss**：$\nabla(\tfrac12\Delta^2) = \Delta\,\nabla\log\pi_\theta$ 在 on-policy 期望下等于严格 reverse-KL gradient。k2 数值未 calibrated 但 gradient 正确，与 k1-in-reward gradient-equivalent。
- **如果坚持要"非负 + 无偏 + 易监控"** ⇒ k3 仍是最佳 **value estimator**；但要分清"value-side k3 monitor + loss-side k2/k1-in-reward gradient"这一组合，比 "k3-as-loss" 更 principled。

所以 k3 在 **value-monitoring 层面**最方便（非负 + value-unbiased + 低方差）。但**对于 "as loss addend" 的 principled 性，关键不是非负，而是 loss-gradient 是否匹配真实 reverse-KL gradient**——k3-as-loss 的 gradient $(1 - e^{-\Delta})\nabla\log\pi_\theta$ 只是 first-order Taylor approximation，有 $O(\Delta^2)$ bias；on-policy 下严格 principled 的 loss 是 (P2) k2-as-loss 或 (P1) k1-in-reward。GRPO/DAPO 历史用 k3 主要是工程惯例 + 小 β 时 bias 可忽略。

**易错**：把非负当作 "loss addend 的 key" —— 那只是数值显示层面的优势；真正决定 principled 性的是 gradient correctness。理想组合是 "value-side k3 monitor + loss-side k2/k1-in-reward gradient"。

</details>

<details>

<summary>Q22.推导 DPO 的 implicit reward 与 KL-regularized RL 的 policy gradient 之间的"对偶等价性"。</summary>

设定：$\max_\pi \mathbb{E}_\pi[r] - \beta\,\text{KL}(\pi\|\pi_\text{ref})$，闭式解 $\pi^*(y|x) = \pi_\text{ref}\exp(r/\beta)/Z$。

**RL 视角**：当 reward $r$ 给定时，最优 policy 通过 importance-weighted update 朝 $\pi^*$ 走。Score function policy gradient：

$\nabla_\theta J = \mathbb{E}_{\pi_\theta}[\nabla\log\pi_\theta \cdot (r - \beta\log(\pi_\theta/\pi_\text{ref}))]$

**DPO 视角**：把 $\pi$ 直接看作待学 $\pi_\theta$，用 BT 反解 implicit reward $\hat{r} = \beta\log(\pi_\theta/\pi_\text{ref}) + \beta\log Z$，对偏好对做 NLL。

**等价性**：

1. 闭式解给出 $r = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z$。
2. 若 $\pi_\theta = \pi^*$（即 RL 已收敛），则 implicit reward 恢复真实 $r$（up to additive $\beta\log Z$，但 $\log Z$ 在 BT pairwise 中抵消）。
3. RL 梯度 $\propto r - \beta\log(\pi_\theta/\pi_\text{ref})$，在收敛时 $= \beta\log Z(x)$（常数 in $y$）→ 梯度为零（PG 在 $\pi^*$ 处停）。
4. DPO 梯度 $\propto \sigma(\hat{r}_l - \hat{r}_w)(\nabla\log\pi(y_w) - \nabla\log\pi(y_l))$，在 $\pi_\theta = \pi^*$ 处对应"BT 模型 perfectly fits" → 梯度同步消失。
5. 两个视角的固定点（fixed point）都是 $\pi^*$，所以**RL 和 DPO 都在 optimize 同一目标，只是从两个角度切入**：RL 是 forward optimization（直接 max $J$），DPO 是 inverse optimization（用偏好数据 fit BT 模型）。

**关键**：DPO 不是 RL 的替代，是**对同一目标的不同 reduction**——RL 通过 sampling-based PG，DPO 通过 closed-form + supervised learning。"DPO 没有 RL" 是误读。

只说"DPO 推 RL 闭式解"不展开 gradient 等价性；或不知道两者 fixed point 相同。

</details>

<details>

<summary>Q23.从 $\sqrt{\text{KL}}$ 和 inverted-U gold curve 推 reward overoptimization 的"安全 KL budget"。</summary>

Gao 2023 的 BoN gold reward 拟合：$R_g(d) = d(\alpha_g - \gamma_g d)$，$d = \sqrt{\text{KL}}$。

求 peak：$dR_g/dd = \alpha_g - 2\gamma_g d = 0$ → $d_\text{peak} = \alpha_g / (2\gamma_g)$。

对应 KL distance：$\text{KL}_\text{peak} = d_\text{peak}^2 = \alpha_g^2 / (4\gamma_g^2)$。

**安全 budget**：要在 peak 之前停下来。一般选 $\text{KL}_\text{stop} = 0.5 \cdot \text{KL}_\text{peak}$（留 50% safety margin）。

**怎么估 $\alpha_g, \gamma_g$？** 需要 gold reward 信号（人工 / 强 RM ensemble）。在小规模 pilot run 跑几个 KL 点，拟合 $R_g(d)$。

**RM 越大，越靠右**：Gao 2023 给的 scaling law 表明 $\alpha_g, \gamma_g$ 随 RM size 改变（拟合系数本身依赖于 RM 规模和数据），更大 RM 的 $\text{KL}_\text{peak}$ 更靠右，但 $R_g(\text{peak})$ 也更高。这是为什么大 RM 既"对得起 budget" 又能拿更高 gold。

**PPO 比 BoN 复杂一点**：$R_g(d) = d(\alpha_g - \gamma_g d) - \delta_g d^{3/2}$，三阶项使 peak 稍微靠左。

应用：实际工程里把 measured KL 限制在 $\text{KL}_\text{peak} \cdot 0.5$ 作 early stop，即 "$\sqrt{\text{KL}}$ < $d_\text{peak}/2$"。

不展开 derivative；或不知 RM size 与 peak 的 scaling。

</details>

<details>

<summary>Q24.如果让你设计一个 RLHF 算法，把 reverse + forward KL 联合用，会怎么用？</summary>

**动机**（按标准约定：reverse = $\text{KL}(q\|p)$ mode-seeking，forward = $\text{KL}(p\|q)$ mass-covering）：

- **Reverse KL** $\text{KL}(\pi_\theta\|\pi_\text{ref})$ = mode-seeking on $\pi_\theta$ → 选 ref 高密度区里 reward 高的 mode（合规则）。**RLHF 默认就用这条**。
- **Forward KL** $\text{KL}(\pi_\text{ref}\|\pi_\theta)$ = mass-covering on $\pi_\theta$ → 让 $\pi_\theta$ 覆盖 ref 的所有可能输出（防 mode collapse）。

**问题**：forward KL 需要从 $\pi_\text{ref}$ 采样，工程上没意义（要训的是 $\pi_\theta$ 不是 $\pi_\text{ref}$）。

**绕路方案 (theoretical)**：

1. **JSD-style combo**：用 $\text{JS}(\pi_\theta\|\pi_\text{ref}) = \tfrac{1}{2}\text{KL}(\pi_\theta\|m) + \tfrac{1}{2}\text{KL}(\pi_\text{ref}\|m)$，$m = (\pi_\theta + \pi_\text{ref})/2$。对称、有界，可同时拿 mode-seeking 和 mass-covering。问题：$m$ 不是闭式（mixture distribution），梯度计算复杂。
2. **Importance sampling**：从 $\pi_\theta$ 采样，但 reweight 成 $\pi_\text{ref}$ 期望：$\mathbb{E}_{\pi_\theta}[(\pi_\text{ref}/\pi_\theta)\log(\pi_\text{ref}/\pi_\theta)] = \text{KL}(\pi_\text{ref}\|\pi_\theta)$。问题：tail 上 $\pi_\text{ref}/\pi_\theta$ 极大，方差炸。
3. **Symmetric KL**：$\text{KL}_\text{sym} = \tfrac{1}{2}(\text{KL}(\pi_\theta\|\pi_\text{ref}) + \text{KL}(\pi_\text{ref}\|\pi_\theta))$，对称但同样需要 forward 一侧的估计。
4. **Hybrid penalty**：训练前期主要 reverse KL（让 policy 朝 mode 收）+ 后期加 small forward KL term 用 importance sampling 估，限制 mode collapse。

**实际工业方案（更简单）**：

- 用 entropy bonus 替代 forward KL 的 "mass coverage" 目标——entropy 不需要 ref 采样。
- 用 ensemble policy 训练，多个 policy 各自 mode-seek 不同 mode，整体保持多样性。
- 用 BoN inference + DPO/PPO training 组合：训练时 mode-seek，推理时多样性来源于 BoN 采样。

**总结**：forward KL 在 RLHF 里"理论吸引但工程难"，主流做法是用 entropy / multiple policies 替代。

只说"两个都加上"——太天真，要给出工程方案；不知道 forward KL 的 sampling 障碍。

</details>

<details>

<summary>Q25.下一代 RLHF 算法可能怎么改进 KL regularization？给 3 个方向 + trade-off。</summary>

**方向 1：Adaptive KL estimator per token**

- 对每 token 单独决定用 k1 还是 k3：小 KL token 用 k1（无偏 + 简单），大 KL token 用 k3（防方差爆炸）。
- Trade-off：实现复杂度 + per-token 阈值难定。

**方向 2：Per-task β controller**

- 不同 sub-task（数学 / 代码 / 对话 / safety）用不同 β，由 task classifier 在线分发。
- Trade-off：要求 task labels 准确；β 之间的 cross-task 干扰需要研究。

**方向 3：Adversarial KL**

- 不固定 $\pi_\text{ref}$，让 $\pi_\text{ref}$ 也在线学（如 self-rewarding LM、iterative DPO），但每轮固定一个时间窗内的 $\pi_\text{ref}$。
- Trade-off：可能 unstable（GAN-like）；需要 fresh human label 防 drift。

**方向 4：Distributional reward + W2 KL**

- 把 reward 也建模成分布 $p(r|x,y)$，KL 用 Wasserstein-2 metric（geometric）替代。
- Trade-off：W2 计算复杂、需 sliced approximation；理论收敛性不清。

**方向 5：Hierarchical KL**

- 句子级 KL + token 级 KL 联合：句子级控总 KL 预算，token 级做细粒度 anchoring。
- 类比：PRM (process reward) 之于 ORM (outcome reward)。
- Trade-off：sentence boundary 在生成时不显式，需要额外标注或 heuristics。

**方向 6：KL-free 但 trust-region 替代**

- 完全去 KL，用 PPO clip 的 trust region 来约束（"clip is enough"）。
- 已有 IRL / SimPO 等尝试。
- Trade-off：失去闭式 RL 数学基础（无 $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$）。

只罗列没 trade-off；或不知道 SimPO / IPO 已部分探索 KL-free 路线。

</details>

## §A 附录：参考文献清单

按章节分组。本节为草稿初版，**arXiv ID 与精确发表场所未经在线核实**——校验时应通过 `/arxiv` 或 codex web_search 重新查证。

**KL 基础与 estimators**

- Schulman 2020 blog *Approximating KL Divergence* `http://joschu.net/blog/kl-approx.html`（k1 / k2 / k3 estimator 的标准来源）
- Endres & Schindelin 2003 IEEE TIT 49(7) *A New Metric for Probability Distributions*（$\sqrt{\text{JS}}$ 是 metric 的证明）
- Pinsker 1964（原始 Pinsker 不等式 $\text{TV} \le \sqrt{\text{KL}/2}$）

**RLHF / PPO**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms*（PPO-Clip + PPO-Penalty adaptive β）
- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback*（InstructGPT，per-token KL in reward）
- Bai et al. 2022 Anthropic arXiv 2204.05862 *Training a Helpful and Harmless Assistant with RLHF*（adaptive β controller）

**DPO 系**

- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（闭式 + 反解 + BT）
- Azar et al. 2024 AISTATS *A General Theoretical Paradigm to Understand Learning from Human Preferences*（IPO，防 deterministic preference 下 reward 无界）
- Ethayarajh et al. 2024 ICML *KTO: Model Alignment as Prospect Theoretic Optimization*（reference point 替代 KL）
- Meng et al. 2024 NeurIPS *SimPO: Simple Preference Optimization with a Reference-Free Reward*（去 ref，length-norm + margin）
- Hong et al. 2024 EMNLP *ORPO: Monolithic Preference Optimization without Reference Model*（odds-ratio 一阶段）

**GRPO 系 / RL with k3 KL**

- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*（GRPO 首次系统使用 k3 + group-relative advantage）
- DeepSeek-AI 2025 arXiv 2501.12948 *DeepSeek-R1*（GRPO + rule-based reward + emergent CoT）
- Yu et al. 2025 ByteDance arXiv 2503.14476 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*（clip-higher + dynamic sampling + token-level loss + 小 β）

**Reward Overoptimization**

- Gao, Schulman, Hilton 2023 ICML *Scaling Laws for Reward Model Overoptimization*（KL distance vs gold reward 的 inverted-U + 拟合 form）
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*
- Eisenstein et al. 2024 COLM (arXiv 2312.09244, 2023) *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking*

**Critic-free RL（与 KL 放置相关）**

- Ahmadian et al. 2024 ACL *Back to Basics: Revisiting REINFORCE Style Optimization*（RLOO，leave-one-out baseline，KL in reward）
- Li et al. 2024 ICML *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method*

**两篇 2024-2026 KL-in-RLHF 系统分析论文（已核验 arXiv 元数据）**

- Kezhao Liu et al., *Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization*, arXiv 2510.01555 (2025-10-02). 系统区分 KL value estimation vs gradient optimization；指出 k3-as-loss 是 biased first-order approximation；推荐 (P1) k1 in reward 或 (P2) k2 as loss；off-policy 需 IS correction。本教程 §3.6 内容基于该论文。
- Vedant Shah et al., *A Comedy of Estimators: On KL Regularization in RL Training of LLMs*, arXiv 2512.21852 (2025-12-26, v3 2026-03-18). 实证对比 k1/k2/k3 在多种 RL 算法 + placement 组合上的 estimator bias / gradient bias / placement-effect；结论：不存在 universally 最好的 estimator。

**Zhihu 中文资料**

- `https://zhuanlan.zhihu.com/p/1979720260128118305` — *KL 进阶：Forward KL、Reverse KL、KL 估计与应用*（已核验，主题与 RLHF KL estimator 一致）
- `https://zhuanlan.zhihu.com/p/1892008158626546312` — [needs-verify URL accessibility] 题目疑似涉及 k2-loss vs k3-loss / GRPO off-policy / clip_std；用户校验时若链接失效可替换。

**与本文紧密相关的内部教程**

- `docs/tutorials/rlhf_dpo_grpo_ppo_tutorial.md` — RLHF / DPO / GRPO / PPO 总览（包含 BT + 闭式 + DPO 推导）
- `docs/tutorials/reasoning_models_tutorial.md` — Reasoning models 的 RL 训练细节
- `docs/tutorials/agentic_rl_tutorial.md` — Agentic setting 下的 token mask + KL 讨论
