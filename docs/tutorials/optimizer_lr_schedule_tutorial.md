## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Optimizer / LR Schedule** — 一页拿下面试核心要点（详见后文 §1–§11 推导）。

1. **为什么不止用裸 SGD**：真实 loss 地形**病态**（Hessian 条件数 $\kappa=\lambda_{\max}/\lambda_{\min}$ 极大），裸梯度下降在陡方向来回**震荡**、在缓方向**爬不动**。两味药：**动量**（沿一致方向累积、震荡相消）+ **逐参数自适应步长**（对角预条件，拉平各坐标曲率失配）。优化器谱系就是"SGD → +动量 → +自适应 → Adam（两者都要）"。

2. **SGD / Momentum / Nesterov**：裸 SGD $\theta_t=\theta_{t-1}-\eta g_t$；heavy-ball 动量 $v_t=\mu v_{t-1}+g_t,\ \theta_t=\theta_{t-1}-\eta v_t$（抑制震荡、加速一致方向）；Nesterov 在**前瞻点** $\theta-\eta\mu v$ 求梯度（look-ahead，纠过冲）。动量 $\approx$ 梯度的**指数加权平均**，窗口 $\sim 1/(1-\mu)$。

3. **AdaGrad → RMSProp → Adam**：AdaGrad 逐坐标步长 $\propto 1/\sqrt{\sum g^2}$（**累加全部历史** → LR 单调趋 0，长训练"学不动"）；RMSProp 把累加和换成 **EMA**（修好单调衰减）；Adam = RMSProp 的二阶 EMA + **一阶动量** + **偏差修正**。

4. **Adam 偏差修正**：$m,v$ 从 0 起步早期有偏（偏向 0），除以 $1-\beta^t$ 去偏。因 $\beta_2$（0.999）比 $\beta_1$（0.9）更接近 1，$v$ 偏得更重 → **不修正时早期步长偏大**（默认 $\beta$ 下首步 $\approx\frac{1-\beta_1}{\sqrt{1-\beta_2}}=3.16$ 倍 $\eta$），修正后首步回到 $\eta$。注意 $\epsilon$ 在根号**外**（PyTorch）。

5. **AdamW 解耦权重衰减**：**Adam 的 L2 正则 $\ne$ 权重衰减**。L2 把 $\lambda\theta$ 加进**梯度** → 经 $1/\sqrt{\hat v}$ 缩放 → 历史梯度大的参数被衰减得更少（解耦被破坏）。AdamW（Loshchilov & Hutter）**解耦**：Adam 步之外直接从权重减 $\eta\lambda\theta$。这是现代默认；SGD 下两者等价、Adam 下不等价。

6. **前沿优化器**：**Muon**（把 2D 权重的动量经牛顿-舒尔茨迭代**正交化**，仅隐藏 2D 层，Kimi-K2 在用）、**Lion**（**符号动量**，单状态 → Adam 一半显存）、**Shampoo**（Kronecker 因子化的**全矩阵预条件**）、**SOAP**（在 Shampoo 特征基里跑 Adam）、**Adafactor**（因子化二阶矩 → 亚线性显存）、**LAMB**（逐层自适应，超大 batch）、**Sophia**（轻量二阶，对角 Hessian）。

7. **LR schedule**：**warmup**（早期 $\hat v$ 方差大 + 大 batch/Post-LN 不稳 → 线性拉起）；**cosine**（平滑降到 ~0，+ warm restart，**需预定总步数**）；**inverse-sqrt / Noam**（原始 Transformer 调度）；**WSD**（warmup-stable-decay，恒定段 + 末段短衰减 → **不用预定总步数**、支持续训与中途 checkpoint）；**one-cycle**（super-convergence）。

8. **权重衰减 + LR-batch 缩放 + no-decay 组**：wd 做正则（近来也有"wd $\approx$ 有效 LR 控制器"的视角）；**线性缩放规则**（batch ×$k$ → LR ×$k$，SGD；Adam 常用 $\sqrt k$）；**LayerNorm/RMSNorm gain、bias、embedding 不加 wd**（分 decay/no-decay 两组）。

9. **梯度裁剪 + LLM 超参**：按**全局范数**裁（保方向）vs 按值裁（改方向），防梯度爆炸/loss spike；LLM 预训练默认 $\beta_2=\mathbf{0.95}$（**不是** 0.999，长记忆对尖峰太迟钝）、$\beta_1=0.9$、$\epsilon=\text{1e-8}$、wd $=0.1$、grad-clip $=1.0$。

## §1 为什么需要 SGD 之外的优化器

**裸随机梯度下降（SGD）的根本困境，是真实 loss 地形几乎总是"病态"的。** 把损失在一个极小点附近二阶展开，曲率由 Hessian $H$ 描述；定义**条件数** $\kappa=\lambda_{\max}/\lambda_{\min}$（$H$ 的最大与最小特征值之比）。$\kappa$ 大意味着不同方向的曲率差异巨大——某些方向陡峭如峡谷壁、某些方向平缓如谷底。对一个二次型，梯度下降的收敛速率约为 $\big(\tfrac{\kappa-1}{\kappa+1}\big)$，$\kappa$ 越大越慢。

直觉上，裸 SGD 在病态谷里会**来回震荡**：固定一个全局 $\eta$，为了不在陡方向（大 $\lambda$）发散，$\eta$ 必须取得很小（$\eta\lt 2/\lambda_{\max}$）；但这个小 $\eta$ 在缓方向（小 $\lambda$）上又**爬得极慢**。结果是在峡谷壁之间锯齿形（zig-zag）反复横跳，真正想前进的谷底方向却几乎不动。深网络的 loss 地形动辄 $\kappa\sim 10^4$ 以上，裸 SGD 因此既慢又不稳。

> 💡 **病态 = 各方向曲率不匹配，单一全局 LR 必然两头不讨好**
> 大曲率方向要小步（否则震荡/发散），小曲率方向要大步（否则爬不动）。一个标量 $\eta$ 没法同时满足 → 这是引入**动量**和**逐参数自适应**的根本动机。

两味互补的药，正好对应优化器谱系的两条轴：

- **动量（momentum）**：累积历史梯度的速度。震荡方向上梯度反复反号、在速度里**相互抵消**；一致方向上梯度同号、在速度里**累积加速**。等价于把更新方向做了平滑，把 GD 的收敛常数从 $\kappa$ 改善到 $\sqrt\kappa$ 量级（$\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$）。**注意**：$\kappa\to\sqrt\kappa$ 是**强凸二次型**上精调 heavy-ball/Nesterov 的经典加速结论；真实深网含随机梯度噪声、非凸、曲率含时，它是**直觉而非保证**。
- **逐参数自适应步长（adaptive / per-parameter LR）**：给每个坐标按它自己的梯度历史单独缩放步长——梯度一直很大的坐标用小步、一直很小的用大步。这相当于一个**对角预条件子**，把各坐标的有效曲率拉平，直接缓解 $\kappa$ 大的病。

**谱系一句话**：SGD（只用一阶梯度）→ + 动量（加一阶 EMA）→ + 自适应（加二阶矩 EMA）→ **Adam = 动量 + 自适应 + 偏差修正**。后面 §2–§5 沿这条线走，§6 是前沿（矩阵预条件 / 符号 / 省显存），§7–§9 是配套的 LR 调度、权重衰减、裁剪。

> ⚠️ **"自适应 = 对角预条件"，不是真二阶**
> Adam 这类只缩放每个坐标（对角），抓不到坐标间的耦合（off-diagonal 曲率）。真正逼近全矩阵预条件的是 Shampoo/SOAP（§6）。把 Adam 说成"近似牛顿法"不准确——它只是**对角**预条件，且用的是梯度二阶矩而非 Hessian。

## §2 SGD / Momentum / Nesterov

### 2.1　裸 SGD 与 heavy-ball 动量

裸 SGD 每步只看当前 mini-batch 梯度 $g_t$：

$$\theta_t = \theta_{t-1} - \eta\, g_t.$$

**heavy-ball 动量**（Polyak, 1964）引入一个速度缓冲 $v$，让更新带"惯性"：

$$\boxed{\;v_t = \mu\, v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\, v_t\;}\qquad (\mu\ \text{典型 }0.9).$$

（这是 PyTorch `SGD(momentum=μ)` 的形式：$v$ 初始为 0，首步 $v_1=g_1$，无 $(1-\mu)$ damping。）为什么有用：

- **震荡方向**（陡壁）：相邻步梯度反复反号，在 $v_t=\mu v_{t-1}+g_t$ 的指数加权和里**部分相消** → 横跳被压住。
- **一致方向**（谷底）：相邻步梯度同号、持续累积 → 速度变大、**加速前进**。

### 2.2　Nesterov 加速梯度（NAG）

heavy-ball 在**当前点** $\theta_{t-1}$ 求梯度。**Nesterov accelerated gradient** 改成在"动量将把你带去的"**前瞻点**求梯度：

$$v_t = \mu\, v_{t-1} + g\big(\theta_{t-1} - \eta\mu\, v_{t-1}\big), \qquad \theta_t = \theta_{t-1} - \eta\, v_t.$$

直觉："先看一眼动量要把我推到哪，再在那个点算梯度"——若前方要过冲，前瞻梯度会提前给出修正，所以 NAG 在凸问题上有更好的收敛常数。Sutskever et al.（ICML 2013）实证：**精心设计的（Nesterov 式）动量对训练深网络至关重要**，好的动量 + 初始化能让深网在没有自适应优化器时也训得起来。

### 2.3　动量 ≈ 梯度的指数加权平均

把递推展开（设 $v_0=0$）：

$$v_t = \sum_{i=1}^{t} \mu^{\,t-i}\, g_i.$$

这是历史梯度的**几何（指数）加权和**——越近的梯度权重越大，越远的按 $\mu^{t-i}$ 衰减。等效平均窗口约 $1/(1-\mu)$（$\mu=0.9$ → 约 10 步，$\mu=0.99$ → 约 100 步）。所以动量在做一件朴素的事：**把含噪的瞬时梯度平均成一个更平滑、更可信的下降方向**，顺带把一致趋势放大、把噪声/震荡抹掉。

> 💡 **动量的两副面孔**
> 优化视角：改善病态问题的收敛常数（$\kappa\to\sqrt\kappa$）。统计视角：对梯度做 EMA 去噪。两者都解释了它为什么几乎是免费的标配——$\mu=0.9$ 一加，又快又稳。

## §3 AdaGrad → RMSProp → Adam（自适应谱系）

### 3.1　AdaGrad：逐坐标自适应，但会"学到死"

AdaGrad（Duchi, Hazan & Singer, JMLR 2011）给每个坐标一个**独立**的、随历史梯度自适应的步长。维护逐坐标的**累加平方和** $G_t$：

$$G_t = G_{t-1} + g_t^2 \ (\text{逐元素}), \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t}+\epsilon}\odot g_t.$$

效果：梯度一直很大的坐标 $G$ 大 → 步长小；稀疏/罕见特征 $G$ 小 → 步长大。在凸优化、稀疏特征（NLP 早期、推荐）上很有效。**致命缺陷**：$G_t$ 是**单调递增**的累加和，分母 $\sqrt{G_t}$ 只增不减 → 有效学习率单调趋于 0。在深网络的长训练里，跑着跑着步长就缩到几乎为 0，**学不动了**（学习过早停滞）。

### 3.2　RMSProp：把累加和换成 EMA

RMSProp（Hinton, Coursera *Neural Networks for ML*, Lecture 6e, 2012，未正式发表）一针见血：问题出在"累加全部历史"。把累加和换成**指数移动平均**，让旧梯度被遗忘：

$$E[g^2]_t = \rho\, E[g^2]_{t-1} + (1-\rho)\, g_t^2, \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t}+\epsilon}\odot g_t.$$

EMA 的分母不再无界增长（它跟踪的是"最近的"梯度平方尺度），于是**有效 LR 不会衰减到 0**——AdaGrad 的死结被解开。这就是 Adam 二阶矩项的直接前身。

### 3.3　Adam = RMSProp + 动量 + 偏差修正

Adam（Kingma & Ba, 2014）把两条轴合一：RMSProp 的**二阶矩 EMA**（自适应步长）+ heavy-ball 的**一阶矩 EMA**（动量）+ 一个关键的**偏差修正**（§4）。一句话记忆：

$$\textbf{Adam} = \underbrace{\text{RMSProp}}_{\text{二阶矩 EMA}} + \underbrace{\text{momentum}}_{\text{一阶矩 EMA}} + \underbrace{\text{bias correction}}_{\text{去早期偏差}}.$$

> ✅ **每一步都在补前一步的洞**
> AdaGrad 给了"逐坐标自适应"但会死 → RMSProp 用 EMA 救活 → Adam 再叠动量去噪 + 偏差修正稳住早期。面试能把这条"谁修了谁"的链讲清，比单背 Adam 公式强得多。

## §4 Adam 深入

### 4.1　完整更新式

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t \qquad (\text{一阶矩 / 动量, EMA}),$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \qquad (\text{二阶矩 / 自适应, EMA}),$$

$$\hat m_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \qquad (\textbf{偏差修正}),$$

$$\boxed{\;\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\;}\qquad (\epsilon\ \text{在根号外，PyTorch}).$$

默认 $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$。$m_0=v_0=0$。

### 4.2　为什么需要偏差修正

$m,v$ 都从 0 初始化。早期它们是对真实矩的**有偏**估计——偏向 0，且 $\beta$ 越接近 1 偏得越久。对平稳梯度取期望：$\mathbb{E}[m_t]=(1-\beta_1^t)\,\mathbb{E}[g]$，所以 $\hat m_t=m_t/(1-\beta_1^t)$ 恰好去偏；$\hat v_t$ 同理。

关键、也是高频被深问的一点：**不修正时，早期等效步长不是偏小，而是偏大。** 看第一步（$t=1$）：

$$\frac{m_1}{\sqrt{v_1}} = \frac{(1-\beta_1)\,g_1}{\sqrt{(1-\beta_2)}\,\lvert g_1\rvert} = \frac{1-\beta_1}{\sqrt{1-\beta_2}}\,\mathrm{sign}(g_1).$$

代入默认值：$\frac{1-0.9}{\sqrt{1-0.999}}=\frac{0.1}{0.0316}\approx 3.16$。也就是说**不修正的首步约为目标 $\eta$ 的 3.16 倍**——因为 $v$（$\beta_2$ 更接近 1）被压向 0 比 $m$ 更狠，分母 $\sqrt{v}$ 太小，反而把步长放大了。偏差修正把 $\hat m_1=g_1,\ \hat v_1=g_1^2$ 还原，使首步严格回到 $\eta\cdot\mathrm{sign}(g_1)$，量级 $=\eta$。（§A 的 [d] 会数值印证这个 3.16。）

> ⚠️ **别顺口说"不修正步长太小"**
> 朴素直觉只盯分子 $m$ 偏向 0 → 以为步太小，**漏了分母 $v$ 偏得更狠**。对标准 $\beta_2=0.999$，净效果是早期步长**偏大**（$\approx 3.16\eta$），可能导致初期不稳/发散——这也是为什么早期还要配 warmup（§7）。把方向答反是顶级 lab 最爱抓的点。

> 💡 **$\epsilon$ 在根号外（PyTorch 约定）**
> Adam 的分母是 $\sqrt{\hat v_t}+\epsilon$，$\epsilon$ 在根号**外**——它既防除零，又**给步长设了上界** $\eta/\epsilon$（梯度极小时不至于炸出无穷大步）。有些实现（如 Adafactor）放在根号**内** $\sqrt{\hat v_t+\epsilon}$，数值行为略不同。$\epsilon$ 太小在 fp16/bf16 下分母 $\approx 0$ 易溢出。

## §5 AdamW（最高频的一节）

### 5.1　Adam 的 L2 正则 $\ne$ 权重衰减

经典 **L2 正则**的做法：在损失里加 $\frac{\lambda}{2}\lVert\theta\rVert^2$，等价于给梯度加一项 $\lambda\theta$：

$$g_t' = g_t + \lambda\,\theta_{t-1}.$$

对裸 SGD，这恰好就是权重衰减：$\theta_t=\theta_{t-1}-\eta(g_t+\lambda\theta_{t-1})=(1-\eta\lambda)\theta_{t-1}-\eta g_t$——每步把权重按 $(1-\eta\lambda)$ 收缩。**但对 Adam，灾难发生了**：这个 $\lambda\theta$ 被塞进 $g'$，于是它要一起进 $m,v$，最终被 $1/(\sqrt{\hat v}+\epsilon)$ 缩放：

$$\theta_t = \theta_{t-1} - \eta\,\frac{\widehat{m'_t}}{\sqrt{\hat v_t}+\epsilon},\qquad m'\ \text{含 }\lambda\theta\ \text{的贡献}.$$

后果：**历史梯度大的参数（$\hat v$ 大）拿到的有效衰减反而更小**，历史梯度小的参数被衰减更多——衰减强度被 $\hat v$ 扭曲，和"对所有权重施加均匀收缩"的正则本意完全脱节。这就是"L2 与 weight decay 在自适应优化器下不再等价"。

### 5.2　AdamW：解耦权重衰减

AdamW（Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 1711.05101）的修法极简——**把权重衰减从梯度里拿出来，绕过自适应分母，直接作用在权重上**：

$$\boxed{\;\theta_t = \theta_{t-1} - \eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Big)\;}$$

即先按原始梯度 $g$（**不含** $\lambda\theta$）算 Adam 步，再单独减一个 $\eta\lambda\theta_{t-1}$。等价写法 $\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$——一个干净的、**与 $\hat v$ 无关**的均匀收缩（PyTorch `AdamW` 即把衰减乘上 lr）。

为什么通常更好（更可控）：解耦后**所有参数受到同样的相对收缩 $\lambda$**，正则强度不再被各参数的梯度历史扭曲，更符合"控制权重范数"的本意；Loshchilov & Hutter 实证 AdamW 在多任务上**往往**稳定优于 Adam+L2，且 LR 与 wd 的最优值更容易解耦调参（不必绝对化成"所有任务都更泛化"，但它是更可控、更常用的默认）。**为何 SGD 下两者重合**：SGD 没有 $1/\sqrt{\hat v}$ 这层自适应缩放，$\lambda\theta$ 加进梯度后被原样使用，恰好就是均匀收缩；一旦有了自适应分母，加梯度（L2）和直接收缩（decay）就分道扬镳。

> ✅ **现代默认就是 AdamW**
> Transformer/LLM 训练几乎一律用 AdamW（不是 Adam）。面试问"Adam 和 AdamW 区别"，标准答案是"**L2 经过 $1/\sqrt{\hat v}$ 缩放、解耦衰减不经过**；SGD 等价、Adam 不等价"。只答"AdamW 加了权重衰减"是没抓到点（Adam 也能加 weight_decay，只不过那是 L2）。

> ⚠️ **PyTorch 里 `Adam(weight_decay=λ)` 是 L2，不是 AdamW**
> `torch.optim.Adam(weight_decay=λ)` 做的是**耦合 L2**（把 $\lambda\theta$ 加进梯度）；要解耦衰减必须用 `torch.optim.AdamW`。二者在相同 $\lambda$ 下产生**不同**更新（§A 的 [c] 会数值展示 $\lVert\Delta\rVert\gt 0$）。搞错这个 API 等于没用上 AdamW。

## §6 前沿优化器

每个只记**一个核心 idea**即可——面试报得出"它改了 Adam 的哪一块"就够。

### 6.1　Muon：把动量正交化

**Muon**（Keller Jordan, 2024 技术博客）针对**2D 权重矩阵**：先按动量算出更新矩阵 $M$，再用几次 **Newton-Schulz 迭代**把它**正交化**成 $O\approx\mathrm{NewtonSchulz}(M)$（数学上逼近 $M=U\Sigma V^\top$ 的 $UV^\top$，即最近的半正交矩阵，用迭代代替昂贵的 SVD），用 $O$ 去更新权重。直觉：正交化让更新在**所有奇异方向上同等用力**、不被少数大奇异值主导 → 更新更均衡、有效学习更快。**只用于隐藏层的 2D 矩阵**；embedding、输出头、norm gain、bias 这些仍走 AdamW（正交化对它们无意义）。Moonshot 的 Kimi-K2 用 Muon 训练；可扩展性见 *Muon is Scalable for LLM Training*（2502.16982）。

> 💡 **原始 Muon 多半是博客、无正式 arXiv**
> Muon 的首发是 Keller Jordan 的技术博客 writeup（无独立 arXiv 论文），有正式 arXiv 的是 Moonshot 的 scaling 工作（2502.16982）。引用时要区分，别给原始 Muon 编一个 arXiv id。

### 6.2　Lion：符号动量，单状态

**Lion**（Chen et al., *Symbolic Discovery of Optimization Algorithms*, 2302.06675）由**符号程序搜索**自动发现。更新取动量的**符号**：$u_t=\mathrm{sign}(\beta_1 m_{t-1}+(1-\beta_1)g_t)$，$\theta_t=\theta_{t-1}-\eta(u_t+\lambda\theta_{t-1})$，再更新动量 $m_t=\beta_2 m_{t-1}+(1-\beta_2)g_t$。**只存 1 个状态（动量）** → 优化器显存是 Adam（要存 $m,v$ 两个）的**一半**。sign 让每个坐标走等幅步长；通常配更小的 LR、更大的 wd。

### 6.3　Shampoo / SOAP：全矩阵预条件

- **Shampoo**（Gupta, Koren & Singer, 1802.09568）：超越 Adam 的**对角**预条件，做**全矩阵 / Kronecker 因子化**预条件。对一个权重矩阵维护左、右预条件子 $L=\sum GG^\top$、$R=\sum G^\top G$，用 $L^{-1/4} G\, R^{-1/4}$ 更新——以 Kronecker 积近似全矩阵 AdaGrad，捕捉坐标间耦合。收敛更快但每步贵（要算矩阵的逆根）。
- **SOAP**（Vyas et al., 2409.11321）：在 **Shampoo 的（缓变）特征基**里跑 Adam——把梯度旋到 Shampoo 预条件子的特征空间、在那里做标准 Adam，再旋回来。它把 Shampoo 与 Adam/Adafactor 联系起来，更稳、超参更少。

### 6.4　Adafactor / LAMB / Sophia

- **Adafactor**（Shazeer & Stern, 1804.04235）：把二阶矩矩阵 $v\in\mathbb{R}^{m\times n}$ **因子化**成行、列两个向量（秩-1 重构 $v\approx rc^\top$）→ 显存从 $O(mn)$ 降到 $O(m+n)$（**亚线性**）。T5 用它在有限显存上训大模型。
- **LAMB**（You et al., 1904.00962）：在 Adam 更新方向上再做**逐层自适应**——按每层的 $\lVert\theta\rVert/\lVert\text{update}\rVert$（trust ratio）缩放（LARS 的 Adam 版）。让**超大 batch** 训练稳定（76 分钟训 BERT）。
- **Sophia**（Liu et al., 2305.14342）：**轻量二阶**。用对角 Hessian 估计（Gauss-Newton-Bartlett 或 Hutchinson）做预条件 + 对更新裁剪，每 $k$ 步才更新一次 Hessian → 比 AdamW 更快收敛，二阶开销摊薄。

## §7 学习率调度

### 7.1　Warmup：为什么开头要慢

**warmup** 在训练最初的几百~几千步把 LR 从 0（或很小）**线性拉到峰值**，再交给后续衰减。动机有二：

1. **Adam 早期方差大**：$\hat v$ 在前几步只用了极少样本估计，**方差很大**，导致自适应步长抖动甚至偏大（§4.2 的 3.16 倍效应）。先用小 LR 让 $m,v$ 的统计"热身"到可靠，再放大步长。
2. **大 batch / Post-LN 不稳**：大 batch 初期梯度尺度大；Post-LN Transformer 顶层梯度天然偏大（见 normalization 篇 §5）——一上来就大 LR 容易发散。warmup 压住早期不稳。

### 7.2　Cosine 退火（+ warm restart）

cosine（Loshchilov & Hutter, *SGDR*, 1608.03983）把 LR 从峰值**平滑余弦衰减**到 $\sim\eta_{\min}$：

$$\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\Big(1+\cos\big(\pi\, t/T\big)\Big).$$

早期大 LR 快速探索、后期小 LR 精细收敛（噪声小、易落入更平的极小）。SGDR 还加 **warm restart**：周期性把 LR 拉回峰值，帮助跳出局部解。**代价：必须预先确定总步数 $T$**（$\cos$ 要知道终点）——这正是 WSD 想解决的痛点。

### 7.3　Inverse-sqrt / Noam（原始 Transformer 调度）

原始 Transformer（Vaswani et al., 1706.03762）用的调度，先**线性 warmup** 再 **$1/\sqrt t$ 衰减**：

$$\eta_t = d_{\text{model}}^{-1/2}\cdot\min\!\Big(t^{-1/2},\ t\cdot t_{\text{warmup}}^{-3/2}\Big).$$

$t\lt t_{\text{warmup}}$ 时取后项（线性上升），之后取前项（按 $t^{-1/2}$ 下降）。无需预定总步数，是早期机器翻译/语言模型的标配。

### 7.4　WSD：Warmup-Stable-Decay

WSD（MiniCPM, Hu et al., 2404.06395）把调度切成三段：**warmup → 长时间恒定（stable，峰值 LR）→ 末段短促 decay**。最大卖点：**不用预先 commit 总步数**——恒定段可以无限延长，想停的时候才接一段短 decay（loss 在 decay 段会陡降到一个好水平）。好处：

- 支持**持续训练 / 中途加数据**（恒定段随时续）；
- 每做一次 decay 就得到一个**可用的 checkpoint**，不必为每个目标步数各训一条余弦；
- 便于在固定算力下探索"训多久"，而不是一次性赌死 $T$。

### 7.5　One-cycle / super-convergence

one-cycle（Smith, *Super-Convergence*, 1708.07120）：整个训练就**一个周期**——LR 先升到一个**很大**的峰值再降到远低于初值，动量则反向同步循环（LR 高时动量低）。在某些设置下能"超收敛"（用很大的 LR 极快训完）。它和 cosine、WSD 一样是"先升后降"的家族，区别在单周期 + 超大峰值 LR。

> 🎯 **选调度的一句话**
> 固定预算、追极致质量 → **cosine**（但要预定步数）；不确定要训多久 / 要续训 / 要中途出 checkpoint → **WSD**；经典 Transformer 复现 → **Noam**；想用超大 LR 极速训 → **one-cycle**。所有这些**前面都应接 warmup**。

## §8 权重衰减 + LR-batch 缩放 + no-decay 组

### 8.1　权重衰减做什么（含一个现代视角）

经典视角：权重衰减是**正则**——每步把权重朝 0 收缩，限制权重范数、压模型复杂度、抗过拟合。**现代视角**（在有归一化层的网络里）：因为 LayerNorm/RMSNorm 让权重**尺度不变**（把 $W$ 乘个常数，归一化后输出不变），权重的"绝对大小"本身不直接影响函数，于是 wd 的真正作用更像是在**调有效学习率 / 控制权重范数的平衡点**——wd 越大、权重范数越小、等效梯度步长相对越大。这是近年训练动力学研究里的重要观察（把 wd 当"有效 LR 旋钮"而非单纯正则）。

### 8.2　LR 随 batch size 缩放

大 batch 训练里，LR 必须跟着 batch 调，否则白白浪费算力或不稳：

- **线性缩放规则**（Goyal et al., 1706.02677）：**batch ×$k$ → LR ×$k$**（用于大 batch SGD），并配 warmup。直觉：mini-batch 均值梯度的方差 $\propto 1/B$，把 batch 放大 $k$ 倍后噪声降低，可以放心把步长也放大 $k$ 倍以保持"每个样本贡献的有效更新"大致不变。
- **Adam 的 $\sqrt k$ 缩放**：Adam 的更新近似被 $\sqrt{\hat v}$ 归一化（接近符号化），经验上对 batch 用**平方根**缩放（LR ×$\sqrt k$）更稳，而非线性。
- 相关地，**LARS**（You et al., 1708.03888）/ **LAMB**（§6.4）用逐层 trust ratio 实现超大 batch 下的自适应缩放。

> ⚠️ **这些都是经验法则，不是定理**
> 线性 / $\sqrt{}$ 缩放只在一定 batch 范围内成立；batch 大到一定程度会进入"收益递减"区（critical batch size 之外，再加 batch 不再线性提速）。面试讲缩放规则要带上"近似 / 有上限"的限定。

### 8.3　no-decay 参数组

**不是所有参数都该加权重衰减。** 标准实践把参数分成两组：

- **加 wd**：各线性层 / 卷积的权重矩阵。
- **不加 wd（no-decay）**：**LayerNorm/RMSNorm 的 gain $\gamma$、所有 bias、（通常）embedding**。

原因：这些是**尺度 / 平移参数**，把它们朝 0 收缩会直接损害表示——把 LN 的 gain 压向 0 等于抹掉该层激活的尺度；bias 本就不该被正则（它只管平移）。GPT/LLaMA 等代码库都显式构造 decay / no-decay 两个 param group。

> 💡 **与 μP 的连接（不在此重复推导）**
> μP（maximal update parametrization）让**最优 LR 在宽度上不变**，从而小宽度调好的超参可 zero-shot 迁移到大模型——这是"如何选 LR"的另一条正交线索，细节见 normalization 篇的 μP 一节（§9），此处不再展开。

## §9 梯度裁剪 + LLM 超参细节

### 9.1　梯度裁剪：全局范数 vs 按值

训练里偶发的**梯度爆炸 / loss 尖峰**（尤其 RNN、长序列、大 LR）会一步把权重带飞甚至 NaN。梯度裁剪（Pascanu, Mikolov & Bengio, 1211.5063）给梯度设一个闸：

- **按全局范数裁（clip by global norm，最常用）**：把所有参数的梯度拼成一个大向量，算其 $\lVert g\rVert_2$；若超过阈值 $c$，整体缩放 $g\leftarrow g\cdot c/\lVert g\rVert_2$。**保持方向**、只压幅度。
- **按值裁（clip by value）**：逐分量 clamp 到 $[-v, v]$。会**改变梯度方向**（不同分量被不同程度截断），一般不如全局范数。

为什么稳：偶发的超大梯度被钳到阈值内，单步更新幅度有界 → 不会因一个坏 batch 把训练带崩。LLM 预训练几乎都用 **global-norm clip，阈值 1.0**。

### 9.2　LLM 训练的 Adam 超参默认

LLM 预训练的 Adam/AdamW 配方和"教科书默认"有几处关键不同：

| 超参 | 教科书默认 | **LLM 预训练常用** | 为什么 |
| --- | --- | --- | --- |
| $\beta_1$ | 0.9 | 0.9 | 动量窗口 ~10 步，够用 |
| $\beta_2$ | 0.999 | **0.95**（或 0.99） | 0.999 记忆窗口 ~1000 步**太长**，对 loss spike / 非平稳反应迟钝、易在尖峰后失稳；0.95（窗口 ~20 步）反应更快更稳 |
| $\epsilon$ | 1e-8 | 1e-8（有时 1e-15） | 防分母为 0；过大会削弱自适应 |
| weight decay | 0 ~ 1e-2 | **0.1** | 解耦 wd（AdamW），较强 |
| grad clip | 无 | **1.0**（global norm） | 压 loss 尖峰 |

> ⚠️ **$\beta_2=0.999$ 搬到 LLM 上是经典坑**
> 视觉/小模型上 0.999 没问题；但 LLM 大规模预训练里梯度统计非平稳、偶有尖峰，0.999 的长记忆让 $\hat v$ 更新太慢，遇到 loss spike 难以快速回稳，常见做法是降到 **$\beta_2=0.95$**。这是面试里"你怎么调 LLM 的 Adam"的标准得分点。

## §10 选优化器 + 显存账

### 10.1　优化器状态的显存

混合精度训练里，每个**可训练参数**的显存大致是：bf16 权重 2 B + bf16 梯度 2 B + fp32 master 4 B + **优化器状态**。优化器状态正是各方法的分水岭：

- **SGD+momentum**：1 个状态（动量 $v$）。
- **Adam / AdamW**：**2 个状态（$m, v$）**，fp32 下 $=8$ B/param——这往往是大头，是"省显存"动机的来源。
- **Lion**：1 个状态（动量）→ Adam 的一半。
- **Adafactor**：因子化 $v$ → 亚线性，近似只剩"参数级"开销。

由此催生的省显存路线：

- **8-bit Adam**（Dettmers et al., 2110.02861）：把 $m,v$ **分块量化到 8-bit** 存储（用时反量化），优化器状态显存降到约 $1/4$，质量几乎无损。
- **Adafactor**：因子化二阶矩，亚线性显存（T5）。
- **Lion**：少存一个状态，直接减半。

### 10.2　什么时候用哪个

> 🎯 **"用哪个优化器"决策树**
> - **Transformer / LLM 微调或预训练**：**AdamW**（安全默认，$\beta_2=0.95$、wd 0.1、grad-clip 1.0、配 warmup+cosine/WSD）。
> - **经典视觉（ResNet 等）**：**SGD+momentum** 常**泛化更好**、且省显存——CV 里仍有大量 SOTA 用它。
> - **想要额外训练加速 / 大规模**：**Muon**（2D 隐藏层）、**Shampoo/SOAP**（全矩阵预条件，肯花每步算力）。
> - **显存受限**：**8-bit Adam** / **Adafactor** / **Lion**。
> - **超大 batch**：**LAMB / LARS**（逐层 trust ratio）。

## §11 工程对比 + 常见误区

### 11.1　优化器对比表

| 优化器 | 优化器状态（× 参数） | 额外计算 | 核心 idea | 典型用途 |
| --- | --- | --- | --- | --- |
| **SGD+Momentum** | 1×（动量） | 极低 | heavy-ball 速度 EMA | 经典视觉、追泛化 |
| **Adam / AdamW** | 2×（$m,v$） | 低（逐元素） | 动量 + 对角自适应（+ 解耦 wd） | Transformer / LLM 默认 |
| **Lion** | 1×（动量） | 低（取 sign） | 符号动量、单状态 | 省显存 / 大 batch |
| **Muon** | 1×（动量）+ NS 迭代 | 中（几次小 matmul） | 动量正交化（Newton-Schulz） | LLM 隐藏 2D 层提速 |
| **Shampoo** | 预条件子 $L,R$（较大） | 高（矩阵逆根） | Kronecker 全矩阵预条件 | 大规模训练提速 |
| **Adafactor** | 亚线性（因子化 $v$） | 低 | 行/列因子化二阶矩 | 显存受限 / T5 |

> 💡 **读表**：横看显存（SGD/Lion/Muon 1 状态、Adam 2 状态、Adafactor 亚线性、Shampoo 最重），竖看代价（对角自适应便宜、全矩阵预条件贵）。没有"最优优化器"，只有"对当前 模型 × 显存 × 算力 预算最合适的"。

### 11.2　常见误区（footguns）

> ❌ **误区 1：Adam 的 L2 当成 AdamW 的解耦衰减**
> `Adam(weight_decay=λ)` 是耦合 L2（经 $1/\sqrt{\hat v}$ 缩放），`AdamW` 才是解耦衰减。相同 $\lambda$ 下两者更新**不同**（§5、§A [c]）。

> ❌ **误区 2：忘了偏差修正 / 把它的方向讲反**
> 漏掉 $1/(1-\beta^t)$ 会让早期步长错乱；且不修正时默认 $\beta$ 下首步是**偏大**（$\approx 3.16\eta$）不是偏小（§4.2）。

> ❌ **误区 3：给 LayerNorm/RMSNorm gain、bias 加权重衰减**
> 这些尺度/平移参数不该被收缩；要分 no-decay 组（§8.3）。

> ❌ **误区 4：LLM 训练用 $\beta_2=0.999$**
> 长记忆对 loss 尖峰太迟钝，易失稳；改 **0.95**（§9.2）。

> ❌ **误区 5：大 batch 不调 LR**
> 要按线性（SGD）/ $\sqrt{}$（Adam）缩放并配 warmup，否则浪费算力或发散（§8.2）。

> ❌ **误区 6：用 cosine 却不知要预定总步数**
> cosine 需要终点 $T$；想中途续训 / 不确定步数应用 **WSD**（§7.2、§7.4）。

> ❌ **误区 7：把 Adam 的 $\epsilon$ 放进根号里**
> PyTorch 是 $\sqrt{\hat v}+\epsilon$（根号外，兼作步长上界）；位置记反会改变数值行为（§4.2）。

## §12 25 高频面试题

按难度分三档，点开看答案要点 + 易踩坑。L2/L3 是顶级 lab 深水区（AdamW 解耦、偏差修正方向、$\beta_2$ 选择、缩放规则、Muon/Shampoo/Lion/Sophia 等）。

### L1必会题

<details>

<summary>Q1. 裸 SGD 和带动量的 SGD 有什么区别？动量做什么？</summary>

- 裸 SGD：$\theta\mathrel{-}=\eta g$，每步只看当前梯度
- 动量：$v=\mu v+g,\ \theta\mathrel{-}=\eta v$，累积历史梯度的（指数加权）速度
- 一致方向累积加速、震荡方向反号相消；等效平均窗口 $\sim 1/(1-\mu)$

只把动量说成"加大学习率"，讲不出"沿一致方向累积、震荡方向相消"。

</details>

<details>

<summary>Q2. 写出 Adam 的完整更新式。</summary>

- $m=\beta_1 m+(1-\beta_1)g$（一阶 EMA），$v=\beta_2 v+(1-\beta_2)g^2$（二阶 EMA）
- 偏差修正 $\hat m=m/(1-\beta_1^t)$，$\hat v=v/(1-\beta_2^t)$
- $\theta\mathrel{-}=\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$，$\epsilon$ 在根号**外**（PyTorch）

漏掉偏差修正；或把 $\epsilon$ 放进根号里。

</details>

<details>

<summary>Q3. warmup 是干嘛的？</summary>

- 训练初期把 LR 从 0/很小**线性拉到峰值**（几百~几千步）
- 早期 Adam 的 $\hat v$ 用样本少、方差大 → 自适应步长不稳/偏大；大 batch、Post-LN 初期梯度大
- warmup 压住早期不稳，避免一上来就发散

把 warmup 当玄学，讲不出"早期方差大 / 梯度大"这个机理。

</details>

<details>

<summary>Q4. AdaGrad 是什么？为什么长训练会"学不动"？</summary>

- 每坐标累加**全部**历史梯度平方 $G=\sum g^2$，步长 $\propto\eta/\sqrt{G}$
- $G$ 单调增 → 有效 LR 单调趋 0 → 长训练后步长几乎为 0，停滞
- 适合凸 / 稀疏特征，不适合深网长训练

不知道是"累加全部历史"导致 LR 衰减到 0。

</details>

<details>

<summary>Q5. RMSProp 修了 AdaGrad 的什么？</summary>

- 把"累加和"换成 **EMA**：$E[g^2]=\rho E[g^2]+(1-\rho)g^2$
- EMA 会遗忘旧梯度 → 分母不再无界增长 → LR 不衰减到 0
- 这就是 Adam 二阶矩项的来源

只说"RMSProp 更快"，讲不出"EMA 替代累加和、修好 LR 衰减"。

</details>

<details>

<summary>Q6. 为什么常用 Adam 而不是裸 SGD？</summary>

- 每参数自适应步长（按 $\sqrt{\hat v}$ 做**对角预条件**）+ 动量，对病态 / 各坐标尺度不一更鲁棒
- 收敛快、对 LR 不那么敏感、几乎免调，适合 Transformer
- 代价：2× 优化器状态显存、有时泛化略逊 SGD

只说"Adam 收敛快"，答不出"自适应 = 对角预条件"；或把它说成"近似牛顿法"（它只是对角）。

</details>

<details>

<summary>Q7. 权重衰减 / L2 正则直觉上做什么？对 SGD 和 Adam 一样吗？</summary>

- 每步把权重朝 0 收缩 → 限制权重范数、控制复杂度
- **SGD**：L2 与 weight decay 等价（$\theta\mathrel{-}=\eta(g+\lambda\theta)=(1-\eta\lambda)\theta-\eta g$）
- **Adam**：两者**不等价**（L2 经 $1/\sqrt{\hat v}$ 缩放），要用 AdamW，见 Q11

以为 L2 和 weight decay 永远一回事（对自适应优化器不成立）。

</details>

<details>

<summary>Q8. 余弦退火是什么？为什么要衰减 LR？</summary>

- $\eta_t=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi t/T))$，从峰值平滑降到 $\sim 0$
- 早期大 LR 快速探索，后期小 LR 精细收敛、减小噪声、利于落入更平的极小
- SGDR 加 warm restart（周期性拉回峰值）

不知道要**预先给定总步数 $T$**；或答不出"为何要 decay"。

</details>

<details>

<summary>Q9. Nesterov 和 heavy-ball 动量的区别？</summary>

- heavy-ball：在**当前点** $\theta$ 算梯度，$v=\mu v+g(\theta)$
- Nesterov：在"动量将带你去的"**前瞻点** $\theta-\eta\mu v$ 算梯度（look-ahead）
- 前瞻能提前看到过冲并纠正，收敛常数更好（Sutskever 2013 强调对深网重要）

说不清 NAG 的梯度是在前瞻点而非当前点求的。

</details>

<details>

<summary>Q10. 为什么说动量 ≈ 梯度的 EMA？</summary>

- 展开 $v_t=\sum_{i\le t}\mu^{t-i}g_i$ —— 历史梯度的指数加权（几何）和
- 等效是一个窗口 $\sim 1/(1-\mu)$ 的滑动平均（$\mu=0.9\to\sim 10$ 步）
- 所以动量在"平均掉噪声、保留一致趋势"

只把动量当"惯性"比喻，写不出指数加权求和这个式子。

</details>

### L2进阶题

<details>

<summary>Q11. AdamW vs Adam+L2：解耦到底解耦了什么？（最高频）</summary>

- Adam+L2：把 $\lambda\theta$ 加进**梯度** → 经 $m,v$ 的 $1/\sqrt{\hat v}$ 缩放 → 历史梯度大的参数被衰减得**更少**（解耦被破坏）
- AdamW：Adam 步之外**直接**从权重减 $\eta\lambda\theta$（不过自适应分母）：$\theta\mathrel{-}=\eta(\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta)$
- 解耦后所有参数受同样收缩，泛化更好；SGD 下两者等价、Adam 下不等价

说"AdamW 就是 Adam 加权重衰减"，讲不出"L2 经过 $\sqrt{\hat v}$ 缩放"这个关键差异。

</details>

<details>

<summary>Q12. Adam 为什么要偏差修正？不修会怎样？方向是偏大还是偏小？</summary>

- $m,v$ 初值 0，早期对真实矩有偏（偏向 0），$\beta$ 越接近 1 偏得越久
- 因 $\beta_2$(0.999) 比 $\beta_1$(0.9) 更接近 1，$v$ 偏得更重 → 分母 $\sqrt{v}$ 太小 → 不修正时早期步长**偏大**（默认 $\beta$ 下首步 $\approx\frac{1-\beta_1}{\sqrt{1-\beta_2}}=3.16$ 倍 $\eta$）
- 修正 $\div(1-\beta^t)$ 把首步拉回 $\eta$

顺口说"不修正步长太小"——对标准 $\beta_2=0.999$ 恰恰相反，是**偏大**（只盯分子漏了分母）。

</details>

<details>

<summary>Q13. 余弦 vs WSD，各自取舍？</summary>

- 余弦：平滑降到 $\sim 0$，质量好，但**必须预先确定总步数 $T$**，难做中途续训 / 加数据
- WSD：warmup → 长恒定 → 末段短 decay；**不用预定总步数**，恒定段可任意延长，每次 decay 出一个可用 checkpoint
- WSD 适合持续训练 / 预算不确定；余弦适合固定预算的一次性训练

不知道余弦要预定 $T$，而 WSD 的卖点正是不用预定。

</details>

<details>

<summary>Q14. 为什么 LLM 预训练用 $\beta_2=0.95$ 而不是 0.999？</summary>

- $\beta_2=0.999$ 的有效记忆窗口 $\sim 1/(1-\beta_2)=1000$ 步，太长、对 loss spike 反应迟钝
- LLM 训练梯度非平稳 / 有尖峰，0.95（窗口 $\sim 20$ 步）反应更快、更稳
- 配 $\beta_1=0.9,\ \epsilon=\text{1e-8},\ \text{wd}=0.1,\ \text{grad-clip}=1.0$ 是常见 LLM 配方

套视觉 / 小模型的 0.999，遇到 loss spike 难恢复。

</details>

<details>

<summary>Q15. 线性缩放规则是什么？为什么 Adam 常用 $\sqrt{}$ 缩放？</summary>

- Goyal：大 batch SGD，batch ×$k$ 则 LR ×$k$（保持每样本有效更新），配 warmup
- 直觉：mini-batch 均值梯度方差 $\propto 1/B$，线性放大 LR 维持 SNR / 步长
- Adam 更新近似被 $\sqrt{\hat v}$ 归一化 → 经验上用 $\sqrt k$（平方根）缩放更稳

大 batch 不调 LR（浪费算力）；或对 Adam 也硬套线性 $k$；忘了 critical batch size 之外收益递减。

</details>

<details>

<summary>Q16. 梯度裁剪：按全局范数 vs 按值，为什么能稳？</summary>

- 全局范数：$\lVert g\rVert\gt c$ 时整体缩放 $g\leftarrow g\cdot c/\lVert g\rVert$，**保方向**只压幅度
- 按值：逐分量 clamp 到 $[-v,v]$，会**改方向**
- 裁掉偶发的梯度爆炸 / 尖峰 → 防 loss 飞、防 NaN（Pascanu 2013）；LLM 常用 global-norm 阈值 1.0

说"裁剪就行"，分不清全局范数（保方向）和按值（改方向）。

</details>

<details>

<summary>Q17. no-decay 参数组：哪些参数不加权重衰减？为什么？</summary>

- LayerNorm/RMSNorm 的 gain $\gamma$、所有 bias、（常）embedding 不加 wd
- 它们是尺度 / 平移参数，朝 0 收缩会压垮表示（如 LN gain→0 抹掉激活尺度）
- 标准做法：把参数分成 decay / no-decay 两组（GPT/LLaMA 代码皆如此）

对所有参数无脑统一 wd，把 norm gain / bias 也衰减了。

</details>

<details>

<summary>Q18. AdaGrad → RMSProp → Adam，每一步修了前者什么？</summary>

- AdaGrad：累加**全部** $g^2$ → LR 单调趋 0（长训练死）
- RMSProp：换成 **EMA** → 修好 LR 衰减（不死）
- Adam：RMSProp + **一阶动量** + **偏差修正** → 又快又稳

背得出三个名字，连不成"各修了前者哪个毛病"的链条。

</details>

<details>

<summary>Q19. Adam 里 $\epsilon$ 放哪、起什么作用？</summary>

- PyTorch：$\theta\mathrel{-}=\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$，$\epsilon$ 在根号**外**
- 作用：防分母为 0，**且给步长设上界** $\eta/\epsilon$（梯度极小时不炸）
- 有些实现 / Adafactor 放根号内 $\sqrt{\hat v+\epsilon}$，数值行为略不同；LLM 常 1e-8，有时 1e-15

把 $\epsilon$ 位置记反；或以为它只"防除零"，忽略它会限制最大步长。

</details>

<details>

<summary>Q20. 为什么 Adam 有时泛化不如 SGD？Adam 有收敛性问题吗？</summary>

- 自适应方法倾向收敛到不同（有时更"尖"）的极小，某些视觉任务泛化逊于 SGD+M
- **AMSGrad**（Reddi 2018）给出 Adam 不收敛的反例：$v$ 的 EMA 会让有效 LR 偶尔回升、破坏收敛；修法是取 $v$ 的历史最大值
- 实务：Transformer 仍首选 AdamW；CV 经典网络 SGD+M 常更好

绝对化"Adam 一定最好"；不知道 Adam 有收敛反例（AMSGrad）。

</details>

### L3高级题

<details>

<summary>Q21. Muon 是什么？为什么把动量正交化、为什么只用于 2D 层？</summary>

- 对 2D 权重的动量 $M$ 做正交化 $O\approx\mathrm{NewtonSchulz}(M)$（$\approx$ SVD 的 $UV^\top$，几次牛顿-舒尔茨迭代代替 SVD）
- 直觉：让更新在所有奇异方向上"同等用力"、不被少数大奇异值主导 → 更新更均衡、训练更快
- 仅隐藏 2D 矩阵；embedding / 输出头 / 标量仍用 AdamW；Kimi-K2 在用；原始 Muon 是博客（无独立 arXiv）

说"Muon 给所有参数正交化"（错，仅 2D 隐藏层）；或不知道用牛顿-舒尔茨代替 SVD。

</details>

<details>

<summary>Q22. Shampoo 和 SOAP 在做什么？和 Adam 的"自适应"有何本质不同？</summary>

- Shampoo：**全矩阵 / Kronecker 因子化**预条件——维护左 / 右预条件子 $L=\sum GG^\top,\ R=\sum G^\top G$，用 $L^{-1/4}GR^{-1/4}$ 更新
- 比 Adam 的**对角**预条件更接近全矩阵 AdaGrad（抓坐标耦合），收敛快但每步贵（矩阵逆根）
- SOAP：在 Shampoo 的（缓变）**特征基**里跑 Adam → 连接 Shampoo 与 Adam，更稳更省调参

把 Shampoo 当成"另一个对角自适应"，没抓住"全矩阵 / Kronecker 预条件"。

</details>

<details>

<summary>Q23. Lion 的更新和内存？它怎么来的？</summary>

- Lion = sign of momentum：$u=\mathrm{sign}(\beta_1 m+(1-\beta_1)g)$，$\theta\mathrel{-}=\eta(u+\lambda\theta)$，再 $m=\beta_2 m+(1-\beta_2)g$
- 只存 **1 个状态（动量）** → 优化器显存是 Adam（$m,v$ 两个）的**一半**
- 由符号回归 / 程序搜索自动发现（Chen 2023）；sign 让各坐标等幅更新（通常配更小 LR、更大 wd）

以为 Lion 也存二阶矩；或不知道它是"符号动量、单状态"。

</details>

<details>

<summary>Q24. 显存受限时怎么省优化器状态？</summary>

- Adam 存 $m,v=2\times$ 参数（fp32 $\sim 8$ B/param），加 fp32 master 是大头
- **8-bit Adam**（Dettmers 2021）：把 $m,v$ 分块量化到 8-bit → 优化器态减到约 $1/4$
- **Adafactor**：把二阶矩 $v$ 行 / 列因子化（秩-1 重构）→ 亚线性显存（T5）；**Lion**：1 状态减半

只知道"用 Adam"，被问显存受限怎么办答不出 8-bit / Adafactor / Lion。

</details>

<details>

<summary>Q25. 讲几个前沿 / 理论点：Sophia、权重衰减作为有效 LR、μP 的 LR 迁移。</summary>

- **Sophia**（Liu 2023）：轻量二阶，用对角 Hessian 估计做预条件 + 裁剪，每 $k$ 步更新一次 Hessian → 比 AdamW 更快
- **权重衰减新视角**：在有归一化层（尺度不变）的网络里，wd 主要在调"有效学习率 / 权重范数平衡点"，而非传统正则
- **μP**：把 init / LR / 输出乘子按 fan_in 缩放 → 最优 LR 宽度不变，小模型调好 zero-shot 迁移到大模型（详见 normalization 篇 §9，不再推导）

把这些当玄学；尤其不知道"wd $\approx$ 有效 LR 控制器"和"μP 让 LR 可跨宽度迁移"。

</details>

## §A 附录：sanity check

本 tutorial 的从零实现应满足以下关键不变量（可写一段纯 PyTorch、CPU 几秒的脚本验证）：

1. **[a] 从零 SGD+momentum == `torch.optim.SGD`**：用 $v_t=\mu v_{t-1}+g_t,\ \theta\mathrel{-}=\eta v_t$（PyTorch 约定：$v$ 初值 0、无 damping），跑几步后两者逐元素在浮点误差内相等（`atol≈1e-5`）。
2. **[b] 从零 Adam == `torch.optim.Adam`**：$\hat m=m/(1-\beta_1^t),\ \hat v=v/(1-\beta_2^t),\ \theta\mathrel{-}=\eta\hat m/(\sqrt{\hat v}+\epsilon)$（$\epsilon$ 根号外）应与 PyTorch 逐元素一致。
3. **[c] AdamW $\ne$ Adam+L2，且从零解耦衰减 == `torch.optim.AdamW`**：相同 `weight_decay=λ` 下，`torch.optim.AdamW`（解耦）与 `torch.optim.Adam(weight_decay=λ)`（L2）产生**不同**更新（$\lVert\Delta\rVert\gt 0$）；从零"先 Adam 步、再减 $\eta\lambda\theta$"的解耦实现应与 `AdamW` 一致。这是 §5 的核心。
4. **[d] 偏差修正使首步回到 $\eta$**：$t=1$ **不修正**的等效步长 $\approx 3.16\eta$（$=\frac{1-\beta_1}{\sqrt{1-\beta_2}}$，因 $v$ 被 $\beta_2$ 压向 0 比 $m$ 更狠 → 分母太小 → 步长**偏大**），修正后回到 $\eta$。验证 corrected/uncorrected 比 $=\frac{\sqrt{1-\beta_2}}{1-\beta_1}\approx 0.316$。
5. **[e] cosine-with-warmup 调度形状**：$\eta(0)\approx 0$、在 warmup 边界处取到峰值 $\eta_{\max}$、末端衰减到 $\sim\eta_{\min}$，且 warmup 之后单调不增。
6. **[f] 动量加速病态二次**：在条件数 $\kappa$ 大的二次型上、用稳定边界内的**小** $\eta$ 跑 $N$ 步，GD+momentum 的最终 loss 显著低于裸 GD（动量在缓方向累积加速）。

下面是几段示意代码（CPU 可跑、Chinese 注释、含形状/数值注释；演示其中几条不变量，完整 6 条 [a]–[f] 见下方独立脚本）。

**[a]/[b] SGD+momentum 与 Adam 从零实现，并与 `torch.optim` 对齐：**

```python
import torch

def sgd_momentum_from_scratch(params, grads, lr=0.1, mu=0.9, steps=5):
    """heavy-ball: v = mu*v + g; theta -= lr*v。v 初值 0（首步 v=g），对齐 PyTorch SGD。"""
    theta = params.clone()                      # [d] 参数
    v = torch.zeros_like(theta)                 # [d] 动量缓冲
    for t in range(steps):
        g = grads[t]                            # [d] 第 t 步梯度（这里用固定序列便于对齐）
        v = mu * v + g                          # 速度 = 动量*旧速度 + 当前梯度
        theta = theta - lr * v                  # 沿速度方向更新
    return theta

def adam_from_scratch(grads, theta0, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, steps=5):
    """m,v 的 EMA + 偏差修正 + eps 在根号外，逐元素对齐 PyTorch Adam。"""
    theta = theta0.clone()
    m = torch.zeros_like(theta)                 # 一阶矩 [d]
    v = torch.zeros_like(theta)                 # 二阶矩 [d]
    for t in range(1, steps + 1):
        g = grads[t - 1]                        # [d]
        m = b1 * m + (1 - b1) * g               # 动量 EMA
        v = b2 * v + (1 - b2) * g * g           # 平方梯度 EMA
        m_hat = m / (1 - b1 ** t)               # 偏差修正（去早期偏向 0）
        v_hat = v / (1 - b2 ** t)
        theta = theta - lr * m_hat / (v_hat.sqrt() + eps)   # eps 在根号外
    return theta

# 与 torch.optim 对齐（同一固定梯度序列喂给两边）：
torch.manual_seed(0)
d, steps = 8, 5
theta0 = torch.randn(d)
grads = [torch.randn(d) for _ in range(steps)]   # 固定梯度序列，保证可比

# --- SGD ---
p = theta0.clone().requires_grad_(True)
opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
for t in range(steps):
    opt.zero_grad(); p.grad = grads[t].clone(); opt.step()
ref_sgd = p.detach()
mine_sgd = sgd_momentum_from_scratch(theta0, grads, lr=0.1, mu=0.9, steps=steps)
assert torch.allclose(mine_sgd, ref_sgd, atol=1e-6)    # 从零 == torch.optim.SGD
```

**[d] 偏差修正：不修正首步偏大（默认 $\beta$ 下 $\approx 3.16\eta$），修正后 $=\eta$：**

```python
import torch

b1, b2, eps = 0.9, 0.999, 1e-8
g1 = torch.tensor([2.0])                          # 任意非零首步梯度
m1 = (1 - b1) * g1                                # 未修正一阶矩
v1 = (1 - b2) * g1 * g1                           # 未修正二阶矩
uncorrected = (m1 / (v1.sqrt() + eps)).item()     # 不修正的等效步长 / lr
corrected = ((m1 / (1 - b1)) / ((v1 / (1 - b2)).sqrt() + eps)).item()  # 修正后
ratio = (1 - b1) / (1 - b2) ** 0.5                # = 3.162...，理论首步放大倍数
# 不变量：uncorrected ≈ ratio ≈ 3.16（偏大），corrected ≈ 1.0（= lr 量级）
assert abs(uncorrected - ratio) < 1e-3
assert abs(corrected - 1.0) < 1e-3
```

**[c] AdamW（解耦衰减）$\ne$ Adam+L2（耦合），从零解耦 == `torch.optim.AdamW`：**

```python
import torch

def adamw_decoupled_from_scratch(grads, theta0, lr=0.1, b1=0.9, b2=0.999,
                                 eps=1e-8, wd=0.05, steps=5):
    """解耦：先按原始梯度 g 做 Adam 步，再直接减 lr*wd*theta（不过自适应分母）。"""
    theta = theta0.clone()
    m = torch.zeros_like(theta); v = torch.zeros_like(theta)
    for t in range(1, steps + 1):
        g = grads[t - 1]                          # 不含 wd 的原始梯度
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
        theta = theta - lr * (m_hat / (v_hat.sqrt() + eps) + wd * theta)  # 解耦衰减
    return theta

torch.manual_seed(0)
d, steps, wd = 8, 5, 0.05
theta0 = torch.randn(d)
grads = [torch.randn(d) for _ in range(steps)]

# torch.optim.AdamW（解耦） vs torch.optim.Adam(weight_decay=)（L2/耦合）
def run(opt_cls, **kw):
    p = theta0.clone().requires_grad_(True)
    opt = opt_cls([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd, **kw)
    for t in range(steps):
        opt.zero_grad(); p.grad = grads[t].clone(); opt.step()
    return p.detach()

ref_adamw = run(torch.optim.AdamW)
ref_adam_l2 = run(torch.optim.Adam)
assert (ref_adamw - ref_adam_l2).norm() > 1e-4         # AdamW != Adam+L2（相同 wd 下不同）
mine = adamw_decoupled_from_scratch(grads, theta0, wd=wd, steps=steps)
assert torch.allclose(mine, ref_adamw, atol=1e-6)      # 从零解耦 == torch.optim.AdamW
```

**[e] cosine-with-warmup 调度（检查首/峰/末三点 + 单调）：**

```python
import math

def cosine_warmup_lr(step, warmup, total, lr_max, lr_min=0.0):
    """线性 warmup 到 lr_max，再余弦衰减到 lr_min。需预定总步数 total。"""
    if step < warmup:
        return lr_max * step / warmup                       # 线性上升（step=0 -> ~0）
    progress = (step - warmup) / (total - warmup)           # 0 -> 1
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

warmup, total, lr_max = 100, 1000, 1.0
lrs = [cosine_warmup_lr(t, warmup, total, lr_max) for t in range(total + 1)]
assert lrs[0] < 1e-9                                        # 起点 ~0
assert abs(lrs[warmup] - lr_max) < 1e-9                     # warmup 边界处取峰值
assert lrs[-1] < 1e-6                                       # 末端衰减到 ~lr_min(=0)
assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(warmup, total))  # warmup 后单调不增
```

完整可跑脚本见 [`code/optimizer_lr_schedule.py`](code/optimizer_lr_schedule.py)（纯 PyTorch、CPU 几秒、6 个 `assert` 覆盖 [a]–[f]）。运行 `python code/optimizer_lr_schedule.py` 的**真实输出**：

```text
[a] SGD+momentum from scratch vs torch.optim.SGD: max|Δ| = 1.19e-07  OK
[b] Adam from scratch vs torch.optim.Adam: max|Δ| = 1.19e-07  OK
[c] AdamW vs Adam+L2 (same wd=0.1): max|Δ| = 2.13e-01 (DIFFER, decoupled≠coupled); from-scratch decoupled vs AdamW = 2.16e-07 (MATCH)  OK
[d] bias correction at step 1: corrected/uncorrected = 0.316 (= √(1-β2)/(1-β1) ≈ 0.316); without it the under-estimated v̂ makes the first step ~3.2× too large  OK
[e] cosine+warmup lr: t=0 -> 0.000, t=warmup -> 1.000 (peak), t=end -> 0.000 (min); linear warmup midpoint -> 0.500  OK
[f] ill-conditioned quadratic (κ=20) after 100 steps: GD loss = 4.04e-02  vs  GD+momentum loss = 1.46e-05  (momentum lower)  OK

all optimizer / LR-schedule sanity checks passed ✓
```

> ✅ **读数解释**
> - **[a]/[b]** 从零 SGD/Adam 与 `torch.optim` 数值一致（max$|\Delta|$=1.2e-7），实现无误。
> - **[c]** **AdamW $\ne$ Adam+L2**：相同 wd 下两者差 $0.21$（解耦≠耦合），而从零解耦实现匹配 `AdamW`（2.2e-7）——§5 的核心，可执行印证。
> - **[d]** 偏差修正方向：corrected/uncorrected $=0.316=\frac{\sqrt{1-\beta_2}}{1-\beta_1}$，即**不修正首步约 $3.2\times$ 偏大**（§4.2，不是偏小）——顶级 lab 最爱抓的方向题。
> - **[e]** cosine-warmup 首($\to0$)/峰($\to1$)/末($\to0$)/暖身中点($\to0.5$)形状正确（§7.2）。
> - **[f]** 病态二次（$\kappa=20$）上 GD+momentum 的 loss（1.5e-5）比裸 GD（4e-2）低约 2700×——动量在缓方向累积加速（§2）。

## 📚 参考文献

- **Adam** — Kingma & Ba, *Adam: A Method for Stochastic Optimization*, arXiv 1412.6980 (2014), ICLR 2015.
- **AdamW (Decoupled Weight Decay)** — Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, arXiv 1711.05101 (2017), ICLR 2019.
- **AdaGrad** — Duchi, Hazan & Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*, JMLR 12 (2011)（**无 arXiv id**，按 JMLR 出处引用）.
- **RMSProp** — Hinton, *Neural Networks for Machine Learning*, Coursera Lecture 6e (2012)（课程讲义，**无正式论文 / arXiv id**）.
- **Nesterov accelerated gradient** — Nesterov, *A method of solving a convex programming problem with convergence rate $O(1/k^2)$*, Soviet Math. Doklady 27 (1983)（**无 arXiv id**）.
- **Momentum for deep nets** — Sutskever, Martens, Dahl & Hinton, *On the importance of initialization and momentum in deep learning*, ICML 2013（PMLR proceedings；**无独立 arXiv id**，按会议出处引用）.
- **AMSGrad (Adam 收敛性)** — Reddi, Kale & Kumar, *On the Convergence of Adam and Beyond*, ICLR 2018（OpenReview；无确认的独立 arXiv id，按会议出处引用）.
- **SGDR / cosine annealing** — Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, arXiv 1608.03983 (2016), ICLR 2017.
- **Transformer / Noam schedule** — Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762 (2017), NeurIPS 2017.
- **Linear scaling rule** — Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, arXiv 1706.02677 (2017).
- **LARS** — You, Gitman & Ginsburg, *Large Batch Training of Convolutional Networks*, arXiv 1708.03888 (2017).
- **LAMB** — You et al., *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes*, arXiv 1904.00962 (2019), ICLR 2020.
- **Lion** — Chen et al., *Symbolic Discovery of Optimization Algorithms*, arXiv 2302.06675 (2023), NeurIPS 2023.
- **Shampoo** — Gupta, Koren & Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization*, arXiv 1802.09568 (2018), ICML 2018.
- **SOAP** — Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam*, arXiv 2409.11321 (2024).
- **Adafactor** — Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, arXiv 1804.04235 (2018), ICML 2018.
- **Sophia** — Liu et al., *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*, arXiv 2305.14342 (2023), ICLR 2024.
- **One-cycle / Super-Convergence** — Smith & Topin, *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*, arXiv 1708.07120 (2017).
- **WSD / MiniCPM** — Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, arXiv 2404.06395 (2024), COLM 2024.
- **8-bit Adam** — Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv 2110.02861 (2021), ICLR 2022.
- **Gradient clipping** — Pascanu, Mikolov & Bengio, *On the difficulty of training Recurrent Neural Networks*, arXiv 1211.5063 (2012), ICML 2013.
- **Muon (scaling)** — Liu et al. (Moonshot AI), *Muon is Scalable for LLM Training*, arXiv 2502.16982 (2025).
- **Muon (原始 writeup)** — Keller Jordan et al., *Muon: An optimizer for the hidden layers of neural networks*, 2024 技术博客（**无独立 arXiv 论文**，按博客 writeup 引用）.
