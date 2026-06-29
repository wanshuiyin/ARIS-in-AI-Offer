## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Normalization / Residual / Init** — 一页拿下面试核心要点（详见后文 §1–§11 推导）。

1. **为什么要归一化**：深网络逐层放大 / 缩小激活，方差以 $g^L$ 指数发散或塌缩；归一化把每层激活拉回受控尺度，关键收益是**平滑了 loss landscape**（更小的梯度 Lipschitz / β-smoothness，Santurkar 2018），从而能用更大学习率、堆更深——**不是**"减少 internal covariate shift"那套旧说法。

2. **BatchNorm**：沿 **batch（+空间）维**对每个 channel 归一化，$\hat x=\frac{x-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$；**训练用 batch 统计 + 维护 running mean/var，推理用 running stats** → train$\ne$eval，且强依赖 batch 大小（小 batch / 变长序列 / RL / online 全踩雷）。

3. **LayerNorm**：沿**特征维 per-token** 归一化（与 batch 完全解耦），$y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$；batch=1 也能用、变长序列也能用、train$=$eval——所以 Transformer / RNN 用它而不用 BN。

4. **RMSNorm**：丢掉 re-centering，只按 RMS 缩放 $\bar x=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot\gamma$（无均值、无 $\beta$）；论点是 **re-scaling 不变性比 re-centering 更重要**，省一次 reduction，LLaMA 之后的事实默认。

5. **Pre-LN vs Post-LN**：Post-LN（原始 Transformer，LN 在残差相加**之后**）质量上限略高但**需 LR warmup、深了不稳**；Pre-LN（LN 在残差分支**内部**）有干净恒等梯度路径 → 稳、免 warmup，但**残差流幅度随深度按 $\sqrt L$ 增长**、深层贡献被稀释，必须补一个 final LN。

6. **放置变体**：DeepNorm（放大残差 $\alpha x_l$ + 缩小 init → 训 1000 层）、Sandwich / double LN（Gemma2 分支前后各一个）、**QK-Norm**（点积前归一化 Q、K，压住 attention logits 防爆）。

7. **残差 + 缩放**：$y=x+F(x)$ 的雅可比 $I+\partial F/\partial x$ 给出**梯度高速路**（恒等项保证梯度不消失）；缩放技巧把分支起点压向恒等——$1/\sqrt N$ 深度缩放、LayerScale（learnable per-channel $\lambda$）、ReZero（learnable 标量 init 0）、SkipInit。

8. **初始化**：核心目标是**方差保持**——Xavier（tanh，$\text{Var}(W)=\frac{2}{n_\text{in}+n_\text{out}}$）、Kaiming（ReLU，$\text{Var}(W)=\frac{2}{n_\text{in}}$，那个 2 补偿 ReLU 砍掉的一半方差）；残差网再按深度下调，**GPT-2 把残差投影权重 $\times\frac{1}{\sqrt{2N}}$**；Fixup 仅靠 init 就免归一化训深残差网。

9. **μP / 归一化-free / 工程**：μP 让**最优超参（尤其 LR）宽度不变** → 小宽度调好 zero-shot 迁移到大模型；Fixup / NFNets / DyT 探索去归一化（前沿方向，非定论）；工程上盯紧 final LN、$\epsilon$ 位置、fp32 reduction、fused kernel。

## §1 为什么要归一化 + 残差

**深网络的根本病灶是"尺度在深度上失控"。** 把一个信号过 $L$ 个线性层，如果每层把激活方差乘以一个增益 $g$，那么 $L$ 层后方差变成 $g^L\cdot\text{Var}(\text{input})$。$g\gt 1$ 就指数爆炸、$g\lt 1$ 就指数消失——只有 $g=1$ 这个**临界点**才稳定，而随机初始化几乎不可能恰好命中。反向传播同理：梯度也按某个增益逐层连乘，于是要么梯度爆炸（loss 变 NaN），要么梯度消失（深层学不动）。这就是 2015 年前"很难训超过一二十层网络"的核心障碍。

> 💡 **$g^L$ 是简化模型，别当全部**
> $g^L$ 是个**标量 / 平均场近似**——真实每层增益数据相关、含非线性 / attention / normalization，完整的信号传播稳定性还涉及 Jacobian 谱 / dynamical isometry、均值漂移、跨层相关、训练中参数漂移。但 $g^L$ 抓住了最本质的"指数尺度失控"，是理解归一化 / 残差 / init 的最小入口模型。顶级面试若深问，要点明它是简化，稳定性 $\ne$ 只看方差连乘。

两类结构性药方分别攻这个病（外加 **init / 参数化**，§8–§9，是第三条轴）——三者**概念上分工、工程上强耦合**：norm 控局部尺度、residual 改善梯度路径、init 设定 step-0 临界性；现代深网的稳定性是三者的**联合设计**（见 §5.3 Pre-LN 残差流增长、§8 残差缩放、§10 norm-free 靠 init+scaling）：

- **归一化（normalization）**：在前向里显式把激活拉回固定尺度（零均值、单位方差，再用 $\gamma,\beta$ 学回需要的尺度），直接掐断 $g^L$ 的连乘。
- **残差连接（residual / skip connection）**：把每层写成 $y=x+F(x)$，给梯度开一条**恒等高速路**（§7），让梯度即使绕过 $F$ 也能无损流到浅层。

二者配合（Pre-LN Transformer = 残差 + LayerNorm + 合适 init）才让"几十到上千层"成为常规操作。

### 1.1　internal covariate shift 的叙事——以及它被推翻

BatchNorm 原始论文（Ioffe & Szegedy, 2015）给出的动机是 **internal covariate shift（ICS，内部协变量偏移）**：随着前面层参数在训练中更新，每一层**输入的分布**也在不停漂移，后面层得不断追着这个移动的分布重新适配，拖慢训练；BN 通过固定每层输入的均值方差来"稳住分布"，所以训练更快。这个故事直觉好懂、流传极广——**但它大概率不是 BN 真正起作用的原因。**

Santurkar et al.（*How Does Batch Normalization Help Optimization?*, 2018, arXiv 1805.11604）用两组实验把这个叙事拆了：

1. **故意制造 ICS，BN 照样 work**：在 BN 层**之后**注入随时间变化的随机噪声（人为放大分布漂移，ICS 明显增大），带 BN 的网络训练速度 / 精度几乎不受影响。如果 BN 的价值真在"消除 ICS"，这里早该崩了。
2. **BN 真正改变的是优化地形**：他们证明 BN 让 loss 关于参数更**平滑**——loss 与梯度的 Lipschitz 常数变小（更好的 β-smoothness），梯度更可预测、更稳定。地形更平滑意味着可以放心走更大的步子（更大 LR）而不越过 / 震荡，这才是 BN 加速收敛、允许大 LR 的机制。

> ⚠️ **面试别再背 "BN 减少 covariate shift"**
> 这是被强实证质疑、主流不再视为主因的旧叙事。更稳的说法：归一化的主要收益是**让损失地形更平滑、梯度更良态（well-conditioned）**（并带来支持更大学习率、隐式正则、尺度不变性等多重好处），从而能训更深；"稳定每层输入分布"至多是表象，不是公认的因果机制。把 covariate shift 当成"已证明的唯一原因"是经典踩坑（§11、Q21）。

> 💡 **一句话心智模型** — 归一化不是在"对齐分布"，而是在给优化器**修一条更平的路**（小 Lipschitz）；残差不是在"加特征"，而是在给梯度**修一条不堵车的高速路**（恒等雅可比）。两条都是为了让"很深"这件事在数值上可行。

## §2 BatchNorm

### 2.1　公式：沿 batch 维，逐 channel

设一个 mini-batch $\mathcal{B}=\{x_1,\dots,x_m\}$，对**每个 channel / 特征 $c$ 独立**统计：

$$\mu_c = \frac{1}{m}\sum_{i=1}^m x_{i,c}, \qquad \sigma_c^2 = \frac{1}{m}\sum_{i=1}^m (x_{i,c}-\mu_c)^2,$$

$$\hat x_{i,c} = \frac{x_{i,c}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}, \qquad y_{i,c} = \gamma_c\,\hat x_{i,c} + \beta_c.$$

关键是**沿哪个维度求统计**：对 $[N, C]$ 的全连接特征，沿 $N$（batch）维、每个 $C$ 一组 $(\mu,\sigma)$；对 $[N, C, H, W]$ 的卷积特征，沿 $(N, H, W)$ 求、每个 channel $C$ 一组。所以 BN 的统计量是 **"跨样本"** 的——一个样本的归一化结果**依赖同 batch 里的其他样本**，这是它一切麻烦的根源。$\gamma_c,\beta_c$ 是逐 channel 的可学缩放 / 偏移，让网络能恢复"非零均值 / 非单位方差"的表达（否则强行标准化会限制表达力）。

### 2.2　train vs eval：running stats 与 momentum

推理时往往**一次只来一个样本**，没有 batch 可统计；而且我们要**确定性**输出（同一输入每次结果一致），不能让结果随同 batch 的其他样本变。于是 BN 在训练时**额外维护一份 running（滑动平均）统计**，推理时改用它：

$$\hat\mu \leftarrow (1-\rho)\,\hat\mu + \rho\,\mu_\mathcal{B}, \qquad \hat\sigma^2 \leftarrow (1-\rho)\,\hat\sigma^2 + \rho\,\sigma_\mathcal{B}^2,$$

其中 $\rho$ 是 momentum（PyTorch 里默认 0.1）。于是：

- **训练（train mode）**：用**当前 batch** 的 $\mu_\mathcal{B},\sigma_\mathcal{B}^2$ 归一化，并顺手更新 running stats。
- **推理（eval mode）**：用冻结的 running $\hat\mu,\hat\sigma^2$ 归一化，**不再看 batch**。

> ⚠️ **train$\ne$eval 是 BN 的硬约束（高频考点）**
> BN 是少数几个"训练和推理走不同计算路径"的层。忘记切 `model.eval()` 会让推理用 batch 统计 → 结果随 batch 内容抖动、复现不了；BN 层在小 eval batch 上还会数值异常。这点和 LayerNorm / RMSNorm 形成鲜明对比——后者 train 和 eval 完全同构。

### 2.3　为什么 BN 不适合序列 / 小 batch / RL

BN 的"跨样本统计"在很多场景下直接失效：

- **变长序列（NLP / 语音）**：不同样本长度不同，padding 会把无意义的 0 算进 batch 统计；不同位置 token 的分布也不一样，"沿 batch 求每个位置的均值"既要对齐长度又语义可疑。Transformer / RNN 因此几乎不用 BN。
- **小 batch**：$\mu_\mathcal{B},\sigma_\mathcal{B}^2$ 是用 $m$ 个样本估的，$m$ 小则估计噪声大（方差 $\propto 1/m$），归一化抖得厉害。检测 / 分割 / 大图任务因显存只能开 batch=2~4，BN 严重掉点（这是 GroupNorm 的动机，§6）。极端 $m=1$ 时方差为 0，根本没法做。
- **RL / online / 流式**：数据是**非平稳**且**强相关**的（同一条 trajectory 的相邻状态高度相关），batch 统计既不独立也随策略漂移；加上 RL 里 train/eval 行为切换频繁，BN 的 running stats 极易和实际分布脱节。RL 里普遍改用 LayerNorm 或不用归一化。

> ❌ **"BN 是默认归一化" 是 CV 时代的惯性**
> 在 CNN + 大 batch 分类任务上 BN 仍然好用；但一进序列建模 / 小 batch / RL，BN 的跨样本耦合 + train$\ne$eval 就成了负担。现代 LLM 栈里 BN 基本绝迹，主力是 LayerNorm / RMSNorm。

## §3 LayerNorm

### 3.1　公式：沿特征维，per-token

LayerNorm（Ba, Kiros & Hinton, 2016, arXiv 1607.06450）把统计维度**从 batch 转到特征**。对**单个 token / 单个样本**的隐藏向量 $x\in\mathbb{R}^d$：

$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i-\mu)^2,$$

$$\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}, \qquad y = \gamma\odot\hat x + \beta, \quad \gamma,\beta\in\mathbb{R}^d.$$

对一个 $[B, T, d]$ 的 Transformer 张量，LN 沿**最后一维 $d$** 求 $(\mu,\sigma)$——每个 $(b,t)$ 位置一组统计，**完全不跨样本、不跨 token**。$\gamma,\beta$ 是 $d$ 维逐通道仿射（elementwise affine）。

### 3.2　为什么适配 Transformer / RNN

LN 的统计只在**一个 token 的特征内部**完成，于是 BN 的所有痛点一次消除：

- **无 batch 耦合**：batch=1 也能算（特征维永远有 $d$ 个数），样本之间互不影响。
- **变长序列友好**：每个位置独立归一化，长度不齐、padding 都不影响有效 token 的统计。
- **train$=$eval**：没有 running stats，训练和推理是**同一个确定性函数**，不用切模式、不会脱节。
- **在线 / 流式 / RL 友好**：来一个 token 算一个，天然适配自回归 decode。

> 💡 **BN 沿 batch、LN 沿 feature（一句话区分）**
> 同一个 $[N, C]$ 张量：BN 把它竖着切（固定 channel、跨样本求统计），LN 把它横着切（固定样本、跨 channel 求统计）。BN 问"这个特征在整批里什么尺度"，LN 问"这个样本内部各特征什么尺度"。前者引入跨样本依赖，后者没有——这就是 LN 适配序列的全部理由。

> ⚠️ **LN 的 $\gamma,\beta$ 不是可有可无**
> 强行标准化到零均值单位方差会**砍掉表达力**（比如 sigmoid 前你可能就需要非零均值）；$\gamma,\beta$ 让网络把尺度 / 偏移**学回来**。去掉仿射（`elementwise_affine=False`）在某些任务可行，但默认保留。注意 RMSNorm 只保留 $\gamma$、砍掉 $\beta$（§4）。

## §4 RMSNorm

### 4.1　只 re-scale，不 re-center

RMSNorm（Zhang & Sennrich, 2019, arXiv 1910.07467）的出发点是一个假设：**LayerNorm 的成功主要来自 re-scaling 不变性（把向量缩放到固定尺度），而不是 re-centering（减均值）。** 既然如此，干脆把减均值这一步省掉，只用 **均方根（RMS）** 做缩放：

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}, \qquad \boxed{\;\bar x = \frac{x}{\text{RMS}(x)}\odot\gamma\;}$$

对比 LayerNorm，RMSNorm 做了两处删减：**(1) 不减均值 $\mu$**（少一次 reduction），**(2) 通常不要 $\beta$ 偏移**（只留 $\gamma$ 缩放）。本质上 LN 既给 $x$ 平移又缩放，RMSNorm 只缩放。

### 4.2　为什么 re-scaling 比 re-centering 重要 + 成本

直觉：网络真正怕的是**激活尺度失控**（$g^L$ 那套），而尺度由 $\lVert x\rVert$ / RMS 主导，和均值关系不大。Zhang & Sennrich 的消融显示：在 LN 里**只保留 re-scaling、去掉 re-centering**，多数任务性能几乎不掉；反过来只保留 re-centering 则掉得多。所以 RMSNorm 用更少的运算拿住了 LN 的主要红利。

成本上 RMSNorm 比 LN 省：

- **少一次均值 reduction**：LN 要先求 $\mu$ 再求 $\sigma^2$（两遍 / 一遍带补偿），RMSNorm 直接一遍平方和。
- **少存 $\beta$**、少一次加法。
- 这些都是 memory-bound 的逐元素 + reduction 操作，省下来在长序列 / 大模型上累积成可观的吞吐收益。

> ✅ **LLaMA 之后的事实默认**
> LLaMA、Qwen、Mistral、Gemma 等近代 LLM 基本都用 RMSNorm（配 Pre-LN 放置）。面试问"现代 LLM 的归一化"，标准答案是 **Pre-RMSNorm**；能补一句"因为 re-scaling 是主要收益、re-centering 可省"就到位了。

> ⚠️ **RMSNorm 没有 re-centering / 没有 bias**
> 把 LN 代码直接当 RMSNorm 用会出错：RMSNorm **不减均值**（所以它对输入的整体平移**不**不变——加个常数 $c$ 后输出会变，而 LN 不变），也**没有 $\beta$**。一个直接后果：RMSNorm 后的激活均值不被强制归零，下游若假设零均值要小心。

## §5 Pre-LN vs Post-LN（最关键的一节）

这是归一化里最容易被深问的点：**LN 到底放在残差的哪一侧？** 两种放法行为差异巨大。

### 5.1　两种放置

记一个子层（attention 或 FFN）为 $\text{Sublayer}(\cdot)$：

$$\textbf{Post-LN（原始 Transformer, Vaswani 2017）}:\quad x_{l+1} = \text{LN}\big(x_l + \text{Sublayer}(x_l)\big),$$

$$\textbf{Pre-LN（现代 LLM 主流）}:\quad x_{l+1} = x_l + \text{Sublayer}\big(\text{LN}(x_l)\big).$$

差别看似只是 LN 挪了个位置，实则改变了**残差路径的纯净度**：

- **Post-LN**：LN 套在"残差相加之后"的整条流上——残差路径**被 LN 反复挤压**，没有一条干净的恒等通道。
- **Pre-LN**：LN 只作用在进入子层的分支输入上，残差主干 $x_l + (\cdots)$ 是**纯恒等加法**——梯度可以无损地沿主干流到底。

### 5.2　梯度幅度论证（Xiong et al. 2020）

Xiong et al.（*On Layer Normalization in the Transformer Architecture*, 2020, arXiv 2002.04745）在初始化处做了梯度尺度分析，给出干净结论：

- **Post-LN**：在初始化时，**靠近输出层**的参数梯度范数约为 $\Theta\!\big(d\sqrt{\ln d}\big)$，**与深度 $L$ 无关**——即顶层梯度天然偏大，而各层之间梯度尺度严重不均衡。直接上大 LR 会让这些大梯度把训练带飞，所以**必须用 learning-rate warmup**（前几千步把 LR 从 0 慢慢拉起）压住，深堆叠时还经常发散。
- **Pre-LN**：每层参数的梯度范数约为 $O\!\big(d\sqrt{(\ln d)/L}\big)$——**随深度 $L$ 衰减、有界且跨层均衡**。于是优化良态，**可以直接上大 LR、免 warmup**，深堆叠也稳。

> ✅ **一句话抓住 Pre vs Post**
> Pre-LN 给残差留了**干净恒等梯度路**（梯度按 $\sim 1/\sqrt L$ 有界、跨层均衡）→ 稳、免 warmup；Post-LN 把梯度尺度堆在顶层（与 $L$ 无关地偏大）→ 需 warmup、深了不稳。这就是为什么 GPT/LLaMA 一律 Pre-LN。
>
> **注意**："Pre-LN 免 warmup"特指它去掉了 Post-LN 那种**梯度不均衡**导致的 warmup 刚需；实践中现代 Pre-LN 大模型（GPT-3/LLaMA）**仍然配 warmup**——那是 Adam 早期 $\hat v$ 方差大、大 batch 等**另一层**动机（与 LN 放置无关，见优化器篇 §7.1），别把"Pre-LN"读成"完全不需要 warmup"。

### 5.3　Pre-LN 的代价：残差流随深度膨胀

Pre-LN 不是免费的。看残差主干：$x_{l+1} = x_l + F_l\big(\text{LN}(x_l)\big)$。每个分支输出 $F_l(\text{LN}(\cdot))$ 因为输入被 LN 归一化、量级大致 $O(1)$，于是方差逐层**累加**：

$$\text{Var}(x_l) \approx \text{Var}(x_0) + \sum_{j\lt l}\text{Var}\big(F_j\big) \;\propto\; l \quad\Longrightarrow\quad \text{std}(x_l)\propto\sqrt{l}.$$

残差流的幅度随深度按 $\sqrt l$ 增长。后果：每个分支内部的 LN 都要除以这个越来越大的 $\text{std}(x_l)$，于是**深层子层的相对贡献被稀释**——加进去的 $O(1)$ 更新相对于已经 $\sqrt l$ 量级的主干越来越微不足道，网络的"有效深度"饱和，极端时退化成"恒等主干主导、深层近乎摆设"（有人称 representation / identity collapse）。这也是为什么 Pre-LN 网络**必须在最后补一个 final LN**（把膨胀的残差流拉回正常尺度再送进输出头），以及催生了 DeepNorm / Sandwich 等想兼顾两端的变体。

> ⚠️ **Pre-LN 一定要有 final LN（高频踩坑）**
> Pre-LN 结构里，最后一层之后残差流量级已是 $\sqrt L$ 级、且未被任何 LN 收口。不补 final LN 直接接 LM head，输出分布会失控、训练质量明显下降。GPT-2/3、LLaMA 都在 transformer 栈之后、输出投影之前加了这个 final LN——别漏。

### 5.4　为什么 Post-LN 质量上限有时更高

既然 Pre-LN 这么稳，为何还提 Post-LN？因为**当 Post-LN 能训稳时（充分 warmup + 调参），它的最终质量往往略优**。原因正是 §5.3 的反面：Post-LN 每层都把残差流重新归一化，深层贡献不会被稀释，"有效深度"利用得更充分。代价是训练脆弱（窗口窄、依赖 warmup、深了容易崩）。所以工程上的取舍是：**追求稳定 / 可扩展性 → Pre-LN（现代 LLM 默认）；在可控规模上榨质量 → Post-LN 或 DeepNorm 这类"修好的 Post-LN"。**

## §6 放置与变体

围绕"LN 放哪 / 归一化什么"，社区演化出一批变体，面试常被点名：

### 6.1　DeepNorm：把 Post-LN 修到能训 1000 层

DeepNorm（Wang et al., *DeepNet*, 2022, arXiv 2203.00555）是一个**改良版 Post-LN**，目标是同时拿到 Post-LN 的质量和 Pre-LN 的稳定。两个动作：

$$x_{l+1} = \text{LN}\big(\alpha\,x_l + \text{Sublayer}(x_l)\big), \qquad \text{且把子层权重初始化按 } \beta \text{ 缩小}.$$

对一个 $N$ 层单栈，取 $\alpha=(2N)^{1/4}\gt 1$（**放大残差主干**），同时用增益 $\beta=(8N)^{-1/4}\lt 1$ **缩小子层初始化**（编码器-解码器另有公式）。直觉：放大 $x_l$ 让残差主干在相加时占主导（接近 Pre-LN 的恒等稳定性），缩小子层 init 让每步"模型更新量"被**理论上界**住、不随深度爆炸。结果是 Post-LN 的放置 + 受控的更新幅度 → 稳定训练到 **1000 层**。记忆点：**DeepNorm = up-scale 残差 + down-scale init 的 Post-LN**。

### 6.2　Sandwich / double LN：分支前后各一个

Gemma2 等用 **sandwich norm（双 LN）**：在每个子层的残差分支**前后各放一个归一化**：

$$x_{l+1} = x_l + \text{PostNorm}\Big(\text{Sublayer}\big(\text{PreNorm}(x_l)\big)\Big).$$

PreNorm 像 Pre-LN 一样给子层喂归一化输入（稳梯度），PostNorm 再把子层**输出**的量级收口（防止它无界地灌进残差流，缓解 §5.3 的膨胀）。代价是多一个 LN 的算力。可看成"Pre-LN 的稳定 + 对分支输出额外加一道闸"。

### 6.3　QK-Norm：归一化 Q、K 稳住 attention logits

QK-Norm（Henry et al., *Query-Key Normalization*, 2020, arXiv 2010.04245）针对的是**注意力 logits 爆炸**：训练中 $q^\top k$ 可能随尺度漂移到极大，softmax 进入饱和区（near one-hot），梯度消失 / 训练发散（大模型、长训练尤甚，也叫 attention entropy collapse）。修法极简——**在点积之前对每个 head 的 $q,k$ 各做一次归一化**（L2 或 RMS/LayerNorm），再算 logits：

$$\tilde q = \text{Norm}(q),\quad \tilde k = \text{Norm}(k), \qquad \text{logits} = \frac{\tilde q^\top \tilde k}{\tau}\ (\tau\ \text{可学温度}).$$

归一化后 $q,k$ 的模长被钳住，logits 不再随尺度爆炸 → attention 稳定。Gemma2、Chameleon、ViT-22B 等都用了 QK-Norm，已成为大规模训练的标配稳定器之一。

### 6.4　GroupNorm 与 WeightNorm

- **GroupNorm**（Wu & He, 2018, arXiv 1803.08494）：把 channel 分成 $G$ 组，**在每个样本的每组 channel（+空间）内**归一化——**不依赖 batch**。它统一了一族归一化：$G=1$ 退化为 LayerNorm（全 channel 一组），$G=C$ 退化为 InstanceNorm（每 channel 一组）。主战场是**小 batch 的视觉任务**（检测 / 分割，batch=2~4 时 BN 崩、GN 稳）。
- **WeightNorm**（Salimans & Kingma, 2016, arXiv 1602.07868）：不归一化激活，而是**重参数化权重**——把权重向量拆成方向和模长 $w = g\,\frac{v}{\lVert v\rVert}$，单独学一个标量模长 $g$ 和方向 $v$。它解耦了权重的尺度与方向、加速优化，且**无 batch 依赖、推理无开销**（可折回普通权重），但稳定性不如 BN，热度后来被 BN/LN 盖过。

> 💡 **一句话归位四种 norm**
> BN 沿 batch、LN 沿 feature、GN 沿 group（batch 无关、桥接 LN 与 IN）、WN 归一化的是**权重**而非激活。面试问"小 batch 检测用什么归一化"答 GroupNorm；问"现代 LLM"答 RMSNorm；问"稳 attention logits"答 QK-Norm。

## §7 残差连接 + 残差缩放

### 7.1　残差为什么 work：三种视角

残差块 $y = x + F(x)$（He et al., *ResNet*, 2015, arXiv 1512.03385）能训很深，有三个互补解释：

1. **恒等映射 / 梯度高速路**：雅可比 $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$。那个 $I$ 保证——哪怕 $\frac{\partial F}{\partial x}$ 因深度而消失，梯度仍能经恒等项无损传回。$L$ 个残差块串起来，$\frac{\partial x_L}{\partial x_0}=\prod_{l}\big(I+\frac{\partial F_l}{\partial x}\big)$ 展开后含一个**纯恒等项**（外加各阶交叉项），所以梯度**至少**有一条不衰减的通路。这直接破了 $g^L$ 消失。
2. **优化更易 / 学残差比学映射容易**：若理想映射接近恒等，让 $F$ 去拟合"残差"（差量）比让一整层拟合恒等容易得多——初始化在 0 附近就已经接近恒等，优化从一个好起点出发。
3. **浅路径集成（ensemble of shallow paths，Veit et al., 2016, arXiv 1605.06431）**：一个 $L$ 块残差网在前向上等价于 $2^L$ 条不同深度路径的集合（每块选"走 $F$"或"走恒等"），其中**短路径占主导**，有效梯度主要来自这些浅路径——所以深残差网"像很多浅网的集成"，优化自然更容易。

### 7.2　残差流（residual stream）视角

机理可解释性里有个统一图景：残差主干是一条贯穿全网的**共享通信总线（residual stream）**。每个子层**从总线读**（经 LN 取出当前状态）、算一个更新、**把更新加回总线**。信息**默认沿总线恒等保留**，每层只决定"往上加什么"。这个视角解释了很多现象：为什么 Pre-LN 残差流会随深度累加膨胀（§5.3，大家都往同一条总线写）、为什么残差缩放（下面）有用（控制每层往总线写的强度）、为什么能在中间层"读出"语义。

### 7.3　残差缩放：把分支起点压向恒等

深残差网的一个隐患（§5.3）：分支输出无节制地灌进残差流，使其方差线性膨胀、训练初期不稳。一族技巧给残差分支**乘一个小缩放**，让训练**从接近恒等的状态起步**：

- **$1/\sqrt N$ 深度缩放**：把每个分支乘 $1/\sqrt N$（$N$=层数），使 $N$ 层累加后总方差仍 $O(1)$（$N\times(1/\sqrt N)^2=1$），残差流不随深度爆。GPT-2 的 init 技巧（§8.4）正是这一思想的初始化版本。
- **LayerScale**（CaiT, Touvron et al., 2021, arXiv 2103.17239）：给每个分支乘一个**可学的 per-channel 对角** $\text{diag}(\lambda)$，$\lambda$ 初始化成很小的值（如 $10^{-4}\sim10^{-6}$）：$x_{l+1}=x_l+\text{diag}(\lambda)\,F(x_l)$。起点近乎恒等，再让网络**逐通道学**每个分支该贡献多少，显著稳住深 ViT。
- **ReZero**（Bachlechner et al., 2020, arXiv 2003.04887）：更激进——每个分支乘一个**可学标量** $\alpha$，**初始化为 0**：$x_{l+1}=x_l+\alpha\,F(x_l)$。$t=0$ 时网络是**精确恒等**（dynamical isometry，每层输入输出雅可比为 $I$），梯度完美传导，可在无归一化下训很深；训练中 $\alpha$ 自己长起来。
- **SkipInit**（De & Smith, 2020, arXiv 2002.10444）：把残差分支上的 BN 换成一个**可学标量（init 0）**，照样能训深残差网——以此**实证 BN 在残差网里的主要作用，其实是初始化时把分支缩小、让网络偏向恒等函数**（而非"减少 ICS"）。这条和 §1.1 的 debunk 相互印证。

> 💡 **残差缩放的统一直觉**
> LayerScale / ReZero / SkipInit / $1/\sqrt N$ 做的是同一件事：**让残差分支在初始化时接近 0、网络从恒等起步**，于是梯度良态、训练稳定，再让网络自己学该往残差流里加多强。"init 成接近恒等"是训练超深网络的一条暗线（也是 §8 init 的核心）。

## §8 初始化

归一化和残差治标，**初始化治本**——它决定网络在第 0 步的方差 / 梯度尺度是否良态。（注：init 只保证**良态起点**；训练全程的稳定还要靠参数化 / LR / optimizer / 残差缩放 / norm 放置一起兜——见 §1 的"三轴联合设计"。）

### 8.1　目标：方差保持（variance preservation）

希望前向激活方差与反向梯度方差**逐层大致守恒**，别让 $g^L$ 把它们带跑。对一个权重 $W\in\mathbb{R}^{n_\text{out}\times n_\text{in}}$、输入各分量独立同方差的线性层 $z=Wx$：

$$\text{Var}(z_j) = n_\text{in}\,\text{Var}(W)\,\text{Var}(x).$$

要 $\text{Var}(z)=\text{Var}(x)$（前向守恒）需 $n_\text{in}\text{Var}(W)=1$；要反向梯度方差守恒需 $n_\text{out}\text{Var}(W)=1$。两者一般不能同时满足，于是有了不同折中。

### 8.2　Xavier / Glorot：给 tanh / 线性的折中

Xavier/Glorot（Glorot & Bengio, AISTATS 2010）面向**对称、近线性**激活（tanh、线性），取前向与反向两个条件的折中：

$$\text{Var}(W) = \frac{2}{n_\text{in}+n_\text{out}} \quad\Longrightarrow\quad W\sim U\!\Big[-\sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}},\ \sqrt{\tfrac{6}{n_\text{in}+n_\text{out}}}\Big]\ \text{或对应高斯}.$$

前提是激活在 0 附近近似线性（tanh 满足、ReLU 不满足），所以对 ReLU 网它偏小。

### 8.3　Kaiming / He：给 ReLU 补回那个因子 2

Kaiming/He（He et al., 2015, arXiv 1502.01852）指出：**ReLU 把负半轴清零，砍掉约一半方差**——$\mathbb{E}[\text{ReLU}(z)^2]=\tfrac12\text{Var}(z)$（对零均值对称 $z$）。要在 ReLU 后仍守恒，需补一个因子 2：

$$\tfrac12\,n_\text{in}\,\text{Var}(W)=1 \quad\Longrightarrow\quad \boxed{\;\text{Var}(W)=\frac{2}{n_\text{in}}\;}\ (\text{fan\_in 模式},\ \text{std}=\sqrt{2/n_\text{in}}).$$

这个 2 是 Xavier 与 Kaiming 的唯一实质区别，却决定了深 ReLU 网（VGG / ResNet 级）能不能训：用 Xavier 的 $1/n_\text{in}$ 喂 ReLU，每层信号能量被 ReLU 砍半又没补回，$L$ 层后二阶矩 $\mathbb{E}[y^2]\approx (1/2)^L$ **指数消失**；用 Kaiming 才守恒（§A 的 [e] 验证：Kaiming 后 post-ReLU 二阶矩 $\mathbb{E}[y^2]\approx1$，Xavier-for-ReLU $\approx 0.5$ 并逐层砍半）。这里守恒的量是**二阶矩 $\mathbb{E}[y^2]$**（喂给下一层的信号能量），而非 $\text{Var}(y)$——ReLU 让输出均值非零，$\text{Var}(y)=1-1/\pi\approx0.68$。

> ⚠️ **Xavier vs Kaiming 不是"换个公式"，是"是否补 ReLU 的因子 2"**
> tanh / 线性用 Xavier（$\frac{2}{n_\text{in}+n_\text{out}}$）；ReLU / LeakyReLU 用 Kaiming（$\frac{2}{n_\text{in}}$，LeakyReLU 还要按负斜率改增益）。给 ReLU 网用 Xavier 会系统性偏小 → 深了激活 / 梯度消失。

### 8.4　残差网的深度感知下调 + GPT-2 的 $1/\sqrt{2N}$

即便每层都 Xavier/Kaiming，残差流仍会累加膨胀（§5.3）。所以深残差网在 per-layer init 之外还要**按深度下调残差分支**：

- 把分支**最后一层初始化为 0**（分支起点输出 0 → 块起点为恒等，同 ReZero / Fixup 思想）；
- 或把分支整体乘 $1/\sqrt L$。

**GPT-2 的经典技巧**：把**残差投影层**（attention 的输出投影 + FFN 的 down 投影，即"往残差流写"的那两个矩阵）的权重在 init 时**乘 $\frac{1}{\sqrt{2N}}$**，$N$=层数。为什么是 $2N$ 而非 $N$？因为每个 transformer 层往残差流写**两次**（attn 一次、FFN 一次），$N$ 层共 $2N$ 次累加；按 $1/\sqrt{2N}$ 缩放每次写入，使 $2N$ 次累加后残差流方差仍 $O(1)$。nanoGPT 等实现都对 `c_proj` 类权重套这个缩放（§A 的 [f] 验证：不缩放方差随深度线性涨，$1/\sqrt{2N}$ 缩放后有界）。

### 8.5　Fixup：仅靠 init 免归一化训深残差网

Fixup（Zhang et al., 2019, arXiv 1901.09321）把"用 init 控方差"推到极致——**完全不用任何归一化层**，仅靠精心设计的初始化就训出能打 BN-ResNet 的深残差网。三招：

1. 每个残差分支的**最后一层初始化为 0**（分支起点输出 0 → 块为恒等，残差流初始方差不爆）；
2. 分支内其他层的权重**额外乘 $L^{-1/(2m-2)}$** 下调（$m$=每分支层数、$L$=块数），抵消深度累加；
3. 加入少量**可学标量偏置 / 乘子**补偿被砍掉的仿射自由度。

意义：**归一化不是训深网络的充要条件**——把方差 / 梯度的尺度问题在初始化处一次性解决，就能去掉 BN/LN。这条直接通向 §10 的 normalizer-free 路线。（Transformer 上的对应工作是 T-Fixup，主张配好 init 后可免 warmup / 免 LN——见参考文献，arXiv id 待核。）

## §9 μP（maximal update parametrization）

### 9.1　问题：标准参数化下最优 LR 随宽度漂移

标准参数化（SP，即 Xavier/Kaiming + 全网同一个 LR）有个隐蔽缺陷：**当模型宽度 $n$ 变化时，各层激活和参数更新的尺度跟着错配**，于是**最优超参（尤其学习率）会随宽度漂移**。后果很实际——你在小模型上辛苦调出的最佳 LR，搬到大模型上不再最优，每加宽一次就得重调；而大模型一次调参的算力成本极高。

根因（Yang et al., *Tensor Programs V*, 2022, arXiv 2203.03466 给出的极限分析）：在 SP 下让宽度 $n\to\infty$，要么激活 / 更新随 $n$ 爆掉（需调小 LR），要么进入 **lazy / kernel 区**（特征几乎不更新、等于没在学 feature）。两种极限都不是我们想要的"宽了还能稳定地大幅更新特征"。

### 9.2　μP：让超参宽度不变 → zero-shot 迁移

μP（maximal update parametrization）是**唯一**能在任意宽度都保持"最大化且稳定的特征学习"的缩放方案：它把**初始化方差、学习率、输出乘子**都写成宽度（fan_in）的函数，逐层分别缩放，使得——

- 每层激活与**特征更新量**在宽度上都保持 $\Theta(1)$（不爆不退化）；
- **最优学习率变得宽度不变**。

典型缩放（Adam 下，示意）：隐藏层 LR $\propto 1/\text{fan\_in}$、隐藏层 init 方差 $\propto 1/\text{fan\_in}$、输出 logits 乘 $1/\text{fan\_in}$；输入 / 输出层与隐藏层用**不同**的缩放。落地为 **μTransfer**：在一个**小宽度代理模型**上把 LR / warmup / init 等超参调好，**zero-shot 直接迁移**到几十上百倍大的目标模型，省掉在大模型上重调的天价算力（被用于 GPT-3 规模、Cerebras-GPT、MiniCPM 等）。

> 💡 **μP 面试金句**
> SP 下"最优 LR 随宽度漂移"，所以小模型调的参不能直接用到大模型；μP 把 init / LR / 输出乘子按 fan_in 重新缩放，使**每层特征更新在宽度上 $\Theta(1)$、最优 LR 宽度不变** → 小宽度调参、大宽度 zero-shot 迁移（μTransfer）。一句话：**μP 是让超参可跨宽度迁移的参数化。**

## §10 归一化-free 与前沿

归一化层带来 train/eval 差异（BN）、跨设备同步（多卡 BN）、额外 reduction 等麻烦，于是一直有人问：**能不能不要归一化？** 这是研究方向，**非定论**，但思路很有启发。

### 10.1　Fixup / NFNets：用 init + 显式方差控制替代归一化

- **Fixup**（§8.5）：纯靠初始化（分支末层置 0 + 深度下调）训深残差网，无任何归一化层，ImageNet 上逼近 BN-ResNet。证明 norm 非必需。
- **NFNets**（Brock et al., 2021, arXiv 2102.06171）：Normalizer-Free Networks，系统性地去掉 BN。三件套——**Scaled Weight Standardization**（标准化权重而非激活）+ **解析设计的缩放残差块** $x_{l+1}=x_l+\alpha\,F(x_l/\beta_l)$（用解析的 $\alpha,\beta_l$ 精确追踪 / 控制每层方差）+ **Adaptive Gradient Clipping（AGC）**（按参数范数自适应裁剪梯度，找回 BN 在大 batch 下的稳定性）。结果在 ImageNet 上**超过 EfficientNet 且无任何归一化层**，说明 BN 的好处（控尺度 + 正则 + 大 batch 稳定）可以被显式手段分别补回。

### 10.2　DyT（Dynamic Tanh）：用可学 tanh 替掉 LN

DyT（*Transformers without Normalization*, 2025, arXiv 2503.10622，arXiv id 以引用核查为准）来自一个观察：**训练好的 LayerNorm 的输入-输出曲线，长得像一条被压扁的 $\tanh$**（对中间值近线性、对离群值 S 形饱和）。既然 LN 的效果近似一个逐元素 squashing，那干脆**省掉求均值 / 方差的 reduction，直接学一个 tanh**：

$$\text{DyT}(x) = \gamma\odot\tanh(\alpha\,x) + \beta,$$

其中 $\alpha$ 是一个**可学标量**（控制输入尺度、对应 LN 里"除以 std"的作用），$\gamma,\beta$ 是逐通道仿射。论文报告在 ViT / LLM / diffusion 等多处用 DyT 替换 LN/RMSNorm，效果相当，且**去掉了归一化的统计量计算**（不再需要逐 token 的 reduction）。

> 🎯 **前沿，诚实定位**
> Fixup / NFNets / DyT 共同传递一个信息：**归一化在理论上不是训练深网络的必要条件**——它的核心作用（控方差、平滑地形、压离群值）可被"好 init / 显式方差控制 / 可学 squashing"等手段替代。但 **LN/RMSNorm 仍是当前生产系统的稳妥默认**（鲁棒、即插即用、生态成熟）。面试谈这些要点明"研究方向 / 有前景"，别说成"已取代归一化"。

## §11 工程实践 + 复杂度对比 + 常见误区

### 11.1　四种归一化对比

| 归一化 | 归一化维度 | 依赖 batch | train$\ne$eval | 可学参数 | 相对成本 |
| --- | --- | --- | --- | --- | --- |
| **BatchNorm** | 沿 batch（+空间），逐 channel | 是 | **是**（推理用 running stats） | $\gamma,\beta$ | 中（两次 reduction + 维护 running stats + 多卡需同步） |
| **LayerNorm** | 沿特征维，per-token | 否 | 否 | $\gamma,\beta$ | 中（均值 + 方差两次 reduction） |
| **RMSNorm** | 沿特征维 RMS，per-token | 否 | 否 | $\gamma$（无 $\beta$） | 低（一次平方和 reduction，无均值、无偏移） |
| **GroupNorm** | 沿组内 channel（+空间），逐 sample | 否 | 否 | $\gamma,\beta$ | 中（组内 reduction，batch 无关） |

### 11.2　工程细节

- **Pre-LN 别漏 final LN**（§5.3）：transformer 栈之后、LM head 之前必须有一个收口 LN，否则膨胀的残差流直接进输出头，质量掉。
- **$\epsilon$ 的位置**：约定是 $\sqrt{\sigma^2+\epsilon}$（$\epsilon$ 在根号**里**，PyTorch 如此），不是 $\sqrt{\sigma^2}+\epsilon$。$\epsilon$ 太小在 fp16 下会因 $\sigma^2\approx0$ 触发除零 / 溢出；RMSNorm 的 $\epsilon$ 同理在 $\sqrt{\text{mean}(x^2)+\epsilon}$ 里。
- **fp16/bf16 数值**：归一化的 reduction（求和 / 平方和）**务必在 fp32 累加**，即使激活是 bf16——平方和在低精度下极易溢出 / 损失有效位。标准实现都把 norm 内部 upcast 到 fp32 再 downcast。
- **fused kernel**：LN/RMSNorm 是 **memory-bound**（读激活 → reduction → 写回），FLOPs 很小但访存重；用 fused kernel（Apex `FusedLayerNorm`、Triton、FlashNorm 等）把读-算-写并进一个 kernel，省 HBM 往返。RMSNorm 因只有一次 reduction、无均值，天然更便宜。
- **推理成本**：单看 FLOPs 归一化微不足道，但它**打断算子融合 + 带一个 reduction**，在大模型逐 token decode 时累积成可观延迟；这也是 RMSNorm（省一次 reduction）在推理侧受欢迎的原因之一。
- **BN 的多卡坑**：数据并行下每卡只看到 batch 的一部分，BN 统计是"局部 batch"的；要全局统计得用 SyncBatchNorm（跨卡通信，慢）。LN/RMSNorm 无此问题（per-token，天然无需同步）。

### 11.3　常见误区（footguns）

> ❌ **误区 1：把 covariate shift 当成归一化有效的唯一/已证原因**
> 已被 Santurkar 2018 实证推翻（§1.1）。正确归因是"平滑 loss 地形 / 梯度更良态"。这是最经典的面试陷阱。

> ❌ **误区 2：Pre-LN 忘了 final LN**
> Pre-LN 残差流随深度按 $\sqrt L$ 膨胀且末端无 LN 收口，漏掉 final LN 直接喂输出头会掉点（§5.3）。

> ❌ **误区 3：把 LN 代码当 RMSNorm 用**
> RMSNorm **不减均值、没有 $\beta$**，且对输入整体平移**不**不变；照搬 LN 的减均值 / 加 bias 逻辑就错了（§4.2）。

> ❌ **误区 4：在 RL / 变长序列 / 小 batch 上用 BatchNorm**
> 跨样本统计 + train$\ne$eval 在这些场景全面失效，应换 LayerNorm / GroupNorm（§2.3）。

> ❌ **误区 5：给 ReLU 网用 Xavier 初始化**
> 少了补 ReLU 的因子 2，深了激活 / 梯度指数衰减；ReLU 用 Kaiming（§8.3）。

## §12 25 高频面试题

按难度分三档，点开看答案要点 + 易踩坑。L2/L3 是顶级 lab 深水区（Pre/Post 梯度论证、Kaiming 推导、μP、DeepNorm、norm-free、QK-Norm、covariate-shift debunk 等）。

### L1必会题

<details>

<summary>Q1. 为什么深网络需要归一化？归一化到底解决了什么？</summary>

- 深网络激活 / 梯度方差按 $g^L$ 指数爆炸或消失，只有 $g=1$ 临界点才稳，随机 init 难命中
- 归一化把每层激活拉回固定尺度，掐断 $g^L$ 连乘 → 可用更大 LR、堆更深
- 现代正确归因：让 loss 地形更平滑、梯度更良态（Santurkar 2018），**不是**"减少 covariate shift"

只说"加速收敛"，讲不出方差尺度 / loss 地形这一层；或还在背 covariate shift。

</details>

<details>

<summary>Q2. 写出 BatchNorm 的公式，它沿哪个维度统计？</summary>

- 逐 channel 沿 batch（卷积再加空间 $H,W$）求 $\mu_c,\sigma_c^2$，$\hat x=\frac{x-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$，再 $\gamma_c\hat x+\beta_c$
- 统计是**跨样本**的 → 一个样本的归一化依赖同 batch 其他样本
- $[N,C]$ 沿 $N$；$[N,C,H,W]$ 沿 $(N,H,W)$，每 channel 一组

把 BN 说成沿特征维（那是 LN）；或忘了它跨样本这个关键性质。

</details>

<details>

<summary>Q3. BatchNorm 训练和推理（eval）有什么不同？为什么？</summary>

- 训练用**当前 batch 统计**归一化，并以 momentum 更新 running mean/var
- 推理用冻结的 **running stats**（不看 batch），保证确定性 + 单样本可用
- 所以 BN 是 train$\ne$eval 的层，忘切 `eval()` 会让推理结果随 batch 抖动

不知道有 running stats / momentum；或以为推理也用 batch 统计。

</details>

<details>

<summary>Q4. 写出 LayerNorm 公式，它和 BN 沿的维度有何不同？</summary>

- 对单 token 向量 $x\in\mathbb{R}^d$ 沿**特征维 $d$** 求 $\mu,\sigma$，$y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$
- $[B,T,d]$ 沿最后一维，每个 $(b,t)$ 一组统计，**不跨样本、不跨 token**
- BN 沿 batch（跨样本），LN 沿 feature（样本内）——这是本质区别

把 LN 也说成跨 batch；或答不出"per-token、与 batch 解耦"。

</details>

<details>

<summary>Q5. RMSNorm 是什么？和 LayerNorm 差在哪？</summary>

- 只按 RMS 缩放：$\bar x=\frac{x}{\sqrt{\frac1d\sum x_i^2+\epsilon}}\odot\gamma$
- 相比 LN 删两样：**不减均值**（无 re-centering）、**无 $\beta$** 偏移
- 论点：re-scaling 是 LN 主要收益，re-centering 可省 → 更便宜，LLaMA 后默认

只说"RMSNorm 更快"，讲不出"去掉减均值 + 去掉 bias"两处删减。

</details>

<details>

<summary>Q6. LayerNorm 在 Transformer 里放在哪？有哪两种放法？</summary>

- Post-LN：$x_{l+1}=\text{LN}(x_l+\text{Sublayer}(x_l))$，LN 在残差相加**之后**（原始 Transformer）
- Pre-LN：$x_{l+1}=x_l+\text{Sublayer}(\text{LN}(x_l))$，LN 在残差分支**内部**（现代 LLM 主流）
- 现代 LLM 用 Pre-LN + 一个 final LN

只知道"有个 LN"，说不清放在残差的哪一侧、两种放法叫什么。

</details>

<details>

<summary>Q7. 为什么 Transformer / RNN 用 LayerNorm 而不用 BatchNorm？</summary>

- 变长序列 + padding 让"沿 batch 求每个位置统计"语义可疑、难对齐
- batch 耦合：BN 一个样本依赖同 batch 其他样本；序列任务常小 batch / batch=1
- LN per-token、与 batch 解耦、train$=$eval，天然适配序列 / 自回归 decode

只说"习惯用 LN"，答不出 batch 耦合 + 变长 + train/eval 这几条具体原因。

</details>

<details>

<summary>Q8. 残差连接为什么能帮助训练很深的网络？</summary>

- $y=x+F(x)$ 雅可比 $I+\partial F/\partial x$，那个 $I$ 给梯度一条**恒等高速路**，破 $g^L$ 消失
- 学残差比学整映射容易（理想接近恒等时从好起点出发）
- 等价于 $2^L$ 条路径的集成、短路径主导，优化更易（Veit 2016）

只说"防梯度消失"，讲不出雅可比恒等项 / 残差易学 / 浅路径集成这几个视角。

</details>

<details>

<summary>Q9. 归一化里的 $\gamma,\beta$（scale/shift）是干嘛的？去掉行不行？</summary>

- 强行标准化到零均值单位方差会砍掉表达力；$\gamma,\beta$ 让网络把尺度 / 偏移**学回来**
- $\gamma$ 缩放、$\beta$ 平移，逐通道
- RMSNorm 只保 $\gamma$、去掉 $\beta$；去掉全部仿射在某些任务可行但默认保留

以为 $\gamma,\beta$ 可有可无；或不知道 RMSNorm 砍掉了 $\beta$。

</details>

<details>

<summary>Q10. Xavier 和 Kaiming 初始化的区别？各用于什么激活？</summary>

- Xavier：$\text{Var}(W)=\frac{2}{n_\text{in}+n_\text{out}}$，给 **tanh / 线性**（对称近线性激活）
- Kaiming：$\text{Var}(W)=\frac{2}{n_\text{in}}$，给 **ReLU**，那个 2 补偿 ReLU 砍掉的一半方差
- 给 ReLU 网用 Xavier 会偏小 → 深了激活 / 梯度消失

只背两个公式，说不清"差别就是补不补 ReLU 的因子 2"。

</details>

### L2进阶题

<details>

<summary>Q11. Pre-LN vs Post-LN：行为差异 + 梯度论证。</summary>

- Post-LN 需 **LR warmup**、深了不稳，但质量上限略高；Pre-LN 稳、免 warmup，但残差流随深度膨胀、需 final LN
- Xiong 2020：Post-LN 顶层梯度 $\Theta(d\sqrt{\ln d})$（与 $L$ 无关地偏大、跨层不均）→ 必须 warmup
- Pre-LN 每层梯度 $O(d\sqrt{(\ln d)/L})$（随 $L$ 衰减、有界均衡）→ 大 LR 免 warmup

只说"Pre-LN 更稳"，给不出顶层梯度偏大 / 随深度衰减这个定量论证。

</details>

<details>

<summary>Q12. 为什么 RMSNorm 去掉 re-centering 没事？re-scaling 和 re-centering 谁更重要？</summary>

- 网络怕的是激活**尺度**失控（$g^L$），尺度由 RMS / 模长主导，和均值关系不大
- 消融：LN 里只保 re-scaling 性能几乎不掉，只保 re-centering 掉得多 → re-scaling 主导
- 所以 RMSNorm 用更少运算拿住主要红利

只说"省了均值计算"，答不出"re-scaling 是主要收益"这个论点依据。

</details>

<details>

<summary>Q13. 为什么 BatchNorm 不适合变长序列 / 小 batch / RL？</summary>

- 变长：padding 污染统计 + 不同位置分布不同，"沿 batch 求每位置统计"语义差
- 小 batch：统计估计噪声 $\propto1/m$，$m$ 小则归一化抖动，$m=1$ 直接失效
- RL：数据非平稳 + 强相关 + train/eval 频繁切，running stats 与实际脱节

只笼统说"BN 不好用"，讲不出三个场景各自的失效机理。

</details>

<details>

<summary>Q14. 推导 Kaiming 初始化的方差，为什么是 $2/n_\text{in}$？</summary>

- 线性层 $\text{Var}(z)=n_\text{in}\text{Var}(W)\text{Var}(x)$，要前向守恒需 $n_\text{in}\text{Var}(W)=1$
- ReLU 把负半轴清零：$\mathbb{E}[\text{ReLU}(z)^2]=\frac12\text{Var}(z)$，砍掉一半方差
- 补因子 2：$\frac12 n_\text{in}\text{Var}(W)=1\Rightarrow\text{Var}(W)=\frac{2}{n_\text{in}}$

写不出 $\text{Var}(z)=n_\text{in}\text{Var}(W)\text{Var}(x)$，或不知道那个 2 来自 ReLU 砍半。

</details>

<details>

<summary>Q15. 残差为什么能破梯度消失？给出反向传播的式子。</summary>

- $L$ 块串联：$\frac{\partial x_L}{\partial x_0}=\prod_l(I+\frac{\partial F_l}{\partial x})$
- 展开含一个**纯恒等项**（再加各阶交叉项）→ 梯度至少有一条不衰减通路
- 即使 $\partial F/\partial x\to0$，恒等项仍把梯度无损传回浅层

只说"加了 skip"，写不出雅可比连乘里那个恒等项为什么救梯度。

</details>

<details>

<summary>Q16. 残差缩放（$1/\sqrt N$ / LayerScale / ReZero）解决什么？怎么做？</summary>

- 解决：深残差网分支无节制灌入残差流 → 方差线性膨胀、初期不稳
- $1/\sqrt N$：分支乘 $1/\sqrt N$ 使 $N$ 层累加方差仍 $O(1)$；LayerScale：可学 per-channel $\lambda$（init 极小）；ReZero：可学标量 $\alpha$ init 0（起点精确恒等）
- 统一直觉：让分支起点接近 0、网络从恒等起步，梯度良态

只记住名字，说不出"让残差分支起点≈恒等"这个共同机理。

</details>

<details>

<summary>Q17. GPT-2 把残差投影权重乘 $1/\sqrt{2N}$ 是干嘛？为什么是 $2N$ 不是 $N$？</summary>

- 控制残差流方差：每次往残差流写入按 $1/\sqrt{2N}$ 缩放，使累加后方差仍 $O(1)$
- $2N$：每个 transformer 层往残差流写**两次**（attn 输出投影 + FFN down 投影），$N$ 层共 $2N$ 次
- 作用在 attn/FFN 的输出投影（残差投影）权重的 init 上

以为是每层一次（$N$）；或不知道作用在"残差投影"这两个特定矩阵上。

</details>

<details>

<summary>Q18. QK-Norm 是什么？解决什么问题？</summary>

- 点积前对每个 head 的 $q,k$ 各做一次归一化（L2/RMS），再算 logits
- 解决 attention logits 随尺度爆炸 → softmax 饱和 / entropy collapse / 训练发散（大模型尤甚）
- 归一化后 $q,k$ 模长被钳住，logits 不爆；Gemma2 / Chameleon / ViT-22B 在用

只说"归一化 QK"，讲不出它防的是 logits 爆炸 / softmax 饱和。

</details>

<details>

<summary>Q19. DeepNorm 怎么训到 1000 层？它和 Pre/Post-LN 什么关系？</summary>

- 是**改良 Post-LN**：$x_{l+1}=\text{LN}(\alpha x_l+\text{Sublayer}(x_l))$，$\alpha=(2N)^{1/4}\gt1$ 放大残差
- 同时用增益 $\beta=(8N)^{-1/4}\lt1$ **缩小子层 init**，把每步模型更新量界住
- up-scale 残差（接近 Pre-LN 稳定）+ down-scale init（更新有界）→ Post-LN 质量 + 稳定到 1000 层

把 DeepNorm 当成 Pre-LN；或只记 $\alpha$ 不记"还要缩小 init"。

</details>

<details>

<summary>Q20. GroupNorm / InstanceNorm 和 BN/LN 是什么关系？什么时候用 GN？</summary>

- GN 把 channel 分 $G$ 组，每样本每组内归一化，**不依赖 batch**
- $G=1$ 退化为 LayerNorm，$G=C$ 退化为 InstanceNorm（GN 桥接二者）
- 用途：**小 batch 视觉**（检测 / 分割 batch=2~4，BN 崩、GN 稳）

不知道 GN 用 $G$ 桥接 LN 与 IN；或答不出小 batch 检测这个典型场景。

</details>

### L3高级题

<details>

<summary>Q21. internal covariate shift 是 BatchNorm 有效的真正原因吗？</summary>

- 不是（大概率）。这是原始论文的叙事，已被 Santurkar 2018 实证推翻
- 证据一：BN 后注入噪声**故意增大** ICS，训练几乎不受影响
- 证据二：BN 真正让 loss / 梯度的 Lipschitz 变小（地形更平滑）→ 支持大 LR
- SkipInit 进一步显示 BN 在残差网的作用是 init 时缩小分支、偏向恒等

还把 covariate shift 当公认原因；或不知道有实验直接证伪。

</details>

<details>

<summary>Q22. μP 是什么？为什么标准参数化下最优 LR 会随宽度漂移？</summary>

- SP 下宽度变则各层激活 / 更新尺度错配，宽度 $n\to\infty$ 要么爆炸要么进 lazy/kernel 区
- 于是最优 LR 随宽度变，小模型调的参不能直接用到大模型
- μP 把 init 方差 / LR / 输出乘子按 fan_in 缩放，使**特征更新在宽度上 $\Theta(1)$、最优 LR 宽度不变** → μTransfer（小宽度调、大宽度 zero-shot 迁移）

只知道"μP 能迁移超参"，讲不出 SP 为何漂移 / μP 缩放了什么。

</details>

<details>

<summary>Q23. 怎么不用任何归一化层训练深残差网络？（Fixup / NFNets）</summary>

- Fixup：分支**末层 init 0**（块起点恒等）+ 其他层乘 $L^{-1/(2m-2)}$ 下调 + 加可学标量偏置/乘子
- NFNets：Scaled Weight Standardization + 解析缩放残差块 $x+\alpha F(x/\beta)$ + Adaptive Gradient Clipping
- 共同点：把"控方差 + 大 batch 稳定"用 init / 显式手段补回，说明 norm 非必需

只说"调 init"，给不出"末层置 0 + 深度下调"或 NFNets 三件套的具体做法。

</details>

<details>

<summary>Q24. DyT（Dynamic Tanh）是什么？归一化是训练的必要条件吗？</summary>

- 观察：训练好的 LN 输入-输出曲线像被压扁的 $\tanh$（中间近线性、离群值饱和）
- DyT：$\gamma\odot\tanh(\alpha x)+\beta$，可学标量 $\alpha$ 替"除以 std"，**去掉求均值/方差的 reduction**
- 含义：归一化理论上非必要（其作用可被 squashing / init / 显式方差控制替代），但 LN/RMSNorm 仍是稳妥生产默认

把 DyT 说成"已取代归一化"（过头）；或讲不出它来自"LN≈tanh"这个观察。

</details>

<details>

<summary>Q25. 给你一个非常深的网络（几百到上千层），你怎么组合 init / 归一化 / 残差让它稳定训练？</summary>

- **残差**必须有（恒等梯度路），分支起点压向恒等：末层 init 0 或 LayerScale/ReZero、按深度乘 $1/\sqrt N$（GPT-2 的 $1/\sqrt{2N}$）
- **归一化** Pre-LN/RMSNorm（免 warmup、稳）+ **别漏 final LN**；要 Post-LN 质量则上 DeepNorm（放大残差 + 缩小 init，可达 1000 层）
- **init** 按激活选 Xavier/Kaiming 做方差保持，深度感知下调；想免归一化走 Fixup/NFNets
- **超参**用 μP 在小宽度调好再迁移；attention 不稳加 QK-Norm；全程 fp32 reduction
- 核心暗线：**让网络从"接近恒等 + 方差守恒"的良态起点出发**

只报单一技巧；或忽略 final LN / 残差缩放 / 深度感知 init 的组合拳与"恒等起步"这条主线。

</details>

## §A 附录：sanity check

本 tutorial 的代码在 [`docs/tutorials/code/normalization.py`](code/normalization.py) 有最小可跑版本（纯 PyTorch、CPU 几秒、6 个 `assert`，覆盖 [a]–[f]）。它应满足以下关键不变量：

1. **[a] LayerNorm 自实现 == `nn.LayerNorm`**：用总体方差（`unbiased=False`，除以 $d$）、$\epsilon$ 放根号内，仿射后应与 PyTorch 在浮点误差内逐元素相等（`atol≈1e-5`）。
2. **[b] RMSNorm 自实现 == `nn.RMSNorm`，且 RMSNorm 对平移不变性与 LN 不同**：给输入整体加常数 $c$，**LayerNorm 输出几乎不变**（它减均值 → re-centering 不变），而 **RMSNorm 输出会变**（它只 re-scale、不减均值）——这正是 §4 "RMSNorm 没有 re-centering" 的可执行验证。
3. **[c] BatchNorm train$\ne$eval**：train 模式用 batch 统计（输出每特征近似零均值）并把 running mean 从 0 推离；切到 eval 用 running stats，对同一输入输出明显不同——验证 §2.2 的双路径。
4. **[d] Post-LN 把参数梯度堆在顶层（top-heavy），Pre-LN 跨深度均衡**：搭一摞 48 层同构残差块（末尾接一个 linear head——避免 loss 直接作用在 Post-LN 的归一化输出上产生"梯度被压成 0"的假象），一次 forward+backward，量每个块 Linear 权重梯度的**顶/底比值**（last/first）。Pre-LN $\approx0.40$（均衡、略偏底），Post-LN $\approx2.35$（$\gt1$，梯度堆在靠近输出的顶层）——正是 §5.2 Xiong "Post-LN 顶层梯度偏大 → 需 warmup" 的可执行印证。**用顶/底比值而非绝对梯度，因为它不受 loss 选择 / 输出归一化混淆**（这是个隐蔽的实验设计坑）。
5. **[e] Kaiming 保二阶矩、Xavier-for-ReLU 衰减**：单位二阶矩输入（$\mathbb{E}[x^2]=1$）过 `Linear+ReLU`，量 **post-ReLU 的二阶矩 $\mathbb{E}[y^2]$**（喂给下一层的信号能量，也是 He 推导真正传播的量）：Kaiming（$\sqrt{2/n_\text{in}}$）后 $\mathbb{E}[y^2]\approx1$（守恒），Xavier-for-ReLU（$\sqrt{1/n_\text{in}}$）后 $\approx0.5$（每层砍半 → 深了消失）——验证 §8.3 的因子 2。注意测的是 $\mathbb{E}[y^2]$ 而非 $\text{Var}(y)$：ReLU 使输出均值非零，故 $\text{Var}(y)=1-1/\pi\approx0.68$，而逐层守恒 / 传播的量是二阶矩 $\mathbb{E}[y^2]$。
6. **[f] GPT-2 残差缩放 $1/\sqrt{2N}$ 把残差流方差界住**：往残差流累加 $N$ 次 $O(1)$ 更新，不缩放则方差随深度**线性增长**，乘 $1/\sqrt{2N}$ 后方差**有界**——验证 §8.4。

下面是几段示意代码（CPU 可跑、Chinese 注释、与上面脚本逻辑一致）。

**(a) LayerNorm + RMSNorm 从零实现，并与 PyTorch 对齐：**

```python
import torch
import torch.nn as nn

def layernorm_from_scratch(x, weight, bias, eps=1e-5):
    """沿最后一维归一化。x: [..., d]。总体方差(unbiased=False)，eps 放根号内。"""
    mean = x.mean(dim=-1, keepdim=True)                       # [..., 1] 均值
    var  = x.var(dim=-1, unbiased=False, keepdim=True)        # [..., 1] 总体方差(/d)，与 torch 一致
    return (x - mean) / torch.sqrt(var + eps) * weight + bias # 标准化 + 仿射

def rmsnorm_from_scratch(x, weight, eps=1e-6):
    """只按 RMS 缩放：不减均值、无 bias。x: [..., d]。"""
    ms = x.pow(2).mean(dim=-1, keepdim=True)                  # [..., 1] 均方
    return x / torch.sqrt(ms + eps) * weight                  # 除以 RMS，再逐通道缩放

d = 64
x = torch.randn(8, d)                                         # [B, d]
ln  = nn.LayerNorm(d, eps=1e-5)                              # 默认仿射 weight=1, bias=0
assert torch.allclose(layernorm_from_scratch(x, ln.weight, ln.bias), ln(x), atol=1e-5)

rms = nn.RMSNorm(d, eps=1e-6)                                # 需 torch>=2.4
assert torch.allclose(rmsnorm_from_scratch(x, rms.weight), rms(x), atol=1e-5)

# RMSNorm 对整体平移不不变，LayerNorm 不变：
c = 5.0
assert (ln(x + c) - ln(x)).abs().max() < 1e-4                              # LN re-centering 不变
assert (rmsnorm_from_scratch(x + c, rms.weight) - rmsnorm_from_scratch(x, rms.weight)).abs().max() > 1e-2
```

**(b) 一摞深残差块，比较 Pre-LN 与 Post-LN 跨深度的梯度范数：**

```python
import torch
import torch.nn as nn

class ResidualStack(nn.Module):
    """深残差栈 + 末尾 linear head。Pre-LN: h=h+Linear(LN(h)); Post-LN: h=LN(h+Linear(h))。
       head 很关键：否则 Post-LN 输出已被 LN 归一化，loss=mean(out^2) 近似常数会把所有梯度压成 0（混淆）。"""
    def __init__(self, depth, d, pre_ln=True):
        super().__init__()
        self.pre_ln = pre_ln
        self.lns  = nn.ModuleList(nn.LayerNorm(d) for _ in range(depth))
        self.lins = nn.ModuleList(nn.Linear(d, d) for _ in range(depth))
        self.head = nn.Linear(d, d)                           # 末尾投影，去掉输出归一化的混淆

    def forward(self, h):                                     # h: [B, d]
        for ln, lin in zip(self.lns, self.lins):
            h = h + lin(ln(h)) if self.pre_ln else ln(h + lin(h))
        return self.head(h)

def block_grad_topheavy(depth, d, pre_ln):
    """一次 forward+backward，返回 last/first 块权重梯度范数之比（>1 = 梯度堆在顶层）。
       该比值对 loss 选择稳健，正是 Xiong 2020 的论断。"""
    torch.manual_seed(0)                                      # 两种放置用同一 init，公平对比
    stack = ResidualStack(depth, d, pre_ln=pre_ln)
    stack(torch.randn(16, d)).pow(2).mean().backward()       # 标量 loss
    gn = [lin.weight.grad.norm().item() for lin in stack.lins]
    return gn[-1] / gn[0]

r_pre  = block_grad_topheavy(48, 64, pre_ln=True)            # 期望 ≈0.40（均衡）
r_post = block_grad_topheavy(48, 64, pre_ln=False)          # 期望 ≈2.35（top-heavy，>1）
# Post-LN 顶/底比 >1：梯度堆在靠近输出的顶层 -> 必须 warmup。Pre-LN 跨深度均衡。
```

**(c) Kaiming 方差保持（vs 给 ReLU 误用 Xavier）：**

```python
import torch
import torch.nn.functional as F

fan_in, fan_out, N = 512, 512, 4096
x = torch.randn(N, fan_in)                                   # E[x^2]=1 (单位二阶矩) [N, fan_in]
W_kaiming = torch.randn(fan_out, fan_in) * (2.0 / fan_in) ** 0.5   # He: std=sqrt(2/fan_in)
W_xavier  = torch.randn(fan_out, fan_in) * (1.0 / fan_in) ** 0.5   # 给 ReLU 误用 Xavier: std=sqrt(1/fan_in)
# 测 post-ReLU 二阶矩 E[y^2]（喂下一层的信号能量、He 推导传播的量），不是 Var(y)
ms_kaiming = F.relu(x @ W_kaiming.t()).pow(2).mean()        # ReLU 砍半，2/fan_in 补回 -> E[y^2]≈1
ms_xavier  = F.relu(x @ W_xavier.t()).pow(2).mean()         # 没补因子 2 -> E[y^2]≈0.5，逐层砍半
# 期望 E[y^2]_kaiming≈1（守恒）、E[y^2]_xavier≈0.5（深了指数消失）。
# 注：Var(y)≈0.68（ReLU 使均值非零），逐层守恒的是二阶矩 E[y^2] 而非 Var。
```

运行 `python docs/tutorials/code/normalization.py` 的真实输出（CPU，纯 PyTorch，含 [a]–[f] 六个 `assert` 的汇总）：

```text
[a] LayerNorm from scratch vs nn.LayerNorm: max|Δ| = 2.38e-07  OK
[b] RMSNorm vs nn.RMSNorm: max|Δ| = 2.38e-07; mean-shift LN |Δ|=9.54e-07 (~0, re-centers) vs RMS |Δ|=3.05e+00 (>0, only re-scales)  OK
[c] BatchNorm train!=eval: max|Δ| = 8.37e+00; running_mean moved 0.701 from 0; train per-feature mean = 8.15e-08 (~0)  OK
[d] per-block weight-grad top/bottom ratio (last/first over 48 blocks): Pre-LN=0.40 (balanced)  Post-LN=2.35 (top-heavy, >1)  -> Post-LN piles gradient near the output, needs warmup  OK
[e] post-ReLU second moment E[y^2] (input E[x^2]=1): Kaiming = 1.000 (~1, preserved)  Xavier = 0.499 (~0.5, halves per layer)  OK
[f] residual-stream var growth over 50 blocks: unscaled ×51.1  vs  1/sqrt(2N)-scaled ×1.50  OK

all normalization / residual / init sanity checks passed ✓
```

> ✅ **读数解释**
> - **[a]/[b]** 自实现 LN/RMSNorm 与 PyTorch 一致（max$|\Delta|$=2.4e-7）；平移测试里 LN 几乎不变（9.5e-7）、RMSNorm 明显变化（3.05），印证 §4 "RMSNorm 只 re-scale 不 re-center"。
> - **[c]** BatchNorm running_mean 从 0 被推到 0.70、train 模式每特征均值≈0，且 train≠eval，印证 §2.2 双路径。
> - **[d]** 块权重梯度的顶/底比（last/first）：Pre-LN $0.40$（均衡），Post-LN $2.35$（$\gt1$，梯度堆在顶层），印证 §5.2 "Post-LN 顶层梯度偏大 → 需 warmup"（加 head 去掉了输出归一化的混淆）。
> - **[e]** post-ReLU 二阶矩 Kaiming $\mathbb{E}[y^2]=1.00$（守恒）、Xavier-for-ReLU $=0.50$（每层砍半），印证 §8.3 的因子 2。
> - **[f]** 残差流方差不缩放 50 层后 ×51（线性涨）、$1/\sqrt{2N}$ 缩放后 ×1.5（有界），印证 §8.4 GPT-2 技巧。

## 📚 参考文献

- **BatchNorm** — Ioffe & Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, arXiv 1502.03167 (2015), ICML 2015.
- **LayerNorm** — Ba, Kiros & Hinton, *Layer Normalization*, arXiv 1607.06450 (2016).
- **RMSNorm** — Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv 1910.07467 (2019), NeurIPS 2019.
- **How Does BN Help Optimization (covariate-shift debunk)** — Santurkar et al., *How Does Batch Normalization Help Optimization?*, arXiv 1805.11604 (2018), NeurIPS 2018.
- **Pre-LN vs Post-LN (gradient argument)** — Xiong et al., *On Layer Normalization in the Transformer Architecture*, arXiv 2002.04745 (2020), ICML 2020.
- **DeepNorm / DeepNet** — Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*, arXiv 2203.00555 (2022).
- **Kaiming init** — He et al., *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, arXiv 1502.01852 (2015), ICCV 2015.
- **Xavier / Glorot init** — Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010（PMLR proceedings，**无 arXiv id**，待引用核查）.
- **ResNet (residual learning)** — He et al., *Deep Residual Learning for Image Recognition*, arXiv 1512.03385 (2015), CVPR 2016.
- **Residuals as ensembles** — Veit et al., *Residual Networks Behave Like Ensembles of Relatively Shallow Networks*, arXiv 1605.06431 (2016), NeurIPS 2016.
- **Fixup** — Zhang et al., *Fixup Initialization: Residual Learning Without Normalization*, arXiv 1901.09321 (2019), ICLR 2019.
- **ReZero** — Bachlechner et al., *ReZero is All You Need: Fast Convergence at Large Depth*, arXiv 2003.04887 (2020), UAI 2021.
- **LayerScale / CaiT** — Touvron et al., *Going deeper with Image Transformers*, arXiv 2103.17239 (2021), ICCV 2021.
- **SkipInit** — De & Smith, *Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks*, arXiv 2002.10444 (2020), NeurIPS 2020.
- **μP / μTransfer** — Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, arXiv 2203.03466 (2022), NeurIPS 2021.
- **NFNets (normalizer-free)** — Brock et al., *High-Performance Large-Scale Image Recognition Without Normalization*, arXiv 2102.06171 (2021), ICML 2021.
- **GroupNorm** — Wu & He, *Group Normalization*, arXiv 1803.08494 (2018), ECCV 2018.
- **WeightNorm** — Salimans & Kingma, *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks*, arXiv 1602.07868 (2016), NeurIPS 2016.
- **QK-Norm** — Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245 (2020), EMNLP 2020 Findings.
- **DyT (Dynamic Tanh)** — Zhu et al., *Transformers without Normalization*, 2025（CVPR 2025；arXiv 2503.10622，**arXiv id 以引用核查为准**）.
- **T-Fixup** — Huang et al., *Improving Transformer Optimization Through Better Initialization*, ICML 2020（PMLR proceedings；无广泛使用的独立 arXiv id，按会议出处引用）.
- **GPT-2 (residual-projection $1/\sqrt{2N}$ init)** — Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI 技术报告 (2019)（**无 arXiv id**）.
