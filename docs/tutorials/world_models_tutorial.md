## §0 TL;DR Cheat Sheet

> 💡 **一句话** — world model 的面试题从来不是"谁家的世界模型更强"，而是三个机制问题：**模型保留什么信息、怎样预测动作后果、这些预测如何进入决策**。三条技术谱系给出三套答案，而且它们互相重叠。

1. **三条谱系**：(1) latent dynamics / model-based RL——学一个能在里面 rollout 的紧凑状态空间；(2) predict-in-representation-space（JEPA）——只预测编码器的输出，不重建像素；(3) 生成式视频与交互式模拟器——直接生成可交互的未来观测。
2. **谱系不是阵营**：Dreamer 4 同时是生成式动态模型和 imagination RL；Cosmos 3 横跨理解、生成与动作；视频扩散模型同样在 VAE latent 里工作——"latent" 不是 RSSM/JEPA 的专有词。
3. **状态**：环境真实状态 $x_t$ 不可观测，模型状态 $s_t=(h_t,z_t)$ 是它的替身。RSSM 递推全篇固定写成 $h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})$：$h_t$ 提供跨时刻记忆，$z_t$ 承载随机性与新信息。
4. **prior vs posterior**：$p_\theta(z_t\mid h_t)$ 用于预测，$q_\phi(z_t\mid h_t,o_t)$ 吸收当前观测。**imagination 里只能用 prior**——调用后验等于偷看还没发生的观测。
5. **KL balancing**：$\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$。两项都是 $q\Vert p$ 方向，分开的是**谁被停梯度**，不是 KL 方向，更不是"把系数调小"。
6. **决策接口四种**：PlaNet 在潜空间在线规划；Dreamer 在 imagination 里训 actor-critic；MuZero 用 MCTS 且完全不重建观测；TD-MPC2 decoder-free 地学动态+奖励+价值供 MPC 使用。
7. **JEPA 公共结构**：$P_\theta(E_\phi(x),m)\to\mathrm{sg}(E_{\bar\phi}(y))$，非对称 predictor + EMA 目标分支 + 停梯度。这套组合**不是不塌缩定理**：常数 encoder 配常数 predictor 依然是退化解。
8. **生成式路线的关键不是画质而是动作接口**：latent action model 的小 codebook、相机位姿、文本条件、执行器动作，语义完全不同；无动作视频只提供先验，要用于动作选择还得补上接口和证据。
9. **数字带口径**：Genie 3 的 720p / 24 FPS / 数分钟一致性全部出自 2025-08-05 的官方博客，无论文、无公开参数量（11B 是 Genie 1 的）；GameNGen 的速度按版本报——v1 摘要写 ">20 FPS"，v2 正文是**单块 TPU-v5、4 步 DDIM 下的 20 FPS**，29.4 dB 是**下一帧预测**的 PSNR，不是长程 rollout 的保真度，更不是物理正确性证据。
10. **面试杀手题**：KL balancing 为什么不等于调小系数、V-JEPA 2-AC 的 zero-shot 到底 zero 了什么、以及"视频逼真能否证明懂物理"——Physics-IQ 给的是实验性答案（跨模型 $r=-0.46$、$p=0.249$，未检出显著相关），不是"两者无关"的定论。

---

## §1 一个问题，三条谱系

### 1.1　world model 是什么

world model 是智能体**内部学到的环境结构与动态模型**：给定至今的观测与动作，它能预测"如果我这样做，接下来会怎样"。这个定义刻意不规定输出形式，因为**必要的输出取决于用途**——要在模型里做 RL，就得有办法在想象里算回报并截断 episode（Dreamer 一类的做法是学奖励头与 continuation 头，奖励函数或终止规则已知时直接用已知的那份也行）；要做视觉目标导航，只需能比较目标表征的距离；要当可玩的模拟器，就必须生成人能看的观测。

所以"它算不算 world model"多半是假问题。真问题是三个：**保留了什么信息、预测的是哪个量、这个量能支撑哪一种决策**。

### 1.2　三条谱系（会重叠）

| 谱系 | 预测什么 | 典型决策接口 | 代表工作 |
| --- | --- | --- | --- |
| latent dynamics / MBRL | 紧凑潜状态的转移，通常带奖励与 continuation | 潜空间规划、imagination 内 actor-critic、MCTS | PlaNet、Dreamer 系列、MuZero、TD-MPC2 |
| predict-in-representation-space | 冻结或 EMA 编码器输出的表征 | 表征空间 MPC（最小化与目标表征的距离） | I-JEPA、V-JEPA 系列、DINO-WM |
| 生成式视频 / 交互式模拟器 | 未来观测本身（像素或视频 token） | 动作条件生成 + 人/策略在环交互 | Genie 系列、GameNGen、Cosmos、Matrix-Game |

> ⚠️ **按"预测哪个量"分类，不要按门派分类** — Dreamer 4 既是 transformer 生成式动态模型也是 imagination RL；Cosmos 3 同一个模型跨语言、图像、视频、音频与动作；视频扩散模型在 VAE latent 里生成，和 RSSM 的 latent 只是同名。用门派分类会立刻自相矛盾，用"预测哪个量、怎么进决策"分类不会。

### 1.3　入选标准

「优先选择奠定关键机制、代表重要技术路线，或由主要研究团队公开且与具身决策密切相关的工作；同等代表性下优先开放权重。最新发布用于说明方向，其成熟度另行说明。」

被排除的工作与逐条理由见 §A.2——面试里被问到没收录的名字时，知道它为什么被排除，比多背一个名字有用。

### 1.4　名词消歧（面试里最容易各说各话的地方）

- **RL 的环境模型**：$p(s_{t+1}, r_{t+1} \mid s_t, a_t)$，服务于规划与想象。谱系一。
- **LeCun 意义上的世界模型**：在抽象表征空间预测，明确拒绝像素级重建。他的立场文章发表在 OpenReview（v0.9.2，2022-06-27），**没有 arXiv 编号**，引用时别硬编一个。谱系二。
- **"世界模拟器"**：能按动作条件生成可交互视频的生成模型。谱系三。
- **LLM 内部的"世界模型"**：指语言模型激活里是否编码了棋盘、地图这类外部状态，问的是"表征里有没有"，不是"能不能预测动作后果"。另一个问题，不在讨论范围内。
- **代码执行意义上的 "world model"**：指预测程序执行状态的模型，和这里讨论的世界模型只是**术语碰撞**，不是同一条线。

还有两组必须分清的符号与口径：

- **$x_t$ vs $s_t$**：$x_t$ 是环境真实状态，不可观测，模型不承诺恢复它；$s_t$ 是模型自己的状态。说"潜状态就是真实状态"是错的。
- **静态表征 vs 可交互动态**：I-JEPA 是静态图像表征方法，**不直接构成可交互的动态模型**；把它说成"图像版世界模型"会在追问下崩掉。

---

## §2 Convention（全文统一）

### 2.1　时间约定

全篇固定一条时间线：**收到 $o_t, r_t, c_t$ → 形成 $s_t$ → 执行 $a_t$ → 收到 $o_{t+1}, r_{t+1}, c_{t+1}$**。

因此 $r_{t+1}$ 是动作 $a_t$ 的即时奖励，$c_{t+1}\in\{0,1\}$ 是"episode 在 $t+1$ 尚未结束"的标志。论文之间下标差一格，绝大多数来自这条约定不同——**先对齐约定，再比公式系数**。

### 2.2　符号表

| 符号 | 含义 | 注意 |
| --- | --- | --- |
| $x_t$ | 环境真实状态 | 不可观测；模型不承诺恢复 |
| $o_t$ | 观测（图像、本体感觉等） | 部分可观测是常态 |
| $a_t$ | 在 $s_t$ 之后执行的动作 | 语义随接口而变（见 §5.1） |
| $r_{t+1}, c_{t+1}$ | 奖励与 continuation | 按 §2.1 的下标 |
| $s_t=(h_t,z_t)$ | 模型状态 | RSSM 的确定性 + 随机两部分 |
| $p_\theta(z_t\mid h_t)$ | prior | 无观测，用于预测与 imagination |
| $q_\phi(z_t\mid h_t,o_t)$ | posterior | 吸收观测，**不能**在 imagination 里调用 |
| $E_\phi, E_{\bar\phi}, P_\theta$ | 在线 encoder、EMA target encoder、predictor | JEPA 三件套 |
| $\mathrm{sg}[\cdot]$ | stop-gradient | 针对计算路径，不是冻结网络 |

### 2.3　预测头：哪些是必要的

Dreamer 一类的 world model 通常带三个头：**观测**（重建或 token 预测）、**奖励**、**continuation**。后两个是为了"在模型里做 RL"——**在这套 Dreamer 实现里**，想象里的回报由学到的奖励头给出，episode 的截断由学到的 continuation 头给出，去掉哪个都算不下去。

这句必要性只对这套实现成立，别升级成普遍规律：奖励函数或终止规则已知时（很多仿真与游戏环境就是如此），直接调用已知的那份同样能在想象里算回报、截断 episode。观测重建更不是"在模型里做 RL"的必要条件——MuZero 与 TD-MPC2 都不重建观测照样规划（§3.8）。

纯视频生成模型**未必需要奖励与 continuation**：它的用途是生成可看的未来。把"没有奖励头"当成缺陷，或者把"有奖励头"当成世界模型的定义，都会答错。

### 2.4　数字口径

每个外部数字第一次出现时，正文里都交代**任务、指标、模型版本、设置**；速度数字注明硬件；提升区分**百分点**与**相对百分比**。开放状态按发布物分别说——论文、代码、权重、数据是四件事，**计划发布不写成已开放**。只有官方博客或新闻稿支撑的说法，一律标注为"仅博客"。

---

## §3 谱系一：latent dynamics 与 model-based RL

### 3.1　起点：在梦里训练策略

Ha & Schmidhuber 2018（*World Models*，arXiv 1803.10122）把结构拆成三块：VAE 把每帧压成低维 $z$；MDN-RNN 在 $z$ 序列上预测下一步的混合高斯分布；一个**线性** controller 从 $(z,h)$ 读出动作。关键一步在最后——在论文的 **VizDoom** 实验里，controller 完全在 RNN 生成的 rollout（论文称 "dream"）里训练，再放回真实环境；**CarRacing 那个实验不是这样**，它的 controller 直接在真实环境里训练。"策略完全在梦里训练"只对前一个实验成立。

这三块定下了此后的分工：**表征**（把观测压成可预测的量）、**动态**（在这个量上外推）、**决策**（从状态读出动作）。后续工作要么改其中一块，要么把它们合并。

### 3.2　RSSM：为什么状态要一半确定、一半随机

PlaNet（Hafner et al.，arXiv 1811.04551）提出 Recurrent State-Space Model：

$$h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1}),\qquad z_t\sim p_\theta(z_t\mid h_t)\ \text{(prior)},\qquad z_t\sim q_\phi(z_t\mid h_t,o_t)\ \text{(posterior)}$$

$h_t$ **确定性**：它提供不经过随机重采样的递推记忆通道。纯随机状态每步都要重采样，长 horizon 上信息被采样噪声冲掉。别把"确定性"读成"无损"——固定宽度的 GRU 状态照样会压缩和遗忘，它只是不再被每步的采样噪声冲刷。这也不是说别的架构不能有记忆——transformer 用 attention 直接回看历史，记忆同样存在，只是不压在一个递推向量里。

$z_t$ **随机**：环境本身随机、观测不完全，$z_t$ 负责表达"这一步真正的新信息"。反过来也别走极端，确定性动态在确定性环境里可以工作得很好；$z_t$ 的价值随环境随机性与部分可观测程度上升。

prior 与 posterior 的分工是整条谱系的核心：**posterior 看得到观测，prior 看不到**。训练时两者都在，rollout 时没有未来观测，只剩 prior。

### 3.3　目标函数：先看没有系数的负 ELBO

$$\mathcal L_t=\underbrace{-\log p_\theta(o_t\mid h_t,z_t)}_{\text{重建 NLL}}+\underbrace{D_{KL}\big(q_\phi(z_t\mid h_t,o_t)\,\Vert\,p_\theta(z_t\mid h_t)\big)}_{\text{一致项}},\qquad z_t\sim q_\phi(z_t\mid h_t,o_t)$$

这是标准序列 VAE 负 ELBO 的**单步、单样本估计项**：重建那一项本该是 $\mathbb E_{z_t\sim q_\phi}[\cdot]$，这里用一个 $z_t$ 样本代替期望；完整目标还要对整条序列的 $t$ 求和。写全是 $\sum_t \mathbb E_{q}\left[\mathcal L_t\right]$，**没有任何权重**。它读作两句话：解码器要能从 $(h_t,z_t)$ 还原观测；prior 要能在看不到观测的情况下猜到 posterior 猜的东西。

后面所有的 KL 权重、KL balancing、free bits 都是**工程改动**，改的是两个目标在优化过程里的相对影响，不是新的概率推导。

> ⚠️ **KL 项压着两个人** — 一致项同时作用于 $q$ 和 $p$。只挂一个系数时优化器有条廉价出路：让 posterior 退化成 prior，KL 归零、重建全靠 $h_t$——posterior collapse。§9 的 toy 实验 1 在解析可算的设定里把这个塌缩跑出来。

### 3.4　KL balancing 与 free bits

DreamerV2（Hafner et al.，arXiv 2010.02193）把 KL 拆成两份：

$$\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$$

**两项都是 $q\Vert p$ 方向**。balancing 不动 KL 的方向，动的是梯度往哪走：第一项停 $q$ 的梯度，只让 prior 去追 posterior（dynamics learning）；第二项停 $p$ 的梯度，只让 posterior 向 prior 妥协（representation regularisation）。取 $\beta_{dyn}\gg\beta_{rep}$ 等于说：**先让动态学会预测，再让表征为了好预测让步**。

`sg` 针对的是**计算路径**。$q$ 与 $p$ 共享 $h_t$，$h_t$ 的梯度照样从另一条路径流回来——实现上是用 `probs.detach()` 造一个新分布，不是把某块网络 `requires_grad_(False)`。

**free bits** 是另一件事：$\max(\tau, D_{KL})$，把 KL 在下限 $\tau$ 处截平，KL 已经够小时不再继续压，省下的容量留给重建。它作用在**联合 KL**——先对所有 latent 组求和再截断，不是对每个类别概率截断。

DreamerV3 论文 v1（Hafner et al. 2023，arXiv 2301.04104，式 4 与表 W.1）取 $\beta_{dyn}=0.5$、$\beta_{rep}=0.1$；同一篇的 v2 正文写成 $\beta_{dyn}=1$、$\beta_{rep}=0.1$。系数统一取 v1，**报系数时一定要说是哪个版本**。

### 3.5　动作从哪来：在线规划 vs imagination 内的 actor-critic

PlaNet 不训练策略网络。每一步它在**潜空间**在线规划，用的是 **CEM**：从一个高斯分布采一批动作序列，用 prior 向前 rollout、用奖励头打分，取回报最高的一批 elite 序列**重新拟合**这个分布，如此迭代若干轮；最后执行**末轮分布均值**的第一个动作，下一步重来。注意它**不是**"采一次、执行最好的那条样本"——迭代重拟合才是 CEM 的主体。规划全程不解码图像，所以采样数可以开得很大。

PlaNet 还有 **latent overshooting**：不只让一步预测的 prior 贴近 posterior，还让**多步**展开的 prior 贴近对应时刻的 posterior。它是**多步一致性信号**，不是"把训练 horizon 拉长"——单步准确的模型复合几十步后照样会漂到荒谬状态，overshooting 直接惩罚这种漂移。

Dreamer（Hafner et al.，arXiv 1912.01603）换了决策接口：**在 imagination 里训练 actor-critic**。从重放缓冲取真实片段，用 posterior 得到起点 $s_t$，之后只用 prior 向前展开 $H$ 步，在这些想象状态上更新策略与价值。策略更新不再消耗真实交互，样本效率大幅提升；代价是**策略只见过模型认为会发生的事**。

λ-return 按 §2.1 的时间约定写：

$$G_t^\lambda=\hat r_{t+1}+\gamma\hat c_{t+1}\big[(1-\lambda)v(s_{t+1})+\lambda G_{t+1}^\lambda\big],\qquad G_H^\lambda=v(s_H)$$

critic 的回归目标要停梯度。DreamerV3 v1 表 W.1 取 $H=15$，正文的预测序列长度 $T=16$：**一律以 $H$ 计动作转移数，状态序列含起点因而有 $H+1$ 个**。差的这个 1 不是笔误，是"数状态还是数转移"。

> ⚠️ **actor 梯度按版本写，不要合成一个公式** — Dreamer 用**路径梯度**：回报对动作可微，梯度穿过学到的动态传回 actor。DreamerV3 的 actor loss 是 **score-function（REINFORCE 式）**形式，配 return normalization 与熵正则。另外"更新 actor/critic 时冻结 world model 参数"指的是不改动态模型的参数，**不等于**对整个 imagination `no_grad()`——路径梯度必须穿过动态图才存在。

### 3.6　离散潜变量与 DreamerV3 的三件工程

DreamerV2 把 $z_t$ 从高斯换成**一组分类分布**（若干组、每组若干类的 one-hot），前向采样、反向用 straight-through 近似。这是**有偏**的梯度估计——它把不可导的采样当成恒等映射来传梯度。

给分类概率混入 **1% 均匀分布**是**DreamerV3**（v1 正文与附录 C）的改动，不是 DreamerV2 的，报版本时别记串；作用是避免某一类概率被压到 0 导致 $\log$ 爆炸。**均匀混合与 free bits 是两件独立机制**：一个防概率退化，一个防 KL 被压过头。

除了这个均匀混合，DreamerV3 又加了三件互相独立的工程：

1. **symlog**：$\mathrm{symlog}(x)=\mathrm{sign}(x)\log(1+\lvert x\rvert)$，逆变换是 symexp。它把跨数量级的信号压进可训练范围，对称且在 0 附近近似恒等。
2. **twohot**：把连续标量编码到一组固定桶上——找到它落在哪两个相邻桶之间，按距离把权重 1 分给这两个桶。回归于是变成分类，期望值可精确恢复。
3. **return normalization**：取一个批次回报的**第 95 与第 5 分位之差** $S$ 做 EMA，除以 $\max(1,S)$。下限 1 是要害：稀疏奖励任务里回报几乎全为 0，$S$ 会很小，没有下限就会把近零回报的噪声放大成巨大梯度。**这不是普通的 advantage 标准化**（减均值除标准差）——它只做尺度、带下限、跨批次平滑。

v1（式 10 之后的正文与附录 C）里 **reward 与 critic 都用 symlog twohot**：先 symlog 压缩，再 twohot 编码成分类回归。**不要把它推广到所有预测头**——观测/continuation 头是另一回事。

DreamerV3 最常被误引的是它的结论句：**同一套超参数**覆盖 150 多个任务。这句话说的是不必逐环境调参，**不是**一套权重通吃所有环境（每个任务仍各自训练）。它在 Minecraft 里从零收集到钻石，是**通过与环境交互**做到的，不是离线数据。

### 3.7　Dreamer 4：换掉骨架之后

Dreamer 4（arXiv 2509.24527）不再是 RSSM：**tokenizer + transformer 动态 + shortcut forcing**。别把 V3 的 RSSM/KL/分类潜变量描述套上去，那是另一套骨架。

它最吸睛的结论要连着限定条件一起背：在论文的**离线 Minecraft 设定**下取得钻石，训练流程包含行为克隆、奖励建模，以及在模型内部做 RL；论文报告的"单 GPU 实时"指的是**交互推理**，不是训练。

### 3.8　不重建也能规划：MuZero 与 TD-MPC2

MuZero（Schrittwieser et al.，arXiv 1911.08265）只学**规划要用的量**：奖励、价值、策略先验，配 MCTS 搜索。它完全不重建观测，也不要求潜状态能恢复 $x_t$。核心是**面向规划的预测充分性**——只要这些量在搜索树里预测得准，搜索就能选对动作。这是设计原则加实证结果，**不是**严格的 value-equivalence 定理。

TD-MPC2（Hansen et al.，arXiv 2310.16828）把同样的思路搬到连续控制：**decoder-free** 地学潜空间动态、奖励与价值，用它们做 MPC，不重建观测。论文的规模实验里，一个 317M 参数的单一 agent 覆盖 80 个任务。

> 💡 **不重建的代价** — 没有解码器就没有"模型在想什么"的直接可视化，调试只能靠间接指标（奖励预测误差、规划收益）。生成式路线正好相反：能看，但看着对不等于用着对（§7.1）。

### 3.9　模型误差怎么办：PETS 与 MBPO

这两篇常被一起提，解决的却不是同一个问题：

- **PETS**（Chua et al. 2018，arXiv 1805.12114）处理**不确定性表示**：用概率模型的 ensemble 同时表达环境噪声与数据不足带来的不确定性，规划时用轨迹采样把它传播下去。
- **MBPO**（Janner et al. 2019，arXiv 1906.08253）处理**误差累积**：不从头 rollout，而是从真实数据里的状态**分支**出很短的模型 rollout 再做 off-policy 更新，把模型偏差的累积压在可接受范围内。

一句话：PETS 说"我不确定"，MBPO 说"我只敢往前看几步"。

### 3.10　imagination 用什么生成：token 还是像素

IRIS（Micheli et al.，arXiv 2209.00588）用离散 tokenizer + transformer 在 token 序列上想象；DIAMOND（Alonso et al. 2024，arXiv 2405.12399）用**扩散** world model 生成图像级的想象轨迹，其实验支持一个具体结论：**视觉细节会影响控制回报**——被压掉的小物体和细边缘，往往正是奖励或碰撞的触发条件。

DayDreamer（Wu et al.，arXiv 2206.14176）是这条谱系落到真机的代表：在其论文的实验里，四足机器人**约一小时**真实交互就学会行走。这个数字属于那个实验，不能当作"model-based RL 一小时学会任何技能"。

---

## §4 谱系二：在表征空间预测（JEPA）

### 4.1　公共结构

LeCun 的立场文章（OpenReview v0.9.2，2022-06-27）主张：预测不该发生在像素上，因为未来的大部分像素细节**不可预测也不值得预测**。JEPA 系列把这句主张变成一个可训练的结构：

$$P_\theta\big(E_\phi(x),\,m\big)\ \longrightarrow\ \mathrm{sg}\big[E_{\bar\phi}(y)\big]$$

$x$ 是上下文，$y$ 是目标，$m$ 是描述"要预测哪一块"的条件（掩码 token、位置等），$E_{\bar\phi}$ 是在线 encoder 的 EMA 副本，目标分支停梯度。每篇论文的差别在三处——**距离怎么算、掩码怎么造、目标怎么构造**——下面逐篇分开说，不要把某一篇的细节推广成"JEPA 都这样"。

### 4.2　I-JEPA：静态图像上的表征预测

I-JEPA（Assran et al. 2023，arXiv 2301.08243）在单张图像上做：从图像取若干**目标块**，另取一个上下文块，用非对称的 predictor 从上下文表征预测各目标块的表征。损失是 patch 表征之间的**平方 $L_2$**；目标分支来自 EMA encoder 且不反传；掩码是**多块**策略（目标块足够大、上下文块与目标不重叠）。

> ⚠️ **别给 I-JEPA 补 VICReg 项** — 它靠的是非对称 predictor + EMA 目标，不含方差/协方差正则项。把 VICReg 的三项损失安到 I-JEPA 头上是常见的记忆串台。另外，I-JEPA 是**静态图像表征方法**，它不接收动作、不预测时间演化，**不直接构成可交互的动态模型**。

### 4.3　V-JEPA：搬到视频

V-JEPA（Bardes et al. 2024，arXiv 2404.08471）把同一结构搬到视频：masked feature prediction，距离换成 **$L_1$**，同样是 stop-gradient + EMA target + predictor。它**不依赖负样本、不做像素重建、不需要预训练好的图像 encoder**——这三个"不"是它相对对比学习与重建式自监督的定位。

注意此时的预测仍然是 **action-free** 的：模型学的是**预测被遮挡的时空表征**（掩码预测，被遮挡的块可以在时间上位于上下文之前、之中或之后），而不是"给定过去、因果地预测未来"；里面更没有"我做了什么"。

### 4.4　collapse 到底靠什么避免

标准答案是一组机制：非对称结构（predictor 只在一侧）、EMA 的慢速目标分支、停梯度、掩码任务本身的难度。但要点在下一句：

> ⚠️ **停梯度与 EMA 不是"不塌缩定理"** — 常数 encoder 配常数 predictor 依然是损失极低的退化解。这套机制让训练动态**倾向于**避开塌缩，效果依赖初始化、EMA 系数、掩码难度与优化器设置。面试里把它说成"理论保证不塌缩"是错的；说成"实践中靠这几件事 + 监控表征方差"才对。§9.2 的 toy 里能看到：共享权重不加停梯度时，表征方差按几何级数衰减到 0；加了停梯度的 EMA 目标，梯度在对称点处为零，塌缩不再是吸引子——但从零初始化出发它也**停在塌缩解上**。

### 4.5　V-JEPA 2 与 2-AC：动作是在第二阶段进来的

V-JEPA 2（Assran et al. 2025，arXiv 2506.09985）的预训练仍然是 **action-free 的 masked prediction**（$L_1$、EMA 目标、停梯度）。它常被引用的 **77.3** 是 Something-Something v2 上 **attentive probe 的 top-1 准确率**——一个表征质量指标，**不是机器人成功率**。

动作在第二阶段进来：**V-JEPA 2-AC** 冻结预训练 encoder，只训练一个 **action-conditioned predictor**，输入包含机器人动作与末端状态。论文描述的 "<62 h unlabeled robot video" 指的是**没有任务语义标签**，不是没有动作记录——动作条件 predictor 必须有动作才能训。

它的规划接口是**表征空间的 MPC**：给一张目标图像，编码成目标表征；预测若干候选动作序列执行后的表征，最小化与目标表征的 **$L_1$ 距离**；用 CEM 搜索动作序列，执行第一个动作后重新规划。**目标表征不是奖励标签**——它是被最小化的距离的一端，没有任何回报回归。

> ⚠️ **zero-shot 到底 zero 了什么** — 指的是**新部署环境**：在 Franka 机械臂上，给定视觉目标做 reaching / grasp / pick-and-place，**不在部署实验室重新采数据**。它不证明"任意机器人、任意任务、任意长程分解都不用采数据"。答题时把这三个限定词说全。

### 4.6　V-JEPA 2.1：改了哪里、提升怎么拆

V-JEPA 2.1（arXiv 2603.14482）的增量有三条：对 **masked token 与 visible-context token 都施加预测监督**（context 项带自己的权重）；用**多层输出做 deep self-supervision**；扩展了图像/视频 tokenizer 与模型规模（小型号涉及蒸馏）。

机器人部分的提升必须拆开报：论文表 6 在**同一规划设置**下，grasp 从 60% 到 70%（**+10 个百分点**）；把规划设置改掉并放长 horizon 后达到 80%（**+20 个百分点**，此时**不再是纯表征比较**）。每项技能 10 个任务。把这两档混成"提升 20%"是典型的口径事故——既混了百分点与相对百分比，又混了两套设置。

### 4.7　DINO-WM 与一个常见张冠李戴

DINO-WM（Zhou et al. 2024，arXiv 2411.04983）在**冻结的 DINOv2 特征**上学动态，规划时以目标图像的特征作为目标。它证明的是：**表征可以是现成的，动态才是要学的那部分**——不必为了做 world model 重新训一个 encoder。

Navigation World Models（Bar et al. 2024，arXiv 2412.03572）经常被放进 JEPA 一节，其实**不属于这里**：它是在第一人称视频与动作上的 **conditional diffusion transformer**，预测的是**未来观测**，所以归谱系三（见 §5.2）。两者都做"以目标为条件的导航规划"，但预测的量不同，是 §1.1 三问里最关键的一问。

### 4.8　理论进展与它的边界

2026 年的一篇 JEPA 泛化理论（arXiv 2606.27014）把预训练误差与下游规划的 regret 联系起来，设置是：条件谱图 / 动作条件共现矩阵的**低秩分解**。结论只在这套假设下成立——**不是无条件保证"表征损失小则规划好"**。引用它时把设置一起说，否则等于用一篇论文的标题替代论证。

VL-JEPA（arXiv 2512.10942）把语言接进同一框架；这里只提方向，不引用它的任何具体数字。

### 4.9　这条谱系缺的是什么

和谱系一对照着看，缺口很清楚——**限定在上面那个规划实例上**（2-AC 的表征空间 MPC）：**没有奖励、没有 continuation、没有显式的不确定性**，决策全靠"与目标表征的距离"这一个量。这是那套接口的性质，不是整条谱系的定律：同一个表征上完全可以再挂奖励或不确定性估计。

因此它天然适合**目标条件**任务（把手移到这个视觉目标上），而需要长程回报或风险规避时就得自己补上缺的那几个量。别把"稀疏奖励"也算进排除项——稀疏奖励任务常常正好能写成目标条件形式，用目标条件 MPC 做反而顺手。

它也继承了一个评测缺口：probe 准确率衡量的是表征里有没有信息，**不衡量动作后果预测得对不对**。所以这条线的论文必须同时给两类证据——表征指标和闭环成功率，只给前者就只能说明表征好。

---

## §5 谱系三：生成式与交互式世界模型

### 5.1　动作接口：这条谱系真正的技术核心

视频生成模型天然不带动作。**怎么把"我做了什么"接进去**，是这条谱系最值钱的一段。

**latent action model（LAM）**：从相邻观测里**反推**一个"动作"。这里以 Genie 式的**离散 VQ** LAM 为例（DreamGen 用的是量化前的连续 embedding 当伪动作，见 §6）。VQ 瓶颈有三件事：

$$k=\arg\min_j\big\lVert E(o_t,o_{t+1})-e_j\big\rVert^2$$

编码器看 $(o_t,o_{t+1})$ 给出一个连续向量，在 codebook 里量化到最近的码 $e_k$；解码器从**当前信息与 $e_k$** 预测未来观测；训练损失含三项——预测项、codebook 项、commitment 项（straight-through 是通用 VQ 写法，§9.3 给代码）。

为什么 codebook 要小？**容量论据**：一个码至多携带 $\log_2 K$ bits，$K=8$ 时是 3 bits。瓶颈窄到只够表达"最能解释帧间变化的那几个方向"，模型就没有余量去把整帧的外观差异塞进去。

> ⚠️ **小 codebook 不保证学到的是动作** — 容量小只保证"留下的信息少"，不保证"留下的是执行器动作"，也不保证同一个码在不同场景里含义一致。arXiv 2506.15691 给了机制：**动作引起的变化与外生变化在竞争同一个瓶颈容量**，方差大的无关变化（光照、相机抖动、背景运动）可能被优先编码；在线性情形下这与 PCA 保留最大方差方向有直接联系。

Genie（Bruce et al. 2024，arXiv 2402.15391）是把 LAM 用成规模化产品的第一篇：**11B** 参数，三个组件——视频 tokenizer、**8 个码**的 latent action model、动态模型；生成用 **MaskGIT 式的并行掩码采样**，不是逐 token 的 GPT 式解码。这三个组件要分开讲，混成一个"Genie 是个视频模型"就答不出它为什么可交互。

除 LAM 外还有三类接口，**语义完全不同，不能互换**：

| 接口 | 输入什么 | 能支撑的决策 |
| --- | --- | --- |
| 文本条件 | 场景/事件描述 | 内容生成、场景构造；与执行器无对应 |
| 相机运动 / 位姿 | 视角轨迹 | 视点控制、空间一致性；不是"身体做了什么" |
| 执行器动作 | 关节/末端/方向盘指令 | 直接对应可执行控制 |
| LAM 码 | 从视频反推的离散码 | 可交互，但语义需另行验证 |

> ⚠️ **无动作视频只提供先验** — 它能学到"世界通常怎么演化"，这是有价值的先验。但要拿来**选动作**，必须补上动作接口，并给出该接口下的决策证据。把"视频预测得好"直接说成"能控制机器人"，是这条谱系最常见的过度推断。

### 5.2　实时交互与记忆

**GameNGen**（Valevski et al. 2024，arXiv 2408.14837）证明扩散模型可以当交互式引擎，速度与保真度两个数字都要连着口径背：速度在 v1 摘要里写作"**超过 20 FPS**"，v2 正文给的是**单块 TPU-v5、4 步 DDIM 采样下的 20 FPS**；**29.4 dB PSNR** 是**下一帧预测**的指标，衡量的是单步预测，**不是长程 rollout 的保真度**。要点在于这两个数字衡量的是**对 DOOM 这个游戏的模拟速度与保真度**，不是物理正确性——DOOM 的"物理"本身就是一套游戏规则。

**Genie 3**（Google DeepMind 官方博客，2025-08-05，**仅博客**）：**720p、24 FPS**，官方描述的一致性是"**a few minutes**"量级、视觉记忆约**一分钟**。**没有论文，也没有公开参数量**——11B 属于 Genie 1，不要往 Genie 3 上套，也不要用 Genie 1 的三组件结构去推断 Genie 3 的内部实现。

**Project Genie**（官方博客，2026-01-29，**仅博客**）是面向消费者的原型，公布的 60 秒世界长度是**产品原型的限制**，不能当成 Genie 3 的研究上限。

**Matrix-Game 3.0**（arXiv 2604.08995）把重点放在三件工程上：动作控制、**记忆检索**、流式生成。速度口径要完整：**5B** 模型，在**异步部署**下最高 **40 FPS @ 720p**，该部署用 **8 张 GPU 跑 DiT + 1 张做 VAE 解码**。写成"单卡 40 FPS"是错的。

**Navigation World Models**（arXiv 2412.03572）在第一人称视频与动作上训 conditional diffusion transformer，用它评估候选轨迹并做目标导向导航——生成式模型直接充当导航规划器的例子。

> 💡 **实时与记忆是一对矛盾** — 交互要求每帧的计算预算固定，而记住十秒前走过的房间要求信息能跨越很长的时间跨度。最直接的做法——把完整历史原样保留并逐帧处理——计算成本会随时间增长；固定大小的状态或外部记忆不需要每帧上下文无限变长，但把"记什么"的选择交给了训练或检索机制。工程上的三种折中是：截断上下文（快，但转身回头时房间**有可能**变了——截断只是抬高了这种不一致的风险，不保证一定发生）、把历史压进固定大小的状态（快且省，但压缩什么由训练决定）、显式检索历史片段（Matrix-Game 3.0 的记忆检索路线，代价是检索本身要算）。看到"一致性维持 N 分钟"的说法时，先问它是哪一种，以及 N 分钟里有没有人真的转身回头。

### 5.3　多模态与空间条件

**UniSim**（Yang et al.，arXiv 2310.06114）提出"通用动作条件模拟器"的路线：把不同来源的数据（人类活动视频、机器人数据、导航数据等）统一成"观测—动作—观测"的条件生成问题，训练一个可交互的视频模拟器。

**Cosmos** 是一个系列，**不能压成一条架构**：

- **Cosmos WFM**（NVIDIA 2025，arXiv 2501.03575）同时给出**自回归**与**扩散**两条路径，并把 **Predict / Transfer / Reason** 当作**不同任务**而不是一个模型的三个模式。
- **Cosmos-Reason1**（arXiv 2503.15558）是物理常识与具身推理的**视觉语言模型**，**不能**拿来当"动态模型"的证据——它回答问题，不预测动作后果。
- **Cosmos-Predict2.5**（arXiv 2511.00062）公布 2B / 14B 两档配置；它**有论文**，不属于"仅博客"那一档；这些配置**不继承**给 Cosmos 3。
- **Cosmos 3**（arXiv 2606.02800）是 omnimodal 的：语言、图像、视频、音频、动作。它的 **mixture-of-transformers** 指的是**每一层同时含一组自回归 reasoner 参数与一组扩散 generator 参数**——**不是**稀疏 FFN 的 MoE，也**不是**"每个模态一个专家"。规模按其论文 §2.5 与 HF 模型卡是 **Edge 4B / Nano 16B / Super 64B**，三档**均已发布**（Edge 发布于 2026-07-20），许可为 **OpenMDW-1.1**。

**Atlas**（World Labs 官方博客，2026-09-01，**仅博客**）要落到接口上说：在**文本、图像、相机位姿、深度与空间上下文**上做自回归扩散；博客展示了**最多一分钟、1440p** 的结果。展示长度**不是**实时速度的证据，"相机可控"、"稀疏重建"、"机器人接触动力学"是三类不同的证据，不能互相顶替。

### 5.4　驾驶与 Sora 这个对照

驾驶方向只保留**任务差异**：**GAIA-2**（Wayve 2025，arXiv 2503.20523）说明的是一件具体的事——把**动作、道路布局、驾驶条件**作为条件加进去，比无条件视频生成更贴近驾驶决策需要的可控性。**GAIA-3** 目前只有新闻稿层面的信息（**仅博客**），只在发布层面提及。

Sora / Sora 2 是**对照**而非主角：它们的公开材料是高层级报告，且目标是通用视频生成的质量与可用性，与"动作后果与决策"这条主线不是同一个问题（详见 §A.2 的排除理由）。把"视频生成器"和"世界模型"直接划等号，正是 §7.4 那场定义之争的起点。

---

## §6 具身智能里的三条路

生成式模型要真正帮到机器人，必须跨过同一个缺口：**生成的是像素，策略需要的是动作**。目前有三条明确的路。

**路一：视频当计划，动作靠推断。** UniPi（Du et al. 2023，arXiv 2302.00111）先把任务生成成一段"该怎么做"的视频，再用逆动力学之类的方式把视频转成可执行动作。计划在像素空间，落地在动作空间。

**路二：生成数据，补上动作标签。** DreamGen（NVIDIA 2025，arXiv 2505.12705）用生成的视频扩充训练数据，但生成视频没有动作标签——它用 **LAM 或逆动力学模型（IDM）恢复伪动作**（LAM 路线取的是量化前的连续 embedding，不是离散码），把"视频"变成"可训练的策略数据"。这一步是必需的桥梁，没有它，合成视频对策略训练没有直接用途。

**路三：模型本身当环境。** Genie Envisioner（arXiv 2508.05635）把世界模型与策略学习、仿真评估连起来，模型既是数据源也是评估场。**SIMA 2**（Google DeepMind 2025，arXiv 2512.04797）是在 Genie 生成的世界里行动的 **agent**——它展示的是"在生成世界里能玩起来"，**不等于**现实迁移已经成立。

| 路线 | 模型产出什么 | 动作从哪来 | 主要风险 |
| --- | --- | --- | --- |
| 视频当计划 | 一段"该怎么做"的视频 | 逆动力学从相邻帧推 | 视频里的动作物理上不可执行 |
| 生成训练数据 | 大量合成轨迹视频 | LAM 或 IDM 恢复的伪动作 | 伪动作有噪声，误差进策略 |
| 模型当环境 | 可交互的世界 | 策略在模型里自己探索 | 策略利用模型误差，真机不成立 |

> ⚠️ **三条路都没有绕开真实数据，但需要的"真实数据"不是同一种** — 路一的逆动力学要**真实的观测—动作配对**来训。路二的两种伪动作来源正好分在两边（DreamGen §2.3 明确区分）：**IDM 依赖动作监督**，也就是真实的观测—动作配对；**LAM 只需要无动作的真实视频**就能训。两者都要真实数据，但"要真实视频"和"要真实观测—动作配对"是两个量级不同的门槛，别合并成一句。至于 LAM 学出来的码能不能对上执行器动作，是**另一个问题**（§5.1 的语义对齐），不能拿来否定它不需要动作标注这一点。路三的评估最终仍要回到真机。"用世界模型代替真实数据"目前是方向不是结论。

---

## §7 评测与定义之争

### 7.1　视频逼真能证明理解物理吗

**Physics-IQ**（Motamed et al. 2025，arXiv 2501.09038）的设计要记准：真实拍摄 **66 个物理场景、396 段视频**；给模型**条件帧或条件视频**（按被测模型是图生视频还是视频续写而定），让它生成**五秒**后续，再与真实后续比较；指标是**空间 IoU、时空 IoU、加权空间 IoU 与 MSE**。它测的是**条件视频续写**，**不是**物理问答。

它同时用"MLLM 能否区分真实与生成视频"来衡量**视觉真实感**。跨模型看，真实感与物理指标的相关是 $r=-0.46$、$p=0.249$——**在这组模型与这个样本量下未检出显著相关**。

> ⚠️ **这是实验性结论，不能升级成"视觉质量与物理理解无关"** — 未检出显著相关 ≠ 证明独立；模型数量少、相关系数本身是负的中等大小。正确的表述是：**在 Physics-IQ 的这组测量里，看起来真实并没有换来物理指标上的优势**。这句话面试里说出来，比背"两者无关"稳得多。

### 7.2　两个基准，两个问题

**WorldModelBench**（arXiv 2502.20694）测的是**指令遵循**加**物理/常识**的违规检测——生成的内容有没有照着指令走、有没有违反基本常识。**WorldScore**（arXiv 2504.00983）测的是**可控性、质量与动态性**。两者不可互相替代：前者问"它听不听话、犯不犯常识错误"，后者问"它可控吗、好看吗、动得对吗"。

### 7.3　更新的两个基准各自加了什么

**WorldArena 2.0**（arXiv 2605.17912）的增量是**视觉触觉信号、模型内的策略优化，以及真实机器人平台**——即把评测从"看视频"推到"在模型里优化策略并在真机上验证"。**WorldRoamBench**（arXiv 2606.31672）的增量是**交互过程中**的稳定性：动作跟随、视觉一致、物理合理、记忆保持能不能在长交互里同时维持。

两篇都给出了"没有模型全面满足要求"这类结论。**这个结论限定在它们各自测过的模型与协议之内**，不是对整个领域的普适判断。

### 7.4　定义之争：若干提案，不是共识

- **GLP critique**（arXiv 2507.05169）批评的是把生成式视频模型直接当作通用世界模型的路线，主张预测能力与决策效用要分开论证。
- **Definition & Roadmap**（arXiv 2607.06401）给出一份定义与路线图提案，其分类另有一套角色划分（Renderer / Simulator / Planner 等）。**它不是 Predictor / Simulator / Evolver 那份**——这是本领域最容易张冠李戴的一处。
- **L0–L7 position**（arXiv 2606.15032）提出的是**多个交叉轴**上的分级，不是一条直线上的等级。
- **Agentic World Modeling**（arXiv 2604.22748）把世界模型放进 agent 的决策回路来谈，关心的是"模型服务于哪一步决策"；**"预测器 / 模拟器 / 演化器"（Predictor / Simulator / Evolver）出自这一篇**。

> ⚠️ **几套等级不要混用，也不要当共识** — L0–L7（2606.15032）是多轴分级，Predictor/Simulator/Evolver 出自 Agentic World Modeling（2604.22748），Definition & Roadmap（2607.06401）又是另一套划分；**它们都是作者立场，不是领域共识**。§10 的 L1/L2/L3 只表示**面试题难度**，与上述任何分级无关。

### 7.5　两条要说公平的话

**生成式路线的价值**：可见的细节可能直接决定接触、碰撞与奖励——DIAMOND 就是这个论点的实验支持。另外，扩散生成**不是**"用 MSE 预测一个平均未来"，它建模的是条件分布，采样给出的是一个具体的、清晰的未来，而不是多个未来的模糊叠加。

**表征预测路线的价值**：忽略不可预测的细节可以省下大量算力，并让预测集中在对下游有用的量上。但反过来也有硬边界——**表征损失小不等于动作后果预测正确**。表征空间的距离是模型自己定义的，它和"这个动作会不会把杯子打翻"之间没有免费的等价关系。

---

## §8 三条谱系对比

| 维度 | latent dynamics / MBRL | predict-in-representation-space | 生成式 / 交互式 |
| --- | --- | --- | --- |
| 预测的量 | 潜状态转移（多数带奖励与 continuation） | EMA/冻结 encoder 的表征 | 未来观测（像素或视频 token） |
| 记忆 | 递推的 $h_t$（或 transformer 上下文） | 上下文表征 + 动作条件 predictor | 上下文窗口、显式记忆检索 |
| 动作接口 | 环境动作，天然有 | 预训练不带动作；**V-JEPA 2-AC 这条流程**把动作放在第二阶段（冻结 encoder，训动作条件 predictor），不是这条谱系的通则 | 要专门设计：LAM 码 / 相机位姿 / 执行器 / 文本 |
| 决策接口 | 潜空间规划、imagination actor-critic、MCTS | 表征空间 MPC（与目标表征的距离） | 人或策略在环交互；经动作推断落地 |
| 主要损失 | 重建 NLL + KL（或无重建的规划量回归） | 表征距离（$L_1$ / 平方 $L_2$） | 生成损失（扩散 / token 预测） |
| 典型证据 | 任务回报、样本效率 | probe 准确率 + 闭环成功率 | 视频质量、可控性、FPS、一致性时长 |
| 主要失败模式 | 模型误差被策略利用；posterior collapse | 表征塌缩；表征距离与任务效用脱钩 | 长程漂移、记忆丢失、逼真但物理不对 |
| 重叠例子 | Dreamer 4 是 transformer 生成式动态 + imagination RL | DINO-WM 用冻结表征学动态，属谱系一的做法 | Cosmos 3 是带推理能力的生成式模型，跨语言/图像/视频/音频/动作 |

---

## §9 Code Patterns

三段核心 pattern，完整实验在 [`code/world_models_toy.py`](code/world_models_toy.py)。三段共用 `import torch, torch.nn as nn, torch.nn.functional as F`。

### 9.1　RSSM 一步 + balanced KL

```python
def cat_kl(probs_q, probs_p, eps=1e-8):
    """KL(q||p)，分组分类分布，probs: [..., G, C]。
    先对类别求和、再对组求和 = 联合 KL —— free bits 截的是这个量。"""
    return (probs_q * ((probs_q + eps).log() - (probs_p + eps).log())).sum(-1).sum(-1)

class RSSM(nn.Module):
    def __init__(self, deter=512, groups=32, classes=32, a_dim=6, emb_dim=1024, hid=256):
        super().__init__()
        self.groups, self.classes = groups, classes
        z_dim = groups * classes
        self.cell = nn.GRUCell(z_dim + a_dim, deter)          # h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.prior_head = nn.Sequential(nn.Linear(deter, hid), nn.SiLU(), nn.Linear(hid, z_dim))
        self.post_head = nn.Sequential(nn.Linear(deter + emb_dim, hid), nn.SiLU(), nn.Linear(hid, z_dim))

    def _probs(self, logits, uniform_mix=0.01):
        logits = logits.unflatten(-1, (self.groups, self.classes))
        return (1 - uniform_mix) * F.softmax(logits, -1) + uniform_mix / self.classes   # 1% 均匀混合

    @staticmethod
    def _sample(probs):
        idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(probs.shape[:-1])
        z = F.one_hot(idx, probs.shape[-1]).float()
        return z + probs - probs.detach()                     # straight-through：有偏但可导

    def step(self, h_prev, z_prev, a_prev, emb_t=None):
        """emb_t=None 就是 imagination：没有观测，只能走 prior。"""
        h = self.cell(torch.cat([z_prev.flatten(-2), a_prev], -1), h_prev)
        prior = self._probs(self.prior_head(h))
        if emb_t is None:
            return h, self._sample(prior), prior, None
        post = self._probs(self.post_head(torch.cat([h, emb_t], -1)))
        return h, self._sample(post), prior, post

def balanced_kl(post, prior, beta_dyn=0.5, beta_rep=0.1, free_nats=1.0):
    """β 取 DreamerV3 论文 v1 式 4 / 表 W.1；32×32 分类潜变量与 free_nats=1 也沿用论文（表 W.1、式 5）。"""
    kl_dyn = cat_kl(post.detach(), prior)                     # sg(q)||p：只让 prior 追 posterior
    kl_rep = cat_kl(post, prior.detach())                     # q||sg(p)：只让 posterior 让步
    return beta_dyn * kl_dyn.clamp(min=free_nats) + beta_rep * kl_rep.clamp(min=free_nats)
```

`post.detach()` 停的是**这条路径**上的梯度；$h_t$ 由两个头共享，它的梯度照样从另一条路径回来。

这段代码里 $\beta_{dyn},\beta_{rep}$（v1 式 4 / 表 W.1）、32 组 × 32 类的分类潜变量（表 W.1）和 `free_nats=1.0`（式 5 的 1 nat）都沿用 DreamerV3 论文；**用 `GRUCell` 作递推单元、网络宽度这类写法是为了可读性选的**，不是论文规定的唯一实现。

### 9.2　JEPA：非对称 predictor + EMA 目标

```python
@torch.no_grad()
def ema_update(target, online, m=0.999):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(m).add_(p_o.detach(), alpha=1 - m)           # 目标分支只被 EMA 推动

def jepa_loss(enc, enc_ema, pred, x_ctx, y_tgt, mask_tok, p=1):
    """p=2 → I-JEPA 的 patch 表征平方 L2；p=1 → V-JEPA / V-JEPA 2 的 L1。"""
    ctx = enc(x_ctx)                                          # 在线分支，有梯度
    with torch.no_grad():
        tgt = enc_ema(y_tgt)                                  # 停梯度：目标不被拉向预测
    out = pred(ctx, mask_tok)                                 # 非对称：predictor 只在这一侧
    return (out - tgt).abs().mean() if p == 1 else (out - tgt).pow(2).mean()

def latent_mpc_cost(pred_ac, s0, action_seq, z_goal):
    """V-JEPA 2-AC 的规划接口：encoder 冻结，只有 action-conditioned predictor 在滚。"""
    s = s0
    for a in action_seq.unbind(1):                            # action_seq: [B, H, a_dim]
        s = pred_ac(s, a)
    return (s - z_goal).abs().flatten(1).mean(1)              # CEM 用它排序，执行首个动作后重规划
```

### 9.3　latent action model 的 VQ 瓶颈

```python
class LatentActionModel(nn.Module):
    def __init__(self, obs_dim, d=32, K=8):                   # Genie 1 用 8 个码
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * obs_dim, 128), nn.SiLU(), nn.Linear(128, d))
        self.codebook = nn.Embedding(K, d)
        self.dec = nn.Sequential(nn.Linear(obs_dim + d, 128), nn.SiLU(), nn.Linear(128, obs_dim))

    def forward(self, o_t, o_next, beta_commit=0.25):
        e = self.enc(torch.cat([o_t, o_next], -1))            # E(o_t, o_{t+1})：编码器看得到未来
        d2 = (e.pow(2).sum(-1, keepdim=True) - 2 * e @ self.codebook.weight.t()
              + self.codebook.weight.pow(2).sum(-1))          # ||E(.) - e_j||^2
        k = d2.argmin(-1)
        e_k = self.codebook(k)
        e_st = e + (e_k - e).detach()                         # straight-through：通用 VQ 写法
        pred = self.dec(torch.cat([o_t, e_st], -1))           # 解码器只有当前信息 + 码
        loss = (F.mse_loss(pred, o_next)                      # 预测项
                + F.mse_loss(e_k, e.detach())                 # codebook 项
                + beta_commit * F.mse_loss(e, e_k.detach()))  # commitment 项
        return pred, k, loss
```

瓶颈的全部作用是**限制信息量**：$K=8$ 时一个码至多 3 bits。解码器只拿得到 $o_t$ 与 $e_k$，所以 $e_k$ 必须携带"帧间变化里最值得留的那点信息"——但那是不是**动作**，代码不保证（§5.1）。这里的 `beta_commit=0.25` 是 VQ-VAE 常用的经验取值，**照搬过来作为演示**，Genie 并未规定这个数。

---

## §10 25 高频面试题

### L1 必会题（10 题）

<details>

<summary>Q1. 什么是 world model？</summary>

智能体内部学到的环境结构与动态模型：给定至今的观测与动作，预测"这样做之后会怎样"。**必要的输出取决于用途**——要在模型里做 RL，就得能在想象里算回报并截断 episode（Dreamer 这类实现为此学奖励头与 continuation 头；奖励函数或终止规则已知时用已知的那份也行）；要做视觉目标导航只需能比较目标表征的距离；要当可玩模拟器就必须生成人能看的观测。回答时把三个机制问题摆出来：保留什么信息、预测哪个量、这个量怎么进决策。

只说"能预测未来的模型"，或直接把它等同于视频生成器，都不得分。

</details>

<details>

<summary>Q2. RSSM 的 prior 和 posterior 分别是什么？imagination 时用哪个？</summary>

递推是 $h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})$。prior $p_\theta(z_t\mid h_t)$ 看不到当前观测，posterior $q_\phi(z_t\mid h_t,o_t)$ 吸收观测。训练时两者都在：重建走 posterior，KL 把 prior 拉向 posterior。**imagination 里没有未来观测，只能用 prior**——调用后验等于把还没发生的观测偷看进来，想象出的回报就不可信了。

</details>

<details>

<summary>Q3. PlaNet 和 Dreamer 怎样选动作？</summary>

PlaNet 不训练策略网络：每一步在潜空间用 **CEM** 在线规划——从一个高斯分布采一批动作序列、用 prior 向前 rollout、用奖励头打分，取 elite 序列重新拟合分布，迭代若干轮后执行**末轮分布均值**的第一个动作，下一步重来，全程不解码图像。说成"采一次样、执行得分最高的那条样本"就漏掉了 CEM 的迭代重拟合。Dreamer 在 imagination 里训练 actor-critic：用 posterior 取真实起点，只用 prior 展开 $H$ 步，在想象状态上更新策略与价值，部署时策略一次前向出动作。

一句话差别：**PlaNet 把算力花在决策时，Dreamer 花在训练时。**

</details>

<details>

<summary>Q4. 视频看起来逼真，能证明模型理解物理吗？</summary>

Physics-IQ（arXiv 2501.09038）给的是实验性答案：真实拍摄 66 个物理场景、396 段视频，给模型条件帧或条件视频（视被测模型是图生视频还是视频续写而定），让它生成五秒后续，指标是空间 IoU、时空 IoU、加权空间 IoU 与 MSE。它同时用"MLLM 能否区分真假视频"衡量视觉真实感，跨模型相关为 $r=-0.46$、$p=0.249$——**在这组模型与样本量下未检出显著相关**。

说成"研究证明视觉质量与物理理解无关"就错了：未检出显著相关不是证明独立。正确说法是"在 Physics-IQ 的这组测量里，逼真没有换来物理指标上的优势"。

</details>

<details>

<summary>Q5. $h_t$ 和 $z_t$ 各自负责什么？</summary>

$h_t$ 是确定性的，提供**不经过随机重采样的递推记忆通道**；$z_t$ 是随机的，表达环境随机性与当步的新信息。别把确定性说成"无损"——固定宽度的 GRU 状态一样会压缩和遗忘，它只是不被每步采样噪声冲刷。两个方向都别推广过头：**不是只有递推向量才能建模记忆**（transformer 用 attention 直接回看历史），也**不是任何确定性 world model 都不能工作**（确定性环境里它可以很好）。

</details>

<details>

<summary>Q6. Dreamer 一类模型有哪几个预测头？纯视频模型也需要吗？</summary>

观测、奖励、continuation 三个。后两个是为了"在模型里做 RL"：**在 Dreamer 这套实现里**，想象里的回报由学到的奖励头给出、episode 的截断由学到的 continuation 头给出，缺哪个都算不下去。**纯视频生成模型未必需要它们**——它的用途是生成可看、可交互的未来。

但这条必要性只到这套实现为止。奖励函数或终止规则已知时（大量仿真与游戏环境就是），直接用已知的那份同样能在想象里做 RL；观测重建更不是必要条件——MuZero 与 TD-MPC2 都不重建观测照样规划。把"有没有奖励头"当成 world model 的定义线，会在第一个追问里塌掉。

</details>

<details>

<summary>Q7. Genie 3 有多大？和 Genie 1 的 11B 是什么关系？</summary>

Genie 3 **没有论文、也没有公开参数量**；720p、24 FPS、"数分钟"级一致性、约一分钟视觉记忆，全部来自 2025-08-05 的官方博客。**11B 属于 Genie 1**（arXiv 2402.15391），它的三组件（视频 tokenizer、8 个码的 latent action model、动态模型）同样不能拿来推断 Genie 3 的内部结构。Project Genie（2026-01-29 博客）的 60 秒是**消费者原型的限制**，不是 Genie 3 的研究上限。

给 Genie 3 报 11B 是硬伤。

</details>

<details>

<summary>Q8. 三条谱系里说的 "latent" 是同一个东西吗？</summary>

不是。RSSM 的 latent 是带 prior/posterior 的状态变量，服务于在里面 rollout；JEPA 的 latent 是编码器输出的表征，没有生成式先验；视频扩散模型的 latent 是 VAE 压缩后的生成空间。共同点只有一条：**不在像素上算**。用"谁在 latent 里工作"来区分谱系，会把视频模型误分到 JEPA 那边。

</details>

<details>

<summary>Q9. 无动作视频预训练的模型能直接拿来选动作吗？</summary>

不能直接。无动作视频提供的是"世界通常怎么演化"的先验，很有价值。要用于**动作选择**，必须补上动作接口，并给出该接口下的决策证据——文本条件、相机运动、执行器动作、LAM 码的语义完全不同，不能互相顶替。V-JEPA 2 的做法就是分两段：预训练 action-free，第二阶段用带动作的数据训 action-conditioned predictor。

</details>

<details>

<summary>Q10. DreamerV3 的"一套超参数"是什么意思？它的钻石怎么来的？</summary>

指**同一套超参数**覆盖 150 多个任务，即不必逐环境调参；**不是**一套权重通吃所有环境——每个任务仍各自训练。它在 Minecraft 里从零收集到钻石，是**通过与环境交互**做到的。Dreamer 4（arXiv 2509.24527）的"离线拿钻石"是另一篇、另一套设定，两者不能混为一谈。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11. KL balancing 是不是就是把 KL 系数调小？</summary>

不是。$\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$：**两项都是 $q\Vert p$ 方向**，分开的是谁被停梯度。$\beta_{dyn}\gg\beta_{rep}$ 的含义是"先让 prior 去追 posterior（学动态），再让 posterior 为了好预测让步（正则表征）"。调小单一系数只是整体放松一致项，得到的是另一种失衡：要么 posterior 退化、重建全靠 $h_t$，要么 prior 永远追不上 posterior，imagination 一展开就漂。系数按版本报：DreamerV3 v1 的式 4 / 表 W.1 是 $\beta_{dyn}=0.5,\beta_{rep}=0.1$，v2 正文是 $1$ 与 $0.1$。

写成 $D_{KL}(p\Vert q)$、或说 balancing 改变了 KL 方向，都不得分。

</details>

<details>

<summary>Q12. JEPA 怎样避免 collapse？</summary>

靠一组机制的组合：非对称结构（predictor 只在一侧）、EMA 的慢速目标分支、目标分支停梯度、掩码任务本身的难度。关键是下一句——**这不是不塌缩定理**：常数 encoder 配常数 predictor 依然是损失极低的退化解，效果依赖初始化、EMA 系数、掩码难度与优化器设置，实践中要监控表征方差或秩。

§9.2 对应的 toy（见 §A.3 实验 2）把这件事跑出来：共享权重且不停梯度时，上下文表征的方差按几何级数衰减；换成停梯度 + EMA 教师后，对称点处梯度为零，塌缩不再是吸引子——但从塌缩点出发它照样停在那里。

</details>

<details>

<summary>Q13. V-JEPA 2-AC 的 zero-shot 到底指什么？</summary>

指**新部署环境不再采数据**：在 Franka 机械臂上给定视觉目标做 reaching / grasp / pick-and-place，不在部署实验室重新采集。结构上，预训练仍是 action-free 的 masked prediction（$L_1$、EMA、停梯度）；2-AC 冻结 encoder，只训 action-conditioned predictor，输入含机器人动作与末端状态。论文说的 "<62 h unlabeled robot video" 是**没有任务语义标签**，不是没有动作记录。规划是表征空间的 MPC：预测动作序列执行后的表征，最小化与目标图像表征的 $L_1$ 距离，CEM 搜索，执行首个动作后重规划——**目标表征不是奖励标签**。

它不证明任意机器人、任意任务、任意长程分解都免采数据。

</details>

<details>

<summary>Q14. 把 VQ codebook 调得很小，就能自动发现真实动作吗？</summary>

不能保证。小 codebook 的论据是**容量**：一个码至多携带 $\log_2 K$ bits（$K=8$ 即 3 bits），逼模型丢掉帧间的大部分差异。但容量小只保证"留下的信息少"，**不保证留下的是执行器动作**，也不保证同一个码跨场景含义一致。arXiv 2506.15691 给出了机制：动作引起的变化与外生变化（光照、相机抖动、背景运动）**在竞争同一个瓶颈容量**，方差大的无关变化可能被优先编码；线性情形下这与 PCA 保留最大方差方向有直接联系。

</details>

<details>

<summary>Q15. 写出 imagination 里的 λ-return，并说明起点与长度约定。</summary>

按 §2 的时间约定：

$$G_t^\lambda=\hat r_{t+1}+\gamma\hat c_{t+1}\big[(1-\lambda)v(s_{t+1})+\lambda G_{t+1}^\lambda\big],\qquad G_H^\lambda=v(s_H)$$

critic 的回归目标要停梯度。DreamerV3 v1 的表 W.1 取 $H=15$、正文的预测序列长度 $T=16$：**以 $H$ 计动作转移数，状态序列含起点因而有 $H+1$ 个状态**。被追问"到底是 15 还是 16"时，答"数转移是 15、数状态是 16"。

</details>

<details>

<summary>Q16. free bits 和分类潜变量的 1% 均匀混合是同一件事吗？</summary>

不是，两件独立机制。**free bits** 是 $\max(\tau, D_{KL})$，对**联合 KL**（先对所有 latent 组求和）截下限，防止 KL 被压过头把表征容量耗光。**1% 均匀混合**是 **DreamerV3**（v1 正文与附录 C）给分类概率混入均匀分布，防止某一类概率被压到 0 导致 $\log$ 爆炸与梯度病态；它不是 DreamerV2 的改动，DreamerV2 带来的是分类潜变量本身。一个管目标函数的形状，一个管分布的数值健康。

另外，分类潜变量 + straight-through 本身是**有偏**的梯度近似——它把不可导的采样当恒等映射来传梯度，这是第三件事。

</details>

<details>

<summary>Q17. symlog 和 twohot 的分工是什么？</summary>

**symlog** 是尺度压缩：$\mathrm{symlog}(x)=\mathrm{sign}(x)\log(1+\lvert x\rvert)$，逆变换 symexp，对称且在 0 附近近似恒等，用来把跨数量级的信号压进可训练范围。**twohot** 是编码方式：把连续标量按距离分给相邻两个桶，于是回归变成分类，期望值可精确恢复。

DreamerV3 v1（式 10 之后的正文与附录 C）里 **reward 与 critic 都用 symlog twohot**——**不要把它推广到所有预测头**，观测与 continuation 头是另一回事。

</details>

<details>

<summary>Q18. DreamerV3 的 return normalization 和普通 advantage 标准化差在哪？</summary>

它取一个批次回报的**第 95 与第 5 分位之差** $S$ 做 EMA，再除以 $\max(1,S)$。三点不同：只做尺度、不减均值；**有下限 1**；跨批次平滑。下限是要害——稀疏奖励任务里回报几乎全为 0，$S$ 很小，普通的"减均值除标准差"正好会把近零回报的噪声放大成巨大梯度，而 $\max(1,S)$ 把这条路堵死。

</details>

<details>

<summary>Q19. PETS 和 MBPO 解决的是同一个问题吗？</summary>

不是。**PETS**（arXiv 1805.12114）解决**不确定性表示**：概率模型的 ensemble 同时表达环境噪声与数据不足带来的不确定性，规划时用轨迹采样把它传播下去。**MBPO**（arXiv 1906.08253）解决**误差累积**：从真实数据里的状态分支出很短的模型 rollout 再做 off-policy 更新，把模型偏差的复合控制住。

一句话：PETS 说"我不确定"，MBPO 说"我只敢往前看几步"。

</details>

<details>

<summary>Q20. Dreamer 4 为什么不能套用 DreamerV3 的描述？它的离线结论限定条件是什么？</summary>

骨架不同：Dreamer 4 是 **tokenizer + transformer 动态 + shortcut forcing**，没有 RSSM、KL balancing、分类潜变量那一套，硬套等于答错架构。它的钻石是在论文的**离线 Minecraft 设定**下取得的，训练流程包含行为克隆、奖励建模和在模型内部做 RL；论文报告的"单 GPU 实时"指的是**交互推理**，不是训练。DreamerV3 的钻石则来自与环境交互——两个结论不能互相替代。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21. 什么时候该预测像素，什么时候该预测表征？</summary>

判据是**决策依赖哪些信息**，不是哪条路线更先进。

预测像素的理由：细节可能直接决定接触、碰撞与奖励（DIAMOND 的实验就支持这一点——被压掉的小物体和细边缘往往正是触发条件）；需要人看、需要当可交互模拟器、需要一个跨任务通用的观测接口。**扩散生成不是"用 MSE 预测一个平均未来"**，它建模条件分布，采出的是一个具体清晰的未来。

预测表征的理由：忽略不可预测的细节省算力，预测集中在对下游有用的量上；下游本来就是表征空间的距离或 MPC（DINO-WM 在冻结 DINOv2 特征上学动态、V-JEPA 2-AC 用目标表征的 $L_1$）。

两条各有硬边界：像素路线**画得像不等于动得对**；表征路线**表征损失小不等于动作后果正确**——表征空间的距离是模型自己定义的，与"这个动作会不会把杯子打翻"之间没有免费的等价关系。

</details>

<details>

<summary>Q22. MuZero 不重建观测，为什么还能规划？</summary>

因为规划只用得到搜索树里的那几个量：奖励、价值、策略先验。潜状态不必能恢复环境真实状态 $x_t$，只需对这些量**预测充分**。代价有两个：没有解码器就没有"模型在想什么"的可视化通道，调试只能靠奖励预测误差、规划收益这类间接指标；而且"预测充分"是设计原则加实证结果，**不是严格的 value-equivalence 定理**。

TD-MPC2 是同一思路在连续控制上的版本：decoder-free 地学潜空间动态、奖励与价值供 MPC 用，其规模实验里 317M 的单一 agent 覆盖 80 个任务。

</details>

<details>

<summary>Q23. 离线学到的 world model，为什么不能靠"无限 imagination"解决探索？</summary>

因为模型只在数据覆盖的状态—动作分布上可信。想象再多次也不会凭空产生数据里没有的信息；更糟的是策略优化会**主动找到并利用模型高估回报的轨迹**——在模型里回报极高，放回真实环境不成立。这正是 MBPO 用短分支 rollout、PETS 用不确定性传播要压住的东西。

两件事要分开说：**扩大覆盖只能靠补充数据或新的环境交互**；**悲观/不确定性惩罚不扩大覆盖**，它做的是把决策约束回模型有证据的区域，让策略别去赌那些数据没覆盖到的高回报幻觉。把惩罚项说成"扩大覆盖的第二条路"是答反了方向。

Dreamer 4 的离线结果不是反例：它有为那个设定准备的离线数据、行为克隆与奖励建模，结论的作用域是那套设定。

</details>

<details>

<summary>Q24. 怎样判断一个 world model 对机器人是真的有用？</summary>

问四件事。**一、动作接口是什么**——执行器动作、相机位姿、LAM 码还是文本条件？只有第一种直接对应可执行控制。**二、决策证据是什么**——闭环成功率、哪个平台、多少任务、有没有换规划设置？V-JEPA 2.1 表 6 就是现成例子：同一规划设置下 grasp 从 60% 到 70%（+10 个百分点）是表征比较，改了规划设置并放长 horizon 后的 80%（+20 个百分点）已经不是纯表征比较。**三、误差在哪累积**——长 horizon 一致性、记忆保持、接触动力学。**四、成本口径**——速度在什么硬件、什么部署（Matrix-Game 3.0 的最高 40 FPS @ 720p 是 8 卡跑 DiT + 1 卡做 VAE 解码的异步部署，不是单卡）。

视频质量分数、表征 probe 准确率都不能替代前三问——V-JEPA 2 常被引用的 77.3 是 SSv2 的 attentive-probe top-1，不是机器人成功率。

</details>

<details>

<summary>Q25. 给你一份刚发布的 world model 报告，怎样快速定位它？</summary>

先按三问读：**预测哪个量**（像素 / 表征 / 规划量）、**动作接口是什么**、**预测如何进决策**。这三问定谱系，比看它自称什么可靠——Dreamer 4 同时是生成式动态与 imagination RL，Cosmos 3 横跨理解、生成与动作。

再查证据层级：有论文还是只有博客；代码、权重、数据分别开放到哪一步（**计划发布不算已开放**）；每个数字有没有给全任务、指标、模型版本、设置与硬件。

最后放进评测坐标：WorldModelBench 测指令遵循与物理/常识违规，WorldScore 测可控性、质量与动态性，WorldRoamBench 测交互过程中的动作跟随、视觉一致、物理合理与记忆保持，WorldArena 2.0 把视觉触觉、模型内策略优化与真实机器人平台纳进来。它们问的不是同一件事，某一项好不能替另一项背书；这些基准里"没有模型全面满足"的结论，也只限定在它们测过的模型与协议之内。

分级要小心：L0–L7（arXiv 2606.15032）是**多个交叉轴**上的提案；Predictor / Simulator / Evolver 出自 **Agentic World Modeling（arXiv 2604.22748）**，别记到 Definition & Roadmap 头上——后者（arXiv 2607.06401）是又一份提案，用的是另一套角色划分（Renderer / Simulator / Planner 等）。**这几套都不是领域共识**；GLP critique（arXiv 2507.05169）代表的是另一种立场。§10 的 L1/L2/L3 只表示面试题难度。

</details>

---

## §A 附录

### A.1　论文与出处

证据层级：**✅ 论文**（arXiv 公开，引用的细节取自论文）／**⚠️ 仅博客**（只有官方博客或新闻稿，无论文）。第三种情况单独写明：**有论文但只提方向**（未逐项核验其数字），不并进"仅博客"。

**谱系一：latent dynamics / MBRL**

| 工作 | 出处 | 一句话 |
| --- | --- | --- |
| World Models | ✅ 1803.10122 | VAE + MDN-RNN + 线性 controller；**VizDoom 实验**里策略完全在 dream 里训练，CarRacing 的 controller 在真实环境训练 |
| PlaNet | ✅ 1811.04551 | RSSM；latent overshooting；潜空间在线规划 |
| Dreamer | ✅ 1912.01603 | imagination 里训 actor-critic（路径梯度） |
| DreamerV2 | ✅ 2010.02193 | 分类潜变量；KL balancing；从想象中达到 Atari 人类水平 |
| DreamerV3 | ✅ 2301.04104 | symlog / twohot / return normalization；一套超参数覆盖 150+ 任务；Minecraft 钻石来自环境交互 |
| Dreamer 4 | ✅ 2509.24527 | tokenizer + transformer 动态 + shortcut forcing；离线 Minecraft 设定下的钻石；单 GPU 实时**推理** |
| MuZero | ✅ 1911.08265 | 只学奖励/价值/策略量，无重建；MCTS |
| TD-MPC2 | ✅ 2310.16828 | decoder-free 潜动态 + 奖励 + 价值供 MPC；317M 单 agent / 80 任务 |
| DayDreamer | ✅ 2206.14176 | 该实验中四足约 1 小时真实交互学会行走 |
| PETS | ✅ 1805.12114 | 概率 ensemble + 轨迹采样 = 不确定性表示 |
| MBPO | ✅ 1906.08253 | 从真实数据分支的短 rollout = 控制模型偏差 |
| IRIS | ✅ 2209.00588 | 离散 tokenizer + transformer 想象 |
| DIAMOND | ✅ 2405.12399 | 扩散 world model；视觉细节影响控制回报 |

**谱系二：predict-in-representation-space**

| 工作 | 出处 | 一句话 |
| --- | --- | --- |
| LeCun 立场文章 | OpenReview v0.9.2（2022-06-27） | 在抽象表征空间预测；**无 arXiv 编号** |
| I-JEPA | ✅ 2301.08243 | 非对称 predictor + EMA 目标；patch 表征平方 $L_2$；多块 masking |
| V-JEPA | ✅ 2404.08471 | 视频 masked feature prediction；$L_1$；无负样本、无像素重建 |
| V-JEPA 2 / 2-AC | ✅ 2506.09985 | 预训练 action-free；2-AC 冻结 encoder 训动作条件 predictor + 表征空间 MPC |
| V-JEPA 2.1 | ✅ 2603.14482 | masked 与 visible-context token 都受监督；多层 deep self-supervision；规模扩展 |
| DINO-WM | ✅ 2411.04983 | 在冻结 DINOv2 特征上学动态，用目标特征规划 |
| JEPA 泛化理论 | ✅ 2606.27014 | 条件谱图 / 动作条件共现矩阵低秩分解 → 预训练误差与规划 regret |
| VL-JEPA | ✅ 2512.10942 | 把语言接入同一框架；**论文 2512.10942；仅提方向**，不引用其具体数字 |

**谱系三、具身与评测**

| 工作 | 出处 | 一句话 |
| --- | --- | --- |
| Genie | ✅ 2402.15391 | 11B；视频 tokenizer + 8 码 LAM + 动态模型；MaskGIT 式采样 |
| Genie 3 | ⚠️ 博客 2025-08-05 | 720p / 24 FPS / 数分钟一致性 / 约一分钟视觉记忆；无论文、无公开参数量 |
| Project Genie | ⚠️ 博客 2026-01-29 | 消费者原型；60 秒是产品限制 |
| GameNGen | ✅ 2408.14837 | v1 摘要 ">20 FPS"、v2 正文 20 FPS（单 TPU-v5、4 步 DDIM）；29.4 dB 是**下一帧预测** PSNR（DOOM 模拟数字） |
| UniSim | ✅ 2310.06114 | 统一的动作条件交互式模拟器路线 |
| Navigation World Models | ✅ 2412.03572 | 第一人称视频 + 动作上的 conditional diffusion transformer |
| Cosmos WFM | ✅ 2501.03575 | 自回归与扩散两条路径；Predict / Transfer / Reason 是不同任务 |
| Cosmos-Reason1 | ✅ 2503.15558 | 物理常识与具身推理的 VLM，**不是**动态模型证据 |
| Cosmos-Predict2.5 | ✅ 2511.00062 | 2B / 14B 配置；**不继承**给 Cosmos 3 |
| Cosmos 3 | ✅ 2606.02800 | omnimodal；每层 AR reasoner + diffusion generator 参数组；Edge 4B / Nano 16B / Super 64B |
| GAIA-2 | ✅ 2503.20523 | 驾驶：动作、道路布局、驾驶条件作为条件 |
| GAIA-3 | ⚠️ 仅新闻稿 | 只在发布层面提及 |
| Matrix-Game 3.0 | ✅ 2604.08995 | 5B；异步部署（8 卡 DiT + 1 卡 VAE）最高 40 FPS @ 720p；记忆检索、流式生成 |
| Atlas | ⚠️ World Labs 博客 2026-09-01 | 文本/图像/相机位姿/深度/空间上下文上的 AR diffusion；展示最多一分钟 1440p |
| LAM 瓶颈分析 | ✅ 2506.15691 | 动作变化与外生变化竞争瓶颈容量；线性情形与 PCA 相关 |
| UniPi | ✅ 2302.00111 | 视频当计划，动作靠逆动力学推断 |
| DreamGen | ✅ 2505.12705 | 生成视频 → LAM/IDM 恢复伪动作 → 策略数据 |
| Genie Envisioner | ✅ 2508.05635 | 连接世界模型与策略学习、仿真评估 |
| SIMA 2 | ✅ 2512.04797 | 在 Genie 生成世界里行动的 agent；不等于现实迁移 |
| Physics-IQ | ✅ 2501.09038 | 66 场景 / 396 段视频；五秒续写；IoU 与 MSE 指标 |
| WorldModelBench | ✅ 2502.20694 | 指令遵循 + 物理/常识违规 |
| WorldScore | ✅ 2504.00983 | 可控性 / 质量 / 动态性 |
| WorldArena 2.0 | ✅ 2605.17912 | 视觉触觉、模型内策略优化、真实机器人平台 |
| WorldRoamBench | ✅ 2606.31672 | 交互中的动作跟随、视觉一致、物理合理、记忆稳定 |
| GLP critique | ✅ 2507.05169 | 反对把生成式视频模型直接当通用世界模型 |
| Definition & Roadmap | ✅ 2607.06401 | 定义与路线图提案；其分类是另一套角色划分（Renderer / Simulator / Planner 等），**不是** Predictor / Simulator / Evolver |
| L0–L7 position | ✅ 2606.15032 | 多个交叉轴上的分级提案 |
| Agentic World Modeling | ✅ 2604.22748 | 把世界模型放回 agent 决策回路；**Predictor / Simulator / Evolver 出自这一篇** |

**开放状态**：Cosmos 3 的 Edge、Nano、Super 均已发布（Edge 发布于 2026-07-20，见其 HF 模型卡），许可 OpenMDW-1.1；Genie 3、Project Genie、Atlas、GAIA-3 只有官方博客层面的信息（Cosmos-Predict2.5 **有论文**，arXiv 2511.00062）。其余条目只断言"论文已公开"，代码、权重与数据是否开放**以各自官方页面为准**——不逐条断言，因为这四件事经常不同步。

### A.2　排除清单与理由

| 被排除 | 理由 |
| --- | --- |
| Sora / Sora 2 | 只有高层级公开报告；与"动作后果与决策"这条主线不匹配——在 §5.4 作为对照保留 |
| Oasis | 机制已由 GameNGen 代表；可研究性与篇幅 |
| Hunyuan-GameCraft | 同一技术问题已有代表工作覆盖 |
| Runway GWM / Odyssey | 发布公告层面 |
| HunyuanWorld | **有论文**（HunyuanWorld 1.0，arXiv 2507.21809）；排除是因为其主题偏 3D 场景生成与可漫游世界构建，与"动作后果如何进入决策"这条主线不是同一问题，且篇幅有限 |
| STORM / TransDreamer | 机制与 Dreamer / IRIS 重叠 |
| Meta CWM | 术语碰撞：指代码执行意义上的 "world model" |
| Othello-GPT 一类 LLM 内部世界模型 | 问的是"表征里有没有"，不是"能否预测动作后果" |
| Tesla | 无论文 |

### A.3　Runnable toy

脚本：[`code/world_models_toy.py`](code/world_models_toy.py)，纯 PyTorch，CPU 上几秒跑完。三个实验都有解析答案，可以逐项对照——**结论只覆盖各自的显式设定，不是论文复现**。

**实验 1｜RSSM 线性高斯：posterior collapse 与 balanced KL。** 环境 $x_{t+1}=0.5x_t+a_t+\epsilon$，$\epsilon$ 等概率取 $\pm1$。确定性部分 $h=0.5x_t+a_t$ 可精确算出，残差 $\epsilon=o_{t+1}-h$ 是唯一的新信息。标量参数化 $q(z\mid h,o)=\mathcal N(w\epsilon,1)$、$p(z\mid h)=\mathcal N(v,1)$、$\hat o=h+dz$，于是 $\mathcal L_{rec}=\tfrac12[(1-dw)^2+d^2]$、$\mathbb E D_{KL}=\tfrac12(w^2+v^2)$。对比单一 KL 系数 9 与 balanced $\beta_{dyn}=9,\beta_{rep}=0.1$（**toy 取值，不是论文取值**）：从 $w=d=0.5$、$v=1$ 出发，SGD lr 0.05 跑 1,000 步，单系数版收敛到 $\lvert w\rvert,\lvert d\rvert<10^{-6}$——posterior collapse，重建 MSE ≈ 1，等于完全没用上观测；balanced 版收敛到 $v\to0$、$w^2=1/\sqrt{0.1}-1$、$d=w/(1+w^2)$，MSE $=\sqrt{0.1}$。

**实验 2｜JEPA 的捷径与停梯度。** $x=s$、$y=s+\epsilon$，encoder 是 $E_w(u)=wu$，predictor 取恒等。共享 $w$ 且**不**停梯度时 $\mathcal L=\tfrac12w^2$，lr 0.1 的梯度下降给出 $w_n=0.9^n$：从 $w_0=1$ 跑 100 步，上下文表征方差降到 $0.9^{200}$——塌缩是吸引子。换成停梯度 + EMA 教师 $b\leftarrow0.9b+0.1w$：从 $w=b=1$ 出发梯度为零，$w$ 停在 1，损失 0.5 全部来自不可预测的噪声；但从 $w=b=0$ 出发它同样停住。**这正是"机制不是定理"的可执行版本。**

**实验 3｜网格世界上的坐标 LAM。** 编码器取 $\Delta=s'-s$，4 个码的 VQ，解码器是加性的。farthest-first 初始化后做一轮最近邻分配与质心更新，就能精确恢复四个转移向量：验证集重建误差 0、purity 1、$I(K;A)=2$ bits（在码的置换意义下）。这说明在**这个坐标化、无外生变化**的设定里瓶颈确实抓到了动作——一旦加入与动作无关的高方差变化，§5.1 说的竞争就会开始。
