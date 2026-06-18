## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定高效注意力 / SSM / 稀疏注意力** — 一页拿下面试核心要点（详见后文 §1–§11 推导）。

1. **二次瓶颈**：softmax attention 训练 $O(L^2 d)$ 时间，decode 时 KV cache 随上下文线性增长（$O(L)$ 显存、每步 $O(L)$ 计算）。逃逸有三条路：**线性注意力**、**SSM / Mamba**、**稀疏注意力**。

2. **线性注意力**：用 kernel feature map $\phi$ 把 softmax 换成 $\phi(q)^\top\phi(k)$，再靠**矩阵乘结合律**把 $\phi(Q)\big(\phi(K)^\top V\big)$ 从 $O(L^2)$ 降到 $O(L)$。等价于一个**隐藏状态是矩阵** $S_t\in\mathbb{R}^{d_k\times d_v}$ 的 RNN：$S_t = S_{t-1} + \phi(k_t) v_t^\top$。

3. **SSM / Mamba**：连续状态空间 $h_t = A h_{t-1} + B x_t,\; y_t = C h_t$，经离散化（$\Delta$ + 零阶保持）得递推。**Selective SSM（S6）**让 $B, C, \Delta$ 随输入变化（数据相关门控），打破 LTI 换来内容选择能力；配硬件感知 selective scan。

4. **Mamba-2 / SSD**：State Space Duality —— 一个 selective SSM **等价于一种结构化掩码线性注意力**（通过 1-半可分矩阵 / semiseparable matrix）。这让 SSM 能用 matmul 密集的 **chunkwise** 算法在 tensor core 上高效训练。

5. **Delta rule（DeltaNet）**：更新是**改写**而非纯累加——$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，那个 $(I-\beta k k^\top)$ 项先**擦掉旧关联**再写新值（在线最小二乘 / 误差修正）。Gated DeltaNet 再加一个标量/对角衰减门 $\alpha_t$ 做遗忘。

6. **Chunkwise parallel**：把序列切块，**块内**用二次的并行 attention，**块间**用 $O(1)$ 的递推状态跨块传递。这是让线性注意力 / SSM / DeltaNet 既能 $O(L)$ 推理、又能 matmul 并行训练的**核心桥梁**。

7. **稀疏注意力（可训练）**：保留 softmax 但只 attend 一个子集。**NSA**（DeepSeek，原生可训练 + 硬件对齐，三分支：compress + select + sliding window）；**MoBA**（Moonshot，MoE 式把 query 路由到 top-k key block）；**Lightning Attention**（MiniMax-01，IO-aware 线性）；**DeepSeek-V3.2 DSA**（轻量 lightning indexer 打分选 token）。

8. **推理杀手锏**：线性 / SSM 保持**固定大小**递推状态（序列长度上 $O(1)$）→ 常数显存 + 每 token $O(1)$ 解码；softmax 的 KV cache 随长度线性涨；稀疏减少 KV 访问/计算但仍存 KV。

9. **混合架构（hybrid）**：纯线性/SSM 的 **recall（联想回忆 / in-context copy）偏弱** —— recall–throughput 权衡。修法是在大量线性/SSM 层里**交错少数注意力层**（full / sliding / 门控 attention，具体类型和比例各模型不同）：Jamba、Hymba、Qwen3-Next、Kimi-Linear、MiniMax-01 都走这条路，兼顾吞吐与 recall。

## §1 直觉：softmax 注意力的二次瓶颈与三条逃逸路线

回顾标准 attention：$\text{Attn}(Q,K,V) = \text{softmax}\!\big(QK^\top/\sqrt{d}\big)V$，$Q,K,V\in\mathbb{R}^{L\times d}$。两个代价：

- **训练 / prefill**：要算 $L\times L$ 的 score 矩阵，时间 $O(L^2 d)$、显存 $O(L^2)$（FlashAttention 把显存压到 $O(L)$，但**时间仍是 $O(L^2 d)$**——FLOPs 没变）。
- **自回归 decode**：用 KV cache 后每步是 $O(L)$，但 **cache 本身随上下文线性增长**：第 $t$ 步要存 $t$ 个 key/value。长上下文下 KV cache 显存和访存带宽成为真正瓶颈。

二次的根源是 softmax 把 query–key 相似度做了**非线性归一化**，$\text{softmax}(QK^\top)$ 无法被拆成"先压缩 K/V 再和 Q 交互"的形式，所以必须显式物化 $L\times L$ 矩阵。三条逃逸路线，本质都是在问"能不能不物化这个 $L\times L$ 矩阵"：

> 💡 **三条逃逸路线的心智模型**
> - **(a) 线性注意力（linear attention）**：去掉 softmax，用 kernel feature map $\phi$ 近似它，于是 $\text{Attn}\approx \phi(Q)\big(\phi(K)^\top V\big)$ —— 靠**结合律**先把 K、V 压成一个 $d_k\times d_v$ 的小矩阵，再和 Q 交互。时间 $O(L d^2)$、decode 状态 $O(d_k d_v)$ **与 $L$ 无关**。代价：丢了 softmax 归一化，recall 变弱。
> - **(b) SSM / Mamba（state-space model）**：把序列建成一个**线性动力系统**的递推 $h_t = A h_{t-1}+B x_t$，天然 $O(L)$、固定状态；selective SSM 让转移参数数据相关，恢复内容选择。
> - **(c) 稀疏注意力（sparse attention）**：保留 softmax，但每个 query 只 attend 一个**子集**（block / 窗口 / top-k），把 $L^2$ 砍成 $L\cdot s$（$s\ll L$）。可以是固定 pattern，也可以是**可训练**的内容相关选择。

(a) 和 (b) 在数学上高度相通（§4 的 SSD 会把这层窗户纸捅破：线性注意力的递推 ≈ 一种 SSM）；(c) 是另一族思路——它**不放弃 softmax**，只放弃"全连接"。下文 §2–§6 走 (a)(b) 这条"次二次递推"主线，§7 讲 (c) 稀疏，§8–§11 讲推理收益、混合架构、工程与复杂度。

## §2 线性注意力

### 2.1　从 softmax 到 kernel feature map

写出单个 query 的输出（忽略 $\sqrt{d}$，记 $\text{sim}(q,k)$ 为相似度核）：

$$o_i = \frac{\sum_{j} \text{sim}(q_i, k_j)\, v_j}{\sum_{j}\text{sim}(q_i, k_j)}, \qquad \text{softmax 即 } \text{sim}(q,k)=\exp(q^\top k).$$

**线性注意力的核心一步**：假设相似度可以写成一个特征映射的内积 $\text{sim}(q,k) = \phi(q)^\top \phi(k)$，其中 $\phi:\mathbb{R}^{d}\to\mathbb{R}^{d_k}$（feature map，$d_k$ 是特征维）。代入：

$$o_i = \frac{\sum_j \big(\phi(q_i)^\top\phi(k_j)\big) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top\big(\sum_j \phi(k_j) v_j^\top\big)}{\phi(q_i)^\top\big(\sum_j \phi(k_j)\big)}.$$

关键在分子：$\sum_j \phi(k_j)v_j^\top$ 是一个 $\mathbf{d_k\times d_v}$ 的矩阵，**与 query 无关**——可以对所有 $j$ 算一次，所有 query 复用。这就是用**结合律**把 $\big(\phi(Q)\phi(K)^\top\big)V$ 重排成 $\phi(Q)\big(\phi(K)^\top V\big)$：

$$\underbrace{\big(\phi(Q)\phi(K)^\top\big)V}_{O(L^2 d_v)\ :\ \text{先算 }L\times L} \;=\; \underbrace{\phi(Q)\big(\phi(K)^\top V\big)}_{O(L d_k d_v)\ :\ \text{先算 }d_k\times d_v}.$$

> ✅ **结合律是全部魔法所在**
> 当 $L \gg d$ 时，$O(Ld_kd_v)\ll O(L^2 d)$。softmax 不能这么拆，因为 $\exp(q^\top k)$ 不是 $\phi(q)^\top\phi(k)$ 的有限维内积（要无限维 RFF 才能逼近，见 Performer）。

常见 $\phi$ 选择：

- **$\phi(x)=\text{elu}(x)+1$**（Katharopoulos 2020，"Transformers are RNNs"）：保证非负、简单。
- **Performer（FAVOR+）**：用随机特征 $\phi(x)=\exp(\omega^\top x - \|x\|^2/2)/\sqrt{m}$ 等**正随机特征**无偏估计 $\exp(q^\top k)$，可在精度上逼近真 softmax（代价是特征维 $m$ 要够大）。
- **RetNet / GLA**：基本上 $\phi=\text{id}$（或加 normalization），把重点放到**衰减 / 门控**而非 feature map（见 2.4）。

### 2.2　递推形式：线性注意力 = 隐藏状态是矩阵的 RNN

把上面**因果**化（causal，$o_i$ 只看 $j\le i$），并定义一个累积状态。在 causal 情形分子的求和变成前缀和：

$$S_i = \sum_{j\le i}\phi(k_j)v_j^\top \in\mathbb{R}^{d_k\times d_v}, \qquad z_i=\sum_{j\le i}\phi(k_j)\in\mathbb{R}^{d_k}.$$

于是得到**递推**（recurrent form）：

$$\boxed{\;S_t = S_{t-1} + \phi(k_t)\,v_t^\top, \qquad o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}\;}$$

这就是面试金句：**"线性注意力 = 一个隐藏状态是矩阵 $S\in\mathbb{R}^{d_k\times d_v}$ 的 RNN"**。对比普通 RNN 的向量 hidden state，这里状态是**外积累加出来的矩阵**——它存的是一个"$\phi(k)\to v$"的关联记忆（associative memory / fast weight）。

> ⚠️ **状态是矩阵，不是向量（高频追问）**
> 线性注意力的递推状态是 $S\in\mathbb{R}^{d_k\times d_v}$（外积 $\phi(k)v^\top$ 累加）。Decode 每步在序列长度上是 $O(1)$，但在状态规模上是 $O(d_k d_v)$——更新和读取都正比于这个矩阵大小。说成"$O(1)$ 状态"要补一句"$O(1)$ 是对序列长度而言"。

两种视角，同一计算：

| 视角 | 公式 | 复杂度 | 适合 |
| --- | --- | --- | --- |
| **并行 / 二次** | $o = \big(\phi(Q)\phi(K)^\top \odot M\big)V$（$M$ 为 causal mask） | $O(L^2 d)$ | 短序列 / 直观 |
| **递推 / 线性** | $S_t=S_{t-1}+\phi(k_t)v_t^\top,\ o_t=\phi(q_t)^\top S_t$ | $O(Ld_kd_v)$，状态 $O(d_kd_v)$ | decode / 长序列 |

注意：causal 线性注意力的**朴素并行式**仍要物化 $L\times L$（因为 mask 是逐元素的），所以训练时真正用的是 §6 的 **chunkwise** 折中——既不物化 $L^2$，又能用 matmul。

### 2.3　丢失 softmax 归一化的代价

softmax 有两个隐性好处被线性注意力丢掉了：

1. **归一化分母 $\phi(q)^\top z_t$ 不稳定**：softmax 的分母恒正且良定义（$\sum\exp\ge 1$），但 $\phi(q)^\top z_t$ 可能很小甚至接近 0（若 $\phi$ 没保证强正性），导致输出爆炸 / 数值不稳。很多实现干脆**去掉分母**（unnormalized linear attention），改用 §2.4 的衰减 / 门控 + 额外 normalization 来稳住量级。

2. **缺乏"尖锐选择"能力 → recall 弱**：softmax 能把概率质量高度集中到一两个最相关的 key（near one-hot 检索）；而线性注意力的 $\phi(q)^\top\phi(k)$ 是有限维内积，**表达不出任意尖锐的 selection**。当任务需要"从上下文里精确复制某个 token"（associative recall / in-context copy）时，线性注意力的固定大小状态会发生**容量冲突**：所有 $\phi(k)v^\top$ 叠加在同一个 $S$ 里互相干扰，越长越糊。

> ❌ **"线性注意力 $O(L)$ 所以全面更优" —— 错**
> 它在**长上下文检索 / 精确复制**类任务上系统性弱于 softmax，这是固定大小状态的**容量瓶颈**（信息被压进 $d_k\times d_v$ 的矩阵，超过容量就互相覆盖），不是实现问题。这正是后面 DeltaNet（改写式更新）、门控（选择性遗忘）、混合架构（保留少数 full attention 救 recall）要解决的核心痛点。

### 2.4　门控 / 衰减：RetNet 与 GLA

纯累加 $S_t=S_{t-1}+\phi(k_t)v_t^\top$ 有个问题：**老信息永不衰减**，状态被无限叠加污染。加一个衰减就好很多：

- **RetNet（Retentive Network）**：引入**标量指数衰减** $\gamma\in(0,1)$：$S_t = \gamma S_{t-1} + k_t v_t^\top$。它有三种等价形态——**parallel**（训练，类似带衰减 mask 的 attention）、**recurrent**（decode，$O(1)$ 状态）、**chunkwise**（长序列训练，块内并行块间递推）。$\gamma$ 通常**逐 head 固定**（非数据相关）。
- **GLA（Gated Linear Attention）**：把标量衰减升级成**数据相关的对角门** $G_t=\text{diag}(\alpha_t)$，$\alpha_t\in(0,1)^{d_k}$ 由输入算出：$S_t = G_t\,S_{t-1} + k_t v_t^\top$（逐通道遗忘）。GLA 给出了硬件高效的 chunkwise 形式，是把"门控 RNN"和"线性注意力"统一起来的代表作。

可以把这条线看成一个谱系：**纯线性注意力（无衰减）→ RetNet（标量衰减）→ GLA（数据相关对角门）→ Gated DeltaNet（门控 + 改写式更新，见 §5）**，复杂度都在 $O(L)$ 一档，区别在状态如何"遗忘"和"写入"。

## §3 状态空间模型 (SSM) 与 Mamba

### 3.1　连续 SSM 与离散化

状态空间模型来自控制论。**连续时间** SSM 把输入信号 $x(t)$ 映到输出 $y(t)$，中间是一个隐状态 $h(t)\in\mathbb{R}^{N}$：

$$h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t),$$

$A\in\mathbb{R}^{N\times N}$（状态转移），$B\in\mathbb{R}^{N\times 1}$，$C\in\mathbb{R}^{1\times N}$。要用在离散 token 序列上，需**离散化**：引入步长 $\Delta$，用**零阶保持（zero-order hold, ZOH）**把连续系统离散成递推：

$$\bar A = \exp(\Delta A), \qquad \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\,\Delta B \approx \Delta B,$$

$$\boxed{\;h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t\;}$$

这一步把"连续动力系统"变成一个**线性递推**——形式上和 RNN 一样，但转移矩阵 $\bar A$ 是结构化的。$\Delta$ 控制"系统反应快慢"：$\Delta$ 大 ≈ 更看重当前输入、快速更新；$\Delta$ 小 ≈ 状态记忆更持久。

### 3.2　HiPPO 初始化直觉

如果 $A$ 随机初始化，这个递推学不动长程依赖。**HiPPO（High-order Polynomial Projection Operators）**给出一个特殊的结构化 $A$ 矩阵，使隐状态 $h_t$ 成为**对历史输入的最优多项式投影系数**——即 $h_t$ 在数学上是"把过去整段信号用一组正交多项式（如 Legendre）压缩"的结果。直觉：HiPPO 让固定大小的状态以一种"有理有据、信息损失最小"的方式记住历史，这是 S4 / Mamba 能在长序列上 work 的关键先验。面试只需说清"HiPPO = 一个让状态最优压缩历史的结构化 $A$ 初始化"即可。

### 3.3　Selective SSM（S6 / Mamba）：让参数数据相关

经典 S4 是 **LTI（线性时不变）**的：$\bar A, \bar B, C$ 对所有 $t$ 相同，与输入无关。LTI 有个根本缺陷——它**无法做内容选择**（content-based reasoning）：同一个转移矩阵处理所有 token，没法"看到重要 token 就多记一点、看到废话就跳过"。这就是为什么 S4 在语言建模上不如 Transformer。

**Mamba 的核心创新（Selective SSM, 简称 S6）**：让 $B, C, \Delta$ **随输入 $x_t$ 变化**（数据相关 / input-dependent），从而实现**选择性**——根据内容决定记住什么、忽略什么。

$$B_t = \text{Linear}_B(x_t),\quad C_t = \text{Linear}_C(x_t),\quad \Delta_t = \text{softplus}\big(\text{Linear}_\Delta(x_t)\big),$$

$$h_t = \bar A_t\,h_{t-1} + \bar B_t\,x_t,\qquad y_t = C_t\,h_t, \qquad \bar A_t=\exp(\Delta_t A).$$

> ⚠️ **到底是谁数据相关？（极易答错）**
> Mamba 里 **$B, C, \Delta$ 是数据相关的，$A$ 本身仍是结构化、与输入无关的固定参数**（每个 channel 一个学到的 $A$）。$\Delta_t$（数据相关）通过 $\bar A_t=\exp(\Delta_t A)$ 把"有效衰减率"变成数据相关——所以**选择性主要由 $\Delta$ 和 $B,C$ 注入，而不是直接让 $A$ 随输入变**。面试说"Mamba 让 $A$ 数据相关"是不准确的，要点出 $A$ 结构化固定、$\Delta$ 是数据相关的门。

**代价**：一旦 $\bar A_t$ 随时间变化，系统不再是 LTI，**S4 赖以高效训练的全局卷积（FFT）形式失效**了——因为卷积核 $\bar C\bar A^k \bar B$ 不再是固定的。Mamba 的解法是**硬件感知的 selective scan**：把这个含时递推写成一个 GPU kernel，用并行前缀扫描（parallel scan / associative scan）在序列维上并行，且把状态保持在 SRAM、只在必要时写回 HBM（kernel fusion，避免物化 $L\times N$ 的中间状态）。这就是 Mamba"selective + hardware-aware"两个关键词的来历。

### 3.4　Mamba block 的全貌

完整 Mamba block 不只是 SSM，还包了门控和卷积：input → 线性投影扩张 → 一个**短因果 1D 卷积**（局部 mixing）→ SiLU → **selective SSM** → 一个**门控分支**（SiLU(z) 逐元素相乘）→ 投影回去。短卷积负责局部模式、SSM 负责长程、门控负责选择性放行。记住"Mamba ≠ 纯 SSM，而是 conv + selective SSM + gate 的组合块"，比只背 SSM 公式更能体现理解。

## §4 Mamba-2 与 State Space Duality (SSD)

### 4.1　核心二象性：SSM ⇔ 结构化掩码注意力

Mamba-1 的 selective scan 是个手写 CUDA kernel，**不走 tensor core 的 matmul 路径**，所以虽然渐进复杂度低、但常数大、和高度优化的 FlashAttention 比并没有压倒性速度优势。Mamba-2 的论文《Transformers are SSMs》提出 **State Space Duality (SSD)**，把 SSM 和 attention 在数学上接通。

把 selective SSM 的递推**展开**（设 $h_0=0$，标量 $\bar A$ 情形）：

$$y_t = \sum_{s\le t} C_t\Big(\prod_{r=s+1}^{t}\bar A_r\Big)\bar B_s\, x_s = \sum_{s\le t} \underbrace{C_t\Big(\prod_{r=s+1}^{t}\bar A_r\Big)\bar B_s}_{\textstyle M_{ts}}\, x_s.$$

把它写成矩阵形式 $y = M x$，其中 $M$ 是一个**下三角矩阵**，$M_{ts}=C_t\big(\prod_{r=s+1}^t \bar A_r\big)\bar B_s$（$t \lt s$ 时为 0）。这个 $M$ 长得就像一个**带衰减的 causal attention 矩阵**：$C_t$ 像 query、$\bar B_s$ 像 key、中间连乘 $\prod\bar A$ 是"位置衰减"。

> ✅ **SSD 一句话**
> **一个 selective SSM 等价于用一个 1-半可分矩阵（1-semiseparable matrix）$M$ 做的结构化掩码注意力（structured masked attention）。** "半可分"指 $M$ 的下三角任意子块的秩 $\le 1$（由 $C, \bar A, \bar B$ 的连乘结构决定），这正是 SSM 递推能 $O(L)$ 算的代数原因；而把 $M$ 当矩阵直接乘 $x$，又能用 matmul。两条计算路径（线性递推 vs 二次矩阵乘）算的是**同一个 $M$**——这就是 duality。

> ⚠️ **别过度宣称**
> SSD 说的是"SSM ≡ **一种结构化的、带特定衰减掩码的**线性注意力"，**不是**"SSM = 普通 softmax attention"。$M$ 没有 softmax、是半可分结构化的。这个等价是在**线性注意力 / 掩码线性 attention** 的层面成立，措辞要精确。

### 4.2　好处：用 chunkwise matmul 训练 SSM

有了 SSD，Mamba-2 就能用 §6 的 **chunkwise / block-decomposition** 算法训练：

- **块内（intra-chunk / diagonal blocks）**：直接物化小的 $M$ 子块，用 matmul（tensor core）算二次 attention——块很小所以 $L_{\text{chunk}}^2$ 可接受。
- **块间（inter-chunk / off-diagonal blocks）**：因为是半可分（秩 1）结构，跨块只需传递一个**低秩的状态**（就是 SSM 的隐状态 $h$），用矩阵乘批量处理。

效果：Mamba-2 比 Mamba-1 训练快好几倍（吃满 tensor core），且因为状态维 $N$ 可以开得更大（matmul 便宜），表达力也更强。Mamba-2 还顺手简化了结构（$A$ 退化为**标量乘单位阵** $\bar A_t = a_t I$，即每步一个标量衰减），让 SSD 推导和实现都更干净。

> 💡 **面试金句**：Mamba-1 = 硬件感知 selective **scan**（自定义 kernel，不吃 tensor core）；Mamba-2 = 借 **SSD** 把同一个计算重写成 **matmul-heavy chunkwise** 形式（吃满 tensor core），训练更快、状态可更大。二者算法等价，差在"用 scan 还是用 matmul 实现"。

## §5 Delta rule：DeltaNet 与 Gated DeltaNet

### 5.1　从"累加"到"改写"

线性注意力的状态更新是**纯累加**：$S_t = S_{t-1} + v_t k_t^\top$（这里把 $\phi(k)$ 简记 $k$）。问题前面说过——同一个 key 反复出现、或不同 key 互相干扰时，信息只会**叠加污染**，没有"更新 / 覆盖"机制。

**Delta rule（来自 fast-weight / 在线学习）** 换一种写法：把状态 $S$ 看成一个线性映射 $k\mapsto S k$（用 key 检索 value），每来一个 $(k_t,v_t)$，我们希望 $S$ 在 $k_t$ 上的"预测" $S_{t-1}k_t$ 逼近目标 $v_t$。做一步在线梯度下降 / 最小二乘修正：

$$S_t = S_{t-1} - \beta_t\big(\underbrace{S_{t-1}k_t - v_t}_{\text{prediction error}}\big)k_t^\top = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top.$$

$$\boxed{\;S_t = S_{t-1}\big(I - \beta_t\, k_t k_t^\top\big) + \beta_t\, v_t k_t^\top\;}$$

（假设 $\|k_t\|=1$ 归一化）。$\beta_t\in(0,1]$ 是**写入强度 / 学习率**（也可数据相关）。

> ✅ **$(I-\beta k k^\top)$ 是 DeltaNet 的灵魂**
> 这个投影项先把状态在 $k_t$ 方向上的**旧关联擦掉**（$S_{t-1}$ 右乘 $(I-\beta kk^\top)$ 衰减掉 $k_t$ 方向的成分），再写入新的 $\beta v_t k_t^\top$。这就是"**改写 / 覆盖**"而不是"叠加"。
> 对比纯线性注意力：$S_t = S_{t-1} + v_t k_t^\top$（没有那个投影项 = 永远只加不改）。

直观看极端情形：

- $\beta_t = 1$ 且 $k_t$ 与历史所有 key 正交：$(I-k_tk_t^\top)$ 不动历史，等价于"先清掉 $k_t$ 方向（本来就空）再写入"——退化为普通线性注意力的一次写入。
- $k_t$ 重复出现：delta rule 会**覆盖**上一次在 $k_t$ 上写的 value，而纯线性注意力会把两次 value 加起来（错误叠加）。这正是 DeltaNet 在 associative recall 上显著强于普通线性注意力的原因。

### 5.2　Gated DeltaNet：加一个遗忘门

DeltaNet 的 delta rule 解决了"精确改写"，但缺一个**全局遗忘**机制（老信息随时间整体淡出）。**Gated DeltaNet** 把 GLA 式的衰减门和 delta rule 结合：

$$S_t = \alpha_t\, S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,$$

其中 $\alpha_t\in(0,1)$（标量或对角，数据相关）是**衰减 / 遗忘门**。一句话：**$\alpha_t$ 管"整体遗忘多少历史"，$\beta_t$ 管"在 $k_t$ 这个方向上改写多强"**。两个机制正交互补——门控负责连续淡忘，delta rule 负责精确覆写。这让 Gated DeltaNet 在长上下文 recall 和状态利用率上都更强，也是 Qwen3-Next、Kimi-Linear（KDA 是其变体）等近期混合架构选它做线性层的原因。

### 5.3　怎么并行：WY 表示 + chunkwise

delta rule 有个 $(I-\beta kk^\top)$ 连乘，看起来强顺序、难并行。但一个块内 $T$ 步的连乘 $\prod_t (I-\beta_t k_t k_t^\top)$ 可以用 **Householder 矩阵乘积的 WY 表示**（数值线性代数里 QR 分解的经典技巧）展开成 $I - W Y^\top$ 的紧凑形式，从而块内用 matmul 批量算、块间传递状态。这就是 DeltaNet 系能在序列长度上 chunkwise 并行训练的关键（Yang et al. 2024 的主要工程贡献之一）。面试记住"delta rule 的块内并行靠 WY / Householder 表示"即可，不必背展开式。

## §6 Chunkwise parallel：训练时怎么并行

### 6.1　为什么需要它

线性注意力 / SSM / DeltaNet 有两副面孔：

- **递推形式**好在**推理**：$O(1)$ 状态、每 token $O(1)$，但**强顺序**（$S_t$ 依赖 $S_{t-1}$），训练时逐 token 跑满 $L$ 步、GPU 利用率极低。
- **并行 / 二次形式**好在**训练**：能 matmul，但 causal 情形要物化 $L\times L$，长序列显存炸。

**Chunkwise parallel** 是二者的折中桥梁，是让这一整族方法**实际可训练**的核心技巧。思路：把序列切成 $L/C$ 个大小为 $C$ 的 chunk，**块内并行、块间递推**。

### 6.2　两段式：intra-chunk（块内并行）+ inter-chunk（块间递推）

以无门控线性注意力为例（$S_t=S_{t-1}+k_tv_t^\top$，状态 $S\in\mathbb{R}^{d_k\times d_v}$，这里用 $\phi=\text{id}$、$k$ 即 $\phi(k)$）。设第 $c$ 个 chunk 的 query/key/value 为 $Q_c,K_c,V_c\in\mathbb{R}^{C\times d}$，进入该 chunk 前的状态为 $S_c$（= 前面所有 chunk 的 $\sum k v^\top$）。chunk 内第 $i$ 个 query 的输出拆成两部分：

$$o_i = \underbrace{q_i^\top S_c}_{\textbf{inter: 历史块的贡献}} + \underbrace{\sum_{j\le i,\, j\in c} (q_i^\top k_j)\, v_j}_{\textbf{intra: 本块内的 causal attention}}.$$

写成 chunk 级矩阵形式：

$$O_c = \underbrace{Q_c\, S_c}_{[C\times d_k][d_k\times d_v]=[C\times d_v]} \;+\; \underbrace{\big(\,(Q_c K_c^\top)\odot M\,\big)\,V_c}_{\text{块内二次, } M\text{ 是 } C\times C\text{ causal mask}},$$

块结束后**更新跨块状态**：

$$S_{c+1} = S_c + K_c^\top V_c \quad (\in\mathbb{R}^{d_k\times d_v}).$$

> 💡 **三个 matmul 看懂 chunkwise**
> - **inter-chunk** `Q_c @ S_c`：历史被压成一个 $d_k\times d_v$ 矩阵 $S_c$，本块所有 query 一次性读取（这部分**线性**于 $L$）。
> - **intra-chunk** `((Q_c K_c^T) ⊙ M) @ V_c`：只在 $C\times C$ 的小块内做二次 attention（$C$ 是常数，如 64/128，所以**总代价线性**于 $L$）。
> - **state update** `S_c + K_c^T V_c`：把本块信息累加进状态，传给下一块。

复杂度：$O\!\big(\tfrac{L}{C}\cdot(Cd_kd_v + C^2 d)\big)=O(Ld_kd_v + LCd)$，对固定 $C$ 是 $O(L)$，且全是 matmul（吃 tensor core）。**chunk size $C$ 是关键旋钮**：$C$ 太小→块间递推占主导、并行度低；$C$ 太大→块内二次项 $C^2$ 变贵。注意本节的状态更新 $S_{c+1}=S_c+K_c^\top V_c$ **仅对无门控加性版成立**；带门控 / 衰减时要折进相应因子——GLA / Mamba-2 在 $S_c$ 与块内 mask 上乘累积衰减 $\prod\alpha$，**DeltaNet 块内还要带 $(I-\beta_t k_tk_t^\top)$ 改写项**（用 §5.3 的 WY 表示展开）。骨架（块内并行 + 块间传状态）完全一样，只是块内/跨块的算子从纯加性换成"衰减 × 改写"。

### 6.3　代码：递推 vs chunkwise，输出应一致

```python
import torch

def linear_attn_recurrent(Q, K, V):
    """逐 token 递推形式。Q,K,V: [L, d]  ->  O: [L, d]
       状态 S: [d_k, d_v] 矩阵（这里 d_k=d_v=d）。"""
    L, d = Q.shape
    S = torch.zeros(d, d, dtype=Q.dtype)          # [d_k, d_v] 矩阵状态
    O = torch.zeros(L, d, dtype=Q.dtype)
    for t in range(L):
        S = S + torch.outer(K[t], V[t])           # S += k_t v_t^T   外积累加
        O[t] = Q[t] @ S                           # o_t = q_t^T S    [d]
    return O

def linear_attn_chunkwise(Q, K, V, C=4):
    """块内并行 + 块间递推。数学上与 recurrent 完全等价。"""
    L, d = Q.shape
    S = torch.zeros(d, d, dtype=Q.dtype)          # 跨块状态 [d_k, d_v]
    O = torch.zeros(L, d, dtype=Q.dtype)
    tri = torch.tril(torch.ones(C, C, dtype=Q.dtype))  # 块内 causal mask [C, C]
    for s in range(0, L, C):
        e = min(s + C, L)
        Qc, Kc, Vc = Q[s:e], K[s:e], V[s:e]       # [c, d]
        c = e - s
        m = tri[:c, :c]
        inter = Qc @ S                            # 历史块贡献 [c, d]
        intra = ((Qc @ Kc.t()) * m) @ Vc          # 块内 causal attention [c, d]
        O[s:e] = inter + intra
        S = S + Kc.t() @ Vc                        # 更新状态 [d_k, d_v]
    return O

if __name__ == "__main__":
    torch.manual_seed(0)
    L, d = 13, 6
    Q, K, V = torch.randn(L, d), torch.randn(L, d), torch.randn(L, d)
    o1 = linear_attn_recurrent(Q, K, V)
    o2 = linear_attn_chunkwise(Q, K, V, C=4)
    print("max |Δ| =", (o1 - o2).abs().max().item())   # 期望 ~1e-6（纯浮点误差）
    assert torch.allclose(o1, o2, atol=1e-5)
```

两条路径（递推 / chunkwise）算的是同一个因果线性注意力，输出应在浮点误差内一致——这是验证实现正确性的黄金 sanity check（见 §A）。

## §7 稀疏注意力 (sparse / 可训练块稀疏)

### 7.1　另一族思路：保留 softmax，只 attend 子集

前面 §2–§6 是"放弃 softmax 换递推"。稀疏注意力走另一条路：**softmax 不动，但每个 query 只看一个子集** $S(i)\subseteq\{1..L\}$，把 $L^2$ 砍成 $L\cdot |S(i)|$。早期（2019–2020）是**固定 pattern**：

- **Longformer / BigBird**：滑动窗口（局部）+ 少量全局 token + （BigBird 加）随机 token，固定稀疏图，$O(L)$。问题：pattern 是人为设计的，未必匹配数据；且很多 kernel 在现代 GPU 上 IO 不友好。

近两年（2025）的前沿是**可训练 / 内容相关 + 硬件对齐**的稀疏注意力——让模型**自己学**该看哪些块，且 pattern 设计成 GPU 能高效跑。

### 7.2　NSA（Native Sparse Attention, DeepSeek）

**NSA 的两个关键词：原生可训练（natively trainable）+ 硬件对齐（hardware-aligned）。** 不是事后给稠密模型套一个稀疏 mask，而是**从预训练开始就用稀疏结构**，端到端可学。每个 query 的输出是**三个分支的门控加和**：

> 💡 **NSA 的三分支（高频考点，必须背全）**
> - **(1) Compressed（压缩粗粒度）**：把远处的 key/value 按块**压缩**成少量"代表 token"（块内聚合），让 query 用很低成本看到全局粗略信息。
> - **(2) Selected（选择细粒度 top-k 块）**：用压缩分支算出的块重要性，**选 top-k 个最相关的 key block**，在这些块上做完整（细粒度）attention——这是"精确检索"的主力。
> - **(3) Sliding window（滑动窗口）**：固定 attend 最近的若干 token，保证**局部上下文**不丢（局部模式对语言建模很重要）。
> 三个分支各出一个输出，用一个**学到的门控**加权融合。

NSA 的"硬件对齐"指：块大小、选择粒度都按 GPU 的 memory access / tensor core 友好方式设计（块连续访问、avoid 随机 gather 的低效），所以它不仅理论 $O(Ls)$，**实测在长序列上对 full attention 有真实墙钟加速**（训练和解码都加速）——这是它区别于很多"理论稀疏但实际更慢"方法的地方。

### 7.3　MoBA（Mixture of Block Attention, Moonshot）

**MoBA 把 MoE 的"路由"思想搬到 attention 上**：把 key/value 切成若干 block，每个 query 像 MoE router 一样**只路由到 top-k 个最相关的 key block**，在这 k 个块上做 softmax attention，其余块完全跳过。"哪些块相关"由 query 和每个块的代表（如块内 key 的均值）打分决定。好处：

- 路由是**可学的、内容相关的**（不是固定 pattern）；
- 块粒度选择对硬件友好（块连续）；
- 设计成可以在 **full attention 和 MoBA 之间无缝切换**（同一套权重，训练时可混用），便于已有模型迁移。

一句话对比：**NSA = compress + select + window 三分支固定融合；MoBA = MoE 式 router 把 query 分配到 top-k key block**。两者都是 2025 年"可训练稀疏"的代表，思路相通（内容相关地选块）但结构不同。

### 7.4　Lightning Attention（MiniMax-01）与 DeepSeek-V3.2 DSA

- **Lightning Attention（MiniMax-01）**：严格说它是**IO-aware 的线性注意力**实现（不是 softmax 稀疏），用类似 FlashAttention 的 tiling + chunkwise 把线性注意力做到 IO 高效，配合混合架构（间插少量 softmax 层）撑起百万级上下文。放在这里是因为 MiniMax-01 是"线性 + 少量 full attention 混合"的大规模落地代表（见 §9）。
- **DeepSeek-V3.2 的 DSA（DeepSeek Sparse Attention）/ lightning indexer**：用一个**轻量索引器（lightning indexer）**给每个 query 快速打分"该 attend 哪些历史 token"，再只在被选中的 token 上做注意力。核心是用一个**便宜的打分网络**做 token 级选择，把长上下文 attention 的成本降下来，同时尽量不掉质量。

> ⚠️ **别把 Lightning Attention 归成"稀疏 softmax"**
> Lightning Attention 本质是**线性注意力**的高效实现（IO-aware），归在 §2 那一族；DSA / NSA / MoBA 才是"保留 softmax、选子集"的稀疏注意力。面试若把它们混为一谈会露怯——区分点是"有没有放弃 softmax、有没有固定大小递推状态"。

### 7.5　固定 pattern vs 可训练稀疏（一句话对比）

| 维度 | 固定 pattern（Longformer/BigBird 时代） | 可训练稀疏（NSA / MoBA, 2025） |
| --- | --- | --- |
| 稀疏结构 | 人为设计（窗口+全局+随机） | 模型**学**出来（内容相关选块） |
| 训练 | 通常 fine-tune 时套上 | **原生**从预训练就稀疏（NSA） |
| 硬件 | 多数 IO 不友好 | **硬件对齐**（块连续、tensor-core 友好） |
| 效果 | 长文档可用，质量有损 | 接近 full attention，且有真实加速 |

## §8 KV cache 与推理：线性/SSM 的杀手锏

### 8.1　三族方法的推理画像

自回归 decode 阶段（一个一个吐 token），三族方法的资源画像截然不同：

| 方法 | decode 每步计算 | "记忆"随上下文 $L$ 的增长 | 每步访存 |
| --- | --- | --- | --- |
| **softmax + KV cache** | $O(L\,d)$（新 q 对全部历史 k/v） | KV cache **$O(L)$ 线性增长** | 读整个 KV cache（带宽瓶颈） |
| **线性注意力 / SSM** | $O(d_k d_v)$（更新+读固定状态） | **固定大小状态 $O(d_k d_v)$，与 $L$ 无关** | 只读写那个小状态 |
| **稀疏（NSA/MoBA）** | $O(s\,d)$（只看选中的 $s$ 个 token/块） | 仍需**存全部 KV**（$O(L)$），但只**访问**子集 | 读子集（比 full 少） |

> ✅ **这就是线性 / SSM 的最大卖点**
> softmax 的 KV cache 随上下文**线性膨胀**——128K 上下文、70B 模型的 KV cache 能到几十 GB，是长上下文推理的头号显存 / 带宽杀手。线性注意力 / SSM 把历史压进一个**固定大小**的递推状态（$d_k\times d_v$ 或 SSM 的 $N$ 维），**显存与上下文长度无关、每 token 解码 $O(1)$**——这是它们在长上下文 / 高吞吐 serving 上最硬的优势。

### 8.2　KV cache 显存对比（直觉数字）

softmax KV cache（per sample，与 §attention 篇一致）：

$$\text{KV cache} = L \cdot n_{\text{layers}} \cdot 2 \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot \text{bytes}.$$

随 $L$ **线性增长**。而线性注意力 / SSM 的"状态显存"是：

$$\text{state} = n_{\text{layers}} \cdot (\text{每层状态大小}) ,\quad \text{每层} \approx d_k\, d_v\ (\text{线性}) \ \text{或}\ N\, d\ (\text{SSM}),$$

**不含 $L$**。所以哪怕上下文从 4K 涨到 1M，线性 / SSM 层的状态显存纹丝不动——代价就是 §2.3 说的 recall 容量上限。稀疏注意力在这个维度上**不省**（仍存全部 KV），它省的是**计算 / 访存**（只算子集），这是它和线性/SSM 的本质分野。

> ⚠️ **混合架构的 KV cache 别忘了 full attention 层**
> Hybrid（§9）里那几层 full attention **仍然有随 $L$ 增长的 KV cache**——所以混合模型不是"完全常数显存"，而是"大部分层常数 + 少数层线性"。算长上下文显存预算时这点常被忽略。

## §9 混合架构 (hybrid)

### 9.1　recall–throughput 权衡：为什么纯线性/SSM 不够

纯线性 / SSM 模型吞吐高、状态固定，但**联想回忆（associative recall）弱**：把某个事实塞进固定大小状态后，要在很久以后**精确取回**，会受状态容量限制（信息被覆盖 / 干扰）。多个研究（如 Based、Zoology 等系列工作）指出存在一条 **recall–memory（throughput）的 Pareto 前沿**：状态越小越快、但 recall 越差。softmax attention 在这条前沿的"高 recall、低吞吐"端，纯 SSM 在"高吞吐、低 recall"端。

**修法出奇地简单粗暴：在大量线性 / SSM 层里，交错插入少数几层 full softmax attention。** 那几层 full attention 负责"精确检索 / in-context copy"，其余线性/SSM 层负责"廉价地处理长程上下文"。少量 full attention 层就能把 recall 补回到接近纯 Transformer，而整体复杂度和 KV cache 仍由线性/SSM 层主导（接近线性）。

> ✅ **为什么 hybrid 赢**
> 在固定预算下，hybrid 同时拿到了两端的好处：**线性/SSM 层 → 长上下文的吞吐与显存（常数状态）**；**少数 full attention 层 → recall / 精确检索**。经验上一个很小的 full-attention 比例（常见 1:5 到 1:7，即每 6–8 层插 1 层 full）就能逼近纯注意力的质量，而 KV cache 砍掉一大截。这就是 2024–2025 大量工业级长上下文模型选混合的原因。

### 9.2　代表性混合模型与比例

| 模型 | 线性 / SSM 成分 | full attention 成分 | 大致比例 / 特点 |
| --- | --- | --- | --- |
| **Jamba** | Mamba 层 | Transformer 层 + MoE | Mamba:Attention 约 7:1，块内插 1 层 attention；首个大规模 Mamba-Transformer 混合 |
| **Zamba** | Mamba 主干 | **共享**一个全局 attention 块（被多处复用） | 用单个共享 attention 省参数 |
| **Hymba** | SSM head | attention head | **同一层内并行**跑 attention head 与 SSM head（head 级混合，非层级交错） |
| **Qwen3-Next** | **Gated DeltaNet** 层 | gated（输出门控）full attention 层 | 大比例 Gated DeltaNet + 少量门控注意力，主打长上下文效率 |
| **Kimi-Linear** | **KDA**（Kimi Delta Attention，Gated DeltaNet 变体） | 周期性 full attention（MLA） | 线性层:full 约 3:1，长上下文吞吐与 KV cache 大幅优化 |
| **MiniMax-01** | **Lightning Attention**（线性） | softmax attention | 大比例 lightning + 周期性 softmax，支撑百万级上下文 |

> 💡 **Hymba 的 head 级混合 vs 其余的层级交错**
> 多数 hybrid 是**层级**交错（某些层是 SSM、某些层是 attention）；**Hymba 是层内**把 attention head 和 SSM head **并行**放在同一层（fused / parallel heads）。这是两种不同的混合粒度，面试能点出"层级 vs head 级"会加分。

### 9.3　近期趋势

2024→2025 的清晰趋势：**线性层从"纯加性线性注意力 / Mamba"升级到"门控 + 改写式更新"（Gated DeltaNet / KDA）**，因为后者 recall 更强、需要的 full attention 层更少；同时 full attention 那几层也常用上 GQA / MLA / 输出门控等省 KV 的手段。换句话说，hybrid 的两端都在各自变强：线性端靠 delta rule + 门控逼近 attention 的 recall，attention 端靠 MLA/GQA 压 KV cache。

## §10 工程实践与常见误区

- **线性注意力不是免费午餐**：它的 recall 弱是**固定大小状态的容量本质**，不是 bug。需要精确长程检索的任务（多跳 QA、长代码补全、大海捞针）上，纯线性/SSM 会掉点——这也是 hybrid 存在的根本理由。
- **"线性 = $O(N)$ 所以一定更快更好"是错的**：(1) 渐进复杂度低 ≠ 实际更快——常数和访存模式很重要，FlashAttention 高度优化后，短到中等序列上 softmax 往往**更快**；线性/SSM 的优势要到很长序列才显现。(2) 更快 ≠ 更好——recall 短板独立于速度。
- **chunk size 要调**：$C$ 太小→并行度低、块间递推占主导；太大→块内 $C^2$ 项变贵 + 显存涨。典型 $C\in[64,256]$，按序列长度 / 硬件调。
- **没有 softmax 归一化 → 数值稳定性要单独管**：纯累加状态会无界增长 / 量级漂移。实践靠**衰减门（$\gamma,\alpha_t$）**、**对状态或输出做 normalization**（RMSNorm / L2）、**$\beta,\Delta$ 的有界激活（sigmoid/softplus）**来稳住。直接搬 softmax attention 的实现习惯过来容易炸。
- **稀疏的硬件对齐很关键**：理论稀疏 ≠ 实际加速。随机 gather、非连续访问会让稀疏 kernel 比 dense 还慢——这就是 NSA / MoBA 强调"块连续、tensor-core 友好"的原因。评估稀疏方法**一定看真实墙钟 / 端到端吞吐**，别只看 FLOPs。
- **和 RoPE 的关系**：线性注意力 / SSM 不像 softmax attention 那样直接套 RoPE——它们的"位置信息"来自递推结构本身（衰减 $\gamma^{t-s}$、$\Delta$、门 $\alpha$ 提供了隐式的相对位置 / 时序衰减）。混合架构里**full attention 层用 RoPE，线性/SSM 层用各自的衰减机制**；硬把 RoPE 塞进线性注意力的 $\phi(q),\phi(k)$ 要小心（会和衰减项相互作用）。
- **区分"linear attention"和"sub-quadratic 但仍是 attention"**：FlashAttention 是**精确 softmax** 的 IO 优化（仍 $O(L^2)$ FLOPs，只省显存），**不是**线性注意力；Performer/Linformer 才是次二次近似。线性注意力 / SSM 是"**用固定状态递推替代 attention**"，根本上换了计算范式。这三类常被面试者混为一谈。

> ⚠️ **最大 footgun #1：拿渐进复杂度当实际性能**
> "我的方法 $O(L)$，所以一定比 $O(L^2)$ 的 Transformer 快/好"——错两次：实际速度看常数和访存（FlashAttention 很强），质量看 recall（线性有短板）。**任何高效注意力 claim 都要给真实墙钟 + 长上下文 recall benchmark**，否则站不住。

> ❌ **最大 footgun #2：以为 SSD / SSM "就是" attention 或线性注意力"约等于" softmax**
> SSD 是"SSM ≡ **结构化掩码（半可分）线性注意力**"的精确等价，**不是** softmax attention；线性注意力是 softmax 的**有限维近似**，在尖锐选择 / recall 上**本质受限**。把这些等价 / 近似过度宣称成"等同 / 无损替代 softmax"是经典错误，会被追问到底。

## §11 复杂度与资源

下表按"训练时间 / decode 每步 / KV·状态显存 / recall / 长度外推 / 可并行"对比五族方法（$L$=序列长，$d$=隐维，$d_k d_v$=线性状态，$N$=SSM 状态维，$s$=稀疏选中数，$C$=chunk）。表中绝对值/范数用 $\lvert\cdot\rvert$、$\lVert\cdot\rVert$ 记号。

| 方法 | 训练时间 | decode 每步 | KV / 状态显存（随 $L$） | recall | 长度外推 | 可并行训练 |
| --- | --- | --- | --- | --- | --- | --- |
| **softmax (Flash)** | $O(L^2 d)$ | $O(L d)$ | KV cache $O(L)$ 线性增长 | **强** | RoPE+YaRN 中等 | 是（matmul） |
| **线性注意力** | $O(L d_k d_v)$（chunkwise） | $O(d_k d_v)$ | 状态 $O(d_k d_v)$，**与 $L$ 无关** | 弱（容量瓶颈） | 好（衰减自然外推） | 是（chunkwise matmul） |
| **SSM / Mamba-2** | $O(L\,N d)$（SSD chunkwise） | $O(N d)$ | 状态 $O(N d)$，**与 $L$ 无关** | 中（强于纯线性，弱于 softmax） | 好 | 是（SSD matmul） |
| **DeltaNet / Gated** | $O(L d_k d_v)$（WY chunkwise） | $O(d_k d_v)$ | 状态 $O(d_k d_v)$，**与 $L$ 无关** | 中-强（改写式更新提升 recall） | 好 | 是（WY chunkwise） |
| **稀疏 (NSA)** | $O(L\,s\,d)$（$s\ll L$） | $O(s\,d)$（仍存全 KV） | KV $O(L)$（存全部，访问子集） | **强**（保留 softmax） | 取决于窗口/选择 | 是（块 matmul，硬件对齐） |

关键读表要点：

- **状态显存与 $L$ 无关**只有线性 / SSM / DeltaNet 三族做到；稀疏仍存 $O(L)$ KV（省的是计算/访存）。
- **recall** 大致排序（**经验值**，随模型规模 / 训练 recipe / benchmark 浮动，不是定理；尤其 DeltaNet/Gated 与 Mamba-2 的相对位置很依赖具体设置）：softmax ≈ 稀疏(NSA) > DeltaNet/Gated > Mamba-2 > 纯线性注意力。注意 NSA 的"≈ softmax"是 **softmax-over-selected**（保留 softmax，但只在选中的块/token 上算）——selector 命中时接近 full attention 的 recall，但**不是 exact full softmax**，极端大海捞针若选错块仍可能 miss。这正是 hybrid 想把 softmax 的 recall 和线性的吞吐结合的动机。
- **训练全可并行**：五族都能 matmul 并行训练（线性/SSM/DeltaNet 靠 chunkwise，稀疏靠块），区别在常数和访存友好度。
- **decode**：线性/SSM/DeltaNet 是序列长度上的真 $O(1)$；softmax 是 $O(L)$；稀疏是 $O(s)$ 但要维护全 KV。

## §12 25 高频面试题

按难度分三档，点开看答案要点 + 易踩坑。L2/L3 是顶级 lab 深水区（SSD、delta rule、chunkwise、NSA 三分支、hybrid 等）。

### L1必会题

<details>

<summary>Q1. softmax attention 的两个二次/线性瓶颈分别是什么？</summary>

- 训练/prefill：物化 $L\times L$ score 矩阵，时间 $O(L^2 d)$（FlashAttn 省显存但不省 FLOPs）
- decode：KV cache 随上下文**线性增长**（$O(L)$ 显存 + 每步 $O(L)$ 访存）
- 三条逃逸：线性注意力、SSM/Mamba、稀疏注意力

只说"$O(n^2)$"，不区分"训练二次时间"和"decode 线性 KV cache"两个不同瓶颈。

</details>

<details>

<summary>Q2. 线性注意力为什么能做到 $O(L)$？</summary>

- 用 feature map $\phi$ 把 $\text{sim}(q,k)$ 写成内积 $\phi(q)^\top\phi(k)$
- 靠**结合律**把 $\big(\phi(Q)\phi(K)^\top\big)V$ 重排成 $\phi(Q)\big(\phi(K)^\top V\big)$
- 先把 K、V 压成 $d_k\times d_v$ 小矩阵，再和 Q 交互，避免物化 $L\times L$

只说"去掉 softmax"，讲不清"结合律 + 先压 KV"这个机制；或不知道 softmax 不能这么拆（需无限维）。

</details>

<details>

<summary>Q3. "线性注意力是一个 RNN" 这句话怎么理解？它的状态是什么形状？</summary>

- causal 线性注意力可写成递推 $S_t=S_{t-1}+\phi(k_t)v_t^\top$，$o_t=\phi(q_t)^\top S_t$
- 状态 $S\in\mathbb{R}^{d_k\times d_v}$ 是一个**矩阵**（外积累加的关联记忆）
- 不同于普通 RNN 的向量 hidden state

把状态说成向量（错，是矩阵）；或不知道它和并行式是同一计算的两副面孔。

</details>

<details>

<summary>Q4. SSM 的基本递推是什么？为什么要离散化？</summary>

- 连续：$h'(t)=Ah(t)+Bx(t),\ y(t)=Ch(t)$
- 离散（ZOH，步长 $\Delta$）：$h_t=\bar A h_{t-1}+\bar B x_t,\ y_t=Ch_t$，$\bar A=\exp(\Delta A)$
- token 序列是离散的，必须把连续动力系统离散成递推才能用

写不出连续→离散这一步；或不知道 $\Delta$ 是步长 / 控制反应快慢。

</details>

<details>

<summary>Q5. Mamba 相比 S4 的关键创新是什么？</summary>

- Selective SSM（S6）：让 $B,C,\Delta$ **数据相关**（随输入变化）
- 由此获得**内容选择**能力（该记的多记、废话跳过），补上 LTI 的短板
- 代价是不再 LTI、FFT 卷积失效，改用硬件感知 selective scan

说"Mamba 让 $A$ 数据相关"（不准确，$A$ 结构化固定，是 $\Delta,B,C$ 数据相关）。

</details>

<details>

<summary>Q6. 线性注意力 / SSM 在推理上最大的优势是什么？</summary>

- 保持**固定大小**的递推状态（与上下文长度 $L$ 无关）
- 显存常数 + 每 token 解码 $O(1)$，对长上下文 / 高吞吐 serving 极有利
- 对比 softmax 的 KV cache 随 $L$ 线性膨胀

只说"更快"，讲不出"固定状态 → 常数显存 + $O(1)$ decode"这个具体机制。

</details>

<details>

<summary>Q7. 稀疏注意力的基本思想？和线性注意力的本质区别？</summary>

- 稀疏：**保留 softmax**，每个 query 只 attend 一个子集（窗口/块/top-k），$L^2\to Ls$
- 线性：**放弃 softmax**，用固定状态递推替代 attention
- 区别：稀疏仍存全部 KV（$O(L)$ 显存）、recall 强；线性是常数状态、recall 弱

把两者混为一谈，或以为稀疏也能把 KV cache 降到常数（错，稀疏仍存全 KV）。

</details>

<details>

<summary>Q8. 什么是混合架构（hybrid）？为什么需要它？</summary>

- 在大量线性/SSM 层里**交错少数 full softmax attention 层**
- 因为纯线性/SSM 的 **recall（联想回忆 / 精确复制）弱**
- 少量 full attention 补 recall，其余线性层保吞吐，兼得两端

只说"混着用"，讲不出"recall–throughput 权衡"这个核心动机。

</details>

<details>

<summary>Q9. FlashAttention 算线性注意力吗？</summary>

- **不算**。FlashAttention 是**精确 softmax** 的 IO 优化（block tiling + online softmax）
- 它把显存从 $O(L^2)$ 降到 $O(L)$，但 **FLOPs 仍是 $O(L^2)$**，没改计算范式
- 线性注意力 / SSM 才是"用固定状态递推替代 attention"

把 FlashAttention 当成"线性/次二次注意力"——它是精确 attention 的工程优化，复杂度类没变。

</details>

<details>

<summary>Q10. 为什么线性注意力可能比 Transformer recall 差？</summary>

- 历史被压进**固定大小**状态 $S\in\mathbb{R}^{d_k\times d_v}$，容量有限
- 多个 $\phi(k)v^\top$ 叠加互相干扰，长序列下精确检索会"糊"
- softmax 能近 one-hot 尖锐选择，有限维 $\phi(q)^\top\phi(k)$ 表达不出

把 recall 弱归因于"实现不好 / 没调参"，而非固定状态的容量本质。

</details>

### L2进阶题

<details>

<summary>Q11. 写出线性注意力的递推式，并解释去掉 softmax 归一化带来的两个问题。</summary>

- 递推：$S_t=S_{t-1}+\phi(k_t)v_t^\top,\ o_t=\phi(q_t)^\top S_t / (\phi(q_t)^\top z_t)$，$z_t=\sum_{j\le t}\phi(k_j)$
- 问题一：归一化分母 $\phi(q)^\top z_t$ 可能很小/接近 0 → 数值不稳，很多实现干脆去分母 + 额外 normalization
- 问题二：缺尖锐选择能力 → recall 弱（容量瓶颈）

只写无归一化版本，答不出"分母不稳 + 选择能力弱"两个具体后果。

</details>

<details>

<summary>Q12. Mamba 里到底哪些量是数据相关的？$A$ 是吗？为什么选择性主要靠 $\Delta$？</summary>

- 数据相关的是 $B, C, \Delta$；**$A$ 是结构化、与输入无关的固定参数**（每 channel 一个学到的 $A$）
- $\Delta_t$ 数据相关，通过 $\bar A_t=\exp(\Delta_t A)$ 把"有效衰减率"变成数据相关——这是选择性的主要来源
- $B_t, C_t$ 数据相关控制"写入/读出什么"

说"Mamba 让 $A$ 随输入变"（错）；或不知道 $\Delta$ 经 $\exp(\Delta A)$ 间接调制衰减。

</details>

<details>

<summary>Q13. 解释 State Space Duality (SSD)。它说 SSM 等于 softmax attention 吗？</summary>

- 把 selective SSM 递推展开成 $y=Mx$，$M$ 是下三角矩阵，$M_{ts}=C_t(\prod_{r}\bar A_r)\bar B_s$
- $M$ 是 **1-半可分矩阵**（下三角子块秩 $\le 1$），等价于一种**结构化掩码（线性）注意力**
- **不是** softmax attention——$M$ 没 softmax、是半可分结构化的；这是"线性/掩码注意力"层面的等价

过度宣称"SSM = softmax attention"（错，是结构化掩码线性注意力，无 softmax）。

</details>

<details>

<summary>Q14. Mamba-2 借 SSD 在工程上得到什么好处？和 Mamba-1 实现差别？</summary>

- Mamba-1：硬件感知 selective **scan**（自定义 kernel，**不吃 tensor core**）
- Mamba-2：用 SSD 把同一计算重写成 **matmul-heavy chunkwise**（块内物化小 $M$、块间传低秩状态），**吃满 tensor core**
- 好处：训练快数倍、状态维 $N$ 可开更大（matmul 便宜）→ 表达力更强

只说"Mamba-2 更快"，讲不出"SSD → chunkwise matmul → tensor core"这条因果链。

</details>

<details>

<summary>Q15. 推导 delta rule，并说明 $(I-\beta k k^\top)$ 项的作用。</summary>

- 把 $S$ 看成 $k\mapsto Sk$ 的映射，对误差 $(S_{t-1}k_t-v_t)$ 做一步在线最小二乘/梯度下降
- 得 $S_t=S_{t-1}-\beta_t(S_{t-1}k_t-v_t)k_t^\top=S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$
- $(I-\beta kk^\top)$ 先**擦掉 $k_t$ 方向的旧关联**再写新值 → "改写/覆盖"而非"叠加"

只背公式，讲不清它是误差修正/在线最小二乘，或答不出投影项 = "先删后写"。

</details>

<details>

<summary>Q16. DeltaNet 和普通线性注意力的本质区别？为什么 DeltaNet recall 更强？</summary>

- 普通线性：$S_t=S_{t-1}+v_tk_t^\top$，纯加性——同一 key 重复出现会**错误叠加** value
- DeltaNet：$S_t=S_{t-1}(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$，会**覆写**旧关联
- 改写式更新避免了关联干扰，所以 associative recall 显著更强

把 DeltaNet 当成"加了衰减的线性注意力"（错，关键是改写式的投影项，不只是衰减）。

</details>

<details>

<summary>Q17. Gated DeltaNet 在 DeltaNet 上加了什么？$\alpha$ 和 $\beta$ 各管什么？</summary>

- 加了一个数据相关的**衰减/遗忘门** $\alpha_t\in(0,1)$：$S_t=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$
- $\alpha_t$ 管"整体遗忘多少历史"（连续淡忘）
- $\beta_t$ 管"在 $k_t$ 方向改写多强"（精确覆写）；两者正交互补

把 $\alpha$ 和 $\beta$ 混为一谈，或不知道门控（遗忘）和 delta（改写）是两个独立机制。

</details>

<details>

<summary>Q18. 详细解释 chunkwise parallel：intra-chunk 和 inter-chunk 各做什么？</summary>

- 切成大小 $C$ 的 chunk，块内并行、块间递推
- **intra-chunk**：块内 $C\times C$ 的二次 causal attention（$((Q_cK_c^\top)\odot M)V_c$）
- **inter-chunk**：历史压成状态 $S_c$，本块 query 一次性读 $Q_cS_c$；块末更新 $S_{c+1}=S_c+K_c^\top V_c$
- 对固定 $C$ 总复杂度 $O(L)$ 且全 matmul（兼顾 $O(L)$ 推理 + 并行训练）

只说"切块算"，讲不清"块内二次 + 块间递推状态"两段，或不知道这是训练可并行的关键。

</details>

<details>

<summary>Q19. NSA 的三个分支分别是什么？为什么强调"原生可训练 + 硬件对齐"？</summary>

- 三分支：**compress**（远处块压成粗 token）+ **select**（top-k 最相关 key block 做细粒度 attention）+ **sliding window**（最近 token 保局部），门控融合
- 原生可训练：从预训练就用稀疏结构、端到端可学（不是事后套 mask）
- 硬件对齐：块连续、tensor-core 友好 → 真实墙钟加速（不只是理论 FLOPs 少）

只记得"top-k 稀疏"，背不全三分支；或不知道"硬件对齐"是为了真实加速。

</details>

<details>

<summary>Q20. MoBA 和 NSA 都是可训练稀疏，思路有何异同？</summary>

- 同：都**内容相关地选块**、都比 full attention 省、都硬件友好
- 异：**MoBA** = MoE 式 router，把每个 query 路由到 top-k key block（其余跳过）；**NSA** = compress+select+window 三分支固定融合
- MoBA 设计成能和 full attention 无缝切换，便于迁移已有模型

把两者说成一样，或不知道 MoBA 借的是 MoE 路由、NSA 是三分支结构。

</details>

### L3高级题

<details>

<summary>Q21. RetNet、Mamba、GLA、DeltaNet 都是 $O(L)$ 递推，它们在"状态如何遗忘/写入"上分别是什么？</summary>

- **RetNet**：$S_t=\gamma S_{t-1}+k_tv_t^\top$，**标量固定衰减** $\gamma$（非数据相关），纯加性写入
- **GLA**：$S_t=\text{diag}(\alpha_t)S_{t-1}+k_tv_t^\top$，**数据相关对角门**遗忘，纯加性写入
- **Mamba(S6)**：SSM 递推 $h_t=\bar A_t h_{t-1}+\bar B_t x_t$，$\bar A_t=\exp(\Delta_t A)$ 数据相关衰减
- **DeltaNet**：$S_t=S_{t-1}(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$，**改写式**写入（删旧+写新）；Gated 版再加 $\alpha_t$ 遗忘

只会背各自公式，归纳不出"衰减(标量/对角/数据相关) × 写入(加性/改写)"这个统一框架。

</details>

<details>

<summary>Q22. 为什么 selective SSM 不能像 S4 那样用 FFT 全局卷积训练？Mamba 怎么解决？</summary>

- S4 是 LTI，卷积核 $\bar C\bar A^k\bar B$ 固定，可写成全局卷积、用 FFT 在 $O(L\log L)$ 算
- selective SSM 的 $\bar A_t,\bar B_t,C_t$ **随时间变化** → 卷积核不再固定 → FFT 卷积失效
- Mamba 解法：硬件感知 **selective scan**（并行前缀扫描 + kernel fusion，状态留 SRAM、不物化 $L\times N$ 中间态）；Mamba-2 进一步用 SSD chunkwise matmul

答不出"含时 → 卷积核不固定 → FFT 失效"，或不知道 scan/SSD 是替代方案。

</details>

<details>

<summary>Q23. delta rule 的 $\prod_t(I-\beta_t k_tk_t^\top)$ 看似强顺序，怎么在序列长度上并行训练？</summary>

- 块内 $T$ 步的 Householder 乘积可用 **WY 表示**（数值线代 QR 的经典技巧）写成 $I-WY^\top$ 紧凑形式
- 于是块内用 matmul 批量算、块间传递状态（chunkwise）
- 这是 DeltaNet 系能在长序列高效训练的关键工程贡献

只说"难并行"或"用 chunkwise"，答不出"WY / Householder 表示"这个具体机制。

</details>

<details>

<summary>Q24. 混合架构里 full attention 该插多少、插哪？纯混合是否就完全常数显存？</summary>

- 经验：很小的 full-attention 比例（常见 1:5 到 1:7，每 6–8 层插 1 层）就能把 recall 补到接近纯 Transformer
- 插法有**层级交错**（某些层 attention）和 **head 级并行**（Hymba：同层内 attention head + SSM head）两种
- **不是完全常数显存**：那几层 full attention 仍有随 $L$ 增长的 KV cache，整体是"大部分层常数 + 少数层线性"

说"hybrid 完全没有 KV cache / 完全常数显存"（错，full attention 层仍有 $O(L)$ KV）。

</details>

<details>

<summary>Q25. 给定一个 200K 长上下文 + 高并发的 serving 场景，你会怎么在 softmax / 线性-SSM / 稀疏 / hybrid 之间权衡？</summary>

- 纯 softmax：recall 最好但 KV cache 随 $L$ 爆（200K × 高并发 → 显存/带宽不可行）
- 纯线性/SSM：常数状态、吞吐显存最优，但 recall（大海捞针/多跳）可能掉点
- 稀疏（NSA/DSA）：保 softmax recall、省计算/访存，但仍存全 KV（显存不省）
- **hybrid（推荐）**：大部分线性/SSM 层压 KV、少数 full attention（配 MLA/GQA + 可叠稀疏）保 recall——2025 工业级长上下文的主流答案；最终选型要看 recall 需求强度、显存预算、可接受的质量损失，并用真实长上下文 benchmark 验证

只报单一方案不做权衡；或忽略"稀疏不省 KV 显存""hybrid 仍有部分线性 KV"等关键约束。

</details>

## §A 附录：sanity check

本 tutorial 的代码（§6 的 `linear_attn_recurrent` / `linear_attn_chunkwise`，以及下方 delta-rule 与 block-sparse 片段）应满足以下关键不变量。完整可跑脚本见 [`code/linear_sparse_attention.py`](code/linear_sparse_attention.py)（纯 PyTorch、CPU 几秒、6 个 assert）——**真实运行输出附在本节末**。

1. **chunkwise == recurrent（核心等价）**：`linear_attn_chunkwise(Q,K,V,C)` 的输出对任意合法 chunk size $C$（如 1/4/7/全长 $L$），都应与 `linear_attn_recurrent(Q,K,V)` 在浮点误差内一致（`atol≈1e-5`）。这验证了"块内二次 + 块间递推"确实等价于逐 token 递推——chunkwise 的正确性基石。

2. **chunk size 无关性**：把 $C$ 从 1 扫到 $L$，输出不变（只是计算路径变了）。$C=1$ 退化为纯递推，$C=L$ 退化为单块二次 attention，中间任意 $C$ 都应一致。

3. **线性注意力状态形状**：递推状态 $S$ 始终是 $\mathbb{R}^{d_k\times d_v}$ 矩阵（本 demo $d_k=d_v=d$），与序列长度 $L$ 无关；decode 一步只更新/读取这个固定大小矩阵。

4. **delta rule 退化为加性线性注意力**：在 delta-rule 递推 $S_t=S_{t-1}(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$ 中，若**丢掉改写项**（即把 $(I-\beta k_tk_t^\top)$ 换成 $I$）并令 $\beta=1$，就退回普通线性注意力 $S_t=S_{t-1}+v_tk_t^\top$。这验证了"delta rule 的灵魂正是那个 $(I-\beta kk^\top)$ 投影项"——抽掉它，DeltaNet 就塌回 §2 的加性线性注意力。

5. **block-sparse mask 行为**：下方 top-k block 稀疏 mask 应满足——每个 query 行恰好保留 $k$ 个 key block（其余 $-\infty$）、softmax 后被屏蔽列权重为 0、且每行权重和为 1（被选中块内归一化）。

下面是 delta-rule 最小递推与 block-sparse top-k mask 两段示意代码（CPU 可跑、自带形状注释）：

```python
import torch
import torch.nn.functional as F

# ---- 代码片段 2：minimal delta-rule recurrence ----
def delta_rule_recurrent(K, V, beta, use_delta=True):
    """S_t = S_{t-1}(I - beta k k^T) + beta v k^T。
       K,V: [L, d]（设 k 已归一化）, beta: [L] 写入强度 -> 状态 S: [d, d]。
       use_delta=False 且 beta=1 时退回普通线性注意力 S += v k^T（见 §A.4）。"""
    L, d = K.shape
    S = torch.zeros(d, d, dtype=K.dtype)          # [d_k, d_v] 矩阵状态
    states = []
    I = torch.eye(d, dtype=K.dtype)
    for t in range(L):
        k = K[t]                                   # [d]
        if use_delta:
            S = S @ (I - beta[t] * torch.outer(k, k))  # 先擦掉 k 方向旧关联
        S = S + beta[t] * torch.outer(V[t], k)     # 写入新 v k^T  (注意 v k^T 顺序)
        states.append(S.clone())
    return torch.stack(states)                     # [L, d, d]

# ---- 代码片段 3：block-sparse top-k attention mask (NSA / MoBA 的 select 思想) ----
def block_topk_attention(Q, K, V, block=4, topk=2):
    """每个 query 只 attend 与它最相关的 top-k 个 key block（块内 softmax）。
       Q,K,V: [L, d] -> out: [L, d]。block: 每块大小, topk: 选几块。"""
    L, d = Q.shape
    nblk = (L + block - 1) // block
    scores = (Q @ K.t()) / (d ** 0.5)              # [Lq, Lk] 全 score（选块 + 最终 attn 都用它）
    # 块代表分数：每块只对"真实列"求均值（忽略 padding，故 L 不整除 block 也正确）-> [Lq, nblk]
    pad = nblk * block - L
    sc  = F.pad(scores, (0, pad), value=0.0).view(L, nblk, block)         # padding 填 0
    cnt = F.pad(torch.ones_like(scores), (0, pad)).view(L, nblk, block)   # 真实列计数（padding=0）
    blk_score = sc.sum(-1) / cnt.sum(-1).clamp(min=1)     # [Lq, nblk] 仅按真实列取均值
    # 选 top-k 块，得到块级 keep mask
    topk = min(topk, nblk)
    sel = blk_score.topk(topk, dim=-1).indices             # [Lq, topk]
    keep_blk = torch.zeros(L, nblk, dtype=torch.bool)
    keep_blk.scatter_(1, sel, True)                        # [Lq, nblk] True=保留该块
    # 展开回 token 级 mask
    keep = keep_blk.repeat_interleave(block, dim=1)[:, :L] # [Lq, Lk] True=keep
    masked = scores.masked_fill(~keep, float("-inf"))
    w = F.softmax(masked, dim=-1)                          # 被选块内归一化, 每行和=1
    return w @ V, keep                                     # [L, d], mask

if __name__ == "__main__":
    torch.manual_seed(0)
    L, d = 12, 8
    # delta-rule: 抽掉改写项 + beta=1 应退回加性线性注意力
    K = F.normalize(torch.randn(L, d), dim=-1)
    V = torch.randn(L, d)
    beta = torch.ones(L)
    S_delta = delta_rule_recurrent(K, V, beta, use_delta=True)[-1]
    S_lin   = delta_rule_recurrent(K, V, beta, use_delta=False)[-1]  # 加性版
    # 注：二者严格相等的充要条件是每步擦除项 S_{t-1}k_t=0；"各 k 两两正交(且 beta=1)"是最干净的充分条件。
    #     一般情形改写项 S_{t-1}(beta k k^T) 非零 -> 产生差异（这正是 DeltaNet 覆写旧关联的作用）
    print("delta vs additive final-state diff:", (S_delta - S_lin).norm().item())
    # block-sparse: 每行恰好 topk 个 block 被保留
    Q = torch.randn(L, d)
    out, keep = block_topk_attention(Q, K, V, block=4, topk=2)
    nblk = (L + 3) // 4
    per_row_blocks = keep.view(L, nblk, 4).any(dim=-1).sum(dim=-1)   # 每行保留块数
    print("blocks kept per query (expect=2):", per_row_blocks.tolist())
```

运行 `python code/linear_sparse_attention.py` 的**真实输出**（CPU，纯 PyTorch）：

```text
[a] chunkwise == recurrent, C in [1, 4, 7, 13]: max |Δ| = 3.81e-06  OK
[b] chunk-size invariance over C in [1, 4, 7, 13]: all equal = True  OK
[c] state shape over all 13 steps = {(6, 6)} (expect {(6, 6)}), L-independent  OK
[d] delta(use_delta=False, beta=1) == additive S+=vk^T: |Δ| = 0.00e+00  OK
[e] overwrite term: non-orthogonal ||Δ|| = 8.315e+00 (>0), orthonormal |Δ| = 3.58e-07 (~0)  OK
[f] L=12: blocks kept/query = [2] (expect [2]), rows sum to 1 = True  OK
[f] L=13: blocks kept/query = [2] (expect [2]), rows sum to 1 = True  OK

all linear / sparse attention sanity checks passed ✓
```

> ✅ **读数解释**
> - **[a]** chunkwise 与逐 token 递推在浮点误差内一致（max $|\Delta|\approx 3.8\mathrm{e}{-6}$）——chunkwise 正确性的黄金验证，且对 $C=1/4/7/L$ 全成立。
> - **[d]** 抽掉改写项 $(I-\beta kk^\top)$ 且 $\beta=1$ 时**严格**退回加性线性注意力（$|\Delta|=0$，§A.4）。
> - **[e]** 改写项确实在起作用：非正交 key 下 delta 与加性差 $8.3$（非零），正交 key + $\beta=1$ 下严格相等（$3.6\mathrm{e}{-7}\approx0$）——印证"每步 $S_{t-1}k_t=0$ 才相等，两两正交是充分条件"。
> - **[f]** block-sparse 在 $L$ 整除（12）与**不整除**（13）block 时，每个 query 都恰好保留 $\text{topk}=2$ 块、softmax 行和为 1（masked block-mean 让非整除长度也正确）。

## 📚 参考文献

- **Linear Attention / Transformers are RNNs** — Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, arXiv 2006.16236 (2020), ICML 2020.
- **Performer (FAVOR+)** — Choromanski et al., *Rethinking Attention with Performers*, arXiv 2009.14794 (2020), ICLR 2021.
- **RetNet** — Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, arXiv 2307.08621 (2023).
- **GLA (Gated Linear Attention)** — Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*, arXiv 2312.06635 (2023), ICML 2024.
- **Mamba** — Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv 2312.00752 (2023), COLM 2024.
- **Mamba-2 / SSD** — Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, arXiv 2405.21060 (2024), ICML 2024.
- **DeltaNet** — Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, arXiv 2406.06484 (2024), NeurIPS 2024.
- **Gated DeltaNet** — Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv 2412.06464 (2024), ICLR 2025.
- **NSA (Native Sparse Attention)** — Yuan et al. (DeepSeek), *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, arXiv 2502.11089 (2025).
- **MoBA (Mixture of Block Attention)** — Lu et al. (Moonshot AI), *MoBA: Mixture of Block Attention for Long-Context LLMs*, arXiv 2502.13189 (2025).
- **Lightning Attention / MiniMax-01** — MiniMax, *MiniMax-01: Scaling Foundation Models with Lightning Attention*, arXiv 2501.08313 (2025).
- **Jamba** — Lieber et al., *Jamba: A Hybrid Transformer-Mamba Language Model*, arXiv 2403.19887 (2024).
- **Hymba** — Dong et al., *Hymba: A Hybrid-head Architecture for Small Language Models*, arXiv 2411.13676 (2024).
- **Kimi-Linear** — Kimi Team, *Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv 2510.26692 (2025).
- **HiPPO** — Gu et al., *HiPPO: Recurrent Memory with Optimal Polynomial Projections*, arXiv 2008.07669 (2020), NeurIPS 2020.
- **S4** — Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces*, arXiv 2111.00396 (2021), ICLR 2022.
- **DeepSeek-V3.2 / DSA** — DeepSeek-AI, *DeepSeek-V3.2* 技术报告 / 模型卡 —— DeepSeek Sparse Attention（DSA，lightning indexer），2025（以官方技术报告 / model card 为准，无独立 arXiv 论文）。
- **Zamba** — Glorioso et al., *Zamba: A Compact 7B SSM Hybrid Model*, arXiv 2405.16712 (2024).
- **Qwen3-Next** — Qwen Team, *Qwen3-Next*（Gated DeltaNet + gated full attention 混合），2025（技术博客 / 模型报告，无独立 arXiv 论文）。
- **Longformer** — Beltagy et al., *Longformer: The Long-Document Transformer*, arXiv 2004.05150 (2020).
- **BigBird** — Zaheer et al., *Big Bird: Transformers for Longer Sequences*, arXiv 2007.14062 (2020), NeurIPS 2020.
