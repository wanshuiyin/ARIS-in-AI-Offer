## §0 总览

Decoder-only causal LM 用前文预测下一个 token。预训练流水线负责语料处理、监督信号构造、参数更新和训练状态恢复。

1. 训练计算量近似为 $C\approx6ND$，其中 $N$ 是主要矩阵乘法参数规模，$D$ 是呈现并参与监督的 token 数，$C$ 以 FLOPs 计。
2. MoE 的每 token 计算按 active 参数估算，模型状态与 checkpoint 的聚合大小按 total 参数计算。
3. Kaplan 与 Chinchilla 给出的预算分配指数不同，差异涉及拟合方法、训练 horizon 和学习率调度。
4. 小模型可用更多新的相关数据继续降低 loss，并以额外训练计算换取较低的部署成本。
5. 近重复文档先归成去重簇，再将整簇分配到 train/val/test，以减少跨 split 重复造成的验证偏差。
6. 文档独立的 packing 需要 block-diagonal causal mask，并单独处理跨文档的 target 标签。
7. 每次更新的损失按全部有效 target 的 NLL 总和除以有效 target 总数计算。
8. 一次更新的输入 token 数等于每个 DP replica 的序列数、序列长度、累积步数与 DP 规模的乘积。

```text
原始文档 → corpus factory → raw-document shards
raw-document shards → tokenizer → token 级采样 → chunk/truncate → tokenized shards
tokenized shards → sampler → packing / labels / masks → 参数更新
参数更新 → scheduler / token 计数 → checkpoint / resume
```

## §1 训练预算与 scaling law

### 1.1　$C\approx 6ND$：系数从哪来，忽略了什么

decoder-only dense Transformer 的训练计算量常用近似

$$C \approx 6ND$$

这一近似最早由 Kaplan et al. (2020) 系统整理。系数的来源是：一次 multiply-add 记为 2 FLOPs；每个 token 的前向要过一遍所有主要矩阵乘法，包括 QKV 投影、attention 输出投影、FFN 上下投影等，近似消耗 $2N$ FLOPs。$D$ 个 token 的前向合计 $\approx 2ND$。反向传播要同时算激活梯度和权重梯度，量级上约是前向的两倍，即 $\approx 4ND$。前向加反向 $\approx 6ND$，$6$ 同样是近似系数，更严谨地应写作 $k\approx 6$。

**参数计数口径**：$N$ 是 compute-bearing、非 embedding 的参数量。embedding 查表本身是一次索引而不是矩阵乘法，机械地把 embedding 参数计入 $N$ 会系统性高估这部分的计算量。输出词表投影是否单独计数、如何计数，不同论文和实现的约定并不完全一致，读一份具体的 FLOPs 估算时应先确认它的参数计数约定。

$6ND$ 忽略和简化的项分三类。**序列长度项**：attention score 的计算量在每条序列内部随 $L_{\text{seq}}^2$ 增长，$N$ 本身不随序列长度变化；固定训练总 token 数 $D$ 时序列条数近似为 $D/L_{\text{seq}}$，总 attention 开销近似随 $D\times L_{\text{seq}}$ 增长，而不是 $D\times L_{\text{seq}}^2$。序列越长，这一项相对 $6ND$ 里其余线性项的占比越不能忽略。**模型与训练细节**：词表投影在 $V$ 大时输出层矩阵乘法量级可观；norm/softmax 等非矩阵乘法算子没有计入；激活重计算为省显存额外重算一遍前向，实际反向的乘数会比 4 更高；MoE 有 active 与 total 两种参数口径（§1.2）；不同社区对"1 FLOP"是把 multiply-add 各算一次还是合记一次，计数约定有差异。**硬件效率**：$6ND$ 不等于 GPU 实际执行的 FLOPs，也不能直接预测墙钟时间。

粗略估算 GPU-days 需要额外引入 GPU 数量、峰值 FLOPs 和 MFU：

$$t\approx C/(n_{GPU}\times\text{峰值FLOPs}\times MFU)$$

通信开销、重计算、data stall 都会让这只是一个估算而非精确预测。

### 1.2　MoE：每 token 计算用 active，显存用 total

MoE 的每 token 计算量用 **active parameters** 估算，也就是真正参与该 token 前向和反向计算的专家参数。模型容量、显存占用和 checkpoint 文件大小按 **total parameters** 算，total 对应的是聚合的 model-state 大小。单卡实际显存还取决于分片方式（见 Q25）。

一个 671B/37B 风格的模型（DeepSeek-V3 一类的公开配置）：$6ND$ 里的 $N$ 用 37B 量级估算预算，存 checkpoint 按 671B 算。routing、负载均衡与 aux loss 的机制推导见 `moe_tutorial.md`。

### 1.3　Kaplan (2020) vs Chinchilla (2022)：指数不能混为一谈

两篇奠基性论文给出的最优配置指数并不相同。

**Kaplan et al. (2020)**：

$$N_{opt}\propto C^{0.73},\qquad D_{opt}\propto C^{0.27}$$

计算预算增加时，Kaplan 的结论是更多地分给参数而不是数据。

**Hoffmann et al. (2022, "Chinchilla")**：论文用多种估计方法交叉验证，没有给出唯一一组"精确指数"。不同估计方法分别给出约 **0.50/0.50**、**0.49/0.51**、**0.46/0.54** 这样接近但不完全相同的一族结果。其中最常被引用的参数化损失模型写作

$$L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

拟合得到约 $\alpha\approx 0.34,\ \beta\approx 0.28$。$E$ 是拟合出的渐近不可约损失（asymptotic loss floor），大致包含数据本身的熵与该模型族的近似误差，拟合本身无法把两者唯一分解开。$A/N^\alpha$ 是参数不够大付出的代价，$B/D^\beta$ 是数据不够多付出的代价。

在计算预算约束 $C=kND$（$k=6$）下对 $L(N,D)$ 做约束最优化，用 Lagrange 乘子消元可得解析解：

$$N_* = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}\left(\frac{C}{k}\right)^{\frac{\beta}{\alpha+\beta}},\qquad D_* = \left(\frac{\beta B}{\alpha A}\right)^{\frac{1}{\alpha+\beta}}\left(\frac{C}{k}\right)^{\frac{\alpha}{\alpha+\beta}}$$

代入四舍五入的 $\alpha=0.34,\beta=0.28$：$\dfrac{\beta}{\alpha+\beta}=\dfrac{0.28}{0.62}\approx 0.452$，$\dfrac{\alpha}{\alpha+\beta}=\dfrac{0.34}{0.62}\approx 0.548$，即

$$N_*\propto C^{0.452},\qquad D_*\propto C^{0.548}$$

用论文完整精度的 $\alpha,\beta$ 计算，通常报约 0.46/0.54。§6 的 [D06] 用这组指数做了一个可执行的拟合单元测试。

"约 0.50/0.50"是"参数和 token 大致等比例增加"这个粗粒度经验结论，"约 0.46/0.54"是参数化损失模型拟合出的具体指数。"每参数约训练 20 token"则是经验性近似，不是单由 $\alpha,\beta$ 推出的普适常数。$D_*/N_*$ 还依赖常数 $A,B$ 以及 $k$；当 $\alpha\ne\beta$ 时 $D_*/N_*\propto (C/k)^{(\alpha-\beta)/(\alpha+\beta)}$ 会随 $C$ 缓慢漂移。固定 $A,B,\alpha,\beta,k$ 并沿用这个拟合模型时，100 倍算力下比值按此指数增长约 1.56 倍，这条推导在数学上成立。它不能外推成"真实最优比例每 100 倍算力就稳定增长 1.56 倍"这条经验断言：Chinchilla 另外两种独立估计方法（IsoFLOP 曲线拟合、参数/数据比例拟合）给出近似恒定的约 0.5/0.5；$\alpha,\beta$ 相差不大时，拟合误差会显著影响 $(\alpha-\beta)/(\alpha+\beta)$ 这个指数；$A,B,k$ 与数据分布、架构、训练 recipe 跨多个数量级算力时未必不变，而推导的前提正是它们全部固定。

Kaplan 与 Chinchilla 分歧的技术原因涉及数据集、实验覆盖范围、损失拟合方法和训练 horizon 设计。关键差异在训练 schedule：Kaplan 大量使用单条训练曲线中间点的 loss，其 LR schedule 终点并非为该预算专门配置；Chinchilla 为每个预定的 token horizon 单独配置匹配的训练 schedule，例如把 cosine decay 长度对齐目标 token 数，再做比较。

### 1.4　小模型超训与部署成本

过拟合指训练 loss 下降而验证 loss 回升。固定模型规模 $N$ 继续喂新的、非重复、同分布或相关分布的数据，通常仍然能降低 loss，不满足这个定义。真正发生的是：固定 $N$ 继续增大 $D$ 时，最终计算量 $C=kND$ 也在同步增大，走到了一个更大的 $C$；而 $(N,D)$ 相对该 $C$ 对应的 compute-optimal 前沿点 $(N_*,D_*)$ 具有更小的 $N$、更大的 $D$。这个配置被称为 **over-trained**，是相对 compute-optimal 前沿的经济学判断。

工业界大量小模型被故意训练到远超 Chinchilla-optimal 比例，例如同等能力下选更小的模型、喂更多 token。理由是部署侧的推理成本：显存、延迟、单位 token 服务成本。用一次性的额外训练计算，换取模型全生命周期里更便宜的推理——训练计算最优与全生命周期成本最优是两个不同的目标。

### 1.5　给定预算怎么选 $N,D$

1. 先明确优化目标。训练计算最优直接用 $N_*,D_*$ 的解析解；推理感知最优先按部署预算卡死一个更小的 $N$，固定 $C$ 时 $D$ 相应增大，也就是刻意超训一个小模型。
2. 长上下文训练会引入 $6ND$ 忽略的 $L_{\text{seq}}^2$ attention 项，预算里要单独留出这部分。
3. MoE 模型算计算预算用 active 参数，报存储和显存预算用 total 参数。
4. 明确 $D$ 的口径：是训练时呈现的 token 数，还是去重后的有效数据量。语料重复度高会让有效数据量低于名义 $D$，这也是 Chinchilla 之后一系列 data-constrained scaling 工作关注的问题。

## §2 corpus factory：从网页快照到训练分片

语料流水线决定训练能否跑稳跑对，其影响常大于模型结构和超参。

### 2.1　十一步流水线总览 + lineage

| 步骤 | 做什么 | 关键坑 |
| --- | --- | --- |
| 1. 原始快照 + 许可证记录 | 保存抓取时间、来源、许可证元数据 | 版权/许可证追溯 |
| 2. 抽取/编码规范化 | 网页正文抽取、去导航栏/广告、Unicode 规范化 | 往往比后续分类器更影响质量（§2.2） |
| 3. 语言识别 | 按语言分流 | 代码混杂文本、短文本置信度低 |
| 4. 基础质量过滤 | 长度/符号比例/重复 n-gram/困惑度过滤 | 阈值一刀切误删长尾但合法的文本 |
| 5. 精确去重 | 完全重复文档/子串去重 | 只抓字节级重复，抓不住近重复 |
| 6. 近似去重 | MinHash/LSH 找近重复 | 相似度≠语义相似（§2.3） |
| 7. 隐私/安全处理 | PII 脱敏、违法内容过滤 | 需要独立于质量过滤的专门流程 |
| 8. benchmark 去污染 | 与评测集做 n-gram 重叠检测 | 和 train-val 泄漏是两回事（§2.6） |
| 9. **cluster-level 数据切分** | **按第 6 步的去重簇整体分配 split** | **必须在这一步之前完成去重**（§2.5） |
| 10. mixture 采样（文档级配额） | 按域权重决定每个域贡献多少篇文档 | 文档级权重≠token 级权重（§2.7） |
| 11. 确定性分片 | 写出可复现的 raw-document shard 文件 | 需要与 lineage 一起版本化（§2.8） |

**两个阶段**：表里这 11 步都在 raw-document 阶段。第 10 步的 mixture 采样是**文档级**配额，决定每个域贡献多少篇文档；第 11 步产出 raw-document shards。tokenizer 在这之后运行，把 raw-document shards 变成 token stream。tokenized 阶段做 token 级 mixture 采样、长文档 chunk/truncate，写出 tokenized training shards，再交给 §2.10 的 data-stream/sampler 层。split 之后 train 与 val/test 分叉处理。mixture 采样在文档级和 token 级各发生一次，两次的含义并不相同。

**数据 lineage**：每个训练样本理想情况下都应能追溯到源数据集或 URL、快照/爬取版本、抽取与过滤流水线版本、命中的质量过滤器与阈值、所属的去重簇 ID、split 归属，以及最终落在哪个 shard 文件的哪个偏移量。这套记录是几个月后排查"某条输出是不是背了训练数据""某个 benchmark 是不是被污染了"的唯一依据。

### 2.2　抽取与质量过滤

网页正文抽取（剥离导航栏、广告、页脚模板、评论区噪声）和编码/Unicode 规范化处在流水线最上游。上游抽取错误会把大量结构性噪声——导航文本、重复模板、乱码——系统性地压进每一份文档，其影响常常大于后续质量分类器的精细调参。这是经验观察，C4 语料审计（Dodge et al.）与 RefinedWeb/FineWeb（Penedo et al.）的结果间接支持，不是"抽取一定优先于分类器"的普适排序。训练语料的 normalization 规则必须与 tokenizer 训练时用的规则保持一致，否则会出现线上线下切分不一致的漂移。normalization 选项与 byte fallback 见 `tokenization_tutorial.md` §3–§4。

语言识别通常用轻量分类器按文档分流，例如 fastText 风格的 n-gram 模型；代码切换文本和短文本的语言判断置信度低。基础质量过滤叠加若干启发式规则——行长度分布、符号/字母比例、重复 n-gram 占比——和/或一个小型辅助语言模型的困惑度评分，本质是用便宜的信号筛掉最差的一批，而不是精细的语义判断。

### 2.3　精确去重 vs 近似去重

**精确去重**是字节级、哈希级的完全匹配：整篇文档去重，或用后缀数组做跨文档重复子串去重。它只能抓住完全相同的内容，抓不住"标题改了一个词"这类近重复。

**近似去重**（MinHash + LSH）分四步：把文档表示为 shingle（n-gram）集合；用 MinHash（Broder, 1997）估计集合间的 Jaccard 相似度；用 LSH 分桶做候选召回；候选对再用精确 Jaccard 复核。

MinHash 估计的是 shingle 集合的 Jaccard 相似度。两篇讨论同一个概念但用词完全不同的文档，Jaccard 相似度可能很低；两篇大段模板化文本（导航栏、许可证声明、样板代码）但主题无关的文档，Jaccard 相似度反而可能很高。MinHash/LSH 抓的是**表层重复**：镜像站点、近乎逐字复制、模板文档。

LSH banding 只做候选召回。在各 MinHash 行、哈希函数近似独立的理想化假设下，候选概率为 $P=1-(1-s^r)^b$，其中 $b$ 个 band、每个 band $r$ 行。分桶只保证候选概率随相似度单调上升，不保证候选就一定是真近重复。

基于 embedding 阈值的 semantic dedup 可能误删"主题相同但表达方式不同"的合法文本，这种误删对小语种、方言、模板化技术文档的影响不均匀，因为这些文本本身的 embedding 分布更集中。它和近重复去重解决的是不同层级的重复问题。

### 2.4　隐私安全处理 + benchmark 去污染

隐私和安全处理是独立于"质量高低"的一条流程：PII 脱敏（邮箱、电话、密钥）与违法有害内容过滤，用的是专门的检测规则和阈值，不和质量过滤器混在一起调参。

**benchmark 去污染**检测训练语料与目标评测集之间的 n-gram 重叠，移除或标记重叠文档，防止训练时看过评测答案。**13-gram 量级的精确匹配**是 GPT-3（Brown et al., 2020）论文里明确采用的具体做法，被广泛借鉴；不同团队实际采用的 n-gram 长度并不统一，不应把它当成行业统一标准。

### 2.5　先去重再切分：cluster-level split

先跑近似去重，把所有文档划分成连通分量（去重簇），再把**整簇**原子性地分配到 train/val/test，同一个簇里的文档绝不能被拆到不同 split。这里依赖的是**传递闭包**：即使文档 A 只和 B 直接比较过、B 只和 C 直接比较过，A 和 C 从未被直接比较，只要 A~B、B~C，A 和 C 就必须落在同一个 split。§6 的 [D01] 用并查集把这条规则做成了一个可执行断言。

如果先切分再去重，近重复内容（镜像站点、被转载改写的文章、模板化文档的不同实例）完全可能一份落在 train、另一份落在 val。模型训练时"见过"验证集样本的近乎逐字复制品，validation loss 因此**虚低**，不是因为模型泛化得好，而是因为它实质上背过了验证样本的一个改写版本。这会误导 early-stopping、模型选择，以及任何拿这个 validation loss 做的下游质量汇报。去重对下游模型质量影响的系统性讨论见 Lee et al., 2021。

### 2.6　train-val 泄漏、benchmark 污染与跨文档上下文

| 类型 | 定义 | 根因 | 应对 |
| --- | --- | --- | --- |
| train-val 泄漏 | 近重复内容跨 split | 先切分再去重 | cluster-level split（§2.5） |
| benchmark 污染 | 训练语料含评测集内容 | 未做去污染检测 | n-gram 重叠检测（§2.4） |
| packed 序列跨文档历史上下文 | 打包后文档 B 能读到文档 A 的 token | causal mask 允许跨文档读取历史 | **通常不是数据泄漏，是序列语义选择**（§3.4） |

第三种是**跨样本上下文耦合**：后一篇文档读到的是先前的 token 而不是未来标签，属于训练动态与统计独立性层面的建模选择，不是评测意义上的信息泄漏。是否用 block-diagonal mask 切断它，取决于是否要求"每篇文档必须是统计独立样本"这条建模假设。

### 2.7　mixture 采样：域权重未必等于 token 权重

混合采样至少有三种不同的"采样单位"，同一份域权重配置在这三种单位下会产出**不同**的最终 token 分布：

- **按文档采样**：先按域权重选中一个文档池，再从池内均匀或加权抽文档，取整篇；
- **按 token 采样**：直接从拼接后的 token 流里切固定长度的块，与文档边界解耦；
- **先采域再采文档**：先按域权重选域，再在域内选文档。

三者的区别来自不同域的**平均文档长度**不同：平均文档更长的域，在同样的文档数权重下实际贡献的 token 数更多。域权重与 token 权重的关系分两种情况。按文档数或文档权重配置、又被直接当成 token 权重使用时，两者通常不相等，这是偏差的来源。配置项本身就按 token-level 目标占比定义时，期望意义下二者可以相等。

无论采用哪种采样单位，都要在采样后单独统计"实际 token 占比"并汇报，而不是只报配置值。domain reweighting 的系统性方法可参见 DoReMi（Xie et al., 2023）。

这也牵连到 §1 里 $D$ 的口径：scaling law 里的 $D$ 通常指训练时**呈现**给模型的 token 数，也就是总 token 流过 optimizer 的次数，不保证等于语料的唯一 token 总量。如果某个域被反复过采样（多个 epoch），或语料里仍残留大量近重复内容，名义 $D$ 会比"有效"数据量虚高，多出来的部分不会带来与新数据同等的统计信息。重复数据对有效 $D$ 的影响，系统性讨论见 Muennighoff et al. (2023)。

### 2.8　确定性分片 + tokenizer 接口

**确定性分片**：给定同一份语料、同一套 mixture 配置、同一个随机种子，"哪篇文档进哪个 shard 文件、在什么位置"必须是可复现的。这一性质直接支撑确定性 resume，重启后要能精确定位到中断前消费到的位置；也支撑 spike 排查，复现导致某次 spike 的那个具体 batch 依赖于能精确复原当时消费的数据。

corpus factory 面向 tokenizer 的输出是一份很小、被冻结的固定格式：版本锁定的 tokenizer 制品（连同哈希）、固定的 BOS/EOS/PAD ID、固定的 normalization 规则、byte fallback 行为，以及供预算估算用的 bytes/token 比率。tokenizer 本身怎么训练（BPE/WordPiece/Unigram）以及 SentencePiece 的框架细节，见 `tokenization_tutorial.md` §2–§5。

### 2.9　长文档 chunking / truncation

单篇文档的原始长度经常超过训练用的 context length，甚至超过很多倍——一本书、一个长代码仓库都可能远超几千到几万 token 的窗口。这类文档在进入 packing 之前先做 **chunk（切块）**，有四个决策点：

- **是否 overlap**：无重叠切块最简单，代价是每个 chunk 边界处的上下文被硬切断；有重叠切块在相邻 chunk 之间保留一段重叠区（例如几十到几百 token），缓解边界处上下文缺失，代价是同一段原文被多个 chunk 共享。
- **重叠区是否重复计 loss**：相邻两个 chunk 共享的重叠 token，如果两个 chunk 都算 loss，这部分内容会被有效地"过采样"，扭曲它在总 $D$ 与 $B_{target}$ 里的真实权重。通常应该只在能看到更长前文的那个 chunk 里保留这段的 loss，另一个 chunk 里把它标记为纯上下文。
- **人工切块边界是否插 EOS**：希望每个 chunk 被当成独立样本对待时，切块处应插入 EOS/BOS 边界并分配新的 doc-id，让后续的 block-diagonal mask 和边界 loss mask 规则能识别"这其实是同一篇文档人为切出来的两段"；不插边界标记时，多个 chunk 会被当成同一篇文档的连续延续处理。
- **position id / doc-id 如何延续**：每个 chunk 视为独立样本时，每个 chunk 应有自己的 doc-id，并按 §3.4 决定是否重置 position id；chunk 逻辑上仍是同一篇文档的延续时，doc-id 应在多个 chunk 间保持一致，position id 也应连续编号而不是每个 chunk 从 0 重新开始。这个选择要与训练、推理时"文档"的定义方式保持一致，否则会出现训练时当独立样本处理、推理时又假设位置连续这类不一致。

这几个决策没有放之四海皆准的默认值，但要显式记录在 lineage 里。一段训练数据如果被 chunk 过，它的 lineage 应该能追溯到来自哪篇原始文档的第几个 chunk、是否与相邻 chunk 重叠、重叠区是否计 loss。缺了这些记录，事后无法判断某段内容的 loss 权重异常是不是 chunking 策略造成的。

### 2.10　data-stream / sampler 层

tokenized training shards 写到磁盘之后，训练时真正读取它们的是 **data-stream / sampler 层**，梯度累积与确定性 resume 都依赖这一层的状态定义：

- **shard shuffle**：每个 epoch 或每一轮遍历开始前，是否打乱 shard 文件的读取顺序，以及打乱用的随机种子和状态；
- **域采样 RNG**：mixture 在训练时动态采样而不是提前离线混合好时，需要一个独立于其他随机性来源（dropout、数据打乱）的域选择随机数生成器，它的状态也要能被 checkpoint 捕捉；
- **有放回 / 无放回**：域内文档或 shard 是否允许被重复采样。多 epoch 训练、或数据量不足以支撑 compute-optimal $D$ 时通常需要有放回重复采样，此时"重复"本身要被记录，因为它直接影响"名义 $D$ vs 有效数据量"这条口径；
- **DP rank 互斥分片**：每个 DP rank 读取的 shard/样本集合必须两两不相交，否则同一条数据会被多个 rank 同时计入，等价于隐式地重复计数了 $D$；
- **worker/prefetch 状态**：多进程/多线程 dataloader 通常会提前预取若干 batch 到队列里，这个队列本身的状态（已经预取了哪些、还没被消费的有哪些）也是训练状态的一部分。resume 时如果只恢复了"读到了第几个样本"这个游标，却没有正确处理预取队列里已取出但还没被消费的数据，这部分数据会被跳过或重复消费；
- **cursor 位置**：以上所有随机性和顺序决策最终都要收敛到一个可以被精确保存和恢复的游标，通常是"哪个 shard 文件的哪个字节或样本偏移量"。游标本身的精度决定了"样本序列一致"这一层能做到多精确。

### 2.11　cluster split 之后：train 与 val/test 的分叉

cluster-level split 只解决"哪些去重簇归属哪个 split"这一件事；split 完成之后，train 和 val/test 走的是两条完全不同的后续流程。

**train** 继续走 §2.7–§2.10 的完整流程：mixture 采样（域权重可能被动态调整）、shuffle、必要时的有放回重复采样、data-stream/sampler 层的全部状态。

**val/test 必须冻结版本**：一旦确定下来，同一个 val/test 集合在整个训练生命周期里、乃至跨多次训练 run 做公平比较时，样本内容、数量、评分协议都保持不变。它不被动态调整的 mixture 影响，也不被重复采样"稀释"或"加权"。分域报告要求的是每个域各自独立、可比较的一份验证集，而不是跟着 train mixture 走的一份子集。

val/test 若要覆盖多个域，域内部的采样应该在 split 阶段一次性确定并固定下来，而不是像 train 那样每个 epoch 动态重新采样。否则不同 checkpoint 之间的 validation loss 就不再是在同一把尺子上比较的。

## §3 objective-to-tensor：token 流如何变成监督信号

### 3.1　teacher forcing + 位移标签

标准 next-token 训练用 teacher forcing：给定 token 序列 $x_1,\dots,x_T$，位置 $i$ 的**标签**是 $x_{i+1}$，整体右移一位。模型在位置 $i$ 输出 $p(x_{i+1}\mid x_{\le i})$ 的预测分布。loss 是所有**有效**目标位置的 NLL，精确的归一化公式见 §3.5。序列末尾没有下一个 token，对应位置的标签记为 `IGNORE`，与 PyTorch 惯例一致，用哨兵值 $-100$。

### 3.2　causal mask + EOS/BOS 边界规则

普通 causal mask $M[q,k]=\mathbb{1}[k\le q]$ 只保证"看不到未来"。围绕文档边界还有两条监督规则：

- **保留**"文档最后一个内容 token → EOS"这条监督，它是模型学会"什么时候该停"的直接信号；
- **按配置屏蔽**"上一篇文档的 EOS → 下一篇文档第一个 token（或 BOS）"这条监督，这个位置对应的标签只是拼接的产物，不是任何有意义的生成延续。

这两条规则可以由**同一个**基于文档边界的规则自动同时产出。把 EOS 视为它所属文档的最后一个 token，在 doc-id 数组里与该文档其余内容同 id，再用"标签有效当且仅当当前位置与下一位置属于同一文档"这条规则构造标签。内容末 token 到 EOS 天然同属一个文档，规则判定有效；EOS 到下一文档首 token 天然跨文档，规则判定无效。EOS 不需要单独的特判逻辑，§6 的 [D04] 演示了这一点。

chat template 拼装时若同时使用模板自带的 BOS 和 tokenizer 的 `add_special_tokens`，容易重复插入 BOS 并污染监督信号，细节见 `tokenization_tutorial.md` §5.5。

### 3.3　padding：loss mask 必须有，key mask 是否必需取决于布局

padding 需要处理两件不同的事：**key padding mask** 防止 query 位置去 attend 到 padding key；**loss mask** 把 target 是 padding 的位置排除出 loss，否则会对着无意义的填充目标算梯度。

- **loss mask 始终必需**——不做它，loss 一定会被填充目标污染；
- **严格 causal + 右侧 padding 时 key mask 可省**——真实 query 的位置天然小于 padding 起始位置，causal mask（$k\le q$）本身已经挡住这些未来的 padding key，省略它不会改变真实 token 的输出；
- **左 padding、序列内部有空洞等其他布局时 key mask 必需**——causal 条件本身不足以排除 padding key，真实 query 完全可能读到无意义内容。

通用实现同时构造两者；只有在能确认严格处于"右 padding + causal"特例、且愿意长期维护这条隐含假设时，才论证性地省略 key mask。

很多 tokenizer 没有独立的 PAD token，直接把 `pad_id` 设成 `eos_id` 本身，这不一定错。事故发生在下游代码按 `input_ids == pad_id` 无差别地构造 mask：这会把语料里**所有真实的 EOS 位置**也一起当成 padding 屏蔽掉。训练信号里"文档该在哪结束"的监督被系统性移除，模型的停止能力显著削弱，从头训练时尤其致命。mask 要按**真实的 padding 位置**构造——每条序列的有效长度 `valid_len`，或随批次携带的显式 padding 标记——不用"和某个 token ID 相等"来判定是否是 padding。§6 的 [D05] 把这套 mask 构造成了可执行断言。

### 3.4　document packing：attention、标签与位置

多篇文档拼接进同一条训练序列（packing）是提高有效 batch 利用率的常见做法。普通 causal mask 只阻止"读未来"，不会阻止后一篇文档读到前一篇文档的历史 token。很多系统就是故意把打包后的 token 流当成一条 **EOS 分隔的连续流**，用普通 causal mask。

若语义上要求"每篇文档是独立样本"，需要 **block-diagonal causal mask**：

$$M[q,k] = (k\le q)\ \wedge\ (\text{docid}[q]=\text{docid}[k])$$

是否重置每篇文档内部的 position ID，取决于位置编码方案与训练语义，不是通用规则：

- **绝对位置 embedding**：不重置会让后续文档从一个不属于它的巨大位置号开始，通常应该重置；
- **标准 RoPE**：只依赖 query/key 之间的相对位置差。block-diagonal mask 把 attention 限制在文档内部后，给同一文档整体加相同偏移量在理想数学下不改变文档内的相对相位，不会仅因"没从 0 编号"而污染注意力计算；
- **变体与工程考虑**：RoPE 长度外推/scaling 变体、有限精度下的数值行为、希望每篇文档严格从位置 0 开始训练以对齐推理时的位置分布——这些仍让大多数实现选择重置。

阻断 attention 不会自动屏蔽跨文档的预测标签。加上 block-diagonal mask 之后，如果不单独处理标签，文档 A 的最后一个位置默认仍然会被要求预测文档 B 的第一个 token。这是一个不折不扣的跨文档 target，必须用 §3.2 的边界 loss mask 单独排除。

**数值例子**（tokens = `[11,12,13,21,22]`，doc IDs = `[0,0,0,1,1]`）：独立文档语义下位移标签为 `[12,13,IGNORE,22,IGNORE]`；block mask 断言 `M[3,2]=False`（3 号位属于文档 1，2 号位属于文档 0，即使 $k\le q$ 也因跨文档被挡）、`M[4,0]=False`、`M[4,3]=True`；连续流语义下同一位置 `M[3,2]=True`，这是合法的配置选择。

**功能测试**：构造 $QK^\top=0$，softmax 退化为在允许的 key 上均匀分配注意力；文档 0 的三个位置 values 设为 `[100,100,100]`，文档 1 的两个位置 values 设为 `[0,0]`。普通 causal mask 下位置 3（文档 1 首 token）的输出是允许 key `[0,1,2,3]` 的均值 $(100+100+100+0)/4=75$；block mask 下只剩 key `3` 本身可用，输出 $=0$。再把文档 0 的 values 从 100 改成 1000：block 模式下文档 1 的输出完全不变，仍是 0；连续流模式下同一输出从 75 变成 750，因为它本就没有断开与文档 0 的耦合。只检查 mask 的 shape 或每行是否归一化为 1，抓不出这类边界逻辑写错但形状仍然正常的 bug。§6 的 [D03] 是这个测试的可执行版本。

### 3.5　按有效 target 归一化 loss

$$\mathcal{L} = \frac{\sum_i \mathrm{NLL}_i}{\#\{\text{有效 target token}\}}$$

对**全局批次**里所有有效 target token 的 NLL 求和，除以有效 token 总数。实现上要在整个梯度累积窗口内、以及跨 DP rank 通信时，分别累计"NLL 总和"与"有效 token 总数"两个量，只在最后统一相除一次。

另一种写法是先对每个 micro-batch 分别求该批内的均值损失，再对这些均值取平均。各 micro-batch 的有效 token 数不同时——padding 比例不同、packing 效率不同、或正在做 short-to-long 的上下文长度切换——它会隐式地给 token 数少的 micro-batch 加权过重、token 数多的加权过轻，偏离真正按 token 计权的损失。

MoE 模型的总 loss 里通常还要按配置加上 router 的辅助损失项（router aux loss），量级一般远小于主 loss，机制见 `moe_tutorial.md`。

### 3.6　PPL/BPB 精确公式

$$\mathrm{PPL} = \exp\left(\frac{\sum_i \mathrm{NLL}_i}{\#\text{valid target tokens}}\right),\qquad \mathrm{BPB} = \frac{\sum_i \mathrm{NLL}_i}{\#\text{bytes}\cdot\ln 2}$$

PPL 只有在 tokenizer、normalization、评测文本三者都完全一致时才适合直接比较。分母的 token 数本身依赖 tokenizer，即使总 NLL 相同，切得更碎的 tokenizer 也会把 loss/token 拉低。BPB 把分母换成同一份文本的 UTF-8 字节数，对 tokenizer 差异更稳健，但仍受编码与预处理是否一致的影响。完整推导与"loss/token 差一倍、BPB 却相同"的数值例子见 `tokenization_tutorial.md` §1.3 与 §7 [D07]。

## §4 update accounting：global batch 到底是多少

一次 optimizer step 里各环节的概念执行顺序：model/config 接口 → forward → loss（NLL 总和除以有效 target token 数）→ backward → 梯度同步（DP all-reduce，注意 sum/mean 归约语义）。随后是 unscale（AMP）、累积窗口结束后对最终梯度做一次梯度裁剪、optimizer step、scheduler 与已消费 token 计数器推进，最后是原子 checkpoint 写入。

### 4.1　global batch 与三种 token 计数

$B_{micro,seq}$ 是每个 DP replica 每个 micro-batch 的序列数，$L_{seq}$ 是序列长度，$G_{acc}$ 是梯度累积步数，$W_{DP}$ 是数据并行的 world size。当 TP/PP > 1 时，一个 DP replica 本身会跨多张卡，"每张卡"和"每个 DP replica"不是一回事。

**送入模型的 token 数**（processed token slots，不管是否有效参与 loss）：

$$B_{input} = B_{micro,seq}\times L_{seq}\times G_{acc}\times W_{DP}$$

**只有 $W_{DP}$ 直接扩大独立样本数**。张量并行（TP）和流水并行（PP）切分的是**同一份**样本的计算，把它们的 world size 也乘进 $B_{input}$ 是最常见的 global batch 算错来源。通信原语与切分机制见 `distributed_training_tutorial.md`；DP 各 rank 消费的数据必须互不重叠，否则 $D$ 被重复计数。

**实际参与 loss 的有效 target 数**：

$$B_{target} = \sum_i m_i$$

其中 $m_i$ 是第 $i$ 条序列的有效 target mask 计数。$B_{input}$ 与 $B_{target}$ 一般不相等，即使**完全没有 padding**。位移标签下每条序列最后一个位置没有下一个 token，被记为 `IGNORE`，每序列有效 target 数是 $L_{seq}-1$ 而不是 $L_{seq}$。再叠加 padding 或 packing 的文档边界 loss mask，$B_{target}$ 会进一步小于 $B_{input}$。

三个记账口径分别对应不同的量：

| 口径 | 对应量 | 用在哪 |
| --- | --- | --- |
| processed token slots | $B_{input}$（"过了多少个 token 位置"） | 推进 §5.1 的已消费 token 计数器、按 token 对齐 scheduler |
| non-padding input tokens | $B_{input}$ 减去 padding 位置 | 估算 GPU 实际有效吞吐、评估 packing/padding 效率 |
| valid target tokens | $B_{target}$ | §3.5 的损失归一化分母；§1 scaling law 里的 $D$ 在有 padding/边界屏蔽时应更贴近此口径 |

### 4.2　梯度累积与"一个大 batch"严格等价的条件

梯度累积要**严格等价**于直接跑一个大 batch，需要同时满足六个条件：

1. **相同样本**——累积窗口内消费的数据必须和大 batch 一次性会消费的数据完全一致，不能因为分片方式不同而漏样本或重复样本；
2. **按全体有效 token 归一化**——而不是对每个 micro-batch 的均值再求均值；
3. **只执行一次 optimizer step**——梯度在累积窗口内只做加和或平均，权重更新只在窗口结束时发生一次；
4. **最终梯度后处理一致**——梯度裁剪必须作用在累积完成后的**最终梯度**上，不能对每个 micro-batch 的局部梯度分别裁剪；
5. **随机性与批相关状态一致**——dropout 等随机性、以及任何跨样本的运行统计量，都要表现得像是在完整大 batch 上一次性计算的；
6. **匹配通信库的归约语义**——全局梯度应等于 $\nabla\left(\sum_r \mathrm{NLL}_r \big/ \sum_r M_r\right)$，$r$ 遍历 DP rank，$M_r$ 是该 rank 的有效 token 数。

第 6 条在实现上最容易出错。DDP/FSDP 等框架默认对多 rank 梯度做**求平均**而不是求和：若本地已用"NLL 总和 / 本 rank 有效 token 数"算出归一化损失，默认求平均会再隐式地按 world size 多除一次。修法二选一：显式改成 sum-reduce 并按全局有效 token 数归一化，或在本地归一化时预先乘回 $W_{DP}$。本地 loss 和通信层不能重复除以 $W_{DP}$。

以上几点保证的是**数学等价**：在精确实数运算下，累积得到的梯度与一次性大 batch 算出的梯度是同一个量。并不保证浮点数值逐 bit 相同：累加顺序、all-reduce 的归约树形结构仍会带来最后几位有效数字的差异。

### 4.3　BF16、FP16 与 loss scaling

BF16 与 FP32 的指数位宽相同，动态范围相近，预训练里通常比 FP16 更不容易溢出。FP16 动态范围更窄，更依赖 loss scaling 来防止小梯度下溢为零。

## §5 lifecycle：从 warmup 到确定性 resume

### 5.1　warmup/stable/decay/cooldown + 数据阶段变化

warmup 压住训练早期的不稳定，stable/plateau 占据训练主体，decay/cooldown 把学习率按 schedule 逐步降到很小的值。AdamW 的更新公式与 warmup/cosine/WSD 的调度数学见 `optimizer_lr_schedule_tutorial.md`。流水线这一侧关心的是三件事：optimizer state 必须随 checkpoint 一起保存，调度是按 step 还是按 token 数推进，以及 batch size 变化时这两者如何联动。

训练末段常常同时发生两件事，但它们由不同配置控制。decay/cooldown 是 LR 侧的概念；data annealing / late-stage reweighting 是数据侧的独立概念，在训练末段提高高质量、精选数据在 mixture 中的权重（Llama 3 等公开训练报告有描述）。一个训练配方完全可以只做 LR decay 而不调整数据权重，反之亦然。

另一条生命周期轴是**上下文长度课程**：训练大部分预算用较短序列，后期切到长上下文。这会改变 $L_{seq}$，意味着 §4.1 的 global batch 核算和已消费 token 计数器都要在切换点处保持一致地重新计算。

### 5.2　恢复训练、样本顺序与数值复现

"resume 之后训练轨迹和不中断完全一样"对应三个不同强度的结果，各自需要的状态不同。

1. **可继续训练**：模型能从 checkpoint 恢复权重和优化器状态并继续降 loss，不要求数据顺序或数值完全对齐。只需要模型权重、优化器状态、学习率调度与已消费 token 计数这几项。
2. **样本序列一致**：重启后模型看到的数据顺序（哪个样本在哪个 step）与不中断时完全一致，不要求逐 bit 数值相同。在第 1 层之上还必须有 RNG 状态（Python、框架、CUDA 等各处的生成器）、数据迭代器位置、shuffle epoch、sampler 状态，以及数据加载 worker/prefetch 队列的状态——否则多 worker 预取顺序本身就是一个不确定源；多 rank 训练还要各 rank 的 checkpoint 元数据。
3. **数值/bitwise 同轨迹**（通常无法完全达到）：在第 2 层之上还需要 AMP loss-scaler 的状态，否则 resume 后的动态 loss scale 起点不同，早期几步的梯度缩放路径就会分叉；需要恰好在梯度累积窗口内尚未完成的"在途"梯度，checkpoint 只在累积边界打点则可规避这一项；需要确定性 kernel，cuDNN/cuBLAS 等非确定性算子必须显式关闭；还需要完全相同的软件版本与通信拓扑，不同 GPU 数、不同集合通信实现的浮点归约顺序不同就会有 ULP 级别的差异累积。**world-size 发生变化时**，即便重新分片语义写得再仔细，通常也无法保证与不改变 world-size 时逐 bit 相同的轨迹，数据切分方式和梯度归约的树形结构都会变。

缺失某一层要求的对应状态，实际达到的就是更弱的一层。checkpoint 元数据应明确标注自己保证到哪一层。

### 5.3　监控：分域 validation loss

validation loss 应该**分域**单独汇报——代码、数学、低资源语言、长上下文各算各的——而不只是一个总 loss。总 loss 很容易被 token 权重最高的域主导：网页文本占混合语料 80% 时，总 loss 曲线基本只反映网页文本质量。权重较低但战略上重要的能力（代码正确率、数学推理、某个低资源语言、长上下文检索能力）正在退化时，总 loss 曲线上看不出来，等发现时往往已经很难挽回。

### 5.4　loss spike / NaN 排查

按"最便宜先查"排列的推荐顺序，是启发式而不是强制流程，有更强先验时可以调整：

1. **查坏 batch / 数据分片**——看这次 spike 能否在同一个 batch 或 shard 上单独复现，这依赖确定性分片、sampler 的 cursor 状态和 lineage（§2.8），否则连"是哪个 batch"都定位不到；
2. **查 LR 与 resume 状态**——一次配置错误的学习率调度、或一次 resume 后学习率调度计数器被意外重置，都可能制造出实际步数与预期不符的异常更新；
3. **查梯度与激活**——梯度范数异常、激活范数爆炸、某层的 logit 量级失控、残差不稳定；这些诊断量背后的机制推导见 `normalization_init_tutorial.md`，具体监控阈值仍需自行工程化；
4. **查混合精度**——BF16/FP16 的溢出与下溢；
5. **查跨 rank 异常**——某个 DP/TP/PP rank 悄悄发散（例如某张卡硬件出问题）而其余 rank 表面正常，只有分别记录各 rank 的 loss/梯度统计才能看出来；
6. **查硬件故障**——ECC 错误、单卡故障、集合通信抖动。

保持可复现的 batch ID、step 号与 shard 归属，才能事后回放定位。§6 的 [D07] 是自动化第一道防线：一个基于滚动 median/MAD 的检测器，它标出"什么时候该开始排查"，不判断根因。

### 5.5　checkpoint 选择

用于下游部署的 checkpoint，依据分域验证集与专门的评测集组合来选。在固定数据混合下，训练 loss 主要反映拟合进度，不直接对应泛化能力或你真正关心的具体能力剖面。

### 5.6　常见误答

| 误答 | 正确判断 |
| --- | --- |
| $6ND$ 是精确公式，拿 $C$ 除以峰值 FLOPs 就是训练耗时 | 它是算法级近似，忽略了长上下文 attention 项、词表投影、激活重计算与 FLOP 计数约定；估墙钟还要引入 GPU 数量、峰值 FLOPs 与 MFU |
| Chinchilla 给出唯一一组精确指数，它和 Kaplan 的分歧就是"数据要更多" | 不同估计方法给出约 0.50/0.50、0.49/0.51、0.46/0.54，代入舍入后的 $\alpha,\beta$ 得到的解析最优约为 0.452/0.548，分歧还涉及拟合方法、训练 horizon 与 LR schedule 是否匹配预算 |
| "每参数 20 token"是普适常数 | 它是经验近似，$D_*/N_*$ 还依赖拟合常数 $A,B,k$，且当 $\alpha\ne\beta$ 时随 $C$ 缓慢漂移 |
| 训练 token 超过 Chinchilla 比例就是过拟合，说明配方错了 | 继续喂新的、非重复的相关数据通常仍在降 loss，over-trained 是相对 compute-optimal 前沿的经济学判断 |
| MinHash 度量语义相似度，semantic dedup 是它的无条件升级版 | MinHash 估的是 shingle 集合的 Jaccard 表层重复，semantic dedup 处理的是另一层级的重复，且会误删表达不同的合法文本 |
| 先切分、之后再去重也能挡住近重复跨 split | 必须先形成去重簇再按整簇分配 split，并且按传递闭包合并从未直接比较过的文档 |
| 配置里的域权重就是模型实际看到的 token 占比，名义 $D$ 就是有效数据量 | 按文档配置却当 token 权重用时二者通常不相等（按 token 目标占比定义时期望相等），重复呈现的 token 抬高名义 $D$ 却带不来同等的新信息 |
| packing 后能跨文档 attend 是 bug，加上 block-diagonal mask 就实现了文档独立 | 连续流是合法的序列语义选择；要文档独立还须单独屏蔽"上一文档 EOS → 下一文档首 token"这条跨文档标签 |
| 把 GPU 数或 TP/PP world size 乘进 global batch，$B_{input}$ 与 $B_{target}$ 通用 | 只有 DP world size 扩大独立样本数，位移标签下每条序列至多 $L_{seq}-1$ 个有效 target |
| 数学等价即数值逐 bit 相同，resume 跑起来就说明轨迹复现了 | 浮点累加顺序与归约树形结构会改变最后几位；bitwise 同轨迹另需 scaler 状态、在途梯度、确定性 kernel 与相同拓扑，world-size 变化通常做不到 |

## §6 从零实现：语料 → packing → 关键不变量 sanity check

完整可跑脚本见 [`code/pretraining_pipeline.py`](code/pretraining_pipeline.py)，纯 Python 标准库、零第三方依赖、CPU 秒级跑完 [D01]–[D07] 全部 sanity check。四组 demo 串成一条迷你流水线：文本 → dedup → pack + loss mask → synthetic budget planner → spike monitor。脚本用可执行断言覆盖 dedup、packing、Chinchilla 拟合、spike 监测四个不变量，没有真正的前向/反向/optimizer step，也没有 checkpoint 写入与 resume。下面是三段核心代码，document packing 的完整功能测试、EOS 边界（§3.2）、padding（§3.3）作为短测试函数并入同一脚本。

**[D01] MinHash / LSH / cluster-split 传递闭包**（§2.3、§2.5 的可执行版）：

```python
doc_a = set(range(1, 11))                 # {1,...,10}
doc_b = set(range(1, 9)) | {11, 12}        # {1,...,8,11,12}
j_exact = exact_jaccard(doc_a, doc_b)
assert abs(j_exact - 2 / 3) < 1e-12        # |A∩B|=8, |A∪B|=12 -> 8/12

sig_a = minhash_signature(doc_a, num_perm=1024, seed=12345)
sig_b = minhash_signature(doc_b, num_perm=1024, seed=12345)
j_est = minhash_estimate(sig_a, sig_b)     # MinHash ESTIMATE, not exact
tol = 6 * math.sqrt((2 / 3) * (1 / 3) / 1024)   # principled sampling tolerance
assert abs(j_est - 2 / 3) < tol             # close, not asserted exactly equal

p_high = lsh_candidate_probability(2 / 3, b=128, r=8)   # P = 1-(1-s^r)^b
p_low = lsh_candidate_probability(0.05, b=128, r=8)
assert p_high > 0.9 and p_low < 0.1         # high-similarity candidates recalled far more often

uf = UnionFind(4)               # docs: 0=A, 1=B, 2=C, 3=D(unrelated)
uf.union(0, 1); uf.union(1, 2)  # A~B found, B~C found (A,C never directly compared)
assert uf.find(0) == uf.find(2) # transitive closure still merges them into one cluster
# -> every doc in this cluster MUST receive the same train/val/test split label
```

**[D02]/[D03] document packing：位移标签 + block-diagonal mask + 功能测试**（§3.4 的可执行版）：

```python
tokens = [11, 12, 13, 21, 22]
doc_ids = [0, 0, 0, 1, 1]

labels_indep = shifted_labels_independent_docs(tokens, doc_ids)
assert labels_indep == [12, 13, IGNORE, 22, IGNORE]      # doc boundary at index 2 masked
labels_cont = shifted_labels_continuous(tokens)
assert labels_cont == [12, 13, 21, 22, IGNORE]           # continuous stream: no boundary masking

m_block = block_diagonal_causal_mask(doc_ids)
m_cont = continuous_causal_mask(len(tokens))
assert m_block[3][2] is False and m_block[4][0] is False and m_block[4][3] is True
assert m_cont[3][2] is True          # SAME (q,k); continuous mode allows it -- a config choice

values = [100, 100, 100, 0, 0]       # doc0=100s, doc1=0s
assert masked_uniform_attention_output(m_cont[3], values) == 75.0    # (100+100+100+0)/4
assert masked_uniform_attention_output(m_block[3], values) == 0.0    # only key=3 allowed

values2 = [1000, 1000, 1000, 0, 0]   # bump doc0's values 10x
assert masked_uniform_attention_output(m_block[3], values2) == 0.0   # doc1's output: UNCHANGED
assert masked_uniform_attention_output(m_cont[3], values2) == 750.0  # continuous: DOES change
```

**[D06] synthetic Chinchilla 曲线拟合**（§1.3 的可执行版，**是拟合单元测试，不是经验定律的验证**）：

```python
alpha, beta, k_flops = 0.34, 0.28, 6            # exponents straight from §1.3's CORRECTIONS
e_true, a_true, b_true = 1.5, 400.0, 400.0      # SYNTHETIC ground truth, not the paper's fitted values
points = [(n, d, loss_surface(n, d, e_true, a_true, b_true, alpha, beta))
          for n in logspace(1e7, 1e12, 6) for d in logspace(1e8, 1e13, 6)]

e_hat, a_hat, b_hat = fit_e_a_b(points, alpha, beta)    # linear least squares (alpha,beta fixed)
assert abs(e_hat - e_true) < 1e-6 and abs(a_hat - a_true) < 1e-3 and abs(b_hat - b_true) < 1e-3
assert e_hat < min(loss for _, _, loss in points)       # irreducible loss floor
assert a_hat > 0 and b_hat > 0

c_budget = 6e21
n_star, d_star = analytic_optimum(c_budget, k_flops, alpha, beta, a_true, b_true)
n_star_100x, d_star_100x = analytic_optimum(100 * c_budget, k_flops, alpha, beta, a_true, b_true)
assert abs(n_star_100x / n_star - 100 ** (beta / (alpha + beta))) < 1e-6   # ~8.0x
assert abs(d_star_100x / d_star - 100 ** (alpha / (alpha + beta))) < 1e-6  # ~12.5x
```

[D04]（EOS 边界，复用 [D02] 同一条 doc-id 规则）、[D05]（padding 的双重 mask + 无空行检查）、[D07]（rolling median/MAD spike 检测 + NaN/Inf 无条件告警 + spike/sustained-shift 事件分类）作为独立断言组并入 `main()`。

## §7 26 高频面试题

按难度分三档，每题给出答案要点与易踩坑。

### L1必会题

<details>

<summary>Q1. 什么是 $N$、$D$、$C$？$C\approx 6ND$ 里的"6"是怎么来的、为什么只是近似？</summary>

- $N$=每 token 参与主要矩阵乘法的 compute-bearing 参数规模（MoE 取 active，embedding 查表通常不算在内，见 §1.1）；$D$=训练时呈现给模型的 token 数；$C$=近似训练 FLOPs
- 一次 multiply-add 记 2 FLOPs（一乘一加）：前向对每个 token 过一遍主要矩阵乘法，总量约 $2N$/token，$D$ 个 token 即 $2ND$；反向要同时算激活梯度和权重梯度，量级约为前向的两倍，即 $4ND$；前向+反向 $\approx 6ND$——这本身是一种计数约定（有些工具把 multiply-add 记 1 FLOP），不是物理常数
- 忽略了长上下文 attention 的 $L_{seq}^2$ 项（固定 $D$ 时总开销近似随 $D\times L_{seq}$）、词表投影、norm/softmax、激活重计算、FLOP 计数约定差异；不是精确公式，也不能直接换算墙钟时间——粗略估算 GPU-days 需要额外引入 GPU 数量、峰值 FLOPs 和 MFU：$t\approx C/(n_{GPU}\times\text{峰值FLOPs}\times MFU)$，通信开销、重计算、data stall 会让这只是一个估算而非精确预测（§1.1）

把 $6ND$ 当成精确公式、把 $N$ 里连 embedding 参数都不加区分地算进去、或者直接拿 $C$ 除以峰值 FLOPs 当训练耗时（不考虑 MFU），是这题最常见的三个失分点。

</details>

<details>

<summary>Q2. causal LM 的 input/label 是怎么错位的？为什么随机初始化模型的 loss 通常接近 $\ln V$？</summary>

- teacher forcing 下，位置 $i$ 的 label 是 $x_{i+1}$（整体右移一位）；序列最后一个位置没有下一个 token，标记为 `IGNORE`（取 $-100$，§3.1）
- packing/文档边界场景下，"标签有效当且仅当当前位置与下一位置属于同一文档"这条规则统一处理了序列末尾和文档边界两种 IGNORE 来源（§3.2）
- 随机初始化的模型对下一个 token 近似均匀地分配概率，此时每个有效位置的 NLL 近似 $\ln V$（$V$ 是词表大小）——训练刚开始时观察到的 loss 应该接近这个值，如果明显偏离（远高于或出现异常低值），通常提示初始化、label 对齐或数据本身有问题

以为"IGNORE 只出现在 batch 末尾"、漏掉文档边界也会产生 IGNORE，或者说不出 $\ln V$ 这个初始 loss 量级的由来，是这题常见的不完整回答。

</details>

<details>

<summary>Q3. BF16 和 FP16 的动态范围有什么区别？什么情况下需要 loss scaling？</summary>

- BF16 与 FP32 的指数位宽相同，动态范围相近，预训练里通常比 FP16 更不容易溢出；代价是尾数位更少，单个数值的相对精度比 FP16 低
- FP16 的指数位宽比 BF16/FP32 窄，动态范围更小——小梯度容易在 FP16 下溢为零（下溢），大数值也更容易溢出为 Inf
- **loss scaling**（把 loss 乘一个较大的系数再反传，梯度回缩时再除掉）是 FP16 训练里用来把小梯度"抬"到 FP16 可表示范围内的标准做法；BF16 因为动态范围已经和 FP32 接近，通常不需要这类 loss scaling（§4.3）
- 这也是 §5.4 排查 loss spike/NaN 时"混合精度"这一层要检查的内容之一——scaler 状态本身也是 §5.2 确定性 resume 需要考虑的一部分状态

把"混合精度训练"简单等同于"BF16 和 FP16 都需要 loss scaling"，说不出两者动态范围差异带来的不同工程需求，是这题的常见误答。

</details>

<details>

<summary>Q4. 训练 token 数超过 Chinchilla 建议比例，是不是过拟合？</summary>

- 不必然是。仅凭"超过 Chinchilla 建议比例"本身，不能单独判定是过拟合——如果继续喂的是**新的、非重复、同分布或相关分布**的数据，固定模型规模通常仍能降低 loss，不满足"训练 loss 降、验证 loss 升"这个过拟合定义
- 真实发生的是：继续增大 $D$ 时，最终计算量 $C=kND$ 本身也在增大；在这个更大的 $C$ 下，该配置相对它自己所在的 compute-optimal 前沿点具有更小的 $N$、更大的 $D$，被称为"over-trained"——这是相对 compute-optimal 前沿的经济学判断，不是训练出错（§1.4）
- 故意训小模型超训是为了压低部署侧显存/延迟/推理成本，训练计算最优 $\ne$ 全生命周期成本最优

把"over-trained"（相对预算的经济学概念）和"过拟合"（泛化失败）混为一谈，或者忽略"新数据/非重复"这个前提就断言"肯定不是过拟合"，是这题最常见的失分点。

</details>

<details>

<summary>Q5. 什么是 document packing？为什么普通 causal mask 不会自动隔离文档？</summary>

- packing：把多篇文档拼接进一条训练序列以提高利用率
- causal mask 只保证"看不到未来"，不检查 token 是否属于同一篇文档——后一篇文档默认能读到前一篇文档的历史 token，这是**很多系统故意采用**的连续 EOS 分隔流语义，不是 bug
- 若要"文档独立"，需要 block-diagonal mask $M[q,k]=(k\le q)\wedge(\text{docid}[q]=\text{docid}[k])$（§3.4）

把"能跨文档 attend"直接判定为实现错误，说不出这是一个合法的语义选择。

</details>

<details>

<summary>Q6. attention mask 和 loss mask 有什么区别？什么时候两个都必需，什么时候可能冗余？</summary>

- attention mask（key padding / block-diagonal）控制"谁能看谁"，作用在 attention 计算里；loss mask 控制"哪些位置的预测参与梯度"，作用在损失计算里——两者独立，阻断 attention 不会自动屏蔽对应标签
- **block-diagonal 场景必须两个都做**：即使文档 B 不能读文档 A，文档 A 最后位置仍可能被要求预测文档 B 首 token，除非单独用边界 loss mask 排除（§3.2、§3.4）
- **padding 场景要看布局**：loss mask 必须有；但严格 causal + 右侧 padding 时，causal mask 本身已挡住未来的 padding key，key mask 对真实 query 可能冗余——左 padding 或其他布局下则不可省略（§3.3）

以为"两个 mask 任何时候都同等必需"，或反过来"可以无条件省略 key mask"，都是这题的失分点；正确答案要按具体布局分情况讨论。

</details>

<details>

<summary>Q7. global batch size / tokens-per-update 到底怎么算？要分清哪几种口径</summary>

- 送入模型的 token 数：$B_{input}=B_{micro,seq}\times L_{seq}\times G_{acc}\times W_{DP}$（$B_{micro,seq}$ 是每个 DP replica 的序列数，不是"每张卡"）；只有 DP world size 直接扩大独立样本数，TP/PP 切分的是同一份样本的计算，不应重复计入
- 实际参与 loss 的有效 target 数：$B_{target}=\sum_i m_i$（$m_i$ 是每条序列的有效 target mask 计数）——即使完全无 padding，因为每条序列末位置的标签是 `IGNORE`，$B_{target}$ 也是每序列 $L_{seq}-1$ 而非 $L_{seq}$；有 padding/packing 边界屏蔽时会进一步小于 $B_{input}$
- 三种记账口径不能混用：推进已消费 token 计数器/对齐 scheduler 通常用 $B_{input}$；损失归一化分母、scaling law 里的 $D$ 应该更贴近 $B_{target}$（§4.1）

把 TP/PP 的 world size 也乘进 batch size 公式，或者把 $B_{input}$ 和 $B_{target}$ 当成同一个数直接换用，是这题最常见的两类计算错误。

</details>

<details>

<summary>Q8. 为什么不同 tokenizer 的 PPL 不能直接比较？</summary>

- PPL 的分母是 token 数，本身依赖 tokenizer——同样的总 NLL，切得更碎的 tokenizer 算出的 loss/token 更低
- BPB 把分母换成同一份文本的字节数，对 tokenizer 更稳健，但仍要求两边评测的是同一份规范化后的字节流
- 精确公式：$\mathrm{PPL}=\exp(\sum \mathrm{NLL}_i/\#\text{valid tokens})$，$\mathrm{BPB}=\sum \mathrm{NLL}_i/(\#\text{bytes}\cdot\ln 2)$（§3.6，完整推导见 tokenization_tutorial.md §1.3）

只知道"PPL 越低越好"，说不出它依赖 tokenizer 这个前提条件。

</details>

### L2进阶题

<details>

<summary>Q9. Kaplan (2020) 和 Chinchilla (2022) 的 scaling law 结论有什么不同？</summary>

- Kaplan：$N_{opt}\propto C^{0.73}, D_{opt}\propto C^{0.27}$，计算预算增加更多分给参数
- Chinchilla：无唯一"精确指数"，不同方法给出约 0.50/0.50、0.49/0.51、0.46/0.54；参数化模型 $L=E+A/N^\alpha+B/D^\beta$ 拟合 $\alpha\approx0.34,\beta\approx0.28$，解析最优 $N_*\propto C^{0.452}, D_*\propto C^{0.548}$
- 分歧不只是"数据更多"：训练 horizon 设计、LR schedule 是否匹配预算、拟合方法都不同（§1.3）

只答"Chinchilla 说要更多数据"，说不出三组估计方法各自给出的指数，也说不出训练 horizon/LR schedule 这条关键技术分歧。

</details>

<details>

<summary>Q10. 推导 Chinchilla 约束优化下的 $N_*,D_*$ 解析解。</summary>

- 目标：$\min L(N,D)=E+A/N^\alpha+B/D^\beta$ s.t. $C=kND$
- 代入约束消掉一个变量，用 Lagrange 乘子或直接对 $\log N$ 求导找驻点，解得 $N_*=(\alpha A/\beta B)^{1/(\alpha+\beta)}(C/k)^{\beta/(\alpha+\beta)}$，$D_*=(\beta B/\alpha A)^{1/(\alpha+\beta)}(C/k)^{\alpha/(\alpha+\beta)}$
- 代入 $\alpha=0.34,\beta=0.28$ 得指数 0.452/0.548；100 倍计算预算下 $N_*$ 约扩大 $100^{0.452}\approx 8.0$ 倍、$D_*$ 约扩大 $100^{0.548}\approx 12.5$ 倍——这条推导在固定 $A,B,\alpha,\beta,k$ 时数学上成立，但不能外推成经验断言（§1.3、[D06]）

指数是交叉形式：$N_*$ 对应 $\beta/(\alpha+\beta)$、$D_*$ 对应 $\alpha/(\alpha+\beta)$，最常见的失分是把它记反。

</details>

<details>

<summary>Q11. 给定一个计算预算，怎么决定模型参数量和训练 token 数？</summary>

- 先分清目标：训练计算最优（直接套 $N_*,D_*$）还是推理感知最优（先卡死部署预算允许的小 $N$，固定 $C$ 下 $D$ 相应增大，即故意超训）
- MoE 用 active 参数算计算预算，total 参数算显存/checkpoint（§1.2）
- 长上下文会引入 $6ND$ 忽略的 $L_{seq}^2$ 项，需单独预留预算（§1.5）

只会背 $N_*,D_*$ 公式，说不出"训练最优≠推理感知最优"这条分岔。

</details>

<details>

<summary>Q12. attention mask、loss mask、position id 这三个旋钮在 packing 里该怎么独立设计？</summary>

- 这不是"只有连续流 vs 独立文档"两套打包好的 recipe——**attention 边界、target 边界、position id 是三个可以独立组合的旋钮**，不必须绑定在一起选：
  - **attention 边界**：普通 causal（$k\le q$，允许跨文档读历史）或 block-diagonal（$k\le q \wedge \text{docid}[q]=\text{docid}[k]$，文档间互不可见）；
  - **target 边界**：即使用普通 causal attention（文档间可以互相"看"），仍然可以单独用文档边界规则屏蔽"上一文档 EOS → 下一文档首 token"这条跨文档标签（§3.2）——attention 允许跨文档读取历史，不代表这条 target 就必须保留；
  - **position id**：是否重置是位置编码方案决定的选择而非数学必需——标准 RoPE 对同一文档内整体的位置平移在理想数学下不敏感，绝对位置编码通常应该重置，具体取舍见 §3.4
- 两种最常见的组合是"连续流=causal attention+不重置 position id+不额外屏蔽 target"和"独立文档=block-diagonal attention+重置 position id+屏蔽跨文档 target"，但工程上也存在"causal attention 但仍屏蔽跨文档 target"这类中间配置，不能把两个极端当成唯二选项
- 两者都要保留"文档内容末 token → EOS"的监督，只是"跨文档"部分的处理方式不同（§3.2、§3.4）

把这道题当成"背两套固定 recipe"来答，说不出这三个旋钮可以独立组合，也说不出 RoPE 重置 position id 并非数学上必需。

</details>

<details>

<summary>Q13. 梯度累积要满足哪些条件才能等价于一个大 batch？"等价"要分清楚是哪一层</summary>

- 相同样本（累积窗口消费的数据必须和大 batch 一次性会消费的完全一致）
- 按全体有效 token 归一化，不是每个 micro-batch 均值再平均
- 只执行一次 optimizer step；梯度裁剪作用在累积完成后的最终梯度上，不是每个 micro-batch 分别裁剪
- 这只保证**数学等价**（在精确实数运算下，累积出的梯度与一次性大 batch 算出的梯度是同一个公式）。要更进一步，还要看：DDP 默认做梯度**求平均**而不是求和，必须显式用全局有效 token 分母做补偿，否则等价于被多除一次 world size（Q16）；dropout/RNG 状态、任何 batch 相关的运行统计量，都要在两种设置下走完全相同的随机数序列才谈得上比较；AMP 的 loss-scaler 状态、scheduler 的更新次数（每个"大 batch"只应该触发一次 scheduler.step，不是每个 micro-batch 一次）也必须对齐
- 即便以上全部对齐，**浮点归约的顺序**（跨 micro-batch 求和的顺序、跨 DP rank all-reduce 的树形结构）不同仍会导致最后几位有效数字不同——这一步只能到"数学等价"，做不到无条件的**数值 bitwise 等价**（§4.2、§5.2 的三层强度框架同样适用于这里）

漏掉"裁剪必须在累积后做一次"、用均值的均值代替总和除以总数、或者把"数学等价"误当成"数值上逐 bit 相同"，是这题最常见的三个坑。

</details>

<details>

<summary>Q14. 怎么设计一个近似去重（MinHash+LSH）pipeline？LSH 的候选概率公式是什么？</summary>

- 文档→shingle 集合→MinHash 签名（多个哈希函数取最小值）→估计 Jaccard 相似度
- LSH 把签名分成 $b$ 个 band、每 band $r$ 行，在各 MinHash 行/哈希函数近似独立的理想化假设下，候选概率 $P=1-(1-s^r)^b$，只做候选召回
- 候选对通常还要精确 Jaccard 复核，且 MinHash 估计的是集合重叠，不是语义相似度（§2.3、[D01]）

把 LSH 候选直接当成"确认的近重复"，跳过精确 Jaccard 复核这一步。

</details>

<details>

<summary>Q15. 为什么必须"先去重再切分"？</summary>

- 若先切分再去重，近重复内容（镜像站点、改写转载）可能一份在 train、一份在 val
- 模型训练时"背"过验证集的近乎复制品，validation loss 虚低，误导 early-stopping 和模型选择
- 正确做法：先做近似去重得到连通分量（去重簇），再把整簇原子性分配到某个 split；即使 A、C 从未被直接比较，只要通过 B 传递相连，也必须同 split（§2.5、[D01]）

只知道"要去重"，说不出去重与切分的先后顺序，也说不出传递闭包这条隐藏要求。

</details>

<details>

<summary>Q16. loss 归一化最容易踩的坑是什么？</summary>

- 正确公式：$\sum \mathrm{NLL}_i / \#\text{有效 target}$，对全局批次一次性归一化
- 常见错误：先对每个 micro-batch 求均值，再对这些均值求平均——当各 micro-batch 有效 token 数不同（padding/packing/变长）时会引入权重偏差
- 多 rank 场景下还有一个变体错误：DDP 默认对梯度做**求平均**，如果本地已经用"NLL 总和 / 本 rank 有效 token 数"算好了归一化损失，再让 all-reduce 按 rank 数求平均，等价于错误地把分母按 world size 缩放了一次；正确做法是要么本地只算"总和"、all-reduce 后再除以**全局**有效 token 总数，要么显式改成 sum-reduce 并自己控制归一化时机（§3.5、§4.2）
- 正确实现要跨 micro-batch/跨 rank 分别累计"NLL 总和"与"有效 token 总数"，最后统一相除

写出"均值的均值"这种实现，或者让 all-reduce 的默认求平均语义和本地归一化逻辑打架，在变长 batch/多 rank 场景下都会悄悄扭曲损失，且不会报错——是这题最容易被忽视的隐患。

</details>

<details>

<summary>Q17. EOS/BOS 在 packing 边界处的监督规则是什么？</summary>

- 保留：文档最后一个内容 token → EOS 的 loss（教模型何时该停）
- 屏蔽（按配置）：上一篇文档的 EOS → 下一篇文档首 token/BOS 的 loss（这个标签只是拼接产物）
- 两条规则可以由同一个"标签有效当且仅当当前位置与下一位置同文档"的规则自动同时产出，不需要给 EOS 单独写特判（§3.2、[D04]）

以为需要为 EOS 写额外的特殊逻辑，没意识到它可以被同一条 doc-id 边界规则统一处理。

</details>

<details>

<summary>Q18. PAD 和 EOS 共用同一个 token ID 时，构造 mask 要注意什么？拆开两条因果链看</summary>

- PAD=EOS 本身不一定错，很多 tokenizer 没有独立 PAD token
- **若按 `target==pad_id` 构造 loss mask**：会把内容末 token→EOS 这条真实监督也当成"target 是 padding"一起屏蔽掉，直接删除了模型学"该在哪停"的信号——这是最严重的事故链条
- **若按 `key==pad_id` 构造 key mask**：后果不同——它错误地把真实 EOS 位置当成 padding key 藏了起来（真实 query 读不到这个 EOS 的上下文表示），但不会自动删除任何 loss，因为 loss mask 是按 target 位置单独算的，这是"藏 context"而不是"删 loss"的错误
- 正确做法统一为：按每条序列的真实有效长度 `valid_len`（或显式 padding 标记）构造 mask，不用 token ID 相等判定（§3.3、[D05]）

只会背"PAD=EOS 就是错的"这种绝对化结论，说不出按 ID 构造 loss mask 和按 ID 构造 key mask 分别会造成哪一种、性质不同的错误。

</details>

### L3高级题

<details>

<summary>Q19. 设计一条从网页抓取到训练分片的完整数据流水线。</summary>

- **§2.1 的十一步是参考流程，硬约束主要是一条：去重簇必须先形成、再做 cluster-level split**（依赖已经算出的连通分量，§2.5）；隐私/安全处理、质量过滤在工程上常常前移或在多个阶段各跑一轮，不必卡死在表格里的固定位置
- raw-document 阶段之后还有一段常被漏答的 **tokenized 阶段**：tokenizer materialization（把 raw-document shards 实际跑一遍 tokenizer，产出 token stream）、token 级 mixture 采样（校正文档级配额和实际 token 占比的偏差）、长文档 chunk/truncate、以及决定训练时如何读取这些 tokenized shard 的 data-stream/sampler 层（shard shuffle、域采样 RNG、DP rank 互斥分片、worker/prefetch 状态、cursor 位置），见 §2.7–§2.10
- 每步都要保留 lineage：源/快照版本/过滤器版本/去重簇 ID/split 归属/chunk 归属/最终 shard 位置（§2.1、§2.9）
- benchmark 去污染针对评测集，和 train-val 切分是两回事（§2.4、§2.6）

只背下十一步的名字和顺序、把它当成唯一正确答案，说不出"去重先于切分"是唯一的硬约束，也漏掉 tokenizer/chunking/sampler 这几个 tokenized 阶段的环节。

</details>

<details>

<summary>Q20. 要做到"确定性 resume"，checkpoint 需要保存哪些状态？"确定性"具体是哪种确定性？</summary>

- 先分清楚要哪一层（§5.2）：**可继续训练**只需模型权重+优化器状态（Adam 的 $m,v$）+ 学习率调度/已消费 token 计数；**样本序列一致**再加 RNG 状态、数据迭代器位置、shuffle epoch、sampler 状态、dataloader worker/prefetch 状态，多 rank 训练还要各 rank 的 checkpoint 元数据
- **数值/bitwise 同轨迹**这一最强层次额外需要 AMP loss-scaler 状态、在途（未完成累积窗口的）pending 梯度、确定性 kernel 与完全相同的软件版本/通信拓扑——即便如此，world-size 改变通常仍无法保证逐 bit 相同的轨迹；checkpoint 在 world-size 改变时如何重新分片、重分片后能否与原轨迹一致，没有通用答案，`distributed_training_tutorial.md` 也未展开
- 漏掉数据迭代器位置或 RNG 状态，直接跌出"样本序列一致"这一层；只满足前两层而漏掉 AMP scaler/pending 梯度/确定性 kernel，能"重启训练""顺序一致"，但达不到数值上的完全复现

把"确定性 resume"当成非黑即白的单一结论，说不出这三层强度分别对应什么必要状态。

</details>

<details>

<summary>Q21. 预训练突然 loss spike 或 NaN，怎么分层排查？</summary>

- 这是一个**推荐的启发式排查顺序**（按"最便宜先查"排列），不是强制流程，有更强先验时可以调整：坏 batch/数据分片（能否在同一批次复现，依赖确定性分片+lineage）→ LR 与 resume 状态 → 梯度与激活（诊断量见 normalization_init_tutorial.md）→ 混合精度溢出/下溢 → 跨 rank 异常（需分 rank 记录统计量）→ 硬件故障
- 保留可复现的 batch ID/step 号，用于事后回放定位
- 自动化第一道防线：rolling median/MAD 检测器，负责标出"什么时候该开始排查"，不负责判断根因（§5.4、[D07]）

上来就猜"肯定是学习率太大"，跳过"能否在同一 batch 复现"这个最便宜也最该先做的第一步。

</details>

<details>

<summary>Q22. 为什么要按域分别报 validation loss？</summary>

- 总 loss 容易被 token 权重最高的域主导（如网页文本占比大），掩盖代码/数学/低资源语言/长上下文这些低权重但重要能力的退化
- 分域监控才能在还来得及调整 mixture/checkpoint 选择时发现问题
- checkpoint 选择也应依据分域验证集/评测集组合，而不只是训练 loss（§5.3、§5.5）

只看一条总 loss 曲线就下结论"模型训得挺好"，看不到被稀释掉的局部退化。

</details>

<details>

<summary>Q23. mixture 采样的"域权重"和模型实际看到的 token 权重，什么时候相等、什么时候不相等？</summary>

- **分两种情况，不做无条件断言**：域权重按 token-level 目标占比定义（配置项写的就是"这个域应该贡献多少比例的 token"）时，期望意义下两者可以相等；按文档数/文档权重配置、却被当成 token 权重直接使用时才会脱节——因为不同域的平均文档长度不同（§2.7）
- **三种采样单位要分清**：按文档采样（不区分域，直接按文档池的抽取概率抽整篇）、先采域再采文档（两阶段过程，第一层的"域权重"在这里才有明确含义）、按 token 采样（直接从拼接后的 token 流切块，与文档边界解耦）
- **应对**：采样后单独统计并汇报"实际 token 级占比"，而不是只报配置值；这也关联 scaling law 里 $D$ 的口径——重复采样/近重复残留会让"有效数据量"低于名义 $D$

无条件断言"域权重肯定不等于 token 权重"，或者把"按文档采样"和"先采域再采文档"当成同一个概念混着讲，是这题最容易踩的两个坑。

</details>

<details>

<summary>Q24. "三种泄漏"（train-val 泄漏 / benchmark 污染 / packed 跨文档上下文）分别是什么，怎么应对？</summary>

| 类型 | 核心判断 | 方法边界 |
| --- | --- | --- |
| train-val 泄漏 | 先切分再去重导致近重复跨 split → cluster-level split **缓解**（§2.5） | MinHash/LSH+n-gram 会漏大幅改写、跨语言翻译、语义相同但表层不同的"答案变体"——只是显著降低表层重复泄漏概率，不是"解决" |
| benchmark 污染 | 训练语料含评测集内容 → n-gram 重叠去污染**缓解**（§2.4） | 漏检更严重（benchmark 常被人工改写规避检测），且对模板化/样板文本（常见代码片段、法律模板）可能假阳性 |
| packed 跨文档历史上下文 | **通常不是数据泄漏，是"跨样本上下文耦合"这一序列语义选择**（§2.6、§3.4） | 是否用 block-diagonal mask 切断，取决于是否要求"文档独立"这条建模假设 |

把前两种检测说成能"彻底解决"，或把第三种也当成泄漏一律要求切断，是这题的失分点。

</details>

<details>

<summary>Q25. MoE 模型的计算预算和显存预算分别怎么算？</summary>

- **计算预算用 active 参数**：$6ND$ 里的 $N$ 取真正参与该 token 前向/反向计算的专家参数
- **$6N_{active}D$ 本身仍是近似**：不包含 MoE 特有的 all-to-all 通信开销、专家间负载不均衡、capacity padding（超出路由容量的 token 需丢弃或额外处理）的额外计算，细节见 `moe_tutorial.md`
- **聚合 checkpoint/model-state 大小随 total 参数增长**：所有专家权重都要能完整存下来
- **单卡显存不能简单按 total 算**：还取决于 EP（专家并行）/FSDP 等分片方式、参数 dtype、优化器状态（Adam 的 $m,v$ 按分片后规模算）、梯度、activation 显存——同样的 total 参数量，分片策略不同单卡显存相差很大；总 loss 里若有 router aux loss，量级通常远小于主 loss（§1.2、§3.5）

把 active 和 total 参数用混，用 total 参数估算 $6ND$ 训练计算量会显著高估；反过来只用 total 参数简单除以卡数估算单卡显存，忽略分片方式和优化器/activation 占用，是这题另一个常见的过简化。

</details>

<details>

<summary>Q26. 设计一套数据 lineage 方案，支持"半年后审计某条训练数据的来源"。</summary>

- **最低追溯（合格底线，不足以支撑严格审计）**：源数据集/URL → 快照/爬取版本 → 抽取与过滤流水线版本（含阈值）→ 去重簇 ID → split 归属 → 最终 shard 文件+偏移量
- **强审计（在最低追溯之上额外记录）**：sample/content hash（确认"现在的内容"和"当初写入时"一致）、原始快照 hash（不只是版本号）、tokenizer/normalizer 哈希（具体制品而非版本字符串）、代码/容器版本、mixture 采样随机种子、shard checksum、document→segment/token 精确映射（尤其有 chunking 时，§2.9）、许可证与删除请求（takedown）状态
- 强审计记录服务四个场景：benchmark 污染排查、loss spike 复现定位、模型输出可疑内容溯源、合规删除请求响应（§2.1、§2.8、§2.9、§5.4）

只存一个"来源 URL+版本号"就以为够了，说不出 content hash、tokenizer 哈希、mixture 种子、document→token 映射这些同样决定"这条样本现在长什么样"的中间状态，也说不出许可证/删除请求这条合规维度。

</details>

## §A 附录：sanity check

从零实现应满足以下关键不变量，脚本见 [`code/pretraining_pipeline.py`](code/pretraining_pipeline.py)：

1. **[D01] dedup**：精确 Jaccard $J(A,B)=8/12=2/3$（$A=\{1..10\}$，$B=\{1..8,11,12\}$）；固定种子 + 1024 permutation 的 MinHash 估计落在按采样理论设定的容差内（不断言精确相等）；LSH（$b=128,r=8$）候选概率对高相似 pair（$s=2/3$）远高于低相似 pair（$s=0.05$）；并查集验证 A~B、B~C 的传递闭包使 A、C 落入同一去重簇，必须分配同一 split。
2. **[D02] packing 位移标签**：`tokens=[11,12,13,21,22]`，`doc_ids=[0,0,0,1,1]` 下，独立文档语义标签为 `[12,13,IGNORE,22,IGNORE]`；连续流语义标签为 `[12,13,21,22,IGNORE]`（同一份 token 流，两种合法配置给出不同标签）。
3. **[D03] block-diagonal mask + 功能测试**：`M[3,2]=False`、`M[4,0]=False`、`M[4,3]=True`（block 模式）；同一坐标连续流模式 `M[3,2]=True`。$QK^\top=0$ 下，doc0 values 从 `[1,2,3]` 改为 `[1000,2000,3000]`、doc1 values 保持 `[10,20]` 时，block 模式下文档 1 的 q3、q4 输出分别保持 `10.0`、`15.0` 不变，连续流模式下 q3 输出从 `4.0` 变为 `1502.5`。
4. **[D04] EOS 边界**：在 `[11,12,13,EOS,21,22,EOS]` / `doc_ids=[0,0,0,0,1,1,1]` 上，内容末 token（13）→EOS 的标签保留（=EOS），EOS→下一文档首 token（21）的标签被屏蔽（IGNORE）——两者由同一条 doc-id 边界规则自动产生。
5. **[D05] padding**：`valid_len=3` 时 loss mask 为 `[1,1,0,0,0]`；真实 query 无法 attend 到 padding key；所有 attention 行都非空（不会产生全 `-inf` 行导致 softmax NaN）。
6. **[D06] Chinchilla 拟合**：从 log-spaced 网格（$N\in[10^7,10^{12}]$, $D\in[10^8,10^{13}]$，各 6 点）无噪声恢复 $E,A,B$ 与合成真值的误差 $<10^{-3}$；恢复的 $E$ 小于所有有限观测 loss；解析 $N_*,D_*$ 与细网格搜索最小点相差不超过一个网格步长；计算预算扩大 100 倍时 $N_*$ 约扩大 $100^{0.452}\approx 8.0$ 倍、$D_*$ 约扩大 $100^{0.548}\approx 12.5$ 倍。
7. **[D07] spike 监测**：序列 `[2.00,2.02,1.98,2.01,1.99,2.00,2.03,1.97,3.00]` 只有最后一点被标记；平滑下降序列全程不误报；同样量级的单点跳变分别标记为 `spike`（后续回落）与 `sustained_shift`（后续维持高位）两种不同事件类型；NaN/Inf 无条件立即标记为 `non_finite`。

运行 `python3 code/pretraining_pipeline.py` 的**真实输出**：

```text
[D01] dedup: exact J(A,B)=0.6667, MinHash est=0.6250 (tol 0.0884), LSH P_high=0.9939 vs P_low=0.0000; real banding candidates=[(0, 1), (1, 2)], confirmed edges=[(0, 1), (1, 2)], cluster split={0: 'test', 1: 'test', 2: 'test', 3: 'train'}  PASS
[D02] packing labels: independent-doc=[12, 13, -100, 22, -100], continuous=[12, 13, 21, 22, -100]  PASS
[D03] block mask: doc1 out unaffected by doc0 values changing (q3: 10.0->10.0, q4: 15.0->15.0); doc0 out unaffected by doc1 values changing (q0/q1/q2: 1.0/1.5/2.0); continuous mode DOES change (q3: 4.0->1502.5)  PASS
[D04] EOS boundary: labels=[12, 13, 999, -100, 22, 999, -100] (content-end->EOS kept, EOS->next-doc masked)  PASS
[D05] padding: loss_mask=[1, 1, 0, 0, 0], no attention row is empty, PAD==EOS adversarial loss_mask=[1, 1, 0, 0, 0] (real EOS target NOT masked)  PASS
[D06] Chinchilla fit: E=1.5000 A=400.00 B=400.00 (true 1.5/400.0/400.0); N*(from fitted A_hat/B_hat)=4.167e+09 matches grid (4.112e+09) and true-coefficient optimum (4.167e+09); 100x compute -> N* x8.003, D* x12.496  PASS
[D07] spike monitor: main seq flags=[8], decline seq flags=0, spike->spike vs shift->sustained_shift, NaN/Inf-> non_finite  PASS

all pretraining pipeline sanity checks passed
```

## 📚 参考文献

- **Scaling Laws（Kaplan）** — Kaplan et al. (OpenAI), *Scaling Laws for Neural Language Models*, arXiv 2001.08361 (2020).
- **Chinchilla（Hoffmann）** — Hoffmann et al. (DeepMind), *Training Compute-Optimal Large Language Models*, arXiv 2203.15556 (2022).
- **GPT-3** — Brown et al. (OpenAI), *Language Models are Few-Shot Learners*, arXiv 2005.14165 (2020), NeurIPS 2020.
- **Gopher** — Rae et al. (DeepMind), *Scaling Language Models: Methods, Analysis & Insights from Training Gopher*, arXiv 2112.11446 (2021).
- **OPT（训练日志与不稳定性记录）** — Zhang et al. (Meta), *OPT: Open Pre-trained Transformer Language Models*, arXiv 2205.01068 (2022).
- **PaLM（含 loss spike / checkpoint 重启实践）** — Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311 (2022).
- **混合精度训练** — Micikevicius et al. (NVIDIA/Baidu), *Mixed Precision Training*, arXiv 1710.03740 (2017), ICLR 2018.
- **The Pile（数据流水线设计参考）** — Gao et al. (EleutherAI), *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, arXiv 2101.00027 (2020).
- **C4 语料审计（抽取质量问题）** — Dodge et al., *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*, arXiv 2104.08758 (2021).
- **RefinedWeb（网页语料抽取/过滤流水线）** — Penedo et al., *The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*, arXiv 2306.01116 (2023).
- **FineWeb（大规模网页语料 + 消融）** — Penedo et al. (Hugging Face), *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, arXiv 2406.17557 (2024).
- **训练数据去重（近重复对下游影响）** — Lee et al. (Google), *Deduplicating Training Data Makes Language Models Better*, arXiv 2107.06499 (2021), ACL 2022.
- **MinHash / Jaccard 估计原始来源** — Broder, *On the Resemblance and Containment of Documents*, Proceedings of Compression and Complexity of Sequences (1997)（无 arXiv id）.
- **数据混合 / domain reweighting** — Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*, arXiv 2305.10429 (2023), NeurIPS 2023.
- **数据受限 scaling（重复数据对有效 D 的影响）** — Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv 2305.16264 (2023), NeurIPS 2023.
- **Llama 2（含预训练数据/训练配方描述）** — Touvron et al. (Meta), *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288 (2023).
- **Llama 3（含数据流水线/训练 recipe 描述）** — Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783 (2024).
- **DeepSeek-V3（active/total 参数与 MoE 训练实践）** — DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437 (2024).
- **C4 / T5（tokenizer 与语料预处理的经典参考）** — Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, arXiv 1910.10683 (2019), JMLR 2020.
