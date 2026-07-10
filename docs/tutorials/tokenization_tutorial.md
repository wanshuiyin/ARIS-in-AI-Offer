## §0 Pipeline 心智模型 + TL;DR Cheat Sheet

**Tokenizer 不是"一个算法",而是一条流水线。** 面试里大量混淆（"SentencePiece 就是 Unigram""byte-level 是一种分词目标"）都来自把整条流水线压缩成了一个名词。先立好这个心智模型：

```text
（上游：chat template 先把 messages 渲染成文本；registered special 在 API 启用/准入时于进入下面管线前被原子识别/旁路）
原始文本
  → ① normalization       （NFC/NFKC/identity；可能有损！）
  → ② pre-tokenization    （regex 切块 / 空白处理；merge 不跨块边界）
  → ③ 分词模型             （BPE 按 rank 合并 / WordPiece 最长匹配 / Unigram Viterbi）
  → ④ ID mapping          （token → 整数 ID → embedding 行）
  → ⑤ post-processing     （自动补 BOS/EOS/CLS/SEP 这类模板性 token）
decode：ID → token 字符串拼接 →（byte 还原 / ▁→空格）→ 文本（反向，但不保证严格互逆）
```

> 💡 **9 句话搞定 Tokenization** — 一页拿下面试核心要点（详见后文 §1–§7 推导）。

1. **流水线心智模型**：chat template 与 special 识别在上游；tokenizer 主体 = normalization → pre-tokenization → 分词模型 → ID mapping →（BOS/EOS 类）post-processing；decode 反向。**SentencePiece 是框架**（Unigram/BPE 都是它的可选模型），**byte-level 是字母表选择**（不是独立算法）——两组概念不在同一层。

2. **为什么是 subword**：word-level 有 OOV + 词表爆炸，char/byte-level 序列太长（BMP 汉字 1 字 = 3 byte、补充平面 4 byte）；subword 折中——高频词整存、低频词拆碎，配 byte 兜底可做到**任何有效 UTF-8 文本零 OOV**。指标必须说清口径：**fertility**（tokens/word？tokens/char？tokens/byte？）；跨 tokenizer 比较用 **BPB**（bits-per-byte）。

3. **BPE：训练 $\ne$ 编码**。训练 = 每轮统计语料里最高频的相邻 pair、合并、**按顺序记录 merge 表**（局部贪心，非全局最优）；编码 = 按**冻结的 merge rank** 依次合并——既不重新数频率，也不是最长匹配。这是最高频的手写题陷阱。

4. **WordPiece**：merge 依据似然改善（常见教学实现 score $= \frac{f(xy)}{f(x)f(y)}$，取 log 是 PMI-like）；编码用 **longest-match-first**（`##` 接续形式），任一位置匹配失败 → **整个词** 变 `[UNK]`（不是保留成功前缀）。

5. **Unigram 是反向做减法**：从超大 seed 词表出发，EM 估计 piece 概率（E-step 用 forward-backward 汇总**所有** segmentation 的后验，不是只取 Viterbi）+ 按删除损失 prune；推理时才用 Viterbi 取最优切分；天然支持采样 segmentation（subword regularization）。

6. **byte-level 与 SentencePiece 是两条工程路线**：GPT-2 把 256 个 byte 先映射到可打印代理字母表再跑 BPE（`Ġ` = 空格 byte 0x20 的代理字符，常出现在 token 开头是 regex 把空格附给后一块所致）；SentencePiece 用 `▁`（U+2581）把空格符号化，`byte_fallback` 只在普通 piece 覆盖不了时退回 `<0xXX>`——**byte fallback $\ne$ byte-level BPE**。

7. **词表大小是两头账**：embedding $\approx Vd$、不共享输出头再加 $Vd$；V 大 →（其他条件相同时通常）fertility 降 → 序列短 → attention/KV cache 省；但 softmax/embedding 变贵。32k→100k→128k→200k→256k 是**多目标设计点**不是单调时间趋势；报数字要区分 mergeable 数、special 数、`len(tokenizer)`、max-ID+1、embedding 行数——五个量常常不相等。

8. **生产陷阱记四个**：**prefix instability**（`encode("ab")` 未必是 `encode("abc")` 的 token 前缀 → token healing / 流式 / prefix cache 的共同根源）；**数字与空白切分**影响算术与代码；**special token 注入**（用户文本里的 `<|assistant|>` 是否被当 special ID 取决于 API 白名单）；**glitch token**（词表里有、训练中欠曝光，如 SolidGoldMagikarp）。

9. **tokenizer-free 前沿**：ByT5（byte 直入）、MegaByte（固定 byte patch + global/local 分层）、BLT（entropy 驱动的动态 patch）；层次化缓解了 byte 序列过长，但训练效率、推理生态与缓存粒度仍是现实障碍——"tokenizer 已死"言之尚早。

## §1 为什么需要 subword + 怎么度量 tokenizer

### 1.1　word-level 与 char-level 的两难

- **word-level**：词表随语料膨胀（形态丰富语言更炸），任何新词/拼错/罕见实体 → OOV `[UNK]`，信息直接丢失；且"run/running/runs"各占一行 embedding，浪费参数、不共享统计强度。
- **char/byte-level**：byte-level 零 OOV（char-level 词表覆盖不全时仍会 OOV），但序列长度暴涨——英文 1 词 $\approx$ 4–5 字符，BMP 汉字（含扩展 A）1 字 = 3 UTF-8 byte（补充平面、扩展 B 及之后 = 4 byte），attention 的 $O(L^2)$ 和 KV cache 的 $O(L)$ 都直接遭殃；模型还要用宝贵的层数去学"字母拼成词"这种本可白给的结构。
- **subword 折中**：高频词保持整 token（"the"、"人工智能"），低频词拆成统计上可复用的子串（"tokenization" → "token"+"ization"——有时恰好是词素，但三种算法都不保证语言学边界），罕见字符退到 byte。既控制词表规模，又保证覆盖。

**覆盖性（OOV 保证）分三档**，面试常被追问：

1. 传统 WordPiece（BERT 风格）：编码失败 → **整词 `[UNK]`**，最弱；
2. Unigram/SentencePiece：字符覆盖不全（`character_coverage` $\lt 1$）仍会产出 `<unk>`；
3. **完整 256-byte 字母表**（GPT-2 式）或**正确启用 byte fallback** 的 SentencePiece：任何有效 UTF-8 文本都可编码，**强覆盖保证**——零 OOV 不是 subword 的默认福利，是这两种设计换来的。

### 1.2　指标：fertility 必须先说口径

**fertility（分词粒度）**：平均每个"单位"被切成多少 token。文献里狭义 fertility 常特指 tokens/**word**（tokens/char、tokens/byte 更常称压缩率口径），本文取宽义、但说这个词**必须先声明分母**——tokens/**word**、tokens/**char** 还是 tokens/**byte**？中文没有天然空格词界，"tokens/word"依赖外部分词器（不同分词标准结果不同），跨语言比较时优先**同一评测文本**上同时报告：

| 指标 | 定义 | 适用 |
| --- | --- | --- |
| UTF-8 **bytes/token** | 评测文本 byte 数 ÷ token 数 | 跨 tokenizer 最稳定的计量口径（byte 数与 tokenizer 无关；跨语言仍需平行语料+多指标） |
| Unicode **chars/token** | code point 数 ÷ token 数 | 直觉友好，但 emoji/组合字符会骗人 |
| **tokens/word** | token 数 ÷ 外部分词词数 | 仅在声明分词器后可比 |
| UNK rate | `<unk>` 占比 | 覆盖性检查 |

工程上还要看**端到端**：同一语料的总 token 数（直接决定训练/推理成本）、吞吐、首 token 延迟——tokenizer 的好坏最终是系统问题（§4.2）。

### 1.3　BPB：跨 tokenizer 可比的 loss 口径（需统一 byte 流与评分协议）

不同 tokenizer 的 **loss/token 和 perplexity 不可直接比较**——分母（token 数）本身就不同：即使总 NLL 完全相同，仅 token 数不同就能让 loss/token 差出数倍；实际比较时还叠加模型质量本身的差异，两种效应混在一起。公平做法是把总负对数似然摊到**同一份文本的 UTF-8 byte 数** $B$ 上：

$$\mathrm{BPB} = \frac{-\sum_i \log p(t_i \mid t_{\lt i})}{B \cdot \ln 2}\qquad(\log\ \text{取自然对数时除以}\ \ln 2\ \text{换算成 bits}).$$

数值例子（§7 的 [D07] 可执行验证）：同一段文本、总 NLL 同为 4.0 nats、$B=8$ bytes，tokenizer A 切 2 个 token、B 切 4 个 → loss/token 分别为 **2.0 和 1.0**（差一倍！），BPB 却同为 $\frac{4.0}{8\ln 2}\approx 0.7213$——**BPB 相同，loss/token 的"差距"纯属口径幻觉**。

> ⚠️ **BPB 比较的前提：两边实际计分的是同一份 post-normalization byte 流**
> 若某 tokenizer 带有损 normalization（NFKC 改写了字符），两边模型实际计分的文本已经不同——仅"输入同一 raw text"不够，必须保证送进模型的规范化后 byte 流完全一致，并用这份流的 byte 数作 $B$；BOS/EOS 处理、context 切分等评分协议也要对齐。BPB 是常用且可比的口径，不是无条件的"唯一公平"。

## §2 三大算法：BPE / WordPiece / Unigram

**本节铁律：训练目标和编码算法分开讲。** 三个算法各有"训练时学什么"和"编码时怎么用"两半，混在一起讲是最常见的失分方式。

### 2.1　BPE：最高频 pair 迭代合并

BPE 原本是 1994 年的字节压缩算法（Gage, 1994），Sennrich, Haddow & Birch (2015, arXiv 1508.07909, ACL 2016) 把它引入 NMT 做 subword 切分，从此成为 GPT 系、Llama 系、Qwen 系的主流选择。

**训练算法**（自底向上做加法）：

1. 把语料切成初始符号序列（字符或 byte），统计每个"词"的频次；
2. 统计所有**相邻符号 pair** 的加权频次；
3. 选**最高频**的 pair 合并成新符号，**把这条 merge 按顺序记入 merge 表**；
4. 重复 2–3，直到达到目标 merge 数 / 词表大小。

三个必须挂在嘴边的细节：

- **局部贪心**：每轮只选当前最高频 pair，**不保证**最终序列长度全局最优——它是压缩启发式，不是最优化算法；
- **tie-breaking 必须确定性**：频次并列时要有固定规则（如字典序），否则同一语料训出不同 merge 表，复现性崩坏；
- **重叠计数**：`"aaaa"` 里 pair `(a,a)` 按滑窗数出现 3 次，但实际非重叠替换只能换 2 次——计数口径和替换语义要分开说。

**手算例子**：语料 `{"abab"×3, "abac"×1}`，初始切分为字符。第一轮 pair 计数：

| pair | 出现次数 | 加权频次 |
| --- | --- | --- |
| $(a,b)$ | "abab" 内 2 次 ×3 + "abac" 内 1 次 ×1 | $2\times 3+1=\mathbf{7}$ |
| $(b,a)$ | "abab" 内 1 次 ×3 + "abac" 内 1 次 ×1 | $3+1=4$ |
| $(a,c)$ | "abac" 内 1 次 ×1 | $1$ |

→ **merge₁ $=(a,b)\to$ `ab`**（频次 7）。第二轮：词变成 `ab ab`×3、`ab a c`×1，pair 计数 $(ab,ab)=3,\ (ab,a)=1,\ (a,c)=1$ → merge₂ $=(ab,ab)\to$ `abab`。merge 表 = `[(a,b), (ab,ab), ...]`——**顺序就是 rank，rank 就是编码时的优先级**。

**编码算法**（这是和训练完全不同的另一段程序）：给定冻结的 merge 表，把输入切成初始符号后，**反复找当前序列中 rank 最小（最早学到）的可用 pair 合并**，直到没有可用 merge：

$$\text{encode}(x):\quad \text{while } \exists\ \text{pair} \in \text{merge 表}:\ \text{合并 rank 最小者（最左优先）}.$$

例：merge 表 `[(a,b), (b,c)]`（rank 0、1），`encode("abc")`：$(a,b)$ 和 $(b,c)$ 都可用，**rank 0 的 $(a,b)$ 赢** → `["ab","c"]`——即使你觉得 `(b,c)` "也很频繁"。编码阶段**没有频率统计这回事**。

> ⚠️ **手写 BPE 的第一陷阱：把 encode 写成"每次合并当前输入中最高频的 pair"**
> 训练时统计频率，编码时只查 rank。写错的后果：同一字符串在不同上下文里编码不同、与训练时的切分分布脱节、和官方实现对不上。§7 的 [D02] 用 `assert bpe_encode("abc", [("a","b"),("b","c")]) == ["ab","c"]` 把这个语义钉死。

**decode**：toy 语义下把 token 字符串直接拼接（byte-level 则再把 byte 还原成 UTF-8）；真实系统还要处理 `▁`/`@@` 这类边界标记与 byte 代理的还原。BPE 的 decode 仍是三者中最简单的——merge 出来的 token 天然是子串拼接。

### 2.2　WordPiece：似然驱动的 merge + 最长匹配编码

WordPiece（Schuster & Nakajima, ICASSP 2012；BERT 沿用，Devlin et al., 2018, arXiv 1810.04805）与 BPE 同为自底向上 merge，**差别在选 pair 的评分**。

**训练目标**：原始 WordPiece 以**语言模型似然改善**为目标选 merge；常见教学实现（如 HF tokenizers 文档）用的 score 是

$$\mathrm{score}(x,y)=\frac{f(xy)}{f(x)\,f(y)},$$

取 log 后 $\log f(xy)-\log f(x)-\log f(y)$，与 PMI 只差一个依赖总计数口径的常数——所以叫它 **PMI-like** 是准确的，直接说"就是 PMI"或"这是所有历史 WordPiece 的唯一精确定义"都过强（Google 的历史训练实现并未完整规范化公开）。

**手算例子**（BPE 与 WordPiece 选择不同 pair）：

| pair | $f(xy)$ | $f(x)$ | $f(y)$ | BPE 依据（频次） | WordPiece score |
| --- | --- | --- | --- | --- | --- |
| $(a,b)$ | 6 | 10 | 10 | **6**（BPE 选它） | $6/100=0.06$ |
| $(q,u)$ | 2 | 2 | 10 | 2 | $2/20=\mathbf{0.1}$（WordPiece 选它） |

直觉：`q` 几乎只出现在 `qu` 里——归一化后的"黏着度"高，本例 WordPiece 因此选它；而 `a`、`b` 大量独立出现，`ab` 频次高只是"底数大"。**BPE 看绝对频次，WordPiece 看黏着度**（该 score 是 surrogate，不宜直接读成总似然增益）。

**编码算法**：与训练无关，用 **longest-match-first（最长匹配优先）**贪心：

1. 从词首开始，找词表中能匹配的**最长**前缀（词首用普通形式，非词首用 `##` 接续形式，如 `##ing`）；
2. 匹配上则前进，从剩余部分继续；
3. **任一位置找不到任何匹配 → 整个词输出 `[UNK]`**——不是保留已成功的前缀再补一个 `[UNK]`。

例（§7 脚本 [D03] 可执行验证）：词表 `{"a", "##b", "##bc", "[UNK]"}` 时 `wp("abc") = ["a", "##bc"]`（`##bc` 比 `##b` 长，赢）；把 `##bc` 移出词表且不加 `##c`，则 `wp("abc") = ["[UNK]"]`——**整词失败**。

> 💡 **`##` 是"我不是词首"的记号**
> WordPiece 依赖 pre-tokenization 先按空格/标点切出"词"，`##` 标记 subword 在词内的接续位置，decode 时据此去掉。对中日韩文本，BERT pipeline 的常见做法是 BasicTokenizer 把 Han 表意字逐字隔开（假名/谚文并不逐字切，韩语本身也有空格）——这是实现策略而非算法硬约束，但也正是 SentencePiece 想摆脱的那类语言相关预处理（§3.2）。

### 2.3　Unigram：概率模型 + 自顶向下剪枝

Unigram LM 分词（Kudo, 2018, arXiv 1804.10959, ACL 2018）换了范式：**不是学 merge 规则，而是学一个"piece 概率表"**，切分是概率模型下的推断。

**模型**：词表 $V$ 中每个 piece $z_i$ 有概率 $p(z_i)$，$\sum_{z\in V}p(z)=1$；一个切分 $z=(z_1,\dots,z_k)$ 的概率是独立乘积 $P(z)=\prod_i p(z_i)$；**一个字符串的（边际）概率是对所有合法切分求和**：

$$P(x)=\sum_{z\in S(x)}\prod_{i} p(z_i),\qquad \text{训练目标}\ \max_\theta \sum_{x\in\text{语料}}\log P(x).$$

**训练**（自顶向下做减法，与 BPE 方向相反）：

1. 从**过大的 seed 词表**出发（如高频子串 + 全部单字符）；
2. **EM 估计概率**：E-step 用 forward-backward 在 $S(x)$（所有切分）上汇总每个 piece 的**后验期望计数**——注意是对全部切分积分，**不是只取 Viterbi 最优切分**（那是 hard-EM，写错方向是高频扣分点）；M-step 归一化重估 $p(z)$；
3. **剪枝**：对每个 piece 计算"删掉它后语料似然的近似损失"，删掉损失最小的一批（如 20%），回到 2；单字符、special、byte pieces 受**保护**不参与删除；
4. 收敛到目标词表大小。

**推理（编码）**：确定性分词才用 **Viterbi** 取最优切分：

$$z^{*}=\arg\max_{z\in S(x)}\sum_i \log p(z_i).$$

**手算例子**（§7 的 [D04] 可执行验证）：概率表

$$p(\text{a})=0.15,\quad p(\text{b})=0.15,\quad p(\text{ab})=0.40,\quad p(\text{aba})=0.20,\quad p(\text{bab})=0.10.$$

对 `"abab"` 做两条 DP（$f_i$ = 前缀 $x_{1:i}$ 的**全切分**概率和；$v_i$ = 最优单一切分概率）：

| $i$ | 前缀 | forward $f_i=\sum_j f_j\,p(x_{j+1:i})$ | Viterbi $v_i$ | $v_i$ 最优路径 |
| --- | --- | --- | --- | --- |
| 0 | $\varepsilon$ | 1 | 1 | — |
| 1 | `a` | $0.15$ | $0.15$ | `a` |
| 2 | `ab` | $0.40+0.15\times 0.15=0.4225$ | $\max(0.40,\ 0.0225)=0.40$ | `ab` |
| 3 | `aba` | $0.20+0.4225\times 0.15=0.263375$ | $\max(0.20,\ 0.06)=0.20$ | `aba` |
| 4 | `abab` | $0.015+0.169+0.0395\ldots=\mathbf{0.22350625}$ | $\max(0.015,\ \mathbf{0.16},\ 0.03)=0.16$ | `ab`·`ab` |

其中 $f_4 = f_1\,p(\text{bab}) + f_2\,p(\text{ab}) + f_3\,p(\text{b}) = 0.15\times 0.10 + 0.4225\times 0.40 + 0.263375\times 0.15$。两个必背读数：**Viterbi 最优切分 `["ab","ab"]`（概率 0.16）**；**边际概率 0.2235 $\geq$ 最优路径概率**（对所有切分求和必然不小于单条路径——这正是 E-step 基于全切分统计（forward-backward）而非单条 Viterbi 路径的原因）。

**subword regularization 与 BPE-dropout**（两个随机化机制，别混写）：

- **Unigram 采样**：训练 LLM 时按切分后验（或 n-best 近似）**采样** segmentation 而非固定 Viterbi，采样分布 $q(z)\propto P(z)^{\alpha}$，`alpha` 是**逆温度/平滑系数**（$\alpha$ 越大越接近 Viterbi、越小越随机，$T=1/\alpha$）——同一个词每个 epoch 可能切法不同，相当于数据增强 + 让模型见过碎片化拼法；
- **BPE-dropout**（Provilkov et al., 2019, arXiv 1910.13267, ACL 2020）：BPE 本身无概率模型，做法是编码时**以概率 $p$ 随机跳过（阻断）部分 merge**，制造更碎的切分。目的相似（鲁棒性），**机制完全不同**（采样后验 vs 阻断 merge）。

### 2.4　三算法对照表 + 覆盖性

| 维度 | BPE | WordPiece | Unigram |
| --- | --- | --- | --- |
| 训练方向 | 自底向上**加法**（merge） | 自底向上加法（merge） | 自顶向下**减法**（prune） |
| 训练目标 | 最高频 pair（贪心压缩） | 似然改善 / PMI-like score | 语料边际似然（EM） |
| 训练产物 | **有序 merge 表** | 词表（含 `##` 形式） | piece **概率表** |
| 编码算法 | 按冻结 rank 依次合并 | longest-match-first | Viterbi（可采样） |
| OOV 行为 | byte-level 变体零 OOV；字符级可 OOV | 失败 → **整词 `[UNK]`** | 覆盖不全 → `<unk>`（可配 byte fallback） |
| 随机化 | BPE-dropout（阻断 merge） | —（标准实现无） | 原生（采样 segmentation，`alpha` 温度） |
| 代表模型 | GPT 系 / Llama / Qwen | BERT 系 | T5 / ALBERT（SentencePiece-Unigram） |

> ✅ **面试答"三者区别"的骨架**
> 先说训练：BPE 数频次、WordPiece 算似然收益、Unigram 从大词表 EM+剪枝；再说编码：rank 合并 / 最长匹配 / Viterbi——**训练和编码是两套独立程序**；最后补一刀覆盖性与随机化（`[UNK]` 语义、Unigram 原生可采样）。三段说完，深度已经超过大多数候选人。

## §3 Byte-level 与 SentencePiece 工程

### 3.1　GPT-2 byte-level BPE：代理字母表 + regex pre-tokenization

GPT-2（Radford et al., 2019）的实际顺序是：**原始文本 → regex pre-tokenization 切块 → 每块 UTF-8 编码 → `bytes_to_unicode` 映射到 256 个可打印 Unicode 字符的代理字母表 → 块内跑普通 BPE**（先切块、再 byte 映射——顺序反了会改变切块结果）。原因很工程：直接在原始 byte 上跑，会有大量不可打印/控制字符，字符串处理和调试都痛苦；`bytes_to_unicode` 把可打印 byte 保持原样、其余 68 个 byte 平移到 `chr(256+n)`，构成一个 256↔256 的**双射**（可逆）。

- 这层映射只是**实现桥梁**，不意味着"每个最终 token 只含一个 byte"——merge 之后一个 token 可对应很多 byte；
- 空格 byte `0x20` 被映射为 **`Ġ`**（U+0120）、换行 `0x0A` 映射为 `Ċ`——所以可视化 GPT 系 token 时满屏的 `Ġ` 是空格 byte 的代理写法——`vocab.json` 里确实存着大量 `Ġthe` 这样的字符串条目，但那个 `Ġ` 是内部代理字符，不是原文中的 U+0120；
- **regex pre-tokenization 先切块，BPE merge 不跨块边界**。GPT-2 的 regex 按类别切（此处只示意关键片段，完整 regex 随版本演化、不建议默写）：

```text
's|'t|'re|'ve|'m|'ll|'d    # 英文缩略（contraction）单独成块
 ?\p{L}+                   # （可带一个前导空格的）字母串 —— 这是 Ġword 的来源
 ?\p{N}+                   # 数字串：GPT-2 不限长度（数字坑的根源，§5.1）
 ?[^\s\p{L}\p{N}]+         # 标点/符号块
\s+(?!\S)|\s+              # 空白（lookahead 分支影响多空格的归属，此处从简）
```

后续 encoding（cl100k_base / o200k_base / Llama 3 风格）的关键改动之一是把数字分支改成 **`\p{N}{1,3}`（至多 3 位一块）**——注意"至多 3 位 pre-token"不等于"最终 token 一定 3 位"：BPE 仍可能在块内继续拆（`999` → `9`+`99`），只是**不能跨块合并**。

### 3.2　SentencePiece：框架，不是算法

**SentencePiece（Kudo & Richardson, 2018, arXiv 1808.06226, EMNLP 2018 demo）是一个 tokenizer 训练/编码框架**，`model_type` 可选 **unigram / bpe / char / word**——它不是 Unigram 的别名（"SentencePiece 分词"这种说法在面试里要主动澄清用的是哪个模型）。关键工程特性：

- **"无需 pre-tokenization"的准确含义**：它不要求先用语言相关的 word tokenizer 切词，能直接从原始句子训练（对中日韩关键）；但 normalization、空白转义和可配置 split 规则**仍然存在**，说"完全没有预处理"是错的；
- **`▁`（U+2581，LOWER ONE EIGHTH BLOCK）**：把空格提升为普通符号参与建模，decode 时 `▁`→空格。默认还会加 dummy prefix（句首补一个 `▁`）、可能折叠连续空白——**这些默认配置意味着不能仅凭 `▁` 就宣称无损可逆**；
- **normalizer 默认 `nmt_nfkc`**：一套 NFKC 派生的规则集（不完全等于任何语言运行库的标准 NFKC）；很多现代模型显式改用 identity 或自定义规则以保住原文；
- **`byte_fallback=true`**：词表保留 256 个形如 `<0xXX>` 的 byte pieces；normalization 后无法被普通 piece 覆盖的 span 转成 UTF-8 bytes 编码，decode 时把连续 byte pieces 还原成字符。

> ⚠️ **byte fallback $\ne$ byte-level BPE**
> SentencePiece 的 byte fallback 是**兜底**：只在普通 Unicode pieces 覆盖不了时才退回 `<0xXX>`；GPT-2 式 byte-level BPE 是**地基**：从第一步就在完整 256-byte 字母表上学 merge。且 byte fallback 救不回被有损 normalization 改写的原文——覆盖性和保真性是两个独立问题（§7 的 [D05] 用 NFKC 例子把两者切开）。

> 💡 **`Ġ` vs `▁`：两个"空格替身"一句话分清**
> `Ġ` 是 byte `0x20` 经 GPT-2 代理字母表映射后的**代理字符**（byte-level 路线）；`▁` 是 SentencePiece 把空格**显式符号化**的 U+2581（框架路线）。来源机制不同，但都在说同一件事：空格信息必须显式保存，decode 才能复原。

### 3.3　Unicode 陷阱与"可逆"的精确条件

**用户眼里的一个"字"，可能是多个 code point、更多 byte**——这是所有 offset/长度 bug 的祖宗：

- **组合字符**：`é` 可以是单 code point U+00E9，也可以是 `e`+U+0301（组合重音）——NFC 把后者规范合成为前者（单码点形式保持不变）；不做 normalization 时，两种写法在覆盖完备且保真的 tokenizer 下产生**不同 token 序列**；
- **兼容字符**：NFKC 把全角 `Ａ`→`A`、`①`→`1`（方便检索，但**有损**——原文写法永久丢失）；
- **emoji ZWJ 序列**：👨‍👩‍👧 是 5 个 code point（3 个 emoji + 2 个 U+200D 零宽连接符）= 18 个 UTF-8 byte，却是 1 个 grapheme cluster（用户感知的"1 个字符"）；
- **variation selector**（U+FE0F 等）、零宽空格 U+200B、不可见控制符：肉眼不可见，却真实影响 token 序列（前提是 normalizer 保留它们且词表覆盖；覆盖不足则塌成 `<unk>`）——安全审计和去重要同时检查 raw 与 normalized 两层。

**双向可逆性的条件**（面试高频追问）：

- `decode(encode(x)) == x` **并非默认成立**。本质要求是**整条 encode→decode 管线在目标输入域上信息保真**：normalizer/pre-tokenizer 不丢信息、无 `<unk>` 吞字（byte 字母表或 byte fallback 兜底）、special/post-processing 行为对称、decoder 不做 cleanup"美化"。工程上最常用的一组**充分配置**：identity normalization（NFKC 一步就毁）+ 空白保真（dummy prefix / 连续空白折叠关掉）+ 完整覆盖 + 关闭 decoder cleanup——注意这是常用配置而非逻辑充要条件（如 dummy prefix 若能被 decoder 对称移除，也不破坏往返；反过来还要求 `▁` 这类内部标记/special surface 的转义是单射，否则原文里的字面 `▁` 会与空格转义碰撞）；
- `encode(decode(ids)) == ids` **一般不保证**：多个非规范 token 序列可以 decode 出同一字符串（`["a","b"]` 和 `["ab"]` 都还原 `"ab"`），re-encode 只会给出规范切分。这种 ID 往返不保真（encode∘decode 是 canonicalization——在典型规范化配置下它通常是幂等的）与 §5.4 的 prefix instability 是相关但不同的两个性质——后者才是 token healing 的直接根源。

## §4 训练语料、词表大小、多语言与扩词表

### 4.1　tokenizer 的训练语料设计常比"选哪个算法"更重要

同一个 BPE 算法，喂不同语料训出的 tokenizer，fertility 可以差出数倍。实际要设计的旋钮：

- **语言配比**：tokenizer 语料里某语言占比低 → 该语言 merge 少 → fertility 高 → 同样内容烧更多 token（成本、上下文窗口双重惩罚）；
- **代码/数学占比**：决定缩进、运算符、常见标识符是否有整 token（§5.2）;
- **去重与清洗**：重复模板文本会把无意义长串顶进词表（glitch token 温床，§5.3）；
- **normalization 一致性**：训 tokenizer 和训模型必须用同一套 normalization，否则线上线下切分不一致；
- **训练器旋钮按实现分**：`character_coverage`/special 槽位是 SentencePiece 式旋钮，`min_frequency` 是 HF tokenizers 式 BPE/WordPiece trainer 旋钮；byte-level BPE 的基础覆盖固定为 256 bytes——别把旋钮笼统安到所有训练器头上。

### 4.2　词表大小：一笔两头的账

设 hidden size 为 $d$、词表大小为 $V$：**输入 embedding $\approx Vd$ 参数；输入输出不共享（untied）时再加一份 $Vd$ 的输出投影，合计 $\approx 2Vd$**。共享（tied）省参数，但输出端 full softmax 的**计算**规模仍与 $Vd$ 相关。具体感受量级（$d=4096$）：$V=32{,}000$ 时 $Vd\approx 1.3$ 亿参数；$V=128{,}256$ 时 $Vd\approx 5.3$ 亿——untied 再翻倍。

另一头：**其他条件相同、词表近似增量扩展时，V 越大、merge 越多、fertility 通常越低 → 同一文本 token 数越少** → attention（$\propto L^2$）、激活、KV cache（$\propto L$）、生成步数全线受益。两头方向相反，所以：

> 🎯 **判断词表大小不看单一指标**
> 端到端比较 tokens/s、首 token 延迟、显存占用和任务质量，而不是只看"压缩率"或只看"embedding 参数量"。32k→100k→128k→200k→256k **不是行业单调时间趋势**，而是语言覆盖、代码占比、模型宽度（$d$ 大才养得起大 $V$）、部署预算共同决定的**多种设计点**。

### 4.3　主流词表事实表：五个数字别混

**同一个 tokenizer 有五个容易混淆的量**：普通（mergeable）token 数、special token 数、`len(tokenizer)`、最大 token ID+1（可以有空洞）、模型 embedding 行数（可为对齐 padding 到更大值）。它们在下表里**并不总相等**。两条换算：**有效条目数 = mergeable 数 + special 数**；tiktoken 的 `n_vocab` = **max ID + 1**（ID 有空洞时它大于有效条目数），HF 的 `len(tokenizer)` 还受 added-token 语义影响：

| Tokenizer | 普通/mergeable token | special token（公开） | tokenizer 有效条目 | 模型 vocab_size（embedding 行） |
| --- | --- | --- | --- | --- |
| bert-base-uncased（WordPiece） | 30,517（含 ~1k `[unused]` 占位） | 5（`[PAD]` `[UNK]` `[CLS]` `[SEP]` `[MASK]`） | 30,522 | 30,522 |
| bert-base-cased | 28,991（同口径） | 5 | 28,996 | 28,996 |
| GPT-2（byte-level BPE） | 50,256（256 byte 符号 + 约 5 万 merge） | 1（end-of-text 记号） | 50,257 | 50,257 |
| cl100k_base（tiktoken） | 100,256 | 5 | 100,261；**max ID+1 = 100,277（ID 有空洞）** | —（API 模型未公开） |
| o200k_base（tiktoken） | 199,998 | 2 | 200,000；**max ID+1 = 200,019** | — |
| Llama 2（SentencePiece BPE） | 31,997（含 256 个 byte pieces） | 3 meta pieces（`<unk>` UNKNOWN；`<s>` `</s>` CONTROL） | 32,000 | 32,000 |
| Llama 3（tiktoken 风格 byte-level BPE） | 128,000 | 256（保留 special 槽） | 128,256 | 128,256 |
| Qwen2 / Qwen2.5 | 151,643 | 3 核心对话标记 | 151,646 | **151,936**（0.5B–3B）/ **152,064**（7B+）——随 checkpoint 尺寸 padding，两代同规律 |
| Gemma 1 / Gemma 2（SentencePiece） | — | — | 256,000（配置值） | 256,000 |
| Gemma 3 | — | — | 262,144（$=2^{18}$，base tokenizer） | 262,208（4B/12B/27B 文本配置；1B 口径按 checkpoint 核对） |

必须能展开讲的几行（special 的字面写法含竖线、与表格分隔符冲突，故表中从略：GPT-2 唯一 special 是 `<|endoftext|>`；Qwen 的 3 核心是 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>`）：

- **cl100k/o200k 的"词表大小"有两种答案**：有效条目数（100,261 / 200,000）和 max ID+1（100,277 / 200,019）——ID 空间有空洞，`n_vocab` 不等于"实际可用 token 数"。说"GPT-4o 词表 200,019"和"o200k 有 20 万个有效 token"描述的是**不同的量**；GPT-4o 用 o200k 系 encoding 是合理表述，但公开的 `o200k_base` 不一定含服务端 chat 协议的全部隐藏处理，模型↔encoding 映射应按版本冻结核对；
- **Llama 3 不是 Llama 2 扩容**：Llama 2 用 SentencePiece BPE（32,000）；Llama 3 换成 Meta 自训、以 tiktoken 风格实现加载的 byte-level BPE（128,000 普通 + 256 保留 special = 128,256）。"tiktoken-based"说的是**实现接口与 byte-level BPE 形式**，不是复用 OpenAI 的 merge 词表——它不是 cl100k_base 的复制品；
- **Qwen 的三组数**：基础词表 151,643、`len(tokenizer)` 151,646（+3 核心 special）、embedding 行 **151,936（0.5B–3B）/ 152,064（7B 及以上）**——embedding 行数是模型侧按 checkpoint 尺寸 padding 的，Qwen2 与 Qwen2.5 同规律，以具体 checkpoint 的 `config.vocab_size` 为准。多模态变体（如视觉版）还会加视觉 special token，"Qwen 词表是多少"必须先问口径；
- **"BERT 是 30k"只是近似**：uncased 30,522 / cased 28,996；**"Gemma 永远 256k"不准确**：Gemma 3 base tokenizer 是 262,144（$2^{18}$），而 4B/12B/27B 文本配置的 embedding 行是 262,208——又一例"tokenizer 大小 $\ne$ embedding 行数"。

### 4.4　多语言：中文不是"按字还是按词"的二选一

现代 BPE/Unigram 词表里，中文 token 同时混着**单个汉字、多字词（高频如"我们""人工智能"）、byte fallback 碎片（罕见字被拆成 3–4 个 `<0xXX>`——BMP 汉字 3 byte、补充平面 4 byte）、标点组合**——"中文按字切"是过时印象。评价一个 tokenizer 的中文效率，回到 §1.2：同一语料上报 bytes/token、chars/token，要报 tokens/word 就先声明外部分词器（不同分词标准下"词"数差异可观）。跨语言公平性还有一层：tokenizer 语料配比显著影响各语言 fertility（§4.1）——同一个模型，低资源语言的用户实际按 token 计费更贵、可用上下文更短。

### 4.5　扩词表：改 tokenizer 文件只是第一步

给已有 LLM 扩词表（领域术语、新语言、agent 控制符），**光改 tokenizer 不会让模型理解新 token**，完整清单：

1. **保住旧世界**：旧 token ID、旧 merge rank、normalization 全部冻结不动——若新增**普通** token 参与最长匹配/merge，它可能改变旧文本的切分（backward compatibility 崩坏）；控制符类新 token 走 **special 通道**（原子匹配、明确边界）；领域词/新语言 piece 则是普通 lexical token——本来就该参与正常切分，代价是旧文本切分可能改变，用回归测试圈定兼容域，而不是一律标成 special（会被 `skip_special_tokens` 吞掉、还扩大注入面）；
2. **resize 两端**：embedding 和 LM head 都要扩行；权重 tied 时确认扩容后仍共享同一参数；optimizer state（Adam 的 $m,v$）同步扩；
3. **初始化**：普通 lexical token 用其 surface 旧切分 embedding 的均值；special/控制符（surface 无有意义旧切分）用描述文本（description proxy）旧切分的均值——都是常用启发式基线（动机是让新行起点接近旧切分语义）；untied 时输出行/bias 需单独初始化；
4. **继续训练**：新 embedding 行需要梯度喂养，只 mean-init 不训练 = 半个 glitch token；
5. **回归测试**：旧测试集 re-encode 必须逐 ID 一致、special token 行为不变、round-trip 行为与扩表前一致（基线本来可逆的样本才断言等于原文）——§7 脚本的 [D08] 是这套 regression test 的最小可执行子集（覆盖 ID/special/tied，未含 decoder round-trip）。

## §5 生产陷阱与推理栈交互

### 5.1　数字：三代切法与算术

三代数字 pre-tokenization（§3.1 的 regex 分支）：

| 方案 | 代表 | 行为 | 后果 |
| --- | --- | --- | --- |
| 数字串不限长 | GPT-2 | `12345` 交给 BPE 自由 merge | 组合不规则（`123`+`45`、`1`+`2345`……随频率而定），位对齐混乱 |
| 逐位切分 | Llama 2（SentencePiece `split_digits`） | 每个数字一个 token | 位对齐干净，序列变长 |
| 至多 3 位一块 | cl100k / o200k / Llama 3（`\p{N}{1,3}`） | 从左到右贪心取至多 3 位（`12345`→`123`+`45`，**不是**右对齐千分位） | 折中；**块内 BPE 仍可再拆**，只是不跨块合并 |

数字切法影响**位对齐**（个位对个位才好学竖式加法）与**跨 token 进位**——但算术能力不能单因果归到 tokenizer：训练数据里的算术占比、位置编码、是否用 CoT/工具同样关键。面试答"为什么 LLM 算术差"只怪 tokenizer 是半对；只字不提 tokenizer 也是半对。

### 5.2　代码：空白就是语法

代码语料训练出的 tokenizer 通常学到**多空格 run、换行+缩进组合、常见符号串**的 merge——但"是否存在独立的 4 空格 token"取决于具体 merge 表，不能想当然；tab 与空格更不能默认等价（两种缩进 = 两套完全不同的 token 序列）。对 Python 这类缩进敏感语言，tokenizer 对空白的处理直接影响：补全模型的缩进正确率、同一段代码的 token 成本（上下文窗口装得下几个文件）、以及 diff/编辑类任务的边界稳定性。

### 5.3　glitch token：词表与语料的错配

> ⚠️ **SolidGoldMagikarp（经典 glitch token，一段讲完）**
> **定义**：词表中存在、但模型训练时几乎从未以该 ID 出现的 token。**成因（主流解释）**：tokenizer 训练语料与模型训练语料不一致（如某 Reddit 用户名进了 BPE 词表、正文却被清洗掉）→ 该行 embedding/输出行从未被充分学习。**症状**：模型无法复述该词、输出乱码或不可预期行为。**诊断**：扫描 embedding 范数异常值、让模型 repeat 候选 token。**边界**：它是"词表-语料错配"的个案，不是所有 GPT 模型或 byte-level tokenizer 的普遍故障。

### 5.4　prefix instability 与 token healing

**比"结尾空格问题"更一般的规律**：$\text{encode}(s_1)$ **未必是** $\text{encode}(s_1 + s_2)$ 的 token 前缀——追加字符可能触发跨旧边界的 merge。最小例子（§7 的 [D06]）：merge 表 `[a+b, ab+c]` 下 `encode("ab")=["ab"]`，但 `encode("abc")=["abc"]`——前者不是后者的前缀。三个受害者：

- **补全类 prompt**：用户输入以 `"The qu"` 结尾时，结尾的未完成词可能被切出训练分布中罕见的"悬空"边界（具体切法依词表/前导空格/版本而定）——分布错位，补全质量下降；
- **token healing**（补救术）：回退最后一个（或若干可能未完成的）prompt token，把它们的字符串还给解码器，**约束后续生成的 decode 前缀复现被回退的后缀**（单 token 回退时即"首 token 以该后缀开头"；多 token 回退需要 trie/FSM 级的序列约束）。注意它是**推理侧包装技术**，不是所有模型 API 的标准能力；
- **流式输入 / prefix cache**：见 §5.7。

### 5.5　special token：训练侧细节 + 注入边界

**训练侧**（坑比"忘加 BOS"深得多）：

- BOS/EOS/PAD 各司其职；**attention mask 和 loss mask 是两回事**——PAD 通常两个都要屏蔽，对话数据常做 assistant-only loss（只在助手回复段算 loss）；
- **把 PAD 设成 EOS 本身不一定错**，真正的经典事故是随后按 `input_ids == pad_id` 无差别构造 mask，把真实 EOS 的 loss（甚至 attention）一起屏蔽 → 该数据上的 EOS 监督被移除、停止能力被显著削弱（从头训练时尤其致命）——mask 应按真实 padding 位置/序列长度构造；
- **packing**（多文档拼一条序列）要在文档边界正确放 EOS/BOS 并处理跨文档 attention；chat template 套模板时警惕**重复添加 BOS**（模板带一个、tokenizer `add_special_tokens` 又加一个）。

**注入边界**：用户文本里出现字面量 `<|assistant|>` 时，它被编码成 special ID 还是普通文本，**取决于 API**——前提是该字符串已在 encoding 中注册为 special（`allowed_special` 只做准入、不会凭空造 special ID）；tiktoken 对"已注册但未准入"的 special 默认报错（放宽 `disallowed_special` 才按普通文本切），HF 接口则看 special token 配置与模板转义。这既是 correctness 问题（模板拼装错乱），也是 **prompt injection 边界**：渲染用户输入前必须明确"字面量到底会不会变成控制符"，并写测试钉死。

### 5.6　offset mapping：多套坐标系

抽取式 QA、NER、代码编辑器、高亮工具都要回答"这个 token 对应原文哪一段"。坑在于存在**多套坐标**：normalized text 的字符偏移、原始文本的 Unicode code point 偏移、原始文本的 UTF-8 byte 偏移（一些运行时还有 UTF-16 code unit 与 grapheme cluster 口径）——NFKC 这类有损 normalization 会改写字符（`ﬃ`→`ffi` 是 1→3 个 code point；`Ａ`→`A` 是字符值与 byte 长度变、code point 数不变），没有 alignment map 时 normalized 偏移就**对不回**原文；emoji/组合字符再让 code point 与"用户感知字符"脱节（§3.3）。注意 NFC 也不保证 code-point 保长（`e`+U+0301 → `é` 是 2→1）。工程解法：优先 identity/近保真 normalization、保留 normalization 的 **alignment map**（可处理一对多/多对一 span）、要求 tokenizer 返回 offset mapping（如 fast tokenizer 的 `return_offsets_mapping`），并在评测里同时核对 byte 与 code point 两套口径。

### 5.7　推理栈：哪些组件要求 tokenizer 对齐

| 组件 | 对 tokenizer 的要求 | 原因 |
| --- | --- | --- |
| **speculative decoding**（经典逐 token 拒绝采样） | draft 与 target **同一可对齐 token 支持** | 逐 token 比较 $p_{\text{target}}$ 与 $p_{\text{draft}}$，词表不同则概率不可比；异 tokenizer 需 retokenization/字符串级桥接（已非最基础算法） |
| **logit 蒸馏**（逐位置 KL） | 词表 **且** 位置都对齐 | 每个位置的分布支撑必须相同；sequence-level 蒸馏与字符级目标可跨 tokenizer，hidden-state matching 需先做 span 对齐/pooling |
| **KV / prefix cache** | 同一字符串 → **同一 token-ID 前缀** | 命中按精确 ID 前缀；tokenizer 改动本身不"抽象地破坏 cache"，真正的失效条件是同一字符串变成不同 ID 序列、chat template 变了、或模型权重/位置状态变了 |
| **BPB 评测**（§1.3） | 同一份 byte 流口径 | 有损 normalization 先统一，否则比较无意义 |

### 5.8　tokenizer 版本化：只存 vocab.json 是复现事故的开始

要**精确复现输入 ID**，必须一起版本化：normalization 配置、pre-tokenization regex、merge 表/词表文件、special-token 映射与 added-token 元数据（`lstrip/rstrip/normalized` 等）、post-processor/decoder 配置、chat template、库与实现版本，以及以上所有 artifact 的**哈希**。只保存一个 `vocab.json`（丢了 merge 顺序与切块规则）→ 半年后同一文本切出不同 ID，训练/评测/缓存全线静默漂移。把 tokenizer 当模型权重一样对待：进版本管理、进实验记录、变更走 regression test（§4.5 清单）。

## §6 Tokenizer-free 前沿

"tokenizer-free"的准确含义：**不依赖预训练的 subword 词表**——它们仍需要 UTF-8 byte 编码、special ID、序列分块或 patching，并没有"没有任何离散化"这回事。三条主线：

- **ByT5**（Xue et al., 2021, arXiv 2105.13626）：mT5 架构直接吃 UTF-8 bytes（词表 = 256 + specials）。收益：对拼写扰动/噪声更鲁棒、embedding 表几乎免费、天然零 OOV；代价：同一文本序列长数倍，训练与推理都慢，需要更重的 encoder 配比。
- **MegaByte**（Yu et al., 2023, arXiv 2305.07185）：**固定大小 byte patch** 分层——global Transformer 建模 patch 序列、轻量 local 模型在 patch 内逐 byte 预测，把 $O(L^2)$ 的大头摊到 patch 粒度。不是"把所有 byte 直接丢给一个普通 Transformer"。
- **BLT（Byte Latent Transformer）**（Pagnoni et al. (Meta), 2024, arXiv 2412.09871）：**动态 patch**——用轻量 next-byte entropy 模型判边界，**高不可预测区域切得更密**（难的地方多花算力，简单区域大步跨过）；local byte encoder 聚合 patch → global latent Transformer 处理 patch 序列 → local decoder 逐 byte 生成。patch 平均长度可调，等效"每 FLOP 处理的 byte 数"可与 subword 模型竞争。

| 模型 | 输入粒度 | 分块机制 | 主要面向 | 一句话局限 |
| --- | --- | --- | --- | --- |
| ByT5 | UTF-8 byte | 无（全 byte 序列） | seq2seq 理解+生成 | 序列长、慢 |
| MegaByte | byte | **固定** patch | 自回归生成 | patch 边界与内容无关 |
| BLT | byte | **entropy 动态** patch | 自回归生成 | 需额外 entropy 模型、生态新 |
| CANINE（Clark et al., 2021, arXiv 2103.06874） | Unicode code point | 哈希 embedding + 下采样 | **理解任务** encoder | 非自回归 byte LM，不宜混为一谈 |

> 💡 **"tokenizer-free 已经解决规模化"仍是过强表述**
> 层次化 byte 模型确实缓解了长序列成本、展示了有竞争力的 scaling 曲线，但训练效率、推理引擎支持、缓存粒度（prefix cache 按什么命中？）和成熟部署工具链仍是现实障碍。面试的稳妥立场：**看好方向、承认工程差距**，并能讲出 BLT 的 entropy patching 与 MegaByte 固定 patch 的本质差别。

## §7 从零实现（纯 Python 标准库）

完整可跑脚本见 [`code/tokenization.py`](code/tokenization.py)（纯标准库、无 torch、CPU 秒级跑完 8 个 demo [D01]–[D08]）。两个教学简化要心里有数：脚本里的 toy BPE 是 **codepoint 级**（初始符号 = Unicode 字符、无 pre-tokenization；真实 GPT-2 先 regex 切块、块内 byte 映射后再 BPE，见 §3.1，[D01] 单独演示那层 byte 代理）；Unigram 两条 DP 用线性概率便于对读 §2.3 手算表，生产实现应换 **log 空间**防下溢。另两处 toy 范围：脚本的 toy BPE 不冻结基础字母表（未见字符直接成为基础 token；真实实现要么 byte 兜底要么 `<unk>`）；[D05] 的 `byte_fallback()` 演示的是 byte 表示与还原本身（无条件拆 byte），真实 SentencePiece 只对普通 piece 覆盖不了的 span 兜底。正文只展示四个核心 demo 的关键代码段；WordPiece 整词 `[UNK]`（[D03]）、NFKC vs byte fallback（[D05]）、prefix instability（[D06]）、扩词表 regression test（[D08]）作为短测试函数并入同一脚本，不再贴出。

**[D01] GPT-2 式 256-byte 可逆映射**（§3.1 的代理字母表，含 `0x20 → Ġ`）：

```python
def bytes_to_unicode():
    """可打印 byte 保留原字符，其余 68 个 byte 平移到 chr(256+n)——256↔256 双射。"""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)                 # 不可打印 byte → 平移进可打印区
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()
U2B = {u: b for b, u in B2U.items()}

assert len(B2U) == 256 and len(set(B2U.values())) == 256   # 双射 → 可逆
assert B2U[0x20] == "Ġ"                                    # 空格 byte 的代理字符
s = " A\n中文🙂\x00"                                        # 含空格/换行/CJK/emoji/控制符
units = "".join(B2U[b] for b in s.encode("utf-8"))
assert bytes(U2B[u] for u in units).decode("utf-8") == s   # 任意 UTF-8 严格往返
```

**[D02] BPE：训练记录有序 merge，编码只查冻结 rank**（§2.1 手算例子的可执行版）：

```python
def bpe_train(word_freqs, num_merges):
    """训练：每轮选最高频相邻 pair（tie-break：频次降序→字典序），按序记录 merge。"""
    words = {tuple(w): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        counts = Counter()
        for w, f in words.items():
            for i in range(len(w) - 1):
                counts[(w[i], w[i + 1])] += f
        if not counts:
            break
        pair = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merges.append(pair)                    # 顺序 = rank = 编码时的优先级
        words = {merge_word(w, pair): f for w, f in words.items()}
    return merges

def bpe_encode(text, merges):
    """编码：只应用冻结的 merge rank——每次合并 rank 最小的可用 pair（最左优先），
    绝不在输入上重新统计频率。"""
    rank = {p: i for i, p in enumerate(merges)}
    seq = list(text)
    while len(seq) > 1:
        best_rank, best_i = math.inf, -1
        for i in range(len(seq) - 1):
            r = rank.get((seq[i], seq[i + 1]), math.inf)
            if r < best_rank:
                best_rank, best_i = r, i
        if best_rank == math.inf:
            break
        seq = seq[:best_i] + [seq[best_i] + seq[best_i + 1]] + seq[best_i + 2:]
    return seq

merges = bpe_train({"abab": 3, "abac": 1}, num_merges=3)
assert merges[0] == ("a", "b")                 # §2.1 手算：pair 频次 7 = 2×3 + 1×1
text = "ababac"
assert "".join(bpe_encode(text, merges)) == text            # round-trip
assert bpe_encode(text, merges) == bpe_encode(text, merges) # 确定性
# rank 语义钉死：即使 (b,c) 也可用，rank 0 的 (a,b) 先行 —— 不是"重新选更频繁的"
assert bpe_encode("abc", [("a", "b"), ("b", "c")]) == ["ab", "c"]
assert bpe_encode("abc", [("b", "c"), ("a", "b")]) == ["a", "bc"]   # rank 反转，结果跟着 rank 走
```

**[D04] Unigram：Viterbi 最优路径 vs forward 全切分边际**（§2.3 DP 表的可执行版）：

```python
def viterbi(s, p):
    """最优单一切分：argmax 各 piece 概率之积。"""
    best = [(1.0, [])] + [None] * len(s)
    for i in range(1, len(s) + 1):
        cands = [(best[j][0] * p[s[j:i]], best[j][1] + [s[j:i]])
                 for j in range(i) if s[j:i] in p and best[j] is not None]
        best[i] = max(cands) if cands else None
    return best[len(s)][1]

def forward_prob(s, p):
    """P(s) = 对所有切分求和（边际概率），forward DP——它是 E-step 的 forward-backward 中的前半。"""
    f = [1.0] + [0.0] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if s[j:i] in p:
                f[i] += f[j] * p[s[j:i]]
    return f[len(s)]

p = {"a": .15, "b": .15, "ab": .40, "aba": .20, "bab": .10}
assert viterbi("abab", p) == ["ab", "ab"]                    # 最优路径，概率 0.16
assert abs(forward_prob("abab", p) - 0.22350625) < 1e-12     # §2.3 DP 表的 f4
assert forward_prob("abab", p) >= math.prod(p[x] for x in viterbi("abab", p))
```

**[D07] fertility 与 BPB**（§1.2/§1.3 指标口径的可执行版）：

```python
texts = ["def add(x): return x + 1", "把变量加一🙂"]        # 中英代码混合语料
total_bytes = sum(len(t.encode("utf-8")) for t in texts)
char_toks = sum(len(list(t)) for t in texts)                 # char 分词
byte_toks = sum(len(t.encode("utf-8")) for t in texts)       # byte 分词
merges7 = bpe_train({t: 1 for t in texts}, num_merges=12)    # 在语料自身上训 12 条 merge
bpe_toks = sum(len(bpe_encode(t, merges7)) for t in texts)   # toy-BPE（12 merges）
assert byte_toks / total_bytes == 1.0        # byte 分词 fertility 恒为 1 token/byte
assert char_toks < byte_toks                 # CJK/emoji 每字 3–4 byte → char 更省
assert bpe_toks < char_toks                  # merge 进一步压缩（micro-average 口径）

nll, n_bytes = 4.0, 8                        # 同一文本、同样总 NLL（nats）、8 bytes
loss_per_token_a, loss_per_token_b = nll / 2, nll / 4        # 2 个 vs 4 个 token
bpb_a = (2 * loss_per_token_a) / (n_bytes * math.log(2))     # 从 loss/token 独立换算回 BPB
bpb_b = (4 * loss_per_token_b) / (n_bytes * math.log(2))
assert loss_per_token_a == 2.0 and loss_per_token_b == 1.0   # loss/token 差一倍
assert bpb_a == bpb_b == 4.0 / (8 * math.log(2))             # BPB 却完全相同
```

> ✅ **四段代码各钉死一个高频误区**
> [D01]：byte-level 的"零 OOV"来自完整 256-byte 基础字母表全部进词表（双射只保证代理映射可逆）；[D02]：**编码只查 rank**，不重新统计频率；[D04]：**边际概率 $\geq$ 最优路径概率**——E-step 需要 forward-backward 的后验期望计数（D04 演示其中的边际量 forward），推理才用 Viterbi；[D07]：**同一总 NLL 下，仅 token 数不同就能让 loss/token 差数倍**——跨 tokenizer 比较用 BPB。

## §8 25 高频面试题

按难度分三档，点开看答案要点 + 易踩坑。L2/L3 是顶级 lab 深水区（训练 vs 编码语义、BPB 推导、扩词表系统设计、推理栈对齐、tokenizer-free）。答题时**不再重复正文推导**，按"框架 → 关键公式 → 常见错误"组织即可。

### L1必会题

<details>

<summary>Q1. 为什么 LLM 用 subword，而不是 word-level 或 char-level？</summary>

- word-level：OOV + 词表爆炸 + 形态变化不共享统计强度
- char/byte-level：byte-level 零 OOV（char-level 覆盖不全仍可 OOV）但序列长数倍，attention $O(L^2)$ / KV cache 遭殃
- subword：高频词整存、低频词拆碎、byte 兜底 → 词表可控 + 强覆盖（§1.1）

只答"折中"不给两端的具体代价；或不知道零 OOV 需要完整字母表兜底（实践中即 byte 字母表/byte fallback）才成立。

</details>

<details>

<summary>Q2. 手写 BPE：训练和 encode 分别怎么做？（最高频手写题）</summary>

- 训练：迭代统计相邻 pair 加权频次 → 合并最高频者 → **按序记录 merge 表**（tie-break 要确定性）
- encode：按**冻结 rank** 反复合并"当前 rank 最小的可用 pair"——不重新统计频率、不是最长匹配
- decode：token 字符串直接拼接（byte-level 再还原 UTF-8；真实系统还要处理 `▁`/`@@` 这类 BPE 边界标记）；另记：merge 不跨 pre-tokenization 块边界，重叠 pair 按"从左到右非重叠"替换

把 encode 写成"每次合并当前输入中最高频的 pair"——训练与编码语义混淆，是这道题的第一死因（§2.1）。

</details>

<details>

<summary>Q3. 为什么 byte-level BPE 基本没有 OOV？代价是什么？</summary>

- 初始字母表 = 完整 256 bytes，任何有效 UTF-8 文本都能拆到 byte 级 → 强覆盖保证
- GPT-2 用 `bytes_to_unicode` 双射把 byte 映射成可打印代理字符再跑 BPE（§3.1）
- 代价：罕见 Unicode 文本 fertility 高（低资源语言被拆碎）、byte 序列更长；normalization 若有损仍会改写原文

以为"没有 OOV"是 BPE 算法本身的性质（其实是 byte 字母表的性质）；或不知道代理字母表只是实现桥梁。

</details>

<details>

<summary>Q4. token 可视化里的 Ġ 和 ▁ 分别是什么？</summary>

- `Ġ`：GPT-2 代理字母表中 byte `0x20`（空格）的映射结果（U+0120）——任何空格 byte 都映射成它；常见于 token 开头是 regex 把空格附给后续块所致
- `▁`：SentencePiece 的 U+2581，把空格显式符号化参与建模，decode 时还原
- 共同点：都是序列化词表里真实存在的字符、语义都是"显式保存空白"；机制完全不同（byte remap vs 空白转义）——`▁` decode 还原的是 normalization 后的空格（dummy prefix/空白折叠下原始空白不保真）

只把 `Ġ`/`▁` 称作"怪字符"却说不出语义，或把两者机制混为一谈（§3.2 callout）。

</details>

<details>

<summary>Q5. WordPiece 的 `##` 前缀和 `[UNK]` 语义？</summary>

- `##` 标记"非词首接续 subword"（`##ing`），编码用 longest-match-first
- 任一位置匹配失败 → **整个词**输出 `[UNK]`，不是保留成功前缀 + `[UNK]`
- 例：词表 `{a, ##b, ##bc}` 时 `abc → [a, ##bc]`；去掉 `##bc` 后 `abc → [[UNK]]`（§2.2、[D03]）

答成 `[a, ##b, [UNK]]`——不知道 BERT-style 失败语义是整词回退。

</details>

<details>

<summary>Q6. fertility 是什么？怎么衡量一个 tokenizer 的好坏？</summary>

- fertility = 平均每个单位切多少 token，**必须声明分母口径**（狭义文献用法常特指 tokens/word；tokens/char、tokens/byte 更常称压缩率）
- 跨语言比较：内容/领域匹配（最好语义平行）的评测文本上报 bytes/token + chars/token +（声明外部分词器的）tokens/word + UNK rate，并分语言 macro 汇总（§1.2）
- 最终看端到端：同一语料总 token 数、吞吐、任务质量——tokenizer 是系统问题

只报一个"压缩率"数字不说口径；中文场景直接用 tokens/word 却不说分词标准。

</details>

<details>

<summary>Q7. BOS/EOS/PAD 各是干嘛的？把 PAD 设成 EOS 有问题吗？</summary>

- BOS 标记开始、EOS 标记结束（EOS 通常作为生成停止条件）；PAD 补齐 batch，attention mask + loss mask 双屏蔽
- PAD=EOS **本身不一定错**（很多模型没有独立 PAD）；经典事故是按 `input_ids==pad_id` 无差别构造 mask，把真实 EOS 的 loss/attention 一起屏蔽 → EOS 监督被移除、停止能力被显著削弱（从头训练尤甚）——mask 应按真实 padding 位置构造
- 相关坑：chat template 与 `add_special_tokens` 重复加 BOS；packing 时文档边界的 EOS/attention 处理（§5.5）

背"PAD=EOS 是错的"这种绝对化结论，说不出真正的失效链条（EOS 的 loss 被误屏蔽）。

</details>

<details>

<summary>Q8. `decode(encode(x)) == x` 一定成立吗？反过来呢？</summary>

- 不一定。本质是整条管线信息保真：normalizer 不丢信息、无 `<unk>`、special/post-processing 对称、decoder 不 cleanup；常用配置 = identity normalization + 空白保真 + byte 兜底 + 关 cleanup + 内部标记/special surface 转义单射（§3.3）
- NFKC 一步就有损：`Ａ①` → `A1`，原文永久丢失（byte fallback 也救不回）
- `encode(decode(ids)) == ids` **一般不成立**：非规范 token 序列 decode 出同一字符串，re-encode 只回规范切分

以为可逆是默认福利；或没意识到第二个方向（ids 往返）根本不保证；或把这种 ID 往返不保真（canonicalization，典型配置下幂等）与 prefix instability（§5.4，token healing 的直接根源）混为一谈。

</details>

### L2进阶题

<details>

<summary>Q9. BPE、WordPiece、Unigram 的核心区别？（最高频概念题）</summary>

- 训练：BPE 数最高频 pair（贪心加法）；WordPiece 按似然改善/PMI-like score 选 merge；Unigram 从大 seed 词表 EM 估计概率 + 按似然损失剪枝（减法）
- 编码：冻结 rank 合并 / longest-match-first / Viterbi（可采样）
- 补刀：`[UNK]` 语义（整词 vs `<unk>` vs byte 兜底零 OOV）、随机化（BPE-dropout vs 原生采样）、训练方向与产物（§2.4 表）

只背"三个名字 + 谁用谁"；讲不出"训练目标和编码算法是两套独立程序"这条主线。

</details>

<details>

<summary>Q10. SentencePiece 和 Unigram 是什么关系？</summary>

- SentencePiece 是**框架**（训练+编码+normalization+序列化），`model_type` 可选 unigram/bpe/char/word——不是 Unigram 的别名
- 特性：从原始句子直接训练（不需语言相关预切词，但 normalization/空白转义仍存在）；`▁` 空白符号化；默认 normalizer `nmt_nfkc`
- `byte_fallback=true` 保留 256 个 `<0xXX>` pieces 兜底（§3.2）

"SentencePiece 就是 Unigram""SentencePiece 没有任何预处理"——两个都是经典口误。

</details>

<details>

<summary>Q11. byte fallback 和 byte-level BPE 有什么区别？</summary>

- byte fallback（SentencePiece）：**兜底**——普通 Unicode pieces 覆盖不了时才退回 `<0xXX>` byte pieces，decode 时还原
- byte-level BPE（GPT-2/Llama 3）：**地基**——从第一步就在完整 256-byte 字母表上学 merge
- 两者都给强覆盖，但 byte fallback 救不回有损 normalization 改写的原文；碎片化模式也不同（§3.2 callout、[D05]）

把两者当同义词；或以为 byte fallback 能恢复 NFKC 之前的原文（覆盖性 ≠ 保真性）。

</details>

<details>

<summary>Q12. 词表为什么不能无限做大？（推导题）</summary>

- 成本端：embedding $\approx Vd$，untied 输出头再加 $Vd$；tied 省参数但 softmax **计算**仍 $\propto Vd$（§4.2）
- 收益端：V 大 →（通常）fertility 低 → 序列短 → attention $O(L^2)$/KV cache/生成步数全省
- 统计代价还有一头：固定语料下超大词表产生大量低频、欠训练的 embedding/输出行，削弱参数共享与组合泛化，压缩收益也递减
- 两头反向 → 最优 V 取决于语言覆盖、代码占比、模型宽度 $d$、部署预算；32k→256k 是设计点不是时间趋势

只答"embedding 变大"（漏了收益端就解释不了为什么词表一直在变大）；或把 100,277 vs 100,261 这类口径差当同一个数（§4.3）。

</details>

<details>

<summary>Q13. 数字是怎么被 tokenize 的？对算术有什么影响？</summary>

- 三代方案：GPT-2 数字串不限长（组合不规则）→ Llama 2 `split_digits` 逐位 → cl100k/o200k/Llama 3 `\p{N}{1,3}` 至多 3 位一块（§5.1）
- `{1,3}` 是 pre-token 上限：块内 BPE 仍可再拆，但不跨块合并
- 影响位对齐与跨 token 进位；但算术能力不能单因果归 tokenizer（数据、位置编码、CoT/工具同样关键）

把"LLM 算术差"全部甩锅 tokenizer；或以为 `{1,3}` 保证最终 token 恰好 3 位。

</details>

<details>

<summary>Q14. subword regularization 和 BPE-dropout 分别是什么？</summary>

- Unigram 采样：按 $q(z)\propto P(z)^{\alpha}$ 采样 segmentation——`alpha` 是逆温度（越大越接近 Viterbi、越小越随机）；同一词不同 epoch 切法不同，数据增强 + 鲁棒性
- BPE-dropout：BPE 无概率模型，编码时以概率 $p$ 随机**阻断**部分 merge 制造碎切分
- 目的相似、机制不同（采样后验 vs 阻断 merge）；标准部署通常关闭随机化、回到确定性切分（§2.3）

把 `alpha` 和 dropout 概率 $p$ 的语义混写（`alpha` 是逆温度，方向别说反）；或不知道 BPE 本身没有可采样的概率模型。

</details>

<details>

<summary>Q15. 如何公平比较一个 tokenizer 的中文和英文效率？</summary>

- 拒绝单报 tokens/word：中文"词"依赖外部分词标准，口径不声明就没有可比性
- 用内容/领域匹配（最好语义平行）的双语语料，报 UTF-8 bytes/token、chars/token、（声明分词器的）tokens/word、UNK rate、总 token 数与端到端吞吐，并声明 macro/micro 聚合（§1.2、§4.4）
- 记住成因：tokenizer 训练语料的语言配比显著影响各语言 fertility——低资源语言 token 成本更高、有效上下文更短

用英文口径的 tokens/word 直接套中文；或不知道中文 token 实际是"单字+多字词+byte 碎片+标点组合"的混合（§4.4）。

</details>

<details>

<summary>Q16. Unigram 训练里 E-step 用什么？为什么不是 Viterbi？</summary>

- 目标是边际似然 $P(x)=\sum_{z\in S(x)}\prod_i p(z_i)$——E-step 要用 forward-backward 汇总**所有切分**的后验期望计数
- 只取 Viterbi = hard-EM，优化的不是同一个目标；推理（确定性编码）才用 Viterbi（§2.3）
- 可执行直觉：[D04] 里 forward 概率 0.2235 $\gt$ 最优路径 0.16——边际和单条路径就是两个量

"训练和推理都用 Viterbi"——把 soft EM 写成 hard EM，这是设计审里点名的高频错误。

</details>

<details>

<summary>Q17. prefix instability 是什么？token healing 怎么救？</summary>

- 一般规律：$\text{encode}(s_1)$ 未必是 $\text{encode}(s_1+s_2)$ 的 token 前缀——追加字符可触发跨旧边界 merge（[D06]：`enc("ab")=[ab]` 但 `enc("abc")=[abc]`）
- 受害者：补全 prompt 的"悬空"结尾 token（分布错位）、流式输入、prefix cache
- token healing：回退最后若干可能未完成的 token，约束后续生成的 decode 前缀复现该后缀（经典单 token 情形 = 首 token 约束）；是推理侧包装，不是所有 API 标配（§5.4）

只知道"结尾空格"这个特例，讲不出更一般的 prefix instability；或以为 token healing 是模型能力而非推理包装。

</details>

<details>

<summary>Q18. glitch token（SolidGoldMagikarp）是怎么回事？</summary>

- 定义：词表中存在、但模型训练时几乎从未以该 ID 出现的 token → embedding/输出行欠训练
- 成因（典型解释，非已证唯一机制）：tokenizer 语料与模型语料错配（进词表的文本被后续清洗掉）；症状：无法复述、乱码、异常行为
- 诊断：embedding 范数异常扫描、repeat 测试；边界：个案性质，不是所有 GPT/byte-level tokenizer 的普遍故障（§5.3）

当都市传说讲，说不出"词表-语料错配 → 欠训练 embedding"的机理链条。

</details>

### L3高级题

<details>

<summary>Q19. 推导 BPB。为什么不同 tokenizer 的 perplexity 不能直接比？</summary>

- loss/token 的分母（token 数）依赖 tokenizer——即使总 NLL 相同，切得碎的那边 loss/token 也更低（[D07]）；实际差异再叠加模型质量，两效应不可分
- 固定不变量是**同一文本的 UTF-8 byte 数** $B$：$\mathrm{BPB}=\frac{-\sum_i\log p(t_i\mid t_{\lt i})}{B\ln 2}$；总 NLL 相同则 BPB 相同，与切几刀无关（[D07]：loss/token 2.0 vs 1.0，BPB 同为 0.7213）
- 前提：两边评测的是同一 byte 流——有损 normalization 必须先统一口径（§1.3）

比较 PPL 时忘了 exponent 底下的"每 token"就是被比较对象本身；或不检查 normalization 是否改写了评测文本。

</details>

<details>

<summary>Q20. 如何安全地给已有 LLM 扩词表？（系统设计）</summary>

- 冻结旧世界：旧 ID / merge rank / normalization 不动；控制符走 special 通道，领域词是普通 lexical token（会改变旧文本切分——用回归测试圈定兼容域，不要一律当 special）
- resize 两端 + optimizer state：embedding、LM head、Adam 的 $m,v$ 同步扩；tied weights 确认仍共享
- 新行 mean-init（lexical token 用自身 surface 旧切分、special 用描述文本代理旧切分的均值）+ **继续训练**；扩 LM head 后新行进入 softmax 分母——旧 token 分布已变，旧集 NLL/生成行为也要回归；再加编码 regression：旧测试集逐 ID 一致、special 行为不变、round-trip 行为与扩表前一致（§4.5、[D08]）

只改 tokenizer 文件就以为模型认识新 token；或漏掉 LM head/optimizer state/tied 检查中的任何一个。

</details>

<details>

<summary>Q21. 哪些系统组件要求 tokenizer 对齐？分别为什么？</summary>

- 经典 speculative decoding：draft/target 需**同一可对齐 token 支持**（逐 token 拒绝采样要比较两个分布）；异 tokenizer 需要专门的字符串级 proposal-verification 设计，朴素桥接不保精确等价
- 逐位置 logit 蒸馏：词表 + 位置都要对齐；sequence-level KD 与字符级目标可跨 tokenizer，hidden-state KD 需 span 对齐/pooling
- KV/prefix cache：按**精确 token-ID 前缀**命中——失效条件是同一字符串变成不同 ID 序列 / template 变 / 权重变，"字符串相同"不足够（§5.7）

抽象地说"tokenizer 变了 cache 就坏了"，给不出精确失效条件；或不知道 sequence-level 蒸馏可以跨 tokenizer。

</details>

<details>

<summary>Q22. 设计一个 offset mapping 方案：token 如何对回原文？</summary>

- 多套坐标：normalized text 字符偏移 / 原文 code point 偏移 / 原文 UTF-8 byte 偏移（还有 UTF-16 code unit、grapheme cluster）——明确 API 返回的是哪套（§5.6）
- 有损 normalization（NFKC）改写字符（`ﬃ`→`ffi` 1→3 code point；`Ａ`→`A` 值/byte 长度变、code point 数不变），无 alignment map 时对不回原文；NFC 也不保 code-point 长度（`e`+U+0301→`é` 2→1）；emoji ZWJ/组合字符让 code point 与用户感知字符脱节
- 方案：近保真 normalization + 保留 alignment map（处理一对多/多对一 span）+ tokenizer 原生 offset mapping + 双口径（byte 与 code point）测试；下游（NER/抽取式 QA/代码编辑）按各自坐标系消费

只给"用 return_offsets_mapping"一句话，不提 normalization 与 Unicode 的两层坐标错位。

</details>

<details>

<summary>Q23. 让你从头为一个新 LLM 训 tokenizer，语料和配置怎么设计？</summary>

- 语料：语言配比（显著影响各语言 fertility；可有意重平衡低资源语言，记录配比）、代码/数学占比、去重清洗；**清洗与 normalization 语义必须与模型语料一致**，且词表 token 要在模型语料中有足够曝光——词表-语料错配才是 glitch token 的温床（§4.1）
- 配置：词表大小按 $Vd$ 账 + fertility 目标定（§4.2）；数字策略（`{1,3}` vs 逐位）；byte 兜底；special/保留槽位一次留够
- 验收：多语言多口径 fertility、round-trip 测试（identity 管线断言等原文，有损 normalization 断言 decode 到规范形）、小规模受控预训练比 BPB/下游质量/吞吐、regression 基线、版本化全套文件+哈希（§5.8）

一上来就纠结"BPE 还是 Unigram"——语料设计对最终 fertility 的影响通常大于算法选择（§4.1 主旨）。

</details>

<details>

<summary>Q24. tokenizer-free：ByT5 / MegaByte / BLT 各自机制？现实障碍？</summary>

- ByT5：UTF-8 byte 直入（词表 256+specials），鲁棒但序列长数倍；MegaByte：**固定** byte patch，global 建模 patch 序列 + local 建模 patch 内 byte，摊薄 $O(L^2)$
- BLT：**entropy 动态 patch**——轻量 next-byte entropy 模型定边界，难预测区域 patch 更密（按难度分配算力）；local encoder → global latent Transformer → local decoder
- 准确表述："tokenizer-free"= 不依赖预训练 subword 词表（仍需 UTF-8/special ID/patching）；scaling 有竞争力但训练效率、推理生态、缓存粒度仍是障碍（§6）

把三者说成"都是把 byte 丢进 Transformer"；或宣称"tokenizer 已被解决"（§6 点名的过强表述）。

</details>

<details>

<summary>Q25. chat template 与 special token 注入：怎么防？训练侧还有什么坑？</summary>

- 注入面：用户文本里的字面量 `<|assistant|>` 是否变 special ID 取决于 API（先注册、再由 `allowed_special` 准入 / 模板转义）——根本防线是**分通道编码**：模板控制符由代码直接注入 ID，用户内容在关闭 special 识别下单独 encode，测试只是兜底；这是 correctness + prompt injection 双重边界（§5.5）
- 训练侧：attention mask ≠ loss mask；assistant-only loss；packing 文档边界；模板与 tokenizer 重复加 BOS；PAD=EOS 时别把 EOS 的 loss 全屏蔽
- 配套：tokenizer 版本化（normalization+regex+merges+special/added-token 元数据+post-processor/decoder+chat template+库版本+哈希），只存 vocab.json 会静默漂移（§5.8）

只答"过滤用户输入里的特殊字符串"——不知道行为由 tokenizer API 配置决定、且训练侧 mask 错误同样致命。

</details>

## §A 附录：sanity check

本 tutorial 的从零实现应满足以下关键不变量（纯 Python 标准库、无第三方依赖、CPU 秒级，脚本见 [`code/tokenization.py`](code/tokenization.py)）：

1. **[D01] 256-byte 双射**：`bytes_to_unicode` 恰好 256 进 256 出（injective → 可逆），188 个可打印 byte 原样、68 个平移 byte 升序进 U+0100 区（`0x00→U+0100`、`0x0A→Ċ`、`0x20→Ġ`）；含空格/换行/CJK/emoji/控制符的字符串经 UTF-8 → 代理字母表 → 逆映射 → UTF-8 严格还原。
2. **[D02] BPE 训练 vs 编码语义**：语料 `{"abab"×3, "abac"×1}` 上首条 merge 必为 `(a,b)`（频次 7）；`decode(encode(x))==x`、编码确定性、token 数 $\leq$ byte 数；rank 冲突用例 `encode("abc")==["ab","c"]`、rank 反转后变 `["a","bc"]`（编码只跟冻结 rank 走，不做运行时重选）；`"aaa"` 从左非重叠合并为 `["aa","a"]`。
3. **[D03] WordPiece 整词失败**：`{a, ##b, ##bc}` 下 `abc → [a, ##bc]`；去掉 `##bc` 后 `abc → [[UNK]]`（不是 `[a, ##b, [UNK]]`）；`{a, ab, ##bc}` 下贪心选 `ab` 后走入死路 → 整词 `[UNK]`（贪心不回溯，尽管 `a`+`##bc` 本可成功）。
4. **[D04] Unigram 两条 DP**：Viterbi 给 `["ab","ab"]`（0.16）；forward 全切分边际 $=0.22350625$（独立手算验证）且 $\geq$ 最优路径概率。
5. **[D05] 覆盖 ≠ 保真**：NFKC 把 `Ａ①` 改写为 `A1`（有损、不可逆）；byte fallback 把 `龘` 拆成 `<0xE9><0xBE><0x98>` 且可严格还原。
6. **[D06] prefix instability**：merge 表 `[a+b, ab+c]` 下 `encode("ab")=["ab"]` 不是 `encode("abc")=["abc"]` 的 token 前缀。
7. **[D07] 指标口径**：byte 分词 fertility 恒为 1 token/byte；中英混合语料上 toy-BPE $\lt$ char $\lt$ byte（micro-average）；同一文本 NLL=4.0 nats、8 bytes 时 loss/token 2.0 vs 1.0 而 BPB 相同。
8. **[D08] 扩词表 regression**：旧 ID / 旧测试集编码 / special 行为全部不变；新 embedding 行 = 语义代理文本旧切分行的逐维均值；tied 的输入 embedding 与 LM head 仍是同一参数对象。

运行 `python3 code/tokenization.py` 的**真实输出**：

```text
[D01] bytes_to_unicode: 256/256 bijective, 0x20 -> 'Ġ', roundtrip ' A\n中文🙂\x00' OK  PASS
[D02] BPE: merges=[('a', 'b'), ('ab', 'ab'), ('a', 'c')], encode('ababac')=['abab', 'ac'], rank-conflict 'abc'->['ab', 'c']  PASS
[D03] WordPiece: 'abc'->['a','##bc']; drop '##bc' -> ['[UNK]'] (whole word)  PASS
[D04] Unigram: viterbi=['ab', 'ab'] (prob 0.1600), forward=0.22350625 >= best path  PASS
[D05] NFKC('Ａ①')='A1' (lossy); byte_fallback('龘')=<0xE9><0xBE><0x98> roundtrips  PASS
[D06] prefix instability: enc('ab')=['ab'] but enc('abc')=['abc'] -> not a token-prefix  PASS
[D07] fertility tokens/byte: byte=1.000, char=0.698, toy-BPE=0.419; loss/token 2.0 vs 1.0 yet BPB both 0.7213  PASS
[D08] vocab extension: old IDs/encodings/specials unchanged, new row = mean(rows [0, 2]) = [4.0, 5.0, 6.0, 7.0], tied head intact  PASS

all tokenization sanity checks passed ✓
```

## 📚 参考文献

- **BPE（压缩起源）** — Gage, *A New Algorithm for Data Compression*, C Users Journal (1994)（**无 arXiv id**）.
- **BPE for NMT** — Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*, arXiv 1508.07909 (2015), ACL 2016.
- **WordPiece（起源）** — Schuster & Nakajima, *Japanese and Korean Voice Search*, ICASSP 2012（**无 arXiv id**）.
- **GNMT（WordPiece 大规模应用）** — Wu et al., *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*, arXiv 1609.08144 (2016).
- **BERT** — Devlin, Chang, Lee & Toutanova, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, arXiv 1810.04805 (2018), NAACL 2019.
- **Unigram LM / subword regularization** — Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*, arXiv 1804.10959 (2018), ACL 2018.
- **SentencePiece** — Kudo & Richardson, *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*, arXiv 1808.06226 (2018), EMNLP 2018 (System Demonstrations).
- **T5（SentencePiece-Unigram 用户）** — Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, arXiv 1910.10683 (2019), JMLR 2020.
- **ALBERT（SentencePiece-Unigram 用户）** — Lan et al., *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*, arXiv 1909.11942 (2019), ICLR 2020.
- **BPE-dropout** — Provilkov, Emelianenko & Voita, *BPE-Dropout: Simple and Effective Subword Regularization*, arXiv 1910.13267 (2019), ACL 2020.
- **GPT-2（byte-level BPE）** — Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI technical report (2019)（**无 arXiv id**）.
- **tiktoken** — OpenAI, https://github.com/openai/tiktoken（BPE 实现库，cl100k_base / o200k_base encodings；**无论文**）.
- **Llama 2** — Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288 (2023).
- **Llama 3** — Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783 (2024).
- **Qwen2** — Yang et al. (Qwen Team), *Qwen2 Technical Report*, arXiv 2407.10671 (2024).
- **Gemma** — Gemma Team (Google DeepMind), *Gemma: Open Models Based on Gemini Research and Technology*, arXiv 2403.08295 (2024).
- **ByT5** — Xue et al., *ByT5: Towards a token-free future with pre-trained byte-to-byte models*, arXiv 2105.13626 (2021), TACL 2022.
- **MegaByte** — Yu et al. (Meta), *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*, arXiv 2305.07185 (2023), NeurIPS 2023.
- **BLT** — Pagnoni et al. (Meta), *Byte Latent Transformer: Patches Scale Better Than Tokens*, arXiv 2412.09871 (2024).
- **CANINE** — Clark, Garrette, Turc & Wieting, *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation*, arXiv 2103.06874 (2021), TACL 2022.
- **Speculative decoding** — Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, arXiv 2211.17192 (2022), ICML 2023.
- **Speculative sampling** — Chen et al. (DeepMind), *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv 2302.01318 (2023).
- **SolidGoldMagikarp（glitch tokens）** — Rumbelow & Watkins, *SolidGoldMagikarp (plus, prompt generation)*, LessWrong blog post (2023)（**无 arXiv id**，按博客出处引用）.
