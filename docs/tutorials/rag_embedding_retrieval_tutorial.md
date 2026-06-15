## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 RAG + 嵌入** — 一页拿下面试核心要点（详见后文 §2–§11 推导）。

1. **RAG 是什么**：检索增强生成 = 先用 query 从外部知识库**检索**相关片段，再把片段拼进 prompt 让 LLM **生成**答案。应对三件事：知识过期、幻觉、私有数据（Lewis et al. 2020）。

2. **两半互补**：左半「**训嵌入**」是对比学习（双塔 + InfoNCE + 难负例），数学密；右半「**用嵌入**」是 RAG 管线（chunk → 检索 → 重排 → 拼 prompt），系统密。

3. **嵌入主公式 InfoNCE**：$\mathcal{L} = -\log \frac{\exp(\text{sim}(q,d^+)/\tau)}{\sum_i \exp(\text{sim}(q,d_i)/\tau)}$ —— 本质是「正样本 vs 一堆负样本」的 softmax 交叉熵，温度 $\tau$ 调锐度。

4. **双塔 vs 交叉编码器**：双塔（bi-encoder）query/doc 分开编码，**可预计算**、快、用于召回；交叉编码器（cross-encoder）拼一起编码，**不可预计算**、慢但准，用于**重排 top-k**。

5. **难负例（hard negative）**：从检索 top-k 里挑「像但不相关」的当负例，比随机负例信息量大；但有 **false negative 陷阱**——挑到的「难负例」可能其实相关，反成噪声。

6. **稀疏 vs 稠密 vs 混合**：BM25（词项精确匹配，强在罕见词/实体）+ 稠密向量（语义匹配）互补，用 **RRF**（Reciprocal Rank Fusion）按排名融合：$\text{RRF}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$，$k\approx60$。

7. **Matryoshka 表征**：一次训练让嵌入的**前 $m$ 维**也是好嵌入，部署时按需截断维度（省存储/加速召回），靠多粒度损失 $\sum_m \mathcal{L}(\text{emb}[:m])$。

8. **RAG vs 长上下文 vs 微调**：RAG 注入**可更新的外部事实**、可溯源；长上下文塞全文但贵且有「lost in the middle」；微调改**能力/风格**不擅长注入海量新事实。三者常组合。

9. **评测**：别只看生成，要分层评 **检索质量**（recall@k / nDCG）+ **生成忠实度**（faithfulness / context relevance / answer relevance，如 RAGAS）。

## §1 直觉：为什么需要 RAG

LLM 有三个结构性短板：**知识截止**（训练后的事不知道）、**幻觉**（不知道时也会一本正经编）、**私有数据**（你公司的文档它没见过）。直接微调来灌知识又贵又慢、还容易过时。

**RAG（Retrieval-Augmented Generation）把「知识」从参数里搬到外部知识库**：把权重当「推理引擎」，把向量库当「可随时更新的外部记忆」。回答时先检索相关片段、再让 LLM 基于片段作答——知识更新只需更新库（不重训）、答案能溯源（引用来源）、私有数据不进训练集。

这条链路天然分成**互补的两半**：

- **训嵌入（左半，数学密）**：怎么把一句话/一段文档变成一个向量，使「语义相近 → 向量相近」。核心是**对比学习**（双塔 + InfoNCE + 难负例）。
- **用嵌入（右半，系统密）**：怎么用这些向量搭一条能用的管线——切块（chunking）、建索引（ANN）、召回、重排、拼 prompt、评测。

> 💡 **一句话心智模型** — LLM 是「闭卷考试只能靠脑子」，RAG 是「开卷考试，先用 query 去书里翻到相关页（检索），再照着页面作答（生成）」。翻得准不准 = 嵌入 + 检索质量；答得好不好 = 重排 + prompt + LLM。

## §2 文本嵌入：双塔 + 对比学习

### 2.1　双塔（bi-encoder）结构

把 query 和 document 各自过一个编码器（通常共享或孪生 BERT-like 模型），取句向量（`[CLS]` 或 mean pooling），L2 归一化：

$$\mathbf{q} = \text{normalize}(\text{Enc}(\text{query})), \qquad \mathbf{d} = \text{normalize}(\text{Enc}(\text{doc}))$$

相似度用点积（归一化后等于余弦）：$\text{sim}(q,d) = \mathbf{q}^\top \mathbf{d}$。

关键优势：**doc 向量可离线预计算并建索引**，线上只编码 query 一次、做近邻搜索。这是双塔能扛海量库的根本原因（对比 §6 的交叉编码器）。

### 2.2　InfoNCE / 对比损失（必考，要会推）

目标：让 query 和它的正例 doc $d^+$ 相近、和负例 $d^-$ 远离。把它写成「在一堆候选里选出正例」的分类问题——对每个 query，候选集 $\{d^+, d_1^-, \dots, d_{N}^-\}$ 上做 softmax 交叉熵：

$$\boxed{\;\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_{i} \exp(\text{sim}(q, d_i^-)/\tau)}\;}$$

推导视角：这就是把「相似度 $/\tau$」当 logits、正例索引当 label 的 **cross-entropy**。最小化它 = 最大化 $q$ 与 $d^+$ 的相似度、压低与所有负例的相似度（InfoNCE，van den Oord 2018；用于检索即 DPR，Karpukhin 2020）。

**温度 $\tau$ 的作用**：$\tau$ 小 → logits 被放大 → softmax 更尖 → 模型更「在意」最难的负例（梯度更偏向 hardest negative；但正例已压倒性最大时小 $\tau$ 也会使梯度饱和变小）；$\tau$ 大 → 更平滑、各负例权重更均匀。$\tau$ 太小会对噪声/false negative 过敏，太大则区分度不足，是关键超参（常见 $0.01\sim0.07$）。

### 2.3　In-batch 负例与难负例

**In-batch negatives**：一个 batch 有 $B$ 对 $(q_i, d_i^+)$，对 $q_i$ 而言，**同 batch 里其它 $B-1$ 个正例 $d_{j\ne i}^+$ 就当负例**。一次前向就拿到 $B-1$ 个负例，几乎零额外开销（DPR 的关键 trick）。batch 越大、负例越多、对比越强（大 batch 对对比学习很重要）。

**难负例（hard negative）**：随机负例往往太容易（一眼就不相关），梯度小。从 BM25/上一版模型的检索 top-k 里挑「字面/语义像、但其实不相关」的当负例，信息量大得多，是刷榜关键。

> ⚠️ **False negative 陷阱** — 从 top-k 挖的「难负例」可能**其实是相关的**（只是没标注），把它当负例会给错误梯度。缓解：去掉与正例太像的候选（阈值过滤）、用 cross-encoder 给负例打分剔除高分项、或 teacher 蒸馏软标签。难负例是双刃剑，不是越难越好。

## §3 难负例的对偶 + Matryoshka 表征

### 3.1　余弦 / 点积 / 归一化

L2 归一化后 $\lVert \mathbf{q} \rVert = \lVert \mathbf{d} \rVert = 1$，于是 $\mathbf{q}^\top\mathbf{d} = \cos\theta \in [-1,1]$。归一化的好处：相似度只看**方向**不看模长，训练更稳（绝对阈值仍需按模型/语料校准）。多数检索嵌入都归一化后用点积（等价余弦）。

### 3.2　Matryoshka 表征学习（MRL）

普通嵌入是定长（如 768 维），想省存储/加速只能事后降维（PCA）损失大。**Matryoshka（Kusupati 2022）让一次训练就得到「嵌套」嵌入**：前 $m$ 维（$m \in \{64,128,256,\dots,768\}$）各自都是合格嵌入。做法是对多个粒度同时算损失：

$$\mathcal{L}_{\text{MRL}} = \sum_{m \in \mathcal{M}} w_m \, \mathcal{L}_{\text{InfoNCE}}\big(\mathbf{q}[:m],\ \mathbf{d}[:m]\big)$$

部署时按预算截断：粗排用前 64/128 维（快、省）、精排用全维。OpenAI `text-embedding-3`、Nomic、BGE 等都支持 MRL 维度可调。

> 💡 **MRL 为何能成立** — 多粒度损失迫使模型把「最重要的语义」压进前几维（类似按重要性排序的主成分），所以截断前缀仍可用。注意：用于余弦/点积检索时，每个前缀 $\mathbf{q}[:m]$ 通常要**重新 L2 归一化**（否则不同前缀范数不同会改变 logit 尺度）。代价：训练目标更复杂、满维质量可能略降一点点，换来部署弹性。

## §4 从零实现：双塔 + InfoNCE 训练

下面是一个可跑的双塔 + in-batch InfoNCE 训练步（用 toy 编码器，重点是对比损失与难负例逻辑）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    """Toy 双塔：query/doc 共享一个编码器；真实场景换成 BERT-like。"""

    def __init__(self, vocab, dim=128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")  # 词袋均值当句向量
        self.proj = nn.Linear(dim, dim)

    def encode(self, ids, offsets):
        h = self.proj(self.emb(ids, offsets))
        return F.normalize(h, dim=-1)                        # L2 归一化 -> 点积=余弦


def info_nce_in_batch(q, d_pos, tau=0.05):
    """q, d_pos: [B, dim] 已归一化。同 batch 其它正例当负例。"""
    logits = (q @ d_pos.t()) / tau          # [B, B]，第 i 行第 i 列是正例
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)  # 对角线为正 -> 多分类交叉熵


def info_nce_with_hard(q, d_pos, d_hard, tau=0.05):
    """额外拼接每个 query 自己的难负例 d_hard: [B, dim]。"""
    pos = (q * d_pos).sum(-1, keepdim=True)         # [B,1] 正例相似度
    in_batch = q @ d_pos.t()                        # [B,B] in-batch 负例
    in_batch.fill_diagonal_(float("-inf"))          # 去掉自身正例避免重复计入
    hard = (q * d_hard).sum(-1, keepdim=True)       # [B,1] 难负例相似度
    logits = torch.cat([pos, in_batch, hard], dim=1) / tau   # 正例放第 0 列
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)
```

要点：

- `info_nce_in_batch` 的精髓：相似度矩阵 $q d^\top$ 的**对角线是正例**，所以 label 就是 `arange(B)`，一行 cross_entropy 搞定。
- 加难负例时把正例 logit 放到第 0 列、label 全 0；`fill_diagonal_(-inf)` 避免 in-batch 矩阵把自身正例又算一遍。
- `tau` 越小 logits 被放得越大，越偏向最难负例。

## §5 检索：稀疏 / 稠密 / 混合

### 5.1　稀疏检索：BM25

BM25 是词项精确匹配的经典基线，对**罕见词、实体名、代码/ID** 这类「必须字面命中」的 query 仍很强：

$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\big(1 - b + b\,\frac{\lvert d\rvert}{\text{avgdl}}\big)}$$

其中 $f(t,d)$ 是词频，$\lvert d\rvert$ 是文档长度，$\text{avgdl}$ 是平均长度，$k_1$（≈1.2–2.0）控词频饱和、$b$（≈0.75）控长度归一。**词频饱和**：出现 10 次不等于比 1 次重要 10 倍。

### 5.2　稠密检索 + ANN（HNSW）

稠密 = 把 query 编码成向量、在向量库里找最近邻。库一大就不能暴力算全部，用**近似最近邻（ANN）**。主流是 **HNSW**（Malkov & Yashunin 2016）：建多层「可导航小世界」图，高层稀疏跳远、低层密集精搜，查询**经验上近似对数 / 亚线性**（非严格保证，依赖 efSearch、图度数、召回目标）。

> ⚠️ **稠密不是万能** — 稠密向量擅长**语义/同义/改写**，但对「精确实体、罕见专名、序列号」可能不如 BM25（这些词在语义空间里没区分度）。所以工业界很少纯稠密，多走混合。

### 5.3　混合检索 + RRF

把 BM25 和稠密各自的排名用 **Reciprocal Rank Fusion** 融合（只用排名、不用分数，免去量纲对齐）：

$$\boxed{\;\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)}\;}, \qquad k \approx 60$$

$\text{rank}_r(d)$ 是文档 $d$ 在检索器 $r$ 里的名次（1 起）。常数 $k$ 压低头部名次的过度主导。RRF 简单、稳、无需训练，是混合检索的默认融合法（Cormack 2009）。

> 💡 **learned sparse：SPLADE** — 在 BM25（词项）与稠密（语义）之外还有一条中间路线：**SPLADE**（Formal 2021）用 MLM 头把 query/doc 展开成**带学习权重的稀疏词项向量**（含同义/扩展词），既能进倒排索引（快、可解释），又有语义扩展能力，常作为强 baseline 或混合的一路。

## §6 重排（Rerank）：交叉编码器

召回（双塔 + ANN）追求**高 recall、快**，但 top-k 里精排不够。**重排器把召回的 top-k（如 100）重新精确打分、取前几个**。

| | 双塔 bi-encoder | 交叉编码器 cross-encoder | ColBERT（late interaction） |
| --- | --- | --- | --- |
| 编码方式 | query、doc **分开**编码 | `[query; doc]` **拼一起**编码 | doc 存**逐 token** 向量 |
| 交互 | 仅最后点积（无 token 级交互） | 全程 attention（early，强） | token 级 late interaction（MaxSim） |
| 可预计算 doc | ✅（建索引） | ❌（每对都要重算） | ✅（存 token 向量，较大） |
| 速度 / 规模 | 快 / 百万–十亿 | 慢 / 只能重排 top-k | 中 / 较大存储 |
| 用途 | **召回** | **重排** | 召回+重排折中 |

交叉编码器：$\text{score}(q,d) = \text{MLP}(\text{BERT}([q;d])_{\texttt{[CLS]}})$，query 和 doc 的 token 全程互相 attend，所以准；代价是**无法预计算**（doc 表示依赖 query），只能对召回的少量候选跑。

**ColBERT 的 late interaction**：doc 存每个 token 的向量，打分用 MaxSim——query 每个 token 找 doc 里最相似的 token 再求和：

$$\text{score}(q,d) = \sum_{i \in q} \max_{j \in d} \mathbf{E}_{q_i}^\top \mathbf{E}_{d_j}$$

比双塔的单向量更细粒度、又能预计算 doc 端，是召回与重排之间的折中（Khattab & Zaharia 2020）。代价是**存储/延迟**：每个 doc 存逐 token 向量，索引远大于单向量；**ColBERTv2**（量化 + 残差压缩）与 **PLAID**（中心点剪枝加速）是把多向量 late interaction 真正落地到可接受存储/延迟的关键工程（Santhanam 2021/2022）。单向量 vs 多向量是「省存储 vs 高精度」的经典权衡。

## §7 RAG 管线：从 query 到答案

```
query
  │  ① query 改写 / 扩展 (HyDE / multi-query)
  ▼
[稀疏 BM25]  +  [稠密 ANN]
  │                │
  └──── RRF 融合 ───┘   ② 召回 top-100
  ▼
cross-encoder 重排  ③ 取 top-5
  ▼
拼 prompt (context + question)  ④ 含来源标注
  ▼
LLM 生成 + 引用   ⑤ 可选自检: Self-RAG(生成反思) · CRAG(检索质量→纠错回路)
```

- **⓪ Chunking（离线切块）**：文档切成片段再嵌入（属离线预处理，不在上面的在线流程图里）。块太大 → 检索粗、稀释相关信息；块太小 → 上下文断裂。常用 200–500 token + 重叠（overlap），或按语义/标题结构切。**chunk 策略对效果影响极大**，是最该调的旋钮之一。
- **① Query 改写**：**HyDE**（Gao 2022）先让 LLM 生成一个「假设**文档 / 段落**」再用它的嵌入检索（让查询向量分布对齐真实文档；QA 场景这段「伪文档」常像答案，但核心是用伪文档去查真实语料）；multi-query 生成多个改写并合并召回。
- **④ Prompt 拼装**：context 片段 + 原 question + 「只依据 context 回答、给出处」指令；注意 **lost-in-the-middle**（Liu 2023）——放中间的关键片段易被忽略，重要的放头尾。
- **⑤ 自检式 RAG**：**Self-RAG**（Asai 2023）训模型输出「要不要检索 / 检索到的有没有用 / 答案有没有依据」的反思 token；**CRAG**（Yan 2024）用轻量 evaluator 给**已检索文档**的相关性/置信度打分，低置信或歧义时触发纠正检索（web 搜索 / 重写）。

## §8 RAG 进阶：GraphRAG · 对比 · 评测

### 8.1　GraphRAG

朴素 RAG 按片段独立召回，回答「跨多文档的全局性问题」（如「整个报告的主题脉络」）很弱。**GraphRAG**（微软，Edge 2024）先用 LLM 把语料抽成**实体-关系知识图 + 社区摘要**，全局问题走「社区摘要」而非散片段。强在 query-focused summarization，代价是建图贵。

### 8.2　RAG vs 长上下文 vs 微调

| | RAG | 长上下文 | 微调 |
| --- | --- | --- | --- |
| 注入什么 | 可更新的**外部事实** | 当次塞进的全文 | **能力 / 风格 / 格式** |
| 知识更新 | 改库即可、可溯源 | 每次重塞、无持久 | 重训、易过时 |
| 成本 | 检索 + 短 prompt | 长 prompt 贵、慢 | 训练贵、推理便宜 |
| 短板 | 检索错则答错 | lost-in-the-middle | 不擅长灌海量新事实 |

实务里三者常组合：微调调风格 + RAG 注事实 + 适度长上下文容纳召回片段。

### 8.3　评测（别只看生成）

分两层评：

- **检索层**：recall@k、nDCG、MRR——召回的片段对不对；选嵌入/检索器可参考 **MTEB**（嵌入综合榜，Muennighoff 2022）与 **BEIR**（零样本检索泛化，Thakur 2021）这类标准 benchmark，而非只看端到端分。
- **生成层**：**faithfulness**（答案是否忠于 context，不编）、**context relevance**（召回是否相关）、**answer relevance**（答案是否答到点）。RAGAS（Es 2023）用 LLM 自动算这几项。

> ❌ **经典翻车** — 只看「答案读起来对不对」，不分层。检索召回了错片段、LLM 据此「忠实地」给出错答案，端到端看像幻觉，根因却在检索。必须把检索和生成分开归因。

## §9 从零实现：mini-RAG 管线

```python
import re, math
import torch, torch.nn.functional as F

def chunk(text, size=40, overlap=10):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += size - overlap                      # 滑窗重叠，避免上下文断裂
    return out

def embed(texts, encoder):                       # encoder: str -> [dim] 归一化向量
    return F.normalize(torch.stack([encoder(t) for t in texts]), dim=-1)

def dense_topk(qv, dv, k):                        # 余弦 top-k
    sims = dv @ qv                                # [N]
    return sims.topk(min(k, len(dv))).indices.tolist()

def bm25_scores(query, docs, k1=1.5, b=0.75):
    toks = [d.lower().split() for d in docs]
    avgdl = sum(len(t) for t in toks) / len(toks)
    N = len(docs)
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    scores = []
    for d in toks:
        s, dl = 0.0, len(d)
        for w in set(query.lower().split()):
            if w not in df:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            f = d.count(w)
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores

def rrf(rank_lists, k=60):                        # rank_lists: [[doc_idx 按名次]...]
    score = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=score.get, reverse=True)

def mini_rag(query, corpus, encoder, top_recall=10, top_final=3):
    chunks = [c for doc in corpus for c in chunk(doc)]
    dv = embed(chunks, encoder)
    qv = F.normalize(encoder(query), dim=-1)
    dense_rank = dense_topk(qv, dv, top_recall)                       # 稠密名次
    bm25 = bm25_scores(query, chunks)                                 # 先算一次，避免 O(N^2)
    bm25_rank = sorted(range(len(chunks)), key=lambda i: bm25[i], reverse=True)[:top_recall]
    fused = rrf([dense_rank, bm25_rank])[:top_final]                  # 混合 + 截断
    context = "\n".join(f"[{r}] {chunks[i]}" for r, i in enumerate(fused, 1))
    return f"基于以下片段回答（标注出处编号）：\n{context}\n\n问题：{query}"
```

这段把 §5–§7 串起来：切块 → 稠密 + BM25 双路召回 → RRF 融合 → 拼成带出处的 prompt（真实场景在融合后再加 cross-encoder 重排）。

## §10 工程实践与常见 bug

- **chunk 大小是头号旋钮**：太大检索粗、太小断上下文；先试 256/512 token + 10–20% overlap，按评测调。
- **比调 chunk size 更高价值的改进**：**Contextual Retrieval**（Anthropic 2024，嵌入前给每个 chunk 补一段 LLM 生成的全文上下文）与 **late chunking**（Jina 2024，先整文过长上下文 encoder、再切块池化）都针对「普通切块丢失文档级上下文」这一根因，往往比单纯调块大小收益更大。
- **嵌入模型要对域**：通用嵌入在专业领域（医疗/法律/代码）可能弱，考虑领域微调或选对口模型（BGE-M3 / E5 / 领域模型）。
- **别纯稠密**：实体/罕见词/精确匹配场景务必加 BM25 混合，否则「检索不到明明有的那条」。
- **难负例 false negative**（§2.3）：挖负例要去重、过滤，否则训歪。
- **lost in the middle**（§7）：关键片段放 context 头尾，别堆中间。
- **重排别省**：召回 recall 够但精度不足时，cross-encoder 重排 top-k 往往是高 ROI 的一步（是否最高取决于召回质量 / 延迟预算 / 候选规模）。
- **索引会过期**：库更新后要增量重嵌入 / 重建索引；陈旧索引 = 陈旧答案。
- **评测要分层**（§8.3）：检索和生成分开归因，否则定位不了问题在哪。
- **温度 $\tau$ 与 batch**：对比学习里大 batch（更多 in-batch 负例）+ 合适 $\tau$ 很关键，小 batch 效果差。

> ⚠️ **「加了 RAG 还是幻觉」** — 多半不是 LLM 的锅：检索召回了无关/错误片段，LLM「忠实地」据此作答。先查检索（recall@k、片段对不对），再怀疑生成。

## §11 复杂度与资源

| 阶段 | 计算 | 备注 |
| --- | --- | --- |
| 嵌入 doc（离线） | $O(N \cdot C_{\text{enc}})$ | 一次性，可批量；MRL 可省存储 |
| 建 ANN 索引（HNSW） | $O(N \log N)$ 建图 | 内存换查询速度 |
| 稠密召回（线上） | 经验近似对数 / 亚线性 per query | ANN 近似、非精确、非严格保证 |
| BM25 召回 | 倒排索引 $O(\lvert q\rvert \cdot \text{postings})$ | 词项级 |
| RRF 融合 | $O(R \cdot k_{\text{recall}})$ | 只排名，极廉价 |
| cross-encoder 重排 | $O(k_{\text{recall}} \cdot C_{\text{enc}})$ | 每个候选一次前向，**只对 top-k** |
| LLM 生成 | $O(L_{\text{ctx}}^2)$ prefill | context 越长越贵（见 attention 篇） |

双塔的核心红利：doc 编码 $O(N)$ 但**离线一次**，线上每 query 只编码 1 次 + ANN 查询；交叉编码器无法预计算，所以**只能用在 top-k 重排**，绝不能拿来扫全库。

## §12 25 高频面试题

按难度分三档，点开看答案要点 + 易踩坑。

### L1必会题（任何用过 RAG 的岗位）

<details>

<summary>Q1. 什么是 RAG？解决什么问题？</summary>

- 检索增强生成：先检索相关片段、再让 LLM 基于片段生成
- 缓解：知识截止 / 幻觉 / 私有数据（缓解事实性幻觉 + 提升可溯源，非保证消除幻觉）
- 知识更新只需更新库，可溯源

只说「让 LLM 查资料」，说不清「检索→拼 prompt→生成」三步和它应对的具体问题。

</details>

<details>

<summary>Q2. RAG 的基本流程有哪些步骤？</summary>

- 离线：文档切块 → 嵌入 → 建索引
- 线上：query 嵌入 → 召回 top-k → （重排）→ 拼 prompt → LLM 生成
- 可选：query 改写、来源标注、自检

漏掉切块/重排，或把召回和重排混为一谈。

</details>

<details>

<summary>Q3. 双塔（bi-encoder）和交叉编码器（cross-encoder）区别？</summary>

- 双塔：query/doc 分开编码，可预计算 doc 向量，快，用于召回
- 交叉编码器：拼一起编码，全程 attention，准但不可预计算，用于重排
- 召回用双塔、重排用交叉编码器是标准搭配

说交叉编码器也能建索引扫全库（错，它不可预计算）。

</details>

<details>

<summary>Q4. 为什么需要重排（rerank）？</summary>

- 召回追求高 recall + 快，精度不够
- 交叉编码器对 top-k 精确打分，提升精度
- 只对召回的少量候选跑，代价可控

以为召回就够了，不知道重排常是高 ROI 的一步。

</details>

<details>

<summary>Q5. 稀疏（BM25）和稠密检索各自强在哪？</summary>

- BM25：词项精确匹配，强在罕见词/实体/序列号
- 稠密：语义/同义/改写匹配
- 互补，工业界多用混合

说稠密全面碾压 BM25（错，精确匹配场景 BM25 常更好）。

</details>

<details>

<summary>Q6. 什么是 chunking？块大小怎么影响效果？</summary>

- 把文档切成片段再嵌入
- 太大：检索粗、稀释相关信息
- 太小：上下文断裂
- 常 256/512 token + 重叠

把 chunk 当无关紧要的预处理，不知道它是头号调参旋钮。

</details>

<details>

<summary>Q7. 相似度为什么常用余弦 / 归一化点积？</summary>

- L2 归一化后点积 = 余弦
- 只看方向不看模长，更稳（阈值仍需校准）
- 多数嵌入归一化后用点积

不知道归一化后点积就是余弦，或答不出归一化的好处。

</details>

<details>

<summary>Q8. RAG 和直接把文档塞进长上下文有何不同？</summary>

- RAG：先检索相关片段，prompt 短、可溯源、库可更新
- 长上下文：塞全文，贵、慢、有 lost-in-the-middle
- 大规模 / 动态知识下 RAG 通常更经济可控（全文也塞不下）

以为有长上下文就不需要 RAG（库一大就塞不下、也贵）。

</details>

<details>

<summary>Q9. 向量库为什么用近似最近邻（ANN）？</summary>

- 库大时暴力算全部 $O(N)$ 太慢
- HNSW 等 ANN 把查询降到经验上近似对数 / 亚线性（非严格保证）
- 用一点召回率换巨大速度

说向量检索是精确最近邻（大库下基本都是近似）。

</details>

<details>

<summary>Q10. 「加了 RAG 还是幻觉」可能的原因？</summary>

- 多半是检索召回了无关/错误片段，LLM 据此「忠实」作答
- 也可能 chunk 太碎、prompt 没强约束「只依据 context」
- 先查检索质量，再怀疑生成

直接归咎 LLM，不去查检索那一层。

</details>

### L2进阶题（research / 工程深入）

<details>

<summary>Q11. 写出 InfoNCE 损失并解释温度 $\tau$。</summary>

- $\mathcal{L} = -\log \frac{\exp(\text{sim}(q,d^+)/\tau)}{\sum_i \exp(\text{sim}(q,d_i)/\tau)}$
- 本质是相似度当 logits、正例当 label 的交叉熵
- $\tau$ 小 → softmax 更尖 → 更看重最难负例；太小对噪声敏感

只会写公式，讲不出「它就是 softmax 交叉熵」和 $\tau$ 的锐度作用。

</details>

<details>

<summary>Q12. In-batch negatives 是什么？为什么高效？</summary>

- batch 内每个 query 用其它样本的正例当负例
- 一次前向拿到 $B-1$ 个负例，几乎零额外成本
- 相似度矩阵对角线是正例，label = arange(B)

说要单独前向算负例（错，in-batch 复用同 batch）。

</details>

<details>

<summary>Q13. 难负例为什么重要？false negative 陷阱是什么？</summary>

- 随机负例太易、梯度小；难负例（像但不相关）信息量大
- false negative：挖到的「难负例」可能其实相关 → 错误梯度
- 缓解：阈值过滤、cross-encoder 剔除、软标签蒸馏

只说「难负例好」，不知道它可能引入 false negative 噪声。

</details>

<details>

<summary>Q14. RRF 怎么融合多路检索？为什么用排名不用分数？</summary>

- $\text{RRF}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$，$k\approx60$
- 只用名次，免去不同检索器分数量纲对齐
- 简单、稳、无需训练

试图直接加 BM25 分数和余弦分数（量纲不可比）。

</details>

<details>

<summary>Q15. Matryoshka 表征解决什么？怎么训？</summary>

- 一次训练得到嵌套嵌入，前 $m$ 维也是好嵌入
- 多粒度损失 $\sum_m \mathcal{L}(\text{emb}[:m])$
- 部署按预算截断维度（截断后需重新 L2 归一化；省存储/加速粗排）

以为是事后 PCA 降维（MRL 是训练时就把重要语义压进前缀）。

</details>

<details>

<summary>Q16. BM25 里 $k_1$ 和 $b$ 分别控制什么？</summary>

- $k_1$：词频饱和（出现多次的边际收益递减）
- $b$：文档长度归一化强度（$b=1$ 全归一、$b=0$ 不归一）
- 典型 $k_1\approx1.2$–$2$，$b\approx0.75$

说不清词频饱和与长度归一这两个机制。

</details>

<details>

<summary>Q17. ColBERT 的 late interaction 与双塔、交叉编码器的关系？</summary>

- 双塔：单向量、仅最后点积（无 token 级交互）、可预计算
- 交叉编码器：early、全 attention、最强、不可预计算
- ColBERT：token 级 late interaction，存 token 向量、MaxSim $\sum_i\max_j E_{q_i}^\top E_{d_j}$，介于两者、可预计算 doc

把 ColBERT 当普通双塔，不知道它是 token 级 MaxSim。

</details>

<details>

<summary>Q18. HyDE 是什么思路？为什么有用？</summary>

- 先让 LLM 生成一个「假设文档 / 段落」，用它的嵌入去检索
- 让查询向量分布对齐真实文档（伪文档比原始问题更像语料里的文档）
- 零样本稠密检索常因此提升

以为 HyDE 是改写 query 关键词（其实是生成假设文档再嵌入）。

</details>

<details>

<summary>Q19. 怎么评测一个 RAG 系统？</summary>

- 检索层：recall@k、nDCG、MRR
- 生成层：faithfulness、context relevance、answer relevance（如 RAGAS）
- 分层归因：先看检索对不对，再看生成

只看端到端答案对错，不分检索/生成两层。

</details>

<details>

<summary>Q20. lost in the middle 是什么？怎么缓解？</summary>

- 长 context 里放中间的关键信息易被模型忽略（Liu 2023）
- 缓解：重排后把最相关片段放头尾、压缩 context、减少无关片段
- 也是「召回越多越好」不成立的原因之一

以为 context 越长越全越好，忽略中间信息被弱化。

</details>

### L3高级题（顶级 lab / 深水区）

<details>

<summary>Q21. 为什么交叉编码器不能拿来扫全库？复杂度差在哪？</summary>

- 交叉编码器 doc 表示依赖 query，**无法离线预计算**
- 每个 (query, doc) 对都要一次完整前向，全库 = $O(N \cdot C_{\text{enc}})$/query
- 双塔 doc 离线编码、线上只 ANN 查询（经验近似对数 / 亚线性）；所以交叉编码器只配重排 top-k

说交叉编码器也能建索引（它的表示和 query 耦合，建不了静态索引）。

</details>

<details>

<summary>Q22. 对比学习里 batch size 为什么影响很大？和负例的关系？</summary>

- in-batch 负例数 = batch-1，batch 越大负例越多、对比越强
- 大 batch 让 InfoNCE 的分母覆盖更多负例，估计更准
- 小 batch 负例少、易过拟合；可用 cross-batch memory / 难负例补偿

不知道 batch 大小直接决定 in-batch 负例数量。

</details>

<details>

<summary>Q23. GraphRAG 相比朴素 RAG 强在哪？代价？</summary>

- 朴素 RAG 按片段独立召回，弱在「跨文档全局问题」
- GraphRAG 用 LLM 抽实体-关系图 + 社区摘要，全局问题走摘要
- 代价：建图/摘要的离线 LLM 成本高、维护复杂

把 GraphRAG 当「换个向量库」，不知道它建的是知识图 + 社区摘要。

</details>

<details>

<summary>Q24. Self-RAG / CRAG 这类「自检式 RAG」在做什么？</summary>

- Self-RAG：训模型输出反思 token（要不要检索 / 片段有没有用 / 答案有没有依据）
- CRAG：在**检索层**用 evaluator 给已检索文档的相关性/置信度打分（推理时无标注全集、测不了真 recall），低置信时触发纠正检索（web 搜索、重写）
- 目标：让 RAG 在「检索没用/错」时不盲目据此作答

以为是更好的检索器；其实是加自我评估/纠正回路——Self-RAG 侧重生成时反思，CRAG 侧重评估检索质量并触发纠正检索。

</details>

<details>

<summary>Q25. 同一套知识，什么时候选 RAG、什么时候选微调、什么时候靠长上下文？</summary>

- 注入大量**可更新外部事实** + 要溯源 → RAG
- 改**能力/风格/格式/输出结构** → 微调
- 单次少量文档、要全局推理且放得下 → 长上下文
- 实务常组合：微调调风格 + RAG 注事实 + 长上下文容纳召回

绝对化地说「RAG 全面优于微调」或反之，不看「注入事实 vs 改能力」的本质区分。

</details>

## §A 附录：sanity check

`info_nce_in_batch` 与 mini-RAG 的关键不变量（可写脚本验证）：

- **对角线即正例**：相似度矩阵 $qd^\top$ 的对角线应是每行最大（训练后），`cross_entropy(logits, arange(B))` 收敛即对齐。
- **归一化**：编码输出 $\lVert\mathbf{q}\rVert\approx1$，点积落在 $[-1,1]$。
- **温度方向**：固定 logits、调小 $\tau$，softmax 概率应更集中（熵更低）。
- **RRF 单调**：某文档在任一检索器名次提前，其 RRF 分数不应下降。
- **混合召回 ⊇ 关键命中**：BM25 能命中的精确实体片段，混合结果里应保留（纯稠密可能漏）。

下面是 [`code/rag_embedding.py`](code/rag_embedding.py) 在 **PyTorch 2.10 / CPU** 上的**真实运行**输出（每行带 `assert`，全过才打印汇总）：

```
[a] in-batch InfoNCE: argmax==idx, loss aligned 0.0000 < shuffled 22.9564  OK
[b] DualEncoder ||v|| = 1.0000 (~1), max|dot| = 1.0000 (<=1)  OK
[c] entropy(tau=0.05) = 0.0000 < entropy(tau=0.5) = 1.2945  OK
[d] with-hard loss = 0.0000 finite, logits (6, 8) == (B, B+2)  OK
[e] RRF(doc7): base = 0.03226 -> better = 0.03252 (non-decreasing); rrf top = 7  OK
[f] BM25 top = 1 (entity chunk 1); fused top2 = [1, 2]  OK

all RAG / embedding sanity checks passed ✓
```

其中 [a] 对齐时 InfoNCE loss≈0、把正例打乱后骤升到 ~23，证明损失确实在「对齐」上响应（不只是构造了对角阵）；[c] 同一组 logits 下 $\tau=0.05$ 的 softmax 熵（≈0）远低于 $\tau=0.5$（≈1.29），验证小温度更锐；[f] 含罕见实体 `Zephyrnaut9000` 的片段被 **BM25 排到第 1**、混合（RRF）结果 top-2 保留，演示混合召回对精确实体的稳健（§5.3；注：此 toy 例里稠密恰好也召回了它，真实场景罕见实体/序列号稠密更易漏）。

---

## 📜 Runnable Code

本 tutorial 的嵌入 / 检索核心在 [`docs/tutorials/code/rag_embedding.py`](code/rag_embedding.py) 有最小可跑版本：

- [`rag_embedding.py`](code/rag_embedding.py) — `DualEncoder`（L2 归一化双塔）+ `info_nce_in_batch` / `info_nce_with_hard`（in-batch + 难负例对比损失）+ `bm25_scores` / `rrf` / 混合召回，含 6 个 `assert` sanity check（对角线即正例 / 归一化 / 温度锐度 / with-hard 形状 / RRF 单调 / 混合保留实体块）。

纯 PyTorch，CPU 几秒跑完、无需 GPU：`python docs/tutorials/code/rag_embedding.py`。上方 §A 的输出即该脚本的真实运行结果。

---

## 📚 参考文献

- **RAG** — Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv 2005.11401 (2020), NeurIPS 2020.
- **DPR** — Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, arXiv 2004.04906 (2020), EMNLP 2020.
- **InfoNCE / CPC** — van den Oord et al., *Representation Learning with Contrastive Predictive Coding*, arXiv 1807.03748 (2018).
- **SimCSE** — Gao et al., *SimCSE: Simple Contrastive Learning of Sentence Embeddings*, arXiv 2104.08821 (2021), EMNLP 2021.
- **ColBERT** — Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, arXiv 2004.12832 (2020), SIGIR 2020.
- **Matryoshka Representation Learning** — Kusupati et al., arXiv 2205.13147 (2022), NeurIPS 2022.
- **BGE-M3** — Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*, arXiv 2402.03216 (2024).
- **HyDE** — Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, arXiv 2212.10496 (2022), ACL 2023.
- **Self-RAG** — Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, arXiv 2310.11511 (2023), ICLR 2024.
- **CRAG** — Yan et al., *Corrective Retrieval Augmented Generation*, arXiv 2401.15884 (2024).
- **GraphRAG** — Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv 2404.16130 (2024).
- **RAGAS** — Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv 2309.15217 (2023).
- **RRF** — Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009.
- **HNSW** — Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, arXiv 1603.09320 (2016), IEEE TPAMI 2020.
- **Lost in the Middle** — Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv 2307.03172 (2023), TACL 2024.
- **BM25** — Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in IR (2009).
- **SPLADE** — Formal et al., *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*, arXiv 2107.05720 (2021), SIGIR 2021.
- **ColBERTv2** — Santhanam et al., *ColBERTv2: Efficient and Effective Retrieval via Lightweight Late Interaction*, arXiv 2112.01488 (2021), NAACL 2022.
- **PLAID** — Santhanam et al., *PLAID: An Efficient Engine for Late Interaction Retrieval*, arXiv 2205.09707 (2022), CIKM 2022.
- **Contextual Retrieval** — Anthropic, *Introducing Contextual Retrieval* (engineering blog, 2024).
- **Late Chunking** — Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*, arXiv 2409.04701 (2024).
- **BEIR** — Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, arXiv 2104.08663 (2021), NeurIPS 2021 D&B.
- **MTEB** — Muennighoff et al., *MTEB: Massive Text Embedding Benchmark*, arXiv 2210.07316 (2022), EACL 2023.
