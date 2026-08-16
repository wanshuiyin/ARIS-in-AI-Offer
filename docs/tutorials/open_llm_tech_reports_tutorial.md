# 近期开放权重大模型技术报告教程

> 检索截止：2026-08-11。本文优先采用模型团队发布的技术报告、模型卡和代码仓库。
> “开放权重”不等于 OSI 意义上的“开源”：有些模型公开权重和推理代码，但训练数据、完整训练代码或许可证仍有限制。

## 0. 先给结论：这一代模型在解决什么问题

近期前沿开放模型不再只靠“更多参数 + 更多 token”。技术主线已经变成五条彼此耦合的轴：

1. **稀疏宽度扩展**：用 MoE 把总参数做大，但控制每个 token 的激活参数量。
2. **混合序列建模**：用线性/递归/稀疏注意力处理绝大多数层，周期性保留全局注意力。
3. **跨层信息流**：传统残差只有一条累加状态；新方法让深层显式读取浅层表示。
4. **训练与部署协同**：Muon、负载均衡、FP4/FP8 QAT、MTP 不再是附属优化，而是模型设计的一部分。
5. **面向 Agent 的后训练**：SFT 只负责冷启动，主能力来自长轨迹 RL、可验证环境、不同推理预算以及多专家能力合并。

如果只记住一句话：**前沿模型正在把“每个 token 做一次昂贵的全局 Transformer 计算”改造成“按需路由专家、按需读取历史、按需分配推理预算”的系统。**

## 1. 如何读一份大模型技术报告

按下面六层读，避免被 benchmark 表格牵着走：

| 层次 | 核心问题 | 常见指标 |
|---|---|---|
| 规模 | 总容量和单 token 成本分别多大？ | 总参数、激活参数、层数、训练 token |
| 序列混合 | 当前 token 如何读取历史？ | Full/MLA/Linear/Sparse Attention、上下文长度、KV cache |
| 通道混合 | 一个 token 经过哪些 FFN/专家？ | 专家数、Top-k、共享专家、负载均衡 |
| 深度混合 | 深层如何保留浅层信息？ | Residual、Hyper-Connection、AttnRes |
| 优化与数据 | 如何稳定训练并安排数据？ | AdamW/Muon、LR schedule、数据配比、context curriculum |
| 后训练 | 能力和行为如何形成？ | SFT、RL、reward、distillation、reasoning effort |

### 1.1 先算三个比例

设总参数为 $P_{total}$，单 token 激活参数为 $P_{active}$，专家总数为 $E$，每 token 选择 $k$ 个专家：

$$
r_{active}=\frac{P_{active}}{P_{total}},\qquad
r_{expert}=\frac{k}{E},\qquad
\text{TPP}=\frac{N_{tokens}}{P_{total}}.
$$

- $r_{active}$ 越低，说明容量与单 token 计算解耦越强；但路由、通信和显存容量问题更难。
- $r_{expert}$ 只描述专家稀疏度，不等于整个模型的激活比例，因为注意力、共享专家、embedding 仍会激活。
- TPP 不能脱离模型结构直接横向比较；MoE 的“总参数”和“激活参数”会导出两种不同口径。

## 2. 模型地图

| 模型 | 发布时间/材料状态 | 规模 | 上下文 | 最值得学的点 |
|---|---|---:|---:|---|
| Kimi-K3 | 2026-07，完整官方报告 | 2.8T / 104B active | 1M | KDA、AttnRes、Stable LatentMoE、长轨迹 RL |
| LongCat-2.0 | 2026-07，官方技术博客/模型卡 | 1.6T / 约 48B active | 1M | LSA、跨层索引、N-gram Embedding、国产 ASIC 训练 |
| DeepSeek-V4 | 2026-04，官方报告/模型卡 | Pro 1.6T/49B；Flash 284B/13B | 1M | CSA+HCA、mHC、Muon、GRPO+OPD |
| Qwen3.6 | 2026-04，模型卡；架构继承 Qwen3.5 | Dense 27B；MoE 35B-A3B | 原生 256K，可扩展约 1M | Gated DeltaNet + Gated Attention、原生多模态、preserved thinking |
| GLM-4.7 | 2026，发布博客；完整架构报告仍以 GLM-4.5 为基线 | 355B/32B；Flash 30B/3B | 版本相关 | thinking/tool-use 接口与 Agent 后训练；不宜据博客臆测新架构 |

下面重点讲前四个；GLM-4.7 放在“证据边界”中说明。

## 3. Kimi-K3：把 token、channel、depth 三个方向一起扩展

### 3.1 总体结构

Kimi-K3 是 93 层、2.78T 总参数、104.2B 激活参数的原生多模态 MoE。hidden size 为 7168，词表 160K。主干由 69 个 KDA 层和 24 个 Gated MLA 层构成；每个基本块采用 **3 个 KDA + 1 个 Gated MLA**，最后额外放置全局注意力层。

可以把一次 block 理解成：

```text
历史 token --KDA--> 低成本连续状态更新 --┐
历史 token --MLA--> 周期性全局内容读取 ----┼--> AttnRes 跨层读取 --> Stable LatentMoE
图像/视频 --MoonViT-V2 + projector-------┘
```

三个扩展方向分别是：

- **序列方向**：KDA + Gated MLA。
- **深度方向**：Attention Residuals。
- **通道方向**：Stable LatentMoE。

### 3.2 KDA：有选择地遗忘和写入

Kimi Delta Attention 维护固定大小的递归状态 $S_t$。简化后：

$$
S_t=(I-\beta_t k_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
     +\beta_t k_tv_t^\top,
\qquad
o_t=S_t^\top q_t.
$$

直觉：

- $\alpha_t$ 是逐通道遗忘门，决定旧状态保留多少。
- $\beta_t$ 决定当前 key-value 写入强度。
- $(I-\beta_t k_tk_t^\top)$ 会先擦除与当前 key 冲突的旧记忆，再写入新 value，因此比普通线性 attention 的简单累加更能处理覆盖关系。

K3 将 log-decay 限制在 $(-5,0)$，使 BF16 下的 chunk 内缩放保持有限，从而让对角块也能使用 Tensor Core 矩阵乘法。KDA 是固定状态、近线性序列成本，但其记忆被压缩；因此每四层插入一次 MLA，重新提供不受限的全局 token-to-token 交互。

### 3.3 Gated MLA：用低维 latent 压缩 KV cache

MLA 先把 token 压缩为 $c_t=W_cx_t$，缓存低维 $c_t$，需要时再上投影为各头的 key/value。K3 的 MLA 不使用显式位置编码；位置和近因信息主要由 KDA 的递归门承担。

这体现了很重要的分工：

- KDA 负责便宜、连续、位置敏感的记忆更新；
- MLA 负责较稀疏但高容量的全局检索；
- 输出 gate 决定当前 token 真正读取哪些通道。

### 3.4 AttnRes：把“层深”也当成一条可注意的序列

普通残差为 $h_l=h_{l-1}+f_l(h_{l-1})$，所有过去信息被挤进单个状态。AttnRes 让第 $l$ 层对更早层的表示做 softmax 加权：

$$
\alpha_{i\rightarrow l}=
\frac{\exp(w_l^\top\operatorname{RMSNorm}(k_i))}
{\sum_{j<l}\exp(w_l^\top\operatorname{RMSNorm}(k_j))},
\qquad
h_l=\sum_{i<l}\alpha_{i\rightarrow l}v_i.
$$

K3 实际使用 block 级版本：93 层按约 12 层一组，跨 block 注意、block 内累加，把保存全部层输出的 $O(Ld)$ 开销降到 $O(Nd)$。它解决的是深层网络的“信息沉淀与冲淡”，而不是长上下文问题。

### 3.5 Stable LatentMoE：896 选 16 为什么还能训练稳定

K3 有 896 个 routed experts，每 token 选 16 个，另有 2 个 shared experts。路由专家在 3584 维 latent 空间工作，而主 hidden size 是 7168：这用较窄的专家通道换取更大的专家池。

极稀疏 MoE 的三个难点及对应解法：

1. **专家链路激活爆炸**：用有界的 SiTU-GLU 替代 SwiGLU，使单坐标输出有上界。
2. **近千专家难以均衡**：Quantile Balancing 直接估计每个专家的路由阈值分位点，而不是依赖慢速的 sign bias 更新。
3. **专家并行通信不均衡**：MoonEP 通过冗余专家放置和 token 调度形成静态、完全均衡的计算形状。

### 3.6 预训练策略

- 文本覆盖 Web、Code、Math、Knowledge；视觉覆盖 caption、图文交错、OCR、感知、视频和 visual coding。
- 图文从训练开始共同进入 next-token prediction，不是训练完 LLM 后再接视觉塔。
- optimizer：矩阵参数使用 **Per-Head Muon**；Q/K/V 的 momentum 按 head 分块正交化，避免大梯度 head 支配整个矩阵更新。
- cosine LR，1% linear warmup，weight decay 0.1。
- 上下文课程：8K → 64K 预训练，再在 cooldown 阶段扩展到 256K → 1M。
- 长上下文数据不仅是长文拼接，还构造必须跨远距离取证才能完成的任务，否则模型可能只学到局部模式。

### 3.7 后训练策略

```text
SFT 冷启动
  ↓
3 个领域 × 3 个 reasoning effort 的 9 个 RL 专家
  ↓
Multi-Teacher On-Policy Distillation
  ↓
统一 Kimi-K3
```

三个 RL 领域是 general、general agents、coding agents；推理档位为 low/high/max。长轨迹采用 partial rollout：完成一定比例就先更新，未完成轨迹保留 KV cache 和 sandbox 状态，在后续 iteration 恢复。

Reasoning-effort RL 不是简单截断输出，而是给每题设置基准预算 $b_0(x)$，当轨迹超过 $\tau b_0(x)$ 时覆盖奖励；逐渐减小 $\tau$ 得到 high/low 专家。最终用多教师 on-policy distillation 合并九个策略。

值得迁移的工程思想：**如果 Agent rollout 很长，就把 rollout 当成可恢复的有状态任务，而不是一次性 completion。**

## 4. DeepSeek-V4：为 1M 上下文重做注意力与残差

### 4.1 规模与目标

- V4-Pro：1.6T 总参数，49B active。
- V4-Flash：284B 总参数，13B active。
- 均为 1M context；预训练超过 32T token。
- 延续 DeepSeekMoE 和 MTP，核心变化在 attention、residual 和 optimizer。

### 4.2 CSA + HCA 混合注意力

DeepSeek-V4 用两种角色不同的层组合长上下文：

- **Compressed Sparse Attention (CSA)**：通过轻量 indexer 从历史位置中选择少量关键 token，再只对选中 KV 做精确 attention。目标是降低随上下文长度增长的计算量。
- **Heavily Compressed Attention (HCA)**：把历史压缩到更小的 latent/cache 表示，提供便宜、稳定的全局通路。

学习时要区分两个“压缩”：

```text
CSA：压缩被读取的位置数量（selection sparsity）
HCA：压缩每个历史位置的表示宽度或总体状态（representation compression）
```

两者结合的原因是纯 sparse selection 可能漏召回，纯 compressed memory 又可能损失精细信息。官方报告给出的系统级结果是：1M context 下，V4-Pro 的单 token inference FLOPs 约为 V3.2 的 27%，KV cache 约为 10%。这不是说整体训练成本下降到相同比例。

### 4.3 mHC：让残差混合既更强又受约束

Manifold-Constrained Hyper-Connections 不再只保留单条 residual stream，而是在多条隐藏流之间学习混合。关键不是“多几条残差”，而是把混合矩阵限制在特定流形/约束集合，使信号尺度在深层传播时稳定。

它和 K3 AttnRes 的共同目标都是改善跨深度信息流，但机制不同：

- AttnRes：显式注意并检索过去层/block 的输出。
- mHC：维护多条状态流并学习受约束的混合变换。

### 4.4 Muon 与 MoE 路由

V4 将 Muon 用于主要二维矩阵参数，通过对 momentum 做近似正交化，使更新方向更均衡；其他不适合矩阵正交化的参数仍需 AdamW 类优化。MoE 延续 DeepSeek 的无辅助损失均衡思想，避免显式 balance loss 与语言建模目标冲突。

### 4.5 训练与后训练

官方披露的后训练分两阶段：

1. 各领域分别进行 SFT 和基于 **GRPO** 的 RL，形成 domain experts。
2. 用 **on-policy distillation** 把各领域专家合并回统一模型。

这和 K3 的“9 个 effort/domain 专家再 MOPD”同属一个趋势：先允许策略专业化，再以在线采样分布做能力合并。相比在同一个 policy 上混合全部 reward，它降低不同任务梯度直接冲突的风险。

部署侧，instruct 权重将 MoE expert 压到 FP4、其他多数参数保持 FP8；这再次说明量化误差需要在后训练阶段显式适配，而不能只靠发布后的 PTQ。

## 5. Qwen3.5/3.6：混合线性注意力进入实用型 Dense/MoE 模型

### 5.1 为什么把 3.5 和 3.6 一起读

Qwen3.6 的公开材料把主要能力升级描述为 agentic coding 和 preserved thinking，架构基础来自 Qwen3.5。不要把 3.6 benchmark 提升自动归因于新 backbone；目前公开证据更支持“沿用混合架构，强化训练和产品行为”。

### 5.2 以 Qwen3.6-27B 为具体例子

- Dense 27B，64 层，hidden size 5120。
- 结构为 16 次重复：`3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN)`。
- 原生 context 262,144，可扩展至约 1.01M。
- 自带视觉 encoder，属于原生 vision-language foundation。
- MTP 进行多步训练，为 speculative decoding 做准备。

Gated DeltaNet 与 KDA 同属 delta-rule 递归家族：用固定状态承载大部分长序列混合，再周期性插入 softmax attention 修复精确信息检索。这里最值得学的不是某一个公式，而是 **3:1 混合层比例已成为工程上可落地的折中**。

### 5.3 多模态训练和 Agent RL

Qwen3.5 官方概述披露：

- 在 trillions 级多模态 token 上做 early fusion；
- Gated Delta Networks 与 sparse MoE 结合；
- RL 扩展到 million-agent environments，并采用异步环境/rollout 编排；
- 支持 201 种语言和方言。

Preserved Thinking 的含义是保留历史回合的 reasoning context，避免每次工具调用后重新推导。它是训练格式、chat template、KV cache 和 Agent harness 的联合设计，不只是 UI 是否显示思维链。

## 6. LongCat-2.0：把稀疏注意力的 indexer 本身也工程化

### 6.1 模型概览

LongCat-2.0 为 1.6T 总参数、约 48B active 的 MoE，预训练超过 35T token；官方称完整训练运行在 AI ASIC superpod 上，并用数千亿 1M-context token 强化长程能力。权重采用 MIT License。

### 6.2 LongCat Sparse Attention 的三层优化

标准动态稀疏注意力的隐性成本是：为了选 Top-k 历史 token，indexer 自己可能要扫描很大候选集，而且选中的 KV 地址高度离散。LSA 依次处理三个瓶颈：

1. **Streaming-aware Indexing (SI)**：把连续块访问与动态随机选择结合，让 HBM 访问更连续。
2. **Cross-Layer Indexing (CLI)**：相邻层共享一次索引；训练时用跨层蒸馏，让被共享的 index 仍能服务后续层。
3. **Hierarchical Indexing (HI)**：先 block 粗召回，再在候选 block 内做 token 级细排。

可以把复杂度理解成：

```text
全扫描：所有历史 token -> 精确打分 -> Top-k
HI：历史 blocks -> 低成本粗排 -> 少量候选 token -> 精排
CLI：若干相邻层复用上面一次结果
SI：把最终读取排成硬件友好的连续访问
```

这三个技巧分别优化算法候选规模、跨层重复计算和内存访问，并非互相替代。

### 6.3 N-gram Embedding

LongCat-2.0 额外加入 135B N-gram Embedding 参数。普通 token embedding 只查当前 token；N-gram embedding 根据局部 token 组合查稀疏表，为常见短语、代码片段或局部模式提供额外容量。

它与 MoE 的区别：

- MoE 根据 hidden state 动态路由，发生在网络深处，计算与通信较重。
- N-gram embedding 根据离散局部模式稀疏查表，发生在输入侧，计算便宜但泛化形态不同。

当 MoE 已经非常稀疏，再继续增加专家可能遭遇通信和路由收益递减；把一部分参数预算投到正交的稀疏维度，是 LongCat 的核心判断。

## 7. 四条横向技术主线

### 7.1 混合注意力已成为主流答案

| 模型 | 廉价主通路 | 高容量校正通路 |
|---|---|---|
| Kimi-K3 | KDA recurrent state | 周期性 Gated MLA |
| Qwen3.5/3.6 | Gated DeltaNet | 周期性 Gated Attention |
| DeepSeek-V4 | HCA/压缩全局状态 | CSA 精细稀疏读取 |
| LongCat-2.0 | 分层 sparse index | 选中 token 的精确 attention |

共同规律：**便宜通路保证覆盖，高容量通路保证精度。** 单用线性记忆容易信息压缩，单用动态稀疏选择容易漏召回，单用 full attention 又不可扩展。

### 7.2 MoE 的瓶颈已经从 FLOPs 转向系统

MoE 降低的是每 token 的算术量，不会自动降低：

- 模型权重总显存；
- all-to-all 通信；
- expert load imbalance；
- 小 batch 下的专家 GEMM 利用率；
- 分布式故障恢复成本。

因此 K3 强调 Quantile Balancing 和 MoonEP，DeepSeek 强调无辅助损失路由与低精度专家，LongCat 则探索在 MoE 之外加入 N-gram sparse capacity。

### 7.3 后训练从“偏好对齐”转为“环境中的策略学习”

新一代 RL 的训练单位不再只是 prompt-response，而是：

$$
\tau=(s_0,a_0,o_1,a_1,o_2,\ldots,s_T),
$$

其中 action 可以是工具调用，observation 是真实环境反馈，reward 由代码测试、检索证据、最终文件、像素相似度或隐藏 verifier 给出。系统要保存的不只是文本，还包括 sandbox、文件系统、应用状态和 KV cache。

### 7.4 推理预算成为显式条件变量

low/high/max 并非简单设置 `max_tokens`。正确做法是：

- 数据中含不同难度和不同合理长度的轨迹；
- reward 同时约束正确性与预算；
- 模型通过系统消息或 option token 感知 effort；
- 不同 effort policy 最终被蒸馏进同一模型。

## 8. 学习与复现路线

### 第一周：建立结构直觉

1. 手写一个标准 causal attention。
2. 手写一个 delta-rule recurrent layer，比较序列长度从 1K 到 64K 时的显存曲线。
3. 每 3 个 recurrent layer 插 1 个 full attention，观察 needle retrieval 是否恢复。

### 第二周：做一个小型 MoE

1. 8 个 expert、Top-2 routing、1 个 shared expert。
2. 记录每步各专家 token count、最大/最小负载和 all-to-all payload。
3. 比较 auxiliary balance loss、router bias update、分位数阈值三种方式。

### 第三周：实现深度信息流实验

在相同参数量下比较：

- 普通 residual；
- learnable weighted residual；
- block-level AttnRes；
- 多流 Hyper-Connection 的简化版。

不要只看最终 loss，还要画 layer contribution、gradient norm 和 representation similarity。

### 第四周：做可验证 Agent RL 的最小闭环

选择一个有确定 verifier 的环境，例如小型代码修复：

```text
问题 + repo state
  -> 模型调用 shell/edit/test
  -> verifier 跑隐藏测试
  -> reward
  -> GRPO 或其他 group-relative policy update
```

随后加入三个 ablation：固定 token budget、按题自适应 budget、partial rollout。记录成功率、平均 token、平均工具步数、wall time 和 reward-hacking 率。

## 9. 阅读报告时的常见误区

1. **总参数不等于推理成本**：MoE 要同时看 active params，但 active params 也不完全等于 FLOPs。
2. **1M context 不等于有效使用 1M**：需要 needle、multi-hop、长程 Agent 和真实 KV/cache 成本共同验证。
3. **稀疏 attention 图不等于因果证据**：indexer 选中某 token 只说明路由相关，不能证明该 token 对答案有因果贡献。
4. **模型分数不能脱离 harness**：SWE-bench 的工具、超时、上下文压缩和采样参数会显著改变结果。
5. **新版 benchmark 上升不等于架构升级**：若只有模型卡/博客而无完整报告，应把提升归因保留在“训练、数据、harness 或架构的综合作用”。
6. **开放权重不等于可复现训练**：必须分别检查权重、代码、数据、训练 recipe 和许可证。

## 10. 关于 GLM-4.7 与“最新”口径

GLM-4.7 已公开权重和发布博客，强调 interleaved thinking、preserved thinking、turn-level thinking、coding 和 tool use；但其公开链接仍把 GLM-4.5 报告作为完整架构报告。因此目前可以确认产品行为和模型规格，不能把未披露的训练配方或架构变化写成事实。学习 backbone 时应读 GLM-4.5 报告（355B/32B active、23T token、MoE、SFT/RL）；学习最新 Agent 接口时再看 GLM-4.7 发布材料。

## 11. 一手资料

- [Kimi-K3 官方仓库与技术报告](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi-K3 官方模型卡](https://huggingface.co/moonshotai/Kimi-K3)
- [DeepSeek-V4-Pro 官方模型卡与技术报告入口](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek-V4 Transformers 架构实现说明](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md)
- [Qwen3.6 官方仓库](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [LongCat-2.0 官方仓库](https://github.com/meituan-longcat/LongCat-2.0)
- [LongCat-2.0 官方模型卡](https://huggingface.co/meituan-longcat/LongCat-2.0)
- [GLM-4.5 Technical Report](https://arxiv.org/abs/2508.06471)
- [GLM-4.7/4.6/4.5 官方仓库](https://github.com/zai-org/GLM-4.5)

## 12. 最后形成一张心智图

```text
输入容量
├── text / image / video early fusion
├── N-gram sparse embedding
└── 大词表与视觉 token 压缩

主干容量
├── 序列：recurrent / compressed / sparse / global hybrid attention
├── 深度：AttnRes / mHC
└── 通道：shared expert + routed MoE

稳定训练
├── Muon / Per-Head Muon
├── bounded activation
├── load balancing
└── progressive context curriculum

形成能力
├── SFT cold start
├── verifiable environment RL
├── reasoning-effort conditioning
└── multi-teacher on-policy distillation

落地部署
├── FP4 expert + FP8 activation
├── MTP / speculative decoding
├── prefix/KV/state cache
└── persistent sandbox and resumable rollout
```

当你能沿这五层解释任意一个新模型，并指出哪些是报告事实、哪些只是合理推断，就已经具备快速读懂下一份技术报告的能力。
