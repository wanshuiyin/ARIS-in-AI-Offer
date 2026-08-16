# 鞠少波秋招简历知识地图

> 目标：简历中的每个技术名词、方法和数字，都能回答“是什么、为什么、怎么做、如何验证、有什么局限、你负责什么”。

## 0. 面试回答的统一框架

任何项目问题都按六步回答：

1. **问题**：业务或研究痛点是什么，原方案为什么不够。
2. **目标**：优化什么能力，约束是什么。
3. **方法**：数据、模型、训练、评测分别做了什么。
4. **机制**：为什么理论上会有效，关键公式或数据流是什么。
5. **证据**：对照组、消融、指标和样例怎样支持结论。
6. **边界**：失败模式、未验证假设、团队结果与个人贡献。

回答项目时先讲主链路，再接受面试官下钻。不要一上来堆模型名和指标。

---

# 一、P0：所有项目共同依赖的基础

## 1. 多模态大模型完整结构

必须能从输入画到输出：

```text
图片/视频
  → decode / sample frames
  → ViT / vision encoder
  → patch tokens
  → spatial merge / projector
  → 与文本 tokens 拼接
  → LLM
  → next-token logits
  → autoregressive generation
```

必须掌握：

- ViT 的 patch embedding、2D position encoding、self-attention。
- CLIP/SigLIP 式视觉预训练与生成式 VLM 的区别。
- Projector、Q-Former、cross-attention、early/late fusion 的差别。
- 动态分辨率如何改变 patch 数和视觉 token 数。
- 图片、多图和视频如何插入文本序列，media placeholder 如何与实际媒体对齐。
- causal mask、position ids、M-RoPE、padding mask。
- 视频帧数、分辨率、视觉 token 数、序列长度和 FLOPs/显存的关系。
- 训练时 teacher forcing 与推理时 autoregressive decoding 的差别。

对应教程：

- [Attention](./attention_tutorial.html)
- [Transformer Block](./transformer_block_tutorial.html)
- [VLM / Multimodal](./vlm_multimodal_tutorial.html)
- [Tokenization](./tokenization_tutorial.html)

典型追问：

- 一张图片是如何变成 LLM token 的？
- 动态视觉 token 为什么能节省计算？
- 多图/视频的 position ids 怎么构造？
- media placeholder 数量与视觉特征不一致会发生什么？
- 视觉 token 翻倍时，attention 计算量如何变化？

## 2. Transformer 与 Qwen3.5-35A3B MoE

必须掌握：

- Pre-LN/RMSNorm Transformer block 的完整数据流。
- MHA、MQA、GQA 的参数量、KV cache 和通信差别。
- RoPE/M-RoPE 的作用和边界。
- Dense FFN 与 MoE FFN 的区别。
- Router logits、softmax、Top-K expert selection。
- 总参数量与激活参数量的区别：“35A3B”约表示 35B 总参数、每 token 激活约 3B 级参数。
- routed expert 与 shared expert。
- expert capacity、load balance、routing collapse。
- EP 的 token dispatch/combine all-to-all。
- 为什么 `ep_size=1` 时仍然是 MoE，但没有 Expert Parallel 通信。
- Qwen3.5 的 full attention 与 linear/gated attention 混合结构应如何理解。
- 若实际模型继承公开 Qwen3.5-35B-A3B recipe，必须以真实 config 为准核对层数、层内组织方式、routed
  expert 数、Top-K 与 shared expert 数。类似的混合架构（如公开 Qwen3-Next-80B-A3B）常见做法是按
  `k × [3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)]` 的 3:1 比例堆叠，可作为理解混合结构的
  参照，但**不能把某个具体层数（例如 40 层）当作 DouyinVL 内部模型的既定事实**；面试时应直接读 config，
  并说明内部版本与公开配置是否一致。不能只根据模型名猜配置。

对应教程：

- [Transformer Block](./transformer_block_tutorial.html)
- [MoE](./moe_tutorial.html)
- [线性 / 稀疏注意力](./linear_sparse_attention_tutorial.html)

典型追问：

- 35B 总参数为什么只有约 3B 激活？
- MoE 是省显存还是省计算？
- Top-K router 如何训练？为什么会负载不均？
- EP 与 FSDP 切 expert 参数有什么本质区别？
- linear attention 为什么不能直接解释为标准 softmax attention map？

## 3. 训练目标与基本优化

必须掌握：

- next-token prediction 和交叉熵：

  $$
  \mathcal L_{\mathrm{NTP}}
  =-\sum_t \log p_\theta(x_t\mid x_{<t})
  $$

- causal LM 的 label shift。
- padding、prompt、媒体占位符是否进入 loss。
- Adam/AdamW、weight decay、gradient clipping。
- warmup、cosine decay、学习率与 global batch 的关系。
- bf16/fp16/fp32 master states。
- gradient accumulation。
- activation checkpointing。
- loss spike、gradient norm、NaN/Inf 的排查顺序。

对应教程：

- [优化器与 LR Schedule](./optimizer_lr_schedule_tutorial.html)
- [归一化 / 残差 / 初始化](./normalization_init_tutorial.html)
- [LLM Pretraining Pipeline](./llm_pretraining_pipeline_tutorial.html)

---

# 二、P0：DouyinVL 千卡 CPT 项目

## 4. CPT 与 SFT 的区别

必须能回答：

| 阶段 | 主要数据 | 训练目标 | 主要解决问题 |
|---|---|---|---|
| Pretraining | 大规模通用数据 | next-token prediction | 通用表征与生成能力 |
| CPT | 领域/模态混合语料 | 通常仍是 next-token prediction | 领域知识、模态和分布迁移 |
| SFT | instruction-response | response token loss | 指令遵循与输出格式 |
| Preference/RL | 偏好或在线 rollout | DPO/PPO/GRPO 等 | 行为、推理和奖励对齐 |

必须掌握：

- 为什么 CPT 能做领域迁移。
- CPT 为什么可能损伤 instruction following。
- 灾难性遗忘的来源和缓解：保留通用数据、控制学习率、采样配比、回归评测。
- token 配方与“样本数配方”的区别。
- recipe pool 规模与模型实际消费量的区别。
- epoch/cap 为什么能防止小数据集被过采样。
- 多源数据如何按 domain、modality、task、quality 分层采样。

简历口径：

- `1.547T tokens / 888.73M 样本`是 CPT v9 数据池或配方规模。
- `约 1T tokens`是实际千卡训练消费量。
- 两者不能混写成“模型训练了 1.547T tokens”。

典型追问：

- CPT 和 SFT 的 loss 有何不同？
- 1.547T 配方池为什么只训练约 1T？
- 如何判断某个数据桶需要加权还是降权？
- 如何证明提升来自数据配比，而不是训练随机性？
- 如何监控 CPT 引发的通用能力遗忘？

## 5. 多源多模态数据治理

简历中的每个检查项都要能讲出“错误样例—风险—检测方法—处理策略”：

### 视觉依赖性

- 问题是否必须看图/视频才能回答。
- 用 text-blind model 或去媒体对照检测文本可答性。
- 防止模型只靠题面、OCR、ASR、标题或数据模板猜答案。

### 答案唯一性

- MCQ 是否存在多个合理答案。
- 开放式 QA 是否有别名、粒度或时态歧义。
- 用规则、judge 和人工抽检结合。

### OCR/ASR 泄漏

- OCR/ASR 文本可能直接包含答案。
- Dropout 的目的不是随机破坏数据，而是建立视觉依赖。
- 必须解释 dropout 概率、适用样本和防御指标。

### 媒体—占位符一致性

- 文本中的 image/video placeholder 数量、顺序必须与媒体特征一致。
- 错配会造成 shape error，更隐蔽时会造成样本语义错位。

### 近重复污染

- exact hash、perceptual hash、MinHash/LSH、embedding similarity。
- 必须按源样本先切 Train/Val，再展开 prompt/intent variant。
- benchmark contamination 与普通训练集重复的区别。

### Schema 完整性

- 必填字段、类型、长度、媒体路径、时间戳、答案、source id。
- 数据校验应在昂贵训练前完成。

典型追问：

- 为什么 Option Shuffle 能减少位置偏置？如何验证？
- OCR Dropout 会不会伤害 OCR 能力？
- 如何检测 train/val 的同源视频泄漏？
- 200M 样本如何做分布式去重和质量统计？

## 6. 动态视觉 Token

必须掌握：

- 分辨率、patch size、spatial merge 与视觉 token 数的关系。
- 视频的帧数 × 每帧 token 数。
- 固定 token budget 下如何在帧覆盖与单帧细节间分配。
- aspect ratio bucketing。
- 动态 batching 与 token-based batching。
- 极长样本如何拖慢整个 batch。

要能回答：

> 为什么不能所有图片都用最高分辨率、所有视频都采最多帧？

答案要包含：

- attention/activation成本；
- batch内长度不均造成padding浪费；
- 吞吐下降；
- 细节与时间覆盖之间的预算权衡；
- 按任务和媒体复杂度动态分配。

## 7. 1024 张 910B NPU 分布式训练

真实配置：

```text
FSDP2 full shard
dp_replicate_size = 8
dp_shard_size = 1024 / 8 = 128
ulysses_size = 1
ep_size = 1
micro_batch_size = 2
global_batch_size = 2048
```

必须能画出：

```text
1024 NPUs
  = 8 replicated groups
  × 128-way FSDP shard
```

每一步必须掌握：

1. 每个 rank 读取 2 个不同样本。
2. 128 卡内，参数/梯度/优化器状态按 FSDP2 分片。
3. 每个 FSDP wrap unit 在 forward 前 all-gather 参数；wrap unit 常按 block 切，但不能不看配置就笼统说成“每层”。
4. `full_shard=true` 时 forward 后重新分片。
5. backward 前再次 all-gather 参数。
6. backward 后 reduce-scatter 梯度。
7. 128-way shard 维完成 reduce-scatter 后，8-way replicate 维还要对对应梯度分片做同步，通常表现为 replicate 维 all-reduce。
8. optimizer 在本地更新参数和 Adam state 分片。
9. DCP 直接保存分布式 checkpoint。

关键区分：

- 这是 FSDP/HSDP，不是 TP：每张卡计算时临时拿到当前 FSDP unit 的完整参数，处理不同样本。
- `ep_size=1`：没有 expert all-to-all。
- `ulysses_size=1`：没有 Ulysses sequence parallel。
- NPU 通信通常对应 HCCL 体系；NCCL 的集合通信概念仍可类比，但不要把硬件后端说错。

Batch 计算：

$$
1024\times2=2048
$$

因此理论 gradient accumulation 为 1。

吞吐口径：

$$
77B/86400 \approx 891K\ \text{tokens/s（全局）}
$$

$$
891K/1024\approx870\ \text{tokens/s/NPU}
$$

`77B tokens/day × 12 days ≈ 924B`，与“约 1T tokens”是近似一致的量级。面试时应说明 77B/day 是项目吞吐口径，1T 和 12 天均为约数；若被要求精确核算，应回到训练日志区分有效 token、padding token、实际运行时间和停机时间。

对应教程：

- [Distributed Training](./distributed_training_tutorial.html)
- [LLM Pretraining Pipeline](./llm_pretraining_pipeline_tutorial.html)

典型追问：

- 为什么 35B 模型还要用 1024 卡？
- FSDP2 与 ZeRO-3 有什么对应关系？
- 为什么选择 128-way shard + 8-way replicate？
- all-gather、reduce-scatter 分别发生在哪？
- 为什么不用 EP？不用 Ulysses 的代价是什么？
- 训练 hang、loss spike、吞吐下降怎么排查？

## 8. Loader、解码与训练稳定性

必须掌握端到端吞吐：

```text
存储读取
→ 样本索引/混合
→ 图片/视频读取
→ 视频解码与采帧
→ resize/tokenize
→ collate/padding
→ host-to-device
→ forward/backward
→ collective communication
```

必须区分：

- data starvation：加速器等待数据。
- compute-bound：模型计算饱和。
- communication-bound：集合通信占主导。
- straggler：某些 rank 因长视频/坏媒体变慢，拖住全局 collective。

排查指标：

- data time、forward time、backward time、optimizer time；
- device utilization；
- samples/s、tokens/s；
- batch token 长度分布；
- decode failure/timeout；
- all-gather/reduce-scatter耗时；
- rank间 step time 的 P50/P99。

典型追问：

- 为什么一个坏视频可能挂住 1024 卡？
- 如何定位是 Loader 慢还是通信慢？
- 超长视频如何采帧？
- checkpoint 保存为什么可能造成训练抖动？

## 9. Benchmark、回归与 badcase 回流

必须掌握：

- 136 个任务、42 个业务任务为什么要分层，而不是只看平均分。
- macro average 与按样本数量加权平均的差别。
- 能力域、数据域、模态域、业务域的分组方法。
- 自动评测、规则评测、LLM-as-judge、人工评测的适用边界。
- regression gate 和防御指标。
- benchmark 污染、judge 偏差、格式错误。
- badcase 如何映射回数据桶、采样配比或训练阶段。

数字归因：

- `79.28 → 79.87`、`79.72 → 82.37`需要能说明评测集合、聚合方式、对照版本和是否同一训练预算。
- `相对 Qwen3.5 基座综合 +10.98pt`属于团队版本结果。
- `抖音推理 Base→Instruct +46.91pt`需要解释 Base/Instruct 的版本、任务集合和为什么增益很大。
- 不要把团队最终指标说成个人单独实验结果。

对应教程：

- [LLM Evaluation & Benchmarking](./llm_evaluation_benchmarking_tutorial.html)

---

# 三、P0：Post-Train 完整链路

## 10. CPT → Long-CoT SFT → Reasoning RL → General RL

必须能说明每个阶段解决不同问题：

### CPT

- 领域知识和多模态分布迁移。
- 不保证自然获得强指令遵循和稳定推理格式。

### Long-CoT SFT

- 用高质量长推理轨迹冷启动。
- 教会模型输出推理过程和任务格式。
- 风险：模仿错误推理、过长输出、风格固化。

### Reasoning RL

- 用可验证或模型判定奖励强化专项推理。
- 需要掌握 PPO/GRPO 的基本目标、advantage、KL约束。
- 风险：reward hacking、长度膨胀、格式投机。

### General RL

- 在通用帮助性、安全性、格式和业务行为上再平衡。
- 防止专项推理强化后只会长思考或牺牲普通任务表现。

个人边界：

- 可以完整解释团队 Post-Train 逻辑。
- 个人参与是协同/部分参与，不表述为全链路独立 owner。

对应教程：

- [RLHF / DPO / GRPO / PPO](./rlhf_dpo_grpo_ppo_tutorial.html)
- [KL Divergence in RLHF](./kl_divergence_rlhf_tutorial.html)
- [Reasoning Models](./reasoning_models_tutorial.html)

---

# 四、P0：Intent Caption 项目

## 11. 条件证据压缩的问题定义

必须能清楚区分普通 Caption 与 Intent Caption：

```text
普通 Caption：
尽量全面描述图像

Intent Caption：
在固定文本预算下，根据 Intent 保留对下游决策有用、
视觉可验证、非冗余的原子事实
```

形式化：

$$
c^*
=\arg\max_{c:\,|c|\le B}
\left[
\text{Evidence}(c)
+\text{IntentRelevance}(c,i)
+\text{Faithfulness}(c)
-\text{Redundancy}(c)
\right]
$$

必须掌握：

- 原子事实、claim、evidence、intent 的定义。
- Evidence Recall 和 Claim Precision 的矛盾。
- 固定长度预算为什么使其成为压缩/选择问题。
- Intent-aware 与把 Intent 文本简单拼进 prompt 的区别。
- 下游任务相关性如何通过 Evidence QA 定义。

## 12. 数据生产与质量门禁

完整链路：

```text
源视觉样本
→ Intent 构造
→ 9类 Evidence QA
→ Visual QA 验证
→ Blind QA 验证
→ Claim Validation
→ World Knowledge 标注
→ 源样本级 Train/Val 划分
→ 展开不同 intent/variant
```

必须解释：

- Visual QA：看图后问题是否可回答。
- Blind QA：不看图时是否仍可回答，用于测视觉依赖。
- Claim Validation：caption claim 是否由视觉证据支持。
- World Knowledge：视觉事实与外部知识如何分离。
- 为什么先按3,995个源样本切分，再展开为19,665条训练数据。
- 为什么115,793条Evidence QA不能当成115,793个独立视觉样本。

## 13. IEPR 奖励

对外名称：

> IEPR = Intent-aware Evidence Preservation Reward

必须能解释各项：

$$
R_{\mathrm{IEPR}}
=w_rR_{\mathrm{evidence}}
+w_pR_{\mathrm{precision}}
+w_iR_{\mathrm{intent}}
+w_lR_{\mathrm{length}}
-w_dR_{\mathrm{redundancy}}
$$

重点不是背权重，而是解释：

- 只有 Recall：模型会堆积事实、变长。
- 只有 Precision：模型可能输出极短、保守内容。
- 只有 Intent Relevance：模型可能重复 Intent 或生成不可验证内容。
- 长度效率约束文本预算。
- 冗余惩罚防止同义重复刷分。

必须掌握 multi-objective reward 的：

- 尺度归一化；
- reward权重；
- clipping；
-缺失judge结果；
- reward方差；
- reward hacking；
- 分项reward监控。

## 14. GRPO 强化学习

必须掌握：

1. 同一个prompt采样一组responses。
2. Judge和规则计算每个response的reward。
3. 在组内做相对标准化：

   $$
   A_i=\frac{r_i-\mu_G}{\sigma_G+\epsilon}
   $$

4. 提高正advantage response概率，降低负advantage response概率。
5. 用ratio clipping和KL约束训练稳定性。

必须能比较：

| 方法 | 是否需要Critic | 数据来源 | 核心信号 |
|---|---:|---|---|
| SFT | 否 | 离线答案 | token imitation |
| DPO | 否 | 离线偏好对 | chosen vs rejected |
| PPO | 是 | on-policy rollout | critic advantage |
| GRPO | 否 | on-policy group | group-relative reward |

## 15. Actor–Judge–Reward–Update 链路

必须能画出：

```text
Prompt + Image
  → Actor生成G个caption
  → Caption Judge
  → Visual Judge
  → Evidence QA评分
  → Claim/Intent/Length/Redundancy组合reward
  → group-relative advantage
  → GRPO update
```

工程追问：

- grouped judge 为什么比逐条judge省请求？
- grouped judge的顺序偏置如何控制？
- 流式reward如何避免等待所有样本？
- 请求缓存的key应包含哪些版本信息？
- 429为什么不能盲目拆成更多并发单请求？
- 模型训练吞吐和外部judge TPM为什么是两个独立瓶颈？

## 16. 如何解释训练趋势

简历写的是：

- 后段相较前段 Reward +13.22pp；
- QA Recall +16.75pp；
- 幻觉率下降2.63pp。

必须说明：

- 这是100-step内部前后阶段趋势。
- 它不天然等于同一held-out slice上的严格Step 0→Step 100因果增益。
- 需要固定request、Judge版本、prompt、采样参数和评测集合，才可做严格配对比较。
- 同时检查输出长度是否增加，避免“更长所以Recall更高”。

## 17. 两阶段拒绝采样与RFT

链路：

```text
greedy生成
→ 筛低Recall/高幻觉困难prompt
→ 对困难prompt做Best-of-N
→ Precision/Relevance/幻觉作为约束
→ 在可接受候选中选最高Recall
→ 形成RFT/SFT数据
```

必须掌握：

- 为什么先筛困难样本，避免对简单样本浪费采样预算。
- Best-of-N提高样本质量但增加推理成本。
- selection bias：只保留高reward答案可能放大judge偏差。
- RFT是offline imitation，不等于online RL。

## 18. 统一 Benchmark 与评测审计

简历中的独立评测资产不能只当作一个数字带过：

- 399 个独立视觉样本；
- 2,692 条通用 Evidence QA；
- 187 条世界知识题；
- Recall、Claim Precision、Intent Relevance、Evidence Alignment、World F1 等多维指标。

必须掌握：

- **独立统计单位**是源视觉样本、intent、QA 还是 claim；置信区间和显著性检验应在真正独立的单位上做。
- **Evidence Recall**衡量参考证据覆盖，**Claim Precision**衡量输出 claim 的视觉支持率，二者不能相互替代。
- **Evidence Alignment**要说明是在 claim–evidence、caption–QA 还是 judge score 层面定义，必须以真实实现为准。
- **World F1**要说明实体别名、同义答案、多粒度答案和开放式文本归一化规则。
- 为什么需要把视觉事实和世界知识分开评估，防止模型用语言先验“答对”却没有看图。
- Intent 答案泄漏、QA 类型失衡、实体别名冲突、世界知识污染分别会怎样让指标虚高。
- Judge 版本、prompt、输入顺序、解码参数、缓存版本和失败样本处理必须被冻结，才能比较模型或 checkpoint。
- 训练集、RFT 候选池和 Benchmark 要按源样本隔离，不能只按展开后的 QA 文本去重。

典型追问：

- 399 个视觉样本是否足以支撑结论？置信区间怎么估计？
- 2,692 条 QA 能否被当作 2,692 个独立样本做 bootstrap？
- Evidence Alignment 与 Recall、Precision 的区别是什么？
- World F1 如何处理别名和开放式答案？
- 如何证明 Intent Caption 的收益不是 Judge 偏好或答案泄漏造成的？

---

# 五、P0：ForestPrune

## 19. 视觉 Token 压缩基础

必须掌握：

- token pruning、token merging、KV compression的区别。
- 训练无关方法与训练式压缩方法的差别。
- prefill FLOPs、KV cache、显存和延迟随视觉token数的变化。
- 为什么逐帧独立压缩会忽略跨帧冗余。
- 为什么高压缩率可能反而提高长视频效果：节省预算后可输入更多帧。

## 20. ForestPrune 方法链路

必须能从头叙述：

```text
多帧视觉tokens
→ 计算语义相似性
→ 加入空间邻近约束
→ 加入时间连续性约束
→ 构建跨帧token trees/forest
→ 用树深和节点角色估计重要性
→ 优先删除局部、短期叶节点
→ 保留跨帧持续语义轨迹
→ 满足全局token budget
```

必须回答：

- 为什么是forest而不是单棵tree？
- 候选 tree nodes 如何从每帧 token 中得到，为什么不直接把全部 token 都作为节点？
- edge 如何由语义相似度、空间距离和时间顺序共同构造？
- 树深为什么能表示信息持续性？
- root、trunk/internal、leaf、tail 分别如何形式化定义？
- root 数量远大于最终预算时为什么要先合并 trees？
- pruning如何保证恰好满足预算？
- 算法时间/空间复杂度。
- 是否依赖query；若不依赖，可能丢失哪些任务相关细节？
- 与FrameFusion、FastV、ToMe等方法的公平比较条件。

真实算法口径：

1. 对候选节点特征 $\mathbf F_{\mathrm{nod}}$ 计算 cosine adjacency：

   $$
   A=\mathbf F_{\mathrm{nod}}\mathbf F_{\mathrm{nod}}^\top
   $$

2. 根据节点二维坐标计算空间距离矩阵 $D$。
3. 用语义阈值 $\tau_s$ 和空间阈值 $\tau_p$ 构造有效连接：

   $$
   c_{ij}=1
   \quad\Longleftrightarrow\quad
   a_{ij}\ge \tau_s
   \ \land\
   d_{ij}\le \tau_p
   $$

   随后再结合时间顺序和论文中的 `LinkNode` 规则构建无环 token trees。不能回答成泛化的“阈值或 Top-K 均可”。

4. 当 root 数量远大于保留预算 $K$ 时，先按 root similarity 合并 trees。
5. 按 tree depth 排序，逐步移除 leaf nodes；若只剩 root/trunk 仍未达到预算，再移除 tail nodes。
6. 极高压缩率下若只剩 roots，则按论文的时间规则选择最终保留节点，直到输出节点数满足 $K$。

边界：

- 论文中的“globally optimal pruning decision”应解释为**在完整视频和统一预算下做跨帧全局决策**，不要擅自声称求得了组合优化问题的可证明全局最优解。
- 构造完整 adjacency/distance matrix 可能引入关于候选节点数的二次成本，因此候选节点压缩既影响效果，也影响 pruning 自身开销。

数字口径：

- “压缩90%保留95.8%平均准确率”要说明分母是未压缩baseline表现，不是95.8绝对分。
- “MLVU +10.1%”要确认是百分点还是相对百分比，简历当前使用 `%`，面试时按论文表格口径回答。
- “pruning时间下降81.4%”要说明硬件、token规模、是否含视觉编码。
- 64帧扩展到128/256/512时，要说明固定的是视觉token预算，而不是计算完全不变。
- 扩帧实验的完整数字要能复述：Video-MME `62.8 → 66.5`，MLVU `71.3 → 72.5`，并说明各自对应的帧数和设置。

---

# 六、P0：CAPD

## 21. OPD 与蒸馏基础

必须掌握：

- 知识蒸馏的teacher/student、soft target、temperature、KL loss。
- **OPD = On-Policy Distillation**：student 或其行为策略先生成 rollout，teacher 再在 student 实际访问到的 prefix/state 上提供逐 token 分布监督。
- teacher 是否冻结与 on-policy/off-policy 是两个维度：冻结 teacher 并不意味着 offline；关键看训练 state/prefix 来自 student rollout 还是固定数据/teacher trajectory。
- 一般形式应显式写出 student-visited state distribution。令 $y\sim\pi_b(\cdot|x)$，其中 $\pi_b$ 通常是当前 student 或冻结的 old-policy snapshot，$s_t=(x,y_{<t})$：

  $$
  \mathcal L_{\mathrm{OPD}}
  =
  \mathbb E_{x\sim\mathcal D,\ y\sim\pi_b}
  \left[
  \sum_t
  D_f\left(
  \pi_S(\cdot\mid s_t),
  \pi_T(\cdot\mid s_t)
  \right)
  \right]
  $$

- CAPD 真实实现采用 forward KL、reverse KL、token NLL 还是其他 $f$-divergence，必须按代码和论文填写；不能把普通离线 KD 公式当成 OPD 的完整定义。
- off-policy SFT/KD 在 teacher 或固定数据 prefix 上训练，存在 exposure bias；OPD 的核心是让训练 state distribution 更接近 student 推理分布。
- teacher更大不代表每个token都一定正确。

## 22. 为什么直接Attention蒸馏不够？

必须清楚表述：

- Attention描述模型内部“信息路由到哪里”。
- 高attention不等于该证据对输出具有因果影响。
- 直接attention matching可能复制teacher的冗余或非因果路由。
- 反事实移除更接近“这个时间片段是否影响边界决策”。

但也要保留边界：

> 反事实遮挡结果依赖遮挡方式和分布外程度；它比裸attention更接近因果证据，但不能自动等价于严格因果识别。

## 23. CAPD 方法链路

必须能画出：

```text
完整视频输入 x
  → Frozen Teacher
  → next-token分布 p_full

逐组移除连续时间片段 x\g
  → Frozen Teacher，共G个反事实输入
  → p_minus_g

计算影响：
  I_g = D(p_full, p_minus_g)

用途1：
  用I_g校准teacher attention policy

用途2：
  提高视觉依赖更强的timestamp token蒸馏权重

Student：
  同时学习“根据什么证据输出”
  与“输出什么”
```

重要口径：

- 若有G个时间组，teacher逻辑输入数是 `G+1`：1个完整输入 + G个逐组移除输入。
- 不要误说成只做G次输入。
- timestamp token为什么视觉依赖更强，需要通过定义或实证说明。
- 反事实影响距离可以是KL、JS或其他分布差异；必须以真实实现为准。
- Attention policy loss与token distribution loss如何组合。
- CAPD“不修改 on-policy rollout”表示保留原 OPD 的 student-visited prefixes，只在 teacher supervision 侧增加完整输入与反事实输入；不要误说成 CAPD 是 offline distillation。
- 必须说明 attention policy 具体来自哪些层、head、query/token，如何聚合到时间组，以及 student/teacher 架构不同时如何对齐。
- 必须区分反事实输入的逻辑数量 $G+1$ 与实际可批处理的 forward 次数；二者不是同一个计时口径。

## 24. CAPD实验设计

必须掌握：

- teacher：GRPO后训练Qwen3-VL-32B。
- student：Qwen3-VL-8B。
- 2,500条样本、1 epoch、全参数训练。
- baseline：GRPO、Vanilla OPD、RAL-AD。
- 时间组粒度、组件、attention-loss权重消融。
- TimeLens九项Recall如何聚合。
- Video-MME、LongVideoBench、LVBench用于通用能力保持，而不是直接证明时间定位提升。
- 迁移到约3,000条业务数据和Qwen3.5 2B学生，说明适配性，但跨模型结论需说明配置差异。
- 需要报告 CAPD 相比 Vanilla OPD 的额外 teacher 计算、显存、缓存和 wall-clock 成本，不能只比较准确率。
- “全面超过”应落实到每项指标、平均值还是统计显著性；若没有多 seed 或置信区间，应明确实验边界。

数字口径：

- 九项Recall平均55.1。
- 相比GRPO提升5.9pt，即12.0%相对提升。
- `5.9pt`与`12.0%`不能混淆：

  $$
  \frac{5.9}{55.1-5.9}\approx12.0\%
  $$

---

# 七、P1：其他科研成果

## 25. One-shot Clip Retrieval

必须掌握：

- 长视频为什么需要clip chunking与retrieval。
- query-guided retrieval与query-agnostic压缩的区别。
- one-shot 的精确定义：同一组 frame–query cross-modal similarities 同时用于 query-guided chunking 和 clip retrieval，避免额外模型和重复特征比较；不是“MLLM 只 forward 一次”，也不等于“只检索一个 clip”。
- cross-modal embedding和相似度。
- peak score 选择 cluster centers、动态规划切分连续 clip，以及 clip relevance 的计算。
- coarse-to-fine训练：先用其他视频的 clips 作粗粒度负例，再用同一长视频中的相似 clips 作细粒度 hard negatives。
- SynLongVideo 的 visual relevance 与 instruction divergence，以及为什么普通 image-caption 预训练 embedding 不擅长 question-guided clip retrieval。
- 连续clip为何比离散帧更适合时间事件。
- 2.2分钟处理小时视频时，是否包含解码、embedding和LLM推理。

核心公式：

1. 对密集采样帧计算 query similarity：

   $$
   s_i=\cos\left(f_I^i,f_q\right)
   $$

2. 用局部峰值强度 $g_i$ 选择 cluster centers。论文定义为：

   $$
   g_i
   =
   2s_i
   -
   \min_{j<i}\left\{s_j\mid s_j\le s_i\right\}
   -
   \min_{i<k\le t}\left\{s_k\mid s_k\le s_i\right\}
   $$

3. 相邻 centers 之间用动态规划选择边界 $b_j^*$，目标是在保持连续性的同时最大化左右子序列的语义一致性。面试时应能解释状态、候选边界和复杂度，而不是只说“按相似度切段”。
4. clip relevance 复用已计算的 frame similarities，例如：

   $$
   r_i=\max\left\{s_j\mid I_j\in V_i\right\}
   $$

5. query 变化时，视频 frame embeddings 可以缓存，但 $s_i$、cluster centers、动态规划边界和检索结果都与 query 绑定，通常需要重新计算。

典型追问：

- retrieval漏掉关键clip怎么办？
- recall与压缩率如何权衡？
- query变化是否需要重新编码视频？
- 与RAG的对应关系是什么？
- peak score 为什么适合找 query-related semantic changes？
- 动态规划边界相比固定窗口或 shot boundary 有什么优势？
- coarse-to-fine 的两类负例分别解决什么问题？

对应教程：

- [RAG + Embedding / Retrieval](./rag_embedding_retrieval_tutorial.html)

## 26. XPlainDF

这是第五作者项目，优先掌握整体逻辑，不必冒充核心owner。

至少要能解释：

- Deepfake Detection与可解释检测。
- specialist–generalist fusion。
- specialist先验可能错误，为什么会污染VLM。
- IB：Information Bottleneck基本思想。
- EDL：用evidence构造Dirichlet，表达uncertainty。
- reliability-aware evidence routing。
- global–spatial evidence routing。
- RTEA 的准确全称、输入输出和在系统中的位置。
- counterfactual evidence learning。
- 该项目中 OPD 的准确全称和训练分布；不能默认套用 CAPD 中的 On-Policy Distillation。
- model soup。
- Macro F1为什么适合类别不均衡。
- Overall Score如何定义必须以论文为准。

回答边界：

> 明确说清自己参与的模块、实验或分析，不把整个框架都说成个人设计。

---

# 八、P1：工程栈必须能落到原理

简历技能栏中的每个词至少准备一个“原理 + 项目例子 + 故障案例”。

## PyTorch

- autograd、leaf tensor、计算图；
- Dataloader worker、pinned memory、prefetch；
- AMP与GradScaler；
- activation checkpointing；
- distributed process group；
- state_dict与distributed checkpoint。

## Transformers

- tokenizer/chat template；
- `input_ids`、`attention_mask`、`position_ids`、`labels`；
- `generate`与训练forward；
- KV cache；
- stopping criteria、EOS；
- 自定义modeling与remote code。

## VERL / Open-VERL

- Actor、rollout、reward、reference policy；
- colocated与分离部署；
- on-policy数据生命周期；
- rollout cache；
- reward异步化；
- checkpoint与恢复。

## vLLM

- continuous batching；
- PagedAttention；
- prefill/decode；
- tensor parallel serving；
- KV cache管理；
-吞吐和首token延迟的区别。

## DeepSpeed/FSDP/ZeRO-3

- ZeRO 1/2/3分别切什么；
- FSDP2 per-parameter DTensor；
- all-gather/reduce-scatter；
- HSDP二维mesh；
- checkpoint和meta initialization。

## Spark

- partition、shuffle、skew；
- map/reduce、join；
- 200M样本去重和统计；
- 小文件问题；
- retry与幂等。

## Linux/Git

- GPU/NPU进程、日志、磁盘、网络排障；
- shell管道和退出码；
- Git branch、rebase、cherry-pick、bisect；
- 可复现实验的commit/config/data版本绑定。

---

# 九、P0：简历数字防守表

| 数字 | 必须准备的解释 |
|---|---|
| 2TB、200M+ | 新增数据的存储/样本统计口径，是否压缩后、是否含variant |
| 1.547T、888.73M | CPT v9配方池规模，不等于实际消费量 |
| 1T、12天、77B/day | 实际训练约数、有效运行时间、token统计口径 |
| 1024张910B NPU | 8×128 FSDP2/HSDP mesh、每卡micro-batch=2 |
| 79.28→79.87 | 聚合任务、同预算对照、绝对百分点 |
| 79.72→82.37 | 抖音知识图文子集定义与评测方式 |
| 136/42任务 | benchmark层级、业务任务与公开任务关系 |
| +10.98pt | 团队版本相对Qwen3.5基座 |
| +46.91pt | 抖音推理Base→Instruct，需解释版本和任务聚合 |
| 3,995→19,665 | 源样本先切分，再展开intent/variant |
| 115,793 QA | QA条数，不是独立视觉样本数 |
| 100-step/12,800 rollout | 每step/group/样本关系和有效rollout定义 |
| +13.22/+16.75/-2.63pp | 训练后段相较前段趋势，不自动等于held-out因果提升 |
| 399/2,692/187 | Intent Caption Benchmark 的独立视觉样本、通用 Evidence QA、世界知识题口径 |
| 90% token/95.8% | 压缩比例与相对保留准确率 |
| +10.1%、-81.4% | 百分比/百分点、计时范围、硬件 |
| 64→128/256/512 | 固定视觉 token 预算下扩帧；vision encoder、pruning 和解码成本不一定固定 |
| 62.8→66.5、71.3→72.5 | ForestPrune 扩帧后的 Video-MME、MLVU 指标及对应帧数 |
| 2,500/1 epoch | CAPD训练数据与训练轮次 |
| 55.1/+5.9pt/+12.0% | 绝对分、百分点、相对提升三种口径 |
| 约3,000/2B | CAPD 业务迁移数据量和 Qwen3.5 架构学生规模 |
| 5模型/5基准/+6.1%/+5.2% | OneClip-RAG 的适配范围及 LongVideoBench、TVQA-Long 增益口径 |
| 2.2分钟/RTX 4090 | 小时级视频计时范围、视频时长分布、是否包含解码/embedding/MLLM |
| 0.9878/0.862 | XPlainDF 的 Macro F1、Overall Score 定义、数据集和 baseline |

---

# 十、个人贡献和团队结果边界

## 可以明确强调

- DouyinVL：覆盖数据/基础设施、CPT、scale-up、评测和版本迭代，不应只说成数据工程。
- Intent Caption：项目负责人；问题定义、数据门禁、IEPR、RL链路和评测体系。
- ForestPrune：第一作者。
- CAPD：第一作者。

## 必须谨慎

- DouyinVL最终benchmark增益是团队版本结果。
- Post-Train可以讲完整逻辑，但个人是协同/部分参与，不说全链路owner。
- Intent Caption内部前后段趋势不能包装成严格配对Step 0提升。
- CAPD中的attention不是天然因果证据；反事实影响也要说明遮挡假设。
- XPlainDF是第五作者，明确个人参与模块。
- Pattern Recognition工作是第二作者，区分主导贡献与协作贡献。

---

# 十一、推荐学习顺序

## 第一阶段：能守住DouyinVL

1. VLM完整数据流。
2. Qwen3.5-35A3B MoE。
3. CPT/SFT/RL区别。
4. 数据配方、去重、泄漏、动态视觉token。
5. FSDP2、1024卡真实mesh和通信。
6. Benchmark与badcase回流。

## 第二阶段：能守住Intent Caption

1. 条件证据压缩定义。
2. Evidence QA与数据门禁。
3. IEPR多目标奖励。
4. GRPO公式和工程链路。
5. Judge、缓存、429、评测偏差。
6. 统一 Benchmark、Evidence Alignment 与 World F1。
7. RFT与Best-of-N。

## 第三阶段：能守住论文

1. 视频视觉token和推理成本。
2. ForestPrune完整算法与复杂度。
3. On-Policy Distillation 和 attention distillation。
4. CAPD反事实定义、两个loss和实验消融。
5. OneClip-RAG 的 peak score、动态规划边界和 coarse-to-fine loss。
6. XPlainDF 的不确定性建模、RTEA 与证据路由。

## 第四阶段：模拟面试

每个核心项目分别准备：

- 30秒一句话版本；
- 2分钟STAR/问题—方法—结果版本；
- 10分钟白板深挖版本；
- 10个基础追问；
- 5个实验归因追问；
- 3个失败/局限问题；
- 1个“如果再做一次怎么改”。

---

# 十二、自测标准

只有同时满足以下条件，才算“基础扎实”：

- 不看资料画出VLM、FSDP2和GRPO数据流。
- 手算global batch、FSDP shard size、吞吐和相对提升。
- 写出NTP、KL distillation、GRPO advantage和IEPR的基本公式。
- 写出 OPD 的 student-rollout 期望、ForestPrune 的连接条件、OneClip-RAG 的 peak score。
- 解释每个指标的分母、聚合方法和对照版本。
- 给出至少一个失败样例和排查路径。
- 明确哪些是团队成果、个人主导、协同参与和仍未严格验证的结论。

---

# 十三、典型追问参考答案

> 以下共 90 道参考问答，覆盖前文典型追问，并针对项目负责人和第一作者经历补充公式、实验归因、工程成本与失败边界。面试时先说第一句话的结论，再根据追问展开机制、证据和限制。

## 27. 多模态模型

### Q1：一张图片是如何变成 LLM token 的？

图片先经过解码、缩放和归一化，再按 patch 切分。ViT 把每个 patch 线性映射成视觉 hidden state，经过多层 self-attention 得到视觉特征；随后 spatial merge、projector 或 resampler 将其压缩并映射到 LLM hidden dimension。文本中的 image placeholder 被替换为这段视觉 embedding，与文本 embedding 拼成统一序列，再输入 LLM。视觉 token 本质上是连续向量，不是 tokenizer 词表中的离散 token id。

### Q2：动态视觉 token 为什么能节省计算？

固定高分辨率会让简单图片也产生大量无用 patch。动态视觉 token 根据原始分辨率、长宽比、媒体复杂度或任务预算决定 resize 和 merge 后的 token 数。标准 attention 对序列长度 $L$ 的主要计算和 attention matrix 显存为 $O(L^2)$，MLP约为 $O(L)$；减少视觉 token 不仅减少视觉编码成本，也减少后续所有 LLM 层的 prefill 计算和 activation。

### Q3：多图/视频的 position ids 怎么构造？

先为文本和各媒体块建立与实际拼接顺序一致的位置。普通 RoPE 可使用连续一维位置；多模态 RoPE通常把视觉位置拆成时间、高度、宽度坐标，再按模型规定映射到 rotary维度。多图需要保证每个placeholder对应正确的视觉块；视频还需保持帧序和帧内空间坐标。具体实现必须遵循模型processor生成的 `position_ids` 或 `rope_deltas`，不能自行假设简单连续编号。

### Q4：media placeholder数量与视觉特征不一致会发生什么？

显式实现通常会在scatter或replace阶段报shape/count mismatch；更危险的实现可能把错误媒体特征填到另一个placeholder，训练仍能运行但图文语义错位。前置检查应验证媒体数量、placeholder数量、顺序、每个媒体的token span及有效路径，并在collate后再次检查展开长度。

### Q5：视觉 token 翻倍时，attention计算量如何变化？

若总序列几乎由视觉token主导，从 $L$ 增加到 $2L$，标准self-attention的 $QK^\top$ 和 $AV$ 计算约变为4倍，attention matrix显存也约变为4倍；MLP和投影部分约变为2倍。若文本长度不可忽略，应比较 $(L_t+2L_v)^2/(L_t+L_v)^2$，不能直接断言整个模型总FLOPs严格4倍。

## 28. Qwen3.5 MoE

### Q6：35B总参数为什么只有约3B激活？

MoE层拥有大量expert参数，但router对每个token只选择Top-K routed experts，再加shared expert和attention等dense部分。因此checkpoint需要存储约35B总参数，但一次token forward只经过被选中的少数experts，激活参数量约为3B级。不同token可选择不同expert，但单个token不会执行全部35B参数。

### Q7：MoE是省显存还是省计算？

主要省的是“相同总参数容量下的每token计算”，不天然省参数显存。所有experts仍需被存储，所以总权重、optimizer state可能很大；如果没有EP或FSDP，单卡仍可能放不下。MoE用稀疏激活获得更高参数容量，但会增加router、负载均衡和all-to-all等工程成本。

### Q8：Top-K router如何训练？为什么会负载不均？

Router用token hidden state计算expert logits，经softmax或归一化后选择Top-K experts；主任务loss通过被选路径反传到router，通常还加入load-balancing或router regularization。负载不均来自数据分布和expert早期随机优势形成的正反馈：某些expert更常被选、得到更多训练、随后更容易继续被选。容量限制、auxiliary loss、router bias或token dropping可缓解，但也可能影响质量。

### Q9：EP与FSDP切expert参数有什么本质区别？

EP按expert语义分工：某张卡长期持有一组experts，token根据router结果通过all-to-all发送到目标卡，多卡合作处理同一batch。FSDP按存储维度切参数：expert权重平时分片，计算该FSDP unit前all-gather参数，每张卡仍处理自己的数据。EP通信token activation，FSDP主要通信参数和梯度。

### Q10：linear attention为什么不能直接解释为标准softmax attention map？

标准attention显式形成 $A=\operatorname{softmax}(QK^\top/\sqrt d)$，每个query对应一个对key归一化的权重分布。线性/递归attention通常维护压缩状态，例如 $S_t=f(S_{t-1},k_t,v_t)$，输出由query读取状态；它不一定存在唯一的、逐token归一化的 $L\times L$ attention matrix。由Q/K相似度构造的热图通常只是诊断proxy，不能当成真实路由质量或因果贡献。

## 29. CPT与数据

### Q11：CPT和SFT的loss有何不同？

CPT通常继续使用全序列next-token loss，让模型学习领域语料和多模态分布；SFT使用instruction-response样本，常只对assistant response token计算loss，prompt token被mask。两者数学核心都可用cross-entropy，但监督位置、数据组织和目标能力不同：CPT做分布迁移，SFT做指令行为对齐。

### Q12：1.547T配方池为什么只训练约1T？

1.547T表示可采样数据池或recipe展开后的token规模，不等于训练必须完整遍历一遍。实际训练由token budget、step数、global batch、序列长度、数据采样权重和各桶epoch/cap共同决定；大桶可能只消费一部分，小桶可能按上限重复采样。简历应把1.547T称为配方池，把约1T称为实际消费量。

### Q13：如何判断某个数据桶需要加权还是降权？

同时看四类证据：目标能力缺口、该桶数据质量、边际收益和负迁移。先建立固定训练预算的配比实验，观察对应能力域、相邻能力和防御指标；若加权后目标域提升且通用能力可控，可提高权重。若出现过拟合、模板偏置、答案泄漏、重复样本或通用能力下降，应降权、设cap或先治理数据。

### Q14：如何证明提升来自数据配比，而不是训练随机性？

保持模型初始化或基座、训练token预算、优化器、学习率、数据质量、评测请求和推理参数一致，只改变配比；最好使用多个seed或至少做中间checkpoint趋势和重复小规模实验。报告目标指标、总体指标、防御指标和置信区间，并检查实际消费token是否符合设计。单次最终分数上涨只能说明相关性较强，不能完全排除训练噪声。

### Q15：如何监控CPT引发的通用能力遗忘？

建立领域提升指标与通用防御指标的双轨评测，在训练中定期评估语言、知识、OCR、视频、推理、指令遵循和安全/格式能力；同时保留固定请求、固定解码和基座对照。若领域分数上涨但通用任务持续下降，检查通用数据比例、学习率、训练时长和数据模板偏置，并通过增加通用replay、降低领域桶权重或更早停止缓解。

### Q16：为什么Option Shuffle能减少位置偏置？如何验证？

如果正确选项长期集中在A/B位置，模型可以学习位置先验而不理解内容。训练时随机打乱选项并同步更新答案标签，使正确位置近似均匀，迫使模型依据选项语义决策。验证时比较shuffle前后各位置准确率、位置分布熵、固定内容换位一致性，以及text-blind baseline；不能只看总准确率。

### Q17：OCR Dropout会不会伤害OCR能力？

可能会，所以不能对所有样本无差别删除OCR。它主要用于答案被附加OCR文本直接泄漏、但视觉中仍可读取的样本，通过概率dropout让模型不能总依赖旁路文本。应保留专门OCR训练桶和OCR回归指标，按任务类型设置概率，并比较视觉理解、OCR识别和文本泄漏防御三类指标。

### Q18：如何检测Train/Val的同源视频泄漏？

先定义稳定的source id，在抽帧、裁剪、prompt、intent和QA variant展开前按源视频分组切分。对无可靠source id的数据，再组合文件hash、关键帧perceptual hash、音频指纹、时长/元数据和视频embedding近邻检索。所有同源剪辑和衍生variant必须进入同一split；最后对跨split候选做阈值审计和人工抽检。

### Q19：200M样本如何做分布式去重和质量统计？

用Spark等系统先做规范化与稳定指纹：文本exact hash、MinHash/LSH，图片pHash/embedding，视频关键帧指纹和source metadata。按hash或LSH bucket repartition，在分区内生成候选对并用union-find/规则合并重复簇，随后按簇分配split。质量统计用可加和聚合器计算数量、token、长度、模态、来源、失败率和分位数；对skewed key做salting，输出版本化manifest和抽样审计。

### Q20：为什么不能所有图片都用最高分辨率、所有视频都采最多帧？

视觉token会提高vision encoder和LLM prefill成本，标准attention对总序列长度近似平方增长；长短样本混合还会造成padding和straggler浪费。在固定算力下，提高单帧分辨率会压缩可用帧数，反之亦然。因此应按任务、媒体长度、细节需求和token budget动态分配，在空间细节、时间覆盖、吞吐和显存之间做权衡。

## 30. 分布式训练与稳定性

### Q21：为什么35B模型还要用1024卡？

目的不只是“把35B权重放下”，而是在约1T token预算和12天时间约束下获得足够吞吐，同时分摊参数、梯度、Adam states和activation。你们实际使用128-way FSDP shard降低单卡模型状态，再用8-way replication扩大数据吞吐；1024卡还承担micro-batch 2、global batch 2048的大规模计算。卡数由模型、序列、多模态解码、目标token/day和通信效率共同决定。

### Q22：FSDP2与ZeRO-3有什么对应关系？

二者都切分parameter、gradient和optimizer state，语义上都属于full sharding。ZeRO-3常以DeepSpeed状态分区描述；FSDP2使用PyTorch DTensor做per-parameter sharding，并通过module hooks在forward/backward前all-gather、backward后reduce-scatter。它们的核心显存思想相同，但API、参数表示、wrap粒度、prefetch和checkpoint生态不同。

### Q23：为什么选择128-way shard + 8-way replicate？

128-way shard显著降低35B MoE的持久模型状态和optimizer显存；8份replica让全部1024卡处理不同数据，提高global throughput，并把full-shard communicator限制在128卡而不是1024卡。它是在单卡显存、all-gather延迟、跨节点带宽和数据并行吞吐之间的折中。精确原因还应结合物理拓扑和profiling，不能仅凭配置断言是全局最优。

### Q24：all-gather、reduce-scatter分别发生在哪？

FSDP wrap unit执行forward前，128个rank把参数分片all-gather成当前unit完整参数；`full_shard=true`时forward后释放完整参数，backward前再all-gather。每个rank算出局部梯度后，reduce-scatter先对128卡梯度求和，再让每卡只保留对应梯度分片。随后8个replica的同位置梯度分片还要在replicate维做all-reduce同步。

### Q25：为什么不用EP？不用Ulysses的代价是什么？

配置只说明这次recipe将MoE参数交给FSDP，而没有按expert语义做token all-to-all；可能是35A3B在当前显存和通信条件下FSDP已足够，或EP收益不及实现复杂度。`ulysses_size=1`表示每个rank处理样本完整序列，避免额外all-to-all，但超长序列activation不能沿Ulysses维分摊，单卡序列显存和可支持长度受到更多限制。真实选择原因需由profile和框架稳定性验证。

### Q26：训练hang、loss spike、吞吐下降怎么排查？

先确定首个异常step和首个异常rank。Hang检查各rank日志、collective序列是否一致、坏样本/解码超时、OOM后僵尸rank和网络/HCCL错误；loss spike检查样本、有效token、label mask、学习率、gradient norm、NaN/Inf和恢复checkpoint；吞吐下降拆分data、forward、backward、collective、optimizer和checkpoint时间，并比较rank间P50/P99。每次只改变一个因素，用可复现样本和配置验证。

### Q27：为什么一个坏视频可能挂住1024卡？

同步训练每个step都要进入相同collective。若一个rank在视频解码、远程读取或异常重试上长时间阻塞，其余1023个rank即使已完成计算，也会在下一次all-gather/reduce-scatter等待它；若某rank异常退出但其他rank未及时感知，还可能表现为collective超时。需要解码timeout、预验证、错误样本隔离和全rank一致的异常退出机制。

### Q28：如何定位是Loader慢还是通信慢？

分别记录data wait、H2D、forward、backward compute、collective和optimizer时间，同时看device utilization与网络带宽。Loader慢通常表现为step开头GPU空闲、data time升高、CPU/存储/解码繁忙；通信慢表现为NCCL/HCCL collective时间升高、网络饱和或rank straggler。用预生成内存batch可隔离Loader，用缩小world size或通信benchmark可隔离collective。

### Q29：超长视频如何采帧？

先按任务决定均匀、分段、镜头边界、query-guided或运动/语义自适应采样；设置最大帧数和视觉token budget，长视频优先保证时间覆盖，再给关键片段更高密度。必须处理可变帧率、重复帧、黑帧和解码失败，并保留原始时间戳映射，避免采样后无法做temporal grounding。

### Q30：Checkpoint保存为什么可能造成训练抖动？

保存会产生device-to-host拷贝、分片序列化、文件系统元数据和HDFS/网络写入；所有rank若在保存点同步，最慢rank决定停顿时间。大量小文件、共享存储拥塞和optimizer state体积会放大抖动。DCP分片保存、异步checkpoint、限流、独立I/O线程和错峰写入可缓解，但必须确保一致性和恢复可用。

## 31. Judge与RL工程

### Q31：Grouped Judge为什么比逐条Judge省请求？

它把同一prompt或多个候选合并到一次模型调用，复用系统提示、图像输入和网络握手，减少请求数和重复prompt token；judge还能在同一上下文内直接比较候选。但单次请求更大，仍受TPM、上下文长度和解析稳定性限制，所以不是无限合并。

### Q32：Grouped Judge的顺序偏置如何控制？

随机化候选顺序，并保存原始candidate id做反向映射；对关键评测可做多次不同排列或A/B交换，统计一致性。Prompt要求逐候选独立打分而非只选第一个，并用位置分布、交换一致率和人工样本审计检测偏置。

### Q33：流式Reward如何避免等待所有样本？

将rollout、judge和reward计算拆成带request id的异步任务；单个样本或小批次judge完成后立即解析并写入结果队列，训练侧按group完整性或batch门槛消费，而不是等整个epoch。必须保证幂等、超时、重试、缓存版本和缺失结果策略，避免把晚到reward配错样本。

### Q34：请求缓存的Key应包含哪些版本信息？

至少包括sample/request id、媒体内容hash、prompt和system prompt、模型及checkpoint、chat template、生成参数、judge版本、reward版本、评测schema和代码/config版本。只用sample id会在prompt或judge升级后错误复用旧结果。

### Q35：429为什么不能盲目拆成更多并发单请求？

429通常表示RPM/TPM或服务容量已超限。把一个group拆成更多并发请求会增加请求数、重复prompt token和瞬时流量，使限流更严重。正确做法是对原group指数退避、降低并发并服从 `Retry-After`；只有在响应格式损坏而非限流时，才考虑逐条fallback。

### Q36：模型训练吞吐和外部Judge TPM为什么是两个独立瓶颈？

训练吞吐由本地NPU/GPU计算、显存和collective决定；Judge TPM是外部服务单位时间可处理的输入输出token配额。即使训练设备空闲，Judge也可能因TPM饱和阻塞reward；反过来Judge充足也不能解决actor rollout慢。需要分别监控tokens/s、队列长度、请求延迟、429率和TPM占用。

## 32. Intent Caption、GRPO 与评测

### Q37：IEPR为什么必须是多目标奖励？完整回答框架是什么？

Intent Caption同时要求证据覆盖、事实可靠、意图相关、长度受控和低冗余，任一单指标都存在退化解。概念上可写为：

$$
R_{\mathrm{IEPR}}
=
w_rR_{\mathrm{evidence}}
+w_pR_{\mathrm{precision}}
+w_iR_{\mathrm{intent}}
+w_lR_{\mathrm{length}}
-w_dR_{\mathrm{redundancy}}
$$

只有Recall会鼓励堆积事实，只有Precision会鼓励极短保守输出，只有Intent Relevance可能鼓励复述intent或生成不可验证信息。面试时必须进一步给出真实实现中的每个分项、取值范围、归一化、权重、缺失judge处理和clipping；上式只是问题定义，不能替代代码中的精确reward。

### Q38：为什么选GRPO而不是DPO、PPO或纯RFT？

该任务能针对同一图像和intent采样多条caption，并通过Evidence QA、Claim、Intent和长度规则得到在线reward，因此适合组内相对比较。GRPO不需要额外critic，能用同prompt的组均值反映prompt难度；DPO需要预先构造稳定偏好对，PPO需要训练value model，纯RFT只能模仿筛出的答案，不能持续探索并利用在线reward。代价是GRPO需要多条on-policy rollout，且group reward方差、judge吞吐和reward hacking都要单独治理。

### Q39：写出Intent Caption使用的GRPO核心目标。

对同一prompt采样 $G$ 个响应，reward为 $r_i$，组相对advantage为：

$$
A_i
=
\frac{r_i-\mu_G}{\sigma_G+\epsilon}
$$

令行为策略为 $\pi_{\mathrm{old}}$，当前策略为 $\pi_\theta$，第 $t$ 个生成token的ratio为：

$$
\rho_{i,t}
=
\exp\left(
\log\pi_\theta(y_{i,t}\mid s_{i,t})
-
\log\pi_{\mathrm{old}}(y_{i,t}\mid s_{i,t})
\right)
$$

简化的clipped objective为：

$$
\mathcal L_{\mathrm{GRPO}}
=
-
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{M_i}
\sum_t m_{i,t}
\min\left(
\rho_{i,t}A_i,\,
\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i
\right)
+
\beta\mathcal L_{\mathrm{KL}}
$$

其中 $m_{i,t}$ 只选择actor生成token，$M_i=\sum_t m_{i,t}$。真实项目是否使用per-token或per-sequence归一化、哪种KL estimator和loss reduction，必须按Open-VERL配置回答。

### Q40：group内reward标准差为0怎么办？

如果同组reward完全相同，组内比较不提供学习信号。实现上应使用 $\epsilon$ 防止除零，并把该组advantage置零或跳过；不能靠极小分母放大数值噪声。频繁出现零方差通常说明采样温度过低、group size过小、reward过于离散、judge饱和或任务太容易/太难，需要从数据课程、采样和reward分辨率上处理。

### Q41：`100-step、12,800条rollout`应该如何防守？

必须解释“rollout”的计数单位。至少准备：

$$
N_{\mathrm{rollout}}
=
N_{\mathrm{step}}
\times
N_{\mathrm{prompt/step}}
\times
G
\times
N_{\mathrm{actor\ replica}}
$$

并说明12,800是原始生成数、judge成功数、进入训练的有效响应数还是去重后数量；失败、超时、截断和缺失reward是否计入。若100步共12,800条，则平均每步128条，但这不能单独反推出prompt数和group size。

### Q42：Caption Judge与Visual Judge冲突怎么办？

先明确职责：Caption Judge更适合语言完整性、claim拆分和意图相关性，Visual Judge负责视觉可验证性和幻觉。冲突时不能简单平均，应按预先定义的优先级、置信度或拒绝规则组合，例如视觉不支持的claim即使语言质量高也不能获得高总分。还应在人工标注集上分别测judge相关性、交换顺序一致率、假阳性/假阴性，并监控版本升级后的分布漂移。

### Q43：如何证明QA Recall提升不是因为输出变长？

至少同时报告Recall、Claim Precision、平均有效长度、单位token证据数和幻觉率，并在长度匹配的输出子集上做配对比较。可画accuracy–length或Recall–length曲线，或在相同长度预算下比较checkpoint。如果Recall只随长度上涨而Precision和长度效率下降，就不能声称模型学会了更好的证据选择。

### Q44：399个视觉样本和2,692条QA的独立统计单位是什么？

同一视觉样本派生出的多个intent、QA和claim存在相关性，不能把2,692条QA当作完全独立样本直接bootstrap。更稳妥的是以源视觉样本为cluster做cluster bootstrap，或做分层/层级bootstrap：先重采样源样本，再在样本内聚合QA。399个独立视觉样本决定了外层有效样本量，2,692条QA主要提高样本内能力覆盖。

### Q45：Evidence Alignment与Recall、Claim Precision有什么区别？

Recall问“参考证据被覆盖了多少”，Claim Precision问“输出claim中有多少能被视觉支持”；Evidence Alignment进一步问输出claim是否与相应Evidence QA或证据槽正确对应。一个caption可能覆盖很多正确事实但没有覆盖当前intent的关键证据，也可能claim本身正确却对应错证据，因此三者需要分开报告。精确匹配、judge匹配还是embedding匹配必须以真实评测实现为准。

### Q46：World F1如何处理实体别名和开放式答案？

先做答案规范化，包括大小写、标点、单位、时间格式、实体别名和可接受同义词，再基于标准答案集合计算precision、recall和F1。若用LLM judge处理开放答案，应冻结prompt和模型版本并做人类校准。世界知识题还必须与视觉事实题分开，因为模型可能只靠语言先验答对，不能把它当作视觉证据能力。

### Q47：如何防止Intent答案泄漏和世界知识污染让Benchmark虚高？

分别做去intent、去媒体、去OCR/ASR和text-only对照；如果不看图仍能高准确回答，说明存在泄漏。按源样本和衍生variant做split隔离，并检查训练/RFT池与Benchmark的exact、近重复和embedding相似性。对世界知识题还要区分问题是否需要视觉实体识别，避免把纯常识题混入视觉能力增益。

### Q48：两阶段拒绝采样为什么有selection bias？如何缓解？

只保留高judge分候选会把judge偏好、长度偏好和系统性误判蒸馏回actor。缓解方法包括：多judge或规则交叉验证、设置Precision/幻觉硬约束、保留一定多样性、人工审计边界样本、报告筛选前后分布变化，并在独立held-out Benchmark上验证。RFT数据最好保存全部候选和各分项reward，便于后续追溯选择偏差。

### Q49：为什么训练后段相较前段的趋势不是严格因果增益？

前后阶段可能使用不同rollout请求、采样随机性、输出长度、judge负载和失败率，因此两段均值变化不等于固定样本上的Step 0到Step 100提升。严格比较需要固定评测集、judge版本、prompt、解码参数和统计单位，对同一请求做paired evaluation并报告置信区间。当前数字应表述为训练内部趋势，而不是严格held-out因果结论。

## 33. ForestPrune

### Q50：为什么是Forest而不是单棵Tree？

视频中存在多个相对独立的语义对象、区域和事件轨迹，强行连成一棵树会引入低相似度边并破坏结构。Forest允许多个roots分别代表不同语义簇，再通过树深和节点角色建模跨帧持续性。它既能表达多个并行语义轨迹，也允许完全无可靠连接的节点形成新root。

### Q51：候选节点和Edge如何构造？

ForestPrune先从每帧视觉token中选择代表性候选节点，得到 $\mathbf F_{\mathrm{nod}}$；使用全部token会让二次相似度矩阵和噪声连接都显著增加。随后计算：

$$
A=\mathbf F_{\mathrm{nod}}\mathbf F_{\mathrm{nod}}^\top
$$

并根据二维坐标得到空间距离矩阵 $D$。有效连接满足：

$$
c_{ij}=1
\quad\Longleftrightarrow\quad
a_{ij}\ge\tau_s
\ \land\
d_{ij}\le\tau_p
$$

最后结合时间顺序和`LinkNode`规则形成无环trees。真实方法不是泛化的“阈值或Top-K均可”；候选节点策略、父节点选择和连接方向都应按论文算法回答。

### Q52：树深为什么能表示信息持续性？边界是什么？

相似语义若在连续帧中稳定出现，会沿时间方向形成更深路径，因此tree depth是跨帧持续性的proxy。短期噪声往往只能形成浅树或leaf。但遮挡、镜头切换、视角变化和外观突变会打断真实轨迹；一次性关键事件也可能只出现一帧。因此深度不是语义重要性的充分条件，需要与节点角色和结构规则共同使用。

### Q53：Root、trunk/internal、leaf和tail分别是什么？

Root是tree的起点或代表节点；trunk/internal node连接前后节点，承担结构传递；leaf没有后续子节点；tail是当前tree时间上最靠后的末端节点。ForestPrune先删leaf以减少局部冗余，若预算仍未满足再逐步删除tail。不能把所有leaf等同于无用token，因为短暂关键事件也可能是leaf，这是方法的潜在失败模式。

### Q54：Root太多时如何处理？为什么不能直接删到预算？

当root数量远大于保留预算 $K$ 时，如果直接删roots，可能在还未利用语义关系前丢失整个语义簇。论文先基于root similarity合并trees并更新结构，再执行预算裁剪。这样做的目标是先减少重复语义簇，再在tree内部根据深度和节点角色压缩。

### Q55：Pruning如何保证满足预算？

真实流程不是全局Top-$K$打分。算法先按tree depth排序，再逐步移除leaf nodes；当所有tree只剩root和trunk但预算仍未满足时，再移除时间末端tail nodes；极高压缩下若只剩roots，则按论文时间规则选出最终保留节点。该过程持续到保留集合大小等于 $K$，最后应显式检查输出token数。

### Q56：ForestPrune的时间和空间复杂度瓶颈是什么？

设候选节点总数为 $M=T N'$。若显式构造完整相似度矩阵和距离矩阵，核心成本至少包含 $O(M^2d)$ 相似度计算和 $O(M^2)$ 存储；建树、排序和迭代剪枝还会增加额外成本。候选节点预压缩因此不仅是质量选择，也是控制pruning自身复杂度的关键。实际回答应补充实现是否分块、是否只比较特定时间窗口及真实profiling结果。

### Q57：“Globally optimal pruning”应如何谨慎解释？

应解释为ForestPrune相对逐帧局部方法，在完整视频和统一token budget下进行跨帧全局决策。除非论文给出了优化问题的严格全局最优证明，否则不要声称算法求得组合优化意义上的数学全局最优解。更安全的表述是“global budget-aware”或“video-level global pruning decision”。

### Q58：为什么固定token预算下扩帧可能提升效果？哪些成本并未固定？

高比例压缩释放了视觉token预算，使模型能用相近的LLM输入token数覆盖更多帧，从而提高长视频时间覆盖。实验中Video-MME从62.8提升至66.5，MLVU从71.3提升至72.5。但vision encoder仍需处理更多原始帧，解码、特征提取和ForestPrune构图成本也可能增长；固定的是送入MLLM的视觉token预算，不是端到端计算完全不变。

### Q59：ForestPrune是否依赖Query？可能丢失什么？

若方法是query-agnostic，一次压缩结果可服务多个问题，适合缓存和通用视频表示；但全局持续性不等于特定query相关性，可能删除短暂动作、细小文字、边缘对象或只出现一次的关键事件。可通过保守预算、短事件/OCR保护、query-aware二阶段重排或与retrieval结合缓解。

### Q60：与FrameFusion等方法怎样做公平比较？

应固定基础MLLM、输入帧、分辨率、视觉token预算、解码参数、硬件、dtype和计时边界，并同时报告准确率、prefill延迟、端到端延迟、峰值显存和pruning自身耗时。`-81.4% pruning time`必须明确比较对象、硬件、token规模及是否包含vision encoding；`+10.1%`也要明确是论文表中的相对百分比还是绝对百分点。

## 34. CAPD 与 On-Policy Distillation

### Q61：OPD的精确定义是什么？和普通KD差在哪？

OPD是On-Policy Distillation。Student或其old-policy snapshot先生成 $y\sim\pi_b(\cdot|x)$，teacher再在student实际访问到的prefix $s_t=(x,y_{<t})$ 上提供分布监督：

$$
\mathcal L_{\mathrm{OPD}}
=
\mathbb E_{x\sim\mathcal D,\ y\sim\pi_b}
\left[
\sum_t
D_f\left(
\pi_S(\cdot\mid s_t),
\pi_T(\cdot\mid s_t)
\right)
\right]
$$

普通KD通常在固定数据或teacher trajectory上训练，属于off-policy；OPD的关键不是teacher是否冻结，而是state distribution来自student rollout。CAPD真实使用的KL方向和estimator必须按代码回答。

### Q62：为什么直接Attention蒸馏不够？

Attention weight描述模型内部信息路由，不等于某段视觉证据对输出的因果贡献。Teacher可能对冗余token分散注意，且student与teacher的层数、head和结构未必可逐项对齐。直接MSE/KL匹配attention可能复制噪声。CAPD通过移除时间组后next-token分布的变化估计该组对边界决策的影响，再校准attention监督，但仍需承认遮挡引入的分布外偏差。

### Q63：CAPD的反事实影响如何定义？

对完整输入得到teacher分布 $p_{\mathrm{full}}$，对移除时间组 $g$ 的输入得到 $p_{-g}$，定义：

$$
I_g
=
D\left(
p_{\mathrm{full}},
p_{-g}
\right)
$$

其中 $D$ 可能是KL、JS或其他分布距离。面试时必须给出真实实现中的方向、求和token位置、词表范围、是否按timestamp token聚合以及归一化方式；不能只回答“可以用任意距离”。

### Q64：为什么有 $G$ 个时间组却需要 $G+1$ 个逻辑teacher输入？

因为所有反事实影响都需要同一个完整输入分布作为基准：1个完整视频输入加上 $G$ 个分别移除时间组的输入，共 $G+1$ 个逻辑样本。工程上它们可以在batch维合并，因此逻辑输入数量不等于串行forward次数；计时和显存分析必须区分这两个口径。

### Q65：移除时间片段会不会产生分布外输入？

会。删除连续帧可能改变时长、position ids、帧间连续性和媒体token数量，使teacher分布变化部分来自输入格式扰动而非真实证据缺失。需要使用与训练分布尽量一致的mask/replace策略、保持位置与token contract，并通过随机移除、无关片段移除或不同遮挡方式做对照。CAPD比裸attention更接近因果证据，但不等于严格因果识别。

### Q66：Attention policy具体是什么，怎样与时间组影响对齐？

必须从真实实现回答四件事：使用哪些层和heads；query来自哪些输出token；视觉attention如何聚合到frame/group；teacher和student结构不同时如何对齐。概念上可把teacher attention聚合成组级policy $a_g$，再用反事实影响 $I_g$ 构造校准target，但具体是重加权、归一化、KL还是MSE不能猜，必须给出代码中的公式。

### Q67：为什么timestamp token要获得更高蒸馏权重？

时间边界输出直接依赖视频时序证据，相比普通功能词或语言模板token，其正确性更受视觉片段影响。因此CAPD对视觉依赖更强的timestamp token提高蒸馏权重，目标是把teacher的边界决策知识集中传给student。但“更视觉依赖”需要用token类型定义、去视觉对照或消融支持，不能只凭直觉。

### Q68：CAPD的总loss怎样组织？

概念上由输出分布蒸馏与证据/attention policy监督组成：

$$
\mathcal L_{\mathrm{CAPD}}
=
\mathcal L_{\mathrm{OPD}}
+
\lambda_{\mathrm{attn}}
\mathcal L_{\mathrm{policy}}
$$

如果timestamp token另有权重，则应在 $\mathcal L_{\mathrm{OPD}}$ 的token reduction中显式体现。真实回答必须补齐 $\mathcal L_{\mathrm{OPD}}$ 的KL方向、$\mathcal L_{\mathrm{policy}}$ 的定义、影响归一化、mask和 $\lambda_{\mathrm{attn}}$；这五项缺一不可。

### Q69：CAPD为什么仍然是on-policy distillation？

CAPD保留原OPD的student rollout和student-visited prefixes，只在teacher监督侧增加完整视频与反事实视频，并改变attention policy或token weighting。反事实teacher查询不会把训练变成offline；判断on/off-policy看state visitation来自谁，而不是teacher调用了多少次。

### Q70：CAPD的主要额外成本是什么？如何优化？

主要成本是每个样本需要 $G+1$ 个teacher逻辑输入，还可能需要导出attention和full-vocab logits。可通过反事实样本batch化、teacher冻结、预计算/缓存、只保存必要层attention、缩小距离计算词表范围、混合精度和异步数据准备优化。实验应报告相对Vanilla OPD的teacher FLOPs、峰值显存、tokens/s和wall-clock，而不只报告准确率。

### Q71：为什么要与GRPO、Vanilla OPD和RAL-AD比较？

GRPO代表直接用任务reward优化student；Vanilla OPD检验普通输出分布蒸馏；RAL-AD代表attention相关蒸馏baseline。三者分别回答“是否需要蒸馏”“是否需要反事实校准”“是否优于已有attention distillation”。公平比较要求相同student初始化、数据、epoch、输出格式、训练预算和评测协议。

### Q72：TimeLens九项Recall怎样聚合才可信？

需要列出三项基准各自的子任务和Recall定义，先报告逐项结果，再说明55.1是简单macro average、按样本加权还是其他聚合。若九项样本规模不同，macro与micro含义差异很大。`+5.9pt`是绝对百分点，`12.0%`是相对提升，因为baseline约为 $55.1-5.9=49.2$：

$$
\frac{5.9}{49.2}\approx12.0\%
$$

### Q73：Video-MME等通用基准能证明什么，不能证明什么？

它们能检查时间定位专项训练是否明显伤害通用视频理解，但不能直接证明时间边界能力提升。若只观察到平均分“保持”，还要说明是否在测量噪声内、是否有多seed或置信区间。专项结论应主要由TimeLens及边界定位指标支持。

### Q74：从Qwen3-VL-8B迁移到Qwen3.5架构2B说明什么？

它提供了跨模型规模、架构和业务数据分布的适配证据，但不自动证明普适性。应报告哪些模块复用、哪些attention接口重新对齐、时间组和loss权重是否重调，以及业务数据约3,000条时是否仍使用相同训练预算。跨设置结果不能与主实验直接做绝对数值比较。

### Q75：CAPD最重要的失败模式是什么？

包括：遮挡产生OOD输入；影响距离受teacher校准误差影响；短片段移除可能同时破坏多种证据；teacher本身边界错误会被放大；attention接口跨模型不一致；$G+1$ teacher成本过高；timestamp weighting可能牺牲普通文本质量。回答时至少给出一个真实badcase和对应消融或缓解方案。

## 35. OneClip-RAG

### Q76：“One-shot”到底指什么？

One-shot指同一组frame–query cross-modal similarities既用于query-guided video chunking，又复用于clip retrieval，从而把原本分开的切分和检索统一到一次处理流程。它不表示MLLM只forward一次，也不表示只检索一个clip。

### Q77：Frame–query similarity和peak score怎样选择cluster centers？

先计算：

$$
s_i=\cos\left(f_I^i,f_q\right)
$$

再用局部峰值强度：

$$
g_i
=
2s_i
-
\min_{j<i}\left\{s_j\mid s_j\le s_i\right\}
-
\min_{i<k\le t}\left\{s_k\mid s_k\le s_i\right\}
$$

选择 $g_i$ 最大的一组frames作为cluster centers。它偏好相对两侧具有明显query相关性峰值的位置，用于捕获与问题相关的语义变化，而不是普通shot boundary。

### Q78：动态规划边界解决什么问题？

相邻cluster centers之间的中间frames既可能属于左clip，也可能属于右clip。OneClip-RAG在所有候选边界上优化左右连续子序列的语义一致性，得到 $b_j^*$，从而保持clip连续且避免逐帧离散选择。回答时应能解释DP的状态、候选边界、目标函数以及为什么它比固定窗口更适合跨scene的query相关事件。

### Q79：Clip relevance如何计算？有什么局限？

论文复用已计算的frame similarities，例如：

$$
r_i
=
\max\left\{
s_j\mid I_j\in V_i
\right\}
$$

优点是无需额外clip encoder；局限是单个高分frame可能让整段clip得分过高，对长clip长度和噪声不够鲁棒。可讨论mean/top-$k$ pooling或长度校准，但必须区分论文实现与后续改进建议。

### Q80：Coarse-to-fine训练的两阶段负例有什么区别？

Coarse阶段用其他视频的clip frames作负例，先学会从不同视频中找到正确上下文；fine阶段改用同一长视频中内容相似但与当前问题不对应的clips作hard negatives，训练细粒度instruction following。两阶段顺序体现从跨视频区分到视频内精确定位的课程学习。

### Q81：SynLongVideo为什么同时需要visual relevance和instruction divergence？

Visual relevance让拼接的短视频在视觉上连贯，形成有挑战但合理的长视频；instruction divergence防止不同片段对应几乎相同的问题，否则检索模型无法学会区分相关clip。公开论文数据包含44,878个video-question pairs、997个已有长视频和430个合成长视频，平均约7.7分钟；简历未写这些数字，面试时可作为论文细节准备。

### Q82：Query变化时哪些可以缓存，哪些必须重算？

原始视频解码结果和每帧视觉embedding可以缓存；新query只需重新编码文本并计算 $s_i$。但OneClip-RAG的cluster centers、DP边界、clip集合和retrieval结果都由 $s_i$ 决定，因此query变化后通常要重新chunk和retrieve。不能把它回答成普通固定chunk双塔RAG。

### Q83：Retrieval漏掉关键clip怎么办？

可提高dense sampling频率、cluster center数量或retrieval Top-$K$，加入多尺度/重叠clip和均匀采样兜底；对低置信query触发扩大检索。评测必须拆分clip retrieval recall与最终QA accuracy，区分“没有检到证据”和“检到后MLLM仍答错”。增加候选会提高recall，但也增加上下文成本和干扰。

### Q84：2.2分钟的效率数字怎样防守？

必须说明硬件是单张RTX 4090、视频时长分布、base MLLM、采样帧数、输出长度和计时范围。公开结论是平均处理小时级视频约2.2分钟，但面试中应回到论文实验脚本确认是否包含视频解码、frame embedding、chunking、retrieval和MLLM生成。没有确认前不要擅自说成纯检索时间或完整端到端SLA。

### Q85：OneClip-RAG与文本RAG的对应关系和本质差异是什么？

视频clip对应document chunk，VL embedding model对应retriever，相关clips作为外部上下文交给MLLM。但视频chunk边界由query和时间连续性共同决定，原始媒体解码成本高，同一query下的证据具有顺序和动作连续性；因此不能简单照搬固定长度文本chunk和静态向量索引。

## 36. XPlainDF

### Q86：XPlainDF解决的核心问题是什么？

Deepfake specialist可以提供局部伪造证据，但specialist本身可能错误或过度自信；如果不加判断地把其先验交给通用VLM，会导致错误证据传播。XPlainDF的核心是specialist–generalist fusion与reliability-aware evidence routing，让VLM利用可靠证据并拒绝不可靠先验，同时输出可解释检测依据。

### Q87：IB和EDL分别起什么作用？

Information Bottleneck通过限制表示保留与任务相关的信息、压制冗余噪声；Evidential Deep Learning把分类evidence映射为Dirichlet参数，用总evidence和分布浓度表达预测与不确定性。面试时应给出真实模块的输入输出、loss和uncertainty定义，而不能只背概念。

### Q88：Reliability-aware routing怎样防止错误specialist污染VLM？

核心是根据specialist evidence及其不确定性决定证据是否进入、以多大权重进入global/spatial路径。高置信且一致的证据被加强，冲突或高不确定证据被降权或拒绝。具体routing score、阈值、RTEA全称及全局–空间融合位置必须按论文和本人负责模块补齐；当前简历无法支持更具体的公式，不能编造。

### Q89：该项目中的OPD、反事实学习和model soup分别是什么？

这些缩写和方法必须逐项回到论文确认。尤其不能默认这里的OPD与CAPD中的On-Policy Distillation同义。反事实证据学习应说明改变或移除哪类evidence、监督什么不变性；model soup应说明平均哪些checkpoint、为什么权重空间平均有效、是否使用uniform或greedy soup。

### Q90：`0.9878 Macro F1`和`0.862 Overall Score`如何防守？

必须说明数据集是XPlainVerse、类别分布、Macro F1的类别平均方式、Overall Score的组成、对照baseline、是否单模型或model soup、阈值选择和测试协议。作为第五作者，应明确自己负责的模块、实验或分析，不把完整框架和最终数字表述为个人独立完成。
