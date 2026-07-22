## §0 心智模型 + TL;DR Cheat Sheet

**Transformer block 不是一个公式，而是一套分层装配的规则。** 面试里大量混淆（"Pre-LN 是 Post-LN 的下一代""MQA→GQA→MLA 是一条升级链""decoder-only 只是 decoder 去掉一部分"）都来自把这套分层规则压缩成了一个线性时间轴。先立好心智模型：残差流 $x_l$ 依次穿过两三个**子层**（attention、FFN，原始 encoder-decoder 还多一个 cross-attn），默认是**串行**依赖——FFN 读到的是 attention 更新之后的残差流（parallel 是例外，见下）：

```text
残差流 x_l
  │
  ├─▶ [归一化槽位: Pre／Post／branch pre+post] → Attention 槽位（自注意力/交叉注意力，
  │     MHA/GQA/MQA/MLA，±RoPE，±QK-Norm，全局/局部窗口）→ [归一化槽位] ──▶ ADD 回残差流
  │     得到中间态 u = x_l + Attn(N(x_l))
  │
  u ─▶ [归一化槽位] → FFN 槽位（Dense GELU / Gated SwiGLU / MoE）→ [归一化槽位] ──▶ ADD 回残差流
  │     （串行默认：FFN 读 u，不是 x_l；parallel 变体：FFN 和 attention 都读同一份 N(x_l)，不读 u）
  │
残差流 x_{l+1}
```

**槽位不是严格数学正交，而是分布在不同层级；层内基本能独立分析，层间存在真实耦合**——这是本教程和"三条正交轴"这类简化说法的关键区别。几个具体反例：MLA 选的 latent 表示方式会反过来约束 RoPE 怎么拆分；norm 拓扑和 norm 算子在实践中明显绑定出现（Pre-LN 配 RMSNorm、branch pre+post 配 RMSNorm 是常见组合，不是任意搭配）；QK-Norm、attention scale、soft-cap 三者不能无脑叠加，需要重新核对尺度；"KV 轴不涉及模型总参数量"这句话本身也站不住——GQA 的参数公式清楚地表明它确实改变了投影参数。用一张分层地图代替"正交轴"更贴近事实：

| 层级 | 典型决策 | 谁来管 |
| --- | --- | --- |
| 算子内部 | QKV 投影、RoPE、QK-Norm、head sharing（MHA/GQA/MQA/MLA）、gated FFN | attention/FFN 模块自己 |
| 单个 block | serial/parallel 依赖图、norm 放置（Pre/Post/branch pre+post）、residual add、branch scaling/gating | block 的 `forward` |
| stack 调度 | final norm、local/global 交替、哪些层 dense/哪些层 MoE | 堆栈装配代码，不是某一层 block |
| 模型外壳 | token embedding、外部 absolute PE、weight tying、LM head（含 soft-cap） | 整个模型的输入/输出层 |
| 执行/适配 overlay | KV cache、FlashAttention、packed projection、量化、LoRA | 部署/微调阶段的执行策略，不改变上面四层的架构语义 |

后文所有"这个选择属于 block 内、stack 级还是模型外壳"的判断，都按这张表定位，不再笼统地说"某个槽位"。

> 💡 **四个残差拓扑公式，一屏记住，全部统一到"完整 block"粒度**（后文 §1–§7 全是这四个公式的具体化；$A$、$M$ 分别是 attention、FFN 子层，$N_1,N_2,N_A,N_M,N_1^{\text{pre}},N_1^{\text{post}},N_2^{\text{pre}},N_2^{\text{post}}$ 都是归一化算子）
>
> $$\text{Post-LN（Vaswani et al. 2017 原始 Transformer）}:\quad u = N_1\big(x + A(x)\big),\quad y = N_2\big(u + M(u)\big)$$
> $$\text{Pre-LN 串行（至少 GPT-2 已明确采用，后成为现代 LLM 常见形式）}:\quad u = x + A\big(N_1(x)\big),\quad y = u + M\big(N_2(u)\big)$$
> $$\text{Pre-LN 并行（GPT-J / PaLM）}:\quad y = x + A\big(N_A(x)\big) + M\big(N_M(x)\big)$$
> $$\text{branch pre+post（Gemma-2/3）}:\quad u = x + N_1^{\text{post}}\big(A(N_1^{\text{pre}}(x))\big),\quad y = u + N_2^{\text{post}}\big(M(N_2^{\text{pre}}(u))\big)$$
>
> 四者的关键区别不是"LN 挪了个位置"这么简单：Post-LN 里残差主干本身被 LN 反复挤压；Pre-LN 串行给残差主干留了一条纯恒等加法通道，但 FFN 读到的是 attention **更新后**的流 $u$；Pre-LN 并行连这个先后依赖都去掉了——attention 和 FFN 读的是**同一份** $x$（$N_A,N_M$ 可以共享参数，也可以是两个独立模块，"没有先后依赖"才是 parallel 的定义，不要求共享 norm）；branch pre+post 在每个子层分支前后各放一次归一化，第二次归一化只套在分支输出上，残差主干 $x+(\cdots)$、$u+(\cdots)$ 仍是纯加法——这与 Post-LN 把归一化套在"相加之后"完全不同。四种拓扑在 §7 的 [D02]/[D08] 里都有可执行的正反面证明。**norm 放置在哪（Post/Pre/branch pre+post）和 serial/parallel 依赖是否解耦，是两个可以分别讨论的维度，不要合并成一条轴。**

**12 句话搞定 Transformer Block**：

1. **Block = 残差流 + 分层装配的子层**：attention、FFN 各自的内部变体属于同一层级内的独立选择；encoder-decoder 比 decoder-only 多一个 cross-attn 子层。
2. **decoder-only：单层看是"去掉 cross-attn"没错，但整模型看还不够**：原始 decoder 层有 masked self-attn、cross-attn、FFN 三个子层；现代 LLM decoder-only block 通常只有 causal self-attn、FFN 两个子层——这一步"去掉 cross-attn"本身没问题；但完整模型层面还要补上：encoder/memory 路径连带消失，双序列条件生成变成单序列因果建模，这是信息流拓扑和序列因子化方式的变化（§2）。
3. **GPT-2 已经是 Pre-LN**：从 GPT-2-style 到 Llama-style，真正改的是 LayerNorm→RMSNorm、learned absolute PE→RoPE（挪进 attention 内部）、GELU dense FFN→SwiGLU gated FFN、酌情 MHA→GQA，外加去掉大量 bias/dropout（§2.2、§3）。
4. **Pre-LN 栈末尾通常配 final norm**：在简化的独立增量假设下，残差流**方差**约随深度**线性**增长（$O(l)$），标准差/RMS/典型幅值才约按 $\sqrt l$ 增长；真实增长率还取决于初始化、残差缩放和分支相关性。不收口直接接输出头，训练质量通常明显下降——这是 GPT-2/Llama 等标准 recipe 的常见配置，但不是所有 Pre-LN 架构在数学上不可或缺的条件（§2.2，推导见 normalization_init_tutorial.md §5.3）。
5. **MHA→MQA→GQA→MLA 不是一条演进链**：按历史时间是 MHA→MQA→GQA（MQA 确实早于 GQA 出现）；但按 KV-head 压缩强度看，MHA 和 MQA 是同一条 GQA 压缩比数轴的两个端点；MLA 是 DeepSeek 系一条独立的低秩 latent 分支，不应接在这条轴的末端——存在多条不同的排序维度，不是只有时间线一条（§4）。
6. **Dense FFN → MoE 是另一条独立的"容量轴"**：不是"下一代必然替换"——Llama 1–3、原始 Mistral、Gemma 1–3 等代表性模型至今仍多用 dense FFN，但 Mixtral、Llama 4 等已采用 MoE（§5）。
7. **QK-Norm、logit soft-cap、branch pre+post norm、parallel residual、local/sliding attention 大多可以按需组合**，但不是无脑叠加——QK-Norm 与 soft-cap 同时使用要重新核对尺度，norm 拓扑与 norm 算子实践中明显绑定出现（§6）。
8. **FFN 参数匹配是算出来的，不是定律**：dense FFN（$8d^2$）与 gated FFN（$3dh$）参数持平时 $h=8d/3$；真实模型还要按硬件友好倍数舍入，不是所有 Llama 系都严格等于这个数（§3、§7 [D06]）。
9. **KV cache 主要跟着 $n_\text{kv}$ 走，不跟 $n_q$ 走**：MQA/GQA 减少 K/V 投影的参数、计算和 cache 显存/带宽，但不减少 Q projection 的计算量（§4，完整显存核算见 attention_tutorial.md §6 与 kv_cache_speculative_decoding_tutorial.md §2）。
10. **现代 recipe 没有完全收敛**：Pre-RMSNorm + RoPE + gated FFN + 推理友好 GQA 只是"较常见"组合；Post-LN 变体、LayerNorm、MHA、MQA、parallel residual、dense FFN、MoE 全部仍在生产环境里使用（§8 model recipe 对照表）。
11. **Cross-attention 和 self-attention 唯一的结构差异是 Q/K/V 来源**：self-attn 里 Q、K、V 都来自同一条流；cross-attn 里 Q 来自当前流、K/V 来自另一条序列，长度可以不同——除此之外 scaled dot-product、多头拆分、softmax 完全一样，不需要单独一套数学（§1）。
12. **"packed projection"只是 kernel 层面的等价重排**：把 Q/K/V（或 SwiGLU 的 gate/up）三个 Linear 拼成一个大 Linear 纯粹是实现优化，权重按行拼接后结果在对应精度下应一致，不改变任何架构语义——遇到这类问题先确认是在问工程实现还是算法设计（§7 [D07]）。

> 🎯 **本文范围：只讲"装配"，不重新推导零件内部的数学**
> softmax attention 的公式推导、FlashAttention 的 IO 复杂度、Xiong/DeepNorm 的梯度论证、MoE 路由/capacity 的完整数学、RoPE 旋转矩阵与 YaRN/NTK/MLA latent algebra、线性注意力推导、量化误差推导、LoRA 低秩公式——这些零件各自的数学在 attention_tutorial.md、normalization_init_tutorial.md、moe_tutorial.md、long_context_rope_yarn_mla_tutorial.md、linear_sparse_attention_tutorial.md、quantization_tutorial.md、lora_peft_tutorial.md 里都有专门篇幅；prefill/decode 的系统性能瓶颈、KV cache 分页/调度见 llm_inference_serving_tutorial.md。本文只关心这些零件**怎么拼成一个 block、怎么从 2017 版一路装配到现代版**，遇到深度数学或系统问题一律给出 1-2 句结论 + 链接，不重复推导。

## §1 2017 seq2seq Transformer

原始 Transformer（Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762, 2017, NeurIPS 2017）是一个**encoder-decoder**架构：encoder 把源序列编码成一组表示，decoder 在这组表示的条件下自回归生成目标序列。位置编码用固定 sinusoidal PE，在进入 encoder/decoder 堆栈**之前**直接加到 token embedding 上——它不属于任何一个 block，是 block 外围的输入处理（完整推导见 long_context_rope_yarn_mla_tutorial.md §2/§3）。论文原始配置 $d_\text{model}=512$、$d_\text{ff}=2048$（4× expansion）、8 head、6 层 encoder + 6 层 decoder；本教程演示用更小的尺寸（如 $d=96$），这是**教学配置**而非论文配置，两者不要混为一谈。

**encoder layer 和 decoder layer 是两个不同的类，不是同一个"两子层"模板的两种用法**——它们的子层数量、子层种类都不同：

| | Encoder layer | Decoder layer |
| --- | --- | --- |
| 子层 1 | Self-attn（双向，无 causal mask，只有可选 padding mask） | Masked self-attn（causal） |
| 子层 2 | FFN | Cross-attn（Q 来自本层输出，长度 $T$；K/V 来自 encoder 输出，长度 $S$） |
| 子层 3 | — | FFN |
| 归一化拓扑 | 每子层 Post-LN：$y=N(x+F(x))$ | 每子层 Post-LN，三次 |

Decoder 的三个子层顺序固定为 masked self-attn → cross-attn → FFN，每个子层各自套一层 Post-LN。Cross-attn 的 Q、K、V 来源不对称：Q 来自 decoder 自身正在生成的流（长度 $T$），K、V 来自 encoder 输出（长度 $S$），$T$ 与 $S$ 一般不相等——这正是 §7 代码 [D01] 里显式验证的 shape 分离（self-attn 与 cross-attn 的公式细节、mask 处理见 attention_tutorial.md §4.1，此处不重复推导）。

**一个具体 shape 走一遍**（对应 §7 代码 [D01] 用的教学尺寸：batch $B=2$、源序列长度 $S=5$、目标序列长度 $T=7$、$d=96$）：encoder 输入 `[B, S, d]`，经过若干层 encoder layer 后仍是 `[B, S, d]`，作为 memory 传给每一层 decoder；decoder 输入 `[B, T, d]`，masked self-attn 阶段 Q/K/V 都是 `[B, T, d]` 内部算出（causal mask 大小 $T\times T$）；cross-attn 阶段 Q 仍是 `[B, T, d]`，但 K、V 来自 encoder memory，形状 `[B, S, d]`——attention score 矩阵是 $T\times S$（而不是方阵），softmax 沿 $S$ 这一维做；输出投影后还原回 `[B, T, d]`，与 self-attn 子层的输出 shape 一致，才能继续送进 FFN 子层。**$T \ne S$ 完全不影响这条数据路径**，这正是 cross-attn "Q 长度和 K/V 长度解耦"这句话的具体样子。

> ⚠️ **两个类，不是一个类的两种调用方式**
> `VanillaEncoderLayer2017`（2 子层）和 `VanillaSeq2SeqDecoderLayer2017`（3 子层）在 §7 的代码里是两个独立的类。把它们写成同一个"通用 block 类 + 参数开关"虽然工程上可行，但会掩盖"decoder 比 encoder 多一整个 cross-attn 子层"这个结构事实——面试现场手写时，先把两个类的子层列表写清楚，比直接码代码更容易拿分。

## §2 三大家族地图 + decoder-only 分叉 + GPT-2-style 桥接

### 2.0　三大家族地图：encoder-only / decoder-only / encoder-decoder

原始 Transformer 是 encoder-decoder；本文从这里往后全部聚焦 **decoder-only**（GPT/Llama/Qwen 系 LLM 的主流选择），但面试里常被要求横向比较三大家族，这里先给一张压缩地图，不深入 encoder-only 的内部预训练机制：

| 家族 | 代表 | 子层组成 | attention 方向 | 典型训练目标 | 典型用途 |
| --- | --- | --- | --- | --- | --- |
| Encoder-only | BERT、RoBERTa | 每层：self-attn + FFN（无 causal mask） | 双向（可看到整个序列） | 掩码语言建模（MLM）等去噪目标 | 表示学习、分类、检索 embedding |
| Encoder-decoder | 原始 Transformer、T5 | encoder 层：self-attn + FFN；decoder 层：masked self-attn + cross-attn + FFN | encoder 双向，decoder 单向且条件于 encoder | 条件生成（翻译、摘要） | 双序列条件生成任务 |
| Decoder-only | GPT、Llama、Qwen | 每层：causal self-attn + FFN（无 cross-attn） | 单向（causal） | 自回归语言建模 | 通用生成式 LLM |

三者的核心区别不是"参数多少"，而是 attention 的可见方向、子层组成、训练目标这三件事绑在一起变化。本文只展开 decoder-only 这一支的完整装配细节（§2–§9）；encoder-only 的双向掩码建模机制、encoder-decoder 的完整训练目标不在本文范围内，需要时可参考对应模型各自的原始论文。

### 2.1　为什么只说"去掉 cross-attn"还不够

现代 LLM（GPT、Llama、Qwen……）用的 decoder-only block，字面上看是"原始 decoder layer 去掉 cross-attn"。**分两层看这句话**：

- **单层结构层面，这个说法是对的**：原始 decoder 是 masked self-attn → cross-attn → FFN（**三个**子层）；decoder-only block 是 causal self-attn → FFN（**两个**子层）。少的正是 cross-attn 这一个子层，逐层对比没有问题。
- **完整模型层面，这句话不完整**：只说"去掉 cross-attn"容易让人以为只是删掉了一个组件、其余不变。实际上 encoder/memory 路径也整体消失了——source/prompt 与 target 从"双流条件生成"（source 编码后条件生成 target）变成"单流因果建模"（prompt 和生成内容拼进同一条因果序列）。变的是信息流拓扑和序列因子化方式，不是任务被替换成了别的问题。

因此更完整的答案是："单层上确实是去掉 cross-attn，但完整模型层面还需要说清楚 encoder/memory 路径也一并消失、双流条件生成变成单流因果建模"——这比只说"去掉了一部分"、或者反过来把它讲成"完全是另一种架构"都更准确。

补充一点边界：并不是所有 decoder-only LLM 都彻底没有 cross-attention——部分多模态或检索增强模型会在纯 decoder-only backbone 上重新引入 cross-attention（例如让文本流 attend 图像 encoder 的输出，或 attend 检索到的文档表示）。本文和下文所有"decoder-only 没有 cross-attn"的表述，均限定为**纯 decoder-only block**（causal self-attn + FFN 两子层），不覆盖这类混合架构。

> 🎯 **面试怎么答这道根本区别题**
> 先分层回答：单层结构上是"去掉 cross-attn"（3 子层变 2 子层）；完整模型上还要加上"source/target 两条序列变成一条因果序列"这一信息流拓扑变化；再补一句"纯 decoder-only block 才没有 cross-attn，部分多模态/检索增强模型会重新引入它"——比只说"去掉了一部分"多两层准确度。

### 2.2　GPT-2-style 桥接：GPT-2 到 Llama 到底改了什么

**常见误解**：现代 LLM 才把 Post-LN 换成 Pre-LN。**事实**：GPT-2（Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI 技术报告, 2019，无 arXiv id）已经是 Pre-LN，且在堆栈末尾配了一个 final LayerNorm（记作 `ln_f`）——这是 GPT-2/Llama 等标准 Pre-LN recipe 的常见配置：不补这一层，Pre-LN 残差流膨胀后的分布会直接冲进输出头，训练质量通常明显下降；但这不是所有 Pre-LN 架构在数学上不可或缺的条件，通过残差缩放、特殊初始化或归一化输出头，也能构造出没有传统 final norm 的变体（推导见 normalization_init_tutorial.md §5.3，此处不重复）。GPT-2 block 的完整 recipe：

$$x = x + \text{Attn}\big(\text{LN}(x)\big), \qquad x = x + \text{FFN}\big(\text{LN}(x)\big), \qquad \text{（GELU dense FFN，MHA，bias=True，无 cross-attn）}$$

学习到的绝对位置嵌入（learned absolute positional embedding）依旧加在 embedding 上、位于堆栈**之外**——这一点和原始 sinusoidal PE 的"所属层级"是一致的，二者都不属于 block 内部；LLaMA 系采用的 RoPE（最初由 Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2104.09864, 2021 提出，LLaMA 是重要采用者而非提出者）则相反，**位于 attention 内部**，只作用于 Q、K 的点积计算——把它们看成同一 block 成员之间的"无缝替换"是不准确的，位置信息在 GPT-2-style 里由外部嵌入携带，在 Llama-style 里由 attention 内部的旋转携带，两者"归属的层级"不同（完整推导见 long_context_rope_yarn_mla_tutorial.md §2）。

GPT-2-style → Llama-style 桥接表（这才是真正被改动的部分）：

| 维度 | GPT-2-style | Llama-style |
| --- | --- | --- |
| 归一化拓扑 | Pre-LN（已经是） | Pre-LN（不变） |
| 归一化类型 | LayerNorm | RMSNorm |
| 位置编码 | learned absolute（堆栈外，加在 embedding 上） | RoPE（attention 内部，仅作用于 Q/K） |
| FFN | GELU dense（2 矩阵） | SwiGLU gated（3 矩阵） |
| Attention | MHA | MHA 或 GQA（按具体 checkpoint 规格） |
| bias | Linear、LayerNorm 均有 bias | 通常全部去掉 bias |
| dropout | 训练中使用 | 大规模预训练通常去掉或降到极小 |
| final norm | 有（`ln_f`，LayerNorm） | 有（RMSNorm） |

> ✅ **"GPT-2→Llama 改了什么"标准答案骨架**
> 先声明"两者都已经是 Pre-LN"（堵住"Pre-LN 是现代发明"这个常见误答），再按 归一化类型 → 位置编码归属层级 → FFN 结构 → attention 头配置 → bias/dropout 五项逐条对比，最后提一句 final norm 两边都有、只是类型不同。

## §3 Llama-style 核心 recipe

在 GPT-2-style 桥接的基础上，Llama-style（Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288, 2023；Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783, 2024）把每个槽位换成如下具体实现，本节先固定用 MHA 保持数据路径清晰，GQA 留到 §4 单独展开：

$$x = x + \text{Attn}_{\text{RoPE}}\big(\text{RMSNorm}(x)\big), \qquad x = x + \text{SwiGLU}\big(\text{RMSNorm}(x)\big)$$

- **RMSNorm**：只做 re-scale，不做 mean-centering，无 bias——完整公式与"为什么去掉 centering 仍常能保持相当效果"的论证见 normalization_init_tutorial.md §4（Zhang & Sennrich 的理论动机和实验结果表明这**往往**能达到与 LayerNorm 相当的效果，不是"绝不掉点"的普适结论），这里只记结论：$y = \dfrac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\odot w$。
- **RoPE**：作用于 attention 内部的 Q、K，V 不旋转；标准实现通常缓存**已旋转**的 K，增量解码时新 token 按真实 position 施加旋转即可直接用于点积——若某个实现选择缓存未旋转的 K，则必须额外保留每个位置并在使用时按正确 position 旋转，不能把历史 K 当成当前 position 处理（这样做数学上同样正确，只是更慢，因此不是常见选择；MLA 的 cache 契约又完全不同，见 §4.2）；完整复数视角推导、频率选择、YaRN/NTK 扩展见 long_context_rope_yarn_mla_tutorial.md §2/§5/§6，此处只取"装配契约"。
- **SwiGLU**：$\text{down}\big(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)\big)$，三矩阵 gated FFN；GeGLU 同结构换 GELU 门（Shazeer, *GLU Variants Improve Transformer*, arXiv 2002.05202, 2020）。相近参数量/计算预算下，gated FFN **往往**有更好质量——这是一条经验规律而非严格证明的定律，本文不重新论证，只用其参数匹配算术（§7.3 [D06]）。
- **bias、dropout**：Q/K/V/O 和 FFN 三矩阵通常全部去掉 bias；大规模预训练常去掉或大幅降低 dropout。
- **final norm**：栈末尾一次 RMSNorm，收口膨胀的残差流。
- **数值精度边界**：RMS/方差归约、softmax、final logits 这类对数值范围敏感的计算，常见实现会在更高精度（如 fp32）下算、再 cast 回激活 dtype（bf16/fp16）——这是装配正确性的一部分，不属于量化数学，完整的低精度误差分析见 quantization_tutorial.md。

> ⚠️ **block 内部 vs "block 外围 recipe"，别混写**
> Weight tying（输入 embedding 与 LM head 共享参数）、embedding scaling、LM head 的 soft-cap、dropout 的整体开关，这些属于"完整 LM recipe"的一部分，但**严格来说位于 block 之外**——token embedding 和 LM head 不属于任何一个 Transformer block。面试被问"block 长什么样"时把这些混进 block 定义里，会让考官怀疑你没分清"一层 block"和"整个模型"的边界。

"block 外围 recipe"具体包含哪些字段，逐条列清楚：

| 外围字段 | 一句话说明 |
| --- | --- |
| Weight tying | 输入 token embedding 矩阵与 LM head 投影矩阵共享同一份参数，省一份 $Vd$ 的参数量 |
| Embedding scaling | 有些实现把 token embedding 乘一个与 $\sqrt{d}$ 相关的常数，再送进第一层 block |
| LM head soft-cap | 部分模型（如 Gemma 系）在最终 vocabulary logits 上也做一次 §6.2 那种有界变换，机制上和 attention-logit soft-cap 是同一类工具、作用在不同张量上 |
| Dropout 全局开关 | 是否启用 dropout、dropout rate 这些超参数通常是整个模型统一的训练配置；但 dropout **运算本身**通常执行在 attention 权重、FFN 输出、residual branch 内部——"谁来配置"和"算子实际在哪里跑"是两件不同的事，不要笼统合并成"block 外的属性" |

这四项都**不出现**在 §7/§8 描述的任何一个 block 类里——`ModernDecoderBlock` 的 `forward` 只接收残差流 `x`，从不知道外面是否做了 embedding scaling、输出头是否 tied。

这张表和 §0 的分层地图对应：weight tying、embedding scaling、LM head 属于"模型外壳"层；final norm、local/global 交替、dense/MoE 层排布这些跨层 schedule 属于"stack 调度"层，都不属于任一单个 block，也不属于"模型外壳"——三者是三个不同的层级，不要笼统合并成"block 之外的东西"。

## §4 推理内存轴：MHA → MQA/GQA，MLA 是独立分支

**这条轴首先解决的问题是 KV cache 显存**：自回归生成时每一步都要把新 token 的 K、V 追加进 cache、再对整段 cache 做 attention，cache 的大小直接决定了同一张卡上能同时服务多少条请求、能撑多长的上下文。这条轴与 §5 的"容量轴"、§6 的稳定化机制在分析层面大多可以分开讨论，但不是严格互不相干——GQA/MQA 减少 KV head 数的同时，也改变了 K/V 投影本身的参数量和计算量（下文 $2d^2(1+n_\text{kv}/n_q)$ 公式就是证据），"KV 显存"和"模型总参数量"并不是完全无关的两件事，只是二者的量级和影响机制不同：KV cache 主要影响推理时的显存/带宽，K/V 投影参数则是训练时也存在的固定开销。

### 4.1　MHA / MQA / GQA：同一条轴上的不同压缩比

历史顺序必须澄清：

> ⚠️ **不是 MHA→GQA→MQA→MLA 的一条线**
> 正确顺序：MHA 是基线；**MQA（Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv 1911.02150, 2019）早于 GQA 出现**；GQA（Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023）是 MHA 与 MQA 之间的质量-显存折中；**MLA 是 DeepSeek 系列的另一条低秩 latent-compression 分支**，不是"把 $n_\text{kv}$ 再减一次"。四者更适合理解成"KV 显存"这一个设计轴上并存的几种设计点，而不是所有模型依次爬过的历史阶梯。

| 变体 | Q heads | KV heads | 关系 |
| --- | --- | --- | --- |
| MHA | $n_q$ | $n_q$ | 基线，$n_\text{kv}=n_q$ |
| MQA | $n_q$ | $1$ | 所有 Q head 共享同一组 K/V |
| GQA | $n_q$ | $n_\text{kv}$（$1 \lt n_\text{kv} \lt n_q$，且 $n_q \bmod n_\text{kv}=0$） | 每组 $n_q/n_\text{kv}$ 个 Q head 共享一组 K/V |

装配契约（完整显存核算、per-token-per-layer 公式、具体模型 GB 数见 attention_tutorial.md §6 与 kv_cache_speculative_decoding_tutorial.md §2，此处不重复推导）：

- KV cache 大小只随 $n_\text{kv}$ 缩放，**与 $n_q$ 无关**——MQA/GQA 减少 K/V 投影本身的参数和计算量，也减少 KV cache 显存/带宽；但 Q projection 的计算量不随 $n_\text{kv}$ 变化，不能笼统说成"只省显存/带宽"。
- **标准实现通常在施加 RoPE 之后缓存 K，V 从不旋转**；增量解码时新 token 的 Q、K 用真实 position offset 施加 RoPE，历史 cache 里已旋转的 K 不再重算——若某实现选择缓存未旋转的 K，则必须额外保留 position 并在读取时按正确 position 旋转，否则会把所有历史 key 当成同一个 position（这样做数学上可行但更慢，因此不是常见选择）。
- GQA/MQA 的 cache **只存 $n_\text{kv}$ 个 head**；把 K/V 广播到 $n_q$ 个 Q head 只能发生在 attention 计算内部（用 view/reshape/分组 einsum），**不能把广播后的张量写回 cache**——这是最容易在手写实现里踩的坑，§7 [D05] 直接把这条做成可执行断言。

举例感受这条轴的实际收益：LLaMA-2-70B 若用 vanilla MHA，4096 上下文下单样本 KV cache 约 10 GB；换成 GQA（$n_\text{kv}=8$）后降到约 1.25 GB——这说明 GQA 对 70B 级模型的推理服务非常有吸引力，也是许多大模型采用 GQA 而非 MHA 的重要原因之一（模型的最终选择还受质量、训练兼容性、服务系统等因素影响，不是单一因素决定的；精确核算见 attention_tutorial.md §6.1 与 kv_cache_speculative_decoding_tutorial.md §2.2，此处不重复推导）。

### 4.2　MLA：独立的低秩 latent 分支

**MLA（DeepSeek-V2, May 2024, arXiv 2405.04434）是独立分支，不是"GQA 更狠的版本"**：GQA 在 head 维度上做压缩（多个 Q head 共享 K/V head），MLA 在 hidden 维度上做**低秩投影**压缩（把每 token 的 K/V 压进一个 $d_c \ll n_q d_h$ 的 latent 向量），并且必须把 RoPE 单独解耦成一份共享的小维度分量才能保住"absorb 上投影矩阵进 query 侧"的推理加速技巧——完整推导（absorbing trick、为什么 RoPE 不能直接吸收、解耦方案、cache 总量公式）见 long_context_rope_yarn_mla_tutorial.md §9，此处只记结论：MLA 的 cache 契约是"存 latent $c_t^{KV}$ + 一份共享 RoPE key"，而不是"存更少的 K/V head"。

> 💡 **一句话区分三者"压什么"**
> GQA 压的是 **head 数量**（多个 Q head 挤一组 K/V）；MQA 是 GQA 的极端情形（$n_\text{kv}=1$）；MLA 压的是**每个 token 的表示维度**（K/V 整体投影进一个低秩 latent），且天然要解决 RoPE 的位置依赖问题。三者不是同一个刻度上的三个点，是两种不同的压缩思路。

## §5 容量轴：Dense FFN → MoE

**这条轴解决的问题是模型容量**，大体可以和 §4 的显存轴、§6 的稳定化机制分开分析，但同样不是严格正交——MoE 对外的**主激活张量**接口确实不变（$[B,T,d]\to[B,T,d]$），但训练契约还要多传 router loss/统计量，系统部署契约还涉及 expert parallelism、all-to-all 通信，这些都不是"接口完全不变"能概括的（细节见下文与 §7 [D09]）。MoE 把 FFN 换成 $N$ 个 expert + 一个 router，每个 token 只走 $k \ll N$ 个 expert——总参数上升、每 token 激活参数不变。完整的路由公式（token-choice top-k、expert-choice、DeepSeek 的 aux-loss-free bias 更新）、capacity factor、load-balancing loss、token dropping 见 moe_tutorial.md §2–§4，此处只讲"装配契约"：

- **block 对外的主激活张量接口不变**：无论 FFN 是 dense 还是 MoE，都是 `[B, T, d] -> [B, T, d]`，residual add 的位置不变——这也是 §7 [D09] 里"把 `ffn` 换成 `MoEFFN` 而 `ModernDecoderBlock` 代码一行不用改"的直接原因；但训练时的返回值通常多一项 router/负载均衡相关的辅助 loss 或统计量，分布式部署时通常还要引入 expert parallelism 和 all-to-all 通信——"张量 shape 契约不变"和"训练/系统契约不变"是两件不同的事（§9 有专门的三层 contract 拆分）。
- **变的是 FFN 内部**：增加了 router、expert 选择（可能还有 shared expert）、以及训练时的辅助均衡损失/偏置更新；**总参数量和每 token 激活参数量必须分开报告**，这是面试里最容易被追问的一句。

这条轴的收益直觉：Mixtral（Jiang et al., *Mixtral of Experts*, arXiv 2401.04088, 2024）每 token 激活约 12.9B 参数，并在论文报告的多数/全部评测项目上匹配或超过 Llama 2 70B——总参数远大于 13B，其单 token 计算规模大致接近十余 B 的 dense 模型，但并非严格 FLOPs 等价（完整"为什么稀疏激活是好主意"的三层论证见 moe_tutorial.md §1）。

> ⚠️ **Dense → MoE 不是所有现代 block 的"下一步"**
> Llama 1–3、原始 Mistral 7B、Gemma 1–3 等代表性模型采用 dense FFN；但 Mixtral、Llama 4 等已采用 MoE。DeepSeek-V2（DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv 2405.04434, 2024）与 DeepSeek-V3（DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437, 2024）选择 MoE 是用通信/路由/负载均衡的工程复杂度换更大的稀疏参数容量；V3 采用的具体负载均衡机制单独来自 Wang, Gao, Zhao, Sun & Dai, *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*（arXiv 2408.15664, 2024），不应把这篇机制论文当成"V2/V3 为什么选择 MoE"这件事本身的主引用。不是所有模型"迟早都要走的路"，把 Dense→MoE 和 §4 的 KV-cache 轴混成同一条"演进链"，是这部分最常见的失分点。

## §6 数值稳定化 / 连接结构 / 执行 overlay

这一节把之前零散的"稳定化技巧"按 §0 的分层地图重新归类成四类：①残差依赖与 norm/scaling 装配（"单个 block"层）；②attention 数值稳定控制（"算子内部"层，控制的是数值尺度）；③attention 连接结构（同样是"算子内部"层，控制的是可见范围而不是数值）；④执行/适配 overlay（"执行/适配 overlay"层，只做指路）。四类大多可以独立分析，但不是无条件可以任意叠加——同类内部经常存在真实约束（例如 QK-Norm 和 soft-cap 同时使用需要重新核对尺度）。

### 6.1　残差依赖、norm 放置与 branch scaling/gating

#### 6.1.1　branch pre+post norm（常被口误叫"sandwich norm"）

更准确的写法是每个子层分支的**前后各放一次归一化**，且第二次 norm 只作用在分支输出上，不作用在残差和上：

$$y = x + N_\text{post}\big(F(N_\text{pre}(x))\big)$$

这和原始 Post-LN 的 $N(x+F(x))$ 不是同一回事——Post-LN 的归一化套在"残差相加之后"，branch pre+post 的第二次归一化只套在"分支输出"上，残差主干 $x+(\cdots)$ 仍是纯加法。Gemma-2 用这种拓扑并配合 attention-logit/final-logit soft-capping；Gemma-3 沿用这一归一化拓扑，但把 soft-capping 换成了 QK-Norm（完整变体谱系见 normalization_init_tutorial.md §6.2）。

#### 6.1.2　Parallel residual（GPT-J、PaLM）

核心公式已在 §0 给出，$y = x + \text{Attn}(N_A(x)) + \text{MLP}(N_M(x))$——attention 和 FFN 没有本层内的先后依赖，都读同一份归一化输入。GPT-J（Wang & Komatsuzaki, *GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model*, EleutherAI 开源发布, 2021，无正式论文/arXiv id）用单一共享 LayerNorm 实现这个公式；PaLM（Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311, 2022）延续同样思路。**"parallel"描述的是"没有本层内先后依赖"这个结构关系，不要求代码必须让 attn 和 FFN 共享同一组 norm 参数**——用两个独立的归一化模块、只要都读原始 $x$，同样满足 parallel 的定义（§7 [D08] 的正反面证明按这个更宽的定义来写）。

#### 6.1.3　residual / branch scaling 与 gating：另一个独立的装配位置

某些 recipe 不仅决定"在哪里加 norm"，还会在残差分支上乘一个固定或可学习的缩放系数——例如 **LayerScale**（对每个分支输出乘一个逐通道、可学习、小初值的系数）、**DeepNorm**（对残差分支乘一个与层数相关的固定放大系数，配合特定初始化让深层网络更容易训练稳定）。这是另一个独立的可选装配位置：加不加缩放、缩放是固定常数还是可学习参数、作用在哪个分支——完整的初始化与梯度论证见 normalization_init_tutorial.md §7（DeepNorm）及相关章节，本文只记装配契约：**residual/branch scaling 和 norm placement 是两个可以分别选择、也可以同时使用的维度**，不要把它们合并成同一件事。

### 6.2　Attention 数值稳定控制：attention scale、QK-Norm、logit soft-cap

三个机制作用在同一条计算链的不同位置，都属于"算子内部"层：

- **attention scale**：标准的 $1/\sqrt{d_h}$（或可学习温度）缩放点积，控制 softmax 输入的整体幅度量级，是最基础、几乎所有实现都有的数值控制（推导见 §9 Q4）。
- **QK-Norm**（Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245, 2020, EMNLP 2020 Findings）的原始定义是在点积**之前**对每个 head 的 Q、K 分别做 L2 归一化，并用一个可学习的缩放/温度参数**取代**标准固定的 $1/\sqrt{d_h}$ 缩放——后续模型里的具体实现细节可能有所不同（例如是否保留固定缩放、是否额外配合 RMSNorm 式归一化）。
- **logit soft-capping** 在点积**之后**对 attention score 或最终 vocabulary logits 施加类似 $c\tanh(z/c)$ 的有界变换，管住的是已经算出来的分数/logits 幅度。

三者常被面试者混成"都是稳住数值"，但作用的位置和管住的量完全不同，**也不能无脑同时叠加**——同时启用需要重新核对整体尺度，而不是简单相加；完整的"为什么 attention logits 会爆炸"论证见 normalization_init_tutorial.md §6.3。

### 6.3　Attention 连接结构：local / sliding / global

这是 attention **连接结构/可见范围（connectivity pattern）**的设计轴，不是"attention backend"，也和位置编码方案是两回事——FlashAttention 才是典型的执行 backend：它改变 exact attention 的计算顺序与 IO 调度，不定义新的连接图或 block 拓扑（完整 IO 复杂度推导见 attention_tutorial.md）。Mistral（Jiang et al., *Mistral 7B*, arXiv 2310.06825, 2023）用固定窗口的 sliding-window attention；Gemma-2 在层间 1:1 交替 local 窗口和 global 全局 attention；Gemma-3 更接近"多个 local 层后插入一个 global 层"的周期性 pattern（约 5:1，不是严格一层 local 一层 global）。这些都不改变 block 外层的 residual 接口，只改变 attention 内部"能看到多远"；但"哪几层用 local、哪几层用 global"这个排布本身是**stack 调度层**的决定（参见 §0 分层地图），不是单个 block 的私有属性——完整的滑窗/StreamingLLM 机制见 long_context_rope_yarn_mla_tutorial.md §10。

### 6.4　执行 / 适配 overlay：FlashAttention、量化、LoRA（只指路）

这三个机制都属于 §0 分层地图里的"执行/适配 overlay"层，不是和 GQA/MoE 同级、需要互斥选择的架构槽位，而是叠加在已经选定的架构之上、用来加速或适配部署环境的策略：

- **FlashAttention**：只改变 exact attention 的计算顺序、IO 调度和内存访问模式，不改变 attention 的数学结果，也不定义新的连接图——可以叠加在 MHA/GQA/MQA/MLA、任意 local/global pattern 之上。完整 IO-aware 复杂度推导见 attention_tutorial.md。
- **量化**：把权重/激活值表示成更低精度的数值格式，是数值表示层面的 overlay，可以和上面任何架构选择叠加，代价是量化误差；完整误差分析见 quantization_tutorial.md。
- **LoRA**：在已训练权重旁加一对低秩矩阵做参数高效微调，是训练/适配阶段的 overlay，不改变 block 的前向架构；完整低秩公式见 lora_peft_tutorial.md。

三者都不应该被当成"和 GQA、MoE 平级、必须三选一"的架构槽位——它们是可以叠加在几乎任意架构组合之上的独立维度。

### 6.5　小结：这几类机制怎么组合

上面几类机制大多可以独立分析，实践中经常同时出现在同一个模型里（例如 Gemma-3 同时用 branch pre+post norm + QK-Norm + local/global 交替 attention），常见的检查顺序是：先确定 attention 的可见范围（global / local / sliding，§6.3），再确定 Q/K 或 logits 的数值稳定化方式（QK-Norm 或 soft-cap，§6.2，二者同时使用要重新核对尺度，不是无脑叠加），最后确定归一化拓扑与 residual scaling（Pre-LN / branch pre+post / parallel，§6.1）。这几个维度大多可以分别组合，但不是数学上严格无约束的正交——同一模型内的具体取值往往还受训练稳定性、kernel 支持、既有 checkpoint 兼容性等工程因素约束，这也是 §8 model recipe 表里同一列会出现多种取值、但组合空间比"完全自由"窄得多的原因。

> 🎯 **这一节的面试答题原则**
> 遇到任何一个"新稳定化技巧"，先问自己三件事：它作用在哪个张量上（Q/K？attention score？残差流？）、它是否改变 block 的输入输出 shape 契约（通常不改变）、它和 §4/§5 的两条主轴是否大体独立（通常大体独立，但检查一下是否有真实耦合）。把新名词往这三个问题上一挂，回答会比单纯罗列"这是什么"扎实得多。

## §7 Capstone：组装、端到端 forward、参数核算与可执行验证

### 7.1　组装一个 Llama-3-like block，再换成 MoE

把前面几节的槽位按"较常见现代组合"实际拼一遍（更精确的说法是 "Llama-3-like core block"，而非声称精确复现某个具体 checkpoint——除非同时对齐它的 head 数、KV-head 数、intermediate size、RoPE base、bias 设置、dtype 行为）：

1. **归一化槽位（block 内）**：Pre-RMSNorm，两次（attention 前一次、FFN 前一次）。
2. **Attention 槽位**：causal self-attn，Q/K 内部施加 RoPE，$n_q$ 个 query head 分组共享 $n_\text{kv}$ 个 KV head（GQA；$n_\text{kv}=n_q$ 时退化为 MHA，见 §7.4 [D05]），K 旋转后进 cache，V 不旋转，no bias。
3. **FFN 槽位**：SwiGLU gated FFN，$h \approx 8d/3$ 按硬件倍数取整，no bias。
4. **单个 block 组装**：$x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x))$，$x=x+\text{SwiGLU}(\text{RMSNorm}(x))$——这就是**一个 block** 的完整定义，重复 $N$ 次得到整个 decoder stack。
5. **stack 级收尾（不属于任何一个 block）**：$N$ 个 block 全部堆完之后，再做一次 **final RMSNorm**，才送进 LM head：$x_\text{out} = \text{RMSNorm}\big(\text{Block}_N(\cdots\text{Block}_1(x_0))\big)$。final norm 是 stack 调度层的操作，不是第 1–4 步里任何一个 block 的组成部分（参见 §0 分层地图、§3 的 block-vs-外围边界讨论）。

用 §7.4 代码里的教学尺寸把这个 block 的参数量具体算一遍（$d=96$、$n_q=8$、$n_\text{kv}=2$、gated hidden $=256$）：GQA attention 部分 $2d^2(1+n_\text{kv}/n_q)=23{,}040$ 个矩阵参数，SwiGLU FFN 部分 $3dh=73{,}728$ 个矩阵参数，两个 RMSNorm 各只有 $d=96$ 个逐通道缩放参数——单个 block 的矩阵参数合计约 $23{,}040+73{,}728=96{,}768$，FFN 占了绝大部分，这也是为什么"FFN hidden size 怎么选"（§3、§7.3）比"用几个 KV head"对参数总量的影响更大。

**接下来把 FFN 槽位换成 MoE，其余什么都不用动**：

$$x=x+\text{Attn}_\text{RoPE,GQA}(\text{RMSNorm}(x)), \qquad x=x+\text{MoEFFN}(\text{RMSNorm}(x))$$

`MoEFFN` 对外的**主激活张量**接口仍是 `[B,T,d] -> [B,T,d]`——`ModernDecoderBlock` 的代码完全不用感知内部是 dense 矩阵乘法还是 router + expert 选择，这也是 §7.4 [D09] 直接验证的性质。但这只说明 §5 强调的"主激活张量接口不变"，不代表这次替换和训练/系统契约完全无关——训练时通常要多返回 router/负载均衡相关的量，分布式部署时通常要引入 expert parallelism（详见 §5、§9 里的三层 contract 拆分）。

> ✅ **capstone 小结：一个 block 的 recipe 由哪些字段描述**
> 归一化拓扑（Post/Pre-serial/Pre-parallel/branch pre+post）+ 归一化类型（LayerNorm/RMSNorm）+ 位置编码方案与归属层级（learned absolute 在堆栈外/RoPE 在 attention 内）+ attention 变体（MHA/MQA/GQA/MLA，是否局部窗口，是否 QK-Norm，attention scale/temperature）+ FFN 变体（dense/gated/MoE，intermediate size）+ bias、dropout、norm epsilon 的取舍、residual scaling/gating（如 LayerScale，见 §6.1.3）——这些覆盖了常见面试所需的粗粒度 recipe 字段，但**不是这个 block 的全部设计空间**：至少还遗漏了 $d_h/d_q/d_v$ 与 head 数的具体取值、RoPE base/rotary fraction/context extension 方案、激活函数的精确近似实现、attention 与 FFN 是否共享某些参数等更细粒度的选择。**final norm、local/global 交替、dense/MoE 层排布不属于这份 block 字段清单**——它们是 stack 调度层的决定（第 5 步），不要混进单个 block 的定义里。

### 7.2　端到端 forward walkthrough：现代 decoder-only block 的完整数据流

§1 给出了 2017 encoder-decoder decoder 的完整 walkthrough，但现代 decoder-only block 反而缺一份同等粒度的数据流讲解——这里补上，用 §7.1 的教学尺寸（$d=96$，$n_q=8$，$n_\text{kv}=2$，gated hidden $=256$）走一遍完整 tensor 流：

$$x \in [B,T,d] \;\xrightarrow{\text{RMSNorm}}\; h_1 \;\xrightarrow{\text{Q/K/V 投影 + reshape}}\; Q\in[B,n_q,T,d_h],\;K,V\in[B,n_\text{kv},T,d_h] \;\xrightarrow{\text{RoPE}}\; Q',K' \;\xrightarrow{\text{GQA attention}}\; O\in[B,T,d] \;\xrightarrow{\text{residual add}}\; u = x + O$$

$$u \;\xrightarrow{\text{RMSNorm}}\; h_2 \;\xrightarrow{\text{gate/up 投影}}\; \text{SiLU}(\text{gate}(h_2))\odot\text{up}(h_2) \;\xrightarrow{\text{down 投影}}\; F\in[B,T,d] \;\xrightarrow{\text{residual add}}\; y = u + F$$

这就是单个 block 的完整前向；把这个 block 重复 $N$ 次，最后再套一次 stack 级 final RMSNorm（§7.1 第 5 步），才是完整的 decoder-only 主干：$y_\text{final} = \text{RMSNorm}\big(\text{Block}_N(\cdots(\text{Block}_1(x_0)))\big)$。

这条数据流在三种执行模式下的具体形态不同：

| 执行模式 | Q/K/V 长度 | mask | KV cache | 典型场景 |
| --- | --- | --- | --- | --- |
| 训练 / full-forward | Q、K、V 都是整段 $T$ | 一次性下三角 causal mask $[T,T]$ | 不使用 | 训练、teacher-forcing 评估、一次性给完整 prompt 打分 |
| Prefill（推理第一步） | Q、K、V 都是整段 prompt 长度 $S$ | causal mask $[S,S]$ | 写入 cache（K、V 各 $n_\text{kv}$ 份，长度 $S$） | 拿到用户 prompt 后的第一次前向，为后续 decode 建 cache |
| One-token decode（推理逐步） | Q 长度 $1$（新 token）；K、V 从 cache 里的历史长度 $t$ 增长到 $t{+}1$ | 新 token 对 $0..t$ 全部可见，退化成"整行 True"（但静态 cache/left padding/packed sequence 等场景仍可能需要额外 mask，见 §9） | 读旧 cache + 追加新 K/V | 自回归生成的每一步 |

三种模式共享同一套 RMSNorm→RoPE→attention→残差 的代码路径，区别只在于 Q/K/V 的实际长度、mask 的形状、以及是否读写 cache——这也是 §7.4 [D04] 用同一份实现同时验证 full-forward 和 cached decoding 数值一致的原因（如果两条路径用了不同代码，这条测试就失去了交叉验证的意义）。prefill/decode 各自的系统性能瓶颈（算术强度、批处理调度）不在本文范围，见 llm_inference_serving_tutorial.md。

### 7.3　参数量与 KV Cache 公式核算

**FFN 参数量**（忽略 bias）：dense（ReLU/GELU，两矩阵）$8d^2$；gated（SwiGLU/GeGLU，三矩阵）$3dh$。令二者相等解出 $h=8d/3$——这是**参数量匹配算出来的常见默认**，不是架构铁律；真实 Llama 系实现通常从 $4d$ 出发乘 $2/3$，再按硬件友好倍数（以及可选 multiplier）舍入，intermediate size 可能明显偏离 $8d/3$。§7.4 [D06] 用 $d=96$、dense hidden $=384$（$4d$）、gated hidden $=256$（$8d/3$）两边都验证出恰好 73,728 个矩阵参数。

**Attention 参数量**（忽略 bias）：MHA 的 Q/K/V/O 四个投影共 $4d^2$；GQA 为 $2d^2(1+n_\text{kv}/n_q)$（$n_\text{kv}=n_q$ 时退化回 $4d^2$，$n_\text{kv}=1$ 时为 MQA 的极小值）。§7.4 [D06] 用真实 `.numel()` 核对了这两个公式。这两个公式成立都需要假设 $d=n_qd_h$、Q/K/V 使用相同 head dim、输出投影是 $d\times d$、忽略 bias——不满足这些假设时公式需要相应调整。

**KV cache**（下表给的是每 token、每层的**元素数**，不是字节数；完整显存占用还要乘上 batch size $B$、序列长度 $L$、层数 $N_\text{layer}$，以及每元素字节数——fp16/bf16 通常 2 字节，fp8 通常 1 字节；完整跨模型 GB 数见 kv_cache_speculative_decoding_tutorial.md §2.2）：

| 变体 | Cache 大小（元素数/token/layer） |
| --- | --- |
| MHA | $2 n_q d_h$ |
| GQA | $2 n_\text{kv} d_h$ |
| MQA | $2 d_h$ |
| MLA | $d_c + d_h^R$（latent 维度 + 共享 RoPE 分量，无 "$\times 2$"，因为 K/V 共享同一 latent） |

### 7.4　从零实现与可执行验证：`code/transformer_block.py`

完整可跑脚本见 [`code/transformer_block.py`](code/transformer_block.py)（纯 PyTorch、CPU 秒级跑完 9 个 demo [D01]–[D09]）。四个入口类对应本文四个"block 时代"：`VanillaEncoderLayer2017`、`VanillaSeq2SeqDecoderLayer2017`（§1）、`GPT2StyleDecoderLayer`（§2）、`ModernDecoderBlock`（§3/§7.1，`ffn=` 可换成 `DenseFFN`/`GatedFFN`/`MoEFFN`）。正文只展示三段最核心的代码；其余 demo 作为脚本里的独立函数，不再贴出。

**[D02] 残差拓扑测试**（§0 四个公式的可执行版：把子层输出置零，Pre-LN 应严格输出 $x$，Post-LN 应输出 $N(x) \ne x$）：

```python
def zero_sublayer(_x):
    return torch.zeros_like(_x)

post_ln_out = ln(x + zero_sublayer(x))       # y = N(x + F(x)) -> y = N(x)
pre_ln_out  = x + zero_sublayer(ln(x))       # y = x + F(N(x)) -> y = x

assert torch.allclose(pre_ln_out, x, atol=1e-6)          # Pre-LN: 严格恒等
assert torch.allclose(post_ln_out, ln(x), atol=1e-6)     # Post-LN: 严格等于 N(x)
assert not torch.allclose(post_ln_out, x, atol=1e-4)     # 且一般不等于 x
```

**[D04] Full forward vs cached decoding**（eval() + 无 dropout 下，整段前向 与 逐 token 增量解码 必须数值一致——同时检验 causal mask、RoPE position offset、cache append 三处正确性）：

```python
causal = build_causal_mask(torch.arange(T), torch.arange(T))
out_full = block(x, cos, sin, mask=causal)                  # 一次性算完整段

cache = KVCache()
outs = []
for t in range(T):
    q_pos, k_pos = torch.tensor([t]), torch.arange(t + 1)
    mask_t = build_causal_mask(q_pos, k_pos)                 # 新 token 看 0..t 全部
    outs.append(block(x[:, t:t+1, :], cos, sin, mask=mask_t,
                       kv_cache=cache, start_pos=t))
out_cached = torch.cat(outs, dim=1)
assert torch.allclose(out_full, out_cached, atol=1e-4)
assert cache.k.shape[1] == n_kv_heads                        # cache 头数从未涨到 n_q_heads
```

**[D08] Parallel vs sequential dependency**（用 forward hook 抓 FFN 的真实输入张量，和"独立重算出的期望张量"做**精确匹配**——而不是脆弱的"两个随机数不相等"）：

```python
captured = {}
seq_blk.ffn.register_forward_hook(lambda m, i, o: captured.setdefault("x", i[0]))
seq_blk(x, causal)
a, _ = seq_blk.attn(seq_blk.norm1(x), mask=causal)
assert torch.allclose(captured["x"], seq_blk.norm2(x + a), atol=1e-5)   # 串行：FFN 看到"更新后"的流

captured.clear()
par_blk.ffn.register_forward_hook(lambda m, i, o: captured.setdefault("x", i[0]))
par_blk(x, causal)
assert torch.allclose(captured["x"], par_blk.norm(x), atol=1e-6)       # 并行：FFN 看到"原始"的流
```

**脚本应满足的关键不变量**（逐条对应 [D01]–[D09]，这里按"验证意图"描述，具体断言实现以脚本为准）：

1. **[D01] Shape contract**：四个入口类都保持 `[B,T,d]` 残差宽度不变；cross-attn 在 $T\ne S$ 时仍正确返回长度 $T$ 的输出，且应验证真正读取了 memory（改变 memory 会改变输出），而不只是验证 shape。
2. **[D02] 残差拓扑测试**：子层输出置零后，Pre-LN 类 block 严格输出 $x$；Post-LN 类 block 输出 $N(x)$，一般不等于 $x$；两次子层各自的 LN 应分别比较，不能用同一个 $N(x)$ 自证自身。
3. **[D03] Causal 不泄露**：只扰动位置 $t{+}1$ 之后的 token，位置 $0..t$ 的输出必须原封不动；应确认这条不变量确实由 attention 分支贡献。
4. **[D04] Full forward vs cached decoding**：eval() + 无 dropout 下，整段前向与逐 token 增量解码在所有位置上数值一致，且 cache 头维度全程保持 $n_\text{kv}$——这是排查 causal mask / RoPE offset / cache 顺序 bug 最直接的手段，但不是唯一需要覆盖 attention 分支非零贡献的测试。
5. **[D05] MHA/MQA/GQA 退化**：$n_\text{kv}=n_q$ 时的通用实现与独立手写的 plain MHA reference 严格一致；$n_\text{kv}=1$（MQA）和中间取值（如 $n_\text{kv}=2$）都应有独立的 `repeat_interleave` reference 比较，不能只测退化到 $n_\text{kv}=n_q$ 这一个特例。
6. **[D06] 参数公式**：$d=96$ 时 dense FFN（hidden $=4d$）与 gated FFN（hidden $=8d/3$）矩阵参数都恰好 73,728；GQA/MHA attention 矩阵参数与 $2d^2(1+n_\text{kv}/n_q)$、$4d^2$ 公式精确相等；$d_\text{model}$ 不整除头数的非法配置应在构造时就报错。
7. **[D07] Packed projection 数值等价**：packed QKV（及 packed gate+up）与三个（或两个）独立 Linear 用相同权重时输出在数值上一致（不是"逐位"意义上的 bit-identical）。
8. **[D08] Parallel vs sequential**：forward hook 同时抓到 attention 和 FFN 的真实输入，串行 block 精确等于 $N_2(x+\text{Attn}(N_1(x)))$，并行 block 两个分支都精确等于 $N(x)$。
9. **[D09] Backward smoke test**：dense block 全部参数梯度有限；MoE-swapped block 应遍历所有非"本 batch 未使用"expert 的参数并检查非零梯度，router/attention/norm 等其余参数也应正常拿到梯度。

> ✅ **三段代码各钉死一个高频误区**
> [D02]：残差拓扑差异不是"抽象说法"，把子层清零就能拿到精确的 $x$ 或 $N(x)$；[D04]：full-forward 与 cached decoding 必须在所有位置数值一致，这是排查 causal mask / RoPE offset / cache 顺序 bug 最直接的手段，但不是唯一的正确性证明；[D08]：判断"parallel 是否等价 sequential"要重构出 attention 和 FFN **各自真实吃到的输入张量**做精确匹配，而不是简单地断言两个输出不相等。

运行 `python3 code/transformer_block.py` 的**真实输出**（代码修复完成后重新运行并回填,D10-D15 是代码审核阶段新增的针对性回归测试,分别对应构造器参数校验、KV cache 一致性、cross-attention 组合限制、mask 安全性与广播、默认 causal 行为、RoPE cache 的 device/dtype 处理）：

```text
[D01] shapes (enc/dec2017/gpt2/modern) = [torch.Size([2, 7, 96]), torch.Size([2, 7, 96]), torch.Size([2, 7, 96]), torch.Size([2, 7, 96])] (expect 4x (2,7,96)); cross-attn ran with Tq=7 != S=5; changing memory changes cross-attn output (memory is actually read, not ignored) = True  PASS
[D02] formula-level: Pre-LN(F=0)==x: True; Post-LN(F=0)==N(x): True (and != x: True); class-level: zeroed ModernDecoderBlock==x: True, zeroed Post-LN encoder layer != x: True, == ln2(ln1(x)) exactly: True  PASS
[D03] perturbing tokens after t=3: positions 0..3 unchanged = True, positions 4..5 changed = True; perturbing position 0 changes positions 1..5 (attention actually mixes forward, not a dead branch) = True  PASS
[D04] full-forward vs token-by-token cached decode: max|Δ| = 4.77e-07 (tol 1e-04, numerically equivalent, NOT bit-identical); cache CONTENT vs independently projected+rotated K/V: max|ΔK|=3.58e-07, max|ΔV|=4.77e-07; final cache shape head_dim=2 (expect 2) len=6 (expect 6)  PASS
[D05] GQA(n_kv=n_q) == independent reference MHA: max|Δ|=0.00e+00; GQA(n_kv=2, group=4) == independent repeat_interleave reference: max|Δ|=8.94e-08; MQA == same reference (n_kv=1): max|Δ|=8.94e-08, cache head_dim=1 (expect 1), output shape=(2, 5, 96), per-query-head outputs distinct (not collapsed) = True  PASS
[D06] FFN params: dense(4d)=73728, gated(8d/3)=73728 (both expect 73728); attn params: GQA=23040 (formula 2d^2(1+n_kv/n_q)=23040), MHA=36864 (formula 4d^2=36864)  PASS
[D07] packed QKV vs separate: max|Δ|=4.77e-07; packed gate+up vs separate: max|Δ|=0.00e+00  PASS
[D08] sequential: attn input == norm1(x): True, FFN input == norm2(x+attn_out) (updated stream): True (and != norm2(x) alone: True); parallel: attn input == norm(x): True, FFN input == norm(x): True, attn and FFN inputs are the literal SAME tensor: True  PASS
[D09] dense block: all grads finite = True, all non-zero = True; MoE tokens_per_expert=[1, 0, 2, 0] (<4 tokens, so >=1 expert unused by pigeonhole); touched experts non-zero+finite grad, untouched experts no/zero grad: True; router gate non-zero+finite grad: True; rest-of-block (attn+norms) non-zero+finite grad: True  PASS
[D10] d_model % n_q_heads != 0 raises ValueError: True; n_q_heads % n_kv_heads != 0 raises ValueError: True; RoPE + odd head_dim raises ValueError: True; MoEFFN(top_k=2) raises NotImplementedError (no silent top-1 downgrade): True  PASS
[D11] inconsistent start_pos vs non-empty cache raises ValueError: True; consistent start_pos does not raise: True; cache.reset() clears state: True; append() with mismatched batch/shape raises ValueError: True  PASS
[D12] cross-attention + RoPE raises ValueError: True; cross-attention + KV cache raises ValueError: True; plain cross-attention (no RoPE/cache) still works, shape=(1, 4, 32): True  PASS
[D13] fully-masked query row produces finite output (no NaN): True; key_padding_mask correctly isolated per-batch-sample when B==n_kv_heads (2): max|Δ| vs per-sample reference=1.79e-07 -> True; same check when B==Tq (3): max|Δ|=1.64e-07 -> True  PASS
[D14] GPT2StyleDecoderLayer: omitted causal_mask == explicit causal_mask: True, default is genuinely causal (no future->past leakage): True; ModernDecoderBlock: omitted causal_mask == explicit causal_mask: True  PASS
[D15] build_rope_cache(dtype=float64) produces float64 cos/sin: True; apply_rope auto-casts a mismatched (float32) cache to a float64 input's dtype (no crash, correct output dtype): True; matches a natively-float64-built cache: True (max|Δ|=0.00e+00)  PASS

all transformer block sanity checks passed ✓
```

## §8 Model Recipe Atlas

### 8.1　主流模型 Block Recipe 对照表

口径说明：下表"QK-Norm"专指 attention 点积前对每个 head 的 Q/K 显式做归一化，不把 DeepSeek MLA 内部的低秩 latent 投影归一化算作 QK-Norm；"branch pre+post"指 $y=x+N_\text{post}(F(N_\text{pre}(x)))$，不等同于原始 Post-LN；下表只描述 MHA/MQA/GQA 的 KV-head 共享方式，local/sliding/global attention pattern 单独写在备注列；final norm、local/global 排布属于 stack 调度层（参见 §0 分层地图），本表按模型整体 recipe 呈现，不代表它们是某个单一 block 的私有字段；这些代表性模型的 canonical recipe 中**没有一个以 ALiBi 为主要位置方案**。

| 模型 | Norm 拓扑 | Norm 类型 | 位置方案 | Attention | FFN/experts | Parallel attn+FFN | QK-Norm | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | Pre-LN | RMSNorm | RoPE | 7B/13B 为 MHA；70B 为 GQA | SwiGLU，dense | 否 | 否 | 不应把整个 Llama-2 家族统一写成 GQA |
| Llama-3 | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU，dense | 否 | 否 | "Llama-3-like"代码还需匹配具体 checkpoint 配置才算精确复现 |
| Qwen2/2.5 | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU；dense 或 MoE 取决于 checkpoint | 否 | 通常否 | 限定 text model；不能把整个 Qwen 家族统一写成 dense 或统一写成 MoE |
| Mistral | Pre-LN | RMSNorm | RoPE | GQA | SwiGLU，dense | 否 | 否 | 指原始 Mistral dense recipe；还采用 sliding-window attention；Mixtral 是另一个 MoE recipe |
| DeepSeek-V2 | Pre-LN | RMSNorm | decoupled RoPE within MLA | MLA | SwiGLU-style；第一层 dense，之后主要 DeepSeekMoE | 否 | 否 | MLA 内有 latent projection normalization，但不等于 headwise QK-Norm |
| DeepSeek-V3 | Pre-LN | RMSNorm | decoupled RoPE within MLA | MLA | SwiGLU-style；前 3 层 dense，之后主要 DeepSeekMoE | 否 | 否 | block 主干沿用 MLA+MoE；训练/负载均衡改进不应误写成新的 attention 类型 |
| GPT-J | Parallel Pre-LN | LayerNorm | partial RoPE | MHA | GELU，dense | 是 | 否 | 核心形式 $x+A(N(x))+M(N(x))$；GELU 不属于 ReLU/GeGLU/SwiGLU 三分类 |
| PaLM | Parallel Pre-LN | LayerNorm，bias-free recipe | RoPE | MQA | SwiGLU，dense | 是 | 否 | 并行 attn/MLP 使用共同的归一化 residual 输入 |
| Gemma-1 | Pre-LN | RMSNorm | RoPE | 2B 为 MQA，7B 为 MHA | GeGLU，dense | 否 | 否 | 不应把 Gemma-1 全系列统一写成 GQA |
| Gemma-2 | branch pre+post | RMSNorm | RoPE | GQA | GeGLU，dense | 否 | 否 | 层间 1:1 交替 local/global attention，用 attention-logit 与 final-logit soft-capping；post norm 位于 branch 内、residual add 前 |
| Gemma-3 | branch pre+post | RMSNorm | RoPE | 1B 为 MQA（单个 KV head、多个 query head）；4B/12B/27B 为 GQA（约 85% 置信度的独立判断，最终仍建议以具体 config 的 `num_key_value_heads` 为准） | GeGLU，dense | 否 | 是 | local/global attention pattern，约 5:1 周期（多个 local 层后插入一个 global 层，不是严格 1:1 交替）；QK-Norm 是与 Gemma-2 的重要区别之一 |
| GPT-2 | Pre-LN | LayerNorm | learned absolute positional embedding | MHA | GELU，dense | 否 | 否 | 每层串行 attn→MLP，stack 末尾有 final LayerNorm |
| GPT-3 | Pre-LN | LayerNorm | learned absolute positional embedding | MHA | GELU，dense | 否 | 否 | 延续 GPT-2-style block，层间有 dense 与局部带状稀疏 attention pattern 的变化 |

**从这张表能读出的三个规律**（对照 §0 的分层地图）：(1) "Norm 拓扑"和"Norm 类型"两列几乎总是绑定出现——Pre-LN 配 RMSNorm、Parallel Pre-LN 配 LayerNorm、branch pre+post 也配 RMSNorm，说明这两个维度虽然分析时可以分开讨论，实践中的组合空间比理论空间窄得多；(2) "Attention"列里 MHA/MQA/GQA/MLA 四种取值在同一批代表性模型里都能找到例子，佐证了 §4 "这是并存的设计点，不是历史阶梯"的判断；(3) "Parallel attn+FFN"这一列只有 GPT-J、PaLM 两行是"是"，其余全部模型都用串行——parallel 是小众但确实在用的变体，不是历史被淘汰的死路。

### 8.2　十二条最容易翻车的说法

把全文出现过的高频误答收进一张表，考前最后扫一遍：

| 常见说法 | 更准确的说法 |
| --- | --- |
| 现代 LLM 才把 Post-LN 换成 Pre-LN | GPT-2 已经是 Pre-LN，这一步发生得更早（§2.2） |
| decoder-only 就是"decoder 去掉 cross-attn" | 单层结构上这么说没错；完整模型层面还需要加上——encoder/memory 路径也一并消失，双流条件生成变成单流因果建模（§2.1） |
| 残差流方差随深度 $\propto\sqrt l$ 膨胀 | 在简化的独立增量假设下，方差约按 $O(l)$ 线性增长，标准差/RMS 才约按 $\sqrt l$ 增长；真实增长率还取决于初始化和分支相关性（§0、§2.2） |
| Pre-LN 栈末尾必须有 final norm，否则模型不可能训好 | final norm 是 GPT-2/Llama 等标准 recipe 的常见配置，不收口通常明显拖累训练质量，但不是所有 Pre-LN 架构数学上不可或缺的条件；它也是 stack 级操作，不是单个 block 的字段（§2.2、§7.1） |
| MHA→MQA→GQA→MLA 是一条演进链 | 按时间顺序确实是 MHA→MQA→GQA（MQA 早于 GQA），但这只是其中一条排序维度；MLA 是独立的低秩 latent 分支，不接在同一条压缩比数轴末端——"不是一条链"的原因是排序维度不止一条，不是"MQA 早于 GQA"这个史实本身有问题（§4、§9） |
| MLA 只是 GQA 更狠的压缩 | MLA 压缩的是 hidden 维度并需要解耦 RoPE；GQA 压缩的是 head 数量/共享关系，两者约束条件都不同（§4.2、§9） |
| SwiGLU hidden size 必须等于 $8d/3$ | 这是参数量匹配算出来的常见默认，真实模型按硬件友好倍数取整会明显偏离（§3、§7.3） |
| Dense FFN 迟早都要换成 MoE | 这是独立的容量轴，不是必经阶段，Llama 1–3、原始 Mistral、Gemma 1–3 等代表性模型至今仍用 dense，但 Mixtral、Llama 4 等已采用 MoE（§5） |
| "sandwich norm"和原始 Post-LN 是一回事 | 更准确写法是 branch pre+post；第二次 norm 只作用在分支输出上，不作用在残差和上（§6.1） |
| Pre-LN 完全不需要 warmup | Pre-LN 只是去掉了 Post-LN 那种"梯度不均衡"导致的 warmup 刚需；现代 Pre-LN 大模型仍常配 warmup（另有 Adam 相关动机，见 normalization_init_tutorial.md §5.2） |
| MQA/GQA 减少了 Q projection 的计算量 | 只减少 K/V 投影的参数/计算和 KV cache 显存/带宽，Q 侧头数和计算量不变（§4.1） |
| parallel residual 要求 attn/FFN 必须共享同一个 norm 参数 | parallel 的定义是"没有先后依赖"，用两个独立 norm 模块、只要都读原始 $x$ 也满足（§6.1、§7.4 [D08]） |

## §9 30 高频面试题

按难度分三档，点开看答案要点 + 易踩坑。L2/L3 是顶级 lab 深水区（拓扑分叉、装配轴独立性、系统正确性、跨模型 recipe 比较）。答题时**不重复 sibling tutorial 里的深度数学推导**，按"结构判断 → 关键公式 → 常见错误"组织即可。

### L1必会题

<details>

<summary>Q1. Post-LN、Pre-LN 串行、Pre-LN 并行、branch pre+post 四种拓扑的完整 forward 分别怎么写？</summary>

- Post-LN：$u=N_1(x+A(x))$，$y=N_2(u+M(u))$——归一化套在每次残差相加之后
- Pre-LN 串行：$u=x+A(N_1(x))$，$y=u+M(N_2(u))$——残差主干是纯恒等加法，FFN 读的是 $u$（attention 更新后的流）
- Pre-LN 并行（GPT-J/PaLM）：$y=x+A(N_A(x))+M(N_M(x))$——两个子层读同一份 $x$，没有先后依赖
- branch pre+post（Gemma-2/3）：$u=x+N_1^{\text{post}}(A(N_1^{\text{pre}}(x)))$，$y=u+N_2^{\text{post}}(M(N_2^{\text{pre}}(u)))$——分支前后各一次 norm，残差主干仍是纯加法
- Pre-LN 栈末尾通常配一次 **stack 级** final norm，这不是上面任何一个 block 公式本身的一部分（§0、§7.1）

只背出三种，漏掉 branch pre+post；或者把 final norm 当成某个 block 公式里的一项。

</details>

<details>

<summary>Q2. 原始 Transformer decoder 与 decoder-only LLM block 的根本区别是什么？</summary>

- 单层结构层面：确实是"去掉 cross-attn"（3 子层→2 子层），这部分说法没问题
- 完整模型层面：还需要加上——encoder/memory 路径一并消失，双序列条件生成变成单序列因果建模，这是信息流拓扑和序列因子化方式的变化
- 纯 decoder-only block 才没有 cross-attn；部分多模态/检索增强 LLM 会在 backbone 上重新引入 cross-attention（§2.1）

只说"去掉了 cross-attn"就停在单层，没补完整模型层面的信息流变化；或者反过来说这完全是不同的问题设定，掩盖了单层结构确实是去掉一个组件这一事实。

</details>

<details>

<summary>Q3. GPT-2-style block 到 Llama-style block 到底改了什么？</summary>

- 两者**都已经是 Pre-LN**（GPT-2 不是 Post-LN）
- LayerNorm→RMSNorm；learned absolute PE（堆栈外）→RoPE（attention 内部，只作用 Q/K，最初由 Su et al. 2021 的 RoFormer 提出，LLaMA 是采用者）；GELU dense FFN→SwiGLU gated FFN；MHA→按具体规格 MHA/GQA；去掉大部分 bias/dropout
- final norm 两边都有，只是类型不同（LayerNorm vs RMSNorm）（§2.2）

先说"现代 LLM 才把 Post-LN 换成 Pre-LN"——GPT-2 已经是 Pre-LN，这个误答会被立刻追问扣分。

</details>

<details>

<summary>Q4. 为什么 attention 要除以 $\sqrt{d_k}$？</summary>

- 假设 Q、K 各分量独立、零均值、单位方差，则点积 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 是 $d_k$ 个独立乘积项之和，方差可加，约为 $d_k$
- 不缩放时点积幅值随 $d_k$ 增长，送进 softmax 前某些 logit 会远大于其余 logit，softmax 趋于 one-hot、对应位置的梯度接近饱和
- 除以 $\sqrt{d_k}$ 把点积方差重新拉回 $O(1)$ 量级，是"控制 softmax 输入尺度"的具体手段，不是任意选的常数

只背"除以 $\sqrt{d_k}$ 防止梯度消失/饱和"这个结论，说不出独立分量点积方差 $\approx d_k$ 这一步推导。

</details>

<details>

<summary>Q5. LayerNorm、RMSNorm、BatchNorm 有什么区别？为什么 Transformer 通常不用 BatchNorm？</summary>

- LayerNorm：对每个样本、每个位置的特征维做归一化（减均值、除以标准差），逐样本独立，不跨 batch
- RMSNorm：只做 re-scale（除以均方根），不做 mean-centering，省了减均值这一步
- BatchNorm：对同一特征维跨 batch 做归一化，统计量（running mean/var）依赖 batch 内其他样本
- Transformer 的输入是变长序列，batch 内不同位置/样本的统计量不稳定；推理时又常是变长、单样本、逐 token 自回归解码，BatchNorm 的 running statistics 在这种场景下和训练分布不匹配；LayerNorm/RMSNorm 逐样本独立计算，天然适配变长序列和自回归解码

只说"BatchNorm 在 NLP 效果不好"，说不出 batch 统计量依赖、以及变长序列/自回归解码不匹配这一具体原因。

</details>

<details>

<summary>Q6. 完整 attention 的 tensor shape 如何变化？为什么需要多头和 $W_O$？</summary>

- 输入 $x\in[B,T,d]$ → Q/K/V 投影后 reshape 成 $[B,n_\text{head},T,d_h]$（$d=n_\text{head}\cdot d_h$）
- score $=QK^\top/\sqrt{d_h}\in[B,n_\text{head},T,T]$ → 加 mask → softmax（沿最后一维）→ 与 V 相乘得到 $[B,n_\text{head},T,d_h]$
- 多头 concat 回 $[B,T,d]$，再过输出投影 $W_O\in\mathbb{R}^{d\times d}$ 得到最终输出 $[B,T,d]$
- 多头让模型在不同子空间并行学习不同的 attention pattern；$W_O$ 把多个 head 的输出重新混合投影——没有 $W_O$，各 head 的输出只是简单拼接，无法让不同 head 的信息相互组合

漏掉 reshape/concat 这两步；答不出 $W_O$ 的作用是"混合多头信息"而不只是"维度对齐"。

</details>

<details>

<summary>Q7. self-attention 和 cross-attention 的关键区别是什么？</summary>

- self-attention：Q、K、V 来自同一条序列
- cross-attention：Q 来自当前流（长度 $T$），K、V 来自另一条序列/memory（长度 $S$），$T$ 与 $S$ 一般不相等
- 原始 Transformer decoder 用 cross-attn 让 target 序列条件于 source 编码；纯 decoder-only LLM block 结构上**没有**cross-attn（§1、§2.1；完整 mask/softmax 公式见 attention_tutorial.md §4.1）

以为 cross-attention 只是"self-attention 加个 mask"；答不出 Q 长度和 K/V 长度可以不同这一点。

</details>

<details>

<summary>Q8. local/sliding window attention 是 attention 的哪一种设计轴？</summary>

- 是 attention **连接结构/可见范围（connectivity pattern）**的设计轴，不是"attention backend"（FlashAttention 才是执行 backend），也不是位置编码方案
- Mistral 用固定滑窗；Gemma-2 层间 1:1 交替 local/global；Gemma-3 更接近多个 local 层后插入一个 global 层的周期性 pattern（约 5:1，不是严格 1:1）
- 大体可以和任意位置编码方案组合，但外推长度、窗口边界、具体 kernel 实现的兼容性仍需要单独验证，不是无条件"即插即用"（§6.3）

把它叫成"attention backend"，或者和 RoPE/ALiBi 这类位置编码方案混为一谈；以为"大体正交"等于"随便换都不用验证"。

</details>

<details>

<summary>Q9. weight tying、embedding scaling、LM head 属于 block 内部还是外围 recipe？</summary>

- 都属于"模型外壳"层，**不属于任何一个 Transformer block**——token embedding 和 LM head 是整个模型的输入输出层，不是某一层 block 的组成部分
- Weight tying：输入 embedding 与 LM head 共享同一个参数矩阵；embedding scaling：token embedding 乘一个和 $\sqrt{d}$ 相关的常数；LM head soft-cap：某些模型在最终 logits 上也做类似 §6.2 的有界变换（§3）
- 面试被问"block 长什么样"时把它们混进 block 定义，容易让考官怀疑对"层 vs 模型"边界的理解

把 embedding/LM head 当作某个 block 的一部分来描述，混淆了"一层 block"和"整个 LM"的边界。

</details>

<details>

<summary>Q10. SwiGLU 的完整公式和 FFN 的作用是什么？</summary>

- FFN 是 **position-wise** 的：对序列每个位置的向量独立做同一个非线性变换，不跨位置混合信息（跨位置信息混合是 attention 的工作）
- SwiGLU 公式：$\text{down}(\text{SiLU}(\text{gate}(x))\odot\text{up}(x))$——gate、up 是两个独立的线性投影到中间维度 $h$，逐元素相乘后再由 down 投影回 $d$
- gate 分支的 SiLU 起门控作用，决定 up 分支每个维度"通过多少"，这是和 Dense FFN（两矩阵、单个激活函数）的关键区别
- FFN 在整个 block 里承担"逐位置特征变换/信息存储"的角色，和 attention 的"跨位置信息路由"角色互补

只记得"三矩阵"这个数量，说不出 gate 分支的门控语义和 FFN 逐位置这一性质。

</details>

### L2进阶题

<details>

<summary>Q11. 为什么 Pre-LN 更容易训练？为什么栈末尾通常要有 final norm？</summary>

- 残差主干是纯恒等加法，存在一条不经过子层及归一化 Jacobian 的恒等梯度路径，通常改善深层网络的优化条件——但这不保证总梯度完全不衰减或不发生抵消，"梯度无损流到底"是过强的说法
- 代价：在简化的独立增量假设下，残差流**方差**约随深度**线性**增长（$O(l)$），标准差/RMS/典型幅值才约按 $\sqrt l$ 增长；真实增长率还取决于初始化、残差缩放和分支相关性
- 不收口直接接输出头，输出分布通常会明显偏离训练时的尺度，训练质量常见地明显下降；GPT-2/Llama 等标准 recipe 因此配置 final norm，但这不是所有 Pre-LN 架构数学上不可或缺的条件——通过残差缩放/特殊初始化/归一化输出头也能构造没有传统 final norm 的变体
- **不保证最终质量永远优于调好的 Post-LN**——Post-LN 训稳时深度贡献不被稀释，上限有时更高

把方差增长写成 $\propto\sqrt l$（应该是标准差/RMS 才是 $\sqrt l$，方差本身按 $O(l)$ 线性增长）；把 final norm 说成"数学上必须"。

</details>

<details>

<summary>Q12. RoPE 应该插在哪，KV cache 里存什么？</summary>

- RoPE 插在 attention 内部，只作用于 Q、K，**V 不旋转**
- 标准实现通常缓存**已旋转**的 K；若某实现选择缓存未旋转的 K，则必须额外保留其 position 并在使用时按正确 position 旋转，不能把所有历史 K 当成当前 position 处理——这样做数学上同样正确，只是更慢，因此不是常见选择
- MLA 缓存的甚至不是普通完整 K，而是 latent + 解耦的共享 RoPE key 分量（§4.2）
- GQA/MQA 下 cache 只存 $n_\text{kv}$ 份，不应把 K/V 先物理复制成 $n_q$ 份再存（§4，完整旋转矩阵推导见 long_context_rope_yarn_mla_tutorial.md §2）

把"缓存已旋转 K"说成数学上唯一正确的实现；不知道 MLA 的 cache 内容和普通 K/V 完全不同。

</details>

<details>

<summary>Q13. MHA、MQA、GQA 的参数量和 KV cache 怎么变化？</summary>

- 公式成立需要明确写出假设：$d=n_qd_h$，Q/K/V 使用相同 head dim，输出投影是 $d\times d$，忽略 bias
- Attention 参数：MHA 为 $4d^2$；GQA 为 $2d^2(1+n_\text{kv}/n_q)$；$n_\text{kv}=n_q$ 退化回 MHA 公式（§7.3 [D06]）
- KV cache：MHA 为 $2n_q d_h$/token/layer；GQA 为 $2n_\text{kv}d_h$；MQA（$n_\text{kv}=1$）为 $2d_h$——这是**元素数**，不是字节数，完整显存还要乘 $B\times L\times N_\text{layer}\times$ 每元素字节数
- GQA 确实减少 K/V 投影的参数和计算量，但不减少 Q projection 的计算量——不是"只省显存/带宽"这么简单

不说明公式假设就直接套用；把 KV cache 元素数公式当成实际字节数；以为 GQA 完全不影响任何计算量。

</details>

<details>

<summary>Q14. 一个 dense Transformer block 的参数量和 FLOPs 怎么算？</summary>

- 参数量：attention 约 $4d^2$（或 GQA 的 $2d^2(1+n_\text{kv}/n_q)$）+ FFN 约 $8d^2$（dense）或 $3dh$（gated）+ 两个 norm 的逐通道参数（$O(d)$，量级远小于矩阵参数）——单层通常 FFN 参数占大头（§7.1 的具体例子里 FFN 占约 76%）
- FLOPs：attention 的 QKVO 投影是 $O(Td^2)$，score 矩阵计算 + 加权求和是 $O(T^2d)$（$T$ 为序列长度），FFN 是 $O(Tdh)\sim O(Td^2)$
- 序列长度 $T$ 较小时投影主导（$O(Td^2)$ 项），$T$ 很大时 attention 的 $O(T^2d)$ 二次项逐渐主导——这也是长上下文场景 attention 计算成为瓶颈的直接原因

只记住参数量公式，说不出 FLOPs 里 $O(Td^2)$ 和 $O(T^2d)$ 两项谁在什么序列长度下占主导。

</details>

<details>

<summary>Q15. 为什么 SwiGLU 的 hidden size 常常接近 $8d/3$？</summary>

- 忽略 bias 时，dense FFN（两矩阵）参数为 $8d^2$（$4d$ expansion）；gated FFN（gate/up/down 三矩阵）参数为 $3dh$
- 令两者相等：$3dh=8d^2 \Rightarrow h=8d/3$——这是**参数量匹配算出来的常见默认**，不是"gated FFN hidden size 必须等于 $8d/3$"的架构定律
- 真实 Llama 系实现通常从 $4d$ 出发乘 $2/3$，再按硬件友好倍数（及可选 multiplier）取整，intermediate size 可能明显偏离 $8d/3$（§3、§7.3 [D06] 用 $d=96$ 验证两边都是 73,728 参数）

把 $8d/3$ 说成"所有 Llama 模型的 hidden size 严格等于这个值"；答不出这是参数匹配推出来的默认，而非架构硬约束。

</details>

<details>

<summary>Q16. 串行 attn→FFN 和 parallel attn+FFN 是否等价？</summary>

- 不等价：串行 Pre-LN 里 FFN 读的是 attention **更新后**的残差流（$\text{FFN}(N_2(x+\text{Attn}(N_1(x))))$）；parallel 里 FFN 和 attention 读**同一份**归一化原始输入（$x$），互不依赖
- 判定方法不应是"两个输出是否恰好不相等"（脆弱），而应重构出 FFN 真实吃到的输入张量做精确匹配（§7.4 [D08] 用 forward hook 抓输入 + 独立重算对比）
- 并行结构去掉了这条依赖链，训练时可以减少部分同步开销（§6.1）

只回答"应该不一样吧"却给不出可执行的判定方法；或者用"输出不相等"这种弱检验冒充结构证明。

</details>

<details>

<summary>Q17. attention scale、QK-Norm、logit soft-cap 分别解决什么问题？</summary>

- attention scale：标准的 $1/\sqrt{d_h}$（或可学温度）缩放点积，控制 softmax 输入的整体幅度量级，是最基础的数值控制
- QK-Norm：点积**之前**对每个 head 的 Q、K 分别做 L2 归一化，并**用一个可学习的缩放/温度参数取代**标准固定的 $1/\sqrt{d_h}$ 缩放——这是 Henry et al. 原始定义的完整版本，后续模型的具体实现细节可能不同
- logit soft-cap：点积**之后**对 score 或最终 vocabulary logits 施加 $c\tanh(z/c)$ 类有界变换，管住已经算出来的分数/logits
- 三者作用位置不同，**不能无脑叠加而不重新检查尺度**——Gemma-2 用 soft-capping，Gemma-3 换成 QK-Norm 并去掉 soft-cap，不是简单叠加（§6.2，完整 attention entropy collapse 论证见 normalization_init_tutorial.md §6.3）

把 QK-Norm 说成"只是归一化 Q/K"，漏掉它同时替换了固定 scale 这一关键组件；把 QK-Norm 和 soft-cap 当成同一件事的两种叫法。

</details>

<details>

<summary>Q18. packed QKV projection 和分开的三个 Linear 有什么区别？</summary>

- 数学上等价：把三个独立权重矩阵按行拼接成一个大矩阵，输出再按原维度切开，拷贝等价权重后结果应在对应 dtype 的合理容差内一致（不是"逐位对齐"这种更强的说法）（§7.4 [D07]）
- 区别在 kernel 层面：packed 版本通常是一次大矩阵乘法，减少 kernel 启动开销；但 packed **不一定总更快**，实际收益取决于矩阵尺寸、量化格式、并行/切分布局
- SwiGLU 的 gate/up 投影同理可以打包，这是"数学相同、实现不同"的经典工程面试点

把数值等价说成"逐位对齐"；以为 packed 总是更快，不考虑具体尺寸/量化/并行布局的影响。

</details>

<details>

<summary>Q19. MLA 和 GQA 本质区别是什么？</summary>

- GQA 沿 **KV-head 数量/共享关系**这一轴压缩：多个 Q head 共享一组 K/V head，压缩比由 $n_\text{kv}/n_q$ 决定
- MLA 通过低维 latent bottleneck 压缩 KV **内容表示**：把每 token 的 K/V 压进一个 $d_c \ll n_qd_h$ 的 latent 向量——注意 MLA cache 通常不仅有 latent，还包括解耦后的共享 RoPE key，"整个 K/V 都只投影进一个 latent"不够精确
- 若使用 RoPE 且希望保留 absorbing trick，MLA 必须解耦 RoPE 成一份独立的共享分量；GQA 不需要处理这个问题（§4；完整推导见 long_context_rope_yarn_mla_tutorial.md §9）

把 GQA 的压缩说成"缩小 $d_h$"（实际是缩小 KV head 数量，不改变 head dim 本身）；以为 MLA 的 cache 只有 latent 一项。

</details>

<details>

<summary>Q20. prefill 和 decode 的性能瓶颈为什么不同？KV cache 如何改变复杂度？</summary>

- 这是 inference serving 层面的问题，不是 block 装配层面的问题——prefill 阶段整段 prompt 一次算完，权重被多次复用，通常算术强度更高，容易跑满算力；decode 阶段每步只算 1 个新 token，却要重新读整层权重和随历史增长的 KV，小 batch 下往往落在带宽瓶颈区间
- KV cache 把"重新计算全部历史 K/V"变成"只算新 token 的 K/V + 复用缓存"，把 decode 阶段每步的 attention 复杂度从 $O(t^2)$ 降到 $O(t)$（$t$ 为当前位置）
- 完整的 prefill/decode 瓶颈分析、批处理调度、显存管理见 llm_inference_serving_tutorial.md，本文不重复展开（§7.2）

把这当成 block 内部的设计问题来回答，忽略这本质上是系统/调度层面的问题；不知道该去哪本教程查完整推导。

</details>

<details>

<summary>Q21. FlashAttention、PagedAttention、GQA/MQA 分别优化哪一层？</summary>

- FlashAttention：IO-aware 的 exact attention kernel，优化的是"怎么算"——用 tiling/recompute 减少 HBM 访问，不改变 attention 的数学结果，也不改变模型架构（§6.4）
- PagedAttention：KV cache 的显存管理机制，优化的是"怎么存"——用分页表把逻辑 KV 映射到物理显存的不连续 block，降低碎片，属于 serving 系统层面
- GQA/MQA：模型架构层面的选择，优化的是"存多少"——减少 KV head 数量从而减少 cache 本身的大小
- 三者是不同层级的优化，不能混为一谈：前两者是执行/系统层面的工程手段，后者是训练前就要定下来的架构决策；完整机制见 attention_tutorial.md（FlashAttention）与 llm_inference_serving_tutorial.md（PagedAttention）

把三者混成"都是省显存的技术"，不区分它们分别作用在算法执行、显存管理、模型架构哪一层。

</details>

<details>

<summary>Q22. bias 和 dropout 为什么在现代大模型 pretrain 里逐渐被去掉？</summary>

- bias：大规模训练里去掉 bias 通常不掉点，还能省一点参数和计算，成为 Llama 系起的常见默认
- dropout：大规模单 epoch 预训练下，数据量本身提供的正则化效果已经足够，重度 dropout 反而可能拖慢收敛；这是工程经验默认，不是数学上被证明"必须去掉"
- GPT-2-style 仍保留两者，Llama-style 起大幅精简——这是 §2.2 桥接表里明确列出的变化项之一

答成"理论证明去掉更好"；实际上这是大规模预训练场景下的经验默认，不是被严格证明的普适结论。

</details>

<details>

<summary>Q23. "sandwich norm"这个叫法哪里不准确？该怎么写公式？</summary>

- "sandwich norm"是有歧义的口语称呼，更准确的说法是"每个子层分支前后各一次归一化"（branch pre+post normalization）
- 准确公式：$y=x+N_\text{post}(F(N_\text{pre}(x)))$——第二次归一化只作用在**分支输出**上，不作用在残差和上
- 这和原始 Post-LN 的 $N(x+F(x))$ 是两个不同的公式：Post-LN 归一化整条相加后的流，branch pre+post 只收口分支输出、残差主干仍是纯加法（§6.1；Gemma-2/3 采用此拓扑，完整变体谱系见 normalization_init_tutorial.md §6.2）

把 branch pre+post 的公式写成和原始 Post-LN 一样的 $N(x+F(x))$，没意识到"归一化谁"这个关键差异。

</details>

<details>

<summary>Q24. MLA 和 GQA 的 KV cache contract 有什么不同？</summary>

- GQA 的 cache 是 $n_\text{kv}$ 份完整 K/V（每份维度 $d_h$），本质上还是"多份普通 K/V"，只是份数比 $n_q$ 少
- MLA 的 cache 是一份共享 latent $c_t^{KV}$（维度 $d_c$）加一份共享的解耦 RoPE key（维度 $d_h^R$），不是"更少份的普通 K/V"，K、V 也不再各自独立存储——完整 absorbing trick、为什么 RoPE 不能直接吸收、解耦方案的代数推导见 long_context_rope_yarn_mla_tutorial.md §9，本文不重复

把 MLA 的 cache 想象成"$n_\text{kv}=1$ 的 GQA"，忽略它存的是低秩 latent + 解耦 RoPE key，而不是普通 K/V。

</details>

<details>

<summary>Q25. 手写一个支持 KV cache 的增量解码 causal self-attention，最容易漏掉哪些正确性坑？</summary>

- cache 里的 K 必须是标准实现里已经旋转过 RoPE 的（若选择缓存未旋转 K 则需要额外保留 position，见 Q12）；新 token 的 Q、K 要用正确的绝对 position offset
- GQA/MQA 下 cache 的 head 维度必须始终是 $n_\text{kv}$，不能在某次调用里把广播后的张量写回 cache
- 单 token 的 causal mask 在"密集 causal attention + 动态 cache + 无 padding/packing"这个常见场景下确实是"看 0..t 全部"（恒为 True），但静态 cache、left padding、sliding window、packed sequence 等场景仍可能需要额外 mask，不能一概而论
- full-forward 与 cached decoding 数值一致是最重要的回归测试之一，能发现 causal mask/RoPE offset/cache 顺序等常见 bug，但这个测试很强、不代表充分——它不能替代对 mask 广播、cache 一致性校验等场景的单独测试（§7.4 [D04]）

认为"单 token mask 恒为 True"在所有场景下都成立；把 full-forward vs cached-decoding 一致性当成唯一且充分的正确性证明。

</details>

<details>

<summary>Q26. GQA 的 $n_q \bmod n_\text{kv} = 0$ 约束是从哪里来的？MLA 里这个约束还成立吗？</summary>

- GQA 约束来自标准均匀 GQA 的定义和常用 kernel 布局：$n_q$ 个 Q head 均匀分成 $n_\text{kv}$ 组，每组 $n_q/n_\text{kv}$ 个，这要求 $n_q\bmod n_\text{kv}=0$
- 但这不是数学上不可绕过的定律——可以定义非均匀的 Q-head 到 KV-head 映射，只是多数模型配置和高效 kernel 的实现假设了等大小分组
- MLA 通常没有 $n_\text{kv}$ 这个分组变量（它是低秩 latent 压缩，不是 head 分组），这条整除约束**不适用于 MLA**——这也是"MLA 不是 GQA 的进一步压缩"这一判断的另一个直接证据（§4）

把这条约束当成所有 KV 压缩方法通用的数学定律；不知道非均匀 head sharing 原则上是可以定义的，只是工程上不常见。

</details>

<details>

<summary>Q27. 为什么"MHA→MQA→GQA→MLA 是一条演进链"是错误的说法？该怎么正确组织这四者的关系？</summary>

- 按历史时间，MHA→MQA→GQA 确实是一条没问题的时间线（MQA 早于 GQA 出现，这个史实本身完全正确，不能拿它来反驳这条时间线）
- 但按 KV-head 压缩强度看，正确顺序其实是 MHA→GQA→MQA——压缩程度不是时间顺序：MHA 是 $n_\text{kv}=n_q$ 的一端，MQA 是 $n_\text{kv}=1$ 的另一端，GQA 是中间任意取值，MHA 和 MQA 可以看成同一条均匀 GQA 压缩比数轴的两个端点
- MLA 是独立的低秩 latent cache 路线，压缩的是 hidden 维度而不是 head 数量，不应该接在"KV-head 数量"这条轴的末端
- "四者不是一条线"的真正原因是**存在多条不同的排序维度**（时间顺序 vs 压缩强度 vs 压缩机制），不是只有时间线一条；"MQA 早于 GQA"这个史实本身完全正确，不能被当成反驳"这是一条链"的论据——真正要反驳的是"把四者按某个单一维度强行排成一条线"这件事本身（§4）

**这是本次修复的重点，原稿曾出现真实的论证方向错误**：原稿把"MQA 早于 GQA 出现"当作反驳"MHA→MQA→GQA→MLA 是一条链"的证据，但这条时间线本身完全没错，拿它当反驳会自相矛盾——正确的反驳点是"压缩强度顺序（MHA→GQA→MQA）和时间顺序（MHA→MQA→GQA）不是同一条轴，MLA 又是完全不同的第三条轴"，问题不在"MQA 早于 GQA"这个事实本身。

</details>

### L3高级题

<details>

<summary>Q28. 给一个 Llama-3-like dense GQA block，怎么在不改 block 接口的前提下换成 MoE？block 的 contract 哪里变、哪里不变？</summary>

- 只替换 FFN 槽位：把 `SwiGLU FFN` 换成 `router + N 个 expert FFN` 的组合模块；attention 槽位（GQA、RoPE）、归一化槽位（Pre-RMSNorm）全部不变
- 需要区分三层 contract：①**张量 shape 契约不变**——主激活张量仍是 `[B,T,d] -> [B,T,d]`，residual add 位置不变，`ModernDecoderBlock` 换 `ffn=MoEFFN` 不需要改其他代码；②**训练返回值可能改变**——需要额外返回/传递 router loss 或负载均衡统计量；③**分布式系统拓扑通常改变**——expert parallelism、all-to-all 通信，总参数量与激活参数量需要分开报告
- backward 时只有本 batch 实际被路由到的 expert 应该拿到梯度，不能要求所有 expert 每个 batch 都非零梯度（完整路由/capacity 数学见 moe_tutorial.md §2–§4）
- final norm、local/global 层排布等 stack 级配置和这次替换无关，不需要跟着变

以为换成 MoE 只要"张量 shape 不变就万事大吉"，忽略训练返回值和分布式系统拓扑这两层通常也要跟着变；或者反过来以为需要重新设计 attention/归一化槽位。

</details>

<details>

<summary>Q29. Parallel residual（GPT-J/PaLM）和串行 Pre-LN 在训练/推理上各有什么系统性代价差异？</summary>

- 依赖链：parallel 去掉了"FFN 必须等 attention 更新完残差流"这条依赖，理论上给了更多重叠计算/通信的空间；串行结构里 FFN 严格排在 attention 之后
- 质量：两种拓扑在实践中都能训好，没有证据表明其中一种在同等算力/调参下绝对更优——这是工程取舍，不是单向的优劣关系
- 归一化开销：如果 parallel 版本用单一共享 norm（如 GPT-J），比串行版本少一次归一化计算；用两个独立 norm 的 parallel 实现则和串行版本归一化次数相同，只是没有先后依赖（§6.1）

笼统说"parallel 一定更快"或"parallel 一定质量更差"，给不出具体是哪个环节（依赖链 vs 归一化次数）造成的差异。

</details>

<details>

<summary>Q30. 给定一个新模型的技术报告，你会重点核对 block 里的哪些字段来判断它的 recipe？</summary>

- 归一化拓扑（Post-LN / Pre-LN 串行 / Pre-LN 并行 / branch pre+post）+ 归一化类型（LayerNorm/RMSNorm）
- 位置编码方案及其归属层级（学习绝对位置在堆栈外，还是 RoPE 在 attention 内部；是否有 YaRN/NTK 类扩展）
- Attention 变体：$n_q$、$n_\text{kv}$（MHA/MQA/GQA），是否 QK-Norm，是否 local/sliding window——**不能只用 $n_q,n_\text{kv}$ 判断是否 MLA**，MLA 需要单独核对"KV cache 表示/是否有低秩 latent 结构"这个字段
- FFN 变体：dense 还是 gated 还是 MoE，激活函数是什么，MoE 的话总参数 vs 激活参数分别多少
- bias、dropout 的取舍
- final norm、local/global 层排布属于 **stack 级字段**，应该单独核对，不要和上面的 block 内字段混在一起统计
- 逐项核对能覆盖大多数 block recipe 问题，但不存在一个精确的百分比覆盖率——遇到新模型仍可能有本文未覆盖的字段（如 Q/K/V head dimension 是否相同、attention scale/temperature、mask/pattern 细节、cache layout 等）

只看"用了 GQA 还是 MHA"这一个字段就下结论；只用 $n_q,n_\text{kv}$ 判断 MLA（MLA 的核心特征是低秩 latent 结构，不是 head 数量）；把 final norm/local-global 排布也算进"block 字段"里一起统计；声称某个固定百分比的覆盖率。

</details>

## 📚 参考文献

- **Transformer（原始 encoder-decoder）** — Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762 (2017), NeurIPS 2017.
- **GPT-2（decoder-only，Pre-LN，learned absolute PE）** — Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI 技术报告 (2019)（**无 arXiv id**）.
- **GPT-3** — Brown et al., *Language Models are Few-Shot Learners*, arXiv 2005.14165 (2020), NeurIPS 2020.
- **GPT-J（parallel residual）** — Wang & Komatsuzaki, *GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model*, EleutherAI 开源发布 (2021)（**无正式论文/arXiv id**）.
- **PaLM（parallel residual，MQA）** — Chowdhery et al. (Google), *PaLM: Scaling Language Modeling with Pathways*, arXiv 2204.02311 (2022).
- **RoPE** — Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv 2104.09864 (2021)。LLaMA 系是重要采用者，不是提出者。
- **Llama 1** — Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, arXiv 2302.13971 (2023).
- **Llama 2** — Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv 2307.09288 (2023).
- **Llama 3** — Grattafiori et al. (Meta), *The Llama 3 Herd of Models*, arXiv 2407.21783 (2024).
- **Qwen2** — Yang et al. (Qwen Team), *Qwen2 Technical Report*, arXiv 2407.10671 (2024).
- **Qwen2.5** — Qwen Team, *Qwen2.5 Technical Report*, arXiv 2412.15115 (2024).
- **Mistral 7B（sliding-window attention）** — Jiang et al., *Mistral 7B*, arXiv 2310.06825 (2023).
- **Gemma（1）** — Gemma Team (Google DeepMind), *Gemma: Open Models Based on Gemini Research and Technology*, arXiv 2403.08295 (2024).
- **Gemma 2（branch pre+post norm，soft-capping）** — Gemma Team (Google DeepMind), *Gemma 2: Improving Open Language Models at a Practical Size*, arXiv 2408.00118 (2024).
- **Gemma 3（QK-Norm）** — Gemma Team (Google DeepMind), *Gemma 3 Technical Report*, arXiv 2503.19786 (2025).
- **MQA** — Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv 1911.02150 (2019).
- **GQA** — Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv 2305.13245, EMNLP 2023.
- **MLA / DeepSeek-V2** — DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv 2405.04434 (2024).
- **DeepSeek-V3** — DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437 (2024).
- **DeepSeekMoE（fine-grained + shared experts）** — Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, arXiv 2401.06066 (2024), ACL 2024.
- **Auxiliary-Loss-Free Balance** — Wang, Gao, Zhao, Sun & Dai, *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, arXiv 2408.15664 (2024).
- **Mixtral of Experts** — Jiang et al., *Mixtral of Experts*, arXiv 2401.04088 (2024).
- **Pre-LN vs Post-LN 梯度论证** — Xiong et al., *On Layer Normalization in the Transformer Architecture*, arXiv 2002.04745 (2020), ICML 2020.
- **DeepNorm / DeepNet** — Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*, arXiv 2203.00555 (2022).
- **RMSNorm** — Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv 1910.07467 (2019), NeurIPS 2019.
- **QK-Norm** — Henry et al., *Query-Key Normalization for Transformers*, arXiv 2010.04245 (2020), EMNLP 2020 Findings.
- **GELU** — Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, arXiv 1606.08415 (2016).
- **GLU Variants / SwiGLU** — Shazeer, *GLU Variants Improve Transformer*, arXiv 2002.05202 (2020)。提出并系统评估了这些 Transformer FFN 变体；原始 GLU 与 SiLU/Swish 的基础思想各有更早来源。
