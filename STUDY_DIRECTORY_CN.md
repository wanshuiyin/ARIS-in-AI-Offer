# ARIS AI 秋招学习目录

这份目录不是按文件名罗列，而是按“先打地基，再形成专项能力，最后面试输出”的依赖关系组织。仓库当前的主体是 **33 篇中英文成对的专题教程**，并补充 **4 篇中文项目与面试专项资料**；每篇通常包含 TL;DR、公式与直觉、实现或工程细节、对比与易错点，以及分级面试题（L1 / L2 / L3）。同名 `_en` 文件是英文版，`.review.json` 是审阅记录，不需要重复学习。本目录中的教程链接统一指向 `.html` 渲染版（点开即为网页），同名 `.md` 为 Markdown 源文件。

## 专项资料入口

1. [DouyinVL 项目指南](docs/tutorials/douyinvl_tutorial.html) — 从开源基座到抖音域 VLM，覆盖数据、CPT、后训练、Caption、稳定性排障与面试表达。
2. [鞠少波秋招简历知识地图](docs/tutorials/resume_interview_knowledge_map.html) — 按“是什么、为什么、怎么做、如何验证、局限、个人贡献”系统拆解简历技术点。
3. [最新开源 LLM 技术报告教程](docs/tutorials/open_llm_tech_reports_tutorial.html) — 跟踪近期代表性开源模型的架构、训练与工程要点。
4. [LeetCode HOT 100 一天速记](docs/tutorials/leetcode_hot_100_tutorial.html) — Python 高频算法题型的短时复习清单。

## 0. 怎么使用这套资料

每篇建议用三遍：

1. **建立地图**：先读 `§0 TL;DR` 和对比表，能用 3 句话说清“解决什么问题、核心方法、代价是什么”。
2. **吃透机制**：精读公式、数据流和代码；亲手推一次关键公式，运行对应教学脚本。
3. **面试输出**：先口答 L1，再做 L2；L3 用来准备研究岗追问。答案遵循“结论 → 原理 → 取舍 → 工程例子”。

建议给每篇建立四项学习状态：`未学 / 已读 / 能复述 / 能追问`。真正完成的标准不是“看过”，而是能在白板上解释、比较相邻方法并回答边界条件。

## 1. 主干必修：所有 LLM / 多模态岗位共同基础

按以下顺序学习：

1. [Tokenization / 分词器](docs/tutorials/tokenization_tutorial.html) — BPE、WordPiece、Unigram、byte-level、词表与 fertility。
2. [Attention](docs/tutorials/attention_tutorial.html) — Q/K/V、缩放点积、mask、MHA、FlashAttention、位置编码。
3. [Transformer Block](docs/tutorials/transformer_block_tutorial.html) — 残差拓扑、Pre/Post-LN、MHA/MQA/GQA/MLA、FFN 与 MoE。
4. [归一化 / 残差 / 初始化](docs/tutorials/normalization_init_tutorial.html) — LN/RMSNorm、QK-Norm、Xavier/Kaiming、深层稳定性。
5. [优化器与 LR Schedule](docs/tutorials/optimizer_lr_schedule_tutorial.html) — SGD、Adam/AdamW、warmup、cosine/WSD、梯度裁剪。
6. [LLM Pretraining Pipeline](docs/tutorials/llm_pretraining_pipeline_tutorial.html) — scaling law、数据清洗、packing、loss mask、checkpoint resume。
7. [LLM Evaluation & Benchmarking](docs/tutorials/llm_evaluation_benchmarking_tutorial.html) — pass@k、污染、judge、Bradley-Terry/Elo 与评估可信度。

完成这一阶段后，应能完整叙述：原始文本如何变成 token，token 如何经过 Transformer 训练，训练为何稳定，以及模型如何被可信地评估。

## 2. LLM 训练与对齐方向

前置：完成主干必修，尤其是优化器、预训练和评估。

1. [LoRA / PEFT](docs/tutorials/lora_peft_tutorial.html) — LoRA/QLoRA/DoRA/rsLoRA，参数效率与 merge。
2. [RLHF / DPO / GRPO / PPO](docs/tutorials/rlhf_dpo_grpo_ppo_tutorial.html) — 奖励模型、策略优化、偏好优化及工程取舍。
3. [KL Divergence in RLHF](docs/tutorials/kl_divergence_rlhf_tutorial.html) — KL 估计量、放置位置与梯度偏差；用于补强 RLHF 数学细节。
4. [Reasoning Models](docs/tutorials/reasoning_models_tutorial.html) — o1/R1、test-time compute、PRM、推理训练与验证。
5. [LLM On-Policy Distillation](docs/tutorials/llm_opd_tutorial.html) — MiniLLM、GKD、on-policy 数据分布与蒸馏。

核心面试链路：`SFT → reward/preference data → RM 或直接偏好目标 → policy update → KL/稳定性约束 → offline/online evaluation`。

## 3. LLM 架构、长上下文与训练系统方向

### 3.1 架构与上下文

1. [MoE](docs/tutorials/moe_tutorial.html) — routing、负载均衡、容量、专家并行与稀疏计算。
2. [Long Context](docs/tutorials/long_context_rope_yarn_mla_tutorial.html) — RoPE、NTK、YaRN、MLA、StreamingLLM。
3. [线性 / 稀疏注意力](docs/tutorials/linear_sparse_attention_tutorial.html) — linear attention、Mamba/SSD、DeltaNet、NSA/MoBA。

### 3.2 分布式训练与推理部署

1. [Distributed Training](docs/tutorials/distributed_training_tutorial.html) — DDP/FSDP/ZeRO、TP/PP/EP/SP 及通信代价。
2. [Quantization](docs/tutorials/quantization_tutorial.html) — GPTQ、AWQ、SmoothQuant、FP8/NVFP4 与精度权衡。
3. [KV Cache + Speculative Decoding](docs/tutorials/kv_cache_speculative_decoding_tutorial.html) — KV 显存、MQA/GQA/MLA、Medusa/EAGLE。
4. [LLM Inference & Serving Stack](docs/tutorials/llm_inference_serving_tutorial.html) — 请求状态机、PagedAttention、continuous batching、chunked prefill、PD 分离。

这一方向要能做两类计算：训练显存/通信量估算，以及推理阶段 KV cache、prefill/decode 吞吐和延迟估算。

## 4. 多模态与检索方向

1. 先复习 [Attention](docs/tutorials/attention_tutorial.html) 与主干 Transformer。
2. [VLM / Multimodal](docs/tutorials/vlm_multimodal_tutorial.html) — ViT、CLIP/SigLIP、LLaVA、Q-Former、M-RoPE、视频 VLM 与训练 pipeline。
3. [RAG + Embedding / Retrieval](docs/tutorials/rag_embedding_retrieval_tutorial.html) — 双塔、InfoNCE、难负例、BM25/HNSW/RRF、ColBERT、GraphRAG。
4. 需要生成能力时，再接第 5 节的 VAE、Diffusion 与图像/视频生成系统。

重点不是背模型名，而是能画出视觉 encoder、projector/resampler、LLM、训练阶段与 loss 的数据流，并解释分辨率、视觉 token 数与上下文成本的关系。

## 5. 生成模型方向

### 5.1 数学与表示基础

1. [VAE / VQ-VAE / VQ-GAN / FSQ](docs/tutorials/vae_vqvae_vqgan_tutorial.html) — 连续/离散 latent、ELBO、codebook、tokenizer。
2. [Diffusion Foundations](docs/tutorials/diffusion_foundations_tutorial.html) — DDPM、score、DDIM、EDM、CFG。
3. [Flow Matching](docs/tutorials/flow_matching_tutorial.html) — vector field、概率路径、ODE 与采样。

### 5.2 系统与专项

1. [Image Generation Systems](docs/tutorials/image_generation_systems_tutorial.html) — LDM、SD/SDXL、DiT、SD3/FLUX、ControlNet 与图像编辑。
2. [Video Generation](docs/tutorials/video_generation_tutorial.html) — 时空建模、视频 DiT、Sora/Hunyuan/Kling/Wan/Movie Gen。
3. [3D Generation](docs/tutorials/3d_generation_tutorial.html) — NeRF、Instant-NGP、3DGS、SDS、Trellis。
4. [Diffusion Post-Training](docs/tutorials/diffusion_post_training_tutorial.html) — DDPO/DPOK、DRaFT/AlignProp、Diffusion-DPO、Flow-GRPO。
5. [Diffusion / Flow Distillation](docs/tutorials/diffusion_distillation_tutorial.html) — CM/iCT/sCM/CTM、LCM、DMD/DMD2、ADD/LADD。

推荐叙述主线：`像素/视频 → latent tokenizer → 噪声或概率路径 → 去噪器/向量场架构 → sampler → 条件控制 → 后训练 → 蒸馏与部署`。

## 6. Agent 方向

前置：RAG、评估和 RLHF/GRPO；如果面向 coding agent，还需理解推理 serving。

1. [Agent Foundations](docs/tutorials/agent_foundations_tutorial.html) — ReAct、planning、tool use、MCP/A2A、memory、computer use。
2. [RAG + Embedding / Retrieval](docs/tutorials/rag_embedding_retrieval_tutorial.html) — agent 的知识获取与检索地基。
3. [Agentic RL](docs/tutorials/agentic_rl_tutorial.html) — tool-use 轨迹、环境反馈、GRPO 与训练难点。
4. [Multi-Agent & Long-Horizon](docs/tutorials/multi_agent_long_horizon_tutorial.html) — 协作、debate、memory、长程规划与信用分配。
5. [Self-Evolving Agents](docs/tutorials/self_evolving_agents_tutorial.html) — skill/memory 演化、反思、自动课程与安全边界。

面试时应把“模型能力”和“系统能力”分开：模型负责决策与生成，agent runtime 负责状态、工具、权限、重试、预算、观测和评估。

## 7. 具身智能补充

仓库本体没有具身智能教程，README 收录了社区维护的外部资源：**具身智能高频面试题库，413 题、8 卷**，覆盖 VLA、模仿学习、RL、世界模型、腿足控制、3D 感知、工程落地、算法与系统设计。适合在完成 VLM、Agent、RLHF 基础之后，作为独立专项题库使用。

## 8. 可运行代码练习

代码位于 [`docs/tutorials/code/`](docs/tutorials/code/README.md)，共 15 个 Python 脚本。建议至少完成下面 10 个核心练习：

| 顺序 | 脚本 | 要验证的能力 |
|---|---|---|
| 1 | `tokenization.py` | BPE/Unigram/byte 映射不是只会调用 tokenizer |
| 2 | `mha.py` | 手写 MHA、causal mask，并与 PyTorch 对齐 |
| 3 | `transformer_block.py` | 组装完整 block，理解每一步 shape 与残差流 |
| 4 | `normalization.py` | LN/RMSNorm、Pre/Post-LN 梯度与初始化 |
| 5 | `optimizer_lr_schedule.py` | AdamW 与 Adam+L2 区别、bias correction、warmup |
| 6 | `pretraining_pipeline.py` | 数据 packing、loss mask 与预训练数据流 |
| 7 | `llm_eval_metrics.py` | pass@k、judge/排序类评估指标 |
| 8 | `lora.py` | LoRA 初始化、缩放、merge/unmerge 与 DoRA |
| 9 | `rag_embedding.py` | 双塔 InfoNCE、BM25 与 RRF 混合检索 |
| 10 | `inference_serving.py` | 推理请求、batching 与 serving 关键状态 |

专项脚本还有：`axial_attention.py`、`linear_sparse_attention.py`、`flow_matching.py`、`mmdit_block.py`、`toy_mmdit_t2i_pipeline.py`。

## 9. 专题博客：拓展阅读，不作为第一轮必修

1. [NVIDIA Cosmos 3：MoT 架构深度导读](docs/blogs/cosmos3_mot_guide.html) — omnimodal world model 与 Mixture-of-Transformers。
2. [Continuous DLM：表征视角](docs/blogs/continuous_dlm_representation_perspective.html) — 连续扩散语言模型与 Flow-Matching 家族。
3. [扩散 × 表征 × 流形](docs/blogs/diffusion_representation_manifold.html) — SSL、REPA、RAE、JiT、V-JEPA2 等表征路线。

## 10. 推荐学习路线

### 路线 A：LLM 算法 / Post-Training

主干 1–7 → LoRA → RLHF/DPO/GRPO/PPO → KL → Reasoning → OPD → 评估复盘。

### 路线 B：LLM 系统 / Infra

主干 1–6 → MoE → Distributed Training → Long Context → Quantization → KV Cache/Speculative Decoding → Inference Serving。

### 路线 C：多模态 / VLM

主干中的 Tokenization、Attention、Transformer、评估 → VLM → RAG → VAE/VQ → Diffusion Foundations → Image/Video Systems。

### 路线 D：生成模型

主干中的 Attention、Transformer、归一化、优化器 → VAE/VQ → Diffusion → Flow Matching → Image Systems → Video/3D → Post-Training → Distillation。

### 路线 E：Agent

主干中的 Transformer、评估 → RAG → Agent Foundations → RLHF/GRPO → Agentic RL → Multi-Agent → Self-Evolving Agents。

## 11. 后续我们如何一起学

你可以直接用下面几种指令让我进入不同学习模式：

- `从 Attention 开始带我学，先讲 §0–§2，然后考我。`
- `模拟面试官，随机问 Transformer Block 的 L1，不要先给答案。`
- `检查我对 DPO 的回答，按正确性、完整性、表达和追问风险评分。`
- `把 VLM 这一篇压缩成一页复习卡。`
- `结合我的项目经历，帮我准备这道题的项目化回答。`
- `给我制定 14 天 LLM Post-Training 学习计划，每天 2 小时。`

默认学习闭环是：**讲解 → 手推/代码 → 口述 → 追问 → 错题回访**。后续可根据目标岗位、时间和已有基础，把本目录裁剪成个人路线。
