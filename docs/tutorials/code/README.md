# Tutorial Code · runnable PyTorch references

最小可跑 PyTorch 教学脚本，配合 `docs/tutorials/*.md` 的概念讲解阅读。
全部为纯 PyTorch 实现，无外部依赖（除 `torch` 和可选的 `matplotlib`），
默认 CPU 几秒到几十秒可跑完。

## 文件清单

| 脚本 | 主题 | 对应 tutorial | 耗时 (CPU) |
|---|---|---|---|
| `mha.py` | Multi-Head Self-Attention + causal mask + 与 `nn.MultiheadAttention` 对齐验证 | [attention_tutorial.md](../attention_tutorial.md) | <5s |
| `axial_attention.py` | Axial attention（H/W 拆分）+ 复杂度对比表 + 感受野验证 | [attention_tutorial.md](../attention_tutorial.md) | <5s |
| `flow_matching.py` | Rectified Flow on 2D toy data (two moons) + Euler sampling + 轨迹可视化 | [flow_matching_tutorial.md](../flow_matching_tutorial.md) | ~30s |
| `dit.py` | 经典单流 DiT：patchify · adaLN-Zero block · 零初始化 final layer · null class；断言 step-0 恒等 / gate 梯度非零 / γβ 梯度为 0 | [image_generation_systems_tutorial.md](../image_generation_systems_tutorial.md) | <5s |
| `mmdit_block.py` | 双流 MMDiT block（joint attention + AdaLN-Zero + per-stream FFN） | [image_generation_systems_tutorial.md](../image_generation_systems_tutorial.md) | <5s |
| `toy_mmdit_t2i_pipeline.py` | End-to-end skeleton（toy text encoder + VAE + MMDiT + Euler scheduler + true CFG） | [image_generation_systems_tutorial.md](../image_generation_systems_tutorial.md) | <10s |
| `lora.py` | `LoRALinear`（B=0 起点 · α/r 与 rsLoRA 缩放 · merge/unmerge）+ `DoRALinear`（幅度-方向分解）+ 6 个 assert | [lora_peft_tutorial.md](../lora_peft_tutorial.md) | <5s |
| `rag_embedding.py` | 双塔 `DualEncoder` + InfoNCE（in-batch / 难负例）+ BM25 + RRF 混合召回 + 6 个 assert | [rag_embedding_retrieval_tutorial.md](../rag_embedding_retrieval_tutorial.md) | <5s |
| `linear_sparse_attention.py` | linear attention 递推/chunkwise 等价 + delta rule（DeltaNet 改写式更新）+ block-sparse top-k（NSA/MoBA select）+ 6 个 assert | [linear_sparse_attention_tutorial.md](../linear_sparse_attention_tutorial.md) | <5s |
| `normalization.py` | LayerNorm/RMSNorm from-scratch vs `torch` + RMSNorm 去 re-centering + BatchNorm train≠eval + Pre/Post-LN 梯度 top-heavy + Kaiming/Xavier 二阶矩 E[y²] + GPT-2 残差 1/√(2N) + 6 个 assert | [normalization_init_tutorial.md](../normalization_init_tutorial.md) | <5s |
| `optimizer_lr_schedule.py` | SGD-momentum/Adam/AdamW from-scratch vs `torch.optim` + **AdamW≠Adam+L2**(解耦≠耦合) + bias correction 方向(首步 3.16× 偏大) + cosine-warmup 调度 + 动量加速病态二次 + 6 个 assert | [optimizer_lr_schedule_tutorial.md](../optimizer_lr_schedule_tutorial.md) | <5s |
| `diffusion_online_rl.py` | Flow-GRPO 的 ODE→SDE 保边缘（1-D 解析方差递推 vs MC，含忘掉 score 修正的反例）+ DiffusionNFT 反射双支的更新方向与 ±2 最优解 + DGPO 权重平衡的 $\log Z$ 抵消（单位权重反例）+ 全部 assert 对解析答案 | [modern_diffusion_post_training_tutorial.md](../modern_diffusion_post_training_tutorial.md) | <5s |
| `world_models_toy.py` | RSSM 线性高斯 toy：单 KL 塌缩 vs balanced KL 的解析不动点 + JEPA 共缩捷径 vs stop-grad/EMA + gridworld 坐标 LAM 精确恢复动作（purity 1、$I(K;A)=2$ bits）；全部断言对解析解 | [world_models_tutorial.md](../world_models_tutorial.md) | <5s |
| `transformer_block.py` | 从零 Transformer block：残差拓扑 · 因果不泄漏 · full vs KV-cache 一致 · MHA/GQA/MQA 与参数量公式 · packed QKV · RoPE 设备/精度 + 15 个 check | [transformer_block_tutorial.md](../transformer_block_tutorial.md) | <5s |
| `llm_eval_metrics.py` | pass@k 真值 vs 无偏/plug-in 估计 · paired bootstrap · Bradley-Terry 两方 MLE(=log3) · judge 位置偏差，纯 stdlib+numpy | [llm_evaluation_benchmarking_tutorial.md](../llm_evaluation_benchmarking_tutorial.md) | <5s |
| `tokenization.py` | 精确复现 GPT-2 `bytes_to_unicode` · BPE 训练/编码 · WordPiece · Unigram Viterbi/前向 DP · byte fallback 往返 · prefix 不稳定性，纯 stdlib | [tokenization_tutorial.md](../tokenization_tutorial.md) | <1s |
| `pretraining_pipeline.py` | MinHash/LSH 去重 · packing 的 label 与 block mask · EOS 边界 · padding loss mask · Chinchilla 拟合 · loss spike 监控，纯 stdlib | [llm_pretraining_pipeline_tutorial.md](../llm_pretraining_pipeline_tutorial.md) | <5s |
| `inference_serving.py` | 采样算子精确定义(top-k/top-p/min-p/typical，顺序不可交换) · 流式 stop-string · static vs continuous batching · chunked prefill · KV 字节/block 分配/prefix 共享 · roofline，纯 stdlib | [llm_inference_serving_tutorial.md](../llm_inference_serving_tutorial.md) | <5s |

## 运行

```bash
cd docs/tutorials/code
python mha.py
python axial_attention.py
python flow_matching.py          # 需要 matplotlib（可选，没装会跳过画图）
python dit.py
python mmdit_block.py
python toy_mmdit_t2i_pipeline.py # 依赖 mmdit_block.py 在同目录
python lora.py
python rag_embedding.py
python linear_sparse_attention.py
python normalization.py
python optimizer_lr_schedule.py
```

每个脚本都自带 sanity check：shape 验证 + 数值合理性检查 + 必要时跟 PyTorch
内置实现对齐。运行失败时会 `assert` 出来。

## 设计目标

1. **教学清晰 > 性能**：每个 op 都展开写，shape 注释齐全
2. **可独立运行**：默认参数小，CPU 几秒钟跑完，不依赖 GPU
3. **跟 tutorial 一一对应**：每个脚本对应一个或两个 markdown 文档的核心概念
4. **常见架构组件、玩具尺寸**：保留主流公开架构组件（双流 MMDiT、AdaLN-Zero、
   joint attention、Euler scheduler、true CFG），hidden/layer 缩到 toy size

## 不包含什么

- ❌ 真实的预训练权重 / checkpoint
- ❌ 分布式训练 / 大规模数据加载
- ❌ Memory-efficient attention kernels / 低精度算子 / 显存优化
- ❌ Gradient checkpointing

这些都不是教学代码的重点。如果需要看完整工程实现，请参考主流公开的
diffusion / transformer library。

## English notes

All in-code documentation (docstrings, comments) is in English so the
code stays accessible to non-Chinese-reading contributors. This README
itself is bilingual for consistency with the rest of `docs/tutorials/`.

